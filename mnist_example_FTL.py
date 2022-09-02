from __future__ import print_function
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from trades import trades_loss
from tqdm import tqdm
import os
import os.path
from torch.utils.data import Dataset
import copy


class IndexMNIST(Dataset):
    def __init__(self, root, train, download=False, transform=None, target_transform = None):
        self.data = datasets.MNIST(root, train=train, download=download, transform=transform, target_transform=target_transform )
    
    def __getitem__(self, index):
        return index, self.data[index][0], self.data[index][1]

    def __len__(self):
        return self.data.__len__()


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(1, 20, 5, 1)
        self.conv2 = nn.Conv2d(20, 50, 5, 1)
        self.fc1 = nn.Linear(4 * 4 * 50, 500)
        self.fc2 = nn.Linear(500, 10)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.max_pool2d(x, 2, 2)
        x = F.relu(self.conv2(x))
        x = F.max_pool2d(x, 2, 2)
        x = x.view(-1, 4 * 4 * 50)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x


class FTL_PGD():
    def __init__(self, 
                model_list,
                epsilon,
                step_size,
                perturb_steps,
                loss = nn.CrossEntropyLoss()):

        self.model_list = model_list
        self.device = next(model_list[0].parameters()).device
        self.epsilon = epsilon
        self.step_size = step_size
        self.perturb_steps = perturb_steps
        self.loss = loss

    def atk(self, images, labels):

        images = images.clone().detach().to(self.device)
        labels = labels.clone().detach().to(self.device)

        # random start
        adv_images = images.clone().detach()
        adv_images = adv_images + torch.empty_like(adv_images).uniform_(-self.epsilon, self.epsilon)
        adv_images = torch.clamp(adv_images, min = 0, max = 1).detach()

        for _ in range(self.perturb_steps):
            adv_images.requires_grad = True
            cost = 0
            
            for model in self.model_list:
                model.eval()
                outputs = model(adv_images)
                cost += self.loss(outputs, labels.reshape(-1))

            # Update adversarial images
            grad = torch.autograd.grad(cost, adv_images,
                                       retain_graph=False, create_graph=False)[0]

            adv_images = adv_images.detach() + self.step_size*grad.sign()
            delta = torch.clamp(adv_images - images, min=-self.epsilon, max=self.epsilon)
            adv_images = torch.clamp(images + delta, min = 0, max = 1).detach()

        return adv_images


def train(args, model, model_list, data_list, device, train_loader, optimizer, epoch):
    model.train()
    data_list.append({'X': {}, 'y':{}})

    # pop data list
    if len(data_list)> args.data_window:
        data_list.pop(0)
    
    # print(len(model_list))
    for batch_idx, (idx, x, y) in enumerate(train_loader):
        x, y = x.to(device), y.to(device)

        # pop model list
        if len(model_list) > args.model_window:
            model_list.pop(0)

        # generate adversarial example
        attacker = FTL_PGD(model_list, epsilon = args.epsilon, step_size = args.step_size, perturb_steps = args.num_steps)
        x_adv = attacker.atk(x,y)
        for i in range(idx.shape[0]):
            data_list[-1]['X'][idx[i].item()] = x_adv[i,:].cpu().detach()
            data_list[-1]['y'][idx[i].item()] = y[i].cpu().detach()

        # look up old stuff and stack up with the current x
        x_old = []
        y_old = []
        if len(data_list) > 1:
            # print(data_list[0]['y'])
            for i in range(len(data_list) - 1):
                for j in range(idx.shape[0]):
                    x_old.append(data_list[i]['X'][idx[j].item()])
                    y_old.append(data_list[i]['y'][idx[j].item()])

            x_old = torch.stack(x_old)
            y_old = torch.stack(y_old)
            # print(x_old.shape)
            # print(x_adv.shape)
            x_adv = torch.vstack([x_adv, x_old])
            y = torch.hstack([y, y_old])
            # print(x_adv.shape)
            # print(y.shape)

        optimizer.zero_grad()

        # calculate loss
        criterion = nn.CrossEntropyLoss()
        # print(print(x_adv.requires_grad))
        loss = criterion(model(x_adv), y.reshape(-1))
        loss.backward()
        optimizer.step()
        if batch_idx % args.log_interval == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(x), len(train_loader.dataset),
                       100. * batch_idx / len(train_loader), loss.item()))
        # add model into the list
        model_temp = copy.deepcopy(model)
        model_list.append(model_temp)


def test(args, model, device, test_loader):
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for _, data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            test_loss += F.nll_loss(output, target, reduction='sum').item()  # sum up batch loss
            pred = output.argmax(dim=1, keepdim=True)  # get the index of the max log-probability
            correct += pred.eq(target.view_as(pred)).sum().item()

    test_loss /= len(test_loader.dataset)

    print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
        test_loss, correct, len(test_loader.dataset),
        100. * correct / len(test_loader.dataset)))


def main():
    # Training settings
    parser = argparse.ArgumentParser(description='PyTorch MNIST Example')
    parser.add_argument('--batch-size', type=int, default=64, metavar='N',
                        help='input batch size for training (default: 64)')
    parser.add_argument('--test-batch-size', type=int, default=1000, metavar='N',
                        help='input batch size for testing (default: 1000)')
    parser.add_argument('--epochs', type=int, default=10, metavar='N',
                        help='number of epochs to train (default: 10)')
    parser.add_argument('--lr', type=float, default=0.01, metavar='LR',
                        help='learning rate (default: 0.01)')
    parser.add_argument('--momentum', type=float, default=0.5, metavar='M',
                        help='SGD momentum (default: 0.5)')
    parser.add_argument('--no-cuda', action='store_true', default=False,
                        help='disables CUDA training')
    parser.add_argument('--epsilon', default=0.1,
                        help='perturbation')
    parser.add_argument('--num-steps', type=int, default=10,
                        help='perturb number of steps')
    parser.add_argument('--step-size', default=0.02,
                        help='perturb step size')
    parser.add_argument('--beta', default=1.0,
                        help='regularization, i.e., 1/lambda in TRADES')
    parser.add_argument('--seed', type=int, default=1, metavar='S',
                        help='random seed (default: 1)')
    parser.add_argument('--log-interval', type=int, default=10, metavar='N',
                        help='how many batches to wait before logging training status')

    parser.add_argument('--save-model', action='store_true', default=True,
                        help='For Saving the current Model')
    parser.add_argument('--model-window', type=int, default=1,
                        help='how many model we save for generating adversarial example')
    parser.add_argument('--data-window', type=int, default=1,
                        help='how many data we save for training a model')
    args = parser.parse_args()
    use_cuda = not args.no_cuda and torch.cuda.is_available()

    torch.manual_seed(args.seed)

    device = torch.device("cuda" if use_cuda else "cpu")

    kwargs = {'num_workers': 1, 'pin_memory': True} if use_cuda else {}
    train_loader = torch.utils.data.DataLoader(
        IndexMNIST('../data', train=True, download=True,
                       transform=transforms.Compose([
                           transforms.ToTensor(),
                       ])),
        batch_size=args.batch_size, shuffle=True, **kwargs)

    test_loader = torch.utils.data.DataLoader(
        IndexMNIST('../data', train=False, transform=transforms.Compose([
            transforms.ToTensor(),
        ])),
        batch_size=args.test_batch_size, shuffle=True, **kwargs)


    model = Net().to(device)
    optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum)
    
    model_list = [model]
    data_list = []
    for epoch in range(1, args.epochs + 1):

        print('--- training ----')
        train(args, model, model_list, data_list, device, train_loader, optimizer, epoch)
        test(args, model, device, test_loader)
        
        # Save_model
        
        


    if (args.save_model):
        torch.save(model.state_dict(), './checkpoint/'+ 'MNIST'  + '_'+ str(args.model_window) + '_' + str(args.data_window)  + '_epoch_'+ str(epoch)+ '.pth')


if __name__ == '__main__':
    main()
