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

def gen_save_adv_example(attacker, train_data, data_name, epoch):
    for  index, (x, y) in tqdm(enumerate(train_data)):
        x,y = torch.tensor(x).detach(), torch.tensor(y).detach()
        x_adv = attacker.atk(x,y).detach()
        path_x = os.path.join( os.getcwd(),'data', data_name , 'epoch_'+ str(epoch), 'X')
        path_y = os.path.join( os.getcwd(),'data', data_name , 'epoch_'+ str(epoch), 'y')
        os.makedirs(path_x, exist_ok = True)
        os.makedirs(path_y, exist_ok = True)
        torch.save(x_adv,  os.path.join(path_x, f"x_{index}.pth"))
        torch.save(y,  os.path.join(path_y, f"y_{index}.pth"))

class DatasetFTL(Dataset):
    def __init__(self, data_name, epoch, data_window):
        self.data_name = data_name
        self.epoch = epoch
        self.data_window = data_window
        self.n_samples = len(os.listdir(os.path.join(os.getcwd(), 'data', data_name, 'epoch_'+ str(epoch), 'X')))
    
    def __getitem__(self, index):
        x_list = []
        y_list = []
        for i in range(max(self.epoch+1 - self.data_window, 0), self.epoch +1 ):
            x_list.append(torch.load(os.path.join(os.getcwd(), 'data', self.data_name, 'epoch_'+ str(i), 'X', f"x_{index}.pth")))
            y_list.append(torch.load(os.path.join(os.getcwd(), 'data', self.data_name, 'epoch_'+ str(i), 'y', f"y_{index}.pth")))
        x = torch.stack(x_list)
        y = torch.stack(y_list)
        return x,y

    def __len__(self):
        return self.n_samples

def train(args, model, device, train_loader, optimizer, epoch):
    model.train()
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        # print(data.shape)
        data = data.reshape(-1, 1,28,28)
        # print(data.shape)
        # print(target.shape)

        optimizer.zero_grad()

        # calculate loss
        criterion = nn.CrossEntropyLoss()
        loss = criterion(model(data), target.reshape(-1))
        loss.backward()
        optimizer.step()
        if batch_idx % args.log_interval == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                       100. * batch_idx / len(train_loader), loss.item()))


def test(args, model, device, test_loader):
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
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
    parser.add_argument('--num-steps', default=10,
                        help='perturb number of steps')
    parser.add_argument('--step-size', default=0.02,
                        help='perturb step size')
    parser.add_argument('--beta', default=1.0,
                        help='regularization, i.e., 1/lambda in TRADES')
    parser.add_argument('--seed', type=int, default=1, metavar='S',
                        help='random seed (default: 1)')
    parser.add_argument('--log-interval', type=int, default=10, metavar='N',
                        help='how many batches to wait before logging training status')

    parser.add_argument('--save-model', action='store_true', default=False,
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
    train_data = datasets.MNIST('../data', train=True, download=True,
                       transform=transforms.Compose([
                           transforms.ToTensor(),
                       ]))

    test_loader = torch.utils.data.DataLoader(
        datasets.MNIST('../data', train=False, transform=transforms.Compose([
            transforms.ToTensor(),
        ])),
        batch_size=args.test_batch_size, shuffle=True, **kwargs)


    model = Net().to(device)
    optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum)
    
    model_list = [model]
    for epoch in range(1, args.epochs + 1):

        if len(model_list) > args.model_window:
            model_list.pop(0)
        # Generate adversarial example
        print('--- generating adversarial examples ---')
        attacker = FTL_PGD(model_list,
                        epsilon = args.epsilon,
                        step_size = args.step_size,
                        perturb_steps = args.num_steps)
        gen_save_adv_example(attacker, train_data, 'MNIST', epoch)

        # Generate a new trainloader
        dataset_i = DatasetFTL('MNIST', epoch = epoch, data_window = args.data_window)
        train_loader_i = torch.utils.data.DataLoader(dataset=dataset_i, 
                                                    batch_size= args.batch_size, 
                                                    shuffle=True,
                                                    **kwargs)
        # Train on the stacked train_loader
        print('--- training ----')
        train(args, model, device, train_loader_i, optimizer, epoch)
        test(args, model, device, test_loader)

        # Save_model
        torch.save(model.state_dict(), 'checkpoint/'+ 'MNIST' + '_epoch_'+ str(epoch))
        model_temp = copy.deepcopy(model)
        model_list.append(model_temp)


    if (args.save_model):
        torch.save(model.state_dict(), "mnist_cnn.pt")


if __name__ == '__main__':
    main()
