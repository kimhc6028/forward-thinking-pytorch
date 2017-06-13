from __future__ import print_function
import os
import argparse
import pickle
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torch.autograd import Variable

# Training settings
parser = argparse.ArgumentParser(description='PyTorch MNIST Example')
parser.add_argument('--batch-size', type=int, default=64, metavar='N',
                    help='input batch size for training (default: 64)')
parser.add_argument('--test-batch-size', type=int, default=1000, metavar='N',
                    help='input batch size for testing (default: 1000)')
parser.add_argument('--epochs', type=int, default=40, metavar='N',
                    help='number of epochs to train (default: 40)')
parser.add_argument('--lr', type=float, default=0.01, metavar='LR',
                    help='learning rate (default: 0.01)')
parser.add_argument('--momentum', type=float, default=0.5, metavar='M',
                    help='SGD momentum (default: 0.5)')
parser.add_argument('--no-cuda', action='store_true', default=False,
                    help='disables CUDA training')
parser.add_argument('--seed', type=int, default=1, metavar='S',
                    help='random seed (default: 1)')
parser.add_argument('--log-interval', type=int, default=10, metavar='N',
                    help='how many batches to wait before logging training status')
parser.add_argument('--deep', action='store_true', default=False,
                    help='using deep model (20 fully connected layers)')

args = parser.parse_args()
args.cuda = not args.no_cuda and torch.cuda.is_available()

torch.manual_seed(args.seed)
if args.cuda:
    torch.cuda.manual_seed(args.seed)

try:
    os.makedirs('./data')
except:
    print('directory ./data already exists')

kwargs = {'num_workers': 1, 'pin_memory': True} if args.cuda else {}
train_loader = torch.utils.data.DataLoader(
    datasets.MNIST('./data', train=True, download=True,
                   transform=transforms.Compose([
                       transforms.ToTensor(),
                       transforms.Normalize((0.1307,), (0.3081,))
                   ])),
    batch_size=args.batch_size, shuffle=True, **kwargs)
test_loader = torch.utils.data.DataLoader(
    datasets.MNIST('./data', train=False, transform=transforms.Compose([
                       transforms.ToTensor(),
                       transforms.Normalize((0.1307,), (0.3081,))
                   ])),
    batch_size=args.batch_size, shuffle=True, **kwargs)


class Net(nn.Module):
    def __init__(self, deep):
        super(Net, self).__init__()

        self.deep = deep
        if deep:
            self.fc1 = nn.Linear(28*28, 100); self.fc2 = nn.Linear(100, 100); self.fc3 = nn.Linear(100, 100); self.fc4 = nn.Linear(100, 100);
            self.fc5 = nn.Linear(100, 100); self.fc6 = nn.Linear(100, 100); self.fc7 = nn.Linear(100, 100); self.fc8 = nn.Linear(100, 100);
            self.fc9 = nn.Linear(100, 100); self.fc10 = nn.Linear(100, 100); self.fc11 = nn.Linear(100, 100); self.fc12 = nn.Linear(100, 100);
            self.fc13 = nn.Linear(100, 100); self.fc14 = nn.Linear(100, 100); self.fc15 = nn.Linear(100, 100); self.fc16 = nn.Linear(100, 100);
            self.fc17 = nn.Linear(100, 100); self.fc18 = nn.Linear(100, 100); self.fc19 = nn.Linear(100, 100); self.fc20 = nn.Linear(100, 10);
            self.fcs = [self.fc1, self.fc2, self.fc3, self.fc4,
                        self.fc5, self.fc6, self.fc7, self.fc8,
                        self.fc9, self.fc10, self.fc11, self.fc12,
                        self.fc13, self.fc14, self.fc15, self.fc16,
                        self.fc17, self.fc18, self.fc19, self.fc20]

        else:
            self.fc1 = nn.Linear(28*28, 150)
            self.fc2 = nn.Linear(150, 100)
            self.fc3 = nn.Linear(100, 50)
            self.fc4 = nn.Linear(50, 10)
            self.fc5 = nn.Linear(10, 10)

        self.optimizer = optim.SGD(self.parameters(), lr=args.lr, momentum=args.momentum)

        self.train_acc = []
        self.test_acc = []


    def forward(self, x):
        if self.deep:
            x = x.view(-1, 28*28)            
            for fc in self.fcs[:-1]:
                x = F.relu(fc(x))
            x = self.fcs[-1](x)
            return F.log_softmax(x)

        else:
            x = x.view(-1, 28*28)
            x = F.relu(self.fc1(x))
            x = F.relu(self.fc2(x))
            x = F.relu(self.fc3(x))
            x = F.relu(self.fc4(x))
            x = self.fc5(x)
            return F.log_softmax(x)


    def train_(self, epoch):
        self.train()
        for batch_idx, (data, target) in enumerate(train_loader):
            correct = 0
            if args.cuda:
                data, target = data.cuda(), target.cuda()
            data, target = Variable(data), Variable(target)
            self.optimizer.zero_grad()
            output = self(data)
            loss = F.nll_loss(output, target)
            loss.backward()
            pred = output.data.max(1)[1] # get the index of the max log-probability
            correct += pred.eq(target.data).cpu().sum()
            accuracy = 100. * correct / len(data)
            self.optimizer.step()
            if batch_idx % args.log_interval == 0:
                print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}, Accuracy: {}/{} ({:.4f}%)'.format(
                    epoch, batch_idx * len(data), len(train_loader.dataset),
                    100. * batch_idx / len(train_loader), loss.data[0],
                    correct, len(data),
                    accuracy))

            self.train_acc.append(accuracy)


    def test_(self, epoch):
        self.eval()
        test_loss = 0
        correct = 0
        for data, target in test_loader:
            if args.cuda:
                data, target = data.cuda(), target.cuda()
            data, target = Variable(data, volatile=True), Variable(target)
            output = self(data)
            test_loss += F.nll_loss(output, target).data[0]
            pred = output.data.max(1)[1] # get the index of the max log-probability
            correct += pred.eq(target.data).cpu().sum()
            
        test_loss = test_loss
        test_loss /= len(test_loader) # loss function already averages over batch size
        accuracy = 100. * correct / len(test_loader.dataset)
        print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.4f}%)\n'.format(
            test_loss, correct, len(test_loader.dataset),
            accuracy))

        self.test_acc.append(accuracy)


    def save_result(self):
        try:
            os.makedirs('./result')
        except:
            print('directory ./result already exists')
        filename = os.path.join('./result/', 'bp_deep.pickle' if self.deep else 'bp.pickle')
        f = open(filename,'w')
        pickle.dump((self.train_acc, self.test_acc), f)
        f.close()

model = Net(args.deep)
if args.cuda:
    model.cuda()


for epoch in range(1, args.epochs + 1):
    model.train_(epoch)
    model.test_(epoch)
model.save_result()
