from __future__ import print_function
import os
import pickle
import argparse

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
            self.c1 = nn.Linear(28*28, 100); self.c2 = nn.Linear(100, 100);
            self.c3 = nn.Linear(100, 100); self.c4 = nn.Linear(100, 100);
            self.c5 = nn.Linear(100, 100); self.c6 = nn.Linear(100, 100);
            self.c7 = nn.Linear(100, 100); self.c8 = nn.Linear(100, 100);
            self.c9 = nn.Linear(100, 100); self.c10 = nn.Linear(100, 100);
            self.c11 = nn.Linear(100, 100); self.c12 = nn.Linear(100, 100);
            self.c13 = nn.Linear(100, 100); self.c14 = nn.Linear(100, 100);
            self.c15 = nn.Linear(100, 100); self.c16 = nn.Linear(100, 100);
            self.c17 = nn.Linear(100, 100); self.c18 = nn.Linear(100, 100);
            self.c19 = nn.Linear(100, 100); self.c20 = nn.Linear(100, 100);

            self.cf1 = nn.Linear(100, 10); self.cf2 = nn.Linear(100, 10);
            self.cf3 = nn.Linear(100, 10); self.cf4 = nn.Linear(100, 10);
            self.cf5 = nn.Linear(100, 10); self.cf6 = nn.Linear(100, 10);
            self.cf7 = nn.Linear(100, 10); self.cf8 = nn.Linear(100, 10);
            self.cf9 = nn.Linear(100, 10); self.cf10 = nn.Linear(100, 10);
            self.cf11 = nn.Linear(100, 10); self.cf12 = nn.Linear(100, 10);
            self.cf13 = nn.Linear(100, 10); self.cf14 = nn.Linear(100, 10);
            self.cf15 = nn.Linear(100, 10); self.cf16 = nn.Linear(100, 10);
            self.cf17 = nn.Linear(100, 10); self.cf18 = nn.Linear(100, 10);
            self.cf19 = nn.Linear(100, 10); self.cf20 = nn.Linear(100, 10);

            self.standby_c = [self.c1, self.c2, self.c3, self.c4,
                              self.c5, self.c6, self.c7, self.c8,
                              self.c9, self.c10, self.c11, self.c12,
                              self.c13, self.c14, self.c15, self.c16,
                              self.c17, self.c18, self.c19, self.c20]

            self.standby_cf = [self.cf1, self.cf2, self.cf3, self.cf4,
                               self.cf5, self.cf6, self.cf7, self.cf8,
                               self.cf9, self.cf10, self.cf11, self.cf12,
                               self.cf13, self.cf14, self.cf15, self.cf16,
                               self.cf17, self.cf18, self.cf19, self.cf20]


        else:
            self.c1 = nn.Linear(28*28, 150)
            self.c2 = nn.Linear(150, 100)
            self.c3 = nn.Linear(100, 50)
            self.c4 = nn.Linear(50, 10)

            self.cf1 = nn.Linear(150, 10)
            self.cf2 = nn.Linear(100, 10)
            self.cf3 = nn.Linear(50, 10)
            self.cf4 = nn.Linear(10, 10)
            self.standby_c = [self.c1, self.c2, self.c3, self.c4]
            self.standby_cf = [self.cf1, self.cf2, self.cf3, self.cf4]


        self.frozen_c = []
        self.training_c = None
        self.training_cf = None

        self.train_acc = []
        self.test_acc = []


    def forward(self, x):
        x = x.view(-1, 28*28)
        for c in self.frozen_c:
            x = F.relu(c(x))
        x = F.relu(self.training_c(x))
        x = F.log_softmax(self.training_cf(x))
        return x


    def add_layer(self):
        if self.training_c:
            self.training_c.requires_grad = False
            self.frozen_c.append(self.training_c)
        try:
            self.training_c = self.standby_c.pop(0)
            self.training_cf = self.standby_cf.pop(0)
            trainable_params = [{'params': self.training_c.parameters()},
                                {'params': self.training_cf.parameters()}
            ]
            self.optimizer = optim.SGD(trainable_params, lr=args.lr, momentum=args.momentum)
        except:
            print('No more standby layers!')


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
    

    def print_model(self, epoch):
        print('Epoch {} ..............'.format(epoch))

        print('\n')
        print('Frozen layers')
        for frozen in self.frozen_c:
            print(frozen)
            print('====>')
        print('\n')
        print('C: {}'.format(self.training_c))
        print('C_F: {}'.format(self.training_cf))
        print('\n')

    def save_result(self):
        try:
            os.makedirs('./result')
        except:
            print('directory ./result already exists')
        filename = os.path.join('./result/', 'ft_deep.pickle' if self.deep else 'ft.pickle')
        f = open(filename,'w')
        pickle.dump((self.train_acc, self.test_acc), f)
        f.close()

        
model = Net(args.deep)
if args.cuda:
    model.cuda()


for epoch in range(0, args.epochs):
    if epoch % 10 == 0:
        model.add_layer()
        model.print_model(epoch)
    model.train_(epoch)
    model.test_(epoch)
model.save_result()
