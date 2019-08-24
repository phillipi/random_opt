#https://github.com/pytorch/examples/blob/master/mnist/main.py

from __future__ import print_function
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
import numpy as np


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(1, 20, 5, 1)
        #self.conv2 = nn.Conv2d(20, 10, 5, 1)
        self.conv2 = nn.Conv2d(20, 50, 5, 1)
        self.fc1 = nn.Linear(4*4*50, 500)
        self.fc2 = nn.Linear(500, 10)
        #self.fc1 = nn.Linear(4*4*10, 10)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.max_pool2d(x, 2, 2)
        x = F.relu(self.conv2(x))
        x = F.max_pool2d(x, 2, 2)
        #x = x.view(-1, 4*4*10)
        x = x.view(-1, 4*4*50)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        #x = self.fc1(x)
        return F.log_softmax(x, dim=1)
    
def weights_init(m):
    if isinstance(m, nn.Conv2d):
        #torch.nn.init.xavier_uniform(m.weight.data)
        #torch.nn.init.kaiming_uniform_(m.weight.data)
        #torch.nn.init.sparse_(m.weight.data, 0.9, std=0.01)
        torch.nn.init.normal_(m.weight.data, mean=0.0, std=1.0)
        #mask = torch.abs(m.weight.data)<2.0
        #m.weight.data[mask] = 0
        torch.nn.init.normal_(m.bias, mean=0.0, std=1.0)
    elif isinstance(m, nn.Linear):
        #torch.nn.init.xavier_uniform(m.weight.data)
        #torch.nn.init.kaiming_uniform_(m.weight.data)
        #torch.nn.init.sparse_(m.weight.data, 0.9, std=0.01)
        torch.nn.init.normal_(m.weight.data, mean=0.0, std=1.0)
        mask = torch.abs(m.weight.data)<3.0
        m.weight.data[mask] = 0
        torch.nn.init.normal_(m.bias, mean=0.0, std=1.0)

def train(args, seed_start, best_acc, best_seed, N, model, device, train_loader, optimizer, epoch):
    
    model.train()
    
    # load data
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        break # just load one batch
    
    for i in range(0,N):
        
        # rand init
        torch.manual_seed(i+seed_start)
        model.apply(weights_init)
        
        # eval model
        output = model(data)
        pred = output.argmax(dim=1, keepdim=True) # get the index of the max log-probability
        acc = pred.eq(target.view_as(pred)).sum().item()/pred.size(0)
        
        if acc > best_acc:
            best_acc = acc
            best_seed = i
        
        if i % args.log_interval == 0:
            print('(iter {}) best acc: {:.0f}%'.format(i, 100*best_acc))
    
    return best_seed, best_acc

def test(args, model, device, test_loader):
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            test_loss += F.nll_loss(output, target, reduction='sum').item() # sum up batch loss
            pred = output.argmax(dim=1, keepdim=True) # get the index of the max log-probability
            correct += pred.eq(target.view_as(pred)).sum().item()

    test_loss /= len(test_loader.dataset)

    print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
        test_loss, correct, len(test_loader.dataset),
        100. * correct / len(test_loader.dataset)))

def main():
    # Training settings
    parser = argparse.ArgumentParser(description='PyTorch MNIST Example')
    parser.add_argument('--batch-size', type=int, default=256, metavar='N',
                        help='input batch size for training (default: 256)')
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
    parser.add_argument('--seed', type=int, default=1, metavar='S',
                        help='random seed (default: 1)')
    parser.add_argument('--log-interval', type=int, default=1000, metavar='N',
                        help='how many batches to wait before logging training status')
    
    parser.add_argument('--save-model', action='store_true', default=False,
                        help='For Saving the current Model')
    args = parser.parse_args()
    use_cuda = not args.no_cuda and torch.cuda.is_available()

    torch.manual_seed(args.seed)

    device = torch.device("cuda" if use_cuda else "cpu")

    kwargs = {'num_workers': 1, 'pin_memory': True} if use_cuda else {}
    train_loader = torch.utils.data.DataLoader(
        datasets.MNIST('../data', train=True, download=True,
                       transform=transforms.Compose([
                           transforms.ToTensor(),
                           transforms.Normalize((0.1307,), (0.3081,))
                       ])),
        batch_size=args.batch_size, shuffle=True, **kwargs)
    test_loader = torch.utils.data.DataLoader(
        datasets.MNIST('../data', train=False, transform=transforms.Compose([
                           transforms.ToTensor(),
                           transforms.Normalize((0.1307,), (0.3081,))
                       ])),
        batch_size=args.test_batch_size, shuffle=True, **kwargs)


    model = Net().to(device)
    optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum)

    seed_start = 0
    best_acc = 0.0
    best_seed = 0
    N = 40000
    for epoch in range(1, args.epochs + 1):
        best_seed, best_acc = train(args, seed_start, best_acc, best_seed, N, model, device, train_loader, optimizer, epoch)
        seed_start += N
        torch.manual_seed(best_seed)
        model.apply(weights_init)
        test(args, model, device, test_loader)

    if (args.save_model):
        torch.save(model.state_dict(),"mnist_cnn.pt")
        
if __name__ == '__main__':
    main()