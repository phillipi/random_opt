#https://github.com/pytorch/examples/blob/master/mnist/main.py

from __future__ import print_function
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
import numpy as np
import time

class NetMNIST(nn.Module):
    def __init__(self):
        super(NetMNIST, self).__init__()
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

class NetCIFAR10(nn.Module):
    def __init__(self):
        super(NetCIFAR10, self).__init__()
        self.conv1 = nn.Conv2d(3, 20, 5, 1)
        self.conv2 = nn.Conv2d(20, 50, 5, 1)
        self.fc1 = nn.Linear(5*5*50, 500)
        self.fc2 = nn.Linear(500, 10)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.max_pool2d(x, 2, 2)
        x = F.relu(self.conv2(x))
        x = F.max_pool2d(x, 2, 2)
        x = x.view(-1, 5*5*50)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return F.log_softmax(x, dim=1)
        
def weights_init(m):
    if isinstance(m, nn.Conv2d):
        m.reset_parameters()
        
        #torch.nn.init.uniform_(m.weight.data,-1.0,1.0)
        #torch.nn.init.xavier_uniform(m.weight.data)
        #torch.nn.init.kaiming_uniform_(m.weight.data)
        #torch.nn.init.sparse_(m.weight.data, 0.9, std=0.01)
        #torch.nn.init.normal_(m.weight.data, mean=0.0, std=1.0)
        #mask = torch.abs(m.weight.data)<2.0
        #m.weight.data[mask] = 0
        
        #torch.nn.init.normal_(m.bias, mean=0.0, std=1.0)
        #torch.nn.init.constant_(m.bias, 0)
    elif isinstance(m, nn.Linear):
        m.reset_parameters()
        
        #torch.nn.init.xavier_uniform(m.weight.data)
        #torch.nn.init.kaiming_uniform_(m.weight.data)
        #torch.nn.init.sparse_(m.weight.data, 0.9, std=0.01)
        
        #torch.nn.init.normal_(m.weight.data, mean=0.0, std=1.0)
        #mask = torch.abs(m.weight.data)<3.0 # 3.0 seems to be the best...
        
        #torch.nn.init.uniform_(m.weight.data,-1.0,1.0)
        #mask = torch.abs(m.weight.data)<0.99
        
        #m.weight.data[mask] = 0
        
        #torch.nn.init.normal_(m.bias, mean=0.0, std=1.0)
        #torch.nn.init.constant_(m.bias, 0)

def train(args, seed_start, N, model, device, train_loader, epoch):
    
    model.train()
    
    criterion = nn.CrossEntropyLoss()
    
    # load data
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        break # just load one batch
    
    accs = np.zeros(N)
    losses = np.ones(N) * np.inf
    for i in range(0,N):
        
        # rand init
        seed = i+seed_start
        torch.manual_seed(seed)
        model.apply(weights_init)

        # linear layer on top
        '''
        A = torch.inverse(torch.mm(output.t(), output))
        W = torch.mv(torch.mm(A, output.t()), target.float())
        output = torch.mm(W, output)
        '''
        
        # SGD for M steps
        M = 1
        optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum)
        for j in range(0,M):
            
            optimizer.zero_grad()
            output = model(data)
            #loss = F.nll_loss(output, target, reduction='sum')
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()
            
            #pred = output.argmax(dim=1, keepdim=True) # get the index of the max log-probability
            #acc = pred.eq(target.view_as(pred)).sum().item()/pred.size(0)
            
            '''
            train_loss = loss.item()
            _, predicted = output.max(1)
            total = target.size(0)
            correct = predicted.eq(target).sum().item()
            '''
            
            #print('Loss: {:.3f} | Acc: {:.3f}% ({}/{})'.format(train_loss, 100.*correct/total, correct, total))
            
        
        # eval model
        output = model(data)
        #loss = F.nll_loss(output, target, reduction='sum').item() # sum up batch loss
        loss = criterion(output, target)
        pred = output.argmax(dim=1, keepdim=True) # get the index of the max log-probability
        acc = pred.eq(target.view_as(pred)).sum().item()/pred.size(0)
        accs[i] = acc
        losses[i] = loss
        
        '''
        if acc > best_acc:
            
            # double check
            acc = 0
            for batch_idx, (data, target) in enumerate(train_loader):
                data, target = data.to(device), target.to(device)
                output = model(data)
                pred = output.argmax(dim=1, keepdim=True) # get the index of the max log-probability
                acc += pred.eq(target.view_as(pred)).sum().item()/pred.size(0)
                if batch_idx>10:
                    break
            acc = acc/batch_idx
            
            if acc > best_acc:
                best_acc = acc
                best_seed = seed
        '''
        
        if i % args.log_interval == 0:
            best_acc = np.max(accs)
            print('(iter {}) best acc: {:.0f}%'.format(i, 100*best_acc))
            best_loss = np.min(losses)
            print('(iter {}) best loss: {}'.format(i, best_loss))
    
    #return best_seed, best_acc
    return accs, losses

def test(args, models, weights, device, test_loader):
    
    for model in models:
        model.eval()
    
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for batch_idx, (data, target) in enumerate(test_loader):
            data, target = data.to(device), target.to(device)
            
            output = None
            for model_idx, model in enumerate(models):
                if output is None:
                    output = weights[model_idx]*model(data)
                else:
                    output += weights[model_idx]*model(data)
            
            #test_loss += F.nll_loss(output, target, reduction='sum').item() # sum up batch loss
            pred = output.argmax(dim=1, keepdim=True) # get the index of the max log-probability
            correct += pred.eq(target.view_as(pred)).sum().item()/pred.size(0)
            #if batch_idx>10:
            #    break

    test_loss /= batch_idx#len(test_loader.dataset)

    print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
        test_loss, correct, batch_idx,#len(test_loader.dataset),
        100. * correct / batch_idx))#len(test_loader.dataset)))

def train_SGD(train_model, train_loader, optimizer, device, epoch):
    print('\nEpoch: %d' % epoch)
    criterion = nn.CrossEntropyLoss()
    train_model.train()
    train_loss = 0
    correct = 0
    total = 0
    for batch_idx, (inputs, targets) in enumerate(train_loader):
        inputs, targets = inputs.to(device), targets.to(device)
        optimizer.zero_grad()
        outputs = train_model(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()

        train_loss += loss.item()
        _, predicted = outputs.max(1)
        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()

        print('Loss: {:.3f} | Acc: {:.3f}% ({}/{})'.format(train_loss/(batch_idx+1), 100.*correct/total, correct, total))

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
    parser.add_argument('--log-interval', type=int, default=2000, metavar='N',
                        help='how many batches to wait before logging training status')
    parser.add_argument('--dataset', type=str, default='MNIST',
                        help='which dataset to use [MNIST, CIFAR10]')
                        
    parser.add_argument('--save-model', action='store_true', default=False,
                        help='For Saving the current Model')
    args = parser.parse_args()
    use_cuda = not args.no_cuda and torch.cuda.is_available()

    torch.manual_seed(args.seed)

    device = torch.device("cuda" if use_cuda else "cpu")

    kwargs = {'num_workers': 1, 'pin_memory': True} if use_cuda else {}
    
    if args.dataset == 'MNIST':
        Net = NetMNIST
        train_loader = torch.utils.data.DataLoader(
            datasets.MNIST('../data', train=True, download=True,
                           transform=transforms.Compose([
                               transforms.ToTensor(),
                               transforms.Normalize((0.1307,), (0.3081,))
                           ])),
            batch_size=args.batch_size, shuffle=False, **kwargs)
        test_loader = torch.utils.data.DataLoader(
            datasets.MNIST('../data', train=False, transform=transforms.Compose([
                               transforms.ToTensor(),
                               transforms.Normalize((0.1307,), (0.3081,))
                           ])),
            batch_size=args.test_batch_size, shuffle=False, **kwargs)
    else:
        Net = NetCIFAR10
        train_loader = torch.utils.data.DataLoader(
            datasets.CIFAR10('../data', train=True, download=True,
                           transform=transforms.Compose([
                               transforms.ToTensor(),
                               transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
                           ])),
            batch_size=args.batch_size, shuffle=False, **kwargs)
        test_loader = torch.utils.data.DataLoader(
            datasets.CIFAR10('../data', train=False, transform=transforms.Compose([
                               transforms.ToTensor(),
                               transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
                           ])),
            batch_size=args.test_batch_size, shuffle=False, **kwargs)

    #N_models_percent = 0.001
    N_models = 10
    #models = []
    #for i in range(0,N_models):
    #    models.append(Net().to(device))
    #optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum)
    train_model = Net().to(device)

    '''
    optimizer = optim.SGD(train_model.parameters(), lr=args.lr, momentum=args.momentum)
    for epoch in range(1, 10):
        train_SGD(train_model, train_loader, optimizer, device, epoch)
    '''
    
    seed_start_0 = np.random.randint(time.time())
    seed_start = seed_start_0
    N = 10000
    accs = np.array([])
    losses = np.array([])
    for epoch in range(1, 100):#args.epochs + 1):
        new_accs, new_losses = train(args, seed_start, N, train_model, device, train_loader, epoch)
        accs = np.concatenate((accs, new_accs))
        losses = np.concatenate((losses, new_losses))
        seed_start += N
        
        #ii = np.argsort(-accs)
        ii = np.argsort(losses)
        
        #N_models = np.minimum(int(N_models_percent*len(losses)), 4000)
        
        '''
        losses_ = losses[ii[0:N_models]]
        weights = np.exp(-losses_*0.01)
        weights = weights/np.sum(weights)
        #print('weights:',weights[1:100])
        '''
        weights = np.ones(N_models)
        
        models = []
        for i in range(0,N_models):
            models.append(Net().to(device))
            print('top seed {}: {} (loss: {}, acc: {}%)'.format(i, ii[i], losses[ii[i]], accs[ii[i]]))
            torch.manual_seed(ii[i]+seed_start_0)
            models[i].apply(weights_init)
        test(args, models, weights, device, test_loader)
    '''
    best_acc = 0.0
    best_seed = 0
    N = 10000
    for epoch in range(1, args.epochs + 1):
        best_seed, best_acc = train(args, seed_start, best_acc, best_seed, N, model, device, train_loader, optimizer, epoch)
        seed_start += N
        torch.manual_seed(best_seed)
        model.apply(weights_init)
        test(args, model, device, test_loader)
    '''

    #if (args.save_model):
    #    torch.save(model.state_dict(),"mnist_cnn.pt")
        
if __name__ == '__main__':
    main()