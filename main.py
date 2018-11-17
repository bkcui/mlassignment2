'''Train CIFAR10 with PyTorch.'''
from __future__ import print_function

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.backends.cudnn as cudnn

import torchvision
import torchvision.transforms as transforms

import os
import argparse

from models import *
from utils import progress_bar
from utils import ImageFolderWithPaths


parser = argparse.ArgumentParser(description='PyTorch CIFAR10 Training')
parser.add_argument('--lr', default=0.1, type=float, help='learning rate')
parser.add_argument('--resume', '-r', action='store_true', help='resume from checkpoint')
parser.add_argument('--predict', action='store_true', help='forward prop')
args = parser.parse_args()

device = 'cuda' if torch.cuda.is_available() else 'cpu'
best_acc = 0  # best test accuracy
start_epoch = 0  # start from epoch 0 or last checkpoint epoch
batch_size = 25

# Data
print('==> Preparing data..')
transform_train = transforms.Compose([
    transforms.RandomCrop(32, padding=4),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
])

transform_test = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
])


trainset = torchvision.datasets.ImageFolder(root='tr', transform=torchvision.transforms.ToTensor())
trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size, shuffle=True, num_workers=2)

valset = torchvision.datasets.ImageFolder(root='val', transform=torchvision.transforms.ToTensor())
valloader = torch.utils.data.DataLoader(valset, batch_size=batch_size, shuffle=False, num_workers=2)

#classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')
classes = ('cup', 'coffee', 'bed', 'tree', 'bird', 'chair', 'tea', 'bread', 'bicycle', 'sail')

# Model
print('==> Building model..')
# net = VGG('VGG19')
net1 = ResNet18()
# net = PreActResNet18()
net2 = GoogLeNet()
# net = DenseNet121()
# net = ResNeXt29_2x64d()
# net = MobileNet()
# net = MobileNetV2()
#net = DPN92()
# net = ShuffleNetG2()
# net = SENet18()
#net = ShuffleNetV2(1)
#net = BiRNN()
net3 = bi_DenseNet121()
net4 = shake_net()
net1 = net1.to(device)
net2 = net2.to(device)
net3 = net3.to(device)
net4 = net4.to(device)
#checkpoint = torch.load('checkpoint/bidense3.t7')
#net.load_state_dict(checkpoint['net'])
#best_acc = checkpoint['acc']
#start_epoch = checkpoint['epoch']


if device == 'cuda':
    net = torch.nn.DataParallel(net1)
    net = torch.nn.DataParallel(net2)
    net = torch.nn.DataParallel(net3)
    net = torch.nn.DataParallel(net4)
    cudnn.benchmark = True

if args.resume:
    # Load checkpoint.
    print('==> Resuming from checkpoint..')
    checkpoint = torch.load('/content/gdrive/My Drive/Colab Notebooks/ResNet18.t7')
    net1.load_state_dict(checkpoint['net'])
    best_acc = checkpoint['acc']
    start_epoch = checkpoint['epoch']
    print('net 1 acc :', best_acc, 'epoch :', start_epoch)
    checkpoint = torch.load('/content/gdrive/My Drive/Colab Notebooks/GoogLeNet.t7')
    net2.load_state_dict(checkpoint['net'])
    best_acc = checkpoint['acc']
    start_epoch = checkpoint['epoch']
    print('net 2 acc :', best_acc, 'epoch :', start_epoch)
    checkpoint = torch.load('/content/gdrive/My Drive/Colab Notebooks/bidense2.t7')
    net3.load_state_dict(checkpoint['net'])
    best_acc = checkpoint['acc']
    start_epoch = checkpoint['epoch']
    print('net 3 acc :', best_acc, 'epoch :', start_epoch)
    checkpoint = torch.load('/content/gdrive/My Drive/Colab Notebooks/shake.t7')
    net4.load_state_dict(checkpoint['net'])
    best_acc = checkpoint['acc']
    start_epoch = checkpoint['epoch']
    print('net 4 acc :', best_acc, 'epoch :', start_epoch)

criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(net.parameters(), lr=args.lr, momentum=0.9, weight_decay=5e-4)

# Training
def train(epoch):
    print('\nEpoch: %d' % epoch)
    net.train()
    train_loss = 0
    correct = 0
    total = 0
    for batch_idx, (inputs, targets) in enumerate(trainloader):
        inputs, targets = inputs.to(device), targets.to(device)
        optimizer.zero_grad()
        outputs = net(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()

        train_loss += loss.item()
        _, predicted = outputs.max(1)
        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()

        progress_bar(batch_idx, len(trainloader), 'Loss: %.3f | Acc: %.3f%% (%d/%d)'
            % (train_loss/(batch_idx+1), 100.*correct/total, correct, total))
    print(correct/total)

def test(epoch):
    global best_acc
    net.eval()
    test_loss = 0
    correct = 0
    total = 0
    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(valloader):
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = net(inputs)
            loss = criterion(outputs, targets)

            test_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()

            progress_bar(batch_idx, len(valloader), 'Loss: %.3f | Acc: %.3f%% (%d/%d)'
                % (test_loss/(batch_idx+1), 100.*correct/total, correct, total))

    # Save checkpoint.
    acc = 100.*correct/total
    print(acc)
    if acc > best_acc:
        print('Saving..  %f' % acc)
        state = {
            'net': net.state_dict(),
            'acc': acc,
            'epoch': epoch,
        }
        #if not os.path.isdir('checkpoint'):
            #os.mkdir('checkpoint')
        torch.save(state, 'checkpoint/bidense.t7')
        best_acc = acc


def predict():
    testset = ImageFolderWithPaths(root='ts', transform=torchvision.transforms.ToTensor())
    testloader = torch.utils.data.DataLoader(testset, batch_size=1, shuffle=False, num_workers=2)
    net.eval()
    data = "filename,classid\n"
    with torch.no_grad():
        for batch_idx, (inputs, targets, path) in enumerate(testloader):
            inputs, targets = inputs.to(device), targets.to(device)
            images = inputs.reshape(-1, 32, 32).to(device)
            outputs = net(inputs)

            _, predicted = outputs.max(1)
            print(path[0].replace("\\0\\", "/") + ",{0:02d}".format(predicted.data.tolist()[0]))
            data += path[0].replace("\\0\\", "/")  + ',{0:02d}'.format(predicted.data.tolist()[0]) + '\n'

    with open('result.csv', 'w') as result:
        result.write(data)


if __name__ == '__main__':


    learning_rate = args.lr

    if args.predict:
        predict()

    else:
        for epoch in range(start_epoch, start_epoch+200):
            train(epoch)
            test(epoch)