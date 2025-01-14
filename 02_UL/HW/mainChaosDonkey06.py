'''Train CIFAR10 with PyTorch.'''
from __future__ import print_function

import sys
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.backends.cudnn as cudnn

import torchvision
import torchvision.transforms as transforms
import lib.custom_transforms as custom_transforms

import os
import argparse
import time

import models
import datasets
import math
import statistics

from lib.NCEAverage import NCEAverage
from lib.LinearAverage import LinearAverage
from lib.NCECriterion import NCECriterion
from lib.utils import AverageMeter
from testChaosDonkey06 import NN, kNN

parser = argparse.ArgumentParser(description='PyTorch MNIST Training')
parser.add_argument('--lr', default=0.03, type=float, help='learning rate')
parser.add_argument('--resume', '-r', default='', type=str, help='resume from checkpoint')
parser.add_argument('--test-only', action='store_true', help='test only')
parser.add_argument('--low-dim', default=128, type=int,
                    metavar='D', help='feature dimension')
parser.add_argument('--nce-k', default=0, type=int,
                    metavar='K', help='number of negative samples for NCE')
parser.add_argument('--nce-t', default=0.1, type=float,
                    metavar='T', help='temperature parameter for softmax')
parser.add_argument('--nce-m', default=0.5, type=float,
                    metavar='M', help='momentum for non-parametric updates')
parser.add_argument('--epochs', default=100, type=int, metavar='N',
                    help='number of total epochs to run')
parser.add_argument('--batch_size', default=20, type=int, help='training batch size')
parser.add_argument('--batch_size_test', default=100, type=int, help='test batch size')

parser.add_argument('--recons_weigth', default=0.6, type=float, help='weigth of reconstruction')

args = parser.parse_args()

device = 'cuda' if torch.cuda.is_available() else 'cpu'
best_acc = 0  # best test accuracy
start_epoch = 0  # start from epoch 0 or last checkpoint epoch


if __name__ == '__main__':

    # Data
    print('==> Preparing data..')
    transform_train = transforms.Compose([
        transforms.RandomResizedCrop(size=32, scale=(0.2,1.)),
        transforms.ColorJitter(0.4, 0.4, 0.4, 0.4),
        transforms.RandomGrayscale(p=0.2),
        #transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        #transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        transforms.Normalize([0.4914], [0.2023]),
    ])

    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize([0.4914], [0.2023]),
    ])

    #### TODO: Modify this part to change the dataset ######
    trainset = datasets.MNISTInstance(root='./data', train=True, download=True, transform=transform_train)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=args.batch_size, shuffle=True, num_workers=1)

    testset = datasets.MNISTInstance(root='./data', train=False, download=True, transform=transform_train)
    testloader = torch.utils.data.DataLoader(testset, batch_size=args.batch_size_test, shuffle=False, num_workers=1)

    classes = ('0', '1', '2', '3', '4', '5', '6', '7', '8', '9')
    ndata = trainset.__len__()

    print('==> Building model..')

    net = models.__dict__['ChaosDonkey06_AE']( low_dim=args.low_dim, n_hidden = 256,output_dim=1024)

    # define leminiscate
    if args.nce_k > 0:
        lemniscate = NCEAverage(args.low_dim, ndata, args.nce_k, args.nce_t, args.nce_m)
    else:
        lemniscate = LinearAverage(args.low_dim, ndata, args.nce_t, args.nce_m)

    if device == 'cuda':
        net = torch.nn.DataParallel(net, device_ids=range(torch.cuda.device_count()))
        cudnn.benchmark = True

    # Model
    if args.test_only or len(args.resume)>0:
        # Load checkpoint.
        print('==> Resuming from checkpoint..')
        assert os.path.isdir('checkpoint'), 'Error: no checkpoint directory found!'
        checkpoint = torch.load('./checkpoint/'+args.resume)
        net.load_state_dict(checkpoint['net'])
        lemniscate = checkpoint['lemniscate']
        best_acc = checkpoint['acc']
        start_epoch = checkpoint['epoch']
        
    # define loss function
    if hasattr(lemniscate, 'K'):
        criterion = NCECriterion(ndata)
        criterion_AE = nn.MSELoss()

    else:
        criterion = nn.CrossEntropyLoss()
        criterion_AE = nn.MSELoss()

    net.to(device)
    lemniscate.to(device)
    criterion.to(device)
    criterion_AE.to(device)
    

    if args.test_only:
        acc = kNN(0, net, lemniscate, trainloader, testloader, 200, args.nce_t, 1)
        sys.exit(0)

    optimizer = optim.SGD(net.parameters(), lr=args.lr, momentum=0.9, weight_decay=5e-4)

    def adjust_learning_rate(optimizer, epoch):
        """Sets the learning rate to the initial LR decayed by 10 every 30 epochs"""
        lr = args.lr
        if epoch >= 80:
            lr = args.lr * (0.1 ** ((epoch-80) // 40))
        print(lr)
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr

    # Training
    def train(epoch):
        print('\nEpoch: %d' % epoch)
        adjust_learning_rate(optimizer, epoch)

        train_loss = AverageMeter()
        data_time = AverageMeter()
        batch_time = AverageMeter()

        correct = 0
        total = 0

        # switch to train mode
        net.train()

        end = time.time()
        for batch_idx, (inputs, targets, indexes) in enumerate(trainloader):
            data_time.update(time.time() - end)
            inputs, targets, indexes = inputs.to(device), targets.to(device), indexes.to(device)
            optimizer.zero_grad()

            features_lantentspace, decoded_output = net(inputs)
            outputs = lemniscate(features_lantentspace, indexes)

            loss1 = criterion(outputs, indexes)
            flat_inp = inputs.view(-1,inputs.shape[2]*inputs.shape[3]) 
            loss2 = criterion_AE( decoded_output, flat_inp )

            loss = (1-args.recons_weigth)*loss1 + (args.recons_weigth)*loss2

            loss.backward()
            optimizer.step()

            train_loss.update(loss.item(), inputs.size(0))

            
            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()


            if batch_idx%100 == 0:
                print('Epoch: [{}][{}/{}]'
                    'Time: {batch_time.val:.3f} ({batch_time.avg:.3f}) '
                    'Data: {data_time.val:.3f} ({data_time.avg:.3f}) '
                    'Loss: {train_loss.val:.4f} ({train_loss.avg:.4f})'.format(
                    epoch, batch_idx, len(trainloader), batch_time=batch_time, data_time=data_time, train_loss=train_loss))

        print('Epoch: {} | Loss: ({train_loss.avg:.4f})'.format(epoch,train_loss=train_loss))

    file1 = open("./ChaosDonkey06_AEmodel_acc_train.txt","a")  
    file2 = open("./ChaosDonkey06_AEmodel_acc_test.txt","a") 
    for epoch in range(start_epoch, start_epoch+args.epochs):
        train(epoch)
        
        acc = kNN(epoch, net, lemniscate, trainloader, testloader, 200, args.nce_t, 0)

        print('Epoch: {} | Accuracy: ({})'.format(epoch,acc))
        file1.write('{} | {} \n'.format(epoch,acc)) 
        #acc=0
        if acc > best_acc:
            print('Saving..')
            state = {
                'net': net.state_dict(),
                'lemniscate': lemniscate,
                'acc': acc,
                'epoch': epoch,
            }
            if not os.path.isdir('checkpoint'):
                os.mkdir('checkpoint')
            torch.save(state, './checkpoint/ckpt.t7')
            best_acc = acc

        print('best accuracy: {:.2f}'.format(best_acc*100))

    acc = kNN(0, net, lemniscate, trainloader, testloader, 200, args.nce_t, 1)
    file2.write('{} | {} \n'.format(epoch,acc)) 

    print('last accuracy: {:.2f}'.format(acc*100))
