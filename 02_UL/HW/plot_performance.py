import numpy as np
import matplotlib.pyplot as plt

file_name = 'resnet18_acc_train.txt'
fileObj=open(file_name,'r')

performance = fileObj.readlines()
acc_resnet = []
for i, line in enumerate(performance):
    accu = line.split(' | ')[-1]
    acc_resnet.append(float(accu[0:6]))

file_name = 'ChaosDonkey06_AEmodel_acc_train.txt'
fileObj=open(file_name,'r')

performance = fileObj.readlines()
acc_ChaosDonkey = []
for i, line in enumerate(performance):
    accu = line.split(' | ')[-1]
    accu.replace('/n','')
    acc_ChaosDonkey.append(float(accu[0:6]))


plt.plot(range(1,21),acc_resnet,'r',range(1,21),acc_ChaosDonkey[0:20],'k'), plt.xlabel('# Epoch'), plt.ylabel('Accuracy')
plt.legend(['Resnet18','Auto Encoder Net'])


plt.show()


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

parser = argparse.ArgumentParser(description='PyTorch MNIST Training')
parser.add_argument('--lr', default=0.03, type=float, help='learning rate')
parser.add_argument('--resume', '-r', default='', type=str, help='resume from checkpoint')
parser.add_argument('--test-only', action='store_true', help='test only')
parser.add_argument('--low-dim', default=128, type=int,
                    metavar='D', help='feature dimension')

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


trainset = datasets.MNISTInstance(root='./data', train=True, download=True, transform=transform_train)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=args.batch_size, shuffle=True, num_workers=1)

testset = datasets.MNISTInstance(root='./data', train=False, download=True, transform=transform_train)
testloader = torch.utils.data.DataLoader(testset, batch_size=args.batch_size_test, shuffle=False, num_workers=1)

batch_1,label, idx = next(iter(trainloader))                                                                                                     

net = models.__dict__['ChaosDonkey06_AE']( low_dim=args.low_dim, n_hidden = 256,output_dim=1024)
idx = 7
im = batch_1[idx,:,:,:].unsqueeze(0)
label_im = label[idx].numpy()
features_lantentspace, decoded_output = net(im)

import numpy as np
reconstructed_im = decoded_output.detach().numpy().reshape(32,32)
plt.imshow(reconstructed_im), plt.title('Label: '+str(label_im)), plt.title('Reconstructed image')
plt.axis('off')
plt.show()

plt.imshow(np.squeeze(im.numpy())), plt.title('Label: '+str(label_im)), plt.title('Input image')
plt.axis('off')
plt.show()