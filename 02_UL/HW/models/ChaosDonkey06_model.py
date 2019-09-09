import torch
import torch.nn as nn
import torch.nn.functional as F
from lib.normalize import Normalize

from torch.autograd import Variable

class ChaosDonkey06_NN(nn.Module):
    def __init__(self, low_dim=128, high_dim=1024):

        self.conv = nn.Conv2d(1, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn = nn.BatchNorm2d(64)
        self.relu = nn.ReLU()
        self.linear = nn.Linear(64*32**2,low_dim)
        self.decoder = nn.Linear(low_dim,high_dim)
        self.l2norm = Normalize(2)

    def forward(self,x):
        out = self.conv(x)
        out = self.bn(out)
        out = self.relu(out)
        out = out.view(out.size(0), -1)
        out = self.linear(out)
        out_low = self.l2norm(out)
        out_high = self.decoder(out_low)
        return out_low, out_high

def ChaosDonkey06_NN(low_dim=128, high_dim=1024):

    return ChaosDonkey06_NN(low_dim=128, high_dim=1024)