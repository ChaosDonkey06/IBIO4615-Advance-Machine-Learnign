import torch.nn as nn
import math
import torch.utils.model_zoo as model_zoo
from lib.normalize import Normalize


class ChaosDonkey06_AE(nn.Module):
    def __init__(self, low_dim=128, n_hidden = 256,output_dim=1024):
        super(ChaosDonkey06_AE, self).__init__()

        self.low_dim = low_dim
        self.n_hidden = n_hidden
        self.output_dim = output_dim

        self.encoder = nn.Sequential(
                            nn.Conv2d(1, 64, kernel_size=3, stride=1, padding=1, bias=False),
                            nn.BatchNorm2d(64),
                            nn.ReLU(),
                            nn.Linear(64*32**2,self.low_dim)
                        )

        self.decoder = nn.Sequential(
                            nn.Linear(self.low_dim, self.n_hidden),
                            nn.ReLU(),
                            nn.Linear(self.n_hidden, self.output_dim),
                            nn.Sigmoid()
                        )
    
    def forward(self, x):
        x_latent =  self.encoder(x)
        x = self.encoder(x)
        return x_latent, x