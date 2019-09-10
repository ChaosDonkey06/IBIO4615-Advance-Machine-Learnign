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
        self.l2norm = Normalize(2)

        self.encoder = nn.Sequential(
                            nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False),
                            nn.BatchNorm2d(64),
                            nn.ReLU()
                        )
        self.linear = nn.Linear(64*(16**2),low_dim)

        self.decoder = nn.Sequential(
                            nn.Linear(self.low_dim, self.n_hidden),
                            nn.ReLU(),
                            nn.Linear(self.n_hidden, self.output_dim),
                            nn.Sigmoid()
                        )
    
    def forward(self, x):
        x_latent =  self.encoder(x)
        x_latent = x_latent.view(x_latent.size(0), -1)
        x_latent = self.linear(x_latent)   
        x_latent = self.l2norm(x_latent)
        x_latent = x_latent.view(x_latent.size(0), -1)
        x = self.decoder(x_latent)
        x = self.l2norm(x)
        return x_latent, x