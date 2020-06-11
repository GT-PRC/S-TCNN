import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from coordConv import addCoords, addCoords_1D
import gpytorch
import math

cwd = os.getcwd()
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


class Solenoid_STCNN(nn.Module):
    def __init__(self):
        super(Solenoid_STCNN, self).__init__()

        ##Fully Connected Layers
        self.add_coords = addCoords_1D()
        self.lin1 = nn.Linear(8, 25)
        self.lin2 = nn.Linear(25, 25)
        self.lin3 = nn.Linear(25, 25)
        self.lin4 = nn.Linear(25, 25)
        self.norm_fc = nn.BatchNorm1d(25)
        ##DECODER
        self.tconv1 = nn.ConvTranspose1d(25, 10, 31, 1, 0)
        self.norm_dec1 = nn.BatchNorm1d(10)

        self.tconv2 = nn.ConvTranspose1d(10, 10, 31, 2, 0)
        self.norm_dec2 = nn.BatchNorm1d(10)

        self.tconv3 = nn.ConvTranspose1d(10, 10, 16, 2,  0)
        self.norm_dec3 = nn.BatchNorm1d(10)

        self.tconv4 = nn.ConvTranspose1d(11, 2, 21, 1, 10)

    def fully_connected(self, x):
        latent = self.lin1(x)
        latent = latent.tanh()

        latent = self.lin2(latent)
        latent = latent.tanh()

        latent = self.lin3(latent)
        latent = latent.tanh()

        latent = self.lin4(latent)
        z = (self.norm_fc(latent.view(-1, self.lin4.out_features, 1))).tanh()

        return z

    def transposed_conv(self, z):
        latent = self.tconv1(z)
        latent = latent.tanh()
        latent = self.norm_dec1(latent)

        latent = self.tconv2(latent)
        latent = latent.tanh()
        latent = self.norm_dec2(latent)

        latent = self.tconv3(latent)
        latent = latent.tanh()
        latent = self.norm_dec3(latent)

        latent = self.add_coords(latent)
        recons_y = self.tconv4(latent)
        return recons_y

    def forward(self, x):
        z = self.fully_connected(x)
        out = self.transposed_conv(z)
        return out



class Solenoid_STCNN_V2(nn.Module):
    def __init__(self):
        super(Solenoid_STCNN_V2, self).__init__()

        ##Fully Connected Layers
        self.add_coords = addCoords_1D()
        self.lin1 = nn.Linear(8, 30)
        self.lin2 = nn.Linear(30, 30)
        self.lin3 = nn.Linear(30, 30)
        self.lin4 = nn.Linear(30, 30)

        ##Transposed Convolution Layers
        self.tconv1 = nn.ConvTranspose1d(30, 25, 31, 1, 0)
        self.norm_dec1 = nn.BatchNorm1d(25)

        self.tconv2 = nn.ConvTranspose1d(25, 25, 4, 2, 1)
        self.norm_dec2 = nn.BatchNorm1d(25)

        self.tconv3 = nn.ConvTranspose1d(25, 25, 4, 2,  1)
        self.norm_dec3 = nn.BatchNorm1d(25)

        self.tconv4 = nn.ConvTranspose1d(25, 25, 4, 2, 1)
        self.norm_dec4 = nn.BatchNorm1d(25)

        self.tconv5 = nn.ConvTranspose1d(25, 2, 4, 1, 0)

    def fully_connected(self, x):
        latent = self.lin1(x)
        latent = F.elu(latent)

        latent = self.lin2(latent)
        latent = F.elu(latent)

        latent = self.lin3(latent)
        latent = F.elu(latent)

        latent = self.lin4(latent)
        latent = F.elu(latent)
        z = latent.view(-1, self.lin4.out_features, 1)
        return z

    def transposed_conv(self, z):
        latent = self.tconv1(z)
        latent = F.elu(latent)

        latent = self.tconv2(latent)
        latent = F.elu(latent)

        latent = self.tconv3(latent)
        latent = F.elu(latent)

        latent = self.tconv4(latent)
        latent = F.elu(latent)

        recons_y = self.tconv5(latent)
        return recons_y[..., :196]

    def forward(self, x):
        z = self.fully_connected(x)
        out = self.transposed_conv(z)
        return out
