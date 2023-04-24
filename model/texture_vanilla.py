import soft_renderer as sr
import soft_renderer.functional as srf
import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class Encoder(nn.Module):
    def __init__(self, dim_in=4, dim_out=512, dim1=64, dim2=1024, im_size=64):
        super(Encoder, self).__init__()
        dim_hidden = [dim1, dim1*2, dim1*4, dim2, dim2]

        self.conv1 = nn.Conv2d(dim_in, dim_hidden[0], kernel_size=5, stride=2, padding=2)
        self.conv2 = nn.Conv2d(dim_hidden[0], dim_hidden[1], kernel_size=5, stride=2, padding=2)
        self.conv3 = nn.Conv2d(dim_hidden[1], dim_hidden[2], kernel_size=5, stride=2, padding=2)

        self.bn1 = nn.BatchNorm2d(dim_hidden[0])
        self.bn2 = nn.BatchNorm2d(dim_hidden[1])
        self.bn3 = nn.BatchNorm2d(dim_hidden[2])

        self.fc1 = nn.Linear(dim_hidden[2]*math.ceil(im_size/8)**2, dim_hidden[3])
        self.fc2 = nn.Linear(dim_hidden[3], dim_hidden[4])
        self.fc3 = nn.Linear(dim_hidden[4], dim_out)

    def forward(self, x):
        x = F.relu(self.bn1(self.conv1(x)), inplace=True)
        x = F.relu(self.bn2(self.conv2(x)), inplace=True)
        x = F.relu(self.bn3(self.conv3(x)), inplace=True)
        
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x), inplace=True)
        x = F.relu(self.fc2(x), inplace=True)
        x = F.relu(self.fc3(x), inplace=True)
        return x

class ColorGen(nn.Module):
    def __init__(self, dim_in=512, im_size=[64, 64], Nd=15, Nc=1280):
        super(ColorGen, self).__init__()
        # Nd=15 (palette size) should be ~10-20
        # Nc=1280 (sampling points)
        self.Nc = Nc
        self.Nd = Nd
        self.fc1 = nn.Linear(dim_in, 1024)
        self.fc_sampling = nn.Linear(1024, im_size[0]*im_size[1] * self.Nd)
        self.fc_selection = nn.Linear(1024, self.Nd * self.Nc)

    def forward(self, x):
        # print(x.shape)
        x = x.view(x.shape[0], -1)
        # print(x.shape)

        x = F.relu(self.fc1(x), inplace=True)
        # print(x.shape)

        col_selection = self.fc_selection(x)
        col_sampling = self.fc_sampling(x)
        # print(col_sampling.shape)
        return col_sampling, col_selection