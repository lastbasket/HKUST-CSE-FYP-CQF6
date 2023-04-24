import soft_renderer as sr
import soft_renderer.functional as srf
import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class TexFieldVanilla(nn.Module):
    def __init__(self, dim_in=512, im_size=[64, 64], Nd=15, Nf=7774, Nv=3889):
        super(TexFieldVanilla, self).__init__()
        # Nd=15 (palette size) should be ~10-20
        # Nf=1280 (sampling points)
        self.Nf = Nf
        self.Nv = Nv
        self.Nd = Nd
        self.im_size = im_size
        self.fc1 = nn.Linear(dim_in, self.Nv)
        self.conv1 = nn.Conv1d(self.Nv, 64, kernel_size=1)
        self.fc_sampling = nn.Linear(64*4, im_size[0]*im_size[1] * self.Nd)
        self.fc_selection = nn.Linear(64*4, self.Nd * self.Nf)

    def forward(self, x, verts, img):
        # nwe version: concat the feat as the fourth channel and do a conv (element wise mul) to get the new 4 channel (color)
        # x: B, 2048, 1, 1
        # verts: B, 7774, 3
        b, c, h, w = img.shape
        x1 = self.fc1(x.view(x.shape[0],-1)).view(x.shape[0], -1, 1)
        embedding = torch.cat((verts, x1), dim=2).view(verts.shape[0], -1, 4)
        embedding = self.conv1(embedding)
        embedding = embedding.view(embedding.shape[0], -1)

        # positional sampler
        col_sampling = F.softmax(self.fc_sampling(embedding).view(-1, h*w, self.Nd), dim=1)
        col_selection = F.softmax(self.fc_selection(embedding).view(-1, self.Nd, self.Nf), dim=1)

        mat1 = torch.matmul(img.view(-1, 3, h*w), col_sampling)
        textures = torch.matmul(mat1, col_selection).permute(0, 2, 1).contiguous()
        return textures