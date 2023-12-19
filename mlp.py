import os
import glob
import matplotlib.pyplot as plt
import numpy as np
import pickle
import torch.nn.functional as F
import torch
import torch.nn as nn

# MLP part from the generation module
class NeRF(nn.Module):
    def __init__(self, D=8, W=128, input_ch=42, input_ch_views=3, output_ch=1, skips=[4]):
        """ 
        """
        super(NeRF, self).__init__()
        self.D = D
        self.W = W
        self.input_ch = input_ch
        # self.input_ch_views = input_ch_views
        self.skips = skips
        self.first_pts_linear = nn.Linear(input_ch, W)
        self.image_linear = nn.Linear(128, W) # 128
        self.pts_linears = nn.ModuleList(
            [nn.Linear(W, W) if i not in self.skips else nn.Linear(W*2, W) for i in range(D-1)])
        self.output_linear = nn.Linear(W, output_ch)

    def forward(self, x, c):
        # input_pts, input_views = torch.split(x, [self.input_ch, self.input_ch_views], dim=-1)
        x = self.first_pts_linear(x)
        h = x + self.image_linear(c)
        for i, l in enumerate(self.pts_linears):
            # print(i, h.shape)
            h = self.pts_linears[i](h)
            h = F.relu(h)
            if i+1 in self.skips:
                h = torch.cat([x + self.image_linear(c), h], -1)

        outputs = self.output_linear(h)

        return torch.sigmoid(outputs)