import torch
import torch.nn as nn
from torchsummary import summary

class GCAC(nn.Module):
    def __init__(self):
        super(GCAC, self).__init__()
        self.dim = 16
        self.rate = 4
        self.channel_attention = nn.Sequential(
            nn.Linear(self.dim, int(self.dim / self.rate)),
            nn.ReLU(inplace=True),
            nn.Linear(int(self.dim / self.rate), self.dim)
        )
        self.spatial_attention = nn.Sequential(
            nn.Conv2d(self.dim, int(self.dim / self.rate), kernel_size=7, padding=3),
            nn.BatchNorm2d(int(self.dim / self.rate)),
            nn.ReLU(inplace=True),
            nn.Conv2d(int(self.dim / self.rate), self.dim, kernel_size=7, padding=3),
            nn.BatchNorm2d(self.dim)
        )
        self.cS = nn.ModuleList([
            nn.AdaptiveAvgPool2d(1),
            nn.ReLU(),
            nn.Conv2d(self.dim, self.dim // 2, kernel_size=1, bias=False),
            nn.Sigmoid(),
            nn.Conv2d(self.dim // 2, self.dim, kernel_size=1, bias=False)
        ])
        self.sS = nn.ModuleList([
            nn.Conv2d(self.dim, 1, kernel_size=1, bias=False),
            nn.Sigmoid()
        ])
    def channel_shuffle(self, x, groups):
        batchsize, num_channels, height, width = x.size()
        channels_per_group = num_channels // groups
        x = x.view(batchsize, groups, channels_per_group, height, width)
        x = torch.transpose(x, 1, 2).contiguous()
        x = x.view(batchsize, -1, height, width)
        return x
    def forward(self, x):
        b, c, h, w = x.shape
        x_permute = x.permute(0, 2, 3, 1).view(b, -1, c)
        x_att_permute = self.channel_attention(x_permute).view(b, h, w, c)
        x_channel_att = x_att_permute.permute(0, 3, 1, 2).sigmoid()
        x = x * x_channel_att
        x = self.channel_shuffle(x, groups=4)
        x_spatial_att = self.spatial_attention(x).sigmoid()
        x = x * x_spatial_att
        z = self.cS[0](x)
        z = self.cS[2](z)
        z = self.cS[1](z)
        z = self.cS[3](z)
        z = self.cS[4](z)
        c_out = x * z.expand_as(x)
        q = self.sS[0](x)
        q = self.sS[1](q)
        s_out = x * q
        out = c_out + s_out
        return out


