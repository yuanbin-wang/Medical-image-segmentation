import torch
from torch import nn
import torch.nn.functional as F
from torchsummary import summary

class tdao(nn.Module):
    def __init__(self, in_channel):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Conv2d(in_channel, 1, kernel_size=1, bias=False)
        self.relu = nn.ReLU(inplace=True)
    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x)
        y = self.fc(y)
        y = self.relu(y)
        y = nn.functional.interpolate(y, size=(x.size(2), x.size(3)), mode='nearest')
        return x * y.expand_as(x)

class kjian(nn.Module):
    def __init__(self, in_channel):
        super().__init__()
        self.Conv1x1 = nn.Conv2d(in_channel, 1, kernel_size=1, bias=False)
        self.norm = nn.Sigmoid()
    def forward(self, x):
        y = self.Conv1x1(x)
        y = self.norm(y)
        return x * y

class hb(nn.Module):
    def __init__(self, in_channel):
        super().__init__()
        self.td = tdao(in_channel)
        self.kj = kjian(in_channel)
    def forward(self, U):
        U_kj = self.kj(U)
        U_td = self.tdao(U)
        return torch.max(U_td, U_kj)


class MSAS_FC(nn.Module):
    def __init__(self, dim_in, dim_out, rate=1, bn_mom=0.1):
        super(MSAS_FC, self).__init__()
        self.branch1 = nn.Sequential(
            nn.Conv2d(dim_in, dim_out, 1, 1, padding=0, dilation=rate, bias=True),
            nn.BatchNorm2d(dim_out, momentum=bn_mom),
            nn.ReLU(inplace=True),
        )
        self.branch2 = nn.Sequential(
            nn.Conv2d(dim_in, dim_out, 3, 1, padding=6 * rate, dilation=6 * rate, bias=True),
            nn.BatchNorm2d(dim_out, momentum=bn_mom),
            nn.ReLU(inplace=True),
        )
        self.branch3 = nn.Sequential(
            nn.Conv2d(dim_in, dim_out, 3, 1, padding=12 * rate, dilation=12 * rate, bias=True),
            nn.BatchNorm2d(dim_out, momentum=bn_mom),
            nn.ReLU(inplace=True),
        )
        self.branch4 = nn.Sequential(
            nn.Conv2d(dim_in, dim_out, 3, 1, padding=18 * rate, dilation=18 * rate, bias=True),
            nn.BatchNorm2d(dim_out, momentum=bn_mom),
            nn.ReLU(inplace=True),
        )
        self.branch5_conv = nn.Conv2d(dim_in, dim_out, 1, 1, 0, bias=True)
        self.branch5_bn = nn.BatchNorm2d(dim_out, momentum=bn_mom)
        self.branch5_relu = nn.ReLU(inplace=True)

        self.conv_cat = nn.Sequential(
            nn.Conv2d(dim_out * 5, dim_out, 1, 1, padding=0, bias=True),
            nn.BatchNorm2d(dim_out, momentum=bn_mom),
            nn.ReLU(inplace=True),
        )
        self.Hebing=hb(in_channel=dim_out*5)
    def forward(self, x):
        [b, c, row, col] = x.size()
        conv1x1 = self.branch1(x)
        conv3x3_1 = self.branch2(x)
        conv3x3_2 = self.branch3(x)
        conv3x3_3 = self.branch4(x)
        gf = torch.mean(x, 2, True)
        gf = torch.mean(gf, 3, True)
        gf = self.branch5_conv(gf)
        gf = self.branch5_bn(gf)
        gf = self.branch5_relu(gf)
        gf = F.interpolate(gf, (row, col), None, 'bilinear', True)
        f_cat = torch.cat([conv1x1, conv3x3_1, conv3x3_2, conv3x3_3, gf], dim=1)
        wyb=self.Hebing(f_cat)
        wyb_feature_cat=wyb*f_cat
        result = self.conv_cat(wyb_feature_cat)
        return result

