import torch
from torch import nn
from torch.nn import functional as F
from torch.nn import init
from torchsummary import summary

class MSJL(nn.Module):
    def __init__(self):
        super(MSJL, self).__init__()
        self.dim = 16
        self.kernel_size = 3
        self.key_embed = nn.Sequential(
            nn.Conv2d(self.dim, self.dim, kernel_size=self.kernel_size, padding=self.kernel_size // 2, groups=4,
                      bias=False),
            nn.BatchNorm2d(self.dim),
            nn.ReLU()
        )
        self.value_embed = nn.Sequential(
            nn.Conv2d(self.dim, self.dim, 1, bias=False),
            nn.BatchNorm2d(self.dim)
        )
        factor = 4
        self.attention_embed = nn.Sequential(
            nn.Conv2d(2 * self.dim, 2 * self.dim // factor, 1, bias=False),
            nn.BatchNorm2d(2 * self.dim // factor),
            nn.ReLU(),
            nn.Conv2d(2 * self.dim // factor, self.kernel_size * self.kernel_size * self.dim, 1)
        )
        self.channel_attention = self.ChannelAttention(input_channels=self.dim, internal_neurons=self.dim // 4)
        self.initial_depth_conv = nn.Conv2d(self.dim, self.dim, kernel_size=5, padding=2, groups=self.dim)
        self.depth_convs = nn.ModuleList([
            nn.Conv2d(self.dim, self.dim, kernel_size=(1, 7), padding=(0, 3), groups=self.dim),
            nn.Conv2d(self.dim, self.dim, kernel_size=(7, 1), padding=(3, 0), groups=self.dim),
            nn.Conv2d(self.dim, self.dim, kernel_size=(1, 11), padding=(0, 5), groups=self.dim),
            nn.Conv2d(self.dim, self.dim, kernel_size=(11, 1), padding=(5, 0), groups=self.dim),
            nn.Conv2d(self.dim, self.dim, kernel_size=(1, 21), padding=(0, 10), groups=self.dim),
            nn.Conv2d(self.dim, self.dim, kernel_size=(21, 1), padding=(10, 0), groups=self.dim),
        ])
        self.pointwise_conv = nn.Conv2d(self.dim, self.dim, kernel_size=1, padding=0)
        self.act = nn.GELU()
        self.gap = nn.AdaptiveAvgPool2d(1)
        self.conv = nn.Conv1d(1, 1, kernel_size=self.kernel_size, padding=(self.kernel_size - 1) // 2)
        self.sigmoid = nn.Sigmoid()
    def ChannelAttentionModule(self, input_channels, internal_neurons):
        class ChannelAttentionModule(nn.Module):
            def __init__(self, input_channels, internal_neurons):
                super(ChannelAttentionModule, self).__init__()
                self.fc1 = nn.Conv2d(in_channels=input_channels, out_channels=internal_neurons, kernel_size=1, stride=1,
                                     bias=True)
                self.fc2 = nn.Conv2d(in_channels=internal_neurons, out_channels=input_channels, kernel_size=1, stride=1,
                                     bias=True)
                self.input_channels = input_channels
            def forward(self, inputs):
                avg_pool = F.adaptive_avg_pool2d(inputs, output_size=(1, 1))
                max_pool = F.adaptive_max_pool2d(inputs, output_size=(1, 1))
                median_pool = self.global_median_pooling(inputs)
                avg_out = self.fc1(avg_pool)
                avg_out = F.relu(avg_out, inplace=True)
                avg_out = self.fc2(avg_out)
                avg_out = torch.sigmoid(avg_out)
                max_out = self.fc1(max_pool)
                max_out = F.relu(max_out, inplace=True)
                max_out = self.fc2(max_out)
                max_out = torch.sigmoid(max_out)
                median_out = self.fc1(median_pool)
                median_out = F.relu(median_out, inplace=True)
                median_out = self.fc2(median_out)
                median_out = torch.sigmoid(median_out)
                out = avg_out + max_out + median_out
                return out
        return ChannelAttentionModule(input_channels, internal_neurons)
    def global_median_pooling(self, x):
        median_pooled = torch.median(x.view(x.size(0), x.size(1), -1), dim=2)[0]
        median_pooled = median_pooled.view(x.size(0), x.size(1), 1, 1)
        return median_pooled
    def forward(self, x):
        bs, c, h, w = x.shape
        k1 = self.key_embed(x)  # label
        v = self.value_embed(x).view(bs, c, -1)
        y = torch.cat([k1, x], dim=1)
        att = self.attention_embed(y)
        att = att.reshape(bs, c, self.kernel_size * self.kernel_size, h, w)
        att = att.mean(2, keepdim=False).view(bs, c, -1)
        k2 = F.softmax(att, dim=-1) * v  # image
        k2 = k2.view(bs, c, h, w)
        x1 = k1 + k2
        x2 = self.mern(x1)
        x3 = x1 + x2
        y = self.gap(x3)
        y = y.squeeze(-1).permute(0, 2, 1)
        y = self.conv(y)
        y = self.sigmoid(y)
        y = y.permute(0, 2, 1).unsqueeze(-1)
        x4 = x3 * y.expand_as(x3)

        return x4

