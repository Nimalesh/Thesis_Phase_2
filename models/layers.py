import torch
import torch.nn as nn
import torch.nn.functional as F

class CBAM(nn.Module):
    def __init__(self, ch):
        super().__init__()
        self.mlp = nn.Sequential(nn.Conv2d(ch, ch//16, 1), nn.ReLU(), nn.Conv2d(ch//16, ch, 1))
        self.sa = nn.Conv2d(2, 1, 7, padding=3)
    def forward(self, x):
        ch_att = torch.sigmoid(self.mlp(F.adaptive_avg_pool2d(x,1)) + self.mlp(F.adaptive_max_pool2d(x,1)))
        x = x * ch_att
        sp_att = torch.sigmoid(self.sa(torch.cat([x.mean(1,keepdim=True), x.max(1,keepdim=True)[0]], 1)))
        return x * sp_att