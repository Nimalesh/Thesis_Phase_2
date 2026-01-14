import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import efficientnet_b6
from .layers import CBAM
import numpy as np

class MultiTaskLatentAugModel(nn.Module):
    def __init__(self, num_classes=3, latent_sigma=0.05):
        super().__init__()
        self.latent_sigma = latent_sigma
        self.enc = efficientnet_b6(weights=None)
        
       
        dummy = torch.randn(1, 3, 256, 256)
        f = self.enc.features
        x0=f[0](dummy); x1=f[2](f[1](x0)); x2=f[3](x1); x3=f[5](f[4](x2)); x4=f[8](f[7](f[6](x3)))
        ch = [x0.shape[1], x1.shape[1], x2.shape[1], x3.shape[1], x4.shape[1]]
        
        self.cbam = CBAM(ch[-1])
        
        # Decoder (UNet++)
        def conv(i, o): return nn.Sequential(nn.Conv2d(i,o,3,padding=1), nn.BatchNorm2d(o), nn.ReLU(True))
        self.u10 = nn.ConvTranspose2d(ch[1], ch[0], 2, 2); self.c01 = conv(ch[0]*2, ch[0])
        self.u21 = nn.ConvTranspose2d(ch[2], ch[1], 2, 2); self.c11 = conv(ch[1]*2, ch[1])
        self.u32 = nn.ConvTranspose2d(ch[3], ch[2], 2, 2); self.c21 = conv(ch[2]*2, ch[2])
        self.u43 = nn.ConvTranspose2d(ch[4], ch[3], 2, 2); self.c31 = conv(ch[3]*2, ch[3])
        
        self.u11 = nn.ConvTranspose2d(ch[1], ch[0], 2, 2); self.c02 = conv(ch[0]*3, ch[0])
        self.u12 = nn.ConvTranspose2d(ch[1], ch[0], 2, 2); self.c03 = conv(ch[0]*4, ch[0])
        self.u13 = nn.ConvTranspose2d(ch[1], ch[0], 2, 2); self.c04 = conv(ch[0]*5, ch[0])
        
        # We need intermediate upsamplers for full UNet++ logic
        self.u22 = nn.ConvTranspose2d(ch[2], ch[1], 2, 2); self.c12 = conv(ch[1]*3, ch[1])
        self.u33 = nn.ConvTranspose2d(ch[3], ch[2], 2, 2); self.c22 = conv(ch[2]*3, ch[2])
        self.u23 = nn.ConvTranspose2d(ch[2], ch[1], 2, 2); self.c13 = conv(ch[1]*4, ch[1])

        self.final = nn.Conv2d(ch[0], 1, 1)
        self.head = nn.Linear(ch[-1], num_classes)

    def apply_latent_augmentation(self, features, labels):
        if labels is None: return features
        aug_features = features.clone()
        minority_mask = (labels == 0) | (labels == 2)
        if minority_mask.any():
            noise = torch.randn_like(aug_features) * self.latent_sigma
            mask_expanded = minority_mask.view(-1, 1, 1, 1).float()
            aug_features = aug_features + (noise * mask_expanded)
        return aug_features

    def forward(self, x, labels=None):
        f = self.enc.features
        x0=f[0](x); x1=f[2](f[1](x0)); x2=f[3](x1); x3=f[5](f[4](x2)); x4=f[8](f[7](f[6](x3)))
        x4 = self.cbam(x4)
        
        if self.training and labels is not None:
            x4 = self.apply_latent_augmentation(x4, labels)
        
        def up(s, t): return F.interpolate(s, t.shape[2:], mode='bilinear', align_corners=True)
        
        x01 = self.c01(torch.cat([x0, up(self.u10(x1), x0)],1))
        x11 = self.c11(torch.cat([x1, up(self.u21(x2), x1)],1))
        x21 = self.c21(torch.cat([x2, up(self.u32(x3), x2)],1))
        x31 = self.c31(torch.cat([x3, up(self.u43(x4), x3)],1))
        
        x02 = self.c02(torch.cat([x0, x01, up(self.u11(x11), x0)],1))
        x12 = self.c12(torch.cat([x1, x11, up(self.u22(x21), x1)],1))
        x22 = self.c22(torch.cat([x2, x21, up(self.u33(x31), x2)],1))
        
        x03 = self.c03(torch.cat([x0, x01, x02, up(self.u12(x12), x0)],1))
        x13 = self.c13(torch.cat([x1, x11, x12, up(self.u23(x22), x1)],1))
        
        x04 = self.c04(torch.cat([x0, x01, x02, x03, up(self.u13(x13), x0)],1))
        
        seg = F.interpolate(self.final(x04), (x.shape[2], x.shape[3]), mode='bilinear')
        cls = self.head(F.adaptive_avg_pool2d(x4,1).flatten(1))
        return seg, cls