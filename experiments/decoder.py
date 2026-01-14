import torch
import torch.nn as nn
import torch.nn.functional as F

def conv_block(in_ch, out_ch):
    return nn.Sequential(
        nn.Conv2d(in_ch, out_ch, 3, padding=1),
        nn.BatchNorm2d(out_ch),
        nn.ReLU(inplace=True)
    )

class UNetDecoder(nn.Module):
    def __init__(self, ch):
        super().__init__()
        # ch: [s0, s1, s2, s3, s4] -> [1/2, 1/4, 1/8, 1/16, 1/32]
        self.up4 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.c4 = conv_block(ch[4] + ch[3], ch[3])
        self.up3 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.c3 = conv_block(ch[3] + ch[2], ch[2])
        self.up2 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.c2 = conv_block(ch[2] + ch[1], ch[1])
        self.up1 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.c1 = conv_block(ch[1] + ch[0], ch[0])

    def forward(self, stages):
        s0, s1, s2, s3, s4 = stages
        d4 = self.c4(torch.cat([self.up4(s4), s3], 1))
        d3 = self.c3(torch.cat([self.up3(d4), s2], 1))
        d2 = self.c2(torch.cat([self.up2(d3), s1], 1))
        d1 = self.c1(torch.cat([self.up1(d2), s0], 1))
        return d1

class UNetPlusPlusDecoder(nn.Module):
    def __init__(self, ch):
        super().__init__()
        self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.c01 = conv_block(ch[0] + ch[1], ch[0])
        self.c11 = conv_block(ch[1] + ch[2], ch[1])
        self.c21 = conv_block(ch[2] + ch[3], ch[2])
        self.c31 = conv_block(ch[3] + ch[4], ch[3])

        self.c02 = conv_block(ch[0]*2 + ch[1], ch[0])
        self.c12 = conv_block(ch[1]*2 + ch[2], ch[1])
        self.c22 = conv_block(ch[2]*2 + ch[3], ch[2])

        self.c03 = conv_block(ch[0]*3 + ch[1], ch[0])
        self.c13 = conv_block(ch[1]*3 + ch[2], ch[1])

        self.c04 = conv_block(ch[0]*4 + ch[1], ch[0])

    def forward(self, stages):
        s0, s1, s2, s3, s4 = stages
        x01 = self.c01(torch.cat([s0, self.up(s1)], 1))
        x11 = self.c11(torch.cat([s1, self.up(s2)], 1))
        x21 = self.c21(torch.cat([s2, self.up(s3)], 1))
        x31 = self.c31(torch.cat([s3, self.up(s4)], 1))

        x02 = self.c02(torch.cat([s0, x01, self.up(x11)], 1))
        x12 = self.c12(torch.cat([s1, x11, self.up(x21)], 1))
        x22 = self.c22(torch.cat([s2, x21, self.up(x31)], 1))

        x03 = self.c03(torch.cat([s0, x01, x02, self.up(x12)], 1))
        x13 = self.c13(torch.cat([s1, x11, x12, self.up(x22)], 1))

        x04 = self.c04(torch.cat([s0, x01, x02, x03, self.up(x13)], 1))
        return x04

class DeepLabV3PlusDecoder(nn.Module):
    def __init__(self, ch):
        super().__init__()
        self.aspp = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(ch[4], 256, 1), nn.ReLU()
        )
        self.low_level = conv_block(ch[1], 48)
        self.final = nn.Sequential(
            conv_block(256 + 48, 256),
            conv_block(256, 256)
        )

    def forward(self, stages):
        s0, s1, s2, s3, s4 = stages
        aspp_out = F.interpolate(self.aspp(s4), size=s1.shape[2:], mode='bilinear', align_corners=True)
        low_out = self.low_level(s1)
        out = self.final(torch.cat([aspp_out, low_out], 1))
        return F.interpolate(out, size=s0.shape[2:], mode='bilinear', align_corners=True)