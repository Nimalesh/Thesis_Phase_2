import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
from .layers import CBAM

# --- FINAL CORRECTED ENCODER CHANNEL MAP ---
ENCODER_CHANNELS = {
    "resnet18": [64, 64, 128, 256, 512],
    "resnet50": [64, 256, 512, 1024, 2048],
    "resnet101": [64, 256, 512, 1024, 2048],
    "efficientnet_b0": [16, 24, 40, 112, 320],
    "efficientnet_b1": [16, 24, 40, 112, 320],
    "efficientnet_b2": [16, 24, 48, 120, 352],
    "efficientnet_b3": [24, 32, 48, 136, 384],
    "efficientnet_b4": [24, 32, 56, 160, 448],
    "efficientnet_b5": [24, 40, 64, 176, 512],
    "efficientnet_b6": [32, 56, 72, 160, 576],
    "mobilenet_v2": [16, 24, 32, 96, 320],
    "mobilenet_v3_small": [16, 16, 24, 48, 576],
    "mobilenet_v3_large": [16, 24, 40, 112, 960],
    "shufflenet_v2_x0_5": [24, 24, 48, 96, 192],
    "shufflenet_v2_x1_0": [24, 24, 116, 232, 464],
    "shufflenet_v2_x1_5": [24, 24, 176, 352, 704],
    "shufflenet_v2_x2_0": [24, 24, 244, 488, 976],
    "googlenet": [64, 192, 480, 832, 1024],
    "inception_v3": [64, 192, 288, 768, 2048]
}

class UNetBlock(nn.Module):
    def __init__(self, in_ch, skip_ch, out_ch):
        super().__init__()
        self.up = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
            nn.Conv2d(in_ch, out_ch, 3, padding=1)
        )
        self.conv = nn.Sequential(
            nn.Conv2d(out_ch + skip_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True)
        )

    def forward(self, x_up, x_skip):
        x = self.up(x_up)
        if x.shape[2:] != x_skip.shape[2:]:
            diffY = x_skip.size()[2] - x.size()[2]
            diffX = x_skip.size()[3] - x.size()[3]
            x = F.pad(x, [diffX // 2, diffX - diffX // 2, diffY // 2, diffY - diffY // 2])
        return self.conv(torch.cat([x, x_skip], dim=1))

class GenericMultiTaskModel(nn.Module):
    def __init__(self, encoder_name, num_classes=3):
        super().__init__()
        self.encoder_name = encoder_name
        ch = ENCODER_CHANNELS[encoder_name]
        
        # Load weights safely to avoid Hash Errors crashing the whole script
        try:
            base = getattr(models, encoder_name)(weights='DEFAULT')
        except:
            print(f"⚠️ Warning: Could not download weights for {encoder_name}. Using random initialization.")
            base = getattr(models, encoder_name)(weights=None)
        
        # 1. ENCODER SLICING
        if "resnet" in encoder_name:
            self.s0, self.s1, self.s2, self.s3, self.s4 = nn.Sequential(base.conv1, base.bn1, base.relu), nn.Sequential(base.maxpool, base.layer1), base.layer2, base.layer3, base.layer4

        elif "efficientnet" in encoder_name:
            f = base.features
            self.s0, self.s1, self.s2, self.s3, self.s4 = f[0:2], f[2:3], f[3:4], f[4:6], f[6:8]

        elif "mobilenet_v2" in encoder_name:
            f = base.features
            self.s0, self.s1, self.s2, self.s3, self.s4 = f[0:2], f[2:4], f[4:7], f[7:14], f[14:18]

        elif "mobilenet_v3_small" in encoder_name:
            f = base.features
            # Adjusted to include the final expansion layer (f[9:12]) for 576 channels
            self.s0, self.s1, self.s2, self.s3, self.s4 = f[0:1], f[1:2], f[2:4], f[4:9], f[9:]

        elif "mobilenet_v3_large" in encoder_name:
            f = base.features
            # Adjusted to include the final expansion layer (f[13:]) for 960 channels
            self.s0, self.s1, self.s2, self.s3, self.s4 = f[0:2], f[2:4], f[4:7], f[7:13], f[13:]

        elif "shufflenet" in encoder_name:
            self.s0, self.s1, self.s2, self.s3, self.s4 = base.conv1, base.maxpool, base.stage2, base.stage3, base.stage4

        elif "googlenet" in encoder_name:
            self.s0, self.s1 = base.conv1, nn.Sequential(base.maxpool1, base.conv2, base.conv3)
            self.s2, self.s3 = nn.Sequential(base.maxpool2, base.inception3a, base.inception3b), nn.Sequential(base.maxpool3, base.inception4a, base.inception4b, base.inception4c, base.inception4d, base.inception4e)
            self.s4 = nn.Sequential(base.maxpool4, base.inception5a, base.inception5b)

        elif "inception_v3" in encoder_name:
            self.s0, self.s1 = nn.Sequential(base.Conv2d_1a_3x3, base.Conv2d_2a_3x3, base.Conv2d_2b_3x3), nn.Sequential(base.maxpool1, base.Conv2d_3b_1x1, base.Conv2d_4a_3x3)
            self.s2, self.s3 = nn.Sequential(base.maxpool2, base.Mixed_5b, base.Mixed_5c, base.Mixed_5d), nn.Sequential(base.Mixed_6a, base.Mixed_6b, base.Mixed_6c, base.Mixed_6d, base.Mixed_6e)
            self.s4 = nn.Sequential(base.Mixed_7a, base.Mixed_7b, base.Mixed_7c)

        self.cbam = CBAM(ch[4])

        # 2. DECODER
        self.L3 = UNetBlock(ch[4], ch[3], ch[3])
        self.L2 = UNetBlock(ch[3], ch[2], ch[2])
        self.L1 = UNetBlock(ch[2], ch[1], ch[1])
        self.L0 = UNetBlock(ch[1], ch[0], ch[0])
        
        self.final_up = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
            nn.Conv2d(ch[0], 32, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 1, 1)
        )
        self.head = nn.Linear(ch[4], num_classes)

    def forward(self, x):
        if "inception_v3" in self.encoder_name:
            x = F.interpolate(x, size=(299, 299), mode='bilinear', align_corners=True)

        x0, x1, x2, x3, x4 = self.s0(x), self.s1(self.s0(x)), self.s2(self.s1(self.s0(x))), self.s3(self.s2(self.s1(self.s0(x)))), self.s4(self.s3(self.s2(self.s1(self.s0(x)))))
        # Refined forward to ensure sequential flow and avoid re-calculating
        feat0 = self.s0(x)
        feat1 = self.s1(feat0)
        feat2 = self.s2(feat1)
        feat3 = self.s3(feat2)
        feat4 = self.s4(feat3)
        
        feat4 = self.cbam(feat4)

        d3 = self.L3(feat4, feat3)
        d2 = self.L2(d3, feat2)
        d1 = self.L1(d2, feat1)
        d0 = self.L0(d1, feat0)

        seg = self.final_up(d0)
        if seg.shape[2:] != (256, 256):
            seg = F.interpolate(seg, size=(256, 256), mode='bilinear')
            
        cls = self.head(F.adaptive_avg_pool2d(feat4, 1).flatten(1))
        return seg, cls