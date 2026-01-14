import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
from experiments.decoder import UNetDecoder, UNetPlusPlusDecoder, DeepLabV3PlusDecoder
from .layers import CBAM

class MultiTaskDecoderModel(nn.Module):
    def __init__(self, encoder_name, decoder_type, num_classes=3):
        super().__init__()
        self.encoder_name = encoder_name
        
        try:
            base = getattr(models, encoder_name)(weights='DEFAULT')
        except:
            base = getattr(models, encoder_name)(weights=None)
            print(f"Weights failed to download so, initialized {encoder_name} with random weights.")

        if "resnet" in encoder_name:
            self.encoder_blocks = nn.ModuleList([
                nn.Sequential(base.conv1, base.bn1, base.relu), 
                nn.Sequential(base.maxpool, base.layer1),      
                base.layer2,                                   
                base.layer3,                                   
                base.layer4                                    
            ])
        else:
            self.encoder_blocks = base.features

        # Pre-detect stage indices and channels
        self.stage_indices, ch = self._find_endpoints()
        self.cbam = CBAM(ch[-1])

        if decoder_type == "unet":
            self.decoder = UNetDecoder(ch)
            out_ch = ch[0]
        elif decoder_type == "unetplusplus":
            self.decoder = UNetPlusPlusDecoder(ch)
            out_ch = ch[0]
        elif decoder_type == "deeplabv3plus":
            self.decoder = DeepLabV3PlusDecoder(ch)
            out_ch = 256

        self.final_up = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
            nn.Conv2d(out_ch, 32, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 1, 1)
        )
        self.head = nn.Linear(ch[-1], num_classes)

    def _find_endpoints(self):
        x = torch.zeros(1, 3, 256, 256)
        indices = []
        if isinstance(self.encoder_blocks, nn.ModuleList):
            channels = []
            for layer in self.encoder_blocks:
                x = layer(x)
                channels.append(x.shape[1])
            return list(range(len(self.encoder_blocks))), channels

        # EfficientNet Logic
        for i, block in enumerate(self.encoder_blocks):
            prev_res = x.shape[2]
            x = block(x)
            if x.shape[2] < prev_res:
                indices.append(i-1)
        indices.append(len(self.encoder_blocks) - 1)
        
        final_indices = indices[-5:]
        x = torch.zeros(1, 3, 256, 256)
        final_channels = []
        for i, block in enumerate(self.encoder_blocks):
            x = block(x)
            if i in final_indices:
                final_channels.append(x.shape[1])
        return final_indices, final_channels

    def forward(self, x):
        if "inception_v3" in self.encoder_name:
            x = F.interpolate(x, size=(299, 299), mode='bilinear', align_corners=True)

        stages = []
        curr = x
        if isinstance(self.encoder_blocks, nn.ModuleList):
            for layer in self.encoder_blocks:
                curr = layer(curr)
                stages.append(curr)
        else:
            for i, block in enumerate(self.encoder_blocks):
                curr = block(curr)
                if i in self.stage_indices:
                    stages.append(curr)
        
        s0, s1, s2, s3, s4 = stages[-5:]
        s4 = self.cbam(s4)
        
        dec_out = self.decoder([s0, s1, s2, s3, s4])
        seg = self.final_up(dec_out)
        
        if seg.shape[2:] != (256, 256):
            seg = F.interpolate(seg, size=(256, 256), mode='bilinear', align_corners=True)
            
        cls = self.head(F.adaptive_avg_pool2d(s4, 1).flatten(1))
        return seg, cls