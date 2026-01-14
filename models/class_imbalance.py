import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
from experiments.decoder import UNetPlusPlusDecoder
from .layers import CBAM
from experiments.class_imbalance_module import DualDomainBottleneck, LatentAugmentor

class FinalExperimentModel(nn.Module):
    def __init__(self, num_classes=3):
        super().__init__()
        
        try:
            base = models.efficientnet_b6(weights='DEFAULT')
        except:
            print("Hash Error or Download Failed weights. Using random weights for B6.")
            base = models.efficientnet_b6(weights=None)
            
        self.encoder_blocks = base.features
        self.stage_indices, ch = self._detect_resolution_stages()
        print(f"Detected Encoder Channels: {ch}")
        
        self.dual_bottleneck = DualDomainBottleneck(ch[4])
        self.latent_aug = LatentAugmentor()
        self.cbam = CBAM(ch[4])
        
        
        self.decoder = UNetPlusPlusDecoder(ch)
        
        self.final_up = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
            nn.Conv2d(ch[0], 32, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 1, 1)
        )
        self.head = nn.Linear(ch[4], num_classes)

    def _detect_resolution_stages(self):
        """ Automatically identifies the 5 stages for UNet skip-connections """
        x = torch.zeros(1, 3, 256, 256)
        indices = []
        channels = []
        
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

    def forward(self, x, labels=None):
        stages = []
        curr = x
        for i, block in enumerate(self.encoder_blocks):
            curr = block(curr)
            if i in self.stage_indices:
                stages.append(curr)
        
        s0, s1, s2, s3, s4 = stages
        
        # Bottleneck
        s4 = self.dual_bottleneck(s4)
        
        # Latent Augmentation (Training Only)
        if self.training and labels is not None:
            s4 = self.latent_aug(s4, labels)
            
        s4 = self.cbam(s4)
        
        # UNet++ Decoder
        dec_out = self.decoder([s0, s1, s2, s3, s4])
        seg = self.final_up(dec_out)
        
        if seg.shape[2:] != (256, 256):
            seg = F.interpolate(seg, size=(256, 256), mode='bilinear', align_corners=True)
            
        cls = self.head(F.adaptive_avg_pool2d(s4, 1).flatten(1))
        
        return seg, cls