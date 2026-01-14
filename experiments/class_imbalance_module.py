import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.fft

class DualDomainBottleneck(nn.Module):
    def __init__(self, in_ch):
        super().__init__()
        self.spatial_path = nn.Sequential(
            nn.Conv2d(in_ch, in_ch, 1),
            nn.BatchNorm2d(in_ch),
            nn.ReLU(inplace=True)
        )
        
        self.freq_path = nn.Sequential(
            nn.Conv2d(in_ch, in_ch, 1),
            nn.BatchNorm2d(in_ch),
            nn.ReLU(inplace=True)
        )
        
        self.fuse = nn.Conv2d(in_ch * 2, in_ch, 1)

    def forward(self, x):
        spa = self.spatial_path(x)
    
        fft = torch.fft.rfft2(x, norm='ortho')
        mag = torch.abs(fft)
        pha = torch.angle(fft)
        
        mag = mag * 1.05 
        
        real = mag * torch.cos(pha)
        imag = mag * torch.sin(pha)
        ifft = torch.fft.irfft2(torch.complex(real, imag), s=x.shape[2:], norm='ortho')
        
        freq = self.freq_path(ifft)
        return self.fuse(torch.cat([spa, freq], dim=1))

    def __init__(self, in_ch):
        super().__init__()
        self.spatial_path = nn.Sequential(
            nn.Conv2d(in_ch, in_ch, 1),
            nn.BatchNorm2d(in_ch),
            nn.ReLU(inplace=True)
        )
        
        self.freq_path = nn.Sequential(
            nn.Conv2d(in_ch, in_ch, 1),
            nn.BatchNorm2d(in_ch),
            nn.ReLU(inplace=True)
        )
        
        self.fuse = nn.Conv2d(in_ch * 2, in_ch, 1)

    def forward(self, x):
        # 1. Spatial Branch
        spa = self.spatial_path(x)
        
        fft = torch.fft.rfft2(x, norm='ortho')
        mag = torch.abs(fft)
        pha = torch.angle(fft)
        
        mag = mag * 1.1
        real = mag * torch.cos(pha)
        imag = mag * torch.sin(pha)
        ifft = torch.fft.irfft2(torch.complex(real, imag), s=x.shape[2:], norm='ortho')
        
        freq = self.freq_path(ifft)
        
        # 3. Fusion
        return self.fuse(torch.cat([spa, freq], dim=1))

class LatentAugmentor(nn.Module):

    def __init__(self, sigma=0.05, mixup_alpha=0.4):
        super().__init__()
        self.sigma = sigma
        self.alpha = mixup_alpha

    def forward(self, x, labels=None, minority_classes=[0, 2]):
        if not self.training or labels is None:
            return x

        x_aug = x.clone()
        device = x.device

        # 1.Gaussian Noise for minority classes
        mask = torch.tensor([lbl in minority_classes for lbl in labels], device=device)
        if mask.any():
            noise = torch.randn_like(x_aug) * self.sigma
            # Apply noise only to selected indices
            x_aug[mask] = x_aug[mask] + noise[mask]

        # 2.Intra-Class MixUp
        for target_cls in minority_classes:
            idx = (labels == target_cls).nonzero(as_tuple=True)[0]
            if len(idx) > 1:
                # Shuffle indices within the same class
                shuffled_idx = idx[torch.randperm(len(idx))]
                lam = np.random.beta(self.alpha, self.alpha)
                x_aug[idx] = lam * x_aug[idx] + (1 - lam) * x_aug[shuffled_idx]
                
        return x_aug