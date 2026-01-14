import os
import glob
import pandas as pd
from PIL import Image
import torch
from torch.utils.data import Dataset
import torchvision.transforms.functional as TF
from torchvision import transforms

class PRECISEDataset(Dataset):
    def __init__(self, df, train=True, size=256):
        self.df = df
        self.train = train
        self.size = size
        
    def __len__(self): return len(self.df)
    
    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        img = Image.open(row['path']).convert('RGB')
        # Mask logic: search for _mask version of the image
        mask_path = row['path'].replace('.png', '_mask.png')
        
        if os.path.exists(mask_path):
            mask = Image.open(mask_path).convert('L')
        else:
            mask = Image.new('L', img.size)
        
        if self.train:
            if torch.rand(1) < 0.5:
                img = TF.hflip(img); mask = TF.hflip(mask)
            angle = transforms.RandomRotation.get_params([-30, 30])
            img = TF.rotate(img, angle); mask = TF.rotate(mask, angle)

        img = TF.resize(img, [self.size, self.size])
        mask = TF.resize(mask, [self.size, self.size], interpolation=transforms.InterpolationMode.NEAREST)
        
        img_t = TF.normalize(TF.to_tensor(img), [0.485,0.456,0.406], [0.229,0.224,0.225])
        mask_t = (TF.to_tensor(mask) > 0.5).float()
        
        return img_t, torch.tensor(row['label_idx']), mask_t

def get_dataloaders(base_path):
    records = []
    # Folder based labels: normal=0, benign=1, malignant=2
    mapping = {'normal': 0, 'benign': 1, 'malignant': 2}
    for label_name, label_idx in mapping.items():
        folder = os.path.join(base_path, label_name)
        for f in glob.glob(os.path.join(folder, "*.png")):
            if "_mask" not in f:
                records.append({'path': f, 'label_idx': label_idx})
    
    return pd.DataFrame(records)