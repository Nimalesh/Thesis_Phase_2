import os
import gc
import torch
from torch.utils.data import DataLoader
from sklearn.model_selection import KFold

import config
from models.multitask import MultiTaskLatentAugModel
from utils.dataset import get_dataloaders, PRECISEDataset
from utils.losses import DiceFocalLoss
from utils.metrics import compute_metrics

def compute_weights(df):
    c = df['label_idx'].value_counts().sort_index().values
    return torch.tensor(sum(c)/(len(c)*c), dtype=torch.float32)

def train():
    # 1. FORCE CPU DEVICE
    device = torch.device('cpu')
    print(f"Training started on: {device}")
    
    df = get_dataloaders(config.BASE_PATH)
    if len(df) == 0:
        print("Data not found. Check BASE_PATH in config.py")
        return

    weights = compute_weights(df).to(device)
    kf = KFold(n_splits=config.N_SPLITS, shuffle=True, random_state=42)
    
    for fold, (t_idx, v_idx) in enumerate(kf.split(df)):
        print(f"\n=== FOLD {fold+1} ===")
        
        tr_ds = PRECISEDataset(df.iloc[t_idx], train=True, size=config.IMAGE_SIZE)
        va_ds = PRECISEDataset(df.iloc[v_idx], train=False, size=config.IMAGE_SIZE)
        
        # Reduced num_workers for CPU stability
        tr_dl = DataLoader(tr_ds, batch_size=config.BATCH_SIZE, shuffle=True, num_workers=0)
        va_dl = DataLoader(va_ds, batch_size=config.BATCH_SIZE, shuffle=False, num_workers=0)
        
        model = MultiTaskLatentAugModel(num_classes=config.NUM_CLASSES).to(device)
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
        
        # 2. REMOVED GradScaler (Not used on CPU)
        
        criterion_seg = DiceFocalLoss()
        criterion_cls = torch.nn.CrossEntropyLoss(weight=weights)
        
        best_f1 = 0

        for epoch in range(config.NUM_EPOCHS):
            model.train()
            epoch_loss = 0
            
            for imgs, lbls, msks in tr_dl:
                imgs, lbls, msks = imgs.to(device), lbls.to(device), msks.to(device)
                optimizer.zero_grad()
                
                # 3. REMOVED autocast (Not used on CPU)
                s_out, c_out = model(imgs, labels=lbls)
                l_seg = criterion_seg(s_out, msks)
                l_cls = criterion_cls(c_out, lbls)
                loss = config.ALPHA * l_seg + config.BETA * l_cls
                
                loss.backward()
                optimizer.step()
                
                epoch_loss += loss.item()
            
            # Validation
            model.eval()
            all_s, all_c, all_lbls, all_msks = [], [], [], []
            with torch.no_grad():
                for imgs, lbls, msks in va_dl:
                    s, c = model(imgs.to(device))
                    all_s.append(s); all_c.append(c)
                    all_lbls.append(lbls); all_msks.append(msks)
            
            dsc, nsd, hd, auc, f1 = compute_metrics(
                torch.cat(all_s), torch.cat(all_msks), 
                torch.cat(all_c), torch.cat(all_lbls)
            )
            
            print(f"Epoch {epoch+1} | Loss: {epoch_loss/len(tr_dl):.4f} | DSC: {dsc:.3f} | F1: {f1:.3f}")
            
            if f1 > best_f1:
                best_f1 = f1
                torch.save(model.state_dict(), os.path.join(config.SAVE_DIR, f'model_fold_{fold+1}_best.pth'))

        gc.collect()

if __name__ == "__main__":
    train()