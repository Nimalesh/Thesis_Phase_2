import torch
import os
import pandas as pd
from torch.utils.data import DataLoader
from sklearn.model_selection import KFold

import config
from models.class_imbalance import FinalExperimentModel
from utils.dataset import get_dataloaders, PRECISEDataset
from utils.losses import DiceFocalLoss
from utils.metrics import compute_metrics

def compute_weights(df):
    c = df['label_idx'].value_counts().sort_index().values
    return torch.tensor(sum(c)/(len(c)*c), dtype=torch.float32)

def main():
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    print(f"🚀 Starting Advanced Experiment: B6 + UNet++ + FFT + LatentAug on {device}")
    
    config.BATCH_SIZE = 2
    df_data = get_dataloaders(config.BASE_PATH)
    weights = compute_weights(df_data)
    
    kf = KFold(n_splits=config.N_SPLITS, shuffle=True, random_state=42)
    
    
    results_log = [] 

    for fold, (t_idx, v_idx) in enumerate(kf.split(df_data)):
        print(f"\n--- Fold {fold+1}/{config.N_SPLITS} ---")
        tr_dl = DataLoader(PRECISEDataset(df_data.iloc[t_idx], train=True), batch_size=config.BATCH_SIZE, shuffle=True)
        va_dl = DataLoader(PRECISEDataset(df_data.iloc[v_idx], train=False), batch_size=config.BATCH_SIZE, shuffle=False)
        
        model = FinalExperimentModel().to(device)
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
        criterion_seg = DiceFocalLoss()
        criterion_cls = torch.nn.CrossEntropyLoss(weight=weights.to(device))
        
        best_f1 = 0
        for epoch in range(config.NUM_EPOCHS):
            model.train()
            for imgs, lbls, msks in tr_dl:
                imgs, lbls, msks = imgs.to(device), lbls.to(device), msks.to(device)
                optimizer.zero_grad()
                s, c = model(imgs, labels=lbls) 
                loss = config.ALPHA * criterion_seg(s, msks) + config.BETA * criterion_cls(c, lbls)
                loss.backward()
                optimizer.step()
            
            model.eval()
            all_s, all_c, all_l, all_m = [], [], [], []
            with torch.no_grad():
                for imgs, lbls, msks in va_dl:
                    s, c = model(imgs.to(device))
                    all_s.append(s); all_c.append(c); all_l.append(lbls); all_m.append(msks)
            
            dsc, nsd, hd, auc, f1 = compute_metrics(torch.cat(all_s), torch.cat(all_m), torch.cat(all_c), torch.cat(all_l))
            print(f"  Epoch {epoch+1} | DSC: {dsc:.4f} | F1: {f1:.4f}")
            
            results_log.append({
                "fold": fold + 1,
                "epoch": epoch + 1,
                "dsc": dsc,
                "nsd": nsd,
                "hd95": hd,
                "auc": auc,
                "f1": f1
            })
            
            if f1 > best_f1:
                best_f1 = f1
                torch.save(model.state_dict(), f"best_advanced_model_fold{fold+1}.pth")

        df_results = pd.DataFrame(results_log)
        df_results.to_csv("experiments/class_imbalance_handling.csv", index=False)
        print(f"Results up to Fold {fold+1} saved to 'advanced_experiment_results.csv'")

    print("\nClass Imbalance handling completed. Final CSV saved.")
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    print(f"Starting Advanced Experiment on Class Imbalance handling{device}")
    
    config.BATCH_SIZE = 2
    
    df_data = get_dataloaders(config.BASE_PATH)
    weights = compute_weights(df_data)
    
    kf = KFold(n_splits=config.N_SPLITS, shuffle=True, random_state=42)
    results = []

    for fold, (t_idx, v_idx) in enumerate(kf.split(df_data)):
        print(f"\n--- Fold {fold+1}/{config.N_SPLITS} ---")
        
        tr_dl = DataLoader(PRECISEDataset(df_data.iloc[t_idx], train=True), batch_size=config.BATCH_SIZE, shuffle=True)
        va_dl = DataLoader(PRECISEDataset(df_data.iloc[v_idx], train=False), batch_size=config.BATCH_SIZE, shuffle=False)
        
        model = FinalExperimentModel().to(device)
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
        criterion_seg = DiceFocalLoss()
        criterion_cls = torch.nn.CrossEntropyLoss(weight=weights.to(device))
        
        best_f1 = 0
        for epoch in range(config.NUM_EPOCHS):
            model.train()
            for imgs, lbls, msks in tr_dl:
                imgs, lbls, msks = imgs.to(device), lbls.to(device), msks.to(device)
                optimizer.zero_grad()
                s, c = model(imgs, labels=lbls) # Pass labels for Latent Augment
                loss = config.ALPHA * criterion_seg(s, msks) + config.BETA * criterion_cls(c, lbls)
                loss.backward()
                optimizer.step()
            
            model.eval()
            all_s, all_c, all_l, all_m = [], [], [], []
            with torch.no_grad():
                for imgs, lbls, msks in va_dl:
                    s, c = model(imgs.to(device))
                    all_s.append(s); all_c.append(c); all_l.append(lbls); all_m.append(msks)
            
            dsc, nsd, hd, auc, f1 = compute_metrics(torch.cat(all_s), torch.cat(all_m), torch.cat(all_c), torch.cat(all_l))
            print(f"  Epoch {epoch+1} | DSC: {dsc:.4f} | F1: {f1:.4f}")
            
            if f1 > best_f1:
                best_f1 = f1
                torch.save(model.state_dict(), f"best_advanced_model_fold{fold+1}.pth")

    print("\n Class Imbalance handling completed")

if __name__ == "__main__":
    main()