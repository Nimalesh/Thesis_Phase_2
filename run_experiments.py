import os
import gc
import torch
import pandas as pd
import numpy as np
from torch.utils.data import DataLoader
from sklearn.model_selection import KFold

import config
from experiments.encoders import ENCODER_LIST
from models.generic_model import GenericMultiTaskModel
from utils.dataset import get_dataloaders, PRECISEDataset
from utils.losses import DiceFocalLoss
from utils.metrics import compute_metrics

def compute_weights(df):
    c = df['label_idx'].value_counts().sort_index().values
    return torch.tensor(sum(c)/(len(c)*c), dtype=torch.float32)

def run_experiment_for_encoder(encoder_name, df, device, weights):
    kf = KFold(n_splits=config.N_SPLITS, shuffle=True, random_state=42)
    fold_results = []
    
    print(f"\nEXPERIMENT: {encoder_name}")
    
    for fold, (t_idx, v_idx) in enumerate(kf.split(df)):
        print(f"  --- Fold {fold+1}/{config.N_SPLITS} ---")
        

        tr_ds = PRECISEDataset(df.iloc[t_idx], train=True, size=config.IMAGE_SIZE)
        va_ds = PRECISEDataset(df.iloc[v_idx], train=False, size=config.IMAGE_SIZE)
        
        tr_dl = DataLoader(tr_ds, batch_size=config.BATCH_SIZE, shuffle=True, num_workers=0)
        va_dl = DataLoader(va_ds, batch_size=config.BATCH_SIZE, shuffle=False, num_workers=0)
        
        model = GenericMultiTaskModel(encoder_name).to(device)
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
        
        criterion_seg = DiceFocalLoss()
        criterion_cls = torch.nn.CrossEntropyLoss(weight=weights.to(device))
        
        best_fold_f1 = 0
        best_fold_metrics = {}

        for epoch in range(config.NUM_EPOCHS):
            model.train()
            for imgs, lbls, msks in tr_dl:
                imgs, lbls, msks = imgs.to(device), lbls.to(device), msks.to(device)
                optimizer.zero_grad()
                
                # Forward
                s_out, c_out = model(imgs)
                l_seg = criterion_seg(s_out, msks)
                l_cls = criterion_cls(c_out, lbls)
                loss = config.ALPHA * l_seg + config.BETA * l_cls
                
                loss.backward()
                optimizer.step()
            
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
            
            if f1 > best_fold_f1:
                best_fold_f1 = f1
                best_fold_metrics = {
                    "encoder": encoder_name,
                    "fold": fold + 1,
                    "dsc": dsc,
                    "nsd": nsd,
                    "hd95": hd,
                    "auc": auc,
                    "f1": f1
                }

        fold_results.append(best_fold_metrics)
        print(f"    Fold {fold+1} Finished. Best F1: {best_fold_f1:.4f}")
        
        del model, tr_dl, va_dl
        gc.collect()
        if device.type == 'mps': torch.mps.empty_cache()

    df_fold = pd.DataFrame(fold_results)
    avg_results = df_fold.mean(numeric_only=True).to_dict()
    avg_results['encoder'] = encoder_name
    return avg_results

def main():
    # Setup Device
    if torch.backends.mps.is_available(): device = torch.device("mps")
    elif torch.cuda.is_available(): device = torch.device("cuda")
    else: device = torch.device("cpu")
    
    print(f"Starting Experiments on: {device}")
    
    # Setup Data
    df_data = get_dataloaders(config.BASE_PATH)
    weights = compute_weights(df_data)
    
    all_experiments = []
    
    os.makedirs("experiment_logs", exist_ok=True)

    for encoder in ENCODER_LIST:
        try:
            results = run_experiment_for_encoder(encoder, df_data, device, weights)
            all_experiments.append(results)
            
            pd.DataFrame(all_experiments).to_csv("experiment_logs/encoder_experiment_results.csv", index=False)
        except Exception as e:
            print(f"Error training {encoder}: {e}")
            continue

if __name__ == "__main__":
    main()