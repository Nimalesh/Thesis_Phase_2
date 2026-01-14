import os
import torch
import pandas as pd
from torch.utils.data import DataLoader
from sklearn.model_selection import KFold
import config
from models.multitask_decoder_exp import MultiTaskDecoderModel
from utils.dataset import get_dataloaders, PRECISEDataset
from utils.losses import DiceFocalLoss
from utils.metrics import compute_metrics

# --- FORCE EFFICIENTNET B6 ---
BEST_ENCODER = "efficientnet_b6" 
DECODER_TYPES = ["unet", "unetplusplus", "deeplabv3plus"]

def compute_weights(df):
    c = df['label_idx'].value_counts().sort_index().values
    return torch.tensor(sum(c)/(len(c)*c), dtype=torch.float32)

def main():
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    print(f"Running Decoder Experiments with {BEST_ENCODER} on {device}")
    
    df_data = get_dataloaders(config.BASE_PATH)
    weights = compute_weights(df_data)
    
    all_results = []

    for dec_name in DECODER_TYPES:
        print(f"\n🚀 EXPERIMENT: {BEST_ENCODER} + {dec_name}")
        
        kf = KFold(n_splits=config.N_SPLITS, shuffle=True, random_state=42)
        fold_metrics = []

        for fold, (t_idx, v_idx) in enumerate(kf.split(df_data)):
            tr_dl = DataLoader(PRECISEDataset(df_data.iloc[t_idx], train=True), batch_size=config.BATCH_SIZE, shuffle=True)
            va_dl = DataLoader(PRECISEDataset(df_data.iloc[v_idx], train=False), batch_size=config.BATCH_SIZE, shuffle=False)
            
            model = MultiTaskDecoderModel(BEST_ENCODER, dec_name).to(device)
            optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
            criterion_seg = DiceFocalLoss()
            criterion_cls = torch.nn.CrossEntropyLoss(weight=weights.to(device))
            
            best_f1 = 0
            for epoch in range(config.NUM_EPOCHS):
                model.train()
                for imgs, lbls, msks in tr_dl:
                    imgs, lbls, msks = imgs.to(device), lbls.to(device), msks.to(device)
                    optimizer.zero_grad()
                    s, c = model(imgs)
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
                if f1 > best_f1:
                    best_f1 = f1
                    best_fold_res = {"decoder": dec_name, "fold": fold+1, "dsc": dsc, "f1": f1}
            
            fold_metrics.append(best_fold_res)
            print(f"  Fold {fold+1} Best F1: {best_f1:.4f}")

        avg_res = pd.DataFrame(fold_metrics).mean(numeric_only=True).to_dict()
        avg_res["decoder"] = dec_name
        all_results.append(avg_res)
        pd.DataFrame(all_results).to_csv("experiment_logs/decoder_experiment_results.csv", index=False)

if __name__ == "__main__":
    main()