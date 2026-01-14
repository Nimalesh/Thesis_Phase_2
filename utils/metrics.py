import numpy as np
import torch
import torch.nn.functional as F
from sklearn.metrics import roc_auc_score, f1_score, precision_score, recall_score
from scipy.ndimage import distance_transform_edt

def compute_metrics(pred_seg, target_seg, pred_cls, target_cls):
    """
    Computes comprehensive metrics for both Segmentation and Classification.
    """
    # --- Segmentation Metrics ---
    pred_mask = (torch.sigmoid(pred_seg) > 0.5).float().cpu().numpy()
    gt_mask = target_seg.float().cpu().numpy()
    
    dsc_list, nsd_list, hd_list = [], [], []
    
    for i in range(pred_mask.shape[0]):
        pm, gm = pred_mask[i, 0], gt_mask[i, 0]
        
        # 1. Dice Score (DSC)
        inter = np.sum(pm * gm)
        union = np.sum(pm) + np.sum(gm)
        dsc_list.append((2. * inter + 1e-6) / (union + 1e-6))
        
        if np.sum(pm) == 0 and np.sum(gm) == 0:
            nsd_list.append(1.0); hd_list.append(0.0)
            continue
        
        d_p = distance_transform_edt(1 - pm)
        d_g = distance_transform_edt(1 - gm)
        
        # 2. Normalized Surface Dice (NSD)
        pm_border = d_p <= 2
        gm_border = d_g <= 2
        inter_surf = np.sum(pm_border * gm) + np.sum(gm_border * pm)
        union_surf = np.sum(pm_border) + np.sum(gm_border)
        nsd_list.append((inter_surf + 1e-6) / (union_surf + 1e-6))
        
        # 3. HD95
        p_coords = np.argwhere(pm)
        g_coords = np.argwhere(gm)
        if len(p_coords) == 0 or len(g_coords) == 0:
            hd_list.append(0.0)
        else:
            dist_p2g = d_g[p_coords[:,0], p_coords[:,1]]
            dist_g2p = d_p[g_coords[:,0], g_coords[:,1]]
            hd_list.append(np.percentile(np.concatenate([dist_p2g, dist_g2p]), 95))

    # --- Classification Metrics ---
    probs = F.softmax(pred_cls, dim=1).cpu().numpy()
    preds = probs.argmax(1)
    lbls = target_cls.cpu().numpy()
    
    try: auc = roc_auc_score(lbls, probs, multi_class='ovr')
    except: auc = 0.0
    f1 = f1_score(lbls, preds, average='macro')
    
    return np.mean(dsc_list), np.mean(nsd_list), np.mean(hd_list), auc, f1