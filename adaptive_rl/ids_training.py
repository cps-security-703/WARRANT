import os
import warnings

import numpy as np
import torch
import torch.nn as nn

from ids_neural_models import DEVICE, FEATURE_DIM, SEQ_LEN, TransformerIDS, TransformerIDSWrapper  # noqa: F401

warnings.filterwarnings("ignore")
os.environ["KMP_DUPLICATE_LIB_OK"] = "True"


def _inject_fdi_14d(feat: np.ndarray, attack_type: int,
                     step_frac: float = 1.0) -> np.ndarray:
    f          = feat.copy()
    eff_frac   = max(step_frac, 0.25)
    mag        = (np.random.uniform(0.42, 0.62) + 0.18 * eff_frac) * eff_frac

    if attack_type == 0:
        f[7]  = float(np.clip(f[7]  - mag * 0.45, 0.70, 1.30))
        f[11] = float(np.clip(f[11] + mag * 0.35, 0.00, 2.00))
        f[3]  = float(np.clip(f[3]  + mag * 0.30, 0.00, 2.00))
        f[1]  = float(np.clip(f[1]  - mag * 0.20, 0.00, 2.00))

    elif attack_type == 1:
        f[2]  = float(np.clip(f[2]  + mag * 0.55, 0.00, 2.00))
        f[5]  = float(np.clip(f[5]  + mag * 0.50, 0.00, 2.00))
        f[6]  = float(np.clip(f[6]  + mag * 0.48, 0.00, 2.00))
        f[11] = float(np.clip(f[11] + mag * 0.30, 0.00, 2.00))
        f[3]  = float(np.clip(f[3]  + mag * 0.25, 0.00, 2.00))

    elif attack_type == 2:
        f[3]  = float(np.clip(f[3]  - mag * 1.00, 0.00, 1.00))
        f[5]  = float(np.clip(f[5]  - mag * 0.95, 0.00, 1.00))
        f[6]  = float(np.clip(f[6]  - mag * 0.95, 0.00, 1.00))
        f[10] = float(np.clip(f[10] - mag * 0.45, 0.00, 1.00))

    elif attack_type == 3:
        f[0]  = float(np.clip(f[0]  - mag * 0.80, 0.00, 1.00))
        f[11] = float(np.clip(f[11] + mag * 0.55, 0.00, 2.00))
        f[2]  = float(np.clip(f[2]  + mag * 0.40, 0.00, 2.00))
        f[9]  = float(np.clip(f[9]  + mag * 0.30, 0.00, 2.00))

    elif attack_type == 4:
        f[8]  = float(np.clip(f[8]  + mag * 0.18, 0.87, 1.13))
        f[5]  = float(np.clip(f[5]  + mag * 0.40, 0.00, 2.00))
        f[7]  = float(np.clip(f[7]  - mag * 0.30, 0.70, 1.30))
        f[6]  = float(np.clip(f[6]  + mag * 0.25, 0.00, 2.00))

    elif attack_type == 5:
        phase = step_frac * np.pi * 2.0
        osc   = np.sin(phase) * mag * 0.25
        f[5]  = float(np.clip(f[5]  + osc * 22.0, 0.00, 2.00))
        f[7]  = float(np.clip(f[7]  - mag * 0.30, 0.70, 1.30))
        f[3]  = float(np.clip(f[3]  + osc * 14.0, 0.00, 2.00))
        f[2]  = float(np.clip(f[2]  + osc *  8.0, 0.00, 2.00))

    return f



def train_transformer_ids(X_seq_tr: np.ndarray, y_tr: np.ndarray,
                           epochs: int = 80, lr: float = 3e-4,
                           batch: int = 256) -> TransformerIDS:
    model     = TransformerIDS().to(DEVICE)
    opt       = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)
    n_neg     = int((y_tr == 0).sum()); n_pos = int((y_tr == 1).sum())
    pos_w     = torch.tensor([n_neg / max(n_pos, 1)], dtype=torch.float32)
    ce_loss   = nn.CrossEntropyLoss(
        weight=torch.tensor([1.0, pos_w.item()]), label_smoothing=0.05)
    ds     = torch.utils.data.TensorDataset(
        torch.FloatTensor(X_seq_tr), torch.LongTensor(y_tr.astype(int)))
    loader = torch.utils.data.DataLoader(ds, batch_size=batch, shuffle=True)
    n_p    = sum(p.numel() for p in model.parameters())
    print(f"    Training Transformer IDS ({n_p} params) …")
    model.train()
    for ep in range(epochs):
        ep_loss = 0.0
        for xb, yb in loader:
            xb, yb = xb.to(DEVICE), yb.to(DEVICE)
            opt.zero_grad()
            loss = ce_loss(model(xb), yb)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            ep_loss += loss.item()
        scheduler.step()
        if (ep + 1) % 20 == 0:
            print(f"      epoch {ep+1:3d}/{epochs}: loss={ep_loss/len(loader):.5f}")
    model.eval()
    return model

