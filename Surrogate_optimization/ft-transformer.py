# -*- coding: utf-8 -*-


import os
import math
from pathlib import Path
import numpy as np
import pandas as pd
import random
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as nn_init
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import joblib
import matplotlib.pyplot as plt
import seaborn as sns

sns.set(style="whitegrid", font="Times New Roman", font_scale=1.4)

Tensor = torch.Tensor

# -------------------- Utilities --------------------
def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = False
    torch.backends.cudnn.benchmark = True

def mae(y_true, y_pred):
    return np.mean(np.abs(y_true - y_pred), axis=0)

def rmse(y_true, y_pred):
    return np.sqrt(np.mean((y_true - y_pred) ** 2, axis=0))

def mre(y_true, y_pred, eps: float = 1e-8):

    denom = np.where(np.abs(y_true) < eps, eps, y_true)
    return np.mean(np.abs((y_true - y_pred) / denom), axis=0)

class NpDataset(Dataset):
    def __init__(self, x, y):
        self.x = x.astype(np.float32)
        self.y = y.astype(np.float32)

    def __len__(self):
        return len(self.x)

    def __getitem__(self, i):
        return torch.from_numpy(self.x[i]), torch.from_numpy(self.y[i])


def style_axes(ax):

    # Axis labels
    ax.xaxis.label.set_size(16)
    ax.yaxis.label.set_size(16)
    ax.xaxis.label.set_weight("bold")
    ax.yaxis.label.set_weight("bold")

    # Tick labels
    for label in ax.get_xticklabels() + ax.get_yticklabels():
        label.set_fontsize(14)
        label.set_fontweight("bold")

    # Legend
    leg = ax.get_legend()
    if leg is not None:
        for text in leg.get_texts():
            text.set_fontsize(14)
            text.set_fontweight("bold")
        leg.get_title().set_fontsize(14)
        leg.get_title().set_fontweight("bold")

# -------------------- FT-Transformer Core --------------------
class Tokenizer(nn.Module):
    def __init__(self, d_numerical, d_token, bias=True):
        super().__init__()
        self.weight = nn.Parameter(torch.empty(d_numerical + 1, d_token))
        self.bias = nn.Parameter(torch.empty(d_numerical, d_token)) if bias else None
        nn_init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        if self.bias is not None:
            nn_init.kaiming_uniform_(self.bias, a=math.sqrt(5))

    def forward(self, x_num):
        x_num = torch.cat([torch.ones(len(x_num), 1, device=x_num.device), x_num], dim=1)
        x = self.weight[None] * x_num[:, :, None]
        if self.bias is not None:
            bias = torch.cat([torch.zeros(1, self.bias.shape[1], device=x.device), self.bias])
            x = x + bias[None]
        return x

class MultiheadAttention(nn.Module):
    def __init__(self, d, n_heads, dropout):
        super().__init__()
        self.n_heads = n_heads
        self.W_q, self.W_k, self.W_v = nn.Linear(d, d), nn.Linear(d, d), nn.Linear(d, d)
        self.W_out = nn.Linear(d, d)
        self.dropout = nn.Dropout(dropout)

    def _reshape(self, x):
        B, N, D = x.shape
        d_h = D // self.n_heads
        return x.reshape(B, N, self.n_heads, d_h).transpose(1, 2).reshape(B * self.n_heads, N, d_h)

    def forward(self, q, kv):
        q, k, v = self.W_q(q), self.W_k(kv), self.W_v(kv)
        q, k, v = self._reshape(q), self._reshape(k), self._reshape(v)
        att = F.softmax(q @ k.transpose(1, 2) / math.sqrt(k.shape[-1]), dim=-1)
        att = self.dropout(att)
        x = att @ v
        B = len(q) // self.n_heads
        x = x.reshape(B, self.n_heads, -1, v.shape[-1]).transpose(1, 2).reshape(B, -1, self.n_heads * v.shape[-1])
        return self.W_out(x)

class Transformer(nn.Module):
    def __init__(self, d_numerical, n_layers, d_token, n_heads, d_ffn_factor,
                 attention_dropout, ffn_dropout, activation, d_out):
        super().__init__()
        self.tokenizer = Tokenizer(d_numerical, d_token, True)
        self.layers = nn.ModuleList([])
        self.act = F.gelu if activation == "gelu" else F.relu
        d_hidden = int(d_token * d_ffn_factor)
        for _ in range(n_layers):
            self.layers.append(nn.ModuleDict({
                'attn': MultiheadAttention(d_token, n_heads, attention_dropout),
                'norm0': nn.LayerNorm(d_token),
                'norm1': nn.LayerNorm(d_token),
                'fc1': nn.Linear(d_token, d_hidden),
                'fc2': nn.Linear(d_hidden, d_token)
            }))
        self.head = nn.Linear(d_token, d_out)

    def forward(self, x_num):
        x = self.tokenizer(x_num)
        for l in self.layers:
            x = x + l['attn'](l['norm0'](x), l['norm0'](x))
            ff = self.act(l['fc1'](l['norm1'](x)))
            x = x + l['fc2'](ff)
        x = self.head(F.gelu(x[:, 0]))
        return x

# -------------------- Main Training --------------------
def main():
    csv_path = "data/lhs_dynamic_incremental_filtered.csv"
    out_dir = Path("runs/ft_transformer")
    out_dir.mkdir(parents=True, exist_ok=True)
    inputs, outputs = ["d", "w", "n1", "R"], ["Fy_N", "G_N", "Fy_over_G"]

    set_seed(45)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    df = pd.read_csv(csv_path)
    X, Y = df[inputs].values, df[outputs].values
    mask = ~np.isnan(X).any(1) & ~np.isnan(Y).any(1)
    X, Y = X[mask], Y[mask]

    X_train, X_tmp, Y_train, Y_tmp = train_test_split(X, Y, test_size=0.2, random_state=45, shuffle=True)
    X_val, X_test, Y_val, Y_test = train_test_split(X_tmp, Y_tmp, test_size=0.5, random_state=45, shuffle=True)

    x_scaler, y_scaler = StandardScaler().fit(X_train), StandardScaler().fit(Y_train)
    X_train_s, X_val_s, X_test_s = x_scaler.transform(X_train), x_scaler.transform(X_val), x_scaler.transform(X_test)
    Y_train_s, Y_val_s, Y_test_s = y_scaler.transform(Y_train), y_scaler.transform(Y_val), y_scaler.transform(Y_test)

    train_loader = DataLoader(NpDataset(X_train_s, Y_train_s), batch_size=128, shuffle=True)
    val_loader = DataLoader(NpDataset(X_val_s, Y_val_s), batch_size=512, shuffle=False)
    test_loader = DataLoader(NpDataset(X_test_s, Y_test_s), batch_size=512, shuffle=False)

    model = Transformer(
        d_numerical=4, n_layers=4, d_token=128, n_heads=4, d_ffn_factor=4.0,
        attention_dropout=0.1, ffn_dropout=0.1, activation="gelu", d_out=3
    ).to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=300, eta_min=1e-5)
    loss_fn = nn.SmoothL1Loss(beta=1.0)

    best_val, patience, min_delta = float("inf"), 50, 1e-5
    train_losses, val_losses = [], []

    for ep in range(1, 301):
        model.train()
        tr_loss = 0.0
        for xb, yb in train_loader:
            xb, yb = xb.to(device), yb.to(device)
            optimizer.zero_grad()
            loss = loss_fn(model(xb), yb)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()
            tr_loss += loss.item() * len(xb)
        tr_loss /= len(train_loader.dataset)

        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for xb, yb in val_loader:
                xb, yb = xb.to(device), yb.to(device)
                val_loss += loss_fn(model(xb), yb).item() * len(xb)
        val_loss /= len(val_loader.dataset)

        train_losses.append(tr_loss)
        val_losses.append(val_loss)
        scheduler.step()

        print(f"[{ep:03d}] train={tr_loss:.6f} val={val_loss:.6f}")
        if val_loss + min_delta < best_val:
            best_val = val_loss
            torch.save(model.state_dict(), out_dir / "model.pt")
            patience = 50
        else:
            patience -= 1
            if patience <= 0:
                print("Early stopping.")
                break

    model.load_state_dict(torch.load(out_dir / "model.pt", map_location=device))
    joblib.dump(x_scaler, out_dir / "x_scaler.joblib")
    joblib.dump(y_scaler, out_dir / "y_scaler.joblib")

    def predict(loader):
        model.eval()
        preds = []
        with torch.no_grad():
            for xb, _ in loader:
                xb = xb.to(device)
                preds.append(model(xb).cpu().numpy())
        return np.concatenate(preds, axis=0)

    y_pred_s = predict(test_loader)
    y_pred = y_scaler.inverse_transform(y_pred_s)
    y_true = Y_test

    per_mae, per_rmse, per_mre = mae(y_true, y_pred), rmse(y_true, y_pred), mre(y_true, y_pred)
    labels = [r"$F_{m}$", r"$G_{0}$", r"$F_{m}/G_{0}$"]


    print("\n==== Test Metrics (original units) ====")

    for i, name in enumerate(labels[:2]):
        print(
            f"{name:>12s} | MAE={per_mae[i]:.6f} | RMSE={per_rmse[i]:.6f} | "
            f"MRE={per_mre[i]:.6f}"
        )

    # -------------------- Visualization --------------------
    # Only output training/validation loss curve, saved as TrainingConvergence.pdf
    plt.figure(figsize=(6, 4))
    plt.plot(train_losses, label="Training Loss", lw=2)
    plt.plot(val_losses, label="Validation Loss", lw=2, linestyle="--")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    ax = plt.gca()
    ax.legend(loc="best")
    style_axes(ax)
    plt.tight_layout()
    plt.savefig(out_dir / "TrainingConvergence.pdf", dpi=600)
    plt.close()

    print(f"\n✅ Visualizations saved to: {out_dir.resolve()}")

if __name__ == "__main__":
    main()
