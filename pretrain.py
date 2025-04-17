import os
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# 1) Vehicle ID list for PHEV
PHEV_IDS = [9, 11, 371, 379, 388, 398, 417, 431, 443, 449, 453, 457,
            492, 497, 536, 537, 542, 545, 550, 554, 560, 561, 567, 569]
DATA_FILE = 'timeseries_dataset_phev_ev.npz'  # NPZ containing both PHEV & EV

# 2) Loader + filter by ID + drop NaNs
def load_and_filter(path, id_list):
    assert os.path.isfile(path), f"{path} not found!"
    data = np.load(path)
    X, y = data['X'], data['y']
    veh = X[:, 0, 0].astype(int)                # VehId at timestep 0
    mask_id = np.isin(veh, id_list)
    X, y = X[mask_id], y[mask_id]
    mask_nan = ~np.isnan(X).any(axis=tuple(range(1, X.ndim)))
    return X[mask_nan], y[mask_nan]

# 3) Load PHEV data
X_pre, y_pre = load_and_filter(DATA_FILE, PHEV_IDS)
# 4) Drop VehId column
X_pre = X_pre[..., 1:]

# 5) Train/val split & scaling
X_tr, X_val, y_tr, y_val = train_test_split(
    X_pre, y_pre, test_size=0.1, random_state=42
)
nsamples, seq_len, nfeat = X_tr.shape

scaler = StandardScaler().fit(X_tr.reshape(-1, nfeat))
def scale(X):
    flat = X.reshape(-1, nfeat)
    flat = scaler.transform(flat)
    return flat.reshape(-1, seq_len, nfeat)

X_tr, X_val = scale(X_tr), scale(X_val)

# 6) DataLoaders
batch_size = 64
dataset_tr = TensorDataset(
    torch.from_numpy(X_tr).float(),
    torch.from_numpy(y_tr).float().unsqueeze(-1)
)
dataset_val = TensorDataset(
    torch.from_numpy(X_val).float(),
    torch.from_numpy(y_val).float().unsqueeze(-1)
)
train_loader = DataLoader(dataset_tr, batch_size, shuffle=True)
val_loader   = DataLoader(dataset_val,   batch_size)

# 7) Model definitions
class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        pos = torch.arange(0, max_len).unsqueeze(1).float()
        div = torch.exp(torch.arange(0, d_model, 2).float() * -(np.log(10000) / d_model))
        pe[:, 0::2] = torch.sin(pos * div)
        pe[:, 1::2] = torch.cos(pos * div)
        self.register_buffer('pe', pe.unsqueeze(0))
    def forward(self, x):
        return x + self.pe[:, :x.size(1)]

class TransformerRegressor(nn.Module):
    def __init__(self, feature_dim, d_model=64, nhead=4,
                 num_layers=3, dim_ff=128, dropout=0.1):
        super().__init__()
        self.input_proj = nn.Linear(feature_dim, d_model)
        self.pos_enc    = PositionalEncoding(d_model)
        enc_layer = nn.TransformerEncoderLayer(
            d_model, nhead, dim_ff, dropout, batch_first=True
        )
        self.encoder   = nn.TransformerEncoder(enc_layer, num_layers)
        self.regressor = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, 1)
        )
    def forward(self, x):
        x = self.input_proj(x)
        x = self.pos_enc(x)
        x = self.encoder(x)
        x = x.mean(dim=1)
        return self.regressor(x)

# 8) Training + checkpointing

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print("Pretraining on PHEV data using device:", device)
model     = TransformerRegressor(feature_dim=nfeat).to(device)
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
criterion = nn.MSELoss()

best_val = float('inf')
ckpt_path = 'best_pretrain.pt'
for epoch in range(1, 51):
    model.train()
    total_loss = 0
    for Xb, yb in train_loader:
        Xb, yb = Xb.to(device), yb.to(device)
        pred   = model(Xb)
        loss   = criterion(pred, yb)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * Xb.size(0)
    avg_train = total_loss / len(train_loader.dataset)

    model.eval()
    val_loss = 0
    with torch.no_grad():
        for Xb, yb in val_loader:
            Xb, yb = Xb.to(device), yb.to(device)
            val_loss += criterion(model(Xb), yb).item() * Xb.size(0)
    avg_val = val_loss / len(val_loader.dataset)

    print(f"Epoch {epoch:02d} — train: {avg_train:.4f}, val: {avg_val:.4f}")
    if avg_val < best_val:
        best_val = avg_val
        torch.save({
            'model_state': model.state_dict(),
            'scaler':      scaler,
            'nfeat':       nfeat,
            'seq_len':     seq_len
        }, ckpt_path)
        print(f"  → Saved {ckpt_path} (val MSE {best_val:.4f})")