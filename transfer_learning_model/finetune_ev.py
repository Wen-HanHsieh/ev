import os
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

script_dir = os.path.dirname(os.path.abspath(__file__))
os.chdir(script_dir)

# 1) Vehicle ID list for EV
EV_IDS   = [10, 455]
gas_data_percent = 1.0  # choose it from [0.2,0.4,0.6,0.8,1.0]
DATA_FILE = '../timeseries_dataset_phev_ev.npz'
PRETRAIN_CKPT = f'../model_config/best_pretrain_gas_{gas_data_percent}_transformer.pt'

# 2) Loader + filter by ID + drop NaNs
def load_and_filter(path, id_list):
    assert os.path.isfile(path), f"{path} not found!"
    data = np.load(path)
    X, y = data['X'], data['y']
    veh = X[:, 0, 0].astype(int)
    mask = np.isin(veh, id_list)
    X, y = X[mask], y[mask]
    mask_nan = ~np.isnan(X).any(axis=tuple(range(1, X.ndim)))
    return X[mask_nan], y[mask_nan]

# 3) Load EV data
X_ft, y_ft = load_and_filter(DATA_FILE, EV_IDS)
# 4) Drop VehId column
X_ft = X_ft[..., 1:]

# 5) Load pretrained checkpoint (with scaler)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
ckpt = torch.load(PRETRAIN_CKPT,
                  map_location=device,
                  weights_only=False)
scaler  = ckpt['scaler']
nfeat   = ckpt['nfeat']
seq_len = ckpt['seq_len']

# 6) Scale EV data using pretrained scaler
flat = X_ft.reshape(-1, nfeat)
flat = scaler.transform(flat)
X_ft = flat.reshape(-1, seq_len, nfeat)

# 7) Train/val split for fine-tuning
X_tr, X_val, y_tr, y_val = train_test_split(
    X_ft, y_ft, test_size=0.1, random_state=42
)

# 8) DataLoaders
batch_size = 64
train_ds = TensorDataset(
    torch.from_numpy(X_tr).float(),
    torch.from_numpy(y_tr).float().unsqueeze(-1)
)
val_ds   = TensorDataset(
    torch.from_numpy(X_val).float(),
    torch.from_numpy(y_val).float().unsqueeze(-1)
)
train_loader = DataLoader(train_ds, batch_size, shuffle=True)
val_loader   = DataLoader(val_ds,   batch_size)

# 9) Model definitions (same as pretrain)
class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        pos = torch.arange(0, max_len).unsqueeze(1).float()
        div = torch.exp(torch.arange(0, d_model, 2).float() * -(np.log(10000)/d_model))
        pe[:, 0::2] = torch.sin(pos*div)
        pe[:, 1::2] = torch.cos(pos*div)
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
            nn.Linear(d_model, d_model*2),
            nn.Linear(d_model*2, 1)
        )
    def forward(self, x):
        x = self.input_proj(x)
        x = self.pos_enc(x)
        x = self.encoder(x)
        x = x.mean(dim=1)
        return self.regressor(x)

# 10) Load model state & freeze
model = TransformerRegressor(feature_dim=nfeat).to(device)
model.load_state_dict(ckpt['model_state'])

#### uncomment the following codes if partial finetune is required

# for name,p in model.named_parameters():
#     if 'regressor' not in name:
#         p.requires_grad = False

# for param in model.parameters():
#     param.requires_grad = False
# for param in model.encoder.layers[-1].parameters():
#     param.requires_grad = True
# for param in model.regressor.parameters():
#     param.requires_grad = True

# 11) Fine-tuning setup
def optimizer_and_criterion():
    opt = torch.optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=1e-4
    )
    return opt, nn.MSELoss()

optimizer, criterion = optimizer_and_criterion()

# 12) Fine-tuning loop + checkpointing
train_losses = []
val_losses  = []
best_val = float('inf')
ft_ckpt = f'../model_config/best_finetune_from_{gas_data_percent}_gas_to_ev_fft.pt' # or best_finetune_from_{gas_data_percent}_gas_to_ev.pt represents partial finetune
for epoch in range(1, 1001):
    model.train()
    tr_loss = 0
    for Xb, yb in train_loader:
        Xb, yb = Xb.to(device), yb.to(device)
        pred   = model(Xb)
        loss   = criterion(pred, yb)
        optimizer.zero_grad(); loss.backward(); optimizer.step()
        tr_loss += loss.item() * Xb.size(0)
    avg_tr = tr_loss / len(train_loader.dataset)

    model.eval()
    val_loss = 0
    with torch.no_grad():
        for Xb, yb in val_loader:
            Xb, yb = Xb.to(device), yb.to(device)
            val_loss += criterion(model(Xb), yb).item() * Xb.size(0)
    avg_val = val_loss / len(val_loader.dataset)

    train_losses.append(avg_tr)
    val_losses.append(avg_val)
    print(f"[FT{epoch:02d}] train: {avg_tr:.4f}, val: {avg_val:.4f}")
    if avg_val < best_val:
        best_val = avg_val
        torch.save({
            'model_state': model.state_dict(),
            'scaler':      scaler,
            'nfeat':       nfeat,
            'seq_len':     seq_len
        }, ft_ckpt)
        print(f"  → Saved {ft_ckpt} (val MSE {best_val:.4f})")

# plot train loss and val loss
plt.figure(figsize=(8, 5))
plt.plot(train_losses, label='Train Loss')
plt.plot(val_losses, label='Validation Loss')
plt.xlabel('Epoch')
plt.ylabel('MSE Loss')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig(f'../figures/finetune_train_loss_curves_{gas_data_percent}_transformer_full_finetune.pdf',dpi=400)