import os
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
from sklearn.metrics import mean_squared_error, mean_absolute_error, mean_absolute_percentage_error, r2_score

# 1) EV IDs and checkpoint
EV_IDS   = [541]
DATA_FILE = 'timeseries_dataset_phev_ev.npz'
CKPT_PATH = 'best_pretrain.pt'
BATCH_SIZE = 64

def load_and_filter(path, id_list):
    assert os.path.isfile(path), f"{path} not found!"
    data = np.load(path)
    X, y = data['X'], data['y']
    veh = X[:, 0, 0].astype(int)
    mask = np.isin(veh, id_list)
    X, y = X[mask], y[mask]
    mask2 = ~np.isnan(X).any(axis=tuple(range(1, X.ndim)))
    return X[mask2], y[mask2]

# 2) Load model + scaler
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
ckpt = torch.load(
    CKPT_PATH,
    map_location=device,
    weights_only=False
)
nfeat   = ckpt['nfeat']
seq_len = ckpt['seq_len']
scaler  = ckpt['scaler']

# 3) Re‑define model classes (same as train.py)
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
        enc_layer = nn.TransformerEncoderLayer(d_model, nhead, dim_ff,
                                               dropout, batch_first=True)
        self.encoder   = nn.TransformerEncoder(enc_layer, num_layers)
        self.regressor = nn.Sequential(nn.LayerNorm(d_model),
                                       nn.Linear(d_model, 1))
    def forward(self, x):
        x = self.input_proj(x)
        x = self.pos_enc(x)
        x = self.encoder(x)
        x = x.mean(dim=1)
        return self.regressor(x)

model = TransformerRegressor(feature_dim=nfeat).to(device)
model.load_state_dict(ckpt['model_state'])
model.eval()

# 4) Prepare EV data
X, y = load_and_filter(DATA_FILE, EV_IDS)
X = X[..., 1:]  # drop VehId
flat = X.reshape(-1, nfeat)
flat = scaler.transform(flat)
X = flat.reshape(-1, seq_len, nfeat)

ds     = TensorDataset(torch.from_numpy(X).float(),
                       torch.from_numpy(y).float().unsqueeze(-1))
loader = DataLoader(ds, BATCH_SIZE)

# 5) Inference & metrics
preds, trues = [], []
with torch.no_grad():
    for Xb, yb in loader:
        Xb = Xb.to(device)
        p = model(Xb).cpu().numpy().flatten()
        preds.append(p)
        trues.append(yb.numpy().flatten())

preds = np.concatenate(preds)
trues = np.concatenate(trues)

mse  = mean_squared_error(trues, preds)
rmse = np.sqrt(mse)
mae  = mean_absolute_error(trues, preds)
map_ = mean_absolute_percentage_error(trues, preds)
r2 = r2_score(trues, preds)

# Additional info
num_samples = len(trues)
mean_true = np.mean(trues)
std_true = np.std(trues)

print("="*40)
print(f"Evaluation Results on EV Data (ID(s): {EV_IDS})")
print("-"*40)
print(f"  Total sequences evaluated: {num_samples}")
print(f"  Ground truth mean energy usage : {mean_true:.4f} kWh")
print(f"  Ground truth std deviation     : {std_true:.4f} kWh\n")

print(f"  MSE  = {mse:.6f} (kWh²)")
print(f"  RMSE = {rmse:.6f} (kWh)")
print(f"  MAE  = {mae:.6f} (kWh)")
print(f"  MAP  = {map_:.6f}")
print(f"  R^2 = {r2:.4f}")
print("="*40)

print("\nSample predictions vs. ground truth:")
for p, t in zip(preds[:5], trues[:5]):
    print(f"  pred = {p:.4f} kWh, true = {t:.4f} kWh")
print("="*40)
