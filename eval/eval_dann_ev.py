import os
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
from sklearn.metrics import mean_squared_error, mean_absolute_error, mean_absolute_percentage_error, r2_score

script_dir = os.path.dirname(os.path.abspath(__file__))
os.chdir(script_dir)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
EV_IDS = [541]
gas_data_percent = 1.0
DATA_FILE = '../timeseries_dataset_phev_ev.npz'
CKPT_PATH = f'../model_config/best_finetune_from_gas_to_ev_dann.pt'
BATCH_SIZE = 64

def load_and_filter(path, id_list):
    assert os.path.isfile(path), f"{path} not found!"
    data = np.load(path)
    X, y = data['X'], data['y']
    veh = X[:, 0, 0].astype(int)
    mask = np.isin(veh, id_list)
    X, y = X[mask], y[mask]
    mask_nan = ~np.isnan(X).any(axis=tuple(range(1, X.ndim)))
    return X[mask_nan], y[mask_nan]

X, y = load_and_filter(DATA_FILE, EV_IDS)
X = X[..., 1:]  # drop VehId

ckpt = torch.load(CKPT_PATH, map_location=device)
nfeat   = ckpt['nfeat']
seq_len = ckpt['seq_len']
scaler  = ckpt['scaler']

flat = X.reshape(-1, nfeat)
flat = scaler.transform(flat)
X = flat.reshape(-1, seq_len, nfeat)

ds     = TensorDataset(torch.from_numpy(X).float(),
                       torch.from_numpy(y).float().unsqueeze(-1))
loader = DataLoader(ds, BATCH_SIZE)

class GradientReversalFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, lambda_):
        ctx.lambda_ = lambda_
        return x.view_as(x)
    @staticmethod
    def backward(ctx, grad_output):
        return -ctx.lambda_ * grad_output, None

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        pos = torch.arange(0, max_len).unsqueeze(1).float()
        div = torch.exp(torch.arange(0, d_model, 2).float() * -(np.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(pos * div)
        pe[:, 1::2] = torch.cos(pos * div)
        self.register_buffer('pe', pe.unsqueeze(0))
    def forward(self, x):
        return x + self.pe[:, :x.size(1)]

class DANNTransformer(nn.Module):
    def __init__(self, feature_dim, d_model=64, nhead=4, num_layers=3, dim_ff=128, dropout=0.1):
        super().__init__()
        self.input_proj = nn.Linear(feature_dim, d_model)
        self.pos_enc = PositionalEncoding(d_model)
        encoder_layer = nn.TransformerEncoderLayer(d_model, nhead, dim_ff, dropout, batch_first=True)
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers)
        self.regressor = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Dropout(0.3),
            nn.Linear(d_model, d_model * 2),
            nn.Linear(d_model * 2, 1)
        )
        self.domain_classifier = nn.Sequential(
            nn.Linear(d_model, 100),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(100, 2)
        )
    def forward(self, x, lambda_=0.0):
        x = self.input_proj(x)
        x = self.pos_enc(x)
        enc_out = self.encoder(x)
        pooled = enc_out.mean(dim=1)
        energy_pred = self.regressor(pooled)
        rev_feat = GradientReversalFunction.apply(pooled, lambda_)
        domain_pred = self.domain_classifier(rev_feat)
        return energy_pred, domain_pred

model = DANNTransformer(feature_dim=nfeat).to(device)
model.load_state_dict(ckpt['model_state'])
model.eval()

preds, trues = [], []
with torch.no_grad():
    for Xb, yb in loader:
        Xb = Xb.to(device)
        energy_pred, _ = model(Xb, lambda_=0.0)
        preds.append(energy_pred.cpu().numpy().flatten())
        trues.append(yb.numpy().flatten())

preds = np.concatenate(preds)
trues = np.concatenate(trues)

mse  = mean_squared_error(trues, preds)
rmse = np.sqrt(mse)
mae  = mean_absolute_error(trues, preds)
mape = mean_absolute_percentage_error(trues, preds)
r2   = r2_score(trues, preds)

print("="*40)
print(f"Evaluation Results on EV Data (DANN)")
print("-"*40)
print(f"  Total sequences evaluated     : {len(trues)}")
print(f"  Ground truth mean energy usage: {np.mean(trues):.4f} kWh")
print(f"  Std deviation of truth        : {np.std(trues):.4f} kWh\n")

print(f"  MSE   = {mse:.6f}")
print(f"  RMSE  = {rmse:.6f}")
print(f"  MAE   = {mae:.6f}")
print(f"  MAPE  = {mape:.6f}")
print(f"  R²    = {r2:.4f}")
print("="*40)

print("\nSample predictions vs. ground truth:")
for p, t in zip(preds[:5], trues[:5]):
    print(f"  pred = {p:.4f} kWh, true = {t:.4f} kWh")
print("="*40)
