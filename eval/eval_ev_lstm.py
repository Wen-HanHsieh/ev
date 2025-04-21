import os
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
from sklearn.metrics import mean_squared_error, mean_absolute_error, mean_absolute_percentage_error, r2_score

script_dir = os.path.dirname(os.path.abspath(__file__))
os.chdir(script_dir)

# 1) EV IDs and checkpoint
EV_IDS   = [541]
gas_data_percent = 0.2 # choose it from [0.2,0.4,0.6,0.8,1.0]
DATA_FILE = '../timeseries_dataset_phev_ev.npz'
CKPT_PATH = f'../model_config/best_finetune_from_{gas_data_percent}_gas_to_ev_lstm.pt'
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
class BiLSTMRegressor(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, dropout=0.2):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.bidirectional = True
        self.num_directions = 2

        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=dropout if num_layers > 1 else 0.0,
            bidirectional=self.bidirectional,
            batch_first=True
        )
        self.fc = nn.Linear(hidden_size * self.num_directions, 1)

    def forward(self, x):
        batch_size = x.size(0)

        h0 = torch.zeros(self.num_layers * self.num_directions, batch_size, self.hidden_size, device=x.device)
        c0 = torch.zeros(self.num_layers * self.num_directions, batch_size, self.hidden_size, device=x.device)

        out, _ = self.lstm(x, (h0, c0))

        last_time_step = out[:, -1, :]

        out = self.fc(last_time_step)
        return out

def init_weights(m):
    if isinstance(m, nn.Linear) or isinstance(m, nn.LSTM):
        for param in m.parameters():
            if param.dim() > 1:
                nn.init.xavier_uniform_(param)

model = BiLSTMRegressor(nfeat, hidden_size=43, num_layers=3)
model.apply(init_weights)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model.to(device)

criterion = nn.MSELoss()
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-5)
num_epochs = 5

# model = TransformerRegressor(feature_dim=nfeat).to(device)
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