import os
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
from sklearn.model_selection import train_test_split
import re
import matplotlib.pyplot as plt

script_dir = os.path.dirname(os.path.abspath(__file__))
os.chdir(script_dir)

# 1) Vehicle ID list for EV
EV_IDS   = [10, 455]
gas_data_percent = 0.2  # choose it from [0.2,0.4,0.6,0.8,1.0]
DATA_FILE = '../timeseries_dataset_phev_ev.npz'
PRETRAIN_CKPT = f'../model_config/best_pretrain_gas_{gas_data_percent}_lstm.pt'

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

# # 6) Scale EV data using pretrained scaler
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

# 10) Load model state & freeze
input_size = X_tr.shape[2]
hidden_size = 43

model = BiLSTMRegressor(input_size, hidden_size, num_layers=3).to(device)
model.load_state_dict(ckpt['model_state'])

def unfreeze_bilstm_layers(model, num_layers, unfreeze_last_n):
    for name, param in model.named_parameters():
        if 'fc' in name:
            param.requires_grad = True
        elif 'lstm' in name:
            match = re.search(r'_l(\d+)', name)
            if match:
                layer_idx = int(match.group(1))
                if layer_idx >= num_layers - unfreeze_last_n:
                    param.requires_grad = True
                else:
                    param.requires_grad = False
            else:
                param.requires_grad = False
        else:
            param.requires_grad = False

unfreeze_bilstm_layers(model, num_layers=3, unfreeze_last_n=1)

# 11) Fine-tuning setup
def optimizer_and_criterion():
    opt = torch.optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=1e-4
    )
    return opt, nn.MSELoss()

optimizer, criterion = optimizer_and_criterion()

train_losses = []
val_losses  = []

# 12) Fine-tuning loop + checkpointing
best_val = float('inf')
ft_ckpt = f'../model_config/best_finetune_from_{gas_data_percent}_gas_to_ev_lstm.pt'
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
plt.savefig(f'../figures/finetune_train_loss_curves_{gas_data_percent}_lstm_partial_finetune.pdf',dpi=400)