import os
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt

script_dir = os.path.dirname(os.path.abspath(__file__))
os.chdir(script_dir)

# 1) Vehicle ID list for PHEV
ICV_IDS  = [129,130,133,145,147,160,176,181,192,193,216,218,232,248,251,257,266,275,278,285,306,313,
            315,318,329,330,356,409,413,414,416,461,473,480,482,483,485]


GAS_IDS   = ICV_IDS
DATA_FILE = '../timeseries_dataset_icv_hev.npz'  # NPZ containing both ICV and HEV
dataset_percent = 1.0 # using N% data to do the pre-training, choose in from [0.2,0.4,0.6,0.8,1.0]

# 2) Loader + filter by ID + drop NaNs
def load_and_filter(path, id_list, dataset_percent, seed):
    assert os.path.isfile(path), f"{path} not found!"
    data = np.load(path)
    X, y = data['X'], data['y']
    veh = X[:, 0, 0].astype(int)                # VehId at timestep 0
    mask_id = np.isin(veh, id_list)
    X, y = X[mask_id], y[mask_id]
    mask_nan = ~np.isnan(X).any(axis=tuple(range(1, X.ndim)))
    X, y = X[mask_nan], y[mask_nan]
    n_samples = X.shape[0]
    np.random.seed(seed)
    indices = np.random.permutation(n_samples)
    n_need  = int(n_samples*dataset_percent)
    X, y = X[indices[:n_need]], y[indices[:n_need]]
    mask_pos = y > 0
    X, y = X[mask_pos], y[mask_pos]
    return X, y

# 3) Load PHEV data
X_pre, y_pre = load_and_filter(DATA_FILE, GAS_IDS, dataset_percent, 100)
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

# 8) Training + checkpointing

input_size = X_tr.shape[2]
hidden_size = 43

model = BiLSTMRegressor(input_size, hidden_size, num_layers=3)
model.apply(init_weights)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model.to(device)

criterion = nn.MSELoss()
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-5)
num_epochs = 51

train_losses = []
val_losses   = []

best_val = float('inf')
ckpt_path = f'../model_config/best_train_gas_itself_lstm.pt'
for epoch in range(1, num_epochs):
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

    train_losses.append(avg_train)
    val_losses.append(avg_val)

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