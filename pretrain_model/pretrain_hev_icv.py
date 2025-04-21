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

# 1) Vehicle ID list for ICV and HEV
ICV_IDS  = [2,7,8,12,108,110,116,119,120,123,126,128,129,130,131,132,133,135,137,138,139,140,141,142,143,
             145,147,148,149,153,154,155,156,157,159,160,161,162,163,164,165,167,169,172,174,176,179,180,181,
             184,185,187,189,190,191,192,193,195,196,199,200,202,203,205,206,207,208,209,211,213,214,215,216,
             217,218,222,223,225,228,230,232,233,234,235,237,238,240,243,244,246,247,248,249,250,251,252,254,
             255,257,259,260,262,264,265,266,267,268,269,270,271,273,274,275,276,278,280,282,283,285,286,288,
             289,291,293,297,299,300,301,302,303,304,306,307,308,309,311,312,313,315,318,319,321,323,324,325,
             326,329,330,332,333,334,337,340,345,354,356,359,366,367,370,380,394,400,401,409,413,414,415,416,
             426,429,432,433,434,454,459,460,461,462,463,464,465,466,467,469,470,472,473,476,478,480,482,483,
             484,485,486,487,488,489,490,494,498,500,501,502,503,504,505,506,507,516,517,519,521,522,527,528,
             529,530,531,533,534,535,538,539,540,546,547,548,552,557,562,563,571,575,576,577,578,580,581,584,
             587,588,591,592,595,596,597,598,599,600,601,602,603,604,606,607,608,609,616,618,624,625,630]
HEV_IDS   = [5,115,124,125,150,201,212,220,231,241,242,258,272,292,298,328,338,344,346,347,348,349,350,351,353,
              355,357,360,368,369,372,374,375,376,378,381,382,383,384,385,386,387,389,392,393,397,399,402,403,404,
              405,406,407,410,411,418,422,428,430,435,436,437,438,439,440,441,444,445,447,448,450,451,452,456,458,
              468,474,475,477,526,532,543,549,555,558,564,565,566,573,574,579,605,610]
GAS_IDS   = ICV_IDS + HEV_IDS
DATA_FILE = '../timeseries_dataset_icv_hev.npz'  # NPZ containing both ICV and HEV
dataset_percent = 0.2 # using N% data to do the pre-training, choose in from [0.2,0.4,0.6,0.8,1.0]

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

# 3) Load GAS data
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
            nn.Linear(d_model, d_model*2),
            nn.Linear(d_model*2,1)
        )
    def forward(self, x):
        x = self.input_proj(x)
        x = self.pos_enc(x)
        x = self.encoder(x)
        x = x.mean(dim=1)
        return self.regressor(x)

# 8) Training + checkpointing

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print("Pretraining on GAS data using device:", device)
model     = TransformerRegressor(feature_dim=nfeat).to(device)
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
criterion = nn.MSELoss()

train_losses = []
val_losses   = []

best_val = float('inf')
ckpt_path = f'../model_config/best_pretrain_gas_{dataset_percent}_transformer.pt'
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

    # store the losses
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

# plot train loss and val loss
plt.figure(figsize=(8, 5))
plt.plot(train_losses, label='Train Loss')
plt.plot(val_losses, label='Validation Loss')
plt.xlabel('Epoch')
plt.ylabel('MSE Loss')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig(f'../figures/gas_train_loss_curves_{dataset_percent}_transformer.pdf',dpi=400)