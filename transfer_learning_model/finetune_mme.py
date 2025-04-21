import os
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from itertools import cycle

script_dir = os.path.dirname(os.path.abspath(__file__))
os.chdir(script_dir)

# 1) Vehicle ID list for EV
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
EV_IDS   = [10, 455]
GAS_IDS   = ICV_IDS + HEV_IDS
gas_data_percent = 1.0  # choose it from [0.2,0.4,0.6,0.8,1.0]
DATA_FILE_GAS = '../timeseries_dataset_icv_hev.npz'  # NPZ containing both ICV and HEV
DATA_FILE_EV  = '../timeseries_dataset_phev_ev.npz'
PRETRAIN_CKPT = f'../model_config/best_pretrain_gas_{gas_data_percent}_transformer.pt'

# 2) Loader + filter by ID + drop NaNs
def load_and_filter(path, id_list, dataset_percent, seed, attribute):
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
    if attribute == "gas":
        mask_pos = y > 0
        X, y = X[mask_pos], y[mask_pos]
    return X, y

# 3) Load EV data
X_src, y_src = load_and_filter(DATA_FILE_GAS, GAS_IDS, gas_data_percent, 100 ,'gas')
X_tgt, y_tgt = load_and_filter(DATA_FILE_EV, EV_IDS, gas_data_percent, 100, 'ev')
# 4) Drop VehId column
X_src = X_src[..., 1:]
X_tgt = X_tgt[..., 1:]

# 5) Load pretrained checkpoint (with scaler)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
ckpt = torch.load(PRETRAIN_CKPT,
                  map_location=device,
                  weights_only=False)
scaler  = ckpt['scaler']
nfeat   = ckpt['nfeat']
seq_len = ckpt['seq_len']

# 6) Scale EV data using pretrained scaler
flat = X_tgt.reshape(-1, nfeat)
flat = scaler.transform(flat)
X_tgt = flat.reshape(-1, seq_len, nfeat)

# 7) Train/val split for fine-tuning
X_tgt_tr, X_tgt_val, y_tgt_tr, y_tgt_val = train_test_split(
    X_tgt, y_tgt, test_size=0.1, random_state=42
)

# 8) DataLoaders
batch_size = 64
tgt_train_ds = TensorDataset(
    torch.from_numpy(X_tgt_tr).float(),
    torch.from_numpy(y_tgt_tr).float().unsqueeze(-1)
)
tgt_val_ds   = TensorDataset(
    torch.from_numpy(X_tgt_val).float(),
    torch.from_numpy(y_tgt_val).float().unsqueeze(-1)
)

src_ds       = TensorDataset(
    torch.from_numpy(X_src).float(),
    torch.from_numpy(y_src).float().unsqueeze(-1)
)

tgt_train_loader = DataLoader(tgt_train_ds, batch_size, shuffle=True)
tgt_val_loader   = DataLoader(tgt_val_ds,   batch_size)

src_loader       = DataLoader(src_ds)

# 9) Model definitions (same as pretrain)
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
    def forward(self, x, return_features=False):
        x = self.input_proj(x)
        x = self.pos_enc(x)
        x = self.encoder(x)
        x_pooled = x.mean(dim=1)
        if return_features:
            return x_pooled
        return self.regressor(x_pooled)

def gaussian_kernel(x, y, sigma = 1.0):
    x = x.unsqueeze(1) # (N, 1, D)
    y = y.unsqueeze(0) # (1, M, D)
    dist = ((x - y) ** 2).sum(2)
    return torch.exp(-dist / (2 * sigma ** 2))

def mmd_loss(x, y, sigma = 1.0):
    xx = gaussian_kernel(x, x, sigma)
    yy = gaussian_kernel(y, y, sigma)
    xy = gaussian_kernel(x, y, sigma)
    return xx.mean() + yy.mean() - 2 * xy.mean()

# 10) Load model state & freeze
model = TransformerRegressor(feature_dim=nfeat).to(device)
model.load_state_dict(ckpt['model_state'])

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
ft_ckpt = f'../model_config/best_finetune_from_gas_to_ev_mme.pt'
lambda_mmd = 0.5
for epoch in range(1, 501):
    model.train()
    total_loss = 0
    for (x_src, y_src), (x_tgt, y_tgt) in zip(src_loader, tgt_train_loader):
        x_src,y_src = x_src.to(device),y_src.to(device)
        x_tgt,y_tgt = x_tgt.to(device),y_tgt.to(device)

        feat_src    = model(x_src,return_features=True)
        feat_tgt    = model(x_tgt,return_features=True)

        preds_src   = model.regressor(feat_src)
        preds_tgt   = model.regressor(feat_tgt)

        loss_src    = criterion(preds_src, y_src)
        loss_tgt    = criterion(preds_tgt, y_tgt)
        loss_mmd    = mmd_loss(feat_src, feat_tgt)
        loss        = loss_src + loss_tgt + lambda_mmd  * loss_mmd
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * x_tgt.size(0)        
    avg_tr = total_loss / len(tgt_train_loader.dataset)

    model.eval()
    val_loss = 0
    with torch.no_grad():
        for Xb, yb in tgt_val_loader:
            Xb, yb = Xb.to(device), yb.to(device)
            val_loss += criterion(model(Xb), yb).item() * Xb.size(0)
    avg_val = val_loss / len(tgt_val_loader.dataset)

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
plt.savefig(f'../figures/finetune_train_loss_curves_mme.pdf',dpi=400)