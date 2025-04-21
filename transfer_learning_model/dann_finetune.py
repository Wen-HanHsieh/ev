import os
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from itertools import cycle
import matplotlib.pyplot as plt

script_dir = os.path.dirname(os.path.abspath(__file__))
os.chdir(script_dir)

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
ckpt = f'../model_config/best_pretrain_gas_1.0_transformer.pt'


BATCH_SIZE = 64
EPOCHS = 200
LR = 1e-4
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def load_and_filter(path, id_list, dataset_percent, seed, attribute):
    assert os.path.isfile(path), f"{path} not found!"
    data = np.load(path)
    X, y = data['X'], data['y']
    veh = X[:, 0, 0].astype(int)
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

X_src, y_src = load_and_filter(DATA_FILE_GAS, GAS_IDS, gas_data_percent, 100 ,'gas')
X_tgt, y_tgt = load_and_filter(DATA_FILE_EV, EV_IDS, gas_data_percent, 100, 'ev')
X_src = X_src[..., 1:]
X_tgt = X_tgt[..., 1:]

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
ckpt = torch.load(ckpt,
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

def dann_lambda_schedule(epoch, max_epochs):
    p = epoch / max_epochs
    return min(0.3, 2. / (1. + np.exp(-10 * p)) - 1.)

model = DANNTransformer(feature_dim=nfeat).to(DEVICE)

# Freeze first 2 encoder layers
for i, layer in enumerate(model.encoder.layers):
    if i < 2:
        for param in layer.parameters():
            param.requires_grad = False

optimizer = torch.optim.AdamW(model.parameters(), lr=LR)
loss_reg = nn.MSELoss()
loss_dom = nn.CrossEntropyLoss()
tgt_iter = cycle(tgt_train_loader)
best_val = float('inf')
ft_ckpt = f'../model_config/best_finetune_from_gas_to_ev_dann.pt'
alpha = 0.2
train_losses = []
val_losses = []
for epoch in range(1, EPOCHS + 1):
    lambda_ = dann_lambda_schedule(epoch, EPOCHS)
    print(f"[Epoch {epoch:02d}] λ = {lambda_:.4f}, α = {alpha}")
    model.train()
    total_loss = 0

    for (Xb_src, yb_src), (Xb_tgt, yb_tgt) in zip(src_loader, tgt_train_loader):
        Xb_src, yb_src = Xb_src.to(DEVICE), yb_src.to(DEVICE)
        Xb_tgt, yb_tgt = Xb_tgt.to(DEVICE), yb_tgt.to(DEVICE)

        y_pred_src, dom_pred_src = model(Xb_src, lambda_)
        y_pred_tgt, dom_pred_tgt = model(Xb_tgt, lambda_)

        loss_y_src = loss_reg(y_pred_src, yb_src)
        loss_y_tgt = loss_reg(y_pred_tgt, yb_tgt)

        domain_label_src = torch.zeros(Xb_src.size(0), dtype=torch.long, device=DEVICE)
        domain_label_tgt = torch.ones(Xb_tgt.size(0), dtype=torch.long, device=DEVICE)

        loss_dom_src = loss_dom(dom_pred_src, domain_label_src)
        loss_dom_tgt = loss_dom(dom_pred_tgt, domain_label_tgt)

        loss = loss_y_src + lambda_ * (loss_dom_src + loss_dom_tgt) + alpha * loss_y_tgt

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * Xb_src.size(0)

    avg_tr = total_loss / len(tgt_train_loader.dataset)

    # model.eval()
    # val_loss = 0
    # with torch.no_grad():
    #     for Xb, yb in tgt_val_loader:
    #         Xb, yb = Xb.to(DEVICE), yb.to(DEVICE)
    #         y_pred, _ = model(Xb, lambda_=0.0)
    #         val_loss += loss_reg(y_pred, yb).item() * Xb.size(0)

    # avg_val = val_loss / len(tgt_val_loader.dataset)

    model.eval()
    val_loss = 0
    y_preds = []
    y_true = []
    with torch.no_grad():
        for Xb, yb in tgt_val_loader:
            Xb, yb = Xb.to(DEVICE), yb.to(DEVICE)
            y_pred, _ = model(Xb, lambda_=0.0)
            val_loss += loss_reg(y_pred, yb).item() * Xb.size(0)
            y_preds.append(y_pred.cpu().numpy())
            y_true.append(yb.cpu().numpy())
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

plt.figure(figsize=(8, 5))
plt.plot(train_losses, label='Train Loss')
plt.plot(val_losses, label='Validation Loss')
plt.xlabel('Epoch')
plt.ylabel('MSE Loss')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig(f'../figures/dann_train_loss_curves_{gas_data_percent}_partial_finetune.pdf',dpi=400)