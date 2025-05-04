import os
import numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from sklearn.preprocessing import StandardScaler

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

DATA_FILE_GAS = 'timeseries_dataset_icv_hev.npz'
DATA_FILE_EV = 'timeseries_dataset_phev_ev.npz'

gas_data_percent = 1.0  # Adjust as needed

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
    n_need  = int(n_samples * dataset_percent)
    X, y = X[indices[:n_need]], y[indices[:n_need]]
    if attribute == "gas":
        mask_pos = y > 0
        X, y = X[mask_pos], y[mask_pos]
    return X, y

# Load and preprocess
X_src, y_src = load_and_filter(DATA_FILE_GAS, GAS_IDS, gas_data_percent, 100, 'gas')
X_tgt, y_tgt = load_and_filter(DATA_FILE_EV, EV_IDS, gas_data_percent, 100, 'ev')

nfeat = X_src.shape[2]
seq_len = X_src.shape[1]

# Standardization
scaler = StandardScaler().fit(X_src.reshape(-1, nfeat))
def scale(X): return scaler.transform(X.reshape(-1, nfeat)).reshape(-1, seq_len, nfeat)
X_src = scale(X_src)
X_tgt = scale(X_tgt)

# Flatten time-series (e.g., mean across time axis) for t-SNE
X_src_flat = X_src.mean(axis=1)
X_tgt_flat = X_tgt.mean(axis=1)

# Subsample source to match target size
n_target = len(X_tgt_flat)
np.random.seed(42)
idx_src = np.random.choice(len(X_src_flat), n_target, replace=False)
X_src_sampled = X_src_flat[idx_src]

# Combine and label
X_all = np.vstack((X_src_sampled, X_tgt_flat))
y_all = np.array([0]*n_target + [1]*n_target)  # 0 = gas, 1 = EV

# t-SNE
tsne = TSNE(n_components=2, perplexity=30, random_state=42)
X_tsne = tsne.fit_transform(X_all)

# Plot
plt.figure(figsize=(8, 6))
plt.scatter(X_tsne[y_all == 0, 0], X_tsne[y_all == 0, 1], label='Gas Cars', alpha=0.6)
plt.scatter(X_tsne[y_all == 1, 0], X_tsne[y_all == 1, 1], label='EVs', alpha=0.6)
plt.title("2D t-SNE: Gas vs EV Feature Space")
plt.legend()
plt.grid(True)
plt.savefig(f'figures/tSNE_visualization.pdf',dpi=400)