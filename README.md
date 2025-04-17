# 🚗 EV Energy Prediction Pipeline

A simple and effective pipeline for predicting electric vehicle (EV) energy consumption based on time-series driving data.

---

## 📈 Pipeline Overview

| Step        | Script           | Output |
|:------------|:-----------------|:-------|
| Pretraining | `pretrain.py`     | `best_pretrain.pt` |
| Fine-tuning | `finetune.py`     | `best_finetune.pt` |
| Evaluation  | `eval.py`         | Performance metrics |

---

## 📦 Data Format

- **Input shape**: `(samples, seq_length, feature_nums)`
- **Output shape**: `(samples, 1)`

### Features:
- `VehId` (vehicle ID) 🚫 (not used for training)
- `Speed`
- `Temperature`
- `Elevation`
- `Speed Limit Class`
- `Latitude`
- `Longitude`

> ⚠️ **Important**:  
> - `VehId` should be **removed before training** the model!  
> - `VehId` is only kept for **identification purposes**.  
> - In the future, `VehId` can help us integrate **vehicle-specific metadata** (e.g., engine configurations) to improve prediction performance.

---

## 🧪 Evaluation Results
```text
========================================
Evaluation Results on EV Data (ID(s): [541])
----------------------------------------
  Total sequences evaluated: 33
  Ground truth mean energy usage : 0.1723 kWh
  Ground truth std deviation     : 0.1077 kWh

  MSE  = 0.006470 (kWh²)
  RMSE = 0.080437 (kWh)
  MAE  = 0.058496 (kWh)
========================================

Sample predictions vs. ground truth:
  pred = 0.0998 kWh, true = 0.0915 kWh
  pred = 0.1687 kWh, true = 0.1578 kWh
  pred = 0.1332 kWh, true = 0.1735 kWh
  pred = 0.0226 kWh, true = 0.0820 kWh
  pred = 0.2451 kWh, true = 0.2605 kWh
========================================

---
```

## ⚙️ Installation

It is recommended to use **conda** for environment management.

```bash
# Create a new environment
conda create -n ev-transformer python=3.9

# Activate the environment
conda activate ev-transformer

# Install required packages
pip install numpy pandas torch tqdm scikit-learn

...
```


## 🚀 Running the Pipeline
```bash
# Step 1: Pretraining
python pretrain.py

# Step 2: Fine-tuning
python finetune.py

# Step 3: Evaluation
python eval.py