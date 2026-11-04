# Toxic Comment Multi-label Classification

NYCU 數據科學專題 · Group A · Team 5（吳傢澂、歐冠廷、吳承羲）

本專題針對 Jigsaw Toxic Comment Classification 的六類 toxicity labels（toxic、severe_toxic、obscene、threat、insult、identity_hate）建立 multi-label classification 模型，並研究兩個問題：
- 加入 Civil Comments 作為外部資料，能不能提升分類表現？
- 外部資料用 hard label 或 soft label，結果會不會不同？

完整的執行計畫見 [docs/plan.md](docs/plan.md)。

## 實驗設定

| 代號 | 訓練資料 | 外部資料的標籤 |
|---|---|---|
| E0 | Jigsaw train split | — |
| E1 | Jigsaw + Civil Comments（1:1） | Hard：score ≥ 0.5 → 1 |
| E2 | Jigsaw + Civil Comments（1:1） | Soft：保留原始 0–1 score |

- 模型：TF-IDF + Logistic Regression（baseline，只跑 E0），以及 BERT、RoBERTa、DeBERTa-v3（主要分析對象）。
- 主要指標：Mean Column-wise ROC-AUC，在 Jigsaw official labeled test 上計算。

## 進度

| 階段 | 狀態 | 程式 |
|---|---|---|
| Data Understanding | ✅ | EDA，在 repo 外完成 |
| Data Preparation | ✅ | `src/prepare_data.py` |
| Data Modeling | 🔄 baseline 完成；Transformer 程式已通過 smoke test，9 個 runs 待跑 | `src/baseline.py`、`src/train.py` |
| Evaluation | 🔄 程式完成，等待實驗結果 | `src/evaluate.py` |
| Refinement | ⬜ | 主實驗完成後再決定 |
| Deployment（選做） | ⬜ | `src/demo.py` |

## 環境

- Python 3.12
- 訓練需要 NVIDIA GPU。PyTorch 要先單獨安裝 CUDA 版本，再裝其他套件。

```bash
python -m venv .venv
```

```bash
.venv/Scripts/python -m pip install torch==2.11.0 --index-url https://download.pytorch.org/whl/cu128
```

```bash
.venv/Scripts/python -m pip install -r requirements.txt
```

在另一台電腦跑實驗的完整步驟見 [docs/run_guide.md](docs/run_guide.md)。

## 資料

原始資料不放進 git，需要自行下載，並放到下列位置：

```
datasets/
├── jigsaw/                 # Kaggle: Jigsaw Toxic Comment Classification Challenge
│   ├── train.csv
│   ├── test.csv
│   └── test_labels.csv
└── civil_comments/         # Hugging Face: google/civil_comments
    ├── train.csv
    ├── validation.csv
    └── test.csv
```

接著產生處理後的資料：

```bash
.venv/Scripts/python src/prepare_data.py
```

執行時間約 30 秒，產出寫到 `data/processed/`（同樣不放進 git）：

| 檔案 | 內容 |
|---|---|
| `train_e0/e1/e2.parquet` | 三種設定的訓練資料；`source` 欄位標示每列來自 Jigsaw 或 Civil |
| `val.parquet` | Jigsaw 的 10% validation（multi-label stratified，seed 42） |
| `test_jigsaw.parquet` | Jigsaw official labeled test（63,978 筆） |
| `test_civil.parquet` | Civil test，以 ≥ 0.5 二值化，只含五類，用於 cross-dataset 評估 |
| `civil_pool_perm.parquet` | 打亂後的 augmentation pool，給 Refinement 的比例消融使用 |
| `data_summary.md` | 筆數、overlap、各 label 正例數，以及驗收結果 |

處理規則：
- 文字清理只做 strip。
- 所有 label 存成 float32。
- 所有隨機步驟固定 seed 42。重跑結果完全相同。
- 腳本結尾會用 assert 檢查 train 和 val／test 之間沒有 leakage。

## 專案結構

```
├── datasets/          # 原始資料（gitignored）
├── data/processed/    # prepare_data.py 產出（gitignored）
├── outputs/           # 訓練與評估產出（gitignored）
├── docs/
│   ├── plan.md        # 執行計畫
│   └── run_guide.md   # 實驗執行指南
├── src/
│   ├── common.py      # 路徑、label 定義、clean()、metrics、共用工具
│   ├── prepare_data.py
│   ├── baseline.py    # TF-IDF + LR（CPU）
│   ├── train.py       # BERT／RoBERTa／DeBERTa-v3（GPU）
│   └── evaluate.py    # 讀 predictions，產出 results.md
└── requirements.txt
```

## 模型與評估

```bash
.venv/Scripts/python src/baseline.py
```

```bash
.venv/Scripts/python src/train.py --model deberta --setting e0
```

```bash
.venv/Scripts/python src/evaluate.py
```

- `train.py` 的完整參數與執行順序見 [docs/run_guide.md](docs/run_guide.md)。
- `evaluate.py` 讀 `outputs/*/preds_*.parquet`，產出 `outputs/results.md`。
