# 實驗執行指南（給跑實驗的隊友）

這份文件說明如何在你的電腦上跑 Phase 3 的 Transformer 實驗，並把結果傳回來。程式碼和設定都已經固定並 review 過，**請不要修改程式或參數**。遇到問題時，照第 8 節處理，或回報給我們。

設計細節見 [plan.md](plan.md) 第 5 節。這份只講怎麼跑。

---

## 1. 需求

| 項目 | 需求 |
|---|---|
| GPU | NVIDIA，建議 8GB VRAM 以上。要不要加 `--batch-size 8`，以第 4 節 smoke test 量到的 VRAM 為準 |
| 作業系統 | Windows 或 Linux 都可以 |
| Python | 3.12 |
| 硬碟 | 約 35GB 的可用空間，主要是 checkpoints（DeBERTa 每個 run 約 4.2GB，BERT／RoBERTa 約 2.6–2.8GB） |
| 網路 | 第一次跑每個模型時，會從 Hugging Face 下載預訓練權重 |

---

## 2. 安裝

**1. 取得程式**

```bash
git clone https://github.com/calculusfkyou/nycu-data-science-project.git
```

```bash
cd nycu-data-science-project
```

```bash
git switch feature/experiments
```

**2. 建立並啟用虛擬環境**

```bash
python -m venv .venv
```

啟用指令依你的 shell 而定：

| Shell | 指令 |
|---|---|
| Windows PowerShell | `.venv\Scripts\Activate.ps1` |
| Windows Git Bash | `source .venv/Scripts/activate` |
| Linux | `source .venv/bin/activate` |

之後所有指令都在**已啟用**的環境中，直接用 `python` 執行。

**3. 安裝 PyTorch**

先單獨安裝 PyTorch。cu128 版本支援 RTX 20／30／40／50 系列，RTX 50 系列只能用 cu128。

```bash
python -m pip install torch==2.11.0 --index-url https://download.pytorch.org/whl/cu128
```

**4. 安裝其他套件**

```bash
python -m pip install -r requirements.txt
```

**5. 確認 GPU 可用**

```bash
python -c "import torch; print(torch.cuda.get_device_name(0), 'bf16:', torch.cuda.is_bf16_supported(including_emulation=False))"
```

應該要印出你的顯卡名稱。RTX 30 系列以後會是 `bf16: True`。`bf16: False` 也可以跑，程式會自動改用 fp16，但請告訴我們。

---

## 3. 準備資料

原始資料不在 git 裡。請下載後放到下列位置：

```
datasets/
├── jigsaw/                 # Kaggle: Jigsaw Toxic Comment Classification Challenge
│   ├── train.csv
│   ├── test.csv
│   └── test_labels.csv
└── civil_comments/         # Hugging Face: google/civil_comments
    ├── train.csv
    └── test.csv
```

接著產生處理後的資料，大約 30 秒：

```bash
python src/prepare_data.py
```

**重要：確認資料和我們的完全一致。** 打開 `data/processed/data_summary.md`，把「Data fingerprints」表和下表逐一比對：

| File | Fingerprint |
|---|---|
| train_e0 | `a36fdd29a451` |
| train_e1 | `89b35a4e3a66` |
| train_e2 | `5b2ba378adf7` |
| val | `da1f6b1e979b` |
| test_jigsaw | `f045df0c4d76` |
| test_civil | `f9325b3186a1` |
| civil_pool_perm | `58af4f70bc80` |

只要有一個不同，**先不要開始訓練**，把你的 `data_summary.md` 傳給我們。最常見的原因是套件版本不同，或原始資料下載的版本不同。

---

## 4. Smoke test（三個合計約 5 分鐘，不含第一次下載模型的時間）

每個模型先用 1% 的資料跑 1 個 epoch，確認流程完整，並量測速度和 VRAM：

```bash
python src/train.py --model deberta --setting e0 --smoke
```

```bash
python src/train.py --model roberta --setting e0 --smoke
```

```bash
python src/train.py --model bert --setting e0 --smoke
```

每個都要看到最後一行 `Done: ...`。結果在 `outputs/smoke/` 底下。Smoke test 只跑 90 步左右，val 分數低（0.6–0.9）是正常的。

打開 **DeBERTa** 的 `outputs/smoke/deberta_e0/runtime.json`，看 `peak_vram_reserved_gb`（三個模型中 DeBERTa 最吃記憶體）：
- 和你的 VRAM 有 1GB 以上的餘裕：正式 run 用預設設定。
- 接近上限，或 smoke test 出現 `CUDA out of memory`：9 個正式 run **全部**加上 `--batch-size 8`。這不會改變實驗設定，effective batch 仍是 16，資料順序也相同。

---

## 5. 正式實驗（9 個 runs）

**請照這個順序跑。** DeBERTa 是主要分析對象，所以最先跑。

```bash
python src/train.py --model deberta --setting e0
```

```bash
python src/train.py --model deberta --setting e1
```

```bash
python src/train.py --model deberta --setting e2
```

```bash
python src/train.py --model roberta --setting e0
```

```bash
python src/train.py --model roberta --setting e1
```

```bash
python src/train.py --model roberta --setting e2
```

```bash
python src/train.py --model bert --setting e0
```

```bash
python src/train.py --model bert --setting e1
```

```bash
python src/train.py --model bert --setting e2
```

如果 smoke test 判斷需要 `--batch-size 8`，就加在每一行後面，而且 9 個 runs 要一致。

**預估時間**：見第 9 節。E1、E2 的資料量是 E0 的兩倍，時間也約兩倍。

**每個 run 結束後**，確認 `outputs/{model}_{setting}/` 裡有這些檔案：
- `config.json`、`log_history.json`、`runtime.json`、`metrics_val.json`
- `preds_val.parquet`、`preds_test_jigsaw.parquet`、`preds_test_civil.parquet`

**檢查分數**：`Done:` 那行的 best val mean ROC-AUC 應該在 0.98 左右（baseline 是 0.986）。低於 0.95 時，程式會多印一行 `WARNING`，表示訓練很可能發散了。這種情況請回報，不要直接繼續用這個結果。

程式結束前會自動檢查 val 的預測有沒有對齊。檢查沒過會直接報錯，這種情況請回報，不要重跑。

---

## 6. 中斷後接續

停電、當機或手動中斷時，用**完全相同的參數**加上 `--resume`：

```bash
python src/train.py --model deberta --setting e1 --resume
```

- 會從最後一個 checkpoint 接續。每 0.5 epoch 存一次，最多損失半個 epoch 的進度。
- 接續後的資料順序和沒中斷時完全相同，不影響實驗結果。
- 參數和原本不同時，程式會拒絕執行。例如原本有 `--batch-size 8`，接續時也要加。
- 如果在第一個 checkpoint 存檔之前就中斷（DeBERTa E0 大約是開始後 20–30 分鐘內），沒有東西可以接續。程式會提示你刪掉資料夾，從頭重跑。
- 想從頭重跑，就刪掉 `outputs/{model}_{setting}/` 整個資料夾。資料夾已存在又沒加 `--resume` 時，程式也會拒絕執行，避免不小心覆蓋。

---

## 7. 回傳結果

全部跑完後，把 `outputs/` 打包，**排除 checkpoint**（checkpoint 很大，我們不需要）。在 repo 根目錄執行，Windows 和 Linux 都可以用：

```bash
tar -czf results.tar.gz --exclude=checkpoint-* outputs
```

把 `results.tar.gz` 傳給我們，大小應該在 100MB 以內。

Checkpoint 請先保留，等我們確認結果沒問題再刪。之後做 demo 會用到 DeBERTa 的 best checkpoint。

---

## 8. 常見問題

| 狀況 | 處理方式 |
|---|---|
| `CUDA out of memory` | 刪掉該 run 的資料夾，加上 `--batch-size 8` 重跑；之後每個 run 都加 |
| `CUDA GPU not found` | PyTorch 裝成了 CPU 版。重新執行第 2 節第 3 步 |
| `already exists. Pass --resume...` | 該 run 跑過了。要接續就加 `--resume`，要重跑就先刪資料夾 |
| `--resume needs the same arguments` | 接續時的參數和原本不同，改成和原本一樣 |
| `has no checkpoint yet` | 在第一次存檔前就中斷了。刪掉該 run 的資料夾，不加 `--resume` 重跑 |
| `nothing to resume` | 這個 run 還沒開始過，拿掉 `--resume` |
| `WARNING: best val mean ROC-AUC ... < 0.95` | 訓練很可能發散，請回報 |
| `fingerprint` 不一致 | 見第 3 節，先不要訓練 |
| `val predictions misaligned` | 程式的對齊檢查失敗，請回報，不要重跑 |
| Hugging Face 下載失敗 | 檢查網路後重跑同一行指令即可，不需要 `--resume` |
| Loss 變成 `nan` | 通常是 fp16 搭配 DeBERTa 的數值問題，請回報 |

---

## 9. 本機實測數據（參考）

以下是在 RTX 4060 Ti 8GB（bf16）上跑 smoke test 量到的數字。正式 run 的時間由 samples/s 推算，只是粗估；RTX 5070 應該會更快。

| Model | Train samples/s | 峰值 VRAM（reserved） | 預估訓練時間 E0／E1／E2 |
|---|---|---|---|
| DeBERTa-v3 | 56–77 | 8.0GB | 約 2h／3h／3h |
| RoBERTa | 91–137 | 4.3GB | 約 1.3h／2h／2h |
| BERT | 126–144 | 4.3GB | 約 1h／1.4h／1.4h |

- 以上都是預設的 `--batch-size 16`。BERT 也用 8 測過 grad-accum，兩者結果等價；正式 run 請 9 個都用同一個值。
- smoke 只跑 90 步，samples/s 波動不小，表中給範圍。
- 9 個 runs 合計約 17 小時，再加上 val 評估和最後的 fp32 預測（DeBERTa 每個 run 約 30 分鐘，BERT／RoBERTa 較短），約 20 小時。
- 峰值 VRAM 出現在最後的 fp32 預測。DeBERTa 用到 8.0GB，12GB 的卡用預設設定即可；8GB 的卡建議全部加 `--batch-size 8`。

Refinement 階段可能還會請你跑額外的 runs，例如不同 seed 或不同外部資料比例。屆時會另外提供指令。
