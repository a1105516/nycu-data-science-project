# 執行計畫：Toxic Comment Multi-label Classification

Group A · Team 5（吳傢澂、歐冠廷、吳承羲）

本文件把 progress 簡報中的研究設計展開成可以直接照做的執行步驟。凡是簡報沒有定義、由本計畫補充的設定，都會標註 **（補充）**。

---

## 0. 文件依據與現況

**依據**
- `group_A_team_05_progress_revision.pptx`
  - slides 56–62：Analysis workflow 與各 Process
  - slides 76–77：Evaluation Strategy、超參數
- `presentation/graph/08_EDA_Supplement/eda_results.md`：EDA 補充結果，由 `eda_supplement.py` 產生

**現況**

| 階段 | 狀態 | 備註 |
|---|---|---|
| Data Understanding | ✅ 完成 | EDA 與補充分析已完成，結論整理於第 3 節 |
| Data Preparation | ✅ 完成 | `src/prepare_data.py`；產出與統計見 `data/processed/data_summary.md` |
| Data Modeling | ⬜ | |
| Evaluation | ⬜ | |
| Refinement | ⬜ | 等主實驗完成後決定要做哪些 |
| Deployment（選做） | ⬜ | |

---

## 1. 實驗設計總覽

### 1.1 研究問題
- **RQ1**：在相同 model architecture 與 evaluation protocol 下，加入 Civil Comments 能否提升 Jigsaw multi-label classification performance？
- **RQ2**：外部資料採用 hard-label 或 soft-label harmonization，是否會造成不同的 performance，特別是在 minority toxicity labels 上？

### 1.2 訓練設定

| 代號 | 訓練資料 | 外部資料的標籤 |
|---|---|---|
| E0 | Jigsaw train split | — |
| E1 | Jigsaw train split + Civil 樣本 | Hard：score ≥ 0.5 → 1，其餘 → 0 |
| E2 | Jigsaw train split + Civil 樣本 | Soft：保留原始 0–1 score |

E1 和 E2 使用**同一批** Civil 樣本，兩者唯一的差別是標籤處理方式。

### 1.3 實驗矩陣

| Model | E0 | E1 | E2 |
|---|---|---|---|
| TF-IDF + LR（baseline） | ✔ | — | — |
| BERT | ✔ | ✔ | ✔ |
| RoBERTa | ✔ | ✔ | ✔ |
| **DeBERTa-v3（主要分析對象）** | ✔ | ✔ | ✔ |

總計：1 個 baseline + 9 個 Transformer runs。

### 1.4 控制變因

| 項目 | 固定方式 |
|---|---|
| 資料切分 | 只切一次（seed 42），所有 runs 共用 |
| 外部樣本 | 只抽一次（seed 42），E1 和 E2 共用 |
| 模型起始版本 | 每個架構固定使用同一個 Hugging Face checkpoint |
| 超參數 | 同一份設定（第 5.2 節） |
| Random seed | 42 |
| Checkpoint 選擇 | validation 上 Mean Column-wise ROC-AUC 最高者 |
| 最終評估資料 | Jigsaw official labeled test，不參與訓練和 checkpoint 選擇 |

### 1.5 已知限制
E1/E2 的訓練資料量是 E0 的兩倍。三組都跑 3 epochs，所以 E1/E2 的 optimizer steps 也是兩倍。如果觀察到增益，無法完全區分是來自「更多資料」還是「更多 steps」。報告中要寫明這一點。Refinement 的比例消融（第 7 節）可以提供部分佐證。

---

## 2. 環境與專案結構

### 2.1 運算環境
- GPU：NVIDIA RTX 5070 12GB（Blackwell）
- Python 3.12，使用 repo 內的 `.venv`
- PyTorch 必須安裝 CUDA 12.8 版的 wheel，較舊的 CUDA 版本不支援 Blackwell

```bash
.venv/Scripts/python -m pip install torch --index-url https://download.pytorch.org/whl/cu128
```

```bash
.venv/Scripts/python -m pip install -r requirements.txt
```

### 2.2 套件
- 已安裝：pandas、numpy、scikit-learn、matplotlib、datasets
- 需新增：
  - torch（cu128）
  - transformers、accelerate
  - iterative-stratification（multi-label stratified split）
  - sentencepiece、protobuf（DeBERTa-v3 tokenizer 需要）
  - pyarrow（parquet）
  - gradio（demo 用，選做）

`requirements.txt` 中要固定版本號。

### 2.3 目錄結構

```
nycu-data-science-project/
├── datasets/                 # 原始資料（已存在，gitignored）
│   ├── jigsaw/
│   └── civil_comments/
├── data/processed/           # prepare_data.py 產出（gitignored）
├── outputs/                  # 每個 run 一個資料夾（gitignored）
│   ├── {model}_{setting}/
│   └── smoke/{model}_{setting}/
├── src/
│   ├── common.py             # 路徑、LABELS、label mapping、clean()、metric 函式、build_augmented()、fingerprint()
│   ├── prepare_data.py       # Phase 2
│   ├── baseline.py           # Phase 3：TF-IDF + LR
│   ├── train.py              # Phase 3：--model {bert,roberta,deberta} --setting {e0,e1,e2}
│   ├── evaluate.py           # Phase 4：讀 predictions，產出結果表
│   └── demo.py               # Phase 6（選做）
├── docs/
│   ├── plan.md
│   └── run_guide.md          # 給在其他電腦跑實驗的隊友
└── requirements.txt
```

- `.gitignore` 要加上 `data/` 和 `outputs/`。
- 可以沿用 `eda_supplement.py` 的寫法：
  - `LABELS`／`CIVIL` 的對應
  - 用 `str.strip()` 當去重 key
  - 用 set 交集計算 overlap

---

## 3. Phase 1：Data Understanding（已完成）

**Input**：Jigsaw 與 Civil Comments 的 comments、labels、annotation scores。
**Output**：資料品質摘要、類別不平衡概況、標籤對應規則。

下表整理各項發現，以及它們對後續步驟的影響：

| 發現 | 影響的下游決策 |
|---|---|
| 約 90% 的留言沒有任何 label；threat 只佔 0.30%，severe_toxic 佔 1.00% | 主要指標用 ROC-AUC，並一併報告 PR-AUC 與 Macro F1；Accuracy 不採用 |
| 有正例的留言共 16,225 則，對應 35,098 個 positive labels | 確認是 multi-label 問題：六個 sigmoid 輸出，loss 用 BCE |
| Jigsaw train 有 4.05% 留言超過 256 words；Civil 幾乎沒有 | max_len 設為 256，截斷主要影響 Jigsaw 的長留言 |
| Jigsaw train 沒有完全重複的留言；normalize 後有 251 筆近似重複 | 去重只針對 strip 後完全相同的文字；近似重複保留 |
| Civil train 有 3 筆空值 | 建立 augmentation pool 前先移除 |
| Civil train 和 Civil test 有 1,217 個相同文字、和 Jigsaw test 有 24 個、和 Jigsaw train 有 38 個 | 建立 pool 時排除和 Civil test、Jigsaw val／test 相同的文字（第 4 節） |
| Civil severe_toxicity 的 score ≥ 0.5 只有 13 筆，但 5.8% 的 score > 0 | Hard label 幾乎丟掉這一類的資訊，這正是比較 E1／E2 的動機 |
| 1:1 抽樣時，threat、identity_hate 的 soft-label mass 約是 hard positives 的 3–4 倍 | 預期 E2 在 rare labels 上的表現會和 E1 不同，是分析重點 |
| Jigsaw test 中 threat 只有 211 個正例、severe_toxic 只有 367 個 | Rare label 的差異要保守解讀，並在 Refinement 階段做 bootstrap CI |

---

## 4. Phase 2：Data Preparation

**程式**：`src/prepare_data.py`
**原則**：所有筆數都由程式計算。EDA 的數字只用來做 sanity check（印出比較），不寫死在程式裡。

### 4.1 步驟

**Step 1：載入資料**
- Jigsaw train：`comment_text` 和六個 labels
- Jigsaw test：`test.csv` 和 `test_labels.csv` 依 `id` merge，再去掉任一 label 為 -1 的列（EDA 參考值為 63,978 筆）
- Civil train／test：只讀 `text` 和六個對應欄位

**Step 2：基本清理**
- `clean(text) = text.strip()`，定義在 `common.py`，所有 scripts 共用。
- 只做 strip：不壓縮空白、不轉小寫、不移除標點或 URL。大小寫交給各模型的 tokenizer 處理。

**Step 3：Jigsaw 切分**
- 依 `clean()` 後的文字去重，保留第一筆（EDA 結果為 0 筆）。
- 使用 `MultilabelStratifiedShuffleSplit(n_splits=1, test_size=0.1, random_state=42)` 切出 train／val。
- 印出並寫入 `data_summary.md`：
  - 兩邊的實際筆數
  - 各 label 在 train／val 的正例數與比例差異

**Step 4：建立 Civil augmentation pool**（只從 Civil train split 建立）
1. 去掉 `text` 為空值的列。
2. 排除 `clean(text)` 和以下任一集合相同的列：Civil test、Jigsaw val、Jigsaw test。
3. Overlap 分兩個數字記錄：
   - 「重疊的不同文字數」：例如 Civil train ∩ Civil test = 1,217 個文字
   - 「實際被排除的列數」：Civil train 本身有重複文字，所以列數 ≥ 文字數
4. Pool 內部**不去重**，維持原始資料分布。
5. 和 Jigsaw train 相同的文字不排除，因為兩者都是訓練資料，不構成 leakage。數量照樣記錄。

**Step 5：Label mapping 與抽樣**

| Civil Comments | Jigsaw target |
|---|---|
| toxicity | toxic |
| severe_toxicity | severe_toxic |
| obscene | obscene |
| threat | threat |
| insult | insult |
| identity_attack | identity_hate |
| sexual_explicit | （排除） |

- 抽樣數 `n = len(jigsaw_train_split)`，即 1:1。
- 用 seed 42 對 pool 做一次 permutation，取前 n 筆。
  - 這樣 Refinement 的比例消融（0.5:1、2:1）只要取前 0.5n／2n 筆，樣本之間就是巢狀關係（0.5:1 ⊂ 1:1 ⊂ 2:1）。
- E1：`(score >= 0.5).astype(float32)`。門檻沿用 Jigsaw Unintended Bias 競賽的判定方式。
- E2：`score.astype(float32)`。
- 外部樣本和 Jigsaw train split 合併後，用 seed 42 shuffle。E1、E2 用同一個 shuffle 順序。

**Step 6：標籤型別**
- 所有標籤欄位一律存成 **float32**，包括 Jigsaw 的 0/1、E1 的 hard label、E2 的 soft score。
- 原因：`BCEWithLogitsLoss` 和 HF 的 `problem_type="multi_label_classification"` 都要求 float targets。型別統一後，三種設定可以走完全相同的訓練程式碼。

**Step 7：Civil test（cross-dataset evaluation 用）**
- 標籤以 score ≥ 0.5 二值化。
- severe_toxicity 在 test split 中沒有正例，所以 cross-dataset evaluation 只計算其餘五類。
- 移除文字和 Jigsaw train（含 val）相同的列，避免 leakage。實際結果是 10 個文字、13 列，都是 "Thank you very much." 這類常見短句。

### 4.2 產出

| 檔案 | 內容 | 欄位 |
|---|---|---|
| `data/processed/train_e0.parquet` | Jigsaw train split | `text`、六個 labels（float32）、`source` |
| `data/processed/train_e1.parquet` | Jigsaw train + Civil hard | 同上 |
| `data/processed/train_e2.parquet` | Jigsaw train + Civil soft | 同上 |
| `data/processed/val.parquet` | Jigsaw val split | `text`、六個 labels |
| `data/processed/test_jigsaw.parquet` | Jigsaw official labeled test | `id`、`text`、六個 labels |
| `data/processed/test_civil.parquet` | Civil test（二值化） | `text`、五個 labels |
| `data/processed/civil_pool_perm.parquet` | permutation 後的完整 pool（保留原始 score） | 給比例消融使用 |
| `data/processed/data_summary.md` | 統計摘要 | 見下方 |

`source` 欄位的值為 `jigsaw` 或 `civil`，用於事後分析。

`data_summary.md` 包含：
- 各檔筆數
- 各 label 的正例數，E2 另外列出 soft-label mass（score 總和）
- 各 overlap 的「文字數」和「被排除列數」
- 和 EDA 參考值的對照

### 4.3 驗收（寫成 `assert`，在 `prepare_data.py` 結尾執行）
- `train_e0/e1/e2` 的 `text` 和 `val`、`test_jigsaw`、`test_civil` 的 `text` 交集都是 0
- `train_e1.text` 和 `train_e2.text` 完全相同，列順序也相同
- 外部樣本數等於 `len(jigsaw_train_split)`
- 所有標籤欄位的 dtype 都是 float32
- E1 外部列的標籤只有 {0, 1}；E2 外部列的標籤落在 [0, 1]
- Jigsaw 部分在 E0、E1、E2 中完全相同

---

## 5. Phase 3：Data Modeling

### 5.1 Baseline：TF-IDF + One-vs-Rest Logistic Regression
- **程式**：`src/baseline.py`
- **資料**：只用 E0
- **特徵**：
  - word analyzer，ngram (1, 2)，`max_features=100_000`，`sublinear_tf=True`
  - char_wb analyzer，ngram (2, 5)，`max_features=100_000`，`sublinear_tf=True`
  - 兩者用 `scipy.sparse.hstack` 合併，共 200,000 維
- **分類器**：每個 label 各訓練一個 `LogisticRegression`（`solver="liblinear"`）
- **調參**：C 從 {0.5, 1, 2, 4} 中，用 val 的 Mean ROC-AUC 挑選
- **輸出**：和 Transformer 相同格式的 prediction 檔（第 5.4 節），讓 `evaluate.py` 統一處理

### 5.2 Transformer models
**程式**：`src/train.py --model {bert,roberta,deberta} --setting {e0,e1,e2} [--smoke] [--batch-size N] [--resume] [--seed N] [--ratio R]`

- `--seed` 和 `--ratio` 是 Refinement 第 3、4 項用的參數，現在先實作，之後不必再改程式（補充）。
  - `--seed` 只換訓練 seed，資料切分和外部樣本維持 seed 42。
  - `--ratio` 從 `civil_pool_perm.parquet` 取前 ratio × n 筆，用和 `prepare_data.py` 相同的 `common.build_augmented()` 組成 E1／E2。ratio 1 就是 `train_e1`／`train_e2`。
  - 輸出資料夾名稱加上後綴，例如 `deberta_e2_r0.5`、`deberta_e0_s43`。

| Model | Hugging Face checkpoint |
|---|---|
| BERT | `bert-base-uncased` |
| RoBERTa | `roberta-base` |
| DeBERTa-v3 | `microsoft/deberta-v3-base` |

- 使用 `AutoModelForSequenceClassification`，設定 `num_labels=6`、`problem_type="multi_label_classification"`。
- Loss 為 `BCEWithLogitsLoss`，可以直接接受 hard targets 和 soft targets，不另外加 class weight 或 resampling。

**超參數**

| 參數 | 設定 | 來源 |
|---|---|---|
| Optimizer | AdamW | 簡報 |
| Learning rate | 2e-5 | 簡報 |
| Weight decay | 0.01 | 簡報 |
| Epochs | 3 | 簡報 |
| Effective batch size | 16 | 簡報 |
| Max sequence length | 256 tokens | 簡報 |
| Random seed | 42（`transformers.set_seed`）。在載入模型**之前**設定，新 classifier head 的初始化也受 seed 控制 | 簡報 |
| LR schedule | linear decay，warmup 6% | （補充） |
| 精度 | bf16 autocast；GPU 不支援 bf16 時自動改用 fp16，實際使用的精度記錄在 `config.json` | （補充） |
| 模型權重 | 一律以 fp32 載入（`dtype=torch.float32`）。transformers v5 預設沿用 checkpoint 的 dtype，而 `deberta-v3-base` 存成 fp16，會導致 fp16 master weights 發散 | （補充） |
| Per-device batch | 16；OOM 時用 `--batch-size 8`，程式自動設 `gradient_accumulation_steps=2`，effective batch 維持 16 | （補充） |
| Padding | dynamic padding（`DataCollatorWithPadding`）＋ length grouping（v5 寫法：`train_sampling_strategy="group_by_length"`）。v5 開啟後連 eval／predict 也會打亂順序，所以 eval 改用 sequential sampler，預測依長度排序後再還原原順序。訓練的 length-grouped 順序只由 (seed, epoch) 決定，中途 `--resume` 的那個 epoch 和沒中斷時完全相同（v5 內建 sampler 在中途接續時順序會錯） | （補充） |
| 推論精度 | 最終的 val／test 預測用 fp32（不經 bf16 autocast），避免 logits 落在 bf16 格點上造成 ROC-AUC 同分；訓練中的 val 評估仍用 bf16 | （補充） |
| Evaluation 頻率 | 每 0.5 epoch 對 val 評估一次，共 6 個候選 checkpoint；訓練結束時 Trainer 會再評估一次最後一步 | （補充） |
| Checkpoint 保存 | 保留 best 和 last：`save_total_limit=2`、`load_best_model_at_end=True`、`metric_for_best_model="mean_roc_auc"` | （補充） |

- 同一個 run 內，所有 setting 都用相同的超參數，不針對單一 setting 調參。
- `save_steps` 要和 `eval_steps` 對齊，這樣 best checkpoint 才一定會被保存。
- last checkpoint 用於中斷後接續：`--resume` 會轉成 `trainer.train(resume_from_checkpoint=True)`。

### 5.3 執行順序
1. **Smoke test**：`--smoke` 只取 1% 的 train、val 和兩個 test（seed 42），跑 1 epoch，輸出到 `outputs/smoke/`。
   - 確認整個流程能跑完：資料讀取、訓練、評估、存檔、輸出 predictions。
   - 量測 samples/s 和峰值 VRAM。
   - 三個模型各跑一次。
2. **DeBERTa-v3**：E0 → E1 → E2（主要分析對象，最先跑）
3. **RoBERTa**：E0 → E1 → E2
4. **BERT**：E0 → E1 → E2

每個 run 結束後馬上確認 `outputs/{model}_{setting}/` 的產出是否完整，再開始下一個。

```bash
.venv/Scripts/python src/train.py --model deberta --setting e0 --smoke
```

```bash
.venv/Scripts/python src/train.py --model deberta --setting e0
```

### 5.4 每個 run 的產出（`outputs/{model}_{setting}/`）

| 檔案 | 內容 |
|---|---|
| `config.json` | 所有參數、資料檔路徑與筆數、套件版本、git commit hash、GPU 名稱 |
| `checkpoint-*/` | best 和 last 兩個 checkpoint |
| `log_history.json` | 訓練 loss 與每次 val 評估的結果（訓練曲線） |
| `preds_val.parquet` | best checkpoint 對 val 的六類機率 |
| `preds_test_jigsaw.parquet` | best checkpoint 對 Jigsaw test 的六類機率 |
| `preds_test_civil.parquet` | best checkpoint 對 Civil test 的機率 |
| `metrics_val.json` | best checkpoint 在 val 上的指標 |
| `runtime.json` | 訓練時間、推論時間、samples/s、峰值 VRAM、best checkpoint |

- Prediction 檔的格式：`row` 欄是對應 `data/processed/{split}.parquet` 的列號，後面是六個 label 的機率（float32）。test_civil 也輸出六類，評估時只用五類。
- `config.json` 記錄訓練資料和每個 eval split 的 data fingerprint（`common.fingerprint()`），以及 Hugging Face 權重的 commit（`checkpoint_revision`）。`evaluate.py` 會和本機的 `data/processed` 比對（訓練資料用 `common.load_train()` 重建後比對），不一致就停止，避免在不同電腦產生的資料上對錯列，或 E1／E2 用到不同批外部樣本。
- `--resume` 時保留原本的 `config.json`，只在 `resume_sessions` 附加這次的環境紀錄。
- baseline 也輸出相同格式，放在 `outputs/tfidf_lr_e0/`。

### 5.5 GPU 時間粗估
- 公式：時間 ≈ epochs × 訓練筆數 ÷ samples/s，再加上 val 評估和三個資料集的推論。
- E1/E2 的訓練筆數約為 E0 的兩倍，所以每個 run 的時間也約為兩倍。
- 9 個 runs 總計粗估 18–25 小時。smoke test 後用實測的 samples/s 更新這個數字。

---

## 6. Phase 4：Evaluation

**程式**：`src/evaluate.py`。只讀 `outputs/*/preds_*.parquet`，不需要重跑模型。
**產出**：
- `outputs/results.md`：第 6.3 節的結果表，以及第 6.4 節的 FP／FN 與翻轉統計
- `outputs/metrics_long.csv`：每個 run × split 的所有指標（一列一個 run 和 split）
- `outputs/val_curves.png`：訓練曲線
- `outputs/error_samples_deberta.csv`：E0 的 threat、identity_hate false negatives 各 20 筆，附 E0／E2 機率，供人工閱讀

Smoke 輸出可以用 `--outputs outputs/smoke` 測試整個評估流程。Bootstrap CI 和 per-label threshold 屬於 Refinement，等 decision gate 後再加。

### 6.1 指標

| 指標 | 定義 | 角色 |
|---|---|---|
| Mean Column-wise ROC-AUC | 六個 label 各自算 `roc_auc_score` 後取平均 | **主要指標**，也是 Kaggle 官方 metric |
| Per-label ROC-AUC | 每個 label 的 ROC-AUC | 看各類別的表現 |
| Per-label PR-AUC | `average_precision_score`，另外列出六類平均 | 在 imbalance 下比 ROC-AUC 更敏感 |
| Macro F1 | 機率以 0.5 為 threshold 後，`f1_score(average="macro", zero_division=0)` | 看實際分類效果 |

### 6.2 評估資料與用途

| 資料 | 用途 |
|---|---|
| Jigsaw val | 只用於 checkpoint 選擇與 baseline 調參 |
| Jigsaw official labeled test | **E0／E1／E2 的最終比較**，以及和文獻、Kaggle 的 benchmark |
| Civil test（五類） | Cross-dataset 表現 |

### 6.3 結果表
1. **主表**：列為 model × setting（含 baseline），欄為 Jigsaw test 上的 Mean ROC-AUC、Mean PR-AUC、Macro F1。
2. **Per-label 表**：每個 model × setting 的六類 ROC-AUC 和 PR-AUC。
3. **Δ 表**（每個 model 各一張）：
   - E1 − E0：hard-label augmentation 的效果（RQ1）
   - E2 − E0：soft-label augmentation 的效果（RQ1）
   - E2 − E1：相同外部樣本下，兩種 harmonization 的差異（RQ2）
   - 每個 Δ 都分 per-label，重點看 severe_toxic、threat、identity_hate。
4. **Cross-dataset 表**：Civil test 上五類的 ROC-AUC 和 PR-AUC。
5. **訓練曲線圖**：每個 run 的 val Mean ROC-AUC 隨訓練進度的變化。

### 6.4 錯誤分析（以 DeBERTa 為主）
- 在 0.5 threshold 下，比較 E0／E1／E2 每個 label 的 FP、FN 數量。
- 找出 E0 → E2 預測翻轉的樣本（FN 變 TP、TN 變 FP），看翻轉集中在哪些 label。
- threat 和 identity_hate 各抽 20 筆 E0 的 FN，人工閱讀，再看 E2 是否改善。
- 觀察 severe_toxic：E1 的外部樣本幾乎全是負例，檢查這類的表現是否因此下降。

### 6.5 與文獻和 Kaggle 比較

| 參考 | 數值 | 備註 |
|---|---|---|
| Rahman et al. (2023)，baseline 參考 | AUROC 0.973 | 傳統 ML，最佳單一設定 |
| Mazari et al. (2024)，SOTA 參考 | AUROC 98.63% | BERT + Bi-LSTM/Bi-GRU ensemble，自行保留 10% 當 validation |
| Kaggle leaderboard | 撰寫報告時查證 | 和官方 labeled test 的 Mean ROC-AUC 同一個基準 |

各論文的切分方式不同，表中數字只作為參考範圍，不能當成同一基準下的排名。

### 6.6 解讀注意
- Jigsaw test 中 threat 只有 211 個正例、severe_toxic 只有 367 個，這兩類的 Δ 容易受少數樣本影響。
- 主實驗每個設定只跑一個 seed。Δ 很小時，不能直接下結論，要搭配第 7 節的 bootstrap CI 或多 seed 結果。

---

## 7. Phase 5：Refinement

**Decision gate**：9 個主實驗 runs 和 Evaluation 都完成後，看結果再決定要做哪些項目。以下四項都是候選。

| 優先 | 項目 | 做法 | 成本 |
|---|---|---|---|
| 1 | Bootstrap 95% CI | 在 Jigsaw test 上做 1,000 次 paired resampling：同一組 row index 同時用於兩個 setting，算 Δ Mean ROC-AUC 和 per-label Δ，取 2.5／97.5 百分位 | 不需重新訓練，只改 `evaluate.py` |
| 2 | Per-label threshold | 在 val 上為每個 label 搜尋 0.05–0.95（step 0.01）中 F1 最高的 threshold，套用到 test，**另外報告**，不取代 0.5 的主結果 | 不需重新訓練 |
| 3 | DeBERTa 多 seed | E0／E1／E2 用 seed 43、44 各再訓練一次，報告 3 個 seed 的 mean ± std；資料切分和外部樣本維持 seed 42，只換訓練 seed | +6 runs，約 8–10 小時 |
| 4 | Augmentation 比例消融 | DeBERTa E1／E2 改用 0.5:1 和 2:1，從 `civil_pool_perm.parquet` 取前 0.5n／2n 筆，觀察增益和資料量的關係 | +4 runs，約 6–10 小時 |

**判斷規則**
- 第 1、2 項成本很低，主實驗完成後直接做。
- 若 DeBERTa 的 Δ（E1−E0、E2−E0 或 E2−E1）的 bootstrap CI 跨 0，或 |Δ Mean ROC-AUC| < 0.002，優先做第 3 項。
- 第 4 項等第 3 項有結果後再決定。

---

## 8. Phase 6：Deployment（選做）

**程式**：`src/demo.py --run outputs/deberta_e2`（run 依 Evaluation 的結果挑選）

- **Input**：使用者輸入的一則英文留言
- **處理**：使用和訓練完全相同的 `clean()`、tokenizer、max_len 256，載入選定 run 的 best checkpoint 做推論
- **Output**：
  - 六個 toxicity probabilities（以長條圖呈現）
  - 用 0.5 threshold 判定的 labels；如果有做 Refinement 第 2 項，可以切換成 per-label threshold
- **介面**：Gradio，在本機執行，不對外部署

```bash
.venv/Scripts/python src/demo.py --run outputs/deberta_e2
```

---

## 9. 風險與對策

| 風險 | 對策 |
|---|---|
| RTX 5070（Blackwell）和舊版 PyTorch 不相容 | 安裝 cu128 wheel；smoke test 第一步先確認 `torch.cuda.is_available()` 和 bf16 可用 |
| 12GB VRAM 在 batch 16、長度 256 時 OOM | per-device batch 改成 8，搭配 grad-accum 2，effective batch 維持 16；必要時開 gradient checkpointing |
| 長時間訓練中斷 | 保留 last checkpoint，用 `--resume` 接續 |
| E1 的 severe_toxic 外部樣本幾乎全是負例，可能拉低該類表現 | 這是預期內的現象，正好用來比較 E1 和 E2，在錯誤分析中討論 |
| 單一 seed 下差異不顯著 | 使用 Refinement 的 bootstrap CI 和多 seed |
| 時程落後 | 先砍 BERT 的 E1／E2，確保 DeBERTa 三組完整 |
| Civil train CSV 較大（約 600MB） | 只讀需要的欄位；pool 處理完存成 parquet，之後不再讀 CSV |

---

## 10. 產出物與重現性 checklist

**資料**
- [x] `data/processed/*.parquet`，以及 `data_summary.md`
- [x] `prepare_data.py` 的 assert 全部通過

**模型（每個 run）**
- [ ] `config.json`：參數、資料筆數、套件版本、git commit hash
- [ ] `log_history.json`、`runtime.json`
- [ ] `preds_val`、`preds_test_jigsaw`、`preds_test_civil` 三個 parquet
- [ ] best 和 last checkpoint

**結果**
- [ ] `outputs/results.md`：主表、per-label 表、Δ 表、cross-dataset 表
- [ ] 錯誤分析紀錄
- [ ] Refinement 結果（有做的項目）

**重現性**
- [ ] 所有隨機性都固定為 seed 42：切分、抽樣、shuffle、訓練（Refinement 多 seed 除外）
- [ ] `requirements.txt` 固定版本號
- [ ] 每個 run 都記錄 git commit hash，確保結果對應到特定版本的程式碼
