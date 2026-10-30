# Mazari 2024 Jigsaw 資料切分

這份資料依照 Mazari 等人〈BERT-based ensemble learning for multi-aspect hate speech detection〉第 4.1、4.2 節的設定整理：將 Jigsaw 原始訓練資料按 90%／10% 分成訓練和驗證集，另以 Jigsaw 官方測試資料中有標註的留言作為測試集。論文 DOI：`10.1007/s10586-022-03956-x`。

## 資料和程式

| 檔案 | 用途 |
| --- | --- |
| `split_data.py` | 含中文註解的切分程式 |
| `train.csv` | 原始 Jigsaw 訓練集的 90% |
| `validation.csv` | 原始 Jigsaw 訓練集的 10% |
| `test.csv` | 依 ID 合併測試文字與標籤，排除標籤為 -1 的資料 |
| `split_info.json` | 原始檔雜湊、切分設定、各資料集的筆數與標籤正例數 |

三個輸出 CSV 均包含 `id`、`comment_text`，以及六個 Jigsaw 二元標籤。程式不清洗或修改留言內容。

## 重現方式

預設來源是同一層的 `../jigsaw` 資料夾。程式只使用 Python 標準函式庫。原始資料必須包含 `train.csv`、`test.csv` 與 `test_labels.csv`。

輸出檔已存在時，程式會停止而不覆寫。若要重跑並比較結果，可指定另一個新資料夾：

```powershell
python split_data.py --output-dir ../mazari_2024_split_check
```

也可用 `--input-dir` 指定另一個來源資料夾，用 `--seed` 修改隨機種子。

## 論文未提供的細節

- 論文沒有說明切分種子、是否分層或可重現的列清單。這裡自行選用 `seed=42` 的隨機、不分層切分；符合文中比例，但無法保證與作者的實際資料列一致。
- 驗證筆數以 `ceil(總訓練筆數 × 0.1)` 計算。資料列在各輸出檔仍保持原始順序。
- 論文記載原始訓練集 159,570 筆、測試集 63,977 筆；本地來源為 159,571 筆與 63,978 筆有效測試資料。差異原因未獲說明，因此不任意刪除資料以湊成論文數字。
- 本切分僅使用論文的 Jigsaw 資料，不包含 Civil Comments，亦未執行文字去重或模型訓練。
