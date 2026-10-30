"""依 Mazari et al. (2024) 的資料比例切分 Jigsaw 資料集。

論文：BERT-based ensemble learning for multi-aspect hate speech detection
DOI: 10.1007/s10586-022-03956-x，PDF 第 8 頁第 4.1、4.2 節。

論文只明確說明從 Jigsaw 訓練資料取 90% 訓練、10% 驗證，
並使用另外的 Kaggle 測試集；沒有交代切分種子或是否分層。
為了讓這份程式可以重現，本程式自行選用 seed 42，隨機切分而不分層。
因此，本切分符合論文報告的比例，但不保證和作者實際使用的資料列相同。

執行：python split_data.py
另存到新資料夾：python split_data.py --output-dir PATH
本程式只需要 Python 標準函式庫，不需要安裝額外套件。
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import platform
import random
from pathlib import Path


# 六個標籤的順序沿用 Jigsaw 原始資料，不把六類視為互斥分類。
LABELS = ["toxic", "severe_toxic", "obscene", "threat", "insult", "identity_hate"]
TEXT_COLUMNS = ["id", "comment_text"]
DATA_COLUMNS = TEXT_COLUMNS + LABELS
LABEL_COLUMNS = ["id"] + LABELS
VALIDATION_FRACTION = 0.10
DEFAULT_SEED = 42

# 預設使用專案既有的 datasets/jigsaw，結果放在本程式所在的新資料夾。
SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_INPUT_DIR = SCRIPT_DIR.parent / "jigsaw"
DEFAULT_OUTPUT_DIR = SCRIPT_DIR


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", type=Path, default=DEFAULT_INPUT_DIR,
                        help="Jigsaw 原始 CSV 資料夾（預設為 ../jigsaw）")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR,
                        help="輸出資料夾（預設為程式所在資料夾；不覆寫既有結果）")
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED,
                        help="自行指定的隨機種子（預設 42；不是論文公布的種子）")
    return parser.parse_args()


def read_csv(path: Path, expected_columns: list[str]) -> list[dict[str, str]]:
    """把每欄讀成字串，避免 ID 前導零或文字 None、NA 被自動轉換。"""
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames != expected_columns:
            raise ValueError(f"{path.name} 的欄位不符：{reader.fieldnames}")
        rows = list(reader)
    if not rows:
        raise ValueError(f"{path.name} 沒有資料列")
    if any(None in row or any(value is None for value in row.values()) for row in rows):
        raise ValueError(f"{path.name} 有欄位數不正確的資料列")
    return rows


def check_unique_ids(rows: list[dict[str, str]], name: str) -> set[str]:
    """合併與切分都依 ID 進行；發現空白或重複 ID 就停止。"""
    ids = [row["id"] for row in rows]
    if any(not identifier.strip() for identifier in ids):
        raise ValueError(f"{name} 存在空白 ID")
    if len(set(ids)) != len(ids):
        raise ValueError(f"{name} 存在重複 ID")
    return set(ids)


def sha256_file(path: Path) -> str:
    """記錄原始檔與產出檔案的雜湊，日後可核對是否使用同一份資料。"""
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_csv(path: Path, rows: list[dict[str, str]]) -> None:
    """輸出相同欄位的 UTF-8 CSV；CSV 套件會處理留言內的逗號與換行。"""
    with path.open("w", encoding="utf-8", newline="") as handle:
        # 使用 CRLF 讓欄位內單獨出現的 CR 或 LF 都會被 CSV writer 正確引號包住。
        writer = csv.DictWriter(handle, fieldnames=DATA_COLUMNS, lineterminator="\r\n")
        writer.writeheader()
        writer.writerows(rows)


def label_counts(rows: list[dict[str, str]]) -> dict[str, int]:
    return {label: sum(row[label] == "1" for row in rows) for label in LABELS}


def main() -> None:
    args = parse_args()
    input_dir = args.input_dir.resolve()
    output_dir = args.output_dir.resolve()
    source_paths = {name: input_dir / name for name in ("train.csv", "test.csv", "test_labels.csv")}
    output_paths = {name: output_dir / name for name in ("train.csv", "validation.csv", "test.csv", "split_info.json")}

    # 避免誤覆寫來源或先前產生的資料；重跑時請使用另一個 --output-dir。
    if input_dir == output_dir:
        raise ValueError("輸入與輸出資料夾不能相同")
    missing = [str(path) for path in source_paths.values() if not path.is_file()]
    if missing:
        raise FileNotFoundError("找不到原始資料：" + ", ".join(missing))
    existing = [str(path) for path in output_paths.values() if path.exists()]
    if existing:
        raise FileExistsError("輸出檔已存在，未覆寫。請指定新的 --output-dir：" + ", ".join(existing))

    # Jigsaw 留言可能跨行，但以 CSV 解析後一列仍是一則留言。
    csv.field_size_limit(10_000_000)
    original_train = read_csv(source_paths["train.csv"], DATA_COLUMNS)
    original_test = read_csv(source_paths["test.csv"], TEXT_COLUMNS)
    original_test_labels = read_csv(source_paths["test_labels.csv"], LABEL_COLUMNS)

    train_ids = check_unique_ids(original_train, "train.csv")
    test_ids = check_unique_ids(original_test, "test.csv")
    label_ids = check_unique_ids(original_test_labels, "test_labels.csv")
    if test_ids != label_ids:
        raise ValueError("test.csv 與 test_labels.csv 的 ID 無法一對一對應")
    if train_ids & test_ids:
        raise ValueError("原始訓練集與測試集出現相同 ID，請先檢查來源")
    if any(row[label] not in {"0", "1"} for row in original_train for label in LABELS):
        raise ValueError("訓練集含有非 0/1 的標籤")

    # 從官方訓練資料中隨機取 10% 當驗證集。
    # 本程式未使用分層抽樣或文字去重，避免加入論文未交代的處理。
    # 固定 random.Random(42) 只是我們自行補上的重現設定，不是論文原始種子。
    indices = list(range(len(original_train)))
    random.Random(args.seed).shuffle(indices)
    validation_size = math.ceil(len(original_train) * VALIDATION_FRACTION)
    validation_indices = set(indices[:validation_size])
    train_rows = [row for i, row in enumerate(original_train) if i not in validation_indices]
    validation_rows = [row for i, row in enumerate(original_train) if i in validation_indices]

    # 原始測試留言與測試標籤分開存放；依 ID 合併，而不是假設列順序相同。
    labels_by_id = {row["id"]: row for row in original_test_labels}
    test_rows = []
    excluded_unlabeled = 0
    for comment in original_test:
        labels = labels_by_id[comment["id"]]
        values = [labels[label] for label in LABELS]
        # Kaggle 測試標籤為 -1 的資料沒有可用的評估標籤，不納入 test.csv。
        if all(value == "-1" for value in values):
            excluded_unlabeled += 1
            continue
        if any(value not in {"0", "1"} for value in values):
            raise ValueError(f"測試 ID {comment['id']} 有部分 -1 或非 0/1 標籤")
        test_rows.append({**comment, **{label: labels[label] for label in LABELS}})

    # 寫出前確認切分完整、互斥；不對留言文字或標籤進行任何清洗。
    split_train_ids = {row["id"] for row in train_rows}
    validation_ids = {row["id"] for row in validation_rows}
    valid_test_ids = {row["id"] for row in test_rows}
    if split_train_ids & validation_ids or split_train_ids | validation_ids != train_ids:
        raise ValueError("訓練與驗證資料有重疊或遺漏")
    if (split_train_ids | validation_ids) & valid_test_ids:
        raise ValueError("測試資料與訓練／驗證資料的 ID 有重疊")
    if len(test_rows) + excluded_unlabeled != len(original_test):
        raise ValueError("測試資料筆數無法核對")

    output_dir.mkdir(parents=True, exist_ok=True)
    splits = {"train": train_rows, "validation": validation_rows, "test": test_rows}
    for name, rows in splits.items():
        write_csv(output_paths[f"{name}.csv"], rows)

    # 一併記錄來源、數量、類別分布和重現資訊，方便審查及後續訓練。
    info = {
        "paper": "Mazari et al. (2024), DOI 10.1007/s10586-022-03956-x",
        "paper_reported_rows": {"original_train": 159570, "labeled_test": 63977},
        "important_note": f"論文未公布切分種子或是否分層；本次自行以 Python random.Random 洗牌、seed={args.seed}、未分層；不保證與作者切出的資料列相同。",
        "source_files": {name: {"path": str(path), "sha256": sha256_file(path)} for name, path in source_paths.items()},
        "split": {
            "original_train_rows": len(original_train),
            "original_test_rows": len(original_test),
            "validation_fraction_requested": VALIDATION_FRACTION,
            "validation_rounding": "ceil(total * 0.10)",
            "seed": args.seed,
            "random_generator": "Python random.Random (MT19937)",
            "stratified": False,
            "deduplicated_by_text": False,
            "shuffle_output_rows": False,
            "excluded_test_rows_all_minus_one": excluded_unlabeled,
        },
        "outputs": {name: {
            "file": f"{name}.csv",
            "rows": len(rows),
            "positive_labels": label_counts(rows),
            "sha256": sha256_file(output_paths[f"{name}.csv"]),
        } for name, rows in splits.items()},
        "python_version": platform.python_version(),
    }
    with output_paths["split_info.json"].open("w", encoding="utf-8", newline="\n") as handle:
        json.dump(info, handle, ensure_ascii=False, indent=2)
        handle.write("\n")

    for name, rows in splits.items():
        print(f"{name:10s} {len(rows):>8,} 筆 -> {output_paths[f'{name}.csv']}")
    print(f"排除無標註測試資料：{excluded_unlabeled:,} 筆")
    print(f"切分與來源紀錄：{output_paths['split_info.json']}")


if __name__ == "__main__":
    main()
