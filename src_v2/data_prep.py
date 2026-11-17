"""Phase 1: Minimal text cleaning, deduplication, label mapping, and tokenization."""

from __future__ import annotations

import argparse
import re
import unicodedata
from pathlib import Path

import numpy as np
import pandas as pd

from src_v2.config import (
    PROCESSED_DATA_DIR,
    RAW_DATA_DIR,
    REPORTS_DIR,
    ensure_project_dirs,
)

URL_RE = re.compile(r"https?://\S+|www\.\S+", re.IGNORECASE)
WHITESPACE_RE = re.compile(r"\s+")


def clean_text(text: object) -> str:
    """Apply the exact 4-step minimal cleaning pipeline from PIPELINE_SPEC.md §1.2.

    1. Normalize Unicode to NFC.
    2. Replace web URLs with space-padded 'url'.
    3. Collapse consecutive whitespace to single space and strip.
    4. Return empty string if empty (caller drops empty rows).
    """
    if not isinstance(text, str):
        if pd.isna(text):
            return ""
        text = str(text)
    t = unicodedata.normalize("NFC", text)
    t = URL_RE.sub(" url ", t)
    t = WHITESPACE_RE.sub(" ", t).strip()
    return t


def _is_word_char(ch: str) -> bool:
    """True for Unicode letters ('L*') and numbers ('N*')."""
    cat = unicodedata.category(ch)
    return cat[0] in ("L", "N")


def tokenize(cleaned_text: str) -> list[str]:
    """Tokenize cleaned text according to PIPELINE_SPEC.md §1.4.

    - Split by whitespace into raw chunks.
    - Chunk with no Unicode letter/number -> single token (e.g. '=)))', '<3', '...!!!').
    - Chunk with letters/numbers -> peel leading symbol cluster and trailing symbol
      cluster as separate tokens, preserving internal punctuation (e.g. 'đ.m', 'l.ồ.n').
    """
    if not cleaned_text:
        return []
    tokens: list[str] = []
    for chunk in cleaned_text.split(" "):
        if not chunk:
            continue
        if not any(_is_word_char(ch) for ch in chunk):
            tokens.append(chunk)
            continue

        start = 0
        while start < len(chunk) and not _is_word_char(chunk[start]):
            start += 1
        end = len(chunk)
        while end > start and not _is_word_char(chunk[end - 1]):
            end -= 1

        if start > 0:
            tokens.append(chunk[:start])
        tokens.append(chunk[start:end])
        if end < len(chunk):
            tokens.append(chunk[end:])
    return tokens


def _load_raw_split(csv_path: Path) -> pd.DataFrame:
    df = pd.read_csv(csv_path, encoding="utf-8-sig")
    df.columns = [c.strip().replace("\ufeff", "") for c in df.columns]
    required = {"Comment", "Sentiment", "Toxicity"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"{csv_path} is missing required columns: {sorted(missing)}")
    return df[["Comment", "Sentiment", "Toxicity"]].copy()


def _map_labels(df: pd.DataFrame, split_name: str) -> pd.DataFrame:
    sent_num = pd.to_numeric(df["Sentiment"], errors="raise").astype(int)
    tox_num = pd.to_numeric(df["Toxicity"], errors="raise").astype(int)
    if not sent_num.isin([-1, 0, 1]).all():
        bad = sent_num[~sent_num.isin([-1, 0, 1])].unique().tolist()
        raise ValueError(f"{split_name}: invalid Sentiment values {bad}")
    if not tox_num.isin([0, 1, 2, 3]).all():
        bad = tox_num[~tox_num.isin([0, 1, 2, 3])].unique().tolist()
        raise ValueError(f"{split_name}: invalid Toxicity values {bad}")
    df = df.copy()
    df["y_sent_idx"] = sent_num + 1
    df["y_tox_idx"] = tox_num
    return df


def prepare_split(csv_path: Path, split_name: str) -> tuple[pd.DataFrame, dict[str, int]]:
    """Load, clean, map labels, and tokenize a single split without cross-dedup."""
    raw_df = _load_raw_split(csv_path)
    n_raw = len(raw_df)
    df = _map_labels(raw_df, split_name)
    df["text_raw"] = df["Comment"].fillna("").astype(str)
    df["text_clean"] = df["text_raw"].map(clean_text)

    non_empty = df["text_clean"] != ""
    n_empty = int((~non_empty).sum())
    df = df[non_empty].reset_index(drop=True)
    df["tokens"] = df["text_clean"].map(tokenize)
    df["num_tokens"] = df["tokens"].map(len)
    stats = {
        "raw": n_raw,
        "dropped_empty": n_empty,
        "after_empty": len(df),
    }
    return df[["text_raw", "text_clean", "tokens", "num_tokens", "y_sent_idx", "y_tox_idx"]], stats


def run_data_prep(raw_dir: Path = RAW_DATA_DIR, out_dir: Path = PROCESSED_DATA_DIR, report_path: Path | None = None) -> dict[str, dict[str, int]]:
    """Run full Phase 1 preparation across train, valid, and challenge."""
    ensure_project_dirs()
    out_dir.mkdir(parents=True, exist_ok=True)
    if report_path is None:
        report_path = REPORTS_DIR / "data_stats.md"

    train_df, train_stats = prepare_split(raw_dir / "train" / "train.csv", "train")
    valid_df, valid_stats = prepare_split(raw_dir / "valid" / "valid.csv", "valid")
    chal_df, chal_stats = prepare_split(raw_dir / "challenge" / "challenge.csv", "challenge")

    # Internal dedup on train (keep first)
    dup_train = train_df.duplicated(subset=["text_clean"], keep="first")
    train_stats["dropped_internal_dup"] = int(dup_train.sum())
    train_df = train_df[~dup_train].reset_index(drop=True)
    train_stats["final"] = len(train_df)

    # Internal dedup on valid (keep first)
    dup_valid = valid_df.duplicated(subset=["text_clean"], keep="first")
    valid_stats["dropped_internal_dup"] = int(dup_valid.sum())
    valid_df = valid_df[~dup_valid].reset_index(drop=True)

    # Anti-leak dedup: drop valid rows appearing in train
    train_keys = set(train_df["text_clean"])
    leak_valid = valid_df["text_clean"].isin(train_keys)
    valid_stats["dropped_cross_leak"] = int(leak_valid.sum())
    valid_df = valid_df[~leak_valid].reset_index(drop=True)
    valid_stats["final"] = len(valid_df)

    # Challenge: keep 100% of non-empty rows (no internal/cross dedup)
    chal_stats["dropped_internal_dup"] = 0
    chal_stats["dropped_cross_leak"] = 0
    chal_stats["final"] = len(chal_df)

    # Verify zero overlap between train and valid
    overlap = set(train_df["text_clean"]) & set(valid_df["text_clean"])
    assert len(overlap) == 0, f"Found {len(overlap)} overlapping keys between train and valid!"

    train_df.to_parquet(out_dir / "train.parquet", index=False)
    valid_df.to_parquet(out_dir / "valid.parquet", index=False)
    chal_df.to_parquet(out_dir / "challenge.parquet", index=False)

    _write_data_stats_report(report_path, train_df, valid_df, train_stats, valid_stats, chal_stats)
    return {"train": train_stats, "valid": valid_stats, "challenge": chal_stats}


def _write_data_stats_report(
    report_path: Path,
    train_df: pd.DataFrame,
    valid_df: pd.DataFrame,
    train_stats: dict[str, int],
    valid_stats: dict[str, int],
    chal_stats: dict[str, int],
) -> None:
    lines = [
        "# Thống kê Dữ liệu sau Phase 1 (`data_prep.py`)",
        "",
        "> Ghi chú: Để đảm bảo `challenge` là tập mù tuyệt đối, báo cáo này không thống kê phân phối nhãn hay từ vựng của `challenge` (chỉ kiểm tra số dòng trước/sau lọc rỗng).",
        "",
        "## 1. Tổng hợp Khử trùng lặp & Chống rò rỉ",
        "",
        "| Tập dữ liệu | Dòng gốc | Loại rỗng | Trùng nội bộ | Trùng chéo train∩valid | Còn lại cuối |",
        "| :--- | :---: | :---: | :---: | :---: | :---: |",
        f"| `train` | {train_stats['raw']:,} | {train_stats['dropped_empty']} | {train_stats['dropped_internal_dup']} | 0 | **{train_stats['final']:,}** |",
        f"| `valid` | {valid_stats['raw']:,} | {valid_stats['dropped_empty']} | {valid_stats['dropped_internal_dup']} | {valid_stats['dropped_cross_leak']} | **{valid_stats['final']:,}** |",
        f"| `challenge` | {chal_stats['raw']:,} | {chal_stats['dropped_empty']} | 0 (giữ nguyên) | 0 (giữ nguyên) | **{chal_stats['final']:,}** |",
        "",
        "## 2. Phân phối Nhãn (`train` & `valid`)",
        "",
    ]

    for name, df in [("train", train_df), ("valid", valid_df)]:
        sent_counts = (df["y_sent_idx"] - 1).value_counts().sort_index().to_dict()
        tox_counts = df["y_tox_idx"].value_counts().sort_index().to_dict()
        lens = df["num_tokens"].to_numpy()
        p50 = float(np.percentile(lens, 50))
        p95 = int(round(float(np.percentile(lens, 95))))
        p99 = float(np.percentile(lens, 99))
        lines.extend([
            f"### Tập `{name}` ($N = {len(df):,}$)",
            f"- **Sentiment (`-1 / 0 / 1`):** `{sent_counts}`",
            f"- **Toxicity (`0 / 1 / 2 / 3`):** `{tox_counts}`",
            f"- **Độ dài token:** min={int(lens.min())}, median={p50:.1f}, **p95={p95}**, p99={p99:.1f}, max={int(lens.max())}",
            "",
        ])

    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Run Phase 1 data cleaning & tokenization.")
    parser.add_argument("--raw-dir", type=Path, default=RAW_DATA_DIR)
    parser.add_argument("--out-dir", type=Path, default=PROCESSED_DATA_DIR)
    parser.add_argument("--report", type=Path, default=REPORTS_DIR / "data_stats.md")
    args = parser.parse_args()

    stats = run_data_prep(args.raw_dir, args.out_dir, args.report)
    for split, st in stats.items():
        print(f"[{split}] {st}")
    print(f"Report written to {args.report}")


if __name__ == "__main__":
    main()
