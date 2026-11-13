#!/usr/bin/env python3
"""
Gộp và chuẩn bị dataset của nhóm (NT120) thành file chuẩn cho script train.
Theo LABELING_GUIDELINES.md:
  - Cột đầu vào: Comment, Sentiment (-1/0/1), Toxicity (0..3).
  - Khử trùng lặp / leak theo Mục 10 (NFC, bỏ URL, HTML, timestamp; trùng train-valid xoá ở valid;
    trùng với train/valid xoá ở challenge).
  - Nhãn:
      + Sentiment -1/0/1 -> Negative/Neutral/Positive
      + Toxicity: mặc định --tox-mode 4  -> giữ nguyên 4 mức 0/1/2/3 (cột `toxicity`)
                  --tox-mode binary      -> nhị phân theo --tox-min-level (cách cũ)
    Cột `tox_level` (0..3 gốc) luôn được giữ lại.
  - Tiền xử lý: preprocess_fasttext.clean() -> cột text_clean.

Cách chạy:
  python prepare_dataset.py --train-file merged/train/train.csv --val-file merged/valid/valid.csv \
         --challenge-file merged/challenge/challenge.csv --out data_ft.csv
"""
import argparse
import html
import re
import sys
import unicodedata
from pathlib import Path

import emoji
import pandas as pd

SENT_MAP = {
    -1: "Negative", 0: "Neutral", 1: "Positive",
    "-1": "Negative", "0": "Neutral", "1": "Positive",
    "negative": "Negative", "neutral": "Neutral", "positive": "Positive",
    "neg": "Negative", "neu": "Neutral", "pos": "Positive"
}

PRIORITY = {"train": 0, "val": 1, "challenge": 2}

EXCEL_ERR = re.compile(r"^#(NAME|ERROR|REF|VALUE|N/A|DIV/0)[?!/0]*$", re.I)
URL_PATTERN = re.compile(r"https?://\S+|www\.\S+|\b[\w\-]+(?:\.[\w\-]+)+[/\S]*", re.I)
HTML_TAG = re.compile(r"<[^>]+>")
TIMESTAMP = re.compile(r"\b\d{1,2}:\d{2}(?::\d{2})?\b")
HAS_LETTER = re.compile(r"[^\W\d_]")
SPACES = re.compile(r"\s+")

TOX_NAMES = {0: "T0 không độc hại", 1: "T1 thiếu lịch sự", 2: "T2 xúc phạm", 3: "T3 thù ghét/đe doạ"}


def guideline_key(text: str) -> str:
    """Khóa so khớp trùng lặp theo Mục 10.1 (không lowercase, không bỏ dấu câu/emoji)."""
    if not isinstance(text, str):
        return ""
    t = unicodedata.normalize("NFC", text)
    t = URL_PATTERN.sub(" ", t)
    t = html.unescape(t)
    t = HTML_TAG.sub(" ", t)
    t = TIMESTAMP.sub(" ", t)
    t = SPACES.sub(" ", t).strip()
    return t


def junk_reason(text: str):
    if not isinstance(text, str) or not text.strip():
        return "rỗng"
    t = text.strip()
    if EXCEL_ERR.match(t):
        return "lỗi Excel (#ERROR!/#NAME?)"
    if len(URL_PATTERN.findall(t)) >= 3 or len(t) > 1000:
        return "spam nhiều link / quá dài"
    if not HAS_LETTER.search(t) and emoji.emoji_count(t) == 0:
        return "không có chữ (chỉ số/ký hiệu)"
    return None


def discover_files(repo: Path):
    patterns = {
        "train": ["merged/train/train.csv", "merged/train/*/train.csv", "train/train.csv", "train.csv"],
        "val": ["merged/valid/valid.csv", "merged/val/val.csv", "merged/valid/*/valid.csv",
                "valid/valid.csv", "val/val.csv", "valid.csv", "val.csv"],
        "challenge": ["merged/challenge/challenge.csv", "merged/challenge/*/challenge.csv",
                      "challenge/challenge.csv", "challenge.csv"],
    }
    found = {"train": [], "val": [], "challenge": []}
    for split, pats in patterns.items():
        for p in pats:
            matches = sorted(repo.glob(p))
            if matches:
                found[split] = matches
                break
    return found


def read_splits(repo: Path, train_file=None, val_file=None, challenge_file=None):
    file_map = {}
    if train_file or val_file or challenge_file:
        if train_file:
            file_map["train"] = [Path(train_file)]
        if val_file:
            file_map["val"] = [Path(val_file)]
        if challenge_file:
            file_map["challenge"] = [Path(challenge_file)]
    else:
        file_map = discover_files(repo)

    frames = []
    for split, files in file_map.items():
        for f in files:
            if not f.exists():
                print(f"!! Không tìm thấy file: {f}", file=sys.stderr)
                continue
            df = pd.read_csv(f, encoding="utf-8-sig")
            df.columns = [c.strip().replace("\ufeff", "") for c in df.columns]
            missing = {"Comment", "Sentiment", "Toxicity"} - set(df.columns)
            if missing:
                raise SystemExit(f"{f}: thiếu các cột bắt buộc {missing}")
            df["split"] = split
            frames.append(df[["Comment", "Sentiment", "Toxicity", "split"]])

    if not frames:
        raise SystemExit("Không đọc được file dữ liệu nào! Kiểm tra --repo hoặc --train-file.")
    return pd.concat(frames, ignore_index=True)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--repo", default=".")
    ap.add_argument("--train-file")
    ap.add_argument("--val-file")
    ap.add_argument("--challenge-file")
    ap.add_argument("--out", default="data_ft.csv")
    ap.add_argument("--tox-mode", choices=["4", "binary"], default="4",
                    help="4: giữ 4 mức 0..3 (mặc định) | binary: gộp theo --tox-min-level")
    ap.add_argument("--tox-min-level", type=int, default=1, choices=[1, 2, 3],
                    help="Chỉ dùng khi --tox-mode binary: mức >= giá trị này là Toxic")
    ap.add_argument("--no-segment", action="store_true", help="không chạy tách từ pyvi")
    a = ap.parse_args()

    df = read_splits(Path(a.repo), a.train_file, a.val_file, a.challenge_file)
    n0 = len(df)
    print(f"Đọc tổng cộng {n0} dòng từ: {df['split'].value_counts().to_dict()}")

    # 1. Nhãn hợp lệ
    s_raw = df["Sentiment"].astype(str).str.strip().str.lower()
    s_mapped = s_raw.map(SENT_MAP)
    t_num = pd.to_numeric(df["Toxicity"], errors="coerce")
    bad = s_mapped.isna() | t_num.isna() | ~t_num.isin([0, 1, 2, 3])
    if bad.any():
        print(f"\n[1] Loại {bad.sum()} dòng nhãn không hợp lệ hoặc thiếu:")
        print(df[bad].groupby("split").size().to_string())
        df = df[~bad].copy()
        s_mapped = s_mapped[~bad]
        t_num = t_num[~bad]

    df["emotion"] = s_mapped
    df["tox_level"] = t_num.astype(int)
    if a.tox_mode == "4":
        df["toxicity"] = df["tox_level"]                       # giữ nguyên 0..3
    else:
        df["toxicity"] = (df["tox_level"] >= a.tox_min_level).astype(int)

    # 2. Dòng rác
    df["junk"] = df["Comment"].map(junk_reason)
    print(f"\n[2] Dòng rác bị loại: {df['junk'].notna().sum()}")
    if df["junk"].notna().any():
        print(df[df.junk.notna()].groupby(["junk", "split"]).size().to_string())
    df = df[df.junk.isna()].copy()

    # 3. Khử trùng / leak (Mục 10)
    df["key"] = df["Comment"].map(guideline_key)
    valid_key = df["key"] != ""
    df_valid = df[valid_key].copy()
    df_empty_key = df[~valid_key].copy()
    df_valid["prio"] = df_valid["split"].map(PRIORITY).fillna(99)
    df_valid = df_valid.sort_values("prio", kind="stable")
    dup = df_valid.duplicated("key", keep="first")
    print(f"\n[3] Trùng lặp bị loại: {dup.sum()} dòng (theo tập: {df_valid[dup].split.value_counts().to_dict()})")
    df = pd.concat([df_valid[~dup].copy(), df_empty_key], ignore_index=True)

    # 4. Tiền xử lý
    from preprocess_fasttext import clean
    print("\n[4] Đang tiền xử lý văn bản (emoticon -> emoji, teencode, pyvi)...")
    df["text"] = df["Comment"].astype(str).str.strip()
    df["text_clean"] = df["text"].map(lambda x: clean(x, not a.no_segment))

    # 5. Bỏ câu rỗng
    empty = df["text_clean"] == ""
    if empty.any():
        print(f"\n[5] Loại {empty.sum()} dòng rỗng sau khi làm sạch.")
        df = df[~empty].copy()

    df = df.sample(frac=1, random_state=42).reset_index(drop=True)
    df.insert(0, "id", range(len(df)))
    cols = ["id", "text", "text_clean", "emotion", "toxicity", "tox_level", "split"]
    df[cols].to_csv(a.out, index=False, encoding="utf-8-sig")

    # 6. Thống kê
    mode_desc = "4 mức 0..3" if a.tox_mode == "4" else f"nhị phân (mức >= {a.tox_min_level})"
    print(f"\n==== HOÀN TẤT: {len(df)}/{n0} dòng -> {a.out} | Toxicity: {mode_desc} ====")
    for sp in ["train", "val", "challenge"]:
        d = df[df.split == sp]
        if d.empty:
            continue
        ct = pd.crosstab(d.emotion, d.toxicity).reindex(["Negative", "Neutral", "Positive"]).fillna(0).astype(int)
        print(f"\n--- Tập {sp.upper()}: {len(d)} dòng")
        print(ct.to_string())
        print("    Phân bố toxicity:", d.toxicity.value_counts().sort_index().to_dict())


if __name__ == "__main__":
    main()
