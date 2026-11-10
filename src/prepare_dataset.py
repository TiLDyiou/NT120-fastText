#!/usr/bin/env python3
"""
Gộp và chuẩn bị dataset của nhóm (NT120) thành file chuẩn cho script train.
Tuân thủ đầy đủ theo tài liệu LABELING_GUIDELINES.md mới nhất:
  - Cột đầu vào: Comment, Sentiment (-1/0/1), Toxicity (0..3).
  - ĐÃ BỎ hoàn toàn cột ambiguity_flag / flag theo guideline mới.
  - Vệ sinh dữ liệu & Khử trùng lặp chuẩn theo Mục 10:
      + Chuẩn hóa NFC, bỏ URL, bỏ HTML entity/tags, bỏ timestamp video.
      + Trùng trong cùng file: giữ dòng đầu tiên.
      + Comment có trong train và valid: XOÁ Ở VALID (giữ train).
      + Comment có trong train/valid và challenge: XOÁ Ở CHALLENGE.
  - Chuyển đổi nhãn:
      + Sentiment -1/0/1 -> Negative/Neutral/Positive
      + Toxicity 0..3 -> nhị phân theo --tox-min-level (mặc định 1: 0 -> Non-toxic, >=1 -> Toxic)
  - Tiền xử lý văn bản: chạy preprocess_fasttext.clean() -> sinh cột text_clean.

Cách chạy:
  python prepare_dataset.py --repo nt120_colab --out data_ft.csv
hoặc:
  python prepare_dataset.py --train-file merged/train/train.csv --val-file merged/valid/valid.csv --out data_ft.csv
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

# Quy ước ưu tiên giữ lại theo Mục 10.2 của Guideline mới:
# - Trùng train và valid -> xoá ở valid (tức giữ train: train có độ ưu tiên cao hơn)
# - Trùng train/valid và challenge -> xoá ở challenge (challenge ưu tiên thấp nhất khi dedup leak)
PRIORITY = {"train": 0, "val": 1, "challenge": 2}

EXCEL_ERR = re.compile(r"^#(NAME|ERROR|REF|VALUE|N/A|DIV/0)[?!/0]*$", re.I)
URL_PATTERN = re.compile(r"https?://\S+|www\.\S+|\b[\w\-]+(?:\.[\w\-]+)+[/\S]*", re.I)
HTML_TAG = re.compile(r"<[^>]+>")
TIMESTAMP = re.compile(r"\b\d{1,2}:\d{2}(?::\d{2})?\b")
HAS_LETTER = re.compile(r"[^\W\d_]")
SPACES = re.compile(r"\s+")


def guideline_key(text: str) -> str:
    """Khóa so khớp trùng lặp theo đúng Mục 10.1 của LABELING_GUIDELINES.md:
    1. Chuẩn hóa NFC.
    2. Bỏ URL.
    3. Bỏ HTML entity & tag.
    4. Bỏ timestamp video (12:34, 1:02:03).
    5. Gộp khoảng trắng thừa, trim.
    (Không lowercase, không bỏ dấu câu/emoji)."""
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
    """Phát hiện dòng rác không thể dùng huấn luyện."""
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
    """Tự động tìm kiếm file dữ liệu trong repo theo cả cấu trúc mới và cũ."""
    patterns = {
        "train": [
            "merged/train/train.csv",
            "merged/train/*/train.csv",
            "train/train.csv",
            "train.csv",
        ],
        "val": [
            "merged/valid/valid.csv",
            "merged/val/val.csv",
            "merged/valid/*/valid.csv",
            "valid/valid.csv",
            "val/val.csv",
            "valid.csv",
            "val.csv",
        ],
        "challenge": [
            "merged/challenge/challenge.csv",
            "merged/challenge/*/challenge.csv",
            "challenge/challenge.csv",
            "challenge.csv",
        ],
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
        if not files:
            continue
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
        raise SystemExit("Không đọc được file dữ liệu nào! Hãy kiểm tra lại tham số --repo hoặc --train-file.")
    return pd.concat(frames, ignore_index=True)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--repo", default=".", help="thư mục gốc chứa dataset hoặc folder merged")
    ap.add_argument("--train-file", help="đường dẫn trực tiếp file train.csv")
    ap.add_argument("--val-file", help="đường dẫn trực tiếp file valid.csv")
    ap.add_argument("--challenge-file", help="đường dẫn trực tiếp file challenge.csv (nếu có)")
    ap.add_argument("--out", default="data_ft.csv", help="file CSV kết quả sau khi xử lý")
    ap.add_argument("--tox-min-level", type=int, default=1, choices=[1, 2, 3],
                    help="Toxicity >= mức này là Toxic (mặc định 1: gồm cả thiếu lịch sự/chửi thề buột miệng). "
                         "Mức 0: Non-toxic, Mức 1-3: Toxic")
    ap.add_argument("--no-segment", action="store_true", help="không chạy tách từ tiếng Việt pyvi")
    a = ap.parse_args()

    df = read_splits(Path(a.repo), a.train_file, a.val_file, a.challenge_file)
    n0 = len(df)
    print(f"Đọc tổng cộng {n0} dòng từ: {df['split'].value_counts().to_dict()}")

    # 1. Kiểm tra tính hợp lệ của nhãn -----------------------------------------------------------
    s_raw = df["Sentiment"].astype(str).str.strip().str.lower()
    s_mapped = s_raw.map(SENT_MAP)
    t_num = pd.to_numeric(df["Toxicity"], errors="coerce")

    bad = s_mapped.isna() | t_num.isna() | ~t_num.isin([0, 1, 2, 3])
    if bad.any():
        print(f"\n[1] Loại {bad.sum()} dòng nhãn không hợp lệ hoặc bị thiếu:")
        print(df[bad].groupby("split").size().to_string())
        df = df[~bad].copy()
        s_mapped = s_mapped[~bad]
        t_num = t_num[~bad]

    df["emotion"] = s_mapped
    df["tox_level"] = t_num.astype(int)
    df["toxicity"] = (df["tox_level"] >= a.tox_min_level).astype(int)

    # 2. Loại bỏ dòng rác ------------------------------------------------------------------------
    df["junk"] = df["Comment"].map(junk_reason)
    n_junk = df["junk"].notna().sum()
    print(f"\n[2] Dòng rác bị loại: {n_junk}")
    if n_junk > 0:
        print(df[df.junk.notna()].groupby(["junk", "split"]).size().to_string())
    df = df[df.junk.isna()].copy()

    # 3. Khử trùng lặp & Rò rỉ dữ liệu theo Mục 10 Guideline ------------------------------------
    # Tạo khóa khớp dữ liệu chuẩn: NFC, bỏ URL, HTML, timestamp video
    df["key"] = df["Comment"].map(guideline_key)
    # Những câu có key rỗng (chỉ toàn link/timestamp) không đem so khớp
    valid_key = df["key"] != ""
    df_valid = df[valid_key].copy()
    df_empty_key = df[~valid_key].copy()

    # Sắp xếp theo thứ tự ưu tiên: train (0) -> val (1) -> challenge (2)
    # Dòng xuất hiện trước sẽ được giữ lại:
    # - Trùng trong cùng file: giữ dòng đầu tiên
    # - Trùng train và valid: XOÁ Ở VALID (vì train ưu tiên hơn)
    # - Trùng train/valid và challenge: XOÁ Ở CHALLENGE
    df_valid["prio"] = df_valid["split"].map(PRIORITY).fillna(99)
    df_valid = df_valid.sort_values("prio", kind="stable")

    dup = df_valid.duplicated("key", keep="first")
    print(f"\n[3] Trùng lặp bị loại: {dup.sum()} dòng (theo tập: {df_valid[dup].split.value_counts().to_dict()})")
    df_cleaned = df_valid[~dup].copy()

    # Ghép lại với các câu có key đặc biệt (nếu có)
    df = pd.concat([df_cleaned, df_empty_key], ignore_index=True)

    # 4. Tiền xử lý văn bản cho fastText --------------------------------------------------------
    from preprocess_fasttext import clean
    print("\n[4] Đang tiền xử lý văn bản (emoticon -> emoji, teencode, pyvi)...")
    df["text"] = df["Comment"].astype(str).str.strip()
    df["text_clean"] = df["text"].map(lambda x: clean(x, not a.no_segment))

    # 5. Loại bỏ câu rỗng sau tiền xử lý --------------------------------------------------------
    empty = df["text_clean"] == ""
    if empty.any():
        print(f"\n[5] Loại bỏ {empty.sum()} dòng rỗng sau khi làm sạch văn bản.")
        df = df[~empty].copy()

    # Xáo trộn ngẫu nhiên từng tập để tránh nhóm theo nguồn
    df = df.sample(frac=1, random_state=42).reset_index(drop=True)
    df.insert(0, "id", range(len(df)))

    cols = ["id", "text", "text_clean", "emotion", "toxicity", "tox_level", "split"]
    df[cols].to_csv(a.out, index=False, encoding="utf-8-sig")

    # 6. Báo cáo phân bố nhãn -------------------------------------------------------------------
    print(f"\n==== HOÀN TẤT: {len(df)}/{n0} dòng được giữ -> {a.out} (Toxicity=1 khi mức >= {a.tox_min_level}) ====")
    for sp in ["train", "val", "challenge"]:
        d = df[df.split == sp]
        if d.empty:
            continue
        ct = pd.crosstab(d.emotion, d.toxicity.map({0: "Non-toxic", 1: "Toxic"}))
        for c in ["Non-toxic", "Toxic"]:
            if c not in ct:
                ct[c] = 0
        print(f"\n--- Tập {sp.upper()}: {len(d)} dòng | Tỷ lệ Toxic: {d.toxicity.mean():.1%}")
        print(ct[["Non-toxic", "Toxic"]].reindex(["Negative", "Neutral", "Positive"]).fillna(0).astype(int).to_string())
        print("    Chi tiết các mức tox_level gốc:", d.tox_level.value_counts().sort_index().to_dict())


if __name__ == "__main__":
    main()
