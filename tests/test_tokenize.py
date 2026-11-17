"""Unit tests for Phase 1 cleaning, tokenization, and deduplication."""

from __future__ import annotations

import unicodedata
from pathlib import Path

import pandas as pd

from src_v2.data_prep import clean_text, run_data_prep, tokenize


def test_tokenize_spec_examples() -> None:
    assert tokenize(clean_text("Đồ ngu ngục. đ.m =)))")) == [
        "Đồ", "ngu", "ngục", ".", "đ.m", "=)))"
    ]
    assert tokenize(clean_text('"Ngục..." (đ.m) buồn:((')) == [
        '"', "Ngục", '..."', "(", "đ.m", ")", "buồn", ":(("
    ]


def test_tokenize_edge_cases() -> None:
    # Missing space after period stays 1 token per spec §1.4 known limitation
    assert tokenize(clean_text("ngục.mày")) == ["ngục.mày"]
    # Symbol-only / emoji / emoticon chunks (no Letter/Number) stay single tokens.
    # Note: '<3' contains digit '3' (Unicode Number), so per strict §1.4 rule it splits
    # into ['<', '3'] (which also avoids cc.vi.300.bin minn=5 zero-vector on 2-char OOV).
    assert tokenize(clean_text(":)) =))) ??? ...!!!")) == [":))", "=)))", "???", "...!!!"]
    assert tokenize(clean_text("<3")) == ["<", "3"]
    # Case preservation & no repeat collapse & no teencode expansion
    assert tokenize(clean_text("ĐẸPPPP ko dc vcl")) == ["ĐẸPPPP", "ko", "dc", "vcl"]


def test_clean_text_nfc_and_url() -> None:
    nfc = "tiếng Việt"
    nfd = unicodedata.normalize("NFD", nfc)
    assert nfc != nfd
    assert clean_text(nfc) == clean_text(nfd)
    assert tokenize(clean_text(nfc)) == tokenize(clean_text(nfd))

    assert clean_text("xemhttps://fb.com/abc ngay") == "xem url ngay"
    assert clean_text("www.google.com   \n\t ok") == "url ok"
    assert clean_text("   \n\t  ") == ""


def test_dedup_and_challenge_preservation(tmp_path: Path) -> None:
    raw = tmp_path / "raw"
    for sp in ["train", "valid", "challenge"]:
        (raw / sp).mkdir(parents=True)

    pd.DataFrame({
        "Comment": ["câu 1", "câu 1", "câu 2", "   "],
        "Sentiment": [1, 0, -1, 0],
        "Toxicity": [0, 0, 2, 0],
    }).to_csv(raw / "train" / "train.csv", index=False)

    pd.DataFrame({
        "Comment": ["câu 2", "câu 3", "câu 3", ""],
        "Sentiment": [-1, 0, 1, 0],
        "Toxicity": [2, 1, 1, 0],
    }).to_csv(raw / "valid" / "valid.csv", index=False)

    pd.DataFrame({
        "Comment": ["câu 1", "câu 4", "câu 4", "   "],
        "Sentiment": [1, 0, 0, 0],
        "Toxicity": [0, 3, 3, 0],
    }).to_csv(raw / "challenge" / "challenge.csv", index=False)

    out = tmp_path / "processed"
    rep = tmp_path / "stats.md"
    stats = run_data_prep(raw, out, rep)

    assert stats["train"]["final"] == 2
    assert stats["valid"]["final"] == 1
    assert stats["challenge"]["final"] == 3  # Only empty dropped; duplicates & overlap kept!

    tr_out = pd.read_parquet(out / "train.parquet")
    va_out = pd.read_parquet(out / "valid.parquet")
    assert set(tr_out["text_clean"]).isdisjoint(set(va_out["text_clean"]))
    assert tr_out["y_sent_idx"].tolist() == [2, 0]  # 1->2, -1->0
