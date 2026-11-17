"""Unit and DoD verification tests for Phase 2 embedding artifacts."""

from __future__ import annotations

import json
import random
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from src_v2.build_embeddings import PAD_ID, PAD_TOKEN, build_vocab, compute_max_len
from src_v2.config import ARTIFACTS_DIR, DEFAULT_FASTTEXT_BIN, EMBEDDING_DIM, PROCESSED_DATA_DIR


def test_build_vocab_no_unk_and_pad_zero() -> None:
    vocab = build_vocab([["chào", "bạn"], ["bạn", "khỏe"]])
    assert vocab[PAD_TOKEN] == PAD_ID == 0
    assert "<unk>" not in vocab
    assert set(vocab.values()) == {0, 1, 2, 3}


def test_compute_max_len() -> None:
    seqs = [["a"] * i for i in range(1, 101)]
    assert compute_max_len(seqs) == 95


def test_artifacts_integrity_and_isolation() -> None:
    for split in ["train", "trainV1", "full"]:
        d = ARTIFACTS_DIR / split
        if not d.exists() or not (d / "embedding_matrix.npy").exists():
            pytest.skip(f"Artifacts not built yet for {split}")

        token_to_id = json.loads((d / "token_to_id.json").read_text(encoding="utf-8"))
        meta = json.loads((d / "meta.json").read_text(encoding="utf-8"))
        mat = np.load(d / "embedding_matrix.npy")

        assert token_to_id["<pad>"] == 0
        assert "<unk>" not in token_to_id
        assert mat.dtype == np.float32
        assert mat.shape == (len(token_to_id), EMBEDDING_DIM)
        assert meta["vocab_size"] == len(token_to_id)
        assert np.all(mat[0] == 0.0)

    # Check pre-cache isolation for train vs valid
    tr_vocab = json.loads((ARTIFACTS_DIR / "train" / "token_to_id.json").read_text(encoding="utf-8"))
    oov_map = json.loads((ARTIFACTS_DIR / "train" / "valid_oov_map.json").read_text(encoding="utf-8"))
    oov_vecs = np.load(ARTIFACTS_DIR / "train" / "valid_oov_vectors.npy")
    assert set(tr_vocab).isdisjoint(set(oov_map))
    assert oov_vecs.shape == (len(oov_map), EMBEDDING_DIM)

    # Verify V1/V2 stratified split matches valid exactly
    va = pd.read_parquet(PROCESSED_DATA_DIR / "valid.parquet")
    v1 = pd.read_parquet(PROCESSED_DATA_DIR / "valid_v1.parquet")
    v2 = pd.read_parquet(PROCESSED_DATA_DIR / "valid_v2.parquet")
    assert len(v1) + len(v2) == len(va)
    assert set(v1["text_clean"]).isdisjoint(set(v2["text_clean"]))


def test_random_20_tokens_match_fasttext_bin() -> None:
    mat_path = ARTIFACTS_DIR / "train" / "embedding_matrix.npy"
    if not mat_path.exists() or not DEFAULT_FASTTEXT_BIN.exists():
        pytest.skip("Missing embedding_matrix.npy or cc.vi.300.bin")

    import fasttext

    token_to_id = json.loads((ARTIFACTS_DIR / "train" / "token_to_id.json").read_text(encoding="utf-8"))
    mat = np.load(mat_path)
    oov_map = json.loads((ARTIFACTS_DIR / "train" / "valid_oov_map.json").read_text(encoding="utf-8"))
    oov_vecs = np.load(ARTIFACTS_DIR / "train" / "valid_oov_vectors.npy")

    rng = random.Random(123)
    real_tokens = [t for t in token_to_id if t != PAD_TOKEN]
    sample_train = rng.sample(real_tokens, 20)
    sample_oov = rng.sample(list(oov_map.keys()), 5)

    ft = fasttext.load_model(str(DEFAULT_FASTTEXT_BIN))
    try:
        for tok in sample_train:
            expected = ft.get_word_vector(tok).astype(np.float32)
            assert np.allclose(mat[token_to_id[tok]], expected, atol=1e-6)
        for tok in sample_oov:
            expected = ft.get_word_vector(tok).astype(np.float32)
            assert np.allclose(oov_vecs[oov_map[tok]], expected, atol=1e-6)
    finally:
        del ft
