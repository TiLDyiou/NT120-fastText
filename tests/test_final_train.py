"""DoD verification tests for Phase 6 final trained models and inference API."""

from __future__ import annotations

import shutil
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import torch

from src_v2.config import PROCESSED_DATA_DIR, RUNS_DIR
from src_v2.final_train import load_final_seed_model
from src_v2.predict import DynamicPredictor, NT120Predictor


def test_final_models_exist_and_reload_cleanly() -> None:
    sample = pd.read_parquet(PROCESSED_DATA_DIR / "train.parquet").iloc[:100]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    for seed in [1, 2, 3]:
        sdir = RUNS_DIR / "final_models" / f"seed_{seed}"
        if not sdir.exists():
            pytest.skip(f"Final model seed_{seed} not trained yet")

        for fname in [
            "model_weights.pt",
            "token_to_id.json",
            "config.json",
            "train_history.json",
        ]:
            assert (sdir / fname).exists(), f"Missing {fname} in {sdir}"

        models, cfg, vocab = load_final_seed_model(sdir, device)
        assert cfg["max_len_full"] == 54
        assert cfg["t_bar_sent"] > 0
        assert cfg["t_bar_tox"] > 0

        pred = DynamicPredictor(models, vocab, cfg["max_len_full"], oov_provider=None, device=device)
        ls, lt = pred.predict_logits_from_tokens(sample["tokens"].tolist(), batch_size=100)
        assert ls.shape == (100, 3)
        assert lt.shape == (100, 4)
        assert np.isfinite(ls).all() and np.isfinite(lt).all()


def test_reload_without_npy_and_high_level_predictor(tmp_path: Path) -> None:
    seed1 = RUNS_DIR / "final_models" / "seed_1"
    if not seed1.exists():
        pytest.skip("Final model seed_1 not trained yet")

    slim_dir = tmp_path / "seed_1_slim"
    slim_dir.mkdir()
    for fname in ["model_weights.pt", "token_to_id.json", "config.json", "train_history.json"]:
        shutil.copy2(seed1 / fname, slim_dir / fname)

    predictor = NT120Predictor.from_dir(slim_dir, fasttext_bin=None, device="cpu", allow_zero_oov=True)
    texts = [
        "Sản phẩm dùng rất tốt, giao hàng nhanh",
        "Đồ ngu, cút đi",
        "Hôm nay trời mưa",
    ]
    preds = predictor.predict(texts)
    assert preds.shape == (3, 2)
    assert set(preds[:, 0].tolist()).issubset({-1, 0, 1})
    assert set(preds[:, 1].tolist()).issubset({0, 1, 2, 3})

    p_sent, p_tox = predictor.predict_proba(texts)
    assert p_sent.shape == (3, 3)
    assert p_tox.shape == (3, 4)
    assert np.allclose(p_sent.sum(axis=1), 1.0, atol=1e-5)
    assert np.allclose(p_tox.sum(axis=1), 1.0, atol=1e-5)
