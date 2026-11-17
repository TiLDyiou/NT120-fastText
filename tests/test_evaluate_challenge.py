"""Unit test for Phase 7 Cold/Warm start evaluation logic using synthetic data (never touches challenge.csv)."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch

from src_v2.config import EMBEDDING_DIM
from src_v2.evaluate_challenge import evaluate_seed_on_challenge
from src_v2.model import ParallelNet
from src_v2.predict import PreCachedOOVProvider


def test_evaluate_seed_cold_and_warm_start_synthetic(tmp_path: Path) -> None:
    seed_dir = tmp_path / "seed_1"
    seed_dir.mkdir(parents=True)

    vocab = {"<pad>": 0, "hay": 1, "quá": 2, "tệ": 3}
    rng = np.random.default_rng(7)
    emb = rng.normal(0.0, 0.2, size=(4, EMBEDDING_DIM)).astype(np.float32)
    emb[0] = 0.0

    model = ParallelNet(emb, hidden_units=16, dropout=0.0, use_dense300=False, task_mode="multi_task")
    torch.save(model.state_dict(), seed_dir / "model_weights.pt")
    np.save(seed_dir / "embedding_matrix.npy", emb)
    (seed_dir / "token_to_id.json").write_text(json.dumps(vocab), encoding="utf-8")
    (seed_dir / "config.json").write_text(
        json.dumps(
            {
                "seed": 1,
                "max_len_full": 10,
                "t_bar_sent": 1.4,
                "t_bar_tox": 1.2,
                "tau_k": [1.0, 1.0, 1.0, 1.0],
                "model_config": {
                    "hidden_units": 16,
                    "dropout": 0.0,
                    "use_dense300": False,
                    "task_mode": "multi_task",
                },
            }
        ),
        encoding="utf-8",
    )

    oov_vecs = rng.normal(0.0, 0.2, size=(2, EMBEDDING_DIM)).astype(np.float32)
    oov_provider = PreCachedOOVProvider({"mới_1": 0, "mới_2": 1}, oov_vecs)

    raw_texts = [
        "hay quá mới_1",
        "tệ mới_2",
        "hay mới_1",
        "tệ quá",
    ] * 10
    y_sent = np.array([2, 0, 2, 0] * 10, dtype=np.int64)
    y_tox = np.array([0, 2, 1, 3] * 10, dtype=np.int64)

    res = evaluate_seed_on_challenge(
        seed_dir,
        raw_texts,
        y_sent,
        y_tox,
        oov_provider,
        torch.device("cpu"),
        batch_size=8,
    )
    assert res["oov_hits"] == 2
    assert res["cold_throughput"] > 0
    assert res["warm_throughput"] > 0
    assert len(res["preds_df"]) == 40
    assert set(res["metrics"]["fpr_by_sentiment"].keys()) == {
        "Negative (-1)",
        "Neutral (0)",
        "Positive (1)",
    }
