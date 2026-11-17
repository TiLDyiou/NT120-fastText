"""Unit tests for Phase 4 Round 1/2/3 selection, constraints, and tie-break logic."""

from __future__ import annotations

import numpy as np

from src_v2.config import ExperimentConfig
from src_v2.tuning import (
    build_round1_configs,
    canonical_config_key,
    sample_round2_configs,
    select_round1_frame,
    select_round3_winner,
)


def test_round1_comb_selection_and_single_task_uncertainty_guard() -> None:
    configs = build_round1_configs()
    assert len(configs) == 7

    # Case 1: C2 (dense300) and C4a (focal_tox) beat C0 -> C_comb needed
    recs = [
        {**c.to_dict(), "combined_macro_f1": 0.60} for c in configs
    ]
    recs[3]["combined_macro_f1"] = 0.63  # C2
    recs[5]["combined_macro_f1"] = 0.64  # C4a
    c_comb, need_comb, _ = select_round1_frame(recs)
    assert need_comb is True
    assert c_comb.use_dense300 is True
    assert c_comb.loss_mode == "focal_tox"

    # Case 2: C3 (single_task) and C4b (uncertainty) beat C0 -> uncertainty excluded for single_task
    recs2 = [{**c.to_dict(), "combined_macro_f1": 0.60} for c in configs]
    recs2[4]["combined_macro_f1"] = 0.66  # C3 single_task
    recs2[6]["combined_macro_f1"] = 0.65  # C4b uncertainty
    c_comb2, _, _ = select_round1_frame(recs2)
    assert c_comb2.task_mode == "single_task"
    assert c_comb2.loss_mode == "weighted_ce"


def test_round2_sampling_respects_f_lt_e_and_uniqueness() -> None:
    frame = ExperimentConfig(emb_mode="freeze_then_open", loss_mode="focal_tox")
    sampled = sample_round2_configs(frame, existing_keys=set(), n_samples=25, rng_seed=2026)
    assert len(sampled) == 25
    keys = {canonical_config_key(c) for c in sampled}
    assert len(keys) == 25
    for c in sampled:
        c.validate()
        assert c.freeze_epochs < c.epochs


def test_round3_tie_break_logic() -> None:
    y_sent = np.array([0, 1, 2] * 30)
    y_tox = np.array([0, 1, 2, 3] * 22 + [0, 1])

    # Perfect predictions for A, identical predictions for B (so bootstrap diff is 0 and std tied), C worse
    preds_a = [(y_sent.copy(), y_tox.copy()) for _ in range(3)]
    preds_b = [(y_sent.copy(), y_tox.copy()) for _ in range(3)]
    preds_c = [(np.zeros_like(y_sent), np.zeros_like(y_tox)) for _ in range(3)]

    candidates = [
        {
            "config_id": "A_big",
            "config": ExperimentConfig(config_id="A_big", hidden_units=256).to_dict(),
            "mean_f1": 0.701,
            "std_f1": 0.005,
            "mean_sent_f1": 0.70,
            "mean_tox_f1": 0.702,
            "trainable_params": 2000000,
            "total_params": 2000000,
            "preds_3seed": preds_a,
        },
        {
            "config_id": "B_small",
            "config": ExperimentConfig(config_id="B_small", hidden_units=64).to_dict(),
            "mean_f1": 0.700,
            "std_f1": 0.004,
            "mean_sent_f1": 0.70,
            "mean_tox_f1": 0.700,
            "trainable_params": 500000,
            "total_params": 500000,
            "preds_3seed": preds_b,
        },
        {
            "config_id": "C_worse",
            "config": ExperimentConfig(config_id="C_worse", hidden_units=64).to_dict(),
            "mean_f1": 0.500,
            "std_f1": 0.010,
            "mean_sent_f1": 0.50,
            "mean_tox_f1": 0.50,
            "trainable_params": 400000,
            "total_params": 400000,
            "preds_3seed": preds_c,
        },
    ]

    dec = select_round3_winner(candidates, y_sent, y_tox, n_boot=200, boot_seed=42)
    assert dec["winner_config_id"] == "B_small"
    assert dec["comparisons"]["A_vs_B"]["seed_tied"] is True
    assert dec["comparisons"]["A_vs_B"]["bootstrap"]["contains_zero"] is True
