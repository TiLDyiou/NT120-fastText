"""Phase 6: Final retraining of winning config C* on (train + valid) across 3 independent seeds."""

from __future__ import annotations

import argparse
import gc
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch

from src_v2.calibrate import compute_tau_k
from src_v2.config import (
    ARTIFACTS_DIR,
    CONFIGS_DIR,
    FINAL_SEEDS,
    LOCAL_ROOT,
    PROCESSED_DATA_DIR,
    RUNS_DIR,
    ExperimentConfig,
    ensure_project_dirs,
)
from src_v2.model import ParallelNet, compute_tox_class_weights
from src_v2.predict import DynamicPredictor, apply_postproc
from src_v2.train import load_artifact_bundle, train_and_evaluate
from src_v2.utils.logging_utils import get_git_commit


def load_final_seed_model(seed_dir: Path, device: torch.device) -> tuple[list[ParallelNet], dict[str, Any], dict[str, int]]:
    """Reload a saved Phase 6 final seed model directory from disk."""
    cfg_data = json.loads((seed_dir / "config.json").read_text(encoding="utf-8"))
    token_to_id = json.loads((seed_dir / "token_to_id.json").read_text(encoding="utf-8"))
    m_cfg = cfg_data["model_config"]

    weights_obj = torch.load(seed_dir / "model_weights.pt", map_location=device, weights_only=True)
    emb_npy_path = seed_dir / "embedding_matrix.npy"
    if emb_npy_path.exists():
        emb_placeholder = np.load(emb_npy_path)
    else:
        vocab_size = int(cfg_data.get("vocab_size", len(token_to_id)))
        emb_placeholder = np.zeros((vocab_size, 300), dtype=np.float32)

    if m_cfg["task_mode"] == "multi_task":
        model = ParallelNet(
            embedding_matrix=emb_placeholder,
            hidden_units=m_cfg["hidden_units"],
            dropout=m_cfg["dropout"],
            use_dense300=m_cfg["use_dense300"],
            task_mode="multi_task",
        )
        model.load_state_dict(weights_obj)
        model.to(device).eval()
        models = [model]
    else:
        m_sent = ParallelNet(
            embedding_matrix=emb_placeholder,
            hidden_units=m_cfg["hidden_units"],
            dropout=m_cfg["dropout"],
            use_dense300=m_cfg["use_dense300"],
            task_mode="single_task",
            active_task="sentiment",
        )
        m_tox = ParallelNet(
            embedding_matrix=emb_placeholder,
            hidden_units=m_cfg["hidden_units"],
            dropout=m_cfg["dropout"],
            use_dense300=m_cfg["use_dense300"],
            task_mode="single_task",
            active_task="toxicity",
        )
        m_sent.load_state_dict(weights_obj["sentiment"])
        m_tox.load_state_dict(weights_obj["toxicity"])
        m_sent.to(device).eval()
        m_tox.to(device).eval()
        models = [m_sent, m_tox]

    return models, cfg_data, token_to_id


def run_final_training(
    processed_dir: Path = PROCESSED_DATA_DIR,
    artifacts_dir: Path = ARTIFACTS_DIR,
    configs_dir: Path = CONFIGS_DIR,
    runs_dir: Path = RUNS_DIR,
) -> list[dict[str, Any]]:
    """Train winning C* on full_train (train + valid) for seeds 1, 2, 3 and verify disk reload."""
    ensure_project_dirs()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    best_payload = json.loads((configs_dir / "best_config.json").read_text(encoding="utf-8"))
    postproc = json.loads((artifacts_dir / "postproc.json").read_text(encoding="utf-8"))
    base_cfg = ExperimentConfig.from_dict(best_payload["config"])

    train_df = pd.read_parquet(processed_dir / "train.parquet")
    valid_df = pd.read_parquet(processed_dir / "valid.parquet")
    full_df = pd.concat([train_df, valid_df], ignore_index=True)

    bundle_full = load_artifact_bundle(artifacts_dir / "full", oov_prefix=None)
    meta_full = bundle_full["meta"]
    token_to_id_full = bundle_full["token_to_id"]
    max_len_full = int(meta_full["max_len"])
    tox_counts_full = [int(x) for x in meta_full["tox_counts"]]
    f_k_full = [float(x) for x in meta_full["tox_freq"]]
    w_k_full = compute_tox_class_weights(tox_counts_full, base_cfg.alpha, base_cfg.clip_max).tolist()

    t_bar_sent = float(postproc["t_bar_sent"])
    t_bar_tox = float(postproc["t_bar_tox"])
    beta_final = float(postproc["beta_final"])
    tau_k = compute_tau_k(f_k_full, beta_final).tolist()
    git_commit = get_git_commit(LOCAL_ROOT.parent)

    sample_100 = full_df.iloc[:100].reset_index(drop=True)
    summaries: list[dict[str, Any]] = []

    for seed in FINAL_SEEDS:
        seed_dir = runs_dir / "final_models" / f"seed_{seed}"
        seed_dir.mkdir(parents=True, exist_ok=True)
        cfg_s = ExperimentConfig(**{**base_cfg.to_dict(), "round_name": "final_train", "seed": seed})

        print(f"\n[Phase 6] Training final model seed={seed} on full_train (N={len(full_df):,})...")
        res = train_and_evaluate(cfg_s, full_df, bundle_full, eval_df=None, device=device)
        models = res["models"]

        # Predict 100 sample rows right before saving
        live_pred = DynamicPredictor(
            models=models,
            token_to_id=token_to_id_full,
            max_len=max_len_full,
            oov_provider=None,
            device=device,
            clone_models=True,
        )
        live_ls, live_lt = live_pred.predict_logits_from_tokens(sample_100["tokens"].tolist(), batch_size=100)

        # Save artifacts required by PIPELINE_SPEC §6.4
        if cfg_s.task_mode == "multi_task":
            torch.save(models[0].state_dict(), seed_dir / "model_weights.pt")
            trained_emb = models[0].embedding.weight.detach().cpu().numpy().astype(np.float32)
            np.save(seed_dir / "embedding_matrix.npy", trained_emb)
        else:
            torch.save(
                {"sentiment": models[0].state_dict(), "toxicity": models[1].state_dict()},
                seed_dir / "model_weights.pt",
            )
            np.save(
                seed_dir / "embedding_matrix_sent.npy",
                models[0].embedding.weight.detach().cpu().numpy().astype(np.float32),
            )
            np.save(
                seed_dir / "embedding_matrix_tox.npy",
                models[1].embedding.weight.detach().cpu().numpy().astype(np.float32),
            )
            np.save(
                seed_dir / "embedding_matrix.npy",
                models[0].embedding.weight.detach().cpu().numpy().astype(np.float32),
            )

        (seed_dir / "token_to_id.json").write_text(
            json.dumps(token_to_id_full, ensure_ascii=False), encoding="utf-8"
        )
        (seed_dir / "train_history.json").write_text(
            json.dumps(res["history"], ensure_ascii=False, indent=2), encoding="utf-8"
        )

        config_payload = {
            "winner_config_id": best_payload["winner_config_id"],
            "seed": seed,
            "model_config": cfg_s.to_dict(),
            "max_len_full": max_len_full,
            "vocab_size": int(meta_full["vocab_size"]),
            "num_full_train_rows": int(len(full_df)),
            "tox_counts_full": tox_counts_full,
            "w_k_full": w_k_full,
            "f_k_full": f_k_full,
            "t_bar_sent": t_bar_sent,
            "t_bar_tox": t_bar_tox,
            "beta_final": beta_final,
            "tau_k": tau_k,
            "trainable_params": res["trainable_params"],
            "total_params": res["total_params"],
            "train_seconds": res["train_seconds"],
            "git_commit": git_commit,
        }
        (seed_dir / "config.json").write_text(
            json.dumps(config_payload, ensure_ascii=False, indent=2), encoding="utf-8"
        )

        # Reload from disk and verify exact 100-row match (Phase 6 DoD)
        reloaded_models, reloaded_cfg, reloaded_vocab = load_final_seed_model(seed_dir, device)
        reloaded_pred = DynamicPredictor(
            models=reloaded_models,
            token_to_id=reloaded_vocab,
            max_len=reloaded_cfg["max_len_full"],
            oov_provider=None,
            device=device,
            clone_models=True,
        )
        rel_ls, rel_lt = reloaded_pred.predict_logits_from_tokens(sample_100["tokens"].tolist(), batch_size=100)
        max_diff = max(
            float(np.max(np.abs(live_ls - rel_ls))),
            float(np.max(np.abs(live_lt - rel_lt))),
        )
        assert max_diff < 1e-5, f"Reload mismatch for seed {seed}: {max_diff}"

        post_live = apply_postproc(live_ls, live_lt, t_bar_sent, t_bar_tox, np.array(tau_k))
        post_rel = apply_postproc(rel_ls, rel_lt, t_bar_sent, t_bar_tox, np.array(tau_k))
        assert np.array_equal(post_live["pred_sent_idx"], post_rel["pred_sent_idx"])
        assert np.array_equal(post_live["pred_tox_idx"], post_rel["pred_tox_idx"])

        print(f"  seed={seed} saved -> {seed_dir} | time={res['train_seconds']:.1f}s | reload_diff={max_diff:.2e}")
        summaries.append({"seed": seed, "train_seconds": res["train_seconds"], "reload_diff": max_diff})

        del res, models, reloaded_models
        gc.collect()
        if device.type == "cuda":
            torch.cuda.empty_cache()

    return summaries


def main() -> None:
    parser = argparse.ArgumentParser(description="Run Phase 6 final training on train+valid.")
    parser.parse_args()
    run_final_training()


if __name__ == "__main__":
    main()
