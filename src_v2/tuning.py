"""Phase 4: Three-round hyperparameter tuning on valid with rigid selection & tie-break rules."""

from __future__ import annotations

import argparse
import gc
import json
import random
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch

from src_v2.config import (
    ARTIFACTS_DIR,
    CONFIGS_DIR,
    FINAL_SEEDS,
    LOCAL_ROOT,
    PROCESSED_DATA_DIR,
    RANDOM_SEARCH_SPACE,
    REPORTS_DIR,
    RUNS_DIR,
    ExperimentConfig,
    ensure_project_dirs,
)
from src_v2.metrics import paired_bootstrap_f1_diff
from src_v2.train import load_artifact_bundle, train_and_evaluate
from src_v2.utils.logging_utils import append_jsonl, get_git_commit, read_jsonl


def build_round1_configs() -> list[ExperimentConfig]:
    """Build the 7 Round 1 mechanism-ablation configurations per PIPELINE_SPEC §4.2."""
    base = ExperimentConfig(config_id="C0", round_name="round1", seed=1)
    configs = [
        base,
        ExperimentConfig(**{**base.to_dict(), "config_id": "C1a", "emb_mode": "open"}),
        ExperimentConfig(**{**base.to_dict(), "config_id": "C1b", "emb_mode": "frozen"}),
        ExperimentConfig(**{**base.to_dict(), "config_id": "C2", "use_dense300": True}),
        ExperimentConfig(**{**base.to_dict(), "config_id": "C3", "task_mode": "single_task"}),
        ExperimentConfig(**{**base.to_dict(), "config_id": "C4a", "loss_mode": "focal_tox", "gamma": 2.0}),
        ExperimentConfig(**{**base.to_dict(), "config_id": "C4b", "loss_mode": "uncertainty"}),
    ]
    return configs


def select_round1_frame(round1_records: list[dict[str, Any]]) -> tuple[ExperimentConfig, bool, dict[str, Any]]:
    """Determine C_comb from Round 1 results; return (candidate_frame_cfg, need_run_comb, summary)."""
    by_id = {r["config_id"]: r for r in round1_records}
    c0_score = by_id["C0"]["combined_macro_f1"]

    # Group 1: Embedding mode (C0 freeze_then_open vs C1a open vs C1b frozen)
    emb_winner = max(["C0", "C1a", "C1b"], key=lambda cid: by_id[cid]["combined_macro_f1"])
    best_emb_mode = by_id[emb_winner]["emb_mode"] if by_id[emb_winner]["combined_macro_f1"] > c0_score else by_id["C0"]["emb_mode"]

    # Group 2: Dense 300 (C0 False vs C2 True)
    best_dense300 = True if by_id["C2"]["combined_macro_f1"] > c0_score else False

    # Group 3: Task mode (C0 multi_task vs C3 single_task)
    best_task_mode = "single_task" if by_id["C3"]["combined_macro_f1"] > c0_score else "multi_task"

    # Group 4: Loss mode (C0 weighted_ce vs C4a focal_tox vs C4b uncertainty)
    # Note: uncertainty is only meaningful for multi_task
    loss_candidates = ["C0", "C4a"] if best_task_mode == "single_task" else ["C0", "C4a", "C4b"]
    loss_winner = max(loss_candidates, key=lambda cid: by_id[cid]["combined_macro_f1"])
    if by_id[loss_winner]["combined_macro_f1"] > c0_score:
        best_loss_mode = by_id[loss_winner]["loss_mode"]
    else:
        best_loss_mode = "weighted_ce"

    c_comb = ExperimentConfig(
        config_id="C_comb",
        round_name="round1_comb",
        seed=1,
        emb_mode=best_emb_mode,
        use_dense300=best_dense300,
        task_mode=best_task_mode,
        loss_mode=best_loss_mode,
        gamma=2.0,
    )

    # Check if C_comb is identical in frame to any of the 7 already-run configs
    frame_key = lambda d: (d["emb_mode"], d["use_dense300"], d["task_mode"], d["loss_mode"])
    existing_keys = {frame_key(by_id[cid]): cid for cid in ["C0", "C1a", "C1b", "C2", "C3", "C4a", "C4b"]}
    matched_id = existing_keys.get(frame_key(c_comb.to_dict()))
    need_run_comb = matched_id is None

    best_single_id = max(["C0", "C1a", "C1b", "C2", "C3", "C4a", "C4b"], key=lambda cid: by_id[cid]["combined_macro_f1"])
    summary = {
        "best_emb_mode": best_emb_mode,
        "best_dense300": best_dense300,
        "best_task_mode": best_task_mode,
        "best_loss_mode": best_loss_mode,
        "matched_existing_id": matched_id,
        "best_single_id": best_single_id,
        "best_single_score": by_id[best_single_id]["combined_macro_f1"],
    }
    return c_comb, need_run_comb, summary


def canonical_config_key(cfg: ExperimentConfig) -> tuple[Any, ...]:
    """Canonical key ignoring conditional/unused hyperparameters to prevent duplicate runs."""
    freeze_ep = cfg.freeze_epochs if cfg.emb_mode == "freeze_then_open" else 2
    emb_factor = cfg.emb_lr_factor if cfg.emb_mode == "freeze_then_open" else 0.1
    clip_max = cfg.clip_max if cfg.alpha == 0.5 else None
    w_tox = cfg.w_tox if (cfg.loss_mode != "uncertainty" and cfg.task_mode != "single_task") else 1.0
    gamma = cfg.gamma if cfg.loss_mode == "focal_tox" else 2.0
    return (
        cfg.emb_mode,
        freeze_ep,
        emb_factor,
        cfg.use_dense300,
        cfg.task_mode,
        cfg.loss_mode,
        w_tox,
        cfg.alpha,
        clip_max,
        gamma,
        cfg.lr,
        cfg.epochs,
        cfg.dropout,
        cfg.hidden_units,
        cfg.weight_decay,
    )


def sample_round2_configs(
    frame_cfg: ExperimentConfig,
    existing_keys: set[tuple[Any, ...]],
    n_samples: int = 25,
    rng_seed: int = 2026,
) -> list[ExperimentConfig]:
    """Sample n_samples unique non-overlapping Round 2 configs respecting all conditional rules."""
    rng = random.Random(rng_seed)
    sampled: list[ExperimentConfig] = []
    seen = set(existing_keys)
    attempts = 0
    max_attempts = 10000

    while len(sampled) < n_samples and attempts < max_attempts:
        attempts += 1
        epochs = rng.choice(RANDOM_SEARCH_SPACE["epochs"])
        alpha = rng.choice(RANDOM_SEARCH_SPACE["alpha"])
        clip_max = rng.choice(RANDOM_SEARCH_SPACE["clip_max"]) if alpha == 0.5 else None
        w_tox = (
            rng.choice(RANDOM_SEARCH_SPACE["w_tox"])
            if (frame_cfg.loss_mode != "uncertainty" and frame_cfg.task_mode != "single_task")
            else 1.0
        )
        gamma = (
            rng.choice(RANDOM_SEARCH_SPACE["gamma"])
            if frame_cfg.loss_mode == "focal_tox"
            else 2.0
        )

        if frame_cfg.emb_mode == "freeze_then_open":
            valid_f = [f for f in RANDOM_SEARCH_SPACE["freeze_epochs"] if f < epochs]
            freeze_epochs = rng.choice(valid_f)
            emb_lr_factor = rng.choice(RANDOM_SEARCH_SPACE["emb_lr_factor"])
        else:
            freeze_epochs = 2
            emb_lr_factor = 0.1

        cand = ExperimentConfig(
            config_id=f"R2_{len(sampled) + 1:02d}",
            round_name="round2",
            seed=1,
            emb_mode=frame_cfg.emb_mode,
            freeze_epochs=freeze_epochs,
            emb_lr_factor=emb_lr_factor,
            use_dense300=frame_cfg.use_dense300,
            task_mode=frame_cfg.task_mode,
            loss_mode=frame_cfg.loss_mode,
            w_tox=w_tox,
            alpha=alpha,
            clip_max=clip_max,
            gamma=gamma,
            lr=rng.choice(RANDOM_SEARCH_SPACE["lr"]),
            epochs=epochs,
            dropout=rng.choice(RANDOM_SEARCH_SPACE["dropout"]),
            hidden_units=rng.choice(RANDOM_SEARCH_SPACE["hidden_units"]),
            weight_decay=rng.choice(RANDOM_SEARCH_SPACE["weight_decay"]),
        )
        cand.validate()
        key = canonical_config_key(cand)
        if key not in seen:
            seen.add(key)
            sampled.append(cand)

    if len(sampled) < n_samples:
        raise RuntimeError(f"Only sampled {len(sampled)}/{n_samples} unique configs.")
    return sampled


def select_round3_winner(
    candidate_stats: list[dict[str, Any]],
    y_sent_val: np.ndarray,
    y_tox_val: np.ndarray,
    n_boot: int = 1000,
    boot_seed: int = 42,
) -> dict[str, Any]:
    """Select C* from Top-3 multi-seed results using Seed Std + Paired Bootstrap + Tie-break."""
    ranked = sorted(candidate_stats, key=lambda d: (-d["mean_f1"], d["trainable_params"]))
    a, b = ranked[0], ranked[1]
    c = ranked[2] if len(ranked) > 2 else None

    mu_diff_ab = a["mean_f1"] - b["mean_f1"]
    max_std_ab = max(a["std_f1"], b["std_f1"])
    seed_tied = bool(mu_diff_ab <= max_std_ab)

    boot_ab = paired_bootstrap_f1_diff(
        y_sent_val, y_tox_val, a["preds_3seed"], b["preds_3seed"], n_boot=n_boot, seed=boot_seed
    )
    boot_tied = bool(boot_ab["contains_zero"])

    if seed_tied or boot_tied:
        if b["trainable_params"] < a["trainable_params"]:
            winner = b
            reason = "Tie-break triggered (Seed Std or Bootstrap 95% CI contains 0); B has fewer trainable params."
        elif a["trainable_params"] < b["trainable_params"]:
            winner = a
            reason = "Tie-break triggered; A has fewer trainable params."
        else:
            winner = a
            reason = "Tie-break triggered with equal param count; kept higher mean F1 (A)."
    else:
        winner = a
        reason = "A strictly outperforms B on both Seed Std check and Paired Bootstrap 95% CI."

    comparisons = {"A_vs_B": {"seed_tied": seed_tied, "mu_diff": mu_diff_ab, "max_std": max_std_ab, "bootstrap": boot_ab}}
    if c is not None:
        boot_ac = paired_bootstrap_f1_diff(
            y_sent_val, y_tox_val, a["preds_3seed"], c["preds_3seed"], n_boot=n_boot, seed=boot_seed
        )
        comparisons["A_vs_C"] = {
            "seed_tied": bool((a["mean_f1"] - c["mean_f1"]) <= max(a["std_f1"], c["std_f1"])),
            "mu_diff": a["mean_f1"] - c["mean_f1"],
            "max_std": max(a["std_f1"], c["std_f1"]),
            "bootstrap": boot_ac,
        }

    return {
        "winner_config_id": winner["config_id"],
        "winner_config": winner["config"],
        "reason": reason,
        "ranked_summary": [
            {
                "config_id": x["config_id"],
                "mean_f1": x["mean_f1"],
                "std_f1": x["std_f1"],
                "mean_sent_f1": x["mean_sent_f1"],
                "mean_tox_f1": x["mean_tox_f1"],
                "trainable_params": x["trainable_params"],
                "total_params": x["total_params"],
            }
            for x in ranked
        ],
        "comparisons": comparisons,
    }


def _record_from_run(
    cfg: ExperimentConfig,
    run_res: dict[str, Any],
    git_commit: str,
) -> dict[str, Any]:
    ev = run_res["eval"]
    return {
        **cfg.to_dict(),
        "trainable_params": run_res["trainable_params"],
        "total_params": run_res["total_params"],
        "train_seconds": round(run_res["train_seconds"], 3),
        "macro_f1_sent": ev["macro_f1_sent"],
        "macro_f1_tox": ev["macro_f1_tox"],
        "combined_macro_f1": ev["combined_macro_f1"],
        "oov_hits": ev["oov_hits"],
        "git_commit": git_commit,
    }


def _run_or_resume(
    cfg: ExperimentConfig,
    train_df: pd.DataFrame,
    valid_df: pd.DataFrame,
    bundle: dict[str, Any],
    log_path: Path,
    cache_dir: Path,
    git_commit: str,
    device: torch.device,
    save_logits: bool = False,
) -> dict[str, Any]:
    records = read_jsonl(log_path)
    cache_file = cache_dir / f"{cfg.config_id}_seed{cfg.seed}.npz"

    for r in records:
        if r["round_name"] == cfg.round_name and r["config_id"] == cfg.config_id and int(r["seed"]) == int(cfg.seed):
            if not save_logits or cache_file.exists():
                print(f"  [RESUME] {cfg.round_name}/{cfg.config_id} seed={cfg.seed} -> combined={r['combined_macro_f1']:.4f}")
                return r

    print(
        f"  [RUN] {cfg.round_name}/{cfg.config_id} seed={cfg.seed} | "
        f"emb={cfg.emb_mode}(F={cfg.freeze_epochs},fac={cfg.emb_lr_factor}) dense={cfg.use_dense300} "
        f"task={cfg.task_mode} loss={cfg.loss_mode} H={cfg.hidden_units} E={cfg.epochs} lr={cfg.lr}"
    )
    res = train_and_evaluate(cfg, train_df, bundle, eval_df=valid_df, device=device)
    rec = _record_from_run(cfg, res, git_commit)
    append_jsonl(log_path, rec)

    if save_logits:
        cache_dir.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            cache_file,
            logits_sent=res["eval"]["logits_sent"],
            logits_tox=res["eval"]["logits_tox"],
        )

    print(
        f"        -> sent={rec['macro_f1_sent']:.4f} tox={rec['macro_f1_tox']:.4f} "
        f"combined={rec['combined_macro_f1']:.4f} ({rec['train_seconds']:.1f}s, params={rec['trainable_params']:,})"
    )
    del res
    gc.collect()
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return rec


def run_tuning(
    n_random_configs: int = 25,
    processed_dir: Path = PROCESSED_DATA_DIR,
    artifacts_dir: Path = ARTIFACTS_DIR,
    runs_dir: Path = RUNS_DIR,
    configs_dir: Path = CONFIGS_DIR,
    reports_dir: Path = REPORTS_DIR,
) -> dict[str, Any]:
    """Execute Round 1 -> Round 2 -> Round 3 tuning pipeline and persist winning C*."""
    ensure_project_dirs()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Running Phase 4 tuning on device: {device}")

    train_df = pd.read_parquet(processed_dir / "train.parquet")
    valid_df = pd.read_parquet(processed_dir / "valid.parquet")
    bundle = load_artifact_bundle(artifacts_dir / "train", oov_prefix="valid")

    log_path = runs_dir / "tuning_log.jsonl"
    cache_dir = runs_dir / "tuning_cache"
    git_commit = get_git_commit(LOCAL_ROOT.parent)

    # --- ROUND 1 ---
    print("\n=== ROUND 1: Mechanism Ablation (7 configs, seed=1) ===")
    r1_configs = build_round1_configs()
    r1_records = [
        _run_or_resume(cfg, train_df, valid_df, bundle, log_path, cache_dir, git_commit, device)
        for cfg in r1_configs
    ]
    c_comb, need_comb, r1_summary = select_round1_frame(r1_records)

    comb_rec = None
    if need_comb:
        print("\n=== ROUND 1 COMBINED CHECK (C_comb) ===")
        comb_rec = _run_or_resume(c_comb, train_df, valid_df, bundle, log_path, cache_dir, git_commit, device)
        if comb_rec["combined_macro_f1"] >= r1_summary["best_single_score"]:
            frame_cfg = c_comb
            frame_source = "C_comb"
        else:
            best_id = r1_summary["best_single_id"]
            frame_cfg = next(c for c in r1_configs if c.config_id == best_id)
            frame_source = f"best_single ({best_id}) because C_comb={comb_rec['combined_macro_f1']:.4f} < {r1_summary['best_single_score']:.4f}"
    else:
        matched_id = r1_summary["matched_existing_id"]
        frame_cfg = next(c for c in r1_configs if c.config_id == matched_id)
        frame_source = f"existing ({matched_id})"

    print(f"\n[Round 1 Decision] Locked frame from {frame_source}: "
          f"emb={frame_cfg.emb_mode}, dense300={frame_cfg.use_dense300}, "
          f"task={frame_cfg.task_mode}, loss={frame_cfg.loss_mode}")

    # --- ROUND 2 ---
    print(f"\n=== ROUND 2: Random Search ({n_random_configs} configs, seed=1) ===")
    all_r1 = read_jsonl(log_path)
    existing_keys = {
        canonical_config_key(ExperimentConfig.from_dict(r))
        for r in all_r1
        if r["round_name"].startswith("round1")
    }
    r2_configs = sample_round2_configs(frame_cfg, existing_keys, n_samples=n_random_configs, rng_seed=2026)
    for cfg in r2_configs:
        _run_or_resume(cfg, train_df, valid_df, bundle, log_path, cache_dir, git_commit, device)

    # --- ROUND 3 ---
    print("\n=== ROUND 3: Top-3 Multi-Seed Confirmation (seeds 1, 2, 3) ===")
    all_r1_r2 = [
        r for r in read_jsonl(log_path)
        if r["round_name"] in {"round1", "round1_comb", "round2"}
    ]
    # Deduplicate by config_id keeping best/first
    seen_ids: dict[str, dict[str, Any]] = {}
    for r in all_r1_r2:
        if r["config_id"] not in seen_ids:
            seen_ids[r["config_id"]] = r
    top3_records = sorted(seen_ids.values(), key=lambda r: -r["combined_macro_f1"])[:3]
    print("Top-3 candidates from Round 1 + Round 2:")
    for idx, r in enumerate(top3_records, 1):
        print(f"  #{idx}: {r['config_id']} ({r['round_name']}) -> combined={r['combined_macro_f1']:.4f}")

    y_sent_val = valid_df["y_sent_idx"].to_numpy()
    y_tox_val = valid_df["y_tox_idx"].to_numpy()
    candidate_stats: list[dict[str, Any]] = []

    for base_rec in top3_records:
        base_cfg = ExperimentConfig.from_dict(base_rec)
        seed_recs = []
        preds_3seed = []
        for s in FINAL_SEEDS:
            r3_cfg = ExperimentConfig(**{**base_cfg.to_dict(), "round_name": "round3", "seed": s})
            rec = _run_or_resume(
                r3_cfg, train_df, valid_df, bundle, log_path, cache_dir, git_commit, device, save_logits=True
            )
            seed_recs.append(rec)
            npz = np.load(cache_dir / f"{r3_cfg.config_id}_seed{s}.npz")
            preds_3seed.append((npz["logits_sent"].argmax(axis=1), npz["logits_tox"].argmax(axis=1)))

        f1s = [r["combined_macro_f1"] for r in seed_recs]
        sent_f1s = [r["macro_f1_sent"] for r in seed_recs]
        tox_f1s = [r["macro_f1_tox"] for r in seed_recs]
        candidate_stats.append(
            {
                "config_id": base_cfg.config_id,
                "config": base_cfg.to_dict(),
                "mean_f1": float(np.mean(f1s)),
                "std_f1": float(np.std(f1s, ddof=1)),
                "mean_sent_f1": float(np.mean(sent_f1s)),
                "mean_tox_f1": float(np.mean(tox_f1s)),
                "trainable_params": int(seed_recs[0]["trainable_params"]),
                "total_params": int(seed_recs[0]["total_params"]),
                "preds_3seed": preds_3seed,
            }
        )

    decision = select_round3_winner(candidate_stats, y_sent_val, y_tox_val, n_boot=1000, boot_seed=42)

    # Export ablation_table.csv, best_config.json, and tuning_decision.md
    all_records = read_jsonl(log_path)
    ablation_df = pd.DataFrame(all_records)
    ablation_df.to_csv(reports_dir / "ablation_table.csv", index=False, encoding="utf-8-sig")

    best_cfg_payload = {
        "winner_config_id": decision["winner_config_id"],
        "config": decision["winner_config"],
        "round1_frame_source": frame_source,
        "decision_reason": decision["reason"],
        "ranked_summary": decision["ranked_summary"],
    }
    (configs_dir / "best_config.json").write_text(
        json.dumps(best_cfg_payload, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    r1_table_records = r1_records + ([comb_rec] if comb_rec is not None else [])
    _write_tuning_decision_report(
        reports_dir / "tuning_decision.md",
        r1_table_records,
        frame_cfg,
        frame_source,
        decision,
    )
    print(f"\n[PHASE 4 COMPLETE] Winner C* = {decision['winner_config_id']} -> saved to {configs_dir / 'best_config.json'}")
    return decision


def _write_tuning_decision_report(
    report_path: Path,
    r1_records: list[dict[str, Any]],
    frame_cfg: ExperimentConfig,
    frame_source: str,
    decision: dict[str, Any],
) -> None:
    ab = decision["comparisons"]["A_vs_B"]
    lines = [
        "# Báo cáo Quyết định Tuning 3 Vòng (Phase 4)",
        "",
        "## 1. Kết quả Vòng 1 — Khảo sát Cơ chế (`seed = 1`)",
        "",
        "| Mã | Emb Mode | Dense300 | Task Mode | Loss | F1 Sent | F1 Tox | **Combined F1** | Tham số học được |",
        "| :--- | :--- | :---: | :--- | :--- | :---: | :---: | :---: | :---: |",
    ]
    for r in r1_records:
        lines.append(
            f"| `{r['config_id']}` | `{r['emb_mode']}` | `{r['use_dense300']}` | `{r['task_mode']}` | `{r['loss_mode']}` | "
            f"{r['macro_f1_sent']:.4f} | {r['macro_f1_tox']:.4f} | **{r['combined_macro_f1']:.4f}** | {r['trainable_params']:,} |"
        )
    lines.extend([
        "",
        f"- **Chốt khung chuyển sang Vòng 2:** `{frame_source}` (`emb={frame_cfg.emb_mode}`, `dense300={frame_cfg.use_dense300}`, `task={frame_cfg.task_mode}`, `loss={frame_cfg.loss_mode}`).",
        "",
        "## 2. Kết quả Vòng 3 — Xác nhận Đa Seed (`seed ∈ {1, 2, 3}`)",
        "",
        "| Hạng | Cấu hình | Sent F1 (Mean) | Tox F1 (Mean) | **Combined F1 (Mean ± Std)** | Tham số học được |",
        "| :---: | :--- | :---: | :---: | :---: | :---: |",
    ])
    for idx, x in enumerate(decision["ranked_summary"], 1):
        lines.append(
            f"| #{idx} | `{x['config_id']}` | {x['mean_sent_f1']:.4f} | {x['mean_tox_f1']:.4f} | "
            f"**{x['mean_f1']:.4f} ± {x['std_f1']:.4f}** | {x['trainable_params']:,} |"
        )
    lines.extend([
        "",
        "## 3. Kiểm tra Thống kê & Quy tắc Tie-break (A vs B)",
        "",
        f"- **Chênh lệch $\\mu_A - \\mu_B$:** `{ab['mu_diff']:.6f}` vs $\\max(\\sigma_A, \\sigma_B) = {ab['max_std']:.6f}$ → Seed Std Tied = `{ab['seed_tied']}`",
        f"- **Paired Bootstrap 95% CI (1000 lần):** `[{ab['bootstrap']['ci_low']:.6f}, {ab['bootstrap']['ci_high']:.6f}]` → Chứa 0 = `{ab['bootstrap']['contains_zero']}`",
    ])
    if "A_vs_C" in decision["comparisons"]:
        ac = decision["comparisons"]["A_vs_C"]
        lines.append(
            f"- *(Tham chiếu A vs C)* $\\mu_A - \\mu_C = {ac['mu_diff']:.6f}$ (max std `{ac['max_std']:.6f}`, tied=`{ac['seed_tied']}`), "
            f"Bootstrap CI `[{ac['bootstrap']['ci_low']:.6f}, {ac['bootstrap']['ci_high']:.6f}]` (chứa 0=`{ac['bootstrap']['contains_zero']}`)."
        )
    lines.extend([
        "",
        f"### Kết luận chọn cấu hình chiến thắng $C^*$: **`{decision['winner_config_id']}`**",
        f"- **Lý do:** {decision['reason']}",
        "",
    ])
    report_path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Run Phase 4 three-round tuning pipeline.")
    parser.add_argument("--n-random", type=int, default=25, help="Number of Round 2 configs (20-30)")
    args = parser.parse_args()
    run_tuning(n_random_configs=args.n_random)


if __name__ == "__main__":
    main()
