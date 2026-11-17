"""Phase 5: Temperature Scaling, ECE reporting, beta prior adjustment, and V1/V2 transfer check."""

from __future__ import annotations

import argparse
import gc
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from scipy.optimize import minimize

from src_v2.config import (
    ARTIFACTS_DIR,
    BETA_GRID,
    CONFIGS_DIR,
    FINAL_SEEDS,
    PROCESSED_DATA_DIR,
    REPORTS_DIR,
    RUNS_DIR,
    ExperimentConfig,
    ensure_project_dirs,
)
from src_v2.metrics import compute_ece, compute_macro_f1
from src_v2.train import load_artifact_bundle, train_and_evaluate


def _softmax_np(logits: np.ndarray) -> np.ndarray:
    z = np.asarray(logits, dtype=np.float64)
    shifted = z - z.max(axis=1, keepdims=True)
    exps = np.exp(shifted)
    return exps / exps.sum(axis=1, keepdims=True)


def _nll_loss_np(logits: np.ndarray, y_true: np.ndarray, temp: float) -> float:
    z = np.asarray(logits, dtype=np.float64)
    scaled = z / float(temp)
    shifted = scaled - scaled.max(axis=1, keepdims=True)
    log_sum_exp = np.log(np.exp(shifted).sum(axis=1))
    true_logits = shifted[np.arange(len(y_true)), y_true]
    return float(np.mean(log_sum_exp - true_logits))


def fit_temperature(logits: np.ndarray, y_true: np.ndarray) -> float:
    """Find scalar T > 0 minimizing unweighted NLL on validation logits (float64 + exact grad)."""
    z = np.asarray(logits, dtype=np.float64)
    y = np.asarray(y_true, dtype=np.int64)
    row_idx = np.arange(len(y))
    z_true = z[row_idx, y]

    def objective_and_grad(log_t_arr: np.ndarray) -> tuple[float, np.ndarray]:
        temp = float(np.exp(log_t_arr[0]))
        scaled = z / temp
        shifted = scaled - scaled.max(axis=1, keepdims=True)
        exps = np.exp(shifted)
        probs = exps / exps.sum(axis=1, keepdims=True)
        nll = float(np.mean(np.log(exps.sum(axis=1)) - shifted[row_idx, y]))
        # d(NLL)/d(log T) = (1 / T) * mean(z_true - sum_c p_c * z_c)
        expected_z = np.sum(probs * z, axis=1)
        grad_log_t = float(np.mean(z_true - expected_z) / temp)
        return nll, np.array([grad_log_t], dtype=np.float64)

    res = minimize(
        objective_and_grad,
        x0=np.array([0.0], dtype=np.float64),
        method="L-BFGS-B",
        jac=True,
        bounds=[(-5.0, 5.0)],
    )
    return float(np.exp(res.x[0]))


def compute_tau_k(tox_freq: list[float] | np.ndarray, beta: float) -> np.ndarray:
    """Compute tau_k(beta) = (f_0 / f_k)^beta for Toxicity classes k in {0..3}."""
    freqs = np.asarray(tox_freq, dtype=np.float64)
    return (freqs[0] / freqs) ** float(beta)


def evaluate_beta_grid(
    logits_tox_list: list[np.ndarray],
    y_tox: np.ndarray,
    t_tox: float,
    tox_freq: list[float] | np.ndarray,
    beta_grid: list[float] = BETA_GRID,
) -> tuple[float, list[dict[str, float]]]:
    """Evaluate beta grid using calibrated probabilities and pick beta* maximizing mean 3-seed F1."""
    probs_list = [_softmax_np(z / float(t_tox)) for z in logits_tox_list]
    rows: list[dict[str, float]] = []

    for beta in beta_grid:
        tau = compute_tau_k(tox_freq, beta).reshape(1, -1)
        f1s = []
        for probs in probs_list:
            preds = (probs * tau).argmax(axis=1)
            f1s.append(compute_macro_f1(y_tox, preds, labels=[0, 1, 2, 3]))
        rows.append(
            {
                "beta": float(beta),
                "mean_f1_tox": float(np.mean(f1s)),
                "std_f1_tox": float(np.std(f1s, ddof=1)) if len(f1s) > 1 else 0.0,
            }
        )

    best_row = max(rows, key=lambda r: (round(r["mean_f1_tox"], 12), -r["beta"]))
    return float(best_row["beta"]), rows


def decide_beta_transfer(delta_train: float, delta_train_v1: float) -> bool:
    """Hard rule from PIPELINE_SPEC §5.3.5: keep beta* iff both conditions hold."""
    return bool(delta_train_v1 >= 0.0 and delta_train_v1 >= 0.5 * delta_train)


def run_calibration(
    processed_dir: Path = PROCESSED_DATA_DIR,
    artifacts_dir: Path = ARTIFACTS_DIR,
    configs_dir: Path = CONFIGS_DIR,
    runs_dir: Path = RUNS_DIR,
    reports_dir: Path = REPORTS_DIR,
) -> dict[str, Any]:
    """Run Phase 5 Temperature Scaling, beta search, and V1/V2 transfer check."""
    ensure_project_dirs()
    best_payload = json.loads((configs_dir / "best_config.json").read_text(encoding="utf-8"))
    winner_id = best_payload["winner_config_id"]
    base_cfg = ExperimentConfig.from_dict(best_payload["config"])

    valid_df = pd.read_parquet(processed_dir / "valid.parquet")
    v1_df = pd.read_parquet(processed_dir / "valid_v1.parquet")
    v2_df = pd.read_parquet(processed_dir / "valid_v2.parquet")
    train_df = pd.read_parquet(processed_dir / "train.parquet")

    train_meta = json.loads((artifacts_dir / "train" / "meta.json").read_text(encoding="utf-8"))
    tox_freq_train = train_meta["tox_freq"]

    y_sent_val = valid_df["y_sent_idx"].to_numpy()
    y_tox_val = valid_df["y_tox_idx"].to_numpy()

    logits_sent_seeds: list[np.ndarray] = []
    logits_tox_seeds: list[np.ndarray] = []
    cache_dir = runs_dir / "tuning_cache"
    for s in FINAL_SEEDS:
        npz = np.load(cache_dir / f"{winner_id}_seed{s}.npz")
        logits_sent_seeds.append(npz["logits_sent"])
        logits_tox_seeds.append(npz["logits_tox"])

    # --- 5.1 Temperature Scaling ---
    t_sent_per_seed = [fit_temperature(z, y_sent_val) for z in logits_sent_seeds]
    t_tox_per_seed = [fit_temperature(z, y_tox_val) for z in logits_tox_seeds]
    t_bar_sent = float(np.mean(t_sent_per_seed))
    t_bar_tox = float(np.mean(t_tox_per_seed))

    ece_sent_before, ece_sent_after = [], []
    ece_tox_before, ece_tox_after = [], []
    nll_sent_before, nll_sent_after = [], []
    nll_tox_before, nll_tox_after = [], []

    for zs, zt in zip(logits_sent_seeds, logits_tox_seeds):
        ps_before = _softmax_np(zs)
        ps_after = _softmax_np(zs / t_bar_sent)
        pt_before = _softmax_np(zt)
        pt_after = _softmax_np(zt / t_bar_tox)

        # Invariant check: Macro-F1 unchanged after dividing by positive scalar T
        f1_s_before = compute_macro_f1(y_sent_val, ps_before.argmax(axis=1), [0, 1, 2])
        f1_s_after = compute_macro_f1(y_sent_val, ps_after.argmax(axis=1), [0, 1, 2])
        f1_t_before = compute_macro_f1(y_tox_val, pt_before.argmax(axis=1), [0, 1, 2, 3])
        f1_t_after = compute_macro_f1(y_tox_val, pt_after.argmax(axis=1), [0, 1, 2, 3])
        assert abs(f1_s_before - f1_s_after) < 1e-12
        assert abs(f1_t_before - f1_t_after) < 1e-12

        ece_sent_before.append(compute_ece(ps_before, y_sent_val, n_bins=15))
        ece_sent_after.append(compute_ece(ps_after, y_sent_val, n_bins=15))
        ece_tox_before.append(compute_ece(pt_before, y_tox_val, n_bins=15))
        ece_tox_after.append(compute_ece(pt_after, y_tox_val, n_bins=15))

        nll_sent_before.append(_nll_loss_np(zs, y_sent_val, 1.0))
        nll_sent_after.append(_nll_loss_np(zs, y_sent_val, t_bar_sent))
        nll_tox_before.append(_nll_loss_np(zt, y_tox_val, 1.0))
        nll_tox_after.append(_nll_loss_np(zt, y_tox_val, t_bar_tox))

    # --- 5.2 Search beta* on full valid ---
    beta_star, beta_curve = evaluate_beta_grid(
        logits_tox_seeds, y_tox_val, t_bar_tox, tox_freq_train, BETA_GRID
    )

    # --- 5.3 V1/V2 transferability check if beta* > 0 ---
    v1v2_info: dict[str, Any] = {"ran": False}
    if beta_star == 0.0:
        beta_final = 0.0
    else:
        v1v2_info["ran"] = True
        text_to_val_idx = {t: i for i, t in enumerate(valid_df["text_clean"])}
        v1_idx = np.array([text_to_val_idx[t] for t in v1_df["text_clean"]], dtype=np.int64)
        v2_idx = np.array([text_to_val_idx[t] for t in v2_df["text_clean"]], dtype=np.int64)

        y_tox_v1 = y_tox_val[v1_idx]
        y_tox_v2 = y_tox_val[v2_idx]

        logits_tox_train_v1 = [z[v1_idx] for z in logits_tox_seeds]
        logits_tox_train_v2 = [z[v2_idx] for z in logits_tox_seeds]

        # Learn T_V1 on V1 only
        t_v1_seeds = [fit_temperature(z, y_tox_v1) for z in logits_tox_train_v1]
        t_bar_v1 = float(np.mean(t_v1_seeds))

        # Find beta_V1 on V1 using train-only models
        beta_v1, beta_v1_curve = evaluate_beta_grid(
            logits_tox_train_v1, y_tox_v1, t_bar_v1, tox_freq_train, BETA_GRID
        )

        # Evaluate train-only models on V2 with beta=0 and beta=beta_V1
        tau_0 = compute_tau_k(tox_freq_train, 0.0).reshape(1, -1)
        tau_v1_tr = compute_tau_k(tox_freq_train, beta_v1).reshape(1, -1)
        f1_train_v2_b0 = [
            compute_macro_f1(y_tox_v2, (_softmax_np(z / t_bar_v1) * tau_0).argmax(axis=1), [0, 1, 2, 3])
            for z in logits_tox_train_v2
        ]
        f1_train_v2_bv1 = [
            compute_macro_f1(y_tox_v2, (_softmax_np(z / t_bar_v1) * tau_v1_tr).argmax(axis=1), [0, 1, 2, 3])
            for z in logits_tox_train_v2
        ]
        delta_train = float(np.mean(f1_train_v2_bv1) - np.mean(f1_train_v2_b0))

        # Train 3 models on train + V1 and evaluate on V2
        bundle_tv1 = load_artifact_bundle(artifacts_dir / "trainV1", oov_prefix="v2")
        train_v1_df = pd.concat([train_df, v1_df], ignore_index=True)
        tox_freq_tv1 = bundle_tv1["meta"]["tox_freq"]
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        logits_tox_tv1_v2: list[np.ndarray] = []
        for s in FINAL_SEEDS:
            cfg_s = ExperimentConfig(**{**base_cfg.to_dict(), "round_name": "phase5_v1v2", "seed": s})
            print(f"  [Phase 5.3] Training {winner_id} on train+V1 (seed={s})...")
            res = train_and_evaluate(cfg_s, train_v1_df, bundle_tv1, eval_df=v2_df, device=device)
            logits_tox_tv1_v2.append(res["eval"]["logits_tox"])
            del res
            gc.collect()
            if device.type == "cuda":
                torch.cuda.empty_cache()

        tau_v1_tv1 = compute_tau_k(tox_freq_tv1, beta_v1).reshape(1, -1)
        f1_tv1_v2_b0 = [
            compute_macro_f1(y_tox_v2, (_softmax_np(z / t_bar_v1) * tau_0).argmax(axis=1), [0, 1, 2, 3])
            for z in logits_tox_tv1_v2
        ]
        f1_tv1_v2_bv1 = [
            compute_macro_f1(y_tox_v2, (_softmax_np(z / t_bar_v1) * tau_v1_tv1).argmax(axis=1), [0, 1, 2, 3])
            for z in logits_tox_tv1_v2
        ]
        delta_train_v1 = float(np.mean(f1_tv1_v2_bv1) - np.mean(f1_tv1_v2_b0))

        keep_beta = decide_beta_transfer(delta_train, delta_train_v1)
        beta_final = beta_star if keep_beta else 0.0
        v1v2_info.update(
            {
                "t_v1_per_seed": t_v1_seeds,
                "t_bar_v1": t_bar_v1,
                "beta_v1": beta_v1,
                "beta_v1_curve": beta_v1_curve,
                "f1_train_v2_b0_mean": float(np.mean(f1_train_v2_b0)),
                "f1_train_v2_bv1_mean": float(np.mean(f1_train_v2_bv1)),
                "delta_train": delta_train,
                "f1_trainV1_v2_b0_mean": float(np.mean(f1_tv1_v2_b0)),
                "f1_trainV1_v2_bv1_mean": float(np.mean(f1_tv1_v2_bv1)),
                "delta_train_v1": delta_train_v1,
                "keep_beta": keep_beta,
                "v2_class3_support": int((y_tox_v2 == 3).sum()),
            }
        )

    postproc = {
        "winner_config_id": winner_id,
        "t_sent_per_seed": t_sent_per_seed,
        "t_tox_per_seed": t_tox_per_seed,
        "t_bar_sent": t_bar_sent,
        "t_bar_tox": t_bar_tox,
        "ece": {
            "sent_before": [float(np.mean(ece_sent_before)), float(np.std(ece_sent_before, ddof=1))],
            "sent_after": [float(np.mean(ece_sent_after)), float(np.std(ece_sent_after, ddof=1))],
            "tox_before": [float(np.mean(ece_tox_before)), float(np.std(ece_tox_before, ddof=1))],
            "tox_after": [float(np.mean(ece_tox_after)), float(np.std(ece_tox_after, ddof=1))],
        },
        "nll": {
            "sent_before_mean": float(np.mean(nll_sent_before)),
            "sent_after_mean": float(np.mean(nll_sent_after)),
            "tox_before_mean": float(np.mean(nll_tox_before)),
            "tox_after_mean": float(np.mean(nll_tox_after)),
        },
        "beta_star": beta_star,
        "beta_curve": beta_curve,
        "beta_final": beta_final,
        "v1v2_check": v1v2_info,
    }
    (artifacts_dir / "postproc.json").write_text(
        json.dumps(postproc, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    _write_calibration_report(reports_dir / "calibration.md", postproc)
    print(f"[PHASE 5 COMPLETE] T_bar_sent={t_bar_sent:.4f}, T_bar_tox={t_bar_tox:.4f}, beta*={beta_star}, beta_final={beta_final}")
    return postproc


def _write_calibration_report(report_path: Path, postproc: dict[str, Any]) -> None:
    ece = postproc["ece"]
    nll = postproc["nll"]
    lines = [
        "# Báo cáo Hiệu chuẩn & Điều chỉnh Hậu xử lý (Phase 5)",
        "",
        f"Áp dụng trên 3 seed của cấu hình chiến thắng **`{postproc['winner_config_id']}`**.",
        "",
        "## 1. Temperature Scaling & Chỉ số ECE (M = 15 bins)",
        "",
        f"- $\\bar{{T}}_{{\\text{{sent}}}}$ = **{postproc['t_bar_sent']:.4f}** (từng seed: `{[round(x, 4) for x in postproc['t_sent_per_seed']]}`)",
        f"- $\\bar{{T}}_{{\\text{{tox}}}}$ = **{postproc['t_bar_tox']:.4f}** (từng seed: `{[round(x, 4) for x in postproc['t_tox_per_seed']]}`)",
        "",
        "| Nhiệm vụ | ECE Trước $T$ (Mean ± Std) | ECE Sau $\\bar{T}$ (Mean ± Std) | NLL Trước → Sau |",
        "| :--- | :---: | :---: | :---: |",
        f"| **Sentiment (3 lớp)** | {ece['sent_before'][0]:.4f} ± {ece['sent_before'][1]:.4f} | **{ece['sent_after'][0]:.4f} ± {ece['sent_after'][1]:.4f}** | {nll['sent_before_mean']:.4f} → {nll['sent_after_mean']:.4f} |",
        f"| **Toxicity (4 lớp)** | {ece['tox_before'][0]:.4f} ± {ece['tox_before'][1]:.4f} | **{ece['tox_after'][0]:.4f} ± {ece['tox_after'][1]:.4f}** | {nll['tox_before_mean']:.4f} → {nll['tox_after_mean']:.4f} |",
        "",
        "- Đã kiểm chứng `Macro-F1` trước và sau khi chia cho $\\bar{T}$ giữ nguyên tuyệt đối (chênh lệch `< 1e-12`).",
        "",
        "## 2. Đường cong Khảo sát $\\beta$ trên `valid` (Toxicity 4 lớp)",
        "",
        "| $\\beta$ | Macro-F1 Toxicity (Mean ± Std 3 seed) |",
        "| :---: | :---: |",
    ]
    for r in postproc["beta_curve"]:
        mark = " *(tối ưu $\\beta^*$)*" if r["beta"] == postproc["beta_star"] else ""
        lines.append(f"| `{r['beta']:.2f}` | {r['mean_f1_tox']:.4f} ± {r['std_f1_tox']:.4f}{mark} |")

    v1v2 = postproc["v1v2_check"]
    lines.extend(["", "## 3. Kiểm tra Chuyển giao `V1 / V2`", ""])
    if not v1v2["ran"]:
        lines.append(f"- $\\beta^* = 0.0$ nên không cần bước kiểm tra `V1/V2`; chốt $\\beta_{{\\text{{final}}}} = 0.0$.")
    else:
        lines.extend([
            f"- Nhiệt độ học trên `V1`: $\\bar{{T}}_{{\\text{{tox, V1}}}} = {v1v2['t_bar_v1']:.4f}$, chọn $\\beta_{{V1}} = {v1v2['beta_v1']:.2f}$.",
            f"- Trên `V2` (mô hình `train`-only): $\\beta=0$ đạt `{v1v2['f1_train_v2_b0_mean']:.4f}` → $\\beta_{{V1}}$ đạt `{v1v2['f1_train_v2_bv1_mean']:.4f}` ($\\Delta_{{\\text{{train}}}} = {v1v2['delta_train']:+.4f}$).",
            f"- Trên `V2` (mô hình `train + V1`): $\\beta=0$ đạt `{v1v2['f1_trainV1_v2_b0_mean']:.4f}` → $\\beta_{{V1}}$ đạt `{v1v2['f1_trainV1_v2_bv1_mean']:.4f}` ($\\Delta_{{\\text{{train+V1}}}} = {v1v2['delta_train_v1']:+.4f}$).",
            f"- Kiểm tra quy tắc cứng ($\\Delta_{{\\text{{train+V1}}}} \\ge 0$ và $\\Delta_{{\\text{{train+V1}}}} \\ge 0.5 \\cdot \\Delta_{{\\text{{train}}}}$): **`{v1v2['keep_beta']}`** → **Chốt $\\beta_{{\\text{{final}}}} = {postproc['beta_final']}$**.",
            "",
            f"> **Lưu ý thống kê theo Spec §5.3.6:** Tập `V2` chỉ có **{v1v2['v2_class3_support']} mẫu lớp 3**, nên phép kiểm tra này có độ mạnh thống kê thấp; chỉ đánh giá trên `Macro-F1 Toxicity` tổng 4 lớp và không rút kết luận riêng lẻ cho lớp 3. Việc giữ nguyên $(\\bar{{T}}, \\beta_{{\\text{{final}}}})$ sang Phase 6 là phép xấp xỉ có kiểm soát.",
        ])
    report_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Run Phase 5 calibration and V1/V2 beta transfer check.")
    parser.parse_args()
    run_calibration()


if __name__ == "__main__":
    main()
