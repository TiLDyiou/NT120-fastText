"""Phase 7: One-time blind evaluation on challenge.csv with Cold/Warm start latency and full metrics."""

from __future__ import annotations

import argparse
import ctypes
import gc
import json
import platform
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch

from src_v2.config import (
    DEFAULT_FASTTEXT_BIN,
    FINAL_SEEDS,
    RAW_DATA_DIR,
    REPORTS_DIR,
    RUNS_DIR,
    SENT_NAMES,
    TOX_NAMES,
    ensure_project_dirs,
)
from src_v2.data_prep import clean_text, tokenize
from src_v2.final_train import load_final_seed_model
from src_v2.metrics import (
    bootstrap_ci_f1,
    build_detailed_eval_report,
)
from src_v2.predict import DynamicPredictor, FastTextBinOOVProvider, OOVVectorProvider, apply_postproc


def get_hardware_info() -> dict[str, Any]:
    """Collect CPU, RAM, GPU, and VRAM specs for reporting."""
    ram_gb = None
    if sys.platform == "win32":
        class MEMSTAT(ctypes.Structure):
            _fields_ = [
                ("dwLength", ctypes.c_ulong),
                ("dwMemoryLoad", ctypes.c_ulong),
                ("ullTotalPhys", ctypes.c_ulonglong),
                ("ullAvailPhys", ctypes.c_ulonglong),
                ("ullTotalPageFile", ctypes.c_ulonglong),
                ("ullAvailPageFile", ctypes.c_ulonglong),
                ("ullTotalVirtual", ctypes.c_ulonglong),
                ("ullAvailVirtual", ctypes.c_ulonglong),
                ("sullAvailExtendedVirtual", ctypes.c_ulonglong),
            ]
        m = MEMSTAT()
        m.dwLength = ctypes.sizeof(m)
        ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(m))
        ram_gb = round(m.ullTotalPhys / (1024 ** 3), 2)

    gpu_name = "CPU only"
    vram_gb = 0.0
    if torch.cuda.is_available():
        gpu_name = torch.cuda.get_device_name(0)
        vram_gb = round(torch.cuda.get_device_properties(0).total_memory / (1024 ** 3), 2)

    return {
        "os": platform.platform(),
        "cpu": platform.processor() or "unknown",
        "ram_total_gb": ram_gb,
        "gpu": gpu_name,
        "vram_total_gb": vram_gb,
        "torch_version": torch.__version__,
    }


def _sync_device(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def _run_end_to_end_timed(
    raw_texts: list[str],
    predictor: DynamicPredictor,
    t_sent: float,
    t_tox: float,
    tau_k: np.ndarray,
    batch_size: int,
    device: torch.device,
) -> tuple[dict[str, np.ndarray], float]:
    """Run full pipeline (clean_text -> tokenize -> lookup/expand -> forward -> T & beta) and measure elapsed sec."""
    _sync_device(device)
    t0 = time.perf_counter()

    token_lists = [tokenize(clean_text(t)) for t in raw_texts]
    logits_sent, logits_tox = predictor.predict_logits_from_tokens(token_lists, batch_size=batch_size)
    post = apply_postproc(logits_sent, logits_tox, t_sent=t_sent, t_tox=t_tox, tau_k=tau_k)

    _sync_device(device)
    elapsed = time.perf_counter() - t0
    return post, float(elapsed)


def evaluate_seed_on_challenge(
    seed_dir: Path,
    raw_texts: list[str],
    y_sent_idx: np.ndarray,
    y_tox_idx: np.ndarray,
    oov_provider: OOVVectorProvider,
    device: torch.device,
    batch_size: int = 256,
) -> dict[str, Any]:
    """Evaluate one saved final seed model with both Cold start and Warm start timing."""
    models, cfg_data, token_to_id = load_final_seed_model(seed_dir, device)
    max_len_full = int(cfg_data["max_len_full"])
    t_sent = float(cfg_data["t_bar_sent"])
    t_tox = float(cfg_data["t_bar_tox"])
    tau_k = np.asarray(cfg_data["tau_k"], dtype=np.float64)

    predictor = DynamicPredictor(
        models=models,
        token_to_id=token_to_id,
        max_len=max_len_full,
        oov_provider=oov_provider,
        device=device,
        clone_models=False,
    )

    # Cold start: unseen challenge tokens fetched from .bin and cached in embedding
    post_cold, cold_sec = _run_end_to_end_timed(
        raw_texts, predictor, t_sent, t_tox, tau_k, batch_size, device
    )
    cold_oov_hits = predictor.oov_hits

    # Warm start: same predictor instance now has all challenge OOVs cached in RAM/VRAM
    post_warm, warm_sec = _run_end_to_end_timed(
        raw_texts, predictor, t_sent, t_tox, tau_k, batch_size, device
    )
    warm_new_hits = predictor.oov_hits - cold_oov_hits
    assert warm_new_hits == 0, f"Expected 0 new OOV lookups during warm start, got {warm_new_hits}"
    assert np.array_equal(post_cold["pred_sent_idx"], post_warm["pred_sent_idx"])
    assert np.array_equal(post_cold["pred_tox_idx"], post_warm["pred_tox_idx"])

    n = len(raw_texts)
    detailed = build_detailed_eval_report(
        y_sent_idx,
        y_tox_idx,
        post_cold["probs_sent"],
        post_cold["probs_tox"],
        y_tox_pred=post_cold["pred_tox_idx"],
    )
    ci_lo, ci_hi = bootstrap_ci_f1(
        y_tox_idx, post_cold["pred_tox_idx"], labels=[0, 1, 2, 3], n_boot=1000, seed=42
    )

    preds_df = pd.DataFrame(
        {
            "text_raw": raw_texts,
            "text_clean": [clean_text(t) for t in raw_texts],
            "y_sent": y_sent_idx - 1,
            "pred_sent": post_cold["pred_sent_idx"] - 1,
            "p_neg": post_cold["probs_sent"][:, 0],
            "p_neu": post_cold["probs_sent"][:, 1],
            "p_pos": post_cold["probs_sent"][:, 2],
            "y_tox": y_tox_idx,
            "pred_tox": post_cold["pred_tox_idx"],
            "p_t0": post_cold["probs_tox"][:, 0],
            "p_t1": post_cold["probs_tox"][:, 1],
            "p_t2": post_cold["probs_tox"][:, 2],
            "p_t3": post_cold["probs_tox"][:, 3],
        }
    )

    return {
        "seed": int(cfg_data["seed"]),
        "n": n,
        "oov_hits": int(cold_oov_hits),
        "cold_elapsed_sec": cold_sec,
        "cold_throughput": n / cold_sec,
        "cold_latency_ms": (cold_sec / n) * 1000.0,
        "warm_elapsed_sec": warm_sec,
        "warm_throughput": n / warm_sec,
        "warm_latency_ms": (warm_sec / n) * 1000.0,
        "tox_bootstrap_ci95": [ci_lo, ci_hi],
        "metrics": detailed,
        "preds_df": preds_df,
    }


def _mean_std(vals: list[float]) -> tuple[float, float]:
    return float(np.mean(vals)), float(np.std(vals, ddof=1)) if len(vals) > 1 else 0.0


def run_challenge_evaluation(
    challenge_csv: Path = RAW_DATA_DIR / "challenge" / "challenge.csv",
    bin_path: Path = DEFAULT_FASTTEXT_BIN,
    runs_dir: Path = RUNS_DIR,
    reports_dir: Path = REPORTS_DIR,
    batch_size: int = 256,
) -> dict[str, Any]:
    """Run one-time blind evaluation on challenge.csv across seeds 1, 2, 3."""
    ensure_project_dirs()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    hw = get_hardware_info()

    df_raw = pd.read_csv(challenge_csv, encoding="utf-8-sig")
    df_raw.columns = [c.strip().replace("\ufeff", "") for c in df_raw.columns]
    df_raw["text_clean"] = df_raw["Comment"].fillna("").astype(str).map(clean_text)
    df_raw = df_raw[df_raw["text_clean"] != ""].reset_index(drop=True)

    raw_texts = df_raw["Comment"].fillna("").astype(str).tolist()
    y_sent_idx = pd.to_numeric(df_raw["Sentiment"], errors="raise").astype(int).to_numpy() + 1
    y_tox_idx = pd.to_numeric(df_raw["Toxicity"], errors="raise").astype(int).to_numpy()

    import fasttext

    print(f"Loading cc.vi.300.bin for Phase 7 Challenge evaluation: {bin_path}")
    ft_model = fasttext.load_model(str(bin_path))
    oov_provider = FastTextBinOOVProvider(ft_model)

    seed_results: list[dict[str, Any]] = []
    try:
        for seed in FINAL_SEEDS:
            seed_dir = runs_dir / "final_models" / f"seed_{seed}"
            print(f"--- Evaluating final seed_{seed} on Challenge (n={len(raw_texts):,}) ---")
            res = evaluate_seed_on_challenge(
                seed_dir, raw_texts, y_sent_idx, y_tox_idx, oov_provider, device, batch_size=batch_size
            )
            preds_path = reports_dir / f"preds_challenge_seed{seed}.csv"
            res["preds_df"].to_csv(preds_path, index=False, encoding="utf-8-sig")
            seed_results.append({k: v for k, v in res.items() if k != "preds_df"})
            m = res["metrics"]
            print(
                f"  seed={seed} | Sent F1={m['macro_f1_sent']:.4f} | Tox F1={m['macro_f1_tox']:.4f} "
                f"| Comb={m['combined_macro_f1']:.4f} | QWK={m['qwk_tox']:.4f} "
                f"| Cold={res['cold_throughput']:.0f}/s | Warm={res['warm_throughput']:.0f}/s | OOV={res['oov_hits']}"
            )
    finally:
        del ft_model
        gc.collect()

    payload = {
        "hardware": hw,
        "batch_size": batch_size,
        "n_challenge": len(raw_texts),
        "seeds": seed_results,
    }
    (reports_dir / "challenge_metrics.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    _write_challenge_report(reports_dir / "challenge_report.md", payload)
    print(f"\n[PHASE 7 COMPLETE] Challenge report saved to {reports_dir / 'challenge_report.md'}")
    return payload


def _write_challenge_report(report_path: Path, payload: dict[str, Any]) -> None:
    hw = payload["hardware"]
    seeds = payload["seeds"]
    n = payload["n_challenge"]

    sent_f1s = [r["metrics"]["macro_f1_sent"] for r in seeds]
    tox_f1s = [r["metrics"]["macro_f1_tox"] for r in seeds]
    comb_f1s = [r["metrics"]["combined_macro_f1"] for r in seeds]
    qwks = [r["metrics"]["qwk_tox"] for r in seeds]
    ece_s = [r["metrics"]["ece_sent"] for r in seeds]
    ece_t = [r["metrics"]["ece_tox"] for r in seeds]
    cold_thrs = [r["cold_throughput"] for r in seeds]
    cold_lats = [r["cold_latency_ms"] for r in seeds]
    warm_thrs = [r["warm_throughput"] for r in seeds]
    warm_lats = [r["warm_latency_ms"] for r in seeds]

    m_s, sd_s = _mean_std(sent_f1s)
    m_t, sd_t = _mean_std(tox_f1s)
    m_c, sd_c = _mean_std(comb_f1s)
    m_q, sd_q = _mean_std(qwks)
    m_es, sd_es = _mean_std(ece_s)
    m_et, sd_et = _mean_std(ece_t)

    lines = [
        f"# BÁO CÁO ĐÁNH GIÁ TẬP MÙ CHALLENGE (PHASE 7) — $N = {n:,}$ CÂU",
        "",
        "> **Lưu ý:** Chạy **đúng 1 lần duy nhất** trên tập mù `challenge.csv` sau khi đóng băng toàn bộ mô hình và hậu xử lý ở Phase 6. Tuyệt đối không chỉnh sửa và chấm lại.",
        "",
        "## 1. Bảng Điểm Tổng hợp 3 Seed (Mean ± Std)",
        "",
        "| Seed | Sentiment Macro-F1 | **Toxicity Macro-F1 (4 lớp)** | Combined Macro-F1 | QWK (Toxicity) | ECE Sent | ECE Tox | CI 95% Tox F1 |",
        "| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |",
    ]
    for r in seeds:
        m = r["metrics"]
        ci = r["tox_bootstrap_ci95"]
        lines.append(
            f"| `seed_{r['seed']}` | {m['macro_f1_sent']:.4f} | **{m['macro_f1_tox']:.4f}** | "
            f"{m['combined_macro_f1']:.4f} | {m['qwk_tox']:.4f} | {m['ece_sent']:.4f} | {m['ece_tox']:.4f} | "
            f"[{ci[0]:.4f}, {ci[1]:.4f}] |"
        )
    lines.extend([
        f"| **Mean ± Std** | **{m_s:.4f} ± {sd_s:.4f}** | **{m_t:.4f} ± {sd_t:.4f}** | "
        f"**{m_c:.4f} ± {sd_c:.4f}** | **{m_q:.4f} ± {sd_q:.4f}** | **{m_es:.4f} ± {sd_es:.4f}** | **{m_et:.4f} ± {sd_et:.4f}** | — |",
        "",
        "## 2. Chi tiết từng Lớp (Mean ± Std qua 3 Seed)",
        "",
        "### 2.1. Nhiệm vụ Sentiment (3 lớp: `-1, 0, 1`)",
        "| Lớp | F1-Score (Mean ± Std) | Support |",
        "| :--- | :---: | :---: |",
    ])
    for name in SENT_NAMES:
        vals = [r["metrics"]["report_sent"][name]["f1-score"] for r in seeds]
        sup = int(seeds[0]["metrics"]["report_sent"][name]["support"])
        mu, sd = _mean_std(vals)
        lines.append(f"| `{name}` | **{mu:.4f} ± {sd:.4f}** | {sup:,} |")

    lines.extend([
        "",
        "### 2.2. Nhiệm vụ Toxicity (4 lớp: `0, 1, 2, 3`)",
        "| Lớp | F1-Score (Mean ± Std) | Support |",
        "| :--- | :---: | :---: |",
    ])
    for name in TOX_NAMES:
        vals = [r["metrics"]["report_tox"][name]["f1-score"] for r in seeds]
        sup = int(seeds[0]["metrics"]["report_tox"][name]["support"])
        mu, sd = _mean_std(vals)
        lines.append(f"| `{name}` | **{mu:.4f} ± {sd:.4f}** | {sup:,} |")

    lines.extend([
        "",
        f"> **Nguyên tắc diễn giải (§7.2.2):** Do lớp 3 (`T3_HateThreat`, support = {int(seeds[0]['metrics']['report_tox']['T3_HateThreat']['support'])}) rất hiếm, F1 riêng lớp 3 dao động mạnh chỉ với vài mẫu đoán sai. Mọi kết luận đánh giá mô hình chỉ rút ra từ **`Macro-F1` tổng 4 lớp** và **`Confusion matrix` $4 \\times 4$**.",
        "",
        "### 2.3. Ma trận Nhầm lẫn (Seed 1 đại diện)",
        "- **Sentiment `3×3` (hàng = thực tế `[-1, 0, 1]`, cột = dự đoán):**",
        "```text",
        str(np.array(seeds[0]["metrics"]["cm_sent"])),
        "```",
        "- **Toxicity `4×4` (hàng = thực tế `[0, 1, 2, 3]`, cột = dự đoán):**",
        "```text",
        str(np.array(seeds[0]["metrics"]["cm_tox"])),
        "```",
        "",
        "## 3. Tỷ lệ Báo nhầm Mức độ Độc hại (FPR) trên tập Non-toxic (`T0`) theo Nhóm Cảm xúc",
        "",
        "> **Lưu ý:** Đây là định nghĩa FPR đa mức riêng biệt ($0 \\to 1$, $0 \\to 2$, $0 \\to 3$), khác bản chất với định nghĩa nhị phân cũ nên không so sánh trực tiếp số học với báo cáo nhị phân trước đây.",
        "",
        "| Nhóm Cảm xúc | Support `T0` | $\\text{FPR}_{0 \\to 1}$ (Mean ± Std) | $\\text{FPR}_{0 \\to 2}$ (Mean ± Std) | $\\text{FPR}_{0 \\to 3}$ (Mean ± Std) |",
        "| :--- | :---: | :---: | :---: | :---: |",
    ])
    for sname in SENT_NAMES:
        sup = seeds[0]["metrics"]["fpr_by_sentiment"][sname]["support_t0"]
        f1s = [r["metrics"]["fpr_by_sentiment"][sname]["fpr_0_to_1"] * 100.0 for r in seeds]
        f2s = [r["metrics"]["fpr_by_sentiment"][sname]["fpr_0_to_2"] * 100.0 for r in seeds]
        f3s = [r["metrics"]["fpr_by_sentiment"][sname]["fpr_0_to_3"] * 100.0 for r in seeds]
        m1, sd1 = _mean_std(f1s)
        m2, sd2 = _mean_std(f2s)
        m3, sd3 = _mean_std(f3s)
        lines.append(
            f"| `{sname}` | {sup:,} | {m1:.2f}% ± {sd1:.2f}% | {m2:.2f}% ± {sd2:.2f}% | {m3:.2f}% ± {sd3:.2f}% |"
        )

    mc_t, sdc_t = _mean_std(cold_thrs)
    mc_l, sdc_l = _mean_std(cold_lats)
    mw_t, sdw_t = _mean_std(warm_thrs)
    mw_l, sdw_l = _mean_std(warm_lats)

    lines.extend([
        "",
        "## 4. Thông lượng & Độ trễ Suy luận (Cold Start vs. Warm Start)",
        "",
        f"- **Phần cứng:** GPU `{hw['gpu']}` (VRAM `{hw['vram_total_gb']} GB`), CPU `{hw['cpu']}`, RAM `{hw['ram_total_gb']} GB`, `batch_size = {payload['batch_size']}`, số token OOV mới trong `challenge` = `{seeds[0]['oov_hits']}`.",
        "",
        "| Kịch bản | Thời gian TB (s) | Thông lượng (câu/s) | Độ trễ (ms/câu) |",
        "| :--- | :---: | :---: | :---: |",
        f"| **Cold start** (lần đầu tra `.bin` cho token mới) | {np.mean([r['cold_elapsed_sec'] for r in seeds]):.3f} | **{mc_t:.0f} ± {sdc_t:.0f}** | **{mc_l:.3f} ± {sdc_l:.3f}** |",
        f"| **Warm start** (đã có cache bộ nhớ) | {np.mean([r['warm_elapsed_sec'] for r in seeds]):.3f} | **{mw_t:.0f} ± {sdw_t:.0f}** | **{mw_l:.3f} ± {sdw_l:.3f}** |",
        "",
        "## 5. Đối chiếu với Bản Cũ (Keras `Bi-GRU-LSTM-CNN` 4 lớp trong `NT120-fastText/reports/blind_benchmark_report.md`)",
        "",
        "| Chỉ số trên `challenge` (3,237 câu) | Bản cũ v1 (Keras + `.vec` + `pyvi`) | **Bản mới v2 (PyTorch + `.bin` + Song song)** |",
        "| :--- | :---: | :---: |",
        f"| **Sentiment Macro-F1 (3 lớp)** | 0.6423 ± 0.0079 | **{m_s:.4f} ± {sd_s:.4f}** |",
        f"| **Toxicity Macro-F1 (4 lớp)** | 0.5731 ± 0.0531 | **{m_t:.4f} ± {sd_t:.4f}** |",
        f"| **Combined Macro-F1** | 0.6077 ± 0.0250 | **{m_c:.4f} ± {sd_c:.4f}** |",
        f"| **ECE Sentiment (Seed 1)** | 0.2440 | **{seeds[0]['metrics']['ece_sent']:.4f}** |",
        f"| **ECE Toxicity (Seed 1)** | 0.0776 | **{seeds[0]['metrics']['ece_tox']:.4f}** |",
        "",
    ])
    report_path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Run Phase 7 single blind evaluation on challenge.")
    parser.add_argument("--challenge", type=Path, default=RAW_DATA_DIR / "challenge" / "challenge.csv")
    parser.add_argument("--bin", type=Path, default=DEFAULT_FASTTEXT_BIN)
    parser.add_argument("--batch-size", type=int, default=256)
    args = parser.parse_args()
    run_challenge_evaluation(challenge_csv=args.challenge, bin_path=args.bin, batch_size=args.batch_size)


if __name__ == "__main__":
    main()
