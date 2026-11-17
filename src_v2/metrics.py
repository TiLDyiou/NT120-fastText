"""Evaluation metrics, ECE, QWK, paired bootstrap CI, and FPR by Sentiment."""

from __future__ import annotations

from typing import Any

import numpy as np
from sklearn.metrics import (
    classification_report,
    cohen_kappa_score,
    confusion_matrix,
    f1_score,
)

from src_v2.config import SENT_NAMES, TOX_NAMES


def compute_macro_f1(y_true: np.ndarray, y_pred: np.ndarray, labels: list[int]) -> float:
    return float(f1_score(y_true, y_pred, labels=labels, average="macro", zero_division=0))


def compute_combined_f1(
    y_sent_true: np.ndarray,
    y_sent_pred: np.ndarray,
    y_tox_true: np.ndarray,
    y_tox_pred: np.ndarray,
) -> tuple[float, float, float]:
    f1_sent = compute_macro_f1(y_sent_true, y_sent_pred, labels=[0, 1, 2])
    f1_tox = compute_macro_f1(y_tox_true, y_tox_pred, labels=[0, 1, 2, 3])
    return f1_sent, f1_tox, 0.5 * (f1_sent + f1_tox)


def compute_ece(probs: np.ndarray, y_true: np.ndarray, n_bins: int = 15) -> float:
    """Expected Calibration Error with M equal-width bins on [0, 1]."""
    conf = probs.max(axis=1)
    correct = (probs.argmax(axis=1) == y_true).astype(np.float64)
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    ece = 0.0
    n = len(y_true)
    if n == 0:
        return 0.0
    for idx in range(n_bins):
        lo, hi = edges[idx], edges[idx + 1]
        if idx == 0:
            mask = (conf >= lo) & (conf <= hi)
        else:
            mask = (conf > lo) & (conf <= hi)
        count = int(mask.sum())
        if count > 0:
            acc_bin = float(correct[mask].mean())
            conf_bin = float(conf[mask].mean())
            ece += (count / n) * abs(acc_bin - conf_bin)
    return float(ece)


def compute_qwk(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Quadratic Weighted Kappa over Toxicity levels 0..3."""
    return float(cohen_kappa_score(y_true, y_pred, labels=[0, 1, 2, 3], weights="quadratic"))


def bootstrap_ci_f1(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    labels: list[int],
    n_boot: int = 1000,
    seed: int = 42,
) -> tuple[float, float]:
    """95% percentile bootstrap confidence interval for Macro-F1."""
    rng = np.random.default_rng(seed)
    n = len(y_true)
    scores = np.empty(n_boot, dtype=np.float64)
    for b in range(n_boot):
        idx = rng.integers(0, n, size=n)
        scores[b] = f1_score(
            y_true[idx], y_pred[idx], labels=labels, average="macro", zero_division=0
        )
    lo, hi = np.percentile(scores, [2.5, 97.5])
    return float(lo), float(hi)


def paired_bootstrap_f1_diff(
    y_sent: np.ndarray,
    y_tox: np.ndarray,
    preds_a: list[tuple[np.ndarray, np.ndarray]],
    preds_b: list[tuple[np.ndarray, np.ndarray]],
    n_boot: int = 1000,
    seed: int = 42,
) -> dict[str, Any]:
    """Paired bootstrap of mean-3-seed Combined Macro-F1 difference (A - B)."""
    rng = np.random.default_rng(seed)
    n = len(y_sent)
    diffs = np.empty(n_boot, dtype=np.float64)

    for b in range(n_boot):
        idx = rng.integers(0, n, size=n)
        ys_b = y_sent[idx]
        yt_b = y_tox[idx]

        scores_a = []
        for ps, pt in preds_a:
            _, _, comb = compute_combined_f1(ys_b, ps[idx], yt_b, pt[idx])
            scores_a.append(comb)

        scores_b = []
        for ps, pt in preds_b:
            _, _, comb = compute_combined_f1(ys_b, ps[idx], yt_b, pt[idx])
            scores_b.append(comb)

        diffs[b] = float(np.mean(scores_a) - np.mean(scores_b))

    ci_low, ci_high = np.percentile(diffs, [2.5, 97.5])
    contains_zero = bool(ci_low <= 0.0 <= ci_high)
    return {
        "mean_diff": float(diffs.mean()),
        "ci_low": float(ci_low),
        "ci_high": float(ci_high),
        "contains_zero": contains_zero,
    }


def compute_fpr_by_sentiment(
    y_sent_true: np.ndarray,
    y_tox_true: np.ndarray,
    y_tox_pred: np.ndarray,
) -> dict[str, dict[str, Any]]:
    """Multi-level FPR_0->1, FPR_0->2, FPR_0->3 on true T0 samples grouped by true Sentiment."""
    out: dict[str, dict[str, Any]] = {}
    for s_idx, s_name in enumerate(SENT_NAMES):
        mask = (y_tox_true == 0) & (y_sent_true == s_idx)
        n_t0 = int(mask.sum())
        if n_t0 == 0:
            out[s_name] = {"support_t0": 0, "fpr_0_to_1": 0.0, "fpr_0_to_2": 0.0, "fpr_0_to_3": 0.0}
        else:
            preds_sub = y_tox_pred[mask]
            out[s_name] = {
                "support_t0": n_t0,
                "fpr_0_to_1": float((preds_sub == 1).mean()),
                "fpr_0_to_2": float((preds_sub == 2).mean()),
                "fpr_0_to_3": float((preds_sub == 3).mean()),
            }
    return out


def build_detailed_eval_report(
    y_sent_true: np.ndarray,
    y_tox_true: np.ndarray,
    probs_sent: np.ndarray,
    probs_tox: np.ndarray,
    y_tox_pred: np.ndarray | None = None,
) -> dict[str, Any]:
    """Build full classification reports, confusion matrices, ECE, QWK, and FPR."""
    pred_sent = probs_sent.argmax(axis=1)
    pred_tox = y_tox_pred if y_tox_pred is not None else probs_tox.argmax(axis=1)

    f1_sent, f1_tox, comb_f1 = compute_combined_f1(y_sent_true, pred_sent, y_tox_true, pred_tox)
    return {
        "macro_f1_sent": f1_sent,
        "macro_f1_tox": f1_tox,
        "combined_macro_f1": comb_f1,
        "ece_sent": compute_ece(probs_sent, y_sent_true, n_bins=15),
        "ece_tox": compute_ece(probs_tox, y_tox_true, n_bins=15),
        "qwk_tox": compute_qwk(y_tox_true, pred_tox),
        "report_sent": classification_report(
            y_sent_true, pred_sent, labels=[0, 1, 2], target_names=SENT_NAMES, output_dict=True, zero_division=0
        ),
        "report_tox": classification_report(
            y_tox_true, pred_tox, labels=[0, 1, 2, 3], target_names=TOX_NAMES, output_dict=True, zero_division=0
        ),
        "cm_sent": confusion_matrix(y_sent_true, pred_sent, labels=[0, 1, 2]).tolist(),
        "cm_tox": confusion_matrix(y_tox_true, pred_tox, labels=[0, 1, 2, 3]).tolist(),
        "fpr_by_sentiment": compute_fpr_by_sentiment(y_sent_true, y_tox_true, pred_tox),
    }
