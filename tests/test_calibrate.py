"""Unit tests for Phase 5 Temperature Scaling, beta weighting, and V1/V2 transfer decision."""

from __future__ import annotations

import numpy as np

from src_v2.calibrate import compute_tau_k, decide_beta_transfer, evaluate_beta_grid, fit_temperature
from src_v2.metrics import compute_ece


def test_temperature_scaling_improves_overconfident_nll_and_ece() -> None:
    rng = np.random.default_rng(123)
    n = 500
    y = rng.integers(0, 3, size=n)
    # Construct severely overconfident logits (multiplied by 6.0)
    base_logits = rng.normal(0.0, 1.0, size=(n, 3))
    for i in range(n):
        if rng.random() < 0.75:
            base_logits[i, y[i]] += 1.5
    overconf_logits = base_logits * 6.0

    t_opt = fit_temperature(overconf_logits, y)
    assert t_opt > 2.0

    # Also verify float32 logits (guards against NumPy 2.x NEP 50 precision trap)
    t_opt_f32 = fit_temperature(overconf_logits.astype(np.float32), y)
    assert abs(t_opt_f32 - t_opt) < 1e-4

    def sm(z):
        s = z - z.max(1, keepdims=True)
        e = np.exp(s)
        return e / e.sum(1, keepdims=True)

    p_before = sm(overconf_logits)
    p_after = sm(overconf_logits / t_opt)
    # Argmax must remain strictly identical
    assert np.array_equal(p_before.argmax(1), p_after.argmax(1))
    assert compute_ece(p_after, y, n_bins=15) < compute_ece(p_before, y, n_bins=15)


def test_beta_grid_and_v1v2_transfer_decision() -> None:
    tau0 = compute_tau_k([0.88, 0.07, 0.04, 0.01], beta=0.0)
    assert np.allclose(tau0, np.ones(4))
    tau_half = compute_tau_k([0.88, 0.07, 0.04, 0.01], beta=0.5)
    assert tau_half[0] == 1.0
    assert tau_half[3] > tau_half[2] > tau_half[1] > 1.0

    # Verify evaluate_beta_grid returns a valid grid value
    y = np.array([0] * 40 + [1] * 10 + [2] * 5 + [3] * 2)
    z = np.zeros((len(y), 4))
    z[:, 0] = 2.0
    for i in range(len(y)):
        z[i, y[i]] += 1.5
    best_b, rows = evaluate_beta_grid([z, z, z], y, t_tox=1.0, tox_freq=[0.88, 0.07, 0.04, 0.01])
    assert best_b in {0.0, 0.25, 0.5, 0.75, 1.0}
    assert len(rows) == 5

    # Decision boundary checks
    assert decide_beta_transfer(delta_train=0.02, delta_train_v1=0.015) is True
    assert decide_beta_transfer(delta_train=0.02, delta_train_v1=0.005) is False
    assert decide_beta_transfer(delta_train=-0.01, delta_train_v1=-0.002) is False
    assert decide_beta_transfer(delta_train=-0.01, delta_train_v1=0.001) is True
