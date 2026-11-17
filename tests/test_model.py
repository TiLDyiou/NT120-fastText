"""Mandatory Phase 3 unit tests & sanity checks for ParallelNet, masking, freeze, and inference."""

from __future__ import annotations

import numpy as np
import pandas as pd
import torch

from src_v2.config import ARTIFACTS_DIR, EMBEDDING_DIM, PROCESSED_DATA_DIR, ExperimentConfig
from src_v2.model import MultiTaskLoss, ParallelNet, compute_tox_class_weights
from src_v2.predict import DynamicPredictor, PreCachedOOVProvider
from src_v2.train import _train_single_model_instance, load_artifact_bundle, train_and_evaluate
from src_v2.utils.seed import set_seed


def _dummy_emb(vocab_size: int = 20, seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    mat = rng.normal(0.0, 0.2, size=(vocab_size, EMBEDDING_DIM)).astype(np.float32)
    mat[0] = 0.0
    return mat


def test_output_shapes_and_no_softmax() -> None:
    mat = _dummy_emb(30)
    for dense in [False, True]:
        model = ParallelNet(mat, hidden_units=32, dropout=0.0, use_dense300=dense)
        model.eval()
        ids = torch.tensor([[1, 2, 3, 0, 0], [4, 5, 0, 0, 0]], dtype=torch.long)
        lens = torch.tensor([3, 2], dtype=torch.long)
        out = model(ids, lens)
        assert out["sentiment"].shape == (2, 3)
        assert out["toxicity"].shape == (2, 4)
        # Raw logits are not constrained to sum to 1
        assert not torch.allclose(out["sentiment"].sum(dim=-1), torch.ones(2))


def test_padding_invariance_and_backward_gru_boundary() -> None:
    """Padding a sequence to longer L must yield identical logits (checks GRU pack + Conv1D masks)."""
    set_seed(42)
    mat = _dummy_emb(25)
    for dense in [False, True]:
        model = ParallelNet(mat, hidden_units=16, dropout=0.0, use_dense300=dense)
        model.eval()

        short_ids = torch.tensor([[2, 5, 7, 9]], dtype=torch.long)
        short_len = torch.tensor([4], dtype=torch.long)

        long_ids = torch.tensor([[2, 5, 7, 9, 0, 0, 0, 0, 0, 0]], dtype=torch.long)
        long_len = torch.tensor([4], dtype=torch.long)

        # Also test non-zero token IDs past actual_len: mask MUST rely on actual_len, not ids != 0
        dirty_pad_ids = torch.tensor([[2, 5, 7, 9, 11, 12, 13, 14]], dtype=torch.long)
        dirty_len = torch.tensor([4], dtype=torch.long)

        with torch.no_grad():
            o_short = model(short_ids, short_len)
            o_long = model(long_ids, long_len)
            o_dirty = model(dirty_pad_ids, dirty_len)

        assert torch.allclose(o_short["sentiment"], o_long["sentiment"], atol=1e-5)
        assert torch.allclose(o_short["toxicity"], o_long["toxicity"], atol=1e-5)
        assert torch.allclose(o_short["sentiment"], o_dirty["sentiment"], atol=1e-5)
        assert torch.allclose(o_short["toxicity"], o_dirty["toxicity"], atol=1e-5)


def test_freeze_then_open_gradients_and_pad_zero() -> None:
    """Epoch <= F has zero embedding grad (and weight unchanged); Epoch > F updates only non-pad."""
    set_seed(7)
    mat = _dummy_emb(15)
    model = ParallelNet(mat, hidden_units=16, dropout=0.0)
    cfg = ExperimentConfig(
        epochs=3, freeze_epochs=1, emb_mode="freeze_then_open", lr=1e-2, emb_lr_factor=0.2, weight_decay=1e-2
    )
    tox_w = compute_tox_class_weights([10, 4, 2, 1], alpha=0.5, clip_max=10.0)
    loss_mod = MultiTaskLoss("weighted_ce", tox_w)

    ids = torch.tensor([[1, 2, 3, 0], [4, 5, 0, 0]], dtype=torch.long)
    lens = torch.tensor([3, 2], dtype=torch.long)
    ys = torch.tensor([0, 2], dtype=torch.long)
    yt = torch.tensor([0, 1], dtype=torch.long)

    w_init = model.embedding.weight.detach().clone()
    # Run 1-epoch frozen check first on a clone to verify zero weight drift during frozen phase
    cfg_frozen_check = ExperimentConfig(
        epochs=2, freeze_epochs=1, emb_mode="frozen", lr=1e-2, weight_decay=1e-2
    )
    model_frozen = ParallelNet(mat, hidden_units=16, dropout=0.0)
    _train_single_model_instance(
        model_frozen, MultiTaskLoss("weighted_ce", tox_w), ids, lens, ys, yt,
        cfg_frozen_check, torch.device("cpu"), seed=7
    )
    assert torch.equal(model_frozen.embedding.weight, torch.as_tensor(mat))

    hist = _train_single_model_instance(
        model, loss_mod, ids, lens, ys, yt, cfg, torch.device("cpu"), seed=7
    )

    assert hist[0]["emb_trainable"] == 0.0
    assert hist[1]["emb_trainable"] == 1.0
    assert hist[2]["emb_trainable"] == 1.0
    assert abs(hist[0]["lr_embedding"] / hist[0]["lr_backbone"] - 0.2) < 1e-6
    assert abs(hist[1]["lr_embedding"] / hist[1]["lr_backbone"] - 0.2) < 1e-6

    # Check row 0 (<pad>) stayed strictly zero and embedding updated after F
    assert torch.all(model.embedding.weight[0] == 0.0)
    assert not torch.allclose(model.embedding.weight[1], w_init[1])


def test_dynamic_vocab_expansion_order_independence() -> None:
    """Predicting sentence A before or after sentence B (with OOVs) produces identical output."""
    set_seed(9)
    mat = _dummy_emb(5)  # vocab 0..4
    vocab = {"<pad>": 0, "xin": 1, "chào": 2, "tệ": 3, "quá": 4}
    oov_vecs = np.random.default_rng(11).normal(0.0, 0.2, size=(2, EMBEDDING_DIM)).astype(np.float32)
    oov_map = {"lạ_1": 0, "lạ_2": 1}

    model = ParallelNet(mat, hidden_units=16, dropout=0.0)
    pred1 = DynamicPredictor([model], vocab, max_len=6, oov_provider=PreCachedOOVProvider(oov_map, oov_vecs))
    pred2 = DynamicPredictor([model], vocab, max_len=6, oov_provider=PreCachedOOVProvider(oov_map, oov_vecs))

    sent_a = ["xin", "lạ_1", "quá"]
    sent_b = ["chào", "lạ_2", "tệ"]

    ls1, lt1 = pred1.predict_logits_from_tokens([sent_a, sent_b], batch_size=1)
    ls2, lt2 = pred2.predict_logits_from_tokens([sent_b, sent_a], batch_size=1)

    assert np.allclose(ls1[0], ls2[1], atol=1e-6)
    assert np.allclose(lt1[0], lt2[1], atol=1e-6)
    # Original model embedding size must remain untouched (clone_models=True)
    assert model.embedding.weight.shape[0] == 5


def test_overfit_200_samples_real_data() -> None:
    """DoD sanity test: can overfit 200 real training samples cleanly (all classes represented)."""
    full_train = pd.read_parquet(PROCESSED_DATA_DIR / "train.parquet")
    # Pick 50 samples per Toxicity class (200 total) so all 4 Toxicity & 3 Sentiment classes exist
    parts = [full_train[full_train["y_tox_idx"] == k].iloc[:50] for k in range(4)]
    train_df = pd.concat(parts, ignore_index=True)
    bundle = load_artifact_bundle(ARTIFACTS_DIR / "train")
    cfg = ExperimentConfig(
        epochs=30,
        freeze_epochs=2,
        lr=4e-3,
        dropout=0.0,
        hidden_units=64,
        weight_decay=0.0,
        alpha=0.0,
        emb_lr_factor=0.3,
    )
    res = train_and_evaluate(cfg, train_df, bundle, eval_df=train_df)
    hist = res["history"]
    assert hist[-1]["train_loss"] < 0.05, f"Expected loss < 0.05, got {hist[-1]['train_loss']}"
    assert res["eval"]["combined_macro_f1"] > 0.95
