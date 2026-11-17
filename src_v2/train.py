"""Phase 3: PyTorch Dataset, DataLoader, freeze_then_open trainer, and evaluation."""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any, Optional

import numpy as np
import pandas as pd
import torch
import torch.nn as nn

from src_v2.build_embeddings import PAD_ID
from src_v2.config import ExperimentConfig
from src_v2.metrics import compute_combined_f1
from src_v2.model import (
    MultiTaskLoss,
    ParallelNet,
    compute_tox_class_weights,
    count_model_parameters,
)
from src_v2.predict import DynamicPredictor, PreCachedOOVProvider
from src_v2.utils.seed import set_seed


def encode_known_tokens(
    token_lists: list[list[str]],
    token_to_id: dict[str, int],
    max_len: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Encode training sequences using known vocabulary only (truncate tail, pad 0)."""
    n = len(token_lists)
    input_ids = np.full((n, max_len), PAD_ID, dtype=np.int64)
    actual_lens = np.ones(n, dtype=np.int64)
    for row_idx, tokens in enumerate(token_lists):
        trunc = tokens[:max_len]
        actual_lens[row_idx] = max(1, len(trunc))
        for col_idx, tok in enumerate(trunc):
            input_ids[row_idx, col_idx] = token_to_id[tok]
    return torch.from_numpy(input_ids), torch.from_numpy(actual_lens)


def load_artifact_bundle(artifact_dir: Path, oov_prefix: Optional[str] = None) -> dict[str, Any]:
    """Load embedding matrix, vocab, meta, and optional OOV pre-cache."""
    token_to_id = json.loads((artifact_dir / "token_to_id.json").read_text(encoding="utf-8"))
    meta = json.loads((artifact_dir / "meta.json").read_text(encoding="utf-8"))
    embedding_matrix = np.load(artifact_dir / "embedding_matrix.npy")

    oov_provider = None
    if oov_prefix is not None:
        oov_map_path = artifact_dir / f"{oov_prefix}_oov_map.json"
        oov_vec_path = artifact_dir / f"{oov_prefix}_oov_vectors.npy"
        if oov_map_path.exists() and oov_vec_path.exists():
            oov_map = json.loads(oov_map_path.read_text(encoding="utf-8"))
            oov_vecs = np.load(oov_vec_path)
            oov_provider = PreCachedOOVProvider(oov_map, oov_vecs)

    return {
        "token_to_id": token_to_id,
        "meta": meta,
        "embedding_matrix": embedding_matrix,
        "oov_provider": oov_provider,
    }


def _build_optimizer(
    model: ParallelNet,
    loss_module: MultiTaskLoss,
    cfg: ExperimentConfig,
) -> torch.optim.AdamW:
    """Create AdamW with separate backbone and embedding parameter groups."""
    backbone_params = [
        p for name, p in model.named_parameters() if not name.startswith("embedding.")
    ]
    emb_params = [model.embedding.weight]

    emb_lr = cfg.lr * (cfg.emb_lr_factor if cfg.emb_mode == "freeze_then_open" else 1.0)
    param_groups: list[dict[str, Any]] = [
        {"params": backbone_params, "lr": cfg.lr, "weight_decay": cfg.weight_decay},
        {"params": emb_params, "lr": emb_lr, "weight_decay": cfg.weight_decay},
    ]

    loss_params = [p for p in loss_module.parameters() if p.requires_grad]
    if loss_params:
        param_groups.append({"params": loss_params, "lr": cfg.lr, "weight_decay": 0.0})

    return torch.optim.AdamW(param_groups)


def _set_embedding_grad_for_epoch(model: ParallelNet, emb_mode: str, freeze_epochs: int, epoch_1idx: int) -> None:
    if emb_mode == "frozen":
        model.embedding.weight.requires_grad_(False)
    elif emb_mode == "open":
        model.embedding.weight.requires_grad_(True)
    elif emb_mode == "freeze_then_open":
        model.embedding.weight.requires_grad_(epoch_1idx > freeze_epochs)
    else:
        raise ValueError(f"Unknown emb_mode: {emb_mode}")


def _train_single_model_instance(
    model: ParallelNet,
    loss_module: MultiTaskLoss,
    x_ids: torch.Tensor,
    x_lens: torch.Tensor,
    y_sent: torch.Tensor,
    y_tox: torch.Tensor,
    cfg: ExperimentConfig,
    device: torch.device,
    seed: int,
    task_filter: Optional[str] = None,
) -> list[dict[str, float]]:
    """Train one ParallelNet instance for exact E epochs with CosineAnnealingLR."""
    model.to(device)
    loss_module.to(device)
    optimizer = _build_optimizer(model, loss_module, cfg)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=cfg.epochs, eta_min=cfg.eta_min
    )

    n_samples = x_ids.size(0)
    x_ids_dev = x_ids.to(device)
    x_lens_dev = x_lens.to(device)
    ys_dev = y_sent.to(device)
    yt_dev = y_tox.to(device)
    gen = torch.Generator()
    gen.manual_seed(seed)

    history: list[dict[str, float]] = []
    all_clip_params = list(model.parameters()) + [
        p for p in loss_module.parameters() if p.requires_grad
    ]

    for epoch in range(1, cfg.epochs + 1):
        _set_embedding_grad_for_epoch(model, cfg.emb_mode, cfg.freeze_epochs, epoch)
        model.train()
        loss_module.train()

        perm_cpu = torch.randperm(n_samples, generator=gen)
        perm_dev = perm_cpu.to(device)
        x_ids_ep = x_ids_dev[perm_dev]
        x_lens_ep = x_lens_dev[perm_dev]
        x_lens_cpu_ep = x_lens[perm_cpu]
        ys_ep = ys_dev[perm_dev]
        yt_ep = yt_dev[perm_dev]
        total_loss = 0.0

        for start in range(0, n_samples, cfg.batch_size):
            end = min(start + cfg.batch_size, n_samples)
            ids_b = x_ids_ep[start:end]
            len_b = x_lens_ep[start:end]
            len_cpu_b = x_lens_cpu_ep[start:end]
            ys_b = ys_ep[start:end]
            yt_b = yt_ep[start:end]

            optimizer.zero_grad(set_to_none=True)
            out = model(ids_b, len_b, len_cpu_b)
            logits_sent = out.get("sentiment") if task_filter in (None, "sentiment") else None
            logits_tox = out.get("toxicity") if task_filter in (None, "toxicity") else None

            loss = loss_module(logits_sent, logits_tox, ys_b, yt_b)
            loss.backward()
            nn.utils.clip_grad_norm_(all_clip_params, max_norm=cfg.max_grad_norm)
            optimizer.step()

            total_loss += float(loss.item()) * (end - start)

        scheduler.step()
        avg_loss = total_loss / max(n_samples, 1)
        history.append(
            {
                "epoch": float(epoch),
                "train_loss": avg_loss,
                "lr_backbone": float(optimizer.param_groups[0]["lr"]),
                "lr_embedding": float(optimizer.param_groups[1]["lr"]),
                "emb_trainable": float(model.embedding.weight.requires_grad),
            }
        )

    return history


def train_and_evaluate(
    cfg: ExperimentConfig,
    train_df: pd.DataFrame,
    artifact_bundle: dict[str, Any],
    eval_df: Optional[pd.DataFrame] = None,
    device: Optional[torch.device] = None,
) -> dict[str, Any]:
    """Train model(s) according to cfg from scratch and optionally evaluate on eval_df."""
    cfg.validate()
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    set_seed(cfg.seed)

    token_to_id = artifact_bundle["token_to_id"]
    emb_matrix = artifact_bundle["embedding_matrix"]
    meta = artifact_bundle["meta"]
    max_len = int(meta["max_len"])
    tox_counts = [int(x) for x in meta["tox_counts"]]
    tox_weights = compute_tox_class_weights(tox_counts, cfg.alpha, cfg.clip_max)

    x_ids, x_lens = encode_known_tokens(train_df["tokens"].tolist(), token_to_id, max_len)
    y_sent = torch.tensor(train_df["y_sent_idx"].to_numpy(), dtype=torch.long)
    y_tox = torch.tensor(train_df["y_tox_idx"].to_numpy(), dtype=torch.long)

    t0 = time.perf_counter()
    if cfg.task_mode == "multi_task":
        model = ParallelNet(
            embedding_matrix=emb_matrix,
            hidden_units=cfg.hidden_units,
            dropout=cfg.dropout,
            use_dense300=cfg.use_dense300,
            task_mode="multi_task",
        )
        loss_mod = MultiTaskLoss(
            loss_mode=cfg.loss_mode,
            tox_weights=tox_weights,
            w_tox=cfg.w_tox,
            gamma=cfg.gamma,
        )
        history = _train_single_model_instance(
            model, loss_mod, x_ids, x_lens, y_sent, y_tox, cfg, device, seed=cfg.seed
        )
        models = [model]
        trainable_params, total_params = count_model_parameters(model, cfg.emb_mode)
    else:
        model_sent = ParallelNet(
            embedding_matrix=emb_matrix,
            hidden_units=cfg.hidden_units,
            dropout=cfg.dropout,
            use_dense300=cfg.use_dense300,
            task_mode="single_task",
            active_task="sentiment",
        )
        loss_sent = MultiTaskLoss("weighted_ce", tox_weights, w_tox=1.0)
        hist_sent = _train_single_model_instance(
            model_sent, loss_sent, x_ids, x_lens, y_sent, y_tox, cfg, device,
            seed=cfg.seed, task_filter="sentiment"
        )

        set_seed(cfg.seed + 1000)
        model_tox = ParallelNet(
            embedding_matrix=emb_matrix,
            hidden_units=cfg.hidden_units,
            dropout=cfg.dropout,
            use_dense300=cfg.use_dense300,
            task_mode="single_task",
            active_task="toxicity",
        )
        loss_tox = MultiTaskLoss(
            loss_mode=cfg.loss_mode,
            tox_weights=tox_weights,
            w_tox=1.0,
            gamma=cfg.gamma,
        )
        hist_tox = _train_single_model_instance(
            model_tox, loss_tox, x_ids, x_lens, y_sent, y_tox, cfg, device,
            seed=cfg.seed + 1000, task_filter="toxicity"
        )
        history = {"sentiment": hist_sent, "toxicity": hist_tox}  # type: ignore[assignment]
        models = [model_sent, model_tox]
        tr_s, tot_s = count_model_parameters(model_sent, cfg.emb_mode)
        tr_t, tot_t = count_model_parameters(model_tox, cfg.emb_mode)
        trainable_params = tr_s + tr_t
        total_params = tot_s + tot_t

    train_seconds = time.perf_counter() - t0

    result: dict[str, Any] = {
        "models": models,
        "history": history,
        "train_seconds": float(train_seconds),
        "trainable_params": int(trainable_params),
        "total_params": int(total_params),
    }

    if eval_df is not None:
        predictor = DynamicPredictor(
            models=models,
            token_to_id=token_to_id,
            max_len=max_len,
            oov_provider=artifact_bundle.get("oov_provider"),
            device=device,
            clone_models=True,
        )
        logits_sent, logits_tox = predictor.predict_logits_from_tokens(
            eval_df["tokens"].tolist(), batch_size=256
        )
        ys_true = eval_df["y_sent_idx"].to_numpy()
        yt_true = eval_df["y_tox_idx"].to_numpy()
        f1_sent, f1_tox, comb = compute_combined_f1(
            ys_true, logits_sent.argmax(axis=1), yt_true, logits_tox.argmax(axis=1)
        )
        result["eval"] = {
            "macro_f1_sent": f1_sent,
            "macro_f1_tox": f1_tox,
            "combined_macro_f1": comb,
            "logits_sent": logits_sent,
            "logits_tox": logits_tox,
            "oov_hits": predictor.oov_hits,
        }

    return result
