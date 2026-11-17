"""Phase 3: Parallel (Bi-GRU || Multi-kernel Conv1D) network and loss functions."""

from __future__ import annotations

from typing import Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from src_v2.config import EMBEDDING_DIM, NUM_SENT_CLASSES, NUM_TOX_CLASSES


class ParallelNet(nn.Module):
    """Parallel Bi-GRU || Conv1D(k=2, 3) network with strict actual_len masking."""

    def __init__(
        self,
        embedding_matrix: np.ndarray,
        hidden_units: int = 128,
        dropout: float = 0.3,
        use_dense300: bool = False,
        task_mode: str = "multi_task",
        active_task: Optional[str] = None,
    ) -> None:
        super().__init__()
        if task_mode not in {"multi_task", "single_task"}:
            raise ValueError(f"Invalid task_mode: {task_mode}")
        if task_mode == "single_task" and active_task not in {"sentiment", "toxicity"}:
            raise ValueError("single_task requires active_task in {'sentiment', 'toxicity'}")

        self.hidden_units = hidden_units
        self.dropout_p = dropout
        self.use_dense300 = use_dense300
        self.task_mode = task_mode
        self.active_task = active_task

        emb_tensor = torch.as_tensor(embedding_matrix, dtype=torch.float32)
        num_embeddings, emb_dim = emb_tensor.shape
        if emb_dim != EMBEDDING_DIM:
            raise ValueError(f"Expected embedding dim {EMBEDDING_DIM}, got {emb_dim}")

        self.embedding = nn.Embedding(num_embeddings, emb_dim, padding_idx=0)
        with torch.no_grad():
            self.embedding.weight.copy_(emb_tensor)
            self.embedding.weight[0].zero_()

        self.input_dropout = nn.Dropout(p=dropout)
        self.dense300 = (
            nn.Sequential(nn.Linear(emb_dim, emb_dim), nn.ReLU())
            if use_dense300
            else None
        )

        # Branch 4a: Bi-GRU
        self.gru = nn.GRU(
            input_size=emb_dim,
            hidden_size=hidden_units,
            batch_first=True,
            bidirectional=True,
        )

        # Branch 4b: Multi-kernel Conv1D (k=2, k=3)
        self.conv2 = nn.Conv1d(emb_dim, hidden_units, kernel_size=2, padding=0)
        self.conv3 = nn.Conv1d(emb_dim, hidden_units, kernel_size=3, padding=1)

        # 5 + 6: 4H per step -> Max + Avg = 8H
        self.head_dropout = nn.Dropout(p=dropout)
        self.fc_shared = nn.Linear(8 * hidden_units, 64)

        self.head_sent = (
            nn.Linear(64, NUM_SENT_CLASSES)
            if (task_mode == "multi_task" or active_task == "sentiment")
            else None
        )
        self.head_tox = (
            nn.Linear(64, NUM_TOX_CLASSES)
            if (task_mode == "multi_task" or active_task == "toxicity")
            else None
        )

    def forward(
        self,
        input_ids: torch.Tensor,
        actual_len: torch.Tensor,
        actual_len_cpu: Optional[torch.Tensor] = None,
    ) -> dict[str, torch.Tensor]:
        """Forward pass returning raw logits dict ('sentiment' and/or 'toxicity')."""
        _, seq_len = input_ids.shape
        # Mask built strictly from actual_len, never from input_ids != 0
        positions = torch.arange(seq_len, device=input_ids.device).unsqueeze(0)
        mask = positions < actual_len.unsqueeze(1)  # (B, L) bool
        mask_f = mask.unsqueeze(-1).to(dtype=self.embedding.weight.dtype)  # (B, L, 1)

        x = self.embedding(input_ids)  # (B, L, 300)
        x = self.input_dropout(x)
        if self.dense300 is not None:
            x = self.dense300(x)

        # 4a. Bi-GRU branch via pack_padded_sequence
        lengths_for_pack = actual_len_cpu if actual_len_cpu is not None else actual_len.detach().cpu()
        packed = nn.utils.rnn.pack_padded_sequence(
            x,
            lengths_for_pack,
            batch_first=True,
            enforce_sorted=False,
        )
        packed_out, _ = self.gru(packed)
        h_rnn, _ = nn.utils.rnn.pad_packed_sequence(
            packed_out, batch_first=True, total_length=seq_len
        )  # (B, L, 2H)

        # 4b. Multi-kernel Conv1D branch with strict pre/post masking
        x_masked = (x * mask_f).transpose(1, 2)  # (B, 300, L)
        h_cnn2 = F.relu(self.conv2(F.pad(x_masked, (0, 1)))).transpose(1, 2) * mask_f
        h_cnn3 = F.relu(self.conv3(x_masked)).transpose(1, 2) * mask_f

        # 5. Concat features -> (B, L, 4H)
        h_seq = torch.cat([h_rnn, h_cnn2, h_cnn3], dim=-1)

        # 6. Masked Max + Masked Avg Pooling -> (B, 8H)
        h_for_max = h_seq.masked_fill(~mask.unsqueeze(-1), -1e9)
        h_max = h_for_max.max(dim=1).values
        denom = actual_len.unsqueeze(-1).clamp(min=1).to(dtype=h_seq.dtype)
        h_avg = (h_seq * mask_f).sum(dim=1) / denom
        pooled = torch.cat([h_max, h_avg], dim=-1)

        # 7. Shared hidden + output logits (no Softmax)
        feat = F.relu(self.fc_shared(self.head_dropout(pooled)))
        out: dict[str, torch.Tensor] = {}
        if self.head_sent is not None:
            out["sentiment"] = self.head_sent(feat)
        if self.head_tox is not None:
            out["toxicity"] = self.head_tox(feat)
        return out

    def expand_embeddings(self, extra_vectors: np.ndarray) -> None:
        """Append new OOV rows to nn.Embedding without touching existing rows (inference only)."""
        if extra_vectors.size == 0:
            return
        extra_t = torch.as_tensor(
            extra_vectors,
            dtype=self.embedding.weight.dtype,
            device=self.embedding.weight.device,
        )
        with torch.no_grad():
            new_weight = torch.cat([self.embedding.weight.data, extra_t], dim=0)
            new_emb = nn.Embedding(
                new_weight.shape[0],
                new_weight.shape[1],
                padding_idx=0,
            ).to(device=new_weight.device, dtype=new_weight.dtype)
            new_emb.weight.copy_(new_weight)
            new_emb.weight.requires_grad_(self.embedding.weight.requires_grad)
            self.embedding = new_emb


def compute_tox_class_weights(
    tox_counts: list[int],
    alpha: float,
    clip_max: Optional[float],
) -> torch.Tensor:
    """Compute smoothed capped class weights: min((N / (4 * n_k))^alpha, clip_max)."""
    if alpha == 0.0:
        return torch.ones(NUM_TOX_CLASSES, dtype=torch.float32)
    n_total = float(sum(tox_counts))
    weights = []
    for count in tox_counts:
        w = (n_total / (4.0 * max(int(count), 1))) ** alpha
        if clip_max is not None:
            w = min(w, float(clip_max))
        weights.append(float(w))
    return torch.tensor(weights, dtype=torch.float32)


class MultiTaskLoss(nn.Module):
    """Unified loss module for weighted_ce, focal_tox, and uncertainty weighting."""

    def __init__(
        self,
        loss_mode: str,
        tox_weights: torch.Tensor,
        w_tox: float = 1.0,
        gamma: float = 2.0,
    ) -> None:
        super().__init__()
        self.loss_mode = loss_mode
        self.w_tox = w_tox
        self.gamma = gamma
        self.register_buffer("tox_weights", tox_weights.clone().detach())
        if loss_mode == "uncertainty":
            self.s_sent = nn.Parameter(torch.tensor(0.0, dtype=torch.float32))
            self.s_tox = nn.Parameter(torch.tensor(0.0, dtype=torch.float32))
        else:
            self.s_sent = None
            self.s_tox = None

    def compute_tox_loss(self, logits_tox: torch.Tensor, y_tox: torch.Tensor) -> torch.Tensor:
        if self.loss_mode == "focal_tox":
            log_probs = F.log_softmax(logits_tox, dim=-1)
            log_pt = log_probs.gather(1, y_tox.unsqueeze(1)).squeeze(1)
            pt = log_pt.exp()
            w_i = self.tox_weights[y_tox]
            return (w_i * ((1.0 - pt) ** self.gamma) * (-log_pt)).mean()
        return F.cross_entropy(logits_tox, y_tox, weight=self.tox_weights)

    def forward(
        self,
        logits_sent: Optional[torch.Tensor],
        logits_tox: Optional[torch.Tensor],
        y_sent: torch.Tensor,
        y_tox: torch.Tensor,
    ) -> torch.Tensor:
        if logits_sent is not None and logits_tox is None:
            return F.cross_entropy(logits_sent, y_sent)
        if logits_tox is not None and logits_sent is None:
            return self.compute_tox_loss(logits_tox, y_tox)
        assert logits_sent is not None and logits_tox is not None

        loss_sent = F.cross_entropy(logits_sent, y_sent)
        loss_tox = self.compute_tox_loss(logits_tox, y_tox)

        if self.loss_mode == "uncertainty":
            assert self.s_sent is not None and self.s_tox is not None
            return (
                torch.exp(-self.s_sent) * loss_sent
                + 0.5 * self.s_sent
                + torch.exp(-self.s_tox) * loss_tox
                + 0.5 * self.s_tox
            )
        return loss_sent + self.w_tox * loss_tox


def count_model_parameters(model: ParallelNet, emb_mode: str) -> tuple[int, int]:
    """Return (trainable_params, total_params) considering final embedding trainability."""
    total = sum(p.numel() for p in model.parameters())
    if emb_mode == "frozen":
        trainable = sum(
            p.numel() for name, p in model.named_parameters() if not name.startswith("embedding.")
        )
    else:
        trainable = total
    return int(trainable), int(total)
