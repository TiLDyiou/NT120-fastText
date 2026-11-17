"""Phase 3: Inference utilities with Dynamic Vocabulary Expansion (lookup cache)."""

from __future__ import annotations

import copy
from pathlib import Path
from typing import Iterable, Optional, Protocol

import numpy as np
import torch
import torch.nn.functional as F

from src_v2.build_embeddings import PAD_ID
from src_v2.data_prep import clean_text, tokenize
from src_v2.model import ParallelNet


class OOVVectorProvider(Protocol):
    """Protocol for resolving 300-d fastText vectors for unseen tokens."""

    def get_vector(self, token: str) -> np.ndarray:
        ...


class PreCachedOOVProvider:
    """Lookup provider backed by pre-cached .npy vectors (used for valid / V2)."""

    def __init__(self, oov_map: dict[str, int], oov_vectors: np.ndarray) -> None:
        self.oov_map = oov_map
        self.oov_vectors = oov_vectors

    def get_vector(self, token: str) -> np.ndarray:
        if token not in self.oov_map:
            raise KeyError(f"Token {token!r} missing from pre-cached OOV map.")
        return self.oov_vectors[self.oov_map[token]]


class FastTextBinOOVProvider:
    """Lookup provider backed directly by cc.vi.300.bin (used for Phase 7 challenge)."""

    def __init__(self, ft_model) -> None:
        self.ft_model = ft_model

    def get_vector(self, token: str) -> np.ndarray:
        return self.ft_model.get_word_vector(token).astype(np.float32)


class DynamicPredictor:
    """Stateless-output inference wrapper with Dynamic Vocabulary Expansion cache."""

    def __init__(
        self,
        models: list[ParallelNet],
        token_to_id: dict[str, int],
        max_len: int,
        oov_provider: Optional[OOVVectorProvider] = None,
        device: Optional[torch.device] = None,
        clone_models: bool = True,
    ) -> None:
        self.models = [copy.deepcopy(m) if clone_models else m for m in models]
        for m in self.models:
            m.eval()
        self.token_to_id = dict(token_to_id)
        self.max_len = int(max_len)
        self.oov_provider = oov_provider
        self.device = device or next(self.models[0].parameters()).device
        for m in self.models:
            m.to(self.device)
        self.oov_hits = 0

    def _encode_batch_with_expansion(
        self, token_lists: list[list[str]]
    ) -> tuple[torch.Tensor, torch.Tensor]:
        batch_size = len(token_lists)
        input_ids = np.full((batch_size, self.max_len), PAD_ID, dtype=np.int64)
        actual_lens = np.ones(batch_size, dtype=np.int64)
        new_vectors: list[np.ndarray] = []

        for row_idx, tokens in enumerate(token_lists):
            truncated = tokens[: self.max_len]
            actual_lens[row_idx] = max(1, len(truncated))
            for col_idx, tok in enumerate(truncated):
                tok_id = self.token_to_id.get(tok)
                if tok_id is None:
                    if self.oov_provider is None:
                        raise KeyError(
                            f"Encountered unseen token {tok!r} with no OOVVectorProvider."
                        )
                    vec = self.oov_provider.get_vector(tok).astype(np.float32)
                    tok_id = len(self.token_to_id)
                    self.token_to_id[tok] = tok_id
                    new_vectors.append(vec)
                    self.oov_hits += 1
                input_ids[row_idx, col_idx] = tok_id

        if new_vectors:
            extra_mat = np.stack(new_vectors, axis=0)
            for m in self.models:
                m.expand_embeddings(extra_mat)

        return (
            torch.from_numpy(input_ids).to(self.device),
            torch.from_numpy(actual_lens).to(self.device),
        )

    @torch.no_grad()
    def predict_logits_from_tokens(
        self,
        token_lists: list[list[str]],
        batch_size: int = 256,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Predict raw logits (logits_sent, logits_tox) for tokenized sequences."""
        sent_chunks: list[np.ndarray] = []
        tox_chunks: list[np.ndarray] = []

        for start in range(0, len(token_lists), batch_size):
            batch_tokens = token_lists[start : start + batch_size]
            ids_t, len_t = self._encode_batch_with_expansion(batch_tokens)

            if len(self.models) == 1:
                out = self.models[0](ids_t, len_t)
                sent_chunks.append(out["sentiment"].detach().cpu().numpy())
                tox_chunks.append(out["toxicity"].detach().cpu().numpy())
            else:
                out_sent = self.models[0](ids_t, len_t)
                out_tox = self.models[1](ids_t, len_t)
                sent_chunks.append(out_sent["sentiment"].detach().cpu().numpy())
                tox_chunks.append(out_tox["toxicity"].detach().cpu().numpy())

        return np.concatenate(sent_chunks, axis=0), np.concatenate(tox_chunks, axis=0)

    def predict_logits_from_raw(
        self,
        raw_texts: Iterable[str],
        batch_size: int = 256,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Full inference pipeline from raw texts: clean_text -> tokenize -> logits."""
        token_lists = [tokenize(clean_text(t)) for t in raw_texts]
        return self.predict_logits_from_tokens(token_lists, batch_size=batch_size)


def apply_postproc(
    logits_sent: np.ndarray,
    logits_tox: np.ndarray,
    t_sent: float = 1.0,
    t_tox: float = 1.0,
    tau_k: Optional[np.ndarray] = None,
) -> dict[str, np.ndarray]:
    """Convert raw logits into calibrated probabilities and final predictions."""
    probs_sent = F.softmax(torch.from_numpy(logits_sent / float(t_sent)), dim=-1).numpy()
    probs_tox = F.softmax(torch.from_numpy(logits_tox / float(t_tox)), dim=-1).numpy()
    pred_sent_idx = probs_sent.argmax(axis=1)
    if tau_k is not None:
        weighted_probs_tox = probs_tox * np.asarray(tau_k, dtype=np.float64).reshape(1, -1)
        pred_tox_idx = weighted_probs_tox.argmax(axis=1)
    else:
        pred_tox_idx = probs_tox.argmax(axis=1)

    return {
        "probs_sent": probs_sent,
        "probs_tox": probs_tox,
        "pred_sent_idx": pred_sent_idx,
        "pred_tox_idx": pred_tox_idx,
    }


class ZeroFallbackOOVProvider:
    """Fallback OOV provider returning zero vectors when cc.vi.300.bin is unavailable."""

    def get_vector(self, token: str) -> np.ndarray:
        return np.zeros(300, dtype=np.float32)


class NT120Predictor:
    """High-level inference API returning (-1/0/1, 0/1/2/3) labels and calibrated probabilities."""

    def __init__(
        self,
        dynamic_predictor: DynamicPredictor,
        t_sent: float,
        t_tox: float,
        tau_k: np.ndarray,
    ) -> None:
        self.dynamic_predictor = dynamic_predictor
        self.t_sent = float(t_sent)
        self.t_tox = float(t_tox)
        self.tau_k = np.asarray(tau_k, dtype=np.float64)

    @classmethod
    def from_dir(
        cls,
        seed_dir: str | Path,
        fasttext_bin: str | Path | None = None,
        device: Optional[str | torch.device] = None,
        allow_zero_oov: bool = False,
    ) -> "NT120Predictor":
        from src_v2.final_train import load_final_seed_model

        dev = torch.device(device) if device is not None else torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )
        models, cfg_data, token_to_id = load_final_seed_model(Path(seed_dir), dev)

        if fasttext_bin is not None:
            import fasttext

            ft_model = fasttext.load_model(str(fasttext_bin))
            oov_provider: Optional[OOVVectorProvider] = FastTextBinOOVProvider(ft_model)
        elif allow_zero_oov:
            oov_provider = ZeroFallbackOOVProvider()
        else:
            oov_provider = None

        dyn = DynamicPredictor(
            models=models,
            token_to_id=token_to_id,
            max_len=int(cfg_data["max_len_full"]),
            oov_provider=oov_provider,
            device=dev,
            clone_models=False,
        )
        return cls(
            dyn,
            t_sent=float(cfg_data["t_bar_sent"]),
            t_tox=float(cfg_data["t_bar_tox"]),
            tau_k=np.asarray(cfg_data["tau_k"], dtype=np.float64),
        )

    def predict_proba(self, texts: Iterable[str], batch_size: int = 256) -> tuple[np.ndarray, np.ndarray]:
        """Return calibrated (sent_proba [N, 3], tox_proba [N, 4]) without tau_k adjustment."""
        ls, lt = self.dynamic_predictor.predict_logits_from_raw(texts, batch_size=batch_size)
        post = apply_postproc(ls, lt, t_sent=self.t_sent, t_tox=self.t_tox, tau_k=self.tau_k)
        return post["probs_sent"], post["probs_tox"]

    def predict(self, texts: Iterable[str], batch_size: int = 256) -> np.ndarray:
        """Return array of shape (N, 2) with columns [Sentiment (-1/0/1), Toxicity (0..3)]."""
        ls, lt = self.dynamic_predictor.predict_logits_from_raw(texts, batch_size=batch_size)
        post = apply_postproc(ls, lt, t_sent=self.t_sent, t_tox=self.t_tox, tau_k=self.tau_k)
        sent_labels = post["pred_sent_idx"] - 1
        tox_labels = post["pred_tox_idx"]
        return np.stack([sent_labels, tox_labels], axis=1)
