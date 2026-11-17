"""Phase 2: Vocabulary construction, max_len calculation, and fastText .bin matrix extraction."""

from __future__ import annotations

import argparse
import gc
import json
from collections import Counter
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from src_v2.config import (
    ARTIFACTS_DIR,
    DEFAULT_FASTTEXT_BIN,
    EMBEDDING_DIM,
    NUM_SENT_CLASSES,
    NUM_TOX_CLASSES,
    PROCESSED_DATA_DIR,
    ensure_project_dirs,
)

PAD_TOKEN = "<pad>"
PAD_ID = 0


def build_vocab(token_lists: Iterable[Iterable[str]]) -> dict[str, int]:
    """Build token_to_id mapping with <pad>=0 and real tokens 1..|V|. No <unk>."""
    counts: Counter[str] = Counter()
    for tokens in token_lists:
        counts.update(tokens)

    token_to_id: dict[str, int] = {PAD_TOKEN: PAD_ID}
    for tok, _ in sorted(counts.items(), key=lambda kv: (-kv[1], kv[0])):
        if tok == PAD_TOKEN:
            continue
        token_to_id[tok] = len(token_to_id)
    return token_to_id


def compute_max_len(token_lists: Iterable[Iterable[str]]) -> int:
    """Compute rounded 95th percentile of token lengths."""
    lengths = np.fromiter((len(t) for t in token_lists), dtype=np.int64)
    if len(lengths) == 0:
        raise ValueError("Cannot compute max_len on empty sequence list.")
    return max(1, int(round(float(np.percentile(lengths, 95)))))


def _extract_matrix(ft_model, token_to_id: dict[str, int]) -> np.ndarray:
    """Extract float32 embedding matrix of shape (|V|+1, 300) with row 0 = zeros."""
    matrix = np.zeros((len(token_to_id), EMBEDDING_DIM), dtype=np.float32)
    for tok, idx in token_to_id.items():
        if idx == PAD_ID:
            continue
        matrix[idx] = ft_model.get_word_vector(tok).astype(np.float32)
    return matrix


def _extract_oov_cache(ft_model, train_vocab: dict[str, int], eval_token_lists: Iterable[Iterable[str]]) -> tuple[dict[str, int], np.ndarray]:
    """Pre-cache fastText vectors for tokens in V_eval \\ V_train."""
    oov_tokens = sorted({tok for tokens in eval_token_lists for tok in tokens if tok not in train_vocab and tok != PAD_TOKEN})
    oov_map = {tok: i for i, tok in enumerate(oov_tokens)}
    oov_vectors = np.zeros((len(oov_tokens), EMBEDDING_DIM), dtype=np.float32)
    for i, tok in enumerate(oov_tokens):
        oov_vectors[i] = ft_model.get_word_vector(tok).astype(np.float32)
    return oov_map, oov_vectors


def _compute_label_stats(df: pd.DataFrame) -> dict[str, object]:
    n = len(df)
    sent_counts = [int((df["y_sent_idx"] == c).sum()) for c in range(NUM_SENT_CLASSES)]
    tox_counts = [int((df["y_tox_idx"] == k).sum()) for k in range(NUM_TOX_CLASSES)]
    tox_freq = [c / n for c in tox_counts]
    return {
        "num_rows": n,
        "sent_counts": sent_counts,
        "tox_counts": tox_counts,
        "tox_freq": tox_freq,
    }


def _save_bundle(
    out_dir: Path,
    ft_model,
    train_df: pd.DataFrame,
    eval_oov_df: pd.DataFrame | None = None,
    oov_prefix: str = "valid",
) -> dict[str, object]:
    out_dir.mkdir(parents=True, exist_ok=True)
    token_lists = train_df["tokens"].tolist()
    token_to_id = build_vocab(token_lists)
    max_len = compute_max_len(token_lists)
    matrix = _extract_matrix(ft_model, token_to_id)

    np.save(out_dir / "embedding_matrix.npy", matrix)
    (out_dir / "token_to_id.json").write_text(
        json.dumps(token_to_id, ensure_ascii=False), encoding="utf-8"
    )

    meta = {
        "vocab_size": len(token_to_id),
        "num_real_tokens": len(token_to_id) - 1,
        "max_len": max_len,
        **_compute_label_stats(train_df),
    }

    if eval_oov_df is not None:
        oov_map, oov_vecs = _extract_oov_cache(ft_model, token_to_id, eval_oov_df["tokens"].tolist())
        np.save(out_dir / f"{oov_prefix}_oov_vectors.npy", oov_vecs)
        (out_dir / f"{oov_prefix}_oov_map.json").write_text(
            json.dumps(oov_map, ensure_ascii=False), encoding="utf-8"
        )
        meta[f"{oov_prefix}_oov_count"] = len(oov_map)

    (out_dir / "meta.json").write_text(
        json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return meta


def ensure_v1_v2_split(valid_df: pd.DataFrame, processed_dir: Path, seed: int = 42) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Create or load the deterministic 50/50 stratified V1/V2 split of valid (by Toxicity)."""
    v1_path = processed_dir / "valid_v1.parquet"
    v2_path = processed_dir / "valid_v2.parquet"
    if v1_path.exists() and v2_path.exists():
        return pd.read_parquet(v1_path), pd.read_parquet(v2_path)

    v1_df, v2_df = train_test_split(
        valid_df,
        test_size=0.5,
        random_state=seed,
        stratify=valid_df["y_tox_idx"],
    )
    v1_df = v1_df.reset_index(drop=True)
    v2_df = v2_df.reset_index(drop=True)
    v1_df.to_parquet(v1_path, index=False)
    v2_df.to_parquet(v2_path, index=False)
    return v1_df, v2_df


def run_build_embeddings(
    processed_dir: Path = PROCESSED_DATA_DIR,
    artifacts_dir: Path = ARTIFACTS_DIR,
    bin_path: Path = DEFAULT_FASTTEXT_BIN,
    v1v2_seed: int = 42,
) -> dict[str, dict[str, object]]:
    """Load cc.vi.300.bin once and export train, trainV1, and full embedding artifacts."""
    ensure_project_dirs()
    train_df = pd.read_parquet(processed_dir / "train.parquet")
    valid_df = pd.read_parquet(processed_dir / "valid.parquet")
    v1_df, v2_df = ensure_v1_v2_split(valid_df, processed_dir, seed=v1v2_seed)

    train_v1_df = pd.concat([train_df, v1_df], ignore_index=True)
    full_df = pd.concat([train_df, valid_df], ignore_index=True)

    import fasttext

    print(f"Loading fastText binary once from: {bin_path}")
    ft_model = fasttext.load_model(str(bin_path))

    try:
        meta_train = _save_bundle(
            artifacts_dir / "train", ft_model, train_df, eval_oov_df=valid_df, oov_prefix="valid"
        )
        meta_train_v1 = _save_bundle(
            artifacts_dir / "trainV1", ft_model, train_v1_df, eval_oov_df=v2_df, oov_prefix="v2"
        )
        meta_full = _save_bundle(
            artifacts_dir / "full", ft_model, full_df, eval_oov_df=None
        )
    finally:
        del ft_model
        gc.collect()

    summary = {"train": meta_train, "trainV1": meta_train_v1, "full": meta_full}
    print("Embedding extraction complete (cc.vi.300.bin released from RAM):")
    for k, m in summary.items():
        print(f"  [{k}] vocab_size={m['vocab_size']:,}, max_len={m['max_len']}, rows={m['num_rows']:,}")
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Run Phase 2 fastText embedding extraction.")
    parser.add_argument("--processed-dir", type=Path, default=PROCESSED_DATA_DIR)
    parser.add_argument("--artifacts-dir", type=Path, default=ARTIFACTS_DIR)
    parser.add_argument("--bin", type=Path, default=DEFAULT_FASTTEXT_BIN)
    parser.add_argument("--v1v2-seed", type=int, default=42)
    args = parser.parse_args()

    run_build_embeddings(args.processed_dir, args.artifacts_dir, args.bin, args.v1v2_seed)


if __name__ == "__main__":
    main()
