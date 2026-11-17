"""Configuration definitions and project paths for the NT120 fastText v2 pipeline."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Optional

# Root directories
LOCAL_ROOT = Path(__file__).resolve().parent.parent
WORKSPACE_ROOT = LOCAL_ROOT.parent
DATA_DIR = LOCAL_ROOT / "data"
RAW_DATA_DIR = (
    DATA_DIR
    if (DATA_DIR / "train" / "train.csv").exists()
    else (WORKSPACE_ROOT / "NT120-fastText" / "data")
)
PROCESSED_DATA_DIR = DATA_DIR / "processed"
EMBEDDINGS_DIR = LOCAL_ROOT / "embeddings"
DEFAULT_FASTTEXT_BIN = EMBEDDINGS_DIR / "cc.vi.300.bin"

ARTIFACTS_DIR = LOCAL_ROOT / "artifacts"
CONFIGS_DIR = LOCAL_ROOT / "configs"
RUNS_DIR = LOCAL_ROOT / "runs"
REPORTS_DIR = LOCAL_ROOT / "reports"
TESTS_DIR = LOCAL_ROOT / "tests"

# Constants from PIPELINE_SPEC.md
EMBEDDING_DIM = 300
NUM_SENT_CLASSES = 3  # -1 -> 0, 0 -> 1, 1 -> 2
NUM_TOX_CLASSES = 4   # 0, 1, 2, 3
SENT_LABELS = [-1, 0, 1]
SENT_NAMES = ["Negative (-1)", "Neutral (0)", "Positive (1)"]
TOX_LABELS = [0, 1, 2, 3]
TOX_NAMES = ["T0_NonToxic", "T1_Unpolite", "T2_Insult", "T3_HateThreat"]

RANDOM_SEARCH_SPACE = {
    "lr": [1e-3, 5e-4, 2e-4],
    "epochs": [2, 3, 5, 8],
    "dropout": [0.2, 0.3, 0.5],
    "hidden_units": [64, 128, 256],
    "weight_decay": [0.0, 1e-4, 1e-2],
    "alpha": [0.0, 0.5],
    "clip_max": [5.0, 10.0, None],
    "w_tox": [1.0, 2.0],
    "gamma": [1.0, 2.0],
    "freeze_epochs": [1, 2],
    "emb_lr_factor": [0.1, 0.2, 0.3],
}

BETA_GRID = [0.0, 0.25, 0.5, 0.75, 1.0]
FINAL_SEEDS = [1, 2, 3]


@dataclass
class ExperimentConfig:
    """Hyperparameters and structural flags for a single model run."""

    config_id: str = "C0"
    round_name: str = "round1"
    seed: int = 1
    arch: str = "parallel"
    emb_mode: str = "freeze_then_open"  # "freeze_then_open" | "open" | "frozen"
    freeze_epochs: int = 2
    emb_lr_factor: float = 0.1
    use_dense300: bool = False
    task_mode: str = "multi_task"       # "multi_task" | "single_task"
    loss_mode: str = "weighted_ce"      # "weighted_ce" | "focal_tox" | "uncertainty"
    w_tox: float = 1.0
    alpha: float = 0.5
    clip_max: Optional[float] = 10.0
    gamma: float = 2.0
    lr: float = 5e-4
    epochs: int = 5
    dropout: float = 0.3
    hidden_units: int = 128
    weight_decay: float = 1e-4
    batch_size: int = 64
    max_grad_norm: float = 1.0
    eta_min: float = 0.0

    def validate(self) -> None:
        if self.arch != "parallel":
            raise ValueError(f"Unsupported arch: {self.arch}")
        if self.emb_mode not in {"freeze_then_open", "open", "frozen"}:
            raise ValueError(f"Invalid emb_mode: {self.emb_mode}")
        if self.task_mode not in {"multi_task", "single_task"}:
            raise ValueError(f"Invalid task_mode: {self.task_mode}")
        if self.loss_mode not in {"weighted_ce", "focal_tox", "uncertainty"}:
            raise ValueError(f"Invalid loss_mode: {self.loss_mode}")
        if self.task_mode == "single_task" and self.loss_mode == "uncertainty":
            raise ValueError("uncertainty loss is only valid for multi_task")
        if self.emb_mode == "freeze_then_open" and not (1 <= self.freeze_epochs < self.epochs):
            raise ValueError(
                f"freeze_then_open requires 1 <= freeze_epochs < epochs "
                f"(got F={self.freeze_epochs}, E={self.epochs})"
            )

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "ExperimentConfig":
        valid_fields = set(cls.__dataclass_fields__.keys())
        filtered = {k: v for k, v in data.items() if k in valid_fields}
        return cls(**filtered)


def ensure_project_dirs() -> None:
    """Create standard directory layout under local_train."""
    for path in [
        PROCESSED_DATA_DIR,
        EMBEDDINGS_DIR,
        ARTIFACTS_DIR / "train",
        ARTIFACTS_DIR / "trainV1",
        ARTIFACTS_DIR / "full",
        CONFIGS_DIR,
        RUNS_DIR / "final_models",
        REPORTS_DIR,
        TESTS_DIR,
    ]:
        path.mkdir(parents=True, exist_ok=True)
