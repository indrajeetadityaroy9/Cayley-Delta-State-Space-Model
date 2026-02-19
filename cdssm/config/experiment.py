"""Experiment configuration and YAML loading."""

from dataclasses import asdict, dataclass, field, fields
from pathlib import Path

import yaml

from cdssm.config.model import CDSSMConfig


@dataclass
class DataConfig:
    """Dataset configuration."""
    dataset_name: str = "wikitext"
    dataset_config: str = "wikitext-103-raw-v1"
    context_length: int = 1024
    text_field: str = "text"
    num_tokens: int = 300_000_000
    batch_size: int = 8
    num_workers: int = 8


@dataclass
class TrainingConfig:
    """Training hyperparameters."""
    epochs: int = 20
    grad_accum_steps: int = 8
    base_lr: float = 6e-4
    weight_decay: float = 0.1
    betas: tuple[float, float] = (0.9, 0.95)
    warmup_steps: int = 500
    grad_clip: float = 1.0
    min_lr_ratio: float = 0.0


@dataclass
class EvalConfig:
    """Evaluation and benchmarking configuration."""
    checkpoint: str = ""
    tasks: list[str] = field(default_factory=list)
    output_dir: str = "results"


@dataclass
class ExperimentConfig:
    """Complete experiment specification."""
    experiment_name: str = "default"
    seed: int = 42
    checkpoint_dir: str = "checkpoints"
    model: CDSSMConfig = field(default_factory=CDSSMConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    data: DataConfig = field(default_factory=DataConfig)
    eval: EvalConfig = field(default_factory=EvalConfig)

    def to_yaml(self) -> str:
        """Serialize config to YAML string (for stamping into output dirs)."""
        d = asdict(self)
        d["training"]["betas"] = list(d["training"]["betas"])
        return yaml.dump(d, default_flow_style=False, sort_keys=False)


_DATA_FIELDS = {f.name for f in fields(DataConfig)}
_TRAINING_FIELDS = {f.name for f in fields(TrainingConfig)}
_EVAL_FIELDS = {f.name for f in fields(EvalConfig)}


def load_config(path: str | Path) -> ExperimentConfig:
    """Load experiment config from YAML.

    YAML sections map directly to dataclass constructors. Unspecified fields
    use the dataclass defaults — no duplicate defaults here.
    """
    path = Path(path)
    with open(path, "r", encoding="utf-8") as f:
        raw = yaml.safe_load(f)

    data_raw = raw.get("data", {})
    data_cfg = DataConfig(**{k: v for k, v in data_raw.items() if k in _DATA_FIELDS})

    training_raw = raw.get("training", {})
    training_raw["betas"] = tuple(training_raw.get("betas", [0.9, 0.95]))
    training_cfg = TrainingConfig(**{k: v for k, v in training_raw.items() if k in _TRAINING_FIELDS})

    eval_raw = raw.get("eval", {})
    eval_cfg = EvalConfig(**{k: v for k, v in eval_raw.items() if k in _EVAL_FIELDS})

    model_raw = raw.get("model", {})
    model_cfg = CDSSMConfig(
        d_model=model_raw.get("d_model", CDSSMConfig.d_model),
        n_layers=model_raw.get("n_layers", CDSSMConfig.n_layers),
        context_length=data_cfg.context_length,
        vocab_size=CDSSMConfig.vocab_size,
        state_dim=model_raw.get("state_dim", CDSSMConfig.state_dim),
    )

    return ExperimentConfig(
        experiment_name=raw.get("experiment_name", ExperimentConfig.experiment_name),
        seed=raw.get("seed", ExperimentConfig.seed),
        checkpoint_dir=raw.get("checkpoint_dir", ExperimentConfig.checkpoint_dir),
        model=model_cfg,
        training=training_cfg,
        data=data_cfg,
        eval=eval_cfg,
    )
