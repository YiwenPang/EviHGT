"""Portable workflow configuration; relative paths are rooted at the repository."""
from dataclasses import asdict, dataclass
from pathlib import Path
import math
import yaml


ROOT = Path(__file__).resolve().parents[2]


@dataclass(frozen=True)
class Settings:
    manifest: str = "data/processed/corpus/manifest.json"
    output_dir: str = "outputs"
    device: str = "cuda"
    precision: str = "bf16"
    learning_rate: float = 0.00005
    weight_decay: float = 0.00001
    max_grad_norm: float = 1.0
    checkpoint_every: int = 250
    evaluate_every_cycle: bool = True
    refresh_seconds: float = 5.0

    def __post_init__(self):
        for name in ("manifest", "output_dir"):
            if not isinstance(getattr(self, name), str) or not getattr(self, name).strip():
                raise ValueError(f"{name} must be a nonempty path string")
        for name in ("learning_rate", "weight_decay", "max_grad_norm", "refresh_seconds"):
            value = getattr(self, name)
            if type(value) not in (int, float) or not math.isfinite(value):
                raise ValueError(f"{name} must be a finite number")
        if type(self.checkpoint_every) is not int or type(self.evaluate_every_cycle) is not bool:
            raise ValueError("checkpoint_every must be an integer; evaluate_every_cycle must be a Boolean")
        if self.device not in ("cpu", "cuda") or self.precision not in ("bf16", "fp32"):
            raise ValueError("Use device cpu/cuda and precision bf16/fp32")
        if self.device == "cpu" and self.precision != "fp32":
            raise ValueError("CPU execution requires explicit precision: fp32")
        if self.learning_rate <= 0 or self.weight_decay < 0 or self.max_grad_norm <= 0:
            raise ValueError("Invalid optimizer settings")
        if self.checkpoint_every < 1 or self.refresh_seconds <= 0:
            raise ValueError("Checkpoint and refresh intervals must be positive")

    def resolve(self, value):
        path = Path(value).expanduser()
        return (path if path.is_absolute() else ROOT / path).resolve()

    def as_dict(self):
        return asdict(self)


def load_settings(path=None):
    path = ROOT / "configs/default.yaml" if path is None else Path(path).resolve()
    if not path.is_file():
        raise FileNotFoundError(f"Configuration not found: {path}")
    values = yaml.safe_load(path.read_text(encoding="utf8"))
    if not isinstance(values, dict):
        raise ValueError("Configuration must be a YAML mapping")
    return Settings(**values)
