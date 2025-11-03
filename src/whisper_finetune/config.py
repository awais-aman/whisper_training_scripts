"""Training configuration: defaults, overridden by a TOML file, env vars, then CLI flags."""

from __future__ import annotations

import os
import tomllib
from collections.abc import Mapping
from dataclasses import dataclass, fields, replace
from pathlib import Path
from typing import Any

DEVICES = ("auto", "cuda", "mps", "cpu")

# Environment variables read for the settings most likely to differ per machine.
ENV_VARS = {
    "WHISPER_MODEL": "model_name",
    "WHISPER_MANIFEST": "manifest",
    "WHISPER_OUTPUT_DIR": "output_dir",
    "WHISPER_LANGUAGE": "language",
    "WHISPER_DEVICE": "device",
}


@dataclass(frozen=True)
class TrainConfig:
    manifest: Path = Path("data/processed/manifest.jsonl")
    model_name: str = "openai/whisper-small"
    output_dir: Path = Path("models/whisper-small")
    language: str | None = None
    task: str = "transcribe"
    device: str = "auto"
    learning_rate: float = 1e-5
    warmup_steps: int = 50
    max_steps: int = 500
    batch_size: int = 1
    gradient_accumulation_steps: int = 2
    save_steps: int = 500
    save_total_limit: int = 2
    logging_steps: int = 10
    eval_fraction: float = 0.0
    max_label_length: int = 448
    seed: int = 42

    def __post_init__(self) -> None:
        if self.device not in DEVICES:
            raise ValueError(f"device must be one of {', '.join(DEVICES)}, got {self.device!r}")
        if self.task not in ("transcribe", "translate"):
            raise ValueError(f"task must be 'transcribe' or 'translate', got {self.task!r}")
        if not 0.0 <= self.eval_fraction < 1.0:
            raise ValueError(f"eval_fraction must be in [0, 1), got {self.eval_fraction}")
        if self.warmup_steps >= self.max_steps:
            raise ValueError("warmup_steps must be smaller than max_steps or the learning rate never peaks")


def _coerce(updates: Mapping[str, Any]) -> dict[str, Any]:
    types = {f.name: f.type for f in fields(TrainConfig)}
    unknown = set(updates) - set(types)
    if unknown:
        raise ValueError(f"unknown config keys: {', '.join(sorted(unknown))}")
    coerced: dict[str, Any] = {}
    for key, value in updates.items():
        kind = types[key]
        if kind == "Path":
            coerced[key] = Path(value)
        elif kind == "int":
            coerced[key] = int(value)
        elif kind == "float":
            coerced[key] = float(value)
        else:
            coerced[key] = value
    return coerced


def load_config(
    config_file: Path | None = None,
    env: Mapping[str, str] | None = None,
    overrides: Mapping[str, Any] | None = None,
) -> TrainConfig:
    """Merge the config sources; later sources win. ``None`` overrides are ignored."""
    env = os.environ if env is None else env
    merged: dict[str, Any] = {}
    if config_file is not None:
        merged.update(tomllib.loads(config_file.read_text(encoding="utf-8")))
    merged.update({field: env[var] for var, field in ENV_VARS.items() if env.get(var)})
    merged.update({k: v for k, v in (overrides or {}).items() if v is not None})
    return replace(TrainConfig(), **_coerce(merged))


def detect_device(requested: str = "auto") -> str:
    """Return the device to train on: the requested one, or the best available."""
    import torch

    if requested == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("device 'cuda' requested but no CUDA GPU is available")
    if requested != "auto":
        return requested
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"
