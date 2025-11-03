"""Command line entry point: ``whisper-finetune prepare`` and ``whisper-finetune train``."""

from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Sequence
from pathlib import Path

from .config import DEVICES, TrainConfig, detect_device, load_config
from .manifest import WHISPER_SAMPLING_RATE, build_manifest

# CLI flag -> TrainConfig field, for the flags that override the config.
TRAIN_FLAGS = {
    "manifest": Path,
    "model_name": str,
    "output_dir": Path,
    "language": str,
    "task": str,
    "device": str,
    "learning_rate": float,
    "warmup_steps": int,
    "max_steps": int,
    "batch_size": int,
    "gradient_accumulation_steps": int,
    "save_steps": int,
    "save_total_limit": int,
    "logging_steps": int,
    "eval_fraction": float,
    "max_label_length": int,
    "seed": int,
}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="whisper-finetune", description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)

    prepare = commands.add_parser("prepare", help="build a JSONL manifest from a transcript table")
    prepare.add_argument("transcripts", type=Path, help="TSV/CSV (audio path, text) or JSON/JSONL records")
    prepare.add_argument("-o", "--output", type=Path, default=Path("data/processed/manifest.jsonl"))
    prepare.add_argument("--audio-dir", type=Path, help="look audio files up by name in this folder")
    prepare.add_argument("--header", action="store_true", help="the TSV/CSV has a header row")
    prepare.add_argument("--sampling-rate", type=int, default=WHISPER_SAMPLING_RATE)

    train = commands.add_parser("train", help="fine-tune Whisper on a manifest")
    train.add_argument("--config", type=Path, help="TOML file with TrainConfig keys")
    train.add_argument(
        "--dry-run", action="store_true", help="print the resolved settings and exit without loading a model"
    )
    for name, kind in TRAIN_FLAGS.items():
        flag = "--model" if name == "model_name" else "--" + name.replace("_", "-")
        extra = {"choices": DEVICES} if name == "device" else {}
        train.add_argument(flag, dest=name, type=kind, default=None, **extra)
    return parser


def _prepare(args: argparse.Namespace) -> int:
    report = build_manifest(
        args.transcripts,
        args.output,
        audio_dir=args.audio_dir,
        has_header=args.header,
        sampling_rate=args.sampling_rate,
    )
    for path in report.missing_audio:
        print(f"skipped {path}: audio file not found", file=sys.stderr)
    for path in report.empty_text:
        print(f"skipped {path}: empty transcript", file=sys.stderr)
    for path, rate in report.wrong_sampling_rate:
        print(f"skipped {path}: {rate} Hz, expected {args.sampling_rate} Hz", file=sys.stderr)
    print(f"wrote {report.written} rows to {args.output} ({report.skipped} skipped)")
    return 0 if report.written else 1


def describe(config: TrainConfig, device: str) -> dict:
    """The resolved settings that a dry run prints."""
    from .train import build_training_arguments

    training_args = build_training_arguments(config, device)
    keys = (
        "output_dir",
        "learning_rate",
        "warmup_steps",
        "max_steps",
        "per_device_train_batch_size",
        "gradient_accumulation_steps",
        "fp16",
        "use_cpu",
        "eval_strategy",
        "save_steps",
        "seed",
    )
    settings = training_args.to_dict()
    return {
        "device": device,
        "model": config.model_name,
        "manifest": str(config.manifest),
        "language": config.language,
        "task": config.task,
        "training_arguments": {key: settings[key] for key in keys},
    }


def _train(args: argparse.Namespace) -> int:
    overrides = {name: getattr(args, name) for name in TRAIN_FLAGS}
    config = load_config(args.config, overrides=overrides)
    device = detect_device(config.device)
    if args.dry_run:
        print(json.dumps(describe(config, device), indent=2))
        return 0
    if not config.manifest.is_file():
        print(f"manifest not found: {config.manifest} (run `whisper-finetune prepare` first)", file=sys.stderr)
        return 1
    from .train import run_training

    run_training(config, device)
    return 0


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        return _prepare(args) if args.command == "prepare" else _train(args)
    except (ValueError, RuntimeError, FileNotFoundError) as error:
        print(f"error: {error}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
