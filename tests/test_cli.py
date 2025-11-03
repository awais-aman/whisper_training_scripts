import json
from pathlib import Path

import numpy as np
import pytest
import soundfile

from whisper_finetune import cli
from whisper_finetune.config import TrainConfig, detect_device, load_config


def test_config_precedence_defaults_file_env_flags(tmp_path):
    toml = tmp_path / "train.toml"
    toml.write_text('model_name = "openai/whisper-base"\nmax_steps = 1000\nlearning_rate = 3e-5\n')
    env = {"WHISPER_MODEL": "openai/whisper-tiny", "WHISPER_OUTPUT_DIR": "out/tiny"}

    config = load_config(toml, env=env, overrides={"max_steps": 2000, "language": None})

    assert config.model_name == "openai/whisper-tiny"  # env beats file
    assert config.max_steps == 2000  # flag beats file
    assert config.learning_rate == 3e-5  # file beats default
    assert config.output_dir == Path("out/tiny")
    assert config.language is None


@pytest.mark.parametrize(
    ("updates", "message"),
    [
        ({"devices": "cpu"}, "unknown config keys"),
        ({"device": "tpu"}, "device must be"),
        ({"warmup_steps": 500, "max_steps": 500}, "never peaks"),
        ({"eval_fraction": 1.0}, "eval_fraction"),
        ({"task": "summarise"}, "task must be"),
    ],
)
def test_invalid_config_is_rejected(updates, message):
    with pytest.raises(ValueError, match=message):
        load_config(env={}, overrides=updates)


def test_parser_maps_flags_to_config_fields():
    args = cli.build_parser().parse_args(
        ["train", "--model", "openai/whisper-tiny", "--max-steps", "20", "--warmup-steps", "2", "--device", "cpu"]
    )
    assert (args.model_name, args.max_steps, args.warmup_steps, args.device) == ("openai/whisper-tiny", 20, 2, "cpu")
    assert args.learning_rate is None  # unset flags must not override the config


def test_parser_rejects_unknown_device():
    with pytest.raises(SystemExit):
        cli.build_parser().parse_args(["train", "--device", "tpu"])


def test_detect_device_prefers_cuda_then_mps(monkeypatch):
    import torch

    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    monkeypatch.setattr(torch.backends.mps, "is_available", lambda: True)
    assert detect_device("auto") == "mps"
    assert detect_device("cpu") == "cpu"
    with pytest.raises(RuntimeError, match="no CUDA"):
        detect_device("cuda")
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    assert detect_device("auto") == "cuda"


def test_dry_run_builds_training_arguments_without_loading_a_model(monkeypatch, capsys):
    import transformers

    def no_loading(*args, **kwargs):
        raise AssertionError("dry run must not load weights or a processor")

    for loader in (transformers.WhisperForConditionalGeneration, transformers.WhisperProcessor):
        monkeypatch.setattr(loader, "from_pretrained", no_loading)
    monkeypatch.setenv("WHISPER_MODEL", "openai/whisper-tiny")
    assert (
        cli.main(
            [
                "train",
                "--dry-run",
                "--device",
                "cpu",
                "--max-steps",
                "40",
                "--warmup-steps",
                "4",
                "--eval-fraction",
                "0.1",
            ]
        )
        == 0
    )

    printed = json.loads(capsys.readouterr().out)
    assert printed["device"] == "cpu"
    assert printed["model"] == "openai/whisper-tiny"
    args = printed["training_arguments"]
    assert args["max_steps"] == 40 and args["warmup_steps"] == 4
    assert args["fp16"] is False and args["use_cpu"] is True
    assert args["eval_strategy"] == "steps"


def test_default_config_without_eval_split_on_mps():
    from whisper_finetune.train import build_training_arguments

    args = build_training_arguments(TrainConfig(), "mps")
    assert args.fp16 is False and args.eval_strategy == "no"


def test_train_without_manifest_fails_cleanly(tmp_path, capsys):
    assert cli.main(["train", "--device", "cpu", "--manifest", str(tmp_path / "none.jsonl")]) == 1
    assert "run `whisper-finetune prepare` first" in capsys.readouterr().err


def test_prepare_command_end_to_end(tmp_path, capsys):
    soundfile.write(str(tmp_path / "a.wav"), np.zeros(160, dtype="float32"), 16_000)
    (tmp_path / "t.tsv").write_text("a.wav\thi\nmissing.wav\tgone\n")
    out = tmp_path / "manifest.jsonl"

    assert cli.main(["prepare", str(tmp_path / "t.tsv"), "-o", str(out)]) == 0

    captured = capsys.readouterr()
    assert "wrote 1 rows" in captured.out and "1 skipped" in captured.out
    assert "missing.wav: audio file not found" in captured.err
    assert out.read_text().count("\n") == 1
