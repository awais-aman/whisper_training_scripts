"""Run real optimisation steps on a tiny, randomly initialised Whisper (no download)."""

import math

import numpy as np
import torch
from transformers import Seq2SeqTrainer, WhisperConfig, WhisperForConditionalGeneration

from whisper_finetune.config import TrainConfig
from whisper_finetune.features import SpeechSeq2SeqCollator
from whisper_finetune.train import build_training_arguments

START = 1


def tiny_whisper() -> WhisperForConditionalGeneration:
    config = WhisperConfig(
        vocab_size=64,
        num_mel_bins=8,
        d_model=16,
        encoder_layers=1,
        decoder_layers=1,
        encoder_attention_heads=2,
        decoder_attention_heads=2,
        encoder_ffn_dim=32,
        decoder_ffn_dim=32,
        max_source_positions=8,
        max_target_positions=16,
        pad_token_id=0,
        bos_token_id=START,
        eos_token_id=2,
        decoder_start_token_id=START,
    )
    torch.manual_seed(0)
    return WhisperForConditionalGeneration(config)


def test_trainer_runs_with_collator_and_built_arguments(tmp_path):
    model = tiny_whisper()
    rng = np.random.default_rng(0)
    # 8 mel bins x 16 frames: Whisper's encoder expects 2 * max_source_positions frames.
    rows = [
        {"input_features": rng.standard_normal((8, 16)).astype("float32"), "labels": [START, *ids, 2]}
        for ids in ([5, 6, 7], [8, 9], [10, 11, 12, 13])
    ]
    config = TrainConfig(output_dir=tmp_path, max_steps=2, warmup_steps=1, batch_size=2, gradient_accumulation_steps=1)

    trainer = Seq2SeqTrainer(
        model=model,
        args=build_training_arguments(config, "cpu"),
        train_dataset=rows,
        data_collator=SpeechSeq2SeqCollator(START),
    )
    result = trainer.train()

    assert result.global_step == 2
    assert math.isfinite(result.training_loss)
