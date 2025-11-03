"""Per-example feature extraction and the batch collator used by the trainer."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import soundfile
import torch

LABEL_PAD_ID = -100  # ignored by the cross-entropy loss


def load_audio(path: str, expected_rate: int) -> np.ndarray:
    """Read an audio file as mono float32, refusing a sampling-rate mismatch.

    Feeding 44.1 kHz audio to a 16 kHz feature extractor silently produces
    garbage spectrograms, so a mismatch is an error rather than a warning.
    """
    samples, rate = soundfile.read(path, dtype="float32", always_2d=False)
    if rate != expected_rate:
        raise ValueError(f"{path}: sampled at {rate} Hz, the feature extractor expects {expected_rate} Hz")
    if samples.ndim > 1:
        samples = samples.mean(axis=1)
    return samples


def prepare_example(example: dict, *, feature_extractor: Any, tokenizer: Any, max_label_length: int) -> dict:
    """Turn one manifest row into log-mel ``input_features`` and token ``labels``.

    Labels are left unpadded here; the collator pads each batch to its own
    longest transcript instead of every row to ``max_label_length``.
    """
    rate = feature_extractor.sampling_rate
    audio = load_audio(example["audio"], rate)
    features = feature_extractor(audio, sampling_rate=rate).input_features[0]
    labels = tokenizer(example["text"], truncation=True, max_length=max_label_length).input_ids
    return {"input_features": features, "labels": labels}


@dataclass
class SpeechSeq2SeqCollator:
    """Stack features and pad labels with -100 so padding adds nothing to the loss.

    The tokenizer prepends the decoder start token to every label sequence,
    and the model prepends it again when it shifts labels right to build the
    decoder inputs. It is dropped here so it is not predicted twice.
    """

    decoder_start_token_id: int

    def __call__(self, batch: list[dict]) -> dict[str, torch.Tensor]:
        input_features = torch.stack(
            [torch.as_tensor(np.asarray(row["input_features"]), dtype=torch.float32) for row in batch]
        )

        longest = max(len(row["labels"]) for row in batch)
        labels = torch.full((len(batch), longest), LABEL_PAD_ID, dtype=torch.long)
        for i, row in enumerate(batch):
            labels[i, : len(row["labels"])] = torch.as_tensor(row["labels"], dtype=torch.long)

        if longest and bool((labels[:, 0] == self.decoder_start_token_id).all()):
            labels = labels[:, 1:]
        return {"input_features": input_features, "labels": labels}
