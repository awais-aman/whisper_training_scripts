from types import SimpleNamespace

import numpy as np
import pytest
import soundfile
import torch

from whisper_finetune.features import LABEL_PAD_ID, SpeechSeq2SeqCollator, load_audio, prepare_example

START = 50258  # Whisper's <|startoftranscript|>


class FakeFeatureExtractor:
    """Mimics WhisperFeatureExtractor: fixed-size features, records what it was given."""

    sampling_rate = 16_000

    def __init__(self):
        self.calls = []

    def __call__(self, audio, sampling_rate):
        self.calls.append((audio, sampling_rate))
        return SimpleNamespace(input_features=[np.full((4, 6), float(len(audio)), dtype=np.float32)])


class FakeTokenizer:
    """One token per character after the start token, honouring truncation."""

    def __call__(self, text, truncation, max_length):
        ids = [START] + [ord(c) for c in text]
        return SimpleNamespace(input_ids=ids[:max_length] if truncation else ids)


def test_collator_pads_labels_with_ignore_index_and_drops_start_token():
    batch = [
        {"input_features": np.ones((4, 6)), "labels": [START, 1, 2, 3]},
        {"input_features": np.zeros((4, 6)), "labels": [START, 7]},
    ]

    out = SpeechSeq2SeqCollator(decoder_start_token_id=START)(batch)

    assert out["input_features"].shape == (2, 4, 6)
    assert out["input_features"].dtype == torch.float32
    assert out["labels"].tolist() == [[1, 2, 3], [7, LABEL_PAD_ID, LABEL_PAD_ID]]


def test_collator_keeps_first_token_when_it_is_not_the_start_token():
    batch = [
        {"input_features": np.ones((4, 6)), "labels": [5, 6]},
        {"input_features": np.ones((4, 6)), "labels": [START]},
    ]

    out = SpeechSeq2SeqCollator(decoder_start_token_id=START)(batch)

    assert out["labels"].tolist() == [[5, 6], [START, LABEL_PAD_ID]]


def test_prepare_example_uses_extractor_rate_and_truncates_labels(tmp_path):
    wav = tmp_path / "a.wav"
    soundfile.write(str(wav), np.zeros(1600, dtype="float32"), 16_000)
    extractor = FakeFeatureExtractor()

    row = prepare_example(
        {"audio": str(wav), "text": "hello"}, feature_extractor=extractor, tokenizer=FakeTokenizer(), max_label_length=3
    )

    assert extractor.calls[0][1] == 16_000
    assert row["labels"] == [START, ord("h"), ord("e")]
    assert row["input_features"].shape == (4, 6)


def test_load_audio_downmixes_stereo(tmp_path):
    wav = tmp_path / "stereo.wav"
    stereo = np.stack([np.full(100, 0.5), np.full(100, -0.5)], axis=1).astype("float32")
    soundfile.write(str(wav), stereo, 16_000)

    samples = load_audio(str(wav), 16_000)

    assert samples.ndim == 1
    assert np.allclose(samples, 0.0)


def test_load_audio_rejects_wrong_sampling_rate(tmp_path):
    wav = tmp_path / "cd.wav"
    soundfile.write(str(wav), np.zeros(441, dtype="float32"), 44_100)
    with pytest.raises(ValueError, match="44100 Hz"):
        load_audio(str(wav), 16_000)
