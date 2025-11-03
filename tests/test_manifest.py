import json
from pathlib import Path

import numpy as np
import pytest
import soundfile

from whisper_finetune.manifest import build_manifest, read_transcripts


def write_wav(path: Path, rate: int = 16_000, seconds: float = 0.1) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    soundfile.write(str(path), np.zeros(int(rate * seconds), dtype="float32"), rate)


def read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]


def test_tsv_rows_are_matched_by_file_name_in_audio_dir(tmp_path):
    audio_dir = tmp_path / "audio"
    write_wav(audio_dir / "a.wav")
    write_wav(audio_dir / "b.wav")
    tsv = tmp_path / "combined.tsv"
    tsv.write_text("/elsewhere/a.wav\thello world\nclips/b.wav\t  second line \n", encoding="utf-8")
    out = tmp_path / "out" / "manifest.jsonl"

    report = build_manifest(tsv, out, audio_dir=audio_dir)

    assert report.written == 2 and report.skipped == 0
    assert read_jsonl(out) == [
        {"audio": str(audio_dir / "a.wav"), "text": "hello world"},
        {"audio": str(audio_dir / "b.wav"), "text": "second line"},
    ]


def test_bad_rows_are_skipped_and_reported(tmp_path):
    write_wav(tmp_path / "ok.wav")
    write_wav(tmp_path / "silent.wav")
    write_wav(tmp_path / "cd.wav", rate=44_100)
    csv_file = tmp_path / "t.csv"
    csv_file.write_text("audio,text\nok.wav,fine\nsilent.wav,\ngone.wav,lost\ncd.wav,too fast\n", encoding="utf-8")
    out = tmp_path / "m.jsonl"

    report = build_manifest(csv_file, out, has_header=True)

    assert report.written == 1
    assert report.empty_text == ["silent.wav"]
    assert report.missing_audio == ["gone.wav"]
    assert report.wrong_sampling_rate == [("cd.wav", 44_100)]
    assert [row["text"] for row in read_jsonl(out)] == ["fine"]


def test_non_ascii_transcripts_are_kept_verbatim(tmp_path):
    write_wav(tmp_path / "u.wav")
    source = tmp_path / "t.jsonl"
    source.write_text(json.dumps({"audio_path": "u.wav", "text": "آپ کیسے ہیں"}) + "\n", encoding="utf-8")
    out = tmp_path / "m.jsonl"

    build_manifest(source, out)

    assert "آپ کیسے ہیں" in out.read_text(encoding="utf-8")


def test_json_list_input(tmp_path):
    source = tmp_path / "t.json"
    source.write_text(json.dumps([{"audio": "x.wav", "text": "one"}, {"audio_path": "y.wav", "text": "two"}]))
    assert list(read_transcripts(source)) == [("x.wav", "one"), ("y.wav", "two")]


@pytest.mark.parametrize(
    ("name", "content", "message"),
    [
        ("t.txt", "", "unsupported transcript format"),
        ("t.tsv", "only-one-column\n", "expected 2 columns"),
        ("t.json", '{"audio": "a.wav"}', "JSON list"),
        ("t.jsonl", '{"audio": "a.wav"}\n', "'text'"),
    ],
)
def test_malformed_input_is_rejected(tmp_path, name, content, message):
    source = tmp_path / name
    source.write_text(content)
    with pytest.raises(ValueError, match=message):
        list(read_transcripts(source))
