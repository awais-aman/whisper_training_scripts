"""Turn a transcript table into the JSONL manifest that `train` reads.

Each manifest line is ``{"audio": <path>, "text": <transcript>}``. Rows are
dropped (and reported) when the audio file is missing, the transcript is
empty, or the audio is not at the sampling rate Whisper expects, so problems
surface here instead of hours into a training run.
"""

from __future__ import annotations

import csv
import json
from collections.abc import Iterator
from dataclasses import dataclass, field
from pathlib import Path

import soundfile

WHISPER_SAMPLING_RATE = 16_000


@dataclass
class ManifestReport:
    written: int = 0
    missing_audio: list[str] = field(default_factory=list)
    empty_text: list[str] = field(default_factory=list)
    wrong_sampling_rate: list[tuple[str, int]] = field(default_factory=list)

    @property
    def skipped(self) -> int:
        return len(self.missing_audio) + len(self.empty_text) + len(self.wrong_sampling_rate)


def read_transcripts(path: Path, *, has_header: bool = False) -> Iterator[tuple[str, str]]:
    """Yield ``(audio_path, text)`` pairs from a TSV, CSV, JSON or JSONL file.

    TSV/CSV files hold two columns (audio path, transcript); pass
    ``has_header`` when the first row is a header. JSON files hold a list of
    objects and JSONL files one object per line, each with an ``audio`` (or
    ``audio_path``) key and a ``text`` key.
    """
    suffix = path.suffix.lower()
    if suffix in {".tsv", ".csv"}:
        delimiter = "\t" if suffix == ".tsv" else ","
        with path.open(newline="", encoding="utf-8") as handle:
            rows = csv.reader(handle, delimiter=delimiter)
            if has_header:
                next(rows, None)
            for line_no, row in enumerate(rows, start=2 if has_header else 1):
                if not row:
                    continue
                if len(row) < 2:
                    raise ValueError(f"{path}:{line_no}: expected 2 columns, got {len(row)}")
                yield row[0], row[1]
    elif suffix == ".json":
        records = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(records, list):
            raise ValueError(f"{path}: expected a JSON list of objects")
        yield from (_record_pair(record, path) for record in records)
    elif suffix == ".jsonl":
        with path.open(encoding="utf-8") as handle:
            for line in handle:
                if line.strip():
                    yield _record_pair(json.loads(line), path)
    else:
        raise ValueError(f"{path}: unsupported transcript format {suffix!r} (use .tsv, .csv, .json or .jsonl)")


def _record_pair(record: dict, source: Path) -> tuple[str, str]:
    audio = record.get("audio", record.get("audio_path"))
    if audio is None or "text" not in record:
        raise ValueError(f"{source}: every record needs 'audio' (or 'audio_path') and 'text'")
    return str(audio), str(record["text"])


def build_manifest(
    transcripts: Path,
    output: Path,
    *,
    audio_dir: Path | None = None,
    has_header: bool = False,
    sampling_rate: int = WHISPER_SAMPLING_RATE,
) -> ManifestReport:
    """Write the manifest and return what was kept and skipped.

    With ``audio_dir`` set, only the file name of each audio path is kept and
    looked up in that folder (transcripts often carry paths from another
    machine). Otherwise relative paths resolve against the transcript file's
    folder.
    """
    base = transcripts.parent
    report = ManifestReport()
    output.parent.mkdir(parents=True, exist_ok=True)

    with output.open("w", encoding="utf-8") as out:
        for raw_audio, raw_text in read_transcripts(transcripts, has_header=has_header):
            text = raw_text.strip()
            audio = audio_dir / Path(raw_audio).name if audio_dir else base / raw_audio
            if not text:
                report.empty_text.append(raw_audio)
            elif not audio.is_file():
                report.missing_audio.append(raw_audio)
            elif (rate := soundfile.info(str(audio)).samplerate) != sampling_rate:
                report.wrong_sampling_rate.append((raw_audio, rate))
            else:
                out.write(json.dumps({"audio": str(audio), "text": text}, ensure_ascii=False) + "\n")
                report.written += 1
    return report
