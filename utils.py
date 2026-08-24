"""Shared helpers used across the diarization / language-id / MOS pipeline.

All pipeline stages exchange a common JSON schema, one file per source audio:

{
    "audio_filepath": "/abs/or/relative/path/to/file.mp3",
    "duration": 123.45,                     # total duration of the source file (seconds)
    "segments": [
        {
            "index": 0,
            "start": 0.0,
            "end": 3.42,
            "speaker": "SPEAKER_00",
            # later stages append: "language", "language_prob", "scoreq_nr_mos", ...
        },
        ...
    ],
    # later stages may append file-level metadata such as
    # "dominant_speaker" / "dominant_ratio"
}
"""
from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Iterable, Optional, Tuple

import numpy as np

AUDIO_EXTS = (".mp3", ".wav", ".flac", ".m4a", ".ogg")


def find_audio_files(root: str) -> list[Path]:
    """Recursively find audio files under `root`."""
    root_path = Path(root)
    return sorted(
        p for p in root_path.rglob("*")
        if p.is_file() and p.suffix.lower() == ".mp3"
    )


def find_json_files(root: str) -> list[Path]:
    """Recursively find .json files under `root`."""
    return sorted(Path(root).rglob("*.json"))


def relative_json_path(audio_path: Path, audio_root: Path, out_root: Path) -> Path:
    """Mirror an audio file's relative location under `out_root` as a .json file."""
    rel = audio_path.relative_to(audio_root).with_suffix(".json")
    return out_root / rel


def relative_out_path(in_path: Path, in_root: Path, out_root: Path) -> Path:
    """Mirror a file's relative location from `in_root` under `out_root`."""
    rel = in_path.relative_to(in_root)
    return out_root / rel


def load_json(path: Path) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def save_json(path: Path, data: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


# --------------------------------------------------------------------------
# Audio loading
# --------------------------------------------------------------------------

def load_audio_mono(path: str, sr: int = 16000) -> Tuple[np.ndarray, int]:
    """Load an audio file (mp3/wav/...) as a mono float32 numpy array at `sr` Hz.

    Uses torchaudio (backed by ffmpeg/sox depending on backend) for decoding,
    then downmixes to mono and resamples to `sr` if needed.
    """
    import torchaudio

    waveform, orig_sr = torchaudio.load(path)  # (channels, samples), float32 in [-1, 1]

    if waveform.shape[0] > 1:
        waveform = waveform.mean(dim=0, keepdim=True)  # downmix to mono

    if orig_sr != sr:
        resampler = torchaudio.transforms.Resample(orig_freq=orig_sr, new_freq=sr)
        waveform = resampler(waveform)

    samples = waveform.squeeze(0).numpy().astype(np.float32)
    return samples, sr


def crop_samples(
    samples: np.ndarray,
    sr: int,
    start: float,
    end: float,
    max_duration: Optional[float] = None,
) -> np.ndarray:
    """Return the slice of `samples` covering [start, min(end, start+max_duration))."""
    if max_duration is not None:
        end = min(end, start + max_duration)
    start_idx = max(0, int(round(start * sr)))
    end_idx = max(start_idx, int(round(end * sr)))
    return samples[start_idx:end_idx]


def audio_duration_seconds(path: str) -> float:
    import torchaudio

    info = torchaudio.info(path)
    return info.num_frames / float(info.sample_rate)