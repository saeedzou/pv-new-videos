"""Diarize a directory tree of mp3 files with pyannote's community-1 pipeline.

For every ``<root>/.../file.mp3`` this writes ``<diarization_dir>/.../file.json``
containing the per-segment speaker turns:

{
    "audio_filepath": "<root>/.../file.mp3",
    "duration": 123.45,
    "segments": [
        {"index": 0, "start": 0.0, "end": 3.42, "speaker": "SPEAKER_00"},
        ...
    ]
}

Example:
    python diarize.py --root /data/mp3s --diarization_dir /data/diarization \
        --token $HF_TOKEN
"""
from __future__ import annotations

import argparse
import logging
import os
from pathlib import Path

from tqdm import tqdm

from utils import audio_duration_seconds, find_audio_files, relative_json_path, save_json

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

PIPELINE_NAME = "pyannote/speaker-diarization-community-1"


def build_pipeline(token: str, device: str):
    import torch
    from pyannote.audio import Pipeline

    pipeline = Pipeline.from_pretrained(PIPELINE_NAME, token=token)
    pipeline.to(torch.device(device))
    return pipeline


def diarize_file(
    pipeline,
    audio_path: Path,
    min_speakers=None,
    max_speakers=None,
    num_speakers=None,
) -> list[dict]:

    import librosa
    import torch

    waveform, sample_rate = librosa.load(str(audio_path), sr=None, mono=True)
    if waveform.ndim == 1:
        waveform = waveform[None, :]

    waveform = torch.from_numpy(waveform).float()

    audio = {
        "waveform": waveform,
        "sample_rate": sample_rate,
    }

    kwargs = {}

    if num_speakers is not None:
        kwargs["num_speakers"] = num_speakers
    else:
        if min_speakers is not None:
            kwargs["min_speakers"] = min_speakers
        if max_speakers is not None:
            kwargs["max_speakers"] = max_speakers

    diarization = pipeline(audio, **kwargs)

    segments = []

    for index, (turn, _, speaker) in enumerate(
        diarization.speaker_diarization.itertracks(yield_label=True)
    ):
        segments.append({
            "index": index,
            "start": round(float(turn.start), 3),
            "end": round(float(turn.end), 3),
            "speaker": str(speaker),
        })

    return segments


def main():
    parser = argparse.ArgumentParser(description="Diarize mp3 files with pyannote community-1.")
    parser.add_argument("--root", type=str, required=True,
                         help="Root directory containing mp3 files (searched recursively).")
    parser.add_argument("--diarization_dir", type=str, required=True,
                         help="Output directory for diarization JSON files.")
    parser.add_argument("--token", type=str, default=os.environ.get("HF_TOKEN"),
                         help="HuggingFace token. Defaults to the HF_TOKEN env var.")
    parser.add_argument("--device", type=str, default=None,
                         help="Torch device, e.g. 'cuda' or 'cpu'. Auto-detected if unset.")
    parser.add_argument("--num_speakers", type=int, default=None)
    parser.add_argument("--min_speakers", type=int, default=None)
    parser.add_argument("--max_speakers", type=int, default=None)
    parser.add_argument("--overwrite", action="store_true",
                         help="Recompute and overwrite existing diarization JSON files.")
    args = parser.parse_args()

    if not args.token:
        raise ValueError("A HuggingFace token is required (pass --token or set HF_TOKEN).")

    device = args.device
    if device is None:
        import torch
        device = "cuda" if torch.cuda.is_available() else "cpu"
    logger.info(f"Using device: {device}")

    root = Path(args.root)
    out_root = Path(args.diarization_dir)
    out_root.mkdir(parents=True, exist_ok=True)

    audio_files = find_audio_files(str(root))
    logger.info(f"Found {len(audio_files)} mp3 file(s) under {root}")

    pipeline = build_pipeline(args.token, device)

    for audio_path in tqdm(audio_files, desc="Diarizing", unit="file"):
        out_path = relative_json_path(audio_path, root, out_root)
        if out_path.exists() and not args.overwrite:
            continue
        try:
            segments = diarize_file(
                pipeline, audio_path,
                min_speakers=args.min_speakers,
                max_speakers=args.max_speakers,
                num_speakers=args.num_speakers,
            )
            duration = audio_duration_seconds(str(audio_path))
            save_json(out_path, {
                "audio_filepath": str(audio_path),
                "duration": round(duration, 3),
                "segments": segments,
            })
        except Exception as e:
            logger.error(f"Failed to diarize {audio_path}: {e}")

    logger.info(f"Diarization JSON files saved to {out_root}")


if __name__ == "__main__":
    main()
