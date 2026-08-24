"""Compute per-segment SCOREQ no-reference MOS scores.

Adapted from the manifest-based `mos.py` example to instead operate on the
diarization/language pipeline's per-file JSON schema: every segment's audio
(capped at `--max_duration` seconds) is scored with `scoreq.Scoreq` and the
result is attached to the segment as `scoreq_nr_mos`.

Reads and (by default) updates JSON files in place under --filtered_dir; pass
--output_dir to instead write to a separate directory.

Example:
    python mos.py --filtered_dir /data/language_filtered --data_domain natural
"""
from __future__ import annotations

import argparse
import logging
import os
import tempfile
from pathlib import Path

import soundfile as sf
from tqdm import tqdm

from utils import crop_samples, find_json_files, load_audio_mono, load_json, relative_out_path, save_json

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


def score_segment(nr, samples, sr, start: float, end: float, max_duration: float) -> float | None:
    audio = crop_samples(samples, sr, start, end, max_duration=max_duration)
    if audio.size == 0:
        return None
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
        tmp_path = tmp.name
    try:
        sf.write(tmp_path, audio, sr)
        return float(nr.predict(test_path=tmp_path))
    finally:
        try:
            os.remove(tmp_path)
        except OSError:
            pass


def main():
    parser = argparse.ArgumentParser(description="Compute per-segment SCOREQ NR-MOS scores.")
    parser.add_argument("--filtered_dir", type=str, required=True,
                         help="Directory of JSON files (with audio_filepath + segments) to score.")
    parser.add_argument("--output_dir", type=str, default=None,
                         help="Where to write scored JSON files. Defaults to updating "
                              "--filtered_dir in place.")
    parser.add_argument("--data_domain", type=str, choices=["natural", "synthetic"],
                         default="natural", help="Which SCOREQ predictor version to use.")
    parser.add_argument("--max_duration", type=float, default=30.0,
                         help="Max seconds of each segment fed to the MOS predictor.")
    parser.add_argument("--overwrite", action="store_true",
                             help="Recompute and overwrite existing MOS JSON files.")
    args = parser.parse_args()

    import scoreq

    nr = scoreq.Scoreq(data_domain=args.data_domain, mode="nr")

    in_root = Path(args.filtered_dir)
    out_root = Path(args.output_dir) if args.output_dir else in_root
    out_root.mkdir(parents=True, exist_ok=True)

    json_files = find_json_files(str(in_root))
    logger.info(f"Found {len(json_files)} JSON file(s) under {in_root}")

    for json_path in tqdm(json_files, desc="Scoring MOS", unit="file"):
        try:
            data = load_json(json_path)
            audio_path = data["audio_filepath"]
            if not os.path.exists(audio_path):
                logger.warning(f"Audio file not found: {audio_path}. Skipping {json_path}.")
                continue

            out_path = relative_out_path(json_path, in_root, out_root)
            if out_path.exists() and not args.overwrite:
                continue
            samples, sr = load_audio_mono(audio_path)

            for seg in data.get("segments", []):
                try:
                    mos = score_segment(
                        nr, samples, sr, float(seg["start"]), float(seg["end"]),
                        max_duration=args.max_duration,
                    )
                    seg["scoreq_nr_mos"] = round(mos, 4) if mos is not None else None
                except Exception as e:
                    logger.error(f"Error scoring segment {seg.get('index')} in {json_path}: {e}")
                    seg["mos_error"] = str(e)

            save_json(out_path, data)
        except Exception as e:
            logger.error(f"Failed to process {json_path}: {e}")

    logger.info(f"Scored JSON files saved to {out_root}")


if __name__ == "__main__":
    main()
