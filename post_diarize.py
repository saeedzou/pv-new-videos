"""Filter diarization JSON files by dominant-speaker duration ratio.

A file is kept only if its most common speaker (by summed segment duration)
accounts for at least ``--min_ratio`` of the total segments duration in that
file. Kept files are copied (with the dominant-speaker stats attached) to
``--filtered_dir``, mirroring the input's relative directory structure.

Example:
    python post_diarize.py --diarization_dir /data/diarization \
        --filtered_dir /data/diarization_filtered --min_ratio 0.7
"""
from __future__ import annotations

import argparse
import logging
from collections import defaultdict
from pathlib import Path

from tqdm import tqdm

from utils import find_json_files, load_json, relative_out_path, save_json

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


def dominant_speaker_ratio(segments: list[dict]) -> tuple[str | None, float, float]:
    """Return (dominant_speaker, ratio, total_duration) for a list of segments."""
    speaker_duration = defaultdict(float)
    total_duration = 0.0
    for seg in segments:
        dur = max(0.0, float(seg["end"]) - float(seg["start"]))
        speaker_duration[seg["speaker"]] += dur
        total_duration += dur

    if total_duration == 0.0 or not speaker_duration:
        return None, 0.0, 0.0

    dominant_speaker = max(speaker_duration, key=speaker_duration.get)
    ratio = speaker_duration[dominant_speaker] / total_duration
    return dominant_speaker, ratio, total_duration


def main():
    parser = argparse.ArgumentParser(
        description="Keep diarization files whose dominant speaker exceeds min_ratio.")
    parser.add_argument("--diarization_dir", type=str, required=True)
    parser.add_argument("--filtered_dir", type=str, required=True)
    parser.add_argument("--min_ratio", type=float, default=0.7,
                         help="Minimum fraction of total segment duration the "
                              "most common speaker must occupy to keep the file.")
    args = parser.parse_args()

    in_root = Path(args.diarization_dir)
    out_root = Path(args.filtered_dir)
    out_root.mkdir(parents=True, exist_ok=True)

    json_files = find_json_files(str(in_root))
    logger.info(f"Found {len(json_files)} diarization JSON file(s) under {in_root}")

    kept = 0
    for json_path in tqdm(json_files, desc="Filtering by dominant speaker", unit="file"):
        try:
            data = load_json(json_path)
        except Exception as e:
            logger.error(f"Failed to read {json_path}: {e}")
            continue

        segments = data.get("segments", [])
        dominant_speaker, ratio, total_duration = dominant_speaker_ratio(segments)

        if dominant_speaker is None or ratio < args.min_ratio:
            continue

        data["dominant_speaker"] = dominant_speaker
        data["dominant_ratio"] = round(ratio, 4)
        data["segments_total_duration"] = round(total_duration, 3)

        out_path = relative_out_path(json_path, in_root, out_root)
        save_json(out_path, data)
        kept += 1

    logger.info(f"Kept {kept}/{len(json_files)} file(s) (min_ratio={args.min_ratio}) "
                f"-> {out_root}")


if __name__ == "__main__":
    main()
