"""Filter language-annotated JSON files by target-language segment ratio.

A file is kept only if more than ``--min_ratio`` of its segments are
confidently (``language_prob >= --min_prob``) predicted as ``--language``.

Example:
    python post_detect_language.py --language_dir /data/language \
        --filtered_dir /data/language_filtered --language fa \
        --min_ratio 0.7 --min_prob 0.85
"""
from __future__ import annotations

import argparse
import logging
from pathlib import Path

from tqdm import tqdm

from utils import find_json_files, load_json, relative_out_path, save_json

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


def target_language_ratio(segments: list[dict], language: str, min_prob: float) -> tuple[float, int]:
    if not segments:
        return 0.0, 0
    matches = sum(
        1 for seg in segments
        if seg.get("language") == language and (seg.get("language_prob") or 0.0) >= min_prob
    )
    return matches / len(segments), len(segments)


def main():
    parser = argparse.ArgumentParser(
        description="Keep language-annotated files dominated by a target language.")
    parser.add_argument("--language_dir", type=str, required=True)
    parser.add_argument("--filtered_dir", type=str, required=True)
    parser.add_argument("--language", type=str, default="fa",
                         help="Target language code to require (default: fa).")
    parser.add_argument("--min_ratio", type=float, default=0.7,
                         help="Minimum fraction of segments that must match --language.")
    parser.add_argument("--min_prob", type=float, default=0.85,
                         help="Minimum language_prob for a segment to count as a match.")
    args = parser.parse_args()

    in_root = Path(args.language_dir)
    out_root = Path(args.filtered_dir)
    out_root.mkdir(parents=True, exist_ok=True)

    json_files = find_json_files(str(in_root))
    logger.info(f"Found {len(json_files)} language JSON file(s) under {in_root}")

    kept = 0
    for json_path in tqdm(json_files, desc="Filtering by language", unit="file"):
        try:
            data = load_json(json_path)
        except Exception as e:
            logger.error(f"Failed to read {json_path}: {e}")
            continue

        segments = data.get("segments", [])
        ratio, n_segments = target_language_ratio(segments, args.language, args.min_prob)

        if n_segments == 0 or ratio <= args.min_ratio:
            continue

        data["language_ratio"] = round(ratio, 4)
        data["target_language"] = args.language

        out_path = relative_out_path(json_path, in_root, out_root)
        save_json(out_path, data)
        kept += 1

    logger.info(f"Kept {kept}/{len(json_files)} file(s) "
                f"(language={args.language}, min_ratio={args.min_ratio}, "
                f"min_prob={args.min_prob}) -> {out_root}")


if __name__ == "__main__":
    main()
