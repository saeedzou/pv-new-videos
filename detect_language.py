"""Detect the spoken language of each diarized segment.

Adapted from the user's Whisper-based `detect_language.py`: instead of
reading a flat JSON-lines manifest of `audio_filepath`s, this reads the
diarization pipeline's per-file JSON schema (`audio_filepath` + `segments`)
from `--diarization_dir`, crops each segment to at most `--max_duration`
seconds (default 30.0, matching Whisper's single-window language-ID input),
and attaches `language` / `language_prob` to every segment before saving to
`--language_dir`.

Language detection itself is unchanged from the original: it runs Whisper's
encoder + a single decoder step, masks the logits down to the language
tokens, and softmaxes to get a probability per language. By default all of
Whisper's languages are considered; pass `--languages en fa` to restrict the
softmax (and the argmax) to a fixed subset, exactly like the original script
did for {'en', 'fa'}. `--model_name` accepts either a local model directory
(falling back to the Hugging Face `openai/whisper-large-v3` checkpoint if the
path doesn't exist or fails to load) or a Hugging Face repo id.

Example:
    python detect_language.py --diarization_dir /data/diarization_filtered \
        --language_dir /data/language --languages en fa --max_duration 30.0
"""
from __future__ import annotations

import argparse
import logging
import os
from pathlib import Path
from typing import Optional

import torch
from transformers import WhisperForConditionalGeneration, WhisperProcessor, WhisperTokenizer
from tqdm import tqdm

from utils import find_json_files, load_audio_mono, load_json, relative_out_path, save_json

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


# ----------------------------- #
#         Load Components         #
# ----------------------------- #

def load_model_and_tokenizer(model_name=None):
    """Loads the Whisper model, processor, and tokenizer from a local path or Hugging Face."""
    fallback_model_name = "openai/whisper-large-v3"
    resolved_model_name = model_name or fallback_model_name

    if model_name:
        if os.path.exists(model_name):
            print(f"Loading Whisper model from local path: {model_name}")
        else:
            print(f"Whisper model path '{model_name}' was not found. Falling back to Hugging Face: {fallback_model_name}")
            resolved_model_name = fallback_model_name

    try:
        processor = WhisperProcessor.from_pretrained(resolved_model_name)
        model = WhisperForConditionalGeneration.from_pretrained(
            resolved_model_name,
            torch_dtype=torch.float16,
            attn_implementation="sdpa"
        )
        tokenizer = WhisperTokenizer.from_pretrained(resolved_model_name)
        return processor, model, tokenizer
    except Exception as e:
        if resolved_model_name != fallback_model_name:
            print(f"Failed to load Whisper model from local path '{resolved_model_name}': {e}")
            print(f"Falling back to Hugging Face: {fallback_model_name}")
            return load_model_and_tokenizer(fallback_model_name)
        raise


# ----------------------------- #
#      Waveform helpers         #
# ----------------------------- #

def crop_waveform(waveform: torch.Tensor, sr: int, start: float, end: float,
                   max_duration: Optional[float]) -> torch.Tensor:
    """Slice a 1-D waveform tensor to [start, min(end, start+max_duration))."""
    if max_duration is not None:
        end = min(end, start + max_duration)
    start_idx = max(0, int(round(start * sr)))
    end_idx = max(start_idx, int(round(end * sr)))
    return waveform[start_idx:end_idx]


def pad_waveforms(waveforms: list[torch.Tensor]) -> torch.Tensor:
    """Pads a list of waveforms to the same length and stacks them into a tensor."""
    max_len = max(w.shape[0] for w in waveforms)
    padded_waveforms = [
        torch.nn.functional.pad(w, (0, max_len - w.shape[0])) for w in waveforms
    ]
    return torch.stack(padded_waveforms)


# ----------------------------- #
#      Language Detection       #
# ----------------------------- #

def detect_language(model, tokenizer, input_features, possible_languages=None):
    """Detects the language of the input audio features."""
    # Get all language tokens (e.g., "<|en|>", "<|fa|>")
    all_language_tokens = [t for t in tokenizer.additional_special_tokens if len(t) == 6]

    # Filter language tokens if specific possible_languages are provided
    if possible_languages is not None:
        language_tokens_to_consider = [t for t in all_language_tokens if t[2:-2] in possible_languages]
        if len(language_tokens_to_consider) < len(possible_languages):
            raise RuntimeError(f'Some languages in {possible_languages} did not have associated language tokens')
    else:
        language_tokens_to_consider = all_language_tokens

    language_token_ids = tokenizer.convert_tokens_to_ids(language_tokens_to_consider)
    decoder_input_ids = torch.tensor([[50258]] * input_features.shape[0]).to(input_features.device)  # 50258 is the <|startoflm|> token
    with torch.no_grad():
        logits = model(input_features, decoder_input_ids=decoder_input_ids).logits

    # Mask out non-language tokens
    mask = torch.ones(logits.shape[-1], dtype=torch.bool)
    mask[language_token_ids] = False
    logits[:, :, mask] = -float('inf')

    output_probs = logits.softmax(dim=-1).float().cpu()

    results = []
    for i in range(logits.shape[0]):
        # Map token IDs back to language codes (e.g., "en", "fa")
        lang_probs = {
            lang[2:-2]: output_probs[i, 0, token_id].item()
            for token_id, lang in zip(language_token_ids, language_tokens_to_consider)
        }
        # Find the language with the highest probability
        detected_lang = max(lang_probs, key=lang_probs.get)
        detected_lang_prob = lang_probs[detected_lang]

        results.append({
            "detected_lang": detected_lang,
            "detected_lang_prob": detected_lang_prob,
            "all_lang_probs": lang_probs  # Optionally include all probabilities
        })
    return results


# ----------------------------- #
#            Main               #
# ----------------------------- #

def main():
    parser = argparse.ArgumentParser(
        description="Detect per-segment spoken language with Whisper's language-ID head.")
    parser.add_argument("--diarization_dir", type=str, required=True,
                         help="Directory of diarization JSON files (with audio_filepath).")
    parser.add_argument("--language_dir", type=str, required=True,
                         help="Output directory for language-annotated JSON files.")
    parser.add_argument("--model_name", type=str, default=None,
                         help="Local path or Hugging Face repo id of the Whisper model. "
                              "Falls back to openai/whisper-large-v3 if unset or not found.")
    parser.add_argument("--languages", type=str, nargs="*", default=None,
                         help="Restrict detection to these language codes, e.g. --languages en fa. "
                              "Defaults to all of Whisper's languages.")
    parser.add_argument("--max_duration", type=float, default=30.0,
                         help="Max seconds of each segment fed to the language detector.")
    parser.add_argument("--batch_size", type=int, default=16,
                         help="Number of segments per forward pass (batched within a file).")
    parser.add_argument("--device", type=str, default=None,
                         help="'cuda' or 'cpu'. Auto-detected if unset.")
    parser.add_argument("--save_all_probs", action="store_true",
                         help="Also store the per-language probability dict on each segment "
                              "(as 'language_probs'), not just the top language.")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    device = torch.device(args.device) if args.device else torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )
    logger.info(f"Using device={device}")

    possible_languages = set(args.languages) if args.languages else None

    in_root = Path(args.diarization_dir)
    out_root = Path(args.language_dir)
    out_root.mkdir(parents=True, exist_ok=True)

    json_files = find_json_files(str(in_root))
    logger.info(f"Found {len(json_files)} diarization JSON file(s) under {in_root}")

    processor, model, tokenizer = load_model_and_tokenizer(args.model_name)
    model = model.to(device)
    if device.type == "cpu":
        # float16 is generally unsupported/slow for inference on CPU.
        model = model.float()
    model.eval()

    for json_path in tqdm(json_files, desc="Detecting language", unit="file"):
        out_path = relative_out_path(json_path, in_root, out_root)
        if out_path.exists() and not args.overwrite:
            continue

        try:
            data = load_json(json_path)
            audio_path = data["audio_filepath"]
            samples, sr = load_audio_mono(audio_path)
            waveform_full = torch.from_numpy(samples)

            segments = data.get("segments", [])
            for batch_start in range(0, len(segments), args.batch_size):
                batch_segs = segments[batch_start:batch_start + args.batch_size]
                waveforms = [
                    crop_waveform(waveform_full, sr, float(seg["start"]), float(seg["end"]),
                                  args.max_duration)
                    for seg in batch_segs
                ]

                # Drop empty crops (e.g. degenerate zero-length segments) but keep alignment.
                valid = [(seg, wf) for seg, wf in zip(batch_segs, waveforms) if wf.numel() > 0]
                if not valid:
                    continue
                valid_segs, valid_waveforms = zip(*valid)

                batch_waveform = pad_waveforms(list(valid_waveforms))
                inputs = processor(batch_waveform.numpy(), sampling_rate=sr, return_tensors="pt")
                input_features = inputs.input_features.to(device, dtype=model.dtype)

                results = detect_language(model, tokenizer, input_features, possible_languages)

                for seg, result in zip(valid_segs, results):
                    seg["language"] = result["detected_lang"]
                    seg["language_prob"] = round(float(result["detected_lang_prob"]), 4)
                    if args.save_all_probs:
                        seg["language_probs"] = {
                            lang: round(float(p), 4) for lang, p in result["all_lang_probs"].items()
                        }

            save_json(out_path, data)
        except Exception as e:
            logger.error(f"Failed to process {json_path}: {e}")

    logger.info(f"Language-annotated JSON files saved to {out_root}")


if __name__ == "__main__":
    main()