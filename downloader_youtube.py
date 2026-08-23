import argparse
import csv
import functools
import logging
import multiprocessing as mp
import os
import sys
import time
from datetime import datetime, timezone

import yt_dlp

logging.basicConfig(level=logging.INFO)

try:
    from huggingface_hub import HfApi, create_repo, hf_hub_download
    from huggingface_hub.utils import EntryNotFoundError, RepositoryNotFoundError
    HF_AVAILABLE = True
except ImportError:
    HF_AVAILABLE = False


parser = argparse.ArgumentParser(
    formatter_class=argparse.ArgumentDefaultsHelpFormatter
)
parser.add_argument(
    '--input_csv',
    required=True,
    type=str,
    help='local CSV with a video_id column listing the videos to download',
)
parser.add_argument(
    '--output_csv',
    default='download_progress.csv',
    type=str,
    help='local path to write the progress CSV; also the name it is pushed '
         'under (repo root) to --hf_repo_id, and fetched from there on startup '
         'to resume',
)
parser.add_argument(
    '--base_dir',
    default='audio',
    type=str,
    help='where to save downloaded mp3 files locally before they are pushed',
)
parser.add_argument(
    '--sample_rate',
    default=24000,
    type=int,
    help='mp3 sample rate (Hz) to resample audio to',
)
parser.add_argument(
    '--cookies_file',
    default='cookies.txt',
    type=str,
    help='path to a cookies.txt file for yt-dlp (used only if it exists)',
)
parser.add_argument(
    '--num_workers',
    default=1,
    type=int,
    help='multi-process workers for the download process',
)
parser.add_argument(
    '--save_every',
    default=1000,
    type=int,
    help='checkpoint progress after this many newly-updated videos',
)
parser.add_argument(
    '--save_interval_sec',
    default=7200,
    type=int,
    help='also checkpoint if this many seconds elapsed since the last checkpoint',
)
parser.add_argument(
    '--max_video_attempts',
    default=15,
    type=int,
    help='skip a video after this many download attempts',
)
parser.add_argument(
    '--max_runtime_sec',
    default=7200,
    type=int,
    help='stop launching new work after this many seconds and exit cleanly '
         '(checkpoint + push still run)',
)
parser.add_argument(
    '--hf_repo_id',
    required=True,
    type=str,
    help='HF dataset repo that both the raw mp3 files and the progress CSV '
         'get pushed to (and the progress CSV is fetched from, at startup)',
)
parser.add_argument(
    '--repo_audio_subdir',
    default='audio',
    type=str,
    help='sub-directory inside the HF repo that mp3 files are uploaded to',
)
parser.add_argument(
    '--no_push',
    action='store_true',
    help='disable all Hugging Face Hub pushes (progress is still saved locally)',
)
args = parser.parse_args()

PROGRESS_LOCAL_PATH = args.output_csv
PROGRESS_FIELDS = ['video_id', 'status', 'attempts', 'last_attempt', 'last_error', 'uploaded_at']

VIDEO_STATUSES = {'pending', 'downloaded', 'uploaded', 'failed', 'unavailable'}
BOT_CHECK_MARKERS = ("Sign in to confirm you’re not a bot",)
AGE_RESTRICTED_MARKERS = ('Sign in to confirm your age', 'This video is age-restricted')

UNAVAILABLE_MARKERS = (
    'Video unavailable',
    'This video is not available',
    'This video is unavailable',
    'This video is no longer available',
    'has been removed by the uploader',
    'account associated with this video has been terminated',
    'is not available in your country',
    'Private video',
    'video is private',
    'This video is private',
    'This video has been removed',
    'Join this channel',
    'This video is available to this',
)
COPYRIGHT_MARKERS = (
    'copyright claim',
    'copyright grounds',
    'It was removed following a copyright removal',
    'This video was removed due to a counterfeit claim',
    'It was blocked due to the claimed content',
)
GEO_BLOCKED_MARKERS = (
    'not made this video available in your country',
)
AUTH_REQUIRED_MARKERS = ('Please sign in',)

PERMANENT_ERROR_TYPES = {'age_restricted', 'unavailable', 'copyright', 'geo_blocked'}


def classify_error(msg):
    """Map a raw yt-dlp/error string to a coarse error_type."""
    if not msg:
        return 'other'
    if any(marker in msg for marker in BOT_CHECK_MARKERS):
        return 'bot_check'
    if any(marker in msg for marker in AGE_RESTRICTED_MARKERS):
        return 'age_restricted'
    if any(marker in msg for marker in COPYRIGHT_MARKERS):
        return 'copyright'
    if any(marker in msg for marker in GEO_BLOCKED_MARKERS):
        return 'geo_blocked'
    if any(marker in msg for marker in UNAVAILABLE_MARKERS):
        return 'unavailable'
    if any(marker in msg for marker in AUTH_REQUIRED_MARKERS):
        return 'auth_required'
    return 'other'


# ----------------------------------------------------------------------------
# Download jobs (run inside worker processes)
# ----------------------------------------------------------------------------
class _CapturingLogger:
    """yt-dlp logger passed via ydl_opts.

    With ignoreerrors=True, yt-dlp routes extractor failures only to the
    logger instead of raising. We stash the last error line here so the
    caller can classify it (bot-check, age-restricted, unavailable, ...)
    and store the real message instead of a generic exit-code string.
    """

    def __init__(self, stop_event):
        self.stop_event = stop_event
        self.last_error = None

    def debug(self, msg):
        pass

    def warning(self, msg):
        logging.warning(msg)

    def error(self, msg):
        logging.error(msg)
        # With verbose=True, yt-dlp calls logger.error() twice per failure:
        # once with the real message ("ERROR: [youtube] ...: Private video...")
        # and once more with a raw traceback dump. Keep the first real
        # message; ignore traceback-looking follow-ups.
        if self.last_error is not None and _looks_like_traceback(msg):
            return
        self.last_error = msg
        if any(marker in msg for marker in BOT_CHECK_MARKERS):
            self.stop_event.set()


def _looks_like_traceback(msg):
    stripped = msg.lstrip()
    return stripped.startswith('File "') or stripped.startswith('Traceback')


def utc_now():
    return datetime.now(timezone.utc).isoformat()


def build_ydl_opts(base_dir, sample_rate, cookies_file, logger):
    opts = {
        'format': 'bestaudio/best',
        'outtmpl': os.path.join(base_dir, '%(id)s.%(ext)s'),
        'noplaylist': True,
        'ignoreerrors': True,
        'max_sleep_interval': 0.2,
        'verbose': True,
        'quiet': True,
        'extractor_args': {
            'youtube': {'player_client': ['default', 'tv_downgraded', 'web_embedded']}
        },
        'logger': logger,
        # No download_ranges here -- we want the full audio, not a clip.
        'postprocessors': [{
            'key': 'FFmpegExtractAudio',
            'preferredcodec': 'mp3',
            'preferredquality': '2',
        }],
        'postprocessor_args': [
            '-ar', str(sample_rate),
        ],
        'prefer_ffmpeg': True,
    }
    if cookies_file and os.path.exists(cookies_file):
        opts['cookies'] = cookies_file
    return opts


def download_single_audio(video_id, base_dir, sample_rate, cookies_file, stop_event):
    os.makedirs(base_dir, exist_ok=True)
    logging.info(f'[{video_id}] starting audio download attempt')
    capturing_logger = _CapturingLogger(stop_event)
    try:
        ydl_opts = build_ydl_opts(base_dir, sample_rate, cookies_file, capturing_logger)
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            error_code = ydl.download([video_id])

        if stop_event.is_set():
            return {
                'ok': False,
                'error': capturing_logger.last_error,
                'error_type': 'bot_check',
            }

        if error_code == 0:
            logging.info(f'[{video_id}] download succeeded')
            return {
                'ok': True,
                'error': None,
                'error_type': None,
            }

        # Prefer the real message the logger captured; fall back to the
        # generic exit-code string only if yt-dlp never logged anything.
        error_text = capturing_logger.last_error or f'yt-dlp returned exit code {error_code}'
        error_type = classify_error(error_text)
        logging.warning(f'[{video_id}] download failed ({error_type}): {error_text}')
        return {
            'ok': False,
            'error': error_text,
            'error_type': error_type,
        }
    except Exception as exc:
        error_text = str(exc)
        error_type = classify_error(error_text)
        if error_type == 'bot_check':
            stop_event.set()
        logging.exception(f'[{video_id}] download job raised an exception')
        return {
            'ok': False,
            'error': error_text,
            'error_type': error_type,
        }


def process_video(item, base_dir, sample_rate, cookies_file, stop_event, max_video_attempts):
    """Runs inside a worker process for a single video."""
    video_id, video_state = item
    attempts = int((video_state or {}).get('attempts', 0) or 0)
    status = (video_state or {}).get('status', 'pending')

    if status in ('uploaded', 'unavailable') or attempts >= max_video_attempts:
        logging.info(f'[{video_id}] skipping ({status}, attempts={attempts})')
        return {
            'video_id': video_id,
            'skipped': True,
            'ok': False,
            'error': None,
            'error_type': None,
            'attempts': attempts,
        }

    if stop_event.is_set():
        return {
            'video_id': video_id,
            'skipped': True,
            'ok': False,
            'error': None,
            'error_type': None,
            'attempts': attempts,
        }

    result = download_single_audio(video_id, base_dir, sample_rate, cookies_file, stop_event)
    result.update({
        'video_id': video_id,
        'skipped': False,
        'attempts': attempts + 1,
    })
    return result


# ----------------------------------------------------------------------------
# Progress + Hugging Face Hub sync (all called from the MAIN process only)
# ----------------------------------------------------------------------------
def new_video_state(status='pending', attempts=0, last_attempt=None, last_error=None, uploaded_at=None):
    return {
        'status': status,
        'attempts': attempts,
        'last_attempt': last_attempt,
        'last_error': last_error,
        'uploaded_at': uploaded_at,
    }


def sanitize_video_state(state):
    if not isinstance(state, dict):
        return new_video_state()

    status = state.get('status', 'pending')
    if status not in VIDEO_STATUSES:
        status = 'pending'

    attempts = state.get('attempts', 0)
    try:
        attempts = int(attempts or 0)
    except Exception:
        attempts = 0

    return new_video_state(
        status=status,
        attempts=attempts,
        last_attempt=state.get('last_attempt') or None,
        last_error=state.get('last_error') or None,
        uploaded_at=state.get('uploaded_at') or None,
    )


def should_retry_video(video_state, max_video_attempts):
    if not isinstance(video_state, dict):
        return True
    if video_state.get('status') in ('uploaded', 'unavailable'):
        return False
    attempts = int(video_state.get('attempts', 0) or 0)
    return attempts < max_video_attempts


def read_input_video_ids(path):
    if not os.path.exists(path):
        logging.error(f'--input_csv not found: {path}')
        sys.exit(1)

    video_ids = []
    seen = set()
    with open(path, newline='', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        if reader.fieldnames is None or 'video_id' not in reader.fieldnames:
            logging.error(f"--input_csv must have a 'video_id' column: {path}")
            sys.exit(1)
        for row in reader:
            video_id = (row.get('video_id') or '').strip()
            if video_id and video_id not in seen:
                seen.add(video_id)
                video_ids.append(video_id)
    return video_ids


def fetch_remote_progress_csv(repo_id, filename, local_path):
    """Best-effort: fetch the progress CSV already in the repo so we resume
    instead of restarting from scratch. Any failure (repo/file doesn't exist
    yet, no network, no permissions) just means we start fresh -- it is not
    fatal."""
    if not HF_AVAILABLE:
        logging.info('huggingface_hub not installed; starting with empty progress.')
        return None
    try:
        downloaded_path = hf_hub_download(
            repo_id=repo_id,
            filename=filename,
            repo_type='dataset',
            token=os.environ.get('HF_TOKEN'),
        )
        os.makedirs(os.path.dirname(local_path) or '.', exist_ok=True)
        with open(downloaded_path, 'rb') as src, open(local_path, 'wb') as dst:
            dst.write(src.read())
        logging.info(f'Fetched existing progress CSV from {repo_id}:{filename}')
        return local_path
    except (EntryNotFoundError, RepositoryNotFoundError):
        logging.info(f'No existing progress CSV at {repo_id}:{filename}; starting fresh.')
        return None
    except Exception as exc:
        logging.warning(f'Could not fetch progress CSV from {repo_id}:{filename}: {exc}')
        return None


def load_progress_csv(path):
    progress = {}
    if not path or not os.path.exists(path):
        return progress
    try:
        with open(path, newline='', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            for row in reader:
                video_id = (row.get('video_id') or '').strip()
                if not video_id:
                    continue
                progress[video_id] = sanitize_video_state(row)
    except Exception as exc:
        logging.warning(f'Could not parse existing progress CSV, starting fresh: {exc}')
        return {}
    return progress


def normalize_progress(progress, video_ids):
    normalized = {}
    for video_id in video_ids:
        normalized[video_id] = sanitize_video_state(progress.get(video_id))
    return normalized


def save_progress_local(progress, path):
    os.makedirs(os.path.dirname(path) or '.', exist_ok=True)
    tmp_path = path + '.tmp'
    with open(tmp_path, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=PROGRESS_FIELDS)
        writer.writeheader()
        for video_id, state in sorted(progress.items()):
            writer.writerow({
                'video_id': video_id,
                'status': state.get('status'),
                'attempts': state.get('attempts', 0),
                'last_attempt': state.get('last_attempt') or '',
                'last_error': state.get('last_error') or '',
                'uploaded_at': state.get('uploaded_at') or '',
            })
    os.replace(tmp_path, path)


def build_eta_commit_message(progress):
    uploaded = 0
    remaining = 0
    for state in progress.values():
        status = state.get('status')
        if status == 'uploaded':
            uploaded += 1
        elif status != 'unavailable':
            remaining += 1

    elapsed = max(1, time.time() - START_TIME)
    rate = uploaded / elapsed  # videos/sec

    if rate > 0 and remaining > 0:
        eta_seconds = remaining / rate
        eta = time.strftime("%H:%M:%S", time.gmtime(eta_seconds))
    else:
        eta = "unknown"

    return f"Progress: {uploaded}/{uploaded + remaining} uploaded (ETA {eta})"


def push_progress_file(api, local_path, repo_id, progress):
    try:
        api.upload_file(
            path_or_fileobj=local_path,
            path_in_repo=os.path.basename(local_path),
            repo_id=repo_id,
            repo_type='dataset',
            commit_message=build_eta_commit_message(progress),
        )
        logging.info(f'Pushed progress CSV to {repo_id}:{os.path.basename(local_path)}')
    except Exception as exc:
        logging.warning(f'Could not push progress CSV to {repo_id}: {exc}')


def push_audio_files(api, base_dir, repo_id, repo_audio_subdir):
    """Upload each mp3 currently on disk as a raw file (no shard/tar), then
    remove it locally. Returns the list of video_ids that were uploaded
    successfully."""
    try:
        create_repo(repo_id, repo_type='dataset', exist_ok=True, token=os.environ.get('HF_TOKEN'))
    except Exception as exc:
        logging.warning(f'create_repo({repo_id}) failed (may already exist / no perms): {exc}')

    if not os.path.isdir(base_dir):
        return []

    mp3_files = sorted(f for f in os.listdir(base_dir) if f.lower().endswith('.mp3'))
    if not mp3_files:
        return []

    uploaded_video_ids = []
    for fname in mp3_files:
        local_path = os.path.join(base_dir, fname)
        video_id = os.path.splitext(fname)[0]
        path_in_repo = f'{repo_audio_subdir}/{fname}' if repo_audio_subdir else fname
        try:
            api.upload_file(
                path_or_fileobj=local_path,
                path_in_repo=path_in_repo,
                repo_id=repo_id,
                repo_type='dataset',
                token=os.environ.get('HF_TOKEN'),
                commit_message=f'Add {fname}',
            )
            os.remove(local_path)
            uploaded_video_ids.append(video_id)
            logging.info(f'Uploaded {fname} to {repo_id}:{path_in_repo}')
        except Exception as exc:
            logging.warning(
                f'Could not upload {fname} to {repo_id} — it will be retried '
                f'next checkpoint since it was not marked complete: {exc}'
            )

    return uploaded_video_ids


def mark_uploaded(progress, video_ids):
    uploaded_at = utc_now()
    for video_id in video_ids:
        state = progress.setdefault(video_id, new_video_state())
        state['status'] = 'uploaded'
        state['uploaded_at'] = uploaded_at
        state['last_error'] = None


# ----------------------------------------------------------------------------
# Main
# ----------------------------------------------------------------------------
if __name__ == '__main__':
    START_TIME = time.time()
    print('*' * 15)
    print('* Download Starts *')
    print('*' * 15)

    assert args.max_video_attempts >= 1, '--max_video_attempts must be >= 1'

    os.makedirs(args.base_dir, exist_ok=True)

    video_ids = read_input_video_ids(args.input_csv)
    logging.info(f'{len(video_ids)} unique video_id(s) found in {args.input_csv}')

    hf_token = os.environ.get('HF_TOKEN')
    push_enabled = (not args.no_push) and HF_AVAILABLE and bool(hf_token)
    if args.no_push:
        logging.info('--no_push set: progress will only be saved locally.')
    elif not HF_AVAILABLE:
        logging.warning('huggingface_hub is not installed; progress will only be saved locally. '
                        'Run `pip install huggingface_hub` to enable pushes.')
    elif not hf_token:
        logging.warning('HF_TOKEN not set in the environment; progress will only be saved locally.')

    api = HfApi(token=hf_token) if push_enabled else None

    # Fetch existing progress from the repo (if any) so we resume instead of
    # re-downloading everything.
    progress = {}
    if push_enabled:
        fetched_path = fetch_remote_progress_csv(
            args.hf_repo_id, os.path.basename(args.output_csv), PROGRESS_LOCAL_PATH
        )
        if fetched_path:
            progress = load_progress_csv(fetched_path)
    elif os.path.exists(PROGRESS_LOCAL_PATH):
        # Offline / no-push mode: fall back to whatever is on disk locally.
        progress = load_progress_csv(PROGRESS_LOCAL_PATH)

    progress = normalize_progress(progress, video_ids)

    remaining_video_ids = [
        vid for vid in video_ids
        if should_retry_video(progress.get(vid), args.max_video_attempts)
    ]
    logging.info(
        f'{len(remaining_video_ids)}/{len(video_ids)} videos remain retryable '
        f'({len(video_ids) - len(remaining_video_ids)} already uploaded or capped)'
    )

    def checkpoint():
        """Persist progress locally first, then sync the repo-confirmed state to HF."""
        save_progress_local(progress, PROGRESS_LOCAL_PATH)
        if push_enabled:
            uploaded_video_ids = push_audio_files(api, args.base_dir, args.hf_repo_id, args.repo_audio_subdir)
            if uploaded_video_ids:
                mark_uploaded(progress, uploaded_video_ids)
                save_progress_local(progress, PROGRESS_LOCAL_PATH)
            push_progress_file(api, PROGRESS_LOCAL_PATH, args.hf_repo_id, progress)

    items = [(vid, progress.get(vid)) for vid in remaining_video_ids]
    workers = max(1, min(args.num_workers, len(items))) if items else 0

    updated_videos_since_checkpoint = 0
    last_checkpoint_time = time.time()

    manager = mp.Manager()
    stop_event = manager.Event()

    worker_fn = functools.partial(
        process_video,
        base_dir=args.base_dir,
        sample_rate=args.sample_rate,
        cookies_file=args.cookies_file,
        stop_event=stop_event,
        max_video_attempts=args.max_video_attempts,
    )

    bot_check_hit = False
    time_limit_hit = False
    try:
        if items:
            with mp.Pool(processes=workers) as pool:
                for result in pool.imap_unordered(worker_fn, items, chunksize=1):
                    if result.get('skipped'):
                        continue

                    video_id = result['video_id']
                    video_state = progress.setdefault(video_id, new_video_state())
                    video_state['last_attempt'] = utc_now()
                    error_type = result.get('error_type')

                    if error_type != 'bot_check':
                        video_state['attempts'] = int(result.get('attempts', video_state.get('attempts', 0)) or 0)
                        video_state['last_error'] = result.get('error')

                    if result.get('ok'):
                        video_state['status'] = 'downloaded'
                        video_state['last_error'] = None
                    elif error_type in PERMANENT_ERROR_TYPES:
                        # Age-restricted / removed / copyright / geo-blocked --
                        # retrying will never succeed, so stop trying now.
                        video_state['status'] = 'unavailable'
                    elif error_type != 'bot_check':
                        video_state['status'] = 'failed'

                    updated_videos_since_checkpoint += 1

                    if error_type == 'bot_check':
                        logging.error(
                            "YouTube bot-check ('Sign in to confirm you're not a bot') detected -- "
                            'stopping early to avoid further requests.'
                        )
                        bot_check_hit = True
                        pool.terminate()
                        break

                    elapsed = time.time() - last_checkpoint_time
                    if (
                        updated_videos_since_checkpoint >= args.save_every
                        or elapsed >= args.save_interval_sec
                    ):
                        logging.info(
                            f'Checkpoint: progress updated for {updated_videos_since_checkpoint} video(s)'
                        )
                        checkpoint()
                        updated_videos_since_checkpoint = 0
                        last_checkpoint_time = time.time()

                    if time.time() - START_TIME >= args.max_runtime_sec:
                        logging.info(
                            f'Max runtime of {args.max_runtime_sec}s reached -- '
                            'stopping early to checkpoint and exit cleanly.'
                        )
                        time_limit_hit = True
                        pool.terminate()
                        break
        else:
            logging.info('Nothing left to download -- all videos are already uploaded or capped.')
    finally:
        logging.info('Finalizing: saving and pushing any remaining progress')
        checkpoint()

    if bot_check_hit:
        print('*' * 15)
        print('* Stopped: YouTube bot-check triggered *')
        print('*' * 15)
        sys.exit(2)

    if time_limit_hit:
        print('*' * 15)
        print('* Stopped: max runtime reached *')
        print('*' * 15)
        sys.exit(3)

    print('*' * 15)
    print('* Download Complete *')
    print('*' * 15)
