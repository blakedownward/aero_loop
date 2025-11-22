"""Processor: trims annotated WAVs and moves to processed splits.

Reads annotations from data/raw/<batch>/annotations.json for batches
that have been marked complete via a .processed file. For each entry,
trims the clip to [trim_start_s, trim_end_s] and writes into
data/processed/aircraft or data/processed/negative.

Each processed sample is logged to data/processed/labels.csv to avoid
double-processing across runs.
"""

import os
import csv
import json
from datetime import datetime, timedelta
from typing import List, Dict, Tuple

import librosa
import soundfile as sf


SAMPLE_RATE = 16000
DTYPE = 'int16'  # logging only; written as PCM_16


def _repo_root() -> str:
    here = os.path.dirname(os.path.abspath(__file__))
    return os.path.abspath(os.path.join(here, os.pardir, os.pardir))


RAW_PATH = os.path.join(_repo_root(), 'data', 'raw')
PROC_PATH = os.path.join(_repo_root(), 'data', 'processed')
AIR_PATH = os.path.join(PROC_PATH, 'aircraft')
NEG_PATH = os.path.join(PROC_PATH, 'negative')
LABELS_CSV = os.path.join(PROC_PATH, 'labels.csv')
RUN_LOG_CSV = os.path.join(PROC_PATH, 'run_log.csv')


def ensure_dirs():
    os.makedirs(AIR_PATH, exist_ok=True)
    os.makedirs(NEG_PATH, exist_ok=True)


def load_labels_index() -> set:
    idx = set()
    if os.path.isfile(LABELS_CSV):
        try:
            with open(LABELS_CSV, 'r', newline='', encoding='utf-8') as f:
                reader = csv.DictReader(f)
                for row in reader:
                    idx.add((row.get('batch'), row.get('filename')))
        except Exception:
            pass
    return idx


def append_label_row(row: Dict):
    file_exists = os.path.isfile(LABELS_CSV)
    with open(LABELS_CSV, 'a', newline='', encoding='utf-8') as f:
        fieldnames = [
            'timestamp', 'batch', 'filename', 'class', 'src', 'dst',
            'trim_start_s', 'trim_end_s', 'duration_s', 'sample_rate', 'dtype'
        ]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        if not file_exists:
            writer.writeheader()
        writer.writerow(row)


def append_run_log(row: Dict):
    file_exists = os.path.isfile(RUN_LOG_CSV)
    with open(RUN_LOG_CSV, 'a', newline='', encoding='utf-8') as f:
        fieldnames = [
            'run_timestamp', 'batches_count', 'annotated_count', 'processed_count',
            'aircraft_pre_s', 'aircraft_post_s', 'negative_pre_s', 'negative_post_s'
        ]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        if not file_exists:
            writer.writeheader()
        writer.writerow(row)


def find_completed_batches() -> List[str]:
    if not os.path.isdir(RAW_PATH):
        return []
    batches = []
    for name in os.listdir(RAW_PATH):
        bdir = os.path.join(RAW_PATH, name)
        if not os.path.isdir(bdir):
            continue
        if '.processed' in os.listdir(bdir) and 'annotations.json' in os.listdir(bdir):
            batches.append(name)
    return batches


def load_annotations(batch: str) -> List[Dict]:
    path = os.path.join(RAW_PATH, batch, 'annotations.json')
    try:
        with open(path, 'r', encoding='utf-8') as f:
            data = json.load(f)
            return data if isinstance(data, list) else []
    except FileNotFoundError:
        return []


def class_dir(label_bool: bool) -> str:
    return AIR_PATH if label_bool else NEG_PATH


def out_filename(batch: str, filename: str) -> str:
    """Build normalized destination filename per requirements.

    Format: "<id6>_<YYYY-MM-DD>_<HH-MM>.wav"

    - <id6> is first 6 characters of original filename (e.g., hex code or 000000)
    - Date is derived from the batch date, rolling over to next day when
      the file's time-of-day is earlier than the batch's start time-of-day.
    - Any suffix after the time portion in the original filename is dropped.
    """
    try:
        # Parse batch start for rollover logic
        batch_dt = datetime.strptime(batch, '%Y-%m-%d_%H-%M')
        batch_tod = (batch_dt.hour, batch_dt.minute)

        base = os.path.basename(filename)
        name, _ext = os.path.splitext(base)
        parts = name.split('_')
        # Expect patterns like: ID6_DATE_TIME_[...]
        id6 = parts[0][:6] if parts and len(parts[0]) >= 6 else (parts[0] if parts else '000000')
        date_str = parts[1] if len(parts) > 1 else batch_dt.strftime('%Y-%m-%d')
        time_str = parts[2] if len(parts) > 2 else batch_dt.strftime('%H-%M')

        # Parse time-of-day from file
        try:
            f_hour, f_minute = map(int, time_str.split('-'))
        except Exception:
            f_hour, f_minute = batch_tod

        # Determine corrected date based on rollover rule
        file_tod = (f_hour, f_minute)
        corrected_date = batch_dt.date()
        if file_tod < batch_tod:
            corrected_date = (batch_dt + timedelta(days=1)).date()

        return f"{id6}_{corrected_date.isoformat()}_{f_hour:02d}-{f_minute:02d}.wav"
    except Exception:
        # Fallback to batch-prefixed original name if parsing fails
        return f"{batch}__{os.path.basename(filename)}"


def trim_wav(src: str, dst: str, start_s: float, end_s: float) -> Tuple[bool, float]:
    """Trim src to [start_s, end_s] seconds and write to dst.

    Returns (ok, duration_s_written)
    """
    import numpy as np
    
    duration = max(0.0, float(end_s) - float(start_s))
    if duration <= 0.0:
        return False, 0.0
    # Load with resampling to target SAMPLE_RATE
    y, sr = librosa.load(src, sr=SAMPLE_RATE, offset=float(start_s), duration=float(duration), mono=True)
    if y.size == 0:
        return False, 0.0
    
    # Ensure data is in valid range [-1, 1] for PCM_16 conversion
    # Clip to prevent overflow during int16 conversion
    y = np.clip(y, -1.0, 1.0)
    
    # Write as PCM_16 using context manager to ensure proper file flushing
    # This is critical on Windows, especially when running through batch files
    # The context manager ensures the file is fully written and closed before returning
    with sf.SoundFile(dst, mode='w', samplerate=SAMPLE_RATE, channels=1, 
                      subtype='PCM_16', format='WAV') as f:
        f.write(y)
    
    # Verify file was written completely
    # Expected size: WAV header (44 bytes) + audio data (samples * 2 bytes for int16)
    expected_min_size = 44 + (len(y) * 2)
    actual_size = os.path.getsize(dst)
    if actual_size < expected_min_size * 0.9:  # Allow 10% variance for header variations
        print(f"Warning: File {os.path.basename(dst)} may be incomplete "
              f"(expected at least {expected_min_size}, got {actual_size})")
        return False, 0.0
    
    return True, float(y.shape[-1]) / float(SAMPLE_RATE)


def process():
    ensure_dirs()
    processed_idx = load_labels_index()

    batches = find_completed_batches()
    if not batches:
        print('No completed annotation batches found. Nothing to process.')
        return

    # Metrics for run summary
    annotated_count = 0
    processed_count = 0
    pre_s = {'aircraft': 0.0, 'negative': 0.0}
    post_s = {'aircraft': 0.0, 'negative': 0.0}

    for batch in batches:
        ann = load_annotations(batch)
        batch_dir = os.path.join(RAW_PATH, batch)
        for rec in ann:
            filename = rec.get('filename')
            label_bool = bool(rec.get('label'))
            start_s = float(rec.get('trim_start_s', 0))
            end_s = float(rec.get('trim_end_s', start_s))
            flagged = bool(rec.get('flag', False))

            if not filename:
                continue

            # Count all annotated entries for this run (includes flagged)
            annotated_count += 1
            cls_name = 'aircraft' if label_bool else 'negative'
            pre_s[cls_name] += max(0.0, end_s - start_s)

            # Skip flagged entries entirely
            if flagged:
                continue

            # skip if already logged
            if (batch, filename) in processed_idx:
                continue

            src = os.path.join(batch_dir, filename)
            if not os.path.isfile(src):
                continue

            dest_dir = class_dir(label_bool)
            # Compute normalized destination name
            dst_name = out_filename(batch, filename)
            dst = os.path.join(dest_dir, dst_name)

            # Ensure unique filename if collision
            if os.path.exists(dst):
                stem, ext = os.path.splitext(dst_name)
                i = 1
                while True:
                    cand = os.path.join(dest_dir, f"{stem}-{i}{ext}")
                    if not os.path.exists(cand):
                        dst = cand
                        break
                    i += 1

            ok, dur_written = trim_wav(src, dst, start_s, end_s)
            if not ok:
                continue

            row = {
                'timestamp': datetime.utcnow().isoformat(timespec='seconds') + 'Z',
                'batch': batch,
                'filename': filename,
                'class': 'aircraft' if label_bool else 'negative',
                'src': os.path.relpath(src, _repo_root()),
                'dst': os.path.relpath(dst, _repo_root()),
                'trim_start_s': start_s,
                'trim_end_s': end_s,
                'duration_s': round(dur_written, 3),
                'sample_rate': SAMPLE_RATE,
                'dtype': DTYPE,
            }
            append_label_row(row)
            processed_idx.add((batch, filename))
            processed_count += 1
            post_s[cls_name] += float(dur_written)

    # Emit run summary
    run_row = {
        'run_timestamp': datetime.utcnow().isoformat(timespec='seconds') + 'Z',
        'batches_count': len(batches),
        'annotated_count': annotated_count,
        'processed_count': processed_count,
        'aircraft_pre_s': round(pre_s['aircraft'], 3),
        'aircraft_post_s': round(post_s['aircraft'], 3),
        'negative_pre_s': round(pre_s['negative'], 3),
        'negative_post_s': round(post_s['negative'], 3),
    }
    append_run_log(run_row)

    print('Processing complete.')


if __name__ == '__main__':
    process()
