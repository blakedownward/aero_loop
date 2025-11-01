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
from datetime import datetime
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
    # prefix with batch to reduce collision risk
    return f"{batch}__{filename}"


def trim_wav(src: str, dst: str, start_s: float, end_s: float) -> Tuple[bool, float]:
    """Trim src to [start_s, end_s] seconds and write to dst.

    Returns (ok, duration_s_written)
    """
    duration = max(0.0, float(end_s) - float(start_s))
    if duration <= 0.0:
        return False, 0.0
    # Load with resampling to target SAMPLE_RATE
    y, sr = librosa.load(src, sr=SAMPLE_RATE, offset=float(start_s), duration=float(duration), mono=True)
    if y.size == 0:
        return False, 0.0
    # Write as PCM_16
    sf.write(dst, y, SAMPLE_RATE, subtype='PCM_16')
    return True, float(y.shape[-1]) / float(SAMPLE_RATE)


def process():
    ensure_dirs()
    processed_idx = load_labels_index()

    batches = find_completed_batches()
    if not batches:
        print('No completed annotation batches found. Nothing to process.')
        return

    for batch in batches:
        ann = load_annotations(batch)
        batch_dir = os.path.join(RAW_PATH, batch)
        for rec in ann:
            filename = rec.get('filename')
            label_bool = bool(rec.get('label'))
            start_s = float(rec.get('trim_start_s', 0))
            end_s = float(rec.get('trim_end_s', start_s))

            if not filename:
                continue

            # skip if already logged
            if (batch, filename) in processed_idx:
                continue

            src = os.path.join(batch_dir, filename)
            if not os.path.isfile(src):
                continue

            dest_dir = class_dir(label_bool)
            dst = os.path.join(dest_dir, out_filename(batch, filename))

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

    print('Processing complete.')


if __name__ == '__main__':
    process()
