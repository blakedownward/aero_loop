import os
import csv
import shutil


def _repo_root() -> str:
    here = os.path.dirname(os.path.abspath(__file__))
    return os.path.abspath(os.path.join(here, os.pardir, os.pardir))


PROC_DIR = os.path.join(_repo_root(), 'data', 'processed')
TARGET_CSV = os.path.join(PROC_DIR, 'target_labels.csv')
ZENODO_DIR = os.path.join(_repo_root(), 'data', 'zenodo')


def ensure_dirs():
    os.makedirs(os.path.join(ZENODO_DIR, 'aircraft'), exist_ok=True)
    os.makedirs(os.path.join(ZENODO_DIR, 'negative'), exist_ok=True)


def _read_rows(path: str):
    rows = []
    with open(path, 'r', encoding='utf-8', newline='') as f:
        reader = csv.DictReader(f)
        for r in reader:
            rows.append(r)
    return rows


def export():
    if not os.path.isfile(TARGET_CSV):
        raise FileNotFoundError(f"target_labels.csv not found at {TARGET_CSV}")

    ensure_dirs()

    rows = _read_rows(TARGET_CSV)

    exported = []
    for r in rows:
        cls = r.get('class', '').strip().lower()
        if cls not in ('aircraft', 'negative'):
            continue

        # Prefer 'dst' (repo-relative) if present; otherwise, build from class dir + filename
        dst_rel = r.get('dst')
        if dst_rel:
            src_abs = os.path.join(_repo_root(), dst_rel)
        else:
            # Fallback: look under processed/<class>/<filename>
            filename = r.get('filename') or ''
            src_abs = os.path.join(PROC_DIR, cls, os.path.basename(filename))

        if not os.path.isfile(src_abs):
            # Skip missing source files
            continue

        fname = os.path.basename(src_abs)
        dest_abs = os.path.join(ZENODO_DIR, cls, fname)

        # Copy file (overwrite if exists)
        shutil.copyfile(src_abs, dest_abs)

        # Collect metadata for zenodo labels.csv
        duration = r.get('duration_s') or r.get('duration') or ''
        sample_rate = r.get('sample_rate') or ''
        dtype = r.get('dtype') or ''
        exported.append({
            'filename': fname,
            'class': cls,
            'duration': duration,
            'sample_rate': sample_rate,
            'dtype': dtype,
        })

    # Write reduced labels.csv for zenodo
    labels_out = os.path.join(ZENODO_DIR, 'labels.csv')
    with open(labels_out, 'w', encoding='utf-8', newline='') as f:
        fieldnames = ['filename', 'class', 'duration', 'sample_rate', 'dtype']
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in exported:
            writer.writerow(row)


if __name__ == '__main__':
    export()

