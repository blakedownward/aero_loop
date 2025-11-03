import os
import csv
import shutil


def _repo_root() -> str:
    here = os.path.dirname(os.path.abspath(__file__))
    return os.path.abspath(os.path.join(here, os.pardir, os.pardir))


ZENODO_DIR = os.path.join(_repo_root(), 'data', 'zenodo')
EI_DIR = os.path.join(_repo_root(), 'data', 'edge-impulse')
LABELS_SRC = os.path.join(ZENODO_DIR, 'labels.csv')
LABELS_DST = os.path.join(EI_DIR, 'labels.csv')


def ensure_dirs():
    os.makedirs(EI_DIR, exist_ok=True)
    for sub in (
        'aircraft-train', 'aircraft-test', 'negative-train', 'negative-test',
    ):
        os.makedirs(os.path.join(EI_DIR, sub), exist_ok=True)


def read_rows(path: str):
    rows = []
    with open(path, 'r', encoding='utf-8', newline='') as f:
        reader = csv.DictReader(f)
        for r in reader:
            rows.append(r)
    return rows


def export():
    if not os.path.isfile(LABELS_SRC):
        raise FileNotFoundError(f"Missing labels.csv at {LABELS_SRC}")

    ensure_dirs()

    rows = read_rows(LABELS_SRC)

    copied = 0
    missing = 0
    for r in rows:
        cls = (r.get('class') or '').strip().lower()
        split = (r.get('split') or '').strip().lower()
        fname = os.path.basename(r.get('filename') or '')

        if cls not in ('aircraft', 'negative'):
            continue
        if split not in ('train', 'test'):
            continue
        if not fname:
            continue

        src = os.path.join(ZENODO_DIR, cls, fname)
        if not os.path.isfile(src):
            missing += 1
            continue

        dest_dir = os.path.join(EI_DIR, f"{cls}-{split}")
        os.makedirs(dest_dir, exist_ok=True)
        dest = os.path.join(dest_dir, fname)
        shutil.copyfile(src, dest)
        copied += 1

    # Copy labels.csv as-is for reference in the EI folder
    shutil.copyfile(LABELS_SRC, LABELS_DST)

    print(f"Copied {copied} files to {EI_DIR}. Missing source files: {missing}.")


if __name__ == '__main__':
    export()

