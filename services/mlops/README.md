# AeroLoop MLOps — Processor

- Trims annotated WAVs from `data/raw/<batch>` based on `annotations.json` and moves them to `data/processed`.
- Skips work already logged in `data/processed/labels.csv` (idempotent).

## Quick Start
- Install: `pip install librosa soundfile`
- Run (repo root): `python services/mlops/processor.py`

## Inputs
- Completed batches only: `data/raw/<batch>/.processed` and `data/raw/<batch>/annotations.json`.
- Annotation record fields used: `filename`, `label` (True=aircraft), `trim_start_s`, `trim_end`.

## Outputs
- Trimmed WAVs to:
  - `data/processed/aircraft/`
  - `data/processed/negative/`
- Log CSV: `data/processed/labels.csv` (one row per processed file).

## Notes
- Output filenames are prefixed with batch (e.g., `batch__file.wav`).
- Audio is resampled to 16 kHz mono and written as PCM_16.
