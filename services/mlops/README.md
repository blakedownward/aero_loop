# AeroLoop MLOps — Processor

This script is the first of the MLOps tasks, and can be run directly after a batch has been annotated. 

- Trims annotated WAVs from `data/raw/<batch>` based on `annotations.json`, creating a trimmed copy in `data/processed`.
- Skips work already logged in `data/processed/labels.csv`.

## Quick Start
- Install: `pip install -r requirements.txt`
- Run (repo root): `python services/mlops/processor.py`

## Inputs
- Completed batches only: `data/raw/<batch>/.processed` and `data/raw/<batch>/annotations.json`.
- Annotation record fields used: `filename`, `label` (True=aircraft), `trim_start_s`, `trim_end_s`.

## Outputs
- Trimmed WAVs to:
  - `data/processed/aircraft/`
  - `data/processed/negative/`
- Log CSV: `data/processed/labels.csv` (one row per processed file).

## Notes
- Output filenames are prefixed with batch (e.g., `batch__file.wav`).
- Audio is resampled to 16 kHz mono and written as PCM_16.
