# AeroLoop MLOps — Processor

This script is the first of the MLOps tasks, and can be run directly after a batch has been annotated. 

- Trims annotated WAVs from `data/raw/<batch>` based on `annotations.json`, creating a trimmed copy in `data/processed`.
- Skips work already logged in `data/processed/labels.csv`.
 - Skips any annotation records marked with `flag: true`.

## Quick Start
- Install: `pip install -r requirements.txt`
- Run (repo root): `python services/mlops/processor.py`

## Inputs
- Completed batches only: `data/raw/<batch>/.processed` and `data/raw/<batch>/annotations.json`.
- Annotation record fields used: `filename`, `label` (True=aircraft), `trim_start_s`, `trim_end_s`, `flag` (optional).

## Outputs
- Trimmed WAVs to:
  - `data/processed/aircraft/`
  - `data/processed/negative/`
- Log CSVs:
  - `data/processed/labels.csv` (one row per processed file).
  - `data/processed/run_log.csv` (one row per run with summary totals).

## Notes
- Output filenames are normalized to `<id6>_<YYYY-MM-DD>_<HH-MM>.wav` where:
  - `<id6>` is the first 6 chars of the original filename (aircraft hex or `000000`).
  - Date is derived from the batch folder name. If the file's time-of-day is earlier than the batch start time, the date rolls to the next day (e.g., batch `2025-10-31_23-50` with file time `03-51` becomes date `2025-11-01`).
  - Any trailing parts after the time are dropped.
- Audio is resampled to 16 kHz mono and written as PCM_16.
