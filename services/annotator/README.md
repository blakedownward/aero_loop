# AeroLoop Annotator

- Annotates WAV clips in batches under `data/raw/<batch>`.
- Saves labels to `annotations.json` in each batch and marks completion via `.processed`.
- Plays audio, shows waveform, and supports a start/end range slider.

## Quick Start
- Install: `pip install -r requirements.txt`
- Run (repo root): `streamlit run services/annotator/app.py`

## Data Layout
- Input: `data/raw/<batch_name>/*.wav`
- Output (per batch):
  - `annotations.json` (list of records)
  - `.processed` (created automatically when all files annotated)

## Usage
- The slider max equals the clip length (60s fallback).
- Default “Aircraft audible” value:
  - 20s clips → False
  - 60s clips → True
  - Others → True
- Click “Commit” to save and advance; “Next file” skips without saving.
- When all batches are annotated, a “Process Data” button appears to trim and export clips.

## Processing (from UI)
- Clicking “Process Data” runs `services/mlops/processor.py`:
  - Writes trimmed clips to `data/processed/aircraft` or `data/processed/negative`.
  - Appends `data/processed/labels.csv` for idempotency.
- Requires `librosa` and `soundfile` (see Quick Start install).

## Notes
- To re-annotate a finished batch, delete its `.processed` file.
- The app resumes where you left off based on `annotations.json`.
