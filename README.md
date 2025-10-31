**Aero Loop — Audio Annotation App**

- Annotates WAV clips by batch, saving labels to `annotations.json` per batch and marking completion via `.processed`.
- Displays filename, waveform, and an audio player; supports range selection and flags.

**Quick Start**
- Requirements: Python 3.9+, `pip`.
- Install: `pip install -r requirements.txt`
- Run: `streamlit run services/annotator/app.py`

**Data Layout**
- Place audio under `data/raw/<batch_name>/*.wav`.
- App writes `data/raw/<batch_name>/annotations.json`.
- When a batch completes, app creates `data/raw/<batch_name>/.processed` and moves on.

**Usage**
- Use the range slider to set audible start/end (max = clip length; 60s fallback).
- Set “Aircraft audible” and optional “Flag clip”.
- Click “Commit” to save and advance; “Next file” to skip without saving.
- When all batches are done, a “Process Data” button placeholder appears.

**Notes**
- To re-annotate a finished batch, delete its `.processed` file.
- If playback is silent, check system/browser audio settings.
