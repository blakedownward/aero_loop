**Aero Loop — Audio Annotation & MLOps Pipeline**

A complete MLOps pipeline for aircraft audio detection, including annotation, processing, training, and remote deployment to Raspberry Pi.

## Components

### 1. Annotator (`services/annotator/`)
- Annotates WAV clips by batch, saving labels to `annotations.json` per batch and marking completion via `.processed`.
- Displays filename, waveform, and an audio player; supports range selection and flags.

**Quick Start**
- Requirements: Python 3.9+, `pip`.
- Install: `pip install -r services/annotator/requirements.txt`
- Run: `streamlit run services/annotator/app.py`

**Data Layout**
- Place audio under `data/raw/<batch_name>/*.wav`.
- App writes `data/raw/<batch_name>/annotations.json`.
- When a batch completes, app creates `data/raw/<batch_name>/.processed` and moves on.

**Usage**
- Use the range slider to set audible start/end (max = clip length; 60s fallback).
- Set "Aircraft audible" and optional "Flag clip" (flag to delete).
- Click "Commit" to save and advance.
- When all batches are done, a "Process Data" button placeholder appears.

**Notes**
- To re-annotate a finished batch, delete its `.processed` file.
- If playback is silent, check system/browser audio settings.

### 2. Remote Integration (`services/remote/`)
- Downloads finished session data from Raspberry Pi
- Deploys trained models to Raspberry Pi

**Setup**

Create a `.env` file in the repository root with your configuration:

```bash
# Copy the example file
cp .env.example .env
# Then edit .env with your settings
```

Required variables:
- **Edge Impulse**: `EI_API_KEY`, `EI_PROJECT_ID`
- **Raspberry Pi**: `PI_HOST`, `PI_USER`, `PI_SESSIONS_PATH`, `PI_MODEL_PATH`
- **SSH Auth**: Either `PI_SSH_KEY_PATH` or `PI_SSH_PASSWORD`

Optional: You can also use `config/remote_config.json` (see `config/remote_config.json.example`), but `.env` is recommended for all configuration.

**Download Sessions**
```bash
python services/remote/download_sessions.py
```

**Deploy Model**
```bash
python services/remote/deploy_model.py
```

### 3. MLOps Pipeline (`services/mlops/`)

**Processor** (`processor.py`)
- Trims annotated WAVs from `data/raw/<batch>` based on `annotations.json`
- Creates trimmed copies in `data/processed/aircraft/` or `data/processed/negative/`
- Run: `python services/mlops/processor.py`

**Edge Impulse Integration**
- **Uploader** (`ei_uploader.py`): Uploads processed samples to Edge Impulse
  - Requires: `EI_API_KEY` and `EI_PROJECT_ID` in `.env` file
  - Run: `python services/mlops/ei_uploader.py`
  
- **Trainer** (`ei_trainer.py`): Triggers training runs on Edge Impulse
  - Run: `python services/mlops/ei_trainer.py` (start training)
  - Run: `python services/mlops/ei_trainer.py status` (check status)
  
- **Downloader** (`ei_downloader.py`): Downloads trained models from Edge Impulse
  - Run: `python services/mlops/ei_downloader.py`
  
- **Model Evaluator** (`model_evaluator.py`): Compares model performance
  - Automatically called during workflow

**Orchestrator** (`orchestrator.py`)
- Orchestrates the complete MLOps workflow:
  1. Download finished sessions from Pi
  2. Process annotated files (trim, etc.)
  3. Upload to Edge Impulse
  4. Train model
  5. Download and evaluate model
  6. Deploy improved models to Pi

**Usage**
```bash
# Run complete workflow
python services/mlops/orchestrator.py

# Run specific steps
python services/mlops/orchestrator.py --steps download_sessions process_annotations

# Wait for training to complete
python services/mlops/orchestrator.py --wait-training

# Skip annotation step (if already done)
python services/mlops/orchestrator.py --skip-annotation
```

**Installation**
```bash
pip install -r services/mlops/requirements.txt
```

## Complete Workflow

1. **Data Collection**: Raspberry Pi continuously collects audio samples in sessions
2. **Download**: Run `download_sessions.py` to fetch finished sessions to `data/raw/`
3. **Annotation**: Use the annotator app to label samples
4. **Processing**: Run `processor.py` to trim and organize samples
5. **Upload**: Run `ei_uploader.py` to upload to Edge Impulse
6. **Train**: Run `ei_trainer.py` to start training
7. **Evaluate**: Run `ei_downloader.py` to download and evaluate model
8. **Deploy**: If model improved, run `deploy_model.py` to deploy to Pi

Or use the orchestrator to automate steps 2-8:
```bash
python services/mlops/orchestrator.py --wait-training
```

## Session Detection Logic

The system uses multiple methods to detect finished sessions:
1. **Primary**: Check if a newer session exists (indicates previous session is complete)
2. **Secondary**: Check for `.processed` marker file in session directory
3. **Fallback**: Time-based heuristics (sessions older than 6 hours with no recent activity)

## Model Evaluation

Models are evaluated based on:
- **Primary**: Test set accuracy
- **Secondary**: Loss (lower is better)
- **Tertiary**: F1 score

A model is only deployed if it improves on at least one metric.
