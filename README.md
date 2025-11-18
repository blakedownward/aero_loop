**Aero Loop — Audio Annotation & MLOps Pipeline**

A complete MLOps pipeline for aircraft audio detection, including annotation, processing, training, and remote deployment to Raspberry Pi.

## Quick Start

The easiest way to run the pipeline is using the provided workflow scripts. Each script automatically activates the appropriate virtual environment and runs the service.

### Download New Samples (Run frequently - hourly/2-hourly)
```bash
# Windows (PowerShell or Command Prompt)
.\run_download.bat

# Linux/Mac
./run_download.sh
```
Downloads finished audio sessions from Raspberry Pi to `data/raw/`.

### Annotate Samples (Run periodically)
```bash
# Windows (PowerShell or Command Prompt)
.\run_annotator.bat

# Linux/Mac
./run_annotator.sh
```
Launches the Streamlit annotation UI to label downloaded audio samples.

### Train & Deploy (Run when enough samples are ready)
```bash
# Windows (PowerShell or Command Prompt)
.\run_train_deploy.bat

# Linux/Mac
./run_train_deploy.sh
```
Runs the complete MLOps workflow:
1. Process annotated files (trim, organize)
2. Upload to Edge Impulse
3. Train model
4. Evaluate model performance
5. Build & download model (if improved)
6. Deploy to Raspberry Pi (if improved)

**Note**: 
- On Windows PowerShell, use `.\` prefix (e.g., `.\run_download.bat`) - PowerShell requires explicit path for security
- On Linux/Mac, you may need to make scripts executable first: `chmod +x run_*.sh`

## Setup

Before running the scripts, ensure each service has its virtual environment set up:

```bash
# Setup Annotator
cd services/annotator
python -m venv venv
venv\Scripts\activate  # Windows
# or: source venv/bin/activate  # Linux/Mac
pip install -r requirements.txt

# Setup Remote
cd services/remote
python -m venv venv
venv\Scripts\activate  # Windows
# or: source venv/bin/activate  # Linux/Mac
pip install -r requirements.txt

# Setup MLOps
cd services/mlops
python -m venv venv
venv\Scripts\activate  # Windows
# or: source venv/bin/activate  # Linux/Mac
pip install -r requirements.txt
```

Also create a `.env` file in the repository root with your configuration (see [Remote Integration Setup](#2-remote-integration-servicesremote) below).

## Components

### 1. Annotator (`services/annotator/`)
- Annotates WAV clips by batch, saving labels to `annotations.json` per batch and marking completion via `.processed`.
- Displays filename, waveform, and an audio player; supports range selection and flags.

**Quick Start**
- **Recommended**: Use `run_annotator.bat` or `run_annotator.sh` (see Quick Start section above)
- **Manual**: `streamlit run services/annotator/app.py` (after activating venv)

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
- **Recommended**: Use `run_download.bat` or `run_download.sh` (see Quick Start section above)
- **Manual**: `python services/remote/download_sessions.py` (after activating venv)

**Deploy Model**
- **Manual**: `python services/remote/deploy_model.py` (after activating venv)
- **Note**: Model deployment is typically handled automatically by the Train & Deploy workflow

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
  1. Process annotated files (trim, etc.)
  2. Upload to Edge Impulse
  3. Train model
  4. Evaluate model performance
  5. Build & download model (if improved)
  6. Deploy improved models to Pi

**Usage**
- **Recommended**: Use `run_train_deploy.bat` or `run_train_deploy.sh` (see Quick Start section above)
- **Manual**: 
  ```bash
  # Run complete train & deploy workflow
  python services/mlops/orchestrator.py --steps process_annotations upload_to_ei train_model evaluate_model build_and_download deploy_model --wait-training
  
  # Run specific steps
  python services/mlops/orchestrator.py --steps process_annotations upload_to_ei
  
  # Wait for training to complete
  python services/mlops/orchestrator.py --wait-training
  ```

## Complete Workflow

The typical workflow follows three main phases:

1. **Data Collection**: Raspberry Pi continuously collects audio samples in sessions

2. **Download** (Run frequently - hourly/2-hourly):
   ```bash
   .\run_download.bat  # Windows (PowerShell/CMD)
   # or
   ./run_download.sh   # Linux/Mac
   ```
   Fetches finished sessions from Pi to `data/raw/`

3. **Annotation** (Run periodically when new batches are available):
   ```bash
   .\run_annotator.bat  # Windows (PowerShell/CMD)
   # or
   ./run_annotator.sh   # Linux/Mac
   ```
   Label downloaded audio samples using the Streamlit UI

4. **Train & Deploy** (Run when enough annotated samples are ready):
   ```bash
   .\run_train_deploy.bat  # Windows (PowerShell/CMD)
   # or
   ./run_train_deploy.sh   # Linux/Mac
   ```
   Automates the complete MLOps pipeline:
   - Process annotated files (trim, organize)
   - Upload to Edge Impulse
   - Train model
   - Evaluate model performance
   - Build & download model (if improved)
   - Deploy to Raspberry Pi (if improved)

**Note**: The Train & Deploy workflow automatically skips build/download and deployment if the model doesn't improve.

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
