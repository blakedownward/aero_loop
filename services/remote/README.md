# Remote Integration

Remote connection tools for downloading session data from Raspberry Pi and deploying models.

## Setup

1. **Install dependencies:**
   ```bash
   pip install -r services/remote/requirements.txt
   ```

2. **Configure connection:**
   - Create `.env` file in repo root (see `.env.example`)
   - Or use `config/remote_config.json` (see `config/remote_config.json.example`)

## Testing the Downloader

### Quick Test (from repository root)

**Option 1: Direct execution (easiest)**
```bash
# Make sure you have .env configured first
python services/remote/download_sessions.py
```

**Option 2: As a module**
```bash
python -m services.remote.download_sessions
```

### Using a Virtual Environment (recommended for isolation)

```bash
# Create venv in repo root
python -m venv venv

# Activate (Windows)
venv\Scripts\activate

# Activate (Linux/Mac)
source venv/bin/activate

# Install dependencies
pip install -r services/remote/requirements.txt

# Run the script
python services/remote/download_sessions.py
```

### What to Expect

The script will:
1. Connect to your Raspberry Pi via SSH
2. Scan for finished sessions
3. Download finished sessions to `data/raw/`
4. Show progress and summary

If connection fails, check:
- `.env` file has correct `PI_HOST`, `PI_USER`, etc.
- SSH key or password is correct
- Pi is reachable on the network

## Testing the Model Deployer

Same as above, but use:
```bash
python -m services.remote.deploy_model
# or
python services/remote/deploy_model.py
```

## Configuration

The scripts will look for configuration in this order:
1. Environment variables (from `.env` file or system)
2. `config/remote_config.json` file

Required settings:
- `PI_HOST` - Raspberry Pi hostname or IP
- `PI_USER` - SSH username (default: `pi`)
- `PI_SESSIONS_PATH` - Path to session directories on Pi
- `PI_MODEL_PATH` - Path where models are stored on Pi
- Either `PI_SSH_KEY_PATH` or `PI_SSH_PASSWORD` for authentication

