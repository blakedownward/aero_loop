# How Configuration Works

## Overview

The `session_constants.py` file contains all configuration constants that the collector service needs. Here's how it's loaded and used:

## File Structure

```
services/collector/
├── config/
│   ├── session_constants.py.example  ← Template file (in git)
│   └── session_constants.py          ← Your actual config (NOT in git, you create this)
├── src/
│   ├── monitor.py                    ← Imports session_constants
│   ├── inference.py                  ← Imports session_constants
│   └── ...
└── .env                              ← Environment variables (optional)
```

## Import Mechanism

### Step 1: Scripts Add Config Directory to Python Path

Each script (like `monitor.py`, `inference.py`, etc.) does this:

```python
# Get the collector directory
COLLECTOR_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
# Build path to config/ directory
CONFIG_DIR = os.path.join(COLLECTOR_DIR, 'config')
# Add config/ to Python's import path
sys.path.insert(0, CONFIG_DIR)
```

This makes Python look in the `config/` directory when importing modules.

### Step 2: Import session_constants

```python
import session_constants as c
```

Python looks for `session_constants.py` in the `config/` directory (because we added it to `sys.path`).

### Step 3: Use Constants

All constants are accessed via the `c` namespace:

```python
c.MIC_MODE          # Microphone mode
c.DEVICE_COORDS     # Device coordinates
c.SESSION_PATH      # Where to save recordings
c.MODEL_PATH        # Path to TFLite model
c.AOI               # Area of Interest boundaries
# etc.
```

## Setup Process

1. **Copy the example file:**
   ```bash
   cp config/session_constants.py.example config/session_constants.py
   ```

2. **Edit your config:**
   ```bash
   nano config/session_constants.py
   ```

3. **Create .env file (optional but recommended):**
   ```bash
   cp env.example .env
   nano .env  # Set PI_USERNAME, coordinates, etc.
   ```

## How session_constants.py Works

The `session_constants.py` file:

1. **Loads .env file** (if present) using `python-dotenv`
2. **Reads environment variables** like `PI_USERNAME`, `DEVICE_LATITUDE`, etc.
3. **Calculates derived values** like:
   - `DEVICE_COORDS` (from GPS or hardcoded)
   - `AOI` (Area of Interest boundaries)
   - `SESSION_PATH` (where to save recordings)
4. **Exports all constants** that other scripts can import

## Example Flow

```
1. User runs: python3 src/monitor.py

2. monitor.py executes:
   - Adds config/ to sys.path
   - import session_constants as c

3. Python looks for session_constants.py in config/

4. session_constants.py executes:
   - Loads .env file (if exists)
   - Reads PI_USERNAME, DEVICE_LATITUDE, etc.
   - Calculates DEVICE_COORDS
   - Calculates AOI from coordinates
   - Creates SESSION_PATH directory
   - Exports all constants

5. monitor.py can now use:
   - c.DEVICE_COORDS
   - c.AOI
   - c.SESSION_PATH
   - etc.
```

## Why This Design?

- **Separation**: Configuration is separate from code
- **Flexibility**: Easy to change settings without editing code
- **Environment-specific**: Each Pi can have its own `session_constants.py`
- **Git-friendly**: `.example` file is in git, actual config is not

## Troubleshooting

**Error: `ModuleNotFoundError: No module named 'session_constants'`**
- Solution: Copy `config/session_constants.py.example` to `config/session_constants.py`

**Error: `NameError: name 'c' is not defined`**
- Solution: Make sure you imported: `import session_constants as c`

**Constants are wrong/default values**
- Check: Is your `.env` file in the collector directory?
- Check: Did you edit `config/session_constants.py` with your values?

