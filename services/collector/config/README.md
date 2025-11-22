# Configuration Directory

This directory contains configuration files for the collector service.

## Files

- `session_constants.py.example` - Example configuration file
- `mel_16k_512fft_32mel_50to4k.npy` - Mel filterbank for feature extraction

## Setup

1. **Create .env file** (recommended for personal settings):
   ```bash
   cd ..  # Go to collector directory
   cp env.example .env
   # Edit .env with your Pi username and coordinates
   ```

2. **Copy the example configuration:**
   ```bash
   cp session_constants.py.example session_constants.py
   ```

3. **Edit `session_constants.py` with your settings:**
   - Device coordinates (latitude, longitude) - can also be set in .env
   - Area of Interest bounds
   - Microphone settings
   - Paths to models and data directories

4. The `session_constants.py` file should be accessible from the `src/` directory.
   The monitor script will look for it in the parent directory's `config/` folder.

**Note**: The `.env` file takes precedence for `PI_USERNAME`, `DEVICE_LATITUDE`, `DEVICE_LONGITUDE`, and `USE_GPS`.

## Mel Filterbank

The mel filterbank file (`mel_16k_512fft_32mel_50to4k.npy`) is required for feature extraction.
It must match the settings used during model training:
- Sample rate: 16 kHz
- FFT size: 512
- Mel bands: 32
- Frequency range: 50 Hz to 4000 Hz

If this file is missing, copy it from `services/dsp/mel_16k_512fft_32mel_50to4k.npy`.

