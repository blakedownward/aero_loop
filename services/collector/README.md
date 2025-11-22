# Aero Loop Collector Service

Raspberry Pi service for monitoring ADS-B signals and recording aircraft audio.

## Overview

The collector service:
- Connects to Dump1090 (ADS-B decoder) running on port 30003
- Monitors aircraft positions in a configurable area of interest
- Triggers audio recordings when aircraft are detected within trigger distance
- Runs real-time inference on recordings using TensorFlow Lite models
- Saves recordings to session directories for later download

## Prerequisites

- Raspberry Pi (tested on Raspberry Pi OS)
- Python 3.9 or later
- Dump1090 installed and running
- Audio input device (USB microphone or Arduino Nano with PDM mic)
- ADS-B receiver (RTL-SDR dongle)

## Quick Start

1. **Clone the repository:**
   ```bash
   git clone <repository-url>
   cd aero_loop/services/collector
   ```

2. **Run the installation script:**
   ```bash
   chmod +x install.sh
   ./install.sh
   ```

3. **Configure the service:**
   ```bash
   # Create .env file with your settings
   cp env.example .env
   # Edit .env with your Pi username and coordinates
   nano .env
   
   # Copy and edit session constants
   cp config/session_constants.py.example config/session_constants.py
   # Edit config/session_constants.py with additional settings if needed
   ```

4. **Start the service:**
   ```bash
   sudo systemctl start aero-collector
   sudo systemctl enable aero-collector  # Enable on boot
   ```

## Manual Installation

If you prefer to install manually:

1. **Install system dependencies:**
   ```bash
   sudo apt-get update
   sudo apt-get install -y python3-pip python3-venv portaudio19-dev
   ```

2. **Install Dump1090:**
   ```bash
   ./scripts/install_dump1090.sh
   ```

3. **Set up Python environment:**
   ```bash
   python3 -m venv venv
   source venv/bin/activate
   pip install -r requirements.txt
   # Install tflite-runtime separately (see requirements.txt)
   ```

4. **Configure:**
   - Copy `config/session_constants.py.example` to `config/session_constants.py`
   - Edit with your location, microphone settings, and paths

5. **Set up systemd service:**
   ```bash
   sudo ./scripts/setup_systemd.sh
   ```

## Configuration

### Environment Variables (.env file)

Create a `.env` file in the collector directory with:

```bash
# Raspberry Pi Username (typically "pi")
PI_USERNAME=pi

# Device Coordinates (used when USE_GPS=false)
DEVICE_LATITUDE=-34.921230
DEVICE_LONGITUDE=138.599503

# GPS Configuration (fetch device coordinates from GPS service - must be configured on the Pi first)
USE_GPS=true

# GPS Log Path (if using GPS)
GPS_LOG_PATH=/home/${PI_USERNAME}/gps_log.json
```

### Session Constants (config/session_constants.py)

Edit `config/session_constants.py` to configure:

- **Location**: Device coordinates (latitude, longitude) - can also be set in .env
- **Area of Interest**: Bounding box for monitoring aircraft
- **Microphone**: Mode ("standard" or "nano") and device ID
- **Paths**: Session directory, model path, mel filterbank path
- **Thresholds**: Trigger distance, silence buffer, max silence count

**Note**: The `.env` file takes precedence for `PI_USERNAME`, `DEVICE_LATITUDE`, `DEVICE_LONGITUDE`, and `USE_GPS`. Other settings are configured in `session_constants.py`.

## Directory Structure

```
services/collector/
├── README.md              # This file
├── requirements.txt      # Python dependencies
├── install.sh            # Automated installation script
├── src/                  # Main application code
│   ├── monitor.py        # Main monitoring loop
│   ├── inference.py      # Model inference
│   ├── logger.py         # Logging utilities
│   ├── areaofinterest.py # AOI calculations
│   ├── record.py         # Audio recording (standard mic)
│   ├── nano_record.py    # Audio recording (Arduino Nano)
│   └── session_constants.py  # Configuration (symlink to config/)
├── config/               # Configuration files
│   ├── session_constants.py.example
│   └── mel_16k_512fft_32mel_50to4k.npy
├── scripts/              # Installation and setup scripts
│   ├── install_dump1090.sh
│   ├── setup_systemd.sh
│   └── check_dependencies.sh
├── systemd/              # Systemd service files
│   └── aero-collector.service
└── models/               # TensorFlow Lite models (deployed here)
```

## Service Management

```bash
# Check status
sudo systemctl status aero-collector

# View logs
sudo journalctl -u aero-collector -f

# Stop service
sudo systemctl stop aero-collector

# Start service
sudo systemctl start aero-collector

# Restart service
sudo systemctl restart aero-collector
```

## Troubleshooting

### Dump1090 not running
- Check if Dump1090 is installed: `which dump1090`
- Start Dump1090: `dump1090 --interactive --net`
- Verify it's listening on port 30003: `netstat -tuln | grep 30003`

### Audio device not found
- List audio devices: `python3 -c "import sounddevice; print(sounddevice.query_devices())"`
- Update `MIC_ID` in `session_constants.py`

### Model inference errors
- Verify model file exists at configured path
- Check mel filterbank file exists
- Ensure tflite-runtime is installed correctly

### Service won't start
- Check logs: `sudo journalctl -u aero-collector -n 50`
- Verify paths in `session_constants.py` are correct
- Ensure Python virtual environment is activated in service file

## Development

To run manually for testing:

```bash
cd services/collector
source venv/bin/activate
cd src
python3 monitor.py
```

## License

See main repository LICENSE file.

