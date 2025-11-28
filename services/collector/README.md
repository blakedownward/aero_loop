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

- Raspberry Pi (tested on Raspbian OS [32-bit])
- Python 3.9 or later
- Dump1090 installed and running
- Audio input device (USB microphone or Arduino Nano flashed as USB mic)
- ADS-B receiver (RTL-SDR dongle)
- USB GPS (optional)

## Package Structure

```
services/collector/
├── README.md                    # Main documentation
├── requirements.txt            # Python dependencies
├── install.sh                  # Main installation script
├── env.example                 # Environment variables template
│
├── src/                        # Main application code
│   ├── monitor.py              # Main monitoring loop
│   ├── inference.py            # Model inference
│   ├── logger.py               # Logging utilities
│   ├── areaofinterest.py       # AOI calculations
│   ├── record.py               # Audio recording (standard mic)
│   ├── nano_record.py          # Audio recording (Arduino Nano)
│   └── init_recording.py       # Initialization wrapper
│
├── config/                     # Configuration files
│   ├── README.md              # Config documentation
│   ├── session_constants.py.example  # Configuration template
│   └── mel_16k_512fft_32mel_50to4k.npy  # Mel filterbank
│
├── scripts/                    # Installation scripts
│   ├── install_dump1090.sh    # Install Dump1090
│   ├── setup_systemd.sh       # Set up systemd services
│   ├── check_dependencies.sh  # Verify dependencies
│   ├── get_coords.sh          # GPS coordinates script (if USE_GPS=true)
│   └── gpsfix.py              # GPS fix detection script
│
├── systemd/                    # Systemd service files
│   ├── aero-collector.service  # Main collector service
│   ├── rundump.service        # Dump1090 service
│   └── get_coords.service     # GPS service (optional)
│
└── models/                     # TensorFlow Lite models
    └── .gitkeep               # Placeholder
```

## Quick Start

1. **On Raspberry Pi, clone the repository and navigate to collector:**
   ```bash
   cd services/collector
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
   nano config/session_constants.py  # Edit with your settings
   ```

4. **Start the services:**
   ```bash
   # If using GPS (USE_GPS=true in .env):
   sudo systemctl start get_coords
   sudo systemctl enable get_coords
   
   # Start dump1090 service
   sudo systemctl start rundump
   sudo systemctl enable rundump
   
   # Start main collector service
   sudo systemctl start aero-collector
   sudo systemctl enable aero-collector
   ```

## Manual Installation

If you prefer to install manually:

1. **Install system dependencies:**
   ```bash
   sudo apt-get update
   sudo apt-get install -y python3-pip python3-venv portaudio19-dev git
   ```

2. **Install Dump1090:**
   ```bash
   ./scripts/install_dump1090.sh
   ```

3. **Set up Python environment:**
   ```bash
   python3 -m venv venv
   source venv/bin/activate
   pip install --upgrade pip
   pip install -r requirements.txt
   # Install tflite-runtime separately (see requirements.txt)
   ```

4. **Configure:**
   - Copy `env.example` to `.env` and edit with your settings
   - Copy `config/session_constants.py.example` to `config/session_constants.py`
   - Edit with your location, microphone settings, and paths

5. **Set up systemd services:**
   ```bash
   sudo ./scripts/setup_systemd.sh
   ```

## Configuration

### Environment Variables (.env file)

Create a `.env` file in the collector directory with:

```bash
# REQUIRED: Raspberry Pi Username (typically "pi")
PI_USERNAME=pi

# REQUIRED: Device Coordinates (used when USE_GPS=false)
DEVICE_LATITUDE=-34.921230
DEVICE_LONGITUDE=138.599503

# GPS Configuration
# Set to true to use GPS coordinates, false to use hardcoded coordinates above
# Default: false (use hardcoded coordinates)
USE_GPS=false

# GPS Log Path (if using GPS)
# Can use ${PI_USERNAME} variable which will be replaced with the username above
# Default: /home/${PI_USERNAME}/gps_log.json
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

### Key Configuration Points

Before running, you must configure:

1. **Device Location** (`DEVICE_COORDS`): Your Pi's latitude/longitude
   - Set in `.env` file as `DEVICE_LATITUDE` and `DEVICE_LONGITUDE`
   - Or enable GPS with `USE_GPS=true` to fetch from GPS service

2. **Area of Interest** (`AOI`): Bounding box for monitoring aircraft
   - Configured in `config/session_constants.py`
   - Defines the geographic area to monitor

3. **Microphone Settings** (`MIC_MODE`, `MIC_ID`): Audio input configuration
   - `MIC_MODE`: "standard" for USB mic or "nano" for Arduino Nano
   - `MIC_ID`: Device ID from sounddevice query

4. **Paths** (`BASE_DIR`, `MODEL_PATH`, `SESSION_PATH`): File system paths
   - Configured in `config/session_constants.py`
   - Ensure model file exists at `MODEL_PATH`

5. **Model File**: Place your `.tflite` model in the `models/` directory

## Dependencies

### System Dependencies
- Python 3.9+
- Dump1090 (ADS-B decoder)
- PortAudio (for audio recording)
- GPSD and GPS tools (if using GPS)

### Python Dependencies
- numpy
- scipy
- sounddevice
- tflite-runtime
- python-dotenv

### Hardware
- RTL-SDR dongle (for ADS-B reception)
- USB microphone or Arduino Nano (for audio input)
- USB GPS module (optional, if using GPS)

## Service Management

The collector service consists of multiple systemd services:

```bash
# Check status of all services
sudo systemctl status get_coords    # GPS service (if USE_GPS=true)
sudo systemctl status rundump       # Dump1090 service
sudo systemctl status aero-collector  # Main collector service

# View logs
sudo journalctl -u aero-collector -f
sudo journalctl -u rundump -f
sudo journalctl -u get_coords -f    # If GPS enabled

# Stop services
sudo systemctl stop aero-collector
sudo systemctl stop rundump
sudo systemctl stop get_coords      # If GPS enabled

# Start services
sudo systemctl start get_coords      # If GPS enabled (must start first)
sudo systemctl start rundump
sudo systemctl start aero-collector

# Restart services
sudo systemctl restart aero-collector

# Enable services on boot
sudo systemctl enable get_coords    # If GPS enabled
sudo systemctl enable rundump
sudo systemctl enable aero-collector
```

**Note**: Services start in order automatically due to systemd dependencies:
1. `get_coords.service` (if GPS enabled)
2. `rundump.service` (dump1090)
3. `aero-collector.service` (main collector)

## Testing

Before running as a service, test manually:

```bash
cd services/collector
source venv/bin/activate
cd src
python3 monitor.py
```

Or use the initialization wrapper:

```bash
cd services/collector
source venv/bin/activate
python3 src/init_recording.py
```

## Troubleshooting

### Check Dependencies
```bash
./scripts/check_dependencies.sh
```

### Dump1090 not running
- Check if Dump1090 is installed: `which dump1090`
- Verify it's listening on port 30003: `netstat -tuln | grep 30003`
- Check rundump service: `sudo systemctl status rundump`
- View rundump logs: `sudo journalctl -u rundump -f`

### Audio device not found
- List audio devices: `python3 -c "import sounddevice; print(sounddevice.query_devices())"`
- Update `MIC_ID` in `session_constants.py`
- For Arduino Nano, ensure symlink exists: `/dev/ArduinoNanoMic`

### GPS service issues (if USE_GPS=true)
- Check GPS device: `sudo gpsd /dev/ttyUSB0 -F /var/run/gpsd.sock`
- Verify GPS fix: `gpspipe -w -n 10`
- Check get_coords service: `sudo systemctl status get_coords`
- View GPS logs: `sudo journalctl -u get_coords -f`
- Check GPS log file: `cat ~/gps_log.json`

### Model inference errors
- Verify model file exists at configured path
- Check mel filterbank file exists: `ls config/mel_16k_512fft_32mel_50to4k.npy`
- Ensure tflite-runtime is installed correctly
- Check model input/output shapes match expectations

### Service won't start
- Check logs: `sudo journalctl -u aero-collector -n 50`
- Verify paths in `session_constants.py` are correct
- Ensure Python virtual environment is activated in service file
- Check that `.env` file exists and has required values
- Verify `PI_USERNAME` is set correctly in `.env`

### View service logs
```bash
sudo journalctl -u aero-collector -f
```

## Development

To run manually for testing:

```bash
cd services/collector
source venv/bin/activate
cd src
python3 monitor.py
```

## Next Steps

After installation:

1. **Deploy a trained model** to `models/` directory
2. **Configure `session_constants.py`** with your location and settings
3. **Test the service manually** before running as systemd service
4. **Start the systemd services** (get_coords → rundump → aero-collector)
5. **Monitor logs** for successful operation
6. **Verify recordings** are being saved to session directories

## License

See main repository LICENSE file.
