# Collector Service Setup Summary

This document summarizes the collector service package structure and setup process.

## Package Structure

```
services/collector/
├── README.md                    # Main documentation
├── SETUP_SUMMARY.md            # This file
├── requirements.txt            # Python dependencies
├── install.sh                  # Main installation script
│
├── src/                        # Main application code
│   ├── monitor.py              # Main monitoring loop
│   ├── inference.py            # Model inference
│   ├── logger.py               # Logging utilities
│   ├── areaofinterest.py       # AOI calculations
│   ├── record.py               # Audio recording (standard mic)
│   └── nano_record.py          # Audio recording (Arduino Nano)
│
├── config/                     # Configuration files
│   ├── README.md              # Config documentation
│   ├── session_constants.py.example  # Configuration template
│   └── mel_16k_512fft_32mel_50to4k.npy  # Mel filterbank
│
├── scripts/                    # Installation scripts
│   ├── install_dump1090.sh    # Install Dump1090
│   ├── setup_systemd.sh       # Set up systemd service
│   └── check_dependencies.sh  # Verify dependencies
│
├── systemd/                    # Systemd service files
│   └── aero-collector.service  # Service definition
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
   cp config/session_constants.py.example config/session_constants.py
   nano config/session_constants.py  # Edit with your settings
   ```

4. **Start the service:**
   ```bash
   sudo systemctl start aero-collector
   sudo systemctl enable aero-collector
   ```

## Key Configuration Points

Before running, you must configure:

1. **Device Location** (`DEVICE_COORDS`): Your Pi's latitude/longitude
2. **Area of Interest** (`AOI`): Bounding box for monitoring aircraft
3. **Microphone Settings** (`MIC_MODE`, `MIC_ID`): Audio input configuration
4. **Paths** (`BASE_DIR`, `MODEL_PATH`, `SESSION_PATH`): File system paths
5. **Model File**: Place your `.tflite` model in the `models/` directory

## Dependencies

- **System**: Python 3.9+, Dump1090, PortAudio
- **Python**: numpy, scipy, sounddevice, tflite-runtime
- **Hardware**: RTL-SDR dongle, USB microphone or Arduino Nano

## Testing

Before running as a service, test manually:

```bash
cd services/collector
source venv/bin/activate
cd src
python3 monitor.py
```

## Troubleshooting

- **Check dependencies**: `./scripts/check_dependencies.sh`
- **View service logs**: `sudo journalctl -u aero-collector -f`
- **Verify Dump1090**: `netstat -tuln | grep 30003`

## Next Steps

After installation:
1. Deploy a trained model to `models/` directory
2. Configure `session_constants.py` with your location
3. Test the service manually
4. Start the systemd service
5. Monitor logs for successful operation

