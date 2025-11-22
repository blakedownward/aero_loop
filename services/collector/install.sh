#!/bin/bash
# Automated installation script for Aero Loop Collector Service
# Run this script on your Raspberry Pi to set up the collector service

set -e  # Exit on error

echo "=========================================="
echo "Aero Loop Collector Service Installer"
echo "=========================================="

# Get the directory where this script is located
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
cd "$SCRIPT_DIR"

# Check if running as root for system-level operations
if [ "$EUID" -eq 0 ]; then 
    echo "Error: Do not run this script as root. It will use sudo when needed."
    exit 1
fi

echo ""
echo "Step 1: Installing system dependencies..."
sudo apt-get update
sudo apt-get install -y python3-pip python3-venv portaudio19-dev git

echo ""
echo "Step 2: Installing Dump1090..."
if [ -f "scripts/install_dump1090.sh" ]; then
    chmod +x scripts/install_dump1090.sh
    ./scripts/install_dump1090.sh
else
    echo "Warning: install_dump1090.sh not found. Skipping Dump1090 installation."
    echo "You may need to install Dump1090 manually."
fi

echo ""
echo "Step 3: Setting up Python virtual environment..."
if [ ! -d "venv" ]; then
    python3 -m venv venv
fi
source venv/bin/activate

echo ""
echo "Step 4: Installing Python dependencies..."
pip install --upgrade pip
pip install -r requirements.txt

echo ""
echo "Step 5: Installing TensorFlow Lite Runtime..."
# Try to install tflite-runtime for ARM
# This may need to be adjusted based on your Pi model and Python version
PYTHON_VERSION=$(python3 --version | cut -d' ' -f2 | cut -d'.' -f1,2)
ARCH=$(uname -m)

if [ "$ARCH" = "aarch64" ]; then
    echo "Detected ARM64 architecture"
    # Try installing from Google Coral repo
    pip install --extra-index-url https://google-coral.github.io/py-repo/ tflite-runtime || {
        echo "Warning: Could not install tflite-runtime automatically."
        echo "You may need to install it manually. See requirements.txt for instructions."
    }
elif [ "$ARCH" = "armv7l" ]; then
    echo "Detected ARMv7 architecture"
    # Try installing from Google Coral repo
    pip install --extra-index-url https://google-coral.github.io/py-repo/ tflite-runtime || {
        echo "Warning: Could not install tflite-runtime automatically."
        echo "You may need to install it manually. See requirements.txt for instructions."
    }
else
    echo "Warning: Unsupported architecture: $ARCH"
    echo "You may need to install tflite-runtime manually."
fi

echo ""
echo "Step 6: Creating configuration files..."
# Create .env file if it doesn't exist
if [ ! -f ".env" ]; then
    if [ -f "env.example" ]; then
        cp env.example .env
        echo "Created .env from example."
        echo "IMPORTANT: Edit .env with your Pi username and coordinates!"
    else
        echo "Warning: env.example not found. You may need to create .env manually."
    fi
else
    echo ".env already exists. Skipping."
fi

# Create session_constants.py if it doesn't exist
if [ ! -f "config/session_constants.py" ]; then
    if [ -f "config/session_constants.py.example" ]; then
        cp config/session_constants.py.example config/session_constants.py
        echo "Created config/session_constants.py from example."
        echo "IMPORTANT: Edit config/session_constants.py with your settings before running!"
    else
        echo "Error: session_constants.py.example not found!"
        exit 1
    fi
else
    echo "config/session_constants.py already exists. Skipping."
fi

echo ""
echo "Step 7: Creating necessary directories..."
mkdir -p sessions models
mkdir -p "$(date +%Y-%m-%d)" 2>/dev/null || true

echo ""
echo "Step 8: Setting up systemd service..."
if [ -f "scripts/setup_systemd.sh" ]; then
    chmod +x scripts/setup_systemd.sh
    ./scripts/setup_systemd.sh
else
    echo "Warning: setup_systemd.sh not found. Skipping systemd setup."
    echo "You may need to set up the systemd service manually."
fi

echo ""
echo "=========================================="
echo "Installation complete!"
echo "=========================================="
echo ""
echo "Next steps:"
echo "1. Edit .env file with your settings:"
echo "   - PI_USERNAME (your Pi username)"
echo "   - DEVICE_LATITUDE and DEVICE_LONGITUDE (your location)"
echo "   - USE_GPS (true/false)"
echo ""
echo "2. Edit config/session_constants.py with additional settings:"
echo "   - Area of Interest bounds"
echo "   - Microphone settings"
echo "   - Model path"
echo "   - Recording parameters"
echo ""
echo "3. Ensure Dump1090 is running:"
echo "   dump1090 --interactive --net"
echo ""
echo "4. Test the service manually:"
echo "   source venv/bin/activate"
echo "   cd src"
echo "   python3 init_recording.py"
echo ""
echo "5. Start the systemd service:"
echo "   sudo systemctl start aero-collector"
echo "   sudo systemctl enable aero-collector"
echo ""

