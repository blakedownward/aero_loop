#!/bin/bash
# Install Dump1090 ADS-B decoder
# This script installs dump1090-mutability or dump1090-fa

set -e

echo "Installing Dump1090..."

# Check if dump1090 is already installed
if command -v dump1090 &> /dev/null; then
    echo "Dump1090 is already installed: $(which dump1090)"
    dump1090 --version || true
    exit 0
fi

# Try to install dump1090-mutability (simpler, good for basic use)
if sudo apt-get install -y dump1090-mutability 2>/dev/null; then
    echo "Successfully installed dump1090-mutability"
    echo ""
    echo "To start Dump1090:"
    echo "  dump1090 --interactive --net"
    echo ""
    echo "This will start Dump1090 with:"
    echo "  - Interactive mode (shows aircraft on terminal)"
    echo "  - Network mode (listens on port 30003 for raw data)"
    exit 0
fi

# Alternative: Try dump1090-fa (more features)
if sudo apt-get install -y dump1090-fa 2>/dev/null; then
    echo "Successfully installed dump1090-fa"
    echo ""
    echo "To start Dump1090:"
    echo "  dump1090-fa --interactive --net"
    exit 0
fi

# If apt install fails, try building from source
echo "Package installation failed. Attempting to build from source..."

# Install build dependencies
sudo apt-get install -y build-essential librtlsdr-dev libusb-1.0-0-dev pkg-config

# Clone and build dump1090-mutability
cd /tmp
if [ ! -d "dump1090" ]; then
    git clone https://github.com/mutability/dump1090.git
fi
cd dump1090
make
sudo make install

echo ""
echo "Dump1090 installed from source."
echo "To start: dump1090 --interactive --net"

