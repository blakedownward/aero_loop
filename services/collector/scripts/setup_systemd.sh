#!/bin/bash
# Set up systemd service for Aero Loop Collector

set -e

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
COLLECTOR_DIR="$(dirname "$SCRIPT_DIR")"
SERVICE_FILE="$COLLECTOR_DIR/systemd/aero-collector.service"

if [ ! -f "$SERVICE_FILE" ]; then
    echo "Error: Service file not found at $SERVICE_FILE"
    exit 1
fi

# Get the current user (should not be root)
SERVICE_USER=$(whoami)
SERVICE_PATH="$COLLECTOR_DIR/src/monitor.py"
VENV_PYTHON="$COLLECTOR_DIR/venv/bin/python3"

# Verify paths exist
if [ ! -f "$SERVICE_PATH" ]; then
    echo "Error: monitor.py not found at $SERVICE_PATH"
    exit 1
fi

if [ ! -f "$VENV_PYTHON" ]; then
    echo "Error: Python virtual environment not found at $VENV_PYTHON"
    echo "Run install.sh first to set up the virtual environment."
    exit 1
fi

# Create a temporary service file with actual paths
TEMP_SERVICE="/tmp/aero-collector.service"
cat > "$TEMP_SERVICE" << EOF
[Unit]
Description=Aero Loop Collector Service
After=network.target

[Service]
Type=simple
User=$SERVICE_USER
WorkingDirectory=$COLLECTOR_DIR/src
ExecStart=$VENV_PYTHON $SERVICE_PATH
Restart=always
RestartSec=10
StandardOutput=journal
StandardError=journal

[Install]
WantedBy=multi-user.target
EOF

# Copy to systemd directory
echo "Installing systemd service..."
sudo cp "$TEMP_SERVICE" /etc/systemd/system/aero-collector.service

# Reload systemd
sudo systemctl daemon-reload

# Enable service (but don't start yet - user should configure first)
echo ""
echo "Service installed successfully!"
echo ""
echo "Before starting, make sure to:"
echo "1. Edit config/session_constants.py with your settings"
echo "2. Ensure Dump1090 is running"
echo ""
echo "Then start the service with:"
echo "  sudo systemctl start aero-collector"
echo "  sudo systemctl enable aero-collector  # Enable on boot"
echo ""
echo "Check status with:"
echo "  sudo systemctl status aero-collector"
echo "  sudo journalctl -u aero-collector -f"

