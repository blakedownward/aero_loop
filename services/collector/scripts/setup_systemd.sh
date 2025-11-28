#!/bin/bash
# Set up systemd service for Aero Loop Collector

set -e

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
COLLECTOR_DIR="$(dirname "$SCRIPT_DIR")"
AERO_SERVICE_FILE="$COLLECTOR_DIR/systemd/aero-collector.service"
RUNDUMP_SERVICE_FILE="$COLLECTOR_DIR/systemd/rundump.service"
GET_COORDS_SERVICE_FILE="$COLLECTOR_DIR/systemd/get_coords.service"

if [ ! -f "$AERO_SERVICE_FILE" ]; then
    echo "Error: Service file not found at $AERO_SERVICE_FILE"
    exit 1
fi

if [ ! -f "$RUNDUMP_SERVICE_FILE" ]; then
    echo "Error: rundump.service file not found at $RUNDUMP_SERVICE_FILE"
    exit 1
fi

# Load PI_USERNAME and USE_GPS from .env file
USE_GPS=false
if [ -f "$COLLECTOR_DIR/.env" ]; then
    # Extract PI_USERNAME
    PI_USERNAME=$(grep -E "^PI_USERNAME=" "$COLLECTOR_DIR/.env" | cut -d'=' -f2 | tr -d '[:space:]' || echo "")
    # Extract USE_GPS (handle true/false, case insensitive)
    USE_GPS_RAW=$(grep -E "^USE_GPS=" "$COLLECTOR_DIR/.env" | cut -d'=' -f2 | tr -d '[:space:]' | tr '[:upper:]' '[:lower:]' || echo "false")
    if [ "$USE_GPS_RAW" = "true" ]; then
        USE_GPS=true
    fi
fi

# Use PI_USERNAME if set, otherwise fallback to whoami
if [ -z "$PI_USERNAME" ]; then
    SERVICE_USER=$(whoami)
    echo "Warning: PI_USERNAME not found in .env, using current user: $SERVICE_USER"
else
    SERVICE_USER="$PI_USERNAME"
    echo "Using PI_USERNAME from .env: $SERVICE_USER"
fi

# Get repository root (assuming collector is at services/collector)
REPO_ROOT="$(dirname "$(dirname "$COLLECTOR_DIR")")"
SERVICE_PATH="$COLLECTOR_DIR/src/monitor.py"
VENV_PYTHON="$COLLECTOR_DIR/venv/bin/python3"
DUMP1090_PATH="/home/$SERVICE_USER/dump1090/dump1090"
GET_COORDS_SCRIPT="$COLLECTOR_DIR/scripts/get_coords.sh"
GET_COORDS_DIR="$COLLECTOR_DIR/scripts"
GPSFIX_SCRIPT="$COLLECTOR_DIR/scripts/gpsfix.py"

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

# Verify dump1090 exists (warn if not found, but continue)
if [ ! -f "$DUMP1090_PATH" ]; then
    echo "Warning: dump1090 not found at $DUMP1090_PATH"
    echo "The rundump service may not work correctly. Ensure dump1090 is installed."
fi

# Check if GPS service should be installed
INSTALL_GPS_SERVICE=false
if [ "$USE_GPS" = "true" ]; then
    if [ ! -f "$GET_COORDS_SCRIPT" ]; then
        echo "Warning: get_coords.sh not found at $GET_COORDS_SCRIPT"
        echo "GPS service will not be installed."
    elif [ ! -f "$GPSFIX_SCRIPT" ]; then
        echo "Warning: gpsfix.py not found at $GPSFIX_SCRIPT"
        echo "GPS service will not be installed."
    else
        INSTALL_GPS_SERVICE=true
        echo "GPS enabled: Will install get_coords.service"
    fi
else
    echo "GPS disabled (USE_GPS=false): Skipping get_coords.service installation"
fi

# Build dependency strings based on GPS service
if [ "$INSTALL_GPS_SERVICE" = "true" ]; then
    RUNDUMP_AFTER="network.target get_coords.service"
    RUNDUMP_REQUIRES="get_coords.service"
    AERO_AFTER="network.target get_coords.service rundump.service"
    AERO_REQUIRES="get_coords.service rundump.service"
else
    RUNDUMP_AFTER="network.target"
    RUNDUMP_REQUIRES=""
    AERO_AFTER="network.target rundump.service"
    AERO_REQUIRES="rundump.service"
fi

# Create a temporary service file for aero-collector with actual paths
TEMP_AERO_SERVICE="/tmp/aero-collector.service"
cat > "$TEMP_AERO_SERVICE" << EOF
[Unit]
Description=Aero Loop Collector Service
After=$AERO_AFTER
Requires=$AERO_REQUIRES

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

# Create a temporary service file for rundump with actual paths
TEMP_RUNDUMP_SERVICE="/tmp/rundump.service"
cat > "$TEMP_RUNDUMP_SERVICE" << EOF
[Unit]
Description=Service to port dump1090 quietly to stdout.
StartLimitBurst=3
StartLimitInterval=10
After=$RUNDUMP_AFTER
Before=aero-collector.service
$(if [ "$INSTALL_GPS_SERVICE" = "true" ]; then echo "Requires=$RUNDUMP_REQUIRES"; fi)

[Install]
WantedBy=multi-user.target
RequiredBy=aero-collector.service

[Service]
Type=simple
User=$SERVICE_USER
WorkingDirectory=/home/$SERVICE_USER
ExecStart=$DUMP1090_PATH --net --quiet
Restart=on-failure
RestartSec=3
EOF

# Create get_coords service if GPS is enabled
if [ "$INSTALL_GPS_SERVICE" = "true" ]; then
    TEMP_GET_COORDS_SERVICE="/tmp/get_coords.service"
    cat > "$TEMP_GET_COORDS_SERVICE" << EOF
[Unit]
Description=Get GPS coordinates service
After=network.target
Before=rundump.service

[Install]
WantedBy=multi-user.target
RequiredBy=rundump.service

[Service]
Type=simple
User=$SERVICE_USER
WorkingDirectory=$GET_COORDS_DIR
ExecStart=/bin/bash $GET_COORDS_SCRIPT
Restart=on-failure
RestartSec=5
StandardOutput=journal
StandardError=journal
EOF
fi

# Copy to systemd directory
echo "Installing systemd services..."
if [ "$INSTALL_GPS_SERVICE" = "true" ]; then
    sudo cp "$TEMP_GET_COORDS_SERVICE" /etc/systemd/system/get_coords.service
    echo "  - Installed get_coords.service (GPS coordinates)"
fi
sudo cp "$TEMP_RUNDUMP_SERVICE" /etc/systemd/system/rundump.service
sudo cp "$TEMP_AERO_SERVICE" /etc/systemd/system/aero-collector.service

# Reload systemd
sudo systemctl daemon-reload

# Enable services (but don't start yet - user should configure first)
echo ""
echo "Services installed successfully!"
echo ""
echo "Installed services:"
if [ "$INSTALL_GPS_SERVICE" = "true" ]; then
    echo "  - get_coords.service (GPS coordinates)"
fi
echo "  - rundump.service (dump1090)"
echo "  - aero-collector.service (main collector)"
echo ""
echo "Before starting, make sure to:"
echo "1. Edit config/session_constants.py with your settings"
echo "2. Ensure dump1090 is installed at $DUMP1090_PATH"
echo ""
echo "Then start the services with:"
if [ "$INSTALL_GPS_SERVICE" = "true" ]; then
    echo "  sudo systemctl start get_coords"
    echo "  sudo systemctl enable get_coords  # Enable on boot"
fi
echo "  sudo systemctl start rundump"
echo "  sudo systemctl enable rundump  # Enable on boot"
echo "  sudo systemctl start aero-collector"
echo "  sudo systemctl enable aero-collector  # Enable on boot"
echo ""
echo "Note: Services will start in order:"
if [ "$INSTALL_GPS_SERVICE" = "true" ]; then
    echo "  1. get_coords.service (GPS)"
fi
echo "  2. rundump.service (dump1090)"
echo "  3. aero-collector.service (main collector)"
echo ""
echo "Check status with:"
if [ "$INSTALL_GPS_SERVICE" = "true" ]; then
    echo "  sudo systemctl status get_coords"
fi
echo "  sudo systemctl status rundump"
echo "  sudo systemctl status aero-collector"
echo "  sudo journalctl -u aero-collector -f"

