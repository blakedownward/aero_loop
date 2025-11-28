#!/usr/bin/bash

# Get the directory where this script is located
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
cd "$SCRIPT_DIR"

# Get home directory (should be set by systemd service User directive)
HOME_DIR="$HOME"
if [ -z "$HOME_DIR" ]; then
    # Fallback: try to get from /etc/passwd or use /home/pi
    HOME_DIR=$(getent passwd "$USER" | cut -d: -f6)
    if [ -z "$HOME_DIR" ]; then
        HOME_DIR="/home/pi"
    fi
fi

sleep 30

sudo gpspipe -w | python3 ./gpsfix.py >> "$HOME_DIR/gps_debug.log" 2>&1

echo "GPS fix acquired"

sleep 2

sudo gpspipe -l -o "$HOME_DIR/gps_log.json" -n 12 -w

#echo "Turnng off GPS connection"
#sleep 2
#sudo killall -e gpsd

