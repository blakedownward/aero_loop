# Raspberry Pi Service Setup

If your monitoring program stops when you SSH in or disconnect, it's likely running in a foreground terminal session. Here's how to set it up as a systemd service so it runs continuously in the background.

## Quick Setup

1. **Copy the service file to your Pi:**
   ```bash
   # From your local machine
   scp ten90audio.service protopi@<PI_IP>:/tmp/
   ```

2. **SSH into your Pi and install the service:**
   ```bash
   ssh protopi@<PI_IP>
   
   # Move the service file to systemd directory
   sudo mv /tmp/ten90audio.service /etc/systemd/system/
   
   # Reload systemd to recognize the new service
   sudo systemctl daemon-reload
   
   # Enable the service to start on boot
   sudo systemctl enable ten90audio.service
   
   # Start the service now
   sudo systemctl start ten90audio.service
   ```

3. **Check if it's running:**
   ```bash
   sudo systemctl status ten90audio.service
   ```

## Service Management Commands

```bash
# Check status
sudo systemctl status ten90audio.service

# View logs (last 50 lines)
sudo journalctl -u ten90audio.service -n 50

# View logs in real-time
sudo journalctl -u ten90audio.service -f

# Stop the service
sudo systemctl stop ten90audio.service

# Start the service
sudo systemctl start ten90audio.service

# Restart the service
sudo systemctl restart ten90audio.service

# Disable auto-start on boot
sudo systemctl disable ten90audio.service
```

## Important Notes

- **Update paths**: Make sure the `ExecStart` path in `ten90audio.service` matches your actual script location
- **Update user**: Change `User=protopi` if your username is different
- **Check logs**: If the service fails to start, check logs with `journalctl -u ten90audio.service`
- **File permissions**: Ensure the script and directories are readable/writable by the service user

## Alternative: Run with nohup (temporary solution)

If you can't set up systemd right now, you can run it with `nohup`:

```bash
# SSH into Pi
ssh protopi@<PI_IP>

# Navigate to your script directory
cd /home/protopi/ten90audio/ten90audio

# Run with nohup (detaches from terminal)
nohup python3 main.py > output.log 2>&1 &

# Check if it's running
ps aux | grep python3

# To stop it later, find the PID and kill it
kill <PID>
```

However, **systemd is the recommended solution** as it:
- Automatically restarts if the program crashes
- Starts on boot
- Provides better logging
- Survives SSH disconnections

