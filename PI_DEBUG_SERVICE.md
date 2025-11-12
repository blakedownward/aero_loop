# Debugging Pi Service - Not Seeing Log Updates

If your service shows as "active" but you're not seeing log updates, try these steps:

## 1. Check Service Status (Detailed)

```bash
ssh protopi@<PI_IP>
sudo systemctl status ten90audio.service
```

Look for:
- Is it actually "active (running)" or just "active"?
- Any error messages in the status output
- How long has it been running?

## 2. Check Service Logs

```bash
# View recent logs (last 50 lines)
sudo journalctl -u ten90audio.service -n 50

# View logs in real-time
sudo journalctl -u ten90audio.service -f

# View logs since boot
sudo journalctl -u ten90audio.service -b

# View logs with timestamps
sudo journalctl -u ten90audio.service --since "1 hour ago"
```

## 3. Check if Process is Actually Running

```bash
# Check if Python process exists
ps aux | grep python3 | grep -v grep

# Check if it's using CPU/memory
top -p $(pgrep -f "main.py")
```

## 4. Check File Permissions

```bash
# Check if service user can write to output directories
sudo -u protopi touch /home/protopi/ten90audio/ten90audio/test_write.txt
sudo -u protopi rm /home/protopi/ten90audio/ten90audio/test_write.txt

# Check directory permissions
ls -la /home/protopi/ten90audio/ten90audio/
ls -la /home/protopi/ten90audio/ten90audio/sessions/  # or wherever it writes
```

## 5. Check if Program is Actually Working

```bash
# Check if new files are being created
ls -lt /home/protopi/ten90audio/ten90audio/sessions/ | head -5

# Check if log files are being updated
find /home/protopi/ten90audio/ten90audio -name "*.log" -mmin -10
```

## 6. Restart Service and Watch Logs

```bash
# Restart the service
sudo systemctl restart ten90audio.service

# Immediately watch logs
sudo journalctl -u ten90audio.service -f
```

## 7. Check Service Configuration

```bash
# View the actual service file
cat /etc/systemd/system/ten90audio.service

# Check if paths are correct
sudo systemctl show ten90audio.service | grep ExecStart
```

## Common Issues:

1. **Service is "active" but process crashed** - Check logs for errors
2. **Wrong working directory** - Service can't find files
3. **Permission issues** - Service user can't write to output directories
4. **Silent errors** - Program is erroring but not logging
5. **Logs going to wrong place** - Check if program writes to files instead of stdout

## Quick Test: Run Manually to See Errors

```bash
# Stop the service
sudo systemctl stop ten90audio.service

# Run manually to see what happens
cd /home/protopi/ten90audio/ten90audio
python3 main.py

# If it works manually, the issue is with the service config
# If it errors, you'll see the actual error message
```

