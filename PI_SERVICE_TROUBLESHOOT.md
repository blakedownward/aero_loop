# Troubleshooting monitor1090.service - No Log Entries

If `journalctl -u monitor1090.service` shows no entries, check these:

## 1. Check Service Status

```bash
sudo systemctl status monitor1090.service
```

This will show:
- If the service is actually running
- When it was last started
- Any error messages
- The actual command being run

## 2. Check if Service File Exists

```bash
# Check if the service file exists
ls -la /etc/systemd/system/monitor1090.service

# View the service file
cat /etc/systemd/system/monitor1090.service
```

## 3. Check if Service is Enabled/Started

```bash
# Check if service is enabled
systemctl is-enabled monitor1090.service

# Check if service is active
systemctl is-active monitor1090.service

# List all services to see if it exists
systemctl list-units | grep monitor
```

## 4. Try Starting the Service Manually

```bash
# Start the service
sudo systemctl start monitor1090.service

# Immediately check status
sudo systemctl status monitor1090.service

# Check logs again
sudo journalctl -u monitor1090.service -n 50
```

## 5. Check if Process is Running

```bash
# Check for the actual process
ps aux | grep -i monitor
ps aux | grep python3

# Check what the service is supposed to run
sudo systemctl show monitor1090.service | grep ExecStart
```

## 6. Common Issues:

### Issue: Service file doesn't exist or is misconfigured
**Solution:** Check the service file path and content

### Issue: Service is not started
**Solution:** `sudo systemctl start monitor1090.service`

### Issue: Service crashes immediately
**Solution:** Check the ExecStart path is correct, check file permissions

### Issue: Service runs but produces no output
**Solution:** The program might be writing to files instead of stdout/stderr

## 7. Test Running the Program Manually

```bash
# Stop the service first
sudo systemctl stop monitor1090.service

# Find where the script is
# (Check the ExecStart line in the service file)
# Then run it manually to see errors:
cd /path/to/script
python3 main.py
```

## 8. Check Service Logs with Different Filters

```bash
# All logs for this service (no limit)
sudo journalctl -u monitor1090.service

# Logs since boot
sudo journalctl -u monitor1090.service -b

# Logs since a specific time
sudo journalctl -u monitor1090.service --since "today"
sudo journalctl -u monitor1090.service --since "1 hour ago"
```

