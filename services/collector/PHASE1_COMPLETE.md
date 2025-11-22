# Phase 1 Complete: Program Ready for Testing

## ✅ Completed Updates

### 1. **session_constants.py** - GPS and Hardcoded Coordinates Support
- ✅ Added `USE_GPS` flag to toggle between GPS and hardcoded coordinates
- ✅ Implemented `try_device_coords()` function to read GPS log file
- ✅ Falls back to hardcoded coordinates if GPS is disabled or unavailable
- ✅ Matches your Pi's session ID format: `TODAY + '_' + START_TIME`
- ✅ Uses `fetch_aoi()` to calculate AOI from device coordinates and distance

### 2. **areaofinterest.py** - Haversine Library Integration
- ✅ Updated to use `haversine` library (matches your Pi implementation)
- ✅ Added `fetch_aoi()` function to calculate AOI boundaries
- ✅ Uses `inverse_haversine` and `Direction` for accurate calculations
- ✅ Maintains `in_aoi()` and `fetch_dist()` functions

### 3. **init_recording.py** - Wrapper Script
- ✅ Created entry point wrapper (matches your Pi's `monitor1090.sh` → `init_recording.py` flow)
- ✅ Supports both "monitor" and "env" modes
- ✅ Currently implements "monitor" mode (env mode placeholder)

### 4. **requirements.txt** - Dependencies
- ✅ Added `haversine` library
- ✅ All other dependencies maintained

### 5. **Import Path Fixes**
- ✅ Fixed all modules to import `session_constants` from `config/` directory
- ✅ Updated `record.py` and `nano_record.py` to use durations from `session_constants`
- ✅ Updated `monitor.py` to use `REFRESH_RATE` from `session_constants`
- ✅ All modules now properly reference configuration

## 📋 Configuration File Structure

The `session_constants.py.example` now includes:
- GPS toggle (`USE_GPS = False` for hardcoded, `True` for GPS)
- Hardcoded coordinates (`DEFAULT_LAT`, `DEFAULT_LON`)
- GPS log path (`GPS_LOG_PATH`)
- All parameters from your `config.json`:
  - `AOI_DISTANCE`, `TRIGGER_DIST`, `MAX_SILENCE`, `SILENCE_BUFFER`
  - `AC_REC_DURATION`, `SILENCE_REC_DURATION`
  - `REC_MODE`, `MIC_MODE`, `MIC_ID`, `LOC_ID`
  - Session paths and model paths

## 🧪 Testing Steps

### 1. **Set up configuration:**
```bash
cd services/collector
cp config/session_constants.py.example config/session_constants.py
# Edit config/session_constants.py with your settings
```

### 2. **Install dependencies:**
```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
# Install tflite-runtime separately (see requirements.txt notes)
```

### 3. **Test manually:**
```bash
cd src
python3 init_recording.py
```

Or test monitor directly:
```bash
cd src
python3 monitor.py
```

## 🔍 What to Check

1. **Configuration loads correctly** - No import errors
2. **GPS/Hardcoded coordinates work** - Check which coordinates are used
3. **AOI calculation works** - Verify AOI boundaries are calculated
4. **Dump1090 connection** - Ensure it can connect to port 30003
5. **Audio recording** - Test both standard and nano modes
6. **Inference** - Verify model loading and prediction

## ⚠️ Known Issues / Next Steps

1. **GPS functionality** - GPS reading code is in place but needs GPS log file to test
2. **ENV mode** - Placeholder only, not yet implemented
3. **Service files** - Phase 2 will integrate the service files you added
4. **Model path** - Ensure model file exists at configured path

## 📝 Notes

- Session ID format now matches your Pi: `YYYY-MM-DD_HH-MM`
- All durations and parameters come from `session_constants`
- GPS is optional - set `USE_GPS = False` to use hardcoded coordinates
- The `fetch_aoi()` function calculates AOI from center point and distance (matches your Pi)

