#!/bin/bash
# Check if all dependencies are installed and configured

echo "Checking dependencies..."
echo ""

ERRORS=0

# Check Python
if command -v python3 &> /dev/null; then
    PYTHON_VERSION=$(python3 --version)
    echo "✓ Python: $PYTHON_VERSION"
else
    echo "✗ Python 3 not found"
    ERRORS=$((ERRORS + 1))
fi

# Check virtual environment
if [ -d "venv" ]; then
    echo "✓ Virtual environment exists"
    source venv/bin/activate
    
    # Check Python packages
    python3 -c "import numpy" 2>/dev/null && echo "✓ numpy installed" || { echo "✗ numpy not installed"; ERRORS=$((ERRORS + 1)); }
    python3 -c "import scipy" 2>/dev/null && echo "✓ scipy installed" || { echo "✗ scipy not installed"; ERRORS=$((ERRORS + 1)); }
    python3 -c "import sounddevice" 2>/dev/null && echo "✓ sounddevice installed" || { echo "✗ sounddevice not installed"; ERRORS=$((ERRORS + 1)); }
    python3 -c "import tflite_runtime" 2>/dev/null && echo "✓ tflite_runtime installed" || { echo "✗ tflite_runtime not installed"; ERRORS=$((ERRORS + 1)); }
    
    deactivate
else
    echo "✗ Virtual environment not found (run install.sh)"
    ERRORS=$((ERRORS + 1))
fi

# Check Dump1090
if command -v dump1090 &> /dev/null; then
    echo "✓ Dump1090 installed: $(which dump1090)"
    
    # Check if it's running
    if netstat -tuln 2>/dev/null | grep -q ":30003"; then
        echo "✓ Dump1090 is running on port 30003"
    else
        echo "⚠ Dump1090 is installed but not running on port 30003"
        echo "  Start it with: dump1090 --interactive --net"
    fi
else
    echo "✗ Dump1090 not found"
    ERRORS=$((ERRORS + 1))
fi

# Check configuration
if [ -f "config/session_constants.py" ]; then
    echo "✓ Configuration file exists"
else
    echo "✗ Configuration file not found (copy from session_constants.py.example)"
    ERRORS=$((ERRORS + 1))
fi

# Check model file
MODEL_PATH=$(python3 -c "import sys; sys.path.insert(0, 'config'); import session_constants; print(session_constants.MODEL_PATH)" 2>/dev/null || echo "")
if [ -n "$MODEL_PATH" ] && [ -f "$MODEL_PATH" ]; then
    echo "✓ Model file found: $MODEL_PATH"
elif [ -n "$MODEL_PATH" ]; then
    echo "⚠ Model file not found at: $MODEL_PATH"
else
    echo "⚠ Could not check model path (configuration may be incomplete)"
fi

# Check mel filterbank
MEL_PATH=$(python3 -c "import sys; sys.path.insert(0, 'config'); import session_constants; print(session_constants.MEL_FILTERBANK_PATH)" 2>/dev/null || echo "")
if [ -n "$MEL_PATH" ] && [ -f "$MEL_PATH" ]; then
    echo "✓ Mel filterbank found: $MEL_PATH"
elif [ -n "$MEL_PATH" ]; then
    echo "✗ Mel filterbank not found at: $MEL_PATH"
    ERRORS=$((ERRORS + 1))
else
    echo "⚠ Could not check mel filterbank path"
fi

echo ""
if [ $ERRORS -eq 0 ]; then
    echo "All dependencies are installed! ✓"
    exit 0
else
    echo "Found $ERRORS issue(s). Please fix them before running the service."
    exit 1
fi

