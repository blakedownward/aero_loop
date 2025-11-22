"""
Audio recording functions for Arduino Nano with PDM microphone.

This module handles reading audio data from an Arduino Nano connected via Serial USB.
The Arduino should be running the nano_mic.ino sketch that streams PDM audio data.
"""

import os
import sys
import serial
import numpy as np
from scipy.io.wavfile import write as wav_write
import time

# Add config directory to path for imports
COLLECTOR_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CONFIG_DIR = os.path.join(COLLECTOR_DIR, 'config')
sys.path.insert(0, CONFIG_DIR)

import session_constants as c

# Recording parameters
SAMPLE_RATE = c.SAMPLE_RATE
CHANNELS = 1  # mono

# Serial port configuration
SERIAL_PORT = '/dev/ttyACM0'  # Common Arduino port on Linux
BAUD_RATE = 115200


def _read_nano_audio(duration, sample_rate):
    """
    Read audio data from Arduino Nano via Serial USB.
    
    Args:
        duration: Recording duration in seconds
        sample_rate: Sample rate in Hz
    
    Returns:
        numpy array of audio samples (int16)
    """
    try:
        ser = serial.Serial(SERIAL_PORT, BAUD_RATE, timeout=1)
        time.sleep(0.5)  # Wait for serial connection to stabilize
        
        num_samples = int(duration * sample_rate)
        samples = []
        
        # Read samples from serial
        # Format: Arduino sends 16-bit samples as two bytes (high, low) with newline separator
        bytes_per_sample = 2
        bytes_to_read = num_samples * bytes_per_sample
        
        data = ser.read(bytes_to_read)
        ser.close()
        
        # Convert bytes to int16 samples
        # Arduino sends: '\n' + high_byte + low_byte for each sample
        # We need to parse this format
        samples = []
        i = 0
        while i < len(data) and len(samples) < num_samples:
            if i < len(data) - 2 and data[i] == ord('\n'):
                # Next two bytes are the sample
                high_byte = data[i + 1]
                low_byte = data[i + 2]
                sample = (high_byte << 8) | low_byte
                # Convert unsigned to signed
                if sample > 32767:
                    sample = sample - 65536
                samples.append(sample)
                i += 3
            else:
                i += 1
        
        # Pad or truncate to exact length
        if len(samples) < num_samples:
            samples.extend([0] * (num_samples - len(samples)))
        elif len(samples) > num_samples:
            samples = samples[:num_samples]
        
        return np.array(samples, dtype=np.int16)
        
    except Exception as e:
        print(f"Error reading from Arduino Nano: {e}")
        # Return silence on error
        return np.zeros(int(duration * sample_rate), dtype=np.int16)


def record_ac(filepath, duration=None, sample_rate=None):
    """
    Record aircraft audio from Arduino Nano.
    
    Args:
        filepath: Path to save the WAV file
        duration: Recording duration in seconds (defaults to AC_REC_DURATION)
        sample_rate: Sample rate in Hz (defaults to SAMPLE_RATE)
    """
    if duration is None:
        duration = c.AC_REC_DURATION
    if sample_rate is None:
        sample_rate = SAMPLE_RATE
    
    print(f"Recording aircraft audio from Nano: {filepath}")
    
    recording = _read_nano_audio(duration, sample_rate)
    
    # Save as WAV file
    wav_write(filepath, sample_rate, recording)


def record_silence(filepath, duration=None, sample_rate=None):
    """
    Record silence/background audio from Arduino Nano.
    
    Args:
        filepath: Path to save the WAV file
        duration: Recording duration in seconds (defaults to SILENCE_REC_DURATION)
        sample_rate: Sample rate in Hz (defaults to SAMPLE_RATE)
    """
    if duration is None:
        duration = c.SILENCE_REC_DURATION
    if sample_rate is None:
        sample_rate = SAMPLE_RATE
    
    print(f"Recording silence from Nano: {filepath}")
    
    recording = _read_nano_audio(duration, sample_rate)
    
    # Save as WAV file
    wav_write(filepath, sample_rate, recording)

