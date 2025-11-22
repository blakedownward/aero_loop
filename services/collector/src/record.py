"""
Audio recording functions for standard USB/ALSA microphones.
"""

import os
import sys
import sounddevice as sd
import numpy as np
from scipy.io.wavfile import write as wav_write

# Add config directory to path for imports
COLLECTOR_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CONFIG_DIR = os.path.join(COLLECTOR_DIR, 'config')
sys.path.insert(0, CONFIG_DIR)

import session_constants as c

# Recording parameters
SAMPLE_RATE = c.SAMPLE_RATE
CHANNELS = 1  # mono


def record_ac(filepath, duration=None, sample_rate=None):
    """
    Record aircraft audio.
    
    Args:
        filepath: Path to save the WAV file
        duration: Recording duration in seconds (defaults to AC_REC_DURATION)
        sample_rate: Sample rate in Hz (defaults to SAMPLE_RATE)
    """
    if duration is None:
        duration = c.AC_REC_DURATION
    if sample_rate is None:
        sample_rate = SAMPLE_RATE
    
    print(f"Recording aircraft audio: {filepath}")
    
    # Record audio
    if c.MIC_ID is not None:
        recording = sd.rec(
            int(duration * sample_rate),
            samplerate=sample_rate,
            channels=CHANNELS,
            device=c.MIC_ID,
            dtype='int16'
        )
    else:
        recording = sd.rec(
            int(duration * sample_rate),
            samplerate=sample_rate,
            channels=CHANNELS,
            dtype='int16'
        )
    
    sd.wait()  # Wait until recording is finished
    
    # Save as WAV file
    wav_write(filepath, sample_rate, recording)


def record_silence(filepath, duration=None, sample_rate=None):
    """
    Record silence/background audio.
    
    Args:
        filepath: Path to save the WAV file
        duration: Recording duration in seconds (defaults to SILENCE_REC_DURATION)
        sample_rate: Sample rate in Hz (defaults to SAMPLE_RATE)
    """
    if duration is None:
        duration = c.SILENCE_REC_DURATION
    if sample_rate is None:
        sample_rate = SAMPLE_RATE
    
    print(f"Recording silence: {filepath}")
    
    # Record audio
    if c.MIC_ID is not None:
        recording = sd.rec(
            int(duration * sample_rate),
            samplerate=sample_rate,
            channels=CHANNELS,
            device=c.MIC_ID,
            dtype='int16'
        )
    else:
        recording = sd.rec(
            int(duration * sample_rate),
            samplerate=sample_rate,
            channels=CHANNELS,
            dtype='int16'
        )
    
    sd.wait()  # Wait until recording is finished
    
    # Save as WAV file
    wav_write(filepath, sample_rate, recording)

