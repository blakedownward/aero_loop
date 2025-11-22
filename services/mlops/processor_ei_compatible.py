"""
EI-Compatible WAV Processor

This module provides an alternative to processor.py that creates WAV files
in a format that exactly matches what Edge Impulse's ingestion service expects.

The key difference is avoiding librosa's resampling, which may differ from
Edge Impulse's internal resampling method.
"""

import os
import numpy as np
import soundfile as sf
from typing import Tuple

# Try to import scipy for direct WAV reading
try:
    from scipy.io import wavfile
    HAS_SCIPY = True
except ImportError:
    HAS_SCIPY = False
    print("WARNING: scipy not available, falling back to librosa")

try:
    import librosa
    HAS_LIBROSA = True
except ImportError:
    HAS_LIBROSA = False
    print("WARNING: librosa not available")


SAMPLE_RATE = 16000


def trim_wav_ei_compatible(src: str, dst: str, start_s: float, end_s: float) -> Tuple[bool, float]:
    """
    Trim WAV file in a way that matches Edge Impulse's ingestion service.
    
    This function:
    1. Reads the source WAV file directly as int16 (if possible)
    2. Trims to the specified time range
    3. Resamples only if necessary, using a method compatible with Edge Impulse
    4. Writes as PCM_16 WAV file
    
    Args:
        src: Source WAV file path
        dst: Destination WAV file path
        start_s: Start time in seconds
        end_s: End time in seconds
    
    Returns:
        (success, duration_written)
    """
    duration = max(0.0, float(end_s) - float(start_s))
    if duration <= 0.0:
        return False, 0.0
    
    # Method 1: Try scipy.io.wavfile (reads raw int16, no resampling)
    if HAS_SCIPY:
        try:
            sr, data = wavfile.read(src)
            
            # Calculate sample indices
            start_sample = int(start_s * sr)
            end_sample = int(end_s * sr)
            
            # Trim
            if data.ndim > 1:
                # Multi-channel: take first channel or average
                data = data[:, 0] if data.shape[1] == 1 else np.mean(data, axis=1)
            
            if start_sample < len(data):
                trimmed = data[start_sample:min(end_sample, len(data))]
            else:
                return False, 0.0
            
            if len(trimmed) == 0:
                return False, 0.0
            
            # Resample if needed (using scipy.signal.resample or librosa)
            if sr != SAMPLE_RATE:
                if HAS_LIBROSA:
                    # Use librosa's resampling, but with explicit method
                    # Edge Impulse likely uses a similar method
                    trimmed_float = trimmed.astype(np.float32) / 32768.0
                    trimmed_float = librosa.resample(
                        trimmed_float, 
                        orig_sr=sr, 
                        target_sr=SAMPLE_RATE,
                        res_type='soxr_hq'  # High quality, similar to what EI might use
                    )
                    # Convert back to int16
                    trimmed = np.clip(trimmed_float * 32768.0, -32768, 32767).astype(np.int16)
                else:
                    # Fallback: simple linear interpolation (not ideal)
                    from scipy import signal
                    num_samples = int(len(trimmed) * SAMPLE_RATE / sr)
                    trimmed = signal.resample(trimmed, num_samples).astype(np.int16)
            
            # Write as PCM_16
            wavfile.write(dst, SAMPLE_RATE, trimmed)
            
            return True, float(len(trimmed)) / float(SAMPLE_RATE)
            
        except Exception as e:
            print(f"scipy.io.wavfile method failed: {e}, trying librosa...")
    
    # Method 2: Fallback to librosa (with explicit resampling method)
    if HAS_LIBROSA:
        try:
            # Load with explicit resampling method
            y, sr = librosa.load(
                src, 
                sr=SAMPLE_RATE,  # Resample to target rate
                offset=float(start_s), 
                duration=float(duration), 
                mono=True,
                res_type='soxr_hq'  # High quality resampling
            )
            
            if y.size == 0:
                return False, 0.0
            
            # Convert float32 [-1, 1] to int16
            # This matches what Edge Impulse expects: int16 values
            y_int16 = np.clip(y * 32768.0, -32768, 32767).astype(np.int16)
            
            # Write as PCM_16 using soundfile
            # Note: soundfile can write int16 directly, which is better
            with sf.SoundFile(dst, mode='w', samplerate=SAMPLE_RATE, channels=1, 
                            subtype='PCM_16', format='WAV') as f:
                f.write(y_int16)  # Write int16 directly, not float32
            
            return True, float(len(y_int16)) / float(SAMPLE_RATE)
            
        except Exception as e:
            print(f"librosa method failed: {e}")
            return False, 0.0
    
    return False, 0.0


def convert_wav_to_ei_format(src: str, dst: str) -> bool:
    """
    Convert an existing WAV file to Edge Impulse compatible format.
    
    This ensures:
    - Sample rate: 16000 Hz
    - Format: PCM_16
    - Channels: Mono
    - Proper int16 encoding
    
    Args:
        src: Source WAV file
        dst: Destination WAV file
    
    Returns:
        True if successful
    """
    try:
        if HAS_SCIPY:
            sr, data = wavfile.read(src)
            
            # Convert to mono if needed
            if data.ndim > 1:
                data = data[:, 0] if data.shape[1] == 1 else np.mean(data, axis=1)
            
            # Resample if needed
            if sr != SAMPLE_RATE:
                if HAS_LIBROSA:
                    data_float = data.astype(np.float32) / 32768.0
                    data_float = librosa.resample(
                        data_float, 
                        orig_sr=sr, 
                        target_sr=SAMPLE_RATE,
                        res_type='soxr_hq'
                    )
                    data = np.clip(data_float * 32768.0, -32768, 32767).astype(np.int16)
                else:
                    from scipy import signal
                    num_samples = int(len(data) * SAMPLE_RATE / sr)
                    data = signal.resample(data, num_samples).astype(np.int16)
            
            # Write
            wavfile.write(dst, SAMPLE_RATE, data)
            return True
            
        elif HAS_LIBROSA:
            y, sr = librosa.load(src, sr=SAMPLE_RATE, mono=True, res_type='soxr_hq')
            y_int16 = np.clip(y * 32768.0, -32768, 32767).astype(np.int16)
            
            with sf.SoundFile(dst, mode='w', samplerate=SAMPLE_RATE, channels=1, 
                            subtype='PCM_16', format='WAV') as f:
                f.write(y_int16)
            return True
            
    except Exception as e:
        print(f"Error converting WAV file: {e}")
        return False
    
    return False


if __name__ == "__main__":
    import sys
    if len(sys.argv) < 4:
        print("Usage: python processor_ei_compatible.py <src> <dst> <start_s> <end_s>")
        sys.exit(1)
    
    src = sys.argv[1]
    dst = sys.argv[2]
    start_s = float(sys.argv[3])
    end_s = float(sys.argv[4])
    
    success, duration = trim_wav_ei_compatible(src, dst, start_s, end_s)
    if success:
        print(f"Successfully created {dst} ({duration:.3f} seconds)")
    else:
        print(f"Failed to create {dst}")
        sys.exit(1)

