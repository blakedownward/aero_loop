"""
Investigate the difference between WAV file and EI raw input.

This script analyzes:
1. The raw input format from EI
2. How the WAV file is loaded
3. Potential normalization/quantization differences
"""

import os
import sys
import numpy as np
from pathlib import Path

# Add current directory to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

try:
    from scipy.io import wavfile
    HAS_SCIPY_WAVFILE = True
except ImportError:
    HAS_SCIPY_WAVFILE = False

try:
    import soundfile as sf
    HAS_SOUNDFILE = True
except ImportError:
    HAS_SOUNDFILE = False

try:
    import librosa
    HAS_LIBROSA = True
except ImportError:
    HAS_LIBROSA = False

from feature_generator import decode_ei_raw, SR


def load_wav_scipy(filepath: str) -> tuple:
    """Load WAV using scipy.io.wavfile (reads raw int16/int32)."""
    if not HAS_SCIPY_WAVFILE:
        raise ImportError("scipy.io.wavfile not available")
    sr, data = wavfile.read(filepath)
    return sr, data


def load_wav_soundfile(filepath: str) -> tuple:
    """Load WAV using soundfile (reads as float)."""
    if not HAS_SOUNDFILE:
        raise ImportError("soundfile not available")
    data, sr = sf.read(filepath)
    return sr, data


def load_wav_librosa(filepath: str) -> tuple:
    """Load WAV using librosa (reads as float, resamples)."""
    if not HAS_LIBROSA:
        raise ImportError("librosa not available")
    data, sr = librosa.load(filepath, sr=None, mono=True)
    return sr, data


def analyze_raw_input():
    """Analyze the EI raw input format."""
    script_dir = Path(__file__).parent
    ei_raw_path = script_dir / "ei_raw_input.npy"
    
    if not ei_raw_path.exists():
        print(f"ERROR: {ei_raw_path} not found")
        return
    
    ei_raw = np.load(ei_raw_path)
    print("=" * 80)
    print("EI Raw Input Analysis")
    print("=" * 80)
    print(f"Shape: {ei_raw.shape}")
    print(f"Dtype: {ei_raw.dtype}")
    print(f"Min: {ei_raw.min()}")
    print(f"Max: {ei_raw.max()}")
    print(f"Mean: {ei_raw.mean():.6f}")
    print(f"Std: {ei_raw.std():.6f}")
    print(f"Unique values: {len(np.unique(ei_raw))}")
    
    # Check if it's quantized
    if np.issubdtype(ei_raw.dtype, np.integer):
        print(f"\nInteger type detected:")
        print(f"  - Bit depth estimate: {np.ceil(np.log2(ei_raw.max() - ei_raw.min() + 1)):.0f} bits")
        print(f"  - Range: [{ei_raw.min()}, {ei_raw.max()}]")
    
    # Decode it
    ei_decoded = decode_ei_raw(ei_raw)
    print(f"\nAfter decode_ei_raw():")
    print(f"  Shape: {ei_decoded.shape}")
    print(f"  Dtype: {ei_decoded.dtype}")
    print(f"  Min: {ei_decoded.min():.6f}")
    print(f"  Max: {ei_decoded.max():.6f}")
    print(f"  Mean: {ei_decoded.mean():.6f}")
    print(f"  Std: {ei_decoded.std():.6f}")
    
    # Check for quantization artifacts
    unique_vals = len(np.unique(ei_decoded))
    print(f"  Unique values: {unique_vals}")
    if unique_vals < len(ei_decoded) * 0.1:  # Less than 10% unique
        print(f"  [NOTE] Low unique value count suggests quantization")
    
    return ei_raw, ei_decoded


def analyze_wav_file():
    """Analyze the WAV file in different ways."""
    repo_root = Path(__file__).parent.parent.parent
    wav_path = repo_root / "data" / "processed" / "negative" / "000000_2025-11-18_12-33.wav"
    
    if not wav_path.exists():
        print(f"ERROR: {wav_path} not found")
        return None, None, None
    
    print("\n" + "=" * 80)
    print("WAV File Analysis")
    print("=" * 80)
    print(f"File: {wav_path}")
    
    results = {}
    
    # Method 1: scipy.io.wavfile (reads raw integers)
    if HAS_SCIPY_WAVFILE:
        try:
            sr, data_int = load_wav_scipy(wav_path)
            print(f"\n1. scipy.io.wavfile:")
            print(f"   Sample rate: {sr}")
            print(f"   Shape: {data_int.shape}")
            print(f"   Dtype: {data_int.dtype}")
            print(f"   Min: {data_int.min()}")
            print(f"   Max: {data_int.max()}")
            print(f"   Mean: {data_int.mean():.6f}")
            print(f"   Std: {data_int.std():.6f}")
            
            # Convert to float
            if data_int.dtype == np.int16:
                data_float = data_int.astype(np.float32) / 32768.0
            elif data_int.dtype == np.int32:
                data_float = data_int.astype(np.float32) / 2147483648.0
            elif data_int.dtype == np.uint8:
                data_float = (data_int.astype(np.float32) - 128.0) / 128.0
            else:
                data_float = data_int.astype(np.float32)
            
            if data_float.ndim > 1:
                data_float = np.mean(data_float, axis=1)
            
            data_float = np.clip(data_float, -1.0, 1.0).astype(np.float32)
            
            # Resample if needed
            if sr != SR:
                if HAS_LIBROSA:
                    data_float = librosa.resample(data_float, orig_sr=sr, target_sr=SR)
                else:
                    print(f"   [WARN] Sample rate mismatch, cannot resample without librosa")
            
            results['scipy'] = data_float[:32000]  # First 2 seconds
            print(f"   After conversion to float32 [-1, 1]:")
            print(f"     Min: {data_float.min():.6f}")
            print(f"     Max: {data_float.max():.6f}")
            print(f"     Mean: {data_float.mean():.6f}")
            print(f"     Std: {data_float.std():.6f}")
            
        except Exception as e:
            print(f"   ERROR: {e}")
    
    # Method 2: soundfile
    if HAS_SOUNDFILE:
        try:
            sr, data = load_wav_soundfile(wav_path)
            print(f"\n2. soundfile:")
            print(f"   Sample rate: {sr}")
            print(f"   Shape: {data.shape}")
            print(f"   Dtype: {data.dtype}")
            print(f"   Min: {data.min():.6f}")
            print(f"   Max: {data.max():.6f}")
            print(f"   Mean: {data.mean():.6f}")
            print(f"   Std: {data.std():.6f}")
            
            if data.ndim > 1:
                data = np.mean(data, axis=1)
            
            data = np.clip(data.astype(np.float32), -1.0, 1.0)
            
            if sr != SR:
                if HAS_LIBROSA:
                    data = librosa.resample(data, orig_sr=sr, target_sr=SR)
            
            results['soundfile'] = data[:32000]
            
        except Exception as e:
            print(f"   ERROR: {e}")
    
    # Method 3: librosa
    if HAS_LIBROSA:
        try:
            data, sr = librosa.load(wav_path, sr=SR, mono=True)
            print(f"\n3. librosa:")
            print(f"   Sample rate: {sr}")
            print(f"   Shape: {data.shape}")
            print(f"   Dtype: {data.dtype}")
            print(f"   Min: {data.min():.6f}")
            print(f"   Max: {data.max():.6f}")
            print(f"   Mean: {data.mean():.6f}")
            print(f"   Std: {data.std():.6f}")
            
            data = np.clip(data.astype(np.float32), -1.0, 1.0)
            results['librosa'] = data[:32000]
            
        except Exception as e:
            print(f"   ERROR: {e}")
    
    return results


def compare_methods(ei_decoded, wav_results):
    """Compare EI decoded input with different WAV loading methods."""
    print("\n" + "=" * 80)
    print("Comparison: EI Raw Input vs WAV File")
    print("=" * 80)
    
    ei_2s = ei_decoded[:32000]
    
    for method, wav_data in wav_results.items():
        if wav_data is None:
            continue
        
        print(f"\n{method.upper()} method:")
        diff = np.abs(ei_2s - wav_data)
        print(f"  Max abs diff: {diff.max():.6f}")
        print(f"  Mean abs diff: {diff.mean():.6f}")
        print(f"  RMS diff: {np.sqrt(np.mean(diff**2)):.6f}")
        
        # Find where differences are largest
        idx = np.argmax(diff)
        print(f"  Largest diff at index {idx}:")
        print(f"    EI: {ei_2s[idx]:.6f}")
        print(f"    WAV: {wav_data[idx]:.6f}")
        print(f"    Diff: {diff[idx]:.6f}")
        
        # Check correlation
        corr = np.corrcoef(ei_2s, wav_data)[0, 1]
        print(f"  Correlation: {corr:.6f}")
        
        # Check if one is a scaled/offset version of the other
        # Try: wav = a * ei + b
        if len(ei_2s) == len(wav_data):
            # Simple linear regression
            ei_mean = ei_2s.mean()
            wav_mean = wav_data.mean()
            ei_centered = ei_2s - ei_mean
            wav_centered = wav_data - wav_mean
            
            if np.sum(ei_centered**2) > 1e-10:
                scale = np.sum(ei_centered * wav_centered) / np.sum(ei_centered**2)
                offset = wav_mean - scale * ei_mean
                print(f"  Linear fit: WAV ~= {scale:.6f} * EI + {offset:.6f}")
                
                # Check residual
                wav_pred = scale * ei_2s + offset
                residual = np.abs(wav_data - wav_pred)
                print(f"  Residual after linear fit:")
                print(f"    Max: {residual.max():.6f}")
                print(f"    Mean: {residual.mean():.6f}")


def main():
    ei_raw, ei_decoded = analyze_raw_input()
    wav_results = analyze_wav_file()
    
    if ei_decoded is not None and wav_results:
        compare_methods(ei_decoded, wav_results)
    
    print("\n" + "=" * 80)
    print("Investigation complete!")
    print("=" * 80)


if __name__ == "__main__":
    main()

