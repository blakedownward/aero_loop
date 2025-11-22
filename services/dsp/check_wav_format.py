"""
Check WAV file format to ensure compatibility with Edge Impulse ingestion service.

Edge Impulse's ingestion service may preprocess audio files before they reach the DSP block.
This script checks if our WAV files match what Edge Impulse expects.
"""

import os
import sys
import numpy as np
from pathlib import Path

try:
    from scipy.io import wavfile
    HAS_SCIPY = True
except ImportError:
    HAS_SCIPY = False

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


def analyze_wav_format(filepath: str):
    """Analyze WAV file format in detail."""
    print(f"\n{'='*80}")
    print(f"Analyzing: {os.path.basename(filepath)}")
    print(f"{'='*80}")
    
    results = {}
    
    # Method 1: scipy.io.wavfile (reads raw WAV header)
    if HAS_SCIPY:
        try:
            sr, data = wavfile.read(filepath)
            results['scipy'] = {
                'sample_rate': int(sr),
                'dtype': str(data.dtype),
                'shape': data.shape,
                'channels': 1 if data.ndim == 1 else data.shape[1],
                'samples': len(data) if data.ndim == 1 else data.shape[0],
                'min': int(data.min()),
                'max': int(data.max()),
                'mean': float(data.mean()),
                'std': float(data.std()),
            }
            print(f"\n1. scipy.io.wavfile (raw WAV format):")
            print(f"   Sample rate: {results['scipy']['sample_rate']} Hz")
            print(f"   Data type: {results['scipy']['dtype']}")
            print(f"   Channels: {results['scipy']['channels']}")
            print(f"   Samples: {results['scipy']['samples']}")
            print(f"   Duration: {results['scipy']['samples'] / results['scipy']['sample_rate']:.3f} seconds")
            print(f"   Value range: [{results['scipy']['min']}, {results['scipy']['max']}]")
            print(f"   Mean: {results['scipy']['mean']:.2f}, Std: {results['scipy']['std']:.2f}")
            
            # Check if it's PCM_16
            if results['scipy']['dtype'] == 'int16':
                print(f"   Format: PCM_16 (16-bit signed integer)")
                if results['scipy']['min'] >= -32768 and results['scipy']['max'] <= 32767:
                    print(f"   [OK] Values in valid int16 range")
                else:
                    print(f"   [WARN] Values outside int16 range!")
            elif results['scipy']['dtype'] == 'int32':
                print(f"   Format: PCM_32 (32-bit signed integer)")
            elif results['scipy']['dtype'] == 'uint8':
                print(f"   Format: PCM_U8 (8-bit unsigned integer)")
            else:
                print(f"   Format: {results['scipy']['dtype']} (unusual format)")
                
        except Exception as e:
            print(f"   ERROR: {e}")
    
    # Method 2: soundfile (reads with proper WAV parsing)
    if HAS_SOUNDFILE:
        try:
            info = sf.info(filepath)
            data, sr = sf.read(filepath)
            
            results['soundfile'] = {
                'sample_rate': int(sr),
                'subtype': info.subtype,
                'format': info.format,
                'channels': info.channels,
                'samples': len(data),
                'dtype': str(data.dtype),
                'min': float(data.min()),
                'max': float(data.max()),
                'mean': float(data.mean()),
                'std': float(data.std()),
            }
            
            print(f"\n2. soundfile (WAV metadata):")
            print(f"   Sample rate: {results['soundfile']['sample_rate']} Hz")
            print(f"   Format: {results['soundfile']['format']}")
            print(f"   Subtype: {results['soundfile']['subtype']}")
            print(f"   Channels: {results['soundfile']['channels']}")
            print(f"   Samples: {results['soundfile']['samples']}")
            print(f"   Duration: {results['soundfile']['samples'] / results['soundfile']['sample_rate']:.3f} seconds")
            print(f"   Data type: {results['soundfile']['dtype']}")
            print(f"   Value range: [{results['soundfile']['min']:.6f}, {results['soundfile']['max']:.6f}]")
            print(f"   Mean: {results['soundfile']['mean']:.6f}, Std: {results['soundfile']['std']:.6f}")
            
            # Check format compatibility
            if results['soundfile']['subtype'] == 'PCM_16':
                print(f"   [OK] PCM_16 format (Edge Impulse compatible)")
            elif results['soundfile']['subtype'] == 'PCM_24':
                print(f"   [WARN] PCM_24 format (may need conversion)")
            elif results['soundfile']['subtype'] == 'PCM_32':
                print(f"   [WARN] PCM_32 format (may need conversion)")
            else:
                print(f"   [INFO] Format: {results['soundfile']['subtype']}")
                
        except Exception as e:
            print(f"   ERROR: {e}")
    
    # Method 3: librosa (loads as float)
    if HAS_LIBROSA:
        try:
            data, sr = librosa.load(filepath, sr=None, mono=True)
            results['librosa'] = {
                'sample_rate': int(sr),
                'samples': len(data),
                'dtype': str(data.dtype),
                'min': float(data.min()),
                'max': float(data.max()),
                'mean': float(data.mean()),
                'std': float(data.std()),
            }
            
            print(f"\n3. librosa (loaded as float):")
            print(f"   Sample rate: {results['librosa']['sample_rate']} Hz")
            print(f"   Samples: {results['librosa']['samples']}")
            print(f"   Duration: {results['librosa']['samples'] / results['librosa']['sample_rate']:.3f} seconds")
            print(f"   Data type: {results['librosa']['dtype']}")
            print(f"   Value range: [{results['librosa']['min']:.6f}, {results['librosa']['max']:.6f}]")
            print(f"   Mean: {results['librosa']['mean']:.6f}, Std: {results['librosa']['std']:.6f}")
            
            # Check if values are in expected range for normalized audio
            if results['librosa']['min'] >= -1.0 and results['librosa']['max'] <= 1.0:
                print(f"   [OK] Values in normalized range [-1, 1]")
            else:
                print(f"   [WARN] Values outside normalized range!")
                
        except Exception as e:
            print(f"   ERROR: {e}")
    
    # Recommendations
    print(f"\n{'='*80}")
    print("Recommendations for Edge Impulse compatibility:")
    print(f"{'='*80}")
    
    if 'soundfile' in results:
        sf_info = results['soundfile']
        recommendations = []
        
        if sf_info['sample_rate'] != 16000:
            recommendations.append(f"⚠ Sample rate is {sf_info['sample_rate']} Hz, Edge Impulse may resample to 16 kHz")
        
        if sf_info['subtype'] != 'PCM_16':
            recommendations.append(f"⚠ Format is {sf_info['subtype']}, recommend PCM_16")
        
        if sf_info['channels'] != 1:
            recommendations.append(f"⚠ {sf_info['channels']} channels, recommend mono (1 channel)")
        
        if not recommendations:
            print("✅ File format appears compatible with Edge Impulse")
        else:
            for rec in recommendations:
                print(rec)
            print(f"\nTo convert to Edge Impulse format:")
            print(f"  - Sample rate: 16000 Hz")
            print(f"  - Format: PCM_16 (16-bit signed integer)")
            print(f"  - Channels: 1 (mono)")
            print(f"  - WAV format")
    
    return results


def main():
    if len(sys.argv) < 2:
        print("Usage: python check_wav_format.py <wav_file>")
        print("\nExample:")
        print("  python check_wav_format.py data/processed/negative/000000_2025-11-18_12-33.wav")
        return 1
    
    filepath = sys.argv[1]
    
    if not os.path.exists(filepath):
        print(f"ERROR: File not found: {filepath}")
        return 1
    
    analyze_wav_format(filepath)
    return 0


if __name__ == "__main__":
    sys.exit(main())

