"""Diagnostic script to check processed audio files for corruption."""

import os
import sys
import numpy as np
import soundfile as sf
from pathlib import Path

# Add repo root to path
_repo_root = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _repo_root)

PROC_PATH = os.path.join(_repo_root, 'data', 'processed')
AIR_PATH = os.path.join(PROC_PATH, 'aircraft')
NEG_PATH = os.path.join(PROC_PATH, 'negative')


def check_wav_file(filepath):
    """Check a single WAV file for corruption."""
    results = {
        'file': os.path.basename(filepath),
        'path': filepath,
        'exists': False,
        'size': 0,
        'valid_header': False,
        'readable': False,
        'all_zeros': False,
        'flat_signal': False,
        'stats': {}
    }
    
    # Check if file exists
    if not os.path.isfile(filepath):
        return results
    
    results['exists'] = True
    results['size'] = os.path.getsize(filepath)
    
    # Check WAV header
    try:
        with open(filepath, 'rb') as f:
            header = f.read(12)
            if len(header) >= 12:
                results['valid_header'] = (
                    header[:4] == b'RIFF' and 
                    header[8:12] == b'WAVE'
                )
    except Exception as e:
        results['header_error'] = str(e)
        return results
    
    # Try to read audio data
    try:
        data, sr = sf.read(filepath)
        results['readable'] = True
        results['sample_rate'] = sr
        results['samples'] = len(data)
        results['channels'] = 1 if data.ndim == 1 else data.shape[1]
        
        # Calculate statistics
        results['stats'] = {
            'min': float(np.min(data)),
            'max': float(np.max(data)),
            'mean': float(np.mean(data)),
            'std': float(np.std(data)),
            'abs_max': float(np.max(np.abs(data)))
        }
        
        # Check if all zeros
        results['all_zeros'] = np.all(data == 0)
        
        # Check if flat (very low variance)
        # A flat signal would have std very close to 0
        results['flat_signal'] = results['stats']['std'] < 0.0001
        
        # Check if data is in expected range for PCM_16
        # PCM_16 should be int16 values, but librosa loads as float32 in [-1, 1]
        # If it's been written as PCM_16 and read back, it should be in int16 range
        if results['stats']['abs_max'] > 1.0:
            results['out_of_range'] = True
        else:
            results['out_of_range'] = False
            
    except Exception as e:
        results['read_error'] = str(e)
        return results
    
    return results


def check_recent_files(num_files=10):
    """Check the most recently processed files."""
    all_files = []
    
    # Get all files from both directories
    for dir_path in [AIR_PATH, NEG_PATH]:
        if os.path.isdir(dir_path):
            for filename in os.listdir(dir_path):
                if filename.endswith('.wav'):
                    filepath = os.path.join(dir_path, filename)
                    mtime = os.path.getmtime(filepath)
                    all_files.append((mtime, filepath))
    
    # Sort by modification time (newest first)
    all_files.sort(reverse=True)
    
    # Check the most recent files
    print(f"Checking {min(num_files, len(all_files))} most recent files...")
    print("=" * 80)
    
    issues = []
    for i, (mtime, filepath) in enumerate(all_files[:num_files], 1):
        print(f"\n[{i}/{num_files}] Checking: {os.path.basename(filepath)}")
        results = check_wav_file(filepath)
        
        if not results['exists']:
            print("  [ERROR] File does not exist")
            issues.append((filepath, "File not found"))
            continue
        
        print(f"  Size: {results['size']:,} bytes")
        print(f"  Valid WAV header: {'OK' if results['valid_header'] else 'FAIL'}")
        
        if not results['valid_header']:
            issues.append((filepath, "Invalid WAV header"))
        
        if results['readable']:
            print(f"  Sample rate: {results['sample_rate']} Hz")
            print(f"  Samples: {results['samples']:,}")
            print(f"  Channels: {results['channels']}")
            print(f"  Stats:")
            print(f"    Min: {results['stats']['min']:.6f}")
            print(f"    Max: {results['stats']['max']:.6f}")
            print(f"    Mean: {results['stats']['mean']:.6f}")
            print(f"    Std: {results['stats']['std']:.6f}")
            print(f"    Abs Max: {results['stats']['abs_max']:.6f}")
            
            if results['all_zeros']:
                print("  [ERROR] WARNING: All samples are zero!")
                issues.append((filepath, "All zeros"))
            elif results['flat_signal']:
                print("  [ERROR] WARNING: Signal is flat (very low variance)!")
                issues.append((filepath, "Flat signal"))
            else:
                print("  [OK] Signal appears to have variation")
            
            if results['out_of_range']:
                print("  [WARN] WARNING: Data out of expected range")
        else:
            print(f"  [ERROR] Cannot read audio data: {results.get('read_error', 'Unknown error')}")
            issues.append((filepath, f"Read error: {results.get('read_error', 'Unknown')}"))
    
    # Summary
    print("\n" + "=" * 80)
    print("SUMMARY")
    print("=" * 80)
    
    if issues:
        print(f"Found {len(issues)} files with issues:")
        for filepath, issue in issues:
            print(f"  - {os.path.basename(filepath)}: {issue}")
    else:
        print("[OK] All checked files appear to be valid")
    
    return issues


if __name__ == '__main__':
    print("Audio File Diagnostic Tool")
    print("=" * 80)
    print(f"Checking files in: {PROC_PATH}")
    print()
    
    issues = check_recent_files(num_files=20)
    
    if issues:
        print("\n[WARNING] ISSUES DETECTED: Some files may be corrupted")
        sys.exit(1)
    else:
        print("\n[OK] No issues detected in checked files")
        sys.exit(0)

