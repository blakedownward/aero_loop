"""Compare working files from yesterday vs broken files from today."""

import os
import sys
import numpy as np
import soundfile as sf
import struct

# Add repo root to path
_repo_root = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _repo_root)

PROC_PATH = os.path.join(_repo_root, 'data', 'processed')
AIR_PATH = os.path.join(PROC_PATH, 'aircraft')
NEG_PATH = os.path.join(PROC_PATH, 'negative')


def analyze_wav_structure(filepath):
    """Analyze WAV file structure in detail."""
    results = {
        'file': os.path.basename(filepath),
        'size': os.path.getsize(filepath),
        'header': {},
        'audio_stats': {}
    }
    
    # Read WAV header
    with open(filepath, 'rb') as f:
        # RIFF header
        riff = f.read(4)
        file_size = struct.unpack('<I', f.read(4))[0]
        wave = f.read(4)
        
        results['header']['riff'] = riff.decode('ascii', errors='ignore')
        results['header']['file_size'] = file_size
        results['header']['wave'] = wave.decode('ascii', errors='ignore')
        
        # Find fmt chunk
        while True:
            chunk_id = f.read(4)
            if not chunk_id:
                break
            chunk_id_str = chunk_id.decode('ascii', errors='ignore')
            chunk_size = struct.unpack('<I', f.read(4))[0]
            
            if chunk_id_str == 'fmt ':
                # Read fmt chunk
                audio_format = struct.unpack('<H', f.read(2))[0]
                num_channels = struct.unpack('<H', f.read(2))[0]
                sample_rate = struct.unpack('<I', f.read(4))[0]
                byte_rate = struct.unpack('<I', f.read(4))[0]
                block_align = struct.unpack('<H', f.read(2))[0]
                bits_per_sample = struct.unpack('<H', f.read(2))[0]
                
                results['header']['audio_format'] = audio_format
                results['header']['num_channels'] = num_channels
                results['header']['sample_rate'] = sample_rate
                results['header']['byte_rate'] = byte_rate
                results['header']['block_align'] = block_align
                results['header']['bits_per_sample'] = bits_per_sample
                
                # Skip any extra fmt data
                if chunk_size > 16:
                    f.read(chunk_size - 16)
            elif chunk_id_str == 'data':
                data_size = chunk_size
                results['header']['data_size'] = data_size
                break
            else:
                # Skip unknown chunks
                f.read(chunk_size)
    
    # Read audio data
    try:
        data, sr = sf.read(filepath)
        results['audio_stats'] = {
            'samples': len(data),
            'sample_rate': sr,
            'channels': 1 if data.ndim == 1 else data.shape[1],
            'dtype': str(data.dtype),
            'min': float(np.min(data)),
            'max': float(np.max(data)),
            'mean': float(np.mean(data)),
            'std': float(np.std(data)),
            'abs_max': float(np.max(np.abs(data)))
        }
    except Exception as e:
        results['read_error'] = str(e)
    
    return results


def compare_files(file1_path, file2_path):
    """Compare two WAV files."""
    print("=" * 80)
    print("FILE COMPARISON")
    print("=" * 80)
    
    file1 = analyze_wav_structure(file1_path)
    file2 = analyze_wav_structure(file2_path)
    
    print(f"\nFile 1: {file1['file']}")
    print(f"File 2: {file2['file']}")
    print("\n" + "-" * 80)
    
    # Compare headers
    print("\nHEADER COMPARISON:")
    print(f"  File Size: {file1['size']:,} vs {file2['size']:,} (diff: {file2['size'] - file1['size']:,})")
    
    h1 = file1['header']
    h2 = file2['header']
    
    print(f"  Audio Format: {h1.get('audio_format')} vs {h2.get('audio_format')}")
    print(f"  Channels: {h1.get('num_channels')} vs {h2.get('num_channels')}")
    print(f"  Sample Rate: {h1.get('sample_rate')} vs {h2.get('sample_rate')}")
    print(f"  Bits per Sample: {h1.get('bits_per_sample')} vs {h2.get('bits_per_sample')}")
    print(f"  Block Align: {h1.get('block_align')} vs {h2.get('block_align')}")
    print(f"  Byte Rate: {h1.get('byte_rate')} vs {h2.get('byte_rate')}")
    print(f"  Data Size: {h1.get('data_size')} vs {h2.get('data_size')}")
    
    # Compare audio stats
    print("\nAUDIO DATA COMPARISON:")
    s1 = file1['audio_stats']
    s2 = file2['audio_stats']
    
    print(f"  Samples: {s1.get('samples')} vs {s2.get('samples')}")
    print(f"  Dtype: {s1.get('dtype')} vs {s2.get('dtype')}")
    print(f"  Min: {s1.get('min'):.6f} vs {s2.get('min'):.6f}")
    print(f"  Max: {s1.get('max'):.6f} vs {s2.get('max'):.6f}")
    print(f"  Mean: {s1.get('mean'):.6f} vs {s2.get('mean'):.6f}")
    print(f"  Std: {s1.get('std'):.6f} vs {s2.get('std'):.6f}")
    print(f"  Abs Max: {s1.get('abs_max'):.6f} vs {s2.get('abs_max'):.6f}")
    
    # Check for differences
    differences = []
    if h1.get('audio_format') != h2.get('audio_format'):
        differences.append(f"Audio format differs: {h1.get('audio_format')} vs {h2.get('audio_format')}")
    if h1.get('bits_per_sample') != h2.get('bits_per_sample'):
        differences.append(f"Bits per sample differs: {h1.get('bits_per_sample')} vs {h2.get('bits_per_sample')}")
    if s1.get('dtype') != s2.get('dtype'):
        differences.append(f"Data type differs: {s1.get('dtype')} vs {s2.get('dtype')}")
    
    if differences:
        print("\n[WARNING] DIFFERENCES FOUND:")
        for diff in differences:
            print(f"  - {diff}")
    else:
        print("\n[OK] Files appear structurally similar")
    
    return differences


if __name__ == '__main__':
    # Find a working file from yesterday (Nov 18 or earlier)
    # and a broken file from today (Nov 19)
    
    working_file = None
    broken_file = None
    
    # Look for files from different dates
    for dir_path in [AIR_PATH, NEG_PATH]:
        if os.path.isdir(dir_path):
            for filename in os.listdir(dir_path):
                if filename.endswith('.wav'):
                    if '2025-11-18' in filename or '2025-11-17' in filename:
                        if working_file is None:
                            working_file = os.path.join(dir_path, filename)
                    elif '2025-11-19' in filename:
                        if broken_file is None:
                            broken_file = os.path.join(dir_path, filename)
    
    if working_file and broken_file:
        print(f"Comparing working file: {os.path.basename(working_file)}")
        print(f"With broken file: {os.path.basename(broken_file)}")
        print()
        compare_files(working_file, broken_file)
    else:
        print("Could not find files to compare")
        if not working_file:
            print("  - No working file found (looking for 2025-11-17 or 2025-11-18)")
        if not broken_file:
            print("  - No broken file found (looking for 2025-11-19)")

