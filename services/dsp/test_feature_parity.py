"""
Test script to verify feature generation matches Edge Impulse studio output.

This script:
1. Loads the actual WAV file from data/processed/negative/
2. Processes the first 2 seconds using our feature generator
3. Compares with the saved test vectors (ei_raw_input.npy and ei_output.npy)
4. Identifies any discrepancies
"""

import os
import sys
import numpy as np
from pathlib import Path

# Add current directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# Try to import audio loading libraries
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

# Import feature generator functions
from feature_generator import (
    decode_ei_raw,
    mfe_ei_compatible,
    best_flat_match,
    SR,
    NFFT,
    HOP,
    FRAMES,
    NOISE_FLOOR_DB
)


def load_wav_file(filepath: str, target_sr: int = 16000) -> np.ndarray:
    """
    Load WAV file and return as float32 mono array at target sample rate.
    Tries scipy.io.wavfile first (most compatible), then soundfile, then librosa.
    """
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"WAV file not found: {filepath}")
    
    # Try scipy.io.wavfile first (most compatible, no dependencies)
    if HAS_SCIPY_WAVFILE:
        try:
            sr, data = wavfile.read(filepath)
            # Convert to float32 in [-1, 1]
            if data.dtype == np.int16:
                data = data.astype(np.float32) / 32768.0
            elif data.dtype == np.int32:
                data = data.astype(np.float32) / 2147483648.0
            elif data.dtype == np.uint8:
                data = (data.astype(np.float32) - 128.0) / 128.0
            else:
                data = data.astype(np.float32)
            
            # Resample if needed
            if sr != target_sr:
                if HAS_LIBROSA:
                    data = librosa.resample(data, orig_sr=sr, target_sr=target_sr)
                else:
                    raise ValueError(f"Sample rate mismatch: got {sr}, need {target_sr}. librosa required for resampling.")
            
            # Ensure mono
            if data.ndim > 1:
                data = np.mean(data, axis=1)
            
            return np.clip(data, -1.0, 1.0).astype(np.float32)
        except Exception as e:
            print(f"scipy.io.wavfile failed: {e}, trying alternatives...")
    
    # Try soundfile
    if HAS_SOUNDFILE:
        try:
            data, sr = sf.read(filepath)
            if sr != target_sr:
                if HAS_LIBROSA:
                    data = librosa.resample(data, orig_sr=sr, target_sr=target_sr)
                else:
                    raise ValueError(f"Sample rate mismatch: got {sr}, need {target_sr}.")
            if data.ndim > 1:
                data = np.mean(data, axis=1)
            return np.clip(data.astype(np.float32), -1.0, 1.0)
        except Exception as e:
            print(f"soundfile failed: {e}, trying librosa...")
    
    # Try librosa
    if HAS_LIBROSA:
        try:
            data, sr = librosa.load(filepath, sr=target_sr, mono=True)
            return np.clip(data.astype(np.float32), -1.0, 1.0)
        except Exception as e:
            raise RuntimeError(f"All audio loading methods failed. Last error: {e}")
    
    raise RuntimeError("No audio loading library available. Install scipy, soundfile, or librosa.")


def find_wav_file(filename: str) -> str:
    """Find WAV file in data/processed/negative/ directory."""
    repo_root = Path(__file__).parent.parent.parent
    wav_path = repo_root / "data" / "processed" / "negative" / filename
    
    if wav_path.exists():
        return str(wav_path)
    
    # Try without exact match (case-insensitive, partial match)
    neg_dir = repo_root / "data" / "processed" / "negative"
    if neg_dir.exists():
        for f in neg_dir.glob("*.wav"):
            if filename.lower() in f.name.lower():
                return str(f)
    
    raise FileNotFoundError(f"Could not find {filename} in {neg_dir}")


def main():
    print("=" * 80)
    print("Feature Generation Parity Test")
    print("=" * 80)
    
    # 1. Load saved test vectors
    script_dir = Path(__file__).parent
    ei_raw_path = script_dir / "ei_raw_input.npy"
    ei_output_path = script_dir / "ei_output.npy"
    
    if not ei_raw_path.exists():
        raise FileNotFoundError(f"Test vector not found: {ei_raw_path}")
    if not ei_output_path.exists():
        raise FileNotFoundError(f"Test vector not found: {ei_output_path}")
    
    print(f"\n1. Loading saved test vectors...")
    ei_raw_input = np.load(ei_raw_path)
    ei_output = np.load(ei_output_path)
    
    print(f"   ei_raw_input.shape={ei_raw_input.shape}, dtype={ei_raw_input.dtype}")
    print(f"   ei_output.shape={ei_output.shape}, dtype={ei_output.dtype}")
    
    # 2. Find and load the actual WAV file
    wav_filename = "000000_2025-11-18_12-33.wav"
    print(f"\n2. Loading WAV file: {wav_filename}")
    try:
        wav_path = find_wav_file(wav_filename)
        print(f"   Found: {wav_path}")
    except FileNotFoundError as e:
        print(f"   ERROR: {e}")
        print(f"   Please ensure the file exists in data/processed/negative/")
        return 1
    
    # Load first 2 seconds (32000 samples at 16kHz)
    wav_data = load_wav_file(wav_path, target_sr=SR)
    print(f"   Loaded {len(wav_data)} samples at {SR} Hz")
    
    # Extract first 2 seconds
    samples_2s = SR * 2
    if len(wav_data) < samples_2s:
        print(f"   WARNING: File has only {len(wav_data)} samples, expected at least {samples_2s}")
        wav_2s = wav_data
    else:
        wav_2s = wav_data[:samples_2s]
    
    print(f"   Using {len(wav_2s)} samples for comparison")
    
    # 3. Decode EI raw input and compare with WAV
    print(f"\n3. Comparing raw input vectors...")
    ei_decoded = decode_ei_raw(ei_raw_input)
    print(f"   EI decoded: shape={ei_decoded.shape}, dtype={ei_decoded.dtype}")
    print(f"   WAV data:   shape={wav_2s.shape}, dtype={wav_2s.dtype}")
    
    if len(ei_decoded) == len(wav_2s):
        raw_diff = np.abs(ei_decoded - wav_2s)
        print(f"   Raw input max diff: {raw_diff.max():.6f}")
        print(f"   Raw input mean diff: {raw_diff.mean():.6f}")
        if raw_diff.max() < 0.001:
            print("   [OK] Raw inputs match closely")
        else:
            print("   [WARN] Raw inputs differ - this may be expected if EI exported differently")
            # Show some sample differences
            idx = np.argmax(raw_diff)
            print(f"   Largest diff at index {idx}: EI={ei_decoded[idx]:.6f}, WAV={wav_2s[idx]:.6f}")
    else:
        print(f"   [WARN] Length mismatch: EI={len(ei_decoded)}, WAV={len(wav_2s)}")
    
    # 4. Generate features from WAV
    print(f"\n4. Generating features from WAV file...")
    feats_wav = mfe_ei_compatible(wav_2s)
    print(f"   Generated features shape: {feats_wav.shape}")
    
    # 5. Generate features from EI raw input (for comparison)
    print(f"\n5. Generating features from EI raw input...")
    feats_ei = mfe_ei_compatible(ei_decoded)
    print(f"   Generated features shape: {feats_ei.shape}")
    
    # 6. Compare with EI output using best_flat_match
    print(f"\n6. Comparing with EI studio output...")
    print("   Testing different flattening orders and offsets...")
    
    seg_vec, info = best_flat_match(feats_wav, ei_output, max_offset=6)
    print(f"\n   Best match from WAV:")
    print(f"   - Offset: {info['offset']}")
    print(f"   - Mel flip: {info['mel_flip']}")
    print(f"   - Flatten order: {info['order']}")
    print(f"   - Max abs diff: {info['score'][0]:.6f}")
    print(f"   - Mean abs diff: {info['score'][1]:.6f}")
    
    # Also test with EI decoded input
    seg_vec_ei, info_ei = best_flat_match(feats_ei, ei_output, max_offset=6)
    print(f"\n   Best match from EI raw input:")
    print(f"   - Offset: {info_ei['offset']}")
    print(f"   - Mel flip: {info_ei['mel_flip']}")
    print(f"   - Flatten order: {info_ei['order']}")
    print(f"   - Max abs diff: {info_ei['score'][0]:.6f}")
    print(f"   - Mean abs diff: {info_ei['score'][1]:.6f}")
    
    # 7. Direct comparison using first 62 frames
    print(f"\n7. Direct comparison (first 62 frames, column-major flatten)...")
    if feats_wav.shape[1] < FRAMES:
        print(f"   ERROR: Not enough frames: {feats_wav.shape[1]} < {FRAMES}")
        return 1
    
    seg_wav = feats_wav[:, :FRAMES].astype(np.float32)
    ours_wav = seg_wav.reshape(-1, order='F')  # Column-major (frame-by-frame)
    
    seg_ei = feats_ei[:, :FRAMES].astype(np.float32)
    ours_ei = seg_ei.reshape(-1, order='F')
    
    theirs = ei_output.astype(np.float32)
    
    abs_diff_wav = np.abs(ours_wav - theirs)
    abs_diff_ei = np.abs(ours_ei - theirs)
    
    print(f"   From WAV file:")
    print(f"   - Max abs diff: {abs_diff_wav.max():.6f}")
    print(f"   - Mean abs diff: {abs_diff_wav.mean():.6f}")
    
    print(f"   From EI raw input:")
    print(f"   - Max abs diff: {abs_diff_ei.max():.6f}")
    print(f"   - Mean abs diff: {abs_diff_ei.mean():.6f}")
    
    # Tolerance check
    tol = 0.0021  # EI features are multiples of 1/256 ≈ 0.00390625; allow ≈ half a bin
    print(f"\n8. Tolerance check (tol={tol})...")
    
    if abs_diff_wav.max() <= tol:
        print(f"   [PASS] WAV file features match EI studio within tolerance")
    else:
        print(f"   [FAIL] WAV file features exceed tolerance")
        topk = np.argsort(-abs_diff_wav)[:10]
        print(f"   Top 10 largest differences:")
        for k in topk:
            print(f"     idx {k:4d}: ours={ours_wav[k]:.6f} | theirs={theirs[k]:.6f} | diff={abs_diff_wav[k]:.6f}")
    
    if abs_diff_ei.max() <= tol:
        print(f"   [PASS] EI raw input features match EI studio within tolerance")
    else:
        print(f"   [FAIL] EI raw input features exceed tolerance")
        topk = np.argsort(-abs_diff_ei)[:10]
        print(f"   Top 10 largest differences:")
        for k in topk:
            print(f"     idx {k:4d}: ours={ours_ei[k]:.6f} | theirs={theirs[k]:.6f} | diff={abs_diff_ei[k]:.6f}")
    
    # 9. Additional diagnostics
    print(f"\n9. Additional diagnostics...")
    print(f"   Feature statistics (from WAV, first 62 frames):")
    print(f"   - Min: {seg_wav.min():.6f}")
    print(f"   - Max: {seg_wav.max():.6f}")
    print(f"   - Mean: {seg_wav.mean():.6f}")
    print(f"   - Std: {seg_wav.std():.6f}")
    
    print(f"   EI output statistics:")
    print(f"   - Min: {theirs.min():.6f}")
    print(f"   - Max: {theirs.max():.6f}")
    print(f"   - Mean: {theirs.mean():.6f}")
    print(f"   - Std: {theirs.std():.6f}")
    
    # Check if differences are systematic
    diff_ratio = abs_diff_wav / (np.abs(theirs) + 1e-10)
    large_relative_diff = np.sum(diff_ratio > 0.1)  # >10% relative difference
    print(f"   - Features with >10% relative difference: {large_relative_diff}/{len(theirs)}")
    
    print("\n" + "=" * 80)
    print("Test complete!")
    print("=" * 80)
    
    return 0


if __name__ == "__main__":
    sys.exit(main())

