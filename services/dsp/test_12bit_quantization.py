"""Test 12-bit quantization against EI raw input format."""

import numpy as np
from pathlib import Path
from scipy.io import wavfile
from quantize_to_12bit import quantize_to_12bit_ei_format

# Load EI raw input
script_dir = Path(__file__).parent
ei_raw_path = script_dir / "ei_raw_input.npy"
wav_path = Path(__file__).parent.parent.parent / "data" / "processed" / "negative" / "000000_2025-11-18_12-33.wav"

print("=" * 80)
print("Testing 12-bit Quantization Against EI Raw Input")
print("=" * 80)

# Load EI raw input
ei_raw = np.load(ei_raw_path)
print(f"\nEI Raw Input:")
print(f"  Range: [{ei_raw.min()}, {ei_raw.max()}]")
print(f"  Unique values: {len(np.unique(ei_raw))}")
print(f"  Dtype: {ei_raw.dtype}")
print(f"  Shape: {ei_raw.shape}")

# Load WAV file (first 2 seconds = 32000 samples)
sr, wav = wavfile.read(str(wav_path))
wav_2s = wav[:32000]  # First 2 seconds

print(f"\nWAV File (first 2s):")
print(f"  Range: [{wav_2s.min()}, {wav_2s.max()}]")
print(f"  Unique values: {len(np.unique(wav_2s))}")
print(f"  Dtype: {wav_2s.dtype}")

# Quantize to 12-bit
quantized = quantize_to_12bit_ei_format(wav_2s)

print(f"\nQuantized WAV (12-bit):")
print(f"  Range: [{quantized.min()}, {quantized.max()}]")
print(f"  Unique values: {len(np.unique(quantized))}")
print(f"  Dtype: {quantized.dtype}")

# Compare
print(f"\nComparison:")
range_match = quantized.min() >= ei_raw.min() and quantized.max() <= ei_raw.max()
unique_similar = abs(len(np.unique(quantized)) - len(np.unique(ei_raw))) <= 10
dtype_match = quantized.dtype == ei_raw.dtype

print(f"  Range match (quantized within EI range): {range_match}")
print(f"  Unique values similar (within 10): {unique_similar}")
print(f"  Dtype match: {dtype_match}")

if range_match and unique_similar and dtype_match:
    print("\n[PASS] Quantization matches EI format!")
else:
    print("\n[WARN] Quantization may need adjustment")

# Test decode_ei_raw on both
from inference import decode_ei_raw

ei_decoded = decode_ei_raw(ei_raw)
quantized_decoded = decode_ei_raw(quantized)

print(f"\nAfter decode_ei_raw():")
print(f"  EI decoded range: [{ei_decoded.min():.6f}, {ei_decoded.max():.6f}]")
print(f"  Quantized decoded range: [{quantized_decoded.min():.6f}, {quantized_decoded.max():.6f}]")
print(f"  Mean diff: {np.abs(ei_decoded - quantized_decoded).mean():.6f}")
print(f"  Max diff: {np.abs(ei_decoded - quantized_decoded).max():.6f}")

