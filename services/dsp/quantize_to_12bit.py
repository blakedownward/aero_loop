"""
Quantize WAV files to 12-bit format to match Raspberry Pi ADC output.

Based on our analysis, Edge Impulse's raw input appears to be from a 12-bit ADC
(only 17 unique values in 32,000 samples). This script converts 16-bit WAV files
to 12-bit format to match what the Pi produces during inference.
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
    print("ERROR: scipy required for this script")
    sys.exit(1)

try:
    import soundfile as sf
    HAS_SOUNDFILE = True
except ImportError:
    HAS_SOUNDFILE = False


def quantize_to_12bit(data_16bit: np.ndarray) -> np.ndarray:
    """
    Quantize 16-bit audio data to 12-bit format.
    
    Edge Impulse's DSP block expects: signal = (signal / 2**15).astype(np.float32)
    But if the source is 12-bit ADC, the actual range is [-2048, 2047] or similar.
    
    Args:
        data_16bit: int16 array in range [-32768, 32767]
    
    Returns:
        int32 array in 12-bit range (stored in int32 container, as EI does)
    """
    # 12-bit ADC range: typically [-2048, 2047] for signed, or [0, 4095] for unsigned
    # Based on EI raw input analysis: range was [-2049, 1792]
    # This suggests a 12-bit signed ADC with some offset
    
    # Method 1: Direct quantization to 12-bit range
    # Scale 16-bit to 12-bit: divide by 16 (2^4)
    data_12bit = (data_16bit // 16).astype(np.int32)
    
    # Clip to 12-bit signed range: [-2048, 2047]
    data_12bit = np.clip(data_12bit, -2048, 2047)
    
    return data_12bit


def quantize_to_12bit_ei_format(data_16bit: np.ndarray) -> np.ndarray:
    """
    Quantize to match Edge Impulse's observed format.
    
    From our analysis:
    - EI raw input: int32 with values in range [-2049, 1792]
    - Only 17 unique values
    - This suggests 12-bit ADC with specific scaling
    
    Args:
        data_16bit: int16 array
    
    Returns:
        int32 array matching EI format
    """
    # Convert to float, normalize, then quantize to 12-bit range
    data_float = data_16bit.astype(np.float32) / 32768.0  # Normalize to [-1, 1]
    
    # Quantize to 12-bit: multiply by 2048, then round
    # This gives us values in approximately [-2048, 2047] range
    data_12bit = np.round(data_float * 2048.0).astype(np.int32)
    
    # Clip to observed EI range (slightly wider than standard 12-bit)
    data_12bit = np.clip(data_12bit, -2049, 2047)
    
    return data_12bit


def convert_wav_to_12bit(src: str, dst: str, method: str = 'ei_format') -> bool:
    """
    Convert 16-bit WAV file to 12-bit format matching Raspberry Pi ADC.
    
    Args:
        src: Source WAV file (16-bit)
        dst: Destination WAV file (will be written as int32 with 12-bit values)
        method: 'ei_format' (matches EI observed format) or 'direct' (simple 12-bit)
    
    Returns:
        True if successful
    """
    try:
        # Read source file
        sr, data = wavfile.read(src)
        
        # Convert to mono if needed
        if data.ndim > 1:
            data = data[:, 0] if data.shape[1] == 1 else np.mean(data, axis=1)
        
        # Ensure int16
        if data.dtype != np.int16:
            if np.issubdtype(data.dtype, np.floating):
                # Float data: convert to int16 first
                data = np.clip(data * 32768.0, -32768, 32767).astype(np.int16)
            else:
                data = data.astype(np.int16)
        
        # Quantize to 12-bit
        if method == 'ei_format':
            data_12bit = quantize_to_12bit_ei_format(data)
        else:
            data_12bit = quantize_to_12bit(data)
        
        # Write as int32 (as Edge Impulse stores it)
        # Note: WAV format doesn't support int32 directly, so we'll write as int16
        # but the values will be in 12-bit range
        # OR we can write as int32 and let Edge Impulse handle it
        
        # Option 1: Write as int16 with 12-bit values (scaled up)
        # This preserves the quantization but uses standard WAV format
        data_scaled = (data_12bit * 16).astype(np.int16)  # Scale 12-bit to 16-bit range
        wavfile.write(dst, sr, data_scaled)
        
        print(f"Converted {os.path.basename(src)} to 12-bit format")
        print(f"  Original range: [{data.min()}, {data.max()}]")
        print(f"  12-bit range: [{data_12bit.min()}, {data_12bit.max()}]")
        print(f"  Unique values: {len(np.unique(data_12bit))}")
        print(f"  Written to: {dst}")
        
        return True
        
    except Exception as e:
        print(f"Error converting WAV file: {e}")
        import traceback
        traceback.print_exc()
        return False


def analyze_12bit_quantization(src: str):
    """Analyze how 12-bit quantization affects a WAV file."""
    try:
        sr, data = wavfile.read(src)
        
        if data.ndim > 1:
            data = data[:, 0] if data.shape[1] == 1 else np.mean(data, axis=1)
        
        if data.dtype != np.int16:
            data = data.astype(np.int16)
        
        print(f"\nAnalyzing: {os.path.basename(src)}")
        print(f"Original (16-bit):")
        print(f"  Range: [{data.min()}, {data.max()}]")
        print(f"  Unique values: {len(np.unique(data))}")
        print(f"  Mean: {data.mean():.2f}, Std: {data.std():.2f}")
        
        # Quantize to 12-bit
        data_12bit_ei = quantize_to_12bit_ei_format(data)
        data_12bit_direct = quantize_to_12bit(data)
        
        print(f"\n12-bit (EI format):")
        print(f"  Range: [{data_12bit_ei.min()}, {data_12bit_ei.max()}]")
        print(f"  Unique values: {len(np.unique(data_12bit_ei))}")
        print(f"  Mean: {data_12bit_ei.mean():.2f}, Std: {data_12bit_ei.std():.2f}")
        
        print(f"\n12-bit (Direct):")
        print(f"  Range: [{data_12bit_direct.min()}, {data_12bit_direct.max()}]")
        print(f"  Unique values: {len(np.unique(data_12bit_direct))}")
        print(f"  Mean: {data_12bit_direct.mean():.2f}, Std: {data_12bit_direct.std():.2f}")
        
        # Compare with EI raw input
        script_dir = Path(__file__).parent
        ei_raw_path = script_dir / "ei_raw_input.npy"
        if ei_raw_path.exists():
            ei_raw = np.load(ei_raw_path)
            print(f"\nEI Raw Input (for comparison):")
            print(f"  Range: [{ei_raw.min()}, {ei_raw.max()}]")
            print(f"  Unique values: {len(np.unique(ei_raw))}")
            print(f"  Mean: {ei_raw.mean():.2f}, Std: {ei_raw.std():.2f}")
            print(f"  Dtype: {ei_raw.dtype}")
            
            # Check if our quantization matches
            if len(np.unique(data_12bit_ei)) <= 20:
                print(f"\n[MATCH] EI format quantization produces similar quantization level!")
        
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()


def main():
    if len(sys.argv) < 2:
        print("Usage:")
        print("  python quantize_to_12bit.py <wav_file> [output_file]")
        print("  python quantize_to_12bit.py --analyze <wav_file>")
        print("\nExamples:")
        print("  python quantize_to_12bit.py input.wav output_12bit.wav")
        print("  python quantize_to_12bit.py --analyze input.wav")
        return 1
    
    if sys.argv[1] == '--analyze':
        if len(sys.argv) < 3:
            print("ERROR: --analyze requires a WAV file")
            return 1
        analyze_12bit_quantization(sys.argv[2])
        return 0
    
    src = sys.argv[1]
    if not os.path.exists(src):
        print(f"ERROR: File not found: {src}")
        return 1
    
    if len(sys.argv) >= 3:
        dst = sys.argv[2]
    else:
        # Auto-generate output filename
        src_path = Path(src)
        dst = str(src_path.parent / f"{src_path.stem}_12bit{src_path.suffix}")
    
    success = convert_wav_to_12bit(src, dst, method='ei_format')
    return 0 if success else 1


if __name__ == "__main__":
    sys.exit(main())

