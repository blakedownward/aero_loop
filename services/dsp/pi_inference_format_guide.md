# Raspberry Pi Inference Format Guide

## The Problem

**Training vs Inference Mismatch:**
- **Training data**: 16-bit WAV files uploaded to Edge Impulse Studio
- **Pi inference**: 12-bit ADC output (based on analysis: only 17 unique values)
- **Result**: Model works in studio but fails on Pi

## Root Cause

From our analysis of Edge Impulse's raw input:
- **Format**: `int32` with values in range `[-2049, 1792]`
- **Unique values**: Only **17 unique values** in 32,000 samples
- **Bit depth**: Appears to be **12-bit ADC** (not 16-bit)

The Raspberry Pi's ADC produces 12-bit audio, but we're training with 16-bit WAV files. This quantization mismatch causes:
- Different feature values during inference
- Model performance degradation on device
- Mismatch between training and inference data distribution

## Solution: Quantize Training Data to 12-bit

To match what the Pi produces during inference, we need to **quantize our training WAV files to 12-bit format** before uploading to Edge Impulse.

### Step 1: Quantize WAV Files

Use the `quantize_to_12bit.py` script:

```bash
# Analyze a WAV file
python services/dsp/quantize_to_12bit.py --analyze data/processed/negative/sample.wav

# Convert to 12-bit format
python services/dsp/quantize_to_12bit.py data/processed/negative/sample.wav sample_12bit.wav
```

### Step 2: Update Processor

Modify `services/mlops/processor.py` to quantize WAV files to 12-bit during processing:

```python
# After trimming and resampling, add quantization:
from services.dsp.quantize_to_12bit import quantize_to_12bit_ei_format

# ... existing code ...
y, sr = librosa.load(src, sr=SAMPLE_RATE, ...)
y = np.clip(y, -1.0, 1.0)

# Convert to int16 first
y_int16 = np.clip(y * 32768.0, -32768, 32767).astype(np.int16)

# Quantize to 12-bit (matching Pi ADC)
y_12bit = quantize_to_12bit_ei_format(y_int16)

# Scale back to 16-bit range for WAV file (but values are quantized)
y_quantized = (y_12bit * 16).astype(np.int16)

# Write as PCM_16 (but with 12-bit quantization)
wavfile.write(dst, SAMPLE_RATE, y_quantized)
```

### Step 3: Retrain Model

1. Re-process all training data with 12-bit quantization
2. Re-upload to Edge Impulse
3. Retrain the model
4. Deploy to Pi and test

## Understanding the Quantization

**12-bit ADC characteristics:**
- Range: Typically `[-2048, 2047]` for signed 12-bit
- Edge Impulse stores in `int32` container but values are 12-bit
- Only ~4096 possible values (vs 65536 for 16-bit)

**Our quantization approach:**
1. Normalize 16-bit to float32 [-1, 1]
2. Quantize to 12-bit range: multiply by 2048, round, clip
3. This produces values matching Pi's ADC output

## Verification

After quantizing, verify:
1. WAV files have reduced unique value count (similar to EI raw input)
2. Feature generation produces similar values to Pi inference
3. Model performance improves on device

## Alternative: Check Pi ADC Configuration

Before quantizing all training data, verify:
1. What ADC is the Pi actually using?
2. What is the actual bit depth?
3. Is there any pre-processing on the Pi before DSP?

If the Pi is actually using 16-bit ADC, then the issue is elsewhere (gain, filtering, etc.).

## Current Status

- ✅ Feature generation algorithm is correct
- ✅ Identified 12-bit quantization mismatch
- ✅ Created quantization script
- ⚠️ Need to update processor to quantize training data
- ⚠️ Need to retrain model with quantized data

