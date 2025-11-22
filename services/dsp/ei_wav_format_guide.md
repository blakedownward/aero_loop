# Edge Impulse WAV File Format Guide

## The Problem

Edge Impulse's **ingestion service** preprocesses WAV files before they reach the DSP block. This preprocessing can cause differences between:
- What we upload (our processed WAV files)
- What the DSP block receives (after ingestion preprocessing)
- What we test locally (direct WAV file processing)

## Edge Impulse Ingestion Service Behavior

Based on the DSP block code analysis, Edge Impulse's ingestion service likely:

1. **Reads the WAV file** (using their own audio library)
2. **Resamples if needed** (even if already at target rate, may use different method)
3. **Converts to int16** (if not already)
4. **Normalizes** (divides by 2^15 to get float32 in [-1, 1])
5. **Passes to DSP block** as int16 values

## Our Current Process

Looking at `services/mlops/processor.py`:

```python
# Load with librosa (resamples to 16kHz, loads as float32 in [-1, 1])
y, sr = librosa.load(src, sr=SAMPLE_RATE, offset=float(start_s), duration=float(duration), mono=True)

# Clip to [-1, 1]
y = np.clip(y, -1.0, 1.0)

# Write as PCM_16
with sf.SoundFile(dst, mode='w', samplerate=SAMPLE_RATE, channels=1, 
                  subtype='PCM_16', format='WAV') as f:
    f.write(y)  # soundfile converts float32 to int16
```

## The Issue

**librosa.load()** uses a specific resampling method (default: `res_type='kaiser_best'`), which may differ from Edge Impulse's resampling method. This can cause:
- Different sample values even at the same sample rate
- Downsampling errors if Edge Impulse detects format inconsistencies
- Feature mismatches between local testing and Edge Impulse studio

## Solution: Match Edge Impulse's Expected Format

To ensure our WAV files match what Edge Impulse expects:

### Option 1: Use scipy.io.wavfile (Direct int16)

Instead of using librosa (which resamples and converts to float), we should:
1. Read the original WAV file directly as int16
2. Resample using a method that matches Edge Impulse (if needed)
3. Write directly as int16 without float conversion

### Option 2: Match librosa Resampling Method

If we must use librosa, we should:
1. Use `res_type='soxr_hq'` or `res_type='kaiser_best'` (librosa default)
2. Ensure we're not double-resampling
3. Write with explicit int16 conversion

### Option 3: Pre-process to Match EI Raw Input Format

Since Edge Impulse's raw input appears to be from a 12-bit ADC (based on our analysis), we might need to:
1. Quantize our 16-bit WAV to 12-bit range
2. Match the exact format Edge Impulse expects

## Recommended Approach

**Create WAV files that match Edge Impulse's ingestion expectations:**

1. **Sample Rate**: 16000 Hz (already correct)
2. **Format**: PCM_16 (already correct)
3. **Channels**: Mono (already correct)
4. **Resampling Method**: Use the same method Edge Impulse uses

The key is ensuring that when Edge Impulse reads our WAV file, it produces the same int16 values that we expect.

## Testing

To verify our WAV files match Edge Impulse's format:

1. Upload a sample to Edge Impulse
2. Export the "raw input features" from Edge Impulse studio
3. Compare with our local WAV file processing
4. Adjust our preprocessing to match

## Current Status

Our WAV files are correctly formatted (16kHz, PCM_16, mono), but the **resampling method** used by librosa may differ from Edge Impulse's ingestion service, causing the downsampling errors.

