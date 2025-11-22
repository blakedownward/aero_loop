# Feature Generation Parity Test Report

## Summary

**Date**: 2025-01-XX  
**Test Sample**: `000000_2025-11-18_12-33.wav` (first 2 seconds)  
**Environment**: pivenv virtual environment

## Key Findings

### ✅ Feature Generation Algorithm is CORRECT

When using the **EI raw input** (`ei_raw_input.npy`), the feature generation matches Edge Impulse studio output **perfectly**:
- Max absolute difference: **0.000050** (well within tolerance of 0.0021)
- Mean absolute difference: **0.000019**
- **Status**: ✅ **PASS**

This confirms that the `feature_generator.py` implementation is correct and matches Edge Impulse's DSP pipeline.

### ⚠️ WAV File Processing Shows Discrepancy

When loading the actual WAV file and processing it, there's a significant difference:
- Max absolute difference: **0.410181** (exceeds tolerance)
- Mean absolute difference: **0.057505**
- **Status**: ❌ **FAIL**

### Root Cause Analysis

The discrepancy is **NOT** in the feature generation algorithm, but in the **raw input data**:

1. **EI Raw Input Characteristics**:
   - Format: `int32` with values in range `[-2049, 1792]`
   - Only **17 unique values** in 32,000 samples
   - Appears to be from a **12-bit ADC** or heavily quantized source
   - After decoding: range `[-0.062531, 0.054688]`, std `0.013218`

2. **WAV File Characteristics**:
   - Format: `int16` PCM (standard WAV format)
   - Full 16-bit resolution with many unique values
   - Range: `[-0.148468, 0.148438]`, std `0.018032`

3. **Comparison**:
   - Correlation between EI raw input and WAV file: **0.001502** (essentially uncorrelated)
   - Max absolute difference: **0.101593**
   - Mean absolute difference: **0.017550**

### Conclusion

The **feature generation script is working correctly**. The discrepancy between on-device performance and Edge Impulse studio is likely due to:

1. **Different input sources**: The EI raw input appears to be from a different source (12-bit ADC) than the processed WAV file (16-bit PCM).

2. **Signal mismatch**: The very low correlation (0.001502) suggests the EI raw input and WAV file may be:
   - From different time periods
   - From different recording sessions
   - Processed differently (e.g., different gain, filtering, or normalization)

3. **Quantization differences**: The EI raw input is heavily quantized (17 unique values), while the WAV file has full resolution.

## Recommendations

1. ✅ **Feature generator is correct** - No changes needed to `feature_generator.py`

2. 🔍 **Verify input source**: Ensure that when testing on-device, you're using the same raw input format that Edge Impulse studio uses. The EI raw input appears to be from a 12-bit ADC, which may differ from the processed WAV files.

3. 📊 **Test with matching inputs**: To verify on-device performance matches Edge Impulse studio:
   - Export raw input from Edge Impulse studio for the same sample
   - Use that raw input for on-device testing
   - Or ensure the on-device ADC matches the format used in Edge Impulse studio

4. 🔧 **Check ADC configuration**: If on-device performance differs, verify:
   - ADC bit depth (12-bit vs 16-bit)
   - Gain settings
   - Pre-processing steps (filtering, normalization)
   - Sample rate and timing alignment

## Test Results Details

### Feature Generation from EI Raw Input
```
Best match parameters:
- Offset: 0
- Mel flip: False
- Flatten order: F (column-major, frame-by-frame)
- Max abs diff: 0.000050
- Mean abs diff: 0.000019
- Status: ✅ PASS (within tolerance 0.0021)
```

### Feature Generation from WAV File
```
Best match parameters:
- Offset: 0
- Mel flip: False
- Flatten order: F (column-major, frame-by-frame)
- Max abs diff: 0.410181
- Mean abs diff: 0.057505
- Status: ❌ FAIL (exceeds tolerance 0.0021)
```

### Feature Statistics Comparison
```
From WAV file (first 62 frames):
- Min: 0.000000
- Max: 0.433594
- Mean: 0.117115
- Std: 0.090827

From EI studio output:
- Min: 0.000000
- Max: 0.230500
- Mean: 0.069058
- Std: 0.057607

Features with >10% relative difference: 1564/1984 (78.8%)
```

## Files Generated

- `test_feature_parity.py`: Comprehensive test script comparing WAV file and EI raw input
- `investigate_raw_input.py`: Detailed analysis of raw input format differences
- `feature_parity_report.md`: This report

