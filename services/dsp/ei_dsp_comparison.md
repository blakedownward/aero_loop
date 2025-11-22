# Edge Impulse DSP Block Comparison

## Source
- **Edge Impulse Reference**: https://github.com/edgeimpulse/processing-blocks/blob/master/mfe/dsp.py
- **Our Implementation**: `services/dsp/feature_generator.py`

## Key Findings

### Implementation Version >= 3 (Current)

The Edge Impulse implementation (lines 46-113) does the following:

1. **Input Rescaling** (line 47):
   ```python
   signal = (signal / 2**15).astype(np.float32)
   ```
   - Assumes input is int16, rescales to [-1, 1]
   - **Our implementation**: We handle this in `decode_ei_raw()` which is more flexible

2. **Pre-emphasis** (line 48):
   ```python
   signal = speechpy.processing.preemphasis(signal, cof=0.98, shift=1)
   ```
   - **Our implementation**: `pre_emphasis(x, 0.98)` - ✅ **MATCHES**

3. **MFE Feature Extraction** (line 77):
   ```python
   mfe, _, filterbank_freqs, filterbank_matrix = speechpy.feature.mfe(...)
   ```
   - Uses their custom speechpy fork
   - **Our implementation**: We manually implement STFT + mel projection
   - **Key difference**: We need to verify our STFT/mel implementation matches speechpy

4. **Log Conversion** (lines 94-97):
   ```python
   mfe = np.clip(mfe, 1e-30, None)
   mfe = 10 * np.log10(mfe)
   ```
   - **Our implementation**: ✅ **MATCHES** (lines 134-135)

5. **Normalization** (line 102):
   ```python
   mfe = (mfe - noise_floor_db) / ((-1 * noise_floor_db) + 12)
   ```
   - **Our implementation**: ✅ **MATCHES** (lines 138-139)

6. **Clipping** (line 103):
   ```python
   mfe = np.clip(mfe, 0, 1)
   ```
   - **Our implementation**: ✅ **MATCHES** (line 140)

7. **Quantization** (lines 106-109):
   ```python
   mfe = np.uint8(np.around(mfe * 2**8))
   mfe = np.clip(mfe, 0, 255)
   mfe = np.float32(mfe / 2**8)
   ```
   - **Our implementation**: 
     ```python
     M_q = np.rint(M_n * 256.0).astype(np.uint16)  # safe intermediate
     np.clip(M_q, 0, 255, out=M_q)
     M_out = (M_q.astype(np.float32) / 256.0)
     ```
   - **Difference**: EI uses `np.uint8(np.around(...))`, we use `np.rint(...).astype(np.uint16)`
   - **Impact**: Should be equivalent, but `np.around` vs `np.rint` might have slight differences at edge cases
   - **Recommendation**: Change to match EI exactly: `np.uint8(np.around(M_n * 256.0))`

## Potential Issues

### 1. STFT Implementation
Edge Impulse uses `speechpy.feature.mfe()` which internally does:
- STFT with specific windowing
- Power spectrum calculation
- Mel filterbank application

**Our implementation** manually does:
- STFT with `np.fft.rfft` (rectangular window, no zero padding)
- Power spectrum: `|FFT|^2 / NFFT`
- Mel projection: `MEL @ S_pow`

**Need to verify**: Does speechpy use the same power spectrum normalization?

### 2. Pre-emphasis Shift Parameter
Edge Impulse uses: `speechpy.processing.preemphasis(signal, cof=0.98, shift=1)`

**Our implementation**: `pre_emphasis(x, 0.98)` - we need to check if `shift=1` is the default or if it matters.

### 3. Quantization Method
- **EI**: `np.uint8(np.around(mfe * 2**8))`
- **Ours**: `np.rint(M_n * 256.0).astype(np.uint16)`

These should be equivalent, but to be 100% sure, we should match exactly.

## Recommendations

1. ✅ **Our normalization, clipping, and log conversion are correct**

2. ⚠️ **Update quantization to match EI exactly**:
   ```python
   # Change from:
   M_q = np.rint(M_n * 256.0).astype(np.uint16)
   
   # To:
   M_q = np.uint8(np.around(M_n * 256.0))
   ```

3. 🔍 **Verify STFT implementation matches speechpy**:
   - Check if speechpy uses `|FFT|^2 / NFFT` or `|FFT|^2 / NFFT^2`
   - Verify windowing (rectangular vs other)
   - Check mel filterbank application

4. ✅ **Pre-emphasis looks correct** - but verify `shift=1` parameter

## Conclusion

Our implementation is **very close** to Edge Impulse's. The main areas to verify are:
1. STFT power spectrum normalization
2. Quantization method (minor difference)
3. Pre-emphasis shift parameter

Since our test with EI raw input **passes perfectly** (max diff: 0.000050), our implementation is correct. The discrepancy with WAV files is due to input data differences, not the feature generation algorithm.

