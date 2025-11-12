This doc explains the suspected class-inversion issue plus the secondary DSP mismatches that may affect parity with Edge Impulse Studio.

---

## 🔍 Model Parity & Class Inversion Notes

**Context:** Edge Impulse (EI) binary classifier deployed on a 32-bit Raspberry Pi 4 with a hand-rolled DSP pipeline replicating EI’s MFE/MFCC blocks.
**Observed Issue:** Predictions appear “reversed” — aircraft clips score low, while negative/noise clips score high.

---

### 1. Likely Root Cause: **Single-Output Sigmoid ≠ “Positive vs Negative”**

#### What’s happening

* EI exports a **single-node sigmoid** for binary models.
* That node represents **one specific class**, not “positive” in a generic sense.
* The class it maps to is determined by **label ordering during training/export**, not by meaning.

Example:

| EI label order             | Tensor output node meaning                     | Correct probability mapping |
| -------------------------- | ---------------------------------------------- | --------------------------- |
| `["aircraft", "negative"]` | `p = sigmoid(logit)` may equal **P(negative)** | `P(aircraft) = 1 - p`       |
| `["negative", "aircraft"]` | `p` may equal **P(aircraft)**                  | `P(aircraft) = p`           |

If your aircraft samples consistently yield **low values**, the model is outputting `P(negative)`.

#### Fix

```python
p_negative = float(pred[0])
p_aircraft = 1.0 - p_negative
is_aircraft = p_aircraft >= THRESHOLD     # default 0.5 or calibrated
```

---

### 2. DSP Parity Issues (Won’t *invert* classes, but will erode accuracy)

#### 2.1 Pre-emphasis uses `np.roll`, causing wrap-around

Your current code wraps the last sample into the first frame, which EI does **not** do.

```python
# Problematic (wraps end → start)
sig = np.roll(x, 1)
sig[0] = 0.0
sig = x - a * sig
```

✅ Correct EI-compatible version:

```python
def pre_emphasis(x, a=0.98):
    y = x.astype(np.float32, copy=True)
    y[1:] = y[1:] - a * y[:-1]
    # y[0] is untouched
    return y
```

---

#### 2.2 Missing per-frame window (Hann/Hamming)

EI applies a window before the FFT; your `mfe_ei_compatible()` path uses raw frames.

✅ Fix (inside frame loop):

```python
frame = frame * np.hamming(len(frame))
```

---

#### 2.3 Normalization bug in `logmelspec()`

You normalize `sig = (signal - mean)/std` but compute STFT on the **unnormalized** `signal`.

```python
# Current (bug)
Z = sft.stft(signal)

# Correct
Z = sft.stft(sig)
```

Even if that function isn’t used now, it will break parity if re-enabled.

---

### 3. Validation Strategy Before Changing Anything

| Step | What to do                                  | Expected outcome                                   |
| ---- | ------------------------------------------- | -------------------------------------------------- |
| 1    | Run 10 known aircraft clips, log raw `pred` | Should be **low** if node = `P(negative)`          |
| 2    | Compute `1 - pred` and re-log               | Should flip correctly                              |
| 3    | Sweep threshold on a small labeled set      | Confirms optimal threshold (not always 0.5)        |
| 4    | Patch DSP issues one-by-one                 | Compare deltas in confidence, not just correctness |

---

### 4. Optional Helper Script (Threshold Sweep)

If useful, I can supply a `calibrate_threshold.py` script that:

* loads N aircraft + N negative WAVs,
* runs your exact DSP → model pipeline,
* computes ROC, Youden’s J index, best threshold, etc.

---

### 5. TL;DR

| Issue                     | Impact              | Fix                       |
| ------------------------- | ------------------- | ------------------------- |
| Output node = wrong class | **Inverts result**  | `p_aircraft = 1 - p`      |
| Pre-emphasis wrap         | Small accuracy loss | Replace with true EI form |
| Missing windowing         | SNR/feature drift   | Apply Hann/Hamming        |
| Wrong signal used in STFT | Silent drift        | Use normalized `sig`      |

If inversion disappears after `1 - p`, the rest becomes **accuracy tuning**, not correctness.

---

