# parity_check_mfe.py
import numpy as np

# ----------------------------
# 1) PASTE edge impulse arrays
# ----------------------------
# Raw input features from EI Studio (length should be 32000)
RAW_INPUT = np.load("ei_raw_input.npy")

# Output features from EI Studio (length should be 1984, flattened 32x62)
EI_OUTPUT = np.load("ei_output.npy")

# ----------------------------
# 2) Config that must match training
# ----------------------------
SR = 16000
NFFT = 512
HOP = 512            # 32 ms
N_MELS = 32
FRAMES = 62
NOISE_FLOOR_DB = -52.0

# Load the same mel bank used in EI (sr=16k, n_fft=512, n_mels=32, fmin=50, fmax=4000)
MEL = np.load("mel_16k_512fft_32mel_50to4k.npy").astype(np.float32)  # shape (32, 257)

# ----------------------------
# 3) Helpers: input decode + EI MFE
# ----------------------------
def decode_ei_raw(raw: np.ndarray) -> np.ndarray:
    """
    EI 'Raw input features' for audio can be:
      - uint8 0..255 (unsigned 8-bit samples) -> map to float32 in [-1, 1]
      - int16 -32768..32767 -> map to float32 in [-1, 1]
    We auto-detect form by range.
    """
    x = np.asarray(raw)
    if x.size == 0:
        return x.astype(np.float32)

    # Floats: assume already normalised but clamp to [-1, 1]
    if np.issubdtype(x.dtype, np.floating):
        return np.clip(x.astype(np.float32, copy=False), -1.0, 1.0)

    # Unsigned 8-bit (common EI export for raw audio)
    if np.issubdtype(x.dtype, np.unsignedinteger) and int(x.max()) <= 255:
        y = x.astype(np.float32, copy=False)
        y = (y - 128.0) / 128.0
        return np.clip(y, -1.0, 1.0).astype(np.float32)

    # Signed integer: normalise to Q15 regardless of storage width.
    # Many EI exports store int16 audio in int32 containers (e.g. 12-bit ADC).
    xi = x.astype(np.int64, copy=False)
    max_abs = float(np.max(np.abs(xi)))
    if max_abs == 0.0:
        return np.zeros_like(xi, dtype=np.float32)

    # Choose divisor: treat anything above 8-bit as int16 (Q15).
    divisor = 128.0 if max_abs <= 128.0 else 32768.0
    xi = np.clip(xi, -divisor, divisor - 1)
    return (xi / divisor).astype(np.float32)

def pre_emphasis(x: np.ndarray, cof: float = 0.98) -> np.ndarray:
    if x.size == 0:
        return x.astype(np.float32)
    rolled = np.roll(x, 1).astype(np.float32, copy=False)
    x = x.astype(np.float32, copy=False)
    return x - cof * rolled

def best_flat_match(seg_32x62, ei_vec_1984, max_offset=4):
    """
    seg_32x62: (32, T>=62) EI-processed features in [0,1]
    ei_vec_1984: (1984,) EI's flattened reference (32x62)
    Tries flatten order (C/F), mel flip (False/True), and small frame offsets.
    Returns (best_vec, details_dict)
    """
    import numpy as np
    assert seg_32x62.ndim == 2 and seg_32x62.shape[0] == 32
    T = seg_32x62.shape[1]
    assert T >= 62, f"need at least 62 frames, got {T}"

    theirs = ei_vec_1984.astype(np.float32, copy=False)
    best = None

    for off in range(0, min(max_offset+1, T-62+1)):
        window = seg_32x62[:, off:off+62]           # (32,62)
        for mel_flip in (False, True):
            W = window[::-1, :] if mel_flip else window
            # two flatten conventions:
            #  - 'C': row-major (mel-major)
            #  - 'F': column-major (time-major: frame-by-frame)
            for order in ('C', 'F'):
                ours = W.reshape(-1, order=order)
                diff = np.abs(ours - theirs)
                score = (float(diff.max()), float(diff.mean()))
                if (best is None) or (score < best['score']):
                    best = {
                        'score': score,
                        'offset': off,
                        'mel_flip': mel_flip,
                        'order': order,
                        'ours': ours.copy(),
                    }
    return best['ours'], best


def mfe_ei_compatible(signal_f32_mono_16k: np.ndarray) -> np.ndarray:
    """
    EI MFE (no librosa), returning **EI-normalized** features in [0,1],
    with EI's 8-bit quantize/dequant baked in.
    Output shape: (32, T)
    """
    x = signal_f32_mono_16k.astype(np.float32, copy=False)

    # EI uses pre-emphasis for modern MFE
    x = pre_emphasis(x, 0.98)

    # Framing (rectangular window, no zero padding) to mirror Edge Impulse speechpy fork
    n_frames = 1 + max(0, (x.shape[0] - NFFT) // HOP)
    if n_frames <= 0:
        raise RuntimeError("input too short for configured STFT")

    S_pow = np.empty((NFFT // 2 + 1, n_frames), dtype=np.float32)
    for i in range(n_frames):
        start = i * HOP
        frame = x[start : start + NFFT]
        spec = np.fft.rfft(frame, n=NFFT, norm=None)
        # speechpy.power_spectrum => |FFT|^2 / fft_points
        S_pow[:, i] = (np.abs(spec) ** 2 / NFFT).astype(np.float32)

    # mel projection
    M_pow = MEL @ S_pow                 # (32, T)

    # avoid log(0), to dB
    np.clip(M_pow, 1e-30, None, out=M_pow)
    M_db = 10.0 * np.log10(M_pow, dtype=np.float32)

    # EI normalization: (mfe - noise_floor) / ((-noise_floor)+12), clip [0,1]
    rng = (-NOISE_FLOOR_DB) + 12.0
    M_n = (M_db - NOISE_FLOOR_DB) / rng
    np.clip(M_n, 0.0, 1.0, out=M_n)

    # EI quantize/dequant to 8-bit
    M_q = np.rint(M_n * 256.0).astype(np.uint16)  # safe intermediate
    np.clip(M_q, 0, 255, out=M_q)
    M_out = (M_q.astype(np.float32) / 256.0)      # back to float32 [0,1]

    return M_out

# ----------------------------
# 4) Run parity test
# ----------------------------
def main():
    assert RAW_INPUT.size == 32000, f"Expected 32000 raw samples, got {RAW_INPUT.size}"
    assert EI_OUTPUT.size == 1984, f"Expected 1984 output features, got {EI_OUTPUT.size}"

    x = decode_ei_raw(RAW_INPUT)  # float32 mono [-1,1] @ 16k (EI exports are already 16k for audio)
    feats = mfe_ei_compatible(x)  # (32, T)

    seg_vec, info = best_flat_match(feats, EI_OUTPUT, max_offset=6)
    print("best:", info)
    print(f"max_abs_diff={info['score'][0]:.6f}, mean_abs_diff={info['score'][1]:.6f}")

    # Take the first 62 frames (2.0 s @ hop=512 & win=512)
    if feats.shape[1] < FRAMES:
        raise RuntimeError(f"Not enough frames: got T={feats.shape[1]}, need {FRAMES}")
    seg = feats[:, :FRAMES].astype(np.float32)       # (32, 62)
    ours = seg.reshape(-1, order='F')   # (32, 62) -> (1984,) stacking columns

    theirs = EI_OUTPUT.astype(np.float32)

    # Compare
    abs_diff = np.abs(ours - theirs)
    print(f"ours.shape={ours.shape}, theirs.shape={theirs.shape}")
    print(f"max_abs_diff={abs_diff.max():.6f}, mean_abs_diff={abs_diff.mean():.6f}")

    # EI features are multiples of 1/256 ≈ 0.00390625; allow ≈ half a bin
    tol = 0.0021
    if abs_diff.max() <= tol:
        print("[PASS] parity within tolerance")
    else:
        # helpful stats to see if mismatch is global or band-limited
        print("[FAIL] parity mismatch")
        # show a few largest offenders
        topk = np.argsort(-abs_diff)[:10]
        for k in topk:
            print(f"idx {k:4d}: ours={ours[k]:.6f} | theirs={theirs[k]:.6f} | diff={abs_diff[k]:.6f}")

if __name__ == "__main__":
    main()
