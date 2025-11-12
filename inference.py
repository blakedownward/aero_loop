#!/usr/bin/python

# import statements
import os
import sys

# print(f'CWD: {os.getcwd()}')
# print(f'path0: {sys.path[0]}')


import json
import numpy as np
from scipy.signal import ShortTimeFFT
from scipy.signal.windows import hann
from scipy.io.wavfile import read

from tflite_runtime.interpreter import Interpreter

print(sys.modules['tflite_runtime'])

# temp imports for debugging
# import tensorflow as tf


def debug_stats(S_db: np.ndarray, label="feat"):
    p_floor = float(np.mean(S_db <= -52.0)) * 100.0
    print(
        f"[{label}] shape={S_db.shape}, mean={S_db.mean():.2f} dB, "
        f"std={S_db.std():.2f}, min={S_db.min():.1f}, max={S_db.max():.1f}, "
        f"%@floor={p_floor:.1f}%")



def lite_predict(interpreter, input_features):
    # interpreter.allocate_tensors() should be called ONCE outside
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()

    idx = input_details[0]['index']
    exp_shape = tuple(input_details[0]['shape'])   # (1, 1984)

    x = input_features
    # x is (mel, frames). Assert mel=32.
    if x.ndim != 2:
        raise ValueError(f"Expected 2D features (mel, frames), got {x.shape}")
    mel, T = x.shape
    if mel != 32:
        raise ValueError(f"Expected 32 mel bins, got {mel}")

    # Flatten to (1, 1984)
    x = x.reshape(1, -1).astype(np.float32)
    if x.shape != exp_shape:
        raise ValueError(f"Input shape {x.shape} != expected {exp_shape}")

    interpreter.set_tensor(idx, x)
    interpreter.invoke()
    pred = interpreter.get_tensor(output_details[0]['index'])
    return pred


# import the mel-filterbank
# this table was built specifically with "sr=16000, n_fft=512, n_mels=32, fmin=50, fmax=4000"
MEL_BANDS = np.load('/home/protopi/ten90audio/ten90audio/mel_16k_512fft_32mel_50to4k.npy').astype(np.float32)


def pre_emphasis(x: np.ndarray, coeff: float = 0.97) -> np.ndarray:

    if x.size == 0:
        return x.astype(np.float32)

    y = np.empty_like(x, dtype=np.float32)

    y[0] = x[0]
    y[1:] = x[1:] - coeff * x[:-1]

    return y
    
    
def segment_db_zeroref(seg_db, floor_db=-52.0):
    seg = seg_db.astype(np.float32, copy=True)
    seg -= np.max(seg)
    np.maximum(seg, floor_db, out=seg)
    return seg
    
    
def loudness_normalize(x, target_dbfs=-20.0, eps=1e-8):
    rms = np.sqrt(np.mean(x**2) + eps)
    target_rms = 10**(target_dbfs/20.0)
    g = target_rms / max(rms, eps)
    y = np.clip(x * g, -1.0, 1.0)
    return y.astype(np.float32)
    

def logmelspec(
    signal,
    mel_bands=MEL_BANDS,
    sr: int = 16000,
    n_fft: int = 512,
    win_length: int = 512,
    hop: int = 512,
    db_floor: float = -52.0,
    use_preemph: bool = True,
    drop_edge_frames: bool = False
) -> np.ndarray:

    if np.issubdtype(signal.dtype, np.integer):
        sig = signal.astype(np.float32) / 32768.0
    else:
        sig = signal.astype(np.float32)

    sig -= np.mean(sig)
    
    sig = loudness_normalize(sig, -20.0)

    if use_preemph:
        sig = pre_emphasis(sig, 0.97)



    window = hann(M=win_length, sym=False)

    sft = ShortTimeFFT(
        win=window,
        hop=hop,
        fs=sr,
        fft_mode='onesided',
        scale_to='magnitude')

    Z = sft.stft(sig)

    S_power = (np.abs(Z)**2).astype(np.float32)

    mel_power = mel_bands @ S_power

    if drop_edge_frames and mel_power.shape[1] >= 2:
        mel_power = mel_power[:, 1:-1]

    floor_lin = 10.0 ** (db_floor / 10.0)
    np.maximum(mel_power, floor_lin, out=mel_power)
    mel_db = 10.0 * np.log10(mel_power, dtype=np.float32)


    return mel_db.astype(np.float32)


def predict_file(filepath: str):
    sr, signal = read(filepath)
    
    print(f"Sample Rate: {sr}")

    # Build features: shape (32, T)
    S = logmelspec(signal=signal)  # drop_edge_frames=False
    debug_stats(S, "logmel")
    n_mels, total_T = S.shape
    assert n_mels == 32, f"Expected 32 mels, got {n_mels}"



    # Derive frames_per_segment from model input (1984 / 32 = 62)
    interpreter = Interpreter(model_path="/home/protopi/ten90audio/ten90audio/tflite_learn_815690_3.tflite")
    interpreter.allocate_tensors()
    input_len = int(interpreter.get_input_details()[0]['shape'][1])
    frames_per_segment = input_len // 32    # 62
    print("Feature shape:", S.shape)            # expect (32, T)
    print("Frames/seg:", frames_per_segment)    # expect 62

    # Build non-overlapping full segments
    starts = list(range(0, total_T - frames_per_segment + 1, frames_per_segment))

    preds = []
    for s in starts:
        sub_db = S[:, s:s + frames_per_segment]  # (32, 62)
        sub_db = segment_db_zeroref(sub_db)
        # x = sub_db.reshape(1, -1)
        # print(x.shape)
        y = lite_predict(interpreter, sub_db)
        preds.append(y.ravel()[0])

    # Optionally run one extra right-aligned tail if you want coverage:
    if total_T >= frames_per_segment and (total_T % frames_per_segment) != 0:
        tail = S[:, -frames_per_segment:]
        y = lite_predict(interpreter, tail)
        preds.append(y.ravel()[0])

    return np.array(preds, dtype=np.float32)




    
    
    
    
#!/usr/bin/env python3
"""
Edge Impulse parity feature pipeline + TFLite inference wrapper.

This script mirrors the feature generation used in `services/dsp/feature_generator.py`
so you can run inference on-device (e.g. Raspberry Pi) with identical preprocessing.
"""

# from __future__ import annotations

import argparse
import os
import sys
from collections import defaultdict
from pathlib import Path
from typing import DefaultDict, Iterable, List, Optional, Tuple

import numpy as np
from scipy.io.wavfile import read as wav_read
from tflite_runtime.interpreter import Interpreter


# ---------------------------------------------------------------------------
# Constants (must match Edge Impulse MFE settings)
# ---------------------------------------------------------------------------
SR = 16000
NFFT = 512
HOP = 512
N_MELS = 32
FRAMES = 62          # 32 mel bins * 62 frames -> 1984 features
NOISE_FLOOR_DB = -52.0

_DEFAULT_MEL_PATH = Path(__file__).resolve().with_name("mel_16k_512fft_32mel_50to4k.npy")
MEL_PATH = Path(os.environ.get("EI_MEL_PATH", _DEFAULT_MEL_PATH))
_DEFAULT_MODEL_PATH = Path(
    os.environ.get(
        "EI_MODEL_PATH",
        Path(__file__).resolve().with_name("tflite_learn_815690_3.tflite"),
    )
)
print(f"Mel Path: {MEL_PATH}")
MEL_BANDS = np.load(MEL_PATH).astype(np.float32)
INFERENCING_CATEGORIES = ("aircraft", "negative")
POSITIVE_LABEL = "aircraft"


# ---------------------------------------------------------------------------
# Helpers copied from parity checker
# ---------------------------------------------------------------------------
def decode_ei_raw(raw: np.ndarray) -> np.ndarray:
    """Map EI exported raw samples to float32 mono [-1, 1]."""
    x = np.asarray(raw)
    if x.size == 0:
        return x.astype(np.float32)

    if np.issubdtype(x.dtype, np.floating):
        return np.clip(x.astype(np.float32, copy=False), -1.0, 1.0)

    if np.issubdtype(x.dtype, np.unsignedinteger) and int(x.max()) <= 255:
        y = x.astype(np.float32, copy=False)
        y = (y - 128.0) / 128.0
        return np.clip(y, -1.0, 1.0).astype(np.float32)

    xi = x.astype(np.int64, copy=False)
    max_abs = float(np.max(np.abs(xi)))
    if max_abs == 0.0:
        return np.zeros_like(xi, dtype=np.float32)

    divisor = 128.0 if max_abs <= 128.0 else 32768.0
    xi = np.clip(xi, -divisor, divisor - 1)
    return (xi / divisor).astype(np.float32)


def pre_emphasis(x: np.ndarray, cof: float = 0.98) -> np.ndarray:
    if x.size == 0:
        return x.astype(np.float32)
    y = x.astype(np.float32, copy=True)
    y[1:] = y[1:] - cof * y[:-1]
    return y


def mfe_ei_compatible(signal_f32_mono_16k: np.ndarray) -> np.ndarray:
    """
    Reproduce EI's mel feature extraction pipeline.
    Returns mel features in [0,1] after EI's quantize/dequant.
    Output shape: (32, T)
    """
    x = signal_f32_mono_16k.astype(np.float32, copy=False)
    x = pre_emphasis(x, 0.98)
    window = np.hamming(NFFT).astype(np.float32)

    n_frames = 1 + max(0, (x.shape[0] - NFFT) // HOP)
    if n_frames <= 0:
        raise RuntimeError(f"Signal too short for NFFT={NFFT}, hop={HOP}")

    S_pow = np.empty((NFFT // 2 + 1, n_frames), dtype=np.float32)
    for i in range(n_frames):
        start = i * HOP
        frame = x[start : start + NFFT]
        frame = frame * window
        spec = np.fft.rfft(frame, n=NFFT, norm=None)
        S_pow[:, i] = (np.abs(spec) ** 2 / NFFT).astype(np.float32)

    M_pow = MEL_BANDS @ S_pow
    np.clip(M_pow, 1e-30, None, out=M_pow)
    M_db = 10.0 * np.log10(M_pow, dtype=np.float32)

    rng = (-NOISE_FLOOR_DB) + 12.0
    M_n = (M_db - NOISE_FLOOR_DB) / rng
    np.clip(M_n, 0.0, 1.0, out=M_n)

    M_q = np.rint(M_n * 256.0).astype(np.uint16)
    np.clip(M_q, 0, 255, out=M_q)
    return M_q.astype(np.float32) / 256.0


# ---------------------------------------------------------------------------
# Inference helpers
# ---------------------------------------------------------------------------
def debug_stats(mat: np.ndarray, label: str = "feat") -> None:
    p_floor = float(np.mean(mat <= 0.0)) * 100.0
    print(
        f"[{label}] shape={mat.shape}, mean={mat.mean():.4f}, "
        f"std={mat.std():.4f}, min={mat.min():.4f}, max={mat.max():.4f}, "
        f"%@0={p_floor:.1f}%"
    )


def load_wav(path: Path) -> np.ndarray:
    sr, signal = wav_read(str(path))
    if sr != SR:
        raise ValueError(f"Expected {SR} Hz audio, got {sr} Hz")
    return signal


def run_interpreter(interpreter: Interpreter, feature_block: np.ndarray) -> float:
    """feature_block: (32, 62) array in [0,1]."""
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()

    idx = input_details[0]["index"]
    expected_shape = tuple(input_details[0]["shape"])  # (1, 1984)

    if feature_block.shape != (N_MELS, FRAMES):
        raise ValueError(f"Expected (32, 62) features, got {feature_block.shape}")

    x = feature_block.reshape(1, -1).astype(np.float32)
    if x.shape != expected_shape:
        raise ValueError(f"Interpreter expects {expected_shape}, got {x.shape}")

    interpreter.set_tensor(idx, x)
    interpreter.invoke()
    pred = interpreter.get_tensor(output_details[0]["index"])
    return float(pred.ravel()[0])


def aircraft_probability(raw_pred: float, target_label: str = POSITIVE_LABEL) -> float:
    """Map raw sigmoid output to aircraft probability based on label ordering."""
    if len(INFERENCING_CATEGORIES) != 2:
        return float(raw_pred)
    if target_label not in INFERENCING_CATEGORIES:
        raise ValueError(f"Label '{target_label}' not in {INFERENCING_CATEGORIES}")
    idx = INFERENCING_CATEGORIES.index(target_label)
    if idx == 0:
        return float(raw_pred)
    if idx == 1:
        return 1.0 - float(raw_pred)
    return float(raw_pred)


def predict_file(
    wav_path: Path,
    model_path: Path,
    debug: bool = False,
) -> np.ndarray:
    signal = load_wav(wav_path)
    feats = mfe_ei_compatible(decode_ei_raw(signal))
    if debug:
        debug_stats(feats, "mfe")

    interpreter = Interpreter(model_path=str(model_path))
    interpreter.allocate_tensors()
    input_len = int(interpreter.get_input_details()[0]["shape"][1])
    frames_per_segment = input_len // N_MELS
    if frames_per_segment != FRAMES:
        raise ValueError(f"Model expects {frames_per_segment} frames, expected {FRAMES}")

    total_frames = feats.shape[1]
    if total_frames < FRAMES:
        raise ValueError(f"Not enough frames ({total_frames}) for a single prediction.")

    starts = list(range(0, total_frames - FRAMES + 1, FRAMES))
    preds: List[float] = []
    for s in starts:
        block = feats[:, s : s + FRAMES]
        raw = run_interpreter(interpreter, block)
        preds.append(aircraft_probability(raw))

    if total_frames >= FRAMES and (total_frames % FRAMES) != 0:
        tail = feats[:, -FRAMES:]
        raw = run_interpreter(interpreter, tail)
        preds.append(aircraft_probability(raw))

    return np.array(preds, dtype=np.float32)


def max_pred(wav_path: str, model_path: str):
    preds = predict_file(wav_path=wav_path, model_path=model_path)
    # print(preds)
    return np.max(preds)


def multi_pred(wav_path: str, model_path: str):
    preds = predict_file(wav_path=wav_path, model_path=model_path)
    print(f"Min: {preds.min()}")
    print(f"Max: {preds.max()}")
    print(f"Mean: {preds.mean()}")




# ---------------------------------------------------------------------------
# CLI helpers
# ---------------------------------------------------------------------------
def parse_args(argv: Optional[Iterable[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run EI parity MFE + TFLite inference or evaluate a manifest."
    )
    parser.add_argument(
        "wav",
        type=Path,
        nargs="?",
        help="Path to a single WAV file (16 kHz mono) to score.",
    )
    parser.add_argument(
        "--model",
        type=Path,
        default=_DEFAULT_MODEL_PATH,
        help="Path to TFLite model (default: %(default)s).",
    )
    parser.add_argument(
        "--manifest",
        type=Path,
        help="Optional CSV file with 'label,path' rows for batch evaluation.",
    )
    parser.add_argument(
        "--max",
        action="store_true",
        help="Print only the maximum probability when running a single file.",
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Print feature stats while generating mel features.",
    )
    args = parser.parse_args(argv)
    if not args.manifest and args.wav is None:
        parser.error("Provide either a WAV path or --manifest file.")
    return args


def load_manifest(manifest_path: Path) -> List[Tuple[str, Path]]:
    entries: List[Tuple[str, Path]] = []
    with manifest_path.open("r", encoding="utf-8") as handle:
        for idx, raw_line in enumerate(handle, start=1):
            line = raw_line.split("#", 1)[0].strip()
            if not line:
                continue
            if "," not in line:
                raise ValueError(
                    f"Manifest {manifest_path} line {idx} missing comma: {raw_line.strip()}"
                )
            label, wav = [part.strip() for part in line.split(",", 1)]
            if not label or not wav:
                raise ValueError(
                    f"Manifest {manifest_path} line {idx} must define 'label,path'."
                )
            entries.append((label, Path(wav)))
    if not entries:
        raise ValueError(f"Manifest {manifest_path} contained no rows.")
    return entries


def evaluate_manifest(
    manifest_entries: List[Tuple[str, Path]],
    model_path: Path,
    debug: bool = False,
) -> None:
    per_label: DefaultDict[str, List[float]] = defaultdict(list)
    print(f"Evaluating {len(manifest_entries)} clips with model {model_path}")
    for label, wav_path in manifest_entries:
        preds = predict_file(wav_path=wav_path, model_path=model_path, debug=debug)
        clip_min = float(preds.min())
        clip_max = float(preds.max())
        clip_mean = float(preds.mean())
        per_label[label].append(clip_max)
        print(
            f"{label:10s} {wav_path} -> min={clip_min:.4f} max={clip_max:.4f} "
            f"mean={clip_mean:.4f} segments={len(preds)}"
        )
    print("\nAggregate max-probability stats per label:")
    for label, values in per_label.items():
        arr = np.array(values, dtype=np.float32)
        print(
            f"{label:10s} clips={len(values):3d} "
            f"max/mean={arr.mean():.4f} max/min={arr.min():.4f} max/max={arr.max():.4f}"
        )


def main(argv: Optional[Iterable[str]] = None) -> None:
    args = parse_args(argv)
    model_path = args.model
    if args.manifest:
        entries = load_manifest(args.manifest)
        evaluate_manifest(entries, model_path, debug=args.debug)
        return

    preds = predict_file(args.wav, model_path, debug=args.debug)
    if args.max:
        print(f"max={preds.max():.6f}")
    else:
        print("predictions:", preds.tolist())
        print(
            f"min={preds.min():.6f}, max={preds.max():.6f}, mean={preds.mean():.6f}"
        )


if __name__ == "__main__":
    main()
