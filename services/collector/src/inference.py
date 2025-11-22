#!/usr/bin/env python3
"""
Edge Impulse parity feature pipeline + TFLite inference wrapper.

This script mirrors the feature generation used in Edge Impulse
so you can run inference on-device (e.g. Raspberry Pi) with identical preprocessing.
"""

import os
import sys
from pathlib import Path
from typing import Optional

import numpy as np
from scipy.io.wavfile import read as wav_read
from tflite_runtime.interpreter import Interpreter

# Add config directory to path for imports
COLLECTOR_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CONFIG_DIR = os.path.join(COLLECTOR_DIR, 'config')
sys.path.insert(0, CONFIG_DIR)

import session_constants as c

# ---------------------------------------------------------------------------
# Constants (must match Edge Impulse MFE settings)
# ---------------------------------------------------------------------------
SR = 16000
NFFT = 512
HOP = 512
N_MELS = 32
FRAMES = 62          # 32 mel bins * 62 frames -> 1984 features
NOISE_FLOOR_DB = -52.0

# Load mel filterbank from config path
MEL_PATH = Path(c.MEL_FILTERBANK_PATH)
if not MEL_PATH.exists():
    raise FileNotFoundError(f"Mel filterbank not found at {MEL_PATH}")
MEL_BANDS = np.load(MEL_PATH).astype(np.float32)

INFERENCING_CATEGORIES = ("aircraft", "negative")
POSITIVE_LABEL = "aircraft"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def quantize_to_12bit_ei_format(signal: np.ndarray) -> np.ndarray:
    """
    Quantize int16 signal to 12-bit format matching Edge Impulse's conversion.
    
    Edge Impulse converts 16-bit WAV files to 12-bit internally during ingestion.
    This function replicates that conversion so Pi inference matches training data.
    """
    # Normalize to float [-1, 1]
    signal_float = signal.astype(np.float32) / 32768.0
    # Quantize to 12-bit: multiply by 2048, round, clip
    signal_12bit = np.round(signal_float * 2048.0).astype(np.int32)
    # Clip to observed EI range (slightly wider than standard 12-bit)
    signal_12bit = np.clip(signal_12bit, -2049, 2047)
    return signal_12bit


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


def load_wav(path: Path) -> np.ndarray:
    """Load WAV file and verify sample rate."""
    sr, signal = wav_read(str(path))
    if sr != SR:
        raise ValueError(f"Expected {SR} Hz audio, got {sr} Hz")
    return signal


def predict_file(
    wav_path: str,
    model_path: str,
    debug: bool = False,
) -> np.ndarray:
    """
    Predict aircraft probability for a WAV file.
    
    Args:
        wav_path: Path to WAV file (16 kHz mono)
        model_path: Path to TFLite model file
        debug: Print debug information
    
    Returns:
        Array of predictions (one per segment)
    """
    signal = load_wav(Path(wav_path))
    # Quantize to 12-bit to match Edge Impulse's internal conversion
    signal = quantize_to_12bit_ei_format(signal)
    feats = mfe_ei_compatible(decode_ei_raw(signal))
    
    if debug:
        print(f"Features shape: {feats.shape}, mean={feats.mean():.4f}, std={feats.std():.4f}")

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
    preds = []
    for s in starts:
        block = feats[:, s : s + FRAMES]
        raw = run_interpreter(interpreter, block)
        preds.append(aircraft_probability(raw))

    if total_frames >= FRAMES and (total_frames % FRAMES) != 0:
        tail = feats[:, -FRAMES:]
        raw = run_interpreter(interpreter, tail)
        preds.append(aircraft_probability(raw))

    return np.array(preds, dtype=np.float32)


def max_pred(wav_path: str, model_path: str) -> float:
    """Get maximum prediction for a WAV file."""
    preds = predict_file(wav_path=wav_path, model_path=model_path)
    return np.max(preds)

