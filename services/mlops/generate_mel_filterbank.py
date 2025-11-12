#!/usr/bin/env python3
# generate_mel_filterbank.py

"""
Utility to synthesize the Mel filterbank that Edge Impulse uses for
audio feature extraction.  The implementation sticks to numpy/scipy
and mirrors the triangle construction used inside the DSP blocks.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Iterable

import numpy as np


def _hz_to_mel_slaney(hz: np.ndarray) -> np.ndarray:
    """Convert frequencies in Hz to the Slaney mel scale."""
    hz = np.asarray(hz, dtype=np.float64)
    return 2595.0 * np.log10(1.0 + hz / 700.0)


def _mel_to_hz_slaney(mel: np.ndarray) -> np.ndarray:
    """Inverse Slaney mel-to-Hz conversion."""
    mel = np.asarray(mel, dtype=np.float64)
    return 700.0 * (10.0 ** (mel / 2595.0) - 1.0)


def _triangular_weights(bin_edges: np.ndarray, n_bins: int) -> np.ndarray:
    """Internal helper that builds the un-normalised triangular mel filters."""
    n_mels = len(bin_edges) - 2
    weights = np.zeros((n_mels, n_bins), dtype=np.float64)

    for band in range(1, n_mels + 1):
        left, centre, right = bin_edges[band - 1], bin_edges[band], bin_edges[band + 1]

        # Guard against flat / overlapping triangles. EI's toolchain clips in the same way.
        if centre <= left:
            centre = left + 1
        if right <= centre:
            right = centre + 1
        centre = min(centre, n_bins - 1)
        right = min(right, n_bins - 1)

        if centre > left:
            ramp_up = (np.arange(left, centre, dtype=np.float64) - left) / (
                (centre - left) or 1
            )
            weights[band - 1, left:centre] = ramp_up

        weights[band - 1, centre] = 1.0

        if right > centre:
            ramp_down = (right - np.arange(centre + 1, right + 1, dtype=np.float64)) / (
                (right - centre) or 1
            )
            weights[band - 1, centre + 1 : right + 1] = np.maximum(ramp_down, 0.0)

    return weights


def build_mel_filterbank(
    sr: int = 16000,
    n_fft: int = 512,
    n_mels: int = 32,
    fmin: float = 50.0,
    fmax: float | None = 4000.0,
    norm: str | None = None,
) -> np.ndarray:
    """
    Construct a mel filterbank matrix with shape (n_mels, 1 + n_fft // 2).

    Parameters match Edge Impulse defaults.  The implementation deliberately
    mirrors librosa's Slaney mode so that we can regenerate EI's shipped
    filterbanks without depending on librosa.
    """
    if n_fft <= 0:
        raise ValueError("n_fft must be positive")
    if n_mels <= 0:
        raise ValueError("n_mels must be positive")
    if fmin < 0.0:
        raise ValueError("fmin must be >= 0")

    if fmax is None:
        fmax = sr / 2.0
    else:
        fmax = min(fmax, sr / 2.0)

    n_fft_bins = 1 + n_fft // 2
    fft_freqs = np.linspace(0.0, sr / 2.0, n_fft_bins, dtype=np.float64)

    mel_edges = np.linspace(
        _hz_to_mel_slaney(np.array([fmin], dtype=np.float64))[0],
        _hz_to_mel_slaney(np.array([fmax], dtype=np.float64))[0],
        n_mels + 2,
        dtype=np.float64,
    )
    hz_edges = _mel_to_hz_slaney(mel_edges)

    bin_edges = np.floor((n_fft + 1) * hz_edges / sr).astype(int)
    bin_edges = np.clip(bin_edges, 0, n_fft_bins - 1)

    mel_filters = _triangular_weights(bin_edges, n_fft_bins)

    norm_mode = (norm or "").lower()
    if norm_mode in ("slaney", "area"):
        area = hz_edges[2 : n_mels + 2] - hz_edges[:n_mels]
        # Avoid division by zero for extremely narrow filters.
        area = np.where(area == 0.0, 1.0, area)
        mel_filters *= (2.0 / area)[:, np.newaxis]
    elif norm_mode in ("", "none"):
        pass
    else:
        raise ValueError(f"Unsupported norm '{norm}'. Use 'slaney' or 'none'.")

    return mel_filters.astype(np.float32)


def save_mel_bank(
    out_npy: str,
    out_json: str | None = None,
    *,
    sr: int = 16000,
    n_fft: int = 512,
    n_mels: int = 32,
    fmin: float = 50.0,
    fmax: float | None = 4000.0,
    norm: str | None = None,
) -> np.ndarray:
    """Persist the mel filterbank to disk (optionally also to JSON for inspection)."""
    matrix = build_mel_filterbank(
        sr=sr,
        n_fft=n_fft,
        n_mels=n_mels,
        fmin=fmin,
        fmax=fmax,
        norm=norm,
    )

    out_path = Path(out_npy)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    np.save(out_path, matrix.astype(np.float32))

    if out_json is not None:
        payload = {
            "sr": sr,
            "n_fft": n_fft,
            "n_mels": n_mels,
            "fmin": fmin,
            "fmax": fmax,
            "norm": norm,
            "matrix": matrix.tolist(),
        }
        with open(out_json, "w", encoding="utf-8") as handle:
            json.dump(payload, handle)

    print(f"Saved mel bank to: {out_path.resolve()}")
    print(f"shape={matrix.shape}, dtype={matrix.dtype}")
    return matrix


def _parse_args(argv: Iterable[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate a Slaney-style mel filterbank.")
    parser.add_argument("--sr", type=int, default=16000, help="Audio sample rate (Hz)")
    parser.add_argument("--n_fft", type=int, default=512, help="FFT size")
    parser.add_argument("--n_mels", type=int, default=32, help="Number of mel bands")
    parser.add_argument("--fmin", type=float, default=50.0, help="Lowest band edge (Hz)")
    parser.add_argument(
        "--fmax", type=float, default=4000.0, help="Highest band edge (Hz); <= sr/2"
    )
    parser.add_argument(
        "--norm",
        type=str,
        default="none",
        choices=("slaney", "area", "none"),
        help="Normalisation mode ('none' matches EI processing).",
    )
    parser.add_argument(
        "--out-npy",
        type=str,
        default="mel_16k_512fft_32mel_50to4k.npy",
        help="Output .npy path for the matrix",
    )
    parser.add_argument(
        "--out-json",
        type=str,
        default=None,
        help="Optional JSON dump for debugging",
    )
    return parser.parse_args(argv)


def main(argv: Iterable[str] | None = None) -> None:
    args = _parse_args(argv)
    save_mel_bank(
        out_npy=args.out_npy,
        out_json=args.out_json,
        sr=args.sr,
        n_fft=args.n_fft,
        n_mels=args.n_mels,
        fmin=args.fmin,
        fmax=args.fmax,
        norm=args.norm if args.norm != "none" else None,
    )


if __name__ == "__main__":
    main()
