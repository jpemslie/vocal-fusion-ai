"""
VocalFusion pure-numpy audio utilities.

These functions have no dependencies on heavy ML libraries (torch, demucs, etc.)
and can be imported and tested in any environment.
"""

import hashlib
import json
import os
from pathlib import Path

import numpy as np
from scipy.signal import butter, sosfilt

from vf.constants import SR


# ---------------------------------------------------------------------------
# File identity / fingerprint cache
# ---------------------------------------------------------------------------

def file_id(path: str) -> str:
    """Stable fingerprint for an audio file: MD5 of (size + first 8 KB)."""
    stat = os.stat(path)
    with open(path, "rb") as f:
        head = f.read(8192)
    return hashlib.md5(str(stat.st_size).encode() + head).hexdigest()[:12]


def fp_path(fid: str, cache_dir: str) -> Path:
    return Path(cache_dir) / fid / "_fingerprint.json"


def load_fp(fid: str, cache_dir: str) -> dict:
    p = fp_path(fid, cache_dir)
    if p.exists():
        try:
            return json.loads(p.read_text())
        except Exception:
            return {}
    return {}


def save_fp(fid: str, cache_dir: str, data: dict) -> None:
    p = fp_path(fid, cache_dir)
    p.parent.mkdir(parents=True, exist_ok=True)
    existing = load_fp(fid, cache_dir)
    existing.update({
        k: (float(v) if isinstance(v, (np.floating, np.integer)) else v)
        for k, v in data.items()
    })
    p.write_text(json.dumps(existing, indent=2))


# ---------------------------------------------------------------------------
# Basic signal utilities
# ---------------------------------------------------------------------------

def to_mono(y: np.ndarray) -> np.ndarray:
    """(T, 2) → (T,)  or  (T,) → (T,), always float32."""
    return y.mean(axis=1).astype(np.float32) if y.ndim == 2 else y.astype(np.float32)


def rms(y: np.ndarray) -> float:
    """RMS power (includes near-zero floor to avoid log(0))."""
    return float(np.sqrt(np.mean(y ** 2) + 1e-12))


def active_rms(y: np.ndarray, threshold_db: float = -48.0) -> float:
    """RMS computed only over samples above threshold_db relative to peak."""
    mono = to_mono(y)
    cutoff = float(np.max(np.abs(mono)) + 1e-12) * 10 ** (threshold_db / 20)
    active = mono[np.abs(mono) > cutoff]
    return float(np.sqrt(np.mean(active ** 2) + 1e-12)) if len(active) >= SR else rms(y)


# ---------------------------------------------------------------------------
# M/S encode / decode
# ---------------------------------------------------------------------------

def ms_encode(stereo: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """(T, 2) stereo → (M, S) each (T,) float32."""
    M = (stereo[:, 0] + stereo[:, 1]) / np.sqrt(2)
    S = (stereo[:, 0] - stereo[:, 1]) / np.sqrt(2)
    return M.astype(np.float32), S.astype(np.float32)


def ms_decode(M: np.ndarray, S: np.ndarray) -> np.ndarray:
    """(M, S) each (T,) → (T, 2) stereo float32."""
    L = (M + S) / np.sqrt(2)
    R = (M - S) / np.sqrt(2)
    return np.stack([L, R], axis=1).astype(np.float32)


def mono_lf(audio: np.ndarray, cutoff_hz: float = 150.0) -> np.ndarray:
    """
    Collapse the Side channel below cutoff_hz to mono.

    Sub-bass should be mono for club system compatibility.  This function
    zeroes the low-frequency portion of the Side channel via M/S encoding:
    high-pass the Side at cutoff_hz, then decode — LF collapses to mono.
    """
    nyq = SR / 2.0
    sos_hp = butter(4, cutoff_hz / nyq, btype="high", output="sos")
    M, S = ms_encode(audio)
    S_hf = sosfilt(sos_hp, S.astype(np.float64)).astype(np.float32)
    return ms_decode(M, S_hf)


# ---------------------------------------------------------------------------
# Signal health check (debug / QC)
# ---------------------------------------------------------------------------

def check(y: np.ndarray, label: str) -> np.ndarray:
    """
    Print peak/RMS and warn on NaN/Inf.  Replaces corrupt samples with zeros.
    Used throughout the mixing pipeline for inline QC.
    """
    has_nan = bool(np.any(np.isnan(y)))
    has_inf = bool(np.any(np.isinf(y)))
    peak    = float(np.nanmax(np.abs(y))) if not (has_nan and has_inf) else float("nan")
    rms_val = float(np.sqrt(np.nanmean(y ** 2) + 1e-12))
    flags   = (" NaN!" if has_nan else "") + (" Inf!" if has_inf else "")
    print(f"      [DBG] {label}: peak={peak:.4f} rms={rms_val:.5f}{flags}", flush=True)
    if has_nan or has_inf:
        print(f"      [DBG] *** CORRUPTED AT {label} — replacing with zeros ***", flush=True)
        y = np.nan_to_num(y, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)
    return y
