"""
VocalFusion — Reference Track Learning System
==============================================
Analyzes chart/commercial songs and extracts a "sonic fingerprint" (reference
profile) that the rest of the system uses as a target for mixing decisions.

Usage:
    python reference.py <folder_of_chart_songs> [output_profile.json]

The profile JSON is used by listen.py and fuse.py to compute per-feature deltas
between the current output and the median of the reference corpus.
"""

import os
import json
import numpy as np
import librosa
import pyloudnorm as pyln
import soundfile as sf
from pathlib import Path
from scipy.signal import butter, sosfilt

SR = 44100

REFERENCE_PROFILE_PATH = os.path.join(os.path.dirname(__file__), "reference_profile.json")

AUDIO_EXTENSIONS = {".mp3", ".flac", ".wav", ".m4a"}

_PROFILE_REQUIRED_KEYS = {"version", "n_tracks", "bands", "loudness", "dynamics", "stereo"}

# ── Filter helpers (consistent with listen.py) ────────────────────────────────

def _bp(y: np.ndarray, lo_hz: float, hi_hz: float, order: int = 4) -> np.ndarray:
    """Bandpass Butterworth filter."""
    nyq = SR / 2.0
    sos = butter(order, [lo_hz / nyq, min(hi_hz / nyq, 0.999)], btype="band", output="sos")
    return sosfilt(sos, y)


def _band_rms_db(y: np.ndarray, lo_hz: float, hi_hz: float) -> float:
    """RMS energy (dBFS) of a bandpass-filtered signal."""
    filtered = _bp(y, lo_hz, hi_hz)
    rms = float(np.sqrt(np.mean(filtered ** 2) + 1e-12))
    return float(20.0 * np.log10(rms + 1e-12))


def _stereo_width(L: np.ndarray, R: np.ndarray, lo_hz: float, hi_hz: float) -> float:
    """
    Width = 1 - correlation(L_band, R_band).
    0 = perfectly mono, 1 = completely decorrelated (maximum width).
    """
    Lb = _bp(L, lo_hz, hi_hz)
    Rb = _bp(R, lo_hz, hi_hz)
    if Lb.std() < 1e-9 or Rb.std() < 1e-9:
        return 0.0
    corr = float(np.corrcoef(Lb, Rb)[0, 1])
    return float(np.clip(1.0 - corr, 0.0, 2.0))


# ── Core analysis ──────────────────────────────────────────────────────────────

def analyze_track(path: str) -> "dict | None":
    """
    Analyze a single audio file and return its sonic fingerprint dict.
    Returns None if analysis fails (corrupt file, too short, etc).

    Extracts:
    - 7 spectral bands (dBFS RMS): sub(20-60Hz), bass(60-250Hz), lo_mid(250-500Hz),
      mid(500-2kHz), hi_mid(2-6kHz), presence(6-10kHz), air(10-20kHz)
    - loudness: integrated LUFS, LRA, crest_db, PLR, true_peak
    - dynamics: spectral_slope (linear regression of log-spaced band energies),
      spectral_crest (peak/RMS of magnitude spectrum), transient_clarity (HFC onset strength),
      dynamic_arc (std of 10s-window LUFS across track)
    - stereo: width_low (20-250Hz), width_mid (250-4kHz), width_high (4kHz+)
      where width = 1 - correlation(L,R) for that band (0=mono, 1=wide)

    Uses librosa.load at sr=44100, mono=False (keep stereo).
    Uses pyloudnorm for LUFS/LRA/true_peak/PLR.
    Uses scipy.signal.butter bandpass filters for band RMS.
    Skips files shorter than 30 seconds.
    """
    try:
        # Load stereo at SR=44100
        y, file_sr = librosa.load(path, sr=SR, mono=False)
        # librosa.load with mono=False returns shape (channels, samples) or (samples,) for mono
        if y.ndim == 1:
            # Mono file — duplicate to stereo
            y = np.stack([y, y], axis=0)
        # y is now (2, N)
        L = y[0]
        R = y[1]
        mono = (L + R) / 2.0

        # Skip files shorter than 30 seconds
        duration = len(mono) / SR
        if duration < 30.0:
            return None

        # ── LOUDNESS ──────────────────────────────────────────────────────────
        meter = pyln.Meter(SR)
        # pyloudnorm expects shape (N,) for mono or (N, 2) for stereo
        stereo_for_lufs = np.stack([L, R], axis=1)  # (N, 2)
        try:
            lufs_i = float(meter.integrated_loudness(stereo_for_lufs))
        except Exception:
            lufs_i = float(meter.integrated_loudness(mono))

        try:
            lra = float(meter.loudness_range(stereo_for_lufs))
        except Exception:
            try:
                lra = float(meter.loudness_range(mono))
            except Exception:
                lra = 0.0

        true_peak = float(20.0 * np.log10(np.abs(y).max() + 1e-12))
        rms_val = float(np.sqrt(np.mean(mono ** 2) + 1e-12))
        peak_val = float(np.abs(mono).max() + 1e-12)
        crest_db = float(20.0 * np.log10(peak_val / rms_val))
        plr = float(np.clip(true_peak - lufs_i, -5.0, 30.0)) if np.isfinite(lufs_i) else 9.0

        # ── SPECTRAL BANDS (dBFS RMS via bandpass filter) ─────────────────────
        sub_db      = _band_rms_db(mono,   20.0,    60.0)
        bass_db     = _band_rms_db(mono,   60.0,   250.0)
        lo_mid_db   = _band_rms_db(mono,  250.0,   500.0)
        mid_db      = _band_rms_db(mono,  500.0,  2000.0)
        hi_mid_db   = _band_rms_db(mono, 2000.0,  6000.0)
        presence_db = _band_rms_db(mono, 6000.0, 10000.0)
        air_db      = _band_rms_db(mono, 10000.0, 20000.0)

        # ── DYNAMICS ──────────────────────────────────────────────────────────
        # Single STFT pass for spectral dynamics features
        S_mag = np.abs(librosa.stft(mono, n_fft=2048, hop_length=512))
        freqs = librosa.fft_frequencies(sr=SR, n_fft=2048)

        # Spectral slope: linear regression over log-spaced octave band energies
        octave_bands = [(200, 400), (400, 800), (800, 1600),
                        (1600, 3200), (3200, 6400), (6400, 12800)]
        octave_dbs = []
        for lo, hi in octave_bands:
            mask = (freqs >= lo) & (freqs < hi)
            if mask.any():
                val = float(20.0 * np.log10(float(S_mag[mask].mean()) + 1e-9))
            else:
                val = -60.0
            octave_dbs.append(val)
        coeffs = np.polyfit(np.arange(len(octave_dbs), dtype=float), octave_dbs, 1)
        spectral_slope = float(coeffs[0])

        # Spectral crest: peak / RMS of the magnitude spectrum (mean across frames)
        frame_peak = S_mag.max(axis=0)
        frame_rms  = np.sqrt((S_mag ** 2).mean(axis=0)) + 1e-12
        spectral_crest = float(np.mean(frame_peak / frame_rms))

        # Transient clarity: p95/p10 of onset strength envelope (dB)
        try:
            flux = librosa.onset.onset_strength(y=mono, sr=SR, hop_length=512)
            p95 = float(np.percentile(flux, 95))
            p10 = float(np.percentile(flux, 10))
            transient_clarity = float(np.clip(20.0 * np.log10((p95 + 1e-6) / (p10 + 1e-6)), 0.0, 40.0))
        except Exception:
            transient_clarity = 10.0

        # Dynamic arc: std of per-10s-window integrated LUFS across the track
        arc_win_samples = SR * 10
        n_windows = len(mono) // arc_win_samples
        arc_lufs_vals = []
        if n_windows >= 2:
            for i in range(n_windows):
                seg = mono[i * arc_win_samples:(i + 1) * arc_win_samples]
                try:
                    wl = float(meter.integrated_loudness(seg))
                    if np.isfinite(wl) and wl > -70.0:
                        arc_lufs_vals.append(wl)
                except Exception:
                    pass
        dynamic_arc = float(np.std(arc_lufs_vals)) if len(arc_lufs_vals) >= 2 else 0.0

        # ── STEREO WIDTH ──────────────────────────────────────────────────────
        width_low  = _stereo_width(L, R,  20.0,  250.0)
        width_mid  = _stereo_width(L, R, 250.0, 4000.0)
        width_high = _stereo_width(L, R, 4000.0, 20000.0)

        return {
            # Spectral bands
            "sub":       sub_db,
            "bass":      bass_db,
            "lo_mid":    lo_mid_db,
            "mid":       mid_db,
            "hi_mid":    hi_mid_db,
            "presence":  presence_db,
            "air":       air_db,
            # Loudness
            "lufs_i":    lufs_i,
            "lra":       lra,
            "crest_db":  crest_db,
            "plr":       plr,
            "true_peak": true_peak,
            # Dynamics
            "spectral_slope":    spectral_slope,
            "spectral_crest":    spectral_crest,
            "transient_clarity": transient_clarity,
            "dynamic_arc":       dynamic_arc,
            # Stereo
            "width_low":  width_low,
            "width_mid":  width_mid,
            "width_high": width_high,
        }

    except Exception:
        return None


# ── Profile building ───────────────────────────────────────────────────────────

def build_profile(folder: str, output_path: str = REFERENCE_PROFILE_PATH) -> dict:
    """
    Analyze all audio files (mp3, flac, wav, m4a) in folder (recursively),
    compute median + p25 + p75 for every feature across all tracks,
    save as JSON to output_path, and return the profile dict.

    Requires at least 3 valid tracks or raises ValueError.
    """
    folder_path = Path(folder)
    audio_files = sorted([
        p for p in folder_path.rglob("*")
        if p.suffix.lower() in AUDIO_EXTENSIONS
    ])
    total = len(audio_files)
    print(f"Analyzing {total} tracks from {folder}...")

    results = []
    for p in audio_files:
        try:
            features = analyze_track(str(p))
            if features is None:
                print(f"✗ {p.name}: skipped (too short or unreadable)")
            else:
                print(f"✓ {p.name}")
                results.append(features)
        except Exception as e:
            print(f"✗ {p.name}: {e}")

    n_valid = len(results)
    if n_valid < 3:
        raise ValueError(
            f"Only {n_valid} valid track(s) found in '{folder}'. "
            f"Need at least 3 to build a reliable profile."
        )

    # Aggregate all features into arrays
    keys = list(results[0].keys())
    arrays = {k: np.array([r[k] for r in results], dtype=float) for k in keys}

    def _stats(arr: np.ndarray) -> dict:
        return {
            "median": float(np.median(arr)),
            "p25":    float(np.percentile(arr, 25)),
            "p75":    float(np.percentile(arr, 75)),
        }

    profile = {
        "version":  1,
        "n_tracks": n_valid,
        "bands": {
            "sub":      _stats(arrays["sub"]),
            "bass":     _stats(arrays["bass"]),
            "lo_mid":   _stats(arrays["lo_mid"]),
            "mid":      _stats(arrays["mid"]),
            "hi_mid":   _stats(arrays["hi_mid"]),
            "presence": _stats(arrays["presence"]),
            "air":      _stats(arrays["air"]),
        },
        "loudness": {
            "lufs_i":    _stats(arrays["lufs_i"]),
            "lra":       _stats(arrays["lra"]),
            "crest_db":  _stats(arrays["crest_db"]),
            "plr":       _stats(arrays["plr"]),
            "true_peak": _stats(arrays["true_peak"]),
        },
        "dynamics": {
            "spectral_slope":    _stats(arrays["spectral_slope"]),
            "spectral_crest":    _stats(arrays["spectral_crest"]),
            "transient_clarity": _stats(arrays["transient_clarity"]),
            "dynamic_arc":       _stats(arrays["dynamic_arc"]),
        },
        "stereo": {
            "width_low":  _stats(arrays["width_low"]),
            "width_mid":  _stats(arrays["width_mid"]),
            "width_high": _stats(arrays["width_high"]),
        },
    }

    # Save to disk
    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(profile, f, indent=2)

    print(f"Profile built from {n_valid}/{total} tracks → {output_path}")
    return profile


# ── Profile I/O ───────────────────────────────────────────────────────────────

def load_profile(path: str = REFERENCE_PROFILE_PATH) -> "dict | None":
    """
    Load and return the reference profile JSON, or None if file doesn't exist.
    Validates version == 1 and required top-level keys are present.
    """
    if not os.path.exists(path):
        return None
    try:
        with open(path, "r", encoding="utf-8") as f:
            profile = json.load(f)
        if profile.get("version") != 1:
            return None
        if not _PROFILE_REQUIRED_KEYS.issubset(profile.keys()):
            return None
        return profile
    except Exception:
        return None


# ── Delta computation ─────────────────────────────────────────────────────────

def compute_delta(output_features: dict, profile: dict) -> dict:
    """
    Given a dict of features from analyze_track() and a loaded profile,
    compute per-feature delta = output_value - profile[section][key]["median"].

    Returns a flat dict: {"sub_delta": X, "bass_delta": X, ..., "lufs_i_delta": X, ...}
    Positive delta = our output is louder/higher than reference.
    Negative delta = our output is quieter/lower than reference.
    """
    delta = {}

    # Bands
    for key in ("sub", "bass", "lo_mid", "mid", "hi_mid", "presence", "air"):
        if key in output_features and key in profile.get("bands", {}):
            delta[f"{key}_delta"] = float(output_features[key] - profile["bands"][key]["median"])

    # Loudness
    for key in ("lufs_i", "lra", "crest_db", "plr", "true_peak"):
        if key in output_features and key in profile.get("loudness", {}):
            delta[f"{key}_delta"] = float(output_features[key] - profile["loudness"][key]["median"])

    # Dynamics
    for key in ("spectral_slope", "spectral_crest", "transient_clarity", "dynamic_arc"):
        if key in output_features and key in profile.get("dynamics", {}):
            delta[f"{key}_delta"] = float(output_features[key] - profile["dynamics"][key]["median"])

    # Stereo
    for key in ("width_low", "width_mid", "width_high"):
        if key in output_features and key in profile.get("stereo", {}):
            delta[f"{key}_delta"] = float(output_features[key] - profile["stereo"][key]["median"])

    return delta


# ── Human-readable summary ────────────────────────────────────────────────────

def format_profile_summary(profile: dict) -> str:
    """
    Return a human-readable string summarizing the profile for use in prompts.
    """
    n = profile.get("n_tracks", 0)
    b = profile.get("bands", {})
    lo = profile.get("loudness", {})
    dy = profile.get("dynamics", {})
    st = profile.get("stereo", {})

    def m(section: dict, key: str) -> float:
        return section.get(key, {}).get("median", 0.0)

    lines = [
        f"Reference profile ({n} chart tracks):",
        (
            f"  Spectral: sub={m(b,'sub'):.1f}dB bass={m(b,'bass'):.1f}dB "
            f"mid={m(b,'mid'):.1f}dB hi_mid={m(b,'hi_mid'):.1f}dB air={m(b,'air'):.1f}dB"
        ),
        (
            f"  Loudness: LUFS={m(lo,'lufs_i'):.1f} LRA={m(lo,'lra'):.1f} "
            f"PLR={m(lo,'plr'):.1f}dB crest={m(lo,'crest_db'):.1f}dB"
        ),
        (
            f"  Dynamics: slope={m(dy,'spectral_slope'):.2f} arc={m(dy,'dynamic_arc'):.3f}"
        ),
        (
            f"  Stereo: low={m(st,'width_low'):.2f} mid={m(st,'width_mid'):.2f} "
            f"high={m(st,'width_high'):.2f}"
        ),
    ]
    return "\n".join(lines)


# ── CLI entry point ───────────────────────────────────────────────────────────

if __name__ == "__main__":
    import sys
    if len(sys.argv) < 2:
        print("Usage: python reference.py <folder_of_chart_songs> [output_profile.json]")
        sys.exit(1)
    folder = sys.argv[1]
    out = sys.argv[2] if len(sys.argv) > 2 else REFERENCE_PROFILE_PATH
    build_profile(folder, out)
