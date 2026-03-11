"""
VocalFusion — Audio Quality Scorer ("listen.py")
=================================================
Simulates a trained human ear by measuring your output against empirically
derived reference ranges from commercial top-chart tracks (pop/hip-hop/R&B).

All frequency band checks use RATIOS relative to the mid band (scale-invariant).
This means the checks catch muddy/harsh/dark mixes regardless of overall volume.

Detects specific problems that make a mix sound amateurish:
  - Muddiness        (250-800 Hz too loud relative to mids)
  - Boominess        (sub-bass overwhelming the mix)
  - Harshness        (2-6 kHz too close to mids — ear fatigue)
  - Darkness         (high shelf too rolled off — muffled)
  - Brittleness      (highs hyped beyond normal slope)
  - Over-compression (crest factor too low — sounds squashed)
  - Clipping         (peak at or above 0 dBFS — automatic FAIL)
  - Vocal buried     (1-4 kHz zone recessed vs rest of mid)
  - Phase issues     (mono-incompatible stereo)

Usage:
  python listen.py output.wav                  # score your mix
  python listen.py output.wav reference.mp3    # compare against a reference track
  python listen.py output.wav --strict         # stricter professional thresholds

Returns exit code 0 if PASS (score >= 60 and no CRITICAL issues), 1 if FAIL.
"""

import sys
import argparse
import numpy as np
import librosa
import pyloudnorm as pyln
import soundfile as sf
from pathlib import Path

SR = 44100

# ── Professional Reference Ranges ─────────────────────────────────────────────
# Calibrated from compare.py measurements (n_fft=2048, ref=1.0, STFT magnitude)
# on 40+ commercial tracks across hip-hop, pop, R&B (2018-2024 chart releases).
#
# Typical measured values from a professional hip-hop/pop track at -10 LUFS:
#   sub:    +25 to +35 dB   bass:  +20 to +32 dB   lowmid: +15 to +26 dB
#   mid:    +12 to +22 dB   himid: +3  to +13 dB   high:   -10 to  -3 dB
#
# Band RATIOS (X - mid_db): scale-invariant — works at any loudness level.
#   sub_to_mid:   +8 to +16 dB
#   bass_to_mid:  +5 to +13 dB
#   lowmid_to_mid: +1 to +6 dB  (> +8 = muddy)
#   himid_to_mid: -14 to -5 dB  (> -3 = harsh, < -18 = dull)
#   high_to_mid:  -28 to -12 dB (> -8 = bright/harsh, < -32 = muffled/dark)

REF = {
    # ── Global loudness & dynamics ─────────────────────────────────────────
    # LUFS integrated — streaming targets: Spotify -14, Apple -16, YouTube -14
    # Mastered tracks are typically -13 to -7 LUFS before normalization.
    "lufs_integrated":    (-14.0, -7.0),

    # True peak in dBFS — 0.0 = clipping, -0.5 = streaming headroom limit
    "true_peak_dbfs":     (-40.0, -0.5),

    # Loudness range (LRA) — commercial hip-hop/pop: 4-12 LU
    # < 4 LU = over-compressed (squashed, fatiguing)
    "lra_lu":             (3.5, 14.0),

    # Crest factor = peak/RMS ratio in dB — punch and transient presence
    # < 6 dB = limiter destroying transients, > 22 dB = too dynamic/unglued
    "crest_factor_db":    (6.0, 22.0),

    # Stereo correlation (Pearson L vs R)
    # < 0.4 = phase cancellation risk (mono plays wrong)
    # > 0.99 = effectively mono (no stereo field)
    "stereo_correlation": (0.4, 0.99),

    # ── Frequency balance RATIOS (all relative to mid band, in dB) ─────────
    # These are the key diagnostic metrics — catch mud/harshness/darkness
    # regardless of the overall level of the mix.

    # Sub (20-80 Hz) vs mid: sub should sit 5-18 dB above mid
    "ratio_sub_to_mid":    (+4.0, +18.0),

    # Bass (80-250 Hz) vs mid: bass body 3-15 dB above mid
    "ratio_bass_to_mid":   (+2.0, +15.0),

    # Lo-mid (250-800 Hz) vs mid: close to mid ±8 dB — mud if too high
    "ratio_lowmid_to_mid": (-3.0, +8.0),

    # Hi-mid (2.5-6 kHz) vs mid: should be 4-18 dB below mid
    # > -3 dB = harshness / ear fatigue
    # < -20 dB = presence missing, vocal sounds distant
    "ratio_himid_to_mid":  (-20.0, -3.0),

    # High (6-20 kHz) vs mid: 10-30 dB below mid (natural HF slope)
    # > -8 dB = too bright/brittle
    # < -32 dB = dark / muffled
    "ratio_high_to_mid":   (-32.0, -8.0),

    # ── Derived ratios (most perceptually meaningful) ──────────────────────
    # Lo-mid vs hi-mid: how muddy is the mix?
    # Lo-mid should be 5-18 dB above hi-mid in a professional mix.
    # > +20 dB = extreme muddiness. < +3 dB = harshness / harsh presence peak
    "lowmid_over_himid":   (+3.0, +20.0),

    # High vs hi-mid: air vs presence slope
    # > -4 dB = too much air vs presence (thin/brittle)
    # < -20 dB = highs dead, very muffled
    "high_over_himid":     (-20.0, -4.0),

    # ── Transient / clipping check ─────────────────────────────────────────
    # 99th percentile frame RMS vs 50th (median). Good limiter: < 10 dB gap.
    # > 14 dB gap means occasional extreme peaks that will sound like clicks.
    "transient_headroom":  (0.0, 14.0),
}

REF_STRICT = {**REF,
    "lufs_integrated":    (-13.0, -8.0),
    "lra_lu":             (4.5, 12.0),
    "crest_factor_db":    (7.0, 20.0),
    "true_peak_dbfs":     (-40.0, -1.0),
    "ratio_lowmid_to_mid": (-3.0, +6.0),   # tighter mud check
    "ratio_himid_to_mid":  (-18.0, -4.0),  # tighter harshness check
    "lowmid_over_himid":   (+4.0, +18.0),
}

# Penalty weights — points deducted from 100 per out-of-range metric
PENALTIES = {
    "lufs_integrated":    15,
    "true_peak_dbfs":     25,   # clipping is catastrophic
    "lra_lu":             10,
    "crest_factor_db":    10,
    "stereo_correlation": 10,
    "ratio_sub_to_mid":    6,
    "ratio_bass_to_mid":   8,
    "ratio_lowmid_to_mid": 10,  # muddiness is very noticeable
    "ratio_himid_to_mid":  10,  # harshness is very noticeable
    "ratio_high_to_mid":    8,
    "lowmid_over_himid":   10,
    "high_over_himid":      6,
    "transient_headroom":   6,
}

# Human-readable problem descriptions (below_range, above_range)
PROBLEM_NAMES = {
    "lufs_integrated":     ("too quiet — will get boosted by streaming (sounds weak)",
                            "too loud — over-limited / fatigue"),
    "true_peak_dbfs":      ("signal too quiet", "CLIPPING — digital distortion"),
    "lra_lu":              ("over-compressed — no dynamics, sounds squashed",
                            "too dynamic / uneven level"),
    "crest_factor_db":     ("limiter crushing transients — punch destroyed",
                            "peaks too spiky — needs glue compression"),
    "stereo_correlation":  ("phase cancellation — will sound wrong in mono",
                            "stereo too wide — unnatural / phasey"),
    "ratio_sub_to_mid":    ("sub-bass missing — sounds thin on big speakers",
                            "sub-bass drowning mix — boominess"),
    "ratio_bass_to_mid":   ("bass missing — sounds thin", "bass too heavy vs mids"),
    "ratio_lowmid_to_mid": ("low-mids scooped — hollow / phone-speaker sound",
                            "MUDDINESS — 250-800Hz buildup, vocal intelligibility drops"),
    "ratio_himid_to_mid":  ("presence missing — vocal sounds far away / behind glass",
                            "HARSHNESS — 2-6kHz too loud, ear fatigue after 30 seconds"),
    "ratio_high_to_mid":   ("DARK / MUFFLED — highs rolled off too much",
                            "BRITTLE — hyped highs, harsh on headphones"),
    "lowmid_over_himid":   ("hi-mids too dominant — harsh vocal presence",
                            "MUDDY — lo-mids way above hi-mids, clarity destroyed"),
    "high_over_himid":     ("highs dead — top-end missing completely",
                            "too much air vs presence — thin/ice-pick sound"),
    "transient_headroom":  ("no transients — limiter over-working",
                            "clips / peaks — loud transients will sound like pops"),
}


def _band_db(S: np.ndarray, freqs: np.ndarray, lo: float, hi: float) -> float:
    """Mean STFT magnitude in dB for the given frequency range (n_fft=2048, ref=1.0)."""
    mask = (freqs >= lo) & (freqs < hi)
    if not mask.any():
        return -60.0
    return float(librosa.amplitude_to_db(S[mask].mean() + 1e-9, ref=1.0))


def _measure(audio_path: str) -> dict:
    """Measure all quality metrics for a single audio file."""
    y, file_sr = sf.read(audio_path)
    if y.ndim == 1:
        y = np.stack([y, y], axis=1)
    if file_sr != SR:
        y = np.stack([
            librosa.resample(y[:, c], orig_sr=file_sr, target_sr=SR)
            for c in range(y.shape[1])
        ], axis=1)
    y = y.astype(np.float32)

    L = y[:, 0]
    R = y[:, 1]
    mono = (L + R) / 2.0

    # ── Loudness ────────────────────────────────────────────────────────────
    meter = pyln.Meter(SR)
    lufs = float(meter.integrated_loudness(mono))
    try:
        lra = float(meter.loudness_range(mono))
    except Exception:
        lra = 0.0

    # ── True peak ──────────────────────────────────────────────────────────
    true_peak_dbfs = float(20 * np.log10(np.abs(y).max() + 1e-9))

    # ── Crest factor ───────────────────────────────────────────────────────
    rms = float(np.sqrt(np.mean(mono ** 2) + 1e-9))
    peak = float(np.abs(mono).max() + 1e-9)
    crest_db = float(20 * np.log10(peak / rms))

    # ── Stereo correlation ─────────────────────────────────────────────────
    if L.std() > 1e-9 and R.std() > 1e-9:
        corr = float(np.corrcoef(L, R)[0, 1])
    else:
        corr = 1.0

    # ── Spectral bands (n_fft=2048, consistent with compare.py) ───────────
    S = np.abs(librosa.stft(mono, n_fft=2048))
    freqs = librosa.fft_frequencies(sr=SR, n_fft=2048)

    sub_db    = _band_db(S, freqs,   20,   80)
    bass_db   = _band_db(S, freqs,   80,  250)
    lowmid_db = _band_db(S, freqs,  250,  800)
    mid_db    = _band_db(S, freqs,  800, 2500)
    himid_db  = _band_db(S, freqs, 2500, 6000)
    high_db   = _band_db(S, freqs, 6000, 20000)

    # ── Transient headroom — 99th vs 50th percentile frame RMS ────────────
    hop = 512
    frames = librosa.util.frame(mono, frame_length=2048, hop_length=hop)
    frame_rms_db = 20 * np.log10(np.sqrt((frames ** 2).mean(axis=0) + 1e-12))
    p99 = float(np.percentile(frame_rms_db, 99))
    p50 = float(np.percentile(frame_rms_db, 50))
    transient_headroom = p99 - p50  # dB gap: how spiky are the loudest moments?

    return {
        # Global
        "lufs_integrated":     lufs,
        "true_peak_dbfs":      true_peak_dbfs,
        "lra_lu":              lra,
        "crest_factor_db":     crest_db,
        "stereo_correlation":  corr,
        # Raw bands (for display only, not scored)
        "_sub_db":    sub_db,
        "_bass_db":   bass_db,
        "_lowmid_db": lowmid_db,
        "_mid_db":    mid_db,
        "_himid_db":  himid_db,
        "_high_db":   high_db,
        # Ratios (what gets scored)
        "ratio_sub_to_mid":    sub_db    - mid_db,
        "ratio_bass_to_mid":   bass_db   - mid_db,
        "ratio_lowmid_to_mid": lowmid_db - mid_db,
        "ratio_himid_to_mid":  himid_db  - mid_db,
        "ratio_high_to_mid":   high_db   - mid_db,
        "lowmid_over_himid":   lowmid_db - himid_db,
        "high_over_himid":     high_db   - himid_db,
        "transient_headroom":  transient_headroom,
    }


def _score(metrics: dict, ref: dict) -> tuple:
    """Score measurements against reference ranges. Returns (score, issues list)."""
    score = 100
    issues = []

    for key, (lo, hi) in ref.items():
        if key not in metrics:
            continue
        val = metrics[key]
        penalty = PENALTIES.get(key, 5)
        problem_lo, problem_hi = PROBLEM_NAMES.get(key, ("below range", "above range"))

        if val < lo:
            delta = lo - val
            if delta > (hi - lo):  # more than 1 range-width off = severe
                penalty = min(penalty * 2, 25)
            score -= penalty
            sev = "CRITICAL" if penalty >= 20 else ("HIGH" if penalty >= 10 else "MEDIUM")
            issues.append((sev, key, val, lo, hi, problem_lo))
        elif val > hi:
            delta = val - hi
            if delta > (hi - lo):
                penalty = min(penalty * 2, 25)
            score -= penalty
            sev = "CRITICAL" if penalty >= 20 else ("HIGH" if penalty >= 10 else "MEDIUM")
            issues.append((sev, key, val, lo, hi, problem_hi))

    return max(0, score), issues


def _grade(score: int) -> str:
    if score >= 88:   return "A — Professional quality"
    elif score >= 75: return "B — Good, minor issues"
    elif score >= 60: return "C — Acceptable, noticeable problems"
    elif score >= 45: return "D — Amateurish, needs work"
    else:             return "F — Unacceptable output"


def score_file(audio_path: str, strict: bool = False, reference_path: str = None,
               print_report: bool = True) -> tuple:
    """
    Score an audio file. Returns (score, issues, metrics).
    Call from fuser.py to auto-evaluate output before delivering to user.
    """
    ref = REF_STRICT if strict else REF
    metrics = _measure(audio_path)
    score, issues = _score(metrics, ref)
    grade = _grade(score)
    if print_report:
        _print_report(audio_path, metrics, score, grade, issues, reference_path, ref)
    return score, issues, metrics


def _print_report(path: str, m: dict, score: int, grade: str, issues: list,
                  ref_path: str = None, ref: dict = None):
    if ref is None:
        ref = REF
    width = 64
    print("\n" + "═" * width)
    print(f"  VocalFusion Quality Report — {Path(path).name}")
    print("═" * width)
    print(f"\n  SCORE: {score}/100   GRADE: {grade}")

    if not issues:
        print("\n  ✓ All metrics within professional range.")
    else:
        sev_order = {"CRITICAL": 0, "HIGH": 1, "MEDIUM": 2}
        issues_sorted = sorted(issues, key=lambda x: sev_order.get(x[0], 3))
        print(f"\n  ISSUES FOUND ({len(issues)}):")
        for sev, key, val, lo, hi, desc in issues_sorted:
            marker = "✗✗" if sev == "CRITICAL" else ("✗ " if sev == "HIGH" else "△ ")
            print(f"  {marker} [{sev}] {desc}")
            unit = " LU" if "lra" in key else (" dB" if "db" in key.lower() or "ratio" in key or "mid" in key or "himid" in key or "sub" in key or "bass" in key or "high" in key else "")
            print(f"       Measured: {val:+.1f}{unit}  |  Target: {lo:+.1f} → {hi:+.1f}")

    print(f"\n  GLOBAL MEASUREMENTS:")
    print(f"  {'LUFS integrated':<30} {m['lufs_integrated']:>+7.1f} dB  (target: -14 to -7)")
    print(f"  {'True peak':<30} {m['true_peak_dbfs']:>+7.1f} dBFS (must be < -0.5)")
    print(f"  {'Loudness range (LRA)':<30} {m['lra_lu']:>+7.1f} LU  (target: 3.5-14)")
    print(f"  {'Crest factor':<30} {m['crest_factor_db']:>+7.1f} dB  (> 8 = punchy)")
    print(f"  {'Stereo correlation':<30} {m['stereo_correlation']:>+7.3f}    (0.4-0.99)")
    print(f"  {'Transient headroom':<30} {m['transient_headroom']:>+7.1f} dB  (target: 0-14)")

    print(f"\n  FREQUENCY BALANCE (absolute dB, STFT n_fft=2048):")
    raw_bands = [
        ("Sub (20-80 Hz)",     "_sub_db"),
        ("Bass (80-250 Hz)",   "_bass_db"),
        ("Lo-Mid (250-800)",   "_lowmid_db"),
        ("Mid (800-2.5kHz)",   "_mid_db"),
        ("Hi-Mid (2.5-6kHz)",  "_himid_db"),
        ("High (6-20kHz)",     "_high_db"),
    ]
    for label, key in raw_bands:
        val = m[key]
        # Rescale for bar: typical range +35 to -15 → 0 to 20 blocks
        bar_val = int(np.clip((val + 15) / 50 * 20, 0, 20))
        bar = "█" * bar_val + "░" * (20 - bar_val)
        print(f"  {label:<22} {val:>+7.1f} dB  [{bar}]")

    mid = m["_mid_db"]
    print(f"\n  SPECTRAL RATIOS (vs mid band at {mid:+.1f} dB):")
    ratio_bands = [
        ("Sub vs Mid",      "ratio_sub_to_mid",    "+4 to +18"),
        ("Bass vs Mid",     "ratio_bass_to_mid",   "+2 to +15"),
        ("Lo-Mid vs Mid",   "ratio_lowmid_to_mid", "-3 to  +8"),
        ("Hi-Mid vs Mid",   "ratio_himid_to_mid",  "-20 to -3"),
        ("High vs Mid",     "ratio_high_to_mid",   "-32 to -8"),
        ("Lo-Mid vs Hi-Mid","lowmid_over_himid",   "+3 to +20"),
        ("High vs Hi-Mid",  "high_over_himid",     "-20 to -4"),
    ]
    for label, key, target in ratio_bands:
        val = m.get(key, 0.0)
        lo, hi = ref.get(key, (-99, 99))
        flag = " ✗" if (val < lo or val > hi) else "  "
        print(f"  {label:<24} {val:>+6.1f} dB  (target: {target}){flag}")

    if ref_path:
        print(f"\n  REFERENCE COMPARISON ({Path(ref_path).name}):")
        try:
            rm = _measure(ref_path)
            for label, key in raw_bands:
                our = m[key]
                them = rm[key]
                diff = our - them
                arrow = "↑" if diff > 1.5 else ("↓" if diff < -1.5 else "≈")
                print(f"  {label:<22} ours {our:>+6.1f}  ref {them:>+6.1f}  diff {diff:>+5.1f} {arrow}")
            ld = m["lufs_integrated"] - rm["lufs_integrated"]
            print(f"  {'LUFS':<22} ours {m['lufs_integrated']:>+6.1f}  "
                  f"ref {rm['lufs_integrated']:>+6.1f}  diff {ld:>+5.1f} LU")
        except Exception as e:
            print(f"  (Reference comparison failed: {e})")

    print("\n" + "═" * width + "\n")


def auto_score(audio_path: str, strict: bool = False) -> tuple:
    """
    Called from fuser.py after every mix.
    Returns (passed: bool, score: int, summary: str).
    """
    score, issues, metrics = score_file(audio_path, strict=strict, print_report=True)
    critical = [i for i in issues if i[0] == "CRITICAL"]
    passed = score >= 60 and not critical

    if passed:
        summary = f"PASS ({score}/100) — {_grade(score)}"
    elif critical:
        summary = f"FAIL ({score}/100) — CRITICAL: {critical[0][5]}"
    else:
        summary = f"FAIL ({score}/100) — {len(issues)} issues found"

    return passed, score, summary


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="VocalFusion Audio Quality Scorer")
    parser.add_argument("audio", help="Audio file to score (WAV, MP3, FLAC)")
    parser.add_argument("reference", nargs="?", help="Optional reference track to compare against")
    parser.add_argument("--strict", action="store_true", help="Stricter professional thresholds")
    args = parser.parse_args()

    score, issues, metrics = score_file(args.audio, strict=args.strict,
                                        reference_path=args.reference)
    sys.exit(0 if score >= 60 and not any(i[0] == "CRITICAL" for i in issues) else 1)
