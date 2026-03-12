"""
VocalFusion — Commercial Quality Grader
========================================
Determines whether an audio file sounds like a professional/chart-ready track
or an amateur production. Based on research into what separates pro from amateur:

  1. Loudness management (LUFS, peak, LRA)
  2. Spectral balance (pink-noise envelope, low-end control, air)
  3. Transient definition (kick punch, crest factor)
  4. Stereo imaging (width, mono-compatibility)
  5. Dynamic variation (musical movement, section consistency)
  6. Production signatures (harmonic richness, stereo M/S balance)
  7. Artifact cleanliness (clipping, clicks, phase issues)

Usage:
  python grade.py song.mp3
  python grade.py song.mp3 --verbose
  python grade.py *.mp3             # batch mode

Exit codes: 0 = pro/chart-ready, 1 = needs work, 2 = amateur
"""

import sys
import argparse
import numpy as np
import librosa
import pyloudnorm as pyln
import soundfile as sf
from pathlib import Path
from scipy.signal import butter, sosfilt

SR = 44100


def _load(path: str) -> np.ndarray:
    y, sr = sf.read(path, always_2d=True)
    if sr != SR:
        y = np.stack([
            librosa.resample(y[:, c].astype(np.float32), orig_sr=sr, target_sr=SR)
            for c in range(y.shape[1])
        ], axis=1)
    return y.astype(np.float32)


def _band_rms(y_mono: np.ndarray, lo: float, hi: float) -> float:
    nyq = SR / 2.0
    sos = butter(4, [lo / nyq, hi / nyq], btype="band", output="sos")
    band = sosfilt(sos, y_mono)
    rms = float(np.sqrt(np.mean(band ** 2) + 1e-12))
    return float(20 * np.log10(rms + 1e-12))


def _band_db_stft(S: np.ndarray, freqs: np.ndarray, lo: float, hi: float) -> float:
    mask = (freqs >= lo) & (freqs < hi)
    if not mask.any():
        return -60.0
    return float(librosa.amplitude_to_db(S[mask].mean() + 1e-9, ref=1.0))


def analyze(path: str) -> dict:
    """Compute all metrics that distinguish professional from amateur tracks."""
    y = _load(path)
    L, R = y[:, 0], y[:, 1]
    mono = (L + R) / 2.0

    # ── Loudness ─────────────────────────────────────────────────────────────
    meter = pyln.Meter(SR)
    try:
        lufs = float(meter.integrated_loudness(mono))
    except Exception:
        lufs = -20.0
    try:
        lra = float(meter.loudness_range(mono))
    except Exception:
        lra = 0.0

    true_peak = float(20 * np.log10(max(float(np.abs(y).max()), 1e-9)))
    rms = float(np.sqrt(np.mean(mono ** 2) + 1e-9))
    peak = float(np.abs(mono).max() + 1e-9)
    crest_db = float(20 * np.log10(peak / rms))

    # ── Stereo imaging ────────────────────────────────────────────────────────
    corr = float(np.corrcoef(L, R)[0, 1]) if (L.std() > 1e-9 and R.std() > 1e-9) else 1.0

    # M/S balance: how much content is in the sides vs the center?
    M = (L + R) / np.sqrt(2.0)
    S = (L - R) / np.sqrt(2.0)
    m_rms = float(np.sqrt(np.mean(M ** 2) + 1e-12))
    s_rms = float(np.sqrt(np.mean(S ** 2) + 1e-12))
    stereo_width_db = float(20 * np.log10((s_rms + 1e-9) / (m_rms + 1e-9)))
    # Pro tracks: sides are 6-18 dB quieter than mid (-18 to -6 dB)
    # Amateur mono: sides near -60 dB. Over-widened: > -4 dB.

    # ── Spectral analysis ─────────────────────────────────────────────────────
    n_fft = 2048
    S_stft = np.abs(librosa.stft(mono, n_fft=n_fft))
    freqs = librosa.fft_frequencies(sr=SR, n_fft=n_fft)

    sub_db    = _band_db_stft(S_stft, freqs,   20,   80)
    bass_db   = _band_db_stft(S_stft, freqs,   80,  250)
    lowmid_db = _band_db_stft(S_stft, freqs,  250,  800)
    mid_db    = _band_db_stft(S_stft, freqs,  800, 2500)
    himid_db  = _band_db_stft(S_stft, freqs, 2500, 6000)
    high_db   = _band_db_stft(S_stft, freqs, 6000,10000)
    air_db    = _band_db_stft(S_stft, freqs,10000,20000)

    # ── Spectral slope (how well it follows pink noise) ───────────────────────
    # Pro mixes: -3 to -8 dB/octave from 200Hz to 10kHz (pink noise ≈ -3 dB/oct)
    octave_bands = [(200,400),(400,800),(800,1600),(1600,3200),(3200,6400),(6400,12800)]
    octave_db = [_band_db_stft(S_stft, freqs, lo, hi) for lo, hi in octave_bands]
    x = np.arange(len(octave_db), dtype=float)
    try:
        coeffs = np.polyfit(x, octave_db, 1)
        spectral_slope = float(coeffs[0])
    except Exception:
        spectral_slope = -5.0

    # ── Spectral centroid variability (musical movement) ─────────────────────
    # A boring/flat track has near-constant spectral centroid.
    # A professional track with dynamics, drops, builds has HIGH variability.
    # Measured as coefficient of variation of centroid over 1-second windows.
    try:
        centroids = librosa.feature.spectral_centroid(y=mono, sr=SR, hop_length=512)[0]
        cv_centroid = float(centroids.std() / (centroids.mean() + 1e-9))
    except Exception:
        cv_centroid = 0.1

    # ── Harmonic richness (warmth/fullness from overtones) ────────────────────
    # Pro tracks have harmonic content (saturation, natural instruments).
    # Computed as ratio of high harmonics (4-8kHz) to fundamental zone (200-800Hz).
    # Too little = thin/clinical. Too much = harsh/distorted.
    harm_ratio = himid_db - lowmid_db  # hi-mid vs lo-mid in dB

    # ── Transient definition (kick/snare punch) ───────────────────────────────
    try:
        flux = librosa.onset.onset_strength(y=mono, sr=SR, hop_length=512)
        p99 = float(np.percentile(flux, 99))
        p50 = float(np.percentile(flux, 50))
        transient_clarity = float(np.clip((p99 - p50) / (p50 + 1e-6), 0.0, 5.0))
    except Exception:
        transient_clarity = 0.5

    # Kick headroom: how much louder the kick transients are vs sustained bass
    try:
        sos_kick = butter(4, [60.0/(SR/2), 150.0/(SR/2)], btype="band", output="sos")
        kick_band = sosfilt(sos_kick, mono)
        hop = 512
        frames = librosa.util.frame(kick_band, frame_length=1024, hop_length=hop)
        frms_db = 20 * np.log10(np.sqrt((frames**2).mean(axis=0) + 1e-12))
        kick_headroom = float(np.percentile(frms_db, 95) - np.percentile(frms_db, 20))
    except Exception:
        kick_headroom = 8.0

    # ── Section consistency (loudness variance across track) ──────────────────
    try:
        win = SR * 15
        n_windows = len(mono) // win
        if n_windows >= 2:
            wlufs = []
            for i in range(n_windows):
                seg = mono[i*win:(i+1)*win]
                try:
                    wl = float(meter.integrated_loudness(seg))
                    if np.isfinite(wl) and wl > -70:
                        wlufs.append(wl)
                except Exception:
                    pass
            section_std = float(np.std(wlufs)) if len(wlufs) >= 2 else 0.0
        else:
            section_std = 0.0
    except Exception:
        section_std = 0.0

    # ── Mud index: lo-mid vs mid ──────────────────────────────────────────────
    try:
        low_e = float(S_stft[(freqs >= 200) & (freqs < 600)].mean() + 1e-9)
        mid_e = float(S_stft[(freqs >= 1000) & (freqs < 3000)].mean() + 1e-9)
        mud_index = float(low_e / mid_e)
    except Exception:
        mud_index = 2.0

    # ── Click artifact score ──────────────────────────────────────────────────
    diff = np.diff(mono)
    diff_rms = float(np.sqrt(np.mean(diff**2) + 1e-12))
    n_clicks = int(np.sum(np.abs(diff) > 10.0 * diff_rms))
    click_score = float(n_clicks / (len(diff) + 1e-9))

    # ── Tempo stability ───────────────────────────────────────────────────────
    try:
        _, beats = librosa.beat.beat_track(y=mono, sr=SR, hop_length=512)
        if len(beats) >= 3:
            btimes = librosa.frames_to_time(beats, sr=SR, hop_length=512)
            ibis = np.diff(btimes)
            cv = float(ibis.std() / (ibis.mean() + 1e-9))
            tempo_stability = float(1.0 - min(cv, 1.0))
        else:
            tempo_stability = 1.0
    except Exception:
        tempo_stability = 0.8

    return {
        "lufs": lufs,
        "true_peak": true_peak,
        "lra": lra,
        "crest_db": crest_db,
        "stereo_corr": corr,
        "stereo_width_db": stereo_width_db,
        "spectral_slope": spectral_slope,
        "cv_centroid": cv_centroid,
        "transient_clarity": transient_clarity,
        "kick_headroom": kick_headroom,
        "section_std": section_std,
        "mud_index": mud_index,
        "click_score": click_score,
        "tempo_stability": tempo_stability,
        "harm_ratio": harm_ratio,
        # Raw bands for display
        "_sub": sub_db, "_bass": bass_db, "_lowmid": lowmid_db,
        "_mid": mid_db, "_himid": himid_db, "_high": high_db, "_air": air_db,
        # Ratios vs mid
        "r_bass": bass_db - mid_db,
        "r_lowmid": lowmid_db - mid_db,
        "r_himid": himid_db - mid_db,
        "r_high": high_db - mid_db,
        "r_air": air_db - mid_db,
    }


def grade(m: dict) -> tuple[int, list]:
    """
    Score 0-100 based on how professional the track sounds.
    Returns (score, issues_list) where each issue is (severity, description, detail).
    """
    score = 100
    issues = []

    def deduct(pts: int, sev: str, msg: str, detail: str = ""):
        nonlocal score
        score -= pts
        issues.append((sev, msg, detail))

    # ── CRITICAL: will sound broken regardless of anything else ──────────────
    if m["true_peak"] > 0.0:
        deduct(30, "CRITICAL", "CLIPPING — digital distortion / float overflow",
               f"True peak {m['true_peak']:.1f} dBFS (must be below 0)")
    elif m["true_peak"] >= -0.3:
        deduct(8, "HIGH", "At ceiling — streaming codec distortion risk",
               f"True peak {m['true_peak']:.1f} dBFS (streaming target: -1.0 dBFS)")
    elif m["true_peak"] >= -0.8:
        deduct(3, "MEDIUM", "Close to ceiling — codec headroom tight",
               f"True peak {m['true_peak']:.1f} dBFS")

    if m["stereo_corr"] < 0.1:
        deduct(25, "CRITICAL", "SEVERE PHASE CANCELLATION — sounds hollow in mono",
               f"Stereo correlation {m['stereo_corr']:.2f}")

    if m["click_score"] > 0.02:
        deduct(20, "CRITICAL", "SEVERE CLICK/POP ARTIFACTS — audible pops throughout",
               f"{m['click_score']*100:.2f}% of samples are clicks")
    elif m["click_score"] > 0.005:
        deduct(12, "HIGH", "Click/pop artifacts present",
               f"{m['click_score']*100:.3f}% click rate")

    # ── Loudness: streaming platform requirements ─────────────────────────────
    if m["lufs"] < -24.0:
        deduct(20, "HIGH", "TOO QUIET — streaming will amplify and it'll sound weak",
               f"Integrated LUFS {m['lufs']:.1f} (need > -24 minimum, pro = -14 to -8)")
    elif m["lufs"] < -18.0:
        deduct(12, "HIGH", "Under-loud — will get boosted up by streaming, revealing noise floor",
               f"Integrated LUFS {m['lufs']:.1f}")
    elif m["lufs"] < -15.0:
        deduct(5, "MEDIUM", "Slightly quiet for streaming platforms",
               f"LUFS {m['lufs']:.1f} (target -14 to -8)")
    elif m["lufs"] > -5.0:
        deduct(15, "HIGH", "OVER-LIMITED — crushed dynamics, sounds fatiguing",
               f"LUFS {m['lufs']:.1f} (over-compressed)")

    # ── Dynamics ──────────────────────────────────────────────────────────────
    if m["lra"] < 1.5:
        deduct(15, "HIGH", "BRICK-WALLED — no dynamics, sounds completely flat/fatiguing",
               f"LRA {m['lra']:.1f} LU (target 4-12 LU)")
    elif m["lra"] < 3.0:
        deduct(8, "MEDIUM", "Over-compressed — limited dynamics, sounds squashed",
               f"LRA {m['lra']:.1f} LU")
    elif m["lra"] > 18.0:
        deduct(8, "MEDIUM", "Too dynamic — inconsistent levels, needs compression",
               f"LRA {m['lra']:.1f} LU")

    if m["crest_db"] < 5.0:
        deduct(12, "HIGH", "SMASHED — transients destroyed, beat blurs at high volume",
               f"Crest factor {m['crest_db']:.1f} dB (target 8-18 dB)")
    elif m["crest_db"] < 7.0:
        deduct(6, "MEDIUM", "Over-limited — transients are soft, lacks punch",
               f"Crest factor {m['crest_db']:.1f} dB")

    # ── Stereo image ──────────────────────────────────────────────────────────
    if m["stereo_corr"] < 0.25:
        deduct(12, "HIGH", "PHASE ISSUES — out-of-phase elements, bad in mono",
               f"Correlation {m['stereo_corr']:.2f}")
    elif m["stereo_corr"] > 0.97:
        deduct(6, "MEDIUM", "Essentially MONO — no stereo width",
               f"Correlation {m['stereo_corr']:.2f} (sounds flat, no space)")

    if m["stereo_width_db"] > -3.0:
        deduct(8, "HIGH", "OVER-WIDENED — phasey, collapes badly in mono",
               f"Sides only {abs(m['stereo_width_db']):.1f} dB below mid")
    elif m["stereo_width_db"] < -25.0:
        deduct(5, "MEDIUM", "Very narrow stereo image — lacks space and dimension",
               f"Sides {abs(m['stereo_width_db']):.1f} dB below mid")

    # ── Spectral balance ──────────────────────────────────────────────────────
    if m["spectral_slope"] > -1.0:
        deduct(10, "HIGH", "TOO BRIGHT / HARSH — spectrum not rolling off naturally",
               f"Slope {m['spectral_slope']:.1f} dB/oct (pro = -3 to -8)")
    elif m["spectral_slope"] < -14.0:
        deduct(10, "HIGH", "TOO DARK / MUFFLED — highs completely dead",
               f"Slope {m['spectral_slope']:.1f} dB/oct")
    elif m["spectral_slope"] > -2.0 or m["spectral_slope"] < -11.0:
        deduct(5, "MEDIUM", "Unbalanced spectral slope",
               f"Slope {m['spectral_slope']:.1f} dB/oct")

    if m["r_bass"] > 22.0:
        deduct(8, "HIGH", "BASS TOO DOMINANT — low end overwhelming the mix",
               f"Bass {m['r_bass']:+.1f} dB vs mid (target +2 to +20)")
    elif m["r_bass"] < 0.0:
        deduct(6, "MEDIUM", "THIN / NO BASS — mix sounds weak on speakers",
               f"Bass {m['r_bass']:+.1f} dB vs mid")

    if m["mud_index"] > 6.5:
        deduct(10, "HIGH", "MUDDY — 200-600Hz buildup, vocal intelligibility destroyed",
               f"Mud index {m['mud_index']:.2f} (target 1.0-5.5)")
    elif m["mud_index"] > 5.5:
        deduct(5, "MEDIUM", "Slightly muddy low-mids",
               f"Mud index {m['mud_index']:.2f}")
    elif m["mud_index"] < 0.6:
        deduct(6, "MEDIUM", "SCOOPED — hollow, lacks body, phone-speaker sound",
               f"Mud index {m['mud_index']:.2f}")

    if m["r_himid"] > 0.0:
        deduct(10, "HIGH", "HARSH / EAR-FATIGUING — 2-6kHz too loud, causes listening fatigue",
               f"Hi-mid {m['r_himid']:+.1f} dB vs mid (should be below mid)")
    elif m["r_himid"] > -2.0:
        deduct(5, "MEDIUM", "Slightly forward hi-mids — borderline harsh",
               f"Hi-mid {m['r_himid']:+.1f} dB vs mid")

    if m["r_air"] < -35.0:
        deduct(6, "MEDIUM", "NO AIR — 10kHz+ completely absent, sounds flat and dull",
               f"Air {m['r_air']:+.1f} dB vs mid (target -20 to -5)")

    # ── Transient definition ──────────────────────────────────────────────────
    if m["transient_clarity"] < 0.15:
        deduct(8, "HIGH", "SMASHED TRANSIENTS — kicks/snares have no definition",
               f"Transient clarity {m['transient_clarity']:.2f} (target > 0.15)")
    elif m["transient_clarity"] < 0.3:
        deduct(4, "MEDIUM", "Soft transients — beat lacks punch",
               f"Transient clarity {m['transient_clarity']:.2f}")

    if m["kick_headroom"] < 3.0:
        deduct(7, "HIGH", "KICK BURIED — kick and bass blend together, no punch at volume",
               f"Kick headroom {m['kick_headroom']:.1f} dB (target >5 dB)")
    elif m["kick_headroom"] < 5.0:
        deduct(3, "MEDIUM", "Kick slightly buried in bass",
               f"Kick headroom {m['kick_headroom']:.1f} dB")

    # ── Musical variation ─────────────────────────────────────────────────────
    if m["cv_centroid"] < 0.08:
        deduct(8, "MEDIUM", "FLAT / NO MOVEMENT — no spectral variation, sounds like a loop",
               f"Spectral centroid CV {m['cv_centroid']:.3f} (pro tracks: > 0.12)")
    elif m["cv_centroid"] < 0.12:
        deduct(3, "MEDIUM", "Limited musical variation / dynamics",
               f"CV {m['cv_centroid']:.3f}")

    if m["section_std"] > 8.0:
        deduct(6, "MEDIUM", "INCONSISTENT LEVELS — mix gets much louder/quieter across sections",
               f"Section std {m['section_std']:.1f} LU")

    if m["tempo_stability"] < 0.6:
        deduct(8, "HIGH", "TEMPO DRIFT — unstable beat, time-stretch artifacts audible",
               f"Tempo stability {m['tempo_stability']:.2f}")

    return max(0, score), issues


def _tier(score: int) -> tuple[str, str]:
    if score >= 88:
        return "S", "Chart-ready / Professional"
    elif score >= 78:
        return "A", "Commercial quality"
    elif score >= 65:
        return "B", "Good — minor issues"
    elif score >= 50:
        return "C", "Acceptable — noticeable problems"
    elif score >= 35:
        return "D", "Amateur — significant issues"
    else:
        return "F", "Not release-ready"


def report(path: str, verbose: bool = False) -> int:
    """Analyze a track and print a quality report. Returns numeric score."""
    print(f"\n{'═'*60}")
    print(f"  🎵  {Path(path).name}")
    print(f"{'═'*60}")

    try:
        m = analyze(path)
    except Exception as e:
        print(f"  ✗  ERROR reading file: {e}")
        return 0

    score, issues = grade(m)
    tier, desc = _tier(score)

    # Determine duration
    try:
        info = sf.info(path)
        dur_s = info.duration
        dur = f"{int(dur_s//60)}:{int(dur_s%60):02d}"
    except Exception:
        dur = "??:??"

    print(f"\n  SCORE: {score}/100   GRADE: {tier} — {desc}")
    print(f"  Duration: {dur}   LUFS: {m['lufs']:.1f}   LRA: {m['lra']:.1f} LU")
    print()

    if not issues:
        print("  ✓  No issues detected — professional quality throughout")
    else:
        for sev, msg, detail in sorted(issues, key=lambda x: {"CRITICAL":0,"HIGH":1,"MEDIUM":2}[x[0]]):
            sym = "✗✗" if sev == "CRITICAL" else ("✗ " if sev == "HIGH" else "△ ")
            print(f"  {sym} [{sev}] {msg}")
            if detail and verbose:
                print(f"        {detail}")

    if verbose:
        print(f"\n  SPECTRAL BREAKDOWN:")
        print(f"  Sub  20-80Hz      {m['_sub']:+.1f} dB")
        print(f"  Bass 80-250Hz     {m['_bass']:+.1f} dB")
        print(f"  LoMid 250-800Hz   {m['_lowmid']:+.1f} dB")
        print(f"  Mid 800-2.5kHz    {m['_mid']:+.1f} dB")
        print(f"  HiMid 2.5-6kHz   {m['_himid']:+.1f} dB")
        print(f"  High 6-10kHz     {m['_high']:+.1f} dB")
        print(f"  Air  10-20kHz    {m['_air']:+.1f} dB")
        print(f"\n  DYNAMICS:")
        print(f"  Crest factor      {m['crest_db']:+.1f} dB")
        print(f"  Stereo corr       {m['stereo_corr']:.3f}  (1.0=mono, 0.0=max-wide)")
        print(f"  Stereo width      {m['stereo_width_db']:+.1f} dB (sides vs mid)")
        print(f"  Spectral slope    {m['spectral_slope']:.1f} dB/oct")
        print(f"  Centroid var      {m['cv_centroid']:.3f}  (low=boring, high=dynamic)")
        print(f"  Transient clarity {m['transient_clarity']:.2f}")
        print(f"  Kick headroom     {m['kick_headroom']:.1f} dB")
        print(f"  Mud index         {m['mud_index']:.2f}")
        print(f"  Tempo stability   {m['tempo_stability']:.2f}")
        print(f"  Section std       {m['section_std']:.1f} LU")

    print(f"\n{'─'*60}")

    return score


def batch_report(paths: list, verbose: bool = False):
    """Analyze multiple files and show ranked results."""
    results = []
    for p in paths:
        score = report(p, verbose=verbose)
        tier, desc = _tier(score)
        results.append((score, tier, Path(p).name))

    if len(results) > 1:
        print(f"\n{'═'*60}")
        print(f"  RANKING  ({len(results)} tracks)")
        print(f"{'═'*60}")
        for score, tier, name in sorted(results, reverse=True):
            bar = "█" * (score // 5) + "░" * (20 - score // 5)
            print(f"  {score:3d}/100  [{tier}]  [{bar}]  {name[:35]}")
        print()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="VocalFusion Commercial Quality Grader")
    parser.add_argument("files", nargs="+", help="Audio files to analyze (mp3/wav/flac)")
    parser.add_argument("--verbose", "-v", action="store_true",
                        help="Show detailed breakdown")
    args = parser.parse_args()

    paths = [str(p) for f in args.files for p in Path(".").glob(f) or [Path(f)]]
    if not paths:
        paths = args.files

    if len(paths) == 1:
        score = report(paths[0], verbose=args.verbose)
        tier, _ = _tier(score)
        sys.exit(0 if tier in ("S", "A") else (1 if tier == "B" else 2))
    else:
        batch_report(paths, verbose=args.verbose)
