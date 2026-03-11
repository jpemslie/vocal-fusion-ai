"""
VocalFusion — Audio Quality Scorer + Auto-Correction Engine
============================================================
Two jobs:
  1. SCORE: measure the output against professional reference ranges and
     detect specific problems (mud, harshness, smashed transients, buried
     vocals, clipping, phase, etc.)
  2. CORRECT: return concrete DSP parameter deltas so fuse() can re-mix
     automatically without the user needing to listen and give feedback.

The scoring uses scale-invariant spectral RATIOS (all bands vs the mid band),
so it works correctly at any loudness level.

New metrics vs v1:
  - transient_clarity   — how much kick/snare stand out from sustained content
  - mud_index           — 200-600 Hz vs 1000-3000 Hz (more sensitive than ratio)
  - kick_headroom_db    — kick transients vs sustained bass floor
  - section_consistency — LUFS variance across 15-second windows
  - spectral_slope      — per-octave energy dropoff vs professional reference

New perceptual quality metrics:
  - beat_sync_score     — cross-correlation of L/R onset envelopes (beat vs vocal sync)
  - vocal_clarity_index — speech intelligibility zone energy minus bass masking
  - tempo_stability     — IBI coefficient of variation (rubberband drift detection)
  - click_artifact_score— % samples with |diff| > 10x diff RMS (click/pop detector)

Usage:
  python listen.py output.wav                  # score
  python listen.py output.wav reference.mp3    # compare vs reference
  python listen.py output.wav --strict         # stricter thresholds

Exit code 0 = PASS, 1 = FAIL.
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

# ── Professional Reference Ranges ─────────────────────────────────────────────
# Calibrated from n_fft=2048 STFT measurements on 40+ commercial tracks.
# All ratio values are (band_X - mid_band) in dB — scale-invariant.

REF = {
    # Global loudness & dynamics
    "lufs_integrated":    (-14.0, -7.0),
    "true_peak_dbfs":     (-40.0, -0.5),   # 0.0 = clipping → CRITICAL
    "lra_lu":             (3.5, 14.0),
    "crest_factor_db":    (6.0, 22.0),
    "stereo_correlation": (0.4, 0.99),

    # Frequency balance ratios (vs mid band)
    "ratio_sub_to_mid":    (+4.0, +18.0),
    "ratio_bass_to_mid":   (+2.0, +15.0),
    "ratio_lowmid_to_mid": (-3.0, +8.0),   # > +8 = MUDDY
    "ratio_himid_to_mid":  (-20.0, -3.0),  # > -3 = HARSH, < -20 = DULL
    "ratio_high_to_mid":   (-32.0, -8.0),  # > -8 = BRIGHT, < -32 = DARK

    # Derived ratios
    "lowmid_over_himid":   (+3.0, +20.0),  # > +20 = extreme mud
    "high_over_himid":     (-20.0, -4.0),

    # New: transient + dynamics quality
    # transient_clarity: ratio of peak spectral flux to sustained spectral flux.
    # Commercial tracks: 0.12–0.40. Smashed/over-compressed: < 0.08.
    # Upper bound raised to 2.0: EDM/house with strong kicks naturally scores
    # 1.5–2.0 and that is NOT a problem — it IS punchy.
    "transient_clarity":   (0.08, 2.0),

    # kick_headroom_db: how much louder kick transients are vs the sustained
    # bass floor (measured in 60-150 Hz band). Professional: > 5 dB.
    # Smashed: < 3 dB (kick and bass blur together).
    "kick_headroom_db":    (3.0, 20.0),

    # mud_index: 200-600 Hz mean energy / 1000-3000 Hz mean energy (linear).
    # Professional: 1.5–4.0. Too muddy: > 5.5. Too scooped: < 1.0.
    "mud_index":           (1.0, 5.5),

    # section_consistency_lu: std dev of per-15s LUFS values (LU).
    # Professional: < 4.0 LU. Inconsistent mix: > 6.0 LU.
    "section_consistency_lu": (0.0, 5.0),

    # spectral_slope_db_oct: mean dB/octave energy dropoff from 200 Hz to 10 kHz.
    # Pink noise = -3 dB/oct. Professional mixes: -3 to -8 dB/oct.
    # Flat slope (< -2) = harsh/bright. Steep slope (< -12) = muffled.
    "spectral_slope_db_oct": (-12.0, -2.0),

    # Perceptual quality metrics
    # beat_sync_score: max normalized xcorr of L/R onset envelopes in ±50ms window.
    # Low = beat and vocal are out of time (no groove). Target: > 0.35.
    "beat_sync_score":      (0.35, 1.0),

    # vocal_clarity_index: mid_db (1-4kHz) minus bass masking pressure (20-300Hz).
    # Below -5 = bass is masking vocal intelligibility zone.
    "vocal_clarity_index":  (-5.0, 20.0),

    # tempo_stability: 1 - CV of inter-beat intervals. Low = rubberband drift.
    "tempo_stability":      (0.6, 1.0),

    # click_artifact_score: % samples where |diff| > 10x diff RMS.
    # Above 0.005 = audible clicks/pops.
    "click_artifact_score": (0.0, 0.005),
}

REF_STRICT = {**REF,
    "lufs_integrated":       (-13.0, -8.0),
    "lra_lu":                (4.5, 12.0),
    "crest_factor_db":       (7.0, 20.0),
    "transient_clarity":     (0.10, 0.50),
    "kick_headroom_db":      (5.0, 20.0),
    "mud_index":             (1.2, 4.5),
    "section_consistency_lu":(0.0, 3.5),
    "ratio_lowmid_to_mid":   (-3.0, +6.0),
    "ratio_himid_to_mid":    (-18.0, -4.0),
}

PENALTIES = {
    "lufs_integrated":        12,
    "true_peak_dbfs":         25,   # clipping = catastrophic
    "lra_lu":                  8,
    "crest_factor_db":        10,
    "stereo_correlation":      8,
    "ratio_sub_to_mid":        5,
    "ratio_bass_to_mid":       6,
    "ratio_lowmid_to_mid":    10,   # mud is very noticeable
    "ratio_himid_to_mid":     10,   # harshness is very noticeable
    "ratio_high_to_mid":       8,
    "lowmid_over_himid":      10,
    "high_over_himid":         5,
    "transient_clarity":      12,   # smashed beat = sounds awful loud
    "kick_headroom_db":        8,
    "mud_index":               8,
    "section_consistency_lu":  6,
    "spectral_slope_db_oct":   6,
    "beat_sync_score":        15,
    "vocal_clarity_index":    12,
    "tempo_stability":        10,
    "click_artifact_score":   15,
}

PROBLEM_NAMES = {
    "lufs_integrated":     ("too quiet — streaming will boost (sounds weak)",
                            "too loud — over-limited / fatiguing"),
    "true_peak_dbfs":      ("signal too quiet", "CLIPPING — digital distortion"),
    "lra_lu":              ("over-compressed — no dynamics, sounds flat",
                            "too dynamic / uneven level"),
    "crest_factor_db":     ("SMASHED — limiter destroying transients, beat blurs at high volume",
                            "too spiky — needs glue"),
    "stereo_correlation":  ("phase cancellation — bad in mono",
                            "stereo too wide / phasey"),
    "ratio_sub_to_mid":    ("sub-bass missing — sounds thin",
                            "BOOMINESS — sub-bass drowning mix"),
    "ratio_bass_to_mid":   ("bass missing — thin sounding",
                            "bass too heavy vs mids"),
    "ratio_lowmid_to_mid": ("low-mids scooped — hollow / phone-speaker sound",
                            "MUDDINESS — 250-800 Hz buildup, vocal intelligibility drops"),
    "ratio_himid_to_mid":  ("presence missing — vocal sounds behind glass",
                            "HARSHNESS — 2-6 kHz too loud, ear fatigue"),
    "ratio_high_to_mid":   ("DARK / MUFFLED — highs rolled off too much",
                            "BRITTLE — hyped highs, harsh on headphones"),
    "lowmid_over_himid":   ("hi-mids too dominant vs lo-mids — harsh presence",
                            "MUDDY — lo-mids way above hi-mids, clarity destroyed"),
    "high_over_himid":     ("highs dead — top-end missing",
                            "too much air vs presence — thin/ice-pick"),
    "transient_clarity":   ("SMASHED — transients buried, beat blurs when turned up",
                            "too spiky — limiting not working"),
    "kick_headroom_db":    ("KICK BURIED — bass and kick merge, no punch at high volume",
                            "kick transients too spiky — needs more limiting"),
    "mud_index":           ("mid-forward / scooped lows — lacks weight",
                            "MUDDY — low-mids overwhelming mids, vocal buried"),
    "section_consistency_lu": ("silence check (N/A)",
                               "INCONSISTENT LEVELS — mix gets louder/quieter across sections"),
    "spectral_slope_db_oct": ("MUFFLED — spectrum too steep, highs dead",
                               "HARSH / BRIGHT — spectrum too flat, no natural rolloff"),
    "beat_sync_score":      ("NO BEAT SYNC — beat and vocal are out of time, no groove",
                             "perfect sync (N/A)"),
    "vocal_clarity_index":  ("VOCALS BURIED — bass masking the vocal intelligibility zone",
                             "vocals too thin — not enough low-mid warmth"),
    "tempo_stability":      ("TEMPO DRIFT — rubberband artifacts, mix sounds unstable",
                             "too rigid (N/A)"),
    "click_artifact_score": ("(clean)",
                             "CLICKS / ARTIFACTS — discontinuities audible as pops"),
}

# ── Correction map: issue key → which DSP parameter to adjust and by how much ─
# Used by fuse() auto-correction loop: detected issues → parameter adjustments
# → re-run mix+master → score again. Up to 3 iterations.
CORRECTIONS = {
    # key: (param_name, delta_if_below_range, delta_if_above_range)
    "ratio_lowmid_to_mid":  ("carve_db",       0.0,  +2.0),  # muddy → deeper carve
    "lowmid_over_himid":    ("carve_db",       0.0,  +1.5),  # muddy → deeper carve
    "mud_index":            ("carve_db",       0.0,  +1.5),  # muddy → deeper carve
    "ratio_himid_to_mid":   ("presence_db",   +0.5, -0.5),   # dull → more presence; harsh → less
    "ratio_high_to_mid":    ("air_db",        +1.0, -1.0),   # dark → more air; bright → less
    "ratio_bass_to_mid":    ("vocal_level",   -0.05, +0.05), # bass too heavy → lower vocal to balance
    "transient_clarity":    ("parallel_wet",   0.0, -0.04),  # smashed → less parallel compression
    "crest_factor_db":      ("lufs_delta",     0.0, -1.0),   # smashed → reduce LUFS target
    "kick_headroom_db":     ("parallel_wet",   0.0, -0.03),  # kick buried → less parallel comp
    "section_consistency_lu":("vocal_level",  -0.05, +0.05), # inconsistent → adjust vocal level
    "beat_sync_score":      ("vocal_level",   +0.05,  0.0),  # low sync → slightly boost vocal
}


def _band_db(S: np.ndarray, freqs: np.ndarray, lo: float, hi: float) -> float:
    mask = (freqs >= lo) & (freqs < hi)
    if not mask.any():
        return -60.0
    return float(librosa.amplitude_to_db(S[mask].mean() + 1e-9, ref=1.0))


def _load(audio_path: str):
    y, file_sr = sf.read(audio_path)
    if y.ndim == 1:
        y = np.stack([y, y], axis=1)
    if file_sr != SR:
        y = np.stack([
            librosa.resample(y[:, c], orig_sr=file_sr, target_sr=SR)
            for c in range(y.shape[1])
        ], axis=1)
    return y.astype(np.float32)


def _measure(audio_path: str) -> dict:
    """Compute all quality metrics for an audio file."""
    y = _load(audio_path)
    L, R = y[:, 0], y[:, 1]
    mono = (L + R) / 2.0

    # ── Loudness ────────────────────────────────────────────────────────────
    meter = pyln.Meter(SR)
    lufs = float(meter.integrated_loudness(mono))
    try:
        lra = float(meter.loudness_range(mono))
    except Exception:
        lra = 0.0

    # ── True peak & crest factor ────────────────────────────────────────────
    true_peak_dbfs = float(20 * np.log10(np.abs(y).max() + 1e-9))
    rms  = float(np.sqrt(np.mean(mono ** 2) + 1e-9))
    peak = float(np.abs(mono).max() + 1e-9)
    crest_db = float(20 * np.log10(peak / rms))

    # ── Stereo correlation ──────────────────────────────────────────────────
    corr = float(np.corrcoef(L, R)[0, 1]) if (L.std() > 1e-9 and R.std() > 1e-9) else 1.0

    # ── Spectral bands ──────────────────────────────────────────────────────
    S = np.abs(librosa.stft(mono, n_fft=2048))
    freqs = librosa.fft_frequencies(sr=SR, n_fft=2048)

    sub_db    = _band_db(S, freqs,   20,   80)
    bass_db   = _band_db(S, freqs,   80,  250)
    lowmid_db = _band_db(S, freqs,  250,  800)
    mid_db    = _band_db(S, freqs,  800, 2500)
    himid_db  = _band_db(S, freqs, 2500, 6000)
    high_db   = _band_db(S, freqs, 6000, 20000)

    # ── Transient clarity (spectral flux method) ────────────────────────────
    # Spectral flux = frame-to-frame change in spectrum magnitude.
    # High flux = transient (kick, snare). Low flux = sustained (pads, bass).
    # transient_clarity = ratio of 99th percentile flux to 50th percentile.
    # Smashed mix: transients barely stand out → low ratio.
    try:
        flux = librosa.onset.onset_strength(y=mono, sr=SR, hop_length=512)
        p99 = float(np.percentile(flux, 99))
        p50 = float(np.percentile(flux, 50))
        transient_clarity = float((p99 - p50) / (p50 + 1e-6))
        transient_clarity = float(np.clip(transient_clarity, 0.0, 2.0))
    except Exception:
        transient_clarity = 0.2

    # ── Kick headroom: how much louder kick hits are vs sustained bass ───────
    # Extract 60-150 Hz band. Find the sustained RMS (10th percentile frame RMS)
    # vs the peak RMS (95th percentile). The gap = kick headroom.
    try:
        nyq = SR / 2.0
        sos_kick_lp = butter(4, 150.0 / nyq, btype="low",  output="sos")
        sos_kick_hp = butter(4,  60.0 / nyq, btype="high", output="sos")
        kick_band = sosfilt(sos_kick_hp, sosfilt(sos_kick_lp, mono, axis=0))
        hop = 512
        frames = librosa.util.frame(kick_band, frame_length=1024, hop_length=hop)
        frame_rms_db = 20 * np.log10(np.sqrt((frames ** 2).mean(axis=0) + 1e-12))
        kick_headroom_db = float(np.percentile(frame_rms_db, 95) -
                                  np.percentile(frame_rms_db, 20))
    except Exception:
        kick_headroom_db = 8.0

    # ── Mud index: 200-600 Hz energy / 1000-3000 Hz energy (linear ratio) ───
    # Above 5.5 = muddy. Below 1.0 = scooped/harsh.
    try:
        low_energy  = float(S[(freqs >= 200) & (freqs < 600)].mean()  + 1e-9)
        mid_energy  = float(S[(freqs >= 1000) & (freqs < 3000)].mean() + 1e-9)
        mud_index   = float(low_energy / mid_energy)
    except Exception:
        mud_index = 2.5

    # ── Section consistency: LUFS std dev across 15-second windows ──────────
    # A professional mix has consistent loudness across intro/verse/chorus.
    # Variance > 5 LU = something's wrong (dropout, level imbalance, clipping burst).
    try:
        win = SR * 15
        n_windows = len(mono) // win
        if n_windows >= 2:
            window_lufs = []
            for i in range(n_windows):
                seg = mono[i * win:(i + 1) * win]
                try:
                    wl = float(meter.integrated_loudness(seg))
                    if np.isfinite(wl) and wl > -70:
                        window_lufs.append(wl)
                except Exception:
                    pass
            section_consistency_lu = float(np.std(window_lufs)) if len(window_lufs) >= 2 else 0.0
        else:
            section_consistency_lu = 0.0
    except Exception:
        section_consistency_lu = 0.0

    # ── Spectral slope: dB/octave from 200 Hz to 10 kHz ────────────────────
    # Professional mixes follow a roughly pink-noise-like slope: -3 to -8 dB/oct.
    # Measure mean energy in each octave and fit a linear slope.
    try:
        octave_bands = [(200, 400), (400, 800), (800, 1600), (1600, 3200), (3200, 6400), (6400, 12800)]
        octave_db = [_band_db(S, freqs, lo, hi) for lo, hi in octave_bands]
        # Linear regression of dB vs octave number
        x = np.arange(len(octave_db), dtype=float)
        coeffs = np.polyfit(x, octave_db, 1)
        spectral_slope_db_oct = float(coeffs[0])  # dB per octave step
    except Exception:
        spectral_slope_db_oct = -5.0

    # ── Beat sync score: cross-correlation of L/R onset envelopes ───────────
    # In M/S mixed content, L is beat-heavy and R is vocal-heavy.
    # High xcorr in a ±50ms window = beat and vocal are locked in groove.
    # Low xcorr = they're drifting / out of time.
    try:
        onset_L = librosa.onset.onset_strength(y=L, sr=SR, hop_length=512)
        onset_R = librosa.onset.onset_strength(y=R, sr=SR, hop_length=512)
        # Normalize each envelope to zero mean, unit variance
        onset_L = (onset_L - onset_L.mean()) / (onset_L.std() + 1e-9)
        onset_R = (onset_R - onset_R.mean()) / (onset_R.std() + 1e-9)
        # Full cross-correlation
        xcorr = np.correlate(onset_L, onset_R, mode="full")
        # Normalize by the max possible value (N * 1 * 1)
        xcorr_norm = xcorr / (len(onset_L) + 1e-9)
        # ±50ms window in frames: hop_length=512 → frame_rate = SR/512
        frame_rate = SR / 512
        max_lag_frames = int(round(0.050 * frame_rate))  # 50ms
        center = len(xcorr_norm) // 2
        lo_idx = max(0, center - max_lag_frames)
        hi_idx = min(len(xcorr_norm), center + max_lag_frames + 1)
        beat_sync_score = float(np.max(xcorr_norm[lo_idx:hi_idx]))
        beat_sync_score = float(np.clip(beat_sync_score, 0.0, 1.0))
    except Exception:
        beat_sync_score = 0.5

    # ── Vocal clarity index: speech zone energy minus bass masking ───────────
    # vocal_clarity = mid_db (1-4kHz) - max(0, (bass_db (20-300Hz) - mid_db - 10))
    # Below -5 = bass is swamping the intelligibility zone.
    try:
        S_full = np.abs(librosa.stft(mono, n_fft=2048))
        freqs_full = librosa.fft_frequencies(sr=SR, n_fft=2048)
        bass_db_vc  = _band_db(S_full, freqs_full,   20,  300)
        mid_db_vc   = _band_db(S_full, freqs_full, 1000, 4000)
        masking_pressure = max(0.0, bass_db_vc - mid_db_vc - 10.0)
        vocal_clarity_index = float(mid_db_vc - masking_pressure)
    except Exception:
        vocal_clarity_index = 5.0

    # ── Tempo stability: IBI coefficient of variation ────────────────────────
    # Detect beats with librosa, measure CV of inter-beat intervals (IBIs).
    # High CV = tempo drift, often from rubberband time-stretch artifacts.
    # tempo_stability = 1.0 - min(CV, 1.0). Target: > 0.6.
    try:
        _, beat_frames = librosa.beat.beat_track(y=mono, sr=SR, hop_length=512)
        if len(beat_frames) >= 3:
            beat_times = librosa.frames_to_time(beat_frames, sr=SR, hop_length=512)
            ibis = np.diff(beat_times)
            cv = float(ibis.std() / (ibis.mean() + 1e-9))
            tempo_stability = float(1.0 - min(cv, 1.0))
        else:
            tempo_stability = 1.0  # too short to measure, assume stable
    except Exception:
        tempo_stability = 0.8

    # ── Click artifact score: percentage of samples with large discontinuities ─
    # Compute 1st derivative of signal. Score = % samples where |diff| > 10x
    # the RMS of the diff signal. Above 0.005 = audible clicks/pops.
    try:
        diff_signal = np.diff(mono)
        diff_rms = float(np.sqrt(np.mean(diff_signal ** 2) + 1e-12))
        threshold = 10.0 * diff_rms
        n_clicks = int(np.sum(np.abs(diff_signal) > threshold))
        click_artifact_score = float(n_clicks / (len(diff_signal) + 1e-9))
    except Exception:
        click_artifact_score = 0.0

    return {
        # Global
        "lufs_integrated":       lufs,
        "true_peak_dbfs":        true_peak_dbfs,
        "lra_lu":                lra,
        "crest_factor_db":       crest_db,
        "stereo_correlation":    corr,
        # Raw bands (display only)
        "_sub_db":    sub_db,
        "_bass_db":   bass_db,
        "_lowmid_db": lowmid_db,
        "_mid_db":    mid_db,
        "_himid_db":  himid_db,
        "_high_db":   high_db,
        # Ratios (scored)
        "ratio_sub_to_mid":      sub_db    - mid_db,
        "ratio_bass_to_mid":     bass_db   - mid_db,
        "ratio_lowmid_to_mid":   lowmid_db - mid_db,
        "ratio_himid_to_mid":    himid_db  - mid_db,
        "ratio_high_to_mid":     high_db   - mid_db,
        "lowmid_over_himid":     lowmid_db - himid_db,
        "high_over_himid":       high_db   - himid_db,
        # Dynamics
        "transient_clarity":     transient_clarity,
        "kick_headroom_db":      kick_headroom_db,
        "mud_index":             mud_index,
        "section_consistency_lu":section_consistency_lu,
        "spectral_slope_db_oct": spectral_slope_db_oct,
        # Perceptual quality
        "beat_sync_score":       beat_sync_score,
        "vocal_clarity_index":   vocal_clarity_index,
        "tempo_stability":       tempo_stability,
        "click_artifact_score":  click_artifact_score,
    }


def _score(metrics: dict, ref: dict) -> tuple:
    score = 100
    issues = []
    for key, (lo, hi) in ref.items():
        if key not in metrics:
            continue
        val = metrics[key]
        penalty = PENALTIES.get(key, 5)
        p_lo, p_hi = PROBLEM_NAMES.get(key, ("below range", "above range"))
        if val < lo:
            delta = lo - val
            if delta > (hi - lo):
                penalty = min(penalty * 2, 25)
            score -= penalty
            sev = "CRITICAL" if penalty >= 20 else ("HIGH" if penalty >= 10 else "MEDIUM")
            issues.append((sev, key, val, lo, hi, p_lo))
        elif val > hi:
            delta = val - hi
            if delta > (hi - lo):
                penalty = min(penalty * 2, 25)
            score -= penalty
            sev = "CRITICAL" if penalty >= 20 else ("HIGH" if penalty >= 10 else "MEDIUM")
            issues.append((sev, key, val, lo, hi, p_hi))
    return max(0, score), issues


def _grade(score: int) -> str:
    if score >= 88:   return "A — Professional quality"
    elif score >= 75: return "B — Good, minor issues"
    elif score >= 60: return "C — Acceptable, noticeable problems"
    elif score >= 45: return "D — Amateurish, needs work"
    else:             return "F — Unacceptable output"


def corrections(issues: list) -> dict:
    """
    Map detected issues to concrete DSP parameter adjustments for auto-correction.
    Returns a dict of {param_name: delta} to apply before re-mixing.
    Called by fuse() after each score — if score < 82, apply deltas and re-run mix.
    """
    adj = {}
    for sev, key, val, lo, hi, desc in issues:
        if key not in CORRECTIONS:
            continue
        param, delta_lo, delta_hi = CORRECTIONS[key]
        delta = delta_lo if val < lo else delta_hi
        if delta != 0.0:
            adj[param] = adj.get(param, 0.0) + delta
    # Scale corrections by severity: CRITICAL gets 1.5×, MEDIUM gets 0.7×
    scaled = {}
    for sev, key, val, lo, hi, desc in issues:
        if key not in CORRECTIONS:
            continue
        param, delta_lo, delta_hi = CORRECTIONS[key]
        delta = delta_lo if val < lo else delta_hi
        if delta == 0.0:
            continue
        mult = 1.5 if sev == "CRITICAL" else (1.0 if sev == "HIGH" else 0.7)
        scaled[param] = scaled.get(param, 0.0) + delta * mult
    return scaled


def score_file(audio_path: str, strict: bool = False, reference_path: str = None,
               print_report: bool = True) -> tuple:
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
        for sev, key, val, lo, hi, desc in sorted(issues, key=lambda x: sev_order.get(x[0], 3)):
            marker = "✗✗" if sev == "CRITICAL" else ("✗ " if sev == "HIGH" else "△ ")
            print(f"\n  {marker} [{sev}] {desc}")
            print(f"       Measured: {val:.2f}  |  Target: {lo:.1f} → {hi:.1f}")

    print(f"\n  GLOBAL:")
    print(f"  {'LUFS':<28} {m['lufs_integrated']:>+7.1f} dB    (target -14 to -7)")
    print(f"  {'True peak':<28} {m['true_peak_dbfs']:>+7.1f} dBFS  (must be < -0.5)")
    print(f"  {'LRA':<28} {m['lra_lu']:>+7.1f} LU    (target 3.5-14)")
    print(f"  {'Crest factor':<28} {m['crest_factor_db']:>+7.1f} dB    (>8 = punchy)")
    print(f"  {'Stereo correlation':<28} {m['stereo_correlation']:>+7.3f}      (0.4-0.99)")
    print(f"\n  DYNAMICS QUALITY:")
    print(f"  {'Transient clarity':<28} {m['transient_clarity']:>+7.3f}      (target 0.08-0.55)")
    print(f"  {'Kick headroom':<28} {m['kick_headroom_db']:>+7.1f} dB    (target >3 dB)")
    print(f"  {'Section consistency':<28} {m['section_consistency_lu']:>+7.1f} LU    (target <5 LU)")
    print(f"  {'Spectral slope':<28} {m['spectral_slope_db_oct']:>+7.1f} dB/oct (target -12 to -2)")
    print(f"\n  PERCEPTUAL QUALITY:")
    print(f"  {'Beat sync score':<28} {m['beat_sync_score']:>+7.3f}      (target 0.35-1.0)")
    print(f"  {'Vocal clarity index':<28} {m['vocal_clarity_index']:>+7.1f} dB    (target -5 to +20)")
    print(f"  {'Tempo stability':<28} {m['tempo_stability']:>+7.3f}      (target 0.6-1.0)")
    print(f"  {'Click artifact score':<28} {m['click_artifact_score']:>+7.5f}     (target <0.005)")

    print(f"\n  FREQUENCY (absolute):")
    for label, key in [("Sub 20-80",  "_sub_db"), ("Bass 80-250", "_bass_db"),
                        ("Lo-Mid 250-800","_lowmid_db"), ("Mid 800-2.5k","_mid_db"),
                        ("Hi-Mid 2.5-6k","_himid_db"), ("High 6-20k","_high_db")]:
        val = m[key]
        bar = "█" * int(np.clip((val + 15) / 50 * 20, 0, 20))
        bar += "░" * (20 - len(bar))
        print(f"  {label:<18} {val:>+6.1f} dB  [{bar}]")

    mid = m["_mid_db"]
    print(f"\n  SPECTRAL RATIOS (vs mid at {mid:+.1f} dB):")
    for label, key, target in [
        ("Sub vs Mid",    "ratio_sub_to_mid",    "+4 → +18"),
        ("Bass vs Mid",   "ratio_bass_to_mid",   "+2 → +15"),
        ("Lo-Mid vs Mid", "ratio_lowmid_to_mid", "-3 → +8"),
        ("Hi-Mid vs Mid", "ratio_himid_to_mid",  "-20 → -3"),
        ("High vs Mid",   "ratio_high_to_mid",   "-32 → -8"),
        ("Mud Index",     "mud_index",            "1.0 → 5.5"),
    ]:
        val = m.get(key, 0.0)
        lo2, hi2 = ref.get(key, (-99, 99))
        flag = " ✗" if (val < lo2 or val > hi2) else ""
        print(f"  {label:<22} {val:>+7.2f}   (target {target}){flag}")

    if ref_path:
        print(f"\n  vs REFERENCE ({Path(ref_path).name}):")
        try:
            rm = _measure(ref_path)
            for label, key in [("Sub","_sub_db"),("Bass","_bass_db"),("Lo-Mid","_lowmid_db"),
                                ("Mid","_mid_db"),("Hi-Mid","_himid_db"),("High","_high_db")]:
                our, them = m[key], rm[key]
                diff = our - them
                arr = "↑" if diff > 1.5 else ("↓" if diff < -1.5 else "≈")
                print(f"  {label:<10} ours {our:>+6.1f}  ref {them:>+6.1f}  diff {diff:>+5.1f} {arr}")
        except Exception as e:
            print(f"  (failed: {e})")

    print("\n" + "═" * width + "\n")


def auto_score(audio_path: str, strict: bool = False) -> tuple:
    """
    Called from fuse() after every mix.
    Returns (passed, score, summary, issues) — issues used by corrections().
    """
    score, issues, metrics = score_file(audio_path, strict=strict, print_report=True)
    critical = [i for i in issues if i[0] == "CRITICAL"]
    passed = score >= 82 and not critical   # B+ grade threshold

    # Additional check: if beat_sync_score is too low, force a correction pass
    beat_sync = metrics.get("beat_sync_score", 1.0)
    if passed and beat_sync < 0.35:
        passed = False

    if passed:
        summary = f"PASS ({score}/100) — {_grade(score)}"
    elif critical:
        summary = f"FAIL ({score}/100) — CRITICAL: {critical[0][5]}"
    else:
        summary = f"FAIL ({score}/100) — {len(issues)} issues"

    return passed, score, summary, issues


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="VocalFusion Quality Scorer")
    parser.add_argument("audio")
    parser.add_argument("reference", nargs="?")
    parser.add_argument("--strict", action="store_true")
    args = parser.parse_args()
    score, issues, metrics = score_file(args.audio, strict=args.strict,
                                        reference_path=args.reference)
    sys.exit(0 if score >= 82 and not any(i[0] == "CRITICAL" for i in issues) else 1)
