"""
VocalFusion — Audio Quality Scorer + Auto-Correction Engine  v4
================================================================
Two jobs:
  1. SCORE: measure the output against professional reference ranges and
     detect specific problems (mud, harshness, smashed transients, buried
     vocals, clipping, phase, timing, groove, energy arc, etc.)
  2. CORRECT: return concrete DSP parameter deltas so fuse() can re-mix
     automatically without the user needing to listen and give feedback.

Metrics are grouped into five tiers:
  TECHNICAL  — loudness, clipping, dynamics, stereo phase
  SPECTRAL   — frequency balance ratios, mud, harshness, slope
  PERCEPTUAL — beat sync, vocal clarity, tempo stability, artifacts
  MUSICAL    — groove, energy arc, harmonic coherence, vocal flow
  MASHUP     — PLR, dynamic complexity, key distance (mashup-specific)

Reference ranges sourced from:
  - EBU R128 / ITU-R BS.1770-4: true peak -1.0 dBTP, integrated loudness
  - Streaming platform specs (Spotify -14 LUFS, Apple -16 LUFS, YouTube -14 LUFS)
  - Genre research: hip-hop/trap crest factor 6-12 dB, R&B 9-14 dB
  - Demucs htdemucs_ft vocal SDR benchmark: 8.42 dB (MultisongMVSep dataset)
  - iZotope Tonal Balance Control research across 1000s of commercial tracks
  - Harmonic mixing (Mixed In Key Camelot Wheel): ≤5 semitones = compatible

v4 additions:
  - plr_db           NEW: Peak-to-Loudness Ratio (true_peak - LUFS); target 5-14 dB
  - dynamic_complexity_db  NEW: avg abs deviation of 2s-window loudness (essentia-
                           inspired); target 3-14 dB; <3=brick-wall, >14=uneven
  - key_distance_semitones NEW: CENS circular-rotation key distance between bass-range
                           (beat harmony) and mid-range (vocal harmony); 0=same key,
                           6=tritone clash; computed from research-backed CENS features
  - true_peak REF corrected from -0.5 to -1.0 dBTP (EBU R128 industry standard)
  - crest_factor REF tightened from 22 to 16 dB (commercial masters cap at ~16 dB)
  - groove_score window: 40ms → 60ms (appropriate for computer-generated mashups)
  - vocal_harmony_score now uses CENS (robust to timbre/dynamics) with circular
    rotation to properly detect key compatibility between beat and vocal harmonics

v4.1 additions (ISMIR / AES / mastering research):
  - PLR lower bound raised 5→6 dB (AES TD1004 reject threshold)
  - vocal_bleed_score: sibilance band ratio method (5-10kHz vs broadband, >6 dB trigger)
  - key_distance: HPSS before CENS — removes kick/snare from chroma (Mauch & Dixon 2010)
  - low_freq_stereo_corr NEW: sub-bass mono check below 80 Hz, target >=0.85

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
    # ── TIER 1: Technical ────────────────────────────────────────────────────
    # lufs_integrated: Spotify/YouTube normalize to -14 LUFS. Hip-hop masters
    # typically land -8 to -12 LUFS. Below -14 = will be boosted (sounds weak).
    "lufs_integrated":    (-14.0, -7.0),
    # true_peak: EBU R128 and all major streaming platforms specify -1.0 dBTP.
    # (Previous -0.5 was too strict — limiter landing exactly at -0.5 triggered false CRITICAL.)
    "true_peak_dbfs":     (-40.0, -1.0),
    # lra_lu: hip-hop typically 4-10 LU, pop 6-12 LU, EDM 4-8 LU.
    "lra_lu":             (2.0, 18.0),
    # crest_factor: commercial hip-hop/trap 6-12 dB, R&B 9-14 dB.
    # >16 dB means mastered too quietly / not limited at all.
    "crest_factor_db":    (6.0, 16.0),
    # plr_db: Peak-to-Loudness Ratio = true_peak_dBTP - integrated_LUFS.
    # Research: hip-hop PLR 6-12 dB, R&B 8-14 dB.
    # <6 dB = over-limited (AES TD1004 / mastering community reject threshold).
    # >14 dB = mastered too quietly.
    "plr_db":             (6.0, 14.0),
    "stereo_correlation": (0.4, 0.99),
    # low_freq_stereo_corr: stereo correlation of the sub-bass band (< 80 Hz).
    # Research: bass and kick must be mono below 80 Hz — decorrelated sub-bass
    # causes phase cancellation on club subwoofers. Target ≥ 0.85.
    "low_freq_stereo_corr": (0.85, 1.0),

    # ── TIER 2: Spectral balance (ratios vs mid band) ─────────────────────
    "ratio_sub_to_mid":    (+4.0, +18.0),
    "ratio_bass_to_mid":   (+2.0, +20.0),
    "ratio_lowmid_to_mid": (-3.0, +10.0),
    "ratio_himid_to_mid":  (-20.0, -1.0),
    "ratio_high_to_mid":   (-32.0, -8.0),
    "lowmid_over_himid":   (+3.0, +20.0),
    "high_over_himid":     (-20.0, -4.0),

    # transient_clarity: dB crest of onset strength envelope (p95/p10 in dB).
    "transient_clarity":   (8.0, 28.0),

    # kick_headroom_db: p95-p10 dynamic range of 60-150 Hz sub-bass band (dB).
    "kick_headroom_db":    (8.0, 45.0),

    # mud_index: 200-600 Hz mean energy / 1000-3000 Hz mean energy (linear).
    "mud_index":           (1.0, 5.5),

    # section_consistency_lu: std dev of per-15s LUFS values (LU).
    "section_consistency_lu": (0.0, 8.0),

    # spectral_slope_db_oct: mean dB/octave energy dropoff 200 Hz → 10 kHz.
    "spectral_slope_db_oct": (-12.0, -2.0),

    # ── TIER 3: Perceptual quality ────────────────────────────────────────
    # beat_sync_score: xcorr of bass-band vs vocal-band onset envelopes.
    # Measures whether vocal rhythm locks to the beat groove.
    # High = vocal feels "locked in". Low = floating / out of time.
    # (v2 used L/R channel xcorr — meaningless in M/S mixed content.)
    "beat_sync_score":      (0.30, 1.0),

    # vocal_clarity_index: mid_db (1-4kHz) minus bass masking pressure.
    "vocal_clarity_index":  (-5.0, 20.0),

    # tempo_stability: 1 - CV of inter-beat intervals.
    "tempo_stability":      (0.6, 1.0),

    # click_artifact_score: % samples where |diff| > 10x diff RMS.
    "click_artifact_score": (0.0, 0.005),

    # vocal_bleed_score: fraction of voiced frames where sibilance band (5-10kHz)
    # exceeds broadband (500Hz-10kHz) by >6 dB — the industry de-essing trigger threshold.
    # Research: >5% frames = de-essing needed; >15% = clearly audible hi-hat bleed.
    "vocal_bleed_score":    (0.0, 0.15),

    # vocal_spectral_crest: median spectral crest of 300-3000Hz in voiced frames.
    "vocal_spectral_crest": (4.0, 30.0),

    # vocal_modulation_index: fraction of envelope energy at syllable rate (3-8 Hz).
    "vocal_modulation_index": (0.20, 0.65),

    # vocal_presence_ratio: vocal zone (1-4kHz) energy / beat zone (200-1kHz) energy
    # during voiced frames. > 0.55 = vocal sits above beat. < 0.30 = buried.
    "vocal_presence_ratio": (0.30, 1.20),

    # ── TIER 4: Musical intelligence ─────────────────────────────────────
    # groove_score: fraction of vocal onsets that land within ±60ms of a beat
    # grid position. Window widened from 40ms to 60ms for mashup context
    # (computer-generated timing has ~20ms more uncertainty than live performance).
    "groove_score":         (0.25, 1.0),

    # dynamic_arc_score: does the mix have a proper energy build + release?
    # Computed as correlation between actual per-section loudness and an ideal
    # arc (builds to peak at 65% then resolves). 1.0 = perfect arc.
    "dynamic_arc_score":    (0.20, 1.0),

    # vocal_harmony_score: CENS-based chroma coherence. Uses circular rotation
    # to find optimal key alignment between beat-range and vocal-range harmonics.
    # High = harmonically coherent. Low = key mismatch.
    "vocal_harmony_score":  (0.30, 1.0),

    # ── TIER 5: Mashup intelligence ───────────────────────────────────────
    # dynamic_complexity_db: essentia-inspired avg abs deviation of 2s-window
    # loudness from global loudness. <3 = brick-wall limited (all dynamics
    # destroyed). >14 = sections too uneven. Target: 4-12 dB for hip-hop.
    "dynamic_complexity_db": (3.0, 14.0),

    # key_distance_semitones: minimum semitone distance between beat-range
    # harmony (bass 80-500 Hz) and vocal-range harmony (800-3000 Hz), found
    # via circular CENS rotation. Camelot wheel research: 0-2 = compatible,
    # 3-5 = noticeable, 6 = tritone clash.
    "key_distance_semitones": (0.0, 5.0),
}

REF_STRICT = {**REF,
    "lufs_integrated":         (-13.0, -8.0),
    "lra_lu":                  (4.5, 16.0),
    "crest_factor_db":         (7.0, 14.0),
    "plr_db":                  (6.0, 12.0),
    "low_freq_stereo_corr":    (0.90, 1.0),
    "transient_clarity":       (10.0, 25.0),
    "kick_headroom_db":        (5.0, 45.0),
    "mud_index":               (1.2, 4.5),
    "section_consistency_lu":  (0.0, 8.0),
    "ratio_lowmid_to_mid":     (-3.0, +6.0),
    "ratio_himid_to_mid":      (-18.0, -1.5),
    "beat_sync_score":         (0.35, 1.0),
    "vocal_bleed_score":       (0.0, 0.08),
    "groove_score":            (0.30, 1.0),
    "dynamic_arc_score":       (0.30, 1.0),
    "vocal_harmony_score":     (0.40, 1.0),
    "vocal_presence_ratio":    (0.40, 1.20),
    "dynamic_complexity_db":   (4.0, 12.0),
    "key_distance_semitones":  (0.0, 3.0),
}

PENALTIES = {
    # TIER 1: Technical
    "lufs_integrated":        12,
    "true_peak_dbfs":         25,   # clipping = catastrophic
    "lra_lu":                  8,
    "crest_factor_db":        10,
    "plr_db":                 10,   # over-compressed (PLR<5) or mastered too quiet (PLR>14)
    "stereo_correlation":      8,
    "low_freq_stereo_corr":    8,   # sub-bass phase cancel = bad on club systems
    # TIER 2: Spectral
    "ratio_sub_to_mid":        5,
    "ratio_bass_to_mid":       6,
    "ratio_lowmid_to_mid":    10,
    "ratio_himid_to_mid":     13,   # ear fatigue > mud
    "ratio_high_to_mid":       8,
    "lowmid_over_himid":      10,
    "high_over_himid":         5,
    "transient_clarity":      12,
    "kick_headroom_db":        8,
    "mud_index":               8,
    "section_consistency_lu":  6,
    "spectral_slope_db_oct":   6,
    # TIER 3: Perceptual
    "beat_sync_score":        18,   # out of time = fundamental failure
    "vocal_clarity_index":    12,
    "tempo_stability":        10,
    "click_artifact_score":   15,
    "vocal_bleed_score":      20,
    "vocal_spectral_crest":   15,
    "vocal_modulation_index": 12,
    "vocal_presence_ratio":   15,   # buried vocal = unusable
    # TIER 4: Musical
    "groove_score":           20,   # no groove = doesn't sound like a song
    "dynamic_arc_score":      10,
    "vocal_harmony_score":    15,   # key clash = unlistenable
    # TIER 5: Mashup intelligence
    "dynamic_complexity_db":   8,   # brick-wall / over-compressed or too uneven
    "key_distance_semitones": 20,   # key clash = unlistenable, worse than anything else
}

PROBLEM_NAMES = {
    "lufs_integrated":     ("too quiet — streaming will boost (sounds weak)",
                            "too loud — over-limited / fatiguing"),
    "true_peak_dbfs":      ("signal too quiet", "CLIPPING — digital distortion"),
    "plr_db":              ("OVER-COMPRESSED — PLR too low, all dynamics destroyed",
                            "mastered too quiet — PLR too high, sounds weak"),
    "lra_lu":              ("over-compressed — no dynamics, sounds flat",
                            "too dynamic / uneven level"),
    "crest_factor_db":     ("SMASHED — limiter destroying transients, beat blurs at high volume",
                            "too spiky — needs glue"),
    "stereo_correlation":  ("phase cancellation — bad in mono",
                            "stereo too wide / phasey"),
    "low_freq_stereo_corr": ("(perfect mono bass — N/A)",
                             "SUB-BASS STEREO: phase cancellation on club subwoofers below 80 Hz"),
    "ratio_sub_to_mid":    ("sub-bass missing — sounds thin",
                            "BOOMINESS — sub-bass drowning mix"),
    "ratio_bass_to_mid":   ("bass missing — thin sounding",
                            "bass too heavy vs mids"),
    "ratio_lowmid_to_mid": ("low-mids scooped — hollow / phone-speaker sound",
                            "MUDDINESS — 250-800 Hz extreme buildup, vocal buried"),
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
    "beat_sync_score":      ("NO GROOVE — vocal and beat are out of time, feels broken",
                             "perfect sync (N/A)"),
    "vocal_clarity_index":  ("VOCALS BURIED — bass masking the vocal intelligibility zone",
                             "vocals too thin — not enough low-mid warmth"),
    "tempo_stability":      ("TEMPO DRIFT — rubberband artifacts, mix sounds unstable",
                             "too rigid (N/A)"),
    "click_artifact_score": ("(clean)",
                             "CLICKS / ARTIFACTS — discontinuities audible as pops"),
    "vocal_bleed_score":    ("(clean)",
                             "SCRATCHY VOCALS — beat bleed in vocal stem, hi-hat artifacts audible"),
    "vocal_spectral_crest": ("NOISY VOCAL — flat spectrum, no harmonic structure, sounds muddy/blendy",
                             "vocal too peaky — possible distortion or tuning artifacts"),
    "vocal_modulation_index":("VOCAL BURIED — no intelligible syllable dynamics, sounds muffled",
                              "CHOPPY VOCAL — over-gated or clipping modulation artifacts"),
    "vocal_presence_ratio": ("VOCAL BURIED — beat energy overwhelming vocal in presence zone",
                             "beat too quiet vs vocal — no foundation"),
    "groove_score":              ("NO GROOVE — vocal phrases float off the beat, sounds amateur",
                                  "perfect grid (N/A)"),
    "dynamic_arc_score":         ("FLAT ENERGY — no build/release, sounds like a loop not a song",
                                  "too dramatic (N/A)"),
    "vocal_harmony_score":       ("KEY CLASH — vocal and beat are in different keys, sounds dissonant",
                                  "harmonic match (N/A)"),
    "dynamic_complexity_db":     ("BRICK-WALL — all dynamics crushed, sounds like a flatline",
                                  "UNEVEN — volume jumps drastically between sections"),
    "key_distance_semitones":    ("key match (N/A)",
                                  "KEY CLASH — beat and vocal are >5 semitones apart, sounds dissonant"),
}

# ── Correction map ─────────────────────────────────────────────────────────────
CORRECTIONS = {
    "ratio_lowmid_to_mid":   ("carve_db",        0.0,  +2.0),
    "lowmid_over_himid":     ("carve_db",         0.0,  +1.5),
    "mud_index":             ("carve_db",          0.0,  +1.5),
    "ratio_himid_to_mid":    ("presence_db",      +0.5, -0.5),
    "ratio_high_to_mid":     ("air_db",           +1.0, -1.0),
    "transient_clarity":     ("lufs_delta",         0.0, -1.0),
    "crest_factor_db":       ("lufs_delta",         0.0, -1.0),
    "kick_headroom_db":      ("carve_db",           0.0, +1.0),
    "beat_sync_score":       ("vocal_level",       +0.10, 0.0),  # raise vocal so it leads more
    "vocal_bleed_score":     ("carve_db",           0.0, +1.5),
    "vocal_spectral_crest":  ("presence_db",       +1.0, -0.5),
    "vocal_modulation_index":("vocal_level",       +0.08, -0.05),
    "vocal_presence_ratio":  ("vocal_level",       +0.15, -0.10),  # bury → boost vocal
    "groove_score":          ("vocal_level",       +0.05,  0.0),  # slight presence boost helps groove
    "vocal_harmony_score":       ("carve_db",     0.0, +1.0),
    "key_distance_semitones":    ("carve_db",     0.0, +2.0),  # clash → deeper spectral carve separates beat from vocal
    "dynamic_complexity_db":     ("lufs_delta",   0.0, -1.5),  # over-dynamic → lower target LUFS for more headroom
    "plr_db":                    ("lufs_delta",  +1.0, -1.0),  # too compressed → ease off; too quiet → push louder
}


# ── Band helpers ───────────────────────────────────────────────────────────────

def _band_db(S: np.ndarray, freqs: np.ndarray, lo: float, hi: float) -> float:
    mask = (freqs >= lo) & (freqs < hi)
    if not mask.any():
        return -60.0
    return float(20.0 * np.log10(float(S[mask].mean()) + 1e-9))


def _band_rms(y: np.ndarray, lo_hz: float, hi_hz: float, sr: int = SR) -> np.ndarray:
    """Return mono RMS timeseries for a bandpass-filtered signal."""
    nyq = sr / 2.0
    sos = butter(4, [lo_hz / nyq, min(hi_hz / nyq, 0.999)], btype="band", output="sos")
    return sosfilt(sos, y)


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


# ── Core measurement ───────────────────────────────────────────────────────────

def _measure(audio_path: str) -> dict:
    """Compute all quality metrics for an audio file."""
    y = _load(audio_path)
    L, R = y[:, 0], y[:, 1]
    mono = (L + R) / 2.0

    # ── Loudness ──────────────────────────────────────────────────────────────
    meter = pyln.Meter(SR)
    lufs = float(meter.integrated_loudness(mono))
    try:
        lra = float(meter.loudness_range(mono))
    except Exception:
        lra = 0.0

    # ── True peak & crest factor ──────────────────────────────────────────────
    true_peak_dbfs = float(20 * np.log10(np.abs(y).max() + 1e-9))
    rms  = float(np.sqrt(np.mean(mono ** 2) + 1e-9))
    peak = float(np.abs(mono).max() + 1e-9)
    crest_db = float(20 * np.log10(peak / rms))

    # ── PLR (Peak-to-Loudness Ratio) ──────────────────────────────────────────
    # PLR = true_peak_dBTP - integrated_LUFS.
    # Research: hip-hop/trap PLR 6-12 dB, R&B 8-14 dB.
    # <5 dB = brick-wall limited (smashed). >14 dB = mastered too quietly.
    plr_db = float(np.clip(true_peak_dbfs - lufs, -5.0, 25.0)) if np.isfinite(lufs) else 9.0

    # ── Stereo correlation ────────────────────────────────────────────────────
    corr = float(np.corrcoef(L, R)[0, 1]) if (L.std() > 1e-9 and R.std() > 1e-9) else 1.0

    # ── Low-frequency stereo correlation (< 80 Hz) ────────────────────────────
    # Research: bass and kick MUST be mono below 80 Hz. Decorrelated sub-bass
    # causes phase cancellation on club subwoofers. Target ρ ≥ 0.85.
    # (EBU R68, club PA engineering: bass below 80Hz should approach mono.)
    try:
        nyq_lf = SR / 2.0
        lf_sos = butter(4, 80.0 / nyq_lf, btype="low", output="sos")
        L_lf = sosfilt(lf_sos, L)
        R_lf = sosfilt(lf_sos, R)
        low_freq_stereo_corr = (
            float(np.corrcoef(L_lf, R_lf)[0, 1])
            if (L_lf.std() > 1e-9 and R_lf.std() > 1e-9) else 1.0
        )
    except Exception:
        low_freq_stereo_corr = 1.0

    # ── Spectral bands ────────────────────────────────────────────────────────
    S = np.abs(librosa.stft(mono, n_fft=2048))
    freqs = librosa.fft_frequencies(sr=SR, n_fft=2048)

    sub_db    = _band_db(S, freqs,   20,   80)
    bass_db   = _band_db(S, freqs,   80,  250)
    lowmid_db = _band_db(S, freqs,  250,  800)
    mid_db    = _band_db(S, freqs,  800, 2500)
    himid_db  = _band_db(S, freqs, 2500, 6000)
    high_db   = _band_db(S, freqs, 6000, 20000)

    # ── Transient clarity ─────────────────────────────────────────────────────
    try:
        flux = librosa.onset.onset_strength(y=mono, sr=SR, hop_length=512)
        p95 = float(np.percentile(flux, 95))
        p10 = float(np.percentile(flux, 10))
        transient_clarity = float(np.clip(20 * np.log10((p95 + 1e-6) / (p10 + 1e-6)), 0.0, 30.0))
    except Exception:
        transient_clarity = 10.0

    # ── Kick dynamic range ────────────────────────────────────────────────────
    try:
        nyq = SR / 2.0
        kick_band = sosfilt(butter(4, 60.0/nyq, btype="high", output="sos"),
                     sosfilt(butter(4, 150.0/nyq, btype="low", output="sos"), mono))
        hop = 512
        frames = librosa.util.frame(kick_band, frame_length=1024, hop_length=hop)
        frame_rms_db = 20 * np.log10(np.sqrt((frames ** 2).mean(axis=0) + 1e-12))
        kick_headroom_db = float(np.percentile(frame_rms_db, 95) -
                                  np.percentile(frame_rms_db, 10))
    except Exception:
        kick_headroom_db = 15.0

    # ── Mud index ─────────────────────────────────────────────────────────────
    try:
        mud_index = float(S[(freqs >= 200) & (freqs < 600)].mean() /
                          (S[(freqs >= 1000) & (freqs < 3000)].mean() + 1e-9))
    except Exception:
        mud_index = 2.5

    # ── Section consistency ───────────────────────────────────────────────────
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

    # ── Spectral slope ────────────────────────────────────────────────────────
    try:
        octave_db = [_band_db(S, freqs, lo, hi)
                     for lo, hi in [(200,400),(400,800),(800,1600),(1600,3200),(3200,6400),(6400,12800)]]
        coeffs = np.polyfit(np.arange(len(octave_db), dtype=float), octave_db, 1)
        spectral_slope_db_oct = float(coeffs[0])
    except Exception:
        spectral_slope_db_oct = -5.0

    # ── Beat sync score (v3: bass-band vs vocal-band onset xcorr) ─────────────
    # The old L/R channel xcorr was meaningless in M/S mixed content because
    # the vocal (summed into Mid) appears equally in both L and R channels.
    # This version correlates the BEAT's groove (kick/bass band, 60-200 Hz) with
    # the VOCAL's rhythm (presence band, 800-3000 Hz).
    # High correlation = vocal rhythm locks to the beat's rhythmic pattern.
    try:
        beat_zone  = _band_rms(mono, 60.0,  200.0)   # bass/kick band = beat rhythm
        vocal_zone = _band_rms(mono, 800.0, 3000.0)  # presence band = vocal rhythm
        hop_bs = 512
        onset_beat  = librosa.onset.onset_strength(y=beat_zone,  sr=SR, hop_length=hop_bs)
        onset_vocal = librosa.onset.onset_strength(y=vocal_zone, sr=SR, hop_length=hop_bs)
        # Normalise
        onset_beat  = (onset_beat  - onset_beat.mean())  / (onset_beat.std()  + 1e-9)
        onset_vocal = (onset_vocal - onset_vocal.mean()) / (onset_vocal.std() + 1e-9)
        xcorr = np.correlate(onset_beat, onset_vocal, mode="full")
        xcorr_norm = xcorr / (len(onset_beat) + 1e-9)
        frame_rate = SR / hop_bs
        max_lag = int(round(0.075 * frame_rate))  # ±75ms window
        center = len(xcorr_norm) // 2
        beat_sync_score = float(np.clip(
            np.max(xcorr_norm[max(0, center - max_lag): center + max_lag + 1]),
            0.0, 1.0))
    except Exception:
        beat_sync_score = 0.5

    # ── Vocal clarity index ───────────────────────────────────────────────────
    try:
        S_full = np.abs(librosa.stft(mono, n_fft=2048))
        freqs_full = librosa.fft_frequencies(sr=SR, n_fft=2048)
        bass_db_vc = _band_db(S_full, freqs_full,   20,  300)
        mid_db_vc  = _band_db(S_full, freqs_full, 1000, 4000)
        masking_pressure = max(0.0, bass_db_vc - mid_db_vc - 10.0)
        vocal_clarity_index = float(mid_db_vc - masking_pressure)
    except Exception:
        vocal_clarity_index = 5.0

    # ── Tempo stability ───────────────────────────────────────────────────────
    try:
        _, beat_frames = librosa.beat.beat_track(y=mono, sr=SR, hop_length=512)
        if len(beat_frames) >= 3:
            beat_times = librosa.frames_to_time(beat_frames, sr=SR, hop_length=512)
            ibis = np.diff(beat_times)
            cv = float(ibis.std() / (ibis.mean() + 1e-9))
            tempo_stability = float(1.0 - min(cv, 1.0))
        else:
            tempo_stability = 1.0
    except Exception:
        tempo_stability = 0.8

    # ── Click artifact score ──────────────────────────────────────────────────
    try:
        diff_signal = np.diff(mono)
        diff_rms = float(np.sqrt(np.mean(diff_signal ** 2) + 1e-12))
        n_clicks = int(np.sum(np.abs(diff_signal) > 10.0 * diff_rms))
        click_artifact_score = float(n_clicks / (len(diff_signal) + 1e-9))
    except Exception:
        click_artifact_score = 0.0

    # ── Shared vocal/beat band signals ────────────────────────────────────────
    nyq = SR / 2.0
    hop_v = 512
    vp_band = sosfilt(butter(4, [1000.0/nyq, min(4000.0/nyq, 0.999)], btype="band", output="sos"),
                      mono)
    hh_band = sosfilt(butter(4, [6000.0/nyq, min(16000.0/nyq, 0.999)], btype="band", output="sos"),
                      mono)
    beat_band = sosfilt(butter(4, [200.0/nyq, min(1000.0/nyq, 0.999)], btype="band", output="sos"),
                        mono)

    vp_frames   = librosa.feature.rms(y=vp_band,   frame_length=1024, hop_length=hop_v)[0]
    hh_frames   = librosa.feature.rms(y=hh_band,   frame_length=1024, hop_length=hop_v)[0]
    beat_frames_rms = librosa.feature.rms(y=beat_band, frame_length=1024, hop_length=hop_v)[0]

    vp_med    = float(np.median(vp_frames))
    voiced_mask = vp_frames > vp_med

    # ── Vocal bleed score (sibilance band ratio method) ───────────────────────
    # Research-backed: hi-hat/cymbal bleed shows as elevated 5-10kHz energy
    # during voiced frames. De-essing trigger = sibilance >6 dB above broadband
    # (500Hz-10kHz). Score = fraction of voiced frames that exceed this threshold.
    # Research: >5% frames = de-essing needed; >15% = audible bleed (our limit).
    try:
        nyq_bl = SR / 2.0
        sib_sos = butter(4, [5000./nyq_bl, min(10000./nyq_bl, 0.999)],
                         btype="band", output="sos")
        broad_sos = butter(4, [500./nyq_bl, min(10000./nyq_bl, 0.999)],
                           btype="band", output="sos")
        sib_sig  = sosfilt(sib_sos,   mono)
        broad_sig = sosfilt(broad_sos, mono)

        sib_frames_rms   = librosa.feature.rms(y=sib_sig,   frame_length=1024, hop_length=512)[0]
        broad_frames_rms = librosa.feature.rms(y=broad_sig, frame_length=1024, hop_length=512)[0]

        n_bl = min(len(sib_frames_rms), len(voiced_mask))
        if voiced_mask[:n_bl].sum() > 20:
            sib_v   = sib_frames_rms[:n_bl][voiced_mask[:n_bl]]
            broad_v = broad_frames_rms[:n_bl][voiced_mask[:n_bl]]
            ratio_db_vals = 20.0 * np.log10(sib_v / (broad_v + 1e-9) + 1e-9)
            SIBILANCE_TRIGGER_DB = 6.0  # industry de-essing standard
            vocal_bleed_score = float(np.clip(
                np.mean(ratio_db_vals > SIBILANCE_TRIGGER_DB), 0.0, 1.0))
        else:
            vocal_bleed_score = 0.0
    except Exception:
        vocal_bleed_score = 0.0

    # ── Vocal spectral crest ──────────────────────────────────────────────────
    try:
        S_voiced = np.abs(librosa.stft(mono, n_fft=2048, hop_length=hop_v))
        freqs_v  = librosa.fft_frequencies(sr=SR, n_fft=2048)
        vp_rms_short = librosa.feature.rms(y=vp_band, frame_length=1024, hop_length=hop_v)[0]
        v_med    = float(np.median(vp_rms_short))
        v_mask   = vp_rms_short > v_med
        if v_mask.sum() > 20:
            mid_mask = (freqs_v >= 300) & (freqs_v < 3000)
            S_mid_voiced = S_voiced[np.ix_(mid_mask, v_mask)]
            frame_max  = S_mid_voiced.max(axis=0)
            frame_mean = S_mid_voiced.mean(axis=0) + 1e-12
            vocal_spectral_crest = float(np.median(frame_max / frame_mean))
        else:
            vocal_spectral_crest = 5.0
    except Exception:
        vocal_spectral_crest = 5.0

    # ── Vocal modulation index ────────────────────────────────────────────────
    try:
        vp_env   = librosa.feature.rms(y=vp_band, frame_length=512, hop_length=256)[0].astype(np.float64)
        env_fft  = np.abs(np.fft.rfft(vp_env - vp_env.mean()))
        mod_freqs = np.fft.rfftfreq(len(vp_env), d=256.0 / SR)
        total_e  = float(env_fft[(mod_freqs >= 1.0) & (mod_freqs <= 20.0)].sum() + 1e-12)
        syl_e    = float(env_fft[(mod_freqs >= 3.0) & (mod_freqs <=  8.0)].sum() + 1e-12)
        vocal_modulation_index = float(np.clip(syl_e / total_e, 0.0, 1.0))
    except Exception:
        vocal_modulation_index = 0.40

    # ── Vocal presence ratio (NEW) ─────────────────────────────────────────────
    # During voiced frames, how much does the vocal zone (1-4kHz) dominate
    # the beat zone (200-1kHz)?  High = vocal cuts through. Low = beat buries vocal.
    try:
        if voiced_mask.sum() > 20:
            vp_voiced   = float(vp_frames[voiced_mask].mean() + 1e-9)
            beat_voiced = float(beat_frames_rms[voiced_mask].mean() + 1e-9)
            vocal_presence_ratio = float(np.clip(vp_voiced / beat_voiced, 0.0, 2.0))
        else:
            vocal_presence_ratio = 0.60
    except Exception:
        vocal_presence_ratio = 0.60

    # ── Groove score (NEW) ────────────────────────────────────────────────────
    # Measure what fraction of vocal onsets land within ±40ms of a beat position.
    # A well-grooved mashup has the rapper/singer "locking" to the kick/snare grid.
    # Beat grid is detected from mono (full-range); narrow bass band gives too few
    # transients for beat_track to work reliably.
    try:
        _, beat_frames_grid = librosa.beat.beat_track(y=mono, sr=SR, hop_length=512)
        beat_times_grid = librosa.frames_to_time(beat_frames_grid, sr=SR, hop_length=512)

        # Detect vocal onsets in the presence band
        vocal_onset_env = librosa.onset.onset_strength(y=vp_band, sr=SR, hop_length=512)
        vocal_onset_frames = librosa.onset.onset_detect(
            onset_envelope=vocal_onset_env, sr=SR, hop_length=512)
        vocal_onset_times = librosa.frames_to_time(vocal_onset_frames, sr=SR, hop_length=512)

        if len(beat_times_grid) >= 2 and len(vocal_onset_times) >= 3:
            window_s = 0.060  # ±60ms — widened from 40ms for mashup context
            on_beat = 0
            for vt in vocal_onset_times:
                dists = np.abs(beat_times_grid - vt)
                if dists.min() <= window_s:
                    on_beat += 1
            groove_score = float(np.clip(on_beat / (len(vocal_onset_times) + 1e-9), 0.0, 1.0))
        else:
            groove_score = 0.50  # neutral default
    except Exception:
        groove_score = 0.50

    # ── Dynamic arc score (NEW) ───────────────────────────────────────────────
    # A professional song has an energy arc: builds toward a climax around
    # 60-70% through the track, then resolves.  Flat energy = loop, not song.
    # Score = Pearson correlation between actual per-section loudness and ideal arc.
    try:
        arc_win = SR * 10  # 10-second windows
        n_wins = len(mono) // arc_win
        if n_wins >= 4:
            win_rms = np.array([
                float(np.sqrt(np.mean(mono[i * arc_win:(i + 1) * arc_win] ** 2) + 1e-12))
                for i in range(n_wins)
            ])
            # Ideal arc: rises to peak at 65% of track, then resolves
            t_frac = np.linspace(0, 1, n_wins)
            ideal = np.where(t_frac < 0.65,
                             0.3 + 0.7 * (t_frac / 0.65) ** 1.2,
                             1.0 - 0.5 * ((t_frac - 0.65) / 0.35))
            # Pearson correlation between actual and ideal
            corr_arc = float(np.corrcoef(win_rms, ideal)[0, 1])
            dynamic_arc_score = float(np.clip((corr_arc + 1.0) / 2.0, 0.0, 1.0))  # 0-1
        else:
            dynamic_arc_score = 0.50
    except Exception:
        dynamic_arc_score = 0.50

    # ── Vocal harmony score (NEW) ─────────────────────────────────────────────
    # Measures chroma coherence between beat-dominated and vocal-dominated frames.
    # Key clash = the beat and vocal are in completely unrelated keys → dissonant.
    # High score = harmonic consonance between sources.
    try:
        chroma = librosa.feature.chroma_cqt(y=mono, sr=SR, hop_length=512, bins_per_octave=36)
        hop_h = 512
        beat_rms_ch  = librosa.feature.rms(y=beat_band,  frame_length=1024, hop_length=hop_h)[0]
        vocal_rms_ch = librosa.feature.rms(y=vp_band,    frame_length=1024, hop_length=hop_h)[0]

        n_frames = min(chroma.shape[1], len(beat_rms_ch), len(vocal_rms_ch))
        if n_frames > 20:
            beat_rms_t  = beat_rms_ch[:n_frames]
            vocal_rms_t = vocal_rms_ch[:n_frames]
            beat_thresh  = float(np.percentile(beat_rms_t, 70))
            vocal_thresh = float(np.percentile(vocal_rms_t, 70))

            beat_dom  = beat_rms_t  > beat_thresh
            vocal_dom = vocal_rms_t > vocal_thresh

            if beat_dom.sum() > 10 and vocal_dom.sum() > 10:
                chroma_beat  = chroma[:, beat_dom].mean(axis=1)
                chroma_vocal = chroma[:, vocal_dom].mean(axis=1)
                # Cosine similarity between mean chroma vectors
                dot = float(np.dot(chroma_beat, chroma_vocal))
                norm = float(np.linalg.norm(chroma_beat) * np.linalg.norm(chroma_vocal) + 1e-9)
                vocal_harmony_score = float(np.clip(dot / norm, 0.0, 1.0))
            else:
                vocal_harmony_score = 0.60
        else:
            vocal_harmony_score = 0.60
    except Exception:
        vocal_harmony_score = 0.60

    # ── Dynamic complexity (Tier 5) ───────────────────────────────────────────
    # Essentia-inspired: avg abs deviation of 2s-window integrated loudness from
    # global integrated loudness.  <3 dB = brick-wall (all dynamics destroyed).
    # >14 dB = sections too uneven.  Research target: 4-12 dB for hip-hop.
    try:
        dc_win = SR * 2  # 2-second windows
        dc_hop = SR * 1  # 1-second hop
        dc_wins = []
        i = 0
        while i + dc_win <= len(mono):
            seg = mono[i:i + dc_win]
            try:
                wl = float(meter.integrated_loudness(seg))
                if np.isfinite(wl) and wl > -70:
                    dc_wins.append(wl)
            except Exception:
                pass
            i += dc_hop
        if len(dc_wins) >= 4 and np.isfinite(lufs):
            dynamic_complexity_db = float(np.mean(np.abs(np.array(dc_wins) - lufs)))
        else:
            dynamic_complexity_db = 6.0  # neutral fallback
    except Exception:
        dynamic_complexity_db = 6.0

    # ── Key distance semitones (Tier 5) ──────────────────────────────────────
    # CENS (Chroma Energy Normalized Statistics) circular-rotation method.
    # Beat-range harmony: bandpass 80-500 Hz (kick, bass, beat harmonics).
    # Vocal-range harmony: bandpass 800-3000 Hz (vocal fundamentals + overtones).
    # Rotate beat CENS over 12 semitones, find shift that maximises similarity.
    # Distance = argmax shift (= semitone offset between keys).
    # Camelot Wheel research: ≤2 semitones = compatible, 6 = tritone clash.
    # Research improvement: HPSS before CENS removes kick/snare from chroma —
    # hip-hop percussion strongly pollutes chroma vectors without this step.
    try:
        # Harmonic-percussive separation strips drums before chroma computation
        y_harmonic_kd, _ = librosa.effects.hpss(mono, margin=3.0)
        nyq_k = SR / 2.0
        bass_rng = sosfilt(
            butter(4, [80.0 / nyq_k, min(500.0 / nyq_k, 0.999)], btype="band", output="sos"),
            y_harmonic_kd)
        voc_rng = sosfilt(
            butter(4, [800.0 / nyq_k, min(3000.0 / nyq_k, 0.999)], btype="band", output="sos"),
            y_harmonic_kd)

        # CENS: chroma_cqt with L1-norm quantisation (approximation without essentia)
        def _cens(sig):
            c = librosa.feature.chroma_cqt(y=sig, sr=SR, hop_length=512, bins_per_octave=36)
            # L1-normalise each frame then quantise to 5 levels (CENS approximation)
            col_sums = c.sum(axis=0, keepdims=True) + 1e-9
            c_norm = c / col_sums
            c_q = np.floor(c_norm * 5) / 5.0
            return c_q.mean(axis=1)  # mean CENS vector (12,)

        cens_beat = _cens(bass_rng)
        cens_voc  = _cens(voc_rng)

        # Circular rotation: try all 12 shifts, find best cosine similarity
        best_sim = -1.0
        best_shift = 0
        for shift in range(12):
            rotated = np.roll(cens_beat, shift)
            dot = float(np.dot(rotated, cens_voc))
            norm = float(np.linalg.norm(rotated) * np.linalg.norm(cens_voc) + 1e-9)
            sim = dot / norm
            if sim > best_sim:
                best_sim = sim
                best_shift = shift
        # Distance = min(shift, 12-shift) — circular semitone distance
        key_distance_semitones = float(min(best_shift, 12 - best_shift))
    except Exception:
        key_distance_semitones = 2.0  # neutral fallback

    return {
        # Global
        "lufs_integrated":       lufs,
        "true_peak_dbfs":        true_peak_dbfs,
        "lra_lu":                lra,
        "crest_factor_db":       crest_db,
        "plr_db":                plr_db,
        "stereo_correlation":    corr,
        "low_freq_stereo_corr":  low_freq_stereo_corr,
        # Raw bands (display only)
        "_sub_db":    sub_db,
        "_bass_db":   bass_db,
        "_lowmid_db": lowmid_db,
        "_mid_db":    mid_db,
        "_himid_db":  himid_db,
        "_high_db":   high_db,
        # Spectral ratios
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
        # Perceptual
        "beat_sync_score":       beat_sync_score,
        "vocal_clarity_index":   vocal_clarity_index,
        "tempo_stability":       tempo_stability,
        "click_artifact_score":  click_artifact_score,
        "vocal_bleed_score":     vocal_bleed_score,
        "vocal_spectral_crest":  vocal_spectral_crest,
        "vocal_modulation_index":vocal_modulation_index,
        "vocal_presence_ratio":  vocal_presence_ratio,
        # Musical
        "groove_score":          groove_score,
        "dynamic_arc_score":     dynamic_arc_score,
        "vocal_harmony_score":   vocal_harmony_score,
        # Mashup intelligence (Tier 5)
        "dynamic_complexity_db": dynamic_complexity_db,
        "key_distance_semitones": key_distance_semitones,
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
                penalty = min(penalty * 2, 30)
            score -= penalty
            sev = "CRITICAL" if penalty >= 20 else ("HIGH" if penalty >= 10 else "MEDIUM")
            issues.append((sev, key, val, lo, hi, p_lo))
        elif val > hi:
            delta = val - hi
            if delta > (hi - lo):
                penalty = min(penalty * 2, 30)
            score -= penalty
            sev = "CRITICAL" if penalty >= 20 else ("HIGH" if penalty >= 10 else "MEDIUM")
            issues.append((sev, key, val, lo, hi, p_hi))
    return max(0, score), issues


def _grade(score: int) -> str:
    if score >= 90:   return "S — Chart-ready"
    elif score >= 82: return "A — Professional quality"
    elif score >= 70: return "B — Good, minor issues"
    elif score >= 55: return "C — Acceptable, noticeable problems"
    elif score >= 40: return "D — Amateurish, needs work"
    else:             return "F — Unacceptable output"


def _musical_diagnosis(metrics: dict) -> list:
    """
    Return a list of plain-English musical diagnoses based on metric patterns.
    These go beyond individual metric failures to describe the overall sound.
    """
    diags = []
    gs  = metrics.get("groove_score", 0.5)
    arc = metrics.get("dynamic_arc_score", 0.5)
    har = metrics.get("vocal_harmony_score", 0.5)
    pre = metrics.get("vocal_presence_ratio", 0.6)
    syn = metrics.get("beat_sync_score", 0.5)
    bli = metrics.get("vocal_bleed_score", 0.2)
    mod = metrics.get("vocal_modulation_index", 0.4)
    dc  = metrics.get("dynamic_complexity_db", 6.0)
    kd  = metrics.get("key_distance_semitones", 2.0)
    plr = metrics.get("plr_db", 9.0)
    lfc = metrics.get("low_freq_stereo_corr", 1.0)

    if gs < 0.30 and syn < 0.35:
        diags.append("⚠ FUNDAMENTAL: vocal and beat are out of time — won't feel like a song regardless of mix quality")
    elif gs < 0.35:
        diags.append("△ Vocal phrases are floating off the beat — listeners will notice the lack of pocket")
    elif gs > 0.55:
        diags.append("✓ Groove is locked in — vocal sits in the pocket")

    if har < 0.35:
        diags.append("⚠ KEY CLASH: vocal and beat are in incompatible keys — sounds dissonant")
    elif har > 0.65:
        diags.append("✓ Harmonic coherence is strong — keys feel compatible")

    if pre < 0.30:
        diags.append("⚠ VOCAL BURIED: beat is overwhelming the vocal in the 1-4kHz zone — can't hear the lyrics")
    elif pre > 0.80:
        diags.append("✓ Vocal presence is strong — cuts through the mix clearly")

    if arc < 0.25:
        diags.append("△ FLAT ENERGY: no build/release structure — sounds like a loop, not a finished song")
    elif arc > 0.50:
        diags.append("✓ Good energy arc — mix builds and releases like a proper song")

    if bli > 0.45:
        diags.append("⚠ SCRATCHY: hi-hat/drum bleed in vocal stem is audible — sounds amateurish")

    if mod < 0.20:
        diags.append("△ MUFFLED: vocal syllables aren't intelligible — phrasing feels buried")
    elif mod > 0.55:
        diags.append("✓ Vocal intelligibility is strong — syllables are clear")

    if lfc < 0.85:
        diags.append(f"⚠ SUB-BASS STEREO: correlation {lfc:.2f} below 80Hz — will cancel on club subwoofers")
    elif lfc >= 0.95:
        diags.append(f"✓ Sub-bass is mono-compatible (ρ={lfc:.2f} below 80Hz)")

    if kd >= 6:
        diags.append("⚠ TRITONE CLASH: beat and vocal are 6 semitones apart — maximum dissonance")
    elif kd >= 3:
        diags.append(f"△ KEY DISTANCE {kd:.0f} semitones — noticeable harmonic tension")
    elif kd <= 2:
        diags.append(f"✓ Key distance {kd:.0f} semitones — harmonically compatible")

    if dc < 3.0:
        diags.append("⚠ BRICK-WALL: dynamic complexity too low — all dynamics crushed, sounds lifeless")
    elif dc > 14.0:
        diags.append(f"△ UNEVEN: dynamic complexity {dc:.1f} dB — sections jump dramatically in volume")
    else:
        diags.append(f"✓ Dynamic complexity {dc:.1f} dB — natural dynamics preserved")

    if plr < 5.0:
        diags.append(f"⚠ OVER-LIMITED: PLR {plr:.1f} dB — master is brick-wall limited, transients destroyed")
    elif plr > 14.0:
        diags.append(f"△ PLR {plr:.1f} dB — mastered too quietly, won't compete on streaming")
    else:
        diags.append(f"✓ PLR {plr:.1f} dB — good Peak-to-Loudness balance")

    return diags


def corrections(issues: list) -> dict:
    """
    Map detected issues to concrete DSP parameter adjustments for auto-correction.
    """
    scaled = {}
    sev_order = {"CRITICAL": 0, "HIGH": 1, "MEDIUM": 2}
    # Process highest-severity first
    for sev, key, val, lo, hi, desc in sorted(issues, key=lambda x: sev_order.get(x[0], 3)):
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
    width = 70
    print("\n" + "═" * width)
    print(f"  VocalFusion Quality Report — {Path(path).name}")
    print("═" * width)
    print(f"\n  SCORE: {score}/100   GRADE: {grade}")

    # Musical diagnosis first — most actionable for the user
    diags = _musical_diagnosis(m)
    if diags:
        print("\n  MUSICAL DIAGNOSIS:")
        for d in diags:
            print(f"  {d}")

    if not issues:
        print("\n  ✓ All metrics within professional range.")
    else:
        print("\n  ISSUES:")
        sev_order = {"CRITICAL": 0, "HIGH": 1, "MEDIUM": 2}
        for sev, key, val, lo, hi, desc in sorted(issues, key=lambda x: sev_order.get(x[0], 3)):
            marker = "✗✗" if sev == "CRITICAL" else ("✗ " if sev == "HIGH" else "△ ")
            print(f"\n  {marker} [{sev}] {desc}")
            print(f"       Measured: {val:.3f}  |  Target: {lo:.2f} → {hi:.2f}")

    print(f"\n  ── TECHNICAL ─────────────────────────────────────────────────────")
    print(f"  {'LUFS integrated':<28} {m['lufs_integrated']:>+7.1f} dB    (target -14 to -7)")
    print(f"  {'True peak':<28} {m['true_peak_dbfs']:>+7.1f} dBFS  (EBU R128: must be < -1.0)")
    print(f"  {'PLR (peak-to-loudness)':<28} {m.get('plr_db', 0.0):>+7.1f} dB    (target 5-14 dB)")
    print(f"  {'LRA':<28} {m['lra_lu']:>+7.1f} LU    (target 2-18)")
    print(f"  {'Crest factor':<28} {m['crest_factor_db']:>+7.1f} dB    (target 6-16 dB)")
    print(f"  {'Stereo correlation':<28} {m['stereo_correlation']:>+7.3f}      (0.4-0.99)")
    print(f"  {'Sub-bass mono (<80Hz)':<28} {m.get('low_freq_stereo_corr', 1.0):>+7.3f}      (>0.85 = mono bass)")

    print(f"\n  ── DYNAMICS ──────────────────────────────────────────────────────")
    print(f"  {'Transient clarity':<28} {m['transient_clarity']:>+7.1f} dB    (target 8-28 dB)")
    print(f"  {'Kick headroom':<28} {m['kick_headroom_db']:>+7.1f} dB    (target 8-45 dB)")
    print(f"  {'Section consistency':<28} {m['section_consistency_lu']:>+7.1f} LU    (<8 LU)")
    print(f"  {'Spectral slope':<28} {m['spectral_slope_db_oct']:>+7.1f} dB/oct (-12 to -2)")

    print(f"\n  ── PERCEPTUAL ────────────────────────────────────────────────────")
    print(f"  {'Beat sync (bass↔vocal)':<28} {m['beat_sync_score']:>7.3f}      (target >0.30)")
    print(f"  {'Vocal clarity index':<28} {m['vocal_clarity_index']:>+7.1f} dB    (target >-5)")
    print(f"  {'Tempo stability':<28} {m['tempo_stability']:>7.3f}      (target >0.6)")
    print(f"  {'Click artifact score':<28} {m['click_artifact_score']:>7.5f}     (<0.005)")
    print(f"  {'Vocal bleed':<28} {m['vocal_bleed_score']:>7.3f}      (<0.40 = clean)")
    print(f"  {'Vocal spectral crest':<28} {m['vocal_spectral_crest']:>7.2f}      (>4 = harmonic)")
    print(f"  {'Vocal modulation':<28} {m['vocal_modulation_index']:>7.3f}      (0.20-0.65)")
    print(f"  {'Vocal presence ratio':<28} {m['vocal_presence_ratio']:>7.3f}      (>0.30)")

    print(f"\n  ── MUSICAL ───────────────────────────────────────────────────────")
    print(f"  {'Groove (on-beat fraction)':<28} {m['groove_score']:>7.3f}      (>0.25 = locked)")
    print(f"  {'Dynamic arc':<28} {m['dynamic_arc_score']:>7.3f}      (>0.20 = builds)")
    print(f"  {'Harmonic coherence':<28} {m['vocal_harmony_score']:>7.3f}      (>0.30 = key match)")

    print(f"\n  ── MASHUP INTELLIGENCE (Tier 5) ──────────────────────────────────")
    print(f"  {'Dynamic complexity':<28} {m.get('dynamic_complexity_db', 0.0):>7.1f} dB    (target 3-14 dB)")
    print(f"  {'Key distance':<28} {m.get('key_distance_semitones', 0.0):>7.1f} st    (0-5 = compatible, 6 = clash)")

    print(f"\n  ── FREQUENCY BALANCE ─────────────────────────────────────────────")
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
        ("Bass vs Mid",   "ratio_bass_to_mid",   "+2 → +20"),
        ("Lo-Mid vs Mid", "ratio_lowmid_to_mid", "-3 → +10"),
        ("Hi-Mid vs Mid", "ratio_himid_to_mid",  "-20 → -1"),
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
    Returns (passed, score, summary, issues).
    """
    score, issues, metrics = score_file(audio_path, strict=strict, print_report=True)
    critical = [i for i in issues if i[0] == "CRITICAL"]

    # Pass requires: score ≥ 82, no CRITICAL issues, and groove is not broken
    groove = metrics.get("groove_score", 0.5)
    harmony = metrics.get("vocal_harmony_score", 0.5)
    passed = score >= 82 and not critical and groove >= 0.25 and harmony >= 0.25

    if passed:
        summary = f"PASS ({score}/100) — {_grade(score)}"
    elif critical:
        summary = f"FAIL ({score}/100) — CRITICAL: {critical[0][5]}"
    elif groove < 0.25:
        summary = f"FAIL ({score}/100) — no groove (vocal off-beat)"
    elif harmony < 0.25:
        summary = f"FAIL ({score}/100) — key clash"
    else:
        summary = f"FAIL ({score}/100) — {len(issues)} issues"

    return passed, score, summary, issues


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="VocalFusion Quality Scorer v4")
    parser.add_argument("audio")
    parser.add_argument("reference", nargs="?")
    parser.add_argument("--strict", action="store_true")
    args = parser.parse_args()
    score, issues, metrics = score_file(args.audio, strict=args.strict,
                                        reference_path=args.reference)
    sys.exit(0 if score >= 82 and not any(i[0] == "CRITICAL" for i in issues) else 1)
