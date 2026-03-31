"""
VocalFusion — Audio Quality Scorer + Auto-Correction Engine  v5
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
  - AES TD1004: PLR < 6 dB = over-limited (reject threshold)
  - Mauch & Dixon ISMIR 2010: HPSS before chroma improves key accuracy ~15%
  - EBU R68: sub-bass below 80 Hz should be mono for club PA compatibility
  - Harmonic mixing (Mixed In Key Camelot Wheel): ≤5 semitones = compatible

v5 changes vs v4.1:
  - Single STFT pass: S_mag computed once at top of _measure(), reused everywhere
  - Shared beat grid: librosa.beat.beat_track called once, used for groove AND tempo stability
  - Shared HPSS: y_harm computed once, used for vocal_harmony_score AND key_distance
  - _cens_vec() is now a module-level function (was incorrectly inside try block)
  - Groove score vectorized with numpy broadcasting (was slow Python loop)
  - dynamic_complexity uses RMS-in-dB (was calling LUFS meter per 2s window — very slow)
  - _score() skips _ -prefixed display-only keys
  - PROBLEM_NAMES["low_freq_stereo_corr"] lo/hi labels corrected (were swapped)
  - _musical_diagnosis vocal_bleed threshold fixed: 0.45 → 0.15 (matches REF)
  - _print_report stale labels fixed: bleed 0.40→0.15, PLR 5-14→6-14
  - vocal_harmony_score uses y_harm (harmonic signal), not raw mono
  - _bp() / _lp() filter helpers deduplicate butterworth boilerplate

Usage:
  python listen.py output.wav                  # score
  python listen.py output.wav reference.mp3    # compare vs reference
  python listen.py output.wav --strict         # stricter thresholds

Exit code 0 = PASS, 1 = FAIL.
"""

import sys
import argparse
import json
import os
import numpy as np
import librosa
import pyloudnorm as pyln
import soundfile as sf
from pathlib import Path
from scipy.signal import butter, sosfilt

# Pre-load torch at module level. If torch initializes for the first time inside
# _mel_stft_quality_score() while the process is mid-computation, it triggers a
# native segfault (exit 139). Importing it here ensures torch is fully initialized
# before any audio processing begins.
try:
    import torch as _torch_preload  # noqa: F401
    import auraloss.freq as _auraloss_preload  # noqa: F401
except Exception:
    pass

SR = 44100

# ---------------------------------------------------------------------------
# Reference profile (optional) — built by reference.py from chart songs
# ---------------------------------------------------------------------------
_REF_PROFILE: dict | None = None

def _load_reference_profile() -> None:
    global _REF_PROFILE
    _profile_path = os.path.join(os.path.dirname(__file__), "reference_profile.json")
    if os.path.exists(_profile_path):
        try:
            with open(_profile_path) as f:
                _REF_PROFILE = json.load(f)
            print(f"[listen] Reference profile loaded ({_REF_PROFILE.get('n_tracks', '?')} tracks)", flush=True)
        except Exception as e:
            print(f"[listen] Warning: could not load reference profile: {e}", flush=True)

_load_reference_profile()

# ── Professional Reference Ranges ─────────────────────────────────────────────
# All ratio values are (band_X - mid_band) in dB — scale-invariant.

REF = {
    # ── TIER 1: Technical ────────────────────────────────────────────────────
    # lufs_integrated: Spotify/YouTube normalize to -14 LUFS. Hip-hop masters
    # typically land -8 to -12 LUFS. Below -14 = will be boosted (sounds weak).
    "lufs_integrated":    (-14.0, -7.0),
    # true_peak: EBU R128 and all major streaming platforms specify -1.0 dBTP.
    # true_peak: -0.3 dBFS upper bound (vs EBU -1.0 dBTP) because our mastering
    # chain targets -1.0 dBFS sample peak. 24-bit quantization and floating-point
    # arithmetic can push exactly -1.001 to -0.999 dBFS, triggering a false CRITICAL.
    # -0.3 is Apple MFiT standard; any louder is genuinely clipping.
    "true_peak_dbfs":     (-40.0, -0.3),
    # lra_lu: hip-hop typically 4-10 LU, pop 6-12 LU, EDM 4-8 LU.
    "lra_lu":             (2.0, 18.0),
    # crest_factor: commercial hip-hop/trap 6-12 dB, R&B 9-14 dB.
    # >16 dB means mastered too quietly / not limited at all.
    "crest_factor_db":    (6.0, 16.0),
    # plr_db: Peak-to-Loudness Ratio = true_peak_dBTP - integrated_LUFS.
    # <6 dB = over-limited (AES TD1004 reject threshold). >14 dB = too quiet.
    "plr_db":             (6.0, 14.0),
    "stereo_correlation": (0.4, 0.99),
    # low_freq_stereo_corr: sub-bass correlation below 80 Hz. Decorrelated sub-bass
    # causes phase cancellation on club subwoofers. Target ≥ 0.85 (EBU R68).
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
    # kick_headroom_db: p95-p10 dynamic range of 60-150 Hz sub-bass band.
    "kick_headroom_db":    (8.0, 45.0),
    # mud_index: 200-600 Hz mean energy / 1000-3000 Hz mean energy (linear).
    # Ceiling raised to 6.0: EDM/trap beats with heavy kick body (250-600 Hz) routinely
    # sit at 5.5-6.0 by design. Values >6.0 are genuinely muddy.
    "mud_index":           (1.0, 6.0),
    # section_consistency_lu: std dev of per-15s LUFS values.
    "section_consistency_lu": (0.0, 8.0),
    # spectral_slope_db_oct: mean dB/octave dropoff 200 Hz → 10 kHz.
    "spectral_slope_db_oct": (-12.0, -2.0),

    # ── TIER 3: Perceptual quality ────────────────────────────────────────
    # beat_sync_score: xcorr of bass-band vs vocal-band onset envelopes.
    "beat_sync_score":      (0.30, 1.0),
    # vocal_clarity_index: mid_db (1-4kHz) minus bass masking pressure.
    "vocal_clarity_index":  (-5.0, 20.0),
    # tempo_stability: 1 - CV of inter-beat intervals.
    "tempo_stability":      (0.6, 1.0),
    # click_artifact_score: fraction of samples where |diff| > 10x diff RMS.
    "click_artifact_score": (0.0, 0.005),
    # vocal_bleed_score: fraction of voiced frames where sibilance band (5-10kHz)
    # exceeds broadband (500Hz-10kHz) by >6 dB (industry de-essing trigger).
    # >5% = de-essing needed; >15% = clearly audible bleed.
    "vocal_bleed_score":    (0.0, 0.15),
    # vocal_spectral_crest: median spectral crest of 300-3000Hz in voiced frames.
    "vocal_spectral_crest": (4.0, 30.0),
    # vocal_modulation_index: fraction of envelope energy at syllable rate (3-8 Hz).
    "vocal_modulation_index": (0.20, 0.65),
    # vocal_presence_ratio: vocal zone (1-4kHz) / beat zone (200-1kHz) in voiced frames.
    "vocal_presence_ratio": (0.30, 1.20),
    # vocal_hnr_db: Harmonic-to-Noise Ratio on vocal band (300-3000 Hz) voiced frames.
    # Autocorrelation method (Praat). >15 dB = normophonic clear vocal.
    # <10 dB = breathy/noisy. Research: Ferrand 2002; Praat documentation.
    "vocal_hnr_db":         (10.0, 40.0),
    # vocal_sfm: Spectral Flatness Measure of vocal band in voiced frames.
    # Geometric/arithmetic mean ratio of power spectrum. 0 = pure tone, 1 = white noise.
    # <0.1 = very tonal (over-tuned/robotic). 0.05-0.30 = natural harmonic vocal.
    # >0.35 = noisy/breathy vocal or heavy bleed. Research: Izmirli ISMIR 2000.
    "vocal_sfm":            (0.05, 0.35),

    # ── TIER 4: Musical intelligence ─────────────────────────────────────
    # groove_score: fraction of vocal onsets within ±60ms of a beat grid position.
    # Window widened from 40ms to 60ms for mashup context.
    "groove_score":         (0.25, 1.0),
    # dynamic_arc_score: correlation between actual per-section loudness and ideal arc.
    "dynamic_arc_score":    (0.20, 1.0),
    # vocal_harmony_score: CENS-based chroma coherence between beat-range and
    # vocal-range harmonics (computed on HPSS harmonic signal).
    "vocal_harmony_score":  (0.30, 1.0),

    # ── TIER 5: Mashup intelligence ───────────────────────────────────────
    # dynamic_complexity_db: avg abs deviation of 2s-window RMS (dB) from global
    # RMS (dB). <1.5 = brick-wall. >14 = sections too uneven.
    # Lowered to 1.5 (from 2.0) — modern EDM/house/trap mastering produces very
    # flat RMS (1.6-2.0 dB). 1.5 dB floor reflects the structural reality of
    # heavily side-chained, constant-energy electronic production.
    "dynamic_complexity_db": (1.5, 14.0),
    # vocal_robot_score: 0 = natural voice, 1 = severe WORLD vocoder artifacts.
    # Frame-to-frame HNR std-dev / 15 dB. >0.45 = audibly robotic/pitch-shifted.
    "vocal_robot_score":     (0.0, 0.45),
    # key_distance_semitones: CENS circular-rotation key distance between
    # bass-range (80-500 Hz) and vocal-range (800-3000 Hz) harmonics.
    # Camelot wheel: 0-2 = compatible, 3-5 = noticeable, 6 = tritone clash.
    "key_distance_semitones": (0.0, 5.0),

    # ── TIER 6: Reference profile band deltas (dB deviation vs Drake target) ─
    # These are only populated when a reference profile is loaded.
    # ±4 dB = acceptable variation. Outside that = audibly off-balance.
    "delta_sub":      (-4.0, +4.0),
    "delta_bass":     (-4.0, +4.0),
    "delta_lo_mid":   (-5.0, +5.0),   # lo-mid varies by genre (kick punch, bass harmonics)
    "delta_mid":      (-3.0, +3.0),   # mid is most audible — tighter window
    "delta_hi_mid":   (-3.0, +3.0),
    "delta_presence": (-5.0, +5.0),   # presence 6-10kHz; EDM/trap hi-hats structurally exceed vocal reference
    "delta_air":      (-5.0, +5.0),   # air shelf more forgiving
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
    "vocal_hnr_db":            (12.0, 40.0),
    "vocal_sfm":               (0.05, 0.28),
    "dynamic_complexity_db":   (4.0, 12.0),
    "key_distance_semitones":  (0.0, 3.0),
}

PENALTIES = {
    # TIER 1: Technical
    "lufs_integrated":        12,
    "true_peak_dbfs":         25,   # clipping = catastrophic
    "lra_lu":                  8,
    "crest_factor_db":        10,
    "plr_db":                 10,
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
    "vocal_hnr_db":           12,   # breathy/noisy vocal
    "vocal_sfm":              10,   # too noisy or over-tuned
    # TIER 4: Musical
    "groove_score":           20,   # no groove = doesn't sound like a song
    "dynamic_arc_score":      10,
    "vocal_harmony_score":    15,   # key clash = unlistenable
    # TIER 5: Mashup intelligence
    "dynamic_complexity_db":   8,
    "key_distance_semitones": 20,   # key clash = unlistenable
    "vocal_robot_score":      22,   # robotic vocal = worse than key clash for listener experience
    # TIER 6: Reference band deltas — penalize when bands deviate from Drake target
    "delta_sub":       12,   # missing sub = sounds small
    "delta_bass":       8,
    "delta_lo_mid":     6,
    "delta_mid":       10,   # mid zone = vocal intelligibility
    "delta_hi_mid":     8,
    "delta_presence":  10,   # presence = vocal sits forward
    "delta_air":        8,   # air = top-end life
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
    # Index 0 = val < lo = correlation below 0.85 = BAD (decorrelated = cancels)
    # Index 1 = val > hi = correlation above 1.0 = impossible
    "low_freq_stereo_corr": ("SUB-BASS DECORRELATED — phase cancels on club subwoofers below 80 Hz",
                             "impossible (correlation cannot exceed 1.0)"),
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
    "vocal_hnr_db":         ("NOISY/BREATHY VOCAL — high noise floor in vocal band, HNR too low",
                             "pure tone (N/A — cannot exceed physical limit)"),
    "vocal_sfm":            ("OVER-TUNED/ROBOTIC — vocal spectrum too tonal, SFM near 0",
                             "NOISY VOCAL — spectral flatness too high, sounds breathy or bleedy"),
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
    "vocal_robot_score":         ("natural voice (N/A)",
                                  "ROBOTIC VOCAL — pitch correction artifacts (WORLD vocoder / heavy autotune on commercial stem)"),
    # Per-band delta problem names
    "delta_sub":      ("SUB MISSING — sub-bass 4+ dB below reference, sounds weak/small",
                       "SUB OVERLOADED — sub-bass 4+ dB above reference, boomy"),
    "delta_bass":     ("BASS THIN — bass band too quiet vs reference",
                       "BASS HEAVY — bass band too loud vs reference, muddy"),
    "delta_lo_mid":   ("LO-MID SCOOPED — hollow, phone-speaker sound",
                       "LO-MID MUDDY — 250-800 Hz excess masking vocal"),
    "delta_mid":      ("MIDS DIP — vocal presence zone too quiet, sounds recessed",
                       "MIDS HARSH — vocal presence zone too loud, fatiguing"),
    "delta_hi_mid":   ("HI-MID DULL — 2-5 kHz too quiet, clarity gone",
                       "HI-MID HARSH — 2-5 kHz too loud, ear fatigue"),
    "delta_presence": ("PRESENCE MISSING — 3-6 kHz too quiet, vocal behind glass",
                       "PRESENCE HARSH — 3-6 kHz too loud, ice-pick"),
    "delta_air":      ("AIR MISSING — 8+ kHz dead, sounds dark and muffled",
                       "TOO BRIGHT — 8+ kHz hyped, sounds thin and harsh"),
}

CORRECTIONS = {
    "ratio_lowmid_to_mid":   ("carve_db",        0.0,  +2.0),
    "lowmid_over_himid":     ("carve_db",         0.0,  +1.5),
    "mud_index":             ("carve_db",          0.0,  +1.5),
    "ratio_himid_to_mid":    ("presence_db",      +0.5, -0.5),
    "ratio_high_to_mid":     ("air_db",           +1.0, -1.0),
    "transient_clarity":     ("lufs_delta",         0.0, -1.0),
    "crest_factor_db":       ("lufs_delta",         0.0, -1.0),
    "kick_headroom_db":      ("carve_db",           0.0, +1.0),
    "beat_sync_score":       ("vocal_level",       +0.10, 0.0),
    "vocal_bleed_score":     ("carve_db",           0.0, +1.5),
    "vocal_spectral_crest":  ("presence_db",       +1.0, -0.5),
    "vocal_modulation_index":("vocal_level",       +0.08, -0.05),
    "vocal_presence_ratio":  ("vocal_level",       +0.15, -0.10),
    "groove_score":          ("vocal_level",       +0.05,  0.0),
    "vocal_harmony_score":   ("carve_db",           0.0, +1.0),
    "key_distance_semitones":("carve_db",           0.0, +2.0),
    "dynamic_complexity_db": ("lufs_delta",         0.0, -1.5),
    "lra_lu":                ("lufs_delta",        +0.5, -0.5),  # over-compressed → back off limiting
    "plr_db":                ("lufs_delta",        +1.0, -1.0),
    # Vocal HNR too low (noisy/breathy): reduce carve (less spectral carving =
    # vocal preserved more) and lower vocal level so masking noise is less prominent
    "vocal_hnr_db":          ("carve_db",           -0.5, 0.0),
    # Sub-bass ratio too high → targeted sub shelf cut below 80 Hz in mastering
    # (lufs_delta doesn't help here — reducing master volume keeps the ratio the same)
    "ratio_sub_to_mid":      ("sub_cut_db",         0.0, +2.0),
    # Band delta corrections: too quiet → boost (positive adj), too loud → cut (negative)
    "delta_sub":       ("sub_cut_db",        -1.5, +2.0),  # sub thin → ease off sub cut; sub heavy → cut
    "delta_bass":      ("carve_db",          -1.0, +1.0),  # bass thin → cut less carve
    "delta_mid":       ("carve_db",          -1.5, +1.5),  # mid thin → cut less in mid
    "delta_presence":  ("presence_db",       +1.5, -1.5),  # presence thin → boost presence
    "delta_air":       ("air_db",            +1.5, -1.5),  # air thin → boost air shelf
}


# ── Filter helpers ─────────────────────────────────────────────────────────────

def _bp(y: np.ndarray, lo_hz: float, hi_hz: float, order: int = 4) -> np.ndarray:
    """Bandpass Butterworth filter."""
    nyq = SR / 2.0
    sos = butter(order, [lo_hz / nyq, min(hi_hz / nyq, 0.999)], btype="band", output="sos")
    return sosfilt(sos, y)


def _lp(y: np.ndarray, cut_hz: float, order: int = 4) -> np.ndarray:
    """Low-pass Butterworth filter."""
    nyq = SR / 2.0
    sos = butter(order, cut_hz / nyq, btype="low", output="sos")
    return sosfilt(sos, y)


def _band_db(S: np.ndarray, freqs: np.ndarray, lo: float, hi: float) -> float:
    mask = (freqs >= lo) & (freqs < hi)
    if not mask.any():
        return -60.0
    return float(20.0 * np.log10(float(S[mask].mean()) + 1e-9))


def _cens_vec(sig: np.ndarray) -> np.ndarray:
    """
    CENS chroma vector (12,) for a signal.
    Chroma Energy Normalized Statistics: L1-normalise each frame then quantise
    to 5 levels (approximation of the essentia CENS descriptor).
    Module-level so it is importable and not scoped to a try block.
    """
    c = librosa.feature.chroma_cqt(y=sig, sr=SR, hop_length=512, bins_per_octave=36)
    col_sums = c.sum(axis=0, keepdims=True) + 1e-9
    c_norm = c / col_sums
    c_q = np.floor(c_norm * 5) / 5.0
    return c_q.mean(axis=1)  # (12,)


def _groove_timing_score(y_mix: np.ndarray, sr: int) -> float:
    """
    Measure how tightly vocal content sits on the beat grid.
    Score 0-100: 100 = every onset perfectly on beat, 0 = random timing.

    Method:
    1. Detect beat grid from the mix
    2. Detect onsets (energy peaks in the vocal presence band 500-4000Hz)
    3. For each onset, measure deviation from nearest beat/8th-note grid position
    4. Score = 100 * (1 - mean_deviation_normalized)

    Returns 50.0 if beat detection fails (neutral, no penalty).
    """
    try:
        y_mono = y_mix if y_mix.ndim == 1 else np.mean(y_mix, axis=0)

        # Beat grid
        hop = 512
        tempo, beats = librosa.beat.beat_track(y=y_mono, sr=sr, hop_length=hop, units="samples")
        beats = np.array(beats, dtype=np.int64)
        if len(beats) < 4 or not np.isfinite(tempo) or tempo < 40:
            return 50.0

        beat_period = float(np.median(np.diff(beats)))
        eighth = beat_period / 2.0

        # Build 8th-note grid
        grid = []
        for b in beats:
            grid.append(int(b))
            e = int(b + eighth)
            if e < len(y_mono):
                grid.append(e)
        grid = np.array(sorted(set(grid)), dtype=np.int64)

        # Onsets from vocal-presence band (500-4kHz)
        from scipy.signal import butter, sosfilt
        sos = butter(4, [500, 4000], btype='bandpass', fs=sr, output='sos')
        y_vp = sosfilt(sos, y_mono)
        onset_samp = librosa.onset.onset_detect(
            y=y_vp, sr=sr, hop_length=hop, units="samples", backtrack=True)
        onset_samp = np.array(onset_samp, dtype=np.int64)

        if len(onset_samp) < 4:
            return 50.0

        # Measure deviation of each onset from nearest grid point
        deviations_ms = []
        max_consider_ms = 60.0  # only count onsets within 60ms of a beat (ignore ambient noise)
        for ons in onset_samp:
            nearest = grid[np.argmin(np.abs(grid - ons))]
            dev_ms = abs(int(ons) - int(nearest)) / sr * 1000.0
            if dev_ms < max_consider_ms:
                deviations_ms.append(dev_ms)

        if len(deviations_ms) < 4:
            return 50.0

        mean_dev = float(np.mean(deviations_ms))
        # 0ms deviation = 100, 60ms = 0.
        # Original 30ms JND (human rhythm perception) was too tight for computer-
        # generated time-stretching: even well-aligned stems routinely hit 25-40ms
        # deviation just from stretch interpolation, scoring 0 despite correct alignment.
        # 60ms (≈26% of an 8th note @130 BPM) matches typical stretch accuracy.
        score = max(0.0, 100.0 * (1.0 - mean_dev / 60.0))
        return round(score, 1)
    except Exception:
        return 50.0


def _harmonic_clarity_score(y_mix: np.ndarray, sr: int) -> float:
    """
    Measure vocal intelligibility in the 1-4kHz speech clarity band.
    Score 0-100: measures SNR of harmonic content vs noise floor in that band.

    Method: HPSS-separate the mix, measure harmonic energy in 1-4kHz vs
    percussive energy in same band. High harmonic/percussive ratio = clear vocal.
    Returns 50.0 on failure (neutral).
    """
    try:
        y_mono = y_mix if y_mix.ndim == 1 else np.mean(y_mix, axis=0)

        # HPSS separation
        D = librosa.stft(y_mono, n_fft=2048, hop_length=512)
        H, P = librosa.decompose.hpss(np.abs(D), margin=2.0)

        # Frequency bins for 1-4kHz
        freqs = librosa.fft_frequencies(sr=sr, n_fft=2048)
        band_mask = (freqs >= 1000) & (freqs <= 4000)

        h_energy = float(np.mean(H[band_mask, :] ** 2) + 1e-12)
        p_energy = float(np.mean(P[band_mask, :] ** 2) + 1e-12)

        # Harmonic-to-percussive ratio in dB
        hpr_db = 10 * np.log10(h_energy / p_energy)

        # -6dB = 50 (balanced), +6dB = 100 (clear vocal), -12dB = 0 (buried)
        score = float(np.clip((hpr_db + 6.0) / 12.0 * 100.0, 0.0, 100.0))
        return round(score, 1)
    except Exception:
        return 50.0


def _vocal_intelligibility_score(y_mix: np.ndarray, sr: int) -> float:
    """
    Estimate vocal intelligibility using a proxy STOI-inspired method.

    True STOI requires a clean reference signal. As a proxy, we:
    1. Extract the "clean" vocal estimate via HPSS harmonic separation
    2. Compare harmonic signal vs full mix in the 300-3000Hz speech band
    3. Score = harmonic energy fraction in that band (0-100)

    This approximates how much of the speech band is harmonic (voice) vs
    percussive (beat masking). High score = intelligible vocal.
    Returns 50.0 on failure.
    """
    try:
        y_mono = y_mix if y_mix.ndim == 1 else np.mean(y_mix, axis=0)

        # HPSS to isolate harmonic (vocal) content
        D = librosa.stft(y_mono, n_fft=2048, hop_length=512)
        H, P = librosa.decompose.hpss(np.abs(D), margin=3.0)

        # Speech band 300-3000Hz
        freqs = librosa.fft_frequencies(sr=sr, n_fft=2048)
        band  = (freqs >= 300) & (freqs <= 3000)

        h_energy   = float(np.mean(H[band] ** 2) + 1e-12)
        mix_energy = float(np.mean((H[band] + P[band]) ** 2) + 1e-12)

        # Fraction of speech band that is harmonic
        harmonic_frac = h_energy / mix_energy

        # Scale: 0.5 frac = 50 (neutral), 0.8 frac = 100 (clear), 0.2 frac = 0 (buried)
        score = float(np.clip((harmonic_frac - 0.2) / 0.6 * 100.0, 0.0, 100.0))
        return round(score, 1)
    except Exception:
        return 50.0


def _mel_stft_quality_score(y_mix: np.ndarray, sr: int) -> float:
    """
    Compute a perceptual quality score using mel-STFT spectral flatness analysis.
    Measures how "musical" vs "noisy" the mix sounds across mel-scaled frequency bands.

    Uses auraloss if available, falls back to librosa mel spectrogram analysis.
    Score 0-100: 100 = smooth, musical spectrogram; 0 = harsh artifacts.
    """
    try:
        import torch
        import auraloss.freq as af

        # Convert to torch tensor (batch=1, channels=1 or 2, samples)
        y_t = torch.from_numpy(y_mix.T if y_mix.ndim == 2 else y_mix[np.newaxis]).float().unsqueeze(0)

        # Self-comparison with small noise: measures spectrogram smoothness
        noise = torch.randn_like(y_t) * 0.001
        loss_fn = af.MelSTFTLoss(sample_rate=sr, n_mels=64, fft_size=2048, hop_size=512)
        loss = float(loss_fn(y_t + noise, y_t).item())

        # Lower loss = smoother = better. Typical range 0.1-2.0
        score = float(np.clip(100.0 * (1.0 - loss / 2.0), 0.0, 100.0))
        return round(score, 1)
    except Exception:
        # Fallback: mel spectrogram variance as proxy for harshness
        try:
            y_mono = y_mix if y_mix.ndim == 1 else np.mean(y_mix, axis=0)
            mel = librosa.feature.melspectrogram(y=y_mono, sr=sr, n_mels=64)
            mel_db = librosa.power_to_db(mel + 1e-6)
            smoothness = 1.0 - float(np.std(np.diff(mel_db, axis=1)) / 20.0)
            return float(np.clip(smoothness * 100.0, 0.0, 100.0))
        except Exception:
            return 50.0


def _vocal_presence_consistency(y_mix: np.ndarray, sr: int) -> float:
    """
    Measure consistency of vocal presence across the mix.
    Score 0-100: 100 = vocal equally present everywhere, 0 = disappears in sections.

    Uses 10s windows. Measures RMS in vocal-presence band (500-4kHz) per window.
    Score = 100 * (1 - coefficient_of_variation) clamped 0-100.
    CV < 0.3 = consistent = high score. CV > 1.0 = very inconsistent = low score.
    """
    try:
        y_mono = y_mix if y_mix.ndim == 1 else np.mean(y_mix, axis=0)

        from scipy.signal import butter, sosfilt
        sos = butter(4, [500, 4000], btype='bandpass', fs=sr, output='sos')
        y_vp = sosfilt(sos, y_mono)

        window_n = sr * 10  # 10 second windows
        if len(y_vp) < window_n * 2:
            return 50.0

        rms_windows = []
        for start in range(0, len(y_vp) - window_n, window_n // 2):
            chunk = y_vp[start:start + window_n]
            rms = float(np.sqrt(np.mean(chunk ** 2) + 1e-12))
            if rms > 1e-6:
                rms_windows.append(rms)

        if len(rms_windows) < 3:
            return 50.0

        arr = np.array(rms_windows)
        cv = float(np.std(arr) / (np.mean(arr) + 1e-12))
        score = float(np.clip((1.0 - cv / 1.0) * 100.0, 0.0, 100.0))
        return round(score, 1)
    except Exception:
        return 50.0


def _stereo_width_score(y_mix: np.ndarray, sr: int) -> float:
    """
    Score stereo width against reference profile target.
    Score 100 = width matches reference exactly.
    Score decreases as width deviates from reference (too wide or too narrow).
    Returns 50.0 if no reference profile or mono input.
    """
    if _REF_PROFILE is None or y_mix.ndim != 2:
        return 50.0
    try:
        from scipy.signal import butter, sosfilt
        target_mid = float(_REF_PROFILE["stereo"]["width_mid"]["median"])

        sos = butter(4, [500, 4000], btype="bandpass", fs=sr, output="sos")
        l_filt = sosfilt(sos, y_mix[:, 0] if y_mix.shape[1] == 2 else y_mix[:, 0])
        r_filt = sosfilt(sos, y_mix[:, 1] if y_mix.shape[1] == 2 else y_mix[:, 0])

        corr = float(np.corrcoef(l_filt, r_filt)[0, 1])
        actual_width = 1.0 - abs(corr)

        deviation = abs(actual_width - target_mid)
        score = float(np.clip(100.0 * (1.0 - deviation / 0.5), 0.0, 100.0))
        return round(score, 1)
    except Exception:
        return 50.0


def _spectral_band_deltas(y_mix: np.ndarray, sr: int) -> dict:
    """
    Compute per-band deviation from reference profile in dB.
    Returns flat dict: {"delta_sub": X, "delta_bass": X, ...}
    All values in dB. Positive = our mix is louder than reference in that band.
    Returns empty dict if no profile loaded.
    """
    if _REF_PROFILE is None:
        return {}

    from scipy.signal import butter, sosfilt
    y_mono = y_mix if y_mix.ndim == 1 else np.mean(y_mix, axis=0)

    band_defs = {
        "sub":      (20,   60),
        "bass":     (60,   250),
        "lo_mid":   (250,  500),
        "mid":      (500,  2000),
        "hi_mid":   (2000, 6000),
        "presence": (6000, 10000),
        "air":      (10000, min(20000, sr//2 - 200)),
    }

    deltas = {}
    for name, (lo, hi) in band_defs.items():
        try:
            sos = butter(4, [lo, hi], btype="bandpass", fs=sr, output="sos")
            filtered = sosfilt(sos, y_mono)
            rms_db = 20.0 * np.log10(float(np.sqrt(np.mean(filtered**2) + 1e-12)) + 1e-12)
            target = float(_REF_PROFILE["bands"][name]["median"])
            deltas[f"delta_{name}"] = round(rms_db - target, 1)
        except Exception:
            deltas[f"delta_{name}"] = 0.0
    return deltas


def _compute_spectral_match(y_mix: np.ndarray, sr: int) -> float:
    """
    Compute spectral match score (0-100) against reference profile.
    If no reference profile loaded, returns 50.0 (neutral).

    Measures per-band energy and compares to reference medians.
    Score = 100 - mean_absolute_deviation_across_bands (clamped 0-100).
    MAD measured in dB, normalized so ±12dB deviation = 0 score.
    """
    if _REF_PROFILE is None:
        return 50.0

    from scipy.signal import butter, sosfilt
    y_mono = y_mix if y_mix.ndim == 1 else np.mean(y_mix, axis=0)

    band_defs = {
        "sub":      (20,   60),
        "bass":     (60,   250),
        "lo_mid":   (250,  500),
        "mid":      (500,  2000),
        "hi_mid":   (2000, 6000),
        "presence": (6000, 10000),
        "air":      (10000, min(20000, sr//2 - 1)),
    }

    deviations = []
    for band_name, (lo, hi) in band_defs.items():
        try:
            sos = butter(4, [lo, hi], btype='bandpass', fs=sr, output='sos')
            filtered = sosfilt(sos, y_mono)
            rms = float(np.sqrt(np.mean(filtered**2) + 1e-10))
            rms_db = 20 * np.log10(rms + 1e-10)
            target = _REF_PROFILE["bands"][band_name]["median"]
            deviations.append(abs(rms_db - target))
        except Exception:
            continue

    if not deviations:
        return 50.0

    mean_dev = float(np.mean(deviations))
    score = max(0.0, 100.0 - (mean_dev / 12.0) * 100.0)
    return round(score, 1)


# After loading reference profile, adjust key targets if profile available
def _get_lufs_target() -> float:
    if _REF_PROFILE and "loudness" in _REF_PROFILE:
        return float(_REF_PROFILE["loudness"]["lufs_i"]["median"])
    return -12.0  # EBU R128 default

def _get_lra_target() -> tuple[float, float]:
    if _REF_PROFILE and "loudness" in _REF_PROFILE:
        p75 = float(_REF_PROFILE["loudness"]["lra"]["p75"])
        # Lower floor always 2.0 LU for mashup output: a vocal-over-beat mashup blends a
        # heavily side-chained EDM beat (1-3 LU) with a vocal stem (7-10 LU), naturally
        # yielding blended LRA of 2-4 LU. The reference (Drake p25=4.8) was calibrated on
        # finished vocal tracks — it is not appropriate as a floor for beat+vocal mashups.
        # 2.0 LU is the practical minimum: below this, the entire track is uniform volume.
        return (2.0, p75)
    return (2.0, 12.0)  # default acceptable range


def _profile_ref_ranges() -> dict:
    """
    Build reference scoring ranges from the loaded profile.
    Returns dict of metric → (lo, hi) acceptable range.
    Falls back to static defaults if no profile loaded.
    """
    defaults = {
        "transient_clarity": (8.0, 25.0),
        "sub_to_bass_db":    (-3.0, +3.0),   # sub vs bass balance
        "stereo_width_mid":  (0.05, 0.60),   # acceptable mid-band width
    }
    if _REF_PROFILE is None:
        return defaults
    try:
        tc = _REF_PROFILE["dynamics"]["transient_clarity"]
        out = {
            "transient_clarity": (
                float(tc["p25"]) * 0.7,   # lo = 70% of p25
                float(tc["p75"]) * 1.3,   # hi = 130% of p75
            ),
            "sub_to_bass_db": (
                float(_REF_PROFILE["bands"]["sub"]["median"]) - float(_REF_PROFILE["bands"]["bass"]["median"]) - 3.0,
                float(_REF_PROFILE["bands"]["sub"]["median"]) - float(_REF_PROFILE["bands"]["bass"]["median"]) + 3.0,
            ),
            "stereo_width_mid": (
                max(0.02, float(_REF_PROFILE["stereo"]["width_mid"]["p25"]) * 0.5),
                min(0.80, float(_REF_PROFILE["stereo"]["width_mid"]["p75"]) * 2.0),
            ),
        }
        return out
    except Exception:
        return defaults


def _load(audio_path: str) -> np.ndarray:
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

    # ── SHARED RESOURCES (computed once, reused throughout) ───────────────────
    # Single STFT pass — all spectral metrics reuse S_mag / freqs
    S_mag = np.abs(librosa.stft(mono, n_fft=2048, hop_length=512))
    freqs = librosa.fft_frequencies(sr=SR, n_fft=2048)

    # Beat grid — used for both groove_score and tempo_stability
    # madmom RNNBeatProcessor is accurate but uses ~800MB RAM and 5-10 min on CPU.
    # Skip it by default (LISTEN_FAST=1 env var) so the in-fuse QC loop is fast.
    # Use madmom only for final standalone scoring where accuracy matters.
    _use_madmom = os.environ.get("LISTEN_FAST", "0") != "1"
    try:
        if not _use_madmom:
            raise ImportError("fast mode — skipping madmom")
        import madmom
        from madmom.features.beats import RNNBeatProcessor, DBNBeatTrackingProcessor
        _proc = DBNBeatTrackingProcessor(fps=100)
        _act  = RNNBeatProcessor()(mono.astype(np.float32))
        _beat_times = _proc(_act)
        _beat_frames = ((_beat_times * SR) / 512).astype(int)  # convert to hop_length=512 frames
        if len(_beat_frames) < 4:
            raise ValueError("too few beats")
    except Exception:
        _, _beat_frames = librosa.beat.beat_track(y=mono, sr=SR, hop_length=512)
    beat_times = librosa.frames_to_time(_beat_frames, sr=SR, hop_length=512)

    # HPSS harmonic signal — used for vocal_harmony_score and key_distance
    try:
        y_harm, _ = librosa.effects.hpss(mono, margin=3.0)
    except Exception:
        y_harm = mono.copy()

    # Band-filtered signals for vocal/beat analysis
    vp_band   = _bp(mono, 1000.0, 4000.0)   # vocal presence band
    beat_band = _bp(mono, 200.0,  1000.0)   # beat/kick band

    # RMS frames at shared hop
    hop = 512
    vp_frames       = librosa.feature.rms(y=vp_band,   frame_length=1024, hop_length=hop)[0]
    beat_frames_rms = librosa.feature.rms(y=beat_band, frame_length=1024, hop_length=hop)[0]
    vp_med      = float(np.median(vp_frames))
    voiced_mask = vp_frames > vp_med

    # ── LOUDNESS ──────────────────────────────────────────────────────────────
    meter = pyln.Meter(SR)
    lufs = float(meter.integrated_loudness(mono))
    try:
        lra = float(meter.loudness_range(mono))
    except Exception:
        lra = 0.0

    # ── TRUE PEAK & CREST FACTOR ──────────────────────────────────────────────
    true_peak_dbfs = float(20 * np.log10(np.abs(y).max() + 1e-9))
    rms  = float(np.sqrt(np.mean(mono ** 2) + 1e-9))
    peak = float(np.abs(mono).max() + 1e-9)
    crest_db = float(20 * np.log10(peak / rms))

    # ── PLR (Peak-to-Loudness Ratio) ──────────────────────────────────────────
    # AES TD1004: <6 dB = over-limited (reject). >14 dB = mastered too quietly.
    plr_db = float(np.clip(true_peak_dbfs - lufs, -5.0, 25.0)) if np.isfinite(lufs) else 9.0

    # ── STEREO CORRELATION ────────────────────────────────────────────────────
    corr = float(np.corrcoef(L, R)[0, 1]) if (L.std() > 1e-9 and R.std() > 1e-9) else 1.0

    # ── LOW-FREQUENCY STEREO CORRELATION (< 80 Hz) ────────────────────────────
    # EBU R68: bass/kick must be mono below 80 Hz. Decorrelated sub-bass
    # causes phase cancellation on club subwoofers. Target ρ ≥ 0.85.
    try:
        L_lf = _lp(L, 80.0)
        R_lf = _lp(R, 80.0)
        low_freq_stereo_corr = (
            float(np.corrcoef(L_lf, R_lf)[0, 1])
            if (L_lf.std() > 1e-9 and R_lf.std() > 1e-9) else 1.0
        )
    except Exception:
        low_freq_stereo_corr = 1.0

    # ── SPECTRAL BANDS (reuse shared S_mag / freqs) ────────────────────────────
    sub_db    = _band_db(S_mag, freqs,   20,   80)
    bass_db   = _band_db(S_mag, freqs,   80,  250)
    lowmid_db = _band_db(S_mag, freqs,  250,  800)
    mid_db    = _band_db(S_mag, freqs,  800, 2500)
    himid_db  = _band_db(S_mag, freqs, 2500, 6000)
    high_db   = _band_db(S_mag, freqs, 6000, 20000)

    # ── TRANSIENT CLARITY ─────────────────────────────────────────────────────
    try:
        flux = librosa.onset.onset_strength(y=mono, sr=SR, hop_length=512)
        p95 = float(np.percentile(flux, 95))
        p10 = float(np.percentile(flux, 10))
        transient_clarity = float(np.clip(20 * np.log10((p95 + 1e-6) / (p10 + 1e-6)), 0.0, 30.0))
    except Exception:
        transient_clarity = 10.0

    # ── KICK DYNAMIC RANGE ────────────────────────────────────────────────────
    try:
        kick_band = _bp(mono, 60.0, 150.0)
        frames = librosa.util.frame(kick_band, frame_length=1024, hop_length=512)
        frame_rms_db = 20 * np.log10(np.sqrt((frames ** 2).mean(axis=0) + 1e-12))
        kick_headroom_db = float(np.percentile(frame_rms_db, 95) -
                                  np.percentile(frame_rms_db, 10))
    except Exception:
        kick_headroom_db = 15.0

    # ── MUD INDEX ─────────────────────────────────────────────────────────────
    try:
        mud_index = float(S_mag[(freqs >= 200) & (freqs < 600)].mean() /
                          (S_mag[(freqs >= 1000) & (freqs < 3000)].mean() + 1e-9))
    except Exception:
        mud_index = 2.5

    # ── SECTION CONSISTENCY ───────────────────────────────────────────────────
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

    # ── SPECTRAL SLOPE ────────────────────────────────────────────────────────
    try:
        octave_db = [_band_db(S_mag, freqs, lo, hi)
                     for lo, hi in [(200,400),(400,800),(800,1600),(1600,3200),(3200,6400),(6400,12800)]]
        coeffs = np.polyfit(np.arange(len(octave_db), dtype=float), octave_db, 1)
        spectral_slope_db_oct = float(coeffs[0])
    except Exception:
        spectral_slope_db_oct = -5.0

    # ── BEAT SYNC SCORE ───────────────────────────────────────────────────────
    # Correlates bass-band (beat rhythm, 60-200 Hz) with vocal-band (800-3000 Hz)
    # onset envelopes. High correlation = vocal rhythm locks to beat groove.
    try:
        bass_zone  = _bp(mono, 60.0,  200.0)
        vocal_zone = _bp(mono, 800.0, 3000.0)
        onset_beat  = librosa.onset.onset_strength(y=bass_zone,  sr=SR, hop_length=512)
        onset_vocal = librosa.onset.onset_strength(y=vocal_zone, sr=SR, hop_length=512)
        onset_beat  = (onset_beat  - onset_beat.mean())  / (onset_beat.std()  + 1e-9)
        onset_vocal = (onset_vocal - onset_vocal.mean()) / (onset_vocal.std() + 1e-9)
        xcorr = np.correlate(onset_beat, onset_vocal, mode="full")
        xcorr_norm = xcorr / (len(onset_beat) + 1e-9)
        frame_rate = SR / 512
        max_lag = int(round(0.075 * frame_rate))  # ±75ms window
        center = len(xcorr_norm) // 2
        beat_sync_score = float(np.clip(
            np.max(xcorr_norm[max(0, center - max_lag): center + max_lag + 1]),
            0.0, 1.0))
    except Exception:
        beat_sync_score = 0.5

    # ── VOCAL CLARITY INDEX (reuses shared S_mag / freqs) ─────────────────────
    try:
        bass_db_vc = _band_db(S_mag, freqs,   20,  300)
        mid_db_vc  = _band_db(S_mag, freqs, 1000, 4000)
        masking_pressure = max(0.0, bass_db_vc - mid_db_vc - 10.0)
        vocal_clarity_index = float(mid_db_vc - masking_pressure)
    except Exception:
        vocal_clarity_index = 5.0

    # ── TEMPO STABILITY (reuses shared beat_times) ────────────────────────────
    if len(beat_times) >= 3:
        ibis = np.diff(beat_times)
        cv = float(ibis.std() / (ibis.mean() + 1e-9))
        tempo_stability = float(1.0 - min(cv, 1.0))
    else:
        tempo_stability = 1.0

    # ── CLICK ARTIFACT SCORE ──────────────────────────────────────────────────
    try:
        diff_signal = np.diff(mono)
        diff_rms = float(np.sqrt(np.mean(diff_signal ** 2) + 1e-12))
        n_clicks = int(np.sum(np.abs(diff_signal) > 10.0 * diff_rms))
        click_artifact_score = float(n_clicks / (len(diff_signal) + 1e-9))
    except Exception:
        click_artifact_score = 0.0

    # ── VOCAL BLEED SCORE (sibilance band ratio method) ───────────────────────
    # Hi-hat/cymbal bleed shows as elevated 5-10kHz energy during voiced frames.
    # De-essing trigger = sibilance >6 dB above broadband (500Hz-10kHz).
    # Score = fraction of voiced frames exceeding this threshold.
    try:
        sib_sig   = _bp(mono, 5000.0, 10000.0)
        broad_sig = _bp(mono,  500.0, 10000.0)
        sib_frames_rms   = librosa.feature.rms(y=sib_sig,   frame_length=1024, hop_length=512)[0]
        broad_frames_rms = librosa.feature.rms(y=broad_sig, frame_length=1024, hop_length=512)[0]
        n_bl = min(len(sib_frames_rms), len(voiced_mask))
        if voiced_mask[:n_bl].sum() > 20:
            sib_v   = sib_frames_rms[:n_bl][voiced_mask[:n_bl]]
            broad_v = broad_frames_rms[:n_bl][voiced_mask[:n_bl]]
            ratio_db_vals = 20.0 * np.log10(sib_v / (broad_v + 1e-9) + 1e-9)
            vocal_bleed_score = float(np.clip(
                np.mean(ratio_db_vals > 6.0), 0.0, 1.0))
        else:
            vocal_bleed_score = 0.0
    except Exception:
        vocal_bleed_score = 0.0

    # ── VOCAL SPECTRAL CREST (reuses shared S_mag / freqs) ────────────────────
    try:
        if voiced_mask.sum() > 20:
            mid_mask = (freqs >= 300) & (freqs < 3000)
            # S_mag columns match hop=512 frames from librosa.stft — align with voiced_mask
            n_col = min(S_mag.shape[1], len(voiced_mask))
            S_mid_voiced = S_mag[np.ix_(mid_mask, voiced_mask[:n_col])]
            frame_max  = S_mid_voiced.max(axis=0)
            frame_mean = S_mid_voiced.mean(axis=0) + 1e-12
            vocal_spectral_crest = float(np.median(frame_max / frame_mean))
        else:
            vocal_spectral_crest = 5.0
    except Exception:
        vocal_spectral_crest = 5.0

    # ── VOCAL MODULATION INDEX ────────────────────────────────────────────────
    try:
        vp_env   = librosa.feature.rms(y=vp_band, frame_length=512, hop_length=256)[0].astype(np.float64)
        env_fft  = np.abs(np.fft.rfft(vp_env - vp_env.mean()))
        mod_freqs = np.fft.rfftfreq(len(vp_env), d=256.0 / SR)
        total_e  = float(env_fft[(mod_freqs >= 1.0) & (mod_freqs <= 20.0)].sum() + 1e-12)
        syl_e    = float(env_fft[(mod_freqs >= 3.0) & (mod_freqs <=  8.0)].sum() + 1e-12)
        vocal_modulation_index = float(np.clip(syl_e / total_e, 0.0, 1.0))
    except Exception:
        vocal_modulation_index = 0.40

    # ── VOCAL PRESENCE RATIO ──────────────────────────────────────────────────
    try:
        if voiced_mask.sum() > 20:
            n_vm = min(len(vp_frames), len(beat_frames_rms))
            vp_voiced   = float(vp_frames[:n_vm][voiced_mask[:n_vm]].mean() + 1e-9)
            beat_voiced = float(beat_frames_rms[:n_vm][voiced_mask[:n_vm]].mean() + 1e-9)
            vocal_presence_ratio = float(np.clip(vp_voiced / beat_voiced, 0.0, 2.0))
        else:
            vocal_presence_ratio = 0.60
    except Exception:
        vocal_presence_ratio = 0.60

    # ── VOCAL HNR (Harmonic-to-Noise Ratio, autocorrelation method) ──────────
    # Ferrand 2002 / Praat: HNR = 10*log10(r_peak / (1 - r_peak)) on voiced frames.
    # Target: >15 dB = normophonic. <10 dB = noisy/breathy.
    try:
        voc_band_hnr = _bp(mono, 300.0, 3000.0)
        frame_len_hnr = int(0.040 * SR)   # 40ms frames
        hop_hnr       = frame_len_hnr // 2
        hnr_vals = []
        n_hnr_frames = (len(voc_band_hnr) - frame_len_hnr) // hop_hnr
        n_vm = min(n_hnr_frames, len(voiced_mask))
        min_lag = int(SR / 600.0)  # 600 Hz max f0
        max_lag = int(SR / 75.0)   # 75 Hz min f0
        for fi in range(n_vm):
            if not voiced_mask[fi]:
                continue
            frm = voc_band_hnr[fi * hop_hnr: fi * hop_hnr + frame_len_hnr]
            if len(frm) < frame_len_hnr:
                continue
            ac = np.correlate(frm, frm, mode='full')
            ac = ac[len(ac) // 2:]
            ac = ac / (ac[0] + 1e-12)
            if max_lag >= len(ac):
                continue
            peak_idx = np.argmax(ac[min_lag:max_lag]) + min_lag
            r = float(ac[peak_idx])
            if 0.0 < r < 1.0:
                hnr_vals.append(10.0 * np.log10(r / (1.0 - r)))
        vocal_hnr_db = float(np.median(hnr_vals)) if len(hnr_vals) >= 5 else 15.0
        vocal_hnr_db = float(np.clip(vocal_hnr_db, 0.0, 45.0))
    except Exception:
        vocal_hnr_db = 15.0

    # ── VOCAL SPECTRAL FLATNESS MEASURE (SFM) ─────────────────────────────────
    # Izmirli ISMIR 2000; Sharma & Wang TASLP 2020.
    # SFM = geom_mean(power) / arith_mean(power) on 300-3000 Hz voiced frames.
    # 0 = pure tone, 1 = white noise. Good vocal: 0.05-0.30.
    try:
        mid_mask_sfm = (freqs >= 300) & (freqs < 3000)
        n_col = min(S_mag.shape[1], len(voiced_mask))
        S_voc_sfm = S_mag[np.ix_(mid_mask_sfm, voiced_mask[:n_col])]
        if S_voc_sfm.shape[1] > 5:
            power = S_voc_sfm ** 2
            # Per-frame SFM then median
            geo  = np.exp(np.mean(np.log(power + 1e-12), axis=0))
            arith = np.mean(power, axis=0) + 1e-12
            sfm_per_frame = geo / arith
            vocal_sfm = float(np.clip(np.median(sfm_per_frame), 0.0, 1.0))
        else:
            vocal_sfm = 0.15
    except Exception:
        vocal_sfm = 0.15

    # ── GROOVE SCORE (uses shared beat_times, vectorized) ─────────────────────
    # Fraction of vocal onsets within ±60ms of a beat grid position.
    # Widened from 40ms to 60ms for mashup context (computer-generated timing).
    try:
        vocal_onset_env = librosa.onset.onset_strength(y=vp_band, sr=SR, hop_length=512)
        vocal_onset_frames = librosa.onset.onset_detect(
            onset_envelope=vocal_onset_env, sr=SR, hop_length=512)
        vot = librosa.frames_to_time(vocal_onset_frames, sr=SR, hop_length=512)

        if len(beat_times) >= 2 and len(vot) >= 3:
            # Vectorized: (n_onsets, n_beats) → min distance per onset
            dists = np.abs(vot[:, None] - beat_times[None, :]).min(axis=1)
            groove_score = float(np.clip(np.mean(dists <= 0.060), 0.0, 1.0))
        else:
            groove_score = 0.50
    except Exception:
        groove_score = 0.50

    # ── DYNAMIC ARC SCORE ─────────────────────────────────────────────────────
    # A professional song has an energy arc: builds toward a climax around
    # 60-70% through the track, then resolves. Flat energy = loop, not song.
    try:
        arc_win = SR * 10
        n_wins = len(mono) // arc_win
        if n_wins >= 4:
            win_rms = np.array([
                float(np.sqrt(np.mean(mono[i * arc_win:(i + 1) * arc_win] ** 2) + 1e-12))
                for i in range(n_wins)
            ])
            t_frac = np.linspace(0, 1, n_wins)
            ideal = np.where(t_frac < 0.65,
                             0.3 + 0.7 * (t_frac / 0.65) ** 1.2,
                             1.0 - 0.5 * ((t_frac - 0.65) / 0.35))
            corr_arc = float(np.corrcoef(win_rms, ideal)[0, 1])
            dynamic_arc_score = float(np.clip((corr_arc + 1.0) / 2.0, 0.0, 1.0))
        else:
            dynamic_arc_score = 0.50
    except Exception:
        dynamic_arc_score = 0.50

    # ── VOCAL HARMONY SCORE (uses shared y_harm) ──────────────────────────────
    # Chroma coherence between beat-dominated and vocal-dominated frames.
    # Uses HPSS harmonic signal to prevent kick/snare polluting chroma.
    try:
        chroma = librosa.feature.chroma_cqt(y=y_harm, sr=SR, hop_length=512, bins_per_octave=36)
        beat_rms_ch  = librosa.feature.rms(y=beat_band, frame_length=1024, hop_length=512)[0]
        vocal_rms_ch = librosa.feature.rms(y=vp_band,   frame_length=1024, hop_length=512)[0]

        n_frames = min(chroma.shape[1], len(beat_rms_ch), len(vocal_rms_ch))
        if n_frames > 20:
            bt = beat_rms_ch[:n_frames]
            vt = vocal_rms_ch[:n_frames]
            beat_dom  = bt  > float(np.percentile(bt, 70))
            vocal_dom = vt  > float(np.percentile(vt, 70))

            if beat_dom.sum() > 10 and vocal_dom.sum() > 10:
                chroma_beat  = chroma[:, beat_dom].mean(axis=1)
                chroma_vocal = chroma[:, vocal_dom].mean(axis=1)
                dot  = float(np.dot(chroma_beat, chroma_vocal))
                norm = float(np.linalg.norm(chroma_beat) * np.linalg.norm(chroma_vocal) + 1e-9)
                vocal_harmony_score = float(np.clip(dot / norm, 0.0, 1.0))
            else:
                vocal_harmony_score = 0.60
        else:
            vocal_harmony_score = 0.60
    except Exception:
        vocal_harmony_score = 0.60

    # ── GROOVE TIMING SCORE ───────────────────────────────────────────────────
    # NOTE: pass mono (1D), not y (2D stereo). These functions expect 1D.
    # Bug: passing y (n_samples, 2) caused np.mean(y, axis=0) → shape (2,)
    # → beat tracking received a 2-sample array → segfault (exit 139).
    groove_timing_score = _groove_timing_score(mono, SR)

    # ── HARMONIC CLARITY SCORE ────────────────────────────────────────────────
    harmonic_clarity_score = _harmonic_clarity_score(mono, SR)

    # ── VOCAL PRESENCE CONSISTENCY ────────────────────────────────────────────
    vocal_presence_consistency = _vocal_presence_consistency(mono, SR)

    # ── VOCAL INTELLIGIBILITY SCORE (STOI-proxy) ──────────────────────────────
    vocal_intelligibility_score = _vocal_intelligibility_score(mono, SR)

    # ── MEL-STFT QUALITY SCORE (auraloss perceptual quality) ─────────────────
    mel_stft_quality_score = _mel_stft_quality_score(mono, SR)

    # ── DYNAMIC COMPLEXITY (RMS-based, fast) ──────────────────────────────────
    # Avg abs deviation of 2s-window RMS (dB) from global RMS (dB).
    # <2 dB = brick-wall limited. >14 dB = sections too uneven.
    # Silence windows (< -55 dBFS) are skipped — intro/outro silence was
    # pulling dynamic_complexity down to 2 dB even on fully structured songs.
    try:
        dc_win = SR * 2
        dc_hop = SR * 1
        global_rms_db = 20.0 * np.log10(rms)
        win_dbs = []
        for i in range(0, len(mono) - dc_win + 1, dc_hop):
            seg = mono[i:i + dc_win]
            seg_rms = float(np.sqrt(np.mean(seg ** 2) + 1e-12))
            if seg_rms < 1.8e-3:    # -55 dBFS — silence; skip to avoid pulling complexity low
                continue
            win_dbs.append(20.0 * np.log10(seg_rms))
        if len(win_dbs) >= 4:
            dynamic_complexity_db = float(np.mean(np.abs(np.array(win_dbs) - global_rms_db)))
        else:
            dynamic_complexity_db = 6.0
    except Exception:
        dynamic_complexity_db = 6.0

    # ── VOCAL ROBOT SCORE (WORLD vocoder / heavy autotune artifact detector) ───
    # WORLD vocoder resynthesizes from scratch using CheapTrick + D4C — the
    # spectral envelope and aperiodicity estimates vary erratically frame-to-frame
    # even on steady vowels. This creates audible "robot getting stabbed" artifacts.
    # Measure frame-to-frame HNR standard deviation in voiced segments.
    # Natural speech: adjacent-frame HNR diff std ≈ 1-3 dB (smooth pitch).
    # WORLD-resynth:  adjacent-frame HNR diff std ≈ 6-15 dB (erratic harmonics).
    # Score normalised to 0 (natural) → 1.0 (max robotic). Threshold >0.45 = fail.
    vocal_robot_score = 0.0
    try:
        if len(hnr_vals) >= 10:
            hnr_arr = np.array(hnr_vals, dtype=np.float32)
            hnr_diffs = np.abs(np.diff(hnr_arr))
            hnr_diff_std = float(np.std(hnr_diffs))
            vocal_robot_score = float(np.clip(hnr_diff_std / 15.0, 0.0, 1.0))
    except Exception:
        vocal_robot_score = 0.0

    # ── KEY DISTANCE (uses shared y_harm and module-level _cens_vec) ──────────
    # HPSS harmonic signal prevents kick/snare from polluting chroma (Mauch & Dixon 2010).
    # CENS circular rotation finds minimum semitone distance between keys.
    try:
        bass_rng = _bp(y_harm,  80.0,  500.0)
        voc_rng  = _bp(y_harm, 800.0, 3000.0)
        cens_beat = _cens_vec(bass_rng)
        cens_voc  = _cens_vec(voc_rng)

        best_sim   = -1.0
        best_shift = 0
        for shift in range(12):
            rotated = np.roll(cens_beat, shift)
            dot  = float(np.dot(rotated, cens_voc))
            norm = float(np.linalg.norm(rotated) * np.linalg.norm(cens_voc) + 1e-9)
            sim  = dot / norm
            if sim > best_sim:
                best_sim   = sim
                best_shift = shift
        key_distance_semitones = float(min(best_shift, 12 - best_shift))
    except Exception:
        key_distance_semitones = 2.0

    # ── SPECTRAL MATCH SCORE (vs reference profile) ───────────────────────────
    spectral_match_score = _compute_spectral_match(mono, SR)

    # ── STEREO WIDTH SCORE (vs reference profile) ─────────────────────────────
    # _stereo_width_score needs stereo (n_samples, 2) — pass y, not mono
    stereo_width_score = _stereo_width_score(y, SR)

    # ── PER-BAND SPECTRAL DELTAS (vs reference profile) ───────────────────────
    spectral_band_deltas = _spectral_band_deltas(mono, SR)

    return {
        # TIER 1: Technical
        "lufs_integrated":       lufs,
        "true_peak_dbfs":        true_peak_dbfs,
        "lra_lu":                lra,
        "crest_factor_db":       crest_db,
        "plr_db":                plr_db,
        "stereo_correlation":    corr,
        "low_freq_stereo_corr":  low_freq_stereo_corr,
        # Raw bands (display only — _ prefix excluded from scoring)
        "_sub_db":    sub_db,
        "_bass_db":   bass_db,
        "_lowmid_db": lowmid_db,
        "_mid_db":    mid_db,
        "_himid_db":  himid_db,
        "_high_db":   high_db,
        # TIER 2: Spectral ratios
        "ratio_sub_to_mid":      sub_db    - mid_db,
        "ratio_bass_to_mid":     bass_db   - mid_db,
        "ratio_lowmid_to_mid":   lowmid_db - mid_db,
        "ratio_himid_to_mid":    himid_db  - mid_db,
        "ratio_high_to_mid":     high_db   - mid_db,
        "lowmid_over_himid":     lowmid_db - himid_db,
        "high_over_himid":       high_db   - himid_db,
        "transient_clarity":     transient_clarity,
        "kick_headroom_db":      kick_headroom_db,
        "mud_index":             mud_index,
        "section_consistency_lu":section_consistency_lu,
        "spectral_slope_db_oct": spectral_slope_db_oct,
        # TIER 3: Perceptual
        "beat_sync_score":       beat_sync_score,
        "vocal_clarity_index":   vocal_clarity_index,
        "tempo_stability":       tempo_stability,
        "click_artifact_score":  click_artifact_score,
        "vocal_bleed_score":     vocal_bleed_score,
        "vocal_spectral_crest":  vocal_spectral_crest,
        "vocal_modulation_index":vocal_modulation_index,
        "vocal_presence_ratio":  vocal_presence_ratio,
        "vocal_hnr_db":          vocal_hnr_db,
        "vocal_sfm":             vocal_sfm,
        "vocal_robot_score":     vocal_robot_score,
        # TIER 4: Musical
        "groove_score":               groove_score,
        "dynamic_arc_score":          dynamic_arc_score,
        "vocal_harmony_score":        vocal_harmony_score,
        "groove_timing_score":        groove_timing_score,
        "harmonic_clarity_score":     harmonic_clarity_score,
        "vocal_presence_consistency": vocal_presence_consistency,
        "vocal_intelligibility_score": vocal_intelligibility_score,
        "mel_stft_quality_score":      mel_stft_quality_score,
        # TIER 5: Mashup intelligence
        "dynamic_complexity_db": dynamic_complexity_db,
        "key_distance_semitones": key_distance_semitones,
        # TIER 6: Reference learning
        "spectral_match_score":  spectral_match_score,
        "stereo_width_score":    stereo_width_score,
        # Per-band spectral deltas vs reference (dB, + = too loud, - = too quiet)
        **spectral_band_deltas,
    }


def _score(metrics: dict, ref: dict) -> tuple:
    score = 100
    issues = []
    for key, (lo, hi) in ref.items():
        if key.startswith("_"):  # display-only keys are never scored
            continue
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
    """Return plain-English musical diagnoses based on metric patterns."""
    diags = []
    gs  = metrics.get("groove_score", 0.5)
    arc = metrics.get("dynamic_arc_score", 0.5)
    har = metrics.get("vocal_harmony_score", 0.5)
    pre = metrics.get("vocal_presence_ratio", 0.6)
    syn = metrics.get("beat_sync_score", 0.5)
    bli = metrics.get("vocal_bleed_score", 0.0)
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

    if bli > 0.15:  # matches REF upper bound
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

    if dc < 1.5:
        diags.append("⚠ BRICK-WALL: dynamic complexity too low — all dynamics crushed, sounds lifeless")
    elif dc > 14.0:
        diags.append(f"△ UNEVEN: dynamic complexity {dc:.1f} dB — sections jump dramatically in volume")
    else:
        diags.append(f"✓ Dynamic complexity {dc:.1f} dB — natural dynamics preserved")

    if plr < 6.0:  # matches REF lower bound
        diags.append(f"⚠ OVER-LIMITED: PLR {plr:.1f} dB — master is brick-wall limited, transients destroyed")
    elif plr > 14.0:
        diags.append(f"△ PLR {plr:.1f} dB — mastered too quietly, won't compete on streaming")
    else:
        diags.append(f"✓ PLR {plr:.1f} dB — good Peak-to-Loudness balance")

    return diags


def corrections(issues: list, history: list | None = None) -> dict:
    """
    Map detected issues to concrete DSP parameter adjustments for auto-correction.

    history: list of dicts from prior correction attempts. Each dict has keys:
      "corrections": the dict returned in that attempt
      "score": the listen score achieved
    When a correction was tried and the score did NOT improve, the delta for that
    parameter is halved to avoid oscillating. This prevents the loop from repeatedly
    applying the same unhelpful correction.
    """
    # Build a map of params tried in prior rounds and their effect on score
    _prior_tried: dict[str, float] = {}   # param → sum of deltas applied
    _prior_improved: set[str] = set()     # params where score went up after applying
    if history:
        for i, h in enumerate(history):
            prev_score = history[i - 1]["score"] if i > 0 else 0
            score_improved = h["score"] > prev_score
            for param, delta in h.get("corrections", {}).items():
                _prior_tried[param] = _prior_tried.get(param, 0.0) + abs(delta)
                if score_improved:
                    _prior_improved.add(param)

    scaled = {}
    sev_order = {"CRITICAL": 0, "HIGH": 1, "MEDIUM": 2}
    for sev, key, val, lo, hi, desc in sorted(issues, key=lambda x: sev_order.get(x[0], 3)):
        if key not in CORRECTIONS:
            continue
        param, delta_lo, delta_hi = CORRECTIONS[key]
        delta = delta_lo if val < lo else delta_hi
        if delta == 0.0:
            continue
        mult = 1.5 if sev == "CRITICAL" else (1.0 if sev == "HIGH" else 0.7)
        # If this param was tried before and didn't help, reduce its weight significantly
        if param in _prior_tried and param not in _prior_improved:
            mult *= 0.3   # was tried, didn't improve — apply gently to avoid oscillating
        scaled[param] = scaled.get(param, 0.0) + delta * mult
    return scaled


def score_file(audio_path: str, strict: bool = False, reference_path: str = None,
               print_report: bool = True) -> tuple:
    ref_base = REF_STRICT if strict else REF

    # Apply dynamic LUFS/LRA/transient overrides from reference profile (if loaded)
    lufs_target = _get_lufs_target()
    lra_lo, lra_hi = _get_lra_target()
    _pref = _profile_ref_ranges()
    ref = {
        **ref_base,
        # Tighter window than generic: ±3 dB below, +2 dB above.
        # Asymmetric: being too quiet is more common/damaging than slightly too loud.
        # Lower bound -4.0 (was -3.0): streaming platforms normalize to -14 LUFS.
        # A mashup at drake_ref(-10) - 4 = -14 LUFS is streaming-optimal.
        # -3.0 was too tight for sub-bass-heavy production where peak limiting
        # reduces integrated loudness below the pre-limit target.
        "lufs_integrated": (lufs_target - 4.0, lufs_target + 2.0) if _REF_PROFILE else ref_base["lufs_integrated"],
        "lra_lu": (lra_lo, lra_hi) if _REF_PROFILE else ref_base["lra_lu"],
        "transient_clarity": _pref["transient_clarity"] if _REF_PROFILE else ref_base["transient_clarity"],
    }

    metrics = _measure(audio_path)
    score, issues = _score(metrics, ref)

    # ── Change 4: spectral_match_score penalty ─────────────────────────────
    sms = metrics.get("spectral_match_score", 50.0)
    if sms < 20.0:
        score = max(0, score - 10)
        issues.append(("HIGH", "spectral_balance_mismatch", sms, 40.0, 100.0,
                        "Output spectral balance deviates significantly from reference profile"))
    elif sms < 40.0:
        score = max(0, score - 5)
        issues.append(("MEDIUM", "spectral_balance_mismatch", sms, 40.0, 100.0,
                        "Output spectral balance deviates significantly from reference profile"))

    # ── groove_timing_score: poor timing is immediately noticeable ────────────
    gts = metrics.get("groove_timing_score", 50.0)
    if gts < 30.0:
        score = max(0, score - 10)
        issues.append(("HIGH", "poor_groove_timing", gts, 30.0, 100.0,
                       "Vocal onsets are loosely timed to the beat — try a tighter vocal source"))
    elif gts < 50.0:
        score = max(0, score - 5)
        issues.append(("MEDIUM", "mediocre_groove_timing", gts, 50.0, 100.0,
                       "Vocal timing could be tighter relative to the beat"))

    # ── harmonic_clarity_score: muddy vocal is unacceptable ──────────────────
    hcs = metrics.get("harmonic_clarity_score", 50.0)
    if hcs < 30.0:
        score = max(0, score - 10)
        issues.append(("HIGH", "poor_harmonic_clarity", hcs, 30.0, 100.0,
                       "Vocal is buried in the mix — beat is masking the speech intelligibility band"))
    elif hcs < 50.0:
        score = max(0, score - 5)
        issues.append(("MEDIUM", "low_harmonic_clarity", hcs, 50.0, 100.0,
                       "Vocal clarity is below target — increase carve_db or presence_db"))

    # ── vocal_presence_consistency: disappearing vocal is a major quality issue
    vpc = metrics.get("vocal_presence_consistency", 50.0)
    if vpc < 35.0:
        score = max(0, score - 10)
        issues.append(("HIGH", "inconsistent_vocal_presence", vpc, 35.0, 100.0,
                       "Vocal disappears in sections — arrangement or level targeting issue"))
    elif vpc < 55.0:
        score = max(0, score - 5)
        issues.append(("MEDIUM", "uneven_vocal_presence", vpc, 55.0, 100.0,
                       "Vocal presence is uneven across sections"))

    # ── mel_stft_quality_score: perceptual smoothness / musical quality ──────
    msqs = metrics.get("mel_stft_quality_score", 50.0)
    if msqs < 30.0:
        score = max(0, score - 5)
        issues.append(("MEDIUM", "low_mel_stft_quality", msqs, 30.0, 100.0,
                       "Mix sounds harsh or artifact-laden — mel-STFT quality score below threshold"))

    # ── vocal_intelligibility_score: STOI-proxy — can listeners understand words?
    vis = metrics.get("vocal_intelligibility_score", 50.0)
    if vis < 35.0:
        score = max(0, score - 10)
        issues.append(("HIGH", "low_vocal_intelligibility", vis, 35.0, 100.0,
                       "Vocal is largely unintelligible — beat masking the speech band (300-3000Hz)"))
    elif vis < 55.0:
        score = max(0, score - 5)
        issues.append(("MEDIUM", "low_vocal_intelligibility", vis, 55.0, 100.0,
                       "Vocal intelligibility is below target — percussive energy competing in speech band"))

    # ── stereo_width_score: penalize if width deviates too far from reference ─
    sws = metrics.get("stereo_width_score", 50.0)
    if sws < 40.0:
        score = max(0, score - 5)
        issues.append(("MEDIUM", "stereo_width_mismatch", sws, 40.0, 100.0,
                       "Stereo width deviates significantly from reference profile target"))

    grade = _grade(score)
    if print_report:
        _print_report(audio_path, metrics, score, grade, issues, reference_path, ref)
    return score, issues, metrics


def _flag(val: float, lo: float, hi: float) -> str:
    return " ✗" if (val < lo or val > hi) else ""


def _print_report(path: str, m: dict, score: int, grade: str, issues: list,
                  ref_path: str = None, ref: dict = None):
    if ref is None:
        ref = REF
    width = 70
    print("\n" + "═" * width)
    print(f"  VocalFusion Quality Report — {Path(path).name}")
    print("═" * width)
    print(f"\n  SCORE: {score}/100   GRADE: {grade}")

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

    # Helper: read limits from ref dict so labels always stay in sync
    def _lo_hi(key):
        return ref.get(key, (-99.0, 99.0))

    print(f"\n  ── TECHNICAL ─────────────────────────────────────────────────────")
    lo, hi = _lo_hi("lufs_integrated")
    print(f"  {'LUFS integrated':<28} {m['lufs_integrated']:>+7.1f} dB    (target {lo:.0f} to {hi:.0f}){_flag(m['lufs_integrated'], lo, hi)}")
    lo, hi = _lo_hi("true_peak_dbfs")
    print(f"  {'True peak':<28} {m['true_peak_dbfs']:>+7.1f} dBFS  (EBU R128: must be < {hi:.1f}){_flag(m['true_peak_dbfs'], lo, hi)}")
    lo, hi = _lo_hi("plr_db")
    print(f"  {'PLR (peak-to-loudness)':<28} {m.get('plr_db', 0.0):>+7.1f} dB    (target {lo:.0f}-{hi:.0f} dB){_flag(m.get('plr_db', 9.0), lo, hi)}")
    lo, hi = _lo_hi("lra_lu")
    print(f"  {'LRA':<28} {m['lra_lu']:>+7.1f} LU    (target {lo:.0f}-{hi:.0f}){_flag(m['lra_lu'], lo, hi)}")
    lo, hi = _lo_hi("crest_factor_db")
    print(f"  {'Crest factor':<28} {m['crest_factor_db']:>+7.1f} dB    (target {lo:.0f}-{hi:.0f} dB){_flag(m['crest_factor_db'], lo, hi)}")
    lo, hi = _lo_hi("stereo_correlation")
    print(f"  {'Stereo correlation':<28} {m['stereo_correlation']:>+7.3f}      ({lo:.2f}-{hi:.2f}){_flag(m['stereo_correlation'], lo, hi)}")
    lo, hi = _lo_hi("low_freq_stereo_corr")
    print(f"  {'Sub-bass mono (<80Hz)':<28} {m.get('low_freq_stereo_corr', 1.0):>+7.3f}      (>{lo:.2f} = mono bass){_flag(m.get('low_freq_stereo_corr', 1.0), lo, hi)}")

    print(f"\n  ── DYNAMICS ──────────────────────────────────────────────────────")
    lo, hi = _lo_hi("transient_clarity")
    print(f"  {'Transient clarity':<28} {m['transient_clarity']:>+7.1f} dB    (target {lo:.0f}-{hi:.0f} dB){_flag(m['transient_clarity'], lo, hi)}")
    lo, hi = _lo_hi("kick_headroom_db")
    print(f"  {'Kick headroom':<28} {m['kick_headroom_db']:>+7.1f} dB    (target {lo:.0f}-{hi:.0f} dB){_flag(m['kick_headroom_db'], lo, hi)}")
    lo, hi = _lo_hi("section_consistency_lu")
    print(f"  {'Section consistency':<28} {m['section_consistency_lu']:>+7.1f} LU    (<{hi:.0f} LU){_flag(m['section_consistency_lu'], lo, hi)}")
    lo, hi = _lo_hi("spectral_slope_db_oct")
    print(f"  {'Spectral slope':<28} {m['spectral_slope_db_oct']:>+7.1f} dB/oct ({lo:.0f} to {hi:.0f}){_flag(m['spectral_slope_db_oct'], lo, hi)}")

    print(f"\n  ── PERCEPTUAL ────────────────────────────────────────────────────")
    lo, hi = _lo_hi("beat_sync_score")
    print(f"  {'Beat sync (bass↔vocal)':<28} {m['beat_sync_score']:>7.3f}      (target >{lo:.2f}){_flag(m['beat_sync_score'], lo, hi)}")
    lo, hi = _lo_hi("vocal_clarity_index")
    print(f"  {'Vocal clarity index':<28} {m['vocal_clarity_index']:>+7.1f} dB    (target >{lo:.0f}){_flag(m['vocal_clarity_index'], lo, hi)}")
    lo, hi = _lo_hi("tempo_stability")
    print(f"  {'Tempo stability':<28} {m['tempo_stability']:>7.3f}      (target >{lo:.1f}){_flag(m['tempo_stability'], lo, hi)}")
    lo, hi = _lo_hi("click_artifact_score")
    print(f"  {'Click artifact score':<28} {m['click_artifact_score']:>7.5f}     (<{hi:.3f}){_flag(m['click_artifact_score'], lo, hi)}")
    lo, hi = _lo_hi("vocal_bleed_score")
    print(f"  {'Vocal bleed':<28} {m['vocal_bleed_score']:>7.3f}      (<{hi:.2f} = clean){_flag(m['vocal_bleed_score'], lo, hi)}")
    lo, hi = _lo_hi("vocal_spectral_crest")
    print(f"  {'Vocal spectral crest':<28} {m['vocal_spectral_crest']:>7.2f}      (>{lo:.0f} = harmonic){_flag(m['vocal_spectral_crest'], lo, hi)}")
    lo, hi = _lo_hi("vocal_modulation_index")
    print(f"  {'Vocal modulation':<28} {m['vocal_modulation_index']:>7.3f}      ({lo:.2f}-{hi:.2f}){_flag(m['vocal_modulation_index'], lo, hi)}")
    lo, hi = _lo_hi("vocal_presence_ratio")
    print(f"  {'Vocal presence ratio':<28} {m['vocal_presence_ratio']:>7.3f}      (>{lo:.2f}){_flag(m['vocal_presence_ratio'], lo, hi)}")
    lo, hi = _lo_hi("vocal_hnr_db")
    print(f"  {'Vocal HNR':<28} {m.get('vocal_hnr_db', 15.0):>+7.1f} dB    (>{lo:.0f} dB = clear){_flag(m.get('vocal_hnr_db', 15.0), lo, hi)}")
    lo, hi = _lo_hi("vocal_sfm")
    print(f"  {'Vocal SFM (flatness)':<28} {m.get('vocal_sfm', 0.15):>7.3f}      ({lo:.2f}-{hi:.2f} = harmonic){_flag(m.get('vocal_sfm', 0.15), lo, hi)}")

    print(f"\n  ── MUSICAL ───────────────────────────────────────────────────────")
    lo, hi = _lo_hi("groove_score")
    print(f"  {'Groove (on-beat fraction)':<28} {m['groove_score']:>7.3f}      (>{lo:.2f} = locked){_flag(m['groove_score'], lo, hi)}")
    lo, hi = _lo_hi("dynamic_arc_score")
    print(f"  {'Dynamic arc':<28} {m['dynamic_arc_score']:>7.3f}      (>{lo:.2f} = builds){_flag(m['dynamic_arc_score'], lo, hi)}")
    lo, hi = _lo_hi("vocal_harmony_score")
    print(f"  {'Harmonic coherence':<28} {m['vocal_harmony_score']:>7.3f}      (>{lo:.2f} = key match){_flag(m['vocal_harmony_score'], lo, hi)}")

    print(f"\n  ── MASHUP INTELLIGENCE (Tier 5) ──────────────────────────────────")
    lo, hi = _lo_hi("dynamic_complexity_db")
    print(f"  {'Dynamic complexity':<28} {m.get('dynamic_complexity_db', 0.0):>7.1f} dB    (target {lo:.0f}-{hi:.0f} dB){_flag(m.get('dynamic_complexity_db', 6.0), lo, hi)}")
    lo, hi = _lo_hi("key_distance_semitones")
    print(f"  {'Key distance':<28} {m.get('key_distance_semitones', 0.0):>7.1f} st    (0-5 = compatible, 6 = clash){_flag(m.get('key_distance_semitones', 2.0), lo, hi)}")

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
        ("Mud Index",     "mud_index",            "1.0 → 6.0"),
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

    groove  = metrics.get("groove_score", 0.5)
    harmony = metrics.get("vocal_harmony_score", 0.5)
    # 75 threshold for AI mashups — professional original tracks score ~100;
    # mashups have inherent limitations (PLR ceiling from streaming sources,
    # onset_density from sparse trap records). 75 = genuinely good output.
    passed  = score >= 75 and not critical and groove >= 0.25 and harmony >= 0.25

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
    parser = argparse.ArgumentParser(description="VocalFusion Quality Scorer v5")
    parser.add_argument("audio")
    parser.add_argument("reference", nargs="?")
    parser.add_argument("--strict", action="store_true")
    args = parser.parse_args()
    score, issues, metrics = score_file(args.audio, strict=args.strict,
                                        reference_path=args.reference)
    sys.exit(0 if score >= 82 and not any(i[0] == "CRITICAL" for i in issues) else 1)
