"""
VocalFusion Fusion Engine v4
============================
AI-driven adaptive mixing — parameters are derived from the actual audio
content, not hardcoded presets.

Stem separation:  Demucs htdemucs_ft (SDR ~8.5) → BS-Roformer (SDR ~13.0)
                  via audio-separator. Night-and-day cleaner vocals.
Spectral carving: Fixed EQ cut → content-aware dynamic EQ. Computes exactly
                  which frequencies the vocal occupies and carves those precise
                  slots in the beat using a Wiener soft-mask.
Vocal cleanup:    noisereduce spectral gating removes residual bleed from the
                  separated vocal stem before any further processing.
Adaptive params:  Noise gate threshold, compressor settings, and sidechain
                  depth are all computed from the actual audio — not presets.
M/S mixing:       Vocal is summed into the Mid channel only; the beat's Sides
                  are preserved untouched, keeping the stereo field intact.

Vocal chain (v4, professional order):
  HPF 80 Hz → Subtractive EQ → De-esser → FET comp → Opto comp →
  NoiseGate → Additive EQ → Saturation → Pre-delay reverb (HPF'd return)

Mastering (v4):
  Mastering EQ → soft clip → glue compressor → LUFS -9 → brick-wall Limiter -1 dBTP
"""

import hashlib
import json
import logging
import os
import sys
import tempfile
import threading
import time
import shutil
from pathlib import Path

import librosa
import librosa.feature.rhythm
import noisereduce as nr
import numpy as np
import pyloudnorm as pyln
import soundfile as sf
from pedalboard import (
    Compressor, HighpassFilter, HighShelfFilter, LowShelfFilter, NoiseGate,
    PeakFilter, Pedalboard, Reverb, Limiter,
    time_stretch as pb_time_stretch,
)
from scipy.interpolate import interp1d
from scipy.ndimage import gaussian_filter1d
from scipy.optimize import linear_sum_assignment
from scipy.signal import butter, sosfilt, fftconvolve

try:
    import pyrubberband as rb
    HAS_PYRUBBERBAND = True
except ImportError:
    HAS_PYRUBBERBAND = False

try:
    from df.enhance import enhance, init_df as _df_init_df
    import torch as _torch
    HAS_DEEPFILTER = True
except Exception:
    HAS_DEEPFILTER = False

SR = 44100
_BS_ROFORMER    = "model_bs_roformer_ep_317_sdr_12.9755.ckpt"
_MDX_VOCAL      = "Kim_Vocal_2.onnx"          # MDX-Net vocal (SDR ~9.5, ONNX fast on CPU)
_MDX23C_VOCAL   = "MDX23C-8KFFT-InstVoc_HQ.ckpt"  # MDX23C vocal (SDR ~12+, better quality)
_DENOISE_MODEL  = "denoise_mel_band_roformer_aufr33_sdr_27.9959.ckpt"          # post-sep denoiser (SDR 27.99)
_DEVERB_MODEL   = "deverb_bs_roformer_8_384dim_10depth.ckpt"                   # BS-Roformer de-reverb


def _check(y: np.ndarray, label: str) -> np.ndarray:
    """Inline signal health check — prints peak/rms and warns on NaN/Inf."""
    has_nan = bool(np.any(np.isnan(y)))
    has_inf = bool(np.any(np.isinf(y)))
    peak = float(np.nanmax(np.abs(y))) if not (has_nan and has_inf) else float('nan')
    rms  = float(np.sqrt(np.nanmean(y ** 2) + 1e-12))
    flags = (" NaN!" if has_nan else "") + (" Inf!" if has_inf else "")
    print(f"      [DBG] {label}: peak={peak:.4f} rms={rms:.5f}{flags}", flush=True)
    if has_nan or has_inf:
        print(f"      [DBG] *** CORRUPTED AT {label} — replacing with zeros ***",
              flush=True)
        y = np.nan_to_num(y, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)
    return y

_KS_MAJOR = np.array([6.35, 2.23, 3.48, 2.33, 4.38, 4.09,
                       2.52, 5.19, 2.39, 3.66, 2.29, 2.88])
_KS_MINOR = np.array([6.33, 2.68, 3.52, 5.38, 2.60, 3.53,
                       2.54, 4.75, 3.98, 2.69, 3.34, 3.17])
_NOTES = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"]


# ── AI Content Analysis ───────────────────────────────────────────────────────

def _analyze_beat_character(beat_mono: np.ndarray, bpm: float) -> dict:
    """
    Derive continuous style scores from the beat's audio content.
    Returns scores 0–1 that drive ALL adaptive parameter decisions.

    Instead of hard genre buckets (trap/pop/hiphop), we compute a feature
    vector and map it continuously to processing parameters.  This avoids
    misclassification edge cases and produces more nuanced adaptation.

    Scores:
      aggressiveness  0=soft/downtempo  1=hard/trap/drill
      bass_weight     0=light bass      1=heavy 808/sub dominant
      brightness      0=warm/dark       1=bright/crispy hi-hats
    """
    clip = beat_mono[:SR * 30]  # first 30 s is enough

    # ── Spectral features ──────────────────────────────────────────────────────
    S = np.abs(librosa.stft(clip, n_fft=2048, hop_length=512))
    freqs = librosa.fft_frequencies(sr=SR)

    def _band_energy(lo, hi):
        m = (freqs >= lo) & (freqs < hi)
        return float(S[m].mean()) if m.any() else 0.0

    total_e  = _band_energy(20, 20000) + 1e-9
    sub_e    = _band_energy(20,  80)          # kick sub / 808
    bass_e   = _band_energy(80,  250)
    hihat_e  = _band_energy(8000, 16000)      # hi-hats / cymbals
    centroid = float(librosa.feature.spectral_centroid(S=S, sr=SR).mean())

    # Zero crossing rate — hi-hats / noise drive ZCR up
    zcr = float(librosa.feature.zero_crossing_rate(clip).mean())

    # ── Continuous scores ──────────────────────────────────────────────────────
    # Aggressiveness: high BPM + high ZCR + high centroid → trap/drill
    bpm_score        = np.clip((bpm - 80) / 80, 0, 1)   # 80 BPM=0, 160 BPM=1
    zcr_score        = np.clip((zcr - 0.05) / 0.25, 0, 1)
    centroid_score   = np.clip((centroid - 1500) / 3000, 0, 1)
    aggressiveness   = float(np.mean([bpm_score, zcr_score, centroid_score]))

    # Bass weight: sub + bass energy relative to total
    bass_weight = float(np.clip((sub_e + bass_e) / total_e * 5, 0, 1))

    # Brightness: hi-hat energy relative to total
    brightness = float(np.clip(hihat_e / total_e * 20, 0, 1))

    return {
        "aggressiveness": round(aggressiveness, 3),
        "bass_weight":    round(bass_weight, 3),
        "brightness":     round(brightness, 3),
        "bpm":            bpm,
        "centroid_hz":    round(centroid, 0),
        "zcr":            round(zcr, 4),
    }


def _analyze_vocal_character(vox_mono: np.ndarray) -> dict:
    """
    Detect vocal delivery style from the separated vocal stem.

    rap_score 0=pure singing  1=pure rap/spoken-word

    Features:
      - ZCR variance:  rap has rapid ZCR changes (percussive syllables)
      - Spectral flatness: rap is more noise-like (higher flatness)
      - Onset rate: rap has more onsets per second
      - Pitch range: singing spans wider semitone range than rap
    """
    clip = vox_mono[:SR * 30]

    zcr        = librosa.feature.zero_crossing_rate(clip)[0]
    zcr_var    = float(np.var(zcr))

    flatness   = float(librosa.feature.spectral_flatness(y=clip).mean())

    onsets     = librosa.onset.onset_detect(y=clip, sr=SR, hop_length=512, units="time")
    onset_rate = len(onsets) / (len(clip) / SR + 1e-9)  # onsets per second

    # Pitch range and gender detection via PYIN fundamental
    median_f0 = 120.0  # default male
    try:
        f0, voiced, _ = librosa.pyin(clip, fmin=60, fmax=1200,
                                     sr=SR, hop_length=512, fill_na=None)
        f0_voiced = f0[voiced] if voiced is not None else np.array([])
        if len(f0_voiced) > 20:
            pitch_range_semitones = float(
                12 * np.log2(f0_voiced.max() / (f0_voiced.min() + 1e-9)))
            median_f0 = float(np.median(f0_voiced))
        else:
            pitch_range_semitones = 5.0
    except Exception:
        pitch_range_semitones = 8.0

    # Normalize each feature → rap score contribution
    zcr_var_score  = np.clip(zcr_var / 0.02, 0, 1)
    flat_score     = np.clip((flatness - 0.01) / 0.06, 0, 1)
    onset_score    = np.clip((onset_rate - 1.0) / 5.0, 0, 1)
    pitch_score    = np.clip(1.0 - (pitch_range_semitones - 3) / 15, 0, 1)

    rap_score = float(np.mean([zcr_var_score, flat_score, onset_score, pitch_score]))

    return {
        "rap_score":    round(rap_score, 3),
        "onset_rate":   round(onset_rate, 2),
        "flatness":     round(float(flatness), 4),
        "pitch_range":  round(pitch_range_semitones, 1),
        # 200 Hz threshold: AutoTune'd male vocals (Future, Travis Scott, etc.) often
    # sit at 160-200 Hz after pitch correction — use 200 Hz to avoid mis-gendering.
    "gender":       "female" if median_f0 >= 200.0 else "male",
        "median_f0":    round(median_f0, 1),
    }


def _style_params(beat_char: dict, vox_char: dict, beat_fp: dict = None) -> dict:
    """
    Map continuous style scores → concrete DSP parameter values.
    All parameters derived from audio content — nothing hardcoded.

    beat_fp: optional beat sonic fingerprint (from _beat_sonic_fingerprint).
             When provided, reverb is matched to the beat's acoustic space and
             the spectral carve range is extended to cover the vocal's actual F0.
    """
    agg  = beat_char["aggressiveness"]   # 0–1
    bass = beat_char["bass_weight"]      # 0–1
    rap  = vox_char["rap_score"]         # 0–1

    # ── Reverb: match beat's acoustic space ───────────────────────────────────
    # Base: very dry (AI stems already have original reverb baked in).
    # Scale: if the beat lives in a roomy space (reverb_tail high), give the
    # vocal more space to match — otherwise it sounds pasted on top.
    base_room = float(np.interp(rap, [0, 1], [0.15, 0.06]))
    base_wet  = float(np.interp(rap, [0, 1], [0.04, 0.02]))
    if beat_fp is not None:
        reverb_tail = float(beat_fp.get("reverb_tail", 0.3))
        # room_scale: 0.7× for punchy/dry beats, 1.4× for roomy/spacious beats
        room_scale  = float(np.interp(reverb_tail, [0.10, 0.60], [0.70, 1.40]))
        reverb_room = float(np.clip(base_room * room_scale, 0.04, 0.35))
        reverb_wet  = float(np.clip(base_wet  * room_scale, 0.01, 0.09))
    else:
        reverb_room = base_room
        reverb_wet  = base_wet

    # ── Spectral carve frequency range ────────────────────────────────────────
    # Default 200-5000 Hz covers most voices. Extend dynamically:
    #   Bass vocalists (F0 < 150 Hz):  carve down to ~80 Hz to cover fundamentals
    #   Sopranos / high falsetto (F0 > 350 Hz or pitch_range > 22 st): up to 9 kHz
    median_f0     = float(vox_char.get("median_f0",  150.0))
    pitch_range_s = float(vox_char.get("pitch_range", 12.0))
    carve_lo_hz   = float(np.clip(median_f0 * 0.65, 80.0, 200.0))
    if median_f0 > 350 or pitch_range_s > 22:
        carve_hi_hz = float(np.clip(median_f0 * 7.0, 5000.0, 9000.0))
    else:
        carve_hi_hz = 5000.0

    return {
        "_rap_score": rap,  # pass-through for ADT and other downstream use
        # FET compressor: rap/trap → faster, harder
        # Ratio cap 4.0:1 (was 6:1 — too aggressive on AI-separated stems, smears dynamics)
        # Research: AI-separated vocal stems need 2:1-3:1; higher ratios crush micro-dynamics
        # Release min 60ms (was 40ms — too fast, caused pumping artifacts)
        # Non-rap floor 2.0:1 (was 3.5:1 → 2.2:1 — over-squashing caused unnatural dynamics)
        "fet_ratio":    float(np.interp(rap, [0, 1], [2.0, 4.0])),
        "fet_attack":   float(np.interp(rap, [0, 1], [5.0, 3.0])),
        "fet_release":  float(np.interp(agg, [0, 1], [120.0, 60.0])),

        # Opto compressor: more gentle always, but faster for aggressive
        "opto_ratio":   float(np.interp(agg, [0, 1], [2.0, 3.0])),
        "opto_attack":  float(np.interp(agg, [0, 1], [30.0, 15.0])),
        "opto_release": float(np.interp(agg, [0, 1], [300.0, 150.0])),

        # Presence boost: 3 kHz is the key vocal intelligibility / "cut-through"
        # frequency. +2-3.5 dB — the +4dB high shelf was removed (caused harshness);
        # vocals cut through via spectral carve, not aggressive presence boost.
        "presence_db":  float(np.interp(rap, [0, 1], [2.0, 3.5])),
        "presence_hz":  3000.0,  # fixed: 3 kHz is the universal cut-through freq

        # Air shelf: compensate for HF stripped by Demucs mask + de-esser.
        "air_db":       float(np.interp(rap, [0, 1], [2.5, 2.0])),

        # Reverb: matched to beat's acoustic space (see computation above)
        "reverb_room":  reverb_room,
        "reverb_damp":  float(np.interp(rap, [0, 1], [0.70, 0.90])),
        "reverb_wet":   reverb_wet,

        # Spectral carve: more bass-heavy → carve deeper. 8-10 dB for house/EDM.
        "carve_db":     float(np.interp(bass, [0, 1], [6.0, 10.0])),
        # Carve frequency range: content-adaptive (see computation above)
        "carve_lo_hz":  carve_lo_hz,
        "carve_hi_hz":  carve_hi_hz,

        # Sidechain: aggressive beat → more sidechain duck
        "sidechain_mult": float(np.interp(agg, [0, 1], [0.9, 1.2])),

        # Vocal level: vocals need to dominate the mid-frequency zone clearly.
        # 2.0 = vocal is 6 dB louder than beat in presence zone.
        # 3.0 = vocal is 9.5 dB louder — needed for rap over heavy EDM beats.
        "vocal_level":  float(np.interp(rap, [0, 1], [2.0, 3.0])),

        # Complementary EQ: cut instrumental at vocal fundamental zone.
        # Research: male F0 body 200-350 Hz → cut at 280 Hz; female → 380 Hz.
        # This is a fixed reciprocal cut complementing the dynamic Wiener carve.
        "comp_eq_hz":   380.0 if vox_char.get("gender") == "female" else 280.0,
    }


def _beat_sonic_fingerprint(inst_mono: np.ndarray, bpm: float) -> dict:
    """
    Extract the beat's sonic DNA — the characteristics that make it sound the way
    it does.  Used to re-produce the vocal so both elements feel native to the
    same session rather than transplanted from different worlds.

    Returns a dict of 0-1 scores:
      saturation     — 0=clean/digital, 1=warm/driven/saturated
      brightness     — 0=dark/sub-heavy, 1=bright/crispy/airy
      reverb_tail    — 0=punchy/dry, 1=spacious/wet/roomy
      dynamic_feel   — 0=crushed/heavily compressed, 1=open/dynamic
      texture        — 0=smooth/polished, 1=gritty/distorted/rough
      transient_punch— 0=soft/smooth attacks, 1=sharp/hard-hitting kicks
    """
    clip = inst_mono[:SR * 30].astype(np.float32)
    hop  = 512
    n_fft = 2048

    S     = np.abs(librosa.stft(clip, n_fft=n_fft, hop_length=hop))
    freqs = librosa.fft_frequencies(sr=SR, n_fft=n_fft)

    def band_e(lo, hi):
        m = (freqs >= lo) & (freqs < hi)
        return float(S[m].mean()) if m.any() else 1e-9

    # Saturation: ratio of harmonic zone (600-2kHz) to fundamental zone (150-600Hz)
    # Saturated/driven signal generates extra harmonic content above the fundamental.
    sat_raw = band_e(600, 2000) / (band_e(150, 600) + 1e-9)
    saturation = float(np.clip((sat_raw - 0.4) / 1.6, 0, 1))

    # Brightness: spectral slope from 200 Hz to 6 kHz
    oct_bands = [(200,400),(400,800),(800,1600),(1600,3200),(3200,6400)]
    oct_db    = [20*np.log10(band_e(lo, hi)+1e-9) for lo, hi in oct_bands]
    try:
        slope = float(np.polyfit(range(len(oct_db)), oct_db, 1)[0])
    except Exception:
        slope = -5.0
    brightness = float(np.clip((slope + 10.0) / 10.0, 0, 1))  # -10→0, 0→1

    # Reverb tail: ratio of onset envelope valley (tail) to peak (transient)
    onset_env = librosa.onset.onset_strength(y=clip, sr=SR, hop_length=hop)
    p75 = float(np.percentile(onset_env, 75)) + 1e-9
    p25 = float(np.percentile(onset_env, 25))
    reverb_tail = float(np.clip(p25 / p75, 0, 1))

    # Dynamic feel: RMS envelope peak-to-floor ratio
    # Low ratio (≤2) = heavily compressed/flat, high (≥8) = very dynamic
    rms_f = librosa.feature.rms(y=clip, frame_length=n_fft, hop_length=hop)[0]
    rms_ratio = float(np.percentile(rms_f, 90) / (np.percentile(rms_f, 10) + 1e-9))
    dynamic_feel = float(np.clip((rms_ratio - 1.5) / 6.0, 0, 1))

    # Texture: spectral flatness — 0=tonal/smooth, 1=noisy/gritty
    flatness = float(librosa.feature.spectral_flatness(y=clip).mean())
    texture  = float(np.clip(flatness / 0.08, 0, 1))

    # Transient punch: onset peak vs mean — sharp peaks = punchy kick energy
    if len(onset_env) > 10:
        tp = float(np.percentile(onset_env, 95) / (onset_env.mean() + 1e-9))
        transient_punch = float(np.clip((tp - 1.0) / 6.0, 0, 1))
    else:
        transient_punch = 0.5

    return {
        "saturation":      round(saturation, 3),
        "brightness":      round(brightness, 3),
        "reverb_tail":     round(reverb_tail, 3),
        "dynamic_feel":    round(dynamic_feel, 3),
        "texture":         round(texture, 3),
        "transient_punch": round(transient_punch, 3),
        "_bpm":            bpm,
    }


def _estimate_room_ir(inst_mono: np.ndarray, sr: int, n_taps: int = 8192) -> np.ndarray:
    """
    Extract a frequency-dependent room impulse response from the instrumental stem
    using multi-band transient-tail analysis and Schroeder backward integration.

    Returns a mono IR array (float32, length n_taps) normalised to unit peak.
    Falls back to a synthetic frequency-dependent IR if fewer than 3 valid tails
    are found in any band.
    """
    clip = inst_mono[: sr * 30].astype(np.float64)
    hop  = 512

    # ── Band-pass filters (defined outside any try-block — Python scope rule) ──
    def _bp(lo, hi, order=4):
        sos = butter(order, [lo / (sr / 2.0), hi / (sr / 2.0)],
                     btype="band", output="sos")
        return sos

    sos_kick = _bp(60,   300)
    sos_snare= _bp(300, 3000)
    sos_hat  = _bp(3000,12000)
    sos_sub  = _bp(20,   150)           # for contamination check
    sos_hpf  = butter(4, 180.0 / (sr / 2.0), btype="high", output="sos")
    sos_lpf8k= butter(4, 8000.0 / (sr / 2.0), btype="low", output="sos")

    bands = [
        ("kick",  sos_kick,  60,   300,  0.060),
        ("snare", sos_snare, 300, 3000,  0.020),
        ("hat",   sos_hat,  3000,12000,  0.010),
    ]

    early_ms  = int(0.080 * sr)   # 80 ms in samples
    tail_ms   = n_taps             # max tail length = IR length

    all_tails       = []           # (weight, tail_array) tuples for decay fitting
    early_patterns  = []           # early reflection excerpts

    for band_name, sos_band, _, _, _, in bands:
        filt = sosfilt(sos_band, clip)

        # Onset strength in this band
        onset_env = librosa.onset.onset_strength(
            y=filt.astype(np.float32), sr=sr, hop_length=hop,
            aggregate=np.median,
        )
        # Peak picking — top 20 transients
        peak_idx = np.argsort(onset_env)[-20:]
        peak_samps = (peak_idx * hop).astype(int)

        for ps in peak_samps:
            if ps + tail_ms >= len(filt):
                continue
            tail = filt[ps: ps + tail_ms].copy()

            # ── Contamination rejection: kick fundamental bleed ──────────────
            # If a dominant spectral peak below 150 Hz is > 12 dB above broadband
            # floor, this tail is likely a kick ring — skip it.
            sub_filt  = sosfilt(sos_sub, tail)
            sub_rms   = float(np.sqrt(np.mean(sub_filt ** 2)) + 1e-12)
            broad_rms = float(np.sqrt(np.mean(tail ** 2)) + 1e-12)
            sub_db_above = 20.0 * np.log10(sub_rms / broad_rms + 1e-12)
            if sub_db_above > -12.0:          # sub is <12 dB below broadband
                continue                       # kick ring — reject

            # ── Early reflections (first 80 ms) ─────────────────────────────
            early = tail[:early_ms].copy()
            # Normalise by peak so averaging across transients is meaningful
            epk = float(np.max(np.abs(early)) + 1e-12)
            early_patterns.append(early / epk)

            # ── Decay tail (80 ms onward) — weight by peak amplitude ─────────
            decay = tail[early_ms:].copy()
            weight = float(np.max(np.abs(tail[:early_ms]) + 1e-12))
            if np.max(np.abs(decay)) > 1e-9:
                all_tails.append((weight, decay))

    # ── Frequency-dependent RT60 via Schroeder integration ───────────────────
    # Measure at 250 Hz, 1 kHz, 4 kHz, 8 kHz
    rt60_bands = [
        (200,  350,  butter(4, [200.0/(sr/2), 350.0/(sr/2)],  btype="band", output="sos")),
        (800, 1250,  butter(4, [800.0/(sr/2), 1250.0/(sr/2)], btype="band", output="sos")),
        (3150,5000,  butter(4, [3150.0/(sr/2), 5000.0/(sr/2)],btype="band", output="sos")),
        (6300,9000,  butter(4, [6300.0/(sr/2), 9000.0/(sr/2)],btype="band", output="sos")),
    ]

    rt60_values = []   # list of (centre_freq_hz, rt60_s)
    for lo, hi, sos_rt in rt60_bands:
        filt_b = sosfilt(sos_rt, clip)
        # Energy decay curve (Schroeder backward integration)
        sq = filt_b ** 2
        edc = np.cumsum(sq[::-1])[::-1]
        edc_db = 10.0 * np.log10(edc / (edc[0] + 1e-30) + 1e-30)
        # Find -60 dB point (or estimate by fitting -5 to -25 dB slope)
        try:
            idx5  = int(np.argmax(edc_db <= -5.0))
            idx25 = int(np.argmax(edc_db <= -25.0))
            if idx25 > idx5 + 10:
                slope = (edc_db[idx25] - edc_db[idx5]) / ((idx25 - idx5) / float(sr))
                rt60  = -60.0 / slope if slope < -0.01 else 0.4
            else:
                rt60 = 0.4
        except Exception:
            rt60 = 0.4
        rt60 = float(np.clip(rt60, 0.05, 3.0))
        rt60_values.append(((lo + hi) / 2.0, rt60))

    # ── Build composite IR ────────────────────────────────────────────────────
    ir = np.zeros(n_taps, dtype=np.float64)

    # Early reflections (0–80 ms)
    if len(early_patterns) >= 3:
        # Align to same length then average
        min_ep = min(len(e) for e in early_patterns)
        early_avg = np.mean(np.stack([e[:min_ep] for e in early_patterns], axis=0), axis=0)
        ir[:min_ep] = early_avg
    else:
        # Synthetic: single direct + two early reflections
        ir[0] = 1.0
        r1 = min(int(0.012 * sr), n_taps - 1)
        r2 = min(int(0.028 * sr), n_taps - 1)
        ir[r1] += 0.35
        ir[r2] += 0.20

    # Late reverberation (80 ms onward) — frequency-dependent decay
    # Use per-octave-band RT60 to build separate decay envelopes, then sum
    t_late = np.arange(n_taps - early_ms) / float(sr)   # time axis for tail
    if len(rt60_values) >= 2:
        # Interpolate RT60 across frequency, then for each sample index build
        # a weighted sum of per-band decays (approximates freq-dep late field).
        freqs_rt  = np.array([v[0] for v in rt60_values])
        rt60s     = np.array([v[1] for v in rt60_values])
        # Frequency-weighted decay: sum of band-limited exponentials
        late_env = np.zeros_like(t_late)
        weights  = [1.0, 1.2, 0.9, 0.7]   # rough energy weights per band
        for i, (fc, rt60_i) in enumerate(rt60_values):
            decay_rate = np.log(1e-3) / rt60_i   # -60 dB in rt60 seconds
            late_env  += weights[i] * np.exp(decay_rate * t_late)
        late_env /= float(np.max(np.abs(late_env)) + 1e-12)

        # Scale to blend smoothly from early reflections at 80 ms
        ep_val = float(np.max(np.abs(ir[:early_ms])) + 1e-12) * 0.5
        ir[early_ms:] = late_env * ep_val
    elif len(all_tails) >= 3:
        # Enough extracted tails — weighted-average the measured decay
        max_tail = min(n_taps - early_ms, min(len(t[1]) for t in all_tails))
        weighted_sum = np.zeros(max_tail, dtype=np.float64)
        weight_total = 0.0
        for w, tail in all_tails:
            weighted_sum += w * tail[:max_tail]
            weight_total += w
        if weight_total > 0:
            avg_tail = weighted_sum / weight_total
            ep_val = float(np.max(np.abs(ir[:early_ms])) + 1e-12) * 0.5
            norm_t  = float(np.max(np.abs(avg_tail)) + 1e-12)
            ir[early_ms: early_ms + max_tail] = avg_tail / norm_t * ep_val
    else:
        # Full synthetic fallback — single exponential, RT60=0.35 s
        rt60_fb = 0.35
        decay_rate_fb = np.log(1e-3) / rt60_fb
        t_all = np.arange(n_taps) / float(sr)
        ir = np.exp(decay_rate_fb * t_all)
        ir[0] = 1.0

    # ── Post-processing ───────────────────────────────────────────────────────
    # HPF at 180 Hz (sub-bass reverb on vocals = mud)
    ir = sosfilt(sos_hpf, ir)
    # Gentle HF roll-off above 8 kHz (rooms don't reflect ultra-high freqs)
    ir = sosfilt(sos_lpf8k, ir)
    # Normalise to unit peak
    pk = float(np.max(np.abs(ir)) + 1e-12)
    ir = (ir / pk).astype(np.float32)
    return ir


def _apply_space_match(
    vox: np.ndarray,
    ir_estimated: np.ndarray,
    reverb_tail_score: float,
    sr: int,
) -> np.ndarray:
    """
    Convolve the vocal with the estimated room IR to place it in the same
    acoustic space as the instrumental.  Uses psychoacoustic wet scaling so
    a dry punchy beat gets less space than a roomy ambient one.
    """
    wet = float(np.interp(reverb_tail_score, [0.0, 1.0], [0.025, 0.065]))

    # Wet return processing filters (outside try-block — Python scope rule)
    sos_hpf_wet = butter(4, 450.0 / (sr / 2.0), btype="high", output="sos")

    nyq = sr / 2.0
    # -1.5 dB high shelf at 6 kHz
    sos_hshelf = butter(2, 6000.0 / nyq, btype="low", output="sos")   # gentle LP proxy
    # -2 dB peak at 500 Hz Q=1.5 — approximated as a narrow low-shelf notch
    sos_notch  = butter(2, [350.0 / nyq, 700.0 / nyq], btype="band", output="sos")

    ir = ir_estimated.astype(np.float64)
    n_out = vox.shape[0]

    if vox.ndim == 1:
        vox = vox[:, np.newaxis]
    n_ch = vox.shape[1]

    wet_channels = []
    for c in range(n_ch):
        ch = vox[:, c].astype(np.float64)
        # Per-channel convolution with mono IR (same IR on L and R)
        conv = fftconvolve(ch, ir, mode="full")[:n_out]
        # HPF wet return at 450 Hz
        conv = sosfilt(sos_hpf_wet, conv)
        # Gentle high-shelf cut at 6 kHz (-1.5 dB) — low-pass proxy blend
        conv_lp   = sosfilt(sos_hshelf, conv)
        shelf_gain = 10.0 ** (-1.5 / 20.0)
        conv = conv * shelf_gain + conv_lp * (1.0 - shelf_gain)
        # -2 dB notch at 500 Hz (removes boxiness)
        conv_band = sosfilt(sos_notch, conv)
        notch_gain = 10.0 ** (-2.0 / 20.0)
        conv = conv - conv_band * (1.0 - notch_gain)
        wet_channels.append(conv)

    wet_arr  = np.stack(wet_channels, axis=1).astype(np.float32)
    dry_arr  = vox[:n_out].astype(np.float32)
    result   = np.clip(dry_arr * (1.0 - wet) + wet_arr * wet, -1.0, 1.0)
    result   = np.nan_to_num(result, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)
    print(f"      [Space Match] Applied room IR (wet={wet:.3f})", flush=True)
    return result


def _produce_vocal_for_beat(
    vox: np.ndarray,
    fp: dict,
    bpm: float,
    ir_estimated: np.ndarray = None,
) -> np.ndarray:
    """
    Re-produce the vocal to sound like it belongs in the beat's sonic universe.

    A producer listening to a beat before recording vocals would instinctively
    choose:  the right amount of saturation, the right room/reverb character,
    the right tonal brightness, and BPM-synced effects for cohesion.
    We do this analytically from the beat fingerprint (fp).

    Stages:
      1. Character saturation  — match the beat's warmth/harmonic richness
      2. Tonal integration     — spectral tilt so vocal complements (not clashes with) beat
      3. Beat-matched reverb   — room size and pre-delay tuned to beat's space
      4. BPM-synced delay      — 1/8-note echo locks vocal timing to beat grid
      5. Dynamic feel matching — if beat is squashed, vocal gets matching glue comp
      6. Acoustic space match  — convolve with estimated room IR (freq-dependent)
    """
    result = vox.copy().astype(np.float32)

    # ── 1. Character saturation ───────────────────────────────────────────────
    # Parallel tanh drive: warm if beat is saturated/driven, clean if digital.
    # Applied at 10-18% wet — enough to match character, never distort the vocal.
    sat_wet = float(np.interp(fp["saturation"], [0.2, 0.8], [0.0, 0.18]))
    if sat_wet > 0.02:
        drive   = float(np.interp(fp["saturation"], [0.2, 0.8], [0.8, 1.8]))
        vox_M   = ((result[:, 0] + result[:, 1]) / 2.0).astype(np.float64)
        sat_sig = (np.tanh(vox_M * drive) / (drive + 1e-9)).astype(np.float32)
        sat_st  = np.stack([sat_sig, sat_sig], axis=1)
        result  = (result * (1.0 - sat_wet) + sat_st * sat_wet).astype(np.float32)

    # ── 2. Tonal integration ──────────────────────────────────────────────────
    # Dark beat → vocal needs extra presence (+2 dB shelf at 3 kHz) to cut through.
    # Bright beat → vocal needs warmth (+1.5 dB shelf at 250 Hz) to not sound thin.
    dark_presence_db  = float(np.interp(fp["brightness"], [0.2, 0.7], [2.0, 0.0]))
    bright_warmth_db  = float(np.interp(fp["brightness"], [0.3, 0.8], [0.0, 1.5]))
    tonal_eq = Pedalboard([
        LowShelfFilter( cutoff_frequency_hz=250.0,  gain_db=bright_warmth_db),
        HighShelfFilter(cutoff_frequency_hz=3000.0, gain_db=dark_presence_db),
    ])
    result = tonal_eq(result.T.astype(np.float32), SR).T.astype(np.float32)

    # ── 3. Beat-matched reverb ────────────────────────────────────────────────
    # Room size and damping derived from the beat's own reverb character.
    # Pre-delay synced to a 16th note at the beat's BPM — rhythmically cohesive.
    # Damping: bright beat → less damped (airy tail), dark → more damped (murky).
    room   = float(np.interp(fp["reverb_tail"],  [0.0, 1.0], [0.06, 0.28]))
    damp   = float(np.interp(fp["brightness"],   [0.0, 1.0], [0.85, 0.40]))
    wet    = float(np.interp(fp["reverb_tail"],  [0.0, 1.0], [0.02, 0.09]))
    rev_board = Pedalboard([Reverb(room_size=room, damping=damp,
                                   wet_level=wet, dry_level=1.0)])
    predelay_samps = int(60000.0 / bpm / 4.0 * SR / 1000)  # 16th-note pre-delay
    # Pre-delay: shift the signal before feeding into reverb, then recombine
    vox_shifted = np.zeros_like(result)
    if 0 < predelay_samps < len(result):
        vox_shifted[predelay_samps:] = result[:-predelay_samps]
    rev_out = rev_board(vox_shifted.T.astype(np.float32), SR).T.astype(np.float32)
    result  = np.clip(result + rev_out * 0.6, -1.0, 1.0).astype(np.float32)

    # ── 4. BPM-synced 1/8-note delay ─────────────────────────────────────────
    # Subtle slapback at the 8th-note interval locks the vocal to the beat grid.
    # Producers always sync delays to tempo — it's what makes a vocal feel "in"
    # the track rather than floating over it.  6% wet = subtle glue, not echo.
    delay_samps = int(60000.0 / bpm / 2.0 * SR / 1000)   # 8th note
    if 0 < delay_samps < len(result) // 2:
        delay_sig          = np.zeros_like(result)
        delay_sig[delay_samps:] = result[:-delay_samps] * 0.45  # -7dB echo
        result = np.clip(result + delay_sig * 0.06, -1.0, 1.0).astype(np.float32)

    # ── 5. Dynamic feel matching ──────────────────────────────────────────────
    # Heavily compressed beat (dynamic_feel < 0.3) → light glue comp on vocal.
    # Both elements then breathe and pump together instead of fighting each other.
    if fp["dynamic_feel"] < 0.35:
        glue_ratio = float(np.interp(fp["dynamic_feel"], [0.0, 0.35], [2.8, 1.5]))
        glue = Pedalboard([Compressor(threshold_db=-16.0, ratio=glue_ratio,
                                      attack_ms=25.0, release_ms=180.0)])
        result = glue(result.T.astype(np.float32), SR).T.astype(np.float32)

    # ── 6. Acoustic space matching ────────────────────────────────────────────
    # After the Pedalboard reverb (which handles pre-delay and room size character),
    # apply the frequency-dependent room IR extracted from the actual instrumental.
    # This places the vocal in the same physical acoustic space as the beat — the
    # #1 reason mashups sound "pasted on" vs recorded together.
    if ir_estimated is not None and len(ir_estimated) > 100:
        result = _apply_space_match(result, ir_estimated, fp["reverb_tail"], SR)

    return np.nan_to_num(result, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)


def _smart_key_shift(n_semi: int, key_b_root: int, key_b_mode: str,
                     key_a_root: int, key_a_mode: str) -> tuple:
    """
    If the direct semitone shift is large (>3), try alternate harmonic
    relationships that might be more compatible:
      - Try parallel mode: if B is minor, try its relative major (+3 semitones)
      - Try octave-equivalent: n_semi - 12 or n_semi + 12
    Returns (best_n_semi, explanation).
    """
    if abs(n_semi) <= 3:
        return n_semi, "compatible"

    candidates = [n_semi]

    # Relative major/minor: same key signature, different root
    if key_b_mode == "minor":
        rel = semitones_to_shift(key_b_root, "major", key_a_root, key_a_mode)
        candidates.append(rel)
    else:
        rel = semitones_to_shift(key_b_root, "minor", key_a_root, key_a_mode)
        candidates.append(rel)

    # Octave-wrapped alternatives
    for c in list(candidates):
        if c > 6:
            candidates.append(c - 12)
        elif c < -6:
            candidates.append(c + 12)

    # Try all 4 adjacent Camelot wheel keys (±1 position = ±2 semitones, ±5 semitones)
    # These are harmonically compatible and require smaller shifts
    for adj in (2, -2, 5, -5, 7, -7):
        candidates.append(n_semi + adj - round((n_semi + adj) / 12) * 12)

    # Pick the smallest absolute shift
    best = min(candidates, key=abs)
    if abs(best) > 5:
        msg = (f"re-mapped {n_semi:+d} → {best:+d} st "
               f"[WARNING: {abs(best)} semitones — quality may suffer]")
    elif best != n_semi:
        msg = f"re-mapped {n_semi:+d} → {best:+d} semitones (better harmonic fit)"
    else:
        msg = f"{n_semi:+d} semitones (compatible)"
    return best, msg


def _deepfilter_clean(vox_mono: np.ndarray) -> np.ndarray:
    """
    Clean vocal stem using DeepFilterNet (neural noise suppression) if available.
    Handles musical noise and bleed artifacts far better than spectral gating.
    Falls back to noisereduce if DeepFilterNet is not installed.
    """
    if not HAS_DEEPFILTER:
        return _clean_vocal(vox_mono)

    try:
        import soxr
        model, df_state, _ = _df_init_df()
        # DeepFilterNet expects 48 kHz
        vox_48k = soxr.resample(vox_mono.astype(np.float32), SR, 48000).astype(np.float32)
        t = _torch.from_numpy(vox_48k).unsqueeze(0)
        enhanced = enhance(model, df_state, t).squeeze(0).numpy()
        return soxr.resample(enhanced, 48000, SR).astype(np.float32)
    except Exception as e:
        print(f"      [DeepFilter failed ({e}), using noisereduce]", flush=True)
        return _clean_vocal(vox_mono)


def _energy_match_envelope(inst: np.ndarray, vox: np.ndarray,
                            target_ratio: float = 1.2,
                            window_s: float = 2.0) -> np.ndarray:
    """
    Dynamic level matching: scale the vocal in overlapping windows so that
    locally vox_rms ≈ target_ratio × inst_rms.

    This creates natural breathing — the vocal follows the beat's energy
    envelope rather than sitting at a static level.  Loud drop sections
    get a louder vocal; breakdown sections let the beat breathe.

    Gain is smoothed (σ=3 frames) to prevent audible pumping.
    """
    win = int(SR * window_s)
    hop = win // 4
    n   = min(len(inst), len(vox))

    inst_mono = _to_mono(inst[:n])
    vox_mono  = _to_mono(vox[:n])
    vox_out   = vox[:n].copy()

    n_frames = max(1, (n + hop - 1) // hop)
    gains = np.ones(n_frames, dtype=np.float32)

    for i in range(n_frames):
        s, e = i * hop, min(i * hop + win, n)
        ir = _rms(inst_mono[s:e])
        vr = _rms(vox_mono[s:e])
        if vr > 1e-9 and ir > 1e-9:
            # Cap at 1.5× max gain: prevents vocal from overshooting beat in
            # loud sections. The static scalar already set the global level;
            # this is just fine-tuning per-window, not dramatic amplification.
            gains[i] = np.clip((ir * target_ratio) / vr, 0.5, 1.5)

    gains = gaussian_filter1d(gains.astype(np.float64), sigma=3.0).astype(np.float32)

    x_f = np.arange(n_frames, dtype=np.float64) * hop
    x_s = np.arange(n, dtype=np.float64)
    gain_samp = interp1d(
        x_f, gains, kind="linear",
        bounds_error=False, fill_value=(gains[0], gains[-1])
    )(x_s).astype(np.float32)

    return (vox_out * gain_samp[:, np.newaxis]).astype(np.float32)


def _iterative_mix(inst: np.ndarray, vox: np.ndarray,
                   style: dict, sidechain_depth: float,
                   bpm_a: float, max_iter: int = 3) -> np.ndarray:
    """
    Closed-loop mixer: produce a mix, evaluate vocal presence,
    adjust level multiplier, repeat until target is met.

    Eliminates the manual trial-and-error of finding the right vocal level.
    Target vocal presence: 40–65% of combined stem energy.
    """
    level_mult = style["vocal_level"]
    carve_db   = style["carve_db"]

    # ── Pre-mix bass management ────────────────────────────────────────────────
    # House/EDM beats have +20-25 dB of bass vs mids.
    # Two-band approach: harder sub cut (<80 Hz) + medium bass cut (80-350 Hz)
    # Target: bring bass vs mid from +17 to +15, lo-mid vs mid from +9 to +8.
    nyq_m = SR / 2.0
    sos_bass_cut = butter(4, 350.0 / nyq_m, btype="low", output="sos")
    bass_band = sosfilt(sos_bass_cut, inst, axis=0).astype(np.float32)
    inst = (inst - bass_band * (1.0 - 10**(-9.0/20.0))).astype(np.float32)  # -9 dB on <350 Hz

    for iteration in range(max_iter):
        # Apply energy-envelope matching then static scalar
        # Presence check uses MID-FREQUENCY RMS (500-5000 Hz) — not full-band.
        # Full-band RMS on a bass-heavy beat inflates beat's apparent loudness;
        # the vocal looks "50% present" while being completely masked in presence zone.
        def _rms_mid(y_mono):
            nyq_r = SR / 2.0
            sos_hp = butter(4, 500.0 / nyq_r, btype="high", output="sos")
            sos_lp = butter(4, 5000.0 / nyq_r, btype="low",  output="sos")
            band = sosfilt(sos_hp, sosfilt(sos_lp, y_mono, axis=0))
            return float(np.sqrt(np.mean(band**2) + 1e-12))

        ir = _rms_mid(_to_mono(inst))
        vr = _rms_mid(_to_mono(vox))
        vox_scaled = (vox * (ir * level_mult / (vr + 1e-9))).astype(np.float32)

        # Note: _energy_match_envelope is intentionally disabled. For EDM/house beats
        # with flat energy envelopes, local tracking kills vocal LRA (was 2.5 LU).
        # The static mid-RMS scalar above handles presence; iterative loop handles targeting.

        # Process instrumental — pass content-adaptive carve range from style
        inst_c = _adaptive_spectral_carve(
            inst, vox_scaled,
            carve_db=carve_db,
            carve_lo_hz=style.get("carve_lo_hz", 200.0),
            carve_hi_hz=style.get("carve_hi_hz", 5000.0),
        )
        inst_c = _check(inst_c, f"iter{iteration+1}/spectral-carve")
        # Complementary EQ: two cuts carve a clear vocal pocket in the beat.
        # 1. Fundamental zone cut (gender-adaptive): clears body/low-mid masking
        # 2. Presence zone cut at 1200 Hz (Q=0.8): clears vocal intelligibility zone.
        #    At high volume, Fletcher-Munson makes 1-2kHz beat energy mask the vocal
        #    harder. -2.5 dB here is barely audible on its own but makes a clear gap
        #    for the vocal to sit in. This is the #1 fix for "vocals unclear loud".
        _comp_eq = Pedalboard([
            PeakFilter(cutoff_frequency_hz=style["comp_eq_hz"], gain_db=-3.0, q=1.2),
            PeakFilter(cutoff_frequency_hz=1200.0, gain_db=-4.0, q=0.8),
            PeakFilter(cutoff_frequency_hz=2500.0, gain_db=-2.0, q=1.0),  # upper-mid cut on beat
        ])
        inst_c = _comp_eq(inst_c.T.astype(np.float32), SR).T.astype(np.float32)

        # Vocal-activated bass duck: when the vocal is present, drop the beat's
        # 20-350 Hz by up to -8 dB. This is the professional mashup technique —
        # bass drops under the vocal so both can be heard clearly.
        vox_env = np.abs(_to_mono(vox_scaled))
        hop_d = 512
        env_frames = librosa.feature.rms(y=vox_env, frame_length=2048, hop_length=hop_d)[0]
        env_thresh = float(np.percentile(env_frames, 80))  # fires on top 20% of frames (true peaks only)
        # Smooth the gain curve (100ms attack, 200ms release)
        duck_gain = np.zeros(len(env_frames), dtype=np.float32)
        for fi in range(len(env_frames)):
            if env_frames[fi] > env_thresh:
                duck_gain[fi] = 10**(-4.0/20.0)  # -4 dB when vocal is at peak (was -8 dB — too aggressive)
            else:
                duck_gain[fi] = 1.0
        # Smooth: simple one-pole IIR
        a_att = np.exp(-1.0 / (SR * 0.10 / hop_d))
        a_rel = np.exp(-1.0 / (SR * 0.20 / hop_d))
        smooth = np.ones(len(duck_gain), dtype=np.float32)
        for fi in range(1, len(duck_gain)):
            a = a_att if duck_gain[fi] < smooth[fi-1] else a_rel
            smooth[fi] = a * smooth[fi-1] + (1.0 - a) * duck_gain[fi]
        # Interpolate to sample resolution
        n_samp = inst_c.shape[0]
        x_f = np.arange(len(smooth)) * hop_d
        x_s = np.arange(n_samp)
        gain_samp = np.interp(x_s, x_f, smooth).astype(np.float32)
        # Apply only to bass band
        sos_bass_d = butter(4, 350.0 / nyq_m, btype="low",  output="sos")
        sos_rest_d = butter(4, 350.0 / nyq_m, btype="high", output="sos")
        bass_d  = sosfilt(sos_bass_d, inst_c, axis=0).astype(np.float32)
        rest_d  = sosfilt(sos_rest_d, inst_c, axis=0).astype(np.float32)
        inst_c = (bass_d * gain_samp[:, np.newaxis] + rest_d).astype(np.float32)

        # Transient shaper: sustain reduction only (no attack boost).
        # attack_gain_db=0: the function was silently disabled (btype crash) throughout
        # all v17 testing. Setting to 0 matches v17 effective behavior and stops
        # hi-hat/cymbal transient boosting that was pushing High band +9.6 dB over ref.
        inst_c = _transient_shape(inst_c, attack_gain_db=0.0, sustain_gain_db=-2.0)
        inst_c = _check(inst_c, f"iter{iteration+1}/transient-shape")
        # Sub-bass management: kick transient sidechains 20-80Hz sub-bass
        inst_c = _kick_sub_sidechain(inst_c, depth=0.20)  # reduced 0.35→0.20: was causing -2dB bass deficit
        inst_c = _parallel_compress(inst_c)
        # Style-adaptive sidechain window: rap syllables are faster, need tighter tracking
        # Release stays constant (100ms) to prevent pumping between phrases
        sc_window_ms = int(np.interp(style.get("_rap_score", 0.5), [0, 1], [40, 15]))
        inst_c = _sidechain(inst_c, vox_scaled,
                            depth=sidechain_depth * style["sidechain_mult"],
                            window_ms=sc_window_ms,
                            attack_ms=10.0, release_ms=100.0)

        # Evaluate presence using mid-frequency RMS (500-5000 Hz).
        # CRITICAL: use original `inst` (pre-carve), NOT `inst_c` (post-carve).
        # Using inst_c causes a self-defeating feedback loop: the carve reduces
        # inst_c's mid energy → presence looks artificially high → loop REDUCES
        # vocal level_mult → vocal ends up quieter than intended.
        # Evaluating against original inst gives an accurate picture of how loud
        # the vocal is relative to the unprocessed beat before any carve benefit.
        vp = _rms_mid(_to_mono(vox_scaled)) / (
             _rms_mid(_to_mono(inst)) + _rms_mid(_to_mono(vox_scaled)) + 1e-9)

        print(f"      Mix iter {iteration+1}: presence={vp:.0%}  "
              f"level_mult={level_mult:.2f}  carve={carve_db:.1f}dB", flush=True)

        if 0.45 <= vp <= 0.70 or iteration == max_iter - 1:
            break
        elif vp < 0.45:
            level_mult = min(level_mult * 1.20, 4.0)
        else:
            level_mult = max(level_mult * 0.88, 1.0)  # floor at 1.0 — never go below starting point

    # Mono-safe low end: collapse Side channel below 150 Hz for mono compatibility.
    # Research: hip-hop professional standard is correlation >0.90 below 150 Hz.
    # Wide kicks (Side energy 0.7-0.85 correlation) cause 2+ dB cancellation on mono sum.
    inst_c = _mono_lf(inst_c, cutoff_hz=150.0)

    # Final M/S mix with dynamic stereo width
    # Research: during vocal sections, narrow the beat's stereo field slightly
    # so the vocal (always mono/Mid) has room to command attention.
    # During instrumental breaks: full width (vocal env = low → width_gain ≈ 1.0)
    # During vocal phrases:    80% width (vocal env = high → width_gain ≈ 0.80)
    # 80ms window + slow attack/release = smooth, inaudible narrowing transitions.
    vox_mono_ref = _to_mono(vox_scaled)
    width_gain = _sidechain_envelope(vox_mono_ref, len(inst_c), depth=0.20,
                                     window_ms=80, attack_ms=30.0, release_ms=300.0)
    inst_M, inst_S = _ms_encode(inst_c)
    vox_M, _       = _ms_encode(vox_scaled)
    inst_S_dynamic = (inst_S * width_gain).astype(np.float32)
    mix = _ms_decode(inst_M + vox_M, inst_S_dynamic)
    return mix


# ── Helpers ───────────────────────────────────────────────────────────────────

def _file_id(path: str) -> str:
    stat = os.stat(path)
    with open(path, "rb") as f:
        head = f.read(8192)
    return hashlib.md5(str(stat.st_size).encode() + head).hexdigest()[:12]


# ── System 5: Song Fingerprint Database ───────────────────────────────────────
# Cache per-song analysis (BPM, key, genre, stem quality) by SHA256 fingerprint.
# Means analysis only runs once per song, ever — even across reboots/sessions.

def _fp_path(fid: str, cache_dir: str) -> Path:
    return Path(cache_dir) / fid / "_fingerprint.json"

def _load_fp(fid: str, cache_dir: str) -> dict:
    p = _fp_path(fid, cache_dir)
    if p.exists():
        try:
            return json.loads(p.read_text())
        except Exception:
            return {}
    return {}

def _save_fp(fid: str, cache_dir: str, data: dict) -> None:
    p = _fp_path(fid, cache_dir)
    p.parent.mkdir(parents=True, exist_ok=True)
    existing = _load_fp(fid, cache_dir)
    existing.update({k: (float(v) if isinstance(v, (np.floating, np.integer)) else v)
                     for k, v in data.items()})
    p.write_text(json.dumps(existing, indent=2))


def _to_mono(y: np.ndarray) -> np.ndarray:
    return y.mean(axis=1).astype(np.float32) if y.ndim == 2 else y.astype(np.float32)


def _rms(y: np.ndarray) -> float:
    return float(np.sqrt(np.mean(y ** 2) + 1e-12))


def _active_rms(y: np.ndarray, threshold_db: float = -48.0) -> float:
    mono = _to_mono(y)
    cutoff = float(np.max(np.abs(mono)) + 1e-12) * 10 ** (threshold_db / 20)
    active = mono[np.abs(mono) > cutoff]
    return float(np.sqrt(np.mean(active ** 2) + 1e-12)) if len(active) >= SR else _rms(y)


def _ms_encode(stereo: np.ndarray) -> tuple:
    """(samples, 2) → (M, S) each (samples,)."""
    M = (stereo[:, 0] + stereo[:, 1]) / np.sqrt(2)
    S = (stereo[:, 0] - stereo[:, 1]) / np.sqrt(2)
    return M.astype(np.float32), S.astype(np.float32)


def _ms_decode(M: np.ndarray, S: np.ndarray) -> np.ndarray:
    """(M, S) → (samples, 2)."""
    L = (M + S) / np.sqrt(2)
    R = (M - S) / np.sqrt(2)
    return np.stack([L, R], axis=1).astype(np.float32)


def _mono_lf(audio: np.ndarray, cutoff_hz: float = 150.0) -> np.ndarray:
    """
    Collapse the Side channel below cutoff_hz to mono for mono compatibility.

    Sub-bass and low-bass are normally expected to be mono in professional mixes.
    When the beat's Side channel contains significant energy below 150 Hz (e.g.
    a "wide kick"), summing to mono cancels that Side content — causing the kick
    to sound thin or hollow on mono speakers/club systems.

    This function zeroes out the low-frequency portion of the Side channel by:
      1. M/S encode the signal
      2. High-pass the Side channel at cutoff_hz (keep only HF stereo info)
      3. M/S decode — the LF band collapses to mono automatically

    Research: correlation >0.90 below 150 Hz is the professional standard.
    Mono-collapsing the Side below 150 Hz guarantees correlation = 1.0.
    """
    nyq = SR / 2.0
    sos_hp = butter(4, cutoff_hz / nyq, btype="high", output="sos")
    M, S = _ms_encode(audio)
    S_hf = sosfilt(sos_hp, S.astype(np.float64)).astype(np.float32)
    return _ms_decode(M, S_hf)


def _maxx_bass(mix: np.ndarray, fundamental_lo: float = 40.0,
               fundamental_hi: float = 100.0, blend: float = 0.30) -> np.ndarray:
    """
    Waves Maxx Bass-style harmonic bass synthesis for small speaker compatibility.

    Sub-bass (40-80 Hz) is inaudible on earbuds, laptop speakers, and phone speakers.
    This function synthesizes 2nd and 3rd harmonics (80-300 Hz) from the fundamental,
    making bass "felt" on any speaker system.

    Used universally in mastering for streaming (Spotify, Apple Music) where
    listeners use earbuds that can't reproduce below 100 Hz.

    Algorithm:
      1. Bandpass the fundamental range (40-100 Hz)
      2. Apply heavy asymmetric saturation → generates 2x, 3x, 4x harmonics
      3. Bandpass to keep only harmonics (remove fundamental from saturated output)
      4. Blend harmonics into the original at blend level

    mix: (samples, 2) float32 stereo
    """
    nyq = SR / 2.0
    sos_fund = butter(4, [fundamental_lo / nyq, fundamental_hi / nyq],
                      btype="band", output="sos")

    # Harmonics should be above fundamental and below 500Hz
    h2_lo = fundamental_lo * 1.8
    h3_hi = min(fundamental_hi * 3.5, 500.0)
    sos_harm = butter(4, [h2_lo / nyq, h3_hi / nyq],
                      btype="band", output="sos")

    # Vectorized over both channels simultaneously (axis=0 = samples)
    mix_f64 = mix.astype(np.float64)
    fund = sosfilt(sos_fund, mix_f64, axis=0)              # (samples, 2)
    peak = np.max(np.abs(fund), axis=0, keepdims=True) + 1e-9
    fund_norm = fund / peak
    # Asymmetric saturation: positive half → softer (2nd harmonic), negative → harder (3rd)
    saturated = np.where(fund_norm > 0,
                         np.tanh(fund_norm * 4.0 * 0.7),
                         np.tanh(fund_norm * 4.0 * 1.3)) * peak
    harmonics = sosfilt(sos_harm, saturated - fund, axis=0)
    return (mix_f64 + harmonics * blend).astype(np.float32)


def _kick_sub_sidechain(inst: np.ndarray, depth: float = 0.35) -> np.ndarray:
    """
    Sub-bass management: kick transient sidechains the 20-80 Hz sub-bass range.

    In hip-hop, the kick and 808/sub-bass compete for headroom in 40-80 Hz.
    Without management: both hit simultaneously → limiter clamps → both lose punch.

    Method:
      - Detect kick transients from the 80-200 Hz "click" band
      - When a kick fires, duck the 20-80 Hz sub-bass by up to 'depth'
      - Duck curve: fast attack (3ms), slow release (80ms) — kick-style ADSR

    This technique is used universally in professional hip-hop mastering.
    """
    sos_kick = butter(4, [80 / (SR / 2), 200 / (SR / 2)],
                      btype="band", output="sos")
    sos_sub_lp = butter(4, 80 / (SR / 2), btype="low",  output="sos")
    sos_sub_hp = butter(4, 20 / (SR / 2), btype="high", output="sos")

    inst_mono = _to_mono(inst)

    # Kick detection: bandpass 80-200 Hz, envelope follow
    kick_band = sosfilt(sos_kick, inst_mono.astype(np.float64)).astype(np.float32)
    from scipy.signal import lfilter as _lfilter
    a_atk = np.exp(-1.0 / (SR * 0.003))   # 3ms attack
    a_rel = np.exp(-1.0 / (SR * 0.080))   # 80ms release
    rect = np.abs(kick_band).astype(np.float64)
    # Two-pass vectorized asymmetric envelope (fast attack, slow release)
    env_atk = _lfilter([1.0 - a_atk], [1.0, -a_atk], rect)
    env = _lfilter([1.0 - a_rel], [1.0, -a_rel],
                   np.maximum(env_atk, np.maximum.accumulate(rect) * 0.01)).astype(np.float32)

    # Normalize envelope → gain reduction (0 = no duck, depth = max duck)
    env_norm = env / (env.max() + 1e-9)
    gain = 1.0 - depth * env_norm  # 1.0 = no change, (1-depth) = max reduction

    # Apply gain only to sub-bass (20-80 Hz) of each channel
    result = inst.copy()
    for c in range(inst.shape[1]):
        ch = inst[:, c].astype(np.float64)
        sub = sosfilt(sos_sub_hp, sosfilt(sos_sub_lp, ch))
        above_sub = ch - sub
        result[:, c] = (above_sub + sub * gain).astype(np.float32)

    return result.astype(np.float32)


# ── Stem Separation ───────────────────────────────────────────────────────────

def _has_gpu() -> bool:
    """True if CUDA or MPS GPU is available for accelerated inference."""
    try:
        import torch
        return torch.cuda.is_available() or torch.backends.mps.is_available()
    except Exception:
        return False


def _run_roformer_on_stem(stem_path: Path, model_name: str, cache_dir: str,
                          want_label: str = "clean") -> bool:
    """
    Run a Roformer model (denoiser or de-reverb) on an already-separated stem WAV.
    Overwrites stem_path with the cleaned output.  Returns True on success.

    Runs the separator in an isolated subprocess so that any segfault or OOM
    in the heavy PyTorch model does not kill the parent fuse process.
    """
    import subprocess
    tmp_dir = Path(tempfile.mkdtemp(dir=cache_dir))
    result_flag = tmp_dir / "_done.flag"
    chosen_out  = tmp_dir / "_chosen.wav"

    # Build a self-contained script that runs the separator and writes the result
    script = f"""
import sys, shutil, logging
from pathlib import Path
import soundfile as sf
import numpy as np
sys.path.insert(0, {repr(str(Path(__file__).parent))})
from audio_separator.separator import Separator

tmp_dir   = Path({repr(str(tmp_dir))})
stem_path = Path({repr(str(stem_path))})
model_name = {repr(model_name)}
cache_dir  = {repr(str(cache_dir))}
want_label = {repr(want_label)}
SR         = 44100

# Strip parentheses from want_label to get the stem name for output_single_stem
# e.g. "(dry)" -> "dry", "(noreverb)" -> "noreverb"
_single_stem = want_label.strip("()") if want_label.startswith("(") else None

sep = Separator(
    log_level=logging.WARNING,
    output_dir=str(tmp_dir),
    output_format="WAV",
    sample_rate=SR,
    model_file_dir=str(Path(cache_dir) / "_models"),
    output_single_stem=_single_stem,  # skip computing the noise/artifact stem → ~2x faster
)
sep.load_model(model_name)
sep.separate(str(stem_path))

all_wavs = sorted(p for p in tmp_dir.iterdir() if p.suffix.lower() == ".wav")
print("      [Roformer] Output files:", [p.name for p in all_wavs], flush=True)

# Reject stems that ARE the noise/reverb artifact side.
# "(no dry)" must be rejected BEFORE "(dry)" is accepted since "dry" is a substring.
# Use parenthesized labels for precision — avoids false substring matches.
_reject = ["(noise)", "(no dry)", "(reverb)", "(residual)", "(background)", "(no vocals)"]
_accept = [want_label, "(dry)", "(noreverb)", "(no noise)", "(no reverb)", "(clean)", "(vocals)"]

chosen = None
for p in all_wavs:
    n = p.name.lower()
    if any(h in n for h in _accept) and not any(r in n for r in _reject):
        chosen = p; break
if chosen is None:
    # Fallback: first file that doesn't match any reject label
    for p in all_wavs:
        if not any(r in p.name.lower() for r in _reject):
            chosen = p; break
if chosen is None and all_wavs:
    chosen = all_wavs[0]

if chosen and chosen.exists():
    y, _ = sf.read(str(chosen), always_2d=True)
    if np.max(np.abs(y)) > 1e-4:
        shutil.copy2(str(chosen), {repr(str(chosen_out))})
        Path({repr(str(result_flag))}).touch()
        print("      [Roformer] Chose:", chosen.name, flush=True)
    else:
        print("      [Roformer] Output was silent — skipping", flush=True)
else:
    print("      [Roformer] No usable output file found", flush=True)
"""
    # System 4: Watchdog thread — kills subprocess if tmp_dir is stale for >3 min.
    # Without this, audio-separator can hang mid-chunk (MPS/ONNX deadlock) and
    # never exit, blocking the entire fuse for 40 minutes.
    _STALE_LIMIT = 180   # seconds with no new output file activity → kill

    def _watchdog(proc, watch_dir: Path, stale_s: int):
        last_activity = time.time()
        while proc.poll() is None:
            time.sleep(20)
            try:
                # Any WAV or flag written to the tmp dir counts as activity
                latest_mtime = max(
                    (f.stat().st_mtime for f in watch_dir.iterdir()
                     if f.suffix in (".wav", ".flag")),
                    default=0.0,
                )
                if latest_mtime > last_activity:
                    last_activity = latest_mtime
                if time.time() - last_activity > stale_s and proc.poll() is None:
                    print(f"      [{model_name}] watchdog: no activity for {stale_s}s"
                          " — killing stalled subprocess", flush=True)
                    proc.kill()
                    return
            except Exception:
                pass

    try:
        import subprocess as _sp
        proc = _sp.Popen([sys.executable, "-c", script])
        wd = threading.Thread(target=_watchdog,
                              args=(proc, tmp_dir, _STALE_LIMIT), daemon=True)
        wd.start()
        try:
            proc.wait(timeout=2400)   # hard ceiling: 40 min
        except _sp.TimeoutExpired:
            proc.kill()
            print(f"      [{model_name}] hard timeout (40 min) — skipping", flush=True)
            return False

        if proc.returncode not in (0, -11, -9):
            print(f"      [{model_name}] subprocess exited {proc.returncode}", flush=True)

        if result_flag.exists() and chosen_out.exists():
            shutil.copy2(str(chosen_out), str(stem_path))
            return True
        return False
    except Exception as _e:
        print(f"      [{model_name}] failed: {_e}", flush=True)
        return False
    finally:
        shutil.rmtree(str(tmp_dir), ignore_errors=True)


def separate(audio_path: str, cache_dir: str = "vf_data/stems",
             upgrade_vocal: bool = False, clean_vocal: bool = False) -> dict:
    """
    Separate vocals using the best available model:
      GPU available → BS-Roformer via audio-separator (SDR ~13, fast on GPU)
      CPU only      → Demucs htdemucs_ft (SDR ~8.5, fast on CPU; BS-Roformer
                      would take 50+ minutes without hardware acceleration)

    Cached by file fingerprint. Returns stereo (samples, 2) float32 arrays.
    """
    os.makedirs(cache_dir, exist_ok=True)
    fid = _file_id(audio_path)
    cached = Path(cache_dir) / fid

    if not (cached / "vocals.wav").exists():
        cached.mkdir(exist_ok=True)

        if _has_gpu():
            # GPU path: BS-Roformer via audio-separator
            tmp_dir = Path(tempfile.mkdtemp(dir=cache_dir))
            try:
                from audio_separator.separator import Separator
                sep = Separator(
                    log_level=logging.WARNING,
                    output_dir=str(tmp_dir),
                    output_format="WAV",
                    sample_rate=SR,
                    model_file_dir=str(Path(cache_dir) / "_models"),
                )
                sep.load_model(_BS_ROFORMER)
                sep.separate(audio_path)

                vox_src = inst_src = None
                for p in tmp_dir.iterdir():
                    lname = p.name.lower()
                    if "(vocals)" in lname and "(instrumental)" not in lname:
                        vox_src = p
                    elif "(instrumental)" in lname or "(no_vocals)" in lname:
                        inst_src = p

                if vox_src and inst_src:
                    shutil.move(str(vox_src),  str(cached / "vocals.wav"))
                    shutil.move(str(inst_src), str(cached / "no_vocals.wav"))
                else:
                    wavs = sorted(tmp_dir.glob("*.wav"))
                    if len(wavs) >= 2:
                        shutil.move(str(wavs[0]), str(cached / "vocals.wav"))
                        shutil.move(str(wavs[1]), str(cached / "no_vocals.wav"))
                    else:
                        raise RuntimeError("audio-separator produced no output")
            except Exception as e:
                print(f"      [BS-Roformer failed ({e}), falling back to Demucs]",
                      flush=True)
                _separate_demucs(audio_path, cached)
            finally:
                shutil.rmtree(str(tmp_dir), ignore_errors=True)
        else:
            # CPU path: Demucs htdemucs_ft 4-stem for oracle stems.
            print("      CPU path: Demucs htdemucs_ft 4-stem (oracle stems)…", flush=True)
            _separate_demucs(audio_path, cached)

    # MDX-Net vocal upgrade runs AFTER the cache check so it applies even when
    # vocals.wav was already cached from a previous (non-upgraded) Demucs run.
    # Sentinel ensures it only runs once per song.
    if upgrade_vocal and not _has_gpu():
        _sentinel = cached / "_mdx_vocal_upgraded"
        if not _sentinel.exists():
            try:
                _upgrade_vocal_mdx(audio_path, cached, cache_dir)
                _sentinel.touch()
            except Exception as _me:
                print(f"      [MDX-Net vocal upgrade failed: {_me} — using Demucs vocal]",
                      flush=True)

    # ── Neural vocal cleaning cascade ─────────────────────────────────────────
    # Step 1: Mel-Roformer-Denoise — removes bleed/noise from the vocal stem.
    #         SDR 27.99 — extremely effective at removing instrumental bleed.
    # Step 2: BS-Roformer-De-Reverb — removes original song's room reverb,
    #         leaving a dry signal that sits cleanly in the new beat's space.
    # Both run on the already-separated vocals.wav and overwrite it in-place.
    # Sentinels prevent re-running if the file was already cleaned.
    # clean_vocal=False for Song A (beat) — we don't need to clean its vocals.
    _voc_path = cached / "vocals.wav"
    if clean_vocal and _voc_path.exists():
        _denoise_sentinel = cached / "_vocal_denoised"
        if not _denoise_sentinel.exists():
            print("      [Neural Denoiser] Running Mel-Roformer-Denoise on vocal stem…",
                  flush=True)
            ok = _run_roformer_on_stem(_voc_path, _DENOISE_MODEL, cache_dir,
                                       want_label="(dry)")  # model target_instrument=dry → outputs (Dry)/(No Dry)
            if ok:
                _denoise_sentinel.touch()
                print("      [Neural Denoiser] Vocal stem cleaned.", flush=True)
            else:
                print("      [Neural Denoiser] Failed — using undenoised vocal.", flush=True)

        # De-reverb DISABLED: studio-recorded vocals have intentional production reverb
        # that is part of the performance. Removing it makes vocals sound unnatural
        # and the model (63s/chunk × 27 chunks = 28 min) is too slow on CPU.
        # De-reverb is only beneficial for room recordings, not studio productions.
        # _deverb_sentinel = cached / "_vocal_deverbed"
        # (de-reverb code removed)

    stems = {}
    for name in ("vocals", "no_vocals", "drums", "bass", "other"):
        p = cached / f"{name}.wav"
        if not p.exists():
            continue
        y, file_sr = sf.read(str(p), always_2d=True)
        if file_sr != SR:
            y = np.stack([
                librosa.resample(y[:, c], orig_sr=file_sr, target_sr=SR)
                for c in range(y.shape[1])
            ], axis=1)
        stems[name] = y.astype(np.float32)
    return stems


def _separate_demucs(audio_path: str, out_dir: Path) -> None:
    """4-stem Demucs htdemucs_ft: vocals / drums / bass / other + derived no_vocals."""
    import subprocess, sys
    from pathlib import Path as _Path

    fid = out_dir.name
    ext = _Path(audio_path).suffix or ".mp3"
    tmp = out_dir.parent / f"{fid}_src{ext}"
    shutil.copy2(audio_path, tmp)
    try:
        result = subprocess.run(
            [sys.executable, "-m", "demucs",
             "-n", "htdemucs_ft",
             "-o", str(out_dir.parent),
             str(tmp)],
            capture_output=True, text=True,
        )
        if result.returncode != 0:
            raise RuntimeError(f"Demucs failed:\n{result.stderr}")
        # Demucs names its output dir after the input filename (without extension).
        # The temp file is named "{fid}_src{ext}", so Demucs creates "{fid}_src/".
        raw = out_dir.parent / "htdemucs_ft" / f"{fid}_src"
        if raw.exists():
            for f in raw.iterdir():
                shutil.move(str(f), str(out_dir / f.name))
            raw.rmdir()
            try:
                (out_dir.parent / "htdemucs_ft").rmdir()
            except OSError:
                pass

        # Derive no_vocals.wav by summing drums + bass + other
        # This gives us a clean instrumental reference AND the 4 oracle stems.
        stems_4 = {}
        for sname in ("drums", "bass", "other"):
            p = out_dir / f"{sname}.wav"
            if p.exists():
                y, _ = sf.read(str(p), always_2d=True)
                stems_4[sname] = y.astype(np.float32)
        if len(stems_4) == 3:
            min_len = min(s.shape[0] for s in stems_4.values())
            no_vox = sum(s[:min_len] for s in stems_4.values())
            sf.write(str(out_dir / "no_vocals.wav"), no_vox, SR, subtype="PCM_24")
        else:
            # Fallback: if stems missing, duplicate vocals as placeholder (rare)
            vp = out_dir / "vocals.wav"
            if vp.exists() and not (out_dir / "no_vocals.wav").exists():
                import shutil as _sh
                _sh.copy2(str(vp), str(out_dir / "no_vocals.wav"))
    finally:
        if tmp.exists():
            tmp.unlink()


_MDX_VOCAL_2 = "UVR-MDX-NET-Voc_FT.onnx"   # second model for ensemble (SDR ~10.5)


def _run_mdx_model(model_name: str, audio_path: str, tmp_dir: Path,
                   cache_dir: str) -> tuple:
    """Run one MDX-Net model and return (vocals_array, no_vocals_array) float32."""
    from audio_separator.separator import Separator
    run_dir = tmp_dir / model_name.replace(".", "_")
    run_dir.mkdir(exist_ok=True)
    sep = Separator(
        log_level=logging.WARNING,
        output_dir=str(run_dir),
        output_format="WAV",
        sample_rate=SR,
        model_file_dir=str(Path(cache_dir) / "_models"),
        mdx_params={"enable_denoise": True, "overlap": 0.25},  # built-in MDX denoiser
    )
    sep.load_model(model_name)
    sep.separate(audio_path)

    vox_src = inst_src = None
    for p in run_dir.iterdir():
        lname = p.name.lower()
        if "(vocals)" in lname and "(instrumental)" not in lname:
            vox_src = p
        elif "(instrumental)" in lname or "(no_vocals)" in lname:
            inst_src = p

    if not (vox_src and inst_src):
        wavs = sorted(run_dir.glob("*.wav"))
        if len(wavs) >= 2:
            vox_src, inst_src = wavs[0], wavs[1]
        else:
            raise RuntimeError(f"{model_name} produced no output")

    vox_y,  _ = sf.read(str(vox_src),  always_2d=True)
    inst_y, _ = sf.read(str(inst_src), always_2d=True)
    return vox_y.astype(np.float32), inst_y.astype(np.float32)


def _separate_mdx(audio_path: str, out_dir: Path, cache_dir: str) -> None:
    """
    CPU path: Ensemble of two MDX-Net models averaged in the STFT domain.

    Kim_Vocal_2 (SDR ~9.5) and UVR-MDX-NET-Voc_FT (SDR ~10.5) make different
    errors in different frequency bins.  Averaging their STFT magnitudes before
    reconstruction cancels correlated bleed while reinforcing correlated signal.
    Effective SDR improvement: ~1.5-2.5 dB over either model alone.

    Falls back to Kim_Vocal_2 only if the second model fails.
    """
    tmp_dir = Path(tempfile.mkdtemp(dir=cache_dir))
    try:
        print(f"      MDX-Net ensemble: running {_MDX_VOCAL}…", flush=True)
        vox1, inst1 = _run_mdx_model(_MDX_VOCAL,   audio_path, tmp_dir, cache_dir)
        print(f"      MDX-Net ensemble: running {_MDX_VOCAL_2}…", flush=True)
        try:
            vox2, inst2 = _run_mdx_model(_MDX_VOCAL_2, audio_path, tmp_dir, cache_dir)

            # Align lengths (models may produce slightly different lengths)
            min_len = min(vox1.shape[0], vox2.shape[0])
            min_ch  = min(vox1.shape[1], vox2.shape[1])
            vox1, vox2   = vox1[:min_len, :min_ch], vox2[:min_len, :min_ch]
            inst1, inst2 = inst1[:min_len, :min_ch], inst2[:min_len, :min_ch]

            # STFT-domain averaging: average magnitudes, use Model 1 phase.
            # Averaging in amplitude (not dB) gives the correct geometric mean
            # of the two separation masks.
            n_fft = 2048
            vox_avg  = np.zeros_like(vox1)
            inst_avg = np.zeros_like(inst1)
            for c in range(min_ch):
                D_v1 = librosa.stft(vox1[:, c],  n_fft=n_fft)
                D_v2 = librosa.stft(vox2[:, c],  n_fft=n_fft)
                D_i1 = librosa.stft(inst1[:, c], n_fft=n_fft)
                D_i2 = librosa.stft(inst2[:, c], n_fft=n_fft)

                mag_v  = (np.abs(D_v1)  + np.abs(D_v2))  / 2.0
                mag_i  = (np.abs(D_i1)  + np.abs(D_i2))  / 2.0
                phase_v = np.angle(D_v1)
                phase_i = np.angle(D_i1)

                vox_avg[:, c]  = librosa.istft(
                    mag_v * np.exp(1j * phase_v), length=min_len).astype(np.float32)
                inst_avg[:, c] = librosa.istft(
                    mag_i * np.exp(1j * phase_i), length=min_len).astype(np.float32)

            print("      MDX-Net ensemble: averaging complete.", flush=True)
            sf.write(str(out_dir / "vocals.wav"),    vox_avg,  SR, subtype="PCM_24")
            sf.write(str(out_dir / "no_vocals.wav"), inst_avg, SR, subtype="PCM_24")

        except Exception as _e2:
            print(f"      [{_MDX_VOCAL_2} failed ({_e2}), using {_MDX_VOCAL} only]",
                  flush=True)
            sf.write(str(out_dir / "vocals.wav"),    vox1,  SR, subtype="PCM_24")
            sf.write(str(out_dir / "no_vocals.wav"), inst1, SR, subtype="PCM_24")

    finally:
        shutil.rmtree(str(tmp_dir), ignore_errors=True)


def _upgrade_vocal_mdx(audio_path: str, cached: Path, cache_dir: str) -> None:
    """
    Run MDX-Net Kim Vocal 2 in a subprocess and STFT-blend with the Demucs vocal.

    Runs in subprocess so it gets a fresh memory space — avoids OOM when
    Demucs' memory hasn't fully returned to the OS after its subprocess exits.
    The blend script runs entirely standalone (no fuser imports) to keep it lean.
    """
    print(f"      MDX-Net vocal upgrade ({_MDX_VOCAL}, subprocess)…", flush=True)

    _script = f"""
import sys, tempfile, shutil, numpy as np, soundfile as sf, librosa
from pathlib import Path
from audio_separator.separator import Separator

audio_path = sys.argv[1]
cached_dir = sys.argv[2]
cache_dir  = sys.argv[3]
model_name = sys.argv[4]
SR = 44100
n_fft = 2048
MDX_WEIGHT = 0.65
DEM_WEIGHT = 0.35

tmp_dir = Path(tempfile.mkdtemp(dir=cache_dir))
try:
    sep = Separator(
        log_level=30,
        output_dir=str(tmp_dir),
        output_format="WAV",
        sample_rate=SR,
        model_file_dir=str(Path(cache_dir) / "_models"),
    )
    sep.load_model(model_name)
    sep.separate(audio_path)

    vox_src = None
    for p in tmp_dir.iterdir():
        lname = p.name.lower()
        if "(vocals)" in lname and "(instrumental)" not in lname:
            vox_src = p
    if vox_src is None:
        wavs = sorted(tmp_dir.glob("*.wav"))
        if wavs:
            vox_src = wavs[0]
    if vox_src is None:
        raise RuntimeError("MDX-Net produced no vocal output")

    mdx_vox, _ = sf.read(str(vox_src), always_2d=True)
    mdx_vox = mdx_vox.astype(np.float32)
    dem_vox, _ = sf.read(str(Path(cached_dir) / "vocals.wav"), always_2d=True)
    dem_vox = dem_vox.astype(np.float32)

    min_len = min(mdx_vox.shape[0], dem_vox.shape[0])
    min_ch  = min(mdx_vox.shape[1], dem_vox.shape[1])
    ensemble = np.zeros((min_len, min_ch), dtype=np.float32)

    for c in range(min_ch):
        D_m = librosa.stft(mdx_vox[:min_len, c], n_fft=n_fft)
        D_d = librosa.stft(dem_vox[:min_len, c], n_fft=n_fft)
        mag_avg = MDX_WEIGHT * np.abs(D_m) + DEM_WEIGHT * np.abs(D_d)
        ensemble[:, c] = librosa.istft(
            mag_avg * np.exp(1j * np.angle(D_m)), length=min_len, n_fft=n_fft
        ).astype(np.float32)

    sf.write(str(Path(cached_dir) / "vocals.wav"), ensemble, SR, subtype="PCM_24")
    print("MDX_UPGRADE_OK")
finally:
    shutil.rmtree(str(tmp_dir), ignore_errors=True)
"""

    import subprocess as _sp
    result = _sp.run(
        [sys.executable, "-c", _script,
         audio_path, str(cached), cache_dir, _MDX_VOCAL],
        capture_output=True, text=True, timeout=900,
    )
    if result.returncode == 0 and "MDX_UPGRADE_OK" in result.stdout:
        print("      MDX-Net vocal upgrade complete.", flush=True)
    else:
        err = (result.stderr or "")[-400:]
        raise RuntimeError(f"MDX subprocess failed (rc={result.returncode}): {err}")


def _oracle_wiener_clean(vox: np.ndarray,
                          drums: np.ndarray,
                          bass: np.ndarray,
                          other: np.ndarray,
                          drum_weight_hh: float = 3.5,
                          mask_floor: float = 0.08) -> np.ndarray:
    """
    Oracle Wiener mask using all 4 Demucs stems.

    Classical 2-stem Wiener: mask = V² / (V² + I²)
    where I is the blind instrumental estimate — only knows "not vocals".

    Oracle 4-stem Wiener: mask = V² / (V² + D² + B² + O²)
    Each source is measured separately, giving a precise interference map.
    Extra drum-weight in 4–16 kHz (hi-hat range): drums are 2× interference
    in those bins, which directly targets the scratchy bleed artifact.

    Floor at 0.10 (lower than 2-stem 0.15) — oracle accuracy means we can
    suppress more aggressively without killing consonant transients, because
    the mask is precise rather than a coarse estimate.
    """
    import librosa

    N_FFT = 2048
    HH_LO, HH_HI = 4000, 16000   # hi-hat range for extra drum suppression
    DRUM_WEIGHT  = drum_weight_hh  # AI-tunable: drums count N× as interference in hi-hat band
    MASK_FLOOR   = mask_floor      # AI-tunable: lower floor → more aggressive suppression

    freqs = librosa.fft_frequencies(sr=SR, n_fft=N_FFT)
    hh_bins = (freqs >= HH_LO) & (freqs <= HH_HI)

    min_len = min(vox.shape[0], drums.shape[0], bass.shape[0], other.shape[0])
    vox_clean = np.zeros_like(vox)

    for c in range(vox.shape[1]):
        ic_d = min(c, drums.shape[1] - 1)
        ic_b = min(c, bass.shape[1] - 1)
        ic_o = min(c, other.shape[1] - 1)

        D_v = librosa.stft(vox[:min_len, c],          n_fft=N_FFT)
        D_d = librosa.stft(drums[:min_len, ic_d],     n_fft=N_FFT)
        D_b = librosa.stft(bass[:min_len, ic_b],      n_fft=N_FFT)
        D_o = librosa.stft(other[:min_len, ic_o],     n_fft=N_FFT)

        mag_v = np.abs(D_v)
        mag_d = np.abs(D_d)
        mag_b = np.abs(D_b)
        mag_o = np.abs(D_o)

        # Extra drum suppression in hi-hat range
        mag_d_w = mag_d.copy()
        mag_d_w[hh_bins, :] *= DRUM_WEIGHT

        denom = mag_v ** 2 + mag_d_w ** 2 + mag_b ** 2 + mag_o ** 2 + 1e-8
        raw_mask = mag_v ** 2 / denom
        mask = np.maximum(raw_mask, MASK_FLOOR)

        D_clean = (mask * mag_v) * np.exp(1j * np.angle(D_v))
        vox_clean[:, c] = librosa.istft(
            D_clean, length=vox.shape[0], n_fft=N_FFT).astype(np.float32)

    return vox_clean.astype(np.float32)


def _targeted_hihat_suppression(vox: np.ndarray, drums: np.ndarray,
                                 hihat_alpha: float = 0.90) -> np.ndarray:
    """
    Spectral subtraction of drum-stem hi-hat content from vocal in 6-16 kHz.

    After oracle Wiener mask reduces bleed by ~70-80%, this stage handles
    the residual.  Uses the drum stem as a direct noise reference:

        mag_out[hh] = max(|V[hh]| - α·|D[hh]|, β·|V[hh]|)

    α = subtraction strength (0.90 — aggressive but leaves a floor)
    β = spectral floor (0.06 = 6% of original remains — protects sibilants)

    The floor β ensures we never mute genuine vocal sibilants (S/SH/CH),
    which share the hi-hat frequency range.  Without it, suppression creates
    a "hole" that sounds worse than light bleed.
    """
    import librosa

    N_FFT  = 2048
    HH_LO  = 6000    # Hz — upper hi-hat range only (more sibilant-safe than 4kHz)
    HH_HI  = 16000   # Hz
    ALPHA  = hihat_alpha  # AI-tunable spectral subtraction coefficient
    BETA   = 0.06         # floor: never suppress below 6% of original

    freqs   = librosa.fft_frequencies(sr=SR, n_fft=N_FFT)
    hh_bins = (freqs >= HH_LO) & (freqs <= HH_HI)

    min_len = min(vox.shape[0], drums.shape[0])
    out = vox.copy()

    for c in range(vox.shape[1]):
        ic_d = min(c, drums.shape[1] - 1)
        D_v  = librosa.stft(vox[:min_len,   c],     n_fft=N_FFT)
        D_d  = librosa.stft(drums[:min_len, ic_d],  n_fft=N_FFT)

        mag_v = np.abs(D_v)
        mag_d = np.abs(D_d)
        ph_v  = np.angle(D_v)

        mag_out = mag_v.copy()
        mag_out[hh_bins, :] = np.maximum(
            mag_v[hh_bins, :] - ALPHA * mag_d[hh_bins, :],
            BETA  * mag_v[hh_bins, :]
        )

        reconstructed = librosa.istft(
            mag_out * np.exp(1j * ph_v), length=min_len, n_fft=N_FFT
        ).astype(np.float32)
        out[:min_len, c] = reconstructed
        # tail beyond min_len keeps the original vocal (drums ended earlier)

    return out.astype(np.float32)


def _harmonic_vocal_process(vox: np.ndarray, mix_dry: float = 0.50) -> np.ndarray:
    """
    Combined voice harmonic resynthesis + enhancement in one STFT pass.

    Traditional spectral subtraction (Wiener, hi-hat subtraction) removes bleed
    globally without knowing the signal model.  This function knows the signal
    model: a human voice is a sum of harmonics at integer multiples of F0.
    Anything NOT at those harmonic frequencies is by definition NOT the voice.

    Single PYIN pass → per-frame harmonic mask:

    VOICED frames (PYIN detects F0):
      • Keep bins within ±BW_BINS of each F0·k (k=1…N_HARM), boosted +1.5 dB
        to restore energy suppressed by earlier masking stages.
      • Suppress all other bins to FLOOR (10% of original) — not zero, because
        noise-modelled consonants (fricative energy between harmonics) matter.

    UNVOICED frames (no stable F0):
      • Pass through unchanged.  Consonants (S, T, SH, CH) are aperiodic and
        handled by the earlier oracle Wiener + spectral subtraction stages.

    Final MIX = 65% harmonic-processed + 35% cleaned original.
    The 35% dry blend preserves consonant naturalness and prevents the
    resynthesised "pure harmonic" character from sounding too synthetic.

    Effective result: removes residual inter-harmonic bleed that survives the
    oracle Wiener and targeted spectral subtraction stages, while restoring
    any harmonics those stages over-suppressed.
    """
    import librosa

    N_FFT    = 2048
    HOP      = 512
    FMIN     = 60.0
    FMAX     = 1100.0   # soprano can reach ~1050 Hz
    BW_BINS  = 3        # ±bins kept around each harmonic (±65 Hz @ 44100/2048)
    N_HARM   = 24       # track up to 24th harmonic
    FLOOR    = 0.10     # inter-harmonic floor in hi-hat zone
    BOOST    = 10 ** (1.5 / 20.0)   # +1.5 dB harmonic boost
    MIX_DRY  = mix_dry  # AI-tunable: higher = more natural, lower = cleaner/more harmonic

    # KEY: only suppress inter-harmonic content ABOVE HH_BIN.
    # Below that, the voice character (formants, vowels) lives — leave it alone.
    # The 0-4kHz band makes the voice sound human; suppressing it sounds robotic.
    # 4-16kHz is where hi-hat bleed lives — safe to apply harmonic mask there.
    freqs_v = librosa.fft_frequencies(sr=SR, n_fft=N_FFT)
    HH_BIN  = int(np.searchsorted(freqs_v, 4000.0))   # ~186 @ 44100/2048

    mono = _to_mono(vox)
    f0, voiced_flag, _ = librosa.pyin(
        mono, fmin=FMIN, fmax=FMAX, sr=SR,
        hop_length=HOP, frame_length=N_FFT)

    out = np.zeros_like(vox)

    for c in range(vox.shape[1]):
        D   = librosa.stft(vox[:, c], n_fft=N_FFT, hop_length=HOP)
        mag = np.abs(D)
        ph  = np.angle(D)

        mag_proc = mag.copy()
        n_frames = min(len(voiced_flag), mag.shape[1])

        for fi in range(n_frames):
            if not voiced_flag[fi] or np.isnan(f0[fi]):
                continue  # unvoiced: passthrough unchanged

            f0_hz = float(f0[fi])

            # Build harmonic mask for this frame
            is_harmonic = np.zeros(mag.shape[0], dtype=bool)
            for k in range(1, N_HARM + 1):
                freq_hz = f0_hz * k
                if freq_hz >= SR / 2:
                    break
                bi  = int(np.round(freq_hz * N_FFT / SR))
                lo  = max(0, bi - BW_BINS)
                hi  = min(mag.shape[0], bi + BW_BINS + 1)
                is_harmonic[lo:hi] = True

            # Harmonic bins: slight boost (restores Wiener over-suppression)
            mag_proc[is_harmonic, fi] = mag[is_harmonic, fi] * BOOST
            # Inter-harmonic suppression ABOVE 4kHz only — this is where hi-hat bleed lives.
            # Below 4kHz: voice formants, vowel character, fundamental body — leave untouched.
            # Suppressing below 4kHz creates vocoder/robotic effect.
            bin_idx = np.arange(mag.shape[0])
            suppress_mask = ~is_harmonic & (bin_idx >= HH_BIN)
            mag_proc[suppress_mask, fi] = mag[suppress_mask, fi] * FLOOR

        D_proc      = mag_proc * np.exp(1j * ph)
        proc_signal = librosa.istft(
            D_proc, length=vox.shape[0], hop_length=HOP, n_fft=N_FFT
        ).astype(np.float32)

        # Blend: mostly processed (clean), partly original (natural consonants)
        out[:, c] = (
            (1.0 - MIX_DRY) * proc_signal + MIX_DRY * vox[:, c]
        ).astype(np.float32)

    return out.astype(np.float32)


# ── Analysis ──────────────────────────────────────────────────────────────────

def detect_bpm(y_mono: np.ndarray) -> float:
    """
    Genre-agnostic BPM detection using multi-prior consensus + spectral fingerprint.

    Covers the full musical tempo range: 55-220 BPM.
    Spectral content (centroid, sub ratio, onset density) is used to weight
    tempo priors toward the likely zone without hard-coding a genre assumption.
    Consensus voting across all priors picks the most-agreed-upon candidate.
    Half/double tempo correction is onset-rate driven (not genre-hardcoded).
    """
    from scipy.stats import lognorm as _lognorm
    hop    = 512
    clip30 = y_mono[:SR * 30]
    onset_env = librosa.onset.onset_strength(y=y_mono, sr=SR, hop_length=hop)

    # ── Spectral fingerprint → tempo zone ─────────────────────────────────────
    S      = np.abs(librosa.stft(clip30, n_fft=2048, hop_length=hop))
    freqs  = librosa.fft_frequencies(sr=SR)
    total_e  = float(S.mean()) + 1e-9
    sub_e    = float(S[(freqs >= 20)  & (freqs < 80)].mean())
    sub_ratio = sub_e / total_e
    centroid  = float(librosa.feature.spectral_centroid(S=S, sr=SR).mean())
    onset_rate = len(librosa.onset.onset_detect(
        onset_envelope=onset_env, sr=SR, hop_length=hop)) / (len(clip30) / SR + 1e-9)

    # Content-adaptive prior centers — ordered by confidence (most likely first)
    if centroid > 4000 and onset_rate > 6.5:
        prior_centers = [170, 145, 120]    # drum & bass / breakbeat / fast EDM
    elif sub_ratio > 0.25 and centroid > 2500:
        prior_centers = [130, 150, 110]    # trap / EDM / hip-hop
    elif centroid < 1800 and onset_rate < 2.5:
        prior_centers = [75, 90, 60]       # ballad / slow R&B / soul
    elif centroid < 2500 and onset_rate < 3.5:
        prior_centers = [95, 110, 80]      # mid-tempo hip-hop / reggaeton / R&B
    else:
        prior_centers = [110, 128, 95]     # pop / dance / default

    # ── Method 1: beat_track (autocorrelation — no genre prior) ───────────────
    bpm_bt = float(librosa.beat.beat_track(
        onset_envelope=onset_env, sr=SR, hop_length=hop)[0])

    # ── Method 2: multiple lognormal priors across the tempo spectrum ──────────
    candidates = [bpm_bt]
    for center in prior_centers:
        try:
            t = float(librosa.feature.rhythm.tempo(
                onset_envelope=onset_env, sr=SR, hop_length=hop,
                prior=_lognorm(s=0.25, scale=center))[0])
            candidates.append(t)
        except Exception:
            pass
    try:    # also add a completely prior-free estimate
        candidates.append(float(librosa.feature.rhythm.tempo(
            onset_envelope=onset_env, sr=SR, hop_length=hop)[0]))
    except Exception:
        pass

    # ── Octave-normalise all candidates to 55-220 BPM ─────────────────────────
    def _norm(b):
        while b > 220: b /= 2
        while b < 55:  b *= 2
        return float(b)

    normed = [_norm(c) for c in candidates]

    # ── Consensus vote: most agreement within ±6% at any octave ──────────────
    def _octave_agree(a, b):
        for m in (0.5, 1.0, 2.0):
            if abs(a * m - b) / (b + 1e-9) < 0.06:
                return True
        return False

    votes = [sum(_octave_agree(c, o) for o in normed) for c in normed]
    # Tiebreak: prefer the estimate closest to the top content-adaptive center
    best_idx = max(range(len(normed)),
                   key=lambda i: (votes[i], -abs(normed[i] - prior_centers[0])))
    bpm = normed[best_idx]

    # ── Genre-agnostic half/double tempo correction ────────────────────────────
    # High onset rate (>5.5/s) + low BPM → half-tempo was detected, double it
    # Low onset rate (<1.5/s)  + high BPM → double-tempo detected, halve it
    if onset_rate > 5.5 and bpm < 95:
        print(f"      BPM ×2 (onset_rate={onset_rate:.1f}/s): {bpm:.0f}→{bpm*2:.0f}", flush=True)
        bpm *= 2
    elif onset_rate < 1.5 and bpm > 145:
        print(f"      BPM ÷2 (onset_rate={onset_rate:.1f}/s): {bpm:.0f}→{bpm/2:.0f}", flush=True)
        bpm /= 2

    while bpm > 220: bpm /= 2
    while bpm < 55:  bpm *= 2

    return float(bpm)


def _best_ratio(bpm_a: float, bpm_b: float) -> float:
    """
    Compute time-stretch ratio with BPM octave correction.
    Tries half/double bpm_b before clamping — critical for extreme tempo pairs
    (e.g., 60 BPM ballad + 130 BPM trap) where naive ratio would be clamped badly.
    """
    candidates = [
        bpm_a / bpm_b,          # direct
        bpm_a / (bpm_b * 2),    # treat B as half-tempo (double-speed track)
        bpm_a / (bpm_b / 2),    # treat B as double-tempo (half-speed track)
    ]
    # Pick candidate closest to 1.0 (minimum time-stretch needed)
    best = min(candidates, key=lambda r: abs(r - 1.0))
    # Vocal quality cap: rubberband R3 produces audible artifacts beyond ±18%.
    # Beyond that, accept gentle beat drift rather than destroying vocal quality.
    QUALITY_CAP = 1.18
    clamped = float(np.clip(best, 1.0 / QUALITY_CAP, QUALITY_CAP))
    if abs(best - clamped) > 0.01:
        print(f"      [INFO] BPM ratio {best:.3f} capped at {clamped:.3f} — "
              f"large tempo gap ({abs(best-1)*100:.0f}%); protecting vocal quality.",
              flush=True)
    return clamped


def detect_key(y_mono: np.ndarray) -> tuple:
    """
    Multi-window energy-weighted key detection.

    Analyzes the full track plus 3 temporal segments (start / middle / end),
    weights each window by its RMS energy, then votes across all windows.

    Why this matters:
    - Intros often have drums-only or sparse harmonic content → misleading key
    - Choruses (usually in the middle/end) have the clearest chord voicings
    - Energy weighting ensures loud, chord-dense sections dominate the vote
    - chroma_cqt handles overtones better than chroma_cens for dense mixes
    """
    def _ks_vote(chroma_mean):
        chroma_mean = chroma_mean / (chroma_mean.sum() + 1e-9)
        best_s, best_r, best_m = -np.inf, 0, "major"
        for root in range(12):
            rot = np.roll(chroma_mean, -root)
            for profile, mode in ((_KS_MAJOR, "major"), (_KS_MINOR, "minor")):
                s = float(np.dot(rot, profile / profile.sum()))
                if s > best_s:
                    best_s, best_r, best_m = s, root, mode
        return best_r, best_m, best_s

    duration = len(y_mono)
    seg_len  = min(duration, SR * 40)

    windows = [y_mono]  # full track always included
    if duration > SR * 30:
        mid = duration // 2
        windows += [
            y_mono[:seg_len],              # start (intro / verse)
            y_mono[mid:mid + seg_len],     # middle (typically chorus-heavy)
            y_mono[max(0, duration - seg_len):],  # end
        ]

    votes = {}  # (root, mode) → accumulated weighted confidence
    for seg in windows:
        rms_w = float(np.sqrt(np.mean(seg ** 2)) + 1e-9)
        try:
            chroma  = librosa.feature.chroma_cqt(
                y=seg.astype(np.float32), sr=SR, bins_per_octave=36)
            root, mode, conf = _ks_vote(chroma.mean(axis=1))
            k = (root, mode)
            votes[k] = votes.get(k, 0.0) + rms_w * conf
        except Exception:
            try:    # fallback to chroma_cens if CQT fails
                chroma  = librosa.feature.chroma_cens(
                    y=seg.astype(np.float32), sr=SR)
                root, mode, conf = _ks_vote(chroma.mean(axis=1))
                k = (root, mode)
                votes[k] = votes.get(k, 0.0) + rms_w * conf * 0.8
            except Exception:
                pass

    if not votes:
        return 0, "major"
    best = max(votes, key=votes.get)
    return best


def semitones_to_shift(src_root, src_mode, dst_root, dst_mode) -> int:
    src = (src_root + 3) % 12 if src_mode == "minor" else src_root
    dst = (dst_root + 3) % 12 if dst_mode == "minor" else dst_root
    diff = (dst - src) % 12
    return diff - 12 if diff > 6 else diff


def _detect_section_start(y_mono: np.ndarray, section: str = "chorus") -> int:
    """
    Detect the sample position of the first 'chorus' (or 'verse') in the track.

    Uses librosa's recurrence-based segmentation + a composite energy/centroid/onset
    score to classify each segment as chorus-like (high energy) or verse-like (quieter).

    Returns sample index of the first matching section boundary, or 0 on failure.
    """
    try:
        hop = 512
        mfcc = librosa.feature.mfcc(y=y_mono, sr=SR, n_mfcc=13, hop_length=hop)

        # Agglomerative segmentation — aim for ~6-8 sections
        n_sections = 7
        boundaries = librosa.segment.agglomerative(mfcc, k=n_sections)

        if len(boundaries) < 2:
            return 0

        # Composite score per segment: energy + centroid + onset density
        rms      = librosa.feature.rms(y=y_mono, hop_length=hop)[0]
        centroid = librosa.feature.spectral_centroid(y=y_mono, sr=SR, hop_length=hop)[0]
        onsets   = librosa.onset.onset_detect(y=y_mono, sr=SR, hop_length=hop)

        def znorm(x):
            return (x - x.mean()) / (x.std() + 1e-8)

        rms_n  = znorm(rms)
        cen_n  = znorm(centroid)

        # Per-segment onset density
        onset_density = np.zeros(len(rms), dtype=np.float32)
        win = max(1, int(SR * 2.0 / hop))
        for o in onsets:
            onset_density[max(0, o - win // 2): o + win // 2] += 1
        ond_n = znorm(onset_density)

        composite = rms_n + cen_n + ond_n  # higher = more chorus-like

        seg_scores = []
        for i in range(len(boundaries) - 1):
            s, e = boundaries[i], boundaries[i + 1]
            seg_scores.append(float(composite[s:e].mean()))

        # Normalize: positive scores = chorus, negative = verse
        median_score = float(np.median(seg_scores))
        target_high = section == "chorus"  # chorus = above median

        for i, score in enumerate(seg_scores):
            is_chorus = score > median_score
            if is_chorus == target_high:
                # Return first sample of this boundary
                frame_s = int(boundaries[i])
                return int(librosa.frames_to_samples(frame_s, hop_length=hop))

        return 0
    except Exception:
        return 0


def _detect_all_sections(y_mono: np.ndarray, n_sections: int = 7) -> list:
    """
    Detect structural sections in a track and classify each as:
      intro / verse / chorus / drop / breakdown / bridge / outro

    Uses MFCC + chroma agglomerative segmentation, then characterises each
    segment by its RMS energy, spectral brightness and onset density.
    Returns a list of dicts:
      {start_s, end_s, label, energy, brightness, density,
       chroma_density, harmonic_flux, spectral_contrast, dominant_chroma}

    New fields (used by Arrangement Intelligence):
      chroma_density     : fraction of 12 pitch classes active above their mean (0-1)
      harmonic_flux      : mean frame-to-frame chroma change — measures harmonic movement
      spectral_contrast  : mean spectral contrast — high = melodic, low = noise/drums
      dominant_chroma    : index (0-11) of strongest pitch class in the section
    """
    try:
        hop = 512
        duration_s = len(y_mono) / SR

        # ── Feature extraction ────────────────────────────────────────────────
        mfcc   = librosa.feature.mfcc(y=y_mono, sr=SR, n_mfcc=13, hop_length=hop)
        chroma = librosa.feature.chroma_cqt(y=y_mono, sr=SR, hop_length=hop,
                                            bins_per_octave=36)
        feats = np.vstack([
            librosa.util.normalize(mfcc,   norm=2, axis=0),
            librosa.util.normalize(chroma, norm=2, axis=0),
        ])

        # ── Boundary detection ────────────────────────────────────────────────
        k = max(3, min(n_sections, int(duration_s / 25) + 2))
        boundaries    = librosa.segment.agglomerative(feats, k=k)
        boundary_times = librosa.frames_to_time(boundaries, sr=SR, hop_length=hop)
        boundary_times = np.append(boundary_times, duration_s)

        # Per-frame characterisation arrays
        rms_f  = librosa.feature.rms(y=y_mono, frame_length=2048, hop_length=hop)[0]
        cen_f  = librosa.feature.spectral_centroid(y=y_mono, sr=SR, hop_length=hop)[0]
        ons_f  = librosa.onset.onset_strength(y=y_mono, sr=SR, hop_length=hop)

        # ── Build raw section list ────────────────────────────────────────────
        raw = []
        for i in range(len(boundary_times) - 1):
            t_s = float(boundary_times[i])
            t_e = float(boundary_times[i + 1])
            if t_e - t_s < 2.0:
                continue
            f_s = min(librosa.time_to_frames(t_s, sr=SR, hop_length=hop), len(rms_f) - 1)
            f_e = min(librosa.time_to_frames(t_e, sr=SR, hop_length=hop), len(rms_f))
            if f_e <= f_s:
                continue

            # ── Richer features for Arrangement Intelligence ──────────────────
            chroma_seg = chroma[:, f_s:f_e]
            chroma_mean_per_class = chroma_seg.mean(axis=1)  # (12,)
            chroma_density = float(
                (chroma_mean_per_class > chroma_mean_per_class.mean()).sum() / 12.0
            )

            if chroma_seg.shape[1] > 1:
                chroma_diff = np.diff(chroma_seg, axis=1)
                harmonic_flux = float(np.abs(chroma_diff).mean())
            else:
                harmonic_flux = 0.0

            try:
                contrast_seg = librosa.feature.spectral_contrast(
                    y=y_mono[int(t_s * SR):int(t_e * SR) + 1], sr=SR, hop_length=hop
                )
                spectral_contrast = float(contrast_seg.mean())
            except Exception:
                spectral_contrast = 0.0

            dominant_chroma = int(chroma_mean_per_class.argmax())

            raw.append({
                'start_s':           t_s,
                'end_s':             t_e,
                'energy':            float(rms_f[f_s:f_e].mean()),
                'brightness':        float(cen_f[f_s:f_e].mean()),
                'density':           float(ons_f[f_s:f_e].mean()),
                'label':             'unknown',
                'chroma_density':    chroma_density,
                'harmonic_flux':     harmonic_flux,
                'spectral_contrast': spectral_contrast,
                'dominant_chroma':   dominant_chroma,
            })

        if not raw:
            return []

        # ── Normalise features to 0-1 within this track ───────────────────────
        for key in ('energy', 'brightness', 'density'):
            vals = np.array([s[key] for s in raw])
            lo, hi = vals.min(), vals.max()
            rng = hi - lo if hi - lo > 1e-9 else 1.0
            for i, s in enumerate(raw):
                s[key] = float((vals[i] - lo) / rng)

        # ── Classify ──────────────────────────────────────────────────────────
        total_dur = raw[-1]['end_s']
        for sec in raw:
            t_frac = sec['start_s'] / (total_dur + 1e-9)
            e, b, d = sec['energy'], sec['brightness'], sec['density']

            if   t_frac < 0.15 and e < 0.45:            label = 'intro'
            elif t_frac > 0.80 and e < 0.45:            label = 'outro'
            elif e > 0.65 and d > 0.55:                 label = 'chorus'
            elif e > 0.55 and b > 0.65:                 label = 'drop'
            elif e < 0.30 and d < 0.35:                 label = 'breakdown'
            elif e > 0.40:                              label = 'verse'
            else:                                       label = 'bridge'
            sec['label'] = label

        return raw

    except Exception as _e:
        print(f"      [section detect failed: {_e}]", flush=True)
        return []


def _score_section_pair(vox_sec: dict, beat_sec: dict) -> float:
    """
    Score compatibility between a vocal section and a beat section for arrangement.
    Returns 0.0 (incompatible) to 1.0 (perfect match).

    Features:
      - Energy compatibility (0.30): target 0.5-0.7 vocal/beat energy ratio
      - Density match (0.20): syllable rate should match rhythmic complexity
      - Spectral contrast complement (0.15): high-contrast vocal + high-contrast beat = good
      - Harmonic flux compatibility (0.15): both sections should have similar harmonic movement rate
      - Duration ratio (0.10): penalise severe length mismatches
      - Chroma density complement (0.10): harmonically busy vocal + harmonically busy beat
    """
    # Energy compatibility: score highest when vocal is 50-70% of beat energy
    energy_ratio = (vox_sec['energy'] + 1e-9) / (beat_sec['energy'] + 1e-9)
    energy_score = float(np.clip(1.0 - abs(energy_ratio - 0.575) / 0.35, 0.0, 1.0))

    # Density match: both sparse or both dense
    density_score = 1.0 - float(np.clip(abs(vox_sec['density'] - beat_sec['density']), 0, 1))

    # Spectral contrast complement
    contrast_score = 1.0 - float(np.clip(
        abs(vox_sec.get('spectral_contrast', 0.5) - beat_sec.get('spectral_contrast', 0.5)) / 2.0,
        0, 1
    ))

    # Harmonic flux: similar harmonic movement rate sounds natural together
    flux_score = 1.0 - float(np.clip(
        abs(vox_sec.get('harmonic_flux', 0.1) - beat_sec.get('harmonic_flux', 0.1)) * 5.0,
        0, 1
    ))

    # Duration ratio: penalize >3x length mismatch
    vox_dur = vox_sec['end_s'] - vox_sec['start_s']
    beat_dur = beat_sec['end_s'] - beat_sec['start_s']
    dur_ratio = min(vox_dur, beat_dur) / (max(vox_dur, beat_dur) + 1e-9)
    duration_score = float(np.clip(dur_ratio, 0, 1))

    # Chroma density complement
    chroma_score = 1.0 - float(np.clip(
        abs(vox_sec.get('chroma_density', 0.5) - beat_sec.get('chroma_density', 0.5)),
        0, 1
    ))

    return (0.30 * energy_score + 0.20 * density_score + 0.15 * contrast_score +
            0.15 * flux_score + 0.10 * duration_score + 0.10 * chroma_score)


def _arrange_sections(vox_secs: list, beat_secs: list) -> list:
    """
    Intelligently assign vocal sections to beat section slots using:
    1. Narrative tiering — prevents high-energy chorus vocals on low-energy intros
    2. Energy arc — ensures the arrangement builds toward a peak then resolves
    3. Harmonic momentum — avoids jarring chroma jumps at section boundaries
    4. Hungarian algorithm within each tier for optimal within-tier assignment

    Returns list of (vox_sec, beat_sec) pairs in beat temporal order.
    Beat sections are NEVER reordered. Only vocal sections move.
    Returns None if not enough sections for rearrangement.
    """
    if len(vox_secs) < 2 or len(beat_secs) < 2:
        return None

    # Narrative tier map (hard constraint: +-1 tier allowed)
    _TIER = {'intro': 0, 'verse': 1, 'bridge': 1, 'chorus': 2, 'drop': 2,
             'breakdown': 3, 'outro': 4}

    n_vox  = len(vox_secs)
    n_beat = len(beat_secs)

    # Expected energy arc: rises to peak at 65% of song then resolves
    total_dur = beat_secs[-1]['end_s'] if beat_secs else 1.0

    def _ideal_energy(t_frac: float) -> float:
        if t_frac < 0.65:
            return 0.3 + 0.7 * (t_frac / 0.65) ** 1.5
        else:
            return 1.0 - 0.4 * ((t_frac - 0.65) / 0.35)

    # Build (n_vox x n_beat) cost matrix
    cost = np.zeros((n_vox, n_beat), dtype=np.float64)
    for vi, vs in enumerate(vox_secs):
        for bi, bs in enumerate(beat_secs):
            score = _score_section_pair(vs, bs)
            cost[vi, bi] = 1.0 - score

            # HARD BLOCK: tier violation > +-1
            vs_tier = _TIER.get(vs['label'], 2)
            bs_tier = _TIER.get(bs['label'], 2)
            if abs(vs_tier - bs_tier) > 1:
                cost[vi, bi] = 10.0

            # SOFT PENALTY: energy arc deviation
            if cost[vi, bi] < 10.0:
                beat_t_frac = bs['start_s'] / (total_dur + 1e-9)
                ideal_e = _ideal_energy(beat_t_frac)
                arc_penalty = float(np.clip(abs(vs['energy'] - ideal_e) * 0.5, 0.0, 0.4))
                cost[vi, bi] += arc_penalty

    # SOFT PENALTY: momentum — penalize consecutive low-energy vocal assignments
    low_energy_threshold = 0.35
    for bi in range(1, n_beat):
        for vi in range(n_vox):
            if cost[vi, bi] >= 10.0:
                continue
            if vox_secs[vi]['energy'] < low_energy_threshold:
                prev_col_costs = cost[:, bi - 1]
                cheapest_prev_vi = int(np.argmin(prev_col_costs))
                if (prev_col_costs[cheapest_prev_vi] < 10.0 and
                        vox_secs[cheapest_prev_vi]['energy'] < low_energy_threshold):
                    cost[vi, bi] += 0.3

    # Hungarian algorithm for optimal global assignment
    row_ind, col_ind = linear_sum_assignment(cost)

    # Build assignments list: (vox_sec, beat_sec) sorted by beat start time
    assignments = []
    for vi, bi in sorted(zip(row_ind, col_ind), key=lambda x: beat_secs[x[1]]['start_s']):
        assignments.append((vox_secs[vi], beat_secs[bi]))

    # Log assignments for debugging
    print("      [Arrangement Intelligence] Section assignment matrix:", flush=True)
    for vs, bs in assignments:
        score = _score_section_pair(vs, bs)
        vs_tier = _TIER.get(vs['label'], 2)
        bs_tier = _TIER.get(bs['label'], 2)
        print(f"        vox {vs['label']}(tier={vs_tier})@{vs['start_s']:.0f}s "
              f"-> beat {bs['label']}(tier={bs_tier})@{bs['start_s']:.0f}s  "
              f"score={score:.2f}", flush=True)

    return assignments


def _stitch_sections(vox: np.ndarray, assignments: list, sr: int = SR) -> np.ndarray:
    """
    Place vocal sections at beat section positions with equal-power crossfades.

    Critical: operates at SECTION level (15-60s segments), NOT onset level.
    Uses 200ms equal-power cosine fades at boundaries only.
    Never += into the buffer (causes phase cancellation) — uses sequential placement.

    Length handling:
      - Vocal longer than beat slot: truncate at slot end, 200ms fade-out
      - Vocal shorter than beat slot: place at slot start, silence fills remainder
        (creates natural instrumental break — musically correct, not a bug)

    Safety: if output RMS < 40% of input RMS, fall back to original vox (returns None).
    """
    out = np.zeros_like(vox, dtype=np.float32)

    for vox_sec, beat_sec in assignments:
        beat_start = int(beat_sec['start_s'] * sr)
        beat_end   = min(int(beat_sec['end_s'] * sr), len(vox))
        vox_start  = int(vox_sec['start_s'] * sr)
        vox_end    = min(int(vox_sec['end_s'] * sr), len(vox))

        seg = vox[vox_start:vox_end].copy()
        if len(seg) == 0:
            continue

        slot_len = beat_end - beat_start
        if slot_len <= 0:
            continue

        # Truncate if vocal is longer than the beat slot
        if len(seg) > slot_len:
            seg = seg[:slot_len]

        # Equal-power cosine fades at head and tail
        fade_n = min(int(0.20 * sr), len(seg) // 4)
        if fade_n > 0:
            t = np.linspace(0, np.pi / 2, fade_n)
            fade_in_curve  = np.sin(t) ** 2  # 0->1 equal power
            fade_out_curve = np.cos(t) ** 2  # 1->0 equal power
            seg[:fade_n]  *= fade_in_curve[:, np.newaxis]
            seg[-fade_n:] *= fade_out_curve[:, np.newaxis]

        # Place segment into output buffer (sequential, no +=)
        place_end = beat_start + len(seg)
        if place_end > len(out):
            seg = seg[:len(out) - beat_start]
            place_end = len(out)
        out[beat_start:place_end] = seg

    # Safety check: if output RMS is below 40% of input RMS, return None for fallback
    in_rms  = float(np.sqrt(np.mean(vox ** 2)) + 1e-9)
    out_rms = float(np.sqrt(np.mean(out ** 2)) + 1e-9)
    if out_rms < 0.40 * in_rms:
        print("      [Arrangement Intelligence] Safety check: output RMS too low "
              f"({out_rms:.4f} vs {in_rms:.4f}) — returning None for fallback.", flush=True)
        return None

    return np.nan_to_num(out.astype(np.float32))


def _arrangement_gain_curves(beat_secs: list, vox_secs: list,
                              n_samples: int) -> tuple:
    """
    Generate per-sample gain multipliers for the beat and vocal stems based
    on their detected sections.  Shapes the mix so it has real dynamics:

      Beat intro    → 1.00  (full beat before vocals enter)
      Beat verse    → 0.82  (step back: vocal owns this space)
      Beat chorus   → 1.00  (both full — big chorus moment)
      Beat drop     → 1.00  (EDM drop: maximum beat energy)
      Beat breakdown→ 0.55  (dramatic pull-back before the drop)
      Beat bridge   → 0.88
      Beat outro    → 0.90

      Vocal intro   → 0.85  (ease in — beat has just established itself)
      Vocal verse   → 1.00
      Vocal chorus  → 1.08  (push vocal forward on the hook)
      Vocal drop    → 1.05
      Vocal breakdown→ 0.92
      Vocal outro   → 0.88

    Returns (beat_gain, vox_gain) — float32 arrays of length n_samples.
    """
    _BEAT_GAIN = {
        'intro': 1.00, 'verse': 0.82, 'chorus': 1.00,
        'drop':  1.00, 'breakdown': 0.55, 'bridge': 0.88, 'outro': 0.90,
    }
    _VOX_GAIN = {
        'intro': 0.85, 'verse': 1.00, 'chorus': 1.08,
        'drop':  1.05, 'breakdown': 0.92, 'bridge': 1.00, 'outro': 0.88,
    }

    beat_gain = np.ones(n_samples, dtype=np.float32)
    vox_gain  = np.ones(n_samples, dtype=np.float32)

    for sec in beat_secs:
        s = int(sec['start_s'] * SR)
        e = min(int(sec['end_s'] * SR), n_samples)
        if e > s:
            beat_gain[s:e] = _BEAT_GAIN.get(sec['label'], 0.90)

    for sec in vox_secs:
        s = int(sec['start_s'] * SR)
        e = min(int(sec['end_s'] * SR), n_samples)
        if e > s:
            vox_gain[s:e] = _VOX_GAIN.get(sec['label'], 1.00)

    # Smooth every discontinuity with a 200 ms linear ramp to avoid clicks
    fade_n = int(0.20 * SR)
    for arr in (beat_gain, vox_gain):
        jumps = np.where(np.abs(np.diff(arr)) > 0.02)[0]
        for j in jumps:
            s = max(0, j - fade_n // 2)
            e = min(n_samples, j + fade_n // 2)
            if e > s + 1:
                arr[s:e] = np.linspace(arr[s], arr[e - 1], e - s)

    return beat_gain, vox_gain


def _groove_quantize(vox: np.ndarray, inst_mono: np.ndarray,
                     bpm: float, strength: float = 0.35) -> np.ndarray:
    """
    DISABLED: overlapping segment operations cause clicks throughout the mix.
    The overlap between fade-out zeroing and += seg_write creates gaps and
    double-writes when onset ranges overlap (which they always do on dense vocals).
    """
    return vox
    # Original broken implementation below — do not re-enable without rewrite:
    """
    Vocal groove quantization: nudge vocal syllable onsets toward the beat grid.

    Rappers and singers are naturally slightly ahead or behind the beat.
    This function detects each vocal onset, finds the nearest beat or 8th-note
    subdivision, and applies a small time shift to pull it toward the grid.

    'strength' (0-1): 0 = no quantization, 1 = hard snap to grid
    0.35 = subtle tightening (feels tighter but retains natural feel)

    Algorithm:
      1. Detect vocal onset positions (sample-accurate)
      2. Compute the beat grid and 8th-note subdivisions from inst_mono
      3. For each onset, find the nearest grid position
      4. If offset < ±50ms (real timing, not gross error), apply time shift
      5. Reconstruct audio by shifting segments between onsets
    """
    if not HAS_PYRUBBERBAND:
        return vox  # no way to do fine-grain shifts without rubberband

    try:
        hop = 256  # smaller hop for precise onset detection
        vox_mono = _to_mono(vox)

        # Detect beat grid from instrumental
        _, beats = librosa.beat.beat_track(y=inst_mono, sr=SR,
                                           hop_length=hop, units="samples")
        if len(beats) < 2:
            return vox

        # Build 8th-note grid (subdivide each beat by 2)
        beat_period = float(np.median(np.diff(beats)))
        eighth_period = beat_period / 2.0
        grid = []
        for b in beats:
            grid.append(int(b))
            eighth = int(b + eighth_period)
            if eighth < len(vox_mono):
                grid.append(eighth)
        grid = sorted(set(grid))

        # Detect vocal onsets
        onset_samp = librosa.onset.onset_detect(
            y=vox_mono, sr=SR, hop_length=hop, units="samples",
            backtrack=True)

        if len(onset_samp) == 0:
            return vox

        # Compute nudge amounts
        max_nudge_ms = 50.0
        max_nudge_samp = int(max_nudge_ms * SR / 1000)
        nudges = {}  # onset_idx → nudge_samples

        for ons in onset_samp:
            nearest_grid = min(grid, key=lambda g: abs(g - ons))
            raw_offset = nearest_grid - ons
            if abs(raw_offset) <= max_nudge_samp:
                nudge = int(raw_offset * strength)
                nudges[int(ons)] = nudge

        if not nudges:
            return vox

        # Apply nudges: shift audio segments with crossfade to prevent clicks
        result = vox.copy()
        onset_list = sorted(nudges.keys())
        xfade = max(64, int(SR * 0.003))  # 3ms crossfade window

        for i, ons in enumerate(onset_list):
            nudge = nudges[ons]
            if nudge == 0:
                continue
            seg_end = onset_list[i + 1] if i + 1 < len(onset_list) else len(vox)
            seg_len = seg_end - ons

            if seg_len < xfade * 2:
                continue

            new_start = ons + nudge
            new_start = max(0, min(new_start, len(vox) - seg_len))

            # Extract segment from original
            seg = vox[ons:ons + seg_len].copy()

            # Crossfade ramp for smooth transition (prevents click at boundaries)
            fade_in  = np.linspace(0, 1, xfade, dtype=np.float32)
            fade_out = np.linspace(1, 0, xfade, dtype=np.float32)

            # Fade out the old position in result
            result[ons:ons + xfade] = (result[ons:ons + xfade].T * fade_out).T
            result[ons + xfade:seg_end] = 0.0

            # Fade in the new position
            write_end = min(new_start + seg_len, len(result))
            write_len = write_end - new_start
            if write_len <= 0:
                continue
            seg_write = seg[:write_len].copy()
            seg_write[:xfade] = (seg_write[:xfade].T * fade_in).T
            result[new_start:write_end] = (
                result[new_start:write_end] + seg_write).astype(np.float32)

        return result.astype(np.float32)

    except Exception as e:
        print(f"      [Groove quantize failed ({e}), skipping]", flush=True)
        return vox


def _beat_align(inst_mono: np.ndarray, vox_stretched_mono: np.ndarray) -> tuple:
    """
    Align the first vocal onset to the nearest measure boundary in the instrumental.

    Uses the ACTUAL STRETCHED VOCAL STEM to detect when the singer first comes in,
    then aligns that moment to the nearest 4-beat measure start in the instrumental.
    This is fundamentally more reliable than using the full original track's beat
    positions (which may reflect a drum intro or silence before the singer starts).

    Returns (vox_prepend_samples, inst_prepend_samples). One will always be 0.
    """
    try:
        # Detect beat grid in the instrumental
        _, beats_inst = librosa.beat.beat_track(y=inst_mono, sr=SR, units="samples")
        if len(beats_inst) < 4:
            return 0, 0

        # Find the first strong vocal onset in the STRETCHED stem
        # backtrack=True snaps to the energy onset, not the peak
        onset_samples = librosa.onset.onset_detect(
            y=vox_stretched_mono, sr=SR,
            hop_length=512, units="samples",
            backtrack=True,
        )
        if len(onset_samples) == 0:
            return 0, 0
        vox_onset_s = int(onset_samples[0])

        # Find the instrumental beat closest to where the vocal first comes in
        nearest_idx = int(np.argmin(np.abs(beats_inst - vox_onset_s)))

        # Snap to measure boundary (nearest multiple of 4 beats) for phrasing
        measure_idx = round(nearest_idx / 4) * 4
        measure_idx = min(measure_idx, len(beats_inst) - 1)
        target_beat = int(beats_inst[measure_idx])

        offset = target_beat - vox_onset_s

        if offset >= 0:
            return int(offset), 0   # prepend silence to vocal
        else:
            return 0, int(-offset)  # prepend silence to instrumental
    except Exception:
        return 0, 0


# ── Adaptive Parameter Analysis ───────────────────────────────────────────────

def _analyze_vocal_stem(vox: np.ndarray) -> dict:
    """
    Derive mixing parameters from the actual vocal stem content.
    Returns a dict of adaptive settings.
    """
    mono = _to_mono(vox)

    # Compute per-frame RMS (40ms frames)
    win = int(SR * 0.040)
    frames = librosa.util.frame(mono, frame_length=win, hop_length=win // 2)
    frame_rms = np.sqrt(np.mean(frames ** 2, axis=0) + 1e-12)
    frame_rms_db = 20 * np.log10(frame_rms + 1e-12)

    # Noise floor: 10th percentile of frame RMS → gate 8 dB above it
    noise_floor_db = float(np.percentile(frame_rms_db, 10))
    gate_thresh_db = float(np.clip(noise_floor_db + 8.0, -60.0, -20.0))

    # Dynamic range: p90 - p10 of active frames
    active_mask = frame_rms_db > noise_floor_db + 6
    if active_mask.sum() > 10:
        dyn_range = float(np.percentile(frame_rms_db[active_mask], 90) -
                          np.percentile(frame_rms_db[active_mask], 10))
    else:
        dyn_range = 12.0

    # Compressor: more compression if dynamic range > 18 dB
    if dyn_range > 22:
        comp_ratio, comp_thresh = 5.0, -20.0
    elif dyn_range > 14:
        comp_ratio, comp_thresh = 3.5, -18.0
    else:
        comp_ratio, comp_thresh = 2.5, -16.0

    return {
        "gate_thresh_db":   gate_thresh_db,
        "comp_ratio":       comp_ratio,
        "comp_thresh_db":   comp_thresh,
        "dynamic_range_db": dyn_range,
        "noise_floor_db":   noise_floor_db,
    }


# ── System 2: Stem Quality Adaptive Processing ────────────────────────────────
# Measure actual separation quality and right-size every processing parameter
# to match the stem — high-bleed stems get aggressive cleanup; clean stems get
# lighter touch to preserve natural dynamics.

def _assess_stem_quality(vox: np.ndarray, inst: np.ndarray) -> dict:
    """
    Measure bleed, SNR, and vocal confidence of the separated vocal stem.
    Returns a dict with raw metrics + recommended _bleed_params adjustments.
    """
    vox_m = _to_mono(vox)

    def _band_rms(y, lo, hi):
        nyq = SR / 2.0
        lo_c = max(lo / nyq, 1e-4)
        hi_c = min(hi / nyq, 1.0 - 1e-4)
        sos = butter(4, [lo_c, hi_c], btype="band", output="sos")
        return float(np.sqrt(np.mean(sosfilt(sos, y) ** 2) + 1e-12))

    vox_mid_rms   = _band_rms(vox_m, 300, 5000)
    hihat_rms     = _band_rms(vox_m, 6000, 14000)
    bleed_ratio   = hihat_rms / (vox_mid_rms + 1e-9)

    # SNR: ratio of 75th-percentile frame RMS to 10th-percentile (noise floor)
    win        = int(SR * 0.04)
    frames     = librosa.util.frame(vox_m, frame_length=win, hop_length=win // 2)
    frms       = np.sqrt(np.mean(frames ** 2, axis=0) + 1e-12)
    noise_flr  = float(np.percentile(frms, 10))
    signal_lvl = float(np.percentile(frms, 75))
    snr_db     = float(20 * np.log10(signal_lvl / (noise_flr + 1e-12)))

    # Vocal confidence: spectral crest of the vocal stem (higher = more tonal/harmonic)
    S     = np.abs(librosa.stft(vox_m[:SR * 30], n_fft=2048))
    crest = float(S.max(axis=0).mean() / (S.mean(axis=0).mean() + 1e-9))

    # Map metrics → recommended processing parameters
    if bleed_ratio > 0.25:     # heavy bleed (e.g. Demucs on complex EDM)
        wiener_floor   = 0.08
        nr_strength    = 0.40
        bleed_level    = "high"
    elif bleed_ratio > 0.12:   # medium bleed
        wiener_floor   = 0.14
        nr_strength    = 0.25
        bleed_level    = "medium"
    else:                       # clean stem (MDX-Net upgrade successful)
        wiener_floor   = 0.20
        nr_strength    = 0.15
        bleed_level    = "low"

    # Low vocal confidence → cap FET ratio to avoid crushing already-thin vocal
    max_fet_ratio = 2.5 if crest < 6 else 4.0

    return {
        "bleed_ratio":      round(bleed_ratio, 4),
        "snr_db":           round(snr_db, 1),
        "vocal_confidence": round(crest, 2),
        "bleed_level":      bleed_level,
        "needs_denoiser":   snr_db < 20,
        "recommended": {
            "wiener_mask_floor":    wiener_floor,
            "noisereduce_strength": nr_strength,
            "max_fet_ratio":        max_fet_ratio,
        },
    }


def _spectral_overlap(vox_mono: np.ndarray, inst_mono: np.ndarray,
                       n_fft: int = 2048, hop: int = 512) -> float:
    """
    Measure frequency-band overlap between vocal and instrumental.
    Returns 0.0 (no overlap) to 1.0 (identical spectrum).
    Used to set adaptive sidechain depth.
    """
    S_v = np.abs(librosa.stft(vox_mono[:SR * 30], n_fft=n_fft, hop_length=hop))
    S_i = np.abs(librosa.stft(inst_mono[:SR * 30], n_fft=n_fft, hop_length=hop))

    # Normalize and compare in log-frequency bands
    v_band = S_v.mean(axis=1)
    i_band = S_i.mean(axis=1)
    v_norm = v_band / (v_band.sum() + 1e-9)
    i_norm = i_band / (i_band.sum() + 1e-9)

    # Overlap = minimum of the two probability distributions
    return float(np.minimum(v_norm, i_norm).sum())


# ── Vocal Processing ──────────────────────────────────────────────────────────

def _pitch_correct(vox_mono: np.ndarray, target_root: int, target_mode: str,
                   strength: float = 0.65) -> np.ndarray:
    """
    Monophonic pitch correction: snap the vocal to the nearest scale degree
    of the target key.  Applied BEFORE time-stretch so stretch artifacts don't
    interact with pitch-correction artifacts.

    Algorithm:
      1. Detect F0 per frame via PYIN (most robust for monophonic vocals)
      2. For each voiced frame, find the nearest chromatic scale note in the
         target key (chromatic within ±50 cents of scale tones)
      3. Compute the cents deviation from the nearest scale note
      4. Apply proportional pitch shift per frame (strength=0.65 → 65% pull)
         — leaves some natural expression; full 1.0 sounds robotic
      5. Reconstruct audio with pyrubberband per-frame shift (if available)

    Falls back to no-op if PYIN fails or pyrubberband not installed.
    """
    if not HAS_PYRUBBERBAND:
        return vox_mono  # can't do per-frame shift without rubberband

    try:
        # Scale degrees for major/minor (semitone offsets from root)
        major_scale = [0, 2, 4, 5, 7, 9, 11]
        minor_scale = [0, 2, 3, 5, 7, 8, 10]
        scale_degrees = major_scale if target_mode == "major" else minor_scale
        # All chromatic scale notes in the key (all octaves)
        scale_notes = set((target_root + d) % 12 for d in scale_degrees)

        hop = 512
        f0, voiced_flag, _ = librosa.pyin(
            vox_mono, fmin=60, fmax=1200, sr=SR, hop_length=hop, fill_na=None)

        if f0 is None or voiced_flag is None:
            return vox_mono

        # Frame duration in samples
        frame_samples = hop
        n_frames = len(f0)
        out = vox_mono.copy()

        i = 0
        while i < n_frames:
            # Find runs of voiced frames for efficient batch processing
            if not voiced_flag[i] or not np.isfinite(f0[i]) or f0[i] <= 0:
                i += 1
                continue

            # Detect the run length of consecutive voiced frames
            j = i
            while j < n_frames and voiced_flag[j] and np.isfinite(f0[j]) and f0[j] > 0:
                j += 1

            # Compute average pitch for the run
            run_f0 = f0[i:j]
            avg_hz = float(np.mean(run_f0))

            # Convert Hz to MIDI note
            midi_note = 12 * np.log2(avg_hz / 440.0) + 69
            chroma = int(round(midi_note)) % 12

            # Find nearest scale note
            best_dist = 12
            for note in scale_notes:
                dist = (note - chroma + 6) % 12 - 6  # wrapped semitone distance
                if abs(dist) < abs(best_dist):
                    best_dist = dist

            # Cents deviation (positive = too sharp, negative = too flat)
            cents_off = (midi_note - (round(midi_note - best_dist))) * 100 % 100
            if cents_off > 50:
                cents_off -= 100

            # Only correct if deviation is significant (>8 cents) to avoid killing vibrato
            if abs(cents_off) > 8:
                correction_semitones = -(cents_off / 100.0) * strength

                s = i * frame_samples
                e = min(j * frame_samples, len(out))
                segment = out[s:e].astype(np.float32)

                # Apply micro pitch shift to this segment
                corrected = rb.pitch_shift(segment, SR, correction_semitones,
                                           rbargs={'-3': ''})
                out[s:e] = corrected[:e - s].astype(np.float32)

            i = j

        return out.astype(np.float32)

    except Exception as e:
        print(f"      [Pitch correction failed ({e}), skipping]", flush=True)
        return vox_mono


def _breath_reduce(vox_ch: np.ndarray, reduction_db: float = 8.0) -> np.ndarray:
    """
    Reduce breath noise between vocal phrases without gating on the voice itself.

    Breath noise has two characteristics that distinguish it from voiced speech:
      1. High spectral flatness (noise-like, not harmonic)
      2. Low mid-frequency energy (no F1/F2 vowel formants in 300-2500 Hz)

    Frames that are both flat AND mid-frequency-quiet are attenuated.
    This avoids gating on consonants (which are also noisy but have more energy)
    or the beginning of syllables (which have mid-frequency energy rising).

    Not a hard gate — uses smooth gain to prevent pumping.
    """
    hop = 1024
    n_fft = 2048
    vox_mid = vox_ch.mean(axis=0).astype(np.float32)

    # Compute frame-level features
    S = np.abs(librosa.stft(vox_mid, n_fft=n_fft, hop_length=hop))
    freqs = librosa.fft_frequencies(sr=SR, n_fft=n_fft)

    # Spectral flatness per frame (0=pure tone, 1=noise)
    flatness = librosa.feature.spectral_flatness(S=S)[0]

    # Mid-band energy (300-2500 Hz) per frame
    mid_mask = (freqs >= 300) & (freqs < 2500)
    mid_energy = S[mid_mask].mean(axis=0)

    # Normalize mid energy 0-1
    mid_norm = mid_energy / (mid_energy.max() + 1e-9)

    # Breath probability: high flatness AND low mid energy → breath
    breath_prob = flatness * (1.0 - mid_norm)
    breath_prob = gaussian_filter1d(breath_prob.astype(np.float64), sigma=3).astype(np.float32)

    # Gain: 1.0 where voice, reduction where breath
    reduction_lin = 10 ** (-reduction_db / 20.0)
    frame_gain = 1.0 - (1.0 - reduction_lin) * np.clip(breath_prob, 0, 1)

    # Interpolate frame gains to sample resolution
    x_frames = np.arange(len(frame_gain), dtype=np.float64) * hop
    x_samp   = np.arange(len(vox_mid), dtype=np.float64)
    gain_samp = interp1d(
        x_frames, frame_gain, kind="linear",
        bounds_error=False, fill_value=(frame_gain[0], frame_gain[-1])
    )(x_samp).astype(np.float32)

    result = np.zeros_like(vox_ch)
    for c in range(vox_ch.shape[0]):
        result[c] = (vox_ch[c] * gain_samp).astype(np.float32)

    return result.astype(np.float32)


def _consonant_enhance(vox_ch: np.ndarray, boost_db: float = 3.0,
                        lo_hz: float = 4000.0, hi_hz: float = 9000.0,
                        fast_ms: float = 0.8, slow_ms: float = 30.0) -> np.ndarray:
    """
    Consonant transient enhancement: boost "t", "d", "k", "s" attack transients
    in the 4-9 kHz band WITHOUT boosting sustained hi-hat or cymbal noise.

    The key insight: consonants are impulsive (fast attack, rapid decay < 30ms).
    Hi-hats and cymbals are sustained (slow attack or sustained envelope).
    Dual envelope detection isolates transient events from sustained noise.

    This technique gives rap/vocal clarity without adding harshness.
    """
    sos_bp = butter(4, [lo_hz / (SR / 2), min(hi_hz / (SR / 2), 0.999)],
                    btype="band", output="sos")
    result = vox_ch.copy()

    # Use mono sum for detection
    mono = vox_ch.mean(axis=0).astype(np.float64)
    band = sosfilt(sos_bp, mono)

    # Dual envelope: fast = transients, slow = sustained
    # Vectorized with two-pass scipy lfilter (attack → release), same as _transient_shape.
    # Replaces O(n) Python loop — 50-100x faster on 3-5M sample signals.
    a_fast = np.exp(-1.0 / (SR * fast_ms / 1000.0))
    a_slow = np.exp(-1.0 / (SR * slow_ms / 1000.0))
    a_rel  = np.exp(-1.0 / (SR * 40.0   / 1000.0))  # shared release

    from scipy.signal import lfilter as _sc_lf
    rect = np.abs(band).astype(np.float64)
    # Attack pass + release pass per envelope (same pattern as _transient_shape._env_follow)
    _acc  = np.maximum.accumulate(rect) * 0.01
    fast_atk = _sc_lf([1.0 - a_fast], [1.0, -a_fast], rect)
    fast_env = _sc_lf([1.0 - a_rel], [1.0, -a_rel],
                       np.maximum(fast_atk, _acc)).astype(np.float64)
    slow_atk = _sc_lf([1.0 - a_slow], [1.0, -a_slow], rect)
    slow_env = _sc_lf([1.0 - a_rel], [1.0, -a_rel],
                       np.maximum(slow_atk, _acc)).astype(np.float64)

    # Transient mask: where fast >> slow = consonant burst
    diff = fast_env - slow_env
    diff_norm = np.clip(diff / (slow_env + 1e-9), 0, 3.0) / 3.0  # 0-1

    # Gain: boost only during transient events
    boost_lin = 10 ** (boost_db / 20.0)
    gain = (1.0 + (boost_lin - 1.0) * diff_norm).astype(np.float32)

    # Apply gain to all channels (vectorized broadcast)
    result = (result * gain[np.newaxis, :]).astype(np.float32)
    return result


def _harmonic_excite(audio_ch: np.ndarray, crossover_hz: float = 3000.0,
                     drive: float = 2.0, mix_level: float = 0.12) -> np.ndarray:
    """
    Aphex Aural Exciter-style harmonic excitement.

    Correct algorithm (based on research into Aphex patents):
      1. HP filter above crossover to isolate high-frequency content
      2. Normalize → saturate (asymmetric tube style = even+odd harmonics) → restore gain
      3. Subtract original HP band from saturated signal → ONLY new harmonic content
      4. Re-HP the harmonics (remove any low-freq saturation artifacts)
      5. Mix new harmonics at mix_level under dry signal

    Key improvement over simple saturation+HP: subtracting the original preserves
    the original signal entirely while adding only the new harmonics on top.

    audio_ch: (n_channels, n_samples) float32
    """
    # Vectorized over all channels (axis=1 = samples axis for (ch, samples) layout)
    sos_hp = butter(4, crossover_hz / (SR / 2.0), btype="high", output="sos")
    audio_f64 = audio_ch.astype(np.float64)
    hp_band = sosfilt(sos_hp, audio_f64, axis=1)            # (channels, samples)
    peak = np.max(np.abs(hp_band), axis=1, keepdims=True) + 1e-9
    hp_norm = hp_band / peak
    saturated = np.where(hp_norm > 0,
                         np.tanh(hp_norm * drive * 0.8),
                         np.tanh(hp_norm * drive * 1.2)) * peak
    harmonics_only = sosfilt(sos_hp, saturated - hp_band, axis=1)
    return (audio_f64 + harmonics_only * mix_level).astype(np.float32)


def _crepe_pitch_correct(vox_ch: np.ndarray) -> np.ndarray:
    """
    Global tuning correction: detect if vocalist is systematically sharp or flat
    and apply a single tiny pitch-shift (up to ±20 cents) to centre the tuning.

    Tries CREPE (neural, most robust) first, falls back to PYIN median analysis.
    Both approaches only detect the GLOBAL offset — no frame-by-frame correction,
    so there is zero risk of the robotic/vocoder artifacts we saw with per-note PYIN.

    5-cent dead zone: any offset below 5 cents (1/20 semitone) is left alone.
    """
    def _apply_global_cents(voiced_f, label):
        if len(voiced_f) < 30:
            return None
        cents_off = [1200.0 * np.log2(f / (440.0 * 2 ** (round(12 * np.log2(max(f, 1e-6) / 440.0)) / 12)))
                     for f in voiced_f if f > 0]
        if not cents_off:
            return None
        median_c = float(np.median(cents_off))
        if abs(median_c) < 5.0:
            print(f"      {label} tuning: {median_c:+.1f} ¢ offset — within 5¢ tolerance, skipping",
                  flush=True)
            return None
        shift_st = -float(np.clip(median_c, -20.0, 20.0)) / 100.0
        print(f"      {label} tuning: median {median_c:+.1f} ¢ → correcting {shift_st:+.4f} st",
              flush=True)
        if HAS_PYRUBBERBAND:
            return rb.pitch_shift(vox_ch.astype(np.float32), SR, shift_st,
                                   rbargs={'-3': ''}).astype(np.float32)
        return None

    # ── Try CREPE first (needs tensorflow; silent fallback if unavailable) ─────
    try:
        import crepe as _crepe
        clip = vox_ch[:SR * 30].astype(np.float32)
        _, frequency, confidence, _ = _crepe.predict(
            clip, SR, viterbi=True, verbose=0, model_capacity='tiny')
        voiced_f = frequency[(confidence > 0.80) & (frequency > 60) & (frequency < 1100)]
        result = _apply_global_cents(voiced_f, "CREPE")
        if result is not None:
            return result
        return vox_ch
    except ImportError:
        pass  # No tensorflow — fall through to PYIN
    except Exception as _e:
        print(f"      [CREPE error: {_e} — falling back to PYIN global tuning]", flush=True)

    # ── PYIN fallback: median F0 over voiced frames ──────────────────────────
    # Safe for global offset — we're not doing note-level correction, just
    # detecting the systematic sharp/flat bias and removing it.
    try:
        clip = vox_ch[:SR * 20].astype(np.float32)  # 20s sample
        f0, voiced_flag, _ = librosa.pyin(
            clip, fmin=60, fmax=1100, sr=SR, hop_length=512, fill_na=None)
        if voiced_flag is not None:
            voiced_f = f0[voiced_flag & (f0 > 60) & (f0 < 1100)] \
                       if f0 is not None else np.array([])
        else:
            voiced_f = f0[~np.isnan(f0)] if f0 is not None else np.array([])
        result = _apply_global_cents(voiced_f.tolist() if len(voiced_f) else [], "PYIN")
        if result is not None:
            return result
    except Exception as _e:
        print(f"      [PYIN global tuning skipped: {_e}]", flush=True)

    return vox_ch


def _clean_vocal(vox_mono: np.ndarray) -> np.ndarray:
    """
    Spectral gating with noisereduce to remove Demucs bleed artifacts.
    Non-stationary mode handles music-like residue better than stationary.
    """
    return nr.reduce_noise(
        y=vox_mono, sr=SR,
        stationary=False,
        prop_decrease=0.15,   # light touch — oracle Wiener+harmonic already removed bleed; avoid artifacts
        n_fft=2048,
    ).astype(np.float32)


def _deess(vox: np.ndarray, threshold_db: float = -22.0,
           cutoff_hz: float = 6500.0, max_reduction_db: float = 7.0) -> np.ndarray:
    """
    Split-band de-esser: detect sibilance in the 6.5 kHz+ band and apply
    gain reduction ONLY to that frequency band. The rest of the signal is untouched.

    Split-band (vs wideband) is the professional standard:
      - Wideband de-essers: whole vocal ducks when a sibilant fires — audible pump
      - Split-band: only 6.5 kHz+ is reduced — transparent, surgical

    Detection: mono sum of sibilance band → frame RMS → gain curve
    Reduction: applied per-channel, only to HP band; LP band passes through clean

    Placed BEFORE compression to prevent sibilance pumping the compressor.
    """
    nyq = SR / 2.0
    sos_hi = butter(4, cutoff_hz / nyq, btype="high", output="sos")
    sos_lo = butter(4, cutoff_hz / nyq, btype="low",  output="sos")

    mono = _to_mono(vox)

    # Sidechain detection: sibilance band of mono sum
    sib = sosfilt(sos_hi, mono).astype(np.float32)

    # Vectorized RMS envelope (replaces O(n_frames) Python loop)
    win = max(1, int(SR * 0.005))   # 5ms frames
    hop = win // 2
    n = len(sib)
    rms_frames = librosa.feature.rms(y=sib, frame_length=win, hop_length=hop)[0]
    n_frames = len(rms_frames)
    env_db = (20 * np.log10(rms_frames.astype(np.float32) + 1e-12))

    # Gain reduction curve
    over_thresh = env_db - threshold_db
    gain_db = np.clip(-over_thresh * 0.6, -max_reduction_db, 0.0)
    gain_linear = (10 ** (gain_db / 20.0)).astype(np.float32)

    # Interpolate to sample resolution
    x_frames = np.arange(n_frames, dtype=np.float64) * hop
    x_samp   = np.arange(n, dtype=np.float64)
    gain_samp = interp1d(
        x_frames, gain_linear, kind="linear",
        bounds_error=False, fill_value=(gain_linear[0], gain_linear[-1])
    )(x_samp).astype(np.float32)

    # Apply gain ONLY to sibilance band — vectorized over both channels at once
    out_f64 = vox.astype(np.float64)
    hi_all = sosfilt(sos_hi, out_f64, axis=0)   # (samples, 2)
    lo_all = sosfilt(sos_lo, out_f64, axis=0)   # (samples, 2)
    hi_reduced = hi_all * gain_samp[:, np.newaxis]
    return (lo_all + hi_reduced).astype(np.float32)


def _dynamic_eq_vocal(vox_ch: np.ndarray) -> np.ndarray:
    """
    Dynamic EQ: reduce 3kHz harshness only when that band gets loud.

    Loud phrases (chest-voice belts, shouty rap) accumulate energy around 3kHz,
    which fatigues the ear. A static cut would dull quiet phrases. A dynamic cut
    (compressor keyed to the 3kHz band) only kicks in when it matters.

    Implementation: bandpass-split at 2.5-4.5kHz, compute RMS envelope,
    apply gain to the extracted mid band, recombine additively.
    Approach: result = ch + bp_band * (gain - 1.0) subtracts the excess.

    Research: 3-3.5kHz, Q=2.5-3, -2 to -4dB max, 10ms attack, 90ms release.
    """
    from scipy.signal import lfilter as _dq_lf
    nyq = SR / 2.0
    # Bandpass 2.5–4.5 kHz (harshness zone)
    sos_bp = butter(4, [2500.0 / nyq, min(4500.0 / nyq, 0.999)],
                    btype="band", output="sos")

    mono = vox_ch.mean(axis=0).astype(np.float64)
    bp_mono = sosfilt(sos_bp, mono)

    # Vectorized peak-follower envelope (10ms attack, 90ms release)
    a_atk = np.exp(-1.0 / (SR * 0.010))
    a_rel = np.exp(-1.0 / (SR * 0.090))
    rect = np.abs(bp_mono)
    env_atk = _dq_lf([1.0 - a_atk], [1.0, -a_atk], rect)
    env = _dq_lf([1.0 - a_rel], [1.0, -a_rel],
                  np.maximum(env_atk, np.maximum.accumulate(rect) * 0.01)).astype(np.float32)

    # Compute gain reduction (max -3dB, ratio ~2.5:1, dynamic threshold)
    # Threshold computed adaptively from the 90th percentile of the envelope
    thresh_lin = float(np.percentile(env[env > 1e-9], 80)) if np.any(env > 1e-9) else 1.0
    over = np.clip(env / (thresh_lin + 1e-12) - 1.0, 0.0, None)  # 0 when below threshold
    gain_db = np.clip(-over * 0.6 * 6.0, -3.0, 0.0)  # ratio ~2.5:1 max -3dB
    gain_samp = (10 ** (gain_db / 20.0)).astype(np.float32)

    # Apply: extract BP band, scale it, add the difference back
    # result = ch + bp_ch * (gain - 1.0)  → subtracts excess from the harsh band
    bp_all = sosfilt(sos_bp, vox_ch.astype(np.float64), axis=1).astype(np.float32)
    delta = bp_all * (gain_samp[np.newaxis, :] - 1.0)
    return (vox_ch + delta).astype(np.float32)


def _multiband_compress_vocal(vox_ch: np.ndarray, style: dict) -> np.ndarray:
    """
    4-band multiband compression modeled on Waves C6 / iZotope Neutron.

    Standard vocal crossover points:
      Band 1:   80 – 250 Hz  (body/mud) — gentle ratio, controls chest resonance
      Band 2:  250 – 2000 Hz (warmth/body) — moderate, most critical band
      Band 3: 2000 – 8000 Hz (presence/sibilance) — tightest, prevents harshness
      Band 4: 8000 – 20000 Hz (air) — very gentle, protects brightness

    Each band is: LP/HP filter pair → compress → recombine.
    Ratios scale with rap_score: rap needs tighter, more controlled high-mids.
    """
    rap = style.get("_rap_score", 0.5)

    # Band crossovers match iZotope Neutron defaults (research-backed):
    # Band 1:   80-400 Hz  — body/chest (post-HPF; removes boxiness)
    # Band 2:  400-2500 Hz — CORE INTELLIGIBILITY (protects formants/consonants)
    # Band 3: 2500-8000 Hz — presence/de-essing (tightest for sibilance control)
    # Band 4: 8000-20000Hz — air (very gentle; protects brightness)
    bands_def = [
        (80,    400,  -24.0, 3.0, 10.0, 150.0),
        (400,  2500,  -20.0, 2.0,  8.0, 120.0),   # gentlest — protects intelligibility
        (2500, 8000,  -18.0, float(np.interp(rap, [0,1], [3.5, 5.0])), 2.0, 60.0),
        (8000, 20000, -28.0, 2.0, 15.0, 300.0),
    ]

    n_ch, n_samp = vox_ch.shape
    out = np.zeros_like(vox_ch)

    for lo, hi, thresh, ratio, atk, rel in bands_def:
        # Bandpass: LP at hi then HP at lo
        nyq = SR / 2.0
        lo_norm = lo / nyq
        hi_norm = min(hi / nyq, 0.999)

        sos_lp = butter(4, hi_norm, btype="low",  output="sos")
        sos_hp = butter(4, lo_norm, btype="high", output="sos")

        band_ch = np.zeros_like(vox_ch)
        for c in range(n_ch):
            bp = sosfilt(sos_lp, sosfilt(sos_hp, vox_ch[c]))
            band_ch[c] = bp.astype(np.float32)

        # Compress each band channel with pedalboard
        band_board = Pedalboard([
            Compressor(threshold_db=thresh, ratio=ratio,
                       attack_ms=atk, release_ms=rel),
        ])
        compressed = band_board(band_ch.astype(np.float32), SR)
        out += compressed

    return out.astype(np.float32)


def _hpf_signal(audio_ch: np.ndarray, cutoff_hz: float, order: int = 4) -> np.ndarray:
    """
    Apply a high-pass filter to a (channels, samples) array.
    Used for the Abbey Road reverb return HPF trick.
    """
    sos = butter(order, cutoff_hz / (SR / 2), btype="high", output="sos")
    return sosfilt(sos, audio_ch, axis=1).astype(np.float32)


def _process_vocals(vox: np.ndarray, ratio: float, n_semitones: int,
                    params: dict, style=None,
                    target_root: int = 0, target_mode: str = "major",
                    bpm: float = 120.0) -> np.ndarray:
    """
    Minimal vocal pipeline — clarity over complexity.

    Previous 16-stage chain was stacking artifacts on top of AI-separation
    artifacts, creating smear and muddiness. Each removed stage is a stage
    that can't add distortion, phase issues, or timing smear.

    Chain (7 stages only):
      1. HPF 80 Hz + 3-node subtractive EQ
      2. Time-stretch + pitch-shift (pyrubberband R3)
      3. De-esser
      4. ONE compressor (style-adaptive)
      5. Noise gate
      6. Presence boost + air shelf
      7. Dry reverb (5-8% wet, short room)

    Removed vs old chain:
      - Breath reduction      (creates pumping artifacts between words)
      - Multiband compression (over-processing on AI stems)
      - Dynamic EQ            (stacks with comp artifacts)
      - NY parallel compression on vocal (smears transients)
      - Mid-band harmonic exciter (adds distortion to bleed)
      - Early reflections     (adds pre-echo that destroys flow)
      - Slap-back echo        (smears phrases into each other)
      - ADT / pitch-shift copies (chorus smear = "no flow" complaint)
      - Soft-knee entry compressor (redundant with de-esser)
    """
    if style is None:
        style = {
            "fet_ratio": 4.0, "fet_attack": 5.0, "fet_release": 80.0,
            "presence_db": 2.0, "presence_hz": 3500.0,
            "air_db": 1.5, "reverb_room": 0.10, "reverb_damp": 0.80, "reverb_wet": 0.06,
        }

    # (samples, 2) → (2, samples) for pedalboard / pyrubberband
    vox_ch = vox.T.astype(np.float32)

    # Stage 0: Spectral noise reduction — remove residual stem-separation bleed.
    # The Wiener mask handles bins where instrumental dominates.
    # noisereduce (prop_decrease=0.55) handles the noise floor left behind.
    # NaN guard: noisereduce can produce NaN in silent/zero regions (divide-by-zero
    # in spectral gating) — replace with 0 before any further processing.
    vox_ch = np.stack([
        np.nan_to_num(_clean_vocal(vox_ch[c]), nan=0.0, posinf=0.0, neginf=0.0)
        for c in range(vox_ch.shape[0])
    ], axis=0)

    # Stage 0.5: HPSS harmonic masking — removes hi-hat/percussive bleed.
    # Hi-hats are percussive (vertical in spectrogram); vocals are harmonic (horizontal).
    # margin=(1.0, 5.0): a bin must be 5x more percussive to be assigned to P mask.
    # Blend: 80% HPSS-cleaned + 20% original — preserves vocal consonants (t, k, f).
    # Community-validated (UVR): most effective hi-hat bleed removal on separated stems.
    try:
        n_fft_hpss = 2048
        hpss_blend = 0.80
        cleaned_ch = np.zeros_like(vox_ch)
        for c in range(vox_ch.shape[0]):
            D = librosa.stft(vox_ch[c].astype(np.float64), n_fft=n_fft_hpss)
            H, P = librosa.decompose.hpss(np.abs(D), kernel_size=31, margin=(1.0, 5.0))
            soft_mask = librosa.util.softmask(H, H + P, power=2)
            D_harm = D * soft_mask
            y_harm = librosa.istft(D_harm, length=vox_ch.shape[1]).astype(np.float32)
            cleaned_ch[c] = (y_harm * hpss_blend + vox_ch[c] * (1.0 - hpss_blend))
        vox_ch = np.nan_to_num(cleaned_ch, nan=0.0, posinf=0.0, neginf=0.0)
        print("      [Stage 0.5] HPSS harmonic mask applied.", flush=True)
    except Exception as _hpss_e:
        print(f"      [Stage 0.5 HPSS failed: {_hpss_e}]", flush=True)

    # Stage 0b — CREPE global tuning correction (only if library installed)
    for c in range(vox.shape[1]):
        vox[:, c] = _crepe_pitch_correct(vox[:, c])

    # Stage 1: HPF 80 Hz + subtractive EQ + hi-hat bleed roll-off
    # The 8kHz shelf always rolls off the hi-hat zone regardless of pitch shift.
    # Stem separation leaves 8-16kHz hi-hat residue that sounds "scratchy" after
    # compression and presence boost. -3dB at 8kHz (-1 oct per octave above) is
    # gentle enough to preserve sibilance (5-8kHz) while suppressing hi-hat bleed.
    pre_eq = Pedalboard([
        HighpassFilter(cutoff_frequency_hz=80.0),
        PeakFilter(cutoff_frequency_hz=300.0, gain_db=-5.0, q=1.2),  # mud (Demucs residue)
        PeakFilter(cutoff_frequency_hz=450.0, gain_db=-3.0, q=1.4),  # cardboard box
        PeakFilter(cutoff_frequency_hz=500.0, gain_db=-2.0, q=1.5),  # boxy
        HighShelfFilter(cutoff_frequency_hz=8000.0, gain_db=-3.0),   # hi-hat bleed roll-off
    ])
    vox_ch = pre_eq(vox_ch, SR).astype(np.float32)

    # Stage 2: time-stretch + pitch-shift (pyrubberband R3, formant-preserving)
    if n_semitones != 0:
        # Strip hi-hat bleed before shifting (8kHz keeps sibilance, removes 8-16kHz bleed)
        sos_lpf = butter(4, 8000.0 / (SR / 2), btype="low", output="sos")
        vox_ch = np.stack([
            sosfilt(sos_lpf, vox_ch[c]).astype(np.float32)
            for c in range(vox_ch.shape[0])
        ], axis=0)

    if HAS_PYRUBBERBAND and (abs(ratio - 1.0) > 0.005 or n_semitones != 0):
        stretched = []
        for c in range(vox_ch.shape[0]):
            y_s = rb.time_stretch(vox_ch[c], SR, ratio, rbargs={'-3': ''})
            if n_semitones != 0:
                y_s = rb.pitch_shift(y_s, SR, n_semitones, rbargs={'-3': ''})
            stretched.append(y_s)
        vox_ch = np.stack(stretched, axis=0).astype(np.float32)
    elif abs(ratio - 1.0) > 0.005 or n_semitones != 0:
        vox_ch = pb_time_stretch(
            vox_ch, SR,
            stretch_factor=ratio,
            pitch_shift_in_semitones=float(n_semitones),
        ).astype(np.float32)

    # Stage 3: de-esser (before compression — prevents sibilance pumping)
    vox_ch = _deess(vox_ch.T, threshold_db=-22.0).T.astype(np.float32)

    # Stage 4 + 5: ONE compressor + noise gate
    dyn_board = Pedalboard([
        Compressor(
            threshold_db=params["comp_thresh_db"],
            ratio=style["fet_ratio"],
            attack_ms=style["fet_attack"],
            release_ms=style["fet_release"],
        ),
        NoiseGate(
            threshold_db=params["gate_thresh_db"],
            ratio=2.0,       # soft expander — natural word-ending decay
            attack_ms=5.0,
            release_ms=150.0,
        ),
    ])
    vox_ch = dyn_board(vox_ch, SR).astype(np.float32)

    # Stage 5b: parallel tape saturation (adds warmth/harmonics without artifacts)
    # Now safe because bleed is removed by oracle Wiener + harmonic mask.
    # Saturation without bleed = harmonic richness; with bleed = mud/distortion.
    # Approach: soft-clip (tanh) at 20% wet — warms the compressed signal without smearing.
    SAT_DRIVE = 1.4   # drive factor — gentle overdrive into tanh
    SAT_WET   = 0.20  # 20% wet blend — adds body without muddying transients
    sat_sig = np.tanh(vox_ch * SAT_DRIVE) / SAT_DRIVE   # gain-compensated tanh
    vox_ch = ((1.0 - SAT_WET) * vox_ch + SAT_WET * sat_sig).astype(np.float32)

    # Stage 6: presence + air shelf
    # The +4dB shelf at 1500Hz was causing hi-mid harshness (2.5-6kHz too loud).
    # Vocals cut through via the spectral carve + narrow presence peak, not a
    # broad shelf. Use a narrow presence peak only + subtle air shelf.
    post_eq = Pedalboard([
        PeakFilter(cutoff_frequency_hz=style["presence_hz"],
                   gain_db=style["presence_db"], q=1.5),
        HighShelfFilter(cutoff_frequency_hz=10000.0, gain_db=style["air_db"]),
    ])
    vox_ch = post_eq(vox_ch, SR).astype(np.float32)

    # Stage 7: short dry reverb (5-8% wet, HPF'd return to avoid mud)
    reverb_board = Pedalboard([
        Reverb(room_size=style["reverb_room"], damping=style["reverb_damp"],
               wet_level=1.0, dry_level=0.0, width=0.7),
    ])
    reverb_wet = reverb_board(vox_ch, SR).astype(np.float32)
    reverb_wet = _hpf_signal(reverb_wet, cutoff_hz=500.0, order=4)
    wet = float(np.clip(style["reverb_wet"], 0.04, 0.09))
    vox_ch = (vox_ch + reverb_wet * wet).astype(np.float32)

    return vox_ch.T  # (samples, 2)


# ── Instrumental Processing ───────────────────────────────────────────────────

def _transient_shape(inst: np.ndarray,
                     attack_gain_db: float = 4.0,
                     sustain_gain_db: float = -2.0,
                     fast_ms: float = 1.0,
                     slow_ms: float = 80.0,
                     release_ms: float = 80.0) -> np.ndarray:
    """
    SPL Transient Designer-style transient shaping for the instrumental.

    Stem separation (Demucs/BS-Roformer) softens drum transients by 2-4 dB.
    This function restores attack definition by computing two envelope followers:
      - Fast follower (1 ms attack):  tracks transient peaks
      - Slow follower (20 ms attack): tracks sustained body
    The difference fast - slow identifies transient vs sustain content.
    Gain is applied proportionally to boost attack, reduce sustain.

    Operates per-channel to preserve stereo image.
    """
    def _env_follow(audio, attack_ms, rel_ms):
        from scipy.signal import lfilter
        rect = np.abs(audio).astype(np.float64)
        # Asymmetric envelope: fast attack lfilter on rectified signal,
        # then slow release lfilter as a second pass.
        # Two symmetric passes approximate the asymmetric behavior:
        # pass 1 = attack (fast peak tracking)
        # pass 2 = release (slow decay from peaks)
        a_atk = np.exp(-1.0 / (SR * attack_ms / 1000.0))
        a_rel = np.exp(-1.0 / (SR * rel_ms    / 1000.0))
        # Attack pass: one-pole LP on the rectified signal (tracks rises fast)
        env_atk = lfilter([1.0 - a_atk], [1.0, -a_atk], rect)
        # Release pass: one-pole LP on the envelope (holds peaks, decays slowly)
        env = lfilter([1.0 - a_rel], [1.0, -a_rel],
                      np.maximum(env_atk, np.maximum.accumulate(rect) * 0.01))
        return env.astype(np.float32)

    result = np.zeros_like(inst)
    att_lin = 10 ** (attack_gain_db / 20.0)
    sus_lin = 10 ** (sustain_gain_db / 20.0)

    # Frequency-weighted detection: kick band (60-200Hz) + snare band (150-6kHz)
    # gives more accurate transient identification vs using full-bandwidth signal
    nyq = SR / 2.0
    sos_kick  = butter(4, [60 / nyq, 200 / nyq],  btype="band", output="sos")
    sos_snare = butter(4, [150 / nyq, min(6000 / nyq, 0.999)], btype="band", output="sos")
    inst_mono_d = _to_mono(inst).astype(np.float64)
    kick_b  = sosfilt(sos_kick,  inst_mono_d)
    snare_b = sosfilt(sos_snare, inst_mono_d)
    detect  = (0.5 * kick_b + 0.5 * snare_b).astype(np.float32)

    fast_env_d = _env_follow(detect, fast_ms, release_ms)
    slow_env_d = _env_follow(detect, slow_ms, release_ms)

    total_d = fast_env_d + slow_env_d + 1e-12
    transient_mask = fast_env_d / total_d
    sustain_mask   = slow_env_d / total_d
    gain = (transient_mask * att_lin + sustain_mask * sus_lin).astype(np.float32)

    for c in range(inst.shape[1]):
        result[:, c] = (inst[:, c] * gain).astype(np.float32)

    return result.astype(np.float32)


def _adaptive_spectral_carve(inst: np.ndarray, vox: np.ndarray,
                              carve_db: float = 5.0,
                              smooth_sigma: float = 2.5,
                              carve_lo_hz: float = 200.0,
                              carve_hi_hz: float = 5000.0) -> np.ndarray:
    """
    Content-aware spectral carving using a Wiener soft-mask.

    For each time-frequency bin, computes how dominant the vocal is vs the
    beat, then reduces the beat in exactly those frequency slots.
    This replaces the old fixed-frequency EQ cut with a dynamic carve that
    adapts to whatever frequencies the vocal actually uses.

    carve_db:     max cut in beat where vocal is loudest (5 dB = perceptually clean)
    smooth_sigma: temporal smoothing to prevent pumping/zipper artifacts
    carve_lo_hz:  lower frequency boundary (default 200 Hz; extend to ~80 Hz for bass vocalists)
    carve_hi_hz:  upper frequency boundary (default 5000 Hz; extend to ~9kHz for sopranos)
    """
    n_fft, hop = 2048, 512
    inst_mono = _to_mono(inst)
    vox_mono  = _to_mono(vox)

    # Compute STFTs
    inst_stft = librosa.stft(inst_mono, n_fft=n_fft, hop_length=hop)
    vox_stft  = librosa.stft(vox_mono,  n_fft=n_fft, hop_length=hop)

    inst_mag = np.abs(inst_stft)
    inst_phase = np.angle(inst_stft)
    vox_mag  = np.abs(vox_stft)

    # Wiener soft-mask: how much of the combined power is vocal?
    vox_pow  = vox_mag  ** 2
    inst_pow = inst_mag ** 2
    vocal_mask = vox_pow / (vox_pow + inst_pow + 1e-10)  # 0 = no vocal, 1 = all vocal

    # Temporal smoothing (prevents audible pumping)
    vocal_mask = gaussian_filter1d(vocal_mask.astype(np.float64),
                                    sigma=smooth_sigma, axis=1).astype(np.float32)

    # Frequency weighting: carve the vocal intelligibility zone.
    # Range is content-adaptive (carve_lo_hz / carve_hi_hz) so:
    #   - Bass vocalists (F0 ~80-130 Hz): carve extends down to ~80 Hz
    #   - Standard voices: default 200-5000 Hz
    #   - Sopranos / high falsetto: extends up to ~9 kHz
    # Peak weight 2.0× at 3 kHz — ear-canal resonance + presence zone.
    freqs = librosa.fft_frequencies(sr=SR, n_fft=n_fft)
    freq_w = np.zeros(len(freqs), dtype=np.float32)
    _lo = carve_lo_hz
    _hi = carve_hi_hz
    _lo_ramp_end = min(max(_lo * 2.0, 300.0), 400.0)   # ramp from _lo to this freq
    for i, f in enumerate(freqs):
        if f < _lo or f > _hi:
            freq_w[i] = 0.0                             # no carve outside the active range
        elif f < _lo_ramp_end:
            # Gentle low-end ramp: avoids mud artifacts when carve extends below 200 Hz
            freq_w[i] = float((f - _lo) / max(_lo_ramp_end - _lo, 1.0) * 0.4)
        elif f < 600:
            t = float((f - _lo_ramp_end) / max(600.0 - _lo_ramp_end, 1.0))
            freq_w[i] = 0.4 + t * 0.4                  # ramp to 0.8 at 600 Hz
        elif f < 1000:
            freq_w[i] = 0.8 + (f - 600) / 400 * 0.2   # ramp to 1.0 at 1 kHz
        else:
            # 2-4 kHz: peak weight 2.0× at 3 kHz — vocal presence / cut-through zone.
            # Fletcher-Munson: at high volume bass masks this zone → carve deeper.
            w_peak = 1.0 + 1.0 * float(np.clip(
                1.0 - abs(np.log2(f / 3000)) * 1.5, 0, 1))  # peak 2.0× at 3 kHz
            # Taper toward 0 in the top 25% of the carve range (prevents harsh cut-off)
            if f > _hi * 0.75:
                taper = float(np.clip((_hi - f) / (_hi * 0.25), 0.0, 1.0))
                w_peak *= taper
            freq_w[i] = w_peak

    # Effective gain: 1.0 where vocal is absent, (1 - max_cut) where vocal is loud
    max_cut = 1.0 - 10 ** (-carve_db / 20.0)   # carve_db=5 → max_cut≈0.44
    effective_mask = 1.0 - max_cut * vocal_mask * freq_w[:, np.newaxis]

    # Pad/trim mask to match inst_stft time dimension
    if effective_mask.shape[1] != inst_stft.shape[1]:
        if effective_mask.shape[1] < inst_stft.shape[1]:
            pad = inst_stft.shape[1] - effective_mask.shape[1]
            effective_mask = np.pad(effective_mask, ((0,0),(0,pad)), mode='edge')
        else:
            effective_mask = effective_mask[:, :inst_stft.shape[1]]

    # Apply mask to both channels of the stereo instrumental
    result = np.zeros_like(inst)
    for c in range(inst.shape[1]):
        ch_stft = librosa.stft(inst[:, c], n_fft=n_fft, hop_length=hop)
        ch_mag   = np.abs(ch_stft)
        ch_phase = np.angle(ch_stft)
        carved_mag = ch_mag * effective_mask
        carved_stft = carved_mag * np.exp(1j * ch_phase)
        reconstructed = librosa.istft(carved_stft, hop_length=hop, length=len(inst))
        result[:, c] = reconstructed.astype(np.float32)

    # High shelf removed (was 4.5→2.0 dB): transient shaper (now working) already
    # restores instrumental presence. Shelf was contributing to High band excess.
    shelf = Pedalboard([HighShelfFilter(cutoff_frequency_hz=5500.0, gain_db=0.0)])
    result = shelf(result.T.astype(np.float32), SR).T.astype(np.float32)

    return result


def _parallel_compress(inst: np.ndarray) -> np.ndarray:
    """
    NY-style parallel compression: blend 30% heavily compressed signal with
    70% dry. Adds density and sustain without squashing kick/snare transients.
    """
    inst_ch = inst.T.astype(np.float32)
    from pedalboard import Gain
    crush = Pedalboard([
        Compressor(threshold_db=-24.0, ratio=8.0, attack_ms=30.0, release_ms=200.0),
        Gain(gain_db=9.0),   # makeup: bring crushed level up to match dry
    ])
    crushed = crush(inst_ch, SR).T.astype(np.float32)
    # 8% wet: was 20% which sounds dense at low volume but "smashed together" loud.
    # The crushed signal fills in transient gaps — at high SPL this makes every
    # hit blend into the next. 8% adds just enough glue without destroying punch.
    return (0.92 * inst + 0.08 * crushed).astype(np.float32)


def _parallel_compress_vocal(vox_ch: np.ndarray, rap_score: float = 0.5) -> np.ndarray:
    """
    NY-style parallel (New York) compression on vocals.

    Professional technique used on virtually every hip-hop and R&B vocal:
      - Dry path  (62-75%): preserves transients, attack, and natural expression
      - Crush path (25-38%): heavily compressed, level-matched, adds density

    The crush path fills in quiet gaps between syllables and sustains energy
    without causing audible pumping (because the dry path preserves dynamics).

    blend = 25% (singing) → 38% (rap): rap needs more fill between fast syllables.

    Level matching before blend ensures we're adding density, not just gain.
    """
    blend = float(np.interp(rap_score, [0, 1], [0.25, 0.38]))

    crush = Pedalboard([
        # Research: 8:1+, 3-5ms attack, 40-80ms release (50ms starting point)
        # Threshold -25dB: crushes almost everything, levels out dynamics fully
        Compressor(threshold_db=-25.0, ratio=8.0, attack_ms=4.0, release_ms=50.0),
    ])
    crushed = crush(vox_ch, SR).astype(np.float32)

    # Level-match crushed to dry RMS before blending (we want density, not loudness)
    dry_rms     = float(np.sqrt(np.mean(vox_ch ** 2) + 1e-12))
    crushed_rms = float(np.sqrt(np.mean(crushed ** 2) + 1e-12))
    if crushed_rms > 1e-9:
        crushed = (crushed * (dry_rms / crushed_rms)).astype(np.float32)

    return ((1.0 - blend) * vox_ch + blend * crushed).astype(np.float32)


def _sidechain_envelope(vox_mono: np.ndarray, n_out: int,
                        depth: float, window_ms: int = 40,
                        attack_ms: float = 10.0,
                        release_ms: float = 100.0) -> np.ndarray:
    """
    Compute per-sample sidechain gain curve (1.0 = no duck, < 1.0 = ducked).

    Separate attack/release smoothing prevents pumping at vocal phrase boundaries.
    Without it, the gain snaps back as soon as the RMS window drops — audible thump.

    Research: 10ms attack, 80-120ms release = transparent hip-hop sidechain.
    """
    win = max(1, int(SR * window_ms / 1000))
    hop = win // 2
    n = len(vox_mono)
    n_frames = max(1, (n + hop - 1) // hop)

    env = np.array([
        np.sqrt(np.mean(vox_mono[i * hop: min(i * hop + win, n)] ** 2))
        for i in range(n_frames)
    ], dtype=np.float32)
    env /= _rms(vox_mono) + 1e-9
    target_gain = (1.0 - depth * np.clip(env, 0.0, 1.0)).astype(np.float64)

    # Attack/release smoothing in frame domain
    frame_rate = SR / max(hop, 1)
    alpha_attack  = np.exp(-1.0 / (frame_rate * attack_ms  / 1000.0))
    alpha_release = np.exp(-1.0 / (frame_rate * release_ms / 1000.0))
    smoothed = np.zeros(n_frames, dtype=np.float64)
    smoothed[0] = target_gain[0]
    for i in range(1, n_frames):
        if target_gain[i] < smoothed[i - 1]:
            alpha = alpha_attack   # gain going down (ducking onset)
        else:
            alpha = alpha_release  # gain recovering after phrase ends
        smoothed[i] = alpha * smoothed[i - 1] + (1.0 - alpha) * target_gain[i]

    x_frames = np.arange(n_frames, dtype=np.float64) * hop
    x_samp   = np.arange(n_out, dtype=np.float64)
    return interp1d(
        x_frames, smoothed, kind="linear",
        bounds_error=False, fill_value=(smoothed[0], smoothed[-1])
    )(x_samp).astype(np.float32)


def _sidechain(inst: np.ndarray, vox: np.ndarray,
               depth: float, window_ms: int = 40,
               attack_ms: float = 10.0, release_ms: float = 100.0) -> np.ndarray:
    """
    Broadband sidechain: duck everything above 200 Hz when vocal is loud.

    - Sub/bass (<200Hz): no ducking — kick and sub stay punchy
    - Everything above 200Hz: ducked proportionally to vocal level

    Previous triband version (duck 200-5kHz only, preserve 5kHz+) caused hi-hats
    and cymbals to pass at full level during vocal sections, pushing the High band
    +10 dB above reference. v17 architecture ducked the full 200+ Hz range.
    Reverting to v17 spec: sub-bass preserved, everything else ducked.
    """
    sos_lp = butter(4, 200.0 / (SR / 2), btype="low",  output="sos")
    sos_hp = butter(4, 200.0 / (SR / 2), btype="high", output="sos")

    inst_lo   = sosfilt(sos_lp, inst, axis=0).astype(np.float32)  # < 200Hz: unaffected
    inst_high = sosfilt(sos_hp, inst, axis=0).astype(np.float32)  # > 200Hz: ducked

    vox_mono = _to_mono(vox)
    gain = _sidechain_envelope(vox_mono, len(inst), depth, window_ms,
                                attack_ms=attack_ms, release_ms=release_ms)

    return (inst_lo + inst_high * gain[:, np.newaxis]).astype(np.float32)


# ── Mastering ─────────────────────────────────────────────────────────────────

def _lufs_normalize(y: np.ndarray, target: float = -9.0) -> np.ndarray:
    meter = pyln.Meter(SR)
    lufs = meter.integrated_loudness(y)
    if not np.isfinite(lufs) or lufs < -70.0:
        return y
    return (y * 10 ** ((target - lufs) / 20)).astype(np.float32)


def _auto_evaluate(mix: np.ndarray, inst: np.ndarray, vox: np.ndarray,
                   bpm_a: float) -> dict:
    """
    Programmatic quality check — runs after every fusion so issues are caught
    without needing a human listener. Checks:

      1. Beat sync   — what % of vocal onsets land within 60ms of an inst beat
      2. Vocal level — is the vocal audible vs the mix (40-70% of mix RMS)
      3. Spectral    — are all 5 frequency bands within commercial reference ranges
      4. Clipping    — any true peaks above -0.2 dBTP?
      5. LUFS        — is the integrated loudness in the -15 to -8 range? (target -10)
    """
    issues = []
    scores = {}

    mix_mono  = _to_mono(mix)
    vox_mono  = _to_mono(vox)
    inst_mono = _to_mono(inst)

    # ── 1. Beat sync via cross-correlation ────────────────────────────────────
    # Measures whether the vocal's rhythmic groove matches the instrumental's
    # beat grid, using cross-correlation of onset envelopes rather than counting
    # how many vocal onsets hit exact beat positions (which is too strict for
    # singers/rappers who frequently land on off-beats or syncopate).
    try:
        hop = 512
        vox_env  = librosa.onset.onset_strength(y=vox_mono,  sr=SR, hop_length=hop)
        inst_env = librosa.onset.onset_strength(y=inst_mono, sr=SR, hop_length=hop)

        # Normalise both envelopes
        vox_env  = vox_env  / (vox_env.max()  + 1e-9)
        inst_env = inst_env / (inst_env.max() + 1e-9)

        # Cross-correlation — positive lags = vocal is ahead of beat
        n = min(len(vox_env), len(inst_env), SR * 60 // hop)  # cap at 60s
        xcorr = np.correlate(vox_env[:n], inst_env[:n], mode="full")
        mid = len(xcorr) // 2

        # Sync score: correlation at lag 0 vs the global peak correlation
        # 1.0 = perfectly in phase, 0.0 = no rhythmic relationship at all
        sync_score = float(xcorr[mid] / (xcorr.max() + 1e-9))
        best_lag_frames = int(np.argmax(xcorr)) - mid
        best_lag_ms = best_lag_frames * hop / SR * 1000

        scores["beat_sync_pct"] = round(sync_score * 100, 1)
        scores["beat_lag_ms"]   = round(best_lag_ms, 0)

        if sync_score < 0.35:
            issues.append(
                f"BEAT SYNC FAIL: {sync_score:.0%} rhythmic correlation "
                f"(best lag {best_lag_ms:+.0f} ms, want sync_score >35%)")
        elif sync_score < 0.55:
            issues.append(
                f"Beat sync marginal: {sync_score:.0%} "
                f"(best lag {best_lag_ms:+.0f} ms, want >55%)")
    except Exception as e:
        scores["beat_sync_pct"] = None
        issues.append(f"Beat sync check error: {e}")

    # ── 2. Vocal presence (stem-based, mastering-gain-independent) ────────────
    # Compare vocal stem RMS to (inst + vocal) combined, not to the mastered mix,
    # so LUFS normalization doesn't skew the metric.
    vr = _rms(vox_mono)
    ir = _rms(inst_mono)
    vp = vr / (ir + vr + 1e-9)
    scores["vocal_presence"] = round(vp * 100, 1)
    if vp < 0.40:
        issues.append(f"Vocal buried: {vp:.0%} of combined energy (want 40-70%)")
    elif vp > 0.70:
        issues.append(f"Vocal overpowers beat: {vp:.0%} of combined energy (want 40-70%)")

    # ── 3. Spectral balance ───────────────────────────────────────────────────
    clip_s = min(len(mix_mono), SR * 60)
    S = np.abs(librosa.stft(mix_mono[:clip_s], n_fft=2048))
    freqs = librosa.fft_frequencies(sr=SR)

    def _bdb(lo, hi):
        m = (freqs >= lo) & (freqs < hi)
        return float(20.0 * np.log10(float(S[m].mean()) + 1e-9))

    bands = {
        "Bass (20-250 Hz)":      (_bdb(20,   250),  25, 34),
        "Lo-Mid (250-800 Hz)":   (_bdb(250,  800),  17, 24),
        "Mid (800-2.5k Hz)":     (_bdb(800, 2500),  10, 18),
        "Hi-Mid (2.5-6k Hz)":    (_bdb(2500, 6000),  4, 12),
        "High (6-20k Hz)":       (_bdb(6000,20000), -12, -3),
    }
    scores["bands"] = {k: round(v, 1) for k, (v, _, _) in bands.items()}
    for name, (val, lo, hi) in bands.items():
        if val < lo:
            issues.append(f"{name}: {val:.1f} dB (want {lo}–{hi}, TOO LOW)")
        elif val > hi:
            issues.append(f"{name}: {val:.1f} dB (want {lo}–{hi}, TOO HIGH)")

    # ── 4. Clipping ───────────────────────────────────────────────────────────
    peak = float(np.max(np.abs(mix)))
    scores["peak_dBFS"] = round(20 * np.log10(peak + 1e-9), 2)
    if peak > 1.001:
        issues.append(f"CLIPPING: peak = {peak:.5f} (hard clip — limiter failed)")

    # ── 5. Integrated loudness ────────────────────────────────────────────────
    meter = pyln.Meter(SR)
    lufs = meter.integrated_loudness(mix_mono)
    scores["lufs"] = round(lufs, 1) if np.isfinite(lufs) else None
    if np.isfinite(lufs):
        if lufs < -15:
            issues.append(f"Mix too quiet: {lufs:.1f} LUFS (want -15 to -8)")
        elif lufs > -8:
            issues.append(f"Mix too loud: {lufs:.1f} LUFS (want -15 to -8)")

    # ── 6. Phase cancellation check ───────────────────────────────────────────
    # Per-band phase difference between vocal and instrumental.
    # Bands where mean phase diff > 90° have partial cancellation in the mix.
    try:
        n_fft_p = 4096
        n_p = min(len(vox_mono), len(inst_mono), n_fft_p)
        Sv = np.fft.rfft(vox_mono[:n_p], n=n_fft_p)
        Si = np.fft.rfft(inst_mono[:n_p], n=n_fft_p)
        freqs_p = np.fft.rfftfreq(n_fft_p, 1.0 / SR)
        phase_diff = np.abs(np.angle(Sv) - np.angle(Si))
        phase_diff = np.where(phase_diff > np.pi, 2 * np.pi - phase_diff, phase_diff)
        phase_diff_deg = np.degrees(phase_diff)

        p_bands = {"bass": (80, 250), "lo_mid": (250, 800),
                   "mid": (800, 3000), "hi_mid": (3000, 8000)}
        phase_issues = []
        for pname, (plo, phi) in p_bands.items():
            pmask = (freqs_p >= plo) & (freqs_p < phi)
            if pmask.any():
                mean_pd = float(phase_diff_deg[pmask].mean())
                if mean_pd > 110.0:
                    phase_issues.append(f"{pname}:{mean_pd:.0f}°")
        if phase_issues:
            issues.append(f"Phase cancellation risk: {', '.join(phase_issues)} (>110° mean diff)")
        scores["phase_issues"] = phase_issues
    except Exception:
        pass

    # ── 7. Stereo correlation (mono compatibility) ────────────────────────────
    # A correlation below 0.5 means the mix has excessive out-of-phase content
    # and will partially cancel in mono (phone speakers, club PA mono fold).
    # Professional target: correlation > 0.7.
    if mix.ndim == 2 and mix.shape[1] == 2:
        L = mix[:, 0].astype(np.float64)
        R = mix[:, 1].astype(np.float64)
        corr_num = float(np.mean(L * R))
        corr_den = float(np.sqrt(np.mean(L ** 2) * np.mean(R ** 2)) + 1e-12)
        stereo_corr = corr_num / corr_den
        scores["stereo_corr"] = round(stereo_corr, 3)
        if stereo_corr < 0.5:
            issues.append(f"STEREO CORR FAIL: {stereo_corr:.2f} (mono cancel risk, want >0.7)")
        elif stereo_corr < 0.7:
            issues.append(f"Stereo corr marginal: {stereo_corr:.2f} (want >0.7 for mono safe)")

    scores["issues"] = issues
    scores["pass"]   = not any("FAIL" in i or "CLIP" in i for i in issues)
    return scores


def _multiband_master_compress(mix: np.ndarray) -> np.ndarray:
    """
    4-band mastering compression for hip-hop — gentle, transparent control.

    Philosophy: ratios 1.2–2:1, target 1–3 dB GR per band (not the 6–10 dB
    used in mixing bus compression). Goal is tonal balance, not loudness.

    Professional hip-hop crossover points:
      Sub (20-80Hz):   kick/808 compete here; slow attack lets transients through
      LowMid (80-200Hz): body/punch zone; very gentle
      Mid (200-5kHz):  vocal and snare compete; medium
      High (5-20kHz):  hi-hats/air; barely touched

    Applied after mastering EQ, before harmonic saturation and soft clip.
    """
    nyq = SR / 2.0
    # (lo_hz, hi_hz, threshold_db, ratio, attack_ms, release_ms)
    band_defs = [
        (20,    80,   -6.0, 2.0, 90.0, 200.0),   # Sub: slow attack = kick passes through
        (80,    200,  -7.0, 1.5, 50.0, 100.0),   # LowMid: gentle body control
        (200,   5000, -8.0, 1.5, 30.0,  80.0),   # Mid: snare/vocal zone, moderate
        (5000, 20000, -9.0, 1.3, 20.0,  50.0),   # High: hi-hats, very gentle
    ]
    out = np.zeros_like(mix)
    for lo, hi, thresh, ratio, atk, rel in band_defs:
        lo_n = lo / nyq
        hi_n = min(hi / nyq, 0.999)
        sos_lp = butter(4, hi_n, btype="low",  output="sos")
        sos_hp = butter(4, lo_n, btype="high", output="sos")
        band = sosfilt(sos_lp, sosfilt(sos_hp, mix, axis=0), axis=0).astype(np.float32)
        comp = Pedalboard([Compressor(threshold_db=thresh, ratio=ratio,
                                      attack_ms=atk, release_ms=rel)])
        out += comp(band.T.astype(np.float32), SR).T.astype(np.float32)
    return out.astype(np.float32)


def _master(mix: np.ndarray, bpm: float = 120.0) -> np.ndarray:
    """
    Mastering chain (v6):
      M/S EQ → mastering EQ → soft clip → glue comp → sub-bass limiter
      → LUFS -10 normalize → brick-wall limiter -2.0 dBFS

    M/S EQ (new):
      Mid: -1.5 dB @ 350 Hz (remove mud from centered elements), sub preserved
      Sides: +2 dB @ 8 kHz shelf (widen highs), -3 dB @ 100 Hz (mono-safe bass)

    LUFS normalize goes LAST so it accounts for all gain reduction.
    Hip-hop target: -9 LUFS (-8 to -10).
    """
    # ── Safety normalize: bring mix to -6 dBFS peak before processing ─────────
    # The soft-clipper (Chebyshev 1.5x - 0.5x³) hard-clips everything above 1.0
    # because it clips input first: mix_c = np.clip(mix, -1.0, 1.0). When the
    # post-mix peak is 4.77 (+13.6 dBFS), the clipper acts as a brick wall on
    # the top 70% of the signal — producing "crazy static" harmonic distortion.
    # Normalizing to -6 dBFS (peak=0.5) ensures the entire mastering chain
    # (tanh saturation, Chebyshev soft clip, compression) operates in its
    # intended range. LUFS normalize at the end sets final loudness.
    peak_in = float(np.max(np.abs(mix)))
    if peak_in > 0.5:
        mix = (mix * (0.5 / peak_in)).astype(np.float32)
    print(f"      Master input: peak_in={peak_in:.3f} → normalized to -6 dBFS",
          flush=True)

    # M/S EQ: separate processing for Mid and Sides channels
    if mix.ndim == 2 and mix.shape[1] == 2:
        M, S = _ms_encode(mix)  # (samples,) each

        # Mid EQ: cut mud at 350 Hz, low-end warmth, and punch-through shelf.
        # Research: +1 dB shelf at 3-5 kHz on the Mid improves mono compatibility
        # and punch-through on phone speakers / earbuds (where M/S collapses to mono).
        mid_eq = Pedalboard([
            PeakFilter(cutoff_frequency_hz=350.0, gain_db=-1.5,  q=1.2),  # Lo-Mid mud in Mid channel
            LowShelfFilter(cutoff_frequency_hz=120.0, gain_db=0.5),  # sub warmth
            HighShelfFilter(cutoff_frequency_hz=4000.0, gain_db=0.0), # removed: was contributing to Hi-Mid excess
        ])
        M_proc = mid_eq(M[np.newaxis, :].astype(np.float32), SR)[0]

        # Sides EQ: roll off sub-bass (mono-safe: bass should be center),
        # cut muddy low-mids on sides, add high-shelf air to widen presence.
        # Research: sides low-mids (300-600Hz) are often murky; cut 2-3dB here
        # improves clarity without affecting the vocal (which is Mid-only).
        sides_eq = Pedalboard([
            HighpassFilter(cutoff_frequency_hz=120.0),              # 120Hz mono-safe (safer than 100Hz)
            PeakFilter(cutoff_frequency_hz=400.0, gain_db=-2.5, q=0.8),  # muddy sides cut
            HighShelfFilter(cutoff_frequency_hz=8000.0, gain_db=+1.0),    # +1.0dB: add air to sides (was -1.0 to tame old exciters)
        ])
        S_proc = sides_eq(S[np.newaxis, :].astype(np.float32), SR)[0]

        mix = _ms_decode(M_proc.astype(np.float32), S_proc.astype(np.float32))

    # Mastering EQ (broadband) — psychoacoustically optimized
    # 3.2kHz fatigue notch: ear is most sensitive here (ISO 226 equal-loudness peak)
    # A -1.5 dB cut at 3.2kHz dramatically reduces fatigue without perceived loudness loss,
    # freeing headroom for the limiter to work 0.5-1 dB harder.
    master_eq = Pedalboard([
        PeakFilter(cutoff_frequency_hz=250.0,  gain_db=-1.5,  q=0.8),  # mud cut
        PeakFilter(cutoff_frequency_hz=500.0,  gain_db=-2.5,  q=0.7),  # lo-mid warmth control (250-800Hz)
        PeakFilter(cutoff_frequency_hz=700.0,  gain_db=-1.0,  q=1.0),  # upper lo-mid cut
        PeakFilter(cutoff_frequency_hz=3200.0, gain_db=-1.5,  q=2.5),  # ear-fatigue notch
        PeakFilter(cutoff_frequency_hz=3500.0, gain_db=-2.0,  q=1.0),  # hi-mid harshness cut
        PeakFilter(cutoff_frequency_hz=4000.0, gain_db=-1.5,  q=1.0),  # upper presence cut (2.5-6kHz)
        HighShelfFilter(cutoff_frequency_hz=6000.0, gain_db=-1.5),
    ])
    mix = master_eq(mix.T.astype(np.float32), SR).T.astype(np.float32)

    # 4-band mastering compression: gentle tonal balance (1-3 dB GR per band)
    # Placed after mastering EQ so it controls, not changes, the tonal balance
    mix = _multiband_master_compress(mix)

    # Maxx Bass, tanh saturation, and harmonic exciter all REMOVED from mastering.
    # These three nonlinear stages were stacking intermodulation distortion and
    # combined with the Chebyshev soft-clip to produce static/harshness artifacts.
    # The Chebyshev soft-clip below is sufficient for peak control without IMD.

    # Soft clip: Chebyshev 3rd-order (1.5x - 0.5x³) — gentler than tanh,
    # preserves low-level signal shape, clips peaks without hardness
    mix_c = np.clip(mix, -1.0, 1.0)
    mix = (1.5 * mix_c - 0.5 * mix_c ** 3).astype(np.float32)

    # Glue compressor: BPM-synced release (60-70% of beat interval).
    # Research: at 140 BPM (429ms/beat), target release ~250ms.
    # At 80 BPM (750ms/beat), target ~450ms.
    # Attack 10ms: fast enough to catch snare body but passes kick transient (slam).
    beat_ms = 60000.0 / max(bpm, 60.0)
    glue_release_ms = float(np.clip(beat_ms * 0.60, 50.0, 400.0))
    # Glue comp: softer than before (-6/2:1 → -10/1.5:1).
    # -6 dBFS threshold was firing on the entire mix body and smashing the beat.
    # -10 dBFS threshold only catches true peak transients. 1.5:1 is barely
    # audible as compression — it "glues" without "squashing".
    glue = Pedalboard([
        Compressor(threshold_db=-10.0, ratio=1.5, attack_ms=15.0, release_ms=glue_release_ms),
    ])
    mix = glue(mix.T.astype(np.float32), SR).T.astype(np.float32)

    # Sub-bass limiter: limit 20-80Hz band separately before main brick-wall.
    # Prevents kick/808 from eating all the headroom and triggering brick-wall clamp.
    # Professional technique: sub-bass limiter at -3 dBFS (1-2ms attack, 60ms release).
    sos_sub_lp = butter(4, 80.0 / (SR / 2), btype="low",  output="sos")
    sos_sub_hp = butter(4, 20.0 / (SR / 2), btype="high", output="sos")
    sub_limiter = Pedalboard([
        Limiter(threshold_db=-3.0, release_ms=60.0),
    ])
    mix_sub   = sosfilt(sos_sub_hp, sosfilt(sos_sub_lp, mix, axis=0), axis=0).astype(np.float32)
    mix_above = (mix - mix_sub).astype(np.float32)
    mix_sub_lim = sub_limiter(mix_sub.T.astype(np.float32), SR).T.astype(np.float32)
    # Additional sub-bass attenuation: high-energy EDM/house beats can have
    # sub energy 20+ dB above mids. Attenuate by extra 6 dB after limiting.
    mix_sub_lim = (mix_sub_lim * 0.5).astype(np.float32)  # -6 dB on sub band
    mix = (mix_above + mix_sub_lim).astype(np.float32)

    # LUFS normalize BEFORE the brick-wall limiter.
    # CRITICAL ORDER: normalize FIRST, then limit.
    # If limiter came first: the normalize step could raise peaks ABOVE the
    # limiter ceiling, causing clipping (e.g., limiter at -1 dBFS, normalize
    # raises by +3 LU → peaks at +2 dBFS).
    # Correct mastering order: all dynamics → LUFS normalize → brick-wall ceiling.
    # Target: -10 LUFS (streaming-optimized; Spotify/Apple/YouTube normalize to -14 LUFS,
    # so -10 is within 4 LU of the norm — keeps dynamics while remaining competitive).
    # Target -12 LUFS (was -10). 2 dB more headroom before the brick-wall limiter
    # means transients pass through — kick hits harder, snare cracks more, vocals
    # don't get smashed at the ceiling. Streaming platforms normalize to -14 LUFS
    # anyway, so -12 is still competitive without destroying dynamics.
    mix = _lufs_normalize(mix, -12.0)

    # Post-normalize HF gentle control: -3.0 dB at 6kHz (down from -5.5 dB).
    # Previous -5.5 dB was calibrated for harmonic exciters/waveshaper HF excess.
    # Those are removed. Spectral data shows High band was -6.9 dB (too dark).
    # With master EQ shelf reduced to -1.5 and sides EQ adding +1 dB air,
    # the mix overshot to +0.3 dB. Apply -3 dB post-normalize to land at ~-2.7 dB
    # (between inputs at -4.0 and -1.5 dB — natural midpoint).
    post_norm_eq = Pedalboard([HighShelfFilter(cutoff_frequency_hz=6000.0, gain_db=-3.0)])
    mix = post_norm_eq(mix.T.astype(np.float32), SR).T.astype(np.float32)

    # Brick-wall limiter LAST: enforces peak ceiling after LUFS normalization.
    limiter = Pedalboard([Limiter(threshold_db=-2.0, release_ms=50.0)])
    mix = limiter(mix.T.astype(np.float32), SR).T.astype(np.float32)
    # Hard safety clip: numpy failsafe in case pedalboard limiter lets peaks through
    # (observed: on dense EDM material peak can reach 0 dBFS post-limiter).
    # -2 dBFS = amplitude 0.7943. Clip then re-normalize to -2 dBFS to prevent
    # inter-sample spikes pushing true peak above 0 dBTP.
    clip_ceil = 10 ** (-2.0 / 20.0)   # 0.7943
    peak_post = float(np.max(np.abs(mix)))
    if peak_post > clip_ceil:
        mix = np.clip(mix, -clip_ceil, clip_ceil).astype(np.float32)
    return mix


def _master_club(mix: np.ndarray, bpm: float = 120.0) -> np.ndarray:
    """
    Club variant mastering: punchier sub, harder sidechain glue, LUFS -9.
    Optimised for large speaker systems and DJ playback.
    """
    # Safety normalize
    peak_in = float(np.max(np.abs(mix)))
    if peak_in > 0.5:
        mix = (mix * (0.5 / peak_in)).astype(np.float32)

    # M/S EQ — more sub on Mid, wider Sides
    if mix.ndim == 2 and mix.shape[1] == 2:
        M, S = _ms_encode(mix)
        mid_eq = Pedalboard([
            LowShelfFilter(cutoff_frequency_hz=80.0,  gain_db=+1.5),  # sub boost
            PeakFilter(cutoff_frequency_hz=350.0,     gain_db=-0.5, q=1.2),
        ])
        sides_eq = Pedalboard([
            HighpassFilter(cutoff_frequency_hz=120.0),
            PeakFilter(cutoff_frequency_hz=400.0,     gain_db=-2.0, q=0.8),
            HighShelfFilter(cutoff_frequency_hz=8000.0, gain_db=+1.5),  # more air
        ])
        M_p = mid_eq(M[np.newaxis, :].astype(np.float32), SR)[0]
        S_p = sides_eq(S[np.newaxis, :].astype(np.float32), SR)[0]
        mix = _ms_decode(M_p.astype(np.float32), S_p.astype(np.float32))

    # Mastering EQ — keep the lows, cut harshness
    master_eq = Pedalboard([
        PeakFilter(cutoff_frequency_hz=250.0,  gain_db=-0.3, q=0.8),
        PeakFilter(cutoff_frequency_hz=3200.0, gain_db=-1.5, q=2.5),
        PeakFilter(cutoff_frequency_hz=3500.0, gain_db=-2.0, q=1.0),
        HighShelfFilter(cutoff_frequency_hz=6000.0, gain_db=-1.0),
    ])
    mix = master_eq(mix.T.astype(np.float32), SR).T.astype(np.float32)

    mix = _multiband_master_compress(mix)

    # Soft clip
    mix_c = np.clip(mix, -1.0, 1.0)
    mix = (1.5 * mix_c - 0.5 * mix_c ** 3).astype(np.float32)

    # Harder glue comp for club impact
    beat_ms = 60000.0 / max(bpm, 60.0)
    glue_release_ms = float(np.clip(beat_ms * 0.55, 40.0, 350.0))
    glue = Pedalboard([Compressor(threshold_db=-8.0, ratio=2.0,
                                   attack_ms=10.0, release_ms=glue_release_ms)])
    mix = glue(mix.T.astype(np.float32), SR).T.astype(np.float32)

    # Sub-bass limiter
    sos_sub_lp = butter(4, 80.0 / (SR / 2), btype="low",  output="sos")
    sos_sub_hp = butter(4, 20.0 / (SR / 2), btype="high", output="sos")
    mix_sub  = sosfilt(sos_sub_hp, sosfilt(sos_sub_lp, mix, axis=0)).astype(np.float32)
    mix_abv  = (mix - mix_sub).astype(np.float32)
    sub_lim  = Pedalboard([Limiter(threshold_db=-2.0, release_ms=40.0)])
    mix_sub  = sub_lim(mix_sub.T.astype(np.float32), SR).T.astype(np.float32)
    mix_sub  = (mix_sub * 0.6).astype(np.float32)   # -4.4 dB — more sub than radio
    mix = (mix_abv + mix_sub).astype(np.float32)

    # LUFS -9 for club loudness
    mix = _lufs_normalize(mix, -9.0)
    post_eq = Pedalboard([HighShelfFilter(cutoff_frequency_hz=6000.0, gain_db=-2.5)])
    mix = post_eq(mix.T.astype(np.float32), SR).T.astype(np.float32)

    limiter = Pedalboard([Limiter(threshold_db=-0.5, release_ms=40.0)])
    mix = limiter(mix.T.astype(np.float32), SR).T.astype(np.float32)
    clip_ceil = 10 ** (-0.5 / 20.0)
    if float(np.max(np.abs(mix))) > clip_ceil:
        mix = np.clip(mix, -clip_ceil, clip_ceil).astype(np.float32)
    return mix


def _master_intimate(mix: np.ndarray, bpm: float = 120.0) -> np.ndarray:
    """
    Intimate variant mastering: wide stereo, LUFS -14, high dynamic range.
    Optimised for headphone listening and streaming platforms.
    """
    peak_in = float(np.max(np.abs(mix)))
    if peak_in > 0.5:
        mix = (mix * (0.5 / peak_in)).astype(np.float32)

    # M/S EQ — wider Sides, less sub
    if mix.ndim == 2 and mix.shape[1] == 2:
        M, S = _ms_encode(mix)
        mid_eq = Pedalboard([
            PeakFilter(cutoff_frequency_hz=350.0, gain_db=-0.5, q=1.2),
            LowShelfFilter(cutoff_frequency_hz=120.0, gain_db=0.3),
        ])
        sides_eq = Pedalboard([
            HighpassFilter(cutoff_frequency_hz=100.0),
            HighShelfFilter(cutoff_frequency_hz=6000.0, gain_db=+2.0),  # wider, airier
        ])
        M_p = mid_eq(M[np.newaxis, :].astype(np.float32), SR)[0]
        S_p = sides_eq(S[np.newaxis, :].astype(np.float32), SR)[0]
        # Boost Sides for headphone width
        mix = _ms_decode(M_p.astype(np.float32), (S_p * 1.2).astype(np.float32))

    master_eq = Pedalboard([
        PeakFilter(cutoff_frequency_hz=250.0,  gain_db=-0.5, q=0.8),
        PeakFilter(cutoff_frequency_hz=3200.0, gain_db=-1.0, q=2.5),
        PeakFilter(cutoff_frequency_hz=3500.0, gain_db=-1.5, q=1.0),
        HighShelfFilter(cutoff_frequency_hz=6000.0, gain_db=-0.5),  # less HF cut = more air
    ])
    mix = master_eq(mix.T.astype(np.float32), SR).T.astype(np.float32)

    # No multiband compress — preserve dynamics for intimacy
    # Soft clip only
    mix_c = np.clip(mix, -1.0, 1.0)
    mix = (1.5 * mix_c - 0.5 * mix_c ** 3).astype(np.float32)

    # Very gentle glue — just barely touches peaks
    beat_ms = 60000.0 / max(bpm, 60.0)
    glue_release_ms = float(np.clip(beat_ms * 0.70, 80.0, 500.0))
    glue = Pedalboard([Compressor(threshold_db=-14.0, ratio=1.3,
                                   attack_ms=20.0, release_ms=glue_release_ms)])
    mix = glue(mix.T.astype(np.float32), SR).T.astype(np.float32)

    # Sub-bass limiter — lighter
    sos_sub_lp = butter(4, 80.0 / (SR / 2), btype="low",  output="sos")
    sos_sub_hp = butter(4, 20.0 / (SR / 2), btype="high", output="sos")
    mix_sub = sosfilt(sos_sub_hp, sosfilt(sos_sub_lp, mix, axis=0)).astype(np.float32)
    mix_abv = (mix - mix_sub).astype(np.float32)
    sub_lim = Pedalboard([Limiter(threshold_db=-4.0, release_ms=80.0)])
    mix_sub = sub_lim(mix_sub.T.astype(np.float32), SR).T.astype(np.float32)
    mix_sub = (mix_sub * 0.45).astype(np.float32)   # -6.9 dB — tighter sub for headphones
    mix = (mix_abv + mix_sub).astype(np.float32)

    # LUFS -14 (streaming standard — Spotify/Apple target)
    mix = _lufs_normalize(mix, -14.0)

    # Very slight HF control
    post_eq = Pedalboard([HighShelfFilter(cutoff_frequency_hz=6000.0, gain_db=-1.5)])
    mix = post_eq(mix.T.astype(np.float32), SR).T.astype(np.float32)

    limiter = Pedalboard([Limiter(threshold_db=-1.5, release_ms=60.0)])
    mix = limiter(mix.T.astype(np.float32), SR).T.astype(np.float32)
    clip_ceil = 10 ** (-1.5 / 20.0)
    if float(np.max(np.abs(mix))) > clip_ceil:
        mix = np.clip(mix, -clip_ceil, clip_ceil).astype(np.float32)
    return mix


def _fade(y: np.ndarray, fade_s: float = 2.0) -> np.ndarray:
    n = min(int(SR * fade_s), len(y) // 6)
    y = y.copy()
    y[:n]  *= np.linspace(0.0, 1.0, n, dtype=np.float32)[:, np.newaxis] ** 0.5
    y[-n:] *= np.linspace(1.0, 0.0, n, dtype=np.float32)[:, np.newaxis] ** 0.5
    return y


# ── System 1: Pre-Flight Compatibility Scorer ─────────────────────────────────
# Runs before any heavy processing. Flags incompatible pairs early so we don't
# waste 45 minutes producing a bad result. Returns a structured report with
# pass/warn/fail per dimension and recommended adjustments.

_NOTES = ["C","C#","D","D#","E","F","F#","G","G#","A","A#","B"]

def _preflight_check(full_a: np.ndarray, full_b: np.ndarray,
                     bpm_a: float, bpm_b: float,
                     key_a_root: int, key_a_mode: str,
                     key_b_root: int, key_b_mode: str) -> dict:
    """
    Compatibility check before fusing. Returns dict with:
      pass        : bool — False only for fatal mismatches (file silent, etc.)
      warnings    : [str] — non-fatal issues to log
      info        : [str] — all-clear confirmations
      bpm_gap_pct : float
      key_semitones: int
      genre_sim   : float  (0–1, MFCC cosine similarity)
    """
    report: dict = {"pass": True, "warnings": [], "info": []}

    # ── BPM gap ───────────────────────────────────────────────────────────────
    bpm_gap_pct = abs(bpm_a - bpm_b) / max(bpm_a, 1.0) * 100
    bpm_ratio   = max(bpm_a, bpm_b) / max(min(bpm_a, bpm_b), 1.0)
    if bpm_ratio > 2.1:
        report["pass"] = False
        report["warnings"].append(
            f"FATAL: BPM gap {bpm_gap_pct:.0f}% ({bpm_a:.0f} vs {bpm_b:.0f}) — "
            ">2× stretch will make vocal unrecognisable. Aborting.")
    elif bpm_gap_pct > 25:
        report["warnings"].append(
            f"Large BPM gap {bpm_gap_pct:.0f}%: beat will be time-stretched "
            ">25%. Expect some vowel smearing on sustained notes.")
    else:
        report["info"].append(f"BPM gap {bpm_gap_pct:.0f}% — OK")

    # ── Key distance ──────────────────────────────────────────────────────────
    n_semi = abs(semitones_to_shift(key_b_root, key_b_mode, key_a_root, key_a_mode))
    if n_semi > 6:
        report["warnings"].append(
            f"Key distance {n_semi} semitones "
            f"({_NOTES[key_b_root]} {key_b_mode} → {_NOTES[key_a_root]} {key_a_mode}): "
            "smart key-shift will find nearest harmonic alternative.")
    elif n_semi > 3:
        report["warnings"].append(
            f"Key shift {n_semi} semitones — small formant-artifact risk on pitched vocals.")
    else:
        report["info"].append(f"Key distance {n_semi} semitones — compatible")

    # ── Vocal presence (Song B) ───────────────────────────────────────────────
    rms_b_db = float(20 * np.log10(_rms(full_b) + 1e-9))
    if rms_b_db < -40:
        report["pass"] = False
        report["warnings"].append(
            f"FATAL: Song B RMS {rms_b_db:.0f} dBFS — file may be silent or corrupt.")
    else:
        report["info"].append(f"Song B RMS {rms_b_db:.0f} dBFS — OK")

    # ── Genre/tonal similarity (MFCC cosine) ──────────────────────────────────
    try:
        seg = SR * 30   # use first 30 s for speed
        m_a = librosa.feature.mfcc(y=full_a[:seg], sr=SR, n_mfcc=12).mean(axis=1)
        m_b = librosa.feature.mfcc(y=full_b[:seg], sr=SR, n_mfcc=12).mean(axis=1)
        m_a /= np.linalg.norm(m_a) + 1e-9
        m_b /= np.linalg.norm(m_b) + 1e-9
        genre_sim = float(np.dot(m_a, m_b))
    except Exception:
        genre_sim = 0.5   # assume OK if analysis fails
    report["genre_sim"] = genre_sim
    if genre_sim < 0.30:
        report["warnings"].append(
            f"Low genre similarity ({genre_sim:.2f}): songs may be tonally incompatible. "
            "Proceeding — result may require manual EQ.")
    else:
        report["info"].append(f"Genre similarity {genre_sim:.2f} — compatible")

    report["bpm_gap_pct"]   = bpm_gap_pct
    report["key_semitones"] = n_semi
    return report


# ── System 3: Multi-Attempt Parameter Grid Search ─────────────────────────────
# On QC fail, run 5 quick re-mixes with systematically varied parameters and
# keep the highest-scoring one.  Each re-mix is ~25 s (stems already in RAM).
# This escapes the single-correction-path trap of the existing 3-attempt loop.

_MIX_PARAM_GRID = [
    # (vocal_level_delta, carve_db_delta, presence_db_delta, nr_strength_delta)
    ( 0.0,  0.0,  0.0,  0.00),   # baseline — already tried, keeps best
    (+0.4,  0.0, +0.5,  0.00),   # louder vocal + more presence
    (-0.4, +1.5,  0.0,  0.05),   # quieter vocal + deeper carve + more cleanup
    ( 0.0, +2.0, +0.5,  0.00),   # big carve, more presence
    (+0.4, +1.0, -0.5,  0.05),   # louder vocal, moderate carve, slight de-noise
]

def _param_grid_search(inst_remix: np.ndarray, vox_remix: np.ndarray,
                       style: dict, sidechain_depth: float, bpm_a: float,
                       out_path: str, current_score: int) -> tuple:
    """
    Try _MIX_PARAM_GRID parameter variations. Returns (best_mix, best_score, best_style).
    Only improves on current_score — never degrades.
    """
    try:
        from listen import auto_score as _auto_score
    except ImportError:
        return None, current_score, style

    best_mix   = None
    best_score = current_score
    best_style = style.copy()

    print(f"\n  [Grid Search] Trying {len(_MIX_PARAM_GRID)} parameter sets…", flush=True)

    for i, (dvl, dcarve, dpres, dnr) in enumerate(_MIX_PARAM_GRID):
        if i == 0:
            continue   # skip baseline — already scored

        trial_style = style.copy()
        trial_style["vocal_level"] = float(np.clip(
            style["vocal_level"] + dvl, 1.0, 4.0))
        trial_style["carve_db"]    = float(np.clip(
            style["carve_db"] + dcarve, 3.0, 14.0))
        trial_style["presence_db"] = float(np.clip(
            style.get("presence_db", 2.0) + dpres, 0.0, 5.0))

        try:
            trial_mix = _iterative_mix(inst_remix, vox_remix,
                                       trial_style, sidechain_depth, bpm_a)
            trial_mix = _fade(trial_mix, fade_s=2.0)
            trial_mix = _master(trial_mix, bpm=bpm_a)
            sf.write(out_path, trial_mix, SR, subtype="PCM_24")
            _passed, _score, _summary, _ = _auto_score(out_path)
            print(f"    Grid {i+1}/{len(_MIX_PARAM_GRID)}: {_score}/100"
                  f"  vl={trial_style['vocal_level']:.1f}"
                  f"  carve={trial_style['carve_db']:.1f}"
                  f"  pres={trial_style['presence_db']:.1f}"
                  f"  {'✓' if _passed else '✗'}", flush=True)
            if _score > best_score:
                best_score = _score
                best_mix   = trial_mix.copy()
                best_style = trial_style.copy()
                if _passed:
                    print(f"    → Found passing config at grid {i+1}!", flush=True)
                    break
        except Exception as _ge:
            print(f"    Grid {i+1} failed: {_ge}", flush=True)

    return best_mix, best_score, best_style


# ── Main Entry Point ──────────────────────────────────────────────────────────

def fuse(song_a: str, song_b: str, out_path: str,
         stems_cache: str = "vf_data/stems",
         progress_cb=None) -> str:
    """
    Fuse Song A (beat/instrumental) with Song B (vocals).
    Writes stereo PCM WAV to out_path and returns the path.
    """
    def step(n, total, msg):
        print(f"[{n}/{total}] {msg}", flush=True)
        if progress_cb:
            progress_cb(n, total, msg)

    TOTAL = 9

    step(1, TOTAL, "Loading audio for analysis…")
    full_a = librosa.load(song_a, sr=SR, mono=True)[0].astype(np.float32)
    full_b = librosa.load(song_b, sr=SR, mono=True)[0].astype(np.float32)

    # ── System 5: Fingerprint DB — load cached analysis, skip re-detection ────
    fid_a = _file_id(song_a)
    fid_b = _file_id(song_b)
    fp_a  = _load_fp(fid_a, stems_cache)
    fp_b  = _load_fp(fid_b, stems_cache)

    step(2, TOTAL, "Detecting BPM…")
    if "bpm" in fp_a:
        bpm_a = float(fp_a["bpm"])
        print(f"      A: {bpm_a:.1f} BPM (cached)", flush=True)
    else:
        bpm_a = detect_bpm(full_a)
        _save_fp(fid_a, stems_cache, {"bpm": bpm_a})
    if "bpm" in fp_b:
        bpm_b = float(fp_b["bpm"])
        print(f"      B: {bpm_b:.1f} BPM (cached)", flush=True)
    else:
        bpm_b = detect_bpm(full_b)
        _save_fp(fid_b, stems_cache, {"bpm": bpm_b})
    print(f"      A: {bpm_a:.1f} BPM   B: {bpm_b:.1f} BPM", flush=True)

    step(3, TOTAL, "Detecting keys…")
    if "key_root" in fp_a and "key_mode" in fp_a:
        key_a_root, key_a_mode = int(fp_a["key_root"]), fp_a["key_mode"]
        print(f"      A: {_NOTES[key_a_root]} {key_a_mode} (cached)", flush=True)
    else:
        key_a_root, key_a_mode = detect_key(full_a)
        _save_fp(fid_a, stems_cache, {"key_root": key_a_root, "key_mode": key_a_mode})
    if "key_root" in fp_b and "key_mode" in fp_b:
        key_b_root, key_b_mode = int(fp_b["key_root"]), fp_b["key_mode"]
        print(f"      B: {_NOTES[key_b_root]} {key_b_mode} (cached)", flush=True)
    else:
        key_b_root, key_b_mode = detect_key(full_b)
        _save_fp(fid_b, stems_cache, {"key_root": key_b_root, "key_mode": key_b_mode})
    print(f"      A: {_NOTES[key_a_root]} {key_a_mode}   "
          f"B: {_NOTES[key_b_root]} {key_b_mode}", flush=True)

    # ── System 1: Pre-Flight Compatibility Check ──────────────────────────────
    pf = _preflight_check(full_a, full_b, bpm_a, bpm_b,
                          key_a_root, key_a_mode, key_b_root, key_b_mode)
    for msg in pf["info"]:
        print(f"      ✓ {msg}", flush=True)
    for msg in pf["warnings"]:
        print(f"      ⚠ {msg}", flush=True)
    # Save genre similarity to fingerprint DB for future use
    _save_fp(fid_a, stems_cache, {"genre_sim_with_last_b": pf.get("genre_sim", 0.5)})
    if not pf["pass"]:
        raise RuntimeError("Pre-flight check failed: " + " | ".join(pf["warnings"]))

    sep_model = "BS-Roformer" if _has_gpu() else "MDX-Net Kim Vocal 2→Demucs fallback"
    step(4, TOTAL, f"Separating stems — Song A (instrumental) via {sep_model}…")
    stems_a = separate(song_a, stems_cache, upgrade_vocal=False)  # beat — no vocal needed

    step(5, TOTAL, f"Separating stems — Song B (vocals) via {sep_model}…")
    stems_b = separate(song_b, stems_cache, upgrade_vocal=True, clean_vocal=True)   # vocal source — upgrade + neural clean

    inst = stems_a["no_vocals"]   # (samples, 2)
    vox  = stems_b["vocals"]      # (samples, 2)

    # ── System 2: Assess stem quality BEFORE bleed removal to set adaptive params ──
    _sq = _assess_stem_quality(vox, inst)
    print(f"      Stem quality — bleed: {_sq['bleed_level']} ({_sq['bleed_ratio']:.3f})"
          f"  SNR: {_sq['snr_db']:.0f} dB"
          f"  confidence: {_sq['vocal_confidence']:.1f}"
          f"  {'[NEEDS DENOISER]' if _sq['needs_denoiser'] else ''}", flush=True)
    _save_fp(fid_b, stems_cache, {
        "stem_bleed_ratio": _sq["bleed_ratio"],
        "stem_snr_db":      _sq["snr_db"],
        "vocal_confidence": _sq["vocal_confidence"],
    })

    # ── AI-tunable bleed removal parameters (adjusted by director on correction passes) ──
    # Initialised from stem quality assessment — right-sized per song, not hardcoded.
    _bleed_params = {
        "drum_weight_hh":       2.0,
        "wiener_mask_floor":    _sq["recommended"]["wiener_mask_floor"],
        "noisereduce_strength": _sq["recommended"]["noisereduce_strength"],
    }

    # ── Vocal bleed removal ──────────────────────────────────────────────────
    # Oracle path (4-stem Demucs): use drums/bass/other as oracle interference
    # sources for a gentle Wiener mask.  The mask for bin (t,f) is:
    #   mask = V² / (V² + D² + B² + O²)
    # Gentle settings: mask_floor=0.20 (avoids musical-noise tonal static),
    # drum_weight_hh=2.0 (avoids over-suppression in hi-hat band).
    # NOTE: _targeted_hihat_suppression and _harmonic_vocal_process REMOVED —
    # both caused audible artifacts (musical noise / robotic vocoder effect).
    #
    # Fallback path (2-stem / GPU): classical V²/(V²+I²) Wiener + reference
    # noisereduce with song B's instrumental as the noise profile.
    if all(k in stems_b for k in ("drums", "bass", "other")):
        # Oracle Wiener mask — gentler settings to avoid musical noise artifacts.
        # mask_floor=0.20 (was 0.08) — higher floor = less aggressive suppression
        # = less "tonal static" artifact from over-suppressed frequency bins.
        # drum_weight_hh=2.0 (was 3.5) — lighter drum weighting in hi-hat band.
        print("      Oracle Wiener mask (4-stem Demucs)…", flush=True)
        try:
            vox = _oracle_wiener_clean(
                vox, stems_b["drums"], stems_b["bass"], stems_b["other"],
                drum_weight_hh=_bleed_params.get("drum_weight_hh", 2.0),
                mask_floor=_bleed_params.get("wiener_mask_floor", 0.20))
            print("      Oracle Wiener done.", flush=True)
        except Exception as _e:
            print(f"      [Oracle Wiener failed: {_e} — falling back to 2-stem]",
                  flush=True)
            try:
                n_fft_ws = 2048
                vox_clean = np.zeros_like(vox)
                min_len = min(vox.shape[0], inst.shape[0])
                for c in range(vox.shape[1]):
                    ic = min(c, inst.shape[1] - 1)
                    D_v = librosa.stft(vox[:min_len, c], n_fft=n_fft_ws)
                    D_i = librosa.stft(inst[:min_len, ic], n_fft=n_fft_ws)
                    raw_mask = librosa.util.softmask(
                        np.abs(D_v), np.abs(D_i) + 1e-8, power=2)
                    mask = np.maximum(raw_mask, 0.20)
                    D_clean = (mask * np.abs(D_v)) * np.exp(1j * np.angle(D_v))
                    vox_clean[:, c] = librosa.istft(
                        D_clean, length=vox.shape[0]).astype(np.float32)
                vox = vox_clean.astype(np.float32)
            except Exception:
                pass

        # NOTE: Targeted spectral subtraction (_targeted_hihat_suppression) REMOVED.
        # It was causing phase artifacts on top of the Wiener mask — two stages of
        # spectral subtraction creates "musical noise" static that sounds scratchy.
        # The Wiener mask alone handles hi-hat bleed adequately.

        # NOTE: Harmonic resynthesis (_harmonic_vocal_process) REMOVED.
        # PYIN-based harmonic masking above 4kHz removes inter-harmonic content
        # that includes natural consonant noise — making the voice sound robotic/
        # vocoder-like. Better to accept residual bleed than destroy voice character.

        # Resemble Enhance: neural perceptual vocal enhancement.
        # Runs in a subprocess so a crash/segfault can't kill the main fuse process.
        # Enhancement-only mode (lambd=0.05) — denoiser disabled (fights musical content).
        try:
            import resemble_enhance as _re_pkg
            _re_model_path = Path(_re_pkg.__file__).parent / "model_repo"
            _re_has_weights = any(_re_model_path.rglob("*.pt")) or \
                              any(_re_model_path.rglob("*.bin")) or \
                              any(_re_model_path.rglob("*.safetensors"))
            if not _re_has_weights:
                print("      [Neural enhance: model weights not downloaded — skipping]",
                      flush=True)
            else:
                # Write vocal to a temp file, run enhance in subprocess, read result back.
                # Subprocess isolation: if model crashes (MPS/torch conflict), we skip cleanly.
                import sys as _sys_re, tempfile, subprocess as _sp
                print("      Neural vocal enhancement (Resemble Enhance)…", flush=True)
                with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as _tf_in:
                    _re_in = _tf_in.name
                with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as _tf_out:
                    _re_out = _tf_out.name
                sf.write(_re_in, vox, SR, subtype="PCM_24")
                _re_script = (
                    "import sys, numpy as np, soundfile as sf, torch\n"
                    "from resemble_enhance.enhancer.inference import enhance\n"
                    "vox, sr = sf.read(sys.argv[1], always_2d=True)\n"
                    "vox = vox.astype(np.float32)\n"
                    "enhanced = []\n"
                    "for c in range(vox.shape[1]):\n"
                    "    t = torch.from_numpy(vox[:, c])  # 1D — enhance() requires (N,)\n"
                    "    e, _ = enhance(t, sr, device='cpu', nfe=8, solver='midpoint', lambd=0.05, tau=0.5)\n"
                    "    enhanced.append(e.numpy())\n"
                    "out = np.stack(enhanced, axis=1).astype(np.float32)\n"
                    "sf.write(sys.argv[2], out, sr, subtype='PCM_24')\n"
                )
                _re_proc = _sp.run(
                    [_sys_re.executable, "-c", _re_script, _re_in, _re_out],
                    capture_output=True, text=True, timeout=600,
                )
                if _re_proc.returncode == 0:
                    _vox_enh, _ = sf.read(_re_out, always_2d=True)
                    _vox_enh = _vox_enh.astype(np.float32)
                    _peak_before = np.abs(vox).max() + 1e-9
                    _peak_after  = np.abs(_vox_enh).max() + 1e-9
                    if _peak_after / _peak_before < 4.0 and _vox_enh.shape == vox.shape:
                        vox = _vox_enh
                        print("      Neural vocal enhancement done.", flush=True)
                    else:
                        print("      [Neural enhance: output invalid — skipping]", flush=True)
                else:
                    print(f"      [Neural enhance subprocess failed (rc={_re_proc.returncode}) — skipping]",
                          flush=True)
                try:
                    os.unlink(_re_in); os.unlink(_re_out)
                except OSError:
                    pass
        except ImportError:
            pass   # resemble-enhance not installed — silently skip
        except Exception as _ree:
            print(f"      [Neural enhance failed: {_ree} — skipping]", flush=True)

        # Reference-guided noisereduce (moderate — oracle mask handles bulk).
        if "no_vocals" in stems_b:
            print("      Reference-guided vocal cleanup…", flush=True)
            try:
                ref_inst = stems_b["no_vocals"]
                min_len_nr = min(vox.shape[0], ref_inst.shape[0])
                vox_ref = np.zeros_like(vox)
                for c in range(vox.shape[1]):
                    ic = min(c, ref_inst.shape[1] - 1)
                    cleaned = nr.reduce_noise(
                        y=vox[:min_len_nr, c].astype(np.float32),
                        y_noise=ref_inst[:min_len_nr, ic].astype(np.float32),
                        sr=SR,
                        stationary=False,
                        prop_decrease=_bleed_params["noisereduce_strength"],
                        n_fft=2048,
                    )
                    vox_ref[:min_len_nr, c] = np.nan_to_num(
                        cleaned, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)
                    if min_len_nr < vox.shape[0]:
                        vox_ref[min_len_nr:, c] = vox[min_len_nr:, c]
                vox = vox_ref.astype(np.float32)
                print("      Reference-guided cleanup done.", flush=True)
            except Exception as _ref_e:
                print(f"      [Reference noisereduce failed: {_ref_e}]", flush=True)

    else:
        # 2-stem path (GPU BS-Roformer or MDX-Net without oracle stems).
        print("      Two-stem Wiener mask (fallback — no oracle stems)…", flush=True)
        try:
            n_fft_ws = 2048
            vox_clean = np.zeros_like(vox)
            min_len = min(vox.shape[0], inst.shape[0])
            for c in range(vox.shape[1]):
                ic = min(c, inst.shape[1] - 1)
                D_v = librosa.stft(vox[:min_len, c], n_fft=n_fft_ws)
                D_i = librosa.stft(inst[:min_len, ic], n_fft=n_fft_ws)
                mag_v = np.abs(D_v)
                mag_i = np.abs(D_i)
                raw_mask = librosa.util.softmask(mag_v, mag_i + 1e-8, power=2)
                mask = np.maximum(raw_mask, 0.15)
                D_clean = (mask * mag_v) * np.exp(1j * np.angle(D_v))
                vox_clean[:, c] = librosa.istft(D_clean, length=vox.shape[0]).astype(np.float32)
            vox = vox_clean.astype(np.float32)
            print("      Two-stem Wiener done.", flush=True)
        except Exception as _e:
            print(f"      [Two-stem Wiener failed: {_e} — skipping]", flush=True)

        # Reference-guided noisereduce (full strength — no oracle mask).
        if "no_vocals" in stems_b:
            print("      Reference-guided vocal cleanup (song B instrumental)…", flush=True)
            try:
                ref_inst = stems_b["no_vocals"]
                min_len_nr = min(vox.shape[0], ref_inst.shape[0])
                vox_ref = np.zeros_like(vox)
                for c in range(vox.shape[1]):
                    ic = min(c, ref_inst.shape[1] - 1)
                    cleaned = nr.reduce_noise(
                        y=vox[:min_len_nr, c].astype(np.float32),
                        y_noise=ref_inst[:min_len_nr, ic].astype(np.float32),
                        sr=SR,
                        stationary=False,
                        prop_decrease=0.50,
                        n_fft=2048,
                    )
                    vox_ref[:min_len_nr, c] = np.nan_to_num(
                        cleaned, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)
                    if min_len_nr < vox.shape[0]:
                        vox_ref[min_len_nr:, c] = vox[min_len_nr:, c]
                vox = vox_ref.astype(np.float32)
                print("      Reference-guided cleanup done.", flush=True)
            except Exception as _ref_e:
                print(f"      [Reference noisereduce failed: {_ref_e} — skipping]",
                      flush=True)

    ratio   = _best_ratio(bpm_a, bpm_b)
    n_semi  = semitones_to_shift(key_b_root, key_b_mode, key_a_root, key_a_mode)
    print(f"      BPM ratio: {ratio:.4f}   pitch shift: {n_semi:+d} semitones",
          flush=True)

    # Smart key shift: find best harmonic alternative if shift is large
    n_semi, key_msg = _smart_key_shift(n_semi, key_b_root, key_b_mode,
                                        key_a_root, key_a_mode)
    print(f"      Key: {key_msg}", flush=True)

    # Stretch direction: vocals are fragile, beats are robust.
    # If BPM gap > 8%, stretch the INSTRUMENTAL to match the vocal's tempo
    # instead of stretching the vocal. Beats handle time-stretching much
    # better than human voices (no formant/intelligibility sensitivity).
    VOCAL_STRETCH_LIMIT = 1.08   # ±8%: beyond this, stretch the beat instead
    stretch_vocal = ratio
    stretch_inst  = 1.0
    if abs(ratio - 1.0) > (VOCAL_STRETCH_LIMIT - 1.0):
        # Invert: keep vocal at 1.0x, slow/speed beat by inverse ratio
        stretch_inst  = 1.0 / ratio
        stretch_vocal = 1.0
        print(f"      [Stretch mode] Large BPM gap ({abs(ratio-1)*100:.0f}%): "
              f"stretching BEAT by {stretch_inst:.3f}x, vocals stay natural speed.",
              flush=True)
        if HAS_PYRUBBERBAND:
            stretched_inst = []
            for c in range(inst.shape[1]):
                stretched_inst.append(
                    rb.time_stretch(inst[:, c].astype(np.float32), SR,
                                    stretch_inst, rbargs={'-3': ''})
                )
            # Trim/pad all channels to same length
            min_len_si = min(len(s) for s in stretched_inst)
            inst = np.stack([s[:min_len_si] for s in stretched_inst],
                            axis=1).astype(np.float32)
        else:
            from pedalboard.io import AudioFile
            from pedalboard import time_stretch as pb_ts
            inst_ch = pb_time_stretch(
                inst.T.astype(np.float32), SR,
                stretch_factor=stretch_inst,
            ).T.astype(np.float32)
            inst = inst_ch
        print(f"      Beat stretched: {inst.shape[0]/SR:.1f}s at {bpm_a/stretch_inst:.1f} BPM",
              flush=True)

    step(6, TOTAL, "Analyzing audio content for AI-adaptive parameters…")
    # AI content analysis: derive all DSP parameters from actual audio
    beat_char  = _analyze_beat_character(full_a, bpm_a)
    vox_char   = _analyze_vocal_character(_to_mono(stems_b["vocals"]))
    # Beat sonic fingerprint: computed BEFORE _style_params so reverb can match
    # the beat's acoustic space and carve range can extend to cover the vocal F0.
    # Uses full_a (original song A pre-stem-separation) for the most representative
    # fingerprint of the beat's natural sound character.
    print("      Pre-computing beat sonic fingerprint for adaptive parameters…", flush=True)
    beat_fp_early = _beat_sonic_fingerprint(full_a, bpm_a)
    print(f"      Beat fingerprint: reverb_tail={beat_fp_early['reverb_tail']:.2f}  "
          f"saturation={beat_fp_early['saturation']:.2f}  "
          f"dynamic={beat_fp_early['dynamic_feel']:.2f}", flush=True)
    style      = _style_params(beat_char, vox_char, beat_fp=beat_fp_early)
    vox_params = _analyze_vocal_stem(vox)
    overlap    = _spectral_overlap(_to_mono(vox), _to_mono(inst))
    # Sidechain depth: depth=0.37 → beat ducks -4 dB when vocal is at peak.
    # Previous cap of 0.15 → only -1.6 dB max duck — barely audible. The beat
    # never stepped back for the vocal, making everything sound cluttered.
    sidechain_depth = float(np.clip(overlap * 0.6, 0.30, 0.45))
    print(f"      Beat: agg={beat_char['aggressiveness']:.2f}  "
          f"bass={beat_char['bass_weight']:.2f}  "
          f"brightness={beat_char['brightness']:.2f}", flush=True)
    print(f"      Vocal: rap_score={vox_char['rap_score']:.2f}  "
          f"onset_rate={vox_char['onset_rate']:.1f}/s  "
          f"pitch_range={vox_char['pitch_range']:.0f}st  "
          f"gender={vox_char['gender']} (F0={vox_char['median_f0']:.0f}Hz)", flush=True)
    print(f"      Style → FET {style['fet_ratio']:.1f}:1  "
          f"reverb_room={style['reverb_room']:.2f}  "
          f"reverb_wet={style['reverb_wet']:.2f}  "
          f"carve={style['carve_db']:.1f}dB [{style['carve_lo_hz']:.0f}-{style['carve_hi_hz']:.0f}Hz]  "
          f"vocal_level={style['vocal_level']:.2f}  "
          f"comp_eq={style['comp_eq_hz']:.0f}Hz", flush=True)
    print(f"      Gate thresh: {vox_params['gate_thresh_db']:.1f} dB  "
          f"Comp ratio: {vox_params['comp_ratio']:.1f}:1  "
          f"Spectral overlap: {overlap:.3f}  "
          f"Sidechain depth: {sidechain_depth:.2f}", flush=True)

    rb_engine = "pyrubberband R3" if HAS_PYRUBBERBAND else "pedalboard (fallback)"
    predelay_ms = min(60000.0 / max(bpm_a, 60.0) / 16.0, 40.0)
    step(7, TOTAL, f"Processing vocals (noisereduce + pitch-correct + stretch [{rb_engine}] + "
         f"de-esser + compress + presence + reverb {predelay_ms:.0f}ms)…")
    vox = _process_vocals(vox, stretch_vocal, n_semi, vox_params, style,
                          target_root=key_a_root, target_mode=key_a_mode,
                          bpm=bpm_a)
    vox = _check(vox, "post-vocal-chain")

    # ── Vocal Production for Beat ──────────────────────────────────────────────
    # The vocal was produced for Song B's sonic universe. This step re-produces
    # it for Song A's world: matching saturation character, reverb space, tonal
    # tilt, BPM-synced delay, and dynamic feel — making both elements feel like
    # they came from the same session rather than being pasted together.
    print("      Analyzing beat sonic fingerprint…", flush=True)
    fp = _beat_sonic_fingerprint(_to_mono(inst), bpm_a)
    print(f"      Fingerprint: sat={fp['saturation']:.2f}  bright={fp['brightness']:.2f}  "
          f"reverb={fp['reverb_tail']:.2f}  dynamic={fp['dynamic_feel']:.2f}  "
          f"texture={fp['texture']:.2f}  punch={fp['transient_punch']:.2f}", flush=True)

    # Estimate room IR from instrumental for acoustic space matching
    print("      Estimating room IR for acoustic space matching…", flush=True)
    try:
        _room_ir = _estimate_room_ir(_to_mono(inst), SR)
        print(f"      [Space Match] Room IR estimated: {len(_room_ir)/SR*1000:.0f}ms, estimated RT60 from IR",
              flush=True)
    except Exception as _ire:
        print(f"      [Space Match] IR estimation failed ({_ire}), using synthetic reverb only",
              flush=True)
        _room_ir = None

    vox = _produce_vocal_for_beat(vox, fp, bpm_a, ir_estimated=_room_ir)
    vox = _check(vox, "post-vocal-production")

    step(8, TOTAL, "Mixing (chorus-align + beat-snap + spectral carve + M/S + sidechain)…")

    # ── Stage 1: Structural alignment (chorus-to-chorus) ───────────────────────
    # Detect the first chorus start in both tracks. Aligning chorus-to-chorus
    # ensures the most energetic part of the vocal lands on the most energetic
    # part of the beat, rather than aligning by bar-1 which might be an intro.
    silence = lambda n: np.zeros((n, 2), dtype=np.float32)

    chorus_inst = _detect_section_start(full_a, section="chorus")
    chorus_vox  = _detect_section_start(full_b, section="chorus")
    print(f"      Chorus starts → inst: {chorus_inst/SR:.1f}s  "
          f"vocal: {chorus_vox/SR:.1f}s", flush=True)

    if chorus_inst > 0 and chorus_vox > 0:
        # Trim/pad so both choruses start at the same position
        if chorus_inst >= chorus_vox:
            # inst chorus is later — prepend silence to vocal to match
            pad_vox = chorus_inst - chorus_vox
            vox = np.concatenate([silence(pad_vox), vox], axis=0)
            print(f"      Structural align: +{pad_vox/SR*1000:.0f} ms pad to vocal", flush=True)
        else:
            # vocal chorus is later — trim inst start
            trim_inst = chorus_vox - chorus_inst
            inst = inst[trim_inst:]
            print(f"      Structural align: trim {trim_inst/SR*1000:.0f} ms from beat start", flush=True)
    else:
        print(f"      Structural align: section detection inconclusive, using beat-align", flush=True)

    # ── Stage 2: Fine beat-grid alignment (bar-level snap) ─────────────────────
    # After chorus alignment, snap the vocal to the nearest measure boundary
    # within the instrumental's beat grid.
    vox_pre, inst_pre = _beat_align(_to_mono(inst), _to_mono(vox))
    if vox_pre > 0:
        vox = np.concatenate([silence(vox_pre), vox], axis=0)
        print(f"      Beat-align: +{vox_pre/SR*1000:.0f} ms pad to vocal", flush=True)
    elif inst_pre > 0:
        inst = np.concatenate([silence(inst_pre), inst], axis=0)
        print(f"      Beat-align: +{inst_pre/SR*1000:.0f} ms pad to beat", flush=True)
    else:
        print(f"      Beat-align: no fine offset needed", flush=True)

    L = min(len(inst), len(vox))
    inst, vox = inst[:L], vox[:L]

    # Groove quantization disabled — causes clicks and can drift timing
    # vox = _groove_quantize(vox, _to_mono(inst), bpm_a, strength=quant_strength)

    # ── Stage 3: Section-aware arrangement gain automation ─────────────────────
    # Detect all sections in the time-aligned stems and generate per-sample gain
    # curves that create real dynamic structure instead of a flat overlay:
    #   - Beat pulls back (0.82x) in verse sections so the vocal owns that space
    #   - Beat pushes full (1.00x) in chorus/drop — big moment with full energy
    #   - Beat drops hard (0.55x) in breakdowns — builds tension before the drop
    #   - Vocal pushes slightly forward (1.08x) on the hook/chorus
    #   - Both curves are smoothed with 200ms ramps at every boundary (no clicks)
    print("      Detecting sections for arrangement automation…", flush=True)
    beat_secs = _detect_all_sections(_to_mono(inst))
    vox_secs  = _detect_all_sections(_to_mono(vox))
    if beat_secs:
        print("      Beat: " + " | ".join(
            f"{s['label']}@{s['start_s']:.0f}s" for s in beat_secs), flush=True)
    if vox_secs:
        print("      Vocal: " + " | ".join(
            f"{s['label']}@{s['start_s']:.0f}s" for s in vox_secs), flush=True)

    # ── Arrangement Intelligence: rearrange vocal sections for optimal compatibility ──
    if beat_secs and vox_secs and len(beat_secs) >= 2 and len(vox_secs) >= 2:
        print("      [Arrangement Intelligence] Scoring section compatibility…", flush=True)
        assignments = _arrange_sections(vox_secs, beat_secs)
        if assignments is not None:
            print("      [Arrangement Intelligence] Assignments:", flush=True)
            for vs, bs in assignments:
                score = _score_section_pair(vs, bs)
                print(f"        Vocal {vs['label']}@{vs['start_s']:.0f}s "
                      f"→ Beat {bs['label']}@{bs['start_s']:.0f}s  "
                      f"(score={score:.2f})", flush=True)
            vox_arranged = _stitch_sections(vox, assignments, sr=SR)
            if vox_arranged is not None:
                vox = vox_arranged
                print("      [Arrangement Intelligence] Vocal rearranged successfully.", flush=True)
            else:
                print("      [Arrangement Intelligence] Stitch safety check failed "
                      "— keeping original order.", flush=True)
        else:
            print("      [Arrangement Intelligence] Not enough sections for rearrangement.",
                  flush=True)

    if beat_secs or vox_secs:
        beat_gc, vox_gc = _arrangement_gain_curves(beat_secs, vox_secs, L)
        inst = (inst * beat_gc[:, np.newaxis]).astype(np.float32)
        vox  = (vox  * vox_gc[:, np.newaxis]).astype(np.float32)
        print("      Arrangement gain curves applied.", flush=True)

    # Save pre-mix stems so the auto-correction loop can re-mix without re-separating.
    # Stem separation is the slow step (2-5 min). Mix+master is ~20 seconds.
    inst_remix = inst.copy()
    vox_remix  = vox.copy()

    # ── Mix + Master (initial pass) ───────────────────────────────────────────
    mix = _iterative_mix(inst, vox, style, sidechain_depth, bpm_a)
    mix = _check(mix, "post-mix")
    mix_pre_variants = mix.copy()  # save pre-fade/pre-master for variants
    mix = _fade(mix, fade_s=2.0)

    step(9, TOTAL, "Mastering…")
    mix = _master(mix, bpm=bpm_a)
    mix = _check(mix, "post-master")

    os.makedirs(os.path.dirname(os.path.abspath(out_path)), exist_ok=True)
    sf.write(out_path, mix, SR, subtype="PCM_24")
    print(f"Done → {out_path}", flush=True)

    # ── Generate mix variants (Club / Intimate) ──────────────────────────────
    # Club: punchy sub, harder glue, LUFS -9, for DJ/speaker playback
    # Intimate: wide stereo, LUFS -14, high LRA, for headphones/streaming
    _base, _ext = os.path.splitext(out_path)
    out_club      = f"{_base}_club{_ext}"
    out_intimate  = f"{_base}_intimate{_ext}"
    _pre_master_mix = _fade(mix_pre_variants, fade_s=2.0)
    try:
        _mix_club = _master_club(_pre_master_mix.copy(), bpm=bpm_a)
        sf.write(out_club, _mix_club, SR, subtype="PCM_24")
        print(f"Club variant → {out_club}", flush=True)
    except Exception as _ve:
        print(f"[Club variant failed: {_ve}]", flush=True)
        out_club = None
    try:
        _mix_intimate = _master_intimate(_pre_master_mix.copy(), bpm=bpm_a)
        sf.write(out_intimate, _mix_intimate, SR, subtype="PCM_24")
        print(f"Intimate variant → {out_intimate}", flush=True)
    except Exception as _ve:
        print(f"[Intimate variant failed: {_ve}]", flush=True)
        out_intimate = None

    # ── Auto-correction loop (AI Director) ───────────────────────────────────
    # Score the mix. If it fails, the AI Director (Claude API) reads all 23 metrics,
    # reasons about root causes using causal knowledge of the DSP chain, and outputs
    # specific parameter adjustments. Falls back to lookup table if API unavailable.
    # Up to 3 passes — re-runs bleed-removal + mix + master on correction iterations.
    try:
        from listen import auto_score as _auto_score, corrections as _corrections, \
            score_file as _score_file
        try:
            from director import get_corrections as _get_ai_corrections
            _has_director = True
        except ImportError:
            _has_director = False

        best_score = 0
        best_mix   = mix.copy()
        _lufs_target = -12.0   # mutable within this loop
        _correction_history = []   # tracks prior attempts for director context

        for _attempt in range(3):
            print(f"\n── Quality Check (attempt {_attempt + 1}/3) {'─' * 35}",
                  flush=True)
            _passed, _score, _summary, _issues = _auto_score(out_path)
            # Also get raw metrics for the AI director
            _, _, _metrics = _score_file(out_path, print_report=False)
            print(f"  → {_summary}", flush=True)

            if _score > best_score:
                best_score = _score
                best_mix   = mix.copy()

            if _passed:
                print(f"  ✓ Mix passed QC on attempt {_attempt + 1}.", flush=True)
                break

            if _attempt == 2:
                print("  Max correction attempts reached — running grid search.", flush=True)
                # System 3: try parameter grid before giving up
                _gm, _gs, _gstyle = _param_grid_search(
                    inst_remix, vox_remix, style, sidechain_depth, bpm_a,
                    out_path, best_score)
                if _gm is not None and _gs > best_score:
                    best_score = _gs
                    best_mix   = _gm.copy()
                    style      = _gstyle
                    print(f"  Grid search improved score to {best_score}/100", flush=True)
                break

            # Record this attempt in history for the director's context
            _correction_history.append({
                "attempt": _attempt,
                "score": _score,
                "params": {
                    "carve_db": style["carve_db"],
                    "presence_db": style.get("presence_db", 2.0),
                    "air_db": style.get("air_db", 2.5),
                    "vocal_level": style["vocal_level"],
                    "lufs_target": _lufs_target,
                    **_bleed_params,
                },
                "issues": [i[1] for i in _issues],
            })

            # Build current param snapshot for the director
            _current_params = {
                "carve_db": style["carve_db"],
                "presence_db": style.get("presence_db", 2.0),
                "air_db": style.get("air_db", 2.5),
                "vocal_level": style["vocal_level"],
                "lufs_target": _lufs_target,
                **_bleed_params,
            }

            # Ask AI Director (or fall back to lookup table)
            if _has_director:
                _adj = _get_ai_corrections(
                    metrics=_metrics,
                    issues=_issues,
                    current_params=_current_params,
                    attempt_num=_attempt,
                    history=_correction_history[:-1],  # exclude current attempt
                )
            else:
                _adj = {}

            if not _adj:
                # Fall back to static lookup table
                _adj_raw = _corrections(_issues)
                # Convert deltas to absolute values
                _adj = {}
                if "carve_db" in _adj_raw:
                    _adj["carve_db"] = float(np.clip(
                        style["carve_db"] + _adj_raw["carve_db"], 3.0, 12.0))
                if "presence_db" in _adj_raw:
                    _adj["presence_db"] = float(np.clip(
                        style.get("presence_db", 1.5) + _adj_raw["presence_db"], 0.0, 4.0))
                if "air_db" in _adj_raw:
                    _adj["air_db"] = float(np.clip(
                        style.get("air_db", 2.5) + _adj_raw["air_db"], 0.0, 5.0))
                if "vocal_level" in _adj_raw:
                    _adj["vocal_level"] = float(np.clip(
                        style["vocal_level"] + _adj_raw["vocal_level"], 0.5, 4.0))
                if "lufs_delta" in _adj_raw:
                    _adj["lufs_target"] = float(np.clip(
                        _lufs_target + _adj_raw["lufs_delta"], -16.0, -9.0))

            if not _adj:
                print("  No correctable issues found — keeping current mix.", flush=True)
                break

            print(f"  Applying: {_adj}", flush=True)

            # Determine if we need to re-run bleed removal (bleed params changed)
            _bleed_changed = any(k in _adj for k in _bleed_params)

            # Apply mix-layer parameter adjustments
            if "carve_db"    in _adj:
                style["carve_db"]    = float(np.clip(_adj["carve_db"], 3.0, 12.0))
            if "presence_db" in _adj:
                style["presence_db"] = float(np.clip(_adj["presence_db"], 0.0, 4.0))
            if "air_db"      in _adj:
                style["air_db"]      = float(np.clip(_adj["air_db"], 0.0, 5.0))
            if "vocal_level" in _adj:
                style["vocal_level"] = float(np.clip(_adj["vocal_level"], 1.0, 4.0))
            if "lufs_target" in _adj:
                _lufs_target = float(np.clip(_adj["lufs_target"], -16.0, -9.0))

            # Apply bleed-removal parameter adjustments
            for _bp in _bleed_params:
                if _bp in _adj:
                    _bleed_params[_bp] = _adj[_bp]

            print(f"  New params: carve={style['carve_db']:.1f}dB  "
                  f"presence={style.get('presence_db',2.0):.1f}dB  "
                  f"vocal_level={style['vocal_level']:.2f}  "
                  f"lufs={_lufs_target:.1f}", flush=True)

            # If bleed params changed, re-run bleed removal on the original stems
            _vox_for_remix = vox_remix
            if _bleed_changed and all(k in stems_b for k in ("drums", "bass", "other")):
                print("  Re-running bleed removal with updated parameters…", flush=True)
                try:
                    _vox_rebleed = stems_b["vocals"].copy()
                    _vox_rebleed = _oracle_wiener_clean(
                        _vox_rebleed, stems_b["drums"], stems_b["bass"], stems_b["other"],
                        drum_weight_hh=_bleed_params["drum_weight_hh"],
                        mask_floor=_bleed_params["wiener_mask_floor"])
                    if "no_vocals" in stems_b:
                        _ref_i = stems_b["no_vocals"]
                        _ml_nr = min(_vox_rebleed.shape[0], _ref_i.shape[0])
                        _vb_nr = np.zeros_like(_vox_rebleed)
                        for _c in range(_vox_rebleed.shape[1]):
                            _ic = min(_c, _ref_i.shape[1] - 1)
                            _cl = nr.reduce_noise(
                                y=_vox_rebleed[:_ml_nr, _c].astype(np.float32),
                                y_noise=_ref_i[:_ml_nr, _ic].astype(np.float32),
                                sr=SR, stationary=False,
                                prop_decrease=_bleed_params["noisereduce_strength"],
                                n_fft=2048)
                            _vb_nr[:_ml_nr, _c] = np.nan_to_num(
                                _cl, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)
                            if _ml_nr < _vox_rebleed.shape[0]:
                                _vb_nr[_ml_nr:, _c] = _vox_rebleed[_ml_nr:, _c]
                        _vox_rebleed = _vb_nr.astype(np.float32)
                    _vox_for_remix = _vox_rebleed
                    print("  Bleed removal updated.", flush=True)
                except Exception as _bre:
                    print(f"  [Bleed re-run failed: {_bre} — using cached stems]", flush=True)

            # Re-run mix + master with adjusted params
            print("  Re-mixing…", flush=True)
            mix = _iterative_mix(inst_remix, _vox_for_remix, style, sidechain_depth, bpm_a)
            mix = _check(mix, f"remix-{_attempt+2}/mix")
            mix = _fade(mix, fade_s=2.0)
            mix = _master(mix, bpm=bpm_a)
            mix = _check(mix, f"remix-{_attempt+2}/master")
            sf.write(out_path, mix, SR, subtype="PCM_24")

        # Write the highest-scoring version
        sf.write(out_path, best_mix, SR, subtype="PCM_24")
        print(f"\n  Final score: {best_score}/100 — {out_path}", flush=True)

    except Exception as _qe:
        import traceback
        print(f"  [Auto-correction unavailable: {_qe}]", flush=True)
        traceback.print_exc()
        # Legacy 7-point check as fallback
        ev = _auto_evaluate(mix, inst, vox, bpm_a)
        print(f"  Beat sync: {ev['beat_sync_pct']}%  "
              f"Vocal: {ev['vocal_presence']}%  "
              f"LUFS: {ev['lufs']}  "
              f"{'PASS' if ev['pass'] else 'FAIL'}", flush=True)

    return {
        "radio":    out_path,
        "club":     out_club,
        "intimate": out_intimate,
        "score":    int(best_score) if best_score > 0 else None,
    }
