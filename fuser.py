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
import logging
import os
import tempfile
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
from scipy.signal import butter, sosfilt

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
_BS_ROFORMER = "model_bs_roformer_ep_317_sdr_12.9755.ckpt"
_MDX_VOCAL   = "Kim_Vocal_2.onnx"  # MDX-Net vocal (SDR ~9.5+, ONNX — fast on CPU)


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


def _style_params(beat_char: dict, vox_char: dict) -> dict:
    """
    Map continuous style scores → concrete DSP parameter values.
    All parameters derived from audio content — nothing hardcoded.
    """
    agg  = beat_char["aggressiveness"]   # 0–1
    bass = beat_char["bass_weight"]      # 0–1
    rap  = vox_char["rap_score"]         # 0–1

    return {
        "_rap_score": rap,  # pass-through for ADT and other downstream use
        # FET compressor: rap/trap → faster, harder
        # Ratio cap 6:1 (was 8:1 — over-squashed vocal, LRA collapsed to 2.5 LU)
        # Release min 60ms (was 40ms — too fast, caused pumping artifacts)
        # Non-rap floor 2.2:1 (was 3.5:1 — over-squashing caused unnatural dynamics)
        "fet_ratio":    float(np.interp(rap, [0, 1], [2.2, 6.0])),
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

        # Reverb: very dry. AI-separated stems already have the original recording's
        # reverb baked in. Adding more reverb = double-reverb = wash/blur.
        # Max 4% wet for singing, max 2% for rap. Just enough for space, not blur.
        "reverb_room":  float(np.interp(rap, [0, 1], [0.15, 0.06])),
        "reverb_damp":  float(np.interp(rap, [0, 1], [0.70, 0.90])),
        "reverb_wet":   float(np.interp(rap, [0, 1], [0.04, 0.02])),

        # Spectral carve: more bass-heavy → carve deeper. 8-10 dB for house/EDM.
        "carve_db":     float(np.interp(bass, [0, 1], [6.0, 10.0])),

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


def _produce_vocal_for_beat(vox: np.ndarray, fp: dict, bpm: float) -> np.ndarray:
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

        # Process instrumental
        inst_c = _adaptive_spectral_carve(inst, vox_scaled, carve_db=carve_db)
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


def separate(audio_path: str, cache_dir: str = "vf_data/stems") -> dict:
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
            # CPU path: Demucs htdemucs_ft 4-stem.
            # Returns drums/bass/other/vocals — oracle stems enable a far more
            # accurate Wiener mask than MDX-Net's blind 2-stem estimate.
            # MDX-Net ensemble (higher SDR in isolation) was replaced because
            # oracle post-processing closes the quality gap and fixes bleed.
            print("      CPU path: Demucs htdemucs_ft 4-stem separation…", flush=True)
            _separate_demucs(audio_path, cached)

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


def _oracle_wiener_clean(vox: np.ndarray,
                          drums: np.ndarray,
                          bass: np.ndarray,
                          other: np.ndarray) -> np.ndarray:
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
    DRUM_WEIGHT  = 3.5            # drums count 3.5× as interference in hi-hat band
    MASK_FLOOR   = 0.08           # lower floor (oracle accuracy → can suppress more)

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


def _targeted_hihat_suppression(vox: np.ndarray, drums: np.ndarray) -> np.ndarray:
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
    ALPHA  = 0.90    # spectral subtraction coefficient
    BETA   = 0.06    # floor: never suppress below 6% of original

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


def _harmonic_vocal_process(vox: np.ndarray) -> np.ndarray:
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
    MIX_DRY  = 0.50     # 50% original — prevents vocoder character

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
    Robust BPM detection for hip-hop/trap (typically 70-200 BPM actual).

    Uses two-stage approach to handle half-tempo detection:
    1. Primary: librosa.beat.beat_track (most robust for musical content)
    2. Fallback: onset-envelope tempo if beat_track gives unreasonable result
    3. Half-tempo correction: if detected BPM suggests half-tempo, try doubling

    Hip-hop half-tempo problem: 140 BPM trap often detected as 70 BPM because
    the hi-hat pattern runs at 140 but the kick/snare pattern matches 70 BPM
    equally well. We resolve this by checking which is closer to 90-160 BPM range.
    """
    hop = 512
    onset_env = librosa.onset.onset_strength(y=y_mono, sr=SR, hop_length=hop)

    # Method 1: beat_track (robust, uses beat autocorrelation)
    bpm_bt, _ = librosa.beat.beat_track(onset_envelope=onset_env, sr=SR,
                                         hop_length=hop)
    bpm_bt = float(bpm_bt)

    # Method 2: prior-informed tempo — lognormal prior centered at 130 BPM
    # covers 80-200 BPM with high probability, much more reliable for hip-hop
    try:
        from scipy.stats import lognorm as _lognorm
        hip_hop_prior = _lognorm(s=0.35, scale=130)
        bpm_oe = float(librosa.feature.rhythm.tempo(
            onset_envelope=onset_env, sr=SR, hop_length=hop,
            prior=hip_hop_prior)[0])
    except Exception:
        bpm_oe = float(librosa.feature.rhythm.tempo(
            onset_envelope=onset_env, sr=SR, hop_length=hop)[0])

    # Pick the one closer to the hip-hop range (90-170 BPM)
    def distance_to_hiphop(bpm):
        bpm = float(bpm)
        return min(abs(bpm - 90), abs(bpm - 140), abs(bpm - 170))

    bpm = bpm_bt if distance_to_hiphop(bpm_bt) <= distance_to_hiphop(bpm_oe) else bpm_oe

    # Normalize range (handle half/double tempo detection)
    while bpm > 200.0:
        bpm /= 2
    while bpm < 70.0:
        bpm *= 2

    # Half-tempo check: if 70-89 BPM, check if 2× is more plausible for hip-hop
    # Hip-hop BPM range: most trap is 130-170, most hip-hop 85-115
    # If detected is 70-84: likely half of 140-168 (trap) — if onset density suggests
    # high onset rate (many hi-hats), double it
    if 70.0 <= bpm < 88.0:
        onset_rate = len(librosa.onset.onset_detect(
            onset_envelope=onset_env, sr=SR, hop_length=hop)) / (len(y_mono) / SR)
        # Trap has >4 onsets/second due to hi-hats; regular hip-hop has 2-4
        if onset_rate > 4.5:
            bpm *= 2.0
            print(f"      BPM half-tempo corrected: ×2 (onset_rate={onset_rate:.1f}/s)",
                  flush=True)

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
    chroma = librosa.feature.chroma_cens(y=y_mono, sr=SR)
    mean_ch = chroma.mean(axis=1)
    mean_ch /= mean_ch.sum() + 1e-9
    best_score, best_root, best_mode = -np.inf, 0, "major"
    for root in range(12):
        rot = np.roll(mean_ch, -root)
        for profile, mode in ((_KS_MAJOR, "major"), (_KS_MINOR, "minor")):
            score = np.dot(rot, profile / profile.sum())
            if score > best_score:
                best_score, best_root, best_mode = score, root, mode
    return best_root, best_mode


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
    Returns a list of dicts: {start_s, end_s, label, energy, brightness, density}
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
            raw.append({
                'start_s':   t_s,
                'end_s':     t_e,
                'energy':    float(rms_f[f_s:f_e].mean()),
                'brightness':float(cen_f[f_s:f_e].mean()),
                'density':   float(ons_f[f_s:f_e].mean()),
                'label':     'unknown',
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
                              smooth_sigma: float = 2.5) -> np.ndarray:
    """
    Content-aware spectral carving using a Wiener soft-mask.

    For each time-frequency bin, computes how dominant the vocal is vs the
    beat, then reduces the beat in exactly those frequency slots.
    This replaces the old fixed-frequency EQ cut with a dynamic carve that
    adapts to whatever frequencies the vocal actually uses.

    carve_db: max cut in beat where vocal is loudest (5 dB = perceptually clean)
    smooth_sigma: temporal smoothing to prevent pumping/zipper artifacts
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

    # Frequency weighting: carve ONLY the vocal intelligibility zone (300 Hz–5 kHz).
    # Research: 2-4 kHz is the ear canal resonance peak (3.3 kHz) and the
    # single most important band for vocal intelligibility in a dense mix.
    # STOP AT 5 kHz: above 5 kHz is air/cymbals from the beat. Spectral analysis
    # showed the mix was -5 dB dark in 6-20kHz band partly because this carve
    # was extending to 8kHz and cutting beat HF content during vocal sections.
    freqs = librosa.fft_frequencies(sr=SR, n_fft=n_fft)
    freq_w = np.zeros(len(freqs), dtype=np.float32)
    for i, f in enumerate(freqs):
        if f < 200 or f > 5000:
            freq_w[i] = 0.0                             # no carve outside 200-5kHz
        elif f < 300:
            freq_w[i] = (f - 200) / 100 * 0.4         # gentle ramp 200-300 Hz (mud zone)
        elif f < 600:
            freq_w[i] = 0.4 + (f - 300) / 300 * 0.4   # ramp to 0.8 at 600 Hz
        elif f < 1000:
            freq_w[i] = 0.8 + (f - 600) / 400 * 0.2   # ramp to 1.0 at 1kHz
        else:
            # 2-4kHz: peak weight 2.0× at 3kHz — vocal presence / cut-through zone.
            # Fletcher-Munson: at high volume bass masks this zone → carve deeper.
            freq_w[i] = 1.0 + 1.0 * float(np.clip(
                1.0 - abs(np.log2(f / 3000)) * 1.5, 0, 1))  # peak 2.0× at 3kHz

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
        return float(librosa.amplitude_to_db(S[m].mean() + 1e-9, ref=1.0))

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
            PeakFilter(cutoff_frequency_hz=350.0, gain_db=-0.75, q=1.2),  # reduced -1.5→-0.75: Lo-Mid
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
        PeakFilter(cutoff_frequency_hz=250.0,  gain_db=-0.5,  q=0.8),  # mud cut
        PeakFilter(cutoff_frequency_hz=500.0,  gain_db=-1.5,  q=0.7),  # lo-mid warmth control (250-800Hz)
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


def _fade(y: np.ndarray, fade_s: float = 2.0) -> np.ndarray:
    n = min(int(SR * fade_s), len(y) // 6)
    y = y.copy()
    y[:n]  *= np.linspace(0.0, 1.0, n, dtype=np.float32)[:, np.newaxis] ** 0.5
    y[-n:] *= np.linspace(1.0, 0.0, n, dtype=np.float32)[:, np.newaxis] ** 0.5
    return y


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

    step(2, TOTAL, "Detecting BPM…")
    bpm_a = detect_bpm(full_a)
    bpm_b = detect_bpm(full_b)
    print(f"      A: {bpm_a:.1f} BPM   B: {bpm_b:.1f} BPM", flush=True)

    step(3, TOTAL, "Detecting keys…")
    key_a_root, key_a_mode = detect_key(full_a)
    key_b_root, key_b_mode = detect_key(full_b)
    print(f"      A: {_NOTES[key_a_root]} {key_a_mode}   "
          f"B: {_NOTES[key_b_root]} {key_b_mode}", flush=True)

    sep_model = "BS-Roformer" if _has_gpu() else "MDX-Net Kim Vocal 2→Demucs fallback"
    step(4, TOTAL, f"Separating stems — Song A (instrumental) via {sep_model}…")
    stems_a = separate(song_a, stems_cache)

    step(5, TOTAL, f"Separating stems — Song B (vocals) via {sep_model}…")
    stems_b = separate(song_b, stems_cache)

    inst = stems_a["no_vocals"]   # (samples, 2)
    vox  = stems_b["vocals"]      # (samples, 2)

    # ── Vocal bleed removal ──────────────────────────────────────────────────
    # Oracle path (4-stem Demucs): use drums/bass/other as oracle interference
    # sources for an exact Wiener mask.  The mask for bin (t,f) is:
    #   mask = V² / (V² + D² + B² + O²)
    # Drums get 2× weight in the 4-16 kHz hi-hat band — this directly targets
    # the "scratchy" artifact the user has been hearing.
    # After masking, PYIN harmonic enhancement restores overtones the mask
    # suppressed alongside the bleed.
    #
    # Fallback path (2-stem / GPU): classical V²/(V²+I²) Wiener + reference
    # noisereduce with song B's instrumental as the noise profile.
    if all(k in stems_b for k in ("drums", "bass", "other")):
        print("      Oracle Wiener mask (4-stem Demucs)…", flush=True)
        try:
            vox = _oracle_wiener_clean(
                vox, stems_b["drums"], stems_b["bass"], stems_b["other"])
            print("      Oracle Wiener done.", flush=True)
            # Targeted spectral subtraction: remove residual drum hi-hat content
            # directly from the 6-16kHz band of the vocal (after Wiener mask).
            print("      Targeted hi-hat spectral subtraction…", flush=True)
            vox = _targeted_hihat_suppression(vox, stems_b["drums"])
            print("      Hi-hat suppression done.", flush=True)
        except Exception as _e:
            print(f"      [Oracle Wiener failed: {_e} — falling back to 2-stem]",
                  flush=True)
            # 2-stem fallback
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
                    mask = np.maximum(raw_mask, 0.15)
                    D_clean = (mask * np.abs(D_v)) * np.exp(1j * np.angle(D_v))
                    vox_clean[:, c] = librosa.istft(
                        D_clean, length=vox.shape[0]).astype(np.float32)
                vox = vox_clean.astype(np.float32)
            except Exception:
                pass

        # Harmonic resynthesis + enhancement: rebuild vocal from harmonic skeleton.
        # Removes residual inter-harmonic bleed + boosts harmonics the Wiener
        # over-suppressed — both in one PYIN pass.
        print("      Harmonic resynthesis (PYIN F0 skeleton rebuild)…", flush=True)
        try:
            vox = _harmonic_vocal_process(vox)
            print("      Harmonic resynthesis done.", flush=True)
        except Exception as _he:
            print(f"      [Harmonic resynth failed: {_he} — skipping]", flush=True)

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
                        prop_decrease=0.35,   # gentler — oracle mask already did heavy lifting
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
    style      = _style_params(beat_char, vox_char)
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
          f"carve={style['carve_db']:.1f}dB  "
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
    vox = _produce_vocal_for_beat(vox, fp, bpm_a)
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
    mix = _fade(mix, fade_s=2.0)

    step(9, TOTAL, "Mastering…")
    mix = _master(mix, bpm=bpm_a)
    mix = _check(mix, "post-master")

    os.makedirs(os.path.dirname(os.path.abspath(out_path)), exist_ok=True)
    sf.write(out_path, mix, SR, subtype="PCM_24")
    print(f"Done → {out_path}", flush=True)

    # ── Auto-correction loop ─────────────────────────────────────────────────
    # Score the mix. If it fails (< 75/100 or CRITICAL issue), apply the
    # corrections() map to DSP params and re-run just mix+master. Up to 3 passes.
    # The user never needs to listen and tell us what's wrong — we fix it ourselves.
    try:
        from listen import auto_score as _auto_score, corrections as _corrections
        best_score = 0
        best_mix   = mix.copy()
        _lufs_target = -12.0   # mutable within this loop

        for _attempt in range(3):
            print(f"\n── Quality Check (attempt {_attempt + 1}/3) {'─' * 35}",
                  flush=True)
            _passed, _score, _summary, _issues = _auto_score(out_path)
            print(f"  → {_summary}", flush=True)

            if _score > best_score:
                best_score = _score
                best_mix   = mix.copy()

            if _passed:
                print(f"  ✓ Mix passed QC on attempt {_attempt + 1}.", flush=True)
                break

            if _attempt == 2:
                print("  Max correction attempts reached — using best result.", flush=True)
                break

            # Get parameter deltas from issue map
            _adj = _corrections(_issues)
            if not _adj:
                print("  No correctable issues found — keeping current mix.", flush=True)
                break

            print(f"  Auto-correcting: {_adj}", flush=True)

            # Apply corrections to style dict (clamped to safe ranges)
            if "carve_db" in _adj:
                style["carve_db"] = float(np.clip(
                    style["carve_db"] + _adj["carve_db"], 3.0, 12.0))
            if "presence_db" in _adj:
                style["presence_db"] = float(np.clip(
                    style.get("presence_db", 1.5) + _adj["presence_db"], 0.0, 4.0))
            if "air_db" in _adj:
                style["air_db"] = float(np.clip(
                    style.get("air_db", 2.5) + _adj["air_db"], 0.0, 5.0))
            if "vocal_level" in _adj:
                style["vocal_level"] = float(np.clip(
                    style["vocal_level"] + _adj["vocal_level"], 0.5, 2.5))
            if "lufs_delta" in _adj:
                _lufs_target = float(np.clip(_lufs_target + _adj["lufs_delta"], -16.0, -9.0))

            print(f"  New params: carve={style['carve_db']:.1f}dB  "
                  f"vocal_level={style['vocal_level']:.2f}  "
                  f"lufs={_lufs_target:.1f}", flush=True)

            # Re-run mix + master with adjusted params
            print("  Re-mixing…", flush=True)
            mix = _iterative_mix(inst_remix, vox_remix, style, sidechain_depth, bpm_a)
            mix = _check(mix, f"remix-{_attempt+2}/mix")
            mix = _fade(mix, fade_s=2.0)
            mix = _master(mix, bpm=bpm_a)
            mix = _check(mix, f"remix-{_attempt+2}/master")
            sf.write(out_path, mix, SR, subtype="PCM_24")

        # Write the highest-scoring version
        sf.write(out_path, best_mix, SR, subtype="PCM_24")
        print(f"\n  Final score: {best_score}/100 — {out_path}", flush=True)

    except Exception as _qe:
        print(f"  [Auto-correction unavailable: {_qe}]", flush=True)
        # Legacy 7-point check as fallback
        ev = _auto_evaluate(mix, inst, vox, bpm_a)
        print(f"  Beat sync: {ev['beat_sync_pct']}%  "
              f"Vocal: {ev['vocal_presence']}%  "
              f"LUFS: {ev['lufs']}  "
              f"{'PASS' if ev['pass'] else 'FAIL'}", flush=True)

    return out_path
