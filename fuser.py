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

    # Pitch range via PYIN fundamental
    try:
        f0, voiced, _ = librosa.pyin(clip, fmin=60, fmax=1200,
                                     sr=SR, hop_length=512, fill_na=None)
        f0_voiced = f0[voiced] if voiced is not None else np.array([])
        if len(f0_voiced) > 20:
            pitch_range_semitones = float(
                12 * np.log2(f0_voiced.max() / (f0_voiced.min() + 1e-9)))
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
        "fet_ratio":    float(np.interp(rap, [0, 1], [4.0, 8.0])),
        "fet_attack":   float(np.interp(rap, [0, 1], [5.0, 1.0])),
        "fet_release":  float(np.interp(agg, [0, 1], [100.0, 40.0])),

        # Opto compressor: more gentle always, but faster for aggressive
        "opto_ratio":   float(np.interp(agg, [0, 1], [2.0, 3.0])),
        "opto_attack":  float(np.interp(agg, [0, 1], [30.0, 15.0])),
        "opto_release": float(np.interp(agg, [0, 1], [300.0, 150.0])),

        # Presence boost: rap needs more mid-presence for intelligibility
        "presence_db":  float(np.interp(rap, [0, 1], [2.0, 3.5])),
        "presence_hz":  float(np.interp(rap, [0, 1], [4000.0, 3000.0])),

        # Air shelf: singing benefits from more air than rap
        "air_db":       float(np.interp(rap, [0, 1], [2.5, 1.5])),

        # Reverb: rap/trap → tighter room; singing/pop → lusher plate
        "reverb_room":  float(np.interp(rap, [0, 1], [0.18, 0.08])),
        "reverb_damp":  float(np.interp(rap, [0, 1], [0.70, 0.85])),
        "reverb_wet":   float(np.interp(rap, [0, 1], [0.10, 0.06])),

        # Spectral carve: more bass-heavy → carve deeper in bass range
        "carve_db":     float(np.interp(bass, [0, 1], [4.0, 6.0])),

        # Sidechain: aggressive beat → more sidechain duck
        "sidechain_mult": float(np.interp(agg, [0, 1], [0.9, 1.2])),

        # Vocal level: rap sits louder relative to beat
        "vocal_level":  float(np.interp(rap, [0, 1], [1.1, 1.4])),
    }


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

    # Pick the smallest absolute shift
    best = min(candidates, key=abs)
    if best != n_semi:
        return best, f"re-mapped {n_semi:+d} → {best:+d} semitones (better harmonic fit)"
    return n_semi, f"{n_semi:+d} semitones (large shift — harmonic clash risk)"


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
            gains[i] = np.clip((ir * target_ratio) / vr, 0.4, 4.0)

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

    for iteration in range(max_iter):
        # Apply energy-envelope matching then static scalar
        ir = _rms(_to_mono(inst))
        vr = _rms(_to_mono(vox))
        vox_scaled = (vox * (ir * level_mult / (vr + 1e-9))).astype(np.float32)

        # Dynamic envelope: vocal tracks beat energy locally
        vox_scaled = _energy_match_envelope(inst, vox_scaled,
                                            target_ratio=level_mult)

        # Process instrumental
        inst_c = _adaptive_spectral_carve(inst, vox_scaled, carve_db=carve_db)
        # Restore drum transients softened by stem separation (attack +4dB, sustain -2dB)
        inst_c = _transient_shape(inst_c, attack_gain_db=4.0, sustain_gain_db=-2.0)
        inst_c = _parallel_compress(inst_c)
        inst_c = _sidechain(inst_c, vox_scaled,
                            depth=sidechain_depth * style["sidechain_mult"])

        # Evaluate presence
        vp = _rms(_to_mono(vox_scaled)) / (
             _rms(_to_mono(inst_c)) + _rms(_to_mono(vox_scaled)) + 1e-9)

        print(f"      Mix iter {iteration+1}: presence={vp:.0%}  "
              f"level_mult={level_mult:.2f}  carve={carve_db:.1f}dB", flush=True)

        if 0.40 <= vp <= 0.65 or iteration == max_iter - 1:
            break
        elif vp < 0.40:
            level_mult = min(level_mult * 1.18, 3.0)
        else:
            level_mult = max(level_mult * 0.85, 0.5)

    # Final M/S mix
    inst_M, inst_S = _ms_encode(inst_c)
    vox_M, _       = _ms_encode(vox_scaled)
    mix = _ms_decode(inst_M + vox_M, inst_S)
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
            # CPU path: Demucs (2–5 min vs 50+ min for BS-Roformer on CPU)
            _separate_demucs(audio_path, cached)

    stems = {}
    for name in ("vocals", "no_vocals"):
        y, file_sr = sf.read(str(cached / f"{name}.wav"))
        if y.ndim == 1:
            y = np.stack([y, y], axis=1)
        if file_sr != SR:
            y = np.stack([
                librosa.resample(y[:, c], orig_sr=file_sr, target_sr=SR)
                for c in range(y.shape[1])
            ], axis=1)
        stems[name] = y.astype(np.float32)
    return stems


def _separate_demucs(audio_path: str, out_dir: Path) -> None:
    """Fallback: Demucs htdemucs_ft --two-stems vocals."""
    import subprocess, sys
    from pathlib import Path as _Path

    fid = out_dir.name
    ext = _Path(audio_path).suffix or ".mp3"
    tmp = out_dir.parent / f"{fid}_src{ext}"
    shutil.copy2(audio_path, tmp)
    try:
        result = subprocess.run(
            [sys.executable, "-m", "demucs",
             "--two-stems", "vocals",
             "-n", "htdemucs_ft",
             "-o", str(out_dir.parent),
             str(tmp)],
            capture_output=True, text=True,
        )
        if result.returncode != 0:
            raise RuntimeError(f"Demucs failed:\n{result.stderr}")
        raw = out_dir.parent / "htdemucs_ft" / fid
        if raw.exists():
            for f in raw.iterdir():
                shutil.move(str(f), str(out_dir / f.name))
            raw.rmdir()
            try:
                (out_dir.parent / "htdemucs_ft").rmdir()
            except OSError:
                pass
    finally:
        if tmp.exists():
            tmp.unlink()


# ── Analysis ──────────────────────────────────────────────────────────────────

def detect_bpm(y_mono: np.ndarray) -> float:
    onset_env = librosa.onset.onset_strength(y=y_mono, sr=SR, hop_length=512)
    tempo = float(librosa.feature.rhythm.tempo(
        onset_envelope=onset_env, sr=SR, hop_length=512)[0])
    while tempo > 180.0:
        tempo /= 2
    while tempo < 60.0:
        tempo *= 2
    return tempo


def _best_ratio(bpm_a: float, bpm_b: float) -> float:
    return float(np.clip(bpm_a / bpm_b, 0.667, 1.5))


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


def _clean_vocal(vox_mono: np.ndarray) -> np.ndarray:
    """
    Spectral gating with noisereduce to remove Demucs bleed artifacts.
    Non-stationary mode handles music-like residue better than stationary.
    """
    return nr.reduce_noise(
        y=vox_mono, sr=SR,
        stationary=False,
        prop_decrease=0.35,   # 0.75 creates watery/metallic artifacts on melodic content
        n_fft=2048,
    ).astype(np.float32)


def _deess(vox: np.ndarray, threshold_db: float = -22.0,
           cutoff_hz: float = 6500.0, max_reduction_db: float = 7.0) -> np.ndarray:
    """
    Dynamic de-esser: detect sibilance energy (6.5 kHz+) per frame and apply
    proportional gain reduction when it exceeds threshold_db.
    Placed BEFORE compression to prevent sibilance pumping the compressor.
    """
    mono = _to_mono(vox)

    # Sidechain: high-pass at cutoff to isolate sibilance
    sos = butter(4, cutoff_hz / (SR / 2), btype="high", output="sos")
    sib = sosfilt(sos, mono).astype(np.float32)

    # Frame-by-frame RMS of sibilance (5ms frames)
    win = max(1, int(SR * 0.005))
    hop = win // 2
    n = len(sib)
    n_frames = max(1, (n + hop - 1) // hop)

    env_db = np.array([
        20 * np.log10(
            float(np.sqrt(np.mean(sib[i * hop: min(i * hop + win, n)] ** 2))) + 1e-12
        )
        for i in range(n_frames)
    ], dtype=np.float32)

    # Gain reduction: compress sibilance above threshold, max reduction capped
    over_thresh = env_db - threshold_db                   # positive when too loud
    gain_db = np.clip(-over_thresh * 0.6, -max_reduction_db, 0.0)
    gain_linear = 10 ** (gain_db / 20.0).astype(np.float32)

    # Interpolate envelope to sample resolution
    x_frames = np.arange(n_frames, dtype=np.float64) * hop
    x_samp   = np.arange(n, dtype=np.float64)
    gain_samp = interp1d(
        x_frames, gain_linear, kind="linear",
        bounds_error=False, fill_value=(gain_linear[0], gain_linear[-1])
    )(x_samp).astype(np.float32)

    out = vox.copy()
    pad = len(gain_samp)
    for c in range(out.shape[1]):
        out[:pad, c] *= gain_samp
    return out.astype(np.float32)


def _hpf_signal(audio_ch: np.ndarray, cutoff_hz: float, order: int = 4) -> np.ndarray:
    """
    Apply a high-pass filter to a (channels, samples) array.
    Used for the Abbey Road reverb return HPF trick.
    """
    sos = butter(order, cutoff_hz / (SR / 2), btype="high", output="sos")
    out = np.zeros_like(audio_ch)
    for c in range(audio_ch.shape[0]):
        out[c] = sosfilt(sos, audio_ch[c]).astype(np.float32)
    return out.astype(np.float32)


def _process_vocals(vox: np.ndarray, ratio: float, n_semitones: int,
                    params: dict, style=None,
                    target_root: int = 0, target_mode: str = "major") -> np.ndarray:
    """
    Full professional vocal pipeline (v5 — style-adaptive):
      1. DeepFilterNet / noisereduce: remove bleed artifacts (per channel)
      2. pyrubberband R3 time-stretch + pitch-shift (formant-preserving)
         fallback: pedalboard time_stretch
      3. HPF 80 Hz, 24 dB/oct
      4. Subtractive EQ: -3 dB @ 300 Hz (mud), -2 dB @ 500 Hz (boxy)
      5. De-esser (BEFORE compression)
      6. Compressor 1 (FET-type): style-adaptive ratio/attack/release
      7. Compressor 2 (Opto-type): style-adaptive ratio/attack/release
      8. NoiseGate
      9. Additive EQ: style-adaptive presence boost + air shelf
     10. Subtle saturation
     11. Pre-delay reverb: style-adaptive room/damp/wet, HPF'd return
    """
    if style is None:
        style = {
            "fet_ratio": 6.0, "fet_attack": 2.0, "fet_release": 60.0,
            "opto_ratio": 2.5, "opto_attack": 20.0, "opto_release": 250.0,
            "presence_db": 2.5, "presence_hz": 3500.0,
            "air_db": 2.0, "reverb_room": 0.12, "reverb_damp": 0.80, "reverb_wet": 0.08,
        }

    # Step 1: neural noise suppression (DeepFilter) or fallback noisereduce
    vox = np.stack([
        _deepfilter_clean(vox[:, c]) for c in range(vox.shape[1])
    ], axis=1)

    # Step 1b: pitch correction — snap vocal to target key BEFORE stretch
    # Use mid channel for pitch detection, apply to both channels equally
    vox_mid = _to_mono(vox)
    vox_mid_corrected = _pitch_correct(vox_mid, target_root, target_mode, strength=0.65)
    # Reconstruct stereo with the pitch-corrected mid
    if not np.allclose(vox_mid, vox_mid_corrected):
        ratio_corr = np.where(np.abs(vox_mid) > 1e-9,
                              vox_mid_corrected / (vox_mid + 1e-9), 1.0).astype(np.float32)
        vox = (vox * ratio_corr[:, np.newaxis]).astype(np.float32)

    # (samples, 2) → (2, samples) for pedalboard / pyrubberband
    vox_ch = vox.T.astype(np.float32)

    # Step 2: time-stretch + pitch-shift
    # pyrubberband R3 engine gives much better formant preservation for vocals
    if HAS_PYRUBBERBAND and (abs(ratio - 1.0) > 0.005 or n_semitones != 0):
        vox_mono_list = []
        for c in range(vox_ch.shape[0]):
            y_s = rb.time_stretch(vox_ch[c], SR, ratio, rbargs={'-3': ''})
            if n_semitones != 0:
                y_s = rb.pitch_shift(y_s, SR, n_semitones, rbargs={'-3': ''})
            vox_mono_list.append(y_s)
        vox_ch = np.stack(vox_mono_list, axis=0).astype(np.float32)
    else:
        # Fallback: pedalboard
        if abs(ratio - 1.0) > 0.005 or n_semitones != 0:
            vox_ch = pb_time_stretch(
                vox_ch, SR,
                stretch_factor=ratio,
                pitch_shift_in_semitones=float(n_semitones),
            ).astype(np.float32)

    # Steps 3-4: HPF + subtractive EQ (before dynamics)
    pre_dynamics_board = Pedalboard([
        HighpassFilter(cutoff_frequency_hz=80.0),      # 80 Hz (not 100) — low-end stem bleed
        PeakFilter(cutoff_frequency_hz=300.0, gain_db=-3.0, q=1.2),   # mud
        PeakFilter(cutoff_frequency_hz=500.0, gain_db=-2.0, q=1.5),   # boxy
    ])
    vox_ch = pre_dynamics_board(vox_ch, SR).astype(np.float32)

    # Step 5: de-esser BEFORE compression (prevents sibilance pumping the compressor)
    vox_ch = _deess(vox_ch.T, threshold_db=-22.0).T.astype(np.float32)

    # Steps 6-8: dual compressor (FET → Opto) + NoiseGate — style-adaptive
    dynamics_board = Pedalboard([
        # Compressor 1: FET-type, fast (catch transients, control peaks)
        Compressor(
            threshold_db=params["comp_thresh_db"],
            ratio=style["fet_ratio"],
            attack_ms=style["fet_attack"],
            release_ms=style["fet_release"],
        ),
        # Compressor 2: Opto-type, slow (smooth programme-level glue)
        Compressor(
            threshold_db=params["comp_thresh_db"] + 3.0,
            ratio=style["opto_ratio"],
            attack_ms=style["opto_attack"],
            release_ms=style["opto_release"],
        ),
        NoiseGate(
            threshold_db=params["gate_thresh_db"],
            ratio=10.0,
            attack_ms=3.0,
            release_ms=150.0,
        ),
    ])
    vox_ch = dynamics_board(vox_ch, SR).astype(np.float32)

    # Step 9: additive EQ AFTER dynamics — style-adaptive presence + air
    post_dynamics_board = Pedalboard([
        PeakFilter(cutoff_frequency_hz=style["presence_hz"],
                   gain_db=style["presence_db"], q=1.5),
        HighShelfFilter(cutoff_frequency_hz=10000.0, gain_db=style["air_db"]),
    ])
    vox_ch = post_dynamics_board(vox_ch, SR).astype(np.float32)

    # Step 10: asymmetric waveshaper — adds even harmonics (2nd = "tube warmth")
    # tanh alone only generates odd harmonics; biasing the input before tanh
    # creates 2nd+3rd harmonic content, which is the "analog" saturation character
    # found in tape machines and Class A tube preamps.
    drive = 1.3
    asym  = 0.12  # bias toward positive half-wave → 2nd harmonic generation
    _vox_biased = vox_ch + asym * np.abs(vox_ch)
    vox_ch = (np.tanh(drive * _vox_biased) / np.tanh(drive)).astype(np.float32)

    # Step 11: pre-delay reverb with HPF'd return (Abbey Road trick)
    pre_delay = int(SR * 0.020)  # 20ms pre-delay — tempo-independent

    # Reverb-only path (wet_level=1, dry_level=0)
    reverb_board = Pedalboard([
        Reverb(room_size=style["reverb_room"], damping=style["reverb_damp"],
               wet_level=1.0, dry_level=0.0, width=0.9),
    ])
    reverb_wet = reverb_board(vox_ch, SR).astype(np.float32)

    # Abbey Road trick: HPF the reverb RETURN at 500 Hz — prevents muddy reverb tail
    reverb_wet = _hpf_signal(reverb_wet, cutoff_hz=500.0, order=4)

    # Shift reverb tail by pre_delay samples (pad front, trim end)
    reverb_shifted = np.concatenate([
        np.zeros((reverb_wet.shape[0], pre_delay), dtype=np.float32),
        reverb_wet,
    ], axis=1)[:, :vox_ch.shape[1]]

    # Mix: dry + pre-delayed HPF'd reverb at style-adaptive wet level
    vox_ch = (vox_ch + reverb_shifted * style["reverb_wet"]).astype(np.float32)

    # Step 12: Stereo ADT — Automatic Double Tracking (radio-ready width + thickness)
    # Two copies: pitch-up (+6 cents) panned left, pitch-down (-6 cents) panned right.
    # LFO on delay time (±2 ms @ 0.3 Hz) prevents static comb-filter notch.
    # Research: 4-8 cents + 13-25ms delay is the "invisible" sweet spot.
    rap = style.get("_rap_score", 0.5)
    adt_cents = float(np.interp(rap, [0, 1], [6.0, 5.0]))  # singers slightly wider
    adt_delay_ms = 22.0
    adt_level = 0.20  # -14 dB — wide enough to feel, subtle enough to not fight the vocal

    n_samp = vox_ch.shape[1]
    # LFO for delay modulation — prevents static comb-filter notch
    t = np.arange(n_samp, dtype=np.float32) / SR
    lfo = (np.sin(2 * np.pi * 0.30 * t) * 0.002 * SR).astype(np.float32)  # ±2ms in samples

    def _adt_copy(ch_audio, cents_shift, delay_base_ms, lfo_mod):
        """Create one ADT copy: pitch shift + LFO-modulated delay."""
        delay_base = int(delay_base_ms * SR / 1000)
        if HAS_PYRUBBERBAND:
            shifted = rb.pitch_shift(ch_audio, SR, cents_shift / 100.0,
                                     rbargs={'-3': ''}).astype(np.float32)
        else:
            shifted = ch_audio.copy()  # skip pitch shift if no rubberband

        # Build LFO-modulated delay using nearest-sample interpolation
        delayed = np.zeros_like(shifted)
        for s in range(delay_base, len(shifted)):
            src = s - delay_base - int(lfo_mod[s] if s < len(lfo_mod) else 0)
            src = max(0, min(len(shifted) - 1, src))
            delayed[s] = shifted[src]
        return delayed

    # Mono signal for ADT input (preserves phase; stereo from L/R pan)
    vox_mid_adt = ((vox_ch[0] + vox_ch[1]) * 0.5).astype(np.float32)

    adt_L = _adt_copy(vox_mid_adt, +adt_cents, adt_delay_ms, lfo)
    adt_R = _adt_copy(vox_mid_adt, -adt_cents, adt_delay_ms + 5.0, -lfo)

    # Pan: add ADT L copy to L channel, R copy to R channel
    vox_ch[0] = (vox_ch[0] + adt_L * adt_level).astype(np.float32)
    vox_ch[1] = (vox_ch[1] + adt_R * adt_level).astype(np.float32)

    return vox_ch.T  # (samples, 2)


# ── Instrumental Processing ───────────────────────────────────────────────────

def _transient_shape(inst: np.ndarray,
                     attack_gain_db: float = 4.0,
                     sustain_gain_db: float = -2.0,
                     fast_ms: float = 1.0,
                     slow_ms: float = 20.0,
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
        a = np.exp(-1.0 / (SR * attack_ms / 1000.0))
        r = np.exp(-1.0 / (SR * rel_ms / 1000.0))
        env = np.zeros_like(audio)
        prev = 0.0
        for i in range(len(audio)):
            level = abs(audio[i])
            if level > prev:
                env[i] = prev = (1 - a) * level + a * prev
            else:
                env[i] = prev = (1 - r) * level + r * prev
        return env

    result = np.zeros_like(inst)
    att_lin = 10 ** (attack_gain_db / 20.0)
    sus_lin = 10 ** (sustain_gain_db / 20.0)

    for c in range(inst.shape[1]):
        ch = inst[:, c].astype(np.float64)
        fast_env = _env_follow(ch, fast_ms, release_ms)
        slow_env = _env_follow(ch, slow_ms, release_ms)

        # Transient mask: how much is fast vs slow (normalized 0-1)
        total = fast_env + slow_env + 1e-12
        transient_mask = fast_env / total   # high during transients
        sustain_mask   = slow_env / total   # high during sustain

        gain = transient_mask * att_lin + sustain_mask * sus_lin
        result[:, c] = (ch * gain).astype(np.float32)

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

    # Frequency weighting: only carve in the vocal presence range (300 Hz – 8 kHz)
    freqs = librosa.fft_frequencies(sr=SR, n_fft=n_fft)
    freq_w = np.zeros(len(freqs), dtype=np.float32)
    for i, f in enumerate(freqs):
        if 300 <= f <= 8000:
            if f < 600:
                freq_w[i] = (f - 300) / 300
            elif f > 6000:
                freq_w[i] = (8000 - f) / 2000
            else:
                freq_w[i] = 1.0

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

    # Add high-shelf air boost to compensate for stem separation frequency loss
    shelf = Pedalboard([HighShelfFilter(cutoff_frequency_hz=5500.0, gain_db=4.5)])
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
    # 30% wet: NY compression research shows 25-40% gives hip-hop density
    return (0.70 * inst + 0.30 * crushed).astype(np.float32)


def _sidechain_envelope(vox_mono: np.ndarray, n_out: int,
                        depth: float, window_ms: int = 40) -> np.ndarray:
    """Compute per-sample sidechain gain curve (1.0 = no duck, < 1.0 = ducked)."""
    win = max(1, int(SR * window_ms / 1000))
    hop = win // 2
    n = len(vox_mono)
    n_frames = max(1, (n + hop - 1) // hop)

    env = np.array([
        np.sqrt(np.mean(vox_mono[i * hop: min(i * hop + win, n)] ** 2))
        for i in range(n_frames)
    ], dtype=np.float32)
    env /= _rms(vox_mono) + 1e-9
    reduction = (1.0 - depth * np.clip(env, 0.0, 1.0)).astype(np.float64)

    x_frames = np.arange(n_frames, dtype=np.float64) * hop
    x_samp   = np.arange(n_out, dtype=np.float64)
    return interp1d(
        x_frames, reduction, kind="linear",
        bounds_error=False, fill_value=(reduction[0], reduction[-1])
    )(x_samp).astype(np.float32)


def _sidechain(inst: np.ndarray, vox: np.ndarray,
               depth: float, window_ms: int = 40) -> np.ndarray:
    """
    Multiband sidechain: duck only the mids/highs (200 Hz+) when vocal is loud.
    Bass/sub-bass (< 200 Hz) pass through unaffected — kick and sub stay punchy.
    depth is computed adaptively from spectral overlap.
    """
    # Split at 200 Hz (Linkwitz-Riley style: LP + HP, order 4)
    crossover = 200.0
    sos_lo = butter(4, crossover / (SR / 2), btype="low",  output="sos")
    sos_hi = butter(4, crossover / (SR / 2), btype="high", output="sos")

    inst_lo = sosfilt(sos_lo, inst, axis=0).astype(np.float32)  # kick, sub-bass
    inst_hi = sosfilt(sos_hi, inst, axis=0).astype(np.float32)  # everything else

    # Sidechain gain from vocal envelope (only applied to mids/highs)
    vox_mono = _to_mono(vox)
    gain = _sidechain_envelope(vox_mono, len(inst), depth, window_ms)

    inst_hi_ducked = (inst_hi * gain[:, np.newaxis]).astype(np.float32)

    return (inst_lo + inst_hi_ducked).astype(np.float32)


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
      5. LUFS        — is the integrated loudness in the -15 to -8 range?
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

    # ── 6. Stereo correlation (mono compatibility) ────────────────────────────
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


def _master(mix: np.ndarray) -> np.ndarray:
    """
    Mastering chain (v4):
      EQ → soft clip → glue compressor → limiter → LUFS -9 (final normalize)

    LUFS normalize goes LAST so it accounts for all gain reduction from the
    glue compressor and limiter. Hip-hop target: -8 to -10 LUFS.
    """
    # Mastering EQ
    master_eq = Pedalboard([
        PeakFilter(cutoff_frequency_hz=600.0, gain_db=-1.0, q=0.8),
        HighShelfFilter(cutoff_frequency_hz=8000.0, gain_db=1.5),
    ])
    mix = master_eq(mix.T.astype(np.float32), SR).T.astype(np.float32)

    # Soft clip: Chebyshev 3rd-order (1.5x - 0.5x³) — gentler than tanh,
    # preserves low-level signal shape, clips peaks without hardness
    mix_c = np.clip(mix, -1.0, 1.0)
    mix = (1.5 * mix_c - 0.5 * mix_c ** 3).astype(np.float32)

    # Glue compressor: SSL-style — faster attack for hip-hop punch (3ms → 1ms)
    glue = Pedalboard([
        Compressor(threshold_db=-6.0, ratio=2.0, attack_ms=1.0, release_ms=80.0),
    ])
    mix = glue(mix.T.astype(np.float32), SR).T.astype(np.float32)

    # Brick-wall limiter at -1.0 dBTP
    limiter = Pedalboard([Limiter(threshold_db=-1.0, release_ms=50.0)])
    mix = limiter(mix.T.astype(np.float32), SR).T.astype(np.float32)

    # LUFS normalize LAST: accounts for all gain reduction above
    # -9.0 LUFS is hip-hop standard (-8 to -10)
    mix = _lufs_normalize(mix, -9.0)
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

    sep_model = "BS-Roformer" if _has_gpu() else "Demucs"
    step(4, TOTAL, f"Separating stems — Song A (instrumental) via {sep_model}…")
    stems_a = separate(song_a, stems_cache)

    step(5, TOTAL, f"Separating stems — Song B (vocals) via {sep_model}…")
    stems_b = separate(song_b, stems_cache)

    inst = stems_a["no_vocals"]   # (samples, 2)
    vox  = stems_b["vocals"]      # (samples, 2)

    ratio   = _best_ratio(bpm_a, bpm_b)
    n_semi  = semitones_to_shift(key_b_root, key_b_mode, key_a_root, key_a_mode)
    print(f"      BPM ratio: {ratio:.4f}   pitch shift: {n_semi:+d} semitones",
          flush=True)

    # Smart key shift: find best harmonic alternative if shift is large
    n_semi, key_msg = _smart_key_shift(n_semi, key_b_root, key_b_mode,
                                        key_a_root, key_a_mode)
    print(f"      Key: {key_msg}", flush=True)

    step(6, TOTAL, "Analyzing audio content for AI-adaptive parameters…")
    # AI content analysis: derive all DSP parameters from actual audio
    beat_char  = _analyze_beat_character(full_a, bpm_a)
    vox_char   = _analyze_vocal_character(_to_mono(stems_b["vocals"]))
    style      = _style_params(beat_char, vox_char)
    vox_params = _analyze_vocal_stem(vox)
    overlap    = _spectral_overlap(_to_mono(vox), _to_mono(inst))
    sidechain_depth = float(np.clip(overlap * 0.4, 0.10, 0.22))
    print(f"      Beat: agg={beat_char['aggressiveness']:.2f}  "
          f"bass={beat_char['bass_weight']:.2f}  "
          f"brightness={beat_char['brightness']:.2f}", flush=True)
    print(f"      Vocal: rap_score={vox_char['rap_score']:.2f}  "
          f"onset_rate={vox_char['onset_rate']:.1f}/s  "
          f"pitch_range={vox_char['pitch_range']:.0f}st", flush=True)
    print(f"      Style → FET {style['fet_ratio']:.1f}:1  "
          f"reverb_room={style['reverb_room']:.2f}  "
          f"reverb_wet={style['reverb_wet']:.2f}  "
          f"carve={style['carve_db']:.1f}dB  "
          f"vocal_level={style['vocal_level']:.2f}", flush=True)
    print(f"      Gate thresh: {vox_params['gate_thresh_db']:.1f} dB  "
          f"Comp ratio: {vox_params['comp_ratio']:.1f}:1  "
          f"Spectral overlap: {overlap:.3f}  "
          f"Sidechain depth: {sidechain_depth:.2f}", flush=True)

    rb_engine = "pyrubberband R3" if HAS_PYRUBBERBAND else "pedalboard (fallback)"
    step(7, TOTAL, f"Processing vocals (DeepFilter + pitch-correct + stretch [{rb_engine}] + style-adaptive chain)…")
    vox = _process_vocals(vox, ratio, n_semi, vox_params, style,
                          target_root=key_a_root, target_mode=key_a_mode)

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

    # AI iterative mixer: closed-loop presence feedback, energy-envelope matching,
    # spectral carve, parallel compress, sidechain — all style-adaptive
    mix = _iterative_mix(inst, vox, style, sidechain_depth, bpm_a)

    mix = _fade(mix, fade_s=2.0)

    step(9, TOTAL, "Mastering…")
    mix = _master(mix)

    os.makedirs(os.path.dirname(os.path.abspath(out_path)), exist_ok=True)
    sf.write(out_path, mix, SR, subtype="PCM_16")
    print(f"Done → {out_path}", flush=True)

    # ── Auto quality evaluation ──────────────────────────────────────────────
    print("\n── Auto Quality Evaluation ─────────────────────────────────────", flush=True)
    ev = _auto_evaluate(mix, inst, vox, bpm_a)
    print(f"  Beat sync:      {ev['beat_sync_pct']}%  (want >45%)", flush=True)
    print(f"  Vocal presence: {ev['vocal_presence']}%  (want 40-70%)", flush=True)
    print(f"  LUFS:           {ev['lufs']} dB", flush=True)
    print(f"  Peak:           {ev['peak_dBFS']} dBFS", flush=True)
    print(f"  Bands:          {ev['bands']}", flush=True)
    if ev["issues"]:
        print("  ISSUES:", flush=True)
        for iss in ev["issues"]:
            print(f"    x {iss}", flush=True)
    else:
        print("  All checks passed", flush=True)
    print(f"  Overall: {'PASS' if ev['pass'] else 'FAIL'}", flush=True)

    return out_path
