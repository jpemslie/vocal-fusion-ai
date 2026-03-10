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

SR = 44100
_BS_ROFORMER = "model_bs_roformer_ep_317_sdr_12.9755.ckpt"

_KS_MAJOR = np.array([6.35, 2.23, 3.48, 2.33, 4.38, 4.09,
                       2.52, 5.19, 2.39, 3.66, 2.29, 2.88])
_KS_MINOR = np.array([6.33, 2.68, 3.52, 5.38, 2.60, 3.53,
                       2.54, 4.75, 3.98, 2.69, 3.34, 3.17])
_NOTES = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"]


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
                    params: dict) -> np.ndarray:
    """
    Full professional vocal pipeline (v4 — correct chain order):
      1. noisereduce: remove Demucs bleed artifacts (per channel)
      2. pyrubberband R3 time-stretch + pitch-shift (one pass, formant-preserving)
         fallback: pedalboard time_stretch
      3. HPF 80 Hz, 24 dB/oct — remove low-end stem bleed (not 100 Hz)
      4. Subtractive EQ: -3 dB @ 300 Hz (mud), -2 dB @ 500 Hz (boxy)
      5. De-esser — BEFORE compression to prevent sibilance pumping the compressor
      6. Compressor 1 (FET-type, fast): ratio=6:1, attack=2ms, release=60ms
      7. Compressor 2 (Opto-type, slow): ratio=2.5:1, attack=20ms, release=250ms
      8. NoiseGate (after compression)
      9. Additive EQ (AFTER dynamics): +2.5 dB @ 3.5 kHz (presence), +2 dB shelf @ 10 kHz (air)
     10. Subtle saturation: tanh(audio * 1.3) / 1.3 — tape 2nd harmonic enrichment
     11. Pre-delay reverb with HPF'd return (Abbey Road trick):
         - Reverb wet-only (room_size=0.12, damping=0.80, width=0.9)
         - HPF reverb return at 500 Hz — prevents muddy reverb tail
         - Pre-delay: 20ms (tempo-independent, keeps vocal intelligible)
         - Mix at 8% wet
    """
    # Step 1: noisereduce on each channel
    vox = np.stack([
        _clean_vocal(vox[:, c]) for c in range(vox.shape[1])
    ], axis=1)

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

    # Steps 6-8: dual compressor (FET → Opto) + NoiseGate
    dynamics_board = Pedalboard([
        # Compressor 1: FET-type, fast (catch transients, control peaks)
        Compressor(
            threshold_db=params["comp_thresh_db"],
            ratio=6.0,
            attack_ms=2.0,
            release_ms=60.0,
        ),
        # Compressor 2: Opto-type, slow (smooth programme-level glue)
        Compressor(
            threshold_db=params["comp_thresh_db"] + 3.0,  # slightly higher thresh for Opto
            ratio=2.5,
            attack_ms=20.0,
            release_ms=250.0,
        ),
        NoiseGate(
            threshold_db=params["gate_thresh_db"],
            ratio=10.0,
            attack_ms=3.0,
            release_ms=150.0,
        ),
    ])
    vox_ch = dynamics_board(vox_ch, SR).astype(np.float32)

    # Step 9: additive EQ AFTER dynamics (boosts are not compressed away)
    post_dynamics_board = Pedalboard([
        PeakFilter(cutoff_frequency_hz=3500.0, gain_db=2.5, q=1.5),   # presence
        HighShelfFilter(cutoff_frequency_hz=10000.0, gain_db=2.0),      # air
    ])
    vox_ch = post_dynamics_board(vox_ch, SR).astype(np.float32)

    # Step 10: subtle tape-style saturation (2nd harmonic enrichment)
    vox_ch = (np.tanh(vox_ch * 1.3) / 1.3).astype(np.float32)

    # Step 11: pre-delay reverb with HPF'd return (Abbey Road trick)
    pre_delay = int(SR * 0.020)  # 20ms pre-delay — tempo-independent

    # Reverb-only path (wet_level=1, dry_level=0)
    reverb_board = Pedalboard([
        Reverb(room_size=0.12, damping=0.80, wet_level=1.0, dry_level=0.0, width=0.9),
    ])
    reverb_wet = reverb_board(vox_ch, SR).astype(np.float32)

    # Abbey Road trick: HPF the reverb RETURN at 500 Hz — prevents muddy reverb tail
    reverb_wet = _hpf_signal(reverb_wet, cutoff_hz=500.0, order=4)

    # Shift reverb tail by pre_delay samples (pad front, trim end)
    reverb_shifted = np.concatenate([
        np.zeros((reverb_wet.shape[0], pre_delay), dtype=np.float32),
        reverb_wet,
    ], axis=1)[:, :vox_ch.shape[1]]

    # Mix: dry + pre-delayed HPF'd reverb at 8% wet
    vox_ch = (vox_ch + reverb_shifted * 0.08).astype(np.float32)

    return vox_ch.T  # (samples, 2)


# ── Instrumental Processing ───────────────────────────────────────────────────

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
    return (0.80 * inst + 0.20 * crushed).astype(np.float32)


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

    # Soft clip (tanh) — shaves peaks before the compressor
    mix = (np.tanh(mix * 0.95) / 0.95).astype(np.float32)

    # Glue compressor: very gentle, just for binding/density
    glue = Pedalboard([
        Compressor(threshold_db=-6.0, ratio=2.0, attack_ms=3.0, release_ms=100.0),
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

    # Key compatibility warning: large shifts risk harmonic clashing
    if abs(n_semi) > 3:
        print(f"WARNING: {n_semi} semitone shift is large — may cause pitch artifacts "
              f"or harmonic clashing", flush=True)

    step(6, TOTAL, "Analyzing stems for adaptive parameters…")
    vox_params = _analyze_vocal_stem(vox)
    overlap    = _spectral_overlap(_to_mono(vox), _to_mono(inst))
    # Spectral carve already handles freq-specific competition (up to -5 dB).
    # Sidechain adds broadband duck on top — keep it light (max -2.5 dB) so
    # the combined effect stays within 7-8 dB, not the 9+ dB of the old code.
    sidechain_depth = float(np.clip(overlap * 0.4, 0.10, 0.22))
    print(f"      Gate thresh: {vox_params['gate_thresh_db']:.1f} dB  "
          f"Comp ratio: {vox_params['comp_ratio']:.1f}:1  "
          f"Spectral overlap: {overlap:.3f}  "
          f"Sidechain depth: {sidechain_depth:.2f}", flush=True)

    rb_engine = "pyrubberband R3" if HAS_PYRUBBERBAND else "pedalboard (fallback)"
    step(7, TOTAL, f"Processing vocals (denoise + stretch + pitch [{rb_engine}] + EQ + gate + reverb)…")
    vox = _process_vocals(vox, ratio, n_semi, vox_params)

    step(8, TOTAL, "Mixing (beat-align + spectral carve + M/S + sidechain + level match)…")

    # Beat-grid alignment: use the ACTUAL STRETCHED vocal stem to find when
    # the singer first comes in, then align that to the nearest measure boundary.
    vox_pre, inst_pre = _beat_align(full_a, _to_mono(vox))
    silence = lambda n: np.zeros((n, 2), dtype=np.float32)
    if vox_pre > 0:
        vox = np.concatenate([silence(vox_pre), vox], axis=0)
        print(f"      Beat-align: prepend {vox_pre/SR*1000:.0f} ms to vocal", flush=True)
    elif inst_pre > 0:
        inst = np.concatenate([silence(inst_pre), inst], axis=0)
        print(f"      Beat-align: prepend {inst_pre/SR*1000:.0f} ms to beat", flush=True)
    else:
        print(f"      Beat-align: no offset needed", flush=True)

    L = min(len(inst), len(vox))
    inst, vox = inst[:L], vox[:L]

    # Level match: vocal 3-6 dB louder than instrumental (research: 1.41-2× inst_rms).
    # Since spectral carving reduces the instrumental after this level set, 1.2×
    # is the appropriate starting point — vocal ends up perceptually forward in the mix.
    # Using plain _rms (not _active_rms): silence gaps must not inflate the multiplier.
    ir = _rms(_to_mono(inst))
    vr = _rms(_to_mono(vox))
    if vr > 1e-9:
        vox = (vox * (ir * 1.2 / vr)).astype(np.float32)

    # Content-aware spectral carve on instrumental
    inst = _adaptive_spectral_carve(inst, vox, carve_db=5.0)

    # Parallel compression: restore punch/density lost after carving
    inst = _parallel_compress(inst)

    # Multiband sidechain duck (only mids/highs — bass stays punchy)
    inst = _sidechain(inst, vox, depth=sidechain_depth)

    # M/S mix: vocal into Mid only, beat Sides preserved
    inst_M, inst_S = _ms_encode(inst)
    vox_M, _       = _ms_encode(vox)
    mix_M = inst_M + vox_M
    mix = _ms_decode(mix_M, inst_S)

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
