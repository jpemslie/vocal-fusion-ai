"""
VocalFusion Fusion Engine v3
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
"""

import hashlib
import logging
import os
import tempfile
import shutil
from pathlib import Path

import librosa
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
    tempo = float(librosa.beat.tempo(
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


def _beat_align(inst_mono: np.ndarray, vox_mono: np.ndarray,
                vox_stretch_ratio: float) -> tuple:
    """
    Compute the sample offset that aligns the vocal's first beat to the
    nearest beat-grid position in the instrumental.

    After time-stretching the vocal by `vox_stretch_ratio`, its beat positions
    are scaled by 1/ratio. We find how many samples to prepend (silence) to
    the vocal so its first downbeat lines up with a beat in the instrumental.

    Returns (vox_prepend_samples, inst_prepend_samples).
    One of them will always be 0.
    """
    try:
        _, beats_inst = librosa.beat.beat_track(y=inst_mono, sr=SR, units="samples")
        _, beats_vox  = librosa.beat.beat_track(y=vox_mono,  sr=SR, units="samples")
    except Exception:
        return 0, 0

    if len(beats_inst) < 4 or len(beats_vox) < 2:
        return 0, 0

    # After stretching the vocal, its beat positions compress by ratio
    # (stretch_factor > 1 = faster = shorter → beat[i]/ratio)
    vox_first_beat_s = int(beats_vox[0] / vox_stretch_ratio)

    # Find the beat in the instrumental that is closest to vox_first_beat_s
    nearest_idx = int(np.argmin(np.abs(beats_inst - vox_first_beat_s)))

    # Snap to a measure boundary (nearest multiple of 4 beats) for musical feel
    measure_idx = round(nearest_idx / 4) * 4
    measure_idx = min(measure_idx, len(beats_inst) - 1)
    target_inst_beat = int(beats_inst[measure_idx])

    offset = target_inst_beat - vox_first_beat_s  # positive = vocal starts too early

    if offset >= 0:
        # Vocal needs to start later → prepend silence to vocal
        return int(offset), 0
    else:
        # Instrumental needs to start later → prepend silence to inst
        return 0, int(-offset)


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
        prop_decrease=0.75,
        n_fft=2048,
    ).astype(np.float32)


def _deess(vox: np.ndarray, threshold_db: float = -22.0,
           cutoff_hz: float = 6500.0, max_reduction_db: float = 7.0) -> np.ndarray:
    """
    Dynamic de-esser: detect sibilance energy (6.5 kHz+) per frame and apply
    proportional gain reduction when it exceeds threshold_db.
    Placed before reverb so the tail doesn't get de-essed.
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


def _process_vocals(vox: np.ndarray, ratio: float, n_semitones: int,
                    params: dict) -> np.ndarray:
    """
    Full vocal pipeline:
      1. noisereduce: remove Demucs bleed artifacts (per channel)
      2. time_stretch + pitch_shift in ONE JUCE pass (no double-artefact)
      3. HPF 100 Hz — remove rumble / low instrument bleed
      4. Low-shelf cut -2 dB @ 200 Hz — reduce mud
      5. Peak cut -3 dB @ 320 Hz — remove Demucs 'boxy' resonance
      6. Peak boost +3 dB @ 3.5 kHz — presence / intelligibility
      7. Adaptive Compressor — ratio/threshold from measured dynamic range
      8. Adaptive NoiseGate — threshold from measured noise floor
      9. De-esser — dynamic gain reduction above 6.5 kHz (before reverb)
     10. Pre-delay reverb — 18 ms pre-delay separates voice from tail,
         gives the vocal presence and depth rather than washing it out
    """
    # Step 1: noisereduce on each channel
    vox = np.stack([
        _clean_vocal(vox[:, c]) for c in range(vox.shape[1])
    ], axis=1)

    # (samples, 2) → (2, samples) for pedalboard
    vox_ch = vox.T.astype(np.float32)

    # Step 2: combined time-stretch + pitch-shift in one JUCE pass
    if abs(ratio - 1.0) > 0.005 or n_semitones != 0:
        vox_ch = pb_time_stretch(
            vox_ch, SR,
            stretch_factor=ratio,
            pitch_shift_in_semitones=float(n_semitones),
        ).astype(np.float32)

    # Steps 3-8: EQ + dynamics (dry — no reverb yet)
    dry_board = Pedalboard([
        HighpassFilter(cutoff_frequency_hz=100.0),
        LowShelfFilter(cutoff_frequency_hz=200.0, gain_db=-2.0, q=0.7),
        PeakFilter(cutoff_frequency_hz=320.0,  gain_db=-3.0, q=1.0),
        PeakFilter(cutoff_frequency_hz=3500.0, gain_db=3.0,  q=1.5),
        Compressor(
            threshold_db=params["comp_thresh_db"],
            ratio=params["comp_ratio"],
            attack_ms=5.0,
            release_ms=100.0,
        ),
        NoiseGate(
            threshold_db=params["gate_thresh_db"],
            ratio=10.0,
            attack_ms=3.0,
            release_ms=150.0,
        ),
    ])
    vox_dry = dry_board(vox_ch, SR).astype(np.float32)  # (2, samples)

    # Step 9: de-esser on dry signal (before reverb — don't de-ess the tail)
    vox_dry = _deess(vox_dry.T, threshold_db=-22.0).T.astype(np.float32)

    # Step 10: pre-delay reverb (18 ms gap before reverb tail starts)
    pre_delay = int(SR * 0.018)  # 661 samples @ 44100

    # Reverb-only path (wet_level=1, dry_level=0)
    reverb_board = Pedalboard([
        Reverb(room_size=0.15, damping=0.75, wet_level=1.0, dry_level=0.0, width=0.8),
    ])
    reverb_wet = reverb_board(vox_dry, SR).astype(np.float32)

    # Shift reverb tail by pre_delay samples (pad front, trim end)
    reverb_shifted = np.concatenate([
        np.zeros((reverb_wet.shape[0], pre_delay), dtype=np.float32),
        reverb_wet,
    ], axis=1)[:, :vox_dry.shape[1]]

    # Mix: dry + pre-delayed reverb at 8% wet
    vox_ch = (vox_dry + reverb_shifted * 0.08).astype(np.float32)

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


# ── Mastering ────────────────────────────────────────────────────────────────

def _lufs_normalize(y: np.ndarray, target: float = -11.0) -> np.ndarray:
    meter = pyln.Meter(SR)
    lufs = meter.integrated_loudness(y)
    if not np.isfinite(lufs) or lufs < -70.0:
        return y
    return (y * 10 ** ((target - lufs) / 20)).astype(np.float32)


def _master(mix: np.ndarray) -> np.ndarray:
    """Soft clip → LUFS -11 → brick-wall Limiter -1 dBTP."""
    mix = (np.tanh(mix * 0.95) / 0.95).astype(np.float32)
    mix = _lufs_normalize(mix, -11.0)
    limiter = Pedalboard([Limiter(threshold_db=-1.0, release_ms=50.0)])
    mix = limiter(mix.T.astype(np.float32), SR).T.astype(np.float32)
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

    step(6, TOTAL, "Analyzing stems for adaptive parameters…")
    vox_params = _analyze_vocal_stem(vox)
    overlap    = _spectral_overlap(_to_mono(vox), _to_mono(inst))
    # Map overlap (typically 0.3–0.7) to sidechain depth (0.15–0.40)
    sidechain_depth = float(np.clip(overlap * 0.9, 0.15, 0.40))
    print(f"      Gate thresh: {vox_params['gate_thresh_db']:.1f} dB  "
          f"Comp ratio: {vox_params['comp_ratio']:.1f}:1  "
          f"Spectral overlap: {overlap:.3f}  "
          f"Sidechain depth: {sidechain_depth:.2f}", flush=True)

    step(7, TOTAL, "Processing vocals (denoise + stretch + pitch + EQ + gate + reverb)…")
    vox = _process_vocals(vox, ratio, n_semi, vox_params)

    step(8, TOTAL, "Mixing (beat-align + spectral carve + M/S + sidechain + level match)…")

    # Beat-grid alignment: snap vocal's first beat to a measure boundary in the inst
    vox_pre, inst_pre = _beat_align(_to_mono(inst), _to_mono(stems_b["vocals"]),
                                     ratio)
    silence = lambda n: np.zeros((n, 2), dtype=np.float32)
    if vox_pre > 0:
        vox = np.concatenate([silence(vox_pre), vox], axis=0)
        print(f"      Beat-align: prepend {vox_pre/SR*1000:.0f} ms to vocal", flush=True)
    elif inst_pre > 0:
        inst = np.concatenate([silence(inst_pre), inst], axis=0)
        print(f"      Beat-align: prepend {inst_pre/SR*1000:.0f} ms to beat", flush=True)

    L = min(len(inst), len(vox))
    inst, vox = inst[:L], vox[:L]

    # Level match: vocal at 110% of instrumental active RMS
    ir = _active_rms(inst)
    vr = _active_rms(vox)
    if vr > 1e-9:
        vox = (vox * (ir * 1.10 / vr)).astype(np.float32)

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
    return out_path
