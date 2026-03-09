"""
VocalFusion Fusion Engine
=========================
Song A  →  instrumental (no_vocals stem from Demucs)
Song B  →  vocals stem
Output  →  stereo mixed, mastered WAV

Why the previous version sounded bad and what changed
──────────────────────────────────────────────────────
BEFORE: two separate pyrubberband passes (time_stretch THEN pitch_shift)
        → double phase-vocoder artifacts, audible smearing on every vocal
NOW:    pedalboard.time_stretch does both in ONE JUCE pass, no intermediate

BEFORE: no noise gate → Demucs bleed during silences makes it sound like
        two full songs playing simultaneously
NOW:    NoiseGate (-42 dB) cuts bleed between vocal phrases

BEFORE: sidechain depth = 0.20 (~-1 dB duck) → instrumental stays loud,
        vocal gets buried / sounds like two songs at once
NOW:    depth = 0.40 (~-4.5 dB duck) → vocal clearly dominant when singing

BEFORE: no reverb → vocal sounds dry, like a different acoustic space
NOW:    short plate reverb (wet=8%) ties vocal into the instrumental

BEFORE: no low-mid cut → Demucs vocal stem sounds boxy/muddy
NOW:    -3 dB peak cut at 320 Hz removes the Demucs "box" resonance

BEFORE: BPM via beat_track (prone to half/double-time errors)
NOW:    onset-strength tempo + normalise to 60-180 BPM range
"""

import hashlib
import os
import shutil
import subprocess
import sys
from pathlib import Path

import librosa
import numpy as np
import pyloudnorm as pyln
import soundfile as sf
from pedalboard import (
    Compressor, HighpassFilter, NoiseGate,
    PeakFilter, Pedalboard, Reverb, Limiter,
    time_stretch as pb_time_stretch,
)
from scipy.interpolate import interp1d
from scipy.signal import butter, sosfilt

SR = 44100

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
    """RMS of non-silent samples only — avoids Demucs silent gaps distorting level."""
    mono = _to_mono(y)
    cutoff = float(np.max(np.abs(mono)) + 1e-12) * 10 ** (threshold_db / 20)
    active = mono[np.abs(mono) > cutoff]
    return float(np.sqrt(np.mean(active ** 2) + 1e-12)) if len(active) >= SR else _rms(y)


# ── Stem Separation ───────────────────────────────────────────────────────────

def separate(audio_path: str, cache_dir: str = "vf_data/stems") -> dict:
    """
    Run Demucs htdemucs_ft --two-stems vocals.
    Cached by file fingerprint. Returns stereo (samples, 2) float32 arrays.
    """
    os.makedirs(cache_dir, exist_ok=True)
    fid = _file_id(audio_path)
    cached = Path(cache_dir) / fid

    if not (cached / "vocals.wav").exists():
        ext = Path(audio_path).suffix or ".mp3"
        tmp = Path(cache_dir) / f"{fid}{ext}"
        shutil.copy2(audio_path, tmp)
        try:
            result = subprocess.run(
                [sys.executable, "-m", "demucs",
                 "--two-stems", "vocals",
                 "-n", "htdemucs_ft",
                 "-o", cache_dir,
                 str(tmp)],
                capture_output=True, text=True,
            )
            if result.returncode != 0:
                raise RuntimeError(
                    f"Demucs failed (exit {result.returncode}):\n{result.stderr}"
                )
            raw = Path(cache_dir) / "htdemucs_ft" / fid
            raw.rename(cached)
            try:
                (Path(cache_dir) / "htdemucs_ft").rmdir()
            except OSError:
                pass
        finally:
            if tmp.exists():
                tmp.unlink()

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


# ── Analysis ──────────────────────────────────────────────────────────────────

def detect_bpm(y_mono: np.ndarray) -> float:
    """
    Onset-strength based tempo — more accurate than beat_track for hip-hop/trap.
    Normalised to 60-180 BPM to correct librosa half/double-time errors.
    """
    onset_env = librosa.onset.onset_strength(y=y_mono, sr=SR, hop_length=512)
    tempo = float(librosa.feature.rhythm.tempo(onset_envelope=onset_env, sr=SR, hop_length=512)[0])
    while tempo > 180.0:
        tempo /= 2
    while tempo < 60.0:
        tempo *= 2
    return tempo


def _best_ratio(bpm_a: float, bpm_b: float) -> float:
    """Capped time-stretch ratio. rate>1 = faster (higher BPM)."""
    return float(np.clip(bpm_a / bpm_b, 0.667, 1.5))


def detect_key(y_mono: np.ndarray) -> tuple:
    """Krumhansl-Schmuckler on chroma_cens. Returns (root_semitone, mode)."""
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


# ── Vocal Processing ──────────────────────────────────────────────────────────

def _process_vocals(vox: np.ndarray, ratio: float, n_semitones: int) -> np.ndarray:
    """
    Full vocal pipeline in (channels, samples) space:
      1. time_stretch + pitch_shift in ONE JUCE pass (no double-artefact)
      2. HPF 100 Hz — remove rumble / low instrument bleed
      3. Peak cut -3 dB @ 320 Hz — remove Demucs 'boxy' resonance
      4. Peak boost +2.5 dB @ 3.5 kHz — presence / intelligibility
      5. Compressor 3.5:1 — even out phrase dynamics
      6. NoiseGate -42 dB — silence Demucs bleed between vocal phrases
      7. Plate reverb 8% wet — tie vocal into the same acoustic space
    """
    # (samples, 2) → (2, samples) for pedalboard
    vox_ch = vox.T.astype(np.float32)

    # Step 1: combined time-stretch + pitch-shift in one JUCE pass
    if abs(ratio - 1.0) > 0.005 or n_semitones != 0:
        vox_ch = pb_time_stretch(
            vox_ch, SR,
            stretch_factor=ratio,
            pitch_shift_in_semitones=float(n_semitones),
        ).astype(np.float32)

    # Steps 2-7: EQ + dynamics + space
    board = Pedalboard([
        HighpassFilter(cutoff_frequency_hz=100.0),
        PeakFilter(cutoff_frequency_hz=320.0,  gain_db=-3.0, q=1.0),
        PeakFilter(cutoff_frequency_hz=3500.0, gain_db=2.5,  q=1.5),
        Compressor(threshold_db=-18.0, ratio=3.5,
                   attack_ms=5.0, release_ms=100.0),
        NoiseGate(threshold_db=-42.0, ratio=8.0,
                  attack_ms=5.0, release_ms=200.0),
        Reverb(room_size=0.12, damping=0.8,
               wet_level=0.08, dry_level=1.0),
    ])
    vox_ch = board(vox_ch, SR).astype(np.float32)

    # (2, samples) → (samples, 2)
    return vox_ch.T


# ── Instrumental Processing ───────────────────────────────────────────────────

def _carve_pocket(inst: np.ndarray) -> np.ndarray:
    """
    Gently reduce 1–4 kHz in the instrumental to open a pocket for vocals.
    Bandpass-subtract method: -2 dB in the vocal presence range.
    """
    sos = butter(4, [1000.0 / (SR / 2), 4000.0 / (SR / 2)],
                 btype="bandpass", output="sos")
    band = sosfilt(sos, inst, axis=0)
    return (inst - 0.21 * band).astype(np.float32)


def _sidechain(inst: np.ndarray, vox: np.ndarray,
               depth: float = 0.40, window_ms: int = 40) -> np.ndarray:
    """
    Duck instrumental by `depth` when vocals are loud.
    depth=0.40 ≈ -4.5 dB reduction — enough to let vocal cut through cleanly.
    Block-based envelope from vocal mono mix, interpolated to sample level.
    """
    vox_mono = _to_mono(vox)
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
    x_samp   = np.arange(len(inst), dtype=np.float64)
    gain = interp1d(
        x_frames, reduction, kind="linear",
        bounds_error=False, fill_value=(reduction[0], reduction[-1])
    )(x_samp).astype(np.float32)

    return inst * gain[:, np.newaxis]


# ── Mastering ────────────────────────────────────────────────────────────────

def _lufs_normalize(y: np.ndarray, target: float = -14.0) -> np.ndarray:
    meter = pyln.Meter(SR)
    lufs = meter.integrated_loudness(y)
    if not np.isfinite(lufs) or lufs < -70.0:
        return y
    return (y * 10 ** ((target - lufs) / 20)).astype(np.float32)


def _master(mix: np.ndarray) -> np.ndarray:
    """Soft clip → LUFS -14 → true-peak -1 dBTP."""
    mix = (np.tanh(mix * 0.95) / 0.95).astype(np.float32)
    mix = _lufs_normalize(mix, -14.0)
    peak = float(np.max(np.abs(mix)))
    limit = 10 ** (-1.0 / 20)
    if peak > limit:
        mix = (mix * limit / peak).astype(np.float32)
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
    def step(n, msg):
        print(f"[{n}/8] {msg}", flush=True)
        if progress_cb:
            progress_cb(n, 8, msg)

    step(1, "Loading audio for analysis…")
    full_a = librosa.load(song_a, sr=SR, mono=True)[0].astype(np.float32)
    full_b = librosa.load(song_b, sr=SR, mono=True)[0].astype(np.float32)

    step(2, "Detecting BPM…")
    bpm_a = detect_bpm(full_a)
    bpm_b = detect_bpm(full_b)
    print(f"      A: {bpm_a:.1f} BPM   B: {bpm_b:.1f} BPM", flush=True)

    step(3, "Detecting keys…")
    key_a_root, key_a_mode = detect_key(full_a)
    key_b_root, key_b_mode = detect_key(full_b)
    print(f"      A: {_NOTES[key_a_root]} {key_a_mode}   "
          f"B: {_NOTES[key_b_root]} {key_b_mode}", flush=True)

    step(4, "Separating stems — Song A (instrumental)…")
    stems_a = separate(song_a, stems_cache)

    step(5, "Separating stems — Song B (vocals)…")
    stems_b = separate(song_b, stems_cache)

    inst = stems_a["no_vocals"]   # (samples, 2)
    vox  = stems_b["vocals"]      # (samples, 2)

    ratio   = _best_ratio(bpm_a, bpm_b)
    n_semi  = semitones_to_shift(key_b_root, key_b_mode, key_a_root, key_a_mode)
    print(f"      BPM ratio: {ratio:.4f}   pitch shift: {n_semi:+d} semitones",
          flush=True)

    step(6, "Processing vocals (stretch + pitch + EQ + gate + reverb)…")
    vox = _process_vocals(vox, ratio, n_semi)

    step(7, "Mixing…")
    L = min(len(inst), len(vox))
    inst, vox = inst[:L], vox[:L]

    # Level match on active (non-silent) regions
    ir = _active_rms(inst)
    vr = _active_rms(vox)
    if vr > 1e-9:
        vox = (vox * (ir * 0.80 / vr)).astype(np.float32)

    inst = _carve_pocket(inst)
    inst = _sidechain(inst, vox, depth=0.40)

    mix = (inst + vox).astype(np.float32)
    mix = _fade(mix, fade_s=2.0)

    step(8, "Mastering…")
    mix = _master(mix)

    os.makedirs(os.path.dirname(os.path.abspath(out_path)), exist_ok=True)
    sf.write(out_path, mix, SR, subtype="PCM_16")
    print(f"Done → {out_path}", flush=True)
    return out_path
