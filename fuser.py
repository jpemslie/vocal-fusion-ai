"""
VocalFusion Fusion Engine
=========================
Song A  →  instrumental (no_vocals stem from Demucs)
Song B  →  vocals stem
Output  →  mixed, mastered WAV

Quality decisions
─────────────────
• Key detection on the FULL song (not the artifact-prone separated stem)
• BPM ratio tries ×0.5 and ×2 variants to minimise stretching distance
• Pedalboard (JUCE) compressor/highpass — zero-copy, no Python loops
• EQ carving: gentle -2 dB notch at 2.5 kHz in instrumental
• Sidechain duck: block-based envelope (no sample-level Python loops)
• LUFS-normalised master at -14 LUFS / -1 dBTP true peak
"""

import hashlib
import os
import shutil
import subprocess
from pathlib import Path

import librosa
import numpy as np
import pyloudnorm as pyln
import pyrubberband as pyrb
import soundfile as sf
from pedalboard import Compressor, HighpassFilter, Pedalboard
from scipy.interpolate import interp1d
from scipy.signal import butter, sosfilt

SR = 44100

# Krumhansl-Schmuckler tonal hierarchy profiles
_KS_MAJOR = np.array([6.35, 2.23, 3.48, 2.33, 4.38, 4.09,
                       2.52, 5.19, 2.39, 3.66, 2.29, 2.88])
_KS_MINOR = np.array([6.33, 2.68, 3.52, 5.38, 2.60, 3.53,
                       2.54, 4.75, 3.98, 2.69, 3.34, 3.17])
_NOTES = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"]


# ── Helpers ───────────────────────────────────────────────────────────────────

def _log(msg: str) -> None:
    print(msg, flush=True)


def _file_id(path: str) -> str:
    """Stable ID from file size + first 8 KB — avoids reading the whole file."""
    stat = os.stat(path)
    with open(path, "rb") as f:
        head = f.read(8192)
    return hashlib.md5(str(stat.st_size).encode() + head).hexdigest()[:12]


def _load_mono(path: str) -> np.ndarray:
    """Load any audio file as mono float32 at SR."""
    y, sr = librosa.load(path, sr=SR, mono=True)
    return y.astype(np.float32)


def _rms(y: np.ndarray) -> float:
    return float(np.sqrt(np.mean(y ** 2) + 1e-12))


# ── Stem Separation ───────────────────────────────────────────────────────────

def separate(audio_path: str, cache_dir: str = "vf_data/stems") -> dict:
    """
    Run Demucs htdemucs_ft --two-stems vocals.
    Results cached by file fingerprint so each song is separated once.
    Returns {'vocals': ndarray, 'no_vocals': ndarray} — mono float32 at SR.
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
                [
                    "python", "-m", "demucs",
                    "--two-stems", "vocals",
                    "--model", "htdemucs_ft",
                    "--out", cache_dir,
                    str(tmp),
                ],
                capture_output=True,
                text=True,
            )
            if result.returncode != 0:
                raise RuntimeError(
                    f"Demucs failed (exit {result.returncode}):\n{result.stderr}"
                )
            raw = Path(cache_dir) / "htdemucs_ft" / fid
            raw.rename(cached)
            # Clean up empty parent
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
        if y.ndim > 1:
            y = y.mean(axis=1)
        if file_sr != SR:
            y = librosa.resample(y, orig_sr=file_sr, target_sr=SR)
        stems[name] = y.astype(np.float32)
    return stems


# ── Analysis ──────────────────────────────────────────────────────────────────

def detect_bpm(y: np.ndarray) -> float:
    """Detect BPM using librosa beat tracker on the full audio."""
    tempo, _ = librosa.beat.beat_track(y=y, sr=SR)
    return max(float(tempo), 40.0)


def _best_ratio(bpm_a: float, bpm_b: float) -> float:
    """
    Return the time-stretch ratio that brings bpm_b closest to bpm_a,
    considering ×0.5 and ×2 multiples so we never over-stretch.
    Capped at ±50% to avoid quality degradation.
    """
    candidates = [bpm_a / (bpm_b * m) for m in (0.5, 1.0, 2.0)]
    ratio = min(candidates, key=lambda r: abs(r - 1.0))
    return float(np.clip(ratio, 0.667, 1.5))


def detect_key(y: np.ndarray) -> tuple:
    """
    Krumhansl-Schmuckler key detection on the full song audio.
    Returns (root_semitone: int, mode: str)  e.g. (9, 'minor') = A minor.
    """
    chroma = librosa.feature.chroma_cqt(y=y, sr=SR, bins_per_octave=36)
    mean_ch = chroma.mean(axis=1)
    mean_ch /= mean_ch.sum() + 1e-9

    best_score, best_root, best_mode = -np.inf, 0, "major"
    for root in range(12):
        rot = np.roll(mean_ch, -root)
        for profile, mode in ((_KS_MAJOR, "major"), (_KS_MINOR, "minor")):
            p = profile / profile.sum()
            score = np.dot(rot, p)
            if score > best_score:
                best_score, best_root, best_mode = score, root, mode
    return best_root, best_mode


def semitones_to_shift(src_root: int, src_mode: str,
                       dst_root: int, dst_mode: str) -> int:
    """
    Smallest semitone shift to bring src key into dst key.
    Minor keys are compared via their relative major to avoid over-shifting.
    """
    src = (src_root + 3) % 12 if src_mode == "minor" else src_root
    dst = (dst_root + 3) % 12 if dst_mode == "minor" else dst_root
    diff = (dst - src) % 12
    return diff - 12 if diff > 6 else diff


# ── DSP ───────────────────────────────────────────────────────────────────────

def _process_vocals(vox: np.ndarray) -> np.ndarray:
    """Vocal chain via Pedalboard (JUCE): highpass + compression."""
    board = Pedalboard([
        HighpassFilter(cutoff_frequency_hz=80.0),
        Compressor(threshold_db=-18.0, ratio=4.0,
                   attack_ms=10.0, release_ms=80.0),
    ])
    return board(vox, SR).astype(np.float32)


def _carve_pocket(inst: np.ndarray) -> np.ndarray:
    """
    Cut ~2 dB of 1–4 kHz from the instrumental to open a pocket for vocals.
    A gentle bandpass-subtract is mathematically equivalent to a soft notch.
    """
    sos = butter(4,
                 [1000.0 / (SR / 2), 4000.0 / (SR / 2)],
                 btype="bandpass", output="sos")
    band = sosfilt(sos, inst)
    return (inst - 0.21 * band).astype(np.float32)  # 0.21 ≈ -2 dB


def _sidechain(inst: np.ndarray, vox: np.ndarray,
               depth: float = 0.25, window_ms: int = 30) -> np.ndarray:
    """
    Duck instrumental by `depth` when vocals are loud.
    Block-based envelope follower — no sample-level Python loop.
    """
    win = max(1, int(SR * window_ms / 1000))
    hop = win // 2
    n_vox = len(vox)
    n_frames = max(1, (n_vox + hop - 1) // hop)

    env = np.zeros(n_frames, dtype=np.float32)
    for i in range(n_frames):
        s = i * hop
        e = min(s + win, n_vox)
        env[i] = np.sqrt(np.mean(vox[s:e] ** 2))

    vocal_rms = _rms(vox)
    env /= vocal_rms + 1e-9
    reduction = (1.0 - depth * np.clip(env, 0.0, 1.0)).astype(np.float64)

    x_frames = np.arange(n_frames, dtype=np.float64) * hop
    x_samp = np.arange(len(inst), dtype=np.float64)
    gain = interp1d(
        x_frames, reduction, kind="linear",
        bounds_error=False, fill_value=(reduction[0], reduction[-1])
    )(x_samp).astype(np.float32)

    return inst * gain


def _lufs_normalize(y: np.ndarray, target: float = -14.0) -> np.ndarray:
    meter = pyln.Meter(SR)
    lufs = meter.integrated_loudness(y)
    if not np.isfinite(lufs) or lufs < -70.0:
        return y
    return (y * 10 ** ((target - lufs) / 20)).astype(np.float32)


def _master(mix: np.ndarray) -> np.ndarray:
    """Soft clip → LUFS normalise → true-peak limit."""
    mix = (np.tanh(mix * 0.95) / 0.95).astype(np.float32)
    mix = _lufs_normalize(mix, -14.0)
    peak = float(np.max(np.abs(mix)))
    limit = 10 ** (-1.0 / 20)  # -1 dBTP
    if peak > limit:
        mix = (mix * limit / peak).astype(np.float32)
    return mix


def _fade(y: np.ndarray, fade_s: float = 2.0) -> np.ndarray:
    n = min(int(SR * fade_s), len(y) // 6)
    y = y.copy()
    y[:n]  *= np.linspace(0.0, 1.0, n) ** 0.5
    y[-n:] *= np.linspace(1.0, 0.0, n) ** 0.5
    return y


# ── Main Entry Point ──────────────────────────────────────────────────────────

def fuse(song_a: str, song_b: str, out_path: str,
         stems_cache: str = "vf_data/stems") -> str:
    """
    Fuse Song A (provides beat/instrumental) with Song B (provides vocals).
    Writes a WAV to out_path and returns it.
    """
    # Load full songs for analysis — key/BPM detection on clean audio
    _log("[1/8] Loading audio for analysis…")
    full_a = _load_mono(song_a)
    full_b = _load_mono(song_b)

    _log("[2/8] Detecting BPM…")
    bpm_a = detect_bpm(full_a)
    bpm_b = detect_bpm(full_b)
    _log(f"      A: {bpm_a:.1f} BPM   B: {bpm_b:.1f} BPM")

    _log("[3/8] Detecting keys…")
    key_a_root, key_a_mode = detect_key(full_a)
    key_b_root, key_b_mode = detect_key(full_b)
    _log(f"      A: {_NOTES[key_a_root]} {key_a_mode}   "
         f"B: {_NOTES[key_b_root]} {key_b_mode}")

    _log("[4/8] Separating stems — Song A (instrumental)…")
    stems_a = separate(song_a, stems_cache)

    _log("[5/8] Separating stems — Song B (vocals)…")
    stems_b = separate(song_b, stems_cache)

    inst = stems_a["no_vocals"]
    vox  = stems_b["vocals"]

    _log("[6/8] Time-stretching & pitch-shifting vocals…")
    ratio = _best_ratio(bpm_a, bpm_b)
    _log(f"      BPM ratio: {ratio:.4f}")
    if abs(ratio - 1.0) > 0.005:
        vox = pyrb.time_stretch(vox, SR, ratio)

    n_semi = semitones_to_shift(key_b_root, key_b_mode, key_a_root, key_a_mode)
    _log(f"      Pitch shift: {n_semi:+d} semitones")
    if n_semi != 0:
        vox = pyrb.pitch_shift(vox, SR, n_semi)

    _log("[7/8] Mixing…")
    # Match lengths to the shorter of the two
    L = min(len(inst), len(vox))
    inst, vox = inst[:L], vox[:L]

    # Vocal chain: highpass + compression
    vox = _process_vocals(vox)

    # Level match: vocals at 90% of instrumental RMS
    inst_rms = _rms(inst)
    vox_rms  = _rms(vox)
    if vox_rms > 1e-9:
        vox = vox * (inst_rms * 0.90 / vox_rms)

    # EQ: carve 1–4 kHz pocket in instrumental
    inst = _carve_pocket(inst)

    # Sidechain: duck instrumental when vocals hit
    inst = _sidechain(inst, vox, depth=0.2)

    mix = inst + vox
    mix = _fade(mix, fade_s=2.0)

    _log("[8/8] Mastering…")
    mix = _master(mix)

    os.makedirs(os.path.dirname(os.path.abspath(out_path)), exist_ok=True)
    sf.write(out_path, mix, SR, subtype="PCM_16")
    _log(f"Done → {out_path}")
    return out_path
