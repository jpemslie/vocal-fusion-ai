"""
VocalFusion Fusion Engine
=========================
Song A  →  instrumental (no_vocals stem from Demucs)
Song B  →  vocals stem
Output  →  stereo mixed, mastered WAV

Quality decisions
─────────────────
• Full stereo pipeline — stems stay (samples, 2), no mono collapse
• Key/BPM detection uses a mono mix of the original file (clean signal)
• chroma_cens for key detection (energy-normalised, robust to timbre)
• BPM ratio: direct bpm_a/bpm_b, capped ±50% to avoid stretch artefacts
• Pedalboard (JUCE) vocal chain per channel: HPF + compression + air shelf
• EQ carving: gentle -2 dB bandstop at 1–4 kHz in instrumental
• Sidechain duck: envelope from mono mix of vocals, applied to both channels
• Master: soft-clip → LUFS -14 → true-peak -1 dBTP
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
import pyrubberband as pyrb
import soundfile as sf
from pedalboard import Compressor, HighpassFilter, Pedalboard
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
    """(samples, 2) or (samples,) → (samples,) mono float32."""
    if y.ndim == 2:
        return y.mean(axis=1).astype(np.float32)
    return y.astype(np.float32)


def _to_stereo(y: np.ndarray) -> np.ndarray:
    """(samples,) → (samples, 2) by duplicating channel."""
    if y.ndim == 1:
        return np.stack([y, y], axis=1)
    return y


def _rms(y: np.ndarray) -> float:
    return float(np.sqrt(np.mean(y ** 2) + 1e-12))


def _active_rms(y: np.ndarray, threshold_db: float = -48.0) -> float:
    """
    RMS of only the active (non-silent) samples.
    Avoids the silent gaps in Demucs vocal stems dragging down the level match.
    """
    mono = _to_mono(y)
    peak = float(np.max(np.abs(mono)) + 1e-12)
    cutoff = peak * 10 ** (threshold_db / 20)
    active = mono[np.abs(mono) > cutoff]
    if len(active) < SR:          # less than 1s of active audio — fall back
        return _rms(y)
    return float(np.sqrt(np.mean(active ** 2) + 1e-12))


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
                [
                    sys.executable, "-m", "demucs",
                    "--two-stems", "vocals",
                    "-n", "htdemucs_ft",
                    "-o", cache_dir,
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
        # Ensure stereo (samples, 2)
        if y.ndim == 1:
            y = np.stack([y, y], axis=1)
        if file_sr != SR:
            y = np.stack([
                librosa.resample(y[:, c], orig_sr=file_sr, target_sr=SR)
                for c in range(y.shape[1])
            ], axis=1)
        stems[name] = y.astype(np.float32)
    return stems


# ── Analysis (mono) ───────────────────────────────────────────────────────────

def detect_bpm(y_mono: np.ndarray) -> float:
    tempo, _ = librosa.beat.beat_track(y=y_mono, sr=SR)
    return max(float(tempo), 40.0)


def _best_ratio(bpm_a: float, bpm_b: float) -> float:
    """rate > 1 = faster, rate < 1 = slower. Capped ±50%."""
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


def semitones_to_shift(src_root: int, src_mode: str,
                       dst_root: int, dst_mode: str) -> int:
    src = (src_root + 3) % 12 if src_mode == "minor" else src_root
    dst = (dst_root + 3) % 12 if dst_mode == "minor" else dst_root
    diff = (dst - src) % 12
    return diff - 12 if diff > 6 else diff


# ── Stereo DSP ────────────────────────────────────────────────────────────────

def _stretch_stereo(y: np.ndarray, ratio: float) -> np.ndarray:
    """Time-stretch each channel independently via pyrubberband."""
    return np.stack([
        pyrb.time_stretch(y[:, c], SR, ratio)
        for c in range(y.shape[1])
    ], axis=1).astype(np.float32)


def _shift_stereo(y: np.ndarray, n_semitones: int) -> np.ndarray:
    """Pitch-shift each channel independently via pyrubberband."""
    return np.stack([
        pyrb.pitch_shift(y[:, c], SR, n_semitones)
        for c in range(y.shape[1])
    ], axis=1).astype(np.float32)


def _process_vocals(vox: np.ndarray) -> np.ndarray:
    """
    Per-channel vocal chain via Pedalboard (JUCE):
      HPF 80 Hz → compression (4:1) → +2 dB air shelf at 8 kHz
    """
    board = Pedalboard([
        HighpassFilter(cutoff_frequency_hz=80.0),
        Compressor(threshold_db=-18.0, ratio=4.0,
                   attack_ms=10.0, release_ms=80.0),
    ])
    sos_shelf = butter(2, 8000.0 / (SR / 2), btype="high", output="sos")

    channels = []
    for c in range(vox.shape[1]):
        ch = board(vox[:, c], SR)
        ch = ch + 0.26 * sosfilt(sos_shelf, ch)   # +2 dB air
        channels.append(ch.astype(np.float32))
    return np.stack(channels, axis=1)


def _carve_pocket(inst: np.ndarray) -> np.ndarray:
    """
    Cut ~2 dB of 1–4 kHz in the instrumental (bandpass-subtract method).
    Applied simultaneously to both channels via sosfilt axis=0.
    """
    sos = butter(4,
                 [1000.0 / (SR / 2), 4000.0 / (SR / 2)],
                 btype="bandpass", output="sos")
    band = sosfilt(sos, inst, axis=0)
    return (inst - 0.21 * band).astype(np.float32)


def _sidechain(inst: np.ndarray, vox: np.ndarray,
               depth: float = 0.2, window_ms: int = 30) -> np.ndarray:
    """
    Duck instrumental by `depth` when vocals are present.
    Envelope is derived from the vocal mono mix; gain applied to both channels.
    """
    vox_mono = _to_mono(vox)
    win = max(1, int(SR * window_ms / 1000))
    hop = win // 2
    n_vox = len(vox_mono)
    n_frames = max(1, (n_vox + hop - 1) // hop)

    env = np.array([
        np.sqrt(np.mean(vox_mono[i * hop: min(i * hop + win, n_vox)] ** 2))
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

    # Broadcast (samples,) gain over (samples, 2) inst
    return inst * gain[:, np.newaxis]


def _lufs_normalize(y: np.ndarray, target: float = -14.0) -> np.ndarray:
    """pyloudnorm expects (samples,) mono or (samples, channels) stereo."""
    meter = pyln.Meter(SR)
    lufs = meter.integrated_loudness(y)
    if not np.isfinite(lufs) or lufs < -70.0:
        return y
    return (y * 10 ** ((target - lufs) / 20)).astype(np.float32)


def _master(mix: np.ndarray) -> np.ndarray:
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
    fade_in  = np.linspace(0.0, 1.0, n, dtype=np.float32) ** 0.5
    fade_out = np.linspace(1.0, 0.0, n, dtype=np.float32) ** 0.5
    y[:n]  *= fade_in[:, np.newaxis]
    y[-n:] *= fade_out[:, np.newaxis]
    return y


# ── Main Entry Point ──────────────────────────────────────────────────────────

def fuse(song_a: str, song_b: str, out_path: str,
         stems_cache: str = "vf_data/stems",
         progress_cb=None) -> str:
    """
    Fuse Song A (beat/instrumental) with Song B (vocals).
    Writes stereo WAV to out_path and returns the path.
    progress_cb(step, total, message) called at each stage if provided.
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

    step(6, "Time-stretching & pitch-shifting vocals…")
    ratio = _best_ratio(bpm_a, bpm_b)
    print(f"      BPM ratio: {ratio:.4f}", flush=True)
    if abs(ratio - 1.0) > 0.005:
        vox = _stretch_stereo(vox, ratio)

    n_semi = semitones_to_shift(key_b_root, key_b_mode, key_a_root, key_a_mode)
    print(f"      Pitch shift: {n_semi:+d} semitones", flush=True)
    if n_semi != 0:
        vox = _shift_stereo(vox, n_semi)

    step(7, "Mixing…")
    L = min(len(inst), len(vox))
    inst, vox = inst[:L], vox[:L]

    vox = _process_vocals(vox)

    # Level match using active-region RMS (ignores silent gaps in vocal stem)
    ir = _active_rms(inst)
    vr = _active_rms(vox)
    if vr > 1e-9:
        vox = (vox * (ir * 0.90 / vr)).astype(np.float32)

    inst = _carve_pocket(inst)
    inst = _sidechain(inst, vox, depth=0.2)

    mix = (inst + vox).astype(np.float32)
    mix = _fade(mix, fade_s=2.0)

    step(8, "Mastering…")
    mix = _master(mix)

    os.makedirs(os.path.dirname(os.path.abspath(out_path)), exist_ok=True)
    sf.write(out_path, mix, SR, subtype="PCM_16")
    print(f"Done → {out_path}", flush=True)
    return out_path
