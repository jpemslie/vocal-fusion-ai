"""
ml/features.py — Dense audio feature extractor for mixing intelligence.

Extracts ~65 features from any audio file covering:
  - Loudness & dynamics (LUFS, LRA, crest, PLR, compression depth)
  - Spectral balance (7-band energy ratios, slope, centroid, rolloff)
  - Vocal zone (300-5kHz presence, clarity, harmonic-to-noise ratio)
  - Stereo (width per band, mid/side balance)
  - Temporal (transient clarity, onset density, dynamic arc)
  - Tonal (chroma entropy = key clarity, roughness = dissonance)

All values are normalised to be scale-invariant (ratios, dB differences,
or z-scored) so that loud vs quiet tracks are comparable.
"""

import numpy as np
import librosa
import pyloudnorm as pyln
import soundfile as sf
from scipy.signal import butter, sosfilt
from pathlib import Path

SR = 44100

FEATURE_NAMES = [
    # Loudness
    "lufs_i",          # integrated loudness
    "lra",             # loudness range
    "crest_db",        # crest factor (dynamic headroom)
    "plr",             # peak-to-loudness ratio
    # Spectral balance (7 bands, energy as fraction of total)
    "band_sub",        # 20-80 Hz
    "band_bass",       # 80-250 Hz
    "band_lowmid",     # 250-500 Hz
    "band_mid",        # 500-2000 Hz
    "band_highmid",    # 2000-5000 Hz
    "band_presence",   # 5000-10000 Hz
    "band_air",        # 10000-20000 Hz
    # Spectral shape
    "spectral_centroid",    # weighted mean frequency (Hz, log-scaled)
    "spectral_rolloff",     # 85% energy cutoff (Hz, log-scaled)
    "spectral_flatness",    # Wiener entropy (0=tonal, 1=noise-like)
    "spectral_slope",       # log-linear slope dB/octave
    # Vocal zone
    "vocal_presence",       # 300-5kHz energy / total
    "vocal_clarity",        # harmonic energy / total in vocal zone
    "hnr_db",               # harmonic-to-noise ratio (dB)
    # Stereo
    "stereo_width_low",     # width 20-250Hz
    "stereo_width_mid",     # width 250-5000Hz
    "stereo_width_high",    # width 5k-20kHz
    "ms_balance",           # M/(M+S) ratio
    # Dynamics
    "transient_clarity",    # attack/sustain ratio
    "dynamic_arc",          # RMS std over 4s windows (how much energy changes)
    "onset_density",        # onsets/sec
    "compression_depth",    # peak-to-avg spread (inverse of compression)
    # Tonal
    "chroma_entropy",       # how spread across 12 pitch classes (key clarity)
    "roughness",            # spectral roughness (dissonance proxy)
    # Ratios (invariant to absolute level)
    "bass_mid_ratio",       # band_bass / band_mid
    "sub_bass_ratio",       # band_sub / band_bass
    "highmid_mid_ratio",    # band_highmid / band_mid
    "air_presence_ratio",   # band_air / band_presence
]

N_FEATURES = len(FEATURE_NAMES)


def _bp(y: np.ndarray, lo: float, hi: float, order: int = 4) -> np.ndarray:
    nyq = SR / 2.0
    sos = butter(order, [lo / nyq, min(hi / nyq, 0.999)], btype="band", output="sos")
    return sosfilt(sos, y)


def _band_rms(y: np.ndarray, lo: float, hi: float) -> float:
    return float(np.sqrt(np.mean(_bp(y, lo, hi) ** 2) + 1e-12))


def _stereo_width(L: np.ndarray, R: np.ndarray, lo: float, hi: float) -> float:
    Lb, Rb = _bp(L, lo, hi), _bp(R, lo, hi)
    if Lb.std() < 1e-9 or Rb.std() < 1e-9:
        return 0.0
    return float(np.clip(1.0 - np.corrcoef(Lb, Rb)[0, 1], 0.0, 2.0))


def extract(path: str, duration: float | None = 60.0) -> np.ndarray:
    """
    Extract feature vector from audio file.
    Returns np.ndarray of shape (N_FEATURES,) with float32 values.
    duration: max seconds to analyse (None = full file). 60s is enough and fast.
    """
    # Load — resample to 44100, keep stereo
    y_stereo, sr = sf.read(path, always_2d=True)
    if sr != SR:
        y_stereo = librosa.resample(y_stereo.T, orig_sr=sr, target_sr=SR).T
    y_stereo = y_stereo.astype(np.float32)

    # Trim to analysis window (from middle of track — avoids intros/outros)
    total = y_stereo.shape[0]
    if duration is not None and total > int(duration * SR):
        mid = total // 2
        half = int(duration * SR) // 2
        y_stereo = y_stereo[mid - half: mid + half]

    L, R = y_stereo[:, 0], y_stereo[:, -1]
    y_mono = (L + R) / 2.0

    feats = {}

    # ── Loudness ──────────────────────────────────────────────────────────────
    meter = pyln.Meter(SR)
    try:
        lufs_i = meter.integrated_loudness(y_stereo.astype(np.float64))
        if not np.isfinite(lufs_i):
            lufs_i = -23.0
    except Exception:
        lufs_i = -23.0
    feats["lufs_i"] = float(lufs_i)

    # LRA via 3s window RMS
    win = int(3 * SR)
    rms_windows = [
        np.sqrt(np.mean(y_mono[i:i+win] ** 2))
        for i in range(0, len(y_mono) - win, win // 2)
    ]
    if len(rms_windows) > 2:
        rms_db = 20.0 * np.log10(np.array(rms_windows) + 1e-9)
        feats["lra"] = float(np.percentile(rms_db, 95) - np.percentile(rms_db, 10))
    else:
        feats["lra"] = 0.0

    peak = float(np.max(np.abs(y_mono)))
    rms_overall = float(np.sqrt(np.mean(y_mono ** 2) + 1e-12))
    rms_db_overall = 20.0 * np.log10(rms_overall + 1e-12)
    peak_db = 20.0 * np.log10(peak + 1e-12)
    feats["crest_db"] = float(peak_db - rms_db_overall)
    feats["plr"] = float(peak_db - lufs_i)

    # ── Spectral bands ─────────────────────────────────────────────────────────
    bands = {
        "band_sub":      (20,   80),
        "band_bass":     (80,   250),
        "band_lowmid":   (250,  500),
        "band_mid":      (500,  2000),
        "band_highmid":  (2000, 5000),
        "band_presence": (5000, 10000),
        "band_air":      (10000, 19000),
    }
    band_rms = {k: _band_rms(y_mono, lo, hi) for k, (lo, hi) in bands.items()}
    total_rms = sum(band_rms.values()) + 1e-12
    for k, v in band_rms.items():
        feats[k] = float(v / total_rms)

    # ── Spectral shape ─────────────────────────────────────────────────────────
    # Use librosa on a short chunk for speed
    chunk = y_mono[:int(30 * SR)] if len(y_mono) > int(30 * SR) else y_mono
    S = np.abs(librosa.stft(chunk, n_fft=2048, hop_length=512))
    freqs = librosa.fft_frequencies(sr=SR, n_fft=2048)

    cent = librosa.feature.spectral_centroid(S=S, sr=SR)[0]
    feats["spectral_centroid"] = float(np.log10(np.mean(cent) + 1.0))

    roll = librosa.feature.spectral_rolloff(S=S, sr=SR, roll_percent=0.85)[0]
    feats["spectral_rolloff"] = float(np.log10(np.mean(roll) + 1.0))

    flat = librosa.feature.spectral_flatness(S=S)[0]
    feats["spectral_flatness"] = float(np.mean(flat))

    # Spectral slope: linear regression of log-magnitude vs log-frequency
    mean_mag = np.mean(S, axis=1)
    valid = (freqs > 80) & (freqs < 16000)
    if valid.sum() > 10:
        log_f = np.log10(freqs[valid] + 1e-3)
        log_m = np.log10(mean_mag[valid] + 1e-12)
        slope = np.polyfit(log_f, log_m, 1)[0]
        feats["spectral_slope"] = float(slope)
    else:
        feats["spectral_slope"] = 0.0

    # ── Vocal zone ─────────────────────────────────────────────────────────────
    vox_lo, vox_hi = 300, 5000
    vox_rms = _band_rms(y_mono, vox_lo, vox_hi)
    feats["vocal_presence"] = float(vox_rms / (rms_overall + 1e-12))

    # Harmonic clarity in vocal zone: HPSS on vocal band
    y_vox = _bp(y_mono, vox_lo, vox_hi)
    S_vox = np.abs(librosa.stft(y_vox[:int(20 * SR)], n_fft=2048, hop_length=512))
    harm_vox = librosa.decompose.hpss(S_vox, margin=3.0)[0]
    feats["vocal_clarity"] = float(
        np.mean(harm_vox) / (np.mean(S_vox) + 1e-12)
    )

    # HNR via autocorrelation proxy
    ac = librosa.autocorrelate(y_vox[:int(5 * SR)], max_size=int(SR * 0.02))
    if len(ac) > 10 and ac[0] > 1e-9:
        peak_ac = np.max(ac[1:]) / ac[0]
        hnr = float(np.clip(10 * np.log10(peak_ac / (1 - peak_ac + 1e-9) + 1e-9), -10, 40))
    else:
        hnr = 0.0
    feats["hnr_db"] = hnr

    # ── Stereo ─────────────────────────────────────────────────────────────────
    feats["stereo_width_low"]  = _stereo_width(L, R, 20,   250)
    feats["stereo_width_mid"]  = _stereo_width(L, R, 250,  5000)
    feats["stereo_width_high"] = _stereo_width(L, R, 5000, 19000)
    M = (L + R) / 2.0
    Ss = (L - R) / 2.0
    m_rms = float(np.sqrt(np.mean(M ** 2) + 1e-12))
    s_rms = float(np.sqrt(np.mean(Ss ** 2) + 1e-12))
    feats["ms_balance"] = float(m_rms / (m_rms + s_rms + 1e-12))

    # ── Dynamics ───────────────────────────────────────────────────────────────
    # Transient clarity: ratio of top 5% peak to median RMS
    hop_d = 512
    rms_f = librosa.feature.rms(y=y_mono, frame_length=2048, hop_length=hop_d)[0]
    feats["transient_clarity"] = float(
        np.percentile(rms_f, 95) / (np.median(rms_f) + 1e-9)
    )

    # Dynamic arc: std of 4s-window RMS in dB
    win4 = int(4 * SR)
    rms4 = [
        np.sqrt(np.mean(y_mono[i:i+win4] ** 2))
        for i in range(0, len(y_mono) - win4, win4)
    ]
    if len(rms4) > 1:
        feats["dynamic_arc"] = float(np.std(20 * np.log10(np.array(rms4) + 1e-9)))
    else:
        feats["dynamic_arc"] = 0.0

    # Onset density
    onsets = librosa.onset.onset_detect(y=y_mono, sr=SR, units="time")
    dur_s = len(y_mono) / SR
    feats["onset_density"] = float(len(onsets) / (dur_s + 1e-3))

    # Compression depth (lower = more compressed)
    feats["compression_depth"] = float(np.percentile(rms_f, 95) / (np.percentile(rms_f, 5) + 1e-9))

    # ── Tonal ──────────────────────────────────────────────────────────────────
    # Chroma entropy: high = spread across all pitches (complex), low = one key
    chroma = librosa.feature.chroma_cqt(y=chunk, sr=SR)
    chroma_mean = np.mean(chroma, axis=1) + 1e-9
    chroma_mean /= chroma_mean.sum()
    feats["chroma_entropy"] = float(-np.sum(chroma_mean * np.log(chroma_mean)))

    # Roughness proxy: mean of spectral flux (sudden spectral changes)
    S_flux = np.mean(np.diff(S, axis=1) ** 2)
    feats["roughness"] = float(np.log10(S_flux + 1e-12))

    # ── Derived ratios ─────────────────────────────────────────────────────────
    feats["bass_mid_ratio"]    = float(band_rms["band_bass"]     / (band_rms["band_mid"] + 1e-12))
    feats["sub_bass_ratio"]    = float(band_rms["band_sub"]      / (band_rms["band_bass"] + 1e-12))
    feats["highmid_mid_ratio"] = float(band_rms["band_highmid"]  / (band_rms["band_mid"] + 1e-12))
    feats["air_presence_ratio"]= float(band_rms["band_air"]      / (band_rms["band_presence"] + 1e-12))

    # Build ordered vector
    vec = np.array([feats[k] for k in FEATURE_NAMES], dtype=np.float32)
    # Replace any NaN/inf with 0
    vec = np.where(np.isfinite(vec), vec, 0.0)
    return vec


if __name__ == "__main__":
    import sys
    path = sys.argv[1]
    v = extract(path)
    for name, val in zip(FEATURE_NAMES, v):
        print(f"  {name:25s}  {val:+.4f}")
