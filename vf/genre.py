"""
genre.py — Fast rule-based genre/style detector using librosa audio features.
"""

import json
import sys
from pathlib import Path

import librosa
import numpy as np


def detect_genre(path: str, sr: int = 44100) -> tuple[str, dict]:
    """
    Detect the genre/style of an audio file using rule-based classification.

    Parameters
    ----------
    path : str
        Path to the audio file.
    sr : int
        Target sample rate for loading (default 44100).

    Returns
    -------
    tuple[str, dict]
        (genre_label, feature_dict) where genre_label is one of:
        "pop", "hiphop", "rnb", "edm", "rock", "acoustic"
        and feature_dict contains the computed audio features as floats.
    """
    try:
        # Load at most 60 seconds, mix to mono
        y, sr_loaded = librosa.load(path, sr=sr, mono=True, duration=60)

        # --- Tempo ---
        tempo_arr, _ = librosa.beat.beat_track(y=y, sr=sr_loaded)
        # beat_track may return a scalar or 1-element array depending on librosa version
        tempo = float(np.atleast_1d(tempo_arr)[0])

        # --- Spectral centroid ---
        spectral_centroid = librosa.feature.spectral_centroid(y=y, sr=sr_loaded)
        spectral_centroid_mean = float(np.mean(spectral_centroid))

        # --- Spectral rolloff (85%) ---
        spectral_rolloff = librosa.feature.spectral_rolloff(y=y, sr=sr_loaded, roll_percent=0.85)
        spectral_rolloff_mean = float(np.mean(spectral_rolloff))

        # --- Zero-crossing rate ---
        zcr = librosa.feature.zero_crossing_rate(y)
        zcr_mean = float(np.mean(zcr))

        # --- RMS energy ---
        rms = librosa.feature.rms(y=y)
        rms_mean = float(np.mean(rms))

        # --- Mel spectrogram for bass_ratio and brightness ---
        n_mels = 128
        mel_spec = librosa.feature.melspectrogram(y=y, sr=sr_loaded, n_mels=n_mels)
        # Convert power to energy (already in power, use as-is for ratios)
        mel_freqs = librosa.mel_frequencies(n_mels=n_mels, fmin=0.0, fmax=sr_loaded / 2)

        total_energy = float(np.sum(mel_spec))

        # bass_ratio: energy below 250 Hz
        bass_mask = mel_freqs < 250.0
        bass_energy = float(np.sum(mel_spec[bass_mask, :]))
        bass_ratio = bass_energy / total_energy if total_energy > 0 else 0.0

        # brightness: energy above 4000 Hz
        bright_mask = mel_freqs > 4000.0
        bright_energy = float(np.sum(mel_spec[bright_mask, :]))
        brightness = bright_energy / total_energy if total_energy > 0 else 0.0

        # --- Build feature dict (rounded to 4 decimal places) ---
        features: dict = {
            "tempo": round(tempo, 4),
            "spectral_centroid_mean": round(spectral_centroid_mean, 4),
            "spectral_rolloff_mean": round(spectral_rolloff_mean, 4),
            "zcr_mean": round(zcr_mean, 4),
            "rms_mean": round(rms_mean, 4),
            "bass_ratio": round(bass_ratio, 4),
            "brightness": round(brightness, 4),
        }

        # --- Rule-based classification (first match wins) ---

        # 1. EDM
        if tempo >= 120 and brightness > 0.25 and bass_ratio > 0.30:
            return ("edm", features)

        # 2. Hip-hop
        if 70 <= tempo <= 105 and bass_ratio > 0.35:
            return ("hiphop", features)

        # 3. R&B
        if 60 <= tempo <= 110 and spectral_centroid_mean < 2500 and bass_ratio > 0.28:
            return ("rnb", features)

        # 4. Rock
        if zcr_mean > 0.08 and spectral_centroid_mean > 2200 and tempo > 100:
            return ("rock", features)

        # 5. Acoustic
        if rms_mean < 0.08 and brightness < 0.15:
            return ("acoustic", features)

        # 6. Pop (default)
        return ("pop", features)

    except Exception:
        return ("pop", {})


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python vf/genre.py /path/to/file.wav", file=sys.stderr)
        sys.exit(1)

    audio_path = sys.argv[1]
    genre_label, feature_dict = detect_genre(audio_path)

    result = {"genre": genre_label, "features": feature_dict}
    print(json.dumps(result, indent=2))
