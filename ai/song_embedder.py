"""
VocalFusion AI — Song Embedder
================================

Generates compact audio embeddings for each song.
Used by MixPredictor to learn from user ratings.

Feature set (128-dim, always available, no extra deps):
  - MFCC-20 (mean + std = 40 features)
  - Chroma-12 mean (12 features)
  - Tempo normalized (1 feature)
  - Per-stem RMS: vocals, drums, bass, other (4 features)
  - Per-stem spectral centroid: vocals, drums, bass, other (4 features)
  - Vocal zero-crossing rate (1 feature)
  Total: 62 features → zero-padded to 128
"""

import numpy as np
import librosa
from pathlib import Path
from typing import Dict, Optional


class SongEmbedder:
    """Generate and cache compact audio embeddings for songs."""

    DIM = 128  # output embedding dimension

    def __init__(self, cache_dir: Path, sample_rate: int = 44100):
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.sr = sample_rate

    def embed_song(self, song_id: str, stems: Dict[str, np.ndarray]) -> np.ndarray:
        """
        Compute or load cached embedding for a song.
        Returns a 128-dim float32 vector.
        """
        cache_path = self.cache_dir / f"{song_id}.npy"
        if cache_path.exists():
            return np.load(str(cache_path))

        print(f"    Computing embedding for {song_id}...")
        emb = self._compute(stems)
        np.save(str(cache_path), emb)
        return emb

    def load_cached(self, song_id: str) -> Optional[np.ndarray]:
        """Load a cached embedding, or None if not yet computed."""
        path = self.cache_dir / f"{song_id}.npy"
        return np.load(str(path)) if path.exists() else None

    def _compute(self, stems: Dict[str, np.ndarray]) -> np.ndarray:
        """Extract handcrafted features from stems → DIM-dim embedding."""
        vocals = stems.get('vocals')
        drums = stems.get('drums')
        bass = stems.get('bass')
        other = stems.get('other')

        # Build mixed signal for global features
        parts = [s for s in [vocals, drums, bass, other] if s is not None]
        if parts:
            mx = max(len(p) for p in parts)
            mix = np.zeros(mx)
            for p in parts:
                mix[:len(p)] += p
        else:
            mix = np.zeros(self.sr)

        mix60 = mix[:min(len(mix), 60 * self.sr)].astype(np.float32)
        features = []

        # MFCC-20 (mean + std = 40 features)
        try:
            mfcc = librosa.feature.mfcc(y=mix60, sr=self.sr, n_mfcc=20)
            features.extend(np.mean(mfcc, axis=1).tolist())
            features.extend(np.std(mfcc, axis=1).tolist())
        except Exception:
            features.extend([0.0] * 40)

        # Chroma mean (12 features)
        try:
            seg = mix[:min(len(mix), 30 * self.sr)].astype(np.float32)
            chroma = librosa.feature.chroma_cqt(y=seg, sr=self.sr)
            features.extend(np.mean(chroma, axis=1).tolist())
        except Exception:
            features.extend([0.0] * 12)

        # Tempo normalized (1 feature)
        try:
            tempo, _ = librosa.beat.beat_track(
                y=mix[:min(len(mix), 30 * self.sr)], sr=self.sr)
            features.append(float(tempo) / 200.0)
        except Exception:
            features.append(0.5)

        # Per-stem RMS (4 features)
        for stem in [vocals, drums, bass, other]:
            if stem is not None and len(stem) > 0:
                seg = stem[:min(len(stem), 30 * self.sr)]
                rms = float(np.sqrt(np.mean(seg ** 2)))
                features.append(float(np.clip(rms * 10, 0, 1)))
            else:
                features.append(0.0)

        # Per-stem spectral centroid mean (4 features)
        for stem in [vocals, drums, bass, other]:
            if stem is not None and len(stem) > 0:
                try:
                    seg = stem[:min(len(stem), 30 * self.sr)].astype(np.float32)
                    sc = librosa.feature.spectral_centroid(y=seg, sr=self.sr)[0]
                    features.append(float(np.mean(sc)) / 8000.0)
                except Exception:
                    features.append(0.0)
            else:
                features.append(0.0)

        # Vocal zero-crossing rate (1 feature)
        if vocals is not None and len(vocals) > 0:
            try:
                seg = vocals[:min(len(vocals), 10 * self.sr)].astype(np.float32)
                zcr = librosa.feature.zero_crossing_rate(seg)[0]
                features.append(float(np.mean(zcr)))
            except Exception:
                features.append(0.0)
        else:
            features.append(0.0)

        # 40 + 12 + 1 + 4 + 4 + 1 = 62 features → pad to DIM
        arr = np.array(features, dtype=np.float32)
        if len(arr) < self.DIM:
            arr = np.pad(arr, (0, self.DIM - len(arr)))
        else:
            arr = arr[:self.DIM]

        return np.clip(arr, -3.0, 3.0).astype(np.float32)
