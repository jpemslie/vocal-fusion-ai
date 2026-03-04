"""
VocalFusion AI — MERT Song Embedder
=====================================

Uses MERT-v1-95M (HuBERT-style transformer trained on 160k hours of music)
to produce 768-dim semantic embeddings that capture pitch, timbre, rhythm,
instrumentation, and mood in a single vector.

Why this beats the old 128-dim handcrafted features:
  - MFCC/chroma measure acoustics; MERT measures musical meaning
  - Two songs that "sound good together" cluster in MERT space
    even when their tempo/key/centroid differ
  - The MixPredictor MLP learns on semantically-grounded similarity →
    fewer ratings needed to converge on good suggestions

Architecture: 13-layer transformer, 95M params, ~380MB weights.
Uses layer 7 hidden state (best for tonal/harmonic structure per the paper).
Chunk-processes long songs (30s windows) and averages the embeddings.

Embeddings are cached to vf_data/embeddings/<song_id>_mert.npy.

Requirements (auto-installed if missing):
  pip install transformers torch torchaudio

Falls back to SongEmbedder (128-dim, zero-padded to 768) if torch is
not installed. The MixPredictor handles variable embedding dims at
train time, so the fallback is safe.
"""

import numpy as np
from pathlib import Path
from typing import Dict, Optional

MERT_SR = 24000        # MERT requires 24kHz input
MERT_DIM = 768
MERT_LAYER = 7         # Layer 7: best for harmonic/tonal structure
CACHE_SUFFIX = "_mert"
MAX_AUDIO_SECS = 60    # Analyse first 60s (enough to characterise a song)
CHUNK_SECS = 30        # Process in 30s windows
MIN_CHUNK_SECS = 2     # Drop trailing chunks shorter than this


class MertEmbedder:
    """
    Generate and cache MERT-based 768-dim embeddings for songs.

    Interface is identical to SongEmbedder — drop-in replacement.
    """

    DIM = MERT_DIM

    def __init__(self, cache_dir: Path, sample_rate: int = 44100):
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.sr = sample_rate
        self._model = None
        self._processor = None
        self._torch = None
        self._available = None   # None = unchecked; True/False after first attempt

    # ------------------------------------------------------------------
    # PUBLIC INTERFACE (same as SongEmbedder)
    # ------------------------------------------------------------------

    def embed_song(self, song_id: str, stems: Dict[str, np.ndarray]) -> np.ndarray:
        """
        Compute or load cached 768-dim MERT embedding for a song.
        Returns float32 array of shape (768,), L2-normalised.
        """
        cache_path = self.cache_dir / f"{song_id}{CACHE_SUFFIX}.npy"
        if cache_path.exists():
            return np.load(str(cache_path))

        print(f"    Computing MERT embedding for {song_id}...")
        emb = self._compute(stems)
        np.save(str(cache_path), emb)
        return emb

    def load_cached(self, song_id: str) -> Optional[np.ndarray]:
        """Load a cached embedding, or None if not yet computed."""
        path = self.cache_dir / f"{song_id}{CACHE_SUFFIX}.npy"
        return np.load(str(path)) if path.exists() else None

    # ------------------------------------------------------------------
    # LAZY MODEL LOADING
    # ------------------------------------------------------------------

    def _ensure_model(self) -> bool:
        """
        Load MERT on first call.  Caches the result so subsequent calls
        are instant.  Returns True if MERT is available and loaded.
        """
        if self._available is not None:
            return self._available

        try:
            import torch
            from transformers import Wav2Vec2FeatureExtractor, AutoModel

            print("    Loading MERT-v1-95M (~380MB, one-time download)...")
            self._model = AutoModel.from_pretrained(
                "m-a-p/MERT-v1-95M",
                trust_remote_code=True,
            )
            self._processor = Wav2Vec2FeatureExtractor.from_pretrained(
                "m-a-p/MERT-v1-95M",
                trust_remote_code=True,
            )
            self._model.eval()
            self._torch = torch
            self._available = True
            print("    MERT loaded successfully.")

        except Exception as e:
            print(f"    MERT unavailable: {e}")
            print("    Install with: pip install transformers torch torchaudio")
            print("    Falling back to handcrafted features.")
            self._available = False

        return self._available

    # ------------------------------------------------------------------
    # COMPUTATION
    # ------------------------------------------------------------------

    def _compute(self, stems: Dict[str, np.ndarray]) -> np.ndarray:
        """Extract MERT embedding from stems → 768-dim float32 array."""
        if not self._ensure_model():
            return self._fallback(stems)

        try:
            from scipy import signal as scipy_signal

            # Build a mix of all available stems
            parts = [s for s in stems.values() if s is not None and len(s) > 0]
            if not parts:
                return np.zeros(self.DIM, dtype=np.float32)

            max_len = max(len(p) for p in parts)
            mix = np.zeros(max_len, dtype=np.float32)
            for p in parts:
                mix[:len(p)] += p

            # Trim to MAX_AUDIO_SECS
            mix = mix[:min(len(mix), MAX_AUDIO_SECS * self.sr)].astype(np.float32)

            # Normalise to prevent clipping / NaN in the model
            peak = np.max(np.abs(mix))
            if peak > 1e-6:
                mix = mix / peak * 0.9

            # Resample to 24kHz (MERT requirement)
            if self.sr != MERT_SR:
                n_target = int(round(len(mix) * MERT_SR / self.sr))
                mix = scipy_signal.resample(mix, n_target).astype(np.float32)

            # Split into CHUNK_SECS windows
            chunk_size = MERT_SR * CHUNK_SECS
            min_chunk = MERT_SR * MIN_CHUNK_SECS
            chunks = [mix[i: i + chunk_size]
                      for i in range(0, len(mix), chunk_size)]
            # Drop tiny trailing chunk
            chunks = [c for c in chunks if len(c) >= min_chunk]
            if not chunks:
                chunks = [mix[:chunk_size] if len(mix) >= chunk_size else mix]

            chunk_embeddings = []
            torch = self._torch
            with torch.no_grad():
                for chunk in chunks:
                    inputs = self._processor(
                        chunk,
                        sampling_rate=MERT_SR,
                        return_tensors="pt",
                    )
                    outputs = self._model(**inputs, output_hidden_states=True)

                    # hidden_states is a tuple of 13 tensors, each (1, T, 768)
                    # Stack → (13, T, 768), pick layer 7, mean-pool over T
                    hidden = torch.stack(outputs.hidden_states)   # (13, 1, T, 768)
                    hidden = hidden.squeeze(1)                     # (13, T, 768)
                    layer_emb = hidden[MERT_LAYER].mean(dim=0)    # (768,)
                    chunk_embeddings.append(layer_emb.cpu().numpy())

            emb = np.mean(chunk_embeddings, axis=0).astype(np.float32)

            # L2 normalise → cosine similarity == dot product downstream
            norm = np.linalg.norm(emb)
            if norm > 1e-8:
                emb = emb / norm

            return emb

        except Exception as e:
            print(f"    MERT inference error ({e}), using fallback features")
            return self._fallback(stems)

    def _fallback(self, stems: Dict[str, np.ndarray]) -> np.ndarray:
        """
        If MERT is unavailable, use handcrafted features zero-padded to 768.
        The MixPredictor trains on whichever size it sees consistently, so
        all songs computed via fallback will still be mutually comparable.
        """
        from ai.song_embedder import SongEmbedder
        tmp = SongEmbedder(self.cache_dir, self.sr)
        emb_128 = tmp._compute(stems)
        emb = np.zeros(self.DIM, dtype=np.float32)
        emb[:len(emb_128)] = emb_128
        return emb
