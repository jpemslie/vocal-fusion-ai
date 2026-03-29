"""
CLAP-based quality scorer — perceptual, genre-agnostic replacement
for the Mahalanobis/PCA Drake-only reference scorer.

Uses Microsoft's CLAP (Contrastive Language-Audio Pretraining) to embed
audio into a 512-dim semantic space, then measures KNN distance to a
reference set of professional tracks.

Advantages over ml/scorer.py (PCA Mahalanobis):
  - Genre-agnostic: works on hip-hop, pop, classical, metal, etc.
  - Perceptual: CLAP understands tonal quality, not just spectral stats
  - Richer reference: can use 100s of tracks, not just 70 Drake records
  - No feature engineering: raw embeddings from pretrained model

Requirements:
    pip install msclap
    # or: pip install laion-clap

Usage:
    # Build reference embeddings from a folder of professional tracks
    python -m ml.clap_scorer --build-ref \
        --ref-dir /path/to/pro_tracks \
        --out ml/clap_ref.npz

    # Score a file
    python -m ml.clap_scorer --score path/to/output.wav \
        --ref ml/clap_ref.npz

    # Programmatic use
    from ml.clap_scorer import CLAPScorer
    scorer = CLAPScorer("ml/clap_ref.npz")
    result = scorer.score("output.wav")
    print(result)  # {'score': 74.3, 'grade': 'C', 'percentile': 88.1}
"""

import argparse
import sys
from pathlib import Path

import numpy as np


GRADES = [
    (85, "A"),
    (70, "B"),
    (55, "C"),
    (35, "D"),
    ( 0, "F"),
]


def _grade(score: float) -> str:
    for threshold, label in GRADES:
        if score >= threshold:
            return label
    return "F"


def _load_clap_model():
    """Try to load CLAP model. Supports both msclap and laion-clap."""
    # Try msclap first (Microsoft's version)
    try:
        from msclap import CLAP
        model = CLAP(version="2023", use_cuda=False)
        return model, "msclap"
    except ImportError:
        pass

    # Try laion-clap
    try:
        import laion_clap
        model = laion_clap.CLAP_Module(enable_fusion=False)
        model.load_ckpt()
        return model, "laion"
    except ImportError:
        pass

    raise ImportError(
        "CLAP not installed. Run one of:\n"
        "  pip install msclap\n"
        "  pip install laion-clap"
    )


def embed_file(audio_path: str | Path, model, backend: str,
               duration: float = 60.0, sr: int = 44100) -> np.ndarray:
    """
    Embed an audio file using CLAP.
    Returns (512,) float32 embedding.
    """
    import librosa
    y, _ = librosa.load(str(audio_path), sr=sr, mono=True, duration=duration)
    # Trim silence from start/end, use middle 60s
    start = max(0, len(y) // 2 - sr * 30)
    end   = min(len(y), start + int(duration * sr))
    y = y[start:end]

    if backend == "msclap":
        import tempfile, soundfile as sf, os
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
            tmp_path = f.name
        sf.write(tmp_path, y, sr)
        emb = model.get_audio_embeddings([tmp_path])
        os.unlink(tmp_path)
        return np.array(emb[0], dtype=np.float32)

    elif backend == "laion":
        import tempfile, soundfile as sf, os
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
            tmp_path = f.name
        sf.write(tmp_path, y, sr)
        emb = model.get_audio_embedding_from_filelist([tmp_path], use_tensor=False)
        os.unlink(tmp_path)
        return np.array(emb[0], dtype=np.float32)

    raise ValueError(f"Unknown backend: {backend}")


def build_reference(ref_dir: str | Path, out_path: str | Path,
                    duration: float = 60.0):
    """
    Build reference embeddings from a folder of professional tracks.
    Saves to .npz with keys: embeddings (N, 512), paths (N,)
    """
    ref_dir = Path(ref_dir)
    tracks = sorted(list(ref_dir.glob("*.wav")) +
                    list(ref_dir.glob("*.mp3")) +
                    list(ref_dir.glob("*.flac")))

    if not tracks:
        print(f"No audio files found in {ref_dir}")
        sys.exit(1)

    print(f"Building CLAP reference from {len(tracks)} tracks…")
    model, backend = _load_clap_model()
    embeddings = []
    paths = []

    for i, track in enumerate(tracks):
        print(f"  [{i+1}/{len(tracks)}] {track.name}", flush=True)
        try:
            emb = embed_file(track, model, backend, duration)
            embeddings.append(emb)
            paths.append(str(track))
        except Exception as e:
            print(f"    ERROR: {e}")

    if not embeddings:
        print("No embeddings generated.")
        sys.exit(1)

    emb_array = np.stack(embeddings).astype(np.float32)
    np.savez(out_path,
             embeddings=emb_array,
             paths=np.array(paths))
    print(f"\nSaved {len(embeddings)} embeddings → {out_path}")
    return emb_array


class CLAPScorer:
    """
    Score audio files against a CLAP reference set.

    Uses cosine similarity + KNN (k=5) to estimate how close
    the output sounds to professional reference tracks.

    Score 0-100:
        A (85-100): Very close to professional references
        B (70-84):  Good, minor differences
        C (55-69):  Acceptable, noticeable quality gap
        D (35-54):  Below commercial standard
        F (0-34):   Significant quality issues
    """

    def __init__(self, ref_path: str | Path, k: int = 5):
        data = np.load(ref_path)
        self.ref_embs = data["embeddings"].astype(np.float32)  # (N, 512)
        self.ref_paths = data.get("paths", np.array([]))
        self.k = min(k, len(self.ref_embs))

        # L2-normalize for cosine similarity
        norms = np.linalg.norm(self.ref_embs, axis=1, keepdims=True)
        self.ref_embs_norm = self.ref_embs / (norms + 1e-8)

        self._model = None
        self._backend = None

    def _get_model(self):
        if self._model is None:
            self._model, self._backend = _load_clap_model()
        return self._model, self._backend

    def score(self, audio_path: str | Path,
              duration: float = 60.0) -> dict:
        """
        Score an audio file.

        Returns:
            {
                'score':      float 0-100,
                'grade':      str 'A'/'B'/'C'/'D'/'F',
                'percentile': float (beats X% of references),
                'knn_sims':   list of K best cosine similarities,
                'nearest':    list of K nearest reference track paths,
            }
        """
        model, backend = self._get_model()
        emb = embed_file(audio_path, model, backend, duration)

        # Cosine similarity to all references
        emb_norm = emb / (np.linalg.norm(emb) + 1e-8)
        sims = self.ref_embs_norm @ emb_norm  # (N,)

        # KNN: mean of top-K similarities
        top_k_idx = np.argsort(sims)[::-1][:self.k]
        knn_mean_sim = float(sims[top_k_idx].mean())

        # Convert cosine similarity → score
        # cosine sim in CLAP space: 0.5-0.95 for professional audio
        # Map [0.5, 0.95] → [0, 100]
        score = float(np.clip((knn_mean_sim - 0.50) / (0.95 - 0.50) * 100, 0, 100))

        # Percentile: how many references this beats
        ref_self_sims = []
        for i, ref_emb in enumerate(self.ref_embs_norm):
            other_sims = np.delete(sims, i)
            ref_self_sims.append(float(np.sort(
                self.ref_embs_norm @ ref_emb)[::-1][:self.k].mean()))
        percentile = float(np.mean(np.array(ref_self_sims) < knn_mean_sim) * 100)

        return {
            "score":      round(score, 1),
            "grade":      _grade(score),
            "percentile": round(percentile, 1),
            "knn_sims":   [round(float(sims[i]), 4) for i in top_k_idx],
            "nearest":    [str(self.ref_paths[i]) if i < len(self.ref_paths) else ""
                           for i in top_k_idx],
        }

    def print_report(self, result: dict, audio_path: str = ""):
        print(f"\n{'='*50}")
        print(f"CLAP Quality Score: {result['score']:.1f}/100  [{result['grade']}]")
        print(f"Beats {result['percentile']:.0f}% of reference tracks")
        if audio_path:
            print(f"File: {audio_path}")
        print(f"\nNearest references (cosine sim):")
        for sim, path in zip(result["knn_sims"], result["nearest"]):
            name = Path(path).name if path else "?"
            print(f"  {sim:.4f}  {name}")
        print("="*50)


def main():
    p = argparse.ArgumentParser(description="CLAP quality scorer")
    p.add_argument("--build-ref", action="store_true",
                   help="Build reference embeddings from a folder of pro tracks")
    p.add_argument("--ref-dir", default="",
                   help="Directory of professional reference tracks (for --build-ref)")
    p.add_argument("--ref", default="ml/clap_ref.npz",
                   help="Reference embeddings .npz path")
    p.add_argument("--score", default="",
                   help="Audio file to score")
    p.add_argument("--out", default="ml/clap_ref.npz",
                   help="Output path for --build-ref")
    p.add_argument("--duration", type=float, default=60.0,
                   help="Seconds of audio to use for embedding")
    args = p.parse_args()

    if args.build_ref:
        if not args.ref_dir:
            print("--ref-dir required with --build-ref")
            sys.exit(1)
        build_reference(args.ref_dir, args.out, args.duration)

    elif args.score:
        scorer = CLAPScorer(args.ref, k=5)
        result = scorer.score(args.score, args.duration)
        scorer.print_report(result, args.score)

    else:
        p.print_help()


if __name__ == "__main__":
    main()
