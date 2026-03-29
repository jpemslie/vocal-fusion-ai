"""
ml/ref_model.py — Build and save the reference distribution model.

Scans a folder of reference tracks (e.g. reference_tracks/drake/),
extracts features from each, fits:
  1. StandardScaler (zero-mean, unit-variance)
  2. PCA (keeps 95% variance, max 20 components)
  3. Mahalanobis covariance (to detect outlier mixes)

Saves to ml/ref_model.npz.

Usage:
    python -m ml.ref_model [ref_folder] [output.npz]
    # default: reference_tracks/drake/ → ml/ref_model.npz
"""

import os
import sys
import numpy as np
from pathlib import Path
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.covariance import EmpiricalCovariance

sys.path.insert(0, str(Path(__file__).parent.parent))
from ml.features import extract, FEATURE_NAMES, N_FEATURES

AUDIO_EXTS = {".mp3", ".wav", ".flac", ".m4a"}
DEFAULT_REF = Path(__file__).parent.parent / "reference_tracks" / "drake"
DEFAULT_OUT = Path(__file__).parent / "ref_model.npz"


def build(ref_folder: str | Path, out_path: str | Path = DEFAULT_OUT,
          duration: float = 60.0) -> dict:
    ref_folder = Path(ref_folder)
    tracks = [f for f in sorted(ref_folder.iterdir()) if f.suffix.lower() in AUDIO_EXTS]
    if not tracks:
        raise FileNotFoundError(f"No audio files found in {ref_folder}")

    print(f"Extracting features from {len(tracks)} reference tracks…", flush=True)
    vecs = []
    failed = []
    for t in tracks:
        try:
            v = extract(str(t), duration=duration)
            vecs.append(v)
            print(f"  ✓ {t.name}", flush=True)
        except Exception as e:
            print(f"  ✗ {t.name}: {e}", flush=True)
            failed.append(str(t))

    if len(vecs) < 5:
        raise RuntimeError(f"Only {len(vecs)} tracks succeeded — need at least 5")

    X = np.stack(vecs, axis=0)   # (n_tracks, N_FEATURES)

    # Fit scaler
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # PCA
    pca = PCA(n_components=min(20, len(vecs) - 1), whiten=False)
    X_pca = pca.fit_transform(X_scaled)
    n_kept = int(np.searchsorted(np.cumsum(pca.explained_variance_ratio_), 0.95)) + 1
    n_kept = max(n_kept, 3)
    print(f"\nPCA: keeping {n_kept} components ({pca.explained_variance_ratio_[:n_kept].sum()*100:.1f}% variance)")

    # Covariance in PCA space (for Mahalanobis)
    X_pca_kept = X_pca[:, :n_kept]
    cov = EmpiricalCovariance()
    cov.fit(X_pca_kept)

    # Save
    np.savez(
        out_path,
        # Scaler
        scaler_mean=scaler.mean_,
        scaler_scale=scaler.scale_,
        # PCA
        pca_components=pca.components_[:n_kept],   # (n_kept, N_FEATURES)
        pca_mean=pca.mean_,
        n_kept=np.array(n_kept),
        # Covariance
        cov_precision=cov.precision_,
        cov_location=cov.location_,
        # Reference distribution in PCA space (for percentile scoring)
        ref_pca=X_pca_kept,       # (n_tracks, n_kept)
        ref_raw=X,                # (n_tracks, N_FEATURES) — raw for delta reporting
        feature_names=np.array(FEATURE_NAMES),
        n_tracks=np.array(len(vecs)),
    )
    print(f"Model saved → {out_path}  ({len(vecs)} tracks, {n_kept} PCA dims)")
    return {"n_tracks": len(vecs), "n_pca": n_kept, "failed": failed}


if __name__ == "__main__":
    ref = sys.argv[1] if len(sys.argv) > 1 else DEFAULT_REF
    out = sys.argv[2] if len(sys.argv) > 2 else DEFAULT_OUT
    build(ref, out)
