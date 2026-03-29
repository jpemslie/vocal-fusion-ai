"""
ml/scorer.py — Score a mix against the reference manifold.

Loads the model built by ref_model.py and scores any WAV/MP3.

Returns a dict:
    score          0-100  (100 = indistinguishable from reference)
    grade          S/A/B/C/D
    mahal_dist     raw Mahalanobis distance (lower = closer to reference)
    percentile     what % of reference tracks are "worse" than this mix
    deltas         {feature_name: delta_vs_ref_median, ...}  (positive = above ref)
    issues         list of (severity, feature_name, description) sorted worst first

Usage:
    python -m ml.scorer path/to/mix.wav
"""

import sys
import numpy as np
from pathlib import Path
from scipy.spatial.distance import mahalanobis

sys.path.insert(0, str(Path(__file__).parent.parent))
from ml.features import extract, FEATURE_NAMES, N_FEATURES

MODEL_PATH = Path(__file__).parent / "ref_model.npz"

_LOADED: dict | None = None


def _load_model(model_path: str | Path = MODEL_PATH) -> dict:
    global _LOADED
    if _LOADED is None or str(_LOADED.get("_path")) != str(model_path):
        d = np.load(model_path, allow_pickle=False)
        _LOADED = {k: d[k] for k in d.files}
        _LOADED["_path"] = str(model_path)
        # Reconstruct feature_names list
        _LOADED["feature_names_list"] = list(_LOADED["feature_names"].astype(str))
    return _LOADED


def _project(x: np.ndarray, m: dict) -> np.ndarray:
    """Scale → PCA project a single feature vector."""
    x_scaled = (x - m["scaler_mean"]) / (m["scaler_scale"] + 1e-12)
    n_kept = int(m["n_kept"])
    comps = m["pca_components"][:n_kept]       # (n_kept, N_FEATURES)
    x_pca = (x_scaled - m["pca_mean"]) @ comps.T
    return x_pca   # (n_kept,)


# Feature-specific descriptions for human-readable issue reporting
_FEAT_DESCRIPTIONS = {
    "lufs_i":            ("too loud", "too quiet"),
    "lra":               ("dynamics too wide", "too compressed / no dynamics"),
    "crest_db":          ("transients too spiky", "over-compressed / no punch"),
    "plr":               ("peak-to-loudness gap too large", "clipping risk"),
    "band_sub":          ("too much sub-bass (boomy)", "sub-bass too thin"),
    "band_bass":         ("bass too heavy (muddy)", "bass too thin"),
    "band_lowmid":       ("low-mids too thick (boxy/warm)", "low-mids too scooped"),
    "band_mid":          ("mids too prominent (nasal)", "mids too recessed (hollow)"),
    "band_highmid":      ("high-mids too harsh / present", "high-mids too dull"),
    "band_presence":     ("presence/bite too harsh", "presence too dull"),
    "band_air":          ("too much air / brightness", "air too dull"),
    "spectral_centroid": ("mix too bright overall", "mix too dark overall"),
    "spectral_rolloff":  ("high-frequency content too dominant", "high-end rolled off too early"),
    "spectral_flatness": ("too noise-like / diffuse", "too tonal / lacks texture"),
    "spectral_slope":    ("slope too flat (harsh, bright)", "slope too steep (muffled, dull)"),
    "vocal_presence":    ("vocals too loud / overpowering", "VOCALS TOO QUIET / BURIED"),
    "vocal_clarity":     ("vocals too processed / smeared", "vocals lack harmonic clarity"),
    "hnr_db":            ("too much noise in vocal zone", "vocals too dry / no warmth"),
    "stereo_width_low":  ("bass too wide (mono problem)", "bass too narrow"),
    "stereo_width_mid":  ("mid stereo too wide (phasey)", "mids too narrow / mono"),
    "stereo_width_high": ("highs too wide (diffuse)", "highs too narrow"),
    "ms_balance":        ("too much side signal (phasey)", "too mono"),
    "transient_clarity": ("transients too aggressive", "transients too smashed"),
    "dynamic_arc":       ("energy arc too dramatic", "flat energy / no movement"),
    "onset_density":     ("too many onsets (cluttered)", "too sparse / lacks energy"),
    "compression_depth": ("dynamics too wide (unmastered feel)", "over-compressed (lifeless)"),
    "chroma_entropy":    ("harmonically complex (key unclear)", "too tonally static"),
    "roughness":         ("too much spectral roughness (harsh)", "too smooth (lacks texture)"),
    "bass_mid_ratio":    ("bass heavy vs mids", "bass thin vs mids"),
    "sub_bass_ratio":    ("sub-heavy vs bass", "sub-thin vs bass"),
    "highmid_mid_ratio": ("high-mids dominant vs mids", "high-mids too recessed vs mids"),
    "air_presence_ratio":("too much air vs presence", "air lacking vs presence"),
}


def score(path: str, model_path: str | Path = MODEL_PATH,
          duration: float = 60.0) -> dict:
    """
    Score a mix file. Returns full diagnostic dict.
    """
    m = _load_model(model_path)
    feat_names = m["feature_names_list"]

    # Extract features
    x = extract(str(path), duration=duration)

    # Project to PCA space
    x_pca = _project(x, m)

    # Mahalanobis distance
    n_kept = int(m["n_kept"])
    x_pca_k = x_pca[:n_kept]
    try:
        dist = mahalanobis(x_pca_k, m["cov_location"], m["cov_precision"])
    except Exception:
        dist = float(np.linalg.norm(x_pca_k - m["cov_location"]))

    # Percentile: what % of reference tracks have higher distance (i.e., are "worse")
    ref_pca = m["ref_pca"]  # (n_tracks, n_kept)
    ref_dists = np.array([
        mahalanobis(r, m["cov_location"], m["cov_precision"])
        for r in ref_pca
    ])
    pct = float(np.mean(ref_dists > dist) * 100)   # % of refs we beat

    # Score: 100 - clipped distance mapped to 0-100
    # dist=0 (identical to reference centroid) → 100
    # dist=3 (edge of 95% ellipsoid) → 70
    # dist=8 (far outlier) → 0
    score_val = float(np.clip(100 - (dist / 0.09), 0, 100))
    if   score_val >= 85: grade = "A"
    elif score_val >= 70: grade = "B"
    elif score_val >= 55: grade = "C"
    elif score_val >= 35: grade = "D"
    else:                 grade = "F"

    # Per-feature deltas vs reference median
    ref_raw = m["ref_raw"]  # (n_tracks, N_FEATURES)
    ref_medians = np.median(ref_raw, axis=0)
    ref_stds    = np.std(ref_raw, axis=0) + 1e-12
    deltas = {}   # raw difference
    z_scores = {} # z-score (how many std devs from reference)
    for i, name in enumerate(feat_names):
        deltas[name] = float(x[i] - ref_medians[i])
        z_scores[name] = float((x[i] - ref_medians[i]) / ref_stds[i])

    # Build issues list: features where |z| > 1.5 are flagged
    issues = []
    for name, z in z_scores.items():
        if abs(z) < 1.2:
            continue
        desc_pos, desc_neg = _FEAT_DESCRIPTIONS.get(name, ("above reference", "below reference"))
        description = desc_pos if z > 0 else desc_neg
        severity = "HIGH" if abs(z) > 2.5 else "MED" if abs(z) > 1.8 else "LOW"
        issues.append({
            "severity": severity,
            "feature": name,
            "z": round(z, 2),
            "delta": round(deltas[name], 4),
            "ref_median": round(float(ref_medians[feat_names.index(name)]), 4),
            "mix_value":  round(float(x[feat_names.index(name)]), 4),
            "description": description,
        })

    # Sort by |z| descending
    issues.sort(key=lambda d: -abs(d["z"]))

    return {
        "score":       round(score_val, 1),
        "grade":       grade,
        "mahal_dist":  round(dist, 3),
        "percentile":  round(pct, 1),
        "deltas":      deltas,
        "z_scores":    z_scores,
        "issues":      issues,
        "features":    {k: float(v) for k, v in zip(feat_names, x)},
    }


def print_report(result: dict) -> None:
    print(f"\n{'━'*60}")
    print(f"  SCORE  {result['score']:.0f}/100  [{result['grade']}]"
          f"   beats {result['percentile']:.0f}% of reference tracks")
    print(f"  Mahalanobis distance from reference: {result['mahal_dist']:.2f}")
    print(f"{'━'*60}")

    issues = result["issues"]
    if not issues:
        print("  ✓ No significant deviations from reference profile.")
    else:
        print(f"  {'SEV':<5} {'FEATURE':<25} {'Z':>6}  DESCRIPTION")
        print(f"  {'---':<5} {'-------':<25} {'---':>6}  -----------")
        for iss in issues[:12]:
            sev_sym = "🔴" if iss["severity"] == "HIGH" else "🟡" if iss["severity"] == "MED" else "🔵"
            print(f"  {sev_sym} {iss['feature']:<24} {iss['z']:>+6.1f}  {iss['description']}")

    print(f"\n  Key measurements vs reference medians:")
    key_feats = ["lufs_i", "lra", "vocal_presence", "band_bass", "band_highmid",
                 "stereo_width_mid", "crest_db", "hnr_db"]
    m = _load_model()
    ref_raw = m["ref_raw"]
    feat_names = m["feature_names_list"]
    ref_medians = np.median(ref_raw, axis=0)
    for fn in key_feats:
        if fn not in result["features"]:
            continue
        mix_val = result["features"][fn]
        idx = feat_names.index(fn)
        ref_med = float(ref_medians[idx])
        diff = mix_val - ref_med
        arrow = "▲" if diff > 0 else "▼"
        print(f"    {fn:<25} mix={mix_val:+.3f}  ref={ref_med:+.3f}  {arrow}{abs(diff):.3f}")
    print()


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python -m ml.scorer <mix.wav> [model.npz]")
        sys.exit(1)
    path = sys.argv[1]
    mp = sys.argv[2] if len(sys.argv) > 2 else MODEL_PATH
    result = score(path, mp)
    print_report(result)
    print(f"Full score: {result['score']}/100 [{result['grade']}]")
