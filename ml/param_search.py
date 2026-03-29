"""
ml/param_search.py — ML-guided parameter search for fuser.

Tries multiple combinations of mixing parameters, scores each mix against
the reference manifold, and returns the parameters that produce the highest
quality score.

Parameters searched:
    vocal_level_mult   (how loud to push the vocal)
    carve_db           (how deep to carve the beat in the vocal zone)
    reverb_wet         (how much reverb on the vocal)
    auto_tune_strength (how robotic/corrected the pitch is, 0=off)
    sidechain_depth    (how much the beat ducks under the vocal)

Usage (standalone):
    python -m ml.param_search beat.mp3 vocal.m4a output.wav

Integration:
    from ml.param_search import smart_fuse
    result = smart_fuse("beat.mp3", "vocal.m4a", "out.wav", direct_vocal=True)
"""

import os
import sys
import time
import tempfile
import itertools
import numpy as np
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))


# ── Parameter grids ──────────────────────────────────────────────────────────
# Coarse grid first (fast), then refine around the best.
# Each value is a multiplier or absolute value injected into fuser via style overrides.

COARSE_GRID = {
    "vocal_level_mult":      [1.5, 2.5, 3.5],
    "carve_db":              [5.0, 10.0, 15.0],
    "at_strength":           [0.0, 0.15],
    "reverb_wet":            [0.02, 0.08],
    "sidechain_depth":       [0.25, 0.40],
    "noise_reduce_strength": [0.05, 0.15],
    "skip_groove_quantize":  [True, False],
    "harmony_wet":           [0.0, 0.07],
}

FINE_GRID_STEPS = 2   # ±steps around best coarse value


def _param_combinations(grid: dict) -> list[dict]:
    keys = list(grid.keys())
    vals = list(grid.values())
    return [dict(zip(keys, combo)) for combo in itertools.product(*vals)]


def _score_file(path: str) -> float:
    """Return ML quality score (0-100). Returns 0 on failure."""
    try:
        from ml.scorer import score as ml_score
        MODEL = Path(__file__).parent / "ref_model.npz"
        if not MODEL.exists():
            return 0.0
        result = ml_score(path, MODEL, duration=30.0)
        return float(result["score"])
    except Exception as e:
        print(f"    [scorer failed: {e}]", flush=True)
        return 0.0


_STEMS_CACHE = str(Path(__file__).parent.parent / "vf_data" / "stems")


def _fuse_with_params(beat: str, vocal: str, out: str,
                      params: dict, direct_vocal: bool = True) -> bool:
    """
    Run fuser.fuse() with patched style parameters.
    Returns True if the output file was created.
    Uses stems_cache so Song A is only separated once across all candidates.
    """
    try:
        import fuser as _fuser

        # Inject our parameter overrides as a global override dict
        _fuser._PARAM_OVERRIDE = {
            "vocal_level_mult":      params.get("vocal_level_mult", 2.5),
            "at_strength_cap":       params.get("at_strength", 0.25),
            "reverb_wet":            params.get("reverb_wet", 0.06),
            "carve_db":              params.get("carve_db", 10.0),
            "sidechain_depth":       params.get("sidechain_depth", 0.35),
            "noise_reduce_strength": params.get("noise_reduce_strength", 0.15),
            "skip_groove_quantize":  params.get("skip_groove_quantize", False),
            "harmony_wet":           params.get("harmony_wet", 0.14),
        }

        # Pass stems_cache so Song A separation is reused across candidates
        _fuser.fuse(beat, vocal, out, direct_vocal=direct_vocal,
                    stems_cache=_STEMS_CACHE)

        # Clean up override
        _fuser._PARAM_OVERRIDE = {}

        return os.path.exists(out)
    except Exception as e:
        print(f"    [fuse failed: {e}]", flush=True)
        _fuser._PARAM_OVERRIDE = {}
        return False


def smart_fuse(beat: str, vocal: str, out: str,
               direct_vocal: bool = True,
               n_coarse: int = 12,
               verbose: bool = True) -> dict:
    """
    ML-guided parameter search.

    1. Runs up to `n_coarse` coarse-grid combinations.
    2. Scores each with the reference manifold model.
    3. Picks the best, writes it to `out`.
    4. Returns diagnostic info.

    Falls back to default fuse() if the ML model isn't available.
    """
    MODEL = Path(__file__).parent / "ref_model.npz"
    if not MODEL.exists():
        print("[smart_fuse] No reference model found — run ml/ref_model.py first.")
        print("[smart_fuse] Falling back to default fuse()…")
        import fuser
        fuser.fuse(beat, vocal, out, direct_vocal=direct_vocal)
        return {"score": 0, "params": {}, "fallback": True}

    combos = _param_combinations(COARSE_GRID)
    np.random.shuffle(combos)   # randomise order so early termination is unbiased
    combos = combos[:n_coarse]

    if verbose:
        print(f"\n[smart_fuse] ML parameter search: {len(combos)} candidate mixes", flush=True)
        print(f"  Beat:  {Path(beat).name}")
        print(f"  Vocal: {Path(vocal).name}")

    results = []
    tmpdir = tempfile.mkdtemp(prefix="vf_search_")

    for i, params in enumerate(combos):
        tmp_out = os.path.join(tmpdir, f"candidate_{i:03d}.wav")
        t0 = time.time()
        if verbose:
            param_str = "  ".join(f"{k}={v:.2f}" for k, v in params.items())
            print(f"\n  [{i+1}/{len(combos)}] {param_str}", flush=True)

        ok = _fuse_with_params(beat, vocal, tmp_out, params, direct_vocal)
        if not ok:
            continue

        sc = _score_file(tmp_out)
        elapsed = time.time() - t0
        if verbose:
            print(f"    → score {sc:.1f}/100  ({elapsed:.1f}s)", flush=True)
        results.append((sc, params, tmp_out))

    if not results:
        print("[smart_fuse] All candidates failed — falling back to default fuse().")
        import fuser
        fuser.fuse(beat, vocal, out, direct_vocal=direct_vocal)
        return {"score": 0, "params": {}, "fallback": True}

    # Pick best
    results.sort(key=lambda x: -x[0])
    best_score, best_params, best_tmp = results[0]

    if verbose:
        print(f"\n[smart_fuse] Best: score={best_score:.1f}/100")
        for k, v in best_params.items():
            print(f"    {k}: {v}")

    # Copy best to final output
    import shutil
    shutil.copy2(best_tmp, out)

    # Clean up
    shutil.rmtree(tmpdir, ignore_errors=True)

    # Full diagnostic score on final output
    from ml.scorer import score as ml_score, print_report
    final = ml_score(out, MODEL, duration=30.0)
    if verbose:
        print_report(final)

    return {
        "score":        best_score,
        "grade":        final["grade"],
        "params":       best_params,
        "issues":       final["issues"][:6],
        "all_results":  [(s, p) for s, p, _ in results],
    }


if __name__ == "__main__":
    if len(sys.argv) < 4:
        print("Usage: python -m ml.param_search beat.mp3 vocal.m4a output.wav")
        sys.exit(1)
    result = smart_fuse(sys.argv[1], sys.argv[2], sys.argv[3])
    print(f"\nFinal score: {result['score']:.1f}/100 [{result.get('grade','?')}]")
