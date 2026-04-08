"""
benchmark_fuser.py — Compare old (hill-climb) vs new (Optuna) optimizer.

Generates N synthetic audio pairs, runs each through fuse() once with
default params and once with the Optuna optimizer, and reports the
chart_score improvement and wall-clock speedup.

Usage:
  python benchmark_fuser.py                    # 3 pairs, 4 optimizer trials
  python benchmark_fuser.py --n-pairs 10 --trials 8
  python benchmark_fuser.py --skip-fuse        # score-only mode (fast, uses cached)
  python benchmark_fuser.py --audio-dir /path/to/wavs  # use real audio pairs
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
import tempfile
from pathlib import Path

import numpy as np
import soundfile as sf

sys.path.insert(0, str(Path(__file__).parent))

SR = 44100


# ---------------------------------------------------------------------------
# Synthetic audio generators (same as test_integration.py)
# ---------------------------------------------------------------------------

def _make_beat(duration_s: float = 30.0, bpm: float = 120.0, seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    t   = np.linspace(0, duration_s, int(SR * duration_s), endpoint=False)
    beat_s = 60.0 / bpm
    sig = np.zeros(len(t), dtype=np.float32)
    for kt in np.arange(0, duration_s, beat_s):
        idx = int(kt * SR)
        dur = int(0.15 * SR)
        n   = min(dur, len(t) - idx)
        env = np.exp(-np.linspace(0, 20, n))
        sub = np.sin(2 * np.pi * 60 * np.linspace(0, 0.15, n)) * env * 0.8
        sig[idx: idx + n] += sub.astype(np.float32)
    for st in np.arange(beat_s, duration_s, beat_s * 2):
        idx = int(st * SR)
        n   = min(int(0.08 * SR), len(t) - idx)
        snr = rng.standard_normal(n) * 0.4 * np.exp(-np.linspace(0, 15, n))
        sig[idx: idx + n] += snr.astype(np.float32)
    sig += (0.3 * np.sin(2 * np.pi * 80 * t)).astype(np.float32)
    sig  = sig / (np.max(np.abs(sig)) + 1e-9) * 0.85
    return np.stack([sig, sig], axis=1).astype(np.float32)


def _make_vocal(duration_s: float = 30.0, f0_hz: float = 180.0, seed: int = 1) -> np.ndarray:
    from scipy.signal import butter, sosfilt
    rng   = np.random.default_rng(seed)
    noise = rng.standard_normal(int(SR * duration_s)).astype(np.float64)
    sos1  = butter(4, [f0_hz * 0.85 / (SR / 2), f0_hz * 1.15 / (SR / 2)],
                   btype="band", output="sos")
    sos2  = butter(4, [1600 / (SR / 2), 2000 / (SR / 2)], btype="band", output="sos")
    vocal = (sosfilt(sos1, noise) + sosfilt(sos2, noise) * 0.6).astype(np.float32)
    vocal = vocal / (np.max(np.abs(vocal)) + 1e-9) * 0.7
    return np.stack([vocal, vocal], axis=1).astype(np.float32)


# ---------------------------------------------------------------------------
# Single-pair benchmark
# ---------------------------------------------------------------------------

def run_pair(
    beat_path: str,
    vocal_path: str,
    work_dir: Path,
    n_trials: int = 4,
    pair_idx: int = 0,
) -> dict:
    """Run one beat+vocal pair: baseline + Optuna. Return result dict."""
    import fuser as _fuser
    import listen
    from vf.optimizer import FuseOptimizer, PARAM_SPACE, PARAM_DEFAULTS

    stems = str(work_dir / "stems")
    os.makedirs(stems, exist_ok=True)

    results = {}

    # ── Baseline (default params) ──────────────────────────────────────────
    print(f"  [pair {pair_idx}] Baseline fuse…", flush=True)
    t0 = time.monotonic()
    _fuser._PARAM_OVERRIDE.clear()
    base_out = str(work_dir / f"pair{pair_idx}_baseline.wav")
    base_result = _fuser.fuse(
        song_a=beat_path,
        song_b=vocal_path,
        out_path=base_out,
        stems_cache=stems,
        direct_vocal=True,
        seed=42,
    )
    _fuser._PARAM_OVERRIDE.clear()
    base_radio = base_result.get("radio", base_out) if isinstance(base_result, dict) else base_out
    base_time  = time.monotonic() - t0

    base_cs, base_grade, _ = listen.score_any(base_radio)
    results["baseline"] = {
        "chart_score": int(round(base_cs)),
        "grade":       base_grade,
        "time_s":      round(base_time, 1),
        "path":        base_radio,
    }
    print(f"  [pair {pair_idx}] Baseline → {int(round(base_cs))}/100 ({base_grade}) "
          f"in {base_time:.1f}s", flush=True)

    # ── Optuna optimizer ───────────────────────────────────────────────────
    if n_trials <= 1:
        results["optuna"] = dict(results["baseline"])
        results["delta_score"] = 0
        results["speedup"]     = 1.0
        return results

    print(f"  [pair {pair_idx}] Optuna ({n_trials} trials)…", flush=True)
    t1 = time.monotonic()

    att_dir_root = work_dir / f"pair{pair_idx}_optuna"
    att_dir_root.mkdir(parents=True, exist_ok=True)

    attempt_results = []

    def objective_fn(params: dict, attempt_n: int) -> tuple[int, dict]:
        _fuser._PARAM_OVERRIDE.clear()
        _fuser._PARAM_OVERRIDE.update(params)
        att_out = str(att_dir_root / f"attempt_{attempt_n}.wav")
        try:
            res = _fuser.fuse(
                song_a=beat_path,
                song_b=vocal_path,
                out_path=att_out,
                stems_cache=stems,
                direct_vocal=True,
                seed=42 + attempt_n,
            )
            _fuser._PARAM_OVERRIDE.clear()
            radio = res.get("radio", att_out) if isinstance(res, dict) else att_out
        except Exception as e:
            _fuser._PARAM_OVERRIDE.clear()
            print(f"    fuse error: {e}", flush=True)
            return 0, {}
        cs, cg, det = listen.score_any(radio)
        attempt_results.append({"n": attempt_n, "chart_score": int(round(cs)), "grade": cg})
        print(f"    attempt {attempt_n}: {int(round(cs))}/100 ({cg})", flush=True)
        return int(round(cs)), det if isinstance(det, dict) else {}

    opt = FuseOptimizer(PARAM_SPACE, PARAM_DEFAULTS)
    best_params, best_score, opt_log = opt.optimize(
        objective_fn=objective_fn,
        n_trials=n_trials,
        target_score=85,
        seed=42 + pair_idx * 100,
    )
    optuna_time = time.monotonic() - t1

    results["optuna"] = {
        "chart_score":   best_score,
        "grade":         listen.score_any(  # fast re-score best only if we have it
            next(
                (a.get("path", "") for a in attempt_results
                 if a["chart_score"] == best_score), ""
            )
        )[1] if any(a["chart_score"] == best_score for a in attempt_results) else "?",
        "time_s":        round(optuna_time, 1),
        "best_params":   {k: round(v, 3) for k, v in best_params.items()},
        "attempts":      attempt_results,
    }
    results["delta_score"] = best_score - int(round(base_cs))
    results["speedup"]     = round(base_time / max(optuna_time / n_trials, 0.1), 2)

    print(f"  [pair {pair_idx}] Optuna best → {best_score}/100 "
          f"(Δ{results['delta_score']:+d}) in {optuna_time:.1f}s", flush=True)
    return results


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="VocalFusion optimizer benchmark")
    parser.add_argument("--n-pairs",    type=int, default=3,  help="Number of synthetic pairs")
    parser.add_argument("--trials",     type=int, default=4,  help="Optuna trials per pair")
    parser.add_argument("--audio-dir",  type=str, default="", help="Dir with beat_N.wav + vocal_N.wav")
    parser.add_argument("--duration",   type=float, default=30.0, help="Synthetic audio duration (s)")
    parser.add_argument("--output",     type=str, default="benchmark_results.json")
    args = parser.parse_args()

    try:
        import fuser  # noqa: F401
    except ImportError:
        print("ERROR: fuser.py not importable — install runtime dependencies first.")
        sys.exit(1)

    work_dir = Path(tempfile.mkdtemp(prefix="vf_bench_"))
    print(f"Working directory: {work_dir}")
    print(f"Running {args.n_pairs} pair(s) × {args.trials} Optuna trials each\n")

    all_results = []
    pairs = []

    if args.audio_dir:
        audio_dir = Path(args.audio_dir)
        for i in range(args.n_pairs):
            b = audio_dir / f"beat_{i}.wav"
            v = audio_dir / f"vocal_{i}.wav"
            if b.exists() and v.exists():
                pairs.append((str(b), str(v)))
            else:
                print(f"Skipping pair {i}: {b} or {v} not found")

    # Fill remaining pairs with synthetic audio
    for i in range(len(pairs), args.n_pairs):
        beat_path  = str(work_dir / f"beat_{i}.wav")
        vocal_path = str(work_dir / f"vocal_{i}.wav")
        sf.write(beat_path,  _make_beat(args.duration, seed=i * 7),     SR)
        sf.write(vocal_path, _make_vocal(args.duration, seed=i * 7 + 1), SR)
        pairs.append((beat_path, vocal_path))

    for idx, (beat_path, vocal_path) in enumerate(pairs):
        print(f"\n── Pair {idx + 1}/{len(pairs)} ─────────────────────────────")
        try:
            result = run_pair(beat_path, vocal_path, work_dir,
                              n_trials=args.trials, pair_idx=idx)
            all_results.append(result)
        except Exception as e:
            print(f"  ERROR: {e}")
            all_results.append({"error": str(e)})

    # ── Summary ──────────────────────────────────────────────────────────────
    valid = [r for r in all_results if "delta_score" in r]
    print(f"\n{'='*55}")
    print(f"  RESULTS ({len(valid)} pairs completed)")
    print(f"{'='*55}")
    if valid:
        base_scores   = [r["baseline"]["chart_score"] for r in valid]
        optuna_scores = [r["optuna"]["chart_score"]   for r in valid]
        deltas        = [r["delta_score"]              for r in valid]
        print(f"  Baseline avg   : {sum(base_scores)/len(base_scores):.1f}/100")
        print(f"  Optuna avg     : {sum(optuna_scores)/len(optuna_scores):.1f}/100")
        print(f"  Avg Δ score    : {sum(deltas)/len(deltas):+.1f}")
        print(f"  Pairs improved : {sum(1 for d in deltas if d > 0)}/{len(valid)}")
    print(f"{'='*55}\n")

    report = {
        "n_pairs":   len(pairs),
        "n_trials":  args.trials,
        "duration_s": args.duration,
        "pairs":     all_results,
        "summary": {
            "avg_baseline":   sum(r["baseline"]["chart_score"] for r in valid) / max(len(valid), 1),
            "avg_optuna":     sum(r["optuna"]["chart_score"]   for r in valid) / max(len(valid), 1),
            "avg_delta":      sum(r["delta_score"]              for r in valid) / max(len(valid), 1),
        } if valid else {},
    }
    Path(args.output).write_text(json.dumps(report, indent=2, default=str))
    print(f"Full report saved to {args.output}")


if __name__ == "__main__":
    main()
