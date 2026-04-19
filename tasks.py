"""
VocalFusion Celery Tasks
========================
fuse_task   — single fuse + score
refine_task — up to 5 attempts, picks best chart_score
"""
import json
import os
import threading
from pathlib import Path

from celery_app import celery_app

JOBS_DIR   = Path("vf_data/jobs")
MIXES_DIR  = Path("vf_data/mixes")
STEMS_DIR  = Path("vf_data/stems")

for _d in (JOBS_DIR, MIXES_DIR, STEMS_DIR):
    _d.mkdir(parents=True, exist_ok=True)

_write_lock = threading.Lock()


# ---------------------------------------------------------------------------
# Parameter candidates for refinement loop
# ---------------------------------------------------------------------------

REFINE_CANDIDATES = [
    {},                                                                         # 1. baseline
    {"vocal_level_mult": 1.1, "carve_db": 3.5},                                # 2. more presence
    {"vocal_level_mult": 0.9, "carve_db": 2.5, "reverb_wet": 0.12},            # 3. spacious
    {"sidechain_depth": 0.55, "carve_db": 4.0},                                # 4. punchy sidechain
    {"vocal_level_mult": 1.05, "carve_db": 3.0, "sidechain_depth": 0.45},      # 5. balanced
]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _write_state(job_id: str, state: dict) -> None:
    path = JOBS_DIR / f"{job_id}.json"
    with _write_lock:
        path.write_text(json.dumps(state))


def _score_output(path: str) -> dict:
    """Score a WAV and return a metadata dict. Gracefully returns zeros on error."""
    result = {
        "chart_score": 0, "chart_grade": "?",
        "mix_score": 0,   "mix_grade":   "?",
        "metadata": {},
    }
    try:
        from listen import score_file as _sf, score_any as _sa, _grade as _grade
        sc, _, met = _sf(path, print_report=False)
        result["mix_score"] = int(sc)
        result["mix_grade"] = _grade(int(sc))
        cs, cg, cdet = _sa(path)
        result["chart_score"] = int(round(cs))
        result["chart_grade"] = cg
        result["metadata"] = {
            "chart_details":    cdet,
            "sibilance_ratio":  float(met.get("sibilance_ratio", 0.25)),
            "delta_sub":        float(met.get("delta_sub", 0)),
            "delta_bass":       float(met.get("delta_bass", 0)),
            "delta_lo_mid":     float(met.get("delta_lo_mid", 0)),
            "delta_mid":        float(met.get("delta_mid", 0)),
            "delta_hi_mid":     float(met.get("delta_hi_mid", 0)),
            "delta_presence":   float(met.get("delta_presence", 0)),
            "delta_air":        float(met.get("delta_air", 0)),
            "lufs":             float(met.get("lufs_int", -99)),
            "dynamic_range":    float(met.get("dynamic_range_db", 0)),
        }
    except Exception as e:
        print(f"[tasks] scoring error: {e}", flush=True)
    return result


def _run_one(job_id: str, vocal_path: str, inst_path: str, seed: int,
             direct_vocal: bool, attempt: int, overrides: dict) -> dict:
    """Run one fuse attempt, returns score dict including out_path."""
    out_path = str(MIXES_DIR / f"{job_id}_a{attempt}_fusion.wav")

    import fuser as _fuser
    _fuser._PARAM_OVERRIDE.clear()
    if overrides:
        _fuser._PARAM_OVERRIDE.update(overrides)

    _fuser.fuse(
        song_a=inst_path,
        song_b=vocal_path,
        out_path=out_path,
        stems_cache=str(STEMS_DIR),
        direct_vocal=direct_vocal,
        seed=seed + attempt,
    )
    _fuser._PARAM_OVERRIDE.clear()

    result = _score_output(out_path) if Path(out_path).exists() else {
        "chart_score": 0, "chart_grade": "?",
        "mix_score": 0,   "mix_grade":   "?",
        "metadata": {},
    }
    result["out_path"] = out_path
    result["attempt"]  = attempt
    result["overrides"] = overrides
    return result


# ---------------------------------------------------------------------------
# Tasks
# ---------------------------------------------------------------------------

@celery_app.task(bind=True, name="tasks.fuse_task")
def fuse_task(self, job_id: str, vocal_path: str, inst_path: str,
              seed: int, direct_vocal: bool, name_a: str, name_b: str) -> dict:
    state: dict = {
        "status": "running", "progress": 5,
        "message": "Starting fusion…",
        "name_a": name_a, "name_b": name_b, "seed": seed, "steps": [],
    }
    _write_state(job_id, state)

    try:
        result = _run_one(job_id, vocal_path, inst_path, seed, direct_vocal, 0, {})

        canonical = str(MIXES_DIR / f"{job_id}_fusion.wav")
        if Path(result["out_path"]).exists():
            Path(result["out_path"]).rename(canonical)

        state.update({
            "status":      "done",
            "progress":    100,
            "message":     f"Chart: {result['chart_score']} · {result['chart_grade']}",
            "output_url":  f"/download/{job_id}",
            "score":       result["chart_score"],
            "chart_score": result["chart_score"],
            "chart_grade": result["chart_grade"],
            "mix_score":   result["mix_score"],
            "mix_grade":   result["mix_grade"],
            "metadata":    result["metadata"],
        })
        _write_state(job_id, state)

        for p in (vocal_path, inst_path):
            try: os.remove(p)
            except OSError: pass

        return state

    except Exception as e:
        state.update({"status": "error", "message": str(e), "progress": 0})
        _write_state(job_id, state)
        raise


@celery_app.task(bind=True, name="tasks.refine_task")
def refine_task(self, job_id: str, vocal_path: str, inst_path: str,
                seed: int, direct_vocal: bool, name_a: str, name_b: str,
                target_score: int = 85, max_attempts: int = 5) -> dict:
    candidates = REFINE_CANDIDATES[:max_attempts]
    state: dict = {
        "status": "running", "progress": 0,
        "message": f"Refinement loop (up to {len(candidates)} attempts)…",
        "name_a": name_a, "name_b": name_b, "seed": seed,
        "steps": [], "attempts": [],
    }
    _write_state(job_id, state)

    best: dict | None = None

    for i, overrides in enumerate(candidates):
        pct = int(i / len(candidates) * 85)
        state["progress"] = max(5, pct)
        state["message"]  = f"Attempt {i+1}/{len(candidates)}…"
        _write_state(job_id, state)

        result = _run_one(job_id, vocal_path, inst_path, seed, direct_vocal, i, overrides)

        state["attempts"].append({
            "attempt":     i + 1,
            "chart_score": result["chart_score"],
            "chart_grade": result["chart_grade"],
            "overrides":   overrides,
        })
        _write_state(job_id, state)

        if best is None or result["chart_score"] > best["chart_score"]:
            best = result

        if result["chart_score"] >= target_score:
            state["message"] = f"Target {target_score} reached on attempt {i+1}!"
            _write_state(job_id, state)
            break

    # Move best output to canonical path
    canonical = str(MIXES_DIR / f"{job_id}_fusion.wav")
    if best and Path(best["out_path"]).exists():
        Path(best["out_path"]).rename(canonical)

    # Clean up other attempt files
    for i in range(len(candidates)):
        p = MIXES_DIR / f"{job_id}_a{i}_fusion.wav"
        if p.exists():
            p.unlink()

    n_att = len(state["attempts"])
    state.update({
        "status":      "done",
        "progress":    100,
        "message":     f"Best: {best['chart_score']} · {best['chart_grade']} (attempt {best['attempt']+1}/{n_att})",
        "output_url":  f"/download/{job_id}",
        "score":       best["chart_score"],
        "chart_score": best["chart_score"],
        "chart_grade": best["chart_grade"],
        "mix_score":   best["mix_score"],
        "mix_grade":   best["mix_grade"],
        "metadata":    best["metadata"],
    })
    _write_state(job_id, state)

    for p in (vocal_path, inst_path):
        try: os.remove(p)
        except OSError: pass

    return state
