"""
VocalFusion FastAPI Production Server  v3
==========================================
POST /fuse              multipart: song_a, song_b, [seed], [direct_vocal], [pro_mode]
GET  /status/{id}       full job state (polls every ~1.5s from UI)
GET  /download/{id}     radio mix WAV
GET  /download/{id}/{v} club / intimate variant
GET  /job/{id}/attempts list all attempts with params + scores
GET  /health

Professional mode: runs quality-driven parameter optimisation (up to 8 attempts,
stops when chart_score >= 85).  Each attempt is saved to vf_data/mixes/{job_id}/attempt_N/
"""

import json
import logging
import math
import os
import random as _pyrand
import shutil
import threading
import time
import uuid
from pathlib import Path

import numpy as np
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
STEMS_DIR  = Path("vf_data/stems")
OUTPUT_DIR = Path("vf_data/mixes")
UPLOAD_DIR = Path("vf_data/uploads")
LOGS_DIR   = Path("vf_data/logs")

for _d in (STEMS_DIR, OUTPUT_DIR, UPLOAD_DIR, LOGS_DIR):
    _d.mkdir(parents=True, exist_ok=True)

# ---------------------------------------------------------------------------
# App
# ---------------------------------------------------------------------------
app = FastAPI(title="VocalFusion API", version="3.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], allow_methods=["*"],
    allow_headers=["*"], expose_headers=["*"],
)

# ---------------------------------------------------------------------------
# Job store + serialization
# ---------------------------------------------------------------------------
_jobs: dict[str, dict] = {}
_lock  = threading.Lock()
_fuse_sem = threading.Semaphore(1)   # one GPU job at a time


def _set(job_id: str, **kw) -> None:
    with _lock:
        if job_id in _jobs:
            _jobs[job_id].update(kw)


def _get(job_id: str) -> dict | None:
    with _lock:
        j = _jobs.get(job_id)
        return dict(j) if j else None


# ---------------------------------------------------------------------------
# Parameter space — sourced from vf.optimizer (single source of truth)
# ---------------------------------------------------------------------------
from vf.optimizer import (
    PARAM_SPACE,
    PARAM_DEFAULTS,
    FuseOptimizer as _FuseOptimizer,
    compute_reward as _compute_reward,
)
from vf.genre   import detect_genre   as _detect_genre
from vf.presets import get_preset     as _get_preset

TARGET_SCORE  = int(os.environ.get("VF_TARGET_SCORE",  "85"))
MAX_ATTEMPTS  = int(os.environ.get("VF_MAX_ATTEMPTS",  "8"))
PRO_THRESHOLD = int(os.environ.get("VF_PRO_THRESHOLD", "72"))

# Optimizer singleton (shared across requests — stateless between calls)
_optimizer = _FuseOptimizer(PARAM_SPACE, PARAM_DEFAULTS)


# kept for any code that still references _clip_params
def _clip_params(p: dict[str, float]) -> dict[str, float]:
    return {k: float(np.clip(v, *PARAM_SPACE[k])) for k, v in p.items()}



# ---------------------------------------------------------------------------
# Scoring helper
# ---------------------------------------------------------------------------
def _score_wav(path: str) -> tuple[int, str, int, str, dict]:
    """Returns (chart_score, chart_grade, mix_score, mix_grade, metadata)."""
    chart_score, chart_grade = 0, "?"
    mix_score,   mix_grade   = 0, "?"
    metadata: dict = {}
    try:
        from listen import score_file as _sf, score_any as _sa, _grade as _grade
        sc, _, met = _sf(path, print_report=False)
        mix_score  = int(sc)
        mix_grade  = _grade(mix_score)
        cs, cg, cdet = _sa(path)
        chart_score = int(round(cs))
        chart_grade = cg
        metadata = {
            "chart_details":   cdet,
            "sibilance_ratio": float(met.get("sibilance_ratio", 0.25)),
            "delta_sub":       float(met.get("delta_sub", 0)),
            "delta_bass":      float(met.get("delta_bass", 0)),
            "delta_lo_mid":    float(met.get("delta_lo_mid", 0)),
            "delta_mid":       float(met.get("delta_mid", 0)),
            "delta_hi_mid":    float(met.get("delta_hi_mid", 0)),
            "delta_presence":  float(met.get("delta_presence", 0)),
            "delta_air":       float(met.get("delta_air", 0)),
            "lufs":            float(met.get("lufs_int", -99)),
            "dynamic_range":   float(met.get("dynamic_range_db", 0)),
            # Expose per-dimension subscores for multi-objective optimizer
            "spectral":        float(cdet.get("spectral",   50) if isinstance(cdet, dict) else 50),
            "loudness":        float(cdet.get("loudness",   50) if isinstance(cdet, dict) else 50),
            "dynamics":        float(cdet.get("dynamics",   50) if isinstance(cdet, dict) else 50),
            "perceptual":      float(cdet.get("perceptual", 50) if isinstance(cdet, dict) else 50),
            "musical":         float(cdet.get("musical",    50) if isinstance(cdet, dict) else 50),
        }
    except Exception as e:
        print(f"[api] scoring error (non-fatal): {e}", flush=True)
    return chart_score, chart_grade, mix_score, mix_grade, metadata


# ---------------------------------------------------------------------------
# Background worker
# ---------------------------------------------------------------------------
def _worker(
    job_id: str,
    vocal_path: str,
    inst_path:  str,
    seed:       int,
    direct_vocal: bool,
    name_a:     str,
    name_b:     str,
    pro_mode:   bool,
) -> None:
    job_dir = OUTPUT_DIR / job_id
    job_dir.mkdir(parents=True, exist_ok=True)

    # Per-job log file
    log_path = LOGS_DIR / f"fuse_{job_id}.log"
    fh = logging.FileHandler(log_path)
    fh.setFormatter(logging.Formatter("%(asctime)s %(message)s"))
    log = logging.getLogger(f"vf.{job_id}")
    log.setLevel(logging.DEBUG)
    log.addHandler(fh)
    log.info(f"job={job_id} seed={seed} pro_mode={pro_mode} "
             f"vocal={Path(vocal_path).name} inst={Path(inst_path).name}")

    _set(job_id, progress=2, message="Waiting for GPU…")

    with _fuse_sem:
        # ── core fuse helper ──────────────────────────────────────────────
        def _run_fuse(params: dict, attempt_n: int, attempt_seed: int) -> tuple[dict | None, str | None]:
            """Run one fuse with given params. Returns (result_dict, radio_path) or (None, None)."""
            att_dir = job_dir / f"attempt_{attempt_n}"
            att_dir.mkdir(parents=True, exist_ok=True)
            out_path = str(att_dir / "mix.wav")

            try:
                import fuser as _fuser
                _fuser._PARAM_OVERRIDE.clear()
                _fuser._PARAM_OVERRIDE.update(params)

                def on_prog(step, total, msg):
                    base_pct = int((attempt_n - 1) / max(1, _total_attempts) * 75)
                    step_pct = int(step / max(1, total) * (75 / max(1, _total_attempts)))
                    _set(job_id, progress=max(5, base_pct + step_pct), message=f"[{attempt_n}] {msg}")

                result = _fuser.fuse(
                    song_a=inst_path,
                    song_b=vocal_path,
                    out_path=out_path,
                    stems_cache=str(STEMS_DIR),
                    direct_vocal=direct_vocal,
                    seed=attempt_seed,
                    progress_cb=on_prog,
                )
                _fuser._PARAM_OVERRIDE.clear()

                radio = result.get("radio", out_path) if isinstance(result, dict) else (result or out_path)
                return result if isinstance(result, dict) else {"radio": radio}, radio

            except Exception as e:
                log.error(f"attempt {attempt_n} fuse error: {e}")
                try: import fuser as _f; _f._PARAM_OVERRIDE.clear()
                except Exception: pass
                return None, None

        def _run_and_score(params: dict, attempt_n: int) -> tuple[int, dict]:
            """Fuse + score one attempt. Updates job state.

            Returns (chart_score, details_dict) for multi-objective optimizer.
            """
            attempt_seed = seed + attempt_n - 1
            log.info(f"attempt {attempt_n} params={json.dumps(params, default=str)}")
            _set(job_id, progress=max(5, int((attempt_n - 1) / _total_attempts * 75)),
                 message=f"Attempt {attempt_n}/{_total_attempts}: fusing…")

            result, radio = _run_fuse(params, attempt_n, attempt_seed)
            if radio is None or not Path(radio).exists():
                log.warning(f"attempt {attempt_n} produced no output — skipping")
                _append_attempt(job_id, attempt_n, params, 0, "?", 0, "?", {})
                return 0, {}

            _set(job_id, progress=max(10, int((attempt_n - 0.3) / _total_attempts * 75)),
                 message=f"Attempt {attempt_n}/{_total_attempts}: scoring…")

            cs, cg, ms, mg, meta = _score_wav(radio)
            log.info(f"attempt {attempt_n} chart={cs} ({cg}) mix={ms} ({mg})")

            att_dir = job_dir / f"attempt_{attempt_n}"
            (att_dir / "metadata.json").write_text(json.dumps({
                "attempt": attempt_n,
                "params": params,
                "chart_score": cs, "chart_grade": cg,
                "mix_score": ms,   "mix_grade":   mg,
                "metadata": meta,
                "radio": radio,
                "club": result.get("club") if result else None,
                "intimate": result.get("intimate") if result else None,
            }, indent=2))

            _append_attempt(job_id, attempt_n, params, cs, cg, ms, mg, meta)
            return cs, meta

        # ── decide total attempt budget ───────────────────────────────────
        _total_attempts = MAX_ATTEMPTS if pro_mode else 1

        # ── genre detection → warm-start preset ──────────────────────────
        _genre, _genre_feats = _detect_genre(inst_path)
        _genre_preset = _get_preset(_genre)
        log.info(f"detected genre={_genre!r} features={_genre_feats}")
        _set(job_id, genre=_genre)

        # ── Bayesian optimization (or fallback hill-climb) ────────────────
        _set(job_id, progress=3, message=f"Optimizing ({_genre})…")
        best_params, best_score, _opt_log = _optimizer.optimize(
            objective_fn=_run_and_score,
            n_trials=_total_attempts,
            target_score=TARGET_SCORE,
            early_stop_score=90,
            early_stop_after_n=3,
            seed=seed,
            timeout_s=float(os.environ.get("VF_TIMEOUT_S", "540")),
            warm_start_params=[_genre_preset] if pro_mode else None,
        )
        attempts_done = len(_opt_log)

        # ── pick the best attempt ─────────────────────────────────────────
        best_meta = _pick_best(job_dir, attempts_done)
        if best_meta is None:
            _set(job_id, status="error", message="No successful attempts", progress=0)
            _cleanup(vocal_path, inst_path)
            return

        radio_path = best_meta.get("radio")
        if not radio_path or not Path(radio_path).exists():
            _set(job_id, status="error",
                 message="Best attempt output file missing", progress=0)
            _cleanup(vocal_path, inst_path)
            return

        # ── build variant URLs ────────────────────────────────────────────
        variants: dict[str, str] = {}
        for v in ("club", "intimate"):
            vp = best_meta.get(v)
            if vp and Path(vp).exists():
                variants[v] = f"/download/{job_id}/{v}"

        cs   = best_meta["chart_score"]
        cg   = best_meta["chart_grade"]
        ms   = best_meta["mix_score"]
        mg   = best_meta["mix_grade"]
        meta = best_meta["metadata"]

        log.info(f"DONE best_attempt={best_meta['attempt']} chart={cs} ({cg}) mix={ms} ({mg})")

        suffix = f" (best of {attempts_done})" if attempts_done > 1 else ""
        _set(job_id,
             status="done",
             progress=100,
             message=f"Chart: {cs} · {cg}{suffix}",
             output_url=f"/download/{job_id}",
             score=cs,
             chart_score=cs, chart_grade=cg,
             mix_score=ms,   mix_grade=mg,
             metadata=meta,
             variants=variants,
             _best_radio=radio_path,
             _best_meta=best_meta)

    _cleanup(vocal_path, inst_path)
    fh.close()
    log.removeHandler(fh)


# ---------------------------------------------------------------------------
# Attempt helpers
# ---------------------------------------------------------------------------
def _append_attempt(job_id, n, params, cs, cg, ms, mg, meta):
    with _lock:
        if job_id in _jobs:
            atts = _jobs[job_id].setdefault("attempts", [])
            atts.append({
                "n": n, "params": params,
                "chart_score": cs, "chart_grade": cg,
                "mix_score": ms,   "mix_grade":   mg,
            })


def _get_attempt_meta(job_dir: Path, n: int) -> dict:
    p = job_dir / f"attempt_{n}" / "metadata.json"
    if p.exists():
        try:
            return json.loads(p.read_text())
        except Exception:
            pass
    return {"params": PARAM_DEFAULTS}


def _pick_best(job_dir: Path, max_n: int) -> dict | None:
    best = None
    best_score = -1
    for i in range(1, max_n + 1):
        p = job_dir / f"attempt_{i}" / "metadata.json"
        if not p.exists():
            continue
        try:
            m = json.loads(p.read_text())
            if m.get("chart_score", 0) > best_score:
                best_score = m["chart_score"]
                best = m
        except Exception:
            pass
    return best


def _cleanup(*paths: str) -> None:
    for p in paths:
        try:
            os.remove(p)
        except OSError:
            pass


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------
@app.post("/fuse")
async def fuse_endpoint(
    song_a:      UploadFile = File(...),
    song_b:      UploadFile = File(...),
    seed:        int  = Form(42),
    direct_vocal: bool = Form(False),
    pro_mode:    bool = Form(True),
):
    job_id = uuid.uuid4().hex[:8]

    a_suf = Path(song_a.filename or "a.wav").suffix or ".wav"
    b_suf = Path(song_b.filename or "b.wav").suffix or ".wav"
    vocal_path = str(UPLOAD_DIR / f"{job_id}_vocal{a_suf}")
    inst_path  = str(UPLOAD_DIR / f"{job_id}_inst{b_suf}")

    with open(vocal_path, "wb") as f:
        shutil.copyfileobj(song_a.file, f)
    with open(inst_path, "wb") as f:
        shutil.copyfileobj(song_b.file, f)

    name_a = Path(song_a.filename or "vocal").stem
    name_b = Path(song_b.filename or "inst").stem

    with _lock:
        _jobs[job_id] = {
            "status": "running", "progress": 0,
            "message": "Queued…",
            "name_a": name_a, "name_b": name_b,
            "seed": seed, "pro_mode": pro_mode,
            "attempts": [],
        }

    threading.Thread(
        target=_worker,
        args=(job_id, vocal_path, inst_path, seed,
              direct_vocal, name_a, name_b, pro_mode),
        daemon=True,
    ).start()

    return JSONResponse({"job_id": job_id})


@app.get("/status/{job_id}")
def status_endpoint(job_id: str):
    state = _get(job_id)
    if state is None:
        raise HTTPException(status_code=404, detail="Job not found")
    return JSONResponse({k: v for k, v in state.items() if not k.startswith("_")})


@app.get("/download/{job_id}")
@app.get("/download/{job_id}/{variant}")
def download_endpoint(job_id: str, variant: str = "radio"):
    state = _get(job_id)
    if state is None:
        raise HTTPException(status_code=404, detail="Job not found")
    if state.get("status") != "done":
        raise HTTPException(status_code=409, detail="Job not complete yet")

    best = state.get("_best_meta", {})
    path_map = {
        "radio":    state.get("_best_radio") or best.get("radio"),
        "club":     best.get("club"),
        "intimate": best.get("intimate"),
    }
    serve_path = path_map.get(variant) or path_map.get("radio")
    if not serve_path or not Path(serve_path).exists():
        raise HTTPException(status_code=404, detail="Output file not found")

    return FileResponse(
        path=serve_path,
        media_type="audio/wav",
        filename=f"{state.get('name_b','inst')}_x_{state.get('name_a','vocal')}_seed{state.get('seed',42)}.wav",
    )


@app.get("/job/{job_id}/attempts")
def attempts_endpoint(job_id: str):
    job_dir = OUTPUT_DIR / job_id
    if not job_dir.exists():
        raise HTTPException(status_code=404, detail="Job not found")
    attempts = []
    for p in sorted(job_dir.glob("attempt_*/metadata.json")):
        try:
            attempts.append(json.loads(p.read_text()))
        except Exception:
            pass
    return JSONResponse({"job_id": job_id, "attempts": attempts})


@app.get("/health")
def health():
    ref = Path("reference_profile.json")
    n = 0
    if ref.exists():
        try: n = json.loads(ref.read_text()).get("n_tracks", 0)
        except Exception: pass
    return {"status": "ok", "reference_profile": ref.exists(),
            "reference_tracks": n, "active_jobs": len(_jobs)}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("api_fast:app", host="0.0.0.0", port=8000, reload=False)
