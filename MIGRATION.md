# VocalFusion — Migration Guide

Upgrading from the monolithic `fuser.py` + hill-climb optimizer to the new
modular architecture (`vf/` package + Optuna TPE).

---

## What changed

| Area | Before | After |
|------|--------|-------|
| Parameter space | 6 params in `api_fast.py` | 10 params in `vf/optimizer.py` |
| Optimization | Baseline → random search → hill-climb | Optuna TPE (warm-start at defaults) |
| Reward signal | `chart_score` only | Multi-objective: penalises weakest dimension |
| De-esser | Fixed 6.5 kHz crossover | Frequency-tracking (4 500–10 000 Hz adaptive) |
| Determinism | Not guaranteed | `CUBLAS_WORKSPACE_CONFIG` + `cudnn.deterministic` |
| Parameter types | `dict[str, float]` | `vf.types.FuseParams` dataclass (also works as dict) |
| Tests | None | `tests/test_dsp.py`, `tests/test_integration.py` |

---

## Step-by-step upgrade

### 1. Install new dependencies

```bash
pip install -r requirements_upgraded.txt
```

Optuna is the only new hard dependency for the optimizer. Everything else was
already in your environment.

### 2. Verify the new modules load

```bash
python -c "from vf.optimizer import FuseOptimizer, PARAM_SPACE; print('OK', list(PARAM_SPACE))"
python -c "from vf.mixing import deess_adaptive, phase_align_stems, true_peak_dbtp; print('OK')"
python -c "from vf.types import FuseParams; print(FuseParams.defaults())"
```

All three should print `OK` with no import errors.

### 3. Verify fuser.py edits

The following were added to `fuser.py`:

**Determinism block** (after DeepFilterNet init, search for `CUBLAS_WORKSPACE_CONFIG`):
```python
_os_det.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
_torch_det.backends.cudnn.deterministic = True
_torch_det.backends.cudnn.benchmark = False
_torch_det.use_deterministic_algorithms(True, warn_only=True)
```

**Adaptive de-esser** (Stage 3, search for `deess_adaptive`):
```python
try:
    from vf.mixing import deess_adaptive as _deess_adaptive
    vox = _deess_adaptive(vox, ...)
except Exception:
    vox = _deess(vox, ...)
```

**Exciter drive hook** (Stage 5b, search for `_exciter_drive`):
```python
SAT_DRIVE = float(style.get("_exciter_drive", 1.4))
```

**Char tilt hook** (Stage 6 post-EQ, search for `_char_tilt_db`):
```python
_char_tilt_db = float(style.get("_char_tilt_db", 0.0))
if abs(_char_tilt_db) > 0.05:
    # low shelf + high shelf tilt nodes added
```

**Four new `_PARAM_OVERRIDE` handlers** (search for `comp_attack_ms`):
```python
if "comp_attack_ms" in _PARAM_OVERRIDE: style["fet_attack"] = ...
if "comp_release_ms" in _PARAM_OVERRIDE: style["fet_release"] = ...
if "exciter_drive" in _PARAM_OVERRIDE: style["_exciter_drive"] = ...
if "char_tilt_db" in _PARAM_OVERRIDE: style["_char_tilt_db"] = ...
```

If any of these are missing, re-apply the relevant edits from the session log.

### 4. Verify api_fast.py edits

Check these at the top of `api_fast.py`:
```python
from vf.optimizer import FuseOptimizer as _FuseOptimizer, PARAM_SPACE, PARAM_DEFAULTS
_optimizer = _FuseOptimizer(PARAM_SPACE, PARAM_DEFAULTS)
```

And in `_score_wav()`, confirm the return includes dimension keys:
```python
return chart_score, {"spectral": ..., "loudness": ..., "dynamics": ...,
                     "perceptual": ..., "musical": ...}
```

And in `_worker()`, confirm the optimizer call (replacing the old hill-climb):
```python
best_params, best_score, _opt_log = _optimizer.optimize(
    objective_fn=_run_and_score,
    n_trials=_total_attempts,
    ...
)
```

### 5. Run the tests

```bash
# Fast unit tests (no audio files needed, < 30 s)
pytest tests/test_dsp.py -v

# Integration tests (requires all runtime deps; uses synthetic audio)
pytest tests/test_integration.py -v

# Skip the slow fuse() tests on CI:
VF_SKIP_FUSE=1 pytest tests/test_integration.py -v
```

All tests should pass before going to production.

### 6. Restart the API server

```bash
# Kill existing uvicorn processes
pkill -f "uvicorn api_fast"

# Restart (adjust path / conda env as needed)
conda run -n vocal-fusion uvicorn api_fast:app --host 0.0.0.0 --port 8000 &
```

### 7. Smoke test

```bash
curl http://localhost:8000/health
```

Should return `{"status": "ok"}`.

Then run the benchmark to confirm the optimizer produces better scores than the
old baseline:
```bash
python benchmark_fuser.py --n-pairs 3 --trials 4
```

---

## Rolling back

If anything breaks, the change is isolated — `fuser.py` falls back gracefully:

- The `deess_adaptive` call is wrapped in `try/except`; if `vf.mixing` fails
  to import, it falls back to the old `_deess()`.
- If `optuna` is not installed, `FuseOptimizer` falls back to random search +
  hill-climb (same algorithm as before).
- The four new `_PARAM_OVERRIDE` keys are additive; the old 6 keys still work
  unchanged.

To fully roll back `api_fast.py`, restore the previous PARAM_SPACE dict and
`_hill_climb()` function from git history.

---

## New parameters (10 total)

| Parameter | Range | Default | Effect |
|-----------|-------|---------|--------|
| `vocal_level_mult` | 0.5 – 2.0 | 1.0 | Vocal vs beat balance |
| `carve_db` | 0.0 – 8.0 | 3.0 | Subtractive EQ cut in beat |
| `reverb_wet` | 0.0 – 0.5 | 0.15 | Reverb send level |
| `stereo_width` | 0.5 – 2.0 | 1.0 | M/S width of mix |
| `master_ceiling` | -2.0 – -0.1 | -1.0 | True-peak limiter ceiling |
| `de_ess_strength` | 0.5 – 2.0 | 1.0 | De-esser gain multiplier |
| `comp_attack_ms` | 0.5 – 20.0 | 4.0 | FET compressor attack |
| `comp_release_ms` | 30.0 – 200.0 | 80.0 | FET compressor release |
| `exciter_drive` | 0.5 – 3.0 | 1.4 | Tape saturation drive |
| `char_tilt_db` | -3.0 – 3.0 | 0.0 | Spectral tilt (dark ↔ bright) |
