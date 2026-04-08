# VocalFusion — Architecture Reference

This document explains how the system is structured, how to add a new DSP
effect, and how the optimizer works.  It is the single source of truth for
anyone (human or AI assistant) picking up this codebase.

---

## High-level architecture

```
                  ┌─────────────┐
   POST /fuse ───▶│  api_fast   │
                  │  FastAPI    │
                  └──────┬──────┘
                         │ background thread (_worker)
                         ▼
                  ┌─────────────┐      ┌──────────────┐
                  │  fuser.py   │◀────▶│ vf/optimizer │
                  │  (mixing)   │      │  (Optuna TPE)│
                  └──────┬──────┘      └──────────────┘
                         │ uses
              ┌──────────┼──────────┐
              ▼          ▼          ▼
         vf/mixing   vf/types   listen.py
         (DSP util)  (dataclass) (scorer)
```

### Key files

| File | Role |
|------|------|
| `api_fast.py` | HTTP API, job queue, SSE progress stream |
| `fuser.py` | Full mixing chain (~7 000 lines) — do not edit lightly |
| `listen.py` | Audio quality scorer → `score_any(path) → (int, str, dict)` |
| `vf/types.py` | Typed dataclasses for all parameter sets |
| `vf/optimizer.py` | Bayesian optimizer + PARAM_SPACE definition |
| `vf/mixing.py` | Standalone DSP utilities (de-esser, phase-align, etc.) |
| `autonomous_songwriter.py` | End-to-end generative pipeline (MusicGen + Bark) |
| `benchmark_fuser.py` | Benchmark script: baseline vs Optuna on synthetic pairs |

---

## fuser.py vocal chain

Processing stages in order (search for `# Stage N` comments):

| Stage | Name | Key variable / function |
|-------|------|------------------------|
| 1 | HPF 80 Hz | `_hpf()` |
| 2 | Subtractive EQ | `_subtractive_eq()` |
| 3 | De-esser | `deess_adaptive()` / `_deess()` fallback |
| 4 | FET compressor | `_compress()` with `fet_attack` / `fet_release` |
| 5 | Opto compressor | `_compress()` with slow settings |
| 5b | Tape saturation (parallel) | `SAT_DRIVE = style["_exciter_drive"]` |
| 6 | Noise gate | `_gate()` |
| 7 | Additive EQ (presence/air/bass) | post_eq nodes + `char_tilt_db` tilt |
| 8 | Pre-delay reverb | `_reverb()` |
| 9 | Auto-tune | `_autotune()` |

After the vocal chain the mix is assembled, mastered, and written to disk.

### `_PARAM_OVERRIDE` — how external params hook in

`fuser.py` exposes a module-level `dict` called `_PARAM_OVERRIDE`.  Before
`fuse()` runs its mixing chain it reads this dict and injects values into the
`style` dict that controls every DSP parameter:

```python
# In fuser.py, at the top of fuse():
if "vocal_level_mult" in _PARAM_OVERRIDE:
    style["vocal_gain"] = _PARAM_OVERRIDE["vocal_level_mult"]
if "exciter_drive" in _PARAM_OVERRIDE:
    style["_exciter_drive"] = _PARAM_OVERRIDE["exciter_drive"]
# ... etc
```

The optimizer (or any external caller) writes into `_PARAM_OVERRIDE`, calls
`fuse()`, then clears it:

```python
import fuser as _fuser
_fuser._PARAM_OVERRIDE.update({"exciter_drive": 2.1, "char_tilt_db": -1.0})
result = _fuser.fuse(...)
_fuser._PARAM_OVERRIDE.clear()
```

**Never leave stale values in `_PARAM_OVERRIDE`** — always clear after each call.

---

## Parameter space (10 params)

Defined once in `vf/optimizer.py` as `PARAM_SPACE` (a `dict[str, tuple[float,
float]]`) and `PARAM_DEFAULTS` (`dict[str, float]`).

| Key | Range | Default |
|-----|-------|---------|
| `vocal_level_mult` | (0.5, 2.0) | 1.0 |
| `carve_db` | (0.0, 8.0) | 3.0 |
| `reverb_wet` | (0.0, 0.5) | 0.15 |
| `stereo_width` | (0.5, 2.0) | 1.0 |
| `master_ceiling` | (-2.0, -0.1) | -1.0 |
| `de_ess_strength` | (0.5, 2.0) | 1.0 |
| `comp_attack_ms` | (0.5, 20.0) | 4.0 |
| `comp_release_ms` | (30.0, 200.0) | 80.0 |
| `exciter_drive` | (0.5, 3.0) | 1.4 |
| `char_tilt_db` | (-3.0, 3.0) | 0.0 |

**To add a new parameter:**

1. Add it to `PARAM_SPACE` and `PARAM_DEFAULTS` in `vf/optimizer.py`.
2. Add a handler in the `_PARAM_OVERRIDE` block in `fuser.py` that injects it
   into `style`.
3. Add a field to `FuseParams` in `vf/types.py`.
4. Add a unit test in `tests/test_dsp.py` if it touches DSP, or a type test in
   `TestTypes`.

---

## Optimizer

`vf/optimizer.py` — `FuseOptimizer`

### How it works

1. **Trial 0 is always the defaults.** This gives the optimizer a warm-start
   baseline and ensures the first attempt is never worse than the old default.

2. **Optuna TPE sampler** — Tree-structured Parzen Estimator.  After trial 0 it
   uses a probabilistic model of the objective surface to suggest parameter
   combinations that are likely to score high.

3. **Early stopping** — if any trial reaches `target_score` (default 85) the
   study stops immediately.  The `early_stop_score` / `early_stop_after_n`
   arguments add a second tier: if score ≥ 90 after at least 3 trials, stop.

4. **Multi-objective reward** — `compute_reward(chart_score, details)` penalises
   any dimension (spectral/loudness/dynamics/perceptual/musical) that falls below
   40 with a quadratic penalty.  This prevents the optimizer from gaming one
   dimension at the expense of others.

5. **Fallback** — if `optuna` is not installed, `_optimize_fallback()` runs
   random search then hill-climb (identical to the old behaviour).

### Interface

```python
from vf.optimizer import FuseOptimizer, PARAM_SPACE, PARAM_DEFAULTS

opt = FuseOptimizer(PARAM_SPACE, PARAM_DEFAULTS)

def my_objective(params: dict, attempt_n: int) -> tuple[int, dict]:
    # run fuse with params, score output
    return chart_score, {"spectral": s, "loudness": l, ...}

best_params, best_score, log = opt.optimize(
    objective_fn=my_objective,
    n_trials=8,
    target_score=85,
    seed=42,
)
```

---

## Scoring

`listen.py` — `score_any(path: str) -> tuple[int, str, dict]`

Returns:
- `chart_score` — 0–100 integer
- `grade` — letter grade string (A+, A, B+, …, F)
- `details` — dict with keys: `spectral`, `loudness`, `dynamics`, `perceptual`, `musical`

The optimizer uses all five detail keys.  The API streams `chart_score` and
`grade` to the UI in real-time.

---

## DSP utilities (`vf/mixing.py`)

### `deess_adaptive(vox, threshold_db, max_reduction_db, sr)`

Detects the dominant sibilance frequency via energy-weighted centroid of the
4–12 kHz STFT band, then sets the de-esser crossover to 85% of that frequency
(clamped to 4 500–10 000 Hz).  Falls back gracefully if the STFT fails.

### `phase_align_stems(separated, original, sr, max_lag_ms)`

Low-passes both signals to 300 Hz, computes cross-correlation, shifts
`separated` by the detected lag.  Used after stem separation to correct any
sample-level drift introduced by the separator.

### `multiband_compress(audio, median_f0_hz, sr)`

4-band compressor with adaptive crossovers derived from the vocal fundamental:
- Band 1: 0–80 Hz (sub)
- Band 2: 80 Hz – 0.8×F0 (body)
- Band 3: 0.8×F0 – 4×F0 (presence)
- Band 4: 4×F0–Nyquist (air/sibilance)

### `true_peak_dbtp(audio, sr)`

4× polyphase upsample (via `scipy.signal.resample_poly`) then peak measurement.
Compliant with ITU-R BS.1770-4.

---

## Adding a new DSP effect

1. **Write the function in `vf/mixing.py`.**  Accept a numpy array, return a
   numpy array of the same shape.  Keep it pure (no global state).

2. **Write unit tests in `tests/test_dsp.py`.**  Use the helper functions
   (`_sine`, `_stereo`, `_white_noise`, `_band_rms`) already defined there.

3. **Hook it into `fuser.py`.**  Find the stage where it belongs, add a
   `try/except ImportError` wrapper so fuser degrades gracefully if `vf.mixing`
   is unavailable, and add a `_PARAM_OVERRIDE` handler if the effect has tunable
   parameters.

4. **Expose the parameter** via `PARAM_SPACE` in `vf/optimizer.py` and
   `FuseParams` in `vf/types.py`.

5. **Run the tests** and the benchmark to confirm no regression.

---

## Running the stack

```bash
# API server
conda run -n vocal-fusion uvicorn api_fast:app --host 0.0.0.0 --port 8000

# UI dev server
cd ui && npm run dev

# Unit tests (fast)
pytest tests/test_dsp.py -v

# Integration tests (requires runtime deps)
pytest tests/test_integration.py -v

# Benchmark (3 synthetic pairs × 4 trials)
python benchmark_fuser.py --n-pairs 3 --trials 4

# Autonomous song generation
python autonomous_songwriter.py --generate-n 1 --genre pop
```

---

## Environment variables

| Variable | Default | Effect |
|----------|---------|--------|
| `VF_TIMEOUT_S` | 540 | Max seconds for one fuse job |
| `VF_SKIP_FUSE` | 0 | Set to `1` to skip slow fuse tests in CI |
| `CUBLAS_WORKSPACE_CONFIG` | `:4096:8` | Set by fuser.py for determinism |

---

## Project layout

```
VocalFusion/
├── api_fast.py            # FastAPI server
├── fuser.py               # Core mixing engine (do not refactor lightly)
├── listen.py              # Audio scorer
├── autonomous_songwriter.py
├── benchmark_fuser.py
├── requirements_upgraded.txt
├── MIGRATION.md
├── CLAUDE.md              # ← you are here
├── vf/
│   ├── __init__.py
│   ├── types.py           # Typed parameter dataclasses
│   ├── optimizer.py       # PARAM_SPACE + FuseOptimizer
│   └── mixing.py          # DSP utilities
├── tests/
│   ├── __init__.py
│   ├── test_dsp.py        # Unit tests for vf/mixing.py + vf/optimizer.py
│   └── test_integration.py # Full pipeline integration tests
└── ui/
    ├── app/page.tsx       # Next.js frontend
    └── lib/api.ts         # API client
```
