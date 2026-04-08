"""
vf/optimizer.py — Bayesian parameter optimization using Optuna.

Replaces the linear hill-climbing in api_fast.py with TPE-sampled Bayesian
optimization.  Advantages:
  - Explores non-local minima (hill-climbing gets stuck).
  - Multi-objective reward: weights weakest listen.py dimension extra.
  - Early stopping: terminates at score >= 90 after 3 trials.
  - Deterministic: fixed seed → reproducible trial order.
  - Warm-start: always evaluates PARAM_DEFAULTS as trial 0.

Usage (inside api_fast.py _worker):

    from vf.optimizer import FuseOptimizer, PARAM_SPACE, PARAM_DEFAULTS

    opt = FuseOptimizer(PARAM_SPACE, PARAM_DEFAULTS)
    best_params, best_score, log = opt.optimize(
        objective_fn=_run_and_score,   # (params, attempt_n) -> (score, details)
        n_trials=_total_attempts,
        target_score=TARGET_SCORE,
        seed=seed,
    )
"""
from __future__ import annotations

import logging
import time
from typing import Callable

import numpy as np

log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Canonical parameter space  (shared between optimizer and api_fast)
# ---------------------------------------------------------------------------

PARAM_SPACE: dict[str, tuple[float, float]] = {
    # ── Original 6 params ──────────────────────────────────────────────────
    "vocal_level_mult": (0.70, 1.30),
    "carve_db":         (1.0,  6.0),
    "sidechain_depth":  (0.20, 0.70),
    "de_ess_strength":  (0.50, 2.00),
    "bass_shelf_gain":  (-3.0, 6.0),
    "air_boost":        (-2.0, 4.0),
    # ── New params (v2) ────────────────────────────────────────────────────
    "comp_attack_ms":   (1.0,  15.0),   # FET compressor attack
    "comp_release_ms":  (40.0, 180.0),  # FET compressor release
    "exciter_drive":    (0.5,  2.5),    # harmonic exciter tanh scale
    "char_tilt_db":     (-3.0, 3.0),    # spectral tilt (dark ↔ bright)
}

PARAM_DEFAULTS: dict[str, float] = {
    "vocal_level_mult": 1.0,
    "carve_db":         3.0,
    "sidechain_depth":  0.40,
    "de_ess_strength":  1.0,
    "bass_shelf_gain":  0.0,
    "air_boost":        0.0,
    "comp_attack_ms":   5.0,
    "comp_release_ms":  100.0,
    "exciter_drive":    1.4,
    "char_tilt_db":     0.0,
}

# listen.py dimension keys expected in the details dict
_DIM_KEYS = ("spectral", "loudness", "dynamics", "perceptual", "musical")


# ---------------------------------------------------------------------------
# Multi-objective reward
# ---------------------------------------------------------------------------

def compute_reward(chart_score: int, details: dict) -> float:
    """
    Weighted reward that penalises a bad weakest dimension and rewards fixing it.

    The pure chart_score treats all dimensions equally.  In practice a mix
    that scores 70/100 on spectral + 20/100 on dynamics SHOULD score lower
    than one scoring 50/50 — the dynamic problem is audible and unfixable
    in post.  We add a non-linear penalty for the weakest dimension.

    Args:
        chart_score: listen.score_any() chart_score [0, 100].
        details: dimension subscores dict from listen.score_any().

    Returns:
        reward: float ≥ 0.  Higher = better trial.
    """
    reward = float(chart_score)

    if not details:
        return reward

    dim_scores = {k: float(details.get(k, 50.0)) for k in _DIM_KEYS}
    weakest_val = min(dim_scores.values())
    weakest_key = min(dim_scores, key=dim_scores.get)

    # Non-linear penalty when any dimension is very bad
    if weakest_val < 40:
        # Quadratic penalty: missing 40→0 adds up to -20 reward points
        deficit = 40.0 - weakest_val
        reward -= (deficit ** 2) / 80.0

    # Bonus for lifting a previously very weak dimension above 55
    # (encourages the sampler to explore the weak dimension's fix-space)
    if weakest_val >= 55:
        reward += (weakest_val - 55.0) * 0.15

    log.debug(
        "reward=%.2f  chart=%d  weakest=%s=%.1f",
        reward, chart_score, weakest_key, weakest_val,
    )
    return reward


# ---------------------------------------------------------------------------
# Optimizer
# ---------------------------------------------------------------------------

class FuseOptimizer:
    """
    Bayesian optimization over the DSP parameter space using Optuna TPE.

    Warm-starts every study with the default params as the first trial so
    the baseline is always measured and serves as the initial prior.

    Args:
        param_space:    Dict mapping param name → (lo, hi) bounds.
        param_defaults: Dict mapping param name → default value.
    """

    def __init__(
        self,
        param_space:    dict[str, tuple[float, float]] = PARAM_SPACE,
        param_defaults: dict[str, float]               = PARAM_DEFAULTS,
    ):
        self.param_space    = param_space
        self.param_defaults = param_defaults

    # ------------------------------------------------------------------
    def optimize(
        self,
        objective_fn: Callable[[dict, int], tuple[int, dict]],
        n_trials:            int              = 8,
        target_score:        int              = 85,
        early_stop_score:    int              = 90,
        early_stop_after_n:  int              = 3,
        seed:                int              = 42,
        timeout_s:           float            = 600.0,
        warm_start_params:   list[dict] | None = None,
    ) -> tuple[dict, int, list[dict]]:
        """
        Run Bayesian optimization.

        Args:
            objective_fn:       Callable(params_dict, attempt_n) → (chart_score, details_dict).
                                Must handle all side-effects (job state updates, file saving).
            n_trials:           Maximum number of fuse+score iterations.
            target_score:       Stop early when chart_score ≥ this.
            early_stop_score:   Stop immediately if this is hit in the first
                                ``early_stop_after_n`` trials.
            early_stop_after_n: Window for the aggressive early stop check.
            seed:               Optuna sampler seed (reproducible trial order).
            timeout_s:          Hard wall-clock timeout (seconds).
            warm_start_params:  Optional list of extra param dicts to enqueue
                                after the defaults (e.g. a genre-tuned preset).
                                Each is clipped to PARAM_SPACE bounds before use.

        Returns:
            (best_params, best_score, attempt_log)
            attempt_log: list of {n, params, chart_score, reward} dicts.
        """
        # Clip any warm-start params to valid bounds
        warm: list[dict] = []
        for wp in (warm_start_params or []):
            clipped = _clip(
                {k: wp.get(k, self.param_defaults[k]) for k in self.param_space},
                self.param_space,
            )
            warm.append(clipped)

        try:
            import optuna
            optuna.logging.set_verbosity(optuna.logging.WARNING)
            return self._optimize_optuna(
                objective_fn, n_trials, target_score,
                early_stop_score, early_stop_after_n,
                seed, timeout_s, warm,
            )
        except ImportError:
            log.warning("optuna not installed — falling back to random search + hill-climb")
            return self._optimize_fallback(
                objective_fn, n_trials, target_score, seed, timeout_s, warm,
            )

    # ------------------------------------------------------------------
    def _optimize_optuna(
        self,
        objective_fn:        Callable,
        n_trials:            int,
        target_score:        int,
        early_stop_score:    int,
        early_stop_after_n:  int,
        seed:                int,
        timeout_s:           float,
        warm_starts:         list[dict] | None = None,
    ) -> tuple[dict, int, list[dict]]:
        import optuna
        from optuna.samplers import TPESampler

        attempt_log:  list[dict]       = []
        best_params:  dict             = dict(self.param_defaults)
        best_score:   int              = 0
        best_reward:  float            = -999.0
        t0 = time.monotonic()

        # Trial counter shared with closure
        _trial_n = [0]
        # Pre-queued param sets: defaults first, then any genre warm-starts
        _pre_queue = [dict(self.param_defaults)] + (warm_starts or [])

        def _optuna_objective(trial: optuna.Trial) -> float:
            if time.monotonic() - t0 > timeout_s:
                trial.set_user_attr("timed_out", True)
                raise optuna.exceptions.TrialPruned("Wall-clock timeout")

            _trial_n[0] += 1
            attempt_n = _trial_n[0]

            # Pre-queued trials (defaults = 1, genre preset = 2, …)
            if attempt_n <= len(_pre_queue):
                params = dict(_pre_queue[attempt_n - 1])
            else:
                params = {
                    k: trial.suggest_float(k, lo, hi)
                    for k, (lo, hi) in self.param_space.items()
                }

            chart_score, details = objective_fn(params, attempt_n)
            reward = compute_reward(chart_score, details)

            attempt_log.append({
                "n":           attempt_n,
                "params":      dict(params),
                "chart_score": chart_score,
                "reward":      reward,
            })

            nonlocal best_params, best_score, best_reward
            if reward > best_reward:
                best_reward  = reward
                best_score   = chart_score
                best_params  = dict(params)

            log.info(
                "Trial %d/%d → chart=%d  reward=%.2f  best=%d",
                attempt_n, n_trials, chart_score, reward, best_score,
            )

            # Aggressive early stop: score >= 90 in first N trials
            if chart_score >= early_stop_score and attempt_n <= early_stop_after_n:
                log.info("Early stop: score=%d >= %d in trial %d",
                         chart_score, early_stop_score, attempt_n)
                raise optuna.exceptions.TrialPruned("Early stop target hit")

            # Soft target reached
            if chart_score >= target_score:
                log.info("Target score %d reached at trial %d", target_score, attempt_n)

            return reward

        sampler = TPESampler(seed=seed, n_startup_trials=3)
        study   = optuna.create_study(direction="maximize", sampler=sampler)

        # Enqueue defaults + any genre warm-start presets
        for wp in _pre_queue:
            study.enqueue_trial(
                {k: wp.get(k, self.param_defaults[k]) for k in self.param_space},
                skip_if_exists=False,
            )

        try:
            study.optimize(
                _optuna_objective,
                n_trials=n_trials,
                timeout=timeout_s,
                catch=(Exception,),
                show_progress_bar=False,
            )
        except optuna.exceptions.TrialPruned:
            pass

        # Pull best params from study (may differ from our tracking if Optuna
        # ignores pruned trials in best_trial lookup)
        try:
            best_trial = study.best_trial
            # Re-clip to ensure we stay in bounds
            best_params = _clip(
                {k: best_trial.params.get(k, self.param_defaults[k])
                 for k in self.param_space},
                self.param_space,
            )
        except (ValueError, RuntimeError):
            pass   # no successful trials — use what we tracked

        return best_params, best_score, attempt_log

    # ------------------------------------------------------------------
    def _optimize_fallback(
        self,
        objective_fn: Callable,
        n_trials:     int,
        target_score: int,
        seed:         int,
        timeout_s:    float,
        warm_starts:  list[dict] | None = None,
    ) -> tuple[dict, int, list[dict]]:
        """
        Fallback when Optuna is unavailable: pre-queued warm-starts → random search → hill-climb.
        Preserved for environments without the optuna package.
        """
        rng = np.random.default_rng(seed + 200)
        attempt_log: list[dict] = []
        t0 = time.monotonic()

        pre_queue = [dict(self.param_defaults)] + (warm_starts or [])
        best_params = dict(self.param_defaults)
        best_score  = 0
        best_reward = -999.0
        n_done = 0

        # Run all pre-queued param sets first (defaults + genre preset)
        for pq_params in pre_queue:
            if n_done >= n_trials or time.monotonic() - t0 > timeout_s:
                break
            n_done += 1
            chart_score, details = objective_fn(dict(pq_params), n_done)
            reward = compute_reward(chart_score, details)
            attempt_log.append({"n": n_done, "params": dict(pq_params),
                                 "chart_score": chart_score, "reward": reward})
            if reward > best_reward:
                best_reward = reward
                best_score  = chart_score
                best_params = dict(pq_params)
            if chart_score >= target_score:
                return best_params, best_score, attempt_log

        # Random phase (attempts 2-4)
        random_best_params = dict(best_params)
        random_best_reward = best_reward
        for i in range(3):
            if n_done >= n_trials or time.monotonic() - t0 > timeout_s:
                break
            n_done += 1
            params = {
                k: float(rng.uniform(lo, hi))
                for k, (lo, hi) in self.param_space.items()
            }
            cs, det = objective_fn(params, n_done)
            rw = compute_reward(cs, det)
            attempt_log.append({"n": n_done, "params": params,
                                 "chart_score": cs, "reward": rw})
            if rw > random_best_reward:
                random_best_reward = rw
                random_best_params = dict(params)
                best_score  = cs
                best_params = dict(params)
            if cs >= target_score:
                return best_params, best_score, attempt_log

        # Hill-climb from best
        step_frac = 0.10
        for _ in range(6):
            if n_done >= n_trials or time.monotonic() - t0 > timeout_s:
                break
            improved = False
            for key, (lo, hi) in self.param_space.items():
                if n_done >= n_trials:
                    break
                width = hi - lo
                for sign in (+1, -1):
                    if n_done >= n_trials:
                        break
                    candidate = _clip(
                        {**best_params, key: best_params[key] + sign * step_frac * width},
                        self.param_space,
                    )
                    n_done += 1
                    cs, det = objective_fn(candidate, n_done)
                    rw = compute_reward(cs, det)
                    attempt_log.append({"n": n_done, "params": candidate,
                                        "chart_score": cs, "reward": rw})
                    if rw > best_reward:
                        best_reward = rw
                        best_score  = cs
                        best_params = dict(candidate)
                        improved = True
                        break
                if improved:
                    break
            if not improved:
                break

        return best_params, best_score, attempt_log


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _clip(params: dict, space: dict[str, tuple[float, float]]) -> dict:
    return {
        k: float(np.clip(v, space[k][0], space[k][1]))
        for k, v in params.items()
        if k in space
    }
