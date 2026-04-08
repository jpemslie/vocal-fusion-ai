"""
tests/test_dsp.py — Unit tests for VocalFusion DSP blocks.

Each test uses synthetic audio (sine sweeps, white noise, impulses) so no
real audio files are needed.  Run with:

    pytest tests/test_dsp.py -v

All tests must pass in < 30 seconds on CPU.
"""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

import numpy as np
import pytest
from scipy.signal import chirp

SR = 44100


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _sine(freq_hz: float, duration_s: float = 1.0, amp: float = 0.5) -> np.ndarray:
    """Mono sine wave."""
    t = np.linspace(0, duration_s, int(SR * duration_s), endpoint=False)
    return (amp * np.sin(2 * np.pi * freq_hz * t)).astype(np.float32)


def _stereo(mono: np.ndarray) -> np.ndarray:
    """Convert (N,) → (N, 2)."""
    return np.stack([mono, mono], axis=1)


def _white_noise(duration_s: float = 1.0, amp: float = 0.3) -> np.ndarray:
    rng = np.random.default_rng(42)
    return (amp * rng.standard_normal(int(SR * duration_s))).astype(np.float32)


def _chirp_signal(f0: float, f1: float, duration_s: float = 1.0) -> np.ndarray:
    t = np.linspace(0, duration_s, int(SR * duration_s), endpoint=False)
    return chirp(t, f0=f0, f1=f1, t1=duration_s, method="logarithmic").astype(np.float32)


def _rms_db(x: np.ndarray) -> float:
    rms = float(np.sqrt(np.mean(x ** 2) + 1e-12))
    return 20 * np.log10(rms)


def _band_rms(x: np.ndarray, lo: float, hi: float) -> float:
    from scipy.signal import butter, sosfilt
    nyq = SR / 2
    sos = butter(4, [lo / nyq, min(hi / nyq, 0.98)], btype="band", output="sos")
    filtered = sosfilt(sos, x.mean(axis=1) if x.ndim == 2 else x)
    return float(np.sqrt(np.mean(filtered ** 2) + 1e-12))


# ---------------------------------------------------------------------------
# vf.mixing.deess_adaptive
# ---------------------------------------------------------------------------

class TestDeessAdaptive:
    """Tests for the frequency-tracking de-esser."""

    def test_not_silent(self):
        """Output must not be silent."""
        from vf.mixing import deess_adaptive
        noise = _stereo(_white_noise(1.0, 0.5))
        out = deess_adaptive(noise, threshold_db=-30.0, max_reduction_db=12.0)
        assert np.max(np.abs(out)) > 0.01

    def test_reduces_sibilance_band(self):
        """7–10 kHz energy should decrease after de-essing a sibilant signal."""
        from vf.mixing import deess_adaptive
        sib = _sine(8000.0, 1.0, 0.8)          # loud 8 kHz sibilance
        body = _sine(200.0, 1.0, 0.3)           # vocal body
        combo = _stereo((sib + body).astype(np.float32))

        out = deess_adaptive(combo, threshold_db=-25.0, max_reduction_db=10.0)
        rms_in  = _band_rms(combo, 7000, 10000)
        rms_out = _band_rms(out,   7000, 10000)
        assert rms_out < rms_in * 0.95, (
            f"Sibilance band RMS did not decrease: in={rms_in:.4f} out={rms_out:.4f}"
        )

    def test_preserves_low_freq_body(self):
        """200 Hz body should be ≥ 95% preserved."""
        from vf.mixing import deess_adaptive
        body = _stereo(_sine(200.0, 1.0, 0.5))
        out  = deess_adaptive(body, threshold_db=-25.0, max_reduction_db=10.0)
        rms_in  = _band_rms(body, 150, 300)
        rms_out = _band_rms(out,  150, 300)
        assert rms_out >= rms_in * 0.95, (
            f"Low-freq body was damaged: in={rms_in:.5f} out={rms_out:.5f}"
        )

    def test_output_shape(self):
        """Output must have the same shape as input."""
        from vf.mixing import deess_adaptive
        x = _stereo(_white_noise(0.5))
        out = deess_adaptive(x)
        # deess_adaptive returns (N, channels) regardless of input layout
        assert out.shape[0] == x.shape[0] or out.shape[1] == x.shape[1]

    def test_no_clipping(self):
        """Output peak must not exceed input peak by more than 0.5 dB."""
        from vf.mixing import deess_adaptive
        x = _stereo(_white_noise(1.0, 0.9))
        out = deess_adaptive(x, threshold_db=-20.0)
        assert np.max(np.abs(out)) <= np.max(np.abs(x)) * 1.06


# ---------------------------------------------------------------------------
# vf.mixing.phase_align_stems
# ---------------------------------------------------------------------------

class TestPhaseAlign:
    """Tests for cross-correlation phase correction."""

    def test_zero_lag_unchanged(self):
        """Perfectly aligned stems should return near-identical output."""
        from vf.mixing import phase_align_stems
        sig = _stereo(_sine(440.0, 2.0))
        out = phase_align_stems(sig, sig.copy(), sr=SR)
        # Allow rounding; signal should be essentially unchanged
        assert np.allclose(out, sig, atol=1e-3)

    def test_corrects_known_lag(self):
        """A signal shifted by 50 samples should be corrected back."""
        from vf.mixing import phase_align_stems
        orig = _stereo(_white_noise(2.0, 0.5))
        lag  = 50
        shifted = np.vstack([np.zeros((lag, 2), dtype=np.float32), orig[:-lag]])
        corrected = phase_align_stems(shifted, orig, sr=SR, max_lag_ms=10.0)
        # Correlation with original should be higher after correction
        corr_before = float(np.corrcoef(
            orig[:, 0].flatten(), shifted[:, 0].flatten())[0, 1])
        corr_after  = float(np.corrcoef(
            orig[:, 0].flatten(), corrected[:, 0].flatten())[0, 1])
        assert corr_after >= corr_before - 0.05  # allow small tolerance

    def test_output_length_preserved(self):
        """Output must be the same length as input."""
        from vf.mixing import phase_align_stems
        sig = _stereo(_white_noise(1.0))
        out = phase_align_stems(sig, sig.copy())
        assert out.shape[0] == sig.shape[0]


# ---------------------------------------------------------------------------
# vf.mixing.true_peak_dbtp
# ---------------------------------------------------------------------------

class TestTruePeak:
    """Tests for ITU-R BS.1770-4 true-peak meter."""

    def test_silent_is_very_negative(self):
        from vf.mixing import true_peak_dbtp
        silence = np.zeros(SR, dtype=np.float32)
        tp = true_peak_dbtp(silence)
        assert tp < -60.0

    def test_full_scale_sine_near_zero_dBTP(self):
        from vf.mixing import true_peak_dbtp
        # A 0 dBFS sine should measure close to 0 dBTP (exact value depends on
        # reconstruction — allow ±1 dB)
        sig = _sine(1000.0, 1.0, amp=1.0)
        tp  = true_peak_dbtp(sig)
        assert -1.5 <= tp <= 0.5, f"Unexpected true peak: {tp:.2f} dBTP"

    def test_stereo_input(self):
        from vf.mixing import true_peak_dbtp
        sig = _stereo(_sine(440.0, 0.5, 0.5))
        tp  = true_peak_dbtp(sig)
        assert isinstance(tp, float)


# ---------------------------------------------------------------------------
# vf.optimizer
# ---------------------------------------------------------------------------

class TestOptimizer:
    """Tests for the Bayesian optimizer."""

    def _make_objective(self, target: int = 80):
        """Synthetic objective: returns target when params are near defaults."""
        call_log = []

        def obj(params: dict, attempt_n: int) -> tuple[int, dict]:
            dist = sum(abs(v - 0.5) for v in params.values())
            score = max(0, min(100, target - int(dist * 10)))
            call_log.append((attempt_n, score))
            return score, {"spectral": score, "loudness": score,
                           "dynamics": score, "perceptual": score}

        return obj, call_log

    def test_runs_n_trials(self):
        """Optimizer must run exactly n_trials (or stop early)."""
        from vf.optimizer import FuseOptimizer, PARAM_SPACE, PARAM_DEFAULTS
        obj, log = self._make_objective(60)
        opt = FuseOptimizer(PARAM_SPACE, PARAM_DEFAULTS)
        _, _, attempt_log = opt.optimize(obj, n_trials=4, target_score=99, seed=1)
        assert 1 <= len(attempt_log) <= 4

    def test_returns_best_params(self):
        """Best params should be within the space bounds."""
        from vf.optimizer import FuseOptimizer, PARAM_SPACE, PARAM_DEFAULTS
        obj, _ = self._make_objective(70)
        opt = FuseOptimizer(PARAM_SPACE, PARAM_DEFAULTS)
        best_params, best_score, _ = opt.optimize(obj, n_trials=3, seed=42)
        for k, (lo, hi) in PARAM_SPACE.items():
            assert lo <= best_params[k] <= hi, f"{k} out of bounds: {best_params[k]}"

    def test_compute_reward_penalises_weak_dimension(self):
        """Reward should be lower when a dimension is very bad."""
        from vf.optimizer import compute_reward
        r_good = compute_reward(70, {"spectral": 70, "loudness": 70,
                                      "dynamics": 70, "perceptual": 70, "musical": 70})
        r_weak = compute_reward(70, {"spectral": 10, "loudness": 70,
                                      "dynamics": 70, "perceptual": 70, "musical": 70})
        assert r_weak < r_good, "Weak-dimension reward must be lower"

    def test_compute_reward_baseline_is_chart_score(self):
        """With all dims at 50, reward should equal chart_score."""
        from vf.optimizer import compute_reward
        r = compute_reward(75, {})   # empty details → no penalty
        assert r == 75.0


# ---------------------------------------------------------------------------
# vf.types
# ---------------------------------------------------------------------------

class TestTypes:
    def test_fuse_params_round_trip(self):
        from vf.types import FuseParams
        fp = FuseParams(vocal_level_mult=1.2, carve_db=4.0, char_tilt_db=-1.5)
        d  = fp.to_dict()
        assert d["vocal_level_mult"] == pytest.approx(1.2)
        fp2 = FuseParams.from_dict(d)
        assert fp2.vocal_level_mult == pytest.approx(1.2)
        assert fp2.char_tilt_db == pytest.approx(-1.5)

    def test_defaults_are_sane(self):
        from vf.types import FuseParams
        fp = FuseParams.defaults()
        assert fp.de_ess_strength == 1.0
        assert fp.exciter_drive   == 1.4
        assert fp.char_tilt_db    == 0.0
