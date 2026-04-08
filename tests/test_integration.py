"""
tests/test_integration.py — Integration tests for the VocalFusion pipeline.

These tests use short synthetic audio to exercise the full fuse → score path
without requiring real vocal/beat files.  They are intentionally slow-ish
(fuser.py runs stem separation) and require all runtime dependencies.

Skip cleanly on CI if fuser cannot be imported:
    pytest tests/test_integration.py -v --ignore-glob='*slow*'

Environment flag to skip the fuse (for fast CI):
    VF_SKIP_FUSE=1 pytest tests/test_integration.py
"""
import os
import sys
import tempfile
import numpy as np
import pytest
import soundfile as sf

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

SR = 44100
SKIP_FUSE = os.environ.get("VF_SKIP_FUSE", "0") == "1"
skip_slow = pytest.mark.skipif(SKIP_FUSE, reason="VF_SKIP_FUSE=1")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_beat(duration_s: float = 5.0) -> np.ndarray:
    """Synthetic beat: kick + snare + hi-hat + bass."""
    t  = np.linspace(0, duration_s, int(SR * duration_s), endpoint=False)
    rng = np.random.default_rng(0)
    # 4/4 kick at 120 BPM
    beat_s = 60.0 / 120.0
    kick_times = np.arange(0, duration_s, beat_s)
    sig = np.zeros(len(t), dtype=np.float32)
    for kt in kick_times:
        idx  = int(kt * SR)
        dur  = int(0.15 * SR)
        env  = np.exp(-np.linspace(0, 20, min(dur, len(t) - idx)))
        sub  = np.sin(2 * np.pi * 60 * np.linspace(0, 0.15, len(env))) * env * 0.8
        sig[idx: idx + len(sub)] += sub.astype(np.float32)
    # Snare on beat 2 + 4
    for st in np.arange(beat_s, duration_s, beat_s * 2):
        idx  = int(st * SR)
        dur  = int(0.08 * SR)
        snr  = rng.standard_normal(min(dur, len(t) - idx)) * 0.4 * \
               np.exp(-np.linspace(0, 15, min(dur, len(t) - idx)))
        sig[idx: idx + len(snr)] += snr.astype(np.float32)
    # Continuous bass
    sig += (0.3 * np.sin(2 * np.pi * 80 * t)).astype(np.float32)
    stereo = np.stack([sig, sig], axis=1)
    stereo = stereo / (np.max(np.abs(stereo)) + 1e-9) * 0.9
    return stereo.astype(np.float32)


def _make_vocal(duration_s: float = 5.0) -> np.ndarray:
    """Synthetic vocal: filtered noise at speech formants."""
    from scipy.signal import butter, sosfilt
    rng = np.random.default_rng(1)
    noise = rng.standard_normal(int(SR * duration_s)).astype(np.float64)
    # Formant 1 at 700 Hz
    sos1 = butter(4, [600/SR*2, 800/SR*2], btype="band", output="sos")
    # Formant 2 at 1800 Hz
    sos2 = butter(4, [1600/SR*2, 2000/SR*2], btype="band", output="sos")
    vocal = sosfilt(sos1, noise) + sosfilt(sos2, noise) * 0.6
    vocal = vocal.astype(np.float32)
    vocal = vocal / (np.max(np.abs(vocal)) + 1e-9) * 0.7
    return np.stack([vocal, vocal], axis=1).astype(np.float32)


@pytest.fixture(scope="module")
def audio_pair(tmp_path_factory):
    """Write synthetic beat + vocal WAVs, return (beat_path, vocal_path)."""
    d = tmp_path_factory.mktemp("audio")
    beat_path  = str(d / "beat.wav")
    vocal_path = str(d / "vocal.wav")
    sf.write(beat_path,  _make_beat(),  SR, subtype="PCM_24")
    sf.write(vocal_path, _make_vocal(), SR, subtype="PCM_24")
    return beat_path, vocal_path


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

class TestListenScorer:
    """listen.py must be importable and produce valid output."""

    def test_import(self):
        import listen  # noqa: F401

    def test_score_any_synthetic(self, tmp_path):
        """score_any on synthetic audio returns a float score in [0, 100]."""
        import listen
        path = str(tmp_path / "test.wav")
        sig  = _make_beat(10.0)
        sf.write(path, sig, SR, subtype="PCM_24")
        cs, cg, det = listen.score_any(path)
        assert 0 <= cs <= 100, f"Unexpected chart_score: {cs}"
        assert isinstance(cg, str) and len(cg) >= 1
        assert isinstance(det, dict)

    def test_score_file_returns_tuple(self, tmp_path):
        import listen
        path = str(tmp_path / "beat.wav")
        sf.write(path, _make_beat(5.0), SR, subtype="PCM_24")
        result = listen.score_file(path, print_report=False)
        score, issues, metrics = result
        assert isinstance(score, (int, float))
        assert isinstance(metrics, dict)


class TestFuserIntegration:
    """Full fuse() round-trip on synthetic audio."""

    @skip_slow
    def test_fuse_produces_output(self, audio_pair, tmp_path):
        """fuse() must produce a non-silent WAV file."""
        try:
            import fuser as _fuser
        except ImportError:
            pytest.skip("fuser.py not importable (missing dependencies)")

        beat_path, vocal_path = audio_pair
        out_path = str(tmp_path / "mix.wav")
        stems    = str(tmp_path / "stems")
        os.makedirs(stems, exist_ok=True)

        result = _fuser.fuse(
            song_a=beat_path,
            song_b=vocal_path,
            out_path=out_path,
            stems_cache=stems,
            direct_vocal=True,
            seed=42,
        )
        assert result is not None
        radio = result.get("radio", out_path) if isinstance(result, dict) else out_path
        assert os.path.exists(radio), "Radio mix WAV not created"

        audio, fs = sf.read(radio)
        assert fs == SR
        assert audio.shape[0] > SR      # at least 1 second
        assert np.max(np.abs(audio)) > 0.01, "Output is silent"

    @skip_slow
    def test_fuse_score_above_threshold(self, audio_pair, tmp_path):
        """Fused synthetic audio should achieve chart_score > 0."""
        try:
            import fuser as _fuser
            import listen
        except ImportError:
            pytest.skip("Runtime dependencies not available")

        beat_path, vocal_path = audio_pair
        out_path = str(tmp_path / "mix2.wav")
        stems    = str(tmp_path / "stems2")
        os.makedirs(stems, exist_ok=True)

        result = _fuser.fuse(
            song_a=beat_path,
            song_b=vocal_path,
            out_path=out_path,
            stems_cache=stems,
            direct_vocal=True,
            seed=42,
        )
        radio = result.get("radio", out_path) if isinstance(result, dict) else out_path
        cs, cg, _ = listen.score_any(radio)
        assert cs > 0, f"chart_score should be > 0, got {cs}"
        print(f"\n[integration] Synthetic mix chart_score = {cs} ({cg})")


class TestOptimizerIntegration:
    """FuseOptimizer runs with a mocked fuse function."""

    def test_optimizer_converges(self):
        """Optimizer must call objective_fn at least once and return valid params."""
        from vf.optimizer import FuseOptimizer, PARAM_SPACE, PARAM_DEFAULTS

        call_counts = [0]

        def mock_fuse(params, attempt_n):
            call_counts[0] += 1
            # Score proportional to how close we are to defaults
            dist = sum(abs(params[k] - PARAM_DEFAULTS[k])
                       for k in PARAM_SPACE)
            score = max(0, min(100, 75 - int(dist * 5)))
            return score, {"spectral": score, "loudness": score,
                           "dynamics": score, "perceptual": score}

        opt = FuseOptimizer(PARAM_SPACE, PARAM_DEFAULTS)
        bp, bs, log = opt.optimize(mock_fuse, n_trials=3, target_score=99, seed=0)

        assert call_counts[0] >= 1
        assert 0 <= bs <= 100
        for k in PARAM_SPACE:
            assert k in bp
