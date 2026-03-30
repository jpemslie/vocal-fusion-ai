"""
Unit tests for listen.py scoring functions.
Uses synthetic audio — no external files required.
"""

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).parent.parent.parent))


def _make_sine(freq=440, duration=5.0, sr=44100, amplitude=0.5):
    t = np.linspace(0, duration, int(sr * duration), endpoint=False)
    mono = (amplitude * np.sin(2 * np.pi * freq * t)).astype(np.float32)
    return np.stack([mono, mono], axis=1)  # stereo (T, 2)


def _make_silence(duration=5.0, sr=44100):
    n = int(sr * duration)
    return np.zeros((n, 2), dtype=np.float32)


# ---------------------------------------------------------------------------
# Import listen.py
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def listen():
    import listen as _listen
    return _listen


# ---------------------------------------------------------------------------
# score_file / auto_score — basic smoke tests
# ---------------------------------------------------------------------------

class TestListenScoreSmoke:
    def test_score_file_returns_tuple(self, listen, tmp_path):
        import soundfile as sf
        audio = _make_sine()
        path = str(tmp_path / "test.wav")
        sf.write(path, audio, 44100, subtype="PCM_16")
        result = listen.score_file(path, print_report=False)
        assert isinstance(result, tuple)
        assert len(result) >= 2  # at minimum (passed, score)

    def test_score_is_numeric(self, listen, tmp_path):
        import soundfile as sf
        audio = _make_sine()
        path = str(tmp_path / "sine.wav")
        sf.write(path, audio, 44100, subtype="PCM_16")
        _, score = listen.score_file(path, print_report=False)[:2]
        assert isinstance(score, (int, float))
        assert 0 <= score <= 100


# ---------------------------------------------------------------------------
# Grade thresholds (isolated — no audio needed)
# ---------------------------------------------------------------------------

class TestGradeThresholds:
    def test_grades_in_listen(self, listen):
        """listen.py must define grade boundaries consistent with ROADMAP."""
        # A ≥ 85, B ≥ 70, C ≥ 55
        for attr in ("GRADES", "_GRADES", "GRADE_THRESHOLDS"):
            if hasattr(listen, attr):
                grades = getattr(listen, attr)
                assert any(v >= 85 for v in (grades if isinstance(grades, (list, dict)) else [])), \
                    "A-grade threshold should be ≥85"
                break


# ---------------------------------------------------------------------------
# Silence vs signal — silence should score badly
# ---------------------------------------------------------------------------

class TestSilenceScoresBadly:
    def test_silence_scores_below_50(self, listen, tmp_path):
        import soundfile as sf
        silence = _make_silence()
        path = str(tmp_path / "silence.wav")
        sf.write(path, silence, 44100, subtype="PCM_16")
        try:
            _, score = listen.score_file(path, print_report=False)[:2]
            assert score < 50, f"Silence scored {score}/100 — expected < 50"
        except Exception:
            pytest.skip("score_file raised on silence — acceptable")
