"""
Unit tests for audio utility functions in fuser.py.
No audio files required — all tests use synthetic numpy arrays.
"""

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

# ---------------------------------------------------------------------------
# Helpers to avoid importing all of fuser.py (heavy model init) when possible
# ---------------------------------------------------------------------------

def _import_fuser_utils():
    """Import only the pure-numpy utilities from fuser.py."""
    import types

    # Stub out ONLY heavy ML/model imports — NOT scientific libraries like scipy.
    # scipy, librosa, soundfile are real deps available in the test environment.
    stubs = [
        "torch", "torchaudio",
        "demucs", "demucs.apply", "demucs.pretrained",
        "pedalboard",
        "anthropic",
        "noisereduce",
        "audio_separator",
        "onnxruntime",
        "psutil",
        "df", "df.enhance",
    ]
    for mod in stubs:
        if mod not in sys.modules:
            sys.modules[mod] = types.ModuleType(mod)

    # pedalboard needs specific attributes fuser references at import time
    import sys as _sys
    pb = _sys.modules["pedalboard"]
    for cls_name in ["Compressor", "HighpassFilter", "HighShelfFilter",
                     "LowShelfFilter", "NoiseGate", "PeakFilter",
                     "Pedalboard", "Reverb", "Limiter"]:
        if not hasattr(pb, cls_name):
            setattr(pb, cls_name, type(cls_name, (), {"__init__": lambda self, **kw: None}))
    if not hasattr(pb, "time_stretch"):
        pb.time_stretch = lambda *a, **kw: None

    import pyloudnorm  # real dep — ensure it's available
    import fuser  # noqa: F401 — side effects load constants
    return fuser


# ---------------------------------------------------------------------------
# M/S encode/decode
# ---------------------------------------------------------------------------

class TestMSEncodeDecode:
    def setup_method(self):
        self.fuser = _import_fuser_utils()

    def test_roundtrip_silence(self):
        silence = np.zeros((1024, 2), dtype=np.float32)
        M, S = self.fuser._ms_encode(silence)
        out = self.fuser._ms_decode(M, S)
        np.testing.assert_allclose(out, silence, atol=1e-6)

    def test_roundtrip_noise(self):
        rng = np.random.default_rng(42)
        audio = rng.standard_normal((4096, 2)).astype(np.float32) * 0.5
        M, S = self.fuser._ms_encode(audio)
        out = self.fuser._ms_decode(M, S)
        np.testing.assert_allclose(out, audio, atol=1e-5)

    def test_mono_signal_has_zero_side(self):
        """Pure mono (L==R) should have S channel = 0."""
        mono = np.random.default_rng(7).standard_normal(2048).astype(np.float32)
        stereo = np.stack([mono, mono], axis=1)
        _, S = self.fuser._ms_encode(stereo)
        np.testing.assert_allclose(S, 0.0, atol=1e-6)

    def test_out_of_phase_has_zero_mid(self):
        """L = -R (fully out of phase) should give M channel = 0."""
        rng = np.random.default_rng(3)
        ch = rng.standard_normal(2048).astype(np.float32)
        stereo = np.stack([ch, -ch], axis=1)
        M, _ = self.fuser._ms_encode(stereo)
        np.testing.assert_allclose(M, 0.0, atol=1e-6)


# ---------------------------------------------------------------------------
# _to_mono
# ---------------------------------------------------------------------------

class TestToMono:
    def setup_method(self):
        self.fuser = _import_fuser_utils()

    def test_stereo_averages_channels(self):
        stereo = np.array([[1.0, 3.0], [2.0, 4.0]], dtype=np.float32)
        mono = self.fuser._to_mono(stereo)
        np.testing.assert_allclose(mono, [2.0, 3.0], atol=1e-6)

    def test_mono_passthrough(self):
        audio = np.array([0.5, -0.3, 0.1], dtype=np.float32)
        out = self.fuser._to_mono(audio)
        np.testing.assert_allclose(out, audio, atol=1e-6)

    def test_output_is_float32(self):
        stereo = np.ones((10, 2), dtype=np.float64)
        out = self.fuser._to_mono(stereo)
        assert out.dtype == np.float32


# ---------------------------------------------------------------------------
# _rms
# ---------------------------------------------------------------------------

class TestRms:
    def setup_method(self):
        self.fuser = _import_fuser_utils()

    def test_known_signal(self):
        """RMS of a unit-amplitude sine = 1/sqrt(2) ≈ 0.707."""
        t = np.linspace(0, 1, 44100, endpoint=False)
        sine = np.sin(2 * np.pi * 440 * t).astype(np.float32)
        rms = self.fuser._rms(sine)
        assert abs(rms - 1.0 / np.sqrt(2)) < 0.002

    def test_silence_near_zero(self):
        silence = np.zeros(1024, dtype=np.float32)
        assert self.fuser._rms(silence) < 1e-5

    def test_always_positive(self):
        rng = np.random.default_rng(0)
        for _ in range(10):
            y = rng.standard_normal(1024).astype(np.float32)
            assert self.fuser._rms(y) > 0


# ---------------------------------------------------------------------------
# _best_ratio (BPM stretch ratio)
# ---------------------------------------------------------------------------

class TestBestRatio:
    def setup_method(self):
        self.fuser = _import_fuser_utils()

    def test_same_bpm_gives_one(self):
        ratio = self.fuser._best_ratio(120.0, 120.0)
        assert abs(ratio - 1.0) < 0.01

    def test_double_bpm(self):
        """bpm_a=120, bpm_b=60 → stretch by 2.0 (or pick octave-corrected)."""
        ratio = self.fuser._best_ratio(120.0, 60.0)
        # Should be 2.0 OR 1.0 (octave-corrected to same) — either is fine
        assert ratio in (pytest.approx(2.0, abs=0.05), pytest.approx(1.0, abs=0.05))

    def test_ratio_within_allowed_range(self):
        """Stretch ratio should stay in [0.5, 2.0] after clamping."""
        for bpm_a, bpm_b in [(80, 160), (140, 70), (95, 130), (200, 80)]:
            ratio = self.fuser._best_ratio(float(bpm_a), float(bpm_b))
            assert 0.4 <= ratio <= 2.1, f"ratio={ratio} out of range for {bpm_a}/{bpm_b}"


# ---------------------------------------------------------------------------
# Semitone / key shift math
# ---------------------------------------------------------------------------

class TestSemitones:
    def setup_method(self):
        self.fuser = _import_fuser_utils()

    def test_same_key_zero_shift(self):
        assert self.fuser.semitones_to_shift(0, "major", 0, "major") == 0

    def test_shift_is_int(self):
        result = self.fuser.semitones_to_shift(0, "major", 7, "major")
        assert isinstance(result, int)

    def test_shift_within_octave(self):
        """Shift should always be within [-6, 6] (closest path)."""
        for root in range(12):
            shift = self.fuser.semitones_to_shift(0, "major", root, "major")
            assert -6 <= shift <= 6, f"shift={shift} for root={root}"
