"""
VocalFusion AI — VocalEnhancer
================================

EQ and dynamics processing to make vocals cut through the mix clearly.

Techniques applied:
  1. High-pass filter vocals at 100Hz  (remove sub-bass rumble)
  2. Cut 300Hz from vocals             (reduce muddy low-mids)
  3. Boost 2.5kHz on vocals            (presence / intelligibility)
  4. Boost 8kHz on vocals              (air / openness)
  5. Cut 2.5kHz from "other" stem      (EQ carve — creates a pocket for vocals)
  6. Cut 2.5kHz from bass stem         (bass shouldn't fight vocal clarity)
  7. Sidechain compression: drums, bass, other duck when vocals are active
     (stronger than the old light sidechain — creates real space)

The single biggest improvement for buried vocals is #5 (EQ carving).
Cutting 4-6dB at 2.5kHz in the instrumental creates a frequency "pocket"
that the vocals occupy. This is standard professional mixing practice.
"""

import numpy as np
from vocalfusion.dsp import EnhancedDSP


class VocalEnhancer:
    """
    Processes vocals and instrument stems to maximise vocal clarity.

    Call process() after gain staging — it expects stems already at
    their target RMS levels.
    """

    def __init__(self, sample_rate: int = 44100):
        self.sr = sample_rate
        self.dsp = EnhancedDSP(sample_rate)

    def process(self,
                vocals: np.ndarray,
                drums: np.ndarray,
                bass: np.ndarray,
                other: np.ndarray):
        """
        Enhance vocals and carve space in the instrumental stems.

        Args:
            vocals: vocal stem (mono, at target RMS level)
            drums:  drum stem
            bass:   bass stem
            other:  melodic/harmonic instruments stem

        Returns:
            (vocals, drums, bass, other) — all enhanced
        """
        has_vocals = vocals is not None and np.any(vocals != 0)

        # Step 1 — Process vocals
        if has_vocals:
            vocals = self._process_vocals(vocals)

        # Step 2 — EQ-carve instruments to make a pocket for vocals
        if other is not None:
            other = self._carve_other(other)
        if bass is not None:
            bass = self._carve_bass(bass)

        # Step 3 — Sidechain: instruments duck when vocals are active
        if has_vocals:
            if drums is not None:
                drums = self.dsp.sidechain_duck(
                    drums, vocals,
                    threshold_db=-22, ratio=2.5,
                    attack_ms=8, release_ms=180, amount=0.20)
            if bass is not None:
                bass = self.dsp.sidechain_duck(
                    bass, vocals,
                    threshold_db=-22, ratio=2.5,
                    attack_ms=6, release_ms=120, amount=0.25)
            if other is not None:
                other = self.dsp.sidechain_duck(
                    other, vocals,
                    threshold_db=-22, ratio=2.0,
                    attack_ms=10, release_ms=200, amount=0.15)

        return vocals, drums, bass, other

    # ----------------------------------------------------------------

    def _process_vocals(self, vocals: np.ndarray) -> np.ndarray:
        """Vocal EQ chain: clean, then add presence"""
        # Remove sub-bass rumble and proximity effect
        vocals = self.dsp.highpass(vocals, 100)
        # Cut boomy low-mids that make vocals sound thick / honky
        vocals = self.dsp.parametric_eq(vocals, 300, -1.5, q=0.8)
        # Presence boost — the 2-3kHz range is where intelligibility lives
        vocals = self.dsp.parametric_eq(vocals, 2500, 2.0, q=1.2)
        # Air boost — adds openness without harshness
        vocals = self.dsp.parametric_eq(vocals, 8000, 1.5, q=0.8)
        return vocals

    def _carve_other(self, other: np.ndarray) -> np.ndarray:
        """Cut the vocal presence band from melodic instruments.

        This is the most effective technique for making vocals cut through:
        remove 4-6dB at 2.5kHz from the instruments so the vocal boost at
        the same frequency has uncontested space.
        """
        return self.dsp.parametric_eq(other, 2500, -5.0, q=0.9)

    def _carve_bass(self, bass: np.ndarray) -> np.ndarray:
        """Minor presence cut from bass — bass doesn't need 2-4kHz anyway"""
        return self.dsp.parametric_eq(bass, 2500, -3.0, q=1.0)
