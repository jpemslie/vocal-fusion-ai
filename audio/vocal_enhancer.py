"""
VocalFusion AI — VocalEnhancer
================================

Minimal, transparent vocal processing. The rule: less is more.
Demucs already degraded the signal once. Every extra processing step
adds more artifacts. Only apply what genuinely helps.

Vocal chain (pedalboard / JUCE):
  1. HighpassFilter 100Hz  — remove sub-bass rumble (always safe)
  2. PeakFilter 2500Hz +2dB — subtle presence lift (not 3kHz which is harsh)
  3. Compressor (gentle)    — even out loud/quiet words

Instrument carving:
  - "other" stem: cut -4dB at 2.5kHz to make a pocket for vocals

Sidechain:
  - Drums and bass duck slightly when vocals are active
"""

import numpy as np

from pedalboard import (
    Pedalboard,
    HighpassFilter,
    PeakFilter,
)
from audio.dsp import EnhancedDSP


class VocalEnhancer:
    """Transparent vocal processing — minimal processing, maximum clarity."""

    def __init__(self, sample_rate: int = 44100):
        self.sr = sample_rate
        self.dsp = EnhancedDSP(sample_rate)

        # Clarity chain — three moves, nothing more:
        #   1. HPF at 100Hz: Demucs leaks sub-bass mud into the vocal stem.
        #      100Hz is safe (no vocal content below this) and removes all mud.
        #   2. Cut 300Hz -3dB: removes boxy/nasal honk from separation artefacts.
        #   3. Boost 3kHz +4dB: the primary "cut-through" frequency for vocals —
        #      this is what makes a vocal sit forward in a mix.
        #   4. Boost 9kHz +2.5dB: air/presence — adds crispness and life.
        # No compressor — _set_rms() in AIDJ already normalises level.
        self._vocal_board = Pedalboard([
            HighpassFilter(cutoff_frequency_hz=100.0),
            PeakFilter(cutoff_frequency_hz=300.0,  gain_db=-3.0, q=1.0),
            PeakFilter(cutoff_frequency_hz=3000.0, gain_db=2.0,  q=0.8),
            PeakFilter(cutoff_frequency_hz=9000.0, gain_db=2.5,  q=0.7),
        ])

        # Carve a pocket for vocals in the melodic instruments
        self._other_board = Pedalboard([
            PeakFilter(cutoff_frequency_hz=2500.0, gain_db=-4.0, q=0.9),
        ])

    def process(self,
                vocals: np.ndarray,
                drums: np.ndarray,
                bass: np.ndarray,
                other: np.ndarray):
        """
        Enhance vocals and carve space in the instrumental stems.
        Returns: (vocals, drums, bass, other)
        """
        has_vocals = vocals is not None and np.any(vocals != 0)

        if has_vocals:
            vocals = self._process_vocals(vocals)

        if other is not None and np.any(other != 0):
            other = self._process_other(other)

        # Sidechain: instruments duck subtly when vocals are loud.
        # Threshold raised to -16dBFS so ducking only triggers on loud
        # vocal peaks, not constantly. Amounts reduced to avoid hollow sound.
        if has_vocals:
            if drums is not None:
                drums = self.dsp.sidechain_duck(
                    drums, vocals,
                    threshold_db=-16, ratio=2.0,
                    attack_ms=10, release_ms=250, amount=0.10)
            if bass is not None:
                bass = self.dsp.sidechain_duck(
                    bass, vocals,
                    threshold_db=-16, ratio=2.0,
                    attack_ms=10, release_ms=200, amount=0.12)
            if other is not None:
                other = self.dsp.sidechain_duck(
                    other, vocals,
                    threshold_db=-16, ratio=1.8,
                    attack_ms=12, release_ms=250, amount=0.08)

        return vocals, drums, bass, other

    def _process_vocals(self, vocals: np.ndarray) -> np.ndarray:
        try:
            processed = self._vocal_board(vocals.astype(np.float32), self.sr)
            processed = np.nan_to_num(processed, nan=0.0, posinf=0.0, neginf=0.0)
            return processed.astype(np.float64)
        except Exception:
            return vocals

    def _process_other(self, other: np.ndarray) -> np.ndarray:
        try:
            processed = self._other_board(other.astype(np.float32), self.sr)
            processed = np.nan_to_num(processed, nan=0.0, posinf=0.0, neginf=0.0)
            return processed.astype(np.float64)
        except Exception:
            return other
