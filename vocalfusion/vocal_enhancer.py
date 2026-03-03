"""
VocalFusion AI — VocalEnhancer (pedalboard edition)
====================================================

Uses Spotify's `pedalboard` library (JUCE C++ engine — same as pro DAWs)
for vocal processing. Much cleaner than scipy filters.

Chain applied to vocals:
  1. noisereduce  — spectral gate removes Demucs stem bleed
  2. NoiseGate    — kills silence between phrases
  3. HighpassFilter 100Hz — remove sub-bass rumble
  4. LowShelfFilter 250Hz -3dB — reduce muddiness
  5. PeakFilter 400Hz -2dB — cut honky/boxy resonance
  6. PeakFilter 3000Hz +3dB — presence / intelligibility (this is the money)
  7. HighShelfFilter 10kHz +2dB — air / openness
  8. Compressor — gentle glue, keeps loudness even
  9. Limiter — safety ceiling

EQ carving on instruments:
  - "other" stem: cut 3kHz with PeakFilter to create pocket for vocals
  - "bass" stem: minor cut at 3kHz (bass doesn't need the presence band)

Sidechain:
  - Stronger than before: drums 0.22, bass 0.28, other 0.18
"""

import numpy as np
import noisereduce as nr

from pedalboard import (
    Pedalboard,
    NoiseGate,
    HighpassFilter,
    LowShelfFilter,
    PeakFilter,
    HighShelfFilter,
    Compressor,
    Limiter,
)
from vocalfusion.dsp import EnhancedDSP


class VocalEnhancer:
    """
    Professional vocal processing using Spotify's pedalboard (JUCE engine).
    """

    def __init__(self, sample_rate: int = 44100):
        self.sr = sample_rate
        self.dsp = EnhancedDSP(sample_rate)

        # Vocal chain — built once, reused
        self._vocal_board = Pedalboard([
            NoiseGate(threshold_db=-42, ratio=2.0,
                      attack_ms=2.0, release_ms=200.0),
            HighpassFilter(cutoff_frequency_hz=100.0),
            LowShelfFilter(cutoff_frequency_hz=250.0, gain_db=-3.0),
            PeakFilter(cutoff_frequency_hz=400.0, gain_db=-2.0, q=1.0),
            PeakFilter(cutoff_frequency_hz=3000.0, gain_db=3.0, q=1.2),
            HighShelfFilter(cutoff_frequency_hz=10000.0, gain_db=2.0),
            Compressor(threshold_db=-18.0, ratio=3.0,
                       attack_ms=10.0, release_ms=100.0),
            Limiter(threshold_db=-1.0, release_ms=100.0),
        ])

        # Carving chain for "other" instruments — cuts where vocals live
        self._other_board = Pedalboard([
            PeakFilter(cutoff_frequency_hz=3000.0, gain_db=-5.0, q=0.9),
        ])

        # Minor carve for bass
        self._bass_board = Pedalboard([
            PeakFilter(cutoff_frequency_hz=3000.0, gain_db=-3.0, q=1.0),
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

        # Step 1 — denoise + EQ vocals with pedalboard
        if has_vocals:
            vocals = self._process_vocals(vocals)

        # Step 2 — EQ-carve instruments
        if other is not None and np.any(other != 0):
            other = self._process_other(other)
        if bass is not None and np.any(bass != 0):
            bass = self._process_bass(bass)

        # Step 3 — Sidechain: instruments duck when vocals are active
        if has_vocals:
            if drums is not None:
                drums = self.dsp.sidechain_duck(
                    drums, vocals,
                    threshold_db=-22, ratio=3.0,
                    attack_ms=8, release_ms=180, amount=0.22)
            if bass is not None:
                bass = self.dsp.sidechain_duck(
                    bass, vocals,
                    threshold_db=-22, ratio=3.0,
                    attack_ms=6, release_ms=120, amount=0.28)
            if other is not None:
                other = self.dsp.sidechain_duck(
                    other, vocals,
                    threshold_db=-22, ratio=2.5,
                    attack_ms=10, release_ms=200, amount=0.18)

        return vocals, drums, bass, other

    # ----------------------------------------------------------------

    def _process_vocals(self, vocals: np.ndarray) -> np.ndarray:
        """Denoise → pedalboard EQ/compression chain"""
        # Spectral noise reduction first — cleans up Demucs separation bleed
        # Use non-stationary mode (adapts over time — better for singing)
        try:
            vocals = nr.reduce_noise(
                y=vocals, sr=self.sr,
                stationary=False,
                prop_decrease=0.70,       # Reduce 70% of noise (leave some warmth)
                freq_mask_smooth_hz=300,
                time_mask_smooth_ms=50,
            )
        except Exception:
            pass  # If noisereduce fails, continue without it

        # pedalboard expects shape (channels, samples) or (samples,)
        # Our stems are mono 1D arrays
        processed = self._vocal_board(vocals.astype(np.float32), self.sr)
        return processed.astype(np.float64)

    def _process_other(self, other: np.ndarray) -> np.ndarray:
        processed = self._other_board(other.astype(np.float32), self.sr)
        return processed.astype(np.float64)

    def _process_bass(self, bass: np.ndarray) -> np.ndarray:
        processed = self._bass_board(bass.astype(np.float32), self.sr)
        return processed.astype(np.float64)
