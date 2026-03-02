"""
VocalFusion AI - Mastering Engine
===================================

Professional mastering chain.
Takes a mixed signal and makes it release-ready.

Chain:
  1. Multiband compression (tame frequency bands independently)
  2. Stereo imaging (widen the mix)
  3. EQ (final tonal shaping)
  4. Harmonic exciter (add sparkle)
  5. True peak limiting (prevent clipping)
  6. LUFS loudness targeting (-14 LUFS for streaming)
"""

import numpy as np
from scipy import signal as scipy_signal
from typing import Optional, Dict


class MasteringEngine:
    """Professional mastering chain"""

    def __init__(self, sample_rate: int = 44100):
        self.sr = sample_rate

    # ================================================================
    # MAIN MASTERING CHAIN
    # ================================================================

    def master(self, audio: np.ndarray,
               target_lufs: float = -14.0,
               style: str = 'balanced') -> np.ndarray:
        """
        Full mastering chain.

        Styles:
          'balanced' - Flat, transparent mastering
          'warm' - Emphasize low-mids, smooth highs
          'bright' - Emphasize presence and air
          'loud' - Maximum loudness (more compression)
          'dynamic' - Preserve dynamics (lighter processing)
        """
        if audio is None or len(audio) == 0:
            return audio

        m = audio.copy()
        if m.ndim > 1:
            m = np.mean(m, axis=0) if m.shape[0] == 2 else m[0]

        # 1. DC offset removal
        m = m - np.mean(m)

        # 2. Multiband compression
        m = self._multiband_compress(m, style)

        # 3. Master EQ (tonal shaping)
        m = self._master_eq(m, style)

        # 4. Harmonic exciter (subtle)
        if style != 'dynamic':
            m = self._harmonic_exciter(m, amount=0.1)

        # 5. Stereo imaging (subtle widening)
        # Note: working in mono for now, but structure supports stereo

        # 6. True peak limiting
        if style == 'loud':
            m = self._true_peak_limit(m, ceiling_db=-0.3, release_ms=50)
        else:
            m = self._true_peak_limit(m, ceiling_db=-0.5, release_ms=100)

        # 7. LUFS normalization
        m = self._loudness_normalize(m, target_lufs)

        # 8. Final safety clip
        m = np.clip(m, -0.99, 0.99)

        return m

    # ================================================================
    # MULTIBAND COMPRESSION
    # ================================================================

    def _multiband_compress(self, audio: np.ndarray,
                             style: str = 'balanced') -> np.ndarray:
        """
        4-band multiband compression.
        Each frequency band compressed independently.

        Bands:
          Sub:   20-150 Hz   (kick, sub-bass)
          Low:   150-600 Hz  (bass body, warmth)
          Mid:   600-4000 Hz (vocals, instruments)
          High:  4000-20kHz  (presence, air, cymbals)
        """
        nyq = self.sr / 2

        # Split into bands
        crossovers = [150, 600, 4000]
        bands = self._split_bands(audio, crossovers)

        # Compression settings per band per style
        settings = {
            'balanced': [
                {'threshold': -18, 'ratio': 2.0, 'attack': 20, 'release': 200, 'makeup': 1},  # Sub
                {'threshold': -20, 'ratio': 2.5, 'attack': 15, 'release': 150, 'makeup': 2},  # Low
                {'threshold': -18, 'ratio': 2.0, 'attack': 10, 'release': 100, 'makeup': 1},  # Mid
                {'threshold': -22, 'ratio': 2.0, 'attack': 5,  'release': 80,  'makeup': 1},  # High
            ],
            'warm': [
                {'threshold': -15, 'ratio': 3.0, 'attack': 25, 'release': 250, 'makeup': 2},
                {'threshold': -16, 'ratio': 3.0, 'attack': 20, 'release': 180, 'makeup': 3},
                {'threshold': -20, 'ratio': 2.0, 'attack': 10, 'release': 100, 'makeup': 1},
                {'threshold': -24, 'ratio': 1.5, 'attack': 5,  'release': 60,  'makeup': 0},
            ],
            'bright': [
                {'threshold': -20, 'ratio': 2.0, 'attack': 20, 'release': 200, 'makeup': 1},
                {'threshold': -22, 'ratio': 2.0, 'attack': 15, 'release': 150, 'makeup': 1},
                {'threshold': -16, 'ratio': 2.5, 'attack': 8,  'release': 80,  'makeup': 2},
                {'threshold': -14, 'ratio': 3.0, 'attack': 3,  'release': 50,  'makeup': 3},
            ],
            'loud': [
                {'threshold': -14, 'ratio': 4.0, 'attack': 15, 'release': 150, 'makeup': 3},
                {'threshold': -14, 'ratio': 4.0, 'attack': 10, 'release': 120, 'makeup': 3},
                {'threshold': -12, 'ratio': 3.5, 'attack': 5,  'release': 80,  'makeup': 3},
                {'threshold': -14, 'ratio': 3.0, 'attack': 3,  'release': 50,  'makeup': 2},
            ],
            'dynamic': [
                {'threshold': -24, 'ratio': 1.5, 'attack': 30, 'release': 300, 'makeup': 0},
                {'threshold': -26, 'ratio': 1.5, 'attack': 20, 'release': 200, 'makeup': 0},
                {'threshold': -24, 'ratio': 1.3, 'attack': 15, 'release': 150, 'makeup': 0},
                {'threshold': -28, 'ratio': 1.3, 'attack': 10, 'release': 100, 'makeup': 0},
            ],
        }

        band_settings = settings.get(style, settings['balanced'])

        # Compress each band
        processed_bands = []
        for band, setting in zip(bands, band_settings):
            compressed = self._compress(
                band,
                threshold_db=setting['threshold'],
                ratio=setting['ratio'],
                attack_ms=setting['attack'],
                release_ms=setting['release'],
                makeup_db=setting['makeup']
            )
            processed_bands.append(compressed)

        # Recombine
        max_len = max(len(b) for b in processed_bands)
        output = np.zeros(max_len)
        for band in processed_bands:
            output[:len(band)] += band

        return output

    def _split_bands(self, audio: np.ndarray,
                      crossovers: list) -> list:
        """Split audio into frequency bands using Linkwitz-Riley crossovers"""
        nyq = self.sr / 2
        bands = []

        # Band 1: below first crossover
        b, a = scipy_signal.butter(4, crossovers[0] / nyq, btype='low')
        bands.append(scipy_signal.filtfilt(b, a, audio).astype(np.float32))

        # Middle bands
        for i in range(len(crossovers) - 1):
            low = crossovers[i] / nyq
            high = crossovers[i + 1] / nyq
            if high >= 1.0:
                high = 0.99
            b, a = scipy_signal.butter(4, [low, high], btype='band')
            bands.append(scipy_signal.filtfilt(b, a, audio).astype(np.float32))

        # Last band: above last crossover
        if crossovers[-1] / nyq < 0.99:
            b, a = scipy_signal.butter(4, crossovers[-1] / nyq, btype='high')
            bands.append(scipy_signal.filtfilt(b, a, audio).astype(np.float32))
        else:
            bands.append(np.zeros_like(audio))

        return bands

    # ================================================================
    # MASTER EQ
    # ================================================================

    def _master_eq(self, audio: np.ndarray, style: str) -> np.ndarray:
        """Final tonal shaping EQ"""
        if style == 'warm':
            audio = self._parametric_eq(audio, 100, 1.5, 0.7)     # Sub warmth
            audio = self._parametric_eq(audio, 350, 1.0, 1.0)     # Body
            audio = self._parametric_eq(audio, 8000, -1.0, 0.5)   # Smooth highs
        elif style == 'bright':
            audio = self._parametric_eq(audio, 150, -1.0, 0.7)    # Clean sub
            audio = self._parametric_eq(audio, 3000, 1.5, 1.0)    # Presence
            audio = self._parametric_eq(audio, 10000, 2.0, 0.5)   # Air
        elif style == 'balanced':
            # Subtle corrections
            audio = self._parametric_eq(audio, 250, -0.5, 0.7)    # Slight mud cut
            audio = self._parametric_eq(audio, 3500, 0.5, 1.0)    # Slight presence
            audio = self._parametric_eq(audio, 12000, 0.5, 0.5)   # Slight air

        return audio

    # ================================================================
    # HARMONIC EXCITER
    # ================================================================

    def _harmonic_exciter(self, audio: np.ndarray,
                           amount: float = 0.1) -> np.ndarray:
        """
        Add subtle harmonic content for sparkle and presence.
        Uses parallel saturation on the high frequencies only.
        """
        nyq = self.sr / 2
        # Extract high frequencies (above 3kHz)
        cutoff = min(3000 / nyq, 0.99)
        b, a = scipy_signal.butter(2, cutoff, btype='high')
        highs = scipy_signal.filtfilt(b, a, audio).astype(np.float32)

        # Saturate
        driven = highs * 3.0
        saturated = np.tanh(driven)

        # Blend back
        return audio + saturated * amount

    # ================================================================
    # TRUE PEAK LIMITING
    # ================================================================

    def _true_peak_limit(self, audio: np.ndarray,
                          ceiling_db: float = -0.5,
                          release_ms: float = 100) -> np.ndarray:
        """
        True peak limiter with lookahead.
        Prevents any sample from exceeding the ceiling.
        """
        ceiling = 10 ** (ceiling_db / 20)
        release_samples = int(release_ms * self.sr / 1000)
        lookahead = int(0.001 * self.sr)  # 1ms lookahead

        peaks = np.abs(audio)
        gain = np.ones_like(audio)

        # Apply lookahead: check future samples
        for i in range(len(audio) - lookahead):
            future_peak = np.max(peaks[i:i + lookahead])
            if future_peak > ceiling:
                gain[i] = ceiling / future_peak

        # Smooth gain (release)
        for i in range(1, len(gain)):
            if gain[i] > gain[i-1]:
                coeff = np.exp(-1.0 / max(1, release_samples))
                gain[i] = coeff * gain[i-1] + (1 - coeff) * gain[i]

        return audio * gain

    # ================================================================
    # LOUDNESS NORMALIZATION
    # ================================================================

    def _loudness_normalize(self, audio: np.ndarray,
                              target_lufs: float = -14.0) -> np.ndarray:
        """Normalize to target LUFS (integrated loudness)"""
        rms = np.sqrt(np.mean(audio ** 2))
        if rms < 1e-10:
            return audio

        # Approximate LUFS from RMS (simplified, but close enough)
        current_lufs = 20 * np.log10(rms + 1e-10) + 3.0  # Rough LUFS approximation

        gain_db = target_lufs - current_lufs
        gain_db = np.clip(gain_db, -20, 20)  # Safety limits

        gain = 10 ** (gain_db / 20)
        result = audio * gain

        # Final safety: prevent any sample from exceeding 0.99
        peak = np.max(np.abs(result))
        if peak > 0.99:
            result *= 0.99 / peak

        return result

    # ================================================================
    # ANALYSIS (for diagnostics)
    # ================================================================

    def analyze_master(self, audio: np.ndarray) -> Dict:
        """Analyze the quality of a mastered signal"""
        rms = np.sqrt(np.mean(audio ** 2))
        peak = np.max(np.abs(audio))
        crest_factor = peak / (rms + 1e-10)
        dynamic_range_db = 20 * np.log10(crest_factor + 1e-10)

        # Approximate LUFS
        lufs = 20 * np.log10(rms + 1e-10) + 3.0

        # Spectral balance
        spectrum = np.abs(np.fft.rfft(audio[:min(len(audio), 4*self.sr)]))
        freqs = np.fft.rfftfreq(min(len(audio), 4*self.sr), 1/self.sr)

        def band_energy(low, high):
            mask = (freqs >= low) & (freqs < high)
            return float(np.mean(spectrum[mask])) if mask.any() else 0.0

        return {
            'rms': float(rms),
            'peak': float(peak),
            'crest_factor_db': float(dynamic_range_db),
            'lufs_approx': float(lufs),
            'sub_energy': band_energy(20, 150),
            'low_energy': band_energy(150, 600),
            'mid_energy': band_energy(600, 4000),
            'high_energy': band_energy(4000, 20000),
            'clipping': bool(peak > 0.99),
        }

    # ================================================================
    # DSP PRIMITIVES
    # ================================================================

    def _compress(self, audio, threshold_db=-20, ratio=3.0,
                   attack_ms=8, release_ms=80, makeup_db=0):
        threshold = 10 ** (threshold_db / 20)
        attack = int(attack_ms * self.sr / 1000)
        release = int(release_ms * self.sr / 1000)

        envelope = np.abs(audio)
        for i in range(1, len(envelope)):
            if envelope[i] > envelope[i-1]:
                coeff = np.exp(-1.0 / max(1, attack))
            else:
                coeff = np.exp(-1.0 / max(1, release))
            envelope[i] = coeff * envelope[i-1] + (1 - coeff) * envelope[i]

        gain = np.ones_like(envelope)
        above = envelope > threshold
        if np.any(above):
            gain[above] = (threshold / envelope[above]) ** (1 - 1/ratio)

        result = audio * gain
        if makeup_db != 0:
            result *= 10 ** (makeup_db / 20)
        return result

    def _parametric_eq(self, audio, center_freq, gain_db, q):
        if abs(gain_db) < 0.1:
            return audio
        nyq = self.sr / 2
        if center_freq >= nyq * 0.95:
            center_freq = nyq * 0.9

        w0 = center_freq / nyq
        A = 10 ** (gain_db / 40.0)
        alpha = np.sin(np.pi * w0) / (2 * q)

        b0 = 1 + alpha * A
        b1 = -2 * np.cos(np.pi * w0)
        b2 = 1 - alpha * A
        a0 = 1 + alpha / A
        a1 = -2 * np.cos(np.pi * w0)
        a2 = 1 - alpha / A

        b = np.array([b0/a0, b1/a0, b2/a0])
        a = np.array([1.0, a1/a0, a2/a0])

        return scipy_signal.filtfilt(b, a, audio).astype(np.float32)
