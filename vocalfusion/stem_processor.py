"""
VocalFusion AI - Stem Processor
=================================

Professional per-stem processing chains.
Each stem type gets its own dedicated processing chain
optimized for that instrument type.

Features:
  - Vocal chain: de-ess, compress, EQ presence/air, reverb
  - Drum chain: transient shaping, parallel compression, punch EQ
  - Bass chain: sub harmonics, compression, frequency splitting
  - Instrument chain: stereo widening, mid-side EQ, spatial placement
  - Frequency-aware processing that adapts to the other stems
  - Stem-to-stem frequency carving to prevent masking
"""

import numpy as np
from scipy import signal as scipy_signal
from typing import Dict, Optional, Tuple
import librosa


class StemProcessor:
    """Professional per-stem processing"""

    def __init__(self, sample_rate: int = 44100):
        self.sr = sample_rate

    # ================================================================
    # VOCAL PROCESSING CHAIN
    # ================================================================

    def process_vocals(self, vocals: np.ndarray,
                        instrumental: Optional[np.ndarray] = None,
                        style: str = 'modern') -> np.ndarray:
        """
        Professional vocal processing chain.

        Chain order (same as any professional mix):
        1. HPF (remove rumble)
        2. De-ess (tame sibilance)
        3. Compression (even out dynamics)
        4. EQ (presence, air, warmth)
        5. Frequency carving (avoid masking with instrumental)
        6. Saturation (subtle warmth)
        7. Reverb (space)
        8. Final level
        """
        if vocals is None:
            return None

        v = vocals.copy()
        if v.ndim > 1:
            v = librosa.to_mono(v)

        # 1. High-pass filter — nothing below 80Hz is vocal
        v = self._highpass(v, 80, order=4)

        # 2. De-ess — reduce harsh sibilance (5-9kHz)
        v = self._deess(v, threshold_db=-20, frequency=6500)

        # 3. Compression — smooth out dynamics
        if style == 'aggressive':
            v = self._compress(v, threshold_db=-16, ratio=4.0,
                              attack_ms=3, release_ms=50, makeup_db=4)
        else:
            v = self._compress(v, threshold_db=-20, ratio=3.0,
                              attack_ms=8, release_ms=80, makeup_db=3)

        # 4. EQ — shape the vocal tone
        v = self._parametric_eq(v, 200, -1.5, 1.0)    # Cut mud
        v = self._parametric_eq(v, 800, 1.0, 1.5)     # Body/warmth
        v = self._parametric_eq(v, 3000, 3.0, 1.0)    # Presence (the "cut through" frequency)
        v = self._parametric_eq(v, 5000, 1.5, 1.5)    # Clarity
        v = self._parametric_eq(v, 12000, 2.0, 0.7)   # Air

        # 5. Frequency carving against instrumental
        if instrumental is not None:
            v = self._carve_against(v, instrumental, 'vocals')

        # 6. Subtle saturation (warmth)
        v = self._saturate(v, drive=0.15)

        # 7. Reverb — just enough to blend into the mix
        v = self._reverb(v, decay=1.2, mix=0.08)

        return v

    # ================================================================
    # DRUM PROCESSING CHAIN
    # ================================================================

    def process_drums(self, drums: np.ndarray,
                       style: str = 'punchy') -> np.ndarray:
        """
        Professional drum processing chain.

        Chain:
        1. Transient shaping (more snap)
        2. Parallel compression (glue + punch)
        3. EQ (sub, punch, snap, air)
        4. Gate (reduce bleed)
        """
        if drums is None:
            return None

        d = drums.copy()
        if d.ndim > 1:
            d = librosa.to_mono(d)

        # 1. Transient shaping — make drums snappier
        d = self._transient_shape(d, attack_boost=1.3, sustain_cut=0.85)

        # 2. Parallel compression — NYC-style drum crush
        compressed = self._compress(d.copy(), threshold_db=-15, ratio=8.0,
                                    attack_ms=1, release_ms=30, makeup_db=6)
        d = d * 0.7 + compressed * 0.3  # Blend parallel

        # 3. EQ
        if style == 'punchy':
            d = self._parametric_eq(d, 60, 3.0, 1.5)     # Sub thump
            d = self._parametric_eq(d, 100, 2.0, 1.0)    # Kick body
            d = self._parametric_eq(d, 400, -2.0, 1.0)   # Cut boxiness
            d = self._parametric_eq(d, 3000, 1.5, 2.0)   # Snare crack
            d = self._parametric_eq(d, 8000, 1.0, 1.0)   # Hi-hat presence
            d = self._parametric_eq(d, 12000, 1.5, 0.7)  # Air/shimmer
        elif style == 'tight':
            d = self._parametric_eq(d, 50, 2.0, 2.0)     # Focused sub
            d = self._parametric_eq(d, 200, -1.5, 1.0)   # Clean low-mids
            d = self._parametric_eq(d, 5000, 2.0, 1.5)   # Attack
            d = self._parametric_eq(d, 10000, 1.0, 0.7)  # Crisp

        # 4. Light limiting to prevent clipping
        d = self._limit(d, threshold_db=-1.0)

        return d

    # ================================================================
    # BASS PROCESSING CHAIN
    # ================================================================

    def process_bass(self, bass: np.ndarray,
                      drums: Optional[np.ndarray] = None,
                      style: str = 'deep') -> np.ndarray:
        """
        Professional bass processing chain.

        Chain:
        1. HPF (remove sub-rumble below 30Hz)
        2. Compression (even out the low end)
        3. EQ (shape the tone)
        4. Sidechain duck to drums (make room for kick)
        5. Saturation (harmonics for small speakers)
        """
        if bass is None:
            return None

        b = bass.copy()
        if b.ndim > 1:
            b = librosa.to_mono(b)

        # 1. HPF — nothing useful below 30Hz
        b = self._highpass(b, 30, order=2)

        # 2. Compression — bass needs to be rock-solid
        b = self._compress(b, threshold_db=-18, ratio=4.0,
                          attack_ms=10, release_ms=100, makeup_db=3)

        # 3. EQ
        if style == 'deep':
            b = self._parametric_eq(b, 60, 2.0, 1.5)     # Sub weight
            b = self._parametric_eq(b, 150, 1.5, 1.0)    # Fundamental
            b = self._parametric_eq(b, 500, -2.0, 1.0)   # Cut mud
            b = self._parametric_eq(b, 1200, 1.0, 2.0)   # Definition
        elif style == 'punchy':
            b = self._parametric_eq(b, 80, 2.5, 1.0)     # Punch
            b = self._parametric_eq(b, 250, -1.0, 1.0)   # Clean
            b = self._parametric_eq(b, 700, 1.5, 1.5)    # Growl
            b = self._parametric_eq(b, 2000, 1.0, 1.0)   # Pick/pluck

        # 4. Sidechain to drums — duck when kick hits
        if drums is not None:
            b = self._sidechain_duck(b, drums, threshold_db=-22,
                                     ratio=2.5, attack_ms=3, release_ms=120)

        # 5. Subtle saturation for harmonic content
        b = self._saturate(b, drive=0.1)

        # 6. LPF — nothing above 8kHz is useful for bass
        b = self._lowpass(b, 8000, order=2)

        return b

    # ================================================================
    # INSTRUMENT/OTHER PROCESSING CHAIN
    # ================================================================

    def process_other(self, other: np.ndarray,
                       vocals: Optional[np.ndarray] = None,
                       style: str = 'wide') -> np.ndarray:
        """
        Process instrumental/synth/guitar/keys stem.

        Chain:
        1. HPF (remove rumble)
        2. Light compression
        3. EQ (clarity, remove mud)
        4. Frequency carving against vocals
        5. Stereo widening
        """
        if other is None:
            return None

        o = other.copy()
        if o.ndim > 1:
            o = librosa.to_mono(o)

        # 1. HPF
        o = self._highpass(o, 100, order=2)

        # 2. Light compression
        o = self._compress(o, threshold_db=-22, ratio=2.0,
                          attack_ms=15, release_ms=120, makeup_db=2)

        # 3. EQ
        o = self._parametric_eq(o, 250, -2.0, 1.0)    # Cut mud
        o = self._parametric_eq(o, 800, -1.0, 1.5)    # Reduce vocal-range competition
        o = self._parametric_eq(o, 2500, 1.0, 1.0)    # Definition
        o = self._parametric_eq(o, 6000, 1.5, 0.7)    # Sparkle
        o = self._parametric_eq(o, 14000, 1.0, 0.5)   # Air

        # 4. Frequency carving against vocals
        if vocals is not None:
            o = self._carve_against(o, vocals, 'other')

        return o

    # ================================================================
    # FREQUENCY CARVING
    # ================================================================

    def _carve_against(self, source: np.ndarray,
                        reference: np.ndarray,
                        source_type: str) -> np.ndarray:
        """
        Carve frequencies from source that would mask the reference.
        This is what separates amateur from professional mixes.

        For vocals: cut the frequency range where the instrumental is loudest
        For instruments: cut the frequency range where vocals sit
        """
        # Analyze reference spectrum
        ref_len = min(len(reference), 4 * self.sr)
        ref_spectrum = np.abs(np.fft.rfft(reference[:ref_len]))
        freqs = np.fft.rfftfreq(ref_len, 1/self.sr)

        if source_type == 'vocals':
            # Find where the instrumental is most dominant in 200-2000Hz
            band = (freqs >= 200) & (freqs <= 2000)
            if band.any() and len(ref_spectrum) == len(freqs):
                band_spectrum = ref_spectrum[band]
                band_freqs = freqs[band]
                peak_idx = np.argmax(band_spectrum)
                peak_freq = band_freqs[peak_idx]
                # Cut this from vocals to let them breathe
                source = self._parametric_eq(source, float(peak_freq), -2.0, 1.5)

        elif source_type == 'other':
            # Cut the vocal presence range (2-5kHz) from instruments
            source = self._parametric_eq(source, 3000, -2.0, 1.5)
            # Also cut where reference has most energy in 500-1500Hz
            band = (freqs >= 500) & (freqs <= 1500)
            if band.any() and len(ref_spectrum) == len(freqs):
                band_spectrum = ref_spectrum[band]
                band_freqs = freqs[band]
                peak_idx = np.argmax(band_spectrum)
                peak_freq = band_freqs[peak_idx]
                source = self._parametric_eq(source, float(peak_freq), -1.5, 2.0)

        return source

    # ================================================================
    # DSP PRIMITIVES
    # ================================================================

    def _highpass(self, audio, cutoff, order=4):
        nyq = self.sr / 2
        if cutoff >= nyq:
            return audio
        b, a = scipy_signal.butter(order, cutoff / nyq, btype='high')
        return scipy_signal.filtfilt(b, a, audio).astype(np.float32)

    def _lowpass(self, audio, cutoff, order=4):
        nyq = self.sr / 2
        if cutoff >= nyq:
            return audio
        b, a = scipy_signal.butter(order, cutoff / nyq, btype='low')
        return scipy_signal.filtfilt(b, a, audio).astype(np.float32)

    def _parametric_eq(self, audio, center_freq, gain_db, q):
        """Apply a parametric EQ band"""
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

    def _compress(self, audio, threshold_db=-20, ratio=3.0,
                   attack_ms=8, release_ms=80, makeup_db=0):
        """Dynamic range compression"""
        threshold = 10 ** (threshold_db / 20)
        attack = int(attack_ms * self.sr / 1000)
        release = int(release_ms * self.sr / 1000)

        envelope = np.abs(audio)
        # Smooth envelope
        for i in range(1, len(envelope)):
            if envelope[i] > envelope[i-1]:
                coeff = np.exp(-1.0 / max(1, attack))
            else:
                coeff = np.exp(-1.0 / max(1, release))
            envelope[i] = coeff * envelope[i-1] + (1 - coeff) * envelope[i]

        # Apply compression
        gain = np.ones_like(envelope)
        above = envelope > threshold
        if np.any(above):
            gain[above] = (threshold / envelope[above]) ** (1 - 1/ratio)

        result = audio * gain

        # Makeup gain
        if makeup_db != 0:
            result *= 10 ** (makeup_db / 20)

        return result

    def _deess(self, audio, threshold_db=-20, frequency=6500):
        """De-esser: compress only the sibilant frequency range"""
        # Extract sibilant band
        nyq = self.sr / 2
        low = max(frequency - 2000, 100) / nyq
        high = min(frequency + 2000, nyq * 0.95) / nyq

        b, a = scipy_signal.butter(2, [low, high], btype='band')
        sibilant = scipy_signal.filtfilt(b, a, audio).astype(np.float32)

        # Compress the sibilant band
        compressed_sib = self._compress(sibilant, threshold_db=threshold_db,
                                         ratio=6.0, attack_ms=1, release_ms=20)

        # Replace sibilant band with compressed version
        result = audio - sibilant + compressed_sib
        return result

    def _transient_shape(self, audio, attack_boost=1.3, sustain_cut=0.85):
        """Transient shaper: boost attacks, cut sustain"""
        # Detect transients via onset strength
        envelope = np.abs(audio)
        fast_env = np.zeros_like(envelope)
        slow_env = np.zeros_like(envelope)

        fast_coeff = np.exp(-1.0 / max(1, int(0.001 * self.sr)))
        slow_coeff = np.exp(-1.0 / max(1, int(0.05 * self.sr)))

        for i in range(1, len(envelope)):
            fast_env[i] = max(envelope[i], fast_coeff * fast_env[i-1])
            slow_env[i] = max(envelope[i], slow_coeff * slow_env[i-1])

        # Transient = fast > slow
        transient_mask = fast_env > slow_env * 1.2
        sustain_mask = ~transient_mask

        gain = np.ones_like(audio)
        gain[transient_mask] = attack_boost
        gain[sustain_mask] = sustain_cut

        # Smooth gain changes
        smooth_samples = int(0.005 * self.sr)
        kernel = np.ones(smooth_samples) / smooth_samples
        gain = np.convolve(gain, kernel, mode='same')

        return audio * gain

    def _saturate(self, audio, drive=0.15):
        """Soft saturation for warmth"""
        driven = audio * (1 + drive)
        # Soft clip using tanh
        saturated = np.tanh(driven)
        # Blend: mostly dry, bit of saturation
        return audio * (1 - drive) + saturated * drive

    def _sidechain_duck(self, target, sidechain, threshold_db=-22,
                         ratio=3.0, attack_ms=3, release_ms=120):
        """Sidechain compression (duck target when sidechain is active)"""
        threshold = 10 ** (threshold_db / 20)
        attack = int(attack_ms * self.sr / 1000)
        release = int(release_ms * self.sr / 1000)

        min_len = min(len(target), len(sidechain))
        sc_env = np.abs(sidechain[:min_len])

        # Smooth sidechain envelope
        for i in range(1, len(sc_env)):
            if sc_env[i] > sc_env[i-1]:
                coeff = np.exp(-1.0 / max(1, attack))
            else:
                coeff = np.exp(-1.0 / max(1, release))
            sc_env[i] = coeff * sc_env[i-1] + (1 - coeff) * sc_env[i]

        # Compute ducking gain
        gain = np.ones(min_len)
        above = sc_env > threshold
        if np.any(above):
            gain[above] = (threshold / sc_env[above]) ** (1 - 1/ratio)

        result = target.copy()
        result[:min_len] *= gain
        return result

    def _reverb(self, audio, decay=1.5, mix=0.1):
        """Simple reverb using feedback comb + allpass filters"""
        output = audio.copy()

        # Feedback comb filters at different delays
        delays_ms = [29.7, 37.1, 41.1, 43.7]
        for delay_ms in delays_ms:
            delay_samples = int(delay_ms * self.sr / 1000)
            feedback = 0.7 * (1 - 1 / (decay + 0.1))
            delayed = np.zeros_like(audio)
            for i in range(delay_samples, len(audio)):
                delayed[i] = audio[i - delay_samples] + feedback * delayed[i - delay_samples] if i - delay_samples >= 0 else 0
            output += delayed * 0.25

        # Allpass filter for diffusion
        allpass_delay = int(5.0 * self.sr / 1000)
        allpass_out = np.zeros_like(output)
        g = 0.5
        for i in range(allpass_delay, len(output)):
            allpass_out[i] = -g * output[i] + output[i - allpass_delay] + g * allpass_out[i - allpass_delay]

        # LPF the reverb tail
        allpass_out = self._lowpass(allpass_out, 6000, order=1)

        # Mix dry + wet
        return audio * (1 - mix) + allpass_out * mix

    def _limit(self, audio, threshold_db=-1.0):
        """Brick-wall limiter"""
        threshold = 10 ** (threshold_db / 20)
        peaks = np.abs(audio)
        gain = np.ones_like(audio)
        above = peaks > threshold
        if np.any(above):
            gain[above] = threshold / peaks[above]

        # Smooth gain to avoid clicks
        smooth = int(0.001 * self.sr)
        kernel = np.ones(smooth) / smooth
        gain = np.convolve(gain, kernel, mode='same')

        return audio * gain
