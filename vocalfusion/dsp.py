"""
VocalFusion AI - Enhanced DSP Module (v2.1)

Major upgrades over v2.0:
  1. IR-based reverb (replaces Schroeder - sounds professional instead of metallic)
  2. Beat grid alignment (finds downbeats and aligns them - fixes sloppy rhythm)
  3. Smooth gain curves (200-500ms decision windows with 50ms ramps - fixes choppy vocals)
  4. Improved multiband processing
  5. Stereo widening for spatial separation

Usage:
    from vocalfusion.dsp import EnhancedDSP
    dsp = EnhancedDSP(sample_rate=44100)
    reverbed = dsp.reverb(audio, reverb_time=1.8, mix=0.12)
    aligned = dsp.align_beats(audio_a, audio_b, tempo_a, tempo_b)
"""

import numpy as np
from scipy import signal as scipy_signal
from pathlib import Path
from typing import Tuple, Optional, Dict
import librosa


class EnhancedDSP:
    """Production-quality DSP toolkit for VocalFusion"""

    def __init__(self, sample_rate: int = 44100):
        self.sr = sample_rate
        self._ir_cache: Dict[str, np.ndarray] = {}

    # ================================================================
    # CORE UTILITIES
    # ================================================================

    @staticmethod
    def ensure_mono(audio: np.ndarray) -> np.ndarray:
        if audio.ndim > 1:
            return librosa.to_mono(audio)
        return audio

    @staticmethod
    def match_lengths(a: np.ndarray, b: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Pad shorter to match longer"""
        if len(a) == len(b):
            return a, b
        max_len = max(len(a), len(b))
        if len(a) < max_len:
            a = np.pad(a, (0, max_len - len(a)))
        if len(b) < max_len:
            b = np.pad(b, (0, max_len - len(b)))
        return a, b

    # ================================================================
    # 1. REVERB - IR-BASED (biggest quality jump)
    # ================================================================

    def reverb(self, audio: np.ndarray, reverb_time: float = 1.8,
               mix: float = 0.12, ir_path: Optional[str] = None) -> np.ndarray:
        """
        Apply reverb using impulse response convolution.

        If ir_path is given, loads that WAV file as the impulse response.
        If not, generates a high-quality synthetic IR (much better than Schroeder).

        To get free IR files:
          - https://www.openair.hosted.york.ac.uk/
          - https://www.voxengo.com/impulses/
          Download any .wav IR file and pass the path here.
        """
        dry = audio.copy()

        if ir_path and Path(ir_path).exists():
            ir = self._load_ir(ir_path, reverb_time)
        else:
            ir = self._generate_synthetic_ir(reverb_time)

        # Convolve using FFT (fast)
        wet = scipy_signal.fftconvolve(audio, ir, mode='full')[:len(audio)]

        # Normalize wet to match dry loudness
        dry_rms = np.sqrt(np.mean(dry ** 2)) + 1e-10
        wet_rms = np.sqrt(np.mean(wet ** 2)) + 1e-10
        wet = wet * (dry_rms / wet_rms)

        return (1.0 - mix) * dry + mix * wet

    def _load_ir(self, ir_path: str, target_time: float) -> np.ndarray:
        """Load and process an impulse response WAV file"""
        cache_key = f"{ir_path}_{target_time}"
        if cache_key in self._ir_cache:
            return self._ir_cache[cache_key]

        ir, ir_sr = librosa.load(ir_path, sr=self.sr, mono=True)

        # Trim or extend IR to match desired reverb time
        target_samples = int(target_time * self.sr)
        if len(ir) > target_samples:
            # Fade out and truncate
            fade = np.linspace(1.0, 0.0, target_samples // 4)
            ir[target_samples - len(fade):target_samples] *= fade
            ir = ir[:target_samples]
        elif len(ir) < target_samples:
            # Extend with exponential decay
            extension = np.zeros(target_samples - len(ir))
            decay_rate = -6.0 / (target_time * self.sr)  # -60dB over reverb_time
            t = np.arange(len(extension))
            extension = np.random.randn(len(extension)) * 0.01 * np.exp(decay_rate * t)
            ir = np.concatenate([ir, extension])

        # Normalize
        ir = ir / (np.max(np.abs(ir)) + 1e-10)

        self._ir_cache[cache_key] = ir
        return ir

    def _generate_synthetic_ir(self, reverb_time: float) -> np.ndarray:
        """
        Generate a high-quality synthetic impulse response.

        This is MUCH better than Schroeder reverb because:
        - Uses multiple diffusion stages (not just comb+allpass)
        - Models early reflections separately from late reverb tail
        - Exponential decay envelope matched to reverb_time
        - Random noise filtered to sound natural (not metallic)
        """
        total_samples = int(reverb_time * self.sr)

        # --- EARLY REFLECTIONS (first 50-80ms) ---
        # These give the listener a sense of room size
        early_length = int(0.08 * self.sr)  # 80ms
        early = np.zeros(early_length)

        # Place discrete reflections at realistic delays
        # (simulating sound bouncing off nearby walls)
        reflection_times_ms = [5, 11, 17, 23, 29, 37, 43, 53, 61, 73]
        reflection_gains = [0.8, 0.65, 0.55, 0.45, 0.38, 0.32, 0.27, 0.22, 0.18, 0.15]

        for t_ms, gain in zip(reflection_times_ms, reflection_gains):
            sample_idx = int(t_ms * self.sr / 1000)
            if sample_idx < early_length:
                # Alternate polarity for diffusion
                polarity = 1.0 if np.random.random() > 0.5 else -1.0
                early[sample_idx] = gain * polarity

        # Smooth early reflections slightly
        smooth_kernel = np.hanning(int(0.001 * self.sr))  # 1ms smoothing
        smooth_kernel = smooth_kernel / smooth_kernel.sum()
        early = np.convolve(early, smooth_kernel, mode='same')

        # --- LATE REVERB TAIL (80ms to reverb_time) ---
        late_length = total_samples - early_length
        if late_length <= 0:
            return early

        # Generate noise-based tail with exponential decay
        # Use filtered noise (not pure white noise) for natural sound
        noise = np.random.randn(late_length)

        # Shape the noise spectrum: boost low-mids, cut highs
        # This makes it sound like a real room (which absorbs high frequencies)
        # Low-pass at ~6kHz with gentle slope
        nyq = self.sr / 2
        b_lp, a_lp = scipy_signal.butter(2, min(6000 / nyq, 0.99), btype='low')
        noise = scipy_signal.filtfilt(b_lp, a_lp, noise)

        # Slight boost at 200-800Hz (room resonance character)
        b_peak, a_peak = self._make_peak_filter(500, 1.5, 0.7)
        noise = scipy_signal.lfilter(b_peak, a_peak, noise)

        # Apply exponential decay envelope
        # RT60: time for reverb to decay by 60dB
        decay_rate = np.log(0.001) / (reverb_time * self.sr)  # -60dB
        envelope = np.exp(decay_rate * np.arange(late_length))
        tail = noise * envelope

        # --- COMBINE ---
        ir = np.zeros(total_samples)
        ir[:early_length] = early
        ir[early_length:] = tail * 0.5  # Tail is quieter than early reflections

        # Add initial impulse (direct sound)
        ir[0] = 1.0

        # Normalize
        ir = ir / (np.max(np.abs(ir)) + 1e-10)

        return ir

    def _make_peak_filter(self, center_freq: float, gain_db: float,
                          q: float) -> Tuple[np.ndarray, np.ndarray]:
        """Create peak/bell filter coefficients"""
        A = 10 ** (gain_db / 40.0)
        w0 = 2 * np.pi * center_freq / self.sr
        alpha = np.sin(w0) / (2 * q)
        b0 = 1 + alpha * A
        b1 = -2 * np.cos(w0)
        b2 = 1 - alpha * A
        a0 = 1 + alpha / A
        a1 = -2 * np.cos(w0)
        a2 = 1 - alpha / A
        return np.array([b0/a0, b1/a0, b2/a0]), np.array([1.0, a1/a0, a2/a0])

    # ================================================================
    # 2. BEAT GRID ALIGNMENT (fixes sloppy rhythm)
    # ================================================================

    def align_beats(self, audio_a: np.ndarray, audio_b: np.ndarray,
                    tempo_a: float, tempo_b: float) -> Tuple[np.ndarray, np.ndarray]:
        """
        Align the beat grids of two audio signals.

        Steps:
          1. Time-stretch B to match A's tempo (if needed)
          2. Find the first strong downbeat in each
          3. Shift B so its downbeat lands on A's downbeat
          4. Return aligned pair

        This is THE biggest fix for "sounds like two songs playing at once"
        """
        a = self.ensure_mono(audio_a.copy())
        b = self.ensure_mono(audio_b.copy())

        # Step 1: Time-stretch B to match A's tempo
        if abs(tempo_a - tempo_b) > 1.0:  # More than 1 BPM difference
            stretch_ratio = tempo_a / tempo_b
            # Clamp to reasonable range (0.5x to 2x)
            stretch_ratio = np.clip(stretch_ratio, 0.5, 2.0)
            b = librosa.effects.time_stretch(b, rate=stretch_ratio)

        # Step 2: Find first strong downbeat in each
        downbeat_a = self._find_first_downbeat(a, tempo_a)
        downbeat_b = self._find_first_downbeat(b, tempo_a)  # Use A's tempo since B is now stretched

        # Step 3: Calculate offset and shift B
        offset_samples = downbeat_a - downbeat_b

        if offset_samples > 0:
            # B needs to be delayed (pad front)
            b = np.pad(b, (offset_samples, 0))
        elif offset_samples < 0:
            # B needs to be advanced (trim front)
            trim = abs(offset_samples)
            if trim < len(b):
                b = b[trim:]
            else:
                b = np.zeros(len(a))

        # Step 4: Match lengths
        a, b = self.match_lengths(a, b)

        return a, b

    def _find_first_downbeat(self, audio: np.ndarray, tempo: float) -> int:
        """
        Find the sample position of the first strong downbeat.

        Uses onset strength and beat tracking to find where the
        first bar-level strong beat lands.
        """
        # Get beat frames
        tempo_detected, beats = librosa.beat.beat_track(
            y=audio, sr=self.sr, start_bpm=tempo, units='frames')

        if len(beats) < 2:
            # Fallback: use onset detection
            onsets = librosa.onset.onset_detect(y=audio, sr=self.sr, units='samples')
            return int(onsets[0]) if len(onsets) > 0 else 0

        # Get onset strength at each beat position
        onset_env = librosa.onset.onset_strength(y=audio, sr=self.sr)

        # Find the strongest beat among the first 8 beats
        # (the first downbeat of a bar is usually the strongest)
        search_range = min(8, len(beats))
        beat_strengths = []
        for i in range(search_range):
            frame = beats[i]
            if frame < len(onset_env):
                beat_strengths.append(onset_env[frame])
            else:
                beat_strengths.append(0)

        # Pick the strongest
        strongest_idx = np.argmax(beat_strengths)
        downbeat_frame = beats[strongest_idx]

        # Convert frame to samples
        downbeat_sample = librosa.frames_to_samples(downbeat_frame)

        return int(downbeat_sample)

    def fine_align_onsets(self, primary: np.ndarray, secondary: np.ndarray,
                          window_ms: float = 50.0) -> np.ndarray:
        """
        Fine-tune alignment of secondary to primary using cross-correlation.

        After coarse beat alignment, this looks at a small window around
        each beat and micro-shifts secondary for phase-coherent alignment.

        window_ms: search window in milliseconds
        """
        window_samples = int(window_ms * self.sr / 1000)

        # Cross-correlate a short segment around the beginning
        # to find the precise offset
        analysis_length = min(len(primary), len(secondary), self.sr * 4)  # First 4 seconds
        seg_p = primary[:analysis_length]
        seg_s = secondary[:analysis_length]

        # Use onset envelope for correlation (more robust than raw audio)
        env_p = librosa.onset.onset_strength(y=seg_p, sr=self.sr)
        env_s = librosa.onset.onset_strength(y=seg_s, sr=self.sr)

        # Cross-correlate within the search window
        # Convert window from samples to frames
        hop_length = 512  # librosa default
        window_frames = max(1, window_samples // hop_length)

        min_len = min(len(env_p), len(env_s))
        if min_len < window_frames * 2:
            return secondary  # Too short to align

        correlation = np.correlate(
            env_p[:min_len],
            env_s[:min_len],
            mode='full'
        )

        # Find the peak within our search window
        center = len(correlation) // 2
        search_start = max(0, center - window_frames)
        search_end = min(len(correlation), center + window_frames)
        search_region = correlation[search_start:search_end]

        if len(search_region) == 0:
            return secondary

        best_offset_frames = np.argmax(search_region) - (search_end - search_start) // 2
        best_offset_samples = best_offset_frames * hop_length

        # Apply the offset
        if best_offset_samples > 0:
            aligned = np.pad(secondary, (best_offset_samples, 0))[:len(secondary)]
        elif best_offset_samples < 0:
            trim = abs(best_offset_samples)
            aligned = np.pad(secondary[trim:], (0, trim))
        else:
            aligned = secondary

        return aligned

    # ================================================================
    # 3. SMOOTH GAIN CURVES (fixes choppy vocal switching)
    # ================================================================

    def compute_activity_mask(self, audio: np.ndarray,
                              decision_window_ms: float = 300.0,
                              ramp_ms: float = 50.0) -> np.ndarray:
        """
        Compute a smooth activity mask for audio.

        Instead of per-frame (12ms) decisions that cause choppy switching,
        this uses:
          - Longer decision windows (300ms default) for stable decisions
          - Smooth ramps (50ms) between active/inactive transitions
          - Hysteresis to prevent rapid toggling

        Returns: gain curve (0.0 to 1.0) at sample rate, same length as audio
        """
        audio = self.ensure_mono(audio)

        # Compute energy in decision-length windows
        decision_samples = int(decision_window_ms * self.sr / 1000)
        ramp_samples = int(ramp_ms * self.sr / 1000)

        # RMS energy per decision window
        n_windows = max(1, len(audio) // decision_samples)
        window_active = np.zeros(n_windows, dtype=bool)

        # Calculate threshold using percentile of non-silence
        rms_values = []
        for i in range(n_windows):
            start = i * decision_samples
            end = min(start + decision_samples, len(audio))
            window = audio[start:end]
            rms = np.sqrt(np.mean(window ** 2))
            rms_values.append(rms)

        rms_values = np.array(rms_values)
        # Threshold: above 20th percentile of non-silence
        non_silence = rms_values[rms_values > 0.001]
        if len(non_silence) > 0:
            threshold = np.percentile(non_silence, 20)
        else:
            threshold = 0.001

        # Apply hysteresis: higher threshold to turn ON, lower to turn OFF
        threshold_on = threshold * 1.2
        threshold_off = threshold * 0.6

        is_active = False
        for i, rms in enumerate(rms_values):
            if not is_active and rms > threshold_on:
                is_active = True
            elif is_active and rms < threshold_off:
                is_active = False
            window_active[i] = is_active

        # Expand to sample-level
        mask = np.zeros(len(audio))
        for i in range(n_windows):
            start = i * decision_samples
            end = min(start + decision_samples, len(audio))
            mask[start:end] = 1.0 if window_active[i] else 0.0

        # Apply smooth ramps at transitions
        mask = self._smooth_transitions(mask, ramp_samples)

        return mask

    def blend_with_masks(self, audio_a: np.ndarray, audio_b: np.ndarray,
                         mask_a: np.ndarray, mask_b: np.ndarray,
                         lead_gain: float = 0.75,
                         harmony_gain: float = 0.35) -> np.ndarray:
        """
        Blend two audio signals using their activity masks.

        When both are active: A is lead (louder), B is harmony (quieter)
        When only one is active: that one plays at full volume
        When neither: silence

        lead_gain/harmony_gain control the mix when both are singing.
        """
        audio_a, audio_b = self.match_lengths(audio_a, audio_b)
        mask_a = mask_a[:len(audio_a)]
        mask_b = mask_b[:len(audio_b)]

        # Ensure masks are same length as audio
        if len(mask_a) < len(audio_a):
            mask_a = np.pad(mask_a, (0, len(audio_a) - len(mask_a)))
        if len(mask_b) < len(audio_b):
            mask_b = np.pad(mask_b, (0, len(audio_b) - len(mask_b)))

        output = np.zeros(len(audio_a))

        # Both active: lead + harmony
        both_active = (mask_a > 0.5) & (mask_b > 0.5)
        # Only A active
        only_a = (mask_a > 0.5) & (mask_b <= 0.5)
        # Only B active
        only_b = (mask_a <= 0.5) & (mask_b > 0.5)

        # Apply gains with smooth mask values (not just binary)
        output[both_active] = (audio_a[both_active] * lead_gain * mask_a[both_active] +
                               audio_b[both_active] * harmony_gain * mask_b[both_active])
        output[only_a] = audio_a[only_a] * mask_a[only_a]
        output[only_b] = audio_b[only_b] * mask_b[only_b]

        return output

    def _smooth_transitions(self, mask: np.ndarray, ramp_samples: int) -> np.ndarray:
        """Apply smooth ramps at 0→1 and 1→0 transitions in a mask"""
        if ramp_samples <= 0:
            return mask

        smoothed = mask.copy()

        # Find transitions
        diff = np.diff(mask)
        rise_points = np.where(diff > 0.5)[0]  # 0 → 1
        fall_points = np.where(diff < -0.5)[0]  # 1 → 0

        for pt in rise_points:
            start = max(0, pt - ramp_samples // 2)
            end = min(len(smoothed), pt + ramp_samples // 2)
            ramp_len = end - start
            if ramp_len > 0:
                smoothed[start:end] = np.linspace(0.0, 1.0, ramp_len)

        for pt in fall_points:
            start = max(0, pt - ramp_samples // 2)
            end = min(len(smoothed), pt + ramp_samples // 2)
            ramp_len = end - start
            if ramp_len > 0:
                smoothed[start:end] = np.linspace(1.0, 0.0, ramp_len)

        return np.clip(smoothed, 0.0, 1.0)

    # ================================================================
    # CROSSFADE UTILITY
    # ================================================================

    def crossfade(self, a: np.ndarray, b: np.ndarray,
                  fade_ms: float = 1000.0) -> np.ndarray:
        """Crossfade from a into b with configurable duration"""
        fade_samples = int(fade_ms * self.sr / 1000)
        if fade_samples <= 0 or len(a) == 0 or len(b) == 0:
            return np.concatenate([a, b])
        fade_samples = min(fade_samples, len(a), len(b))

        # Use equal-power crossfade (sounds smoother than linear)
        t = np.linspace(0, np.pi / 2, fade_samples)
        fade_out = np.cos(t)   # 1 → 0 with equal power curve
        fade_in = np.sin(t)    # 0 → 1 with equal power curve

        result = np.concatenate([
            a[:-fade_samples],
            a[-fade_samples:] * fade_out + b[:fade_samples] * fade_in,
            b[fade_samples:]
        ])
        return result

    # ================================================================
    # EQ AND FILTERS
    # ================================================================

    def parametric_eq(self, audio: np.ndarray, center_freq: float,
                      gain_db: float, q: float = 1.0) -> np.ndarray:
        """Apply parametric EQ bell filter"""
        if abs(gain_db) < 0.1:
            return audio
        b, a = self._make_peak_filter(center_freq, gain_db, q)
        return scipy_signal.lfilter(b, a, audio)

    def highpass(self, audio: np.ndarray, cutoff: float, order: int = 4) -> np.ndarray:
        nyq = self.sr / 2.0
        if cutoff >= nyq:
            return audio
        b, a = scipy_signal.butter(order, cutoff / nyq, btype='high')
        return scipy_signal.filtfilt(b, a, audio)

    def lowpass(self, audio: np.ndarray, cutoff: float, order: int = 4) -> np.ndarray:
        nyq = self.sr / 2.0
        if cutoff >= nyq:
            return audio
        b, a = scipy_signal.butter(order, cutoff / nyq, btype='low')
        return scipy_signal.filtfilt(b, a, audio)

    def bandpass(self, audio: np.ndarray, low: float, high: float,
                 order: int = 4) -> np.ndarray:
        nyq = self.sr / 2.0
        b, a = scipy_signal.butter(order, [low / nyq, min(high / nyq, 0.99)], btype='band')
        return scipy_signal.filtfilt(b, a, audio)

    # ================================================================
    # COMPRESSION (with proper envelope)
    # ================================================================

    def compress(self, audio: np.ndarray, threshold_db: float = -20,
                 ratio: float = 3.0, attack_ms: float = 10.0,
                 release_ms: float = 100.0,
                 makeup_db: float = 0.0) -> np.ndarray:
        """Sample-accurate compressor with real attack/release"""
        threshold = 10 ** (threshold_db / 20.0)
        envelope = self._compute_envelope(audio, attack_ms, release_ms)

        gain = np.ones_like(envelope)
        above_mask = envelope > threshold
        if above_mask.any():
            above_db = 20 * np.log10(envelope[above_mask] / threshold + 1e-10)
            reduction_db = above_db * (1.0 - 1.0 / ratio)
            gain[above_mask] = 10 ** (-reduction_db / 20.0)

        compressed = audio * gain

        if abs(makeup_db) > 0.01:
            compressed *= 10 ** (makeup_db / 20.0)

        return compressed

    def _compute_envelope(self, audio: np.ndarray, attack_ms: float,
                          release_ms: float) -> np.ndarray:
        """Sample-accurate envelope follower"""
        rectified = np.abs(audio)
        attack_coeff = 1.0 - np.exp(-1.0 / (self.sr * attack_ms / 1000.0))
        release_coeff = 1.0 - np.exp(-1.0 / (self.sr * release_ms / 1000.0))
        envelope = np.zeros_like(rectified)
        envelope[0] = rectified[0]
        for i in range(1, len(rectified)):
            if rectified[i] > envelope[i - 1]:
                envelope[i] = envelope[i - 1] + attack_coeff * (rectified[i] - envelope[i - 1])
            else:
                envelope[i] = envelope[i - 1] + release_coeff * (rectified[i] - envelope[i - 1])
        return envelope

    # ================================================================
    # SIDECHAIN DUCKING (envelope-following)
    # ================================================================

    def sidechain_duck(self, target: np.ndarray, sidechain_source: np.ndarray,
                       threshold_db: float = -25, ratio: float = 4.0,
                       attack_ms: float = 5.0, release_ms: float = 200.0,
                       amount: float = 1.0) -> np.ndarray:
        """
        Envelope-following sidechain ducking.

        target: audio to duck (e.g., instruments)
        sidechain_source: audio that triggers ducking (e.g., vocals)
        amount: 0.0 = no ducking, 1.0 = full ducking
        """
        if target is None or sidechain_source is None:
            return target

        target, sidechain_source = self.match_lengths(target, sidechain_source)

        threshold = 10 ** (threshold_db / 20.0)
        envelope = self._compute_envelope(sidechain_source, attack_ms, release_ms)

        gain_reduction = np.ones_like(envelope)
        above = envelope > threshold
        if above.any():
            above_db = 20 * np.log10(envelope[above] / threshold + 1e-10)
            reduction_db = above_db * (1.0 - 1.0 / ratio)
            gain_reduction[above] = 10 ** (-reduction_db / 20.0)

        # Blend between full ducking and no ducking based on amount
        gain = 1.0 - amount * (1.0 - gain_reduction)

        return target * gain

    # ================================================================
    # MASTERING CHAIN
    # ================================================================

    def multiband_compress(self, audio: np.ndarray,
                           low_threshold: float = -18, low_ratio: float = 2.0,
                           mid_threshold: float = -15, mid_ratio: float = 1.5,
                           high_threshold: float = -12, high_ratio: float = 1.3
                           ) -> np.ndarray:
        """3-band multiband compressor"""
        low = self.lowpass(audio, 250)
        mid = self.bandpass(audio, 250, 4000)
        high = self.highpass(audio, 4000)

        low_c = self.compress(low, low_threshold, low_ratio, 20, 200, 1)
        mid_c = self.compress(mid, mid_threshold, mid_ratio, 10, 150, 1)
        high_c = self.compress(high, high_threshold, high_ratio, 5, 100, 0.5)

        return low_c + mid_c + high_c

    def true_peak_limit(self, audio: np.ndarray, threshold_db: float = -1.0,
                        ceiling_db: float = -0.3) -> np.ndarray:
        """True-peak limiter with 4x oversampled peak detection"""
        threshold = 10 ** (threshold_db / 20.0)
        ceiling = 10 ** (ceiling_db / 20.0)

        # 4x oversample for true peak
        oversampled = scipy_signal.resample(audio, len(audio) * 4)
        true_peak = np.max(np.abs(oversampled))

        if true_peak <= threshold:
            return audio

        gain = ceiling / true_peak
        return audio * gain

    def loudness_normalize(self, audio: np.ndarray,
                           target_lufs: float = -14.0) -> np.ndarray:
        """LUFS-approximation loudness normalization"""
        rms = np.sqrt(np.mean(audio ** 2))
        if rms < 1e-10:
            return audio
        current_lufs = 20 * np.log10(rms)
        gain_db = np.clip(target_lufs - current_lufs, -12, 12)
        return audio * 10 ** (gain_db / 20.0)

    def master(self, audio: np.ndarray, target_lufs: float = -14.0,
               limiter_threshold: float = -1.0,
               limiter_ceiling: float = -0.3) -> np.ndarray:
        """Full mastering chain: multiband compress → normalize → limit"""
        mastered = self.multiband_compress(audio)
        mastered = self.loudness_normalize(mastered, target_lufs)
        mastered = self.true_peak_limit(mastered, limiter_threshold, limiter_ceiling)
        return mastered

    # ================================================================
    # STEREO UTILITIES
    # ================================================================

    def stereo_place(self, audio: np.ndarray, pan: float = 0.0,
                     width: float = 1.0) -> Tuple[np.ndarray, np.ndarray]:
        """
        Create stereo signal from mono with pan and width control.

        pan: -1.0 (full left) to 1.0 (full right), 0.0 = center
        width: 0.0 (mono) to 1.0 (full stereo)

        Returns: (left_channel, right_channel)
        """
        audio = self.ensure_mono(audio)

        # Constant-power panning
        angle = (pan + 1.0) * np.pi / 4  # 0 to pi/2
        left_gain = np.cos(angle)
        right_gain = np.sin(angle)

        left = audio * left_gain
        right = audio * right_gain

        # Add stereo width using Haas effect (slight delay on one side)
        if width > 0:
            delay_samples = int(width * 0.0003 * self.sr)  # Up to 0.3ms
            if delay_samples > 0:
                right = np.pad(right, (delay_samples, 0))[:len(audio)]

        return left, right
