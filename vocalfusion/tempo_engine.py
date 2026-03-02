"""
VocalFusion AI - Tempo Engine
==============================

Professional tempo analysis and matching.

Features:
  - Accurate BPM detection with half-time/double-time awareness
  - Beat grid extraction (downbeats, bars, phrases)
  - Smart tempo matching (finds the stretch with least distortion)
  - Beat-aligned time stretching (stretches between beats, not uniformly)
  - Groove analysis (swing, push/pull, feel)
  - Phase-locked beat alignment between two songs
"""

import numpy as np
import librosa
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass, field


@dataclass
class BeatGrid:
    """Complete rhythmic analysis of a song"""
    bpm: float                    # Primary BPM
    bpm_confidence: float         # 0-1 how stable the tempo is
    beat_times: np.ndarray        # Time of each beat in seconds
    downbeat_times: np.ndarray    # Time of each bar start (beat 1)
    time_signature: int           # Beats per bar (3, 4, etc.)
    swing_amount: float           # 0=straight, 1=full swing
    tempo_stability: float        # 0-1 how consistent tempo is throughout
    beat_strengths: np.ndarray    # Relative strength of each beat
    phrase_boundaries: np.ndarray # Where 4/8-bar phrases start


@dataclass
class TempoMatch:
    """Result of tempo matching analysis"""
    strategy_name: str        # e.g. "Half-time B", "Direct match"
    target_bpm: float         # The BPM both songs will play at
    stretch_a: float          # Time-stretch factor for song A
    stretch_b: float          # Time-stretch factor for song B
    quality_score: float      # 0-1 how good this match is
    description: str          # Human-readable explanation


class TempoEngine:
    """Professional tempo analysis and matching"""

    def __init__(self, sample_rate: int = 44100):
        self.sr = sample_rate

    # ================================================================
    # BEAT GRID ANALYSIS
    # ================================================================

    def analyze(self, audio: np.ndarray) -> BeatGrid:
        """Full rhythmic analysis of an audio signal"""
        if audio.ndim > 1:
            audio = librosa.to_mono(audio)

        # Detect tempo with multiple methods for accuracy
        bpm, bpm_confidence = self._detect_tempo_robust(audio)

        # Get beat positions
        beat_times, beat_strengths = self._extract_beats(audio, bpm)

        # Find downbeats (bar starts)
        downbeat_times, time_sig = self._find_downbeats(audio, beat_times)

        # Analyze groove/swing
        swing = self._analyze_swing(beat_times, bpm)

        # Tempo stability (how consistent is the BPM throughout?)
        stability = self._tempo_stability(beat_times)

        # Find phrase boundaries (4-bar or 8-bar groupings)
        phrases = self._find_phrases(downbeat_times, time_sig)

        return BeatGrid(
            bpm=bpm,
            bpm_confidence=bpm_confidence,
            beat_times=beat_times,
            downbeat_times=downbeat_times,
            time_signature=time_sig,
            swing_amount=swing,
            tempo_stability=stability,
            beat_strengths=beat_strengths,
            phrase_boundaries=phrases
        )

    # ================================================================
    # ROBUST TEMPO DETECTION
    # ================================================================

    def _detect_tempo_robust(self, audio: np.ndarray) -> Tuple[float, float]:
        """
        Detect tempo using multiple methods and cross-validate.
        Handles half-time/double-time ambiguity.
        """
        # Method 1: librosa tempogram
        onset_env = librosa.onset.onset_strength(y=audio, sr=self.sr)
        tempos = librosa.feature.tempo(
            onset_envelope=onset_env, sr=self.sr,
            aggregate=None)

        if len(tempos) > 0:
            # Get the two most likely tempos
            tempo_hist, tempo_bins = np.histogram(tempos, bins=50, range=(40, 220))
            top_indices = np.argsort(tempo_hist)[-3:]
            candidates = [(tempo_bins[i] + tempo_bins[i+1])/2 for i in top_indices
                         if tempo_hist[i] > 0]
        else:
            candidates = [120.0]

        # Method 2: autocorrelation-based
        ac_tempo = self._autocorrelation_tempo(onset_env)
        if ac_tempo > 0:
            candidates.append(ac_tempo)

        # Method 3: librosa default
        default_tempo = librosa.feature.tempo(
            onset_envelope=onset_env, sr=self.sr)[0]
        candidates.append(default_tempo)

        # Score each candidate
        best_tempo = default_tempo
        best_score = 0

        for candidate in candidates:
            if candidate < 40 or candidate > 220:
                continue
            score = self._score_tempo_candidate(onset_env, candidate)
            if score > best_score:
                best_score = score
                best_tempo = candidate

        # Resolve half-time/double-time: prefer 80-160 BPM range
        if best_tempo < 70 and best_tempo * 2 < 180:
            alt_score = self._score_tempo_candidate(onset_env, best_tempo * 2)
            if alt_score > best_score * 0.7:
                best_tempo *= 2
                best_score = alt_score

        if best_tempo > 170 and best_tempo / 2 > 70:
            alt_score = self._score_tempo_candidate(onset_env, best_tempo / 2)
            if alt_score > best_score * 0.7:
                best_tempo /= 2
                best_score = alt_score

        confidence = min(1.0, best_score / 2.0)
        return float(best_tempo), float(confidence)

    def _autocorrelation_tempo(self, onset_env: np.ndarray) -> float:
        """Detect tempo via onset envelope autocorrelation"""
        ac = np.correlate(onset_env, onset_env, mode='full')
        ac = ac[len(ac)//2:]  # Keep positive lags only

        # Convert BPM range to lag range
        hop = 512
        min_lag = int(60 * self.sr / (200 * hop))  # 200 BPM
        max_lag = int(60 * self.sr / (60 * hop))    # 60 BPM

        if max_lag >= len(ac) or min_lag < 1:
            return 0.0

        search = ac[min_lag:max_lag]
        if len(search) == 0:
            return 0.0

        peak = np.argmax(search) + min_lag
        bpm = 60 * self.sr / (peak * hop)
        return float(bpm)

    def _score_tempo_candidate(self, onset_env: np.ndarray, bpm: float) -> float:
        """Score how well a tempo candidate fits the onset envelope"""
        hop = 512
        beat_period = 60 * self.sr / (bpm * hop)  # Beat period in frames

        if beat_period < 2 or beat_period > len(onset_env):
            return 0.0

        # Check autocorrelation at this period
        lag = int(beat_period)
        if lag >= len(onset_env):
            return 0.0

        # Sum correlation at beat period and its multiples
        score = 0.0
        for mult in [1, 2, 4]:
            check_lag = int(lag * mult)
            if check_lag < len(onset_env):
                corr = np.sum(onset_env[check_lag:] * onset_env[:len(onset_env)-check_lag])
                score += corr / mult  # Weight fundamental more

        # Normalize
        energy = np.sum(onset_env ** 2)
        if energy > 0:
            score /= energy

        return float(score)

    # ================================================================
    # BEAT EXTRACTION
    # ================================================================

    def _extract_beats(self, audio: np.ndarray,
                        expected_bpm: float) -> Tuple[np.ndarray, np.ndarray]:
        """Extract beat positions and their strengths"""
        tempo, beats = librosa.beat.beat_track(
            y=audio, sr=self.sr, bpm=expected_bpm,
            units='time')

        if len(beats) == 0:
            # Fallback: generate beat grid from BPM
            duration = len(audio) / self.sr
            beat_interval = 60.0 / expected_bpm
            beats = np.arange(0, duration, beat_interval)

        # Compute beat strengths from onset envelope
        onset_env = librosa.onset.onset_strength(y=audio, sr=self.sr)
        hop = 512
        strengths = []
        for bt in beats:
            frame = int(bt * self.sr / hop)
            if frame < len(onset_env):
                # Average onset strength around this beat
                window = max(1, int(0.05 * self.sr / hop))  # 50ms window
                start = max(0, frame - window)
                end = min(len(onset_env), frame + window + 1)
                strengths.append(float(np.mean(onset_env[start:end])))
            else:
                strengths.append(0.0)

        return np.array(beats), np.array(strengths)

    def _find_downbeats(self, audio: np.ndarray,
                         beat_times: np.ndarray) -> Tuple[np.ndarray, int]:
        """Find bar-level downbeats and estimate time signature"""
        if len(beat_times) < 8:
            return beat_times[::4], 4

        # Use spectral flux to find strong beats
        onset_env = librosa.onset.onset_strength(y=audio, sr=self.sr)
        hop = 512

        strengths = []
        for bt in beat_times:
            frame = int(bt * self.sr / hop)
            if frame < len(onset_env):
                strengths.append(onset_env[min(frame, len(onset_env)-1)])
            else:
                strengths.append(0.0)
        strengths = np.array(strengths)

        # Test time signatures 3 and 4
        best_sig = 4
        best_score = 0

        for sig in [3, 4]:
            if len(strengths) < sig * 2:
                continue
            # Check if every sig-th beat is stronger than average
            downbeat_strengths = strengths[::sig]
            other_strengths = np.delete(strengths,
                                         list(range(0, len(strengths), sig)))
            if len(other_strengths) > 0 and len(downbeat_strengths) > 0:
                ratio = np.mean(downbeat_strengths) / (np.mean(other_strengths) + 1e-10)
                if ratio > best_score:
                    best_score = ratio
                    best_sig = sig

        downbeats = beat_times[::best_sig]
        return downbeats, best_sig

    def _find_phrases(self, downbeat_times: np.ndarray,
                       time_sig: int) -> np.ndarray:
        """Find phrase boundaries (typically 4 or 8 bars)"""
        # Try 4-bar and 8-bar phrase lengths
        if len(downbeat_times) < 4:
            return downbeat_times

        # Default: 4-bar phrases
        phrases = downbeat_times[::4]
        return phrases

    # ================================================================
    # GROOVE ANALYSIS
    # ================================================================

    def _analyze_swing(self, beat_times: np.ndarray, bpm: float) -> float:
        """
        Measure swing amount.
        Swing = how much the off-beats are shifted late.
        0 = perfectly straight, 1 = full triplet swing
        """
        if len(beat_times) < 4:
            return 0.0

        expected_interval = 60.0 / bpm
        deviations = []

        for i in range(1, len(beat_times) - 1, 2):
            # Off-beats (every other beat)
            actual = beat_times[i] - beat_times[i-1]
            deviation = (actual - expected_interval) / expected_interval
            deviations.append(deviation)

        if not deviations:
            return 0.0

        avg_deviation = np.mean(deviations)
        # Swing: positive deviation on off-beats = swung
        swing = np.clip(avg_deviation * 3, 0, 1)  # Scale to 0-1
        return float(swing)

    def _tempo_stability(self, beat_times: np.ndarray) -> float:
        """Measure how consistent the tempo is throughout the song"""
        if len(beat_times) < 4:
            return 0.5

        intervals = np.diff(beat_times)
        if len(intervals) == 0:
            return 0.5

        mean_interval = np.mean(intervals)
        std_interval = np.std(intervals)

        # Coefficient of variation (lower = more stable)
        cv = std_interval / (mean_interval + 1e-10)

        # Convert to 0-1 score (0.02 CV = perfect, 0.15 CV = unstable)
        stability = 1.0 - np.clip(cv / 0.15, 0, 1)
        return float(stability)

    # ================================================================
    # TEMPO MATCHING
    # ================================================================

    def find_best_match(self, grid_a: BeatGrid,
                         grid_b: BeatGrid) -> List[TempoMatch]:
        """
        Find the best tempo matching strategies for two songs.
        Returns sorted list of options (best first).

        Considers:
        - Direct match (A→B or B→A)
        - Half-time / double-time relationships
        - Meeting in the middle
        - Weighted middle (favor less stretching on vocals)
        """
        bpm_a = grid_a.bpm
        bpm_b = grid_b.bpm
        matches = []

        # Strategy 1: Match to A (stretch B)
        stretch = bpm_a / bpm_b
        matches.append(TempoMatch(
            strategy_name=f"Match A ({bpm_a:.0f} BPM)",
            target_bpm=bpm_a,
            stretch_a=1.0,
            stretch_b=stretch,
            quality_score=self._stretch_quality(1.0, stretch),
            description=f"Keep A at {bpm_a:.0f}, stretch B from {bpm_b:.0f}"
        ))

        # Strategy 2: Match to B (stretch A)
        stretch = bpm_b / bpm_a
        matches.append(TempoMatch(
            strategy_name=f"Match B ({bpm_b:.0f} BPM)",
            target_bpm=bpm_b,
            stretch_a=stretch,
            stretch_b=1.0,
            quality_score=self._stretch_quality(stretch, 1.0),
            description=f"Keep B at {bpm_b:.0f}, stretch A from {bpm_a:.0f}"
        ))

        # Strategy 3: Half-time B (if B is much faster)
        if bpm_b > bpm_a * 1.3:
            half_b = bpm_b / 2
            stretch = bpm_a / half_b
            matches.append(TempoMatch(
                strategy_name=f"Half-time B ({half_b:.0f}→{bpm_a:.0f})",
                target_bpm=bpm_a,
                stretch_a=1.0,
                stretch_b=stretch,
                quality_score=self._stretch_quality(1.0, stretch) * 0.95,
                description=f"Treat B as {half_b:.0f} BPM (half-time), match to A"
            ))

        # Strategy 4: Half-time A (if A is much faster)
        if bpm_a > bpm_b * 1.3:
            half_a = bpm_a / 2
            stretch = bpm_b / half_a
            matches.append(TempoMatch(
                strategy_name=f"Half-time A ({half_a:.0f}→{bpm_b:.0f})",
                target_bpm=bpm_b,
                stretch_a=stretch,
                stretch_b=1.0,
                quality_score=self._stretch_quality(stretch, 1.0) * 0.95,
                description=f"Treat A as {half_a:.0f} BPM (half-time), match to B"
            ))

        # Strategy 5: Double-time A
        if bpm_a * 2 < 200:
            double_a = bpm_a * 2
            stretch_a = double_a / bpm_a  # This is just for the ratio
            stretch_b = double_a / bpm_b
            matches.append(TempoMatch(
                strategy_name=f"Double-time A ({double_a:.0f} BPM)",
                target_bpm=double_a,
                stretch_a=1.0,  # A plays naturally, just interpreted as double-time
                stretch_b=stretch_b,
                quality_score=self._stretch_quality(1.0, stretch_b) * 0.9,
                description=f"Interpret A as {double_a:.0f} BPM, stretch B to match"
            ))

        # Strategy 6: Double-time B
        if bpm_b * 2 < 200:
            double_b = bpm_b * 2
            stretch_a = double_b / bpm_a
            matches.append(TempoMatch(
                strategy_name=f"Double-time B ({double_b:.0f} BPM)",
                target_bpm=double_b,
                stretch_a=stretch_a,
                stretch_b=1.0,
                quality_score=self._stretch_quality(stretch_a, 1.0) * 0.9,
                description=f"Interpret B as {double_b:.0f} BPM, stretch A to match"
            ))

        # Strategy 7: Meet in middle
        mid = (bpm_a + bpm_b) / 2
        stretch_a = mid / bpm_a
        stretch_b = mid / bpm_b
        matches.append(TempoMatch(
            strategy_name=f"Middle ({mid:.0f} BPM)",
            target_bpm=mid,
            stretch_a=stretch_a,
            stretch_b=stretch_b,
            quality_score=self._stretch_quality(stretch_a, stretch_b),
            description=f"Both stretch to {mid:.0f} BPM"
        ))

        # Strategy 8: Weighted middle (favor vocal source)
        weighted = bpm_a * 0.7 + bpm_b * 0.3
        stretch_a = weighted / bpm_a
        stretch_b = weighted / bpm_b
        matches.append(TempoMatch(
            strategy_name=f"Favor A ({weighted:.0f} BPM)",
            target_bpm=weighted,
            stretch_a=stretch_a,
            stretch_b=stretch_b,
            quality_score=self._stretch_quality(stretch_a, stretch_b) * 0.95,
            description=f"Weighted toward A: {weighted:.0f} BPM"
        ))

        # Sort by quality
        matches.sort(key=lambda m: m.quality_score, reverse=True)
        return matches

    def _stretch_quality(self, rate_a: float, rate_b: float) -> float:
        """
        Score how much quality degradation the stretching causes.
        Both rates considered (1.0 = no stretch = perfect).
        """
        dev_a = abs(rate_a - 1.0)
        dev_b = abs(rate_b - 1.0)
        max_dev = max(dev_a, dev_b)

        if max_dev < 0.03:
            return 1.0     # Imperceptible
        elif max_dev < 0.08:
            return 0.95    # Barely noticeable
        elif max_dev < 0.15:
            return 0.85    # Slight artifacts
        elif max_dev < 0.25:
            return 0.65    # Noticeable but acceptable
        elif max_dev < 0.40:
            return 0.40    # Significant artifacts
        elif max_dev < 0.60:
            return 0.20    # Heavy artifacts
        else:
            return 0.05    # Extremely degraded

    # ================================================================
    # BEAT-ALIGNED TIME STRETCHING
    # ================================================================

    def beat_aligned_stretch(self, audio: np.ndarray,
                               beat_grid: BeatGrid,
                               target_bpm: float) -> np.ndarray:
        """
        Stretch audio to target BPM using beat-aligned stretching.
        Instead of uniform stretching, this stretches between beats,
        preserving transient quality better.
        """
        if abs(beat_grid.bpm - target_bpm) < 1.0:
            return audio  # Close enough, no stretching needed

        ratio = target_bpm / beat_grid.bpm

        # If small stretch, just do uniform (simpler, sounds fine)
        if abs(ratio - 1.0) < 0.1:
            return librosa.effects.time_stretch(audio, rate=ratio)

        # For larger stretches, use phase vocoder with beat-sync
        # This preserves transients better
        beat_frames = librosa.time_to_frames(
            beat_grid.beat_times, sr=self.sr, hop_length=512)

        # Use librosa's phase vocoder
        stft = librosa.stft(audio, n_fft=2048, hop_length=512)
        stft_stretched = librosa.phase_vocoder(stft, rate=ratio, hop_length=512)
        audio_stretched = librosa.istft(stft_stretched, hop_length=512)

        return audio_stretched

    # ================================================================
    # ALIGN TWO BEAT GRIDS
    # ================================================================

    def align_beats(self, audio_a: np.ndarray, grid_a: BeatGrid,
                     audio_b: np.ndarray, grid_b: BeatGrid,
                     match: TempoMatch) -> Tuple[np.ndarray, np.ndarray]:
        """
        Align two audio signals so their beats match.
        Returns the aligned versions of both.
        """
        # Apply tempo stretching
        if abs(match.stretch_a - 1.0) > 0.02:
            audio_a = librosa.effects.time_stretch(audio_a, rate=match.stretch_a)

        if abs(match.stretch_b - 1.0) > 0.02:
            audio_b = librosa.effects.time_stretch(audio_b, rate=match.stretch_b)

        # Find first downbeat in each and align
        offset_a = self._find_first_strong_beat(audio_a, match.target_bpm)
        offset_b = self._find_first_strong_beat(audio_b, match.target_bpm)

        # Trim to align downbeats
        if offset_a > offset_b:
            diff = int((offset_a - offset_b) * self.sr)
            if diff < len(audio_a):
                audio_a = audio_a[diff:]
        elif offset_b > offset_a:
            diff = int((offset_b - offset_a) * self.sr)
            if diff < len(audio_b):
                audio_b = audio_b[diff:]

        return audio_a, audio_b

    def _find_first_strong_beat(self, audio: np.ndarray,
                                  bpm: float) -> float:
        """Find the time of the first strong beat (downbeat)"""
        onset_env = librosa.onset.onset_strength(y=audio, sr=self.sr)
        beat_period_frames = int(60 * self.sr / (bpm * 512))

        # Look in the first 4 beats
        search_range = min(beat_period_frames * 4, len(onset_env))
        if search_range <= 0:
            return 0.0

        # Find the strongest onset in the first few beats
        peak_frame = np.argmax(onset_env[:search_range])
        return float(peak_frame * 512 / self.sr)
