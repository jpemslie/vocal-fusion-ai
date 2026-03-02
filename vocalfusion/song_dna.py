"""
VocalFusion AI — SongDNA: Deep Song Analysis
===============================================

Analyzes a song's stems to produce a detailed "DNA" profile:
  - Beat grid: sample-accurate beat positions
  - Energy curve: RMS over time (1 value per beat)
  - Vocal map: where vocals are active, phrase boundaries
  - Section boundaries: real boundaries using novelty detection
  - Section profiles: energy, spectral character, has_vocals, role

The existing analysis.json has garbage section data (0.07s "chorus").
This module re-analyzes from the actual audio at mix time.
"""

import numpy as np
import librosa
from dataclasses import dataclass, field
from typing import List, Optional, Dict


@dataclass
class BeatGrid:
    """Sample-accurate beat positions"""
    tempo: float = 120.0
    beat_times: np.ndarray = field(default_factory=lambda: np.array([]))
    downbeat_times: np.ndarray = field(default_factory=lambda: np.array([]))
    beat_samples: np.ndarray = field(default_factory=lambda: np.array([], dtype=int))
    bars: int = 0
    beats_per_bar: int = 4
    tempo_stable: bool = True

    def nearest_beat(self, time_s):
        """Find the nearest beat to a given time"""
        if len(self.beat_times) == 0:
            return time_s
        idx = np.argmin(np.abs(self.beat_times - time_s))
        return self.beat_times[idx]

    def nearest_downbeat(self, time_s):
        """Find the nearest downbeat (bar start) to a given time"""
        if len(self.downbeat_times) == 0:
            return self.nearest_beat(time_s)
        idx = np.argmin(np.abs(self.downbeat_times - time_s))
        return self.downbeat_times[idx]

    def beats_in_range(self, start_s, end_s):
        """Count beats in a time range"""
        mask = (self.beat_times >= start_s) & (self.beat_times <= end_s)
        return int(np.sum(mask))

    def bar_duration(self):
        """Duration of one bar in seconds"""
        return 60.0 / self.tempo * self.beats_per_bar


@dataclass
class VocalPhrase:
    """A detected vocal phrase"""
    start_time: float = 0.0
    end_time: float = 0.0
    energy: float = 0.0  # RMS energy of this phrase
    is_strong: bool = False  # Is this a prominent vocal moment?


@dataclass
class Section:
    """A detected section of a song"""
    start_time: float = 0.0
    end_time: float = 0.0
    duration: float = 0.0
    energy: float = 0.0  # Average RMS energy (0-1)
    spectral_centroid: float = 0.0  # Brightness
    has_vocals: bool = False
    vocal_density: float = 0.0  # 0-1, how much of section has vocals
    classification: str = "unknown"  # verse, chorus, intro, outro, drop, breakdown, buildup
    n_beats: int = 0
    n_bars: int = 0


@dataclass
class SongDNA:
    """Complete analysis profile for one song"""
    duration: float = 0.0
    sample_rate: int = 44100
    beat_grid: BeatGrid = field(default_factory=BeatGrid)
    sections: List[Section] = field(default_factory=list)
    vocal_phrases: List[VocalPhrase] = field(default_factory=list)
    vocal_activity: np.ndarray = field(default_factory=lambda: np.array([]))
    energy_curve: np.ndarray = field(default_factory=lambda: np.array([]))
    energy_curve_times: np.ndarray = field(default_factory=lambda: np.array([]))
    key: str = ""
    key_chroma: np.ndarray = field(default_factory=lambda: np.zeros(12))
    overall_energy: float = 0.0
    has_strong_vocals: bool = False
    has_strong_drums: bool = False


class SongAnalyzer:
    """Analyze a song from its stems to produce SongDNA"""

    def __init__(self, sample_rate: int = 44100):
        self.sr = sample_rate

    def analyze(self, stems: Dict[str, np.ndarray]) -> SongDNA:
        """Full analysis of a song from its stems"""
        dna = SongDNA(sample_rate=self.sr)

        # Get individual stems
        vocals = self._get_mono(stems, 'vocals')
        drums = self._get_mono(stems, 'drums')
        bass = self._get_mono(stems, 'bass')
        other = self._get_mono(stems, 'other')

        # Full mix for some analyses
        full = self._sum_stems(vocals, drums, bass, other)
        dna.duration = len(full) / self.sr

        print(f"      Duration: {dna.duration:.1f}s")

        # 1. Beat grid
        dna.beat_grid = self._detect_beats(full, drums)
        print(f"      Tempo: {dna.beat_grid.tempo:.1f} BPM, "
              f"{dna.beat_grid.bars} bars, "
              f"{len(dna.beat_grid.beat_times)} beats")

        # 2. Energy curve (per beat)
        dna.energy_curve, dna.energy_curve_times = self._compute_energy_curve(
            full, dna.beat_grid)
        dna.overall_energy = float(np.mean(dna.energy_curve)) if len(dna.energy_curve) > 0 else 0

        # 3. Vocal analysis
        dna.vocal_activity = self._detect_vocal_activity(vocals)
        dna.vocal_phrases = self._find_vocal_phrases(vocals, dna.vocal_activity)
        dna.has_strong_vocals = any(p.is_strong for p in dna.vocal_phrases)
        print(f"      Vocals: {len(dna.vocal_phrases)} phrases, "
              f"strong={dna.has_strong_vocals}")

        # 4. Drum strength
        if drums is not None:
            drum_rms = np.sqrt(np.mean(drums ** 2))
            dna.has_strong_drums = drum_rms > 0.02
        print(f"      Drums: strong={dna.has_strong_drums}")

        # 5. Key detection
        dna.key, dna.key_chroma = self._detect_key(full)
        print(f"      Key: {dna.key}")

        # 6. Section detection (the important one)
        dna.sections = self._detect_sections(
            full, vocals, drums, dna.beat_grid,
            dna.energy_curve, dna.energy_curve_times,
            dna.vocal_activity)
        print(f"      Sections: {len(dna.sections)}")
        for s in dna.sections:
            print(f"        {s.start_time:5.1f}-{s.end_time:5.1f}s "
                  f"({s.duration:5.1f}s) | {s.classification:12s} "
                  f"| e={s.energy:.2f} vox={s.has_vocals}")

        return dna

    # ================================================================
    # BEAT GRID DETECTION
    # ================================================================

    def _detect_beats(self, full, drums):
        """Detect beats using drums (preferred) or full mix"""
        grid = BeatGrid()

        # Use drums if strong enough, otherwise full mix
        source = drums if drums is not None and np.sqrt(np.mean(drums**2)) > 0.01 else full

        # Get tempo
        tempo, beat_frames = librosa.beat.beat_track(
            y=source, sr=self.sr, units='frames')

        # Handle tempo as array (newer librosa)
        if hasattr(tempo, '__len__'):
            tempo = float(tempo[0]) if len(tempo) > 0 else 120.0
        tempo = float(tempo)

        # Resolve to reasonable range (80-170 BPM)
        if tempo < 70:
            tempo *= 2
        elif tempo > 180:
            tempo /= 2

        grid.tempo = tempo

        # Get beat times
        beat_times = librosa.frames_to_time(beat_frames, sr=self.sr)
        grid.beat_times = beat_times
        grid.beat_samples = librosa.frames_to_samples(beat_frames)

        # Downbeats: every 4th beat
        if len(beat_times) >= 4:
            # Try to find the strongest beat as the downbeat
            beat_strengths = np.array([
                np.sqrt(np.mean(source[max(0,s-256):s+256]**2))
                for s in grid.beat_samples[:min(16, len(grid.beat_samples))]
            ])
            if len(beat_strengths) >= 4:
                # Find which offset (0,1,2,3) has strongest average
                offsets = [np.mean(beat_strengths[i::4]) for i in range(4)]
                best_offset = int(np.argmax(offsets))
            else:
                best_offset = 0

            grid.downbeat_times = beat_times[best_offset::4]
        else:
            grid.downbeat_times = beat_times

        grid.bars = len(grid.downbeat_times)

        # Check tempo stability
        if len(beat_times) > 4:
            intervals = np.diff(beat_times)
            cv = np.std(intervals) / np.mean(intervals) if np.mean(intervals) > 0 else 1
            grid.tempo_stable = cv < 0.15

        return grid

    # ================================================================
    # ENERGY CURVE
    # ================================================================

    def _compute_energy_curve(self, full, beat_grid):
        """Compute energy at each beat position"""
        if len(beat_grid.beat_times) < 2:
            # Fallback: compute in 0.5s windows
            hop = int(0.5 * self.sr)
            n = max(1, len(full) // hop)
            energy = np.array([
                np.sqrt(np.mean(full[i*hop:(i+1)*hop]**2))
                for i in range(n)
            ])
            times = np.arange(n) * 0.5
        else:
            # Compute RMS at each beat
            energy = []
            times = []
            for i in range(len(beat_grid.beat_times) - 1):
                s = int(beat_grid.beat_times[i] * self.sr)
                e = int(beat_grid.beat_times[i+1] * self.sr)
                s = max(0, min(s, len(full)-1))
                e = max(s+1, min(e, len(full)))
                chunk = full[s:e]
                energy.append(np.sqrt(np.mean(chunk**2)))
                times.append(beat_grid.beat_times[i])
            energy = np.array(energy)
            times = np.array(times)

        # Normalize 0-1
        if len(energy) > 0 and np.max(energy) > 0:
            energy = energy / np.max(energy)

        return energy, times

    # ================================================================
    # VOCAL ACTIVITY DETECTION
    # ================================================================

    def _detect_vocal_activity(self, vocals, hop_s=0.25):
        """Detect where vocals are active, returns array of 0/1 per time window"""
        if vocals is None:
            return np.array([])

        hop = int(hop_s * self.sr)
        n_frames = max(1, len(vocals) // hop)
        activity = np.zeros(n_frames)

        # RMS in each window
        rms_values = np.array([
            np.sqrt(np.mean(vocals[i*hop:(i+1)*hop]**2))
            for i in range(n_frames)
        ])

        if np.max(rms_values) < 1e-6:
            return activity

        # Normalize
        rms_norm = rms_values / np.max(rms_values)

        # Threshold: vocal is active when above 15% of peak
        # Use adaptive threshold based on distribution
        median_rms = np.median(rms_norm[rms_norm > 0.01]) if np.any(rms_norm > 0.01) else 0.1
        threshold = max(0.10, median_rms * 0.5)

        activity = (rms_norm > threshold).astype(float)

        # Smooth: fill short gaps (< 1 second)
        gap_frames = int(1.0 / hop_s)
        for i in range(1, len(activity) - 1):
            if activity[i] == 0:
                # Check if surrounded by active frames within gap distance
                left = max(0, i - gap_frames)
                right = min(len(activity), i + gap_frames)
                if np.any(activity[left:i] > 0) and np.any(activity[i+1:right] > 0):
                    activity[i] = 0.5  # Bridge gap

        # Remove very short activations (< 0.5s)
        min_frames = int(0.5 / hop_s)
        i = 0
        while i < len(activity):
            if activity[i] > 0:
                j = i
                while j < len(activity) and activity[j] > 0:
                    j += 1
                if j - i < min_frames:
                    activity[i:j] = 0
                i = j
            else:
                i += 1

        return activity

    def _find_vocal_phrases(self, vocals, activity, hop_s=0.25):
        """Find distinct vocal phrases from the activity map"""
        if vocals is None or len(activity) == 0:
            return []

        phrases = []
        hop = int(hop_s * self.sr)
        in_phrase = False
        start_frame = 0

        for i in range(len(activity)):
            if activity[i] > 0 and not in_phrase:
                in_phrase = True
                start_frame = i
            elif activity[i] == 0 and in_phrase:
                in_phrase = False
                start_time = start_frame * hop_s
                end_time = i * hop_s

                # Compute energy of this phrase
                s = int(start_time * self.sr)
                e = min(int(end_time * self.sr), len(vocals))
                phrase_audio = vocals[s:e]
                energy = float(np.sqrt(np.mean(phrase_audio**2))) if len(phrase_audio) > 0 else 0

                phrases.append(VocalPhrase(
                    start_time=start_time,
                    end_time=end_time,
                    energy=energy,
                    is_strong=energy > 0.03
                ))

        # Close any open phrase
        if in_phrase:
            start_time = start_frame * hop_s
            end_time = len(activity) * hop_s
            s = int(start_time * self.sr)
            e = min(int(end_time * self.sr), len(vocals))
            phrase_audio = vocals[s:e]
            energy = float(np.sqrt(np.mean(phrase_audio**2))) if len(phrase_audio) > 0 else 0
            phrases.append(VocalPhrase(
                start_time=start_time,
                end_time=end_time,
                energy=energy,
                is_strong=energy > 0.03
            ))

        return phrases

    # ================================================================
    # KEY DETECTION
    # ================================================================

    def _detect_key(self, full):
        """Detect musical key using chroma features"""
        chroma = librosa.feature.chroma_cqt(y=full, sr=self.sr)
        chroma_profile = np.mean(chroma, axis=1)

        # Krumhansl-Kessler profiles
        major_profile = np.array([6.35, 2.23, 3.48, 2.33, 4.38, 4.09,
                                   2.52, 5.19, 2.39, 3.66, 2.29, 2.88])
        minor_profile = np.array([6.33, 2.68, 3.52, 5.38, 2.60, 3.53,
                                   2.54, 4.75, 3.98, 2.69, 3.34, 3.17])

        note_names = ['C', 'C#', 'D', 'D#', 'E', 'F',
                      'F#', 'G', 'G#', 'A', 'A#', 'B']

        best_corr = -2
        best_key = "C major"

        for root in range(12):
            # Rotate profile to test each root
            rotated = np.roll(chroma_profile, -root)

            # Test major
            corr = float(np.corrcoef(rotated, major_profile)[0, 1])
            if corr > best_corr:
                best_corr = corr
                best_key = f"{note_names[root]} major"

            # Test minor
            corr = float(np.corrcoef(rotated, minor_profile)[0, 1])
            if corr > best_corr:
                best_corr = corr
                best_key = f"{note_names[root]} minor"

        return best_key, chroma_profile

    # ================================================================
    # SECTION DETECTION — THE CRITICAL ONE
    # ================================================================

    def _detect_sections(self, full, vocals, drums, beat_grid,
                          energy_curve, energy_times, vocal_activity):
        """
        Detect real sections using multiple novelty signals.

        Approach:
        1. Compute novelty functions from MFCC, chroma, and energy
        2. Find peaks in combined novelty (= section boundaries)
        3. Snap boundaries to nearest downbeat
        4. Classify each section by energy + vocal content + position
        """

        # === Novelty functions ===

        # MFCC novelty (timbral changes)
        mfcc = librosa.feature.mfcc(y=full, sr=self.sr, n_mfcc=13, hop_length=2048)
        mfcc_delta = np.sqrt(np.sum(np.diff(mfcc, axis=1)**2, axis=0))
        mfcc_novelty = self._smooth(mfcc_delta, window=16)

        # Chroma novelty (harmonic changes)
        chroma = librosa.feature.chroma_cqt(y=full, sr=self.sr, hop_length=2048)
        chroma_delta = np.sqrt(np.sum(np.diff(chroma, axis=1)**2, axis=0))
        chroma_novelty = self._smooth(chroma_delta, window=16)

        # Energy novelty (dynamic changes)
        rms = librosa.feature.rms(y=full, hop_length=2048)[0]
        rms_delta = np.abs(np.diff(rms))
        energy_novelty = self._smooth(rms_delta, window=16)

        # Normalize each to 0-1
        mfcc_novelty = self._norm01(mfcc_novelty)
        chroma_novelty = self._norm01(chroma_novelty)
        energy_novelty = self._norm01(energy_novelty)

        # Combined novelty
        min_len = min(len(mfcc_novelty), len(chroma_novelty), len(energy_novelty))
        combined = (mfcc_novelty[:min_len] * 0.40 +
                   chroma_novelty[:min_len] * 0.35 +
                   energy_novelty[:min_len] * 0.25)

        # Convert to time
        novelty_times = librosa.frames_to_time(
            np.arange(min_len), sr=self.sr, hop_length=2048)

        # === Find peaks (section boundaries) ===
        # Minimum section length: 8 seconds (roughly 4 bars at 120bpm)
        min_section_s = 8.0
        min_frames = int(min_section_s * self.sr / 2048)

        # Adaptive threshold: peaks must be above median + 0.5*std
        threshold = np.median(combined) + 0.5 * np.std(combined)

        boundary_frames = []
        last_boundary = -min_frames
        for i in range(1, len(combined) - 1):
            if (combined[i] > combined[i-1] and
                combined[i] > combined[i+1] and
                combined[i] > threshold and
                i - last_boundary >= min_frames):
                boundary_frames.append(i)
                last_boundary = i

        # Convert to times
        boundary_times = [novelty_times[f] for f in boundary_frames if f < len(novelty_times)]

        # Add start and end
        boundary_times = [0.0] + boundary_times + [len(full) / self.sr]

        # Snap to nearest downbeat if we have a beat grid
        if len(beat_grid.downbeat_times) > 0:
            snapped = [0.0]
            for t in boundary_times[1:-1]:
                snapped.append(beat_grid.nearest_downbeat(t))
            snapped.append(len(full) / self.sr)
            # Remove duplicates
            boundary_times = sorted(set(snapped))

        # Ensure minimum section length after snapping
        cleaned = [boundary_times[0]]
        for t in boundary_times[1:]:
            if t - cleaned[-1] >= min_section_s * 0.75:
                cleaned.append(t)
            elif t == boundary_times[-1]:
                # Extend last section
                cleaned[-1] = t
        boundary_times = cleaned

        # === Build section objects ===
        sections = []
        for i in range(len(boundary_times) - 1):
            start_t = boundary_times[i]
            end_t = boundary_times[i + 1]
            dur = end_t - start_t

            if dur < 2.0:
                continue

            s_samp = int(start_t * self.sr)
            e_samp = min(int(end_t * self.sr), len(full))

            # Energy
            chunk = full[s_samp:e_samp]
            sec_energy = float(np.sqrt(np.mean(chunk**2)))

            # Spectral centroid
            sc = librosa.feature.spectral_centroid(y=chunk, sr=self.sr)
            sec_centroid = float(np.mean(sc))

            # Vocal presence
            if len(vocal_activity) > 0:
                hop_s = 0.25
                va_start = int(start_t / hop_s)
                va_end = int(end_t / hop_s)
                va_segment = vocal_activity[va_start:min(va_end, len(vocal_activity))]
                vocal_density = float(np.mean(va_segment > 0)) if len(va_segment) > 0 else 0
            else:
                vocal_density = 0

            # Beats in this section
            n_beats = beat_grid.beats_in_range(start_t, end_t)
            n_bars = n_beats // beat_grid.beats_per_bar

            sec = Section(
                start_time=start_t,
                end_time=end_t,
                duration=dur,
                energy=sec_energy,
                spectral_centroid=sec_centroid,
                has_vocals=vocal_density > 0.25,
                vocal_density=vocal_density,
                n_beats=n_beats,
                n_bars=n_bars,
            )

            sections.append(sec)

        # === Classify sections ===
        if sections:
            max_energy = max(s.energy for s in sections)
            if max_energy > 0:
                for s in sections:
                    s.energy = s.energy / max_energy  # Normalize to 0-1

            self._classify_sections(sections)

        return sections

    def _classify_sections(self, sections):
        """Classify sections based on energy, vocals, and position"""
        n = len(sections)
        if n == 0:
            return

        for i, sec in enumerate(sections):
            position = i / max(n - 1, 1)  # 0 = start, 1 = end

            # Simple rules that actually work:
            if position < 0.08 and sec.energy < 0.5:
                sec.classification = "intro"
            elif position > 0.90 and sec.energy < 0.5:
                sec.classification = "outro"
            elif sec.energy >= 0.75 and sec.has_vocals:
                sec.classification = "chorus"
            elif sec.energy >= 0.80 and not sec.has_vocals:
                sec.classification = "drop"
            elif sec.energy < 0.35:
                sec.classification = "breakdown"
            elif sec.has_vocals and sec.energy < 0.75:
                sec.classification = "verse"
            elif not sec.has_vocals and sec.energy >= 0.5:
                # Check if energy is rising (buildup)
                if i > 0 and sec.energy > sections[i-1].energy + 0.1:
                    sec.classification = "buildup"
                else:
                    sec.classification = "instrumental"
            else:
                sec.classification = "verse"

    # ================================================================
    # UTILITIES
    # ================================================================

    def _get_mono(self, stems, key):
        audio = stems.get(key)
        if audio is None:
            return None
        if audio.ndim > 1:
            return np.mean(audio, axis=0)
        return audio

    def _sum_stems(self, *stems):
        parts = [s for s in stems if s is not None]
        if not parts:
            return np.zeros(self.sr * 10)
        mx = max(len(p) for p in parts)
        out = np.zeros(mx)
        for p in parts:
            out[:len(p)] += p
        return out

    def _smooth(self, x, window=8):
        if len(x) < window:
            return x
        kernel = np.ones(window) / window
        return np.convolve(x, kernel, mode='same')

    def _norm01(self, x):
        if len(x) == 0:
            return x
        mn, mx = np.min(x), np.max(x)
        if mx - mn < 1e-10:
            return np.zeros_like(x)
        return (x - mn) / (mx - mn)
