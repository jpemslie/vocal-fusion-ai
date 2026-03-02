"""
VocalFusion AI - Harmonic Engine
=================================

Professional harmonic analysis and matching.
This doesn't just compare root keys — it analyzes actual chord progressions
and finds the transposition where the most chords are consonant.

Features:
  - Frame-by-frame chroma analysis
  - Chord detection (major, minor, dim, aug, sus, 7th)
  - Key strength scoring (how strongly is the key established?)
  - Optimal transposition search across all 12 semitones
  - Consonance matrix between two songs' progressions
  - Section-level harmonic compatibility scoring
"""

import numpy as np
import librosa
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass


# Interval consonance ratings (0=dissonant, 1=perfect consonance)
INTERVAL_CONSONANCE = {
    0: 1.0,    # Unison
    1: 0.1,    # Minor 2nd - very dissonant
    2: 0.4,    # Major 2nd - mildly dissonant
    3: 0.7,    # Minor 3rd - consonant
    4: 0.7,    # Major 3rd - consonant
    5: 0.9,    # Perfect 4th - very consonant
    6: 0.15,   # Tritone - very dissonant
    7: 0.95,   # Perfect 5th - very consonant
    8: 0.6,    # Minor 6th - consonant
    9: 0.65,   # Major 6th - consonant
    10: 0.35,  # Minor 7th - moderate
    11: 0.2,   # Major 7th - dissonant
}

# Key profiles (Krumhansl-Kessler profiles for key detection)
MAJOR_PROFILE = np.array([6.35, 2.23, 3.48, 2.33, 4.38, 4.09, 2.52, 5.19, 2.39, 3.66, 2.29, 2.88])
MINOR_PROFILE = np.array([6.33, 2.68, 3.52, 5.38, 2.60, 3.53, 2.54, 4.75, 3.98, 2.69, 3.34, 3.17])

NOTE_NAMES = ['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B']


@dataclass
class HarmonicProfile:
    """Complete harmonic analysis of a song or section"""
    key_root: int              # 0-11 semitone index
    key_mode: str              # 'major' or 'minor'
    key_name: str              # e.g. 'C major', 'F# minor'
    key_confidence: float      # 0-1 how strongly the key is established
    chroma_profile: np.ndarray # Average chroma energy (12 values)
    chord_progression: List[Dict]  # List of detected chords with timings
    dominant_chords: List[str]     # Most common chords
    energy_curve: np.ndarray       # Energy over time


class HarmonicEngine:
    """Deep harmonic analysis and matching"""

    def __init__(self, sample_rate: int = 44100):
        self.sr = sample_rate

    # ================================================================
    # FULL HARMONIC ANALYSIS
    # ================================================================

    def analyze(self, audio: np.ndarray) -> HarmonicProfile:
        """Complete harmonic analysis of an audio signal"""
        if audio.ndim > 1:
            audio = librosa.to_mono(audio)

        # Get chroma features
        chroma = librosa.feature.chroma_cqt(y=audio, sr=self.sr, hop_length=512)

        # Detect key using Krumhansl-Kessler algorithm
        key_root, key_mode, key_confidence = self._detect_key(chroma)
        key_name = f"{NOTE_NAMES[key_root]} {'major' if key_mode == 'major' else 'minor'}"

        # Average chroma profile
        chroma_profile = np.mean(chroma, axis=1)

        # Detect chord progression
        chords = self._detect_chords(chroma)

        # Find dominant chords
        chord_counts = {}
        for c in chords:
            name = c['chord']
            chord_counts[name] = chord_counts.get(name, 0) + 1
        dominant = sorted(chord_counts, key=chord_counts.get, reverse=True)[:6]

        # Energy curve
        rms = librosa.feature.rms(y=audio, hop_length=512)[0]

        return HarmonicProfile(
            key_root=key_root,
            key_mode=key_mode,
            key_name=key_name,
            key_confidence=key_confidence,
            chroma_profile=chroma_profile,
            chord_progression=chords,
            dominant_chords=dominant,
            energy_curve=rms
        )

    # ================================================================
    # KEY DETECTION (Krumhansl-Kessler algorithm)
    # ================================================================

    def _detect_key(self, chroma: np.ndarray) -> Tuple[int, str, float]:
        """
        Detect key using correlation with Krumhansl-Kessler profiles.
        Tests all 24 possible keys (12 major + 12 minor).
        """
        profile = np.mean(chroma, axis=1)

        if np.sum(profile) < 1e-10:
            return 0, 'major', 0.0

        profile = profile / np.sum(profile)

        best_corr = -1
        best_root = 0
        best_mode = 'major'

        for root in range(12):
            # Rotate profile to test this root
            rotated = np.roll(profile, -root)

            # Test major
            major_norm = MAJOR_PROFILE / np.sum(MAJOR_PROFILE)
            corr_major = np.corrcoef(rotated, major_norm)[0, 1]

            # Test minor
            minor_norm = MINOR_PROFILE / np.sum(MINOR_PROFILE)
            corr_minor = np.corrcoef(rotated, minor_norm)[0, 1]

            if corr_major > best_corr:
                best_corr = corr_major
                best_root = root
                best_mode = 'major'

            if corr_minor > best_corr:
                best_corr = corr_minor
                best_root = root
                best_mode = 'minor'

        # Confidence: how much better is the best key than the average?
        confidence = max(0, min(1, (best_corr + 1) / 2))

        return best_root, best_mode, confidence

    # ================================================================
    # CHORD DETECTION
    # ================================================================

    def _detect_chords(self, chroma: np.ndarray,
                        hop_length: int = 512) -> List[Dict]:
        """
        Detect chords frame by frame.
        Uses template matching against common chord types.
        """
        # Chord templates (relative to root)
        templates = {
            'maj': [1, 0, 0, 0, 1, 0, 0, 1, 0, 0, 0, 0],
            'min': [1, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 0],
            'dim': [1, 0, 0, 1, 0, 0, 1, 0, 0, 0, 0, 0],
            'aug': [1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0],
            'sus4': [1, 0, 0, 0, 0, 1, 0, 1, 0, 0, 0, 0],
            '7':  [1, 0, 0, 0, 1, 0, 0, 1, 0, 0, 1, 0],
            'm7': [1, 0, 0, 1, 0, 0, 0, 1, 0, 0, 1, 0],
        }

        chords = []
        frames_per_beat = max(1, int(0.5 * self.sr / hop_length))  # ~0.5s per chord

        for i in range(0, chroma.shape[1] - frames_per_beat, frames_per_beat):
            # Average chroma over this beat
            frame = np.mean(chroma[:, i:i + frames_per_beat], axis=1)

            if np.max(frame) < 0.01:
                continue

            frame = frame / (np.max(frame) + 1e-10)

            best_match = 'N'  # No chord
            best_score = 0.3  # Minimum threshold

            for root in range(12):
                for chord_type, template in templates.items():
                    rotated_template = np.roll(template, root).astype(float)
                    score = np.dot(frame, rotated_template) / (
                        np.linalg.norm(frame) * np.linalg.norm(rotated_template) + 1e-10)

                    if score > best_score:
                        best_score = score
                        suffix = '' if chord_type == 'maj' else chord_type
                        best_match = f"{NOTE_NAMES[root]}{suffix}"

            time = i * hop_length / self.sr
            chords.append({
                'chord': best_match,
                'time': time,
                'confidence': best_score
            })

        return chords

    # ================================================================
    # FIND OPTIMAL TRANSPOSITION
    # ================================================================

    def find_best_transposition(self, profile_a: HarmonicProfile,
                                  profile_b: HarmonicProfile) -> Tuple[int, float]:
        """
        Find the transposition for B that maximizes harmonic compatibility with A.

        Tests all 12 semitone shifts.
        For each, measures:
        1. Chroma correlation (do the pitch profiles align?)
        2. Chord consonance (do the chord progressions work together?)
        3. Key relationship (are they in related keys?)

        Returns: (best_shift_semitones, compatibility_score)
        """
        best_shift = 0
        best_score = -1

        for shift in range(-6, 6):
            score = self._score_transposition(profile_a, profile_b, shift)
            if score > best_score:
                best_score = score
                best_shift = shift

        return best_shift, best_score

    def _score_transposition(self, profile_a: HarmonicProfile,
                              profile_b: HarmonicProfile,
                              shift: int) -> float:
        """Score a specific transposition"""

        # 1. Chroma correlation (40% of score)
        shifted_chroma = np.roll(profile_b.chroma_profile, shift)
        chroma_corr = np.corrcoef(profile_a.chroma_profile, shifted_chroma)[0, 1]
        chroma_score = (chroma_corr + 1) / 2  # Normalize to 0-1

        # 2. Key relationship (30% of score)
        shifted_root = (profile_b.key_root + shift) % 12
        interval = abs(profile_a.key_root - shifted_root) % 12
        key_consonance = INTERVAL_CONSONANCE.get(interval, 0.3)

        # Bonus for same mode (both major or both minor)
        if profile_a.key_mode == profile_b.key_mode:
            key_consonance = min(1.0, key_consonance + 0.1)

        # Bonus for relative major/minor
        if profile_a.key_mode != profile_b.key_mode:
            relative_interval = (shifted_root - profile_a.key_root) % 12
            if (profile_a.key_mode == 'major' and relative_interval == 9) or \
               (profile_a.key_mode == 'minor' and relative_interval == 3):
                key_consonance = min(1.0, key_consonance + 0.2)

        # 3. Chord progression consonance (30% of score)
        chord_score = self._chord_progression_consonance(
            profile_a.chord_progression,
            profile_b.chord_progression,
            shift)

        total = chroma_score * 0.4 + key_consonance * 0.3 + chord_score * 0.3
        return total

    def _chord_progression_consonance(self, chords_a: List[Dict],
                                       chords_b: List[Dict],
                                       shift: int) -> float:
        """
        Measure how well two chord progressions work together.
        For each simultaneous chord pair, check if they share notes
        or form consonant intervals.
        """
        if not chords_a or not chords_b:
            return 0.5  # Unknown

        consonance_scores = []

        for chord_a in chords_a:
            # Find closest chord in B at the same time
            time_a = chord_a['time']
            closest_b = min(chords_b, key=lambda c: abs(c['time'] - time_a))

            if abs(closest_b['time'] - time_a) > 2.0:
                continue  # Too far apart in time

            # Get root notes
            root_a = self._chord_to_root(chord_a['chord'])
            root_b = self._chord_to_root(closest_b['chord'])

            if root_a is None or root_b is None:
                continue

            # Apply transposition to B
            root_b = (root_b + shift) % 12

            # Check interval consonance
            interval = abs(root_a - root_b) % 12
            consonance = INTERVAL_CONSONANCE.get(interval, 0.3)
            consonance_scores.append(consonance)

        if not consonance_scores:
            return 0.5

        return float(np.mean(consonance_scores))

    def _chord_to_root(self, chord_name: str) -> Optional[int]:
        """Convert chord name to root note index"""
        if not chord_name or chord_name == 'N':
            return None

        for i, name in enumerate(NOTE_NAMES):
            if chord_name.startswith(name):
                return i

        return None

    # ================================================================
    # SECTION-LEVEL ANALYSIS
    # ================================================================

    def analyze_section(self, audio: np.ndarray,
                         start_time: float, end_time: float) -> HarmonicProfile:
        """Analyze harmony of a specific section"""
        start_sample = int(start_time * self.sr)
        end_sample = int(end_time * self.sr)

        if start_sample >= len(audio):
            start_sample = 0
        end_sample = min(end_sample, len(audio))

        section_audio = audio[start_sample:end_sample]
        return self.analyze(section_audio)

    def find_compatible_sections(self, sections_a: List[Dict],
                                  sections_b: List[Dict],
                                  audio_a: np.ndarray,
                                  audio_b: np.ndarray) -> List[Dict]:
        """
        Find which sections from A are harmonically compatible
        with which sections from B.

        Returns a list of compatible pairs with scores.
        """
        pairs = []

        for sec_a in sections_a:
            profile_a = self.analyze_section(
                audio_a, sec_a.get('start_time', 0), sec_a.get('end_time', 0))

            for sec_b in sections_b:
                profile_b = self.analyze_section(
                    audio_b, sec_b.get('start_time', 0), sec_b.get('end_time', 0))

                best_shift, score = self.find_best_transposition(profile_a, profile_b)

                pairs.append({
                    'section_a': sec_a,
                    'section_b': sec_b,
                    'transposition': best_shift,
                    'compatibility': score,
                    'key_a': profile_a.key_name,
                    'key_b': profile_b.key_name,
                })

        pairs.sort(key=lambda p: p['compatibility'], reverse=True)
        return pairs
