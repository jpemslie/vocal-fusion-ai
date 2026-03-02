"""
VocalFusion AI - Section Engine
================================

Professional song section analysis.

Features:
  - Energy-based section detection
  - Section type classification (intro, verse, chorus, bridge, outro)
  - Energy curve mapping
  - Best section extraction for mashup source material
  - Section compatibility scoring between two songs
  - Transition point detection
"""

import numpy as np
import librosa
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass, field


@dataclass
class Section:
    """A single section of a song"""
    section_type: str       # intro, verse, chorus, bridge, outro, drop, breakdown
    start_time: float       # Start in seconds
    end_time: float         # End in seconds
    duration: float         # Duration in seconds
    energy: float           # Average energy (0-1)
    spectral_centroid: float  # Brightness
    has_vocals: bool        # Whether vocals are active
    confidence: float       # How confident we are in the classification


@dataclass
class SongStructure:
    """Complete structural analysis of a song"""
    sections: List[Section]
    energy_curve: np.ndarray       # Frame-level energy
    spectral_curve: np.ndarray     # Frame-level brightness
    vocal_activity: np.ndarray     # Frame-level vocal presence
    total_duration: float
    peak_energy_time: float        # Where the song peaks
    energy_arc: str                # "building", "constant", "declining", etc.


class SectionEngine:
    """Professional section detection and analysis"""

    def __init__(self, sample_rate: int = 44100):
        self.sr = sample_rate

    # ================================================================
    # FULL STRUCTURE ANALYSIS
    # ================================================================

    def analyze(self, audio: np.ndarray,
                vocal_stem: Optional[np.ndarray] = None) -> SongStructure:
        """Analyze complete song structure"""
        if audio.ndim > 1:
            audio = librosa.to_mono(audio)

        # Compute features
        energy_curve = self._compute_energy_curve(audio)
        spectral_curve = self._compute_spectral_curve(audio)

        # Vocal activity detection
        if vocal_stem is not None:
            vocal_activity = self._detect_vocal_activity(vocal_stem)
        else:
            vocal_activity = self._estimate_vocal_activity(audio)

        # Find section boundaries
        boundaries = self._find_boundaries(audio, energy_curve, spectral_curve)

        # Classify sections
        sections = self._classify_sections(
            audio, boundaries, energy_curve, spectral_curve, vocal_activity)

        # Analyze energy arc
        peak_time = np.argmax(energy_curve) * 512 / self.sr
        energy_arc = self._classify_energy_arc(energy_curve)

        return SongStructure(
            sections=sections,
            energy_curve=energy_curve,
            spectral_curve=spectral_curve,
            vocal_activity=vocal_activity,
            total_duration=len(audio) / self.sr,
            peak_energy_time=peak_time,
            energy_arc=energy_arc
        )

    # ================================================================
    # ENERGY & SPECTRAL CURVES
    # ================================================================

    def _compute_energy_curve(self, audio: np.ndarray) -> np.ndarray:
        """Compute frame-level energy (RMS)"""
        rms = librosa.feature.rms(y=audio, hop_length=512)[0]
        # Normalize to 0-1
        max_rms = np.max(rms)
        if max_rms > 0:
            rms = rms / max_rms
        return rms

    def _compute_spectral_curve(self, audio: np.ndarray) -> np.ndarray:
        """Compute frame-level spectral centroid (brightness)"""
        centroid = librosa.feature.spectral_centroid(
            y=audio, sr=self.sr, hop_length=512)[0]
        # Normalize to 0-1
        max_cent = np.max(centroid)
        if max_cent > 0:
            centroid = centroid / max_cent
        return centroid

    def _detect_vocal_activity(self, vocal_stem: np.ndarray) -> np.ndarray:
        """Detect vocal activity from the vocal stem"""
        if vocal_stem.ndim > 1:
            vocal_stem = librosa.to_mono(vocal_stem)

        rms = librosa.feature.rms(y=vocal_stem, hop_length=512)[0]
        max_rms = np.max(rms)
        if max_rms > 0:
            rms = rms / max_rms

        # Threshold: consider vocals active above 0.1
        activity = (rms > 0.1).astype(float)

        # Smooth with 0.5s window
        smooth_frames = max(1, int(0.5 * self.sr / 512))
        kernel = np.ones(smooth_frames) / smooth_frames
        activity = np.convolve(activity, kernel, mode='same')
        activity = (activity > 0.3).astype(float)

        return activity

    def _estimate_vocal_activity(self, audio: np.ndarray) -> np.ndarray:
        """Estimate vocal activity from full mix using spectral characteristics"""
        # Vocals typically in 200Hz-4kHz range with high harmonic-to-noise ratio
        spectral = np.abs(librosa.stft(audio, n_fft=2048, hop_length=512))
        freqs = librosa.fft_frequencies(sr=self.sr, n_fft=2048)

        # Vocal band energy
        vocal_mask = (freqs >= 200) & (freqs <= 4000)
        vocal_energy = np.mean(spectral[vocal_mask], axis=0)

        # Full band energy
        full_energy = np.mean(spectral, axis=0) + 1e-10

        # Ratio: high ratio = likely vocals present
        ratio = vocal_energy / full_energy
        max_ratio = np.max(ratio)
        if max_ratio > 0:
            ratio = ratio / max_ratio

        # Smooth and threshold
        smooth_frames = max(1, int(0.3 * self.sr / 512))
        kernel = np.ones(smooth_frames) / smooth_frames
        ratio = np.convolve(ratio, kernel, mode='same')
        activity = (ratio > 0.4).astype(float)

        return activity

    # ================================================================
    # BOUNDARY DETECTION
    # ================================================================

    def _find_boundaries(self, audio: np.ndarray,
                          energy: np.ndarray,
                          spectral: np.ndarray) -> List[float]:
        """
        Find section boundaries using novelty detection.
        Combines spectral novelty with energy changes.
        """
        hop = 512

        # Method 1: MFCC-based novelty
        mfcc = librosa.feature.mfcc(y=audio, sr=self.sr,
                                      hop_length=hop, n_mfcc=13)
        novelty_mfcc = self._compute_novelty(mfcc)

        # Method 2: Chroma-based novelty (harmonic changes)
        chroma = librosa.feature.chroma_cqt(y=audio, sr=self.sr,
                                              hop_length=hop)
        novelty_chroma = self._compute_novelty(chroma)

        # Method 3: Energy changes
        novelty_energy = np.abs(np.diff(energy, prepend=energy[0]))

        # Combine novelties
        min_len = min(len(novelty_mfcc), len(novelty_chroma), len(novelty_energy))
        combined = (novelty_mfcc[:min_len] * 0.4 +
                    novelty_chroma[:min_len] * 0.35 +
                    novelty_energy[:min_len] * 0.25)

        # Normalize
        max_nov = np.max(combined)
        if max_nov > 0:
            combined = combined / max_nov

        # Find peaks (section boundaries)
        # Minimum section length: 4 seconds
        min_distance = int(4.0 * self.sr / hop)
        threshold = np.percentile(combined, 75)

        peaks = []
        for i in range(min_distance, len(combined) - min_distance):
            if combined[i] > threshold:
                # Check if it's a local maximum
                window = combined[max(0, i-min_distance//2):
                                  min(len(combined), i+min_distance//2)]
                if combined[i] >= np.max(window):
                    peaks.append(i)

        # Convert to times
        boundaries = [0.0]  # Always start at 0
        for peak in peaks:
            time = peak * hop / self.sr
            # Don't add boundaries too close together
            if time - boundaries[-1] >= 3.0:
                boundaries.append(time)

        # Add end
        total_duration = len(audio) / self.sr
        if total_duration - boundaries[-1] > 2.0:
            boundaries.append(total_duration)
        else:
            boundaries[-1] = total_duration

        return boundaries

    def _compute_novelty(self, feature_matrix: np.ndarray) -> np.ndarray:
        """Compute novelty curve from a feature matrix"""
        # Self-similarity using cosine distance between consecutive frames
        # Use a checkerboard kernel for novelty detection
        norms = np.linalg.norm(feature_matrix, axis=0, keepdims=True)
        norms = np.maximum(norms, 1e-10)
        normalized = feature_matrix / norms

        # Compute frame-to-frame differences
        diffs = np.sum(np.abs(np.diff(normalized, axis=1)), axis=0)
        diffs = np.append(diffs, 0)  # Pad to match length

        # Smooth
        smooth_frames = max(1, int(0.3 * self.sr / 512))
        kernel = np.ones(smooth_frames) / smooth_frames
        diffs = np.convolve(diffs, kernel, mode='same')

        return diffs

    # ================================================================
    # SECTION CLASSIFICATION
    # ================================================================

    def _classify_sections(self, audio: np.ndarray,
                            boundaries: List[float],
                            energy: np.ndarray,
                            spectral: np.ndarray,
                            vocal_activity: np.ndarray) -> List[Section]:
        """Classify each section between boundaries"""
        hop = 512
        sections = []

        for i in range(len(boundaries) - 1):
            start = boundaries[i]
            end = boundaries[i + 1]
            duration = end - start

            # Get feature values for this section
            start_frame = int(start * self.sr / hop)
            end_frame = int(end * self.sr / hop)

            section_energy = np.mean(energy[start_frame:end_frame]) if start_frame < len(energy) else 0.0
            section_spectral = np.mean(spectral[start_frame:end_frame]) if start_frame < len(spectral) else 0.0

            # Vocal activity in this section
            vocal_frames = vocal_activity[start_frame:min(end_frame, len(vocal_activity))]
            has_vocals = np.mean(vocal_frames) > 0.3 if len(vocal_frames) > 0 else False

            # Classify based on features
            section_type, confidence = self._classify_single_section(
                section_energy, section_spectral, has_vocals,
                duration, start, end, len(audio) / self.sr, i, len(boundaries) - 1)

            sections.append(Section(
                section_type=section_type,
                start_time=start,
                end_time=end,
                duration=duration,
                energy=section_energy,
                spectral_centroid=section_spectral,
                has_vocals=has_vocals,
                confidence=confidence
            ))

        return sections

    def _classify_single_section(self, energy, spectral, has_vocals,
                                   duration, start, end, total_duration,
                                   section_idx, total_sections):
        """Classify a single section"""
        # Position-based heuristics
        relative_start = start / total_duration
        relative_end = end / total_duration

        # First section is likely intro
        if section_idx == 0:
            if energy < 0.3 or not has_vocals:
                return 'intro', 0.8
            else:
                return 'verse', 0.5

        # Last section is likely outro
        if section_idx == total_sections - 1:
            if energy < 0.3:
                return 'outro', 0.8
            else:
                return 'chorus', 0.4

        # High energy + vocals = chorus (usually)
        if energy > 0.6 and has_vocals:
            return 'chorus', 0.7

        # High energy + no vocals = drop/instrumental
        if energy > 0.7 and not has_vocals:
            return 'drop', 0.6

        # Low energy + no vocals = breakdown/instrumental
        if energy < 0.3 and not has_vocals:
            return 'breakdown', 0.6

        # Medium energy + vocals = verse
        if has_vocals and energy < 0.6:
            return 'verse', 0.6

        # Low energy, in the middle = bridge
        if energy < 0.4 and relative_start > 0.4 and relative_end < 0.8:
            return 'bridge', 0.5

        # Default: verse
        return 'verse', 0.3

    def _classify_energy_arc(self, energy: np.ndarray) -> str:
        """Classify the overall energy arc of the song"""
        if len(energy) < 10:
            return 'constant'

        # Split into thirds
        third = len(energy) // 3
        energy_1 = np.mean(energy[:third])
        energy_2 = np.mean(energy[third:2*third])
        energy_3 = np.mean(energy[2*third:])

        if energy_2 > energy_1 * 1.3 and energy_2 > energy_3 * 1.2:
            return 'peak_middle'      # Build → Peak → Decline
        elif energy_3 > energy_1 * 1.3:
            return 'building'          # Builds to end
        elif energy_1 > energy_3 * 1.3:
            return 'declining'         # Starts high, declines
        else:
            return 'constant'          # Relatively flat

    # ================================================================
    # BEST SECTION EXTRACTION
    # ================================================================

    def find_best_sections(self, structure: SongStructure,
                            target_duration: float = 60.0,
                            prefer_type: str = 'chorus') -> List[Section]:
        """
        Find the best sections for mashup source material.
        Prioritizes: chorus > drop > verse with vocals > anything else.

        Returns sections that sum to approximately target_duration.
        """
        # Score each section
        scored = []
        for section in structure.sections:
            score = self._score_section_for_mashup(section, prefer_type)
            scored.append((section, score))

        # Sort by score (best first)
        scored.sort(key=lambda x: x[1], reverse=True)

        # Collect sections up to target duration
        selected = []
        total_duration = 0

        for section, score in scored:
            if total_duration >= target_duration:
                break
            if section.duration < 3.0:  # Skip very short sections
                continue

            selected.append(section)
            total_duration += section.duration

        # Sort selected by time (maintain chronological order)
        selected.sort(key=lambda s: s.start_time)

        return selected

    def _score_section_for_mashup(self, section: Section,
                                    prefer_type: str) -> float:
        """Score how good a section is for mashup source material"""
        type_scores = {
            'chorus': 1.0,
            'drop': 0.95,
            'verse': 0.6,
            'pre_chorus': 0.7,
            'bridge': 0.5,
            'breakdown': 0.3,
            'intro': 0.2,
            'outro': 0.15,
        }

        base_score = type_scores.get(section.section_type, 0.4)

        # Bonus for preferred type
        if section.section_type == prefer_type:
            base_score += 0.2

        # Energy bonus (higher energy = better for mashups)
        energy_bonus = section.energy * 0.3

        # Vocal bonus (sections with vocals are usually more interesting)
        vocal_bonus = 0.15 if section.has_vocals else 0

        # Duration bonus (too short or too long is bad)
        if 15 < section.duration < 45:
            duration_bonus = 0.1
        else:
            duration_bonus = 0

        # Confidence bonus
        conf_bonus = section.confidence * 0.1

        return base_score + energy_bonus + vocal_bonus + duration_bonus + conf_bonus

    # ================================================================
    # EXTRACT STEMS FOR A SECTION
    # ================================================================

    def extract_section_stems(self, stems: Dict[str, np.ndarray],
                                section: Section) -> Dict[str, np.ndarray]:
        """Extract stems for a specific section"""
        start_sample = int(section.start_time * self.sr)
        end_sample = int(section.end_time * self.sr)

        extracted = {}
        for name, audio in stems.items():
            if audio is None:
                extracted[name] = None
                continue

            if audio.ndim > 1:
                audio = librosa.to_mono(audio)

            if start_sample < len(audio):
                extracted[name] = audio[start_sample:min(end_sample, len(audio))].copy()
            else:
                extracted[name] = np.zeros(end_sample - start_sample)

        return extracted

    def extract_best_material(self, stems: Dict[str, np.ndarray],
                                structure: SongStructure,
                                target_duration: float = 60.0) -> Dict:
        """
        Extract the best material from a song for mashup use.
        Returns combined stems from the best sections.
        """
        best_sections = self.find_best_sections(structure, target_duration)

        if not best_sections:
            # Fallback: use the highest-energy 30 seconds
            return self._extract_energy_window(stems, target_duration)

        # Extract and concatenate stems from best sections
        combined = {}
        for name in stems:
            parts = []
            for section in best_sections:
                section_stems = self.extract_section_stems(stems, section)
                if section_stems[name] is not None:
                    parts.append(section_stems[name])
            if parts:
                combined[name] = np.concatenate(parts)
            else:
                combined[name] = None

        total_dur = sum(s.duration for s in best_sections)
        return {
            'stems': combined,
            'sections': best_sections,
            'total_duration': total_dur,
            'energy': np.mean([s.energy for s in best_sections])
        }

    def _extract_energy_window(self, stems: Dict, target_duration: float) -> Dict:
        """Fallback: extract highest-energy window"""
        # Mix all stems to find energy
        parts = []
        for v in stems.values():
            if v is not None:
                audio = librosa.to_mono(v) if v.ndim > 1 else v
                parts.append(audio)

        if not parts:
            return {'stems': stems, 'sections': [], 'total_duration': 0, 'energy': 0}

        max_len = max(len(p) for p in parts)
        full_mix = np.zeros(max_len)
        for p in parts:
            full_mix[:len(p)] += p

        window = int(target_duration * self.sr)
        if window >= len(full_mix):
            return {'stems': stems, 'sections': [],
                    'total_duration': len(full_mix)/self.sr,
                    'energy': np.sqrt(np.mean(full_mix**2))}

        # Slide window
        hop = self.sr
        best_start = 0
        best_energy = 0
        for start in range(0, len(full_mix) - window, hop):
            energy = np.sqrt(np.mean(full_mix[start:start+window] ** 2))
            if energy > best_energy:
                best_energy = energy
                best_start = start

        # Extract
        extracted = {}
        for name, audio in stems.items():
            if audio is None:
                extracted[name] = None
                continue
            if audio.ndim > 1:
                audio = librosa.to_mono(audio)
            extracted[name] = audio[best_start:best_start+window].copy()

        return {
            'stems': extracted, 'sections': [],
            'total_duration': target_duration, 'energy': best_energy
        }
