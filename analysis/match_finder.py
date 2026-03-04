"""
VocalFusion AI — MatchFinder: Section Compatibility Analysis
==============================================================

Given two SongDNA profiles, finds which sections from A pair well
with sections from B.

For each possible pairing, scores:
  - Energy match: similar energy levels sound intentional
  - Spectral complement: different frequency ranges = less masking
  - Key compatibility: chroma correlation between the sections
  - Role fit: vocals from one + instrumental from other = ideal

Output: ranked list of MatchPairings, each describing exactly
what to take from each song and how they fit together.
"""

import numpy as np
import librosa
from dataclasses import dataclass, field
from typing import List, Dict, Optional, Tuple

from analysis.song_dna import SongDNA, Section


@dataclass
class SectionPairing:
    """A pairing of one section from each song"""
    section_a_idx: int = 0
    section_b_idx: int = 0
    section_a: Section = field(default_factory=Section)
    section_b: Section = field(default_factory=Section)

    # What to use from each song
    role_a: str = "vocals"     # "vocals", "beat", "both", "none"
    role_b: str = "beat"       # "vocals", "beat", "both", "none"

    # Scores (0-1)
    energy_score: float = 0.0
    spectral_score: float = 0.0
    key_score: float = 0.0
    role_score: float = 0.0
    overall_score: float = 0.0


@dataclass
class MatchResult:
    """Complete matching result between two songs"""
    pairings: List[SectionPairing] = field(default_factory=list)
    best_direction: str = "a_vocals"  # "a_vocals" (A vox + B beat) or "b_vocals"
    direction_score_a: float = 0.0  # Average score for A vocals direction
    direction_score_b: float = 0.0  # Average score for B vocals direction
    tempo_strategy: str = ""
    stretch_ratio: float = 1.0


class MatchFinder:
    """Find compatible section pairings between two songs"""

    def __init__(self, sample_rate: int = 44100):
        self.sr = sample_rate

    def find_matches(self, dna_a: SongDNA, dna_b: SongDNA,
                      stems_a: Dict, stems_b: Dict) -> MatchResult:
        """
        Find all compatible pairings between two songs.

        Returns MatchResult with ranked pairings and best direction.
        """
        result = MatchResult()

        print(f"    Song A: {len(dna_a.sections)} sections, "
              f"tempo={dna_a.beat_grid.tempo:.0f}")
        print(f"    Song B: {len(dna_b.sections)} sections, "
              f"tempo={dna_b.beat_grid.tempo:.0f}")

        # Determine tempo strategy
        result.tempo_strategy, result.stretch_ratio = self._find_tempo_strategy(
            dna_a.beat_grid.tempo, dna_b.beat_grid.tempo)
        print(f"    Tempo strategy: {result.tempo_strategy} "
              f"(ratio={result.stretch_ratio:.3f})")

        # Key compatibility
        key_compat = self._key_compatibility(dna_a.key_chroma, dna_b.key_chroma)
        print(f"    Key compatibility: {key_compat:.2f}")

        # Score all possible pairings
        all_pairings = []

        for i, sec_a in enumerate(dna_a.sections):
            for j, sec_b in enumerate(dna_b.sections):
                # Try both directions
                for direction in ['a_vocals', 'b_vocals']:
                    pairing = self._score_pairing(
                        sec_a, sec_b, i, j, direction, key_compat,
                        dna_a, dna_b, stems_a, stems_b)
                    all_pairings.append(pairing)

        # Sort by overall score
        all_pairings.sort(key=lambda p: p.overall_score, reverse=True)
        result.pairings = all_pairings

        # Determine best direction
        a_vox_scores = [p.overall_score for p in all_pairings if p.role_a == 'vocals']
        b_vox_scores = [p.overall_score for p in all_pairings if p.role_b == 'vocals']

        result.direction_score_a = np.mean(a_vox_scores) if a_vox_scores else 0
        result.direction_score_b = np.mean(b_vox_scores) if b_vox_scores else 0

        if result.direction_score_a >= result.direction_score_b:
            result.best_direction = "a_vocals"
        else:
            result.best_direction = "b_vocals"

        print(f"    Best direction: {result.best_direction} "
              f"(A vox avg={result.direction_score_a:.3f}, "
              f"B vox avg={result.direction_score_b:.3f})")
        print(f"    Top pairings:")
        for p in all_pairings[:5]:
            print(f"      {p.overall_score:.3f} | "
                  f"A[{p.section_a_idx}]({p.section_a.classification}) "
                  f"{p.role_a} + "
                  f"B[{p.section_b_idx}]({p.section_b.classification}) "
                  f"{p.role_b}")

        return result

    def _score_pairing(self, sec_a, sec_b, idx_a, idx_b,
                        direction, key_compat, dna_a, dna_b,
                        stems_a, stems_b):
        """Score a specific pairing of sections"""
        pairing = SectionPairing(
            section_a_idx=idx_a,
            section_b_idx=idx_b,
            section_a=sec_a,
            section_b=sec_b,
        )

        if direction == 'a_vocals':
            pairing.role_a = 'vocals'
            pairing.role_b = 'beat'
        else:
            pairing.role_a = 'beat'
            pairing.role_b = 'vocals'

        # --- Energy compatibility ---
        # Best when the beat section has similar or higher energy than vocal section
        if direction == 'a_vocals':
            vox_energy = sec_a.energy
            beat_energy = sec_b.energy
            vox_has_vocals = sec_a.has_vocals
            beat_has_vocals = sec_b.has_vocals
        else:
            vox_energy = sec_b.energy
            beat_energy = sec_a.energy
            vox_has_vocals = sec_b.has_vocals
            beat_has_vocals = sec_a.has_vocals

        energy_diff = abs(vox_energy - beat_energy)
        pairing.energy_score = max(0, 1.0 - energy_diff * 1.5)

        # Bonus if beat is slightly more energetic (supports vocals)
        if beat_energy > vox_energy:
            pairing.energy_score = min(1.0, pairing.energy_score + 0.1)

        # --- Role fitness ---
        # Best: vocal section has vocals, beat section is instrumental
        pairing.role_score = 0.5  # baseline
        if direction == 'a_vocals':
            if sec_a.has_vocals:
                pairing.role_score += 0.3  # Good: A actually has vocals
            if not sec_b.has_vocals:
                pairing.role_score += 0.2  # Good: B is instrumental
            if sec_a.has_vocals and not sec_b.has_vocals:
                pairing.role_score = 1.0   # Perfect: clean vocal + clean beat
        else:
            if sec_b.has_vocals:
                pairing.role_score += 0.3
            if not sec_a.has_vocals:
                pairing.role_score += 0.2
            if sec_b.has_vocals and not sec_a.has_vocals:
                pairing.role_score = 1.0

        # Penalty: both sections have strong vocals (will clash)
        if sec_a.has_vocals and sec_b.has_vocals:
            pairing.role_score *= 0.5

        # --- Spectral compatibility ---
        # Higher centroid + lower centroid = good complement
        centroid_diff = abs(sec_a.spectral_centroid - sec_b.spectral_centroid)
        max_centroid = max(sec_a.spectral_centroid, sec_b.spectral_centroid, 1)
        pairing.spectral_score = min(1.0, centroid_diff / max_centroid + 0.3)

        # --- Key compatibility ---
        pairing.key_score = key_compat

        # --- Overall ---
        pairing.overall_score = (
            pairing.energy_score * 0.25 +
            pairing.role_score * 0.35 +
            pairing.spectral_score * 0.15 +
            pairing.key_score * 0.25
        )

        # Bonus for matching section types that make musical sense
        type_bonus = self._section_type_bonus(
            sec_a.classification, sec_b.classification, direction)
        pairing.overall_score = min(1.0, pairing.overall_score + type_bonus)

        return pairing

    def _section_type_bonus(self, type_a, type_b, direction):
        """Bonus for musically sensible section combinations"""
        # Chorus vocals over drop beat = great
        if direction == 'a_vocals':
            vox_type, beat_type = type_a, type_b
        else:
            vox_type, beat_type = type_b, type_a

        bonuses = {
            ('chorus', 'drop'): 0.15,
            ('chorus', 'chorus'): 0.10,
            ('verse', 'verse'): 0.08,
            ('verse', 'instrumental'): 0.10,
            ('chorus', 'instrumental'): 0.12,
            ('chorus', 'buildup'): 0.05,
        }
        return bonuses.get((vox_type, beat_type), 0.0)

    def _find_tempo_strategy(self, tempo_a, tempo_b):
        """Find the best tempo matching strategy"""
        # Direct ratio
        direct = tempo_a / tempo_b

        # Half-time / double-time variants
        candidates = [
            ('direct', direct),
            ('half_b', tempo_a / (tempo_b / 2)),
            ('double_b', tempo_a / (tempo_b * 2)),
            ('half_a', (tempo_a / 2) / tempo_b),
            ('double_a', (tempo_a * 2) / tempo_b),
        ]

        # Pick the one closest to 1.0 (least stretching)
        best_name, best_ratio = min(candidates, key=lambda x: abs(x[1] - 1.0))

        # Only accept if stretch is <25%
        if abs(best_ratio - 1.0) > 0.25:
            # Try harder: maybe one is double-time of the other
            extended = candidates + [
                ('half_both', (tempo_a / 2) / (tempo_b / 2)),
            ]
            best_name, best_ratio = min(extended, key=lambda x: abs(x[1] - 1.0))

        return best_name, best_ratio

    def _key_compatibility(self, chroma_a, chroma_b):
        """Score key compatibility by chroma correlation"""
        if len(chroma_a) != 12 or len(chroma_b) != 12:
            return 0.5

        # Test all 12 transpositions and find the best match
        best_corr = -2.0
        for shift in range(12):
            shifted = np.roll(chroma_b, shift)
            corr = float(np.corrcoef(chroma_a, shifted)[0, 1])
            if corr > best_corr:
                best_corr = corr

        # Map correlation to 0-1 score
        # corr > 0.8 = excellent, > 0.5 = good, < 0.2 = poor
        return max(0, min(1.0, (best_corr + 0.5) / 1.5))

    def get_top_pairings(self, result: MatchResult,
                          direction: str = None,
                          n: int = 10) -> List[SectionPairing]:
        """Get the top n pairings, optionally filtered by direction"""
        if direction is None:
            direction = result.best_direction

        filtered = [
            p for p in result.pairings
            if (direction == 'a_vocals' and p.role_a == 'vocals') or
               (direction == 'b_vocals' and p.role_b == 'vocals')
        ]
        return filtered[:n]
