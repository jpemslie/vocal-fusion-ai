"""
VocalFusion AI — DJArranger: Intelligent Timeline Builder
============================================================

Given MatchFinder results, builds a timeline that tells a story.

A good mashup has:
  - An arc: builds energy, peaks, resolves
  - Variety: different sections, not the same combo repeated
  - Transitions: smooth handoffs between sections
  - Surprise: unexpected moments that work
  - Duration: 2.5-4 minutes (not the full length of both songs)

The arranger picks from the best-scoring pairings and arranges
them into a coherent sequence.
"""

import numpy as np
from dataclasses import dataclass, field
from typing import List, Dict, Optional

from vocalfusion.song_dna import SongDNA, Section
from vocalfusion.match_finder import MatchResult, SectionPairing, MatchFinder


@dataclass
class TimelineBlock:
    """A block in the mashup timeline"""
    start_time: float = 0.0
    end_time: float = 0.0
    duration: float = 0.0

    # What plays in this block
    vocal_source: str = "none"      # "A", "B", "none"
    vocal_section_idx: int = -1     # Index into source song's sections
    vocal_start_in_song: float = 0  # Where in the source song to pull from
    vocal_end_in_song: float = 0

    beat_source: str = "A"          # "A", "B"
    beat_section_idx: int = -1
    beat_start_in_song: float = 0
    beat_end_in_song: float = 0

    # Character
    block_type: str = "verse"       # intro, verse, chorus, breakdown, drop, buildup, outro
    energy_target: float = 0.7      # 0-1

    # Transition info
    transition_in: str = "crossfade"   # crossfade, cut, buildup, drop
    transition_out: str = "crossfade"

    # Score from the pairing
    pairing_score: float = 0.0


@dataclass
class MashupTimeline:
    """Complete timeline for a mashup"""
    blocks: List[TimelineBlock] = field(default_factory=list)
    total_duration: float = 0.0
    direction: str = "a_vocals"
    tempo: float = 120.0
    key_shift: int = 0


class DJArranger:
    """Build an intelligent mashup timeline"""

    def __init__(self, sample_rate: int = 44100):
        self.sr = sample_rate

        # Target structure for a ~3 minute mashup
        # This is the energy arc template
        self.arc_template = [
            {'type': 'intro',     'energy': 0.4, 'bars': 8,  'vocal': False},
            {'type': 'verse',     'energy': 0.7, 'bars': 16, 'vocal': True},
            {'type': 'buildup',   'energy': 0.85,'bars': 8,  'vocal': True},
            {'type': 'chorus',    'energy': 1.0, 'bars': 16, 'vocal': True},
            {'type': 'breakdown', 'energy': 0.3, 'bars': 8,  'vocal': False},
            {'type': 'verse',     'energy': 0.7, 'bars': 16, 'vocal': True},
            {'type': 'chorus',    'energy': 1.0, 'bars': 16, 'vocal': True},
            {'type': 'outro',     'energy': 0.3, 'bars': 8,  'vocal': False},
        ]

    def arrange(self, dna_a: SongDNA, dna_b: SongDNA,
                match_result: MatchResult) -> MashupTimeline:
        """
        Build the mashup timeline.

        Strategy:
        1. Determine direction (A vocals or B vocals)
        2. Find the tempo (the beat song's tempo, untouched)
        3. For each slot in the arc template, find the best matching pairing
        4. Ensure variety (don't reuse sections too much)
        5. Calculate exact timing based on bars + tempo
        """
        timeline = MashupTimeline()
        timeline.direction = match_result.best_direction

        if timeline.direction == 'a_vocals':
            vox_dna = dna_a
            beat_dna = dna_b
            vox_label = 'A'
            beat_label = 'B'
        else:
            vox_dna = dna_b
            beat_dna = dna_a
            vox_label = 'B'
            beat_label = 'A'

        timeline.tempo = beat_dna.beat_grid.tempo
        bar_duration = beat_dna.beat_grid.bar_duration()

        print(f"    Direction: {vox_label} vocals + {beat_label} beat")
        print(f"    Beat tempo: {timeline.tempo:.0f} BPM, "
              f"bar = {bar_duration:.2f}s")

        # Get relevant pairings (filtered by direction)
        matcher = MatchFinder(self.sr)
        top_pairings = matcher.get_top_pairings(
            match_result, direction=timeline.direction, n=20)

        if not top_pairings:
            print("    No pairings! Using default arrangement...")
            return self._default_timeline(dna_a, dna_b, timeline)

        # Build section pools
        vox_sections = self._categorize_sections(vox_dna.sections)
        beat_sections = self._categorize_sections(beat_dna.sections)

        # Fill each slot in the arc template
        current_time = 0.0
        used_vox_sections = set()
        used_beat_sections = set()

        for slot in self.arc_template:
            block = TimelineBlock()
            block.start_time = current_time
            block.block_type = slot['type']
            block.energy_target = slot['energy']
            block.duration = slot['bars'] * bar_duration
            block.end_time = current_time + block.duration

            # Find the best pairing for this slot
            pairing = self._find_best_for_slot(
                slot, top_pairings, vox_dna, beat_dna,
                vox_sections, beat_sections,
                used_vox_sections, used_beat_sections,
                timeline.direction)

            if pairing is not None:
                if timeline.direction == 'a_vocals':
                    vox_sec = pairing.section_a
                    beat_sec = pairing.section_b
                    vox_idx = pairing.section_a_idx
                    beat_idx = pairing.section_b_idx
                else:
                    vox_sec = pairing.section_b
                    beat_sec = pairing.section_a
                    vox_idx = pairing.section_b_idx
                    beat_idx = pairing.section_a_idx

                block.pairing_score = pairing.overall_score

                # Vocal source
                if slot['vocal']:
                    block.vocal_source = vox_label
                    block.vocal_section_idx = vox_idx
                    block.vocal_start_in_song = vox_sec.start_time
                    block.vocal_end_in_song = vox_sec.end_time
                    used_vox_sections.add(vox_idx)
                else:
                    block.vocal_source = "none"

                # Beat source (always from beat song)
                block.beat_source = beat_label
                block.beat_section_idx = beat_idx
                block.beat_start_in_song = beat_sec.start_time
                block.beat_end_in_song = beat_sec.end_time
                used_beat_sections.add(beat_idx)

            else:
                # Fallback: use first available section
                block.vocal_source = vox_label if slot['vocal'] else "none"
                block.beat_source = beat_label

            # Set transition types
            if slot['type'] == 'intro':
                block.transition_in = 'fade_in'
            elif slot['type'] == 'drop' or slot['type'] == 'chorus':
                block.transition_in = 'drop'
            elif slot['type'] == 'breakdown':
                block.transition_in = 'filter_sweep'
            elif slot['type'] == 'outro':
                block.transition_out = 'fade_out'
            else:
                block.transition_in = 'crossfade'

            timeline.blocks.append(block)
            current_time = block.end_time

        timeline.total_duration = current_time

        # Print timeline
        print(f"\n    Timeline ({timeline.total_duration:.0f}s):")
        for b in timeline.blocks:
            print(f"      {b.start_time:5.1f}-{b.end_time:5.1f}s "
                  f"| {b.block_type:12s} "
                  f"| vox={b.vocal_source:4s} "
                  f"| beat={b.beat_source:4s} "
                  f"| e={b.energy_target:.1f} "
                  f"| score={b.pairing_score:.2f}")

        return timeline

    def _find_best_for_slot(self, slot, pairings, vox_dna, beat_dna,
                             vox_sections, beat_sections,
                             used_vox, used_beat, direction):
        """Find the best pairing for a timeline slot"""
        target_type = slot['type']
        target_energy = slot['energy']
        needs_vocal = slot['vocal']

        best_pairing = None
        best_fit = -1

        for p in pairings:
            if direction == 'a_vocals':
                vox_sec = p.section_a
                beat_sec = p.section_b
                vox_idx = p.section_a_idx
                beat_idx = p.section_b_idx
            else:
                vox_sec = p.section_b
                beat_sec = p.section_a
                vox_idx = p.section_b_idx
                beat_idx = p.section_a_idx

            # Skip if we need vocals but the vocal section has none
            if needs_vocal and not vox_sec.has_vocals:
                continue

            # Prefer unused sections (variety)
            novelty_bonus = 0
            if vox_idx not in used_vox:
                novelty_bonus += 0.15
            if beat_idx not in used_beat:
                novelty_bonus += 0.10

            # Energy match
            energy_match = 1.0 - abs(target_energy - beat_sec.energy)

            # Section type match
            type_match = self._type_match_score(
                target_type, vox_sec.classification, beat_sec.classification)

            # Combined fitness
            fitness = (
                p.overall_score * 0.35 +
                energy_match * 0.25 +
                type_match * 0.25 +
                novelty_bonus * 0.15
            )

            if fitness > best_fit:
                best_fit = fitness
                best_pairing = p

        return best_pairing

    def _type_match_score(self, target, vox_type, beat_type):
        """How well do these section types match the target slot?"""
        score = 0.5  # baseline

        # Perfect matches
        if target == 'chorus' and vox_type == 'chorus':
            score += 0.4
        if target == 'chorus' and beat_type in ('drop', 'chorus'):
            score += 0.3
        if target == 'verse' and vox_type == 'verse':
            score += 0.4
        if target == 'intro' and beat_type in ('intro', 'verse'):
            score += 0.3
        if target == 'breakdown' and beat_type in ('breakdown', 'intro'):
            score += 0.3
        if target == 'outro' and beat_type in ('outro', 'breakdown'):
            score += 0.3
        if target == 'buildup' and beat_type in ('buildup', 'verse'):
            score += 0.3

        return min(1.0, score)

    def _categorize_sections(self, sections):
        """Group sections by type for quick lookup"""
        categories = {}
        for i, sec in enumerate(sections):
            t = sec.classification
            if t not in categories:
                categories[t] = []
            categories[t].append((i, sec))
        return categories

    def _default_timeline(self, dna_a, dna_b, timeline):
        """Fallback timeline if no good pairings found"""
        bar_dur = 60.0 / timeline.tempo * 4
        blocks = []
        t = 0.0

        for btype, bars, vox in [
            ('intro', 8, 'none'),
            ('verse', 16, 'A'),
            ('chorus', 16, 'A'),
            ('verse', 16, 'A'),
            ('chorus', 16, 'A'),
            ('outro', 8, 'none'),
        ]:
            dur = bars * bar_dur
            blocks.append(TimelineBlock(
                start_time=t,
                end_time=t + dur,
                duration=dur,
                vocal_source=vox,
                beat_source='B',
                block_type=btype,
                energy_target={'intro': 0.4, 'verse': 0.7,
                              'chorus': 1.0, 'outro': 0.3}.get(btype, 0.7),
            ))
            t += dur

        timeline.blocks = blocks
        timeline.total_duration = t
        return timeline
