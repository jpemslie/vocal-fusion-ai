"""
VocalFusion AI — HarmonicMixer
================================

Selects pitch shifts using music theory (circle of fifths / Camelot wheel)
instead of brute-force scoring. This avoids the circular dependency where
the key-clash penalty in the score biases against non-zero shifts even when
a shift would genuinely improve the mashup.

The Camelot wheel assigns each key a position. Adjacent positions are
harmonically compatible. This translates directly to semitone intervals.
"""

from typing import List


# Semitone name to pitch class
KEY_MAP = {
    'C': 0, 'C#': 1, 'Db': 1, 'D': 2, 'D#': 3, 'Eb': 3,
    'E': 4, 'F': 5, 'F#': 6, 'Gb': 6, 'G': 7, 'G#': 8,
    'Ab': 8, 'A': 9, 'A#': 10, 'Bb': 10, 'B': 11
}

# How well two tonics blend, given the interval (semitones) from
# the shifted vocal tonic to the beat tonic.
# Based on circle of fifths: each step = 5 or 7 semitones.
INTERVAL_SCORE = {
    0: 1.00,   # Unison — same key
    7: 0.95,   # Perfect 5th up — 1 step clockwise on circle of fifths
    5: 0.95,   # Perfect 4th up (= P5 down) — 1 step counter-clockwise
    2: 0.75,   # Major 2nd up — 2 steps clockwise
    10: 0.75,  # Minor 7th up (= M2 down) — 2 steps counter-clockwise
    9: 0.70,   # Major 6th up (= m3 down)
    4: 0.70,   # Major 3rd up
    3: 0.65,   # Minor 3rd — relative minor/major relationship
    8: 0.60,   # Minor 6th
    1: 0.15,   # Minor 2nd — semitone clash
    11: 0.15,  # Major 7th — leading-tone tension
    6: 0.10,   # Tritone — maximum dissonance
}

INTERVAL_NAMES = {
    0: 'same key',
    7: 'perfect 5th',
    5: 'perfect 4th',
    2: 'major 2nd',
    10: 'minor 7th',
    9: 'major 6th',
    4: 'major 3rd',
    3: 'minor 3rd (relative)',
    8: 'minor 6th',
    1: 'minor 2nd (clash)',
    11: 'major 7th (clash)',
    6: 'tritone (dissonant)',
}


class HarmonicMixer:
    """
    Determine musically ideal pitch shifts using circle of fifths logic.

    Given a vocal key and a beat key, returns candidate shifts ranked by
    how harmonically compatible they would make the mashup — without ever
    needing to actually render and score the audio.
    """

    def get_compatible_shifts(self, vocal_key: str, beat_key: str,
                               n: int = 5) -> List[int]:
        """
        Return the top-n semitone shifts for vocals to best match the beat.

        Args:
            vocal_key: e.g. "A minor", "C# major", "E minor"
            beat_key:  e.g. "C major", "G major"
            n:         how many candidate shifts to return (tested in order)

        Returns:
            List of shift values (semitones, -6 to +6) sorted best first.
            Shift=0 (no processing) is always included — it's never wrong
            to leave things alone if they're already compatible.
        """
        vocal_root = self._parse_key(vocal_key)
        beat_root = self._parse_key(beat_key)

        scored = []
        for shift in range(-6, 7):
            shifted_vocal = (vocal_root + shift) % 12
            interval = (beat_root - shifted_vocal) % 12
            compat = INTERVAL_SCORE.get(interval, 0.30)
            # Small penalty for larger shifts — prefer minimal pitch movement
            distance_penalty = abs(shift) * 0.025
            score = compat - distance_penalty
            scored.append((shift, score, interval))

        # Best compatibility first
        scored.sort(key=lambda x: x[1], reverse=True)

        top_shifts = [s for s, _, _ in scored[:n]]

        # Guarantee shift=0 is always tested (safe fallback)
        if 0 not in top_shifts:
            top_shifts[-1] = 0

        return top_shifts

    def best_shift(self, vocal_key: str, beat_key: str) -> int:
        """Return single best shift (for display/logging)"""
        return self.get_compatible_shifts(vocal_key, beat_key, n=1)[0]

    def describe_shift(self, vocal_key: str, beat_key: str, shift: int) -> str:
        """Human-readable description of why a shift was chosen"""
        vocal_root = self._parse_key(vocal_key)
        beat_root = self._parse_key(beat_key)
        shifted = (vocal_root + shift) % 12
        interval = (beat_root - shifted) % 12
        name = INTERVAL_NAMES.get(interval, 'unknown')
        score = INTERVAL_SCORE.get(interval, 0.30) - abs(shift) * 0.025
        return f"{shift:+d} st → {name} (score {score:.2f})"

    def _parse_key(self, key_str: str) -> int:
        """Parse a key string like 'A minor' or 'C# major' → pitch class 0-11.

        Minor keys are converted to their relative major for interval math,
        since the relative major shares the same pitch classes.
        """
        if not key_str:
            return 0
        key_str = key_str.strip()
        is_minor = ('minor' in key_str.lower()
                    or key_str.endswith('m')
                    or ' m ' in key_str.lower())
        parts = key_str.split()
        root_str = parts[0] if parts else 'C'
        root = KEY_MAP.get(root_str, 0)
        if is_minor:
            root = (root + 3) % 12  # relative major has the same Camelot position
        return root
