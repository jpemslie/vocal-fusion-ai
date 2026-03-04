"""
VocalFusion AI — VocalQualityFilter
=====================================

Demucs leaks non-vocal content into the vocal stem: drum bleed, cymbal
reverb, low-level room noise. Without filtering, the AI DJ places these
junk segments as "vocal phrases" — creating moments of just hi-hat bleed
or a single unintelligible syllable.

Scoring criteria (each 0–1, weighted sum):

  harmonic_ratio   (weight 0.5)
    Separates the phrase into harmonic + percussive components (HPSS).
    Real vocals are strongly harmonic. Drum bleed is strongly percussive.
    Score = harmonic_rms / (harmonic_rms + percussive_rms)

  pitch_coherence  (weight 0.3)
    Runs librosa pitch tracking. Real vocals have trackable, stable pitch
    across most frames. Noise/bleed has mostly zero or chaotic pitch.
    Score = fraction of frames with confident pitch detection

  duration_score   (weight 0.2)
    Penalises very short phrases. A 0.5s phrase is almost certainly a
    breath or single word with no musical context. ≥2s = full score.

Keeps the top keep_ratio of phrases (default 0.70 = drop bottom 30%).
"""

import numpy as np
import librosa


class VocalQualityFilter:

    def __init__(self, sample_rate: int = 44100, keep_ratio: float = 0.70):
        self.sr         = sample_rate
        self.keep_ratio = keep_ratio

    # ------------------------------------------------------------------ #
    # Public API
    # ------------------------------------------------------------------ #

    def filter_phrases(self, phrases: list, vocals: np.ndarray) -> list:
        """
        Score every phrase and return the top keep_ratio fraction.
        Preserves the original energy-descending sort order on the kept phrases.

        Parameters
        ----------
        phrases : list of dicts with start, end, duration, energy (sample offsets)
        vocals  : full vocal stem array (used to extract per-phrase audio)

        Returns
        -------
        Filtered list (may be shorter than input). If nothing can be scored,
        returns phrases unchanged.
        """
        if not phrases or vocals is None or len(vocals) == 0:
            return phrases

        scored = []
        for ph in phrases:
            s, e = int(ph['start']), int(ph['end'])
            if e <= s or s >= len(vocals):
                continue
            audio = vocals[s: min(e, len(vocals))]
            if len(audio) < int(0.25 * self.sr):
                continue
            score = self._score(audio, ph['duration'])
            scored.append((score, ph))

        if not scored:
            return phrases

        scored.sort(key=lambda x: x[0], reverse=True)
        keep_n   = max(1, int(len(scored) * self.keep_ratio))
        kept     = [ph for _, ph in scored[:keep_n]]
        n_dropped = len(scored) - keep_n

        if n_dropped > 0:
            min_score = scored[keep_n - 1][0]
            print(f"    VocalQualityFilter: kept {keep_n}/{len(scored)} phrases "
                  f"(dropped {n_dropped} below score {min_score:.2f})")

        # Restore energy-descending order for downstream phrase placement
        return sorted(kept, key=lambda p: p['energy'], reverse=True)

    # ------------------------------------------------------------------ #
    # Scoring
    # ------------------------------------------------------------------ #

    def _score(self, audio: np.ndarray, duration_s: float) -> float:
        audio = audio.astype(np.float32)

        harmonic_score  = self._harmonic_ratio(audio)
        pitch_score     = self._pitch_coherence(audio)
        duration_score  = self._duration_score(duration_s)

        return (0.5 * harmonic_score +
                0.3 * pitch_score    +
                0.2 * duration_score)

    def _harmonic_ratio(self, audio: np.ndarray) -> float:
        """Fraction of energy that is harmonic vs percussive."""
        try:
            harmonic, percussive = librosa.effects.hpss(audio, margin=3.0)
            h = float(np.sqrt(np.mean(harmonic    ** 2)))
            p = float(np.sqrt(np.mean(percussive  ** 2)))
            return h / (h + p + 1e-8)
        except Exception:
            return 0.5

    def _pitch_coherence(self, audio: np.ndarray) -> float:
        """Fraction of frames where a confident pitch is detected."""
        try:
            _, magnitudes = librosa.piptrack(
                y=audio, sr=self.sr, threshold=0.1, hop_length=512)
            max_mags  = magnitudes.max(axis=0)
            # Frame is "confident" if its magnitude is above the median
            threshold = float(np.percentile(max_mags, 50))
            confident = float(np.mean(max_mags > threshold))
            return min(1.0, confident * 1.4)
        except Exception:
            return 0.5

    def _duration_score(self, duration_s: float) -> float:
        """Reward for duration: full at ≥2s, linear from 0 at 0.5s to 0.6 at 1s."""
        if duration_s >= 2.0:
            return 1.0
        if duration_s >= 1.0:
            return 0.6 + 0.4 * (duration_s - 1.0)
        if duration_s >= 0.5:
            return 0.6 * (duration_s - 0.5) / 0.5
        return 0.0
