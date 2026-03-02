"""
VocalFusion AI — MixRefiner: Iterative Quality Improvement
=============================================================

The feedback loop. Uses MixIntelligence as ears:
  1. Score the current mix
  2. Find the worst dimensions
  3. Apply targeted, gentle fixes for each
  4. Re-score to verify improvement
  5. If improved, keep. If not, revert.
  6. Repeat until quality plateaus or max iterations hit.

Fixes are GENTLE — small EQ moves, light compression.
The goal is polish, not reconstruction.
"""

import numpy as np
from typing import Dict, Optional, Tuple, List

from vocalfusion.dsp import EnhancedDSP
from vocalfusion.mix_intelligence import MixIntelligence, MixScore


class MixRefiner:
    """Iteratively improve mix quality using AI scoring"""

    def __init__(self, sample_rate: int = 44100):
        self.sr = sample_rate
        self.dsp = EnhancedDSP(sample_rate)
        self.ai = MixIntelligence(sample_rate)

    def refine(self, full_mix: np.ndarray,
               vocals: Optional[np.ndarray] = None,
               max_iterations: int = 3,
               min_improvement: float = 0.005) -> Tuple[np.ndarray, MixScore]:
        """
        Iteratively refine a mix.

        Returns: (refined_audio, final_score)
        """
        current = full_mix.copy()
        best_audio = current.copy()
        best_score = self.ai.evaluate_mix(current, vocals=vocals)
        best_overall = best_score.overall

        print(f"    Initial: {best_overall:.3f}")
        self._print_score(best_score)

        for iteration in range(max_iterations):
            # Find worst dimensions that are below threshold
            worst = best_score.worst_dimensions(n=3)
            fixable = [(name, val) for name, val in worst if val < 0.65]

            if not fixable:
                print(f"    Iteration {iteration+1}: all dimensions above threshold")
                break

            # Try fixing each weak dimension
            candidate = current.copy()
            applied_fixes = []

            for dim_name, dim_val in fixable:
                fixed = self._apply_fix(candidate, vocals, dim_name, dim_val)

                # Check if fix actually helped
                test_score = self.ai.evaluate_mix(fixed, vocals=vocals)

                if test_score.overall > best_overall - 0.01:
                    # Fix didn't make things worse, keep it
                    candidate = fixed
                    applied_fixes.append(f"{dim_name}({dim_val:.2f}→{getattr(test_score, dim_name, dim_val):.2f})")
                else:
                    # Fix made things worse, skip
                    pass

            # Score the candidate with all fixes applied
            new_score = self.ai.evaluate_mix(candidate, vocals=vocals)
            improvement = new_score.overall - best_overall

            if improvement > min_improvement:
                current = candidate
                best_audio = candidate.copy()
                best_score = new_score
                best_overall = new_score.overall
                print(f"    Iteration {iteration+1}: {best_overall:.3f} "
                      f"(+{improvement:.3f}) | Fixed: {', '.join(applied_fixes)}")
            else:
                print(f"    Iteration {iteration+1}: no improvement "
                      f"({new_score.overall:.3f} vs {best_overall:.3f})")
                if improvement < -min_improvement:
                    # Got worse, revert
                    current = best_audio.copy()
                break

        print(f"    Final: {best_overall:.3f}")
        self._print_score(best_score)

        return best_audio, best_score

    def _apply_fix(self, audio, vocals, dim_name, dim_val):
        """Apply a gentle, targeted fix for a weak dimension"""
        fixed = audio.copy()

        if dim_name == 'vocal_clarity':
            # Vocals buried — boost presence, cut instrumental mud
            if vocals is not None and np.any(vocals != 0):
                # Boost the 2-5kHz presence range where vocals live
                fixed = self.dsp.parametric_eq(fixed, 3000, 1.5, 1.2)
                fixed = self.dsp.parametric_eq(fixed, 5000, 1.0, 1.0)
                # Cut competing low-mids
                fixed = self.dsp.parametric_eq(fixed, 400, -1.5, 0.8)

        elif dim_name == 'spectral_balance':
            # Probably muddy or thin
            # Check where the problem is
            low_energy = self._band_energy(fixed, 0, 250)
            mid_energy = self._band_energy(fixed, 250, 4000)
            high_energy = self._band_energy(fixed, 4000, 20000)

            total = low_energy + mid_energy + high_energy
            if total > 0:
                low_pct = low_energy / total
                mid_pct = mid_energy / total
                high_pct = high_energy / total

                if low_pct > 0.5:  # Too boomy
                    fixed = self.dsp.parametric_eq(fixed, 200, -2.0, 0.8)
                if mid_pct > 0.6:  # Too muddy
                    fixed = self.dsp.parametric_eq(fixed, 500, -1.5, 0.7)
                if high_pct < 0.15:  # Too dull
                    fixed = self.dsp.parametric_eq(fixed, 8000, 2.0, 0.7)

        elif dim_name == 'harmonic_clarity':
            # Clashing frequencies — surgical cuts
            fixed = self.dsp.parametric_eq(fixed, 300, -1.0, 1.5)
            fixed = self.dsp.parametric_eq(fixed, 600, -1.0, 1.5)

        elif dim_name == 'beat_coherence':
            # Rhythms not tight — boost transients
            fixed = self.dsp.parametric_eq(fixed, 80, 1.0, 1.0)
            fixed = self.dsp.parametric_eq(fixed, 5000, 1.0, 1.0)

        elif dim_name == 'energy_consistency':
            # Energy drops out — gentle level riding
            fixed = self._gentle_level_ride(fixed, target_window_s=2.0)

        elif dim_name == 'dynamic_range':
            # Either too squashed or too dynamic
            peak = np.max(np.abs(fixed))
            rms = np.sqrt(np.mean(fixed ** 2))
            crest = peak / max(rms, 1e-8)

            if crest > 10:  # Too dynamic
                # Gentle compression via soft clipping
                threshold = rms * 3
                fixed = np.where(
                    np.abs(fixed) > threshold,
                    np.sign(fixed) * (threshold + np.tanh(
                        (np.abs(fixed) - threshold) / threshold) * threshold),
                    fixed)

        elif dim_name == 'phase_coherence':
            # Phase issues — HPF to remove sub-bass phase problems
            fixed = self.dsp.highpass(fixed, 35)

        elif dim_name == 'spectral_separation':
            # Elements blending together — boost definition
            fixed = self.dsp.parametric_eq(fixed, 2500, 1.0, 1.0)
            fixed = self.dsp.parametric_eq(fixed, 10000, 1.5, 0.7)

        return fixed

    def _gentle_level_ride(self, audio, target_window_s=2.0):
        """Gently bring up quiet sections to smooth energy"""
        window = int(target_window_s * self.sr)
        n_windows = max(1, len(audio) // window)

        # Compute RMS per window
        rms = np.array([
            np.sqrt(np.mean(audio[i*window:(i+1)*window] ** 2))
            for i in range(n_windows)
        ])

        if len(rms) == 0 or np.max(rms) == 0:
            return audio

        target = np.median(rms)
        output = audio.copy()

        for i in range(n_windows):
            if rms[i] < target * 0.4 and rms[i] > 1e-8:
                # This window is too quiet
                boost = min(target / rms[i], 2.0)  # Max 6dB boost
                start = i * window
                end = min((i + 1) * window, len(output))

                # Apply with smooth ramp
                ramp_len = min(int(0.1 * self.sr), (end - start) // 4)
                gain = np.ones(end - start) * boost
                if ramp_len > 0:
                    gain[:ramp_len] = np.linspace(1.0, boost, ramp_len)
                    gain[-ramp_len:] = np.linspace(boost, 1.0, ramp_len)

                output[start:end] *= gain

        return output

    def _band_energy(self, audio, low_hz, high_hz):
        """Compute energy in a frequency band"""
        try:
            if low_hz <= 0:
                filtered = self.dsp.lowpass(audio, high_hz)
            elif high_hz >= 20000:
                filtered = self.dsp.highpass(audio, low_hz)
            else:
                filtered = self.dsp.bandpass(audio, low_hz, high_hz)
            return float(np.mean(filtered ** 2))
        except Exception:
            return 0.0

    def _print_score(self, score):
        """Print score breakdown"""
        print(f"      Beat={score.beat_coherence:.2f} "
              f"Spec={score.spectral_balance:.2f} "
              f"Harm={score.harmonic_clarity:.2f} "
              f"Vocal={score.vocal_clarity:.2f} "
              f"Dyn={score.dynamic_range:.2f} "
              f"Phase={score.phase_coherence:.2f} "
              f"Sep={score.spectral_separation:.2f} "
              f"Energy={score.energy_consistency:.2f}")
