"""
VocalFusion AI — EnergyAutomation
=====================================

Gives the mix a professional energy arc — the mix now builds tension,
hits hard on drops, breathes in breakdowns, and releases into grooves.
Without this, every section of the mix sounds equally loud and flat.

Two operations:

  1. Vocal level automation by bar label
     ─────────────────────────────────────
     drop      → ×1.25  (hook vocals need to hit hard)
     groove    → ×1.00  (neutral reference)
     build     → ×0.80  (restrained — tension is about withholding)
     breakdown → ×0.00  (beat-only moment — silence IS the transition)

     Transitions between bars are smoothed over half a bar to avoid clicks.

  2. Filter sweep into drops ("filter opens on the drop")
     ───────────────────────────────────────────────────────
     Real DJs close a low-pass filter in the 2 bars before a drop, then
     open it fully when the drop hits. This creates the "tension → release"
     that makes drops feel physically satisfying.

     Implementation: LP cutoff sweeps 300 Hz → fully open across the
     2 bars pre-drop, applied to all three beat stems (drums, bass, other).
     The sweep is divided into 16 segments; cutoff is fixed per segment
     to avoid phase discontinuities.
"""

import numpy as np
from scipy.signal import butter, sosfilt


# Per-section vocal gain multipliers
SECTION_GAIN = {
    'drop':      1.25,
    'groove':    1.00,
    'build':     0.80,
    'breakdown': 0.00,
}

# How many bars to smooth gain transitions over (prevents clicks)
_SMOOTH_BARS = 0.5

# Number of filter-cutoff steps in the pre-drop sweep
_SWEEP_SEGMENTS = 16


class EnergyAutomation:

    def __init__(self, sample_rate: int = 44100):
        self.sr = sample_rate

    # ------------------------------------------------------------------ #
    # Public API
    # ------------------------------------------------------------------ #

    def apply_vocal_automation(
        self,
        out_vocals: np.ndarray,
        bar_labels: list,
        bar_samples: int,
    ) -> np.ndarray:
        """
        Multiply out_vocals by a per-bar gain envelope.

        Parameters
        ----------
        out_vocals  : vocal track (already level-set by AIDJ)
        bar_labels  : list of section label strings (one per bar)
        bar_samples : number of samples per bar

        Returns modified out_vocals.
        """
        total = len(out_vocals)
        if not bar_labels or bar_samples <= 0 or total == 0:
            return out_vocals

        # Build raw step envelope
        envelope = np.ones(total, dtype=np.float64)
        for i, lbl in enumerate(bar_labels):
            gain  = SECTION_GAIN.get(lbl, 1.00)
            start = i * bar_samples
            end   = min(start + bar_samples, total)
            if start >= total:
                break
            envelope[start:end] = gain

        # Smooth to avoid clicks at bar boundaries
        smooth_n = max(1, int(_SMOOTH_BARS * bar_samples))
        envelope = self._smooth(envelope, smooth_n)

        # Report section distribution
        drop_bars  = sum(1 for l in bar_labels if l == 'drop')
        build_bars = sum(1 for l in bar_labels if l == 'build')
        bd_bars    = sum(1 for l in bar_labels if l == 'breakdown')
        print(f"    EnergyAutomation: vocal gains — "
              f"{drop_bars} drop bars ×1.25, "
              f"{build_bars} build bars ×0.80, "
              f"{bd_bars} breakdown bars ×0.00")

        return out_vocals * envelope

    def apply_drop_filter_sweep(
        self,
        out_drums: np.ndarray,
        out_bass:  np.ndarray,
        out_other: np.ndarray,
        bar_labels: list,
        bar_samples: int,
    ) -> tuple:
        """
        Apply a low-pass filter sweep in the 2 bars leading into each drop.
        Cutoff climbs from 300 Hz → fully open, so the beat "opens up" on
        the drop — a physical release of tension.

        Returns (out_drums, out_bass, out_other) modified in-place.
        """
        if not bar_labels or bar_samples <= 0:
            return out_drums, out_bass, out_other

        total = min(len(out_drums), len(out_bass), len(out_other))
        sweep_bars = 2

        # Find the start bar of each drop section
        drop_starts = [
            i for i in range(1, len(bar_labels))
            if bar_labels[i] == 'drop' and bar_labels[i - 1] != 'drop'
        ]

        if not drop_starts:
            return out_drums, out_bass, out_other

        print(f"    EnergyAutomation: filter sweep into "
              f"{len(drop_starts)} drop(s)")

        nyq = self.sr / 2.0

        for drop_bar in drop_starts:
            sweep_start_bar = max(0, drop_bar - sweep_bars)
            start_s = sweep_start_bar * bar_samples
            end_s   = min(drop_bar * bar_samples, total)
            if start_s >= total or end_s <= start_s:
                continue

            seg_n = (end_s - start_s) // _SWEEP_SEGMENTS

            for seg in range(_SWEEP_SEGMENTS):
                seg_s = start_s + seg * seg_n
                seg_e = min(seg_s + seg_n, end_s)
                if seg_s >= total or seg_e <= seg_s:
                    break

                # Progress 0 → 1 across the sweep; cutoff 300 Hz → Nyquist
                t      = seg / _SWEEP_SEGMENTS
                cutoff = 300.0 + t * (nyq * 0.90 - 300.0)
                cutoff = min(cutoff, nyq * 0.90)

                try:
                    sos = butter(2, cutoff / nyq, btype='low', output='sos')
                    for arr in (out_drums, out_bass, out_other):
                        chunk = arr[seg_s:seg_e].astype(np.float64)
                        arr[seg_s:seg_e] = sosfilt(sos, chunk)
                except Exception:
                    pass   # never crash the mix over an effect

        return out_drums, out_bass, out_other

    # ------------------------------------------------------------------ #
    # Internal helpers
    # ------------------------------------------------------------------ #

    def _smooth(self, env: np.ndarray, n: int) -> np.ndarray:
        """Simple uniform moving-average to smooth gain step transitions."""
        if n <= 1:
            return env
        kernel = np.ones(n) / n
        return np.convolve(env, kernel, mode='same')
