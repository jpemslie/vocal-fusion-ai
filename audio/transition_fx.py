"""
VocalFusion AI — TransitionFX
==============================

Adds professional DJ transitions to vocal phrases:
  1. Reverb tail  — last word decays naturally into the beat instead of
                    cutting off abruptly (single biggest "amateur vs pro" tell)
  2. BPM-synced delay — quarter-note echo repeats with exponential decay,
                        making vocals feel locked to the groove

Both effects are added (not replacing) to the out_vocals array after the
phrase end position, so they never interfere with the next phrase.
"""

import numpy as np
from scipy.signal import fftconvolve


class TransitionFX:

    def __init__(self, sample_rate: int = 44100):
        self.sr = sample_rate

    # ------------------------------------------------------------------ #
    # Public API
    # ------------------------------------------------------------------ #

    def apply_phrase_tails(
        self,
        out_vocals: np.ndarray,
        placements: list,
        beat_bpm: float,
        reverb_dur_s: float = 0.7,
        delay_repeats: int = 2,
    ) -> np.ndarray:
        """
        For every placed phrase, write a reverb + delay tail into out_vocals
        starting right after the phrase ends.

        Parameters
        ----------
        out_vocals    : full-length vocal track array (modified in-place)
        placements    : list of dicts with vocal_start, vocal_end, dest_start
        beat_bpm      : beats-per-minute of the beat song (for delay timing)
        reverb_dur_s  : how long the reverb tail lasts (seconds)
        delay_repeats : number of quarter-note echo repeats
        """
        total = len(out_vocals)
        beat_s = 60.0 / max(beat_bpm, 60.0)   # quarter-note duration in seconds

        n_tailed = 0
        for pl in placements:
            dest_start = int(pl['dest_start'])
            src_len    = int(pl['vocal_end']) - int(pl['vocal_start'])
            dest_end   = min(dest_start + src_len, total)
            if dest_end <= dest_start:
                continue

            # Grab the last 300ms of the written phrase as the reverb source
            window_n  = min(int(0.30 * self.sr), dest_end - dest_start)
            tail_src  = out_vocals[dest_end - window_n: dest_end].copy()

            if np.max(np.abs(tail_src)) < 1e-7:
                continue   # phrase was silent — nothing to tail

            rev  = self._reverb_tail(tail_src, reverb_dur_s)
            self._add_at(out_vocals, rev, dest_end, total)

            delay = self._delay_tail(tail_src, beat_s, delay_repeats)
            self._add_at(out_vocals, delay, dest_end, total)

            n_tailed += 1

        if n_tailed:
            print(f"    TransitionFX: reverb+delay tails on {n_tailed} phrases")

        return out_vocals

    # ------------------------------------------------------------------ #
    # Internal helpers
    # ------------------------------------------------------------------ #

    def _reverb_tail(self, src: np.ndarray, dur_s: float) -> np.ndarray:
        """
        Convolve src with a synthetic exponential-decay IR.
        Returns ONLY the tail portion (samples after len(src)).
        """
        tail_n = int(dur_s * self.sr)
        if tail_n <= 0 or len(src) == 0:
            return np.zeros(0)

        # Build a simple room IR: exponential decay + three early reflections
        t  = np.linspace(0.0, dur_s, tail_n)
        ir = np.exp(-5.0 * t)                  # main decay
        for delay_ms, amplitude in [(12, 0.45), (27, 0.30), (55, 0.18)]:
            n = int(delay_ms * self.sr / 1000)
            if n < tail_n:
                ir[n] += amplitude * np.exp(-5.0 * t[n])

        ir /= np.max(np.abs(ir)) + 1e-8

        full = fftconvolve(src.astype(np.float64), ir)
        tail = full[len(src):]                 # only the decay after phrase ends

        # Scale tail to ~25% of the phrase RMS
        src_rms  = float(np.sqrt(np.mean(src  ** 2))) + 1e-8
        tail_rms = float(np.sqrt(np.mean(tail ** 2))) + 1e-8
        tail     = tail * (src_rms * 0.25 / tail_rms)

        # Fade to zero
        tail *= np.linspace(1.0, 0.0, len(tail))
        return tail.astype(np.float64)

    def _delay_tail(self, src: np.ndarray, beat_s: float,
                    n_repeats: int = 2) -> np.ndarray:
        """
        Quarter-note ping-pong delay: echoes at beat intervals with -9dB decay.
        """
        beat_n = int(beat_s * self.sr)
        if beat_n <= 0 or len(src) == 0:
            return np.zeros(0)

        total_n = beat_n * n_repeats + len(src)
        tail    = np.zeros(total_n, dtype=np.float64)

        src_rms = float(np.sqrt(np.mean(src ** 2))) + 1e-8

        for i in range(1, n_repeats + 1):
            gain   = 0.30 * (0.5 ** (i - 1))   # −10 dB per repeat
            offset = i * beat_n
            end    = min(offset + len(src), total_n)
            tail[offset:end] += src[: end - offset] * gain

        # Scale to ~20% of phrase RMS
        active  = tail[tail != 0]
        if len(active) > 0:
            tail_rms = float(np.sqrt(np.mean(active ** 2))) + 1e-8
            tail     = tail * (src_rms * 0.20 / tail_rms)

        return tail

    def _add_at(self, target: np.ndarray, chunk: np.ndarray,
                offset: int, total: int) -> None:
        """Add chunk into target starting at offset, clipping to total length."""
        if len(chunk) == 0 or offset >= total:
            return
        end = min(offset + len(chunk), total)
        target[offset:end] += chunk[: end - offset]
