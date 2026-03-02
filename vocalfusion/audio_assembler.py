"""
VocalFusion AI — AudioAssembler: Precision Audio Engine
=========================================================

Executes a MashupTimeline by assembling audio from stems.

Key principles:
  1. NEVER stretch drums/bass more than 5%
  2. Only stretch vocals (they handle it much better)
  3. Cut on beat boundaries (no arbitrary cuts)
  4. Proper crossfades at transitions
  5. Energy management per section
  6. Minimal processing — don't destroy the stems
"""

import numpy as np
import librosa
from typing import Dict, Optional, Tuple

from vocalfusion.dsp import EnhancedDSP
from vocalfusion.song_dna import SongDNA
from vocalfusion.dj_arranger import MashupTimeline, TimelineBlock


class AudioAssembler:
    """Assemble a mashup from a timeline and stems"""

    def __init__(self, sample_rate: int = 44100):
        self.sr = sample_rate
        self.dsp = EnhancedDSP(sample_rate)

    def assemble(self, timeline: MashupTimeline,
                  stems_a: Dict[str, np.ndarray],
                  stems_b: Dict[str, np.ndarray],
                  dna_a: SongDNA, dna_b: SongDNA) -> Dict[str, np.ndarray]:
        """
        Assemble the mashup audio from a timeline.

        Steps:
        1. Prepare stems (mono, tempo-match vocals only)
        2. Find best key shift
        3. For each block in timeline, extract the right audio
        4. Apply transitions between blocks
        5. Mix with proper gain staging
        """
        print(f"\n    Assembling {timeline.total_duration:.0f}s mashup...")

        # Mono all stems
        stems_a = {k: self._mono(v) for k, v in stems_a.items()}
        stems_b = {k: self._mono(v) for k, v in stems_b.items()}

        # Determine which song provides beats vs vocals
        if timeline.direction == 'a_vocals':
            vox_stems = stems_a
            beat_stems = stems_b
            vox_dna = dna_a
            beat_dna = dna_b
            vox_tempo = dna_a.beat_grid.tempo
            beat_tempo = dna_b.beat_grid.tempo
        else:
            vox_stems = stems_b
            beat_stems = stems_a
            vox_dna = dna_b
            beat_dna = dna_a
            vox_tempo = dna_b.beat_grid.tempo
            beat_tempo = dna_a.beat_grid.tempo

        # Step 1: Stretch ONLY vocals to match beat tempo
        stretch_ratio = beat_tempo / vox_tempo
        # Check half/double time
        candidates = [stretch_ratio, stretch_ratio * 2, stretch_ratio / 2]
        stretch_ratio = min(candidates, key=lambda r: abs(r - 1.0))

        vox_audio = vox_stems.get('vocals')
        if vox_audio is not None and abs(stretch_ratio - 1.0) > 0.02:
            print(f"    Stretching vocals: {stretch_ratio:.3f}x "
                  f"({vox_tempo:.0f}→{beat_tempo:.0f} BPM)")
            vox_audio = librosa.effects.time_stretch(vox_audio, rate=stretch_ratio)
        elif vox_audio is not None:
            print(f"    Vocals: no stretch needed")

        # Step 2: Find best key shift
        best_shift = self._find_best_key_shift(
            vox_audio, self._make_inst(beat_stems))
        if vox_audio is not None and best_shift != 0:
            print(f"    Key shifting vocals: {best_shift:+d} semitones")
            vox_audio = librosa.effects.pitch_shift(
                vox_audio, sr=self.sr, n_steps=best_shift)
        timeline.key_shift = best_shift

        # Build the beat instrumental (untouched)
        beat_drums = beat_stems.get('drums')
        beat_bass = beat_stems.get('bass')
        beat_other = beat_stems.get('other')
        beat_inst = self._make_inst(beat_stems)

        # Step 3: Assemble each block
        total_samples = int(timeline.total_duration * self.sr)
        out_vocals = np.zeros(total_samples)
        out_instrumental = np.zeros(total_samples)

        for i, block in enumerate(timeline.blocks):
            start = int(block.start_time * self.sr)
            end = min(int(block.end_time * self.sr), total_samples)
            block_len = end - start
            if block_len <= 0:
                continue

            # --- Get vocal audio for this block ---
            if block.vocal_source != 'none' and vox_audio is not None:
                # Extract from the matched section of the vocal song
                vox_start_samp = int(block.vocal_start_in_song * self.sr * stretch_ratio)
                vox_end_samp = vox_start_samp + block_len
                vox_block = self._safe_extract(vox_audio, vox_start_samp, vox_end_samp)

                # If the section is shorter than the block, loop it
                if len(vox_block) < block_len:
                    repeats = (block_len // max(len(vox_block), 1)) + 1
                    vox_block = np.tile(vox_block, repeats)[:block_len]
            else:
                vox_block = np.zeros(block_len)

            # --- Get beat audio for this block ---
            beat_start_samp = int(block.beat_start_in_song * self.sr)
            beat_end_samp = beat_start_samp + block_len
            inst_block = self._safe_extract(beat_inst, beat_start_samp, beat_end_samp)

            # If section shorter than block, loop it
            if len(inst_block) < block_len:
                repeats = (block_len // max(len(inst_block), 1)) + 1
                inst_block = np.tile(inst_block, repeats)[:block_len]

            # --- Apply energy scaling ---
            inst_block *= block.energy_target

            # --- Apply transitions ---
            vox_block, inst_block = self._apply_transition(
                vox_block, inst_block, block, i,
                len(timeline.blocks), block_len)

            # --- Section-specific treatment ---
            if block.block_type == 'breakdown':
                # During breakdowns, strip the drums, keep melodic
                beat_other_sec = self._safe_extract(
                    beat_other, beat_start_samp, beat_end_samp) if beat_other is not None else np.zeros(block_len)
                beat_bass_sec = self._safe_extract(
                    beat_bass, beat_start_samp, beat_end_samp) if beat_bass is not None else np.zeros(block_len)
                if len(beat_other_sec) < block_len:
                    beat_other_sec = np.pad(beat_other_sec, (0, block_len - len(beat_other_sec)))
                if len(beat_bass_sec) < block_len:
                    beat_bass_sec = np.pad(beat_bass_sec, (0, block_len - len(beat_bass_sec)))
                inst_block = beat_other_sec * 0.6 + beat_bass_sec * 0.3
                inst_block *= block.energy_target

            elif block.block_type == 'intro' and i == 0:
                # First block: start with just drums, gradually bring in more
                if beat_drums is not None:
                    drums_sec = self._safe_extract(
                        beat_drums, beat_start_samp, beat_end_samp)
                    if len(drums_sec) < block_len:
                        drums_sec = np.pad(drums_sec, (0, block_len - len(drums_sec)))
                    # First half: drums only. Second half: blend in full instrumental
                    half = block_len // 2
                    inst_block[:half] = drums_sec[:half] * 0.7
                    blend = np.linspace(0, 1, block_len - half)
                    inst_block[half:] = (drums_sec[half:block_len] * 0.7 * (1 - blend) +
                                          inst_block[half:] * blend)

            # --- Write to output ---
            actual = min(block_len, len(vox_block), len(inst_block))
            out_vocals[start:start+actual] += vox_block[:actual]
            out_instrumental[start:start+actual] += inst_block[:actual]

        # Step 4: Mix with proper levels
        print(f"    Mixing...")
        out_vocals = self._set_rms(out_vocals, 0.10)
        out_instrumental = self._set_rms(out_instrumental, 0.06)

        # HPF on vocals (just remove rumble)
        if np.any(out_vocals != 0):
            out_vocals = self.dsp.highpass(out_vocals, 80)

        # Gentle sidechain
        if np.any(out_vocals != 0):
            out_instrumental = self.dsp.sidechain_duck(
                out_instrumental, out_vocals,
                threshold_db=-25, ratio=2.0,
                attack_ms=10, release_ms=200, amount=0.25)

        full_mix = out_vocals + out_instrumental

        # Light master
        full_mix = self._light_master(full_mix)

        dur = len(full_mix) / self.sr
        print(f"    Output: {dur:.1f}s")

        return {
            'vocals': out_vocals,
            'drums': beat_drums,
            'bass': beat_bass,
            'other': beat_other,
            'full_mix': full_mix,
        }

    # ================================================================
    # TRANSITIONS
    # ================================================================

    def _apply_transition(self, vox, inst, block, block_idx,
                           n_blocks, block_len):
        """Apply transition effects at block boundaries"""
        xfade = min(int(0.5 * self.sr), block_len // 4)  # 500ms or 25% of block

        # Fade in
        if block.transition_in == 'fade_in' or block_idx == 0:
            fade_len = min(int(3 * self.sr), block_len // 2)
            fade = np.linspace(0, 1, fade_len)
            vox[:fade_len] *= fade
            inst[:fade_len] *= fade

        elif block.transition_in == 'drop':
            # Brief silence then slam in
            silence = min(int(0.1 * self.sr), block_len // 8)
            if silence > 0:
                # Short dip before the drop
                vox[:silence] *= np.linspace(0.3, 1.0, silence)
                inst[:silence] *= np.linspace(0.3, 1.0, silence)

        elif block.transition_in == 'crossfade':
            if xfade > 0:
                vox[:xfade] *= np.linspace(0, 1, xfade)
                inst[:xfade] *= np.linspace(0, 1, xfade)

        elif block.transition_in == 'filter_sweep':
            # Low-pass filter sweep opening up
            sweep_len = min(int(2 * self.sr), block_len // 2)
            if sweep_len > 0:
                sweep = np.linspace(0.2, 1.0, sweep_len)
                inst[:sweep_len] *= sweep

        # Fade out
        if block.transition_out == 'fade_out' or block_idx == n_blocks - 1:
            fade_len = min(int(4 * self.sr), block_len // 2)
            fade = np.linspace(1, 0, fade_len)
            vox[-fade_len:] *= fade
            inst[-fade_len:] *= fade

        elif block_idx < n_blocks - 1:
            # Standard crossfade out
            if xfade > 0:
                vox[-xfade:] *= np.linspace(1, 0, xfade)
                inst[-xfade:] *= np.linspace(1, 0, xfade)

        return vox, inst

    # ================================================================
    # KEY MATCHING
    # ================================================================

    def _find_best_key_shift(self, vocals, instrumental):
        """Test all 12 shifts, pick best by chroma correlation"""
        if vocals is None or instrumental is None:
            return 0

        test_len = min(10 * self.sr, len(vocals), len(instrumental))
        v = vocals[:test_len]
        inst = instrumental[:test_len]

        inst_chroma = np.mean(
            librosa.feature.chroma_cqt(y=inst, sr=self.sr), axis=1)

        best_shift = 0
        best_corr = -2

        for shift in range(-6, 6):
            if shift == 0:
                sig = v
            else:
                sig = librosa.effects.pitch_shift(v, sr=self.sr, n_steps=shift)

            sig_chroma = np.mean(
                librosa.feature.chroma_cqt(y=sig, sr=self.sr), axis=1)
            corr = float(np.corrcoef(sig_chroma, inst_chroma)[0, 1])

            if corr > best_corr:
                best_corr = corr
                best_shift = shift

        return best_shift

    # ================================================================
    # UTILITIES
    # ================================================================

    def _mono(self, audio):
        if audio is None:
            return None
        return np.mean(audio, axis=0) if audio.ndim > 1 else audio

    def _safe_extract(self, audio, start, end):
        length = end - start
        if audio is None:
            return np.zeros(length)
        if start >= len(audio):
            return np.zeros(length)
        seg = audio[max(0, start):min(end, len(audio))]
        if len(seg) < length:
            return np.pad(seg, (0, length - len(seg)))
        return seg.copy()

    def _make_inst(self, stems):
        parts = []
        for k in ['drums', 'bass', 'other']:
            v = stems.get(k)
            if v is not None:
                v = self._mono(v)
                parts.append(v)
        if not parts:
            return None
        mx = max(len(p) for p in parts)
        out = np.zeros(mx)
        for p in parts:
            out[:len(p)] += p
        return out

    def _set_rms(self, audio, target):
        if audio is None or np.all(audio == 0):
            return audio if audio is not None else np.zeros(self.sr)
        rms = np.sqrt(np.mean(audio ** 2))
        if rms < 1e-8:
            return audio
        return audio * min(target / rms, 8.0)

    def _light_master(self, audio):
        rms = np.sqrt(np.mean(audio ** 2))
        if rms < 1e-8:
            return audio
        target = 10 ** (-14 / 20)
        audio = audio * min(target / rms, 5.0)
        peak = np.max(np.abs(audio))
        if peak > 0.95:
            audio *= 0.95 / peak
        return audio
