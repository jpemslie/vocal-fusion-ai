"""
VocalFusion AI — The AI DJ
=============================

PHILOSOPHY: Every decision is a scored experiment.
Nothing is hardcoded. The system tries things, listens,
and keeps what sounds best.

DECISIONS MADE BY SCORING (not by rules):
  1. Direction: A vocals + B beat, or B vocals + A beat?
  2. Tempo: stretch vocals, stretch beat, or split the difference?
  3. Key: which of 12 semitone shifts scores highest?
  4. Vocal level: what ratio of vocals to beat sounds best?
  5. Beat region: which part of the beat song works best under these vocals?
  6. Vocal timing: should vocals start at the same time as the beat, or offset?
  7. Arrangement: at each moment, do vocals ON or OFF score better?

The MixIntelligence module is the AI's ears.
Every test is a 10-15 second clip scored by all 8 quality dimensions.
"""

import numpy as np
import librosa
import pyrubberband as pyrb
from typing import Dict, Optional, Tuple, List
from dataclasses import dataclass, field

from vocalfusion.dsp import EnhancedDSP
from vocalfusion.mix_intelligence import MixIntelligence
from vocalfusion.song_dna import SongDNA
from vocalfusion.harmonic_mixer import HarmonicMixer
from vocalfusion.vocal_enhancer import VocalEnhancer


# Minimal timeline structures for compatibility with orchestrator
@dataclass
class StemLevels:
    drums: float = 1.0
    bass: float = 1.0
    other: float = 1.0
    vocals: float = 0.0

@dataclass
class TimelineBlock:
    start_time: float = 0.0
    end_time: float = 0.0
    duration: float = 0.0
    beat_start_in_song: float = 0.0
    beat_end_in_song: float = 0.0
    levels_start: StemLevels = field(default_factory=StemLevels)
    levels_end: StemLevels = field(default_factory=StemLevels)
    block_type: str = "verse"
    moment: str = ""

@dataclass
class MashupTimeline:
    blocks: List[TimelineBlock] = field(default_factory=list)
    total_duration: float = 0.0
    direction: str = "a_vocals"
    tempo: float = 120.0
    key_shift: int = 0
    beat_region_start: float = 0.0
    beat_region_end: float = 0.0
    vocal_region_start: float = 0.0
    vocal_region_end: float = 0.0


class AIDJ:
    """
    The AI DJ. Makes decisions by experimenting and scoring.
    """

    def __init__(self, sample_rate: int = 44100):
        self.sr = sample_rate
        self.dsp = EnhancedDSP(sample_rate)
        self.ears = MixIntelligence(sample_rate)
        self.harmonic = HarmonicMixer()
        self.enhancer = VocalEnhancer(sample_rate)
        self.test_duration = 12  # seconds per test clip

    def create_mashup(self, stems_a: Dict[str, np.ndarray],
                       stems_b: Dict[str, np.ndarray],
                       dna_a: SongDNA, dna_b: SongDNA) -> Dict:
        """
        Main entry point. Returns dict with full_mix + stems + quality_scores.
        """
        print(f"\n{'='*60}")
        print(f"  AI DJ — Experiment → Score → Decide")
        print(f"{'='*60}")

        # Mono everything upfront
        stems_a = {k: self._mono(v) for k, v in stems_a.items()}
        stems_b = {k: self._mono(v) for k, v in stems_b.items()}

        # ============================================================
        # DECISION 1: Which direction? (scored experiment)
        # ============================================================
        print(f"\n  DECISION 1: Direction")
        print(f"  {'─'*50}")

        score_a_vox, combo_a = self._test_direction(
            stems_a, stems_b, dna_a, dna_b, 'a_vocals')
        score_b_vox, combo_b = self._test_direction(
            stems_a, stems_b, dna_a, dna_b, 'b_vocals')

        print(f"    A vocals + B beat: {score_a_vox:.3f}")
        print(f"    B vocals + A beat: {score_b_vox:.3f}")

        if score_a_vox >= score_b_vox:
            direction = 'a_vocals'
            best_combo = combo_a
            print(f"    → Using A vocals + B beat")
        else:
            direction = 'b_vocals'
            best_combo = combo_b
            print(f"    → Using B vocals + A beat")

        vox_audio = best_combo['vocals']
        beat_drums = best_combo['drums']
        beat_bass = best_combo['bass']
        beat_other = best_combo['other']
        used_stretch = best_combo['stretch']
        used_shift = best_combo['key_shift']

        # ============================================================
        # DECISION 2: Vocal level (scored experiment)
        # ============================================================
        print(f"\n  DECISION 2: Vocal Level")
        print(f"  {'─'*50}")

        best_vox_rms, best_inst_rms = self._find_best_levels(
            vox_audio, beat_drums, beat_bass, beat_other)
        print(f"    → Vocals: {best_vox_rms:.3f} RMS, "
              f"Inst: {best_inst_rms:.3f} RMS")

        # ============================================================
        # DECISION 3: Best beat region (scored experiment)
        # ============================================================
        print(f"\n  DECISION 3: Best Beat Region")
        print(f"  {'─'*50}")

        if direction == 'a_vocals':
            beat_dna = dna_b
        else:
            beat_dna = dna_a

        beat_region_start, beat_region_end = self._find_best_beat_region(
            vox_audio, beat_drums, beat_bass, beat_other,
            beat_dna, target_duration=180.0)

        print(f"    → Beat region: {beat_region_start:.1f}-{beat_region_end:.1f}s")

        # ============================================================
        # DECISION 4: Best vocal offset (scored experiment)
        # ============================================================
        print(f"\n  DECISION 4: Vocal Timing Offset")
        print(f"  {'─'*50}")

        if direction == 'a_vocals':
            vox_dna = dna_a
        else:
            vox_dna = dna_b

        vocal_offset = self._find_best_vocal_offset(
            vox_audio, beat_drums, beat_bass, beat_other,
            beat_region_start, vox_dna, used_stretch)

        print(f"    → Vocal starts at {vocal_offset:.1f}s into vocal song")

        # ============================================================
        # DECISION 5: Arrangement — when should vocals be ON vs OFF?
        # ============================================================
        print(f"\n  DECISION 5: Vocal Arrangement")
        print(f"  {'─'*50}")

        beat_duration = beat_region_end - beat_region_start
        total_samples = int(beat_duration * self.sr)

        vocal_envelope = self._discover_arrangement(
            vox_audio, beat_drums, beat_bass, beat_other,
            beat_region_start, vocal_offset, used_stretch,
            total_samples, beat_dna)

        # ============================================================
        # BUILD FINAL MIX
        # ============================================================
        print(f"\n  Building final mix...")
        print(f"  {'─'*50}")

        result = self._build_final(
            vox_audio, beat_drums, beat_bass, beat_other,
            beat_region_start, vocal_offset, used_stretch,
            vocal_envelope, total_samples,
            best_vox_rms, best_inst_rms)

        # Score final
        final_score = self.ears.evaluate_mix(
            result['full_mix'], vocals=result.get('vocals'))
        result['quality_scores'] = final_score

        dur = len(result['full_mix']) / self.sr
        print(f"\n{'='*60}")
        print(f"  DONE: {dur:.1f}s | Score: {final_score.overall:.3f}")
        print(f"  Beat={final_score.beat_coherence:.2f} "
              f"Vocal={final_score.vocal_clarity:.2f} "
              f"Harmony={final_score.harmonic_clarity:.2f} "
              f"Spectrum={final_score.spectral_balance:.2f}")
        print(f"{'='*60}\n")

        return result

    # ================================================================
    # DECISION 1: DIRECTION (scored experiment)
    # ================================================================

    def _test_direction(self, stems_a, stems_b, dna_a, dna_b, direction):
        """Test a direction by building a quick test clip and scoring it"""
        if direction == 'a_vocals':
            vox_raw = stems_a.get('vocals')
            beat_d = stems_b.get('drums')
            beat_b = stems_b.get('bass')
            beat_o = stems_b.get('other')
            vox_tempo = dna_a.beat_grid.tempo
            beat_tempo = dna_b.beat_grid.tempo
        else:
            vox_raw = stems_b.get('vocals')
            beat_d = stems_a.get('drums')
            beat_b = stems_a.get('bass')
            beat_o = stems_a.get('other')
            vox_tempo = dna_b.beat_grid.tempo
            beat_tempo = dna_a.beat_grid.tempo

        if vox_raw is None:
            return 0.0, None

        # Tempo: find best stretch (split difference if large)
        stretch = beat_tempo / vox_tempo
        candidates = [stretch, stretch * 2, stretch / 2]
        stretch = min(candidates, key=lambda r: abs(r - 1.0))

        # Only stretch vocals — NEVER stretch the beat (keeps it crisp)
        if abs(stretch - 1.0) > 0.02:
            vox_raw = pyrb.time_stretch(vox_raw, self.sr, stretch)

        # Key: use circle-of-fifths (Camelot wheel) to find compatible shifts.
        # Only test musically justified candidates — avoids the circular scoring
        # problem where the key-clash penalty penalises non-zero shifts even when
        # a shift genuinely improves the mashup.
        if direction == 'a_vocals':
            vocal_key = dna_a.key
            beat_key = dna_b.key
        else:
            vocal_key = dna_b.key
            beat_key = dna_a.key

        candidate_shifts = self.harmonic.get_compatible_shifts(vocal_key, beat_key, n=5)
        print(f"    Key: vocals={vocal_key!r} beat={beat_key!r}")
        print(f"    Candidates: {candidate_shifts}")

        best_shift = 0
        best_key_score = -1

        # Build quick instrumental for key testing
        inst = self._sum_stems(beat_d, beat_b, beat_o)
        clip_len = min(8 * self.sr, len(vox_raw))

        for shift in candidate_shifts:
            if shift == 0:
                test_vox_clip = vox_raw[:clip_len]
            else:
                test_vox_clip = pyrb.pitch_shift(vox_raw[:clip_len], self.sr, shift)
            test_inst_clip = inst[:clip_len] if inst is not None else np.zeros(clip_len)
            test_mix = self._quick_mix(test_vox_clip, test_inst_clip)
            score = self._quick_score(test_mix, test_vox_clip)
            print(f"      shift={shift:+d}: {self.harmonic.describe_shift(vocal_key, beat_key, shift)} score={score:.3f}")
            if score > best_key_score:
                best_key_score = score
                best_shift = shift

        # Apply best key shift
        if best_shift != 0:
            vox_raw = pyrb.pitch_shift(vox_raw, self.sr, best_shift)

        # Score the full direction
        inst = self._sum_stems(beat_d, beat_b, beat_o)
        clip_len = min(self.test_duration * self.sr, len(vox_raw),
                       len(inst) if inst is not None else self.sr)
        # Use middle of both songs (most representative)
        mid_v = len(vox_raw) // 2
        mid_i = len(inst) // 2 if inst is not None else 0
        half = clip_len // 2

        v_clip = vox_raw[max(0, mid_v-half):mid_v+half]
        i_clip = inst[max(0, mid_i-half):mid_i+half] if inst is not None else np.zeros(clip_len)

        test_mix = self._quick_mix(v_clip, i_clip)
        direction_score = self._quick_score(test_mix, v_clip)

        combo = {
            'vocals': vox_raw,
            'drums': beat_d,
            'bass': beat_b,
            'other': beat_o,
            'stretch': stretch,
            'key_shift': best_shift,
        }

        return direction_score, combo

    # ================================================================
    # DECISION 2: VOCAL LEVEL (scored experiment)
    # ================================================================

    def _find_best_levels(self, vocals, drums, bass, other):
        """Try different vocal/beat ratios, score each"""
        inst = self._sum_stems(drums, bass, other)
        if inst is None or vocals is None:
            return 0.12, 0.05

        clip_len = min(self.test_duration * self.sr, len(vocals), len(inst))
        v = vocals[:clip_len]
        i = inst[:clip_len]

        best_score = -1
        best_vox_rms = 0.12
        best_inst_rms = 0.05

        # Test different ratios — vocals must always be prominent
        for vox_rms in [0.12, 0.14, 0.16, 0.18]:
            for inst_rms in [0.04, 0.05, 0.06, 0.07]:
                v_test = self._set_rms(v.copy(), vox_rms)
                i_test = self._set_rms(i.copy(), inst_rms)
                mix = v_test + i_test
                peak = np.max(np.abs(mix))
                if peak > 0.95:
                    mix *= 0.95 / peak
                score = self._quick_score(mix, v_test)
                if score > best_score:
                    best_score = score
                    best_vox_rms = vox_rms
                    best_inst_rms = inst_rms

        print(f"    Tested 16 level combos, best: "
              f"v={best_vox_rms:.2f} i={best_inst_rms:.2f} "
              f"(score={best_score:.3f})")
        return best_vox_rms, best_inst_rms

    # ================================================================
    # DECISION 3: BEST BEAT REGION (scored experiment)
    # ================================================================

    def _find_best_beat_region(self, vocals, drums, bass, other,
                                beat_dna, target_duration=180.0):
        """Test different beat regions, score each"""
        inst = self._sum_stems(drums, bass, other)
        if inst is None:
            return 0.0, target_duration

        duration = len(inst) / self.sr
        if duration <= target_duration:
            return 5.0, max(10.0, duration - 5.0)

        best_score = -1
        best_start = 5.0
        best_end = target_duration + 5.0
        test_clip = min(self.test_duration * self.sr, len(vocals) if vocals is not None else self.sr)

        # Test regions at 15-second intervals
        step = 15.0
        tests = 0
        for start_s in np.arange(5.0, max(6.0, duration - target_duration - 5.0), step):
            end_s = start_s + target_duration
            # Score using a clip from the middle of this region
            mid = (start_s + end_s) / 2
            mid_samp = int(mid * self.sr)
            half = test_clip // 2

            i_clip = inst[max(0, mid_samp-half):mid_samp+half]
            v_clip = vocals[:len(i_clip)] if vocals is not None else np.zeros(len(i_clip))

            if len(i_clip) < self.sr or len(v_clip) < self.sr:
                continue

            mix = self._quick_mix(v_clip, i_clip)
            score = self._quick_score(mix, v_clip)
            tests += 1

            if score > best_score:
                best_score = score
                best_start = start_s
                best_end = end_s

        # Snap to downbeats
        if len(beat_dna.beat_grid.downbeat_times) > 0:
            best_start = beat_dna.beat_grid.nearest_downbeat(best_start)
            best_end = beat_dna.beat_grid.nearest_downbeat(best_end)

        print(f"    Tested {tests} regions, best score: {best_score:.3f}")
        return best_start, best_end

    # ================================================================
    # DECISION 4: VOCAL OFFSET (scored experiment)
    # ================================================================

    def _find_best_vocal_offset(self, vocals, drums, bass, other,
                                 beat_region_start, vox_dna, stretch):
        """Find where in the vocal song to start for best compatibility"""
        if vocals is None:
            return 0.0

        inst = self._sum_stems(drums, bass, other)
        if inst is None:
            return 0.0

        # Get beat clip from the chosen region
        bs = int(beat_region_start * self.sr)
        clip_len = min(self.test_duration * self.sr, len(inst) - bs)
        i_clip = inst[bs:bs + clip_len]

        # Test different starting points in the vocal song
        vox_duration = len(vocals) / self.sr
        best_score = -1
        best_offset = 0.0

        # Test at 10-second intervals across the vocal song
        step = 10.0
        tests = 0
        for offset_s in np.arange(0, max(1, vox_duration - 30), step):
            offset_samp = int(offset_s * self.sr)
            v_clip = vocals[offset_samp:offset_samp + clip_len]
            if len(v_clip) < self.sr:
                continue

            # Make same length
            test_len = min(len(v_clip), len(i_clip))
            mix = self._quick_mix(v_clip[:test_len], i_clip[:test_len])
            score = self._quick_score(mix, v_clip[:test_len])
            tests += 1

            if score > best_score:
                best_score = score
                best_offset = offset_s

        print(f"    Tested {tests} offsets, best score: {best_score:.3f}")
        return best_offset

    # ================================================================
    # DECISION 5: ARRANGEMENT (scored experiment)
    # ================================================================

    def _discover_arrangement(self, vocals, drums, bass, other,
                               beat_region_start, vocal_offset, stretch,
                               total_samples, beat_dna):
        """
        Discover when vocals should be ON vs OFF by scoring.

        Test each 8-bar chunk with vocals ON and OFF.
        Keep whichever scores better. This creates a natural arrangement
        that responds to the actual music instead of following a template.

        Then apply pro DJ structure on top:
        - First chunk always starts without vocals (intro)
        - Before the best vocal chunk, insert a 2-bar vocal solo moment
        - Last chunk fades out
        """
        bar_dur = beat_dna.beat_grid.bar_duration()
        if bar_dur <= 0:
            bar_dur = 60.0 / beat_dna.beat_grid.tempo * 4

        chunk_bars = 8
        chunk_dur = chunk_bars * bar_dur
        chunk_samples = int(chunk_dur * self.sr)
        n_chunks = max(1, total_samples // chunk_samples)

        inst = self._sum_stems(drums, bass, other)
        if inst is None:
            return np.ones(total_samples)

        # For each chunk: test vocals ON vs OFF
        envelope = np.zeros(total_samples)
        chunk_scores_on = []
        chunk_scores_off = []

        bs = int(beat_region_start * self.sr)
        vs = int(vocal_offset * self.sr)

        for i in range(n_chunks):
            chunk_start = i * chunk_samples
            chunk_end = min(chunk_start + chunk_samples, total_samples)
            clen = chunk_end - chunk_start

            # Get beat chunk
            i_start = bs + chunk_start
            i_chunk = inst[i_start:i_start + clen] if i_start + clen <= len(inst) else np.zeros(clen)

            # Get vocal chunk
            v_start = vs + chunk_start
            v_chunk = vocals[v_start:v_start + clen] if vocals is not None and v_start + clen <= len(vocals) else np.zeros(clen)

            # Score with vocals ON
            mix_on = self._quick_mix(v_chunk, i_chunk)
            score_on = self._quick_score(mix_on, v_chunk)

            # Score with vocals OFF
            score_off = self._quick_score(i_chunk, None)

            chunk_scores_on.append(score_on)
            chunk_scores_off.append(score_off)

            # Decision: vocals default ON — only turn off if dramatically worse
            # The whole point of a mashup is having vocals.
            # Bias heavily toward ON.
            vocals_on = score_on > score_off - 0.05  # vocals stay unless WAY worse

            if vocals_on:
                envelope[chunk_start:chunk_end] = 1.0

            status = "ON " if vocals_on else "OFF"
            print(f"    Chunk {i+1:2d}/{n_chunks}: "
                  f"on={score_on:.3f} off={score_off:.3f} → {status}")

        # --- Ensure vocals are ON for at least 60% of the track ---
        total_on = np.sum(envelope[:n_chunks * chunk_samples] > 0.5)
        total_len = n_chunks * chunk_samples
        if total_len > 0 and total_on / total_len < 0.6:
            # Not enough vocals — force the best-scoring ON chunks
            on_scores = list(enumerate(chunk_scores_on))
            on_scores.sort(key=lambda x: x[1], reverse=True)
            for idx, _ in on_scores:
                cs = idx * chunk_samples
                ce = min(cs + chunk_samples, total_samples)
                envelope[cs:ce] = 1.0
                total_on = np.sum(envelope[:total_len] > 0.5)
                if total_on / total_len >= 0.6:
                    break
            print(f"    DJ: Forced vocals to {total_on/total_len*100:.0f}% coverage")

        # --- Apply DJ structure on top ---

        # Rule 1: First chunk is always intro (no vocals)
        if n_chunks > 2:
            envelope[:chunk_samples] = 0.0
            print(f"    DJ: First chunk → intro (no vocals)")

        # Rule 2: Last chunk fades out
        if n_chunks > 2:
            fade_start = (n_chunks - 1) * chunk_samples
            fade_len = total_samples - fade_start
            if fade_len > 0:
                envelope[fade_start:total_samples] *= np.linspace(1, 0, fade_len)
            print(f"    DJ: Last chunk → outro (fade)")

        # Rule 3: Find the best vocal chunk and add a 2-bar vocal solo before it
        if chunk_scores_on:
            best_vocal_chunk = int(np.argmax(chunk_scores_on))
            if best_vocal_chunk > 1:
                solo_bars = 2
                solo_samples = int(solo_bars * bar_dur * self.sr)
                solo_start = best_vocal_chunk * chunk_samples - solo_samples
                if solo_start > 0:
                    # This is a vocal solo moment — we'll mark it with envelope = 2.0
                    # (the builder will interpret 2.0 as "vocal solo" = mute beat)
                    envelope[solo_start:solo_start + solo_samples] = 2.0
                    print(f"    DJ: Vocal solo before chunk {best_vocal_chunk+1}")

        # Rule 4: Second-best vocal chunk gets a "beat drop" before it
        if len(chunk_scores_on) > 3:
            scores_copy = list(chunk_scores_on)
            scores_copy[best_vocal_chunk] = -1  # exclude the one we used
            second_best = int(np.argmax(scores_copy))
            if second_best > 1:
                drop_samples = int(0.1 * self.sr)
                drop_start = second_best * chunk_samples
                if drop_start > drop_samples:
                    # Mark with 3.0 = beat drop
                    envelope[drop_start - drop_samples:drop_start] = 3.0
                    print(f"    DJ: Beat drop before chunk {second_best+1}")

        # Smooth the envelope to avoid clicks
        smooth = int(0.3 * self.sr)
        if smooth > 1:
            kernel = np.hanning(smooth)
            kernel /= kernel.sum()
            # Only smooth the 0/1 transitions, preserve the special markers
            mask_normal = (envelope <= 1.0)
            smooth_env = np.convolve(np.clip(envelope, 0, 1), kernel, mode='same')
            envelope = np.where(mask_normal, smooth_env, envelope)

        return envelope

    # ================================================================
    # BUILD FINAL MIX
    # ================================================================

    def _build_final(self, vocals, drums, bass, other,
                      beat_region_start, vocal_offset, stretch,
                      vocal_envelope, total_samples,
                      vox_rms, inst_rms):
        """Build the final mix using all decisions"""

        bs = int(beat_region_start * self.sr)
        vs = int(vocal_offset * self.sr)

        # Extract continuous streams
        out_drums = self._extract(drums, bs, total_samples)
        out_bass = self._extract(bass, bs, total_samples)
        out_other = self._extract(other, bs, total_samples)
        out_vocals = self._extract(vocals, vs, total_samples)

        # Apply vocal envelope
        # 0.0 = vocals off
        # 0.0-1.0 = normal volume
        # 2.0 = vocal solo (mute beat)
        # 3.0 = beat drop (brief silence then slam)

        normal_env = np.clip(vocal_envelope, 0, 1)
        out_vocals = out_vocals * normal_env

        # Vocal solo moments: mute the beat
        solo_mask = (vocal_envelope >= 1.9) & (vocal_envelope <= 2.1)
        if np.any(solo_mask):
            out_drums[solo_mask] *= 0.0
            out_bass[solo_mask] *= 0.0
            out_other[solo_mask] *= 0.15  # Tiny hint of melody
            # Restore vocal level in solo sections
            out_vocals[solo_mask] = self._extract(vocals, vs, total_samples)[solo_mask]

        # Beat drop moments: brief silence
        drop_mask = (vocal_envelope >= 2.9) & (vocal_envelope <= 3.1)
        if np.any(drop_mask):
            out_drums[drop_mask] *= 0.02
            out_bass[drop_mask] *= 0.02
            out_other[drop_mask] *= 0.02
            out_vocals[drop_mask] *= 0.02

        # Intro fade (first 3 seconds)
        fade_in = min(int(3 * self.sr), total_samples // 4)
        if fade_in > 0:
            fade = np.linspace(0, 1, fade_in)
            out_drums[:fade_in] *= fade
            out_bass[:fade_in] *= fade
            out_other[:fade_in] *= fade

        # Apply scored levels
        out_vocals = self._set_rms(out_vocals, vox_rms)
        out_drums = self._set_rms(out_drums, inst_rms * 1.0)
        out_bass = self._set_rms(out_bass, inst_rms * 0.7)
        out_other = self._set_rms(out_other, inst_rms * 0.5)

        # VocalEnhancer: EQ carving + presence boost + strong sidechain
        out_vocals, out_drums, out_bass, out_other = self.enhancer.process(
            out_vocals, out_drums, out_bass, out_other)

        full_mix = out_vocals + out_drums + out_bass + out_other

        # Light master
        full_mix = self._light_master(full_mix)

        return {
            'vocals': out_vocals,
            'drums': out_drums,
            'bass': out_bass,
            'other': out_other,
            'full_mix': full_mix,
        }

    # ================================================================
    # SCORING HELPERS
    # ================================================================

    def _quick_mix(self, vocals, instrumental):
        """Quick mix for testing: vocals prominent over beat"""
        v = self._set_rms(vocals.copy(), 0.14) if vocals is not None else np.zeros(1)
        i = self._set_rms(instrumental.copy(), 0.06) if instrumental is not None else np.zeros(1)
        tgt = min(len(v), len(i))
        if tgt < 1:
            return np.zeros(self.sr)
        mix = v[:tgt] + i[:tgt]
        peak = np.max(np.abs(mix))
        if peak > 0.95:
            mix *= 0.95 / peak
        return mix

    def _quick_score(self, mix, vocals=None):
        """Score a test clip"""
        try:
            score = self.ears.evaluate_mix(mix, vocals=vocals)
            return score.overall
        except Exception:
            return 0.0

    # ================================================================
    # UTILITIES
    # ================================================================

    def _mono(self, audio):
        if audio is None:
            return None
        return np.mean(audio, axis=0) if audio.ndim > 1 else audio

    def _sum_stems(self, drums, bass, other):
        parts = [s for s in [drums, bass, other] if s is not None]
        if not parts:
            return None
        mx = max(len(p) for p in parts)
        out = np.zeros(mx)
        for p in parts:
            out[:len(p)] += p
        return out

    def _extract(self, audio, start, length):
        if audio is None:
            return np.zeros(length)
        start = max(0, start)
        seg = audio[start:start + length]
        if len(seg) < length:
            return np.pad(seg, (0, length - len(seg)))
        return seg.copy()

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
