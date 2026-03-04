"""
VocalFusion AI — The AI DJ
=============================

PHILOSOPHY: Precision over vagueness. Every element placed at an exact
sample position. No soft gates, no continuous streams, no ambiguity.

PAWSA-STYLE DJ STRUCTURE:
  Beat plays continuously, fully structured around its own energy profile.
  Vocal stem is cut into discrete phrases at hard silence boundaries.
  Phrases placed bar-accurately based on beat energy sections:

    bars  0-7  : beat-only intro (drums build in, bass after 2 bars,
                 melody after 4 bars — staggered entry like a real DJ)
    bars  8+   : vocal phrases placed in temporal order from original song,
                 matched to beat section energy:
                   hook phrases  → drop sections  (high beat energy)
                   groove phrases→ groove sections (medium energy)
                   atmosphere    → build sections  (rising energy)
                   breakdown sections → always beat-only (breath moment)
    last 8 bars: beat-only outro, everything fades out

DECISIONS MADE BY SCORING:
  1. Direction:   A vocals + B beat, or B vocals + A beat?
  2. Beat region: which 3-minute window of the beat song sounds best?
  3. Levels:      what vocal/beat RMS ratio sounds best?
"""

import numpy as np
import librosa
from typing import Dict, Optional, Tuple, List
from dataclasses import dataclass, field

try:
    import pyloudnorm as pyln
    _PYLOUDNORM = True
except ImportError:
    _PYLOUDNORM = False

try:
    from pedalboard import Pedalboard, Compressor as PBCompressor
    _PEDALBOARD = True
except ImportError:
    _PEDALBOARD = False

from audio.dsp import EnhancedDSP
from analysis.mix_intelligence import MixIntelligence
from analysis.song_dna import SongDNA
from audio.vocal_enhancer import VocalEnhancer
from audio.transition_fx import TransitionFX
from audio.vocal_quality_filter import VocalQualityFilter
from audio.energy_automation import EnergyAutomation

try:
    from advanced_mixer import StructureSegmenter
    _SEGMENTER_AVAILABLE = True
except ImportError:
    _SEGMENTER_AVAILABLE = False


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
        self.enhancer = VocalEnhancer(sample_rate)
        self.transition_fx = TransitionFX(sample_rate)
        self.quality_filter = VocalQualityFilter(sample_rate)
        self.energy_auto = EnergyAutomation(sample_rate)
        self.test_duration = 12  # seconds per test clip

    def create_mashup(self, stems_a: Dict[str, np.ndarray],
                       stems_b: Dict[str, np.ndarray],
                       dna_a: SongDNA, dna_b: SongDNA,
                       predicted_params: Optional[Dict] = None) -> Dict:
        """
        Main entry point. Returns dict with full_mix + stems + quality_scores + params_used.

        predicted_params: optional dict from MixPredictor with keys like 'direction'.
            When provided, the AI skips that scored experiment and uses the suggestion.
        """
        print(f"\n{'='*60}")
        print(f"  AI DJ — Experiment → Score → Decide")
        print(f"{'='*60}")

        if predicted_params is None:
            predicted_params = {}

        # Mono everything upfront
        stems_a = {k: self._mono(v) for k, v in stems_a.items()}
        stems_b = {k: self._mono(v) for k, v in stems_b.items()}

        # ============================================================
        # DECISION 1: Which direction? (scored experiment)
        # ============================================================
        print(f"\n  DECISION 1: Direction")
        print(f"  {'─'*50}")

        forced_direction = predicted_params.get('direction')

        if forced_direction in ('a_vocals', 'b_vocals'):
            # Predictor has a data-backed suggestion — use it directly
            print(f"    Predictor suggests: {forced_direction} (skipping full experiment)")
            score, best_combo = self._test_direction(
                stems_a, stems_b, dna_a, dna_b, forced_direction)
            direction = forced_direction
            print(f"    → Score: {score:.3f}")
        else:
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

        # ============================================================
        # KEY MATCHING: pitch-shift vocals to match beat
        # ============================================================
        print(f"\n  KEY MATCHING")
        print(f"  {'─'*50}")

        if direction == 'a_vocals':
            vox_key  = dna_a.key
            beat_key = dna_b.key
        else:
            vox_key  = dna_b.key
            beat_key = dna_a.key

        vox_audio = self._match_key(vox_audio, vox_key, beat_key)

        # ============================================================
        # DECISION 2: Vocal level (scored experiment)
        # ============================================================
        print(f"\n  DECISION 2: Vocal Level")
        print(f"  {'─'*50}")

        if predicted_params.get('vox_rms') and predicted_params.get('inst_rms'):
            best_vox_rms = float(predicted_params['vox_rms'])
            best_inst_rms = float(predicted_params['inst_rms'])
            print(f"    → Using predicted levels: "
                  f"Vocals {best_vox_rms:.3f} RMS, Inst {best_inst_rms:.3f} RMS")
        else:
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
            vox_dna  = dna_a
        else:
            beat_dna = dna_a
            vox_dna  = dna_b

        beat_region_start, beat_region_end = self._find_best_beat_region(
            vox_audio, beat_drums, beat_bass, beat_other,
            beat_dna, target_duration=180.0)

        print(f"    → Beat region: {beat_region_start:.1f}-{beat_region_end:.1f}s")

        # ============================================================
        # BUILD FINAL MIX — phrase-level placement
        # ============================================================
        print(f"\n  Building final mix...")
        print(f"  {'─'*50}")

        result = self._build_final(
            vox_audio, beat_drums, beat_bass, beat_other,
            beat_region_start, beat_region_end,
            beat_dna, best_vox_rms, best_inst_rms,
            vox_dna=vox_dna)

        # Score final
        final_score = self.ears.evaluate_mix(
            result['full_mix'], vocals=result.get('vocals'))
        result['quality_scores'] = final_score

        # Record what was decided — used by MixPredictor for learning
        result['params_used'] = {
            'direction': direction,
            'vox_rms': best_vox_rms,
            'inst_rms': best_inst_rms,
        }

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

        # Downbeat alignment only — no stretch, no pitch shift.
        # Just nudge vocals so bar 1 of the vocals lands on bar 1 of the beat.
        # IMPORTANT: detect vocal downbeat using VOX tempo, not beat tempo —
        # using the wrong BPM makes the phase detection find the wrong bar start.
        inst_temp = self._sum_stems(beat_d, beat_b, beat_o)
        if inst_temp is not None and len(vox_raw) > self.sr:
            vox_db = self.dsp._find_first_downbeat(vox_raw, vox_tempo)
            inst_db = self.dsp._find_first_downbeat(inst_temp, beat_tempo)
            offset = inst_db - vox_db
            max_shift = int(4 * (60.0 / max(beat_tempo, 1)) * 4 * self.sr)
            if 0 < offset < max_shift:
                vox_raw = np.pad(vox_raw, (offset, 0))
                print(f"    Downbeat: delayed vocals by {offset / self.sr * 1000:.0f}ms")
            elif -max_shift < offset < 0:
                vox_raw = vox_raw[abs(offset):]
                print(f"    Downbeat: advanced vocals by {abs(offset) / self.sr * 1000:.0f}ms")

        # Score direction (vocals completely untouched beyond downbeat alignment)
        inst = self._sum_stems(beat_d, beat_b, beat_o)
        clip_len = min(self.test_duration * self.sr, len(vox_raw),
                       len(inst) if inst is not None else self.sr)
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
            'stretch': 1.0,
            'key_shift': 0,
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
        # Use the loudest clip window rather than the first N seconds.
        # Many songs have instrumental intros — testing on silence produces
        # meaningless level scores and the experiment picks arbitrary values.
        v = self._loudest_window(vocals, clip_len)
        i = inst[:clip_len]

        best_score = -1
        best_vox_rms = 0.09
        best_inst_rms = 0.12

        # DJ-style ranges: beat always at or above vocal level
        for vox_rms in [0.07, 0.09, 0.11, 0.13]:
            for inst_rms in [0.10, 0.12, 0.14, 0.16]:
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
    # PHRASE EXTRACTION — cut vocal stem into discrete sung phrases
    # ================================================================

    def _extract_vocal_phrases(self, vocals: np.ndarray) -> list:
        """
        Cut the vocal stem into discrete sung phrases at silence boundaries.
        Returns list of phrase dicts sorted by energy (loudest first).

        Each phrase: {start, end, duration, energy}
          start/end — sample offsets into the original vocal stem
          duration  — seconds
          energy    — RMS (used to rank importance)

        Algorithm:
          1. Compute short-time RMS (50ms frames)
          2. Hard threshold at -40 dB — below = silence, above = active
          3. Bridge gaps < 0.4s so a single breath doesn't split a phrase
          4. Discard regions shorter than 0.8s (false positives / breaths)
        """
        if vocals is None or len(vocals) == 0:
            return []

        MIN_PHRASE_S = 0.5   # discard anything shorter than this

        # ── Method 1: Silero VAD (state-of-the-art, enterprise-grade) ──────────
        # Silero VAD is trained specifically for clean speech/vocal detection.
        # It returns confident phrase boundaries, not just RMS thresholding.
        # Requires 16 kHz input; we resample, detect, then map back to original SR.
        try:
            import torch
            from silero_vad import load_silero_vad, get_speech_timestamps

            audio_16k = librosa.resample(
                vocals.astype(np.float32), orig_sr=self.sr, target_sr=16000)
            audio_tensor = torch.from_numpy(audio_16k)
            model = load_silero_vad()

            # threshold=0.45: slightly permissive to catch breathy vocals
            # min_speech_duration_ms: ignore sub-500ms fragments (breaths, pops)
            # min_silence_duration_ms: gap must be ≥250ms to split two phrases
            timestamps = get_speech_timestamps(
                audio_tensor, model,
                sampling_rate=16000,
                threshold=0.45,
                min_speech_duration_ms=int(MIN_PHRASE_S * 1000),
                min_silence_duration_ms=250,
            )

            ratio = self.sr / 16000
            phrases = []
            for ts in timestamps:
                s = int(ts['start'] * ratio)
                e = min(int(ts['end'] * ratio), len(vocals))
                if e - s < int(MIN_PHRASE_S * self.sr):
                    continue
                energy = float(np.sqrt(np.mean(vocals[s:e] ** 2)))
                phrases.append({'start': s, 'end': e,
                                'duration': (e - s) / self.sr,
                                'energy': energy})

            print(f"    Silero VAD: {len(phrases)} vocal phrases")
            for ph in phrases[:8]:
                print(f"      {ph['start']/self.sr:.1f}s–{ph['end']/self.sr:.1f}s "
                      f"({ph['duration']:.1f}s) energy={ph['energy']:.5f}")
            return sorted(phrases, key=lambda p: p['energy'], reverse=True)

        except Exception as e:
            print(f"    Silero VAD unavailable ({type(e).__name__}), using adaptive RMS gate")

        # ── Method 2: Adaptive RMS gate fallback ───────────────────────────────
        HOP = 256
        FRAME = 1024
        BRIDGE_FRAMES = int(0.4 * self.sr / HOP)   # bridge gaps < 0.4s

        rms = librosa.feature.rms(
            y=vocals.astype(np.float32),
            frame_length=FRAME, hop_length=HOP)[0]
        db = librosa.amplitude_to_db(rms + 1e-8, ref=1.0)

        # Adaptive: 15th-percentile adapts to this song's noise floor
        THRESHOLD_DB = float(np.clip(np.percentile(db, 15), -55.0, -25.0))
        print(f"    Adaptive RMS gate: {THRESHOLD_DB:.1f} dB threshold")
        active = (db > THRESHOLD_DB)

        # Bridge short gaps
        i = 0
        while i < len(active):
            if not active[i]:
                j = i
                while j < len(active) and not active[j]:
                    j += 1
                if (j - i) <= BRIDGE_FRAMES and i > 0 and j < len(active):
                    active[i:j] = True
                i = j
            else:
                i += 1

        min_phr_frames = int(MIN_PHRASE_S * self.sr / HOP)
        phrases = []
        in_phrase = False
        p_start = 0

        for idx, a in enumerate(active):
            if a and not in_phrase:
                p_start = idx
                in_phrase = True
            elif not a and in_phrase:
                if idx - p_start >= min_phr_frames:
                    s = p_start * HOP
                    e = min(idx * HOP, len(vocals))
                    energy = float(np.sqrt(np.mean(vocals[s:e] ** 2)))
                    phrases.append({'start': s, 'end': e,
                                    'duration': (e - s) / self.sr,
                                    'energy': energy})
                in_phrase = False

        if in_phrase:
            s = p_start * HOP
            e = len(vocals)
            if e - s >= min_phr_frames * HOP:
                energy = float(np.sqrt(np.mean(vocals[s:e] ** 2)))
                phrases.append({'start': s, 'end': e,
                                'duration': (e - s) / self.sr,
                                'energy': energy})

        print(f"    RMS gate: {len(phrases)} vocal phrases")
        for ph in phrases[:8]:
            print(f"      {ph['start']/self.sr:.1f}s–{ph['end']/self.sr:.1f}s "
                  f"({ph['duration']:.1f}s) energy={ph['energy']:.5f}")
        return sorted(phrases, key=lambda p: p['energy'], reverse=True)

    # ================================================================
    # BEAT STRUCTURE ANALYSIS
    # ================================================================

    def _analyze_beat_bars(self, drums, bass, other,
                            total_samples: int, beat_dna) -> tuple:
        """
        Compute per-bar energy for the beat stems and label each bar.

        Returns:
          bar_energies : np.ndarray  — normalized 0–1 RMS per bar
          bar_labels   : list[str]   — 'intro'|'build'|'groove'|'drop'|'breakdown'
          bar_samples  : int         — samples per bar
        """
        bar_dur = beat_dna.beat_grid.bar_duration()
        if bar_dur <= 0:
            bar_dur = 60.0 / max(beat_dna.beat_grid.tempo, 60) * 4
        bar_samples = max(1, int(bar_dur * self.sr))
        total_bars  = max(1, total_samples // bar_samples)

        # Compute per-bar RMS arrays for each stem separately
        def _bar_rms(stem):
            out = np.zeros(total_bars)
            if stem is None:
                return out
            for b in range(total_bars):
                s, e = b * bar_samples, min((b + 1) * bar_samples, total_samples)
                seg = stem[s:e]
                out[b] = np.sqrt(np.mean(seg ** 2)) if len(seg) > 0 else 0.0
            return out

        d_bars  = _bar_rms(drums)
        bs_bars = _bar_rms(bass)
        o_bars  = _bar_rms(other)

        # Adaptive weights: the stem with the most dynamic range across bars
        # is the one driving the song's energy structure — weight it most.
        # In house/techno the drums drive; in synth-wave the melodic stem drives.
        def _dyn(arr):
            m = arr.mean()
            return arr.std() / (m + 1e-8)

        d_dyn  = _dyn(d_bars)
        bs_dyn = _dyn(bs_bars)
        o_dyn  = _dyn(o_bars)
        total_dyn = d_dyn + bs_dyn + o_dyn + 1e-8

        energies = (d_bars  * (d_dyn  / total_dyn) +
                    bs_bars * (bs_dyn / total_dyn) +
                    o_bars  * (o_dyn  / total_dyn))

        # 3-bar smoothing to remove single-bar spikes
        if len(energies) >= 3:
            kernel   = np.array([0.25, 0.50, 0.25])
            energies = np.convolve(energies, kernel, mode='same')

        e_max = energies.max()
        energies_norm = energies / e_max if e_max > 0 else energies

        # Percentile-based thresholds: always produces a spread of section types
        # regardless of the song's overall dynamic range.
        p75 = float(np.percentile(energies_norm, 75))   # top 25% = drop
        p40 = float(np.percentile(energies_norm, 40))   # middle  = groove
        p20 = float(np.percentile(energies_norm, 20))   # low     = build
        # below p20 = breakdown

        labels = []
        for e in energies_norm:
            if   e >= p75: labels.append('drop')
            elif e >= p40: labels.append('groove')
            elif e >= p20: labels.append('build')
            else:          labels.append('breakdown')

        from collections import Counter
        counts = Counter(labels)
        print(f"    Beat structure ({total_bars} bars, adaptive thresholds "
              f"drop≥{p75:.2f} groove≥{p40:.2f} build≥{p20:.2f}): {dict(counts)}")

        return energies_norm, labels, bar_samples

    # ================================================================
    # PHRASE PLACEMENT — DJ-aware, energy-matched, temporal order
    # ================================================================

    def _place_phrases_dj_style(self, phrases: list, bar_labels: list,
                                 bar_samples: int, total_samples: int,
                                 beat_dna) -> list:
        """
        Place vocal phrases bar-accurately using DJ logic.

        Rules (zero ambiguity):
          - Phrases kept in TEMPORAL ORDER from original song
          - Phrases categorised by relative energy:
              top 35%    → 'hook'        (place in 'drop' bars)
              mid 35–70% → 'groove'      (place in 'groove' bars)
              bottom 30% → 'atmosphere'  (place in 'build' bars)
          - 'breakdown' bars are ALWAYS beat-only — no vocals ever
          - First 8 bars: beat-only intro — no vocals
          - Last 8 bars:  beat-only outro — no vocals
          - Minimum 2-bar gap between any two phrases
          - Every phrase starts on an exact bar boundary
          - No overlaps, ever
        """
        bar_dur    = bar_samples / self.sr
        total_bars = total_samples // bar_samples

        if total_bars < 12 or not phrases:
            return [{'phrase': phrases[0], 'dest_start': 0}] if phrases else []

        # Tech house standard (Pawsa style): 32-bar intro, 16-bar outro.
        # At 128 BPM, 1 bar ≈ 1.875s → 16 bars ≈ 30s, which is realistic for a 3-min mix.
        # Never fewer than 8 bars for either boundary.
        INTRO_BARS = min(max(8, total_bars // 8), 16)
        OUTRO_BARS = min(max(8, total_bars // 10), 12)

        # PHRASE_QUANTUM: cue points in DJ software are ALWAYS at 8-bar (or 16-bar)
        # boundaries — this is the standard in all DJ software (Rekordbox, Traktor, Serato).
        # Pawsa and all tech house DJs drop vocals precisely on 8-bar phrase starts.
        PHRASE_QUANTUM = 8

        # Percentile-based phrase categorisation.
        # Fixed 0.65/0.30 thresholds fail for songs with narrow or wide dynamic ranges.
        # Quartiles always give a realistic distribution across the three types.
        energies = [p['energy'] for p in phrases]
        q75 = float(np.percentile(energies, 75))   # top quartile = hook
        q50 = float(np.percentile(energies, 50))   # above median = groove
        # below median = atmosphere

        SECTION_PREFS = {
            'hook':       ['drop',   'groove'],
            'groove':     ['groove', 'drop',  'build'],
            'atmosphere': ['build',  'groove'],
        }

        for p in phrases:
            if   p['energy'] >= q75: p['type'] = 'hook'
            elif p['energy'] >= q50: p['type'] = 'groove'
            else:                    p['type'] = 'atmosphere'

        # Occupied array: True = this bar cannot accept a phrase start
        occupied = [False] * total_bars
        for i in range(INTRO_BARS):
            occupied[i] = True
        for i in range(total_bars - OUTRO_BARS, total_bars):
            occupied[i] = True
        for i, lbl in enumerate(bar_labels):
            if lbl == 'breakdown':
                occupied[i] = True   # breakdown bars → beat breathes alone

        # Place phrases in temporal order (narrative stays intact)
        ordered = sorted(phrases, key=lambda p: p['start'])
        placements = []

        for phrase in ordered:
            phrase_bars = max(1, int(np.ceil(phrase['duration'] / bar_dur)))
            # Gap proportional to phrase length — a 0.5-bar stab doesn't need
            # 2 full bars of silence after it; a 4-bar section needs breathing room.
            gap_bars = max(1, min(3, phrase_bars // 2))
            needed   = phrase_bars + gap_bars
            ptype       = phrase.get('type', 'groove')
            preferred   = SECTION_PREFS.get(ptype, ['groove', 'drop', 'build'])

            placed = False

            # Only consider bars that fall on the 8-bar quantum grid.
            # This is the DJ industry standard: cue points always at bar 8, 16, 24...
            # First pass: preferred section type + on-grid.
            # Second pass: any section type + on-grid.
            # Last resort: any free bar (off-grid, only if truly nothing else works).
            candidate_bars = [
                b for b in range(INTRO_BARS, total_bars - OUTRO_BARS)
                if b % PHRASE_QUANTUM == 0 and not occupied[b]
            ]

            for target_lbl in preferred + ['groove', 'drop', 'build']:
                for bar in candidate_bars:
                    if bar >= len(bar_labels) or bar_labels[bar] != target_lbl:
                        continue
                    end_bar = bar + needed
                    if end_bar > total_bars:
                        continue
                    if any(occupied[bar:end_bar]):
                        continue
                    placements.append({'phrase': phrase, 'dest_start': bar * bar_samples})
                    for j in range(bar, min(end_bar, total_bars)):
                        occupied[j] = True
                    placed = True
                    break
                if placed:
                    break

            if not placed:
                # Any on-grid bar regardless of section label
                for bar in candidate_bars:
                    end_bar = bar + needed
                    if end_bar > total_bars:
                        continue
                    if any(occupied[bar:end_bar]):
                        continue
                    placements.append({'phrase': phrase, 'dest_start': bar * bar_samples})
                    for j in range(bar, min(end_bar, total_bars)):
                        occupied[j] = True
                    placed = True
                    break

            if not placed:
                # True last resort: any free bar (off-grid)
                for bar in range(INTRO_BARS, total_bars - OUTRO_BARS):
                    if occupied[bar]:
                        continue
                    end_bar = bar + needed
                    if end_bar > total_bars:
                        continue
                    if any(occupied[bar:end_bar]):
                        continue
                    placements.append({'phrase': phrase, 'dest_start': bar * bar_samples})
                    for j in range(bar, min(end_bar, total_bars)):
                        occupied[j] = True
                    placed = True
                    break

            if not placed:
                print(f"      (skipped {phrase['duration']:.1f}s {ptype} — no slot)")

        # Sort by destination so we process mix in time order
        placements.sort(key=lambda p: p['dest_start'])

        print(f"    Placed {len(placements)}/{len(ordered)} phrases:")
        for pl in placements:
            bar = pl['dest_start'] // bar_samples
            lbl = bar_labels[bar] if bar < len(bar_labels) else '?'
            print(f"      Bar {bar:3d} [{lbl:9s}] @ {pl['dest_start']/self.sr:6.1f}s "
                  f"— {pl['phrase']['duration']:.1f}s {pl['phrase'].get('type','?')}")

        return placements

    # ================================================================
    # STRUCTURE-AWARE PLACEMENT — whole sections, not fragments
    # ================================================================

    def _place_by_structure(self, vocals, vox_dna, beat_dna,
                             beat_region_start, total_samples,
                             stretch_ratio=1.0):
        """
        Map whole vocal song sections to structurally-matching beat sections.

        Instead of scattering VAD fragments at 8-bar slots (which sounds random),
        this places entire verse/chorus blocks at beat drop/groove/build sections.

        TYPE_MAP: what beat section type a vocal section prefers to land on
          chorus  → drop       (high-energy vocal over high-energy beat moment)
          verse   → instrumental (groove section — medium energy, consistent)
          buildup → buildup    (rising tension matches rising tension)
          etc.

        Returns list of {vocal_start, vocal_end, dest_start, vox_type, beat_type}
        — all positions in samples (vocal_* in the stretched vocal array).
        """
        SKIP_VOX  = {'intro', 'outro', 'breakdown', 'unknown'}
        SKIP_BEAT = {'intro', 'outro', 'unknown'}

        TYPE_MAP = {
            'chorus':       ['drop',         'instrumental', 'buildup'],
            'drop':         ['drop',         'instrumental'],
            'verse':        ['instrumental', 'buildup',      'drop'],
            'buildup':      ['buildup',      'instrumental'],
            'instrumental': ['drop',         'instrumental'],
        }

        vox_secs_raw  = getattr(vox_dna,  'sections', []) or []
        beat_secs_raw = getattr(beat_dna, 'sections', []) or []
        if not vox_secs_raw or not beat_secs_raw:
            return []

        bar_dur = beat_dna.beat_grid.bar_duration()
        if bar_dur <= 0:
            bar_dur = 60.0 / max(beat_dna.beat_grid.tempo, 1.0) * 4

        total_dur = total_samples / self.sr
        intro_s   = min(8 * bar_dur, total_dur * 0.15)
        outro_s   = total_dur - min(8 * bar_dur, total_dur * 0.15)

        # Vocal sections scaled to stretched domain (skip short or structural-only)
        vox_secs = []
        for s in vox_secs_raw:
            if s.classification in SKIP_VOX:
                continue
            if s.duration < 4.0:
                continue
            start_sc = s.start_time * stretch_ratio
            end_sc   = s.end_time   * stretch_ratio
            vox_secs.append({
                'type':  s.classification,
                'start': int(start_sc * self.sr),
                'end':   min(int(end_sc * self.sr), len(vocals)),
                'dur':   end_sc - start_sc,
            })

        if not vox_secs:
            return []

        # Beat sections relative to beat region, snapped to bar boundary
        beat_region_end = beat_region_start + total_dur
        beat_secs = []
        for s in beat_secs_raw:
            if s.classification in SKIP_BEAT:
                continue
            if s.end_time <= beat_region_start or s.start_time >= beat_region_end:
                continue
            rel_s = s.start_time - beat_region_start
            rel_e = s.end_time   - beat_region_start
            if rel_e <= intro_s or rel_s >= outro_s:
                continue
            rel_s = max(rel_s, intro_s)
            rel_e = min(rel_e, outro_s)
            if rel_e - rel_s < 4.0:
                continue
            # Snap to nearest bar boundary
            snap = max(0, round(rel_s / bar_dur)) * bar_dur
            if snap >= outro_s:
                continue
            beat_secs.append({'type': s.classification, 'start': snap, 'end': rel_e})

        if not beat_secs:
            return []

        print(f"      {len(vox_secs)} vocal sections × {len(beat_secs)} beat slots")

        # Greedy match: each vocal section (temporal order) → best beat slot
        ALL_TYPES = ['drop', 'instrumental', 'buildup', 'breakdown', 'verse', 'chorus']
        used = set()
        placements = []

        for vsec in vox_secs:
            preferred = TYPE_MAP.get(vsec['type'], ['drop', 'instrumental'])
            search_order = preferred + [t for t in ALL_TYPES if t not in preferred]

            best_j = None
            for target in search_order:
                for j, bsec in enumerate(beat_secs):
                    if j not in used and bsec['type'] == target:
                        best_j = j
                        break
                if best_j is not None:
                    break

            # Ultimate fallback: earliest free slot
            if best_j is None:
                for j in range(len(beat_secs)):
                    if j not in used:
                        best_j = j
                        break

            if best_j is None:
                print(f"      {vsec['type']:12s} ({vsec['dur']:.1f}s): no slot")
                continue

            bsec = beat_secs[best_j]
            dest = int(bsec['start'] * self.sr)
            if dest >= total_samples:
                continue

            used.add(best_j)
            placements.append({
                'vocal_start': vsec['start'],
                'vocal_end':   vsec['end'],
                'dest_start':  dest,
                'vox_type':    vsec['type'],
                'beat_type':   bsec['type'],
            })
            print(f"      {vsec['type']:12s} ({vsec['dur']:.1f}s) "
                  f"→ [{bsec['type']:12s}] @ {bsec['start']:.1f}s")

        placements.sort(key=lambda p: p['dest_start'])
        return placements

    # ================================================================
    # BUILD FINAL MIX — Pawsa-style, sample-accurate
    # ================================================================

    def _build_final(self, vocals, drums, bass, other,
                      beat_region_start, beat_region_end,
                      beat_dna, vox_rms, inst_rms, vox_dna=None):
        """
        Pawsa-style DJ build:

        Beat structure:
          - Drums start immediately but fade in over 4 bars
          - Bass silent for 2 bars, then fades in over 2 bars
          - Melodic ('other') silent for 4 bars, fades in over 4 bars
            → staggered entry builds tension exactly like a DJ intro
          - Everything fades out over the last 8 bars

        Vocal placement (structure-first):
          - Primary: whole verse/chorus sections mapped to beat drop/groove/build
            (keeps lyrics coherent — no random fragment scattering)
          - Fallback: VAD phrase placement if < 2 sections found in song_dna
          - 15ms crossfade at every phrase edge to kill clicks
          - 'other' stem ducked 60% during every vocal section (no bleed)
        """
        bs            = int(beat_region_start * self.sr)
        total_samples = int((beat_region_end - beat_region_start) * self.sr)

        # ── Beat stems ─────────────────────────────────────────────
        out_drums = self._extract(drums, bs, total_samples)
        out_bass  = self._extract(bass,  bs, total_samples)
        out_other = self._extract(other, bs, total_samples)

        # ── Beat structure analysis ─────────────────────────────────
        bar_energies, bar_labels, bar_samples = self._analyze_beat_bars(
            out_drums, out_bass, out_other, total_samples, beat_dna)

        bar_dur = bar_samples / self.sr

        # ── Tempo matching: stretch vocals to beat BPM ──────────────
        # Measure actual stretch ratio by comparing array lengths —
        # this handles half/double-time correction and out-of-range caps.
        if vox_dna is not None:
            orig_len = len(vocals) if vocals is not None else 0
            vocals = self._match_tempo(
                vocals, vox_dna.beat_grid.tempo, beat_dna.beat_grid.tempo)
            stretched_len = len(vocals) if vocals is not None else orig_len
            stretch_ratio = stretched_len / orig_len if orig_len > 0 else 1.0
        else:
            stretch_ratio = 1.0

        # ── Other-stem EQ carve ──────────────────────────────────────
        # Carve a pocket in the beat's melodic stem so it doesn't mask vocals.
        # Do this before placement so the carve is baked in before level-setting.
        if out_other is not None and np.any(out_other != 0):
            out_other = self.enhancer._process_other(out_other)

        # ── Placement: structure-first, fragment fallback ────────────
        # Primary: map whole verse/chorus sections to beat drop/groove/build.
        # Keeps the vocal narrative coherent (no random fragment scattering).
        # Fallback: VAD phrase placement only if song_dna found < 2 sections.
        print(f"    Vocal placement:")
        placements = []

        if vox_dna is not None and getattr(vox_dna, 'sections', None):
            placements = self._place_by_structure(
                vocals, vox_dna, beat_dna,
                beat_region_start, total_samples, stretch_ratio)

        if len(placements) < 2:
            print(f"      Falling back to phrase-based placement...")
            phrases = self._extract_vocal_phrases(vocals)
            phrases = self.quality_filter.filter_phrases(phrases, vocals)
            phrase_pls = self._place_phrases_dj_style(
                phrases, bar_labels, bar_samples, total_samples, beat_dna)
            placements = [{'vocal_start': p['phrase']['start'],
                           'vocal_end':   p['phrase']['end'],
                           'dest_start':  p['dest_start']}
                          for p in phrase_pls]

        out_vocals   = np.zeros(total_samples, dtype=np.float64)
        vocal_active = np.zeros(total_samples, dtype=np.float64)
        FADE_N = int(0.015 * self.sr)

        for pl in placements:
            src_s      = pl['vocal_start']
            src_e      = pl['vocal_end']
            dest_start = pl['dest_start']

            if vocals is None or src_e <= src_s or src_s >= len(vocals):
                continue
            src      = vocals[src_s:min(src_e, len(vocals))].astype(np.float64)
            dest_end = min(dest_start + len(src), total_samples)
            actual   = dest_end - dest_start
            if actual <= 0:
                continue
            src = src[:actual]

            # Apply EQ to only the phrases that actually get placed —
            # not the full 3-minute stem. Demucs artifacts in the silent
            # gaps never enter the mix, so we don't amplify them.
            src = self.enhancer._process_vocals(src)

            fade_n = min(FADE_N, actual // 4)
            if fade_n > 0:
                src[:fade_n]  *= np.linspace(0.0, 1.0, fade_n)
                src[-fade_n:] *= np.linspace(1.0, 0.0, fade_n)

            out_vocals[dest_start:dest_end]   = src
            vocal_active[dest_start:dest_end] = 1.0

        # ── Reverb + delay tails on every vocal phrase ─────────────
        out_vocals = self.transition_fx.apply_phrase_tails(
            out_vocals, placements, beat_bpm=beat_dna.beat_grid.tempo)

        # ── Pawsa-style staggered beat intro ───────────────────────
        # Drums fade in over 4 bars — feel the kick arrive
        drum_fade_n = min(int(4 * bar_dur * self.sr), total_samples // 4)
        if drum_fade_n > 0:
            out_drums[:drum_fade_n] *= np.linspace(0.0, 1.0, drum_fade_n)

        # Bass: silent 2 bars, then fade in over 2 bars
        bass_silent_n = min(int(2 * bar_dur * self.sr), total_samples // 4)
        bass_fade_n   = min(int(2 * bar_dur * self.sr), total_samples // 4)
        out_bass[:bass_silent_n] = 0.0
        if bass_fade_n > 0:
            end = min(bass_silent_n + bass_fade_n, total_samples)
            out_bass[bass_silent_n:end] *= np.linspace(0.0, 1.0, end - bass_silent_n)

        # Melodic (other): silent 4 bars, fade in over 4 bars
        other_silent_n = min(int(4 * bar_dur * self.sr), total_samples // 4)
        other_fade_n   = min(int(4 * bar_dur * self.sr), total_samples // 4)
        out_other[:other_silent_n] = 0.0
        if other_fade_n > 0:
            end = min(other_silent_n + other_fade_n, total_samples)
            out_other[other_silent_n:end] *= np.linspace(0.0, 1.0, end - other_silent_n)

        # ── Outro: everything fades over last 8 bars ────────────────
        outro_n     = min(int(8 * bar_dur * self.sr), total_samples // 4)
        outro_start = max(0, total_samples - outro_n)
        if outro_start < total_samples:
            fade = np.linspace(1.0, 0.0, total_samples - outro_start)
            out_drums[outro_start:]  *= fade
            out_bass[outro_start:]   *= fade
            out_other[outro_start:]  *= fade
            out_vocals[outro_start:] *= fade

        # ── Set levels ──────────────────────────────────────────────
        # Beat is always the loudest element (DJ style: groove is king)
        out_drums  = self._set_rms(out_drums,  inst_rms)
        out_bass   = self._set_rms(out_bass,   inst_rms * 0.85)
        # 'other' contains the beat song's chords/melody — playing it loud
        # under a different song's vocals causes constant harmonic clash.
        # Keep it at 25% as texture only, nearly silent during vocal phrases.
        out_other  = self._set_rms(out_other,  inst_rms * 0.25)
        out_vocals = self._set_rms_active(out_vocals, vox_rms)

        # Nearly mute 'other' during vocal phrases (90% duck) — the beat
        # song's chords actively fight the vocal's melody when both play.
        out_other *= (1.0 - vocal_active * 0.90)

        # ── Energy arc: automate vocal level + filter sweeps into drops ──
        out_vocals = self.energy_auto.apply_vocal_automation(
            out_vocals, bar_labels, bar_samples)
        out_drums, out_bass, out_other = self.energy_auto.apply_drop_filter_sweep(
            out_drums, out_bass, out_other, bar_labels, bar_samples)

        full_mix = out_vocals + out_drums + out_bass + out_other
        full_mix   = np.nan_to_num(full_mix,   nan=0.0, posinf=0.0, neginf=0.0)
        out_vocals = np.nan_to_num(out_vocals, nan=0.0, posinf=0.0, neginf=0.0)
        full_mix   = self._light_master(full_mix)

        return {
            'vocals':   out_vocals,
            'drums':    out_drums,
            'bass':     out_bass,
            'other':    out_other,
            'full_mix': full_mix,
        }

    # ================================================================
    # TEMPO MATCHING
    # ================================================================

    def _match_tempo(self, vocals, vox_bpm, beat_bpm):
        """
        Time-stretch vocals to match beat BPM using pyrubberband R3 engine.

        Without this, lyrics drift off the beat whenever the two songs have
        different tempos (e.g. 116 BPM vocal over 128 BPM beat = 1 beat
        of drift every 8 bars).

        Quality caps:
          - Only stretches if ratio is within 0.86–1.16 (±14%)
          - Checks for half/double time first (ratio near 2:1 or 1:2)
          - R3 engine (--fine flag) is transparent at these small ratios
        """
        if vocals is None or len(vocals) == 0:
            return vocals
        if vox_bpm <= 0 or beat_bpm <= 0:
            return vocals

        ratio = beat_bpm / vox_bpm

        # Half-time / double-time — treat as in-range
        if 1.85 <= ratio <= 2.15:
            ratio /= 2.0
        elif 0.47 <= ratio <= 0.54:
            ratio *= 2.0

        if abs(ratio - 1.0) < 0.005:
            print(f"    Tempo: {vox_bpm:.0f}→{beat_bpm:.0f} BPM (no stretch needed)")
            return vocals

        if not (0.86 <= ratio <= 1.16):
            print(f"    Tempo gap {vox_bpm:.0f}→{beat_bpm:.0f} BPM "
                  f"(ratio {ratio:.2f} outside ±14% quality limit, skipping)")
            return vocals

        try:
            import pyrubberband
            stretched = pyrubberband.time_stretch(
                vocals.astype(np.float32), self.sr, ratio,
                rbargs={"--fine": ""},
            )
            stretched = np.nan_to_num(stretched, nan=0.0, posinf=0.0, neginf=0.0)
            print(f"    Tempo match: {vox_bpm:.0f}→{beat_bpm:.0f} BPM "
                  f"(stretch ×{ratio:.3f})")
            return stretched.astype(np.float64)
        except Exception as e:
            print(f"    Tempo stretch failed ({e}), vocals at original BPM")
            return vocals

    # ================================================================
    # KEY MATCHING
    # ================================================================

    _NOTE_MAP = {
        'C': 0, 'C#': 1, 'Db': 1, 'D': 2, 'D#': 3, 'Eb': 3,
        'E': 4, 'F': 5, 'F#': 6, 'Gb': 6, 'G': 7, 'G#': 8,
        'Ab': 8, 'A': 9, 'A#': 10, 'Bb': 10, 'B': 11,
    }

    def _parse_key(self, key_str: str):
        """Parse 'C# minor' → (semitone_from_C, mode). Returns (None, None) on failure."""
        if not key_str:
            return None, None
        parts = key_str.strip().split()
        root = self._NOTE_MAP.get(parts[0])
        if root is None:
            return None, None
        mode = 'minor' if len(parts) > 1 and 'min' in parts[1].lower() else 'major'
        return root, mode

    def _match_key(self, vocals: np.ndarray, vox_key: str, beat_key: str) -> np.ndarray:
        """Pitch-shift vocals to match beat key. Skips if shift > 6 semitones."""
        vox_root, _ = self._parse_key(vox_key)
        beat_root, _ = self._parse_key(beat_key)

        if vox_root is None or beat_root is None:
            print(f"    Key match: could not parse ('{vox_key}' / '{beat_key}'), skipping")
            return vocals

        diff = beat_root - vox_root
        # Shortest chromatic path
        if diff > 6:
            diff -= 12
        elif diff < -6:
            diff += 12

        if diff == 0:
            print(f"    Key match: {vox_key} → {beat_key} (same root, no shift)")
            return vocals

        if abs(diff) > 6:
            print(f"    Key match: {vox_key} → {beat_key} = {diff:+d}st (>6, skipping)")
            return vocals

        print(f"    Key shift: {vox_key} → {beat_key} = {diff:+d} semitones")
        try:
            import pyrubberband
            shifted = pyrubberband.pitch_shift(
                vocals.astype(np.float32), self.sr, diff,
                rbargs={"--formant": ""},
            )
            shifted = np.nan_to_num(shifted, nan=0.0, posinf=0.0, neginf=0.0)
            return shifted.astype(np.float64)
        except Exception as e:
            print(f"    Key shift failed ({e}), using original pitch")
            return vocals

    # ================================================================
    # SCORING HELPERS
    # ================================================================

    def _quick_mix(self, vocals, instrumental):
        """Quick mix for scoring: beat-forward DJ style (beat louder than vocals)"""
        v = self._set_rms(vocals.copy(), 0.09) if vocals is not None else np.zeros(1)
        i = self._set_rms(instrumental.copy(), 0.12) if instrumental is not None else np.zeros(1)
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

    def _loudest_window(self, audio: np.ndarray, window_samples: int) -> np.ndarray:
        """Return the window_samples-length slice with the highest RMS.
        Searches in steps of 1 second to keep it fast."""
        if audio is None or len(audio) <= window_samples:
            return audio if audio is not None else np.zeros(window_samples)
        step = self.sr  # 1-second steps
        best_rms = -1.0
        best_start = 0
        for start in range(0, len(audio) - window_samples, step):
            rms = float(np.sqrt(np.mean(audio[start:start + window_samples] ** 2)))
            if rms > best_rms:
                best_rms = rms
                best_start = start
        return audio[best_start: best_start + window_samples]

    def _set_rms(self, audio, target):
        if audio is None or np.all(audio == 0):
            return audio if audio is not None else np.zeros(self.sr)
        rms = np.sqrt(np.mean(audio ** 2))
        if rms < 1e-8:
            return audio
        return audio * min(target / rms, 8.0)

    def _set_rms_active(self, audio, target):
        """Like _set_rms but computes RMS only over non-silent samples.
        Used for out_vocals which contains long stretches of silence between
        phrases — using full-array RMS would cause over-boosting."""
        if audio is None or np.all(audio == 0):
            return audio if audio is not None else np.zeros(self.sr)
        active = audio[np.abs(audio) > 1e-6]
        if len(active) < self.sr // 10:   # fewer than 100ms of content
            return audio
        rms = np.sqrt(np.mean(active ** 2))
        if rms < 1e-8:
            return audio
        return audio * min(target / rms, 8.0)

    def _light_master(self, audio):
        """Bus compression → soft-clip → LUFS normalisation → true peak limit."""
        if audio is None or np.sqrt(np.mean(audio ** 2)) < 1e-8:
            return audio

        # 1. Bus compressor — glue the mix
        if _PEDALBOARD:
            try:
                board = Pedalboard([PBCompressor(
                    threshold_db=-18.0, ratio=2.0,
                    attack_ms=10.0, release_ms=100.0)])
                audio = board(audio.astype(np.float32), self.sr).astype(np.float64)
                audio = np.nan_to_num(audio, nan=0.0, posinf=0.0, neginf=0.0)
            except Exception:
                pass

        # 2. Soft-clip peaks > 0.9 (tanh curve — no harsh digital clipping)
        audio = np.tanh(audio / 0.9) * 0.9

        # 3. Normalize to -14 LUFS (true integrated loudness)
        if _PYLOUDNORM:
            try:
                meter = pyln.Meter(self.sr)
                loudness = meter.integrated_loudness(audio.reshape(-1, 1))
                if np.isfinite(loudness) and loudness < -1.0:
                    audio = pyln.normalize.loudness(
                        audio.reshape(-1, 1), loudness, -14.0).ravel()
            except Exception:
                pass
        else:
            rms = np.sqrt(np.mean(audio ** 2))
            if rms > 1e-8:
                audio = audio * min(10 ** (-14 / 20) / rms, 5.0)

        # 4. True peak limit to -1 dBTP (0.891)
        peak = np.max(np.abs(audio))
        if peak > 0.891:
            audio *= 0.891 / peak

        return audio
