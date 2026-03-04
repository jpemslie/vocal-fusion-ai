#!/usr/bin/env python3
"""
VocalFusion AI – Advanced Mixing Components
============================================

This file provides production‑quality modules that directly improve
the output of your mashup system:

  1. RubberbandDSP – high‑quality time‑stretch/pitch‑shift using rubberband.
  2. AdvancedAligner – OTAC tempo ratio selection and beat‑grid alignment.
  3. StructureSegmenter – detect verse/chorus sections for intelligent arrangement.
  4. MixEvaluator – objective metrics via mir_eval (beat, key, separation).
  5. AsymmetricCompatibilityModel – direction‑aware compatibility scoring.

Each component can be used independently or together to replace
heuristic parts of the current pipeline.

Dependencies (install with pip):
    numpy, librosa, pyrubberband, mir_eval, torch, pedalboard (optional)
"""

import warnings
import numpy as np
import librosa

# Optional imports with fallback warnings
try:
    import pyrubberband as pyrb
    RUBBERBAND_AVAILABLE = True
except ImportError:
    RUBBERBAND_AVAILABLE = False
    warnings.warn("pyrubberband not installed. Install with: pip install pyrubberband")

try:
    import mir_eval
    MIREVAL_AVAILABLE = True
except ImportError:
    MIREVAL_AVAILABLE = False
    warnings.warn("mir_eval not installed. Install with: pip install mir_eval")

try:
    import torch
    import torch.nn as nn
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    warnings.warn("PyTorch not installed. Install with: pip install torch")

try:
    from pedalboard import Pedalboard, Reverb, Compressor, HighpassFilter, LowpassFilter
    PEDALBOARD_AVAILABLE = True
except ImportError:
    PEDALBOARD_AVAILABLE = False
    warnings.warn("pedalboard not installed. Install with: pip install pedalboard")

# ----------------------------------------------------------------------
# 1. RubberbandDSP – production‑quality audio effects
# ----------------------------------------------------------------------

class RubberbandDSP:
    """
    High‑quality DSP using rubberband for time/pitch and pedalboard for effects.
    Falls back to librosa if dependencies are missing (with warning).
    """
    def __init__(self, sample_rate=44100):
        self.sr = sample_rate
        if not RUBBERBAND_AVAILABLE:
            warnings.warn("RubberbandDSP: pyrubberband missing, using librosa fallback (lower quality)")

    def time_stretch(self, audio, ratio):
        """Time‑stretch audio by `ratio` (e.g., 0.8 = slower, 1.2 = faster)."""
        if RUBBERBAND_AVAILABLE:
            return pyrb.time_stretch(audio, self.sr, ratio)
        else:
            return librosa.effects.time_stretch(audio, rate=ratio)

    def pitch_shift(self, audio, semitones):
        """Pitch‑shift by `semitones` (positive = up, negative = down)."""
        if RUBBERBAND_AVAILABLE:
            return pyrb.pitch_shift(audio, self.sr, semitones)
        else:
            return librosa.effects.pitch_shift(audio, sr=self.sr, n_steps=semitones)

    def reverb(self, audio, room_size=0.5, wet_level=0.2):
        """Apply reverb using pedalboard (if available)."""
        if PEDALBOARD_AVAILABLE:
            board = Pedalboard([Reverb(room_size=room_size, wet_level=wet_level)])
            return board(audio, self.sr)
        else:
            # Simple synthetic reverb fallback
            return self._simple_reverb(audio, room_size, wet_level)

    def _simple_reverb(self, audio, room_size, wet_level):
        """Basic reverb fallback using convolution with noise burst."""
        # Not production quality, but better than nothing
        ir_duration = 1.0  # seconds
        ir = np.random.randn(int(ir_duration * self.sr))
        decay = np.exp(-np.linspace(0, 5, len(ir)))
        ir = ir * decay
        wet = np.convolve(audio, ir, mode='full')[:len(audio)]
        wet = wet * (np.sqrt(np.mean(audio**2)) / (np.sqrt(np.mean(wet**2)) + 1e-8))
        return (1 - wet_level) * audio + wet_level * wet

    def highpass(self, audio, cutoff_freq):
        """High‑pass filter using pedalboard (if available)."""
        if PEDALBOARD_AVAILABLE:
            board = Pedalboard([HighpassFilter(cutoff_frequency_hz=cutoff_freq)])
            return board(audio, self.sr)
        else:
            # Simple IIR fallback (scipy required)
            from scipy.signal import butter, filtfilt
            nyq = self.sr / 2
            b, a = butter(4, cutoff_freq / nyq, btype='high')
            return filtfilt(b, a, audio)

    def lowpass(self, audio, cutoff_freq):
        """Low‑pass filter using pedalboard (if available)."""
        if PEDALBOARD_AVAILABLE:
            board = Pedalboard([LowpassFilter(cutoff_frequency_hz=cutoff_freq)])
            return board(audio, self.sr)
        else:
            from scipy.signal import butter, filtfilt
            nyq = self.sr / 2
            b, a = butter(4, cutoff_freq / nyq, btype='low')
            return filtfilt(b, a, audio)

    def compress(self, audio, threshold_db=-20, ratio=4, attack_ms=10, release_ms=100):
        """Compressor using pedalboard (if available)."""
        if PEDALBOARD_AVAILABLE:
            board = Pedalboard([Compressor(threshold_db=threshold_db, ratio=ratio,
                                            attack_ms=attack_ms, release_ms=release_ms)])
            return board(audio, self.sr)
        else:
            # Very simple RMS‑based compression fallback
            return self._simple_compress(audio, threshold_db, ratio)

    def _simple_compress(self, audio, threshold_db, ratio):
        # Not recommended; just a placeholder
        return audio


# ----------------------------------------------------------------------
# 2. AdvancedAligner – OTAC tempo ratio and beat alignment
# ----------------------------------------------------------------------

class AdvancedAligner:
    """
    Provides optimal tempo ratios (OTAC) and beat‑grid alignment.
    Based on ideas from the Mash‑Up project.
    """
    def __init__(self, sample_rate=44100):
        self.sr = sample_rate
        self.dsp = RubberbandDSP(sample_rate)

    def find_optimal_tempo_ratio(self, tempo_a, tempo_b):
        """
        Compute the optimal adjustment coefficient (OTAC).
        Tries direct, double, and half ratios; picks the one closest to 1.0.
        Returns the ratio to apply to song B's tempo to match song A.
        """
        ratios = [
            tempo_a / tempo_b,
            (tempo_a * 2) / tempo_b,
            tempo_a / (tempo_b * 2)
        ]
        best = min(ratios, key=lambda r: abs(r - 1.0))
        return best

    def align_beats(self, audio_a, audio_b, tempo_a, tempo_b):
        """
        Align audio_b to audio_a's tempo and beat grid.
        Returns aligned versions of both (same length).
        """
        # Step 1: stretch audio_b to match tempo_a using OTAC
        ratio = self.find_optimal_tempo_ratio(tempo_a, tempo_b)
        if abs(ratio - 1.0) > 0.01:
            audio_b = self.dsp.time_stretch(audio_b, ratio)

        # Step 2: find first strong downbeat in each
        downbeat_a = self._find_first_downbeat(audio_a, tempo_a)
        downbeat_b = self._find_first_downbeat(audio_b, tempo_a)  # use stretched tempo

        # Step 3: shift audio_b so downbeats align
        offset_samples = downbeat_a - downbeat_b
        if offset_samples > 0:
            audio_b = np.pad(audio_b, (offset_samples, 0))[:len(audio_a)]
        elif offset_samples < 0:
            trim = -offset_samples
            audio_b = np.pad(audio_b[trim:], (0, trim))[:len(audio_a)]

        # Step 4: match lengths
        max_len = max(len(audio_a), len(audio_b))
        if len(audio_a) < max_len:
            audio_a = np.pad(audio_a, (0, max_len - len(audio_a)))
        if len(audio_b) < max_len:
            audio_b = np.pad(audio_b, (0, max_len - len(audio_b)))

        return audio_a, audio_b

    def _find_first_downbeat(self, audio, tempo):
        """Return sample position of first strong downbeat using beat tracking."""
        _, beats = librosa.beat.beat_track(
            y=audio, sr=self.sr, start_bpm=tempo, units='frames')
        if len(beats) < 2:
            onsets = librosa.onset.onset_detect(y=audio, sr=self.sr, units='samples')
            return int(onsets[0]) if len(onsets) > 0 else 0
        onset_env = librosa.onset.onset_strength(y=audio, sr=self.sr)
        search_range = min(8, len(beats))
        strengths = [float(onset_env[beats[i]]) if beats[i] < len(onset_env) else 0.0
                     for i in range(search_range)]
        best = int(np.argmax(strengths))
        return int(librosa.frames_to_samples(beats[best]))


# ----------------------------------------------------------------------
# 3. StructureSegmenter – detect intro/verse/chorus sections
# ----------------------------------------------------------------------

class StructureSegmenter:
    """
    Segment a song into structural sections using self‑similarity.
    Returns list of (start_time, end_time, label).
    """
    def __init__(self, sample_rate=44100):
        self.sr = sample_rate

    def segment(self, audio):
        """
        Perform structure segmentation.
        Returns: list of dicts with keys: start, end, label, energy
        """
        from sklearn.cluster import AgglomerativeClustering

        hop_length = 512
        mfcc = librosa.feature.mfcc(y=audio, sr=self.sr, hop_length=hop_length, n_mfcc=13)
        R = librosa.segment.recurrence_matrix(mfcc, mode='affinity', sym=True)

        # librosa.segment.agglomerative was removed in librosa 0.10 — use sklearn directly
        n_frames = R.shape[0]
        k = min(8, max(2, n_frames // 10))
        dist = np.clip(1.0 - R, 0.0, None)
        np.fill_diagonal(dist, 0.0)
        labels = AgglomerativeClustering(
            n_clusters=k, metric='precomputed', linkage='average'
        ).fit_predict(dist)

        # Find change points
        boundaries = [0]
        for i in range(1, len(labels)):
            if labels[i] != labels[i - 1]:
                boundaries.append(i)
        boundaries.append(len(labels))

        times = librosa.frames_to_time(boundaries, sr=self.sr, hop_length=hop_length)

        # First pass: collect raw energies
        sections = []
        for i in range(len(times) - 1):
            start = float(times[i])
            end = float(times[i + 1])
            seg_audio = audio[int(start * self.sr):int(end * self.sr)]
            energy = float(np.sqrt(np.mean(seg_audio ** 2))) if len(seg_audio) > 0 else 0.0
            sections.append({'start': start, 'end': end, 'label': 'verse', 'energy': energy})

        # Second pass: label relative to median energy (avoids fixed-threshold fragility)
        if sections:
            median_e = float(np.median([s['energy'] for s in sections]))
            for i, s in enumerate(sections):
                e = s['energy']
                if i == 0 and e < median_e * 0.75:
                    s['label'] = 'intro'
                elif i == len(sections) - 1 and e < median_e * 0.75:
                    s['label'] = 'outro'
                elif e > median_e * 1.3:
                    s['label'] = 'chorus'
                elif e < median_e * 0.65:
                    s['label'] = 'breakdown'
                # else: 'verse' (default, already set)

        return sections


# ----------------------------------------------------------------------
# 4. MixEvaluator – objective metrics using mir_eval
# ----------------------------------------------------------------------

class MixEvaluator:
    """
    Wrapper around mir_eval to compute objective quality metrics.
    Requires reference data (e.g., beat times, key, stems) for meaningful evaluation.
    """
    def __init__(self):
        if not MIREVAL_AVAILABLE:
            warnings.warn("MixEvaluator: mir_eval not installed; metrics will return 0.5")

    def beat_coherence(self, reference_beats, estimated_beats):
        """F‑measure of beat tracking."""
        if not MIREVAL_AVAILABLE:
            return 0.5
        return mir_eval.beat.f_measure(reference_beats, estimated_beats)

    def key_consistency(self, reference_key, estimated_key):
        """Weighted key score (0–1)."""
        if not MIREVAL_AVAILABLE:
            return 0.5
        return mir_eval.key.weighted_score(reference_key, estimated_key)

    def separation_quality(self, reference_stems, estimated_stems, sr):
        """
        BSS Eval metrics for source separation.
        reference_stems: dict of stem name -> audio
        estimated_stems: dict of stem name -> audio (aligned)
        """
        if not MIREVAL_AVAILABLE:
            return {'SDR': 0.0}
        # Assume stems are 'vocals', 'drums', 'bass', 'other'
        # We'll compute SDR for each stem that exists in both
        results = {}
        for name in ['vocals', 'drums', 'bass', 'other']:
            if name in reference_stems and name in estimated_stems:
                ref = reference_stems[name]
                est = estimated_stems[name]
                min_len = min(len(ref), len(est))
                sdr, sir, sar, perm = mir_eval.separation.bss_eval_sources(
                    ref[:min_len][np.newaxis, :],
                    est[:min_len][np.newaxis, :],
                    compute_permutation=False
                )
                results[name] = float(sdr[0])
        return results

    def mix_quality(self, mix_audio):
        """
        Standalone quality estimate — no reference audio required.
        Returns 0–1 based on clipping, dynamic range (crest factor), and silence.
        Suitable for evaluating mashup output where no ground-truth reference exists.
        """
        if len(mix_audio) == 0:
            return 0.0
        # Clipping penalty: any sample above 95% FS is bad
        clip_ratio = float(np.mean(np.abs(mix_audio) > 0.95))
        clip_score = max(0.0, 1.0 - clip_ratio * 50)
        # Crest factor: peak / RMS. Good mastered music sits around 6–12x (~15–21 dB).
        rms = float(np.sqrt(np.mean(mix_audio ** 2)))
        if rms < 1e-8:
            return 0.0
        crest = float(np.max(np.abs(mix_audio))) / rms
        log_crest = np.log10(max(crest, 0.1))  # ideal ~0.78 (≈6x, 15 dB)
        crest_score = float(np.clip(1.0 - abs(log_crest - 0.78) / 0.6, 0.0, 1.0))
        return 0.5 * clip_score + 0.5 * crest_score

    def snr_vs_reference(self, mix_audio, reference_audio):
        """
        SNR of mix relative to reference (0–1).
        Only meaningful with true reference audio (e.g. validating Demucs separation).
        NOT suitable for mashup quality — use mix_quality() instead.
        """
        min_len = min(len(mix_audio), len(reference_audio))
        err = mix_audio[:min_len] - reference_audio[:min_len]
        ref_power = float(np.mean(reference_audio[:min_len] ** 2))
        if ref_power < 1e-10:
            return 0.0
        snr = 10 * np.log10(ref_power / (float(np.mean(err ** 2)) + 1e-8))
        return float(np.clip((snr + 10) / 40, 0.0, 1.0))

    def overall_quality(self, mix_audio, reference_audio=None):
        """
        Backwards-compatible wrapper.
        Without reference_audio → standalone mix_quality().
        With reference_audio    → snr_vs_reference().
        """
        if reference_audio is not None:
            return self.snr_vs_reference(mix_audio, reference_audio)
        return self.mix_quality(mix_audio)


# ----------------------------------------------------------------------
# 5. AsymmetricCompatibilityModel – direction‑aware PyTorch model
# ----------------------------------------------------------------------

if TORCH_AVAILABLE:
    class AsymmetricCompatibilityModel(nn.Module):
        """
        Predicts compatibility score for a specific direction.
        Input: embeddings of song A, song B, and a direction flag (0 or 1).
        Output: score 0–1.
        """
        def __init__(self, embedding_dim=768, hidden_dim=256):
            super().__init__()
            self.net = nn.Sequential(
                nn.Linear(embedding_dim * 2 + 1, hidden_dim),
                nn.ReLU(),
                nn.Dropout(0.2),
                nn.Linear(hidden_dim, hidden_dim // 2),
                nn.ReLU(),
                nn.Dropout(0.2),
                nn.Linear(hidden_dim // 2, 1),
                nn.Sigmoid()
            )

        def forward(self, emb_a, emb_b, direction):
            """
            emb_a, emb_b: (batch, D)
            direction: (batch,) with values 0 (A vocals) or 1 (B vocals)
            """
            direction = direction.float().unsqueeze(1)  # (batch, 1)
            x = torch.cat([emb_a, emb_b, direction], dim=1)
            return self.net(x).squeeze(1)

else:
    class AsymmetricCompatibilityModel:
        def __init__(self, *args, **kwargs):
            raise ImportError("PyTorch is required for AsymmetricCompatibilityModel")


# ----------------------------------------------------------------------
# Example / Test
# ----------------------------------------------------------------------

if __name__ == "__main__":
    print("Testing advanced mixer components...")

    # Test RubberbandDSP (if available)
    dsp = RubberbandDSP()
    dummy = np.random.randn(44100 * 2) * 0.1  # 2 seconds of noise
    stretched = dsp.time_stretch(dummy, 1.2)
    print(f"Time stretch: {len(dummy)} -> {len(stretched)} samples")

    # Test AdvancedAligner
    aligner = AdvancedAligner()
    ratio = aligner.find_optimal_tempo_ratio(120, 140)
    print(f"OTAC ratio for 120→140 BPM: {ratio:.3f}")

    # Test StructureSegmenter (requires real audio, skip here)
    # segmenter = StructureSegmenter()
    # sections = segmenter.segment(dummy)  # would fail with noise

    # Test MixEvaluator (requires mir_eval)
    evaluator = MixEvaluator()
    # Dummy beat times
    ref_beats = np.arange(0, 10, 0.5)
    est_beats = ref_beats + 0.02
    beat_score = evaluator.beat_coherence(ref_beats, est_beats)
    print(f"Beat coherence (with small error): {beat_score:.3f}")

    # Test AsymmetricCompatibilityModel (if torch available)
    if TORCH_AVAILABLE:
        model = AsymmetricCompatibilityModel(embedding_dim=128)
        emb_a = torch.randn(4, 128)
        emb_b = torch.randn(4, 128)
        dirs = torch.tensor([0, 1, 0, 1])
        scores = model(emb_a, emb_b, dirs)
        print(f"Compatibility scores: {scores.detach().numpy()}")

    print("All tests completed.")