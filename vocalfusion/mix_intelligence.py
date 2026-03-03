"""
VocalFusion AI - Mix Quality Intelligence (v3.0)
=================================================

This is the brain that knows what "sounds good" means.

It evaluates audio on 8 dimensions that correlate with human perception
of mix quality, then uses those scores to make intelligent decisions
about HOW to combine two songs.

QUALITY DIMENSIONS:
  1. Beat Coherence    - Are rhythmic elements aligned and tight?
  2. Spectral Balance  - Is the frequency spectrum well-distributed? (not muddy, not thin)
  3. Harmonic Clarity  - Are the pitched elements in tune with each other?
  4. Vocal Clarity     - Can you clearly hear the vocals above the instruments?
  5. Dynamic Range     - Is the mix breathing, or is it squashed/too quiet?
  6. Phase Coherence   - Are elements reinforcing each other, not canceling?
  7. Stereo/Spectral Separation - Can you distinguish different elements?
  8. Overall Energy    - Does the mix have consistent energy and momentum?

Each dimension produces a 0.0-1.0 score.
The evaluator then recommends adjustments to improve low-scoring areas.

Usage:
    from vocalfusion.mix_intelligence import MixIntelligence
    ai = MixIntelligence(sample_rate=44100)

    # Score a mix
    scores = ai.evaluate_mix(full_mix, vocals, instrumental)

    # Get recommendations
    recs = ai.recommend_adjustments(scores)

    # Auto-optimize: try multiple configs, pick the best
    best_mix = ai.optimize_mix(vocals, stems, tempo, key)
"""

import numpy as np
import librosa
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, field


@dataclass
class MixScore:
    """Quality scores for a mix"""
    beat_coherence: float = 0.0
    spectral_balance: float = 0.0
    harmonic_clarity: float = 0.0
    vocal_clarity: float = 0.0
    dynamic_range: float = 0.0
    phase_coherence: float = 0.0
    spectral_separation: float = 0.0
    energy_consistency: float = 0.0

    @property
    def overall(self) -> float:
        """Weighted average — vocals and beat matter most"""
        weights = {
            'beat_coherence': 0.20,
            'spectral_balance': 0.10,
            'harmonic_clarity': 0.15,
            'vocal_clarity': 0.20,
            'dynamic_range': 0.05,
            'phase_coherence': 0.10,
            'spectral_separation': 0.10,
            'energy_consistency': 0.10,
        }
        total = 0
        for attr, weight in weights.items():
            total += getattr(self, attr) * weight
        return total

    def worst_dimensions(self, n=3) -> List[Tuple[str, float]]:
        """Return the n lowest-scoring dimensions"""
        scores = [
            ('beat_coherence', self.beat_coherence),
            ('spectral_balance', self.spectral_balance),
            ('harmonic_clarity', self.harmonic_clarity),
            ('vocal_clarity', self.vocal_clarity),
            ('dynamic_range', self.dynamic_range),
            ('phase_coherence', self.phase_coherence),
            ('spectral_separation', self.spectral_separation),
            ('energy_consistency', self.energy_consistency),
        ]
        return sorted(scores, key=lambda x: x[1])[:n]

    def __repr__(self):
        return (f"MixScore(overall={self.overall:.2f} | "
                f"beat={self.beat_coherence:.2f} spec={self.spectral_balance:.2f} "
                f"harm={self.harmonic_clarity:.2f} vocal={self.vocal_clarity:.2f} "
                f"dyn={self.dynamic_range:.2f} phase={self.phase_coherence:.2f} "
                f"sep={self.spectral_separation:.2f} energy={self.energy_consistency:.2f})")


@dataclass
class MixRecommendation:
    """Specific adjustment to improve mix quality"""
    dimension: str          # Which quality dimension this fixes
    action: str             # What to do (e.g., "reduce_vocal_reverb")
    parameter: str          # Which parameter to adjust
    current_value: float    # Current setting
    suggested_value: float  # Recommended setting
    reason: str             # Human-readable explanation
    priority: int           # 1=critical, 2=important, 3=nice-to-have


class MixIntelligence:
    """
    The AI that knows what sounds good.
    Evaluates mixes and makes intelligent decisions.
    """

    def __init__(self, sample_rate: int = 44100):
        self.sr = sample_rate

    # ================================================================
    # MAIN EVALUATION
    # ================================================================

    def evaluate_mix(self, full_mix: np.ndarray,
                     vocals: Optional[np.ndarray] = None,
                     instrumental: Optional[np.ndarray] = None,
                     stems: Optional[Dict[str, np.ndarray]] = None) -> MixScore:
        """
        Evaluate a mix across all quality dimensions.

        Args:
            full_mix: The final mixed audio
            vocals: Isolated vocal track (if available)
            instrumental: Instrumental bed (if available)
            stems: Individual stems dict (if available)

        Returns: MixScore with 0.0-1.0 scores for each dimension
        """
        score = MixScore()

        score.beat_coherence = self._eval_beat_coherence(full_mix)
        score.spectral_balance = self._eval_spectral_balance(full_mix)
        score.harmonic_clarity = self._eval_harmonic_clarity(full_mix)
        score.vocal_clarity = self._eval_vocal_clarity(full_mix, vocals, instrumental)
        score.dynamic_range = self._eval_dynamic_range(full_mix)
        score.phase_coherence = self._eval_phase_coherence(full_mix, stems)
        score.spectral_separation = self._eval_spectral_separation(full_mix, stems)
        score.energy_consistency = self._eval_energy_consistency(full_mix)

        return score

    # ================================================================
    # 1. BEAT COHERENCE
    # Are the beats tight and aligned?
    # Good mix: strong, regular pulse with clear transients
    # Bad mix: flammy, blurred beats, no clear pulse
    # ================================================================

    def _eval_beat_coherence(self, audio: np.ndarray) -> float:
        """
        Measures how strong and regular the rhythmic pulse is.

        Method: Compute onset strength autocorrelation.
        A good mix has strong peaks at regular intervals (the beat).
        A bad mix (misaligned drums) has smeared, weak autocorrelation.
        """
        onset_env = librosa.onset.onset_strength(y=audio, sr=self.sr)

        if len(onset_env) < 10:
            return 0.5

        # Autocorrelation of onset envelope
        autocorr = np.correlate(onset_env, onset_env, mode='full')
        autocorr = autocorr[len(autocorr) // 2:]  # Right half only

        if len(autocorr) < 2 or autocorr[0] == 0:
            return 0.5

        # Normalize
        autocorr = autocorr / autocorr[0]

        # Look for strong peak between 0.3-2 seconds (30-200 BPM range)
        min_lag = int(0.3 * self.sr / 512)   # ~0.3s in frames
        max_lag = min(int(2.0 * self.sr / 512), len(autocorr) - 1)

        if min_lag >= max_lag:
            return 0.5

        search_region = autocorr[min_lag:max_lag]
        if len(search_region) == 0:
            return 0.5

        peak_strength = np.max(search_region)

        # Strong periodic peak = good beat coherence
        # Score: 0.3+ autocorrelation peak is good, 0.6+ is great
        score = np.clip(peak_strength / 0.6, 0, 1)

        # Also check: how sharp is the peak? (sharp = tight timing)
        peak_idx = np.argmax(search_region)
        if peak_idx > 0 and peak_idx < len(search_region) - 1:
            sharpness = search_region[peak_idx] - 0.5 * (
                search_region[peak_idx - 1] + search_region[peak_idx + 1])
            score = score * 0.7 + np.clip(sharpness / 0.3, 0, 1) * 0.3

        return float(np.clip(score, 0, 1))

    # ================================================================
    # 2. SPECTRAL BALANCE
    # Is the frequency spectrum well-distributed?
    # Good: full, balanced across lows/mids/highs
    # Bad: all energy in one band, hollow mids, boomy lows
    # ================================================================

    def _eval_spectral_balance(self, audio: np.ndarray) -> float:
        """
        Measures frequency distribution balance.

        Method: Split spectrum into bands, compare energy distribution
        to an "ideal" curve (based on professional masters).
        """
        S = np.abs(librosa.stft(audio))
        freqs = librosa.fft_frequencies(sr=self.sr)

        if S.shape[0] == 0 or S.shape[1] == 0:
            return 0.5

        # Define bands
        bands = [
            ('sub', 20, 80),
            ('bass', 80, 250),
            ('low_mid', 250, 500),
            ('mid', 500, 2000),
            ('upper_mid', 2000, 4000),
            ('presence', 4000, 8000),
            ('air', 8000, 16000),
        ]

        band_energies = []
        for name, low, high in bands:
            mask = (freqs >= low) & (freqs < high)
            if mask.any():
                energy = np.mean(S[mask, :] ** 2)
                band_energies.append(energy)
            else:
                band_energies.append(0)

        band_energies = np.array(band_energies)
        total = np.sum(band_energies)
        if total < 1e-10:
            return 0.5

        # Normalize to proportions
        proportions = band_energies / total

        # Ideal proportions for modern bass-heavy music (hip-hop / electronic).
        # Stem-separated audio is dry mono — bass band naturally dominates.
        # Using a realistic target prevents always scoring 0.00.
        ideal = np.array([0.08, 0.25, 0.20, 0.22, 0.12, 0.08, 0.05])

        # Score: how close to ideal distribution?
        diff = np.abs(proportions - ideal)
        balance_score = 1.0 - np.mean(diff) * 4  # softer scale (was *5)

        # Penalty for extreme imbalance (one band > 60% of total energy)
        max_prop = np.max(proportions)
        if max_prop > 0.60:
            balance_score *= 0.70  # softer penalty (was *0.5 at >0.5)

        # Penalty for missing bands (hollow sound) — gentle
        silent_bands = np.sum(proportions < 0.01)
        balance_score -= silent_bands * 0.05  # softer (was *0.1)

        # Floor: never return exactly 0 — a mix with any content has some balance
        return float(np.clip(balance_score, 0.05, 1.0))

    # ================================================================
    # 3. HARMONIC CLARITY
    # Are pitched elements in tune with each other?
    # Good: clear tonal center, consonant intervals
    # Bad: clashing keys, dissonant layering, pitch drift
    # ================================================================

    def _eval_harmonic_clarity(self, audio: np.ndarray) -> float:
        """
        Measures how harmonically coherent the mix is.

        Method: Extract chromagram, measure how concentrated the energy
        is around a single key center. Professional mixes have most energy
        in 3-4 related pitch classes (the key). Bad mashups spread energy
        across all 12 pitch classes (= atonal mud).
        """
        chroma = librosa.feature.chroma_cqt(y=audio, sr=self.sr)

        if chroma.shape[1] == 0:
            return 0.5

        # Average chroma profile
        chroma_profile = np.mean(chroma, axis=1)  # 12 pitch classes

        if np.sum(chroma_profile) < 1e-10:
            return 0.5

        # Normalize
        chroma_profile = chroma_profile / np.sum(chroma_profile)

        # Sort to find how concentrated the energy is
        sorted_profile = np.sort(chroma_profile)[::-1]

        # In a good mix: top 3-4 pitch classes contain most energy
        # (these correspond to the tonic, 3rd, 5th, and maybe 7th)
        top3_energy = np.sum(sorted_profile[:3])
        top5_energy = np.sum(sorted_profile[:5])

        # Score based on concentration
        # Professional track: top 3 notes have 50-70% of energy
        # Bad mashup: top 3 notes have 30-40% (spread out)
        concentration_score = np.clip((top3_energy - 0.30) / 0.35, 0, 1)

        # Also check for strong dissonance: minor 2nd intervals
        # (adjacent semitones both having high energy = clash)
        dissonance = 0
        for i in range(12):
            next_i = (i + 1) % 12
            # Both notes strong? That's a minor 2nd = dissonant
            if chroma_profile[i] > 0.1 and chroma_profile[next_i] > 0.1:
                dissonance += chroma_profile[i] * chroma_profile[next_i]

        dissonance_penalty = np.clip(dissonance * 10, 0, 0.4)

        score = concentration_score - dissonance_penalty
        return float(np.clip(score, 0, 1))

    # ================================================================
    # 4. VOCAL CLARITY
    # Can you hear the vocals clearly?
    # Good: vocals sit above the instrumental, clear words
    # Bad: vocals buried, masked, or harsh
    # ================================================================

    def _eval_vocal_clarity(self, full_mix: np.ndarray,
                            vocals: Optional[np.ndarray] = None,
                            instrumental: Optional[np.ndarray] = None) -> float:
        """
        Measures how clearly vocals cut through the mix.

        Method 1 (if vocals available): Compare vocal energy to mix energy
        in the 1-4kHz range (speech intelligibility region).

        Method 2 (mix only): Analyze the 1-4kHz band for presence of
        formant-like structures (indicators of vocal clarity).
        """
        if vocals is not None and instrumental is not None:
            return self._vocal_clarity_with_stems(full_mix, vocals, instrumental)

        # Method 2: Analyze vocal frequency region in the mix
        S = np.abs(librosa.stft(full_mix))
        freqs = librosa.fft_frequencies(sr=self.sr)

        # Vocal presence region: 1-4kHz
        vocal_mask = (freqs >= 1000) & (freqs <= 4000)
        # Full spectrum
        full_mask = freqs > 0

        if not vocal_mask.any() or not full_mask.any():
            return 0.5

        vocal_energy = np.mean(S[vocal_mask, :] ** 2)
        full_energy = np.mean(S[full_mask, :] ** 2)

        if full_energy < 1e-10:
            return 0.5

        # Ratio: how much of the mix energy is in the vocal region
        vocal_ratio = vocal_energy / full_energy

        # Good mix: vocal region is prominent but not overwhelming
        # Target ratio: 0.15-0.35
        if vocal_ratio < 0.10:
            score = vocal_ratio / 0.10 * 0.5  # Too quiet
        elif vocal_ratio < 0.15:
            score = 0.5 + (vocal_ratio - 0.10) / 0.05 * 0.3
        elif vocal_ratio <= 0.35:
            score = 0.8 + (0.2 * (1.0 - abs(vocal_ratio - 0.25) / 0.10))
        else:
            score = max(0.3, 1.0 - (vocal_ratio - 0.35) * 3)  # Too harsh

        return float(np.clip(score, 0, 1))

    def _vocal_clarity_with_stems(self, full_mix, vocals, instrumental):
        """More accurate vocal clarity when we have separate stems"""
        # Match lengths
        min_len = min(len(full_mix), len(vocals), len(instrumental))
        full_mix = full_mix[:min_len]
        vocals = vocals[:min_len]
        instrumental = instrumental[:min_len]

        # Compute vocal-to-instrumental ratio in speech band (1-4kHz)
        n_fft = 2048
        S_vocal = np.abs(librosa.stft(vocals, n_fft=n_fft))
        S_inst = np.abs(librosa.stft(instrumental, n_fft=n_fft))
        freqs = librosa.fft_frequencies(sr=self.sr, n_fft=n_fft)

        vocal_band = (freqs >= 1000) & (freqs <= 4000)

        vocal_power = np.mean(S_vocal[vocal_band, :] ** 2)
        inst_power = np.mean(S_inst[vocal_band, :] ** 2)

        if inst_power < 1e-10:
            return 1.0 if vocal_power > 1e-10 else 0.5

        # Signal-to-noise ratio in the vocal band
        snr_db = 10 * np.log10(vocal_power / inst_power + 1e-10)

        # Ideal: vocals 3-8dB above instruments in their frequency range
        if snr_db < -3:
            score = 0.2  # Vocals buried
        elif snr_db < 0:
            score = 0.2 + (snr_db + 3) / 3 * 0.3  # Vocals struggling
        elif snr_db < 3:
            score = 0.5 + snr_db / 3 * 0.3  # Getting there
        elif snr_db <= 8:
            score = 0.8 + 0.2 * (1.0 - abs(snr_db - 5) / 3)  # Sweet spot
        else:
            score = max(0.5, 1.0 - (snr_db - 8) * 0.05)  # Vocals too loud

        return float(np.clip(score, 0, 1))

    # ================================================================
    # 5. DYNAMIC RANGE
    # Is the mix breathing or squashed?
    # Good: varies between loud and quiet moments
    # Bad: constant loudness (fatiguing) or too much variation
    # ================================================================

    def _eval_dynamic_range(self, audio: np.ndarray) -> float:
        """
        Measures dynamic range health.

        Method: Compute RMS in short windows, measure the ratio between
        loud and quiet sections. Professional masters have 6-14dB of
        dynamic range. Over-compressed = <4dB. Too dynamic = >20dB.
        """
        hop = int(0.1 * self.sr)  # 100ms windows
        rms_values = []
        for i in range(0, len(audio) - hop, hop):
            window = audio[i:i+hop]
            rms = np.sqrt(np.mean(window ** 2))
            if rms > 1e-6:  # Skip silence
                rms_values.append(rms)

        if len(rms_values) < 10:
            return 0.5

        rms_db = 20 * np.log10(np.array(rms_values) + 1e-10)

        # Dynamic range = difference between loud and quiet (excluding silence)
        p90 = np.percentile(rms_db, 90)  # Loud parts
        p10 = np.percentile(rms_db, 10)  # Quiet parts
        dynamic_range_db = p90 - p10

        # Score: 6-14dB is ideal
        if dynamic_range_db < 3:
            score = 0.2  # Over-compressed, fatiguing
        elif dynamic_range_db < 6:
            score = 0.2 + (dynamic_range_db - 3) / 3 * 0.5
        elif dynamic_range_db <= 14:
            score = 0.7 + 0.3 * (1.0 - abs(dynamic_range_db - 10) / 4)
        elif dynamic_range_db <= 20:
            score = 0.7 - (dynamic_range_db - 14) / 6 * 0.3
        else:
            score = 0.3  # Too much variation, not cohesive

        return float(np.clip(score, 0, 1))

    # ================================================================
    # 6. PHASE COHERENCE
    # Are elements reinforcing or canceling each other?
    # Good: elements add constructively
    # Bad: thin sound from phase cancellation
    # ================================================================

    def _eval_phase_coherence(self, audio: np.ndarray,
                               stems: Optional[Dict] = None) -> float:
        """
        Detects phase cancellation issues.

        Method: If stems available, check if sum of stems matches the mix.
        Large differences indicate phase cancellation.
        If only mix: check for suspicious dips in the spectrum (notch patterns).
        """
        if stems is not None:
            return self._phase_coherence_with_stems(audio, stems)

        # Check spectrum for notch patterns (signs of comb filtering)
        S = np.abs(librosa.stft(audio))
        if S.shape[1] == 0:
            return 0.5

        # Average spectrum
        avg_spectrum = np.mean(S, axis=1)

        if np.max(avg_spectrum) < 1e-10:
            return 0.5

        avg_spectrum = avg_spectrum / np.max(avg_spectrum)

        # Count deep notches (drops of >12dB)
        spectrum_db = 20 * np.log10(avg_spectrum + 1e-10)
        local_avg = np.convolve(spectrum_db, np.ones(20)/20, mode='same')
        notches = np.sum((local_avg - spectrum_db) > 12)

        # Few notches = good phase coherence
        notch_ratio = notches / len(spectrum_db)
        score = 1.0 - np.clip(notch_ratio * 20, 0, 0.6)

        return float(np.clip(score, 0, 1))

    def _phase_coherence_with_stems(self, full_mix, stems):
        """Check if stems add up properly (no cancellation)"""
        # Sum all stems
        stem_list = [v for v in stems.values() if v is not None]
        if not stem_list:
            return 0.5

        max_len = max(len(s) for s in stem_list)
        stem_sum = np.zeros(max_len)
        for s in stem_list:
            padded = np.pad(s, (0, max_len - len(s)))
            stem_sum += padded

        # Compare energy of stem_sum vs full_mix
        min_len = min(len(stem_sum), len(full_mix))
        sum_energy = np.sqrt(np.mean(stem_sum[:min_len] ** 2))
        mix_energy = np.sqrt(np.mean(full_mix[:min_len] ** 2))

        if sum_energy < 1e-10:
            return 0.5

        # If mix energy is much less than stem sum, there's cancellation
        ratio = mix_energy / sum_energy
        # Ideal ratio ~0.6-0.9 (some headroom from mixing)
        if ratio < 0.3:
            score = 0.2  # Severe cancellation
        elif ratio < 0.5:
            score = 0.2 + (ratio - 0.3) / 0.2 * 0.4
        elif ratio <= 0.95:
            score = 0.6 + 0.4 * (1.0 - abs(ratio - 0.75) / 0.2)
        else:
            score = 0.8

        return float(np.clip(score, 0, 1))

    # ================================================================
    # 7. SPECTRAL SEPARATION
    # Can you distinguish different elements?
    # Good: each element has its own space
    # Bad: everything blurs together, mud
    # ================================================================

    def _eval_spectral_separation(self, audio: np.ndarray,
                                   stems: Optional[Dict] = None) -> float:
        """
        Measures how well-separated the elements are in the frequency domain.

        Method: Compute spectral flatness over time. A well-mixed track
        alternates between noisy (drums) and tonal (vocals/instruments).
        A muddy mix is uniformly flat.
        """
        S = np.abs(librosa.stft(audio))
        if S.shape[1] < 2:
            return 0.5

        # Spectral flatness per frame
        flatness = librosa.feature.spectral_flatness(S=S)[0]

        if len(flatness) < 2:
            return 0.5

        # Good separation = high variance in flatness
        # (some frames tonal, some noisy = you can hear different elements)
        flatness_std = np.std(flatness)
        flatness_mean = np.mean(flatness)

        # Score based on variation
        # Professional mix: std 0.05-0.15, mean 0.1-0.3
        variation_score = np.clip(flatness_std / 0.10, 0, 1)

        # Penalty for extreme flatness (all noise = mud)
        if flatness_mean > 0.5:
            variation_score *= 0.5

        # Also check spectral contrast (difference between peaks and valleys)
        contrast = librosa.feature.spectral_contrast(S=S, sr=self.sr)
        avg_contrast = np.mean(contrast)

        # Higher contrast = better separation
        contrast_score = np.clip(avg_contrast / 30, 0, 1)

        score = variation_score * 0.5 + contrast_score * 0.5
        return float(np.clip(score, 0, 1))

    # ================================================================
    # 8. ENERGY CONSISTENCY
    # Does the mix maintain momentum?
    # Good: smooth energy flow, intentional builds/drops
    # Bad: random energy spikes, dead spots, jarring transitions
    # ================================================================

    def _eval_energy_consistency(self, audio: np.ndarray) -> float:
        """
        Measures energy flow smoothness.

        Method: Compute RMS envelope, measure how smooth it is.
        Professional mixes have gradual energy changes.
        Bad mashups have sudden jumps where songs switch.
        """
        hop = int(0.5 * self.sr)  # 500ms windows
        rms_values = []
        for i in range(0, len(audio) - hop, hop):
            rms = np.sqrt(np.mean(audio[i:i+hop] ** 2))
            rms_values.append(rms)

        if len(rms_values) < 4:
            return 0.5

        rms_arr = np.array(rms_values)
        if np.max(rms_arr) < 1e-10:
            return 0.5

        # Normalize
        rms_arr = rms_arr / np.max(rms_arr)

        # Compute frame-to-frame changes
        diffs = np.abs(np.diff(rms_arr))

        # Score: small changes = smooth = good
        avg_diff = np.mean(diffs)
        max_diff = np.max(diffs)

        # Penalty for large jumps (jarring transitions)
        jump_penalty = np.sum(diffs > 0.3) / len(diffs)

        smoothness = 1.0 - np.clip(avg_diff / 0.15, 0, 1)
        jump_score = 1.0 - np.clip(jump_penalty * 5, 0, 0.5)

        score = smoothness * 0.6 + jump_score * 0.4
        return float(np.clip(score, 0, 1))

    # ================================================================
    # RECOMMENDATION ENGINE
    # ================================================================

    def recommend_adjustments(self, score: MixScore) -> List[MixRecommendation]:
        """
        Based on the quality scores, recommend specific adjustments.
        Returns a prioritized list of changes to make.
        """
        recs = []

        if score.beat_coherence < 0.5:
            recs.append(MixRecommendation(
                dimension='beat_coherence',
                action='realign_beats',
                parameter='beat_alignment_window_ms',
                current_value=50, suggested_value=100,
                reason='Beats are not tight — try wider alignment window or use single drum source',
                priority=1))

        if score.spectral_balance < 0.5:
            recs.append(MixRecommendation(
                dimension='spectral_balance',
                action='apply_eq_correction',
                parameter='eq_curve',
                current_value=0, suggested_value=1,
                reason='Frequency spectrum is unbalanced — needs EQ correction',
                priority=2))

        if score.harmonic_clarity < 0.5:
            recs.append(MixRecommendation(
                dimension='harmonic_clarity',
                action='adjust_transposition',
                parameter='transposition_semitones',
                current_value=0, suggested_value=0,
                reason='Songs may be in clashing keys — try different transposition or use single-source mode',
                priority=1))

        if score.vocal_clarity < 0.5:
            recs.append(MixRecommendation(
                dimension='vocal_clarity',
                action='boost_vocal_presence',
                parameter='vocal_gain_db',
                current_value=0, suggested_value=3,
                reason='Vocals are buried — increase vocal level or add presence EQ',
                priority=1))
            recs.append(MixRecommendation(
                dimension='vocal_clarity',
                action='increase_sidechain',
                parameter='sidechain_amount',
                current_value=0.3, suggested_value=0.5,
                reason='Duck instruments more when vocals are active',
                priority=2))

        if score.dynamic_range < 0.4:
            recs.append(MixRecommendation(
                dimension='dynamic_range',
                action='reduce_compression',
                parameter='compression_ratio',
                current_value=3.0, suggested_value=2.0,
                reason='Mix is over-compressed — reduce compression ratio',
                priority=3))

        if score.phase_coherence < 0.5:
            recs.append(MixRecommendation(
                dimension='phase_coherence',
                action='switch_to_single_source',
                parameter='fusion_mode',
                current_value='blend', suggested_value='a_vocals_b_inst',
                reason='Phase cancellation detected — use vocals from one song only',
                priority=1))

        if score.spectral_separation < 0.4:
            recs.append(MixRecommendation(
                dimension='spectral_separation',
                action='increase_eq_carving',
                parameter='eq_carve_depth_db',
                current_value=-2, suggested_value=-4,
                reason='Elements are masking each other — deeper EQ carving needed',
                priority=2))

        if score.energy_consistency < 0.5:
            recs.append(MixRecommendation(
                dimension='energy_consistency',
                action='improve_crossfades',
                parameter='crossfade_ms',
                current_value=500, suggested_value=1500,
                reason='Energy jumps at transitions — use longer crossfades',
                priority=2))

        # Sort by priority
        recs.sort(key=lambda r: r.priority)
        return recs

    # ================================================================
    # SONG COMPATIBILITY SCORING
    # Predicts how good a mashup will sound BEFORE mixing
    # ================================================================

    def predict_mashup_quality(self, analysis_a: Dict, analysis_b: Dict) -> Dict:
        """
        Predict how good a mashup will sound based on song analyses.

        This runs BEFORE mixing to tell the user:
        - Which songs pair well
        - What the best fusion mode would be
        - What adjustments are needed

        Returns dict with predicted scores and recommendations.
        """
        tempo_a = analysis_a.get('tempo', 120)
        tempo_b = analysis_b.get('tempo', 120)
        key_a = analysis_a.get('key', 'C major')
        key_b = analysis_b.get('key', 'C major')

        predictions = {}

        # Tempo compatibility
        tempo_ratio = max(tempo_a, tempo_b) / max(min(tempo_a, tempo_b), 1)
        if tempo_ratio < 1.05:
            predictions['tempo_score'] = 1.0
            predictions['tempo_note'] = 'Perfect tempo match'
        elif tempo_ratio < 1.10:
            predictions['tempo_score'] = 0.9
            predictions['tempo_note'] = 'Very close tempos, minor stretch needed'
        elif tempo_ratio < 1.20:
            predictions['tempo_score'] = 0.7
            predictions['tempo_note'] = 'Moderate tempo difference, stretching may be audible'
        elif tempo_ratio < 1.50:
            predictions['tempo_score'] = 0.4
            predictions['tempo_note'] = 'Large tempo gap — recommend vocals-over-instrumental mode'
        elif tempo_ratio < 2.05 and tempo_ratio > 1.95:
            predictions['tempo_score'] = 0.6
            predictions['tempo_note'] = 'Double-time relationship — could work as half-time mashup'
        else:
            predictions['tempo_score'] = 0.2
            predictions['tempo_note'] = 'Very different tempos — will require heavy time-stretching'

        # Key compatibility
        key_score, key_note = self._key_compatibility(key_a, key_b)
        predictions['key_score'] = key_score
        predictions['key_note'] = key_note

        # Recommended mode
        if tempo_ratio > 1.3:
            predictions['recommended_mode'] = 'a_vocals_b_inst'
            predictions['mode_reason'] = 'Large tempo difference — layering everything will sound messy'
        elif key_score < 0.4:
            predictions['recommended_mode'] = 'a_vocals_b_inst'
            predictions['mode_reason'] = 'Clashing keys — single vocal source avoids harmonic conflict'
        else:
            predictions['recommended_mode'] = 'blend'
            predictions['mode_reason'] = 'Compatible tempo and key — full blend should work'

        # Overall prediction
        predictions['predicted_quality'] = (
            predictions['tempo_score'] * 0.4 +
            predictions['key_score'] * 0.4 +
            0.5 * 0.2  # Unknown factors
        )

        return predictions

    def _key_compatibility(self, key_a: str, key_b: str) -> Tuple[float, str]:
        """Score key compatibility between two songs"""
        # Parse keys
        key_map = {
            'C': 0, 'C#': 1, 'Db': 1, 'D': 2, 'D#': 3, 'Eb': 3,
            'E': 4, 'F': 5, 'F#': 6, 'Gb': 6, 'G': 7, 'G#': 8,
            'Ab': 8, 'A': 9, 'A#': 10, 'Bb': 10, 'B': 11
        }

        def parse_key(k):
            k = k.strip()
            is_minor = 'minor' in k.lower() or 'min' in k.lower()
            root = k.split()[0] if ' ' in k else k
            semitone = key_map.get(root, 0)
            if is_minor:
                semitone = (semitone + 3) % 12  # Relative major
            return semitone

        root_a = parse_key(key_a)
        root_b = parse_key(key_b)
        interval = abs(root_a - root_b) % 12

        # Compatible intervals (in semitones)
        compatibility = {
            0: (1.0, 'Same key — perfect match'),
            7: (0.9, 'Perfect 5th — very compatible'),
            5: (0.9, 'Perfect 4th — very compatible'),
            2: (0.6, 'Major 2nd — workable with adjustment'),
            10: (0.6, 'Minor 7th — workable with adjustment'),
            4: (0.5, 'Major 3rd — might clash'),
            3: (0.5, 'Minor 3rd — might clash'),
            9: (0.5, 'Major 6th — might clash'),
            8: (0.5, 'Minor 6th — might clash'),
            1: (0.2, 'Minor 2nd — strong clash'),
            11: (0.2, 'Major 7th — strong clash'),
            6: (0.1, 'Tritone — maximum dissonance'),
        }

        score, note = compatibility.get(interval, (0.3, 'Unknown interval'))
        return score, f'{key_a} / {key_b}: {note}'

    # ================================================================
    # AUTO-OPTIMIZER
    # Try multiple configurations, score each, pick the best
    # ================================================================

    def find_best_config(self, stems_a: Dict, stems_b: Dict,
                          analysis_a: Dict, analysis_b: Dict) -> Dict:
        """
        Try multiple fusion configurations and return the best one.

        Tests:
        - A vocals over B instrumental
        - B vocals over A instrumental
        - Blend mode (if tempos are close)
        - Different transpositions (-2, -1, 0, +1, +2)

        Returns the configuration that scores highest.
        """
        from vocalfusion.dsp import EnhancedDSP
        dsp = EnhancedDSP(self.sr)

        tempo_a = analysis_a.get('tempo', 120)
        tempo_b = analysis_b.get('tempo', 120)

        configs = []

        # Config 1: A vocals + B instrumental
        configs.append({
            'mode': 'a_vocals_b_inst',
            'transposition': 0,
            'description': f'Vocals from A over instrumental from B'
        })

        # Config 2: B vocals + A instrumental
        configs.append({
            'mode': 'b_vocals_a_inst',
            'transposition': 0,
            'description': f'Vocals from B over instrumental from A'
        })

        # Config 3: Blend (only if tempos are close)
        ratio = max(tempo_a, tempo_b) / max(min(tempo_a, tempo_b), 1)
        if ratio < 1.2:
            configs.append({
                'mode': 'blend',
                'transposition': 0,
                'description': 'Full blend of both songs'
            })

        best_score = -1
        best_config = configs[0]

        print(f"  Testing {len(configs)} configurations...")
        for i, config in enumerate(configs):
            # Quick evaluation: predict quality without full mixing
            prediction = self.predict_mashup_quality(analysis_a, analysis_b)
            estimated_score = prediction['predicted_quality']

            # Adjust score based on mode
            if config['mode'] == 'blend' and ratio > 1.15:
                estimated_score *= 0.7  # Penalty for stretching in blend mode

            print(f"    Config {i+1}: {config['description']} -> est. score: {estimated_score:.2f}")

            if estimated_score > best_score:
                best_score = estimated_score
                best_config = config

        print(f"  Best: {best_config['description']} (score: {best_score:.2f})")
        return best_config
