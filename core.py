"""
VocalFusion - Complete AI-Powered Vocal Fusion System
Core Architecture and Initial Implementation
"""

import os
import json
import numpy as np
import librosa
import soundfile as sf
import parselmouth  # More accurate pitch tracking
from dataclasses import dataclass, asdict, field
from enum import Enum
from typing import Dict, List, Optional, Tuple, Any
from pathlib import Path
import hashlib
import pickle
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# ============================================================================
# CORE DATA MODELS
# ============================================================================

class VoiceType(str, Enum):
    SOPRANO = "soprano"        # C4-C6
    MEZZO_SOPRANO = "mezzo-soprano"  # A3-A5
    ALTO = "alto"              # F3-F5
    TENOR = "tenor"            # C3-C5
    BARITONE = "baritone"      # G2-G4
    BASS = "bass"              # E2-E4
    UNKNOWN = "unknown"

class VocalQuality(str, Enum):
    BREATHY = "breathy"
    BELTING = "belting"
    FALSETTO = "falsetto"
    VIBRATO = "vibrato"
    STRAIGHT = "straight"
    GROWL = "growl"

class SongSection(str, Enum):
    INTRO = "intro"
    VERSE = "verse"
    PRE_CHORUS = "pre_chorus"
    CHORUS = "chorus"
    BRIDGE = "bridge"
    OUTRO = "outro"
    INSTRUMENTAL = "instrumental"
    SOLO = "solo"

@dataclass
class PitchPoint:
    time: float
    frequency: float
    confidence: float
    midi_note: int
    note_name: str
    is_voiced: bool

@dataclass
class VocalPhrase:
    id: str
    start_time: float
    end_time: float
    duration: float
    pitch_mean: float
    pitch_std: float
    energy_mean: float
    energy_std: float
    words: Optional[List[str]] = None
    emotion: Optional[str] = None
    quality: Optional[VocalQuality] = None

@dataclass
class NoteEvent:
    start_time: float
    end_time: float
    pitch_midi: float
    pitch_hz: float
    pitch_name: str
    duration: float
    velocity: float
    confidence: float

@dataclass
class Formant:
    f1: float  # 300-900 Hz: vowel openness
    f2: float  # 800-2800 Hz: vowel frontness
    f3: float  # 2000-3500 Hz: voice quality
    f4: float  # 3000-5000 Hz: voice brightness

@dataclass
class TimbreProfile:
    mfcc_mean: List[float]  # 13 MFCC coefficients
    mfcc_std: List[float]
    spectral_centroid_mean: float
    spectral_centroid_std: float
    spectral_rolloff_mean: float
    spectral_rolloff_std: float
    zero_crossing_rate_mean: float
    zero_crossing_rate_std: float
    brightness: float  # 0-1
    warmth: float  # 0-1
    presence: float  # 0-1
    air: float  # 0-1

@dataclass
class VocalAnalysis:
    """Complete vocal analysis results"""
    # Basic info
    voice_type: VoiceType
    gender: Optional[str] = None
    age_estimate: Optional[str] = None
    
    # Pitch analysis
    pitch_contour: List[PitchPoint] = field(default_factory=list)
    notes: List[NoteEvent] = field(default_factory=list)
    vibrato_events: List[Dict] = field(default_factory=list)
    pitch_bends: List[Dict] = field(default_factory=list)
    
    # Range analysis
    range_low_hz: float = 0.0
    range_high_hz: float = 0.0
    tessitura_low_hz: float = 0.0
    tessitura_high_hz: float = 0.0
    comfort_range_low_hz: float = 0.0
    comfort_range_high_hz: float = 0.0
    
    # Timing analysis
    phrases: List[VocalPhrase] = field(default_factory=list)
    silences: List[Dict] = field(default_factory=list)
    onsets: List[float] = field(default_factory=list)
    rhythm_pattern: List[float] = field(default_factory=list)
    bpm: float = 0.0
    groove: Dict[str, float] = field(default_factory=dict)
    
    # Spectral analysis
    timbre: TimbreProfile = field(default_factory=lambda: TimbreProfile([], [], 0, 0, 0, 0, 0, 0, 0, 0, 0, 0))
    formants_over_time: List[Formant] = field(default_factory=list)
    harmonics_to_noise_ratio: float = 0.0
    breathiness_index: float = 0.0
    
    # Dynamics analysis
    loudness_curve: List[float] = field(default_factory=list)
    intensity_zones: Dict[str, List[float]] = field(default_factory=dict)
    dynamic_range_db: float = 0.0
    compression_ratio: float = 0.0
    
    # Emotional analysis
    emotional_arc: List[Dict] = field(default_factory=list)
    energy_curve: List[float] = field(default_factory=list)
    tension_curve: List[float] = field(default_factory=list)
    
    # Statistics
    pitch_accuracy: float = 0.0
    timing_accuracy: float = 0.0
    consistency_score: float = 0.0
    expressiveness_score: float = 0.0
    
    def to_dict(self):
        """Convert to serializable dictionary"""
        result = asdict(self)
        # Handle non-serializable fields
        result['pitch_contour'] = [asdict(p) for p in self.pitch_contour]
        result['notes'] = [asdict(n) for n in self.notes]
        result['phrases'] = [asdict(p) for p in self.phrases]
        result['timbre'] = asdict(self.timbre)
        result['formants_over_time'] = [asdict(f) for f in self.formants_over_time]
        return result

@dataclass
class SongAnalysis:
    """Complete song analysis"""
    song_id: str
    title: Optional[str] = None
    artist: Optional[str] = None
    
    # Technical metadata
    duration: float = 0.0
    sample_rate: int = 44100
    channels: int = 2
    bit_depth: int = 16
    file_size: int = 0
    md5_hash: str = ""
    
    # Musical analysis
    key: str = "C major"
    tempo: float = 120.0
    time_signature: str = "4/4"
    mode: str = "major"
    scale_type: str = "diatonic"
    
    # Structure analysis
    sections: List[Dict] = field(default_factory=list)
    section_map: List[Tuple[float, float, str]] = field(default_factory=list)
    chord_progression: List[str] = field(default_factory=list)
    harmonic_complexity: float = 0.0
    
    # Vocal analysis (if present)
    vocals: Optional[VocalAnalysis] = None
    
    # Instrumental analysis
    instruments_detected: List[str] = field(default_factory=list)
    frequency_spectrum: Dict[str, float] = field(default_factory=dict)
    stereo_width: float = 0.0
    dynamics: Dict[str, float] = field(default_factory=dict)
    
    # Energy analysis
    energy_curve: List[float] = field(default_factory=list)
    energy_zones: Dict[str, List[float]] = field(default_factory=dict)
    climax_point: float = 0.0
    
    # Compatibility features
    features: Dict[str, Any] = field(default_factory=dict)

@dataclass
class CompatibilityScore:
    """Detailed compatibility analysis"""
    song_a_id: str
    song_b_id: str
    
    # Individual compatibility scores (0-1, where 1 is perfect)
    key_compatibility: float = 0.0
    tempo_compatibility: float = 0.0
    range_compatibility: float = 0.0
    timbre_compatibility: float = 0.0
    structure_compatibility: float = 0.0
    emotional_compatibility: float = 0.0
    harmonic_compatibility: float = 0.0
    
    # Vocal-specific compatibility
    vocal_blend_score: float = 0.0
    phrase_alignment_score: float = 0.0
    frequency_separation_score: float = 0.0
    
    # Overall scores
    overall_score: float = 0.0
    difficulty_score: float = 0.0  # 0=easy, 1=difficult
    quality_potential: float = 0.0  # Potential quality of fusion
    
    # Technical recommendations
    recommended_transposition_semitones: int = 0
    recommended_tempo_adjustment_ratio: float = 1.0
    recommended_key: str = "C major"
    
    # Arrangement strategies
    arrangement_strategies: List[str] = field(default_factory=list)
    suggested_structure: List[Dict] = field(default_factory=list)
    
    # Challenges and opportunities
    challenges: List[str] = field(default_factory=list)
    opportunities: List[str] = field(default_factory=list)
    
    # Detailed analysis
    frequency_collision_zones: List[Tuple[float, float]] = field(default_factory=list)
    phrase_overlap_analysis: Dict[str, Any] = field(default_factory=dict)
    harmonic_tension_points: List[float] = field(default_factory=list)

# ============================================================================
# CORE ANALYSIS ENGINE
# ============================================================================

class PitchAnalyzer:
    """Advanced pitch analysis using multiple methods for accuracy"""
    
    def __init__(self, sample_rate=44100):
        self.sample_rate = sample_rate
        
    def analyze(self, audio: np.ndarray, method: str = "parselmouth") -> List[PitchPoint]:
        """
        Extract pitch contour with high accuracy
        Methods: 'parselmouth', 'crepe', 'pyin', 'librosa'
        """
        if method == "parselmouth":
            return self._analyze_parselmouth(audio)
        elif method == "crepe":
            return self._analyze_crepe(audio)
        elif method == "pyin":
            return self._analyze_pyin(audio)
        else:
            return self._analyze_librosa(audio)
    
    def _analyze_parselmouth(self, audio: np.ndarray) -> List[PitchPoint]:
        """Use Parselmouth (Praat) for most accurate pitch tracking"""
        # Convert to mono if needed
        if audio.ndim > 1:
            audio = librosa.to_mono(audio)
        
        # Create Praat sound object
        sound = parselmouth.Sound(audio, self.sample_rate)
        
        # Extract pitch with Praat's algorithm
        pitch = sound.to_pitch(
            time_step=0.01,      # 10ms steps
            pitch_floor=75.0,    # Minimum pitch (Hz)
            pitch_ceiling=600.0  # Maximum pitch (Hz)
        )
        
        # Get pitch values
        times = pitch.xs()
        frequencies = pitch.selected_array['frequency']
        strengths = pitch.selected_array['strength']
        
        # Convert to PitchPoint objects
        points = []
        for t, f, s in zip(times, frequencies, strengths):
            if f > 0 and s > 0.5:  # Valid voiced frame
                midi = librosa.hz_to_midi(f)
                note_name = librosa.midi_to_note(int(round(midi)))
                points.append(PitchPoint(
                    time=float(t),
                    frequency=float(f),
                    confidence=float(s),
                    midi_note=int(round(midi)),
                    note_name=note_name,
                    is_voiced=True
                ))
            else:
                points.append(PitchPoint(
                    time=float(t),
                    frequency=0.0,
                    confidence=0.0,
                    midi_note=0,
                    note_name="",
                    is_voiced=False
                ))
        
        return points
    
    def _analyze_crepe(self, audio: np.ndarray) -> List[PitchPoint]:
        """Use CREPE neural network for pitch tracking"""
        try:
            import crepe
            # CREPE expects 16kHz sample rate
            audio_16k = librosa.resample(audio, orig_sr=self.sample_rate, target_sr=16000)
            
            # Run CREPE
            time, frequency, confidence, _ = crepe.predict(
                audio_16k, 
                sr=16000,
                viterbi=True,
                step_size=10  # 10ms steps
            )
            
            points = []
            for t, f, c in zip(time, frequency, confidence):
                if f > 0 and c > 0.5:
                    midi = librosa.hz_to_midi(f)
                    note_name = librosa.midi_to_note(int(round(midi)))
                    points.append(PitchPoint(
                        time=float(t),
                        frequency=float(f),
                        confidence=float(c),
                        midi_note=int(round(midi)),
                        note_name=note_name,
                        is_voiced=True
                    ))
                else:
                    points.append(PitchPoint(
                        time=float(t),
                        frequency=0.0,
                        confidence=0.0,
                        midi_note=0,
                        note_name="",
                        is_voiced=False
                    ))
            
            return points
            
        except ImportError:
            print("CREPE not installed, falling back to Parselmouth")
            return self._analyze_parselmouth(audio)
    
    def _analyze_pyin(self, audio: np.ndarray) -> List[PitchPoint]:
        """Use PYIN algorithm (probabilistic YIN)"""
        # Convert to mono
        if audio.ndim > 1:
            audio = librosa.to_mono(audio)
        
        # Run PYIN
        f0, voiced_flag, voiced_probs = librosa.pyin(
            audio,
            fmin=librosa.note_to_hz('C2'),
            fmax=librosa.note_to_hz('C7'),
            sr=self.sample_rate,
            frame_length=2048,
            hop_length=512
        )
        
        times = librosa.frames_to_time(range(len(f0)), sr=self.sample_rate, hop_length=512)
        
        points = []
        for t, f, vf, vp in zip(times, f0, voiced_flag, voiced_probs):
            if vf and not np.isnan(f):
                midi = librosa.hz_to_midi(f)
                note_name = librosa.midi_to_note(int(round(midi)))
                points.append(PitchPoint(
                    time=float(t),
                    frequency=float(f),
                    confidence=float(vp),
                    midi_note=int(round(midi)),
                    note_name=note_name,
                    is_voiced=True
                ))
            else:
                points.append(PitchPoint(
                    time=float(t),
                    frequency=0.0,
                    confidence=0.0,
                    midi_note=0,
                    note_name="",
                    is_voiced=False
                ))
        
        return points
    
    def _analyze_librosa(self, audio: np.ndarray) -> List[PitchPoint]:
        """Fallback to librosa's pitch tracking"""
        if audio.ndim > 1:
            audio = librosa.to_mono(audio)
        
        # Use librosa's piptrack
        pitches, magnitudes = librosa.piptrack(
            y=audio,
            sr=self.sample_rate,
            fmin=75,
            fmax=600
        )
        
        points = []
        times = librosa.frames_to_time(range(pitches.shape[1]), sr=self.sample_rate)
        
        for i in range(pitches.shape[1]):
            index = magnitudes[:, i].argmax()
            pitch = pitches[index, i]
            
            if pitch > 0:
                midi = librosa.hz_to_midi(pitch)
                note_name = librosa.midi_to_note(int(round(midi)))
                confidence = magnitudes[index, i] / magnitudes[:, i].max()
                
                points.append(PitchPoint(
                    time=float(times[i]),
                    frequency=float(pitch),
                    confidence=float(confidence),
                    midi_note=int(round(midi)),
                    note_name=note_name,
                    is_voiced=True
                ))
            else:
                points.append(PitchPoint(
                    time=float(times[i]),
                    frequency=0.0,
                    confidence=0.0,
                    midi_note=0,
                    note_name="",
                    is_voiced=False
                ))
        
        return points
    
    def extract_notes(self, pitch_points: List[PitchPoint], min_note_duration: float = 0.05) -> List[NoteEvent]:
        """Convert pitch contour to discrete note events"""
        notes = []
        
        # Group consecutive same notes
        current_note = None
        start_time = 0
        start_index = 0
        
        for i, point in enumerate(pitch_points):
            if point.is_voiced:
                if current_note is None:
                    # Start new note
                    current_note = point.midi_note
                    start_time = point.time
                    start_index = i
                elif point.midi_note != current_note:
                    # Note change, finalize previous note
                    if point.time - start_time >= min_note_duration:
                        velocity = np.mean([p.confidence for p in pitch_points[start_index:i]])
                        notes.append(NoteEvent(
                            start_time=start_time,
                            end_time=point.time,
                            pitch_midi=float(current_note),
                            pitch_hz=float(librosa.midi_to_hz(current_note)),
                            pitch_name=librosa.midi_to_note(int(round(current_note))),
                            duration=point.time - start_time,
                            velocity=float(velocity),
                            confidence=float(velocity)
                        ))
                    
                    # Start new note
                    current_note = point.midi_note
                    start_time = point.time
                    start_index = i
            else:
                if current_note is not None:
                    # Silence detected, finalize note
                    if point.time - start_time >= min_note_duration:
                        velocity = np.mean([p.confidence for p in pitch_points[start_index:i]])
                        notes.append(NoteEvent(
                            start_time=start_time,
                            end_time=point.time,
                            pitch_midi=float(current_note),
                            pitch_hz=float(librosa.midi_to_hz(current_note)),
                            pitch_name=librosa.midi_to_note(int(round(current_note))),
                            duration=point.time - start_time,
                            velocity=float(velocity),
                            confidence=float(velocity)
                        ))
                    current_note = None
        
        # Handle final note
        if current_note is not None:
            velocity = np.mean([p.confidence for p in pitch_points[start_index:]])
            notes.append(NoteEvent(
                start_time=start_time,
                end_time=pitch_points[-1].time,
                pitch_midi=float(current_note),
                pitch_hz=float(librosa.midi_to_hz(current_note)),
                pitch_name=librosa.midi_to_note(int(round(current_note))),
                duration=pitch_points[-1].time - start_time,
                velocity=float(velocity),
                confidence=float(velocity)
            ))
        
        return notes

class TimingAnalyzer:
    """Advanced timing and phrase analysis"""
    
    def __init__(self, sample_rate=44100):
        self.sample_rate = sample_rate
    
    def detect_phrases(self, audio: np.ndarray, pitch_points: List[PitchPoint]) -> List[VocalPhrase]:
        """Detect vocal phrases with breath points and musical boundaries"""
        
        # Convert to mono if needed
        if audio.ndim > 1:
            audio = librosa.to_mono(audio)
        
        # 1. Energy-based phrase detection
        energy = librosa.feature.rms(y=audio, frame_length=2048, hop_length=512)[0]
        energy_times = librosa.frames_to_time(range(len(energy)), sr=self.sample_rate, hop_length=512)
        
        # 2. Onset detection for phrase starts
        onset_frames = librosa.onset.onset_detect(
            y=audio, 
            sr=self.sample_rate,
            units='frames',
            hop_length=512
        )
        onset_times = librosa.frames_to_time(onset_frames, sr=self.sample_rate, hop_length=512)
        
        # 3. Find voiced regions from pitch points
        voiced_times = [p.time for p in pitch_points if p.is_voiced]
        voiced_energies = [energy[np.argmin(np.abs(energy_times - t))] for t in voiced_times]
        
        # 4. Dynamic thresholding for phrase detection
        threshold = np.percentile(energy, 20)  # Start with 20th percentile
        
        phrases = []
        in_phrase = False
        phrase_start = 0
        phrase_points = []
        
        for i, (e, t) in enumerate(zip(energy, energy_times)):
            if e > threshold and not in_phrase:
                # Phrase start
                in_phrase = True
                phrase_start = t
                phrase_points = []
            elif e <= threshold and in_phrase:
                # Phrase end
                in_phrase = False
                phrase_duration = t - phrase_start
                
                if phrase_duration > 0.3:  # Minimum phrase length
                    # Calculate phrase statistics
                    phrase_pitch_points = [p for p in pitch_points 
                                          if phrase_start <= p.time <= t and p.is_voiced]
                    
                    if phrase_pitch_points:
                        pitch_values = [p.frequency for p in phrase_pitch_points]
                        pitch_mean = np.mean(pitch_values)
                        pitch_std = np.std(pitch_values)
                        
                        # Get energy for this phrase
                        phrase_energy_indices = (energy_times >= phrase_start) & (energy_times <= t)
                        phrase_energy = energy[phrase_energy_indices]
                        energy_mean = np.mean(phrase_energy) if len(phrase_energy) > 0 else 0
                        energy_std = np.std(phrase_energy) if len(phrase_energy) > 0 else 0
                        
                        # Detect vocal quality
                        quality = self._detect_vocal_quality(audio, phrase_start, t)
                        
                        phrases.append(VocalPhrase(
                            id=f"phrase_{len(phrases):03d}",
                            start_time=float(phrase_start),
                            end_time=float(t),
                            duration=float(phrase_duration),
                            pitch_mean=float(pitch_mean),
                            pitch_std=float(pitch_std),
                            energy_mean=float(energy_mean),
                            energy_std=float(energy_std),
                            quality=quality
                        ))
        
        return phrases
    
    def _detect_vocal_quality(self, audio: np.ndarray, start_time: float, end_time: float) -> VocalQuality:
        """Detect vocal quality for a phrase"""
        start_sample = int(start_time * self.sample_rate)
        end_sample = int(end_time * self.sample_rate)
        phrase_audio = audio[start_sample:end_sample]
        
        if len(phrase_audio) == 0:
            return VocalQuality.STRAIGHT
        
        # Calculate features
        spectral_centroid = librosa.feature.spectral_centroid(y=phrase_audio, sr=self.sample_rate)[0].mean()
        zero_crossing = librosa.feature.zero_crossing_rate(phrase_audio)[0].mean()
        harmonic_ratio = self._harmonic_noise_ratio(phrase_audio)
        
        # Classify based on features
        if harmonic_ratio < 0.3:
            return VocalQuality.BREATHY
        elif spectral_centroid > 2000 and zero_crossing > 0.1:
            return VocalQuality.BELTING
        elif spectral_centroid < 800:
            return VocalQuality.FALSETTO
        else:
            return VocalQuality.STRAIGHT
    
    def _harmonic_noise_ratio(self, audio: np.ndarray) -> float:
        """Calculate harmonic-to-noise ratio"""
        try:
            import scipy.signal as signal
            # Simple HNR approximation
            f, Pxx = signal.welch(audio, self.sample_rate)
            harmonic_power = Pxx[(f > 100) & (f < 1000)].sum()
            noise_power = Pxx[(f > 3000) & (f < 5000)].sum()
            
            if noise_power > 0:
                return harmonic_power / (harmonic_power + noise_power)
            else:
                return 1.0
        except:
            return 0.5
    
    def detect_silences(self, audio: np.ndarray, min_silence_duration: float = 0.1) -> List[Dict]:
        """Detect silent sections and breath points"""
        if audio.ndim > 1:
            audio = librosa.to_mono(audio)
        
        energy = librosa.feature.rms(y=audio, frame_length=2048, hop_length=512)[0]
        energy_times = librosa.frames_to_time(range(len(energy)), sr=self.sample_rate, hop_length=512)
        
        threshold = np.percentile(energy, 5)  # Very low for silence detection
        
        silences = []
        in_silence = False
        silence_start = 0
        
        for e, t in zip(energy, energy_times):
            if e <= threshold and not in_silence:
                # Silence start
                in_silence = True
                silence_start = t
            elif e > threshold and in_silence:
                # Silence end
                in_silence = False
                silence_duration = t - silence_start
                
                if silence_duration >= min_silence_duration:
                    silences.append({
                        'start_time': float(silence_start),
                        'end_time': float(t),
                        'duration': float(silence_duration),
                        'type': 'breath' if silence_duration < 0.8 else 'phrase_gap'
                    })
        
        return silences

class SpectralAnalyzer:
    """Advanced spectral and timbre analysis"""
    
    def __init__(self, sample_rate=44100):
        self.sample_rate = sample_rate
    
    def analyze_timbre(self, audio: np.ndarray) -> TimbreProfile:
        """Extract comprehensive timbre profile"""
        if audio.ndim > 1:
            audio = librosa.to_mono(audio)
        
        # MFCC features
        mfccs = librosa.feature.mfcc(
            y=audio, 
            sr=self.sample_rate,
            n_mfcc=13,
            n_fft=2048,
            hop_length=512
        )
        mfcc_mean = mfccs.mean(axis=1).tolist()
        mfcc_std = mfccs.std(axis=1).tolist()
        
        # Spectral centroid
        spectral_centroid = librosa.feature.spectral_centroid(
            y=audio, 
            sr=self.sample_rate,
            n_fft=2048,
            hop_length=512
        )[0]
        
        # Spectral rolloff
        spectral_rolloff = librosa.feature.spectral_rolloff(
            y=audio, 
            sr=self.sample_rate,
            n_fft=2048,
            hop_length=512
        )[0]
        
        # Zero crossing rate
        zcr = librosa.feature.zero_crossing_rate(audio, frame_length=2048, hop_length=512)[0]
        
        # Calculate derived timbre characteristics
        brightness = self._calculate_brightness(audio)
        warmth = self._calculate_warmth(audio)
        presence = self._calculate_presence(audio)
        air = self._calculate_air(audio)
        
        return TimbreProfile(
            mfcc_mean=mfcc_mean,
            mfcc_std=mfcc_std,
            spectral_centroid_mean=float(spectral_centroid.mean()),
            spectral_centroid_std=float(spectral_centroid.std()),
            spectral_rolloff_mean=float(spectral_rolloff.mean()),
            spectral_rolloff_std=float(spectral_rolloff.std()),
            zero_crossing_rate_mean=float(zcr.mean()),
            zero_crossing_rate_std=float(zcr.std()),
            brightness=float(brightness),
            warmth=float(warmth),
            presence=float(presence),
            air=float(air)
        )
    
    def _calculate_brightness(self, audio: np.ndarray) -> float:
        """Calculate brightness (high frequency content)"""
        fft = np.abs(np.fft.rfft(audio))
        freqs = np.fft.rfftfreq(len(audio), 1/self.sample_rate)
        
        # Energy above 3kHz
        high_freq_energy = fft[freqs > 3000].sum()
        total_energy = fft.sum()
        
        if total_energy > 0:
            return float(high_freq_energy / total_energy)
        return 0.0
    
    def _calculate_warmth(self, audio: np.ndarray) -> float:
        """Calculate warmth (low-mid frequency content)"""
        fft = np.abs(np.fft.rfft(audio))
        freqs = np.fft.rfftfreq(len(audio), 1/self.sample_rate)
        
        # Energy between 100-500Hz
        low_mid_energy = fft[(freqs > 100) & (freqs < 500)].sum()
        total_energy = fft.sum()
        
        if total_energy > 0:
            return float(low_mid_energy / total_energy)
        return 0.0
    
    def _calculate_presence(self, audio: np.ndarray) -> float:
        """Calculate presence (2-4kHz range)"""
        fft = np.abs(np.fft.rfft(audio))
        freqs = np.fft.rfftfreq(len(audio), 1/self.sample_rate)
        
        # Energy in presence range
        presence_energy = fft[(freqs > 2000) & (freqs < 4000)].sum()
        total_energy = fft.sum()
        
        if total_energy > 0:
            return float(presence_energy / total_energy)
        return 0.0
    
    def _calculate_air(self, audio: np.ndarray) -> float:
        """Calculate air (very high frequencies)"""
        fft = np.abs(np.fft.rfft(audio))
        freqs = np.fft.rfftfreq(len(audio), 1/self.sample_rate)
        
        # Energy above 10kHz
        air_energy = fft[freqs > 10000].sum()
        total_energy = fft.sum()
        
        if total_energy > 0:
            return float(air_energy / total_energy)
        return 0.0
    
    def extract_formants(self, audio: np.ndarray, frame_length_ms: float = 25) -> List[Formant]:
        """Extract formant frequencies over time"""
        if audio.ndim > 1:
            audio = librosa.to_mono(audio)
        
        frame_length = int(self.sample_rate * frame_length_ms / 1000)
        hop_length = frame_length // 2
        n_frames = len(audio) // hop_length
        
        formants = []
        
        for i in range(n_frames):
            start = i * hop_length
            end = min(start + frame_length, len(audio))
            frame = audio[start:end]
            
            if len(frame) < 100:
                continue
            
            # Apply window
            frame = frame * np.hanning(len(frame))
            
            # Calculate LPC coefficients for formant estimation
            try:
                import scipy.signal as signal
                
                # Pre-emphasis
                frame = signal.lfilter([1, -0.97], 1, frame)
                
                # Calculate LPC coefficients
                order = 10  # For 5 formants (2*order formants)
                a = librosa.lpc(frame, order=order)
                
                # Find roots
                rts = np.roots(a)
                rts = rts[np.imag(rts) >= 0]
                angz = np.arctan2(np.imag(rts), np.real(rts))
                
                # Convert to frequencies
                frqs = angz * (self.sample_rate / (2 * np.pi))
                
                # Sort and take first 4 formants
                indices = np.argsort(frqs)
                frqs = frqs[indices]
                
                if len(frqs) >= 4:
                    formants.append(Formant(
                        f1=float(frqs[0]),
                        f2=float(frqs[1]),
                        f3=float(frqs[2]),
                        f4=float(frqs[3])
                    ))
                else:
                    # Default values if not enough formants
                    formants.append(Formant(
                        f1=500.0,
                        f2=1500.0,
                        f3=2500.0,
                        f4=3500.0
                    ))
                    
            except Exception as e:
                # Fallback to default
                formants.append(Formant(
                    f1=500.0,
                    f2=1500.0,
                    f3=2500.0,
                    f4=3500.0
                ))
        
        return formants

class DynamicsAnalyzer:
    """Advanced dynamics and emotional analysis"""
    
    def __init__(self, sample_rate=44100):
        self.sample_rate = sample_rate
    
    def analyze_dynamics(self, audio: np.ndarray) -> Dict[str, Any]:
        """Analyze loudness and dynamics"""
        if audio.ndim > 1:
            audio = librosa.to_mono(audio)
        
        # Calculate LUFS-like loudness (simplified)
        window_size = int(0.4 * self.sample_rate)  # 400ms window
        hop_size = window_size // 2
        
        loudness = []
        times = []
        
        for i in range(0, len(audio) - window_size, hop_size):
            window = audio[i:i + window_size]
            # RMS power
            power = np.mean(window ** 2)
            # Convert to dB
            if power > 0:
                db = 10 * np.log10(power)
            else:
                db = -90  # Very quiet
            
            loudness.append(db)
            times.append(i / self.sample_rate)
        
        # Calculate dynamic range
        loudness_db = np.array(loudness)
        dynamic_range = loudness_db.max() - loudness_db.min()
        
        # Compression ratio estimation
        threshold = np.percentile(loudness_db, 50)  # Median as threshold
        above_threshold = loudness_db[loudness_db > threshold]
        if len(above_threshold) > 0:
            reduction = (above_threshold.max() - threshold) / 2
            compression_ratio = 1.0 + (reduction / 10)  # Simplified ratio
        else:
            compression_ratio = 1.0
        
        # Intensity zones
        zones = {
            'whisper': loudness_db[loudness_db < -40].tolist(),
            'soft': loudness_db[(loudness_db >= -40) & (loudness_db < -20)].tolist(),
            'medium': loudness_db[(loudness_db >= -20) & (loudness_db < -10)].tolist(),
            'loud': loudness_db[(loudness_db >= -10) & (loudness_db < -5)].tolist(),
            'very_loud': loudness_db[loudness_db >= -5].tolist()
        }
        
        return {
            'loudness_curve': loudness,
            'times': times,
            'dynamic_range_db': float(dynamic_range),
            'compression_ratio': float(compression_ratio),
            'intensity_zones': zones,
            'peak_loudness': float(loudness_db.max()),
            'average_loudness': float(loudness_db.mean())
        }
    
    def analyze_emotion(self, audio: np.ndarray, pitch_points: List[PitchPoint]) -> Dict[str, Any]:
        """Analyze emotional content"""
        if audio.ndim > 1:
            audio = librosa.to_mono(audio)
        
        # Extract features for emotion
        mfccs = librosa.feature.mfcc(y=audio, sr=self.sample_rate, n_mfcc=20)
        
        # Pitch statistics
        valid_pitches = [p.frequency for p in pitch_points if p.is_voiced]
        if valid_pitches:
            pitch_mean = np.mean(valid_pitches)
            pitch_std = np.std(valid_pitches)
            pitch_range = max(valid_pitches) - min(valid_pitches)
        else:
            pitch_mean = pitch_std = pitch_range = 0
        
        # Tempo and rhythm
        tempo, _ = librosa.beat.beat_track(y=audio, sr=self.sample_rate)
        
        # Energy
        energy = librosa.feature.rms(y=audio)[0]
        energy_mean = energy.mean()
        energy_std = energy.std()
        
        # Simple emotion estimation based on features
        emotion_score = {
            'happiness': min(1.0, (pitch_mean / 400) * (tempo / 180)),  # Higher pitch, faster tempo
            'sadness': min(1.0, (1 - (pitch_mean / 400)) * (1 - (tempo / 180))),  # Lower pitch, slower tempo
            'anger': min(1.0, energy_mean * 10),  # High energy
            'calm': min(1.0, (1 - energy_std) * (1 - pitch_std)),  # Consistent energy and pitch
            'excitement': min(1.0, energy_std * pitch_std * 10)  # Variability
        }
        
        # Dominant emotion
        dominant_emotion = max(emotion_score.items(), key=lambda x: x[1])
        
        # Emotional arc (simplified)
        window_size = len(audio) // 10  # 10 segments
        emotional_arc = []
        
        for i in range(0, len(audio), window_size):
            segment = audio[i:i + window_size]
            if len(segment) < 100:
                continue
                
            segment_energy = np.mean(segment ** 2)
            segment_pitch = [p.frequency for p in pitch_points 
                            if i/self.sample_rate <= p.time <= (i+window_size)/self.sample_rate 
                            and p.is_voiced]
            
            if segment_pitch:
                segment_pitch_mean = np.mean(segment_pitch)
                excitement = min(1.0, segment_energy * np.std(segment_pitch) * 100)
            else:
                segment_pitch_mean = 0
                excitement = 0
            
            emotional_arc.append({
                'time': i / self.sample_rate,
                'energy': float(segment_energy),
                'pitch': float(segment_pitch_mean),
                'excitement': float(excitement)
            })
        
        return {
            'emotion_scores': emotion_score,
            'dominant_emotion': dominant_emotion[0],
            'emotional_arc': emotional_arc,
            'tension_curve': [e['excitement'] for e in emotional_arc]
        }

class VoiceTypeClassifier:
    """Classify voice type based on range and timbre"""
    
    VOICE_RANGES = {
        VoiceType.SOPRANO: (261.63, 1046.50),    # C4-C6
        VoiceType.MEZZO_SOPRANO: (220.00, 880.00),  # A3-A5
        VoiceType.ALTO: (174.61, 698.46),        # F3-F5
        VoiceType.TENOR: (130.81, 523.25),       # C3-C5
        VoiceType.BARITONE: (98.00, 392.00),     # G2-G4
        VoiceType.BASS: (82.41, 329.63)          # E2-E4
    }
    
    def classify(self, pitch_points: List[PitchPoint], timbre: TimbreProfile) -> VoiceType:
        """Classify voice type based on range and timbre characteristics"""
        
        # Extract valid pitches
        valid_pitches = [p.frequency for p in pitch_points if p.is_voiced]
        
        if not valid_pitches:
            return VoiceType.UNKNOWN
        
        # Calculate range
        range_low = np.percentile(valid_pitches, 5)  # 5th percentile for low
        range_high = np.percentile(valid_pitches, 95)  # 95th percentile for high
        
        # Calculate tessitura (most used range)
        tessitura_low = np.percentile(valid_pitches, 25)  # 25th percentile
        tessitura_high = np.percentile(valid_pitches, 75)  # 75th percentile
        
        # Find closest matching voice type
        best_match = VoiceType.UNKNOWN
        best_score = float('inf')
        
        for voice_type, (ref_low, ref_high) in self.VOICE_RANGES.items():
            # Calculate range overlap score
            range_score = abs(range_low - ref_low) + abs(range_high - ref_high)
            
            # Adjust for tessitura
            tessitura_center = (tessitura_low + tessitura_high) / 2
            ref_center = (ref_low + ref_high) / 2
            tessitura_score = abs(tessitura_center - ref_center)
            
            # Consider timbre (higher spectral centroid = higher voice type)
            timbre_score = abs(timbre.spectral_centroid_mean - self._get_expected_centroid(voice_type))
            
            total_score = range_score * 0.5 + tessitura_score * 0.3 + timbre_score * 0.2
            
            if total_score < best_score:
                best_score = total_score
                best_match = voice_type
        
        return best_match
    
    def _get_expected_centroid(self, voice_type: VoiceType) -> float:
        """Get expected spectral centroid for voice type"""
        centroids = {
            VoiceType.SOPRANO: 2000,
            VoiceType.MEZZO_SOPRANO: 1800,
            VoiceType.ALTO: 1600,
            VoiceType.TENOR: 1400,
            VoiceType.BARITONE: 1200,
            VoiceType.BASS: 1000
        }
        return centroids.get(voice_type, 1500)

# ============================================================================
# MAIN VOCAL ANALYSIS ENGINE
# ============================================================================

class VocalAnalysisEngine:
    """Orchestrates all vocal analysis components"""
    
    def __init__(self, sample_rate=44100):
        self.sample_rate = sample_rate
        self.pitch_analyzer = PitchAnalyzer(sample_rate)
        self.timing_analyzer = TimingAnalyzer(sample_rate)
        self.spectral_analyzer = SpectralAnalyzer(sample_rate)
        self.dynamics_analyzer = DynamicsAnalyzer(sample_rate)
        self.voice_classifier = VoiceTypeClassifier()
    
    def analyze_vocals(self, audio: np.ndarray) -> VocalAnalysis:
        """Complete vocal analysis pipeline"""
        
        print("Starting vocal analysis...")
        
        # 1. Pitch analysis
        print("  - Analyzing pitch...")
        pitch_points = self.pitch_analyzer.analyze(audio, method="parselmouth")
        notes = self.pitch_analyzer.extract_notes(pitch_points)
        
        # 2. Timing and phrase analysis
        print("  - Analyzing timing and phrases...")
        phrases = self.timing_analyzer.detect_phrases(audio, pitch_points)
        silences = self.timing_analyzer.detect_silences(audio)
        
        # 3. Spectral analysis
        print("  - Analyzing timbre and formants...")
        timbre = self.spectral_analyzer.analyze_timbre(audio)
        formants = self.spectral_analyzer.extract_formants(audio)
        
        # 4. Dynamics analysis
        print("  - Analyzing dynamics...")
        dynamics = self.dynamics_analyzer.analyze_dynamics(audio)
        emotion = self.dynamics_analyzer.analyze_emotion(audio, pitch_points)
        
        # 5. Range analysis
        print("  - Calculating vocal range...")
        range_low, range_high, tessitura_low, tessitura_high = self._calculate_vocal_range(pitch_points)
        
        # 6. Voice type classification
        print("  - Classifying voice type...")
        voice_type = self.voice_classifier.classify(pitch_points, timbre)
        
        # 7. Onset detection
        onsets = librosa.onset.onset_detect(y=audio, sr=self.sample_rate, units='time')
        
        # 8. Rhythm analysis
        tempo, beats = librosa.beat.beat_track(y=audio, sr=self.sample_rate)
        rhythm_pattern = self._extract_rhythm_pattern(beats, self.sample_rate)
        
        # 9. Calculate statistics
        pitch_accuracy = self._calculate_pitch_accuracy(pitch_points)
        consistency_score = self._calculate_consistency(pitch_points, phrases)
        expressiveness_score = self._calculate_expressiveness(dynamics, emotion)
        
        print("Vocal analysis complete!")
        
        return VocalAnalysis(
            voice_type=voice_type,
            pitch_contour=pitch_points,
            notes=notes,
            phrases=phrases,
            silences=silences,
            range_low_hz=float(range_low),
            range_high_hz=float(range_high),
            tessitura_low_hz=float(tessitura_low),
            tessitura_high_hz=float(tessitura_high),
            comfort_range_low_hz=float(tessitura_low),
            comfort_range_high_hz=float(tessitura_high),
            timbre=timbre,
            formants_over_time=formants,
            harmonics_to_noise_ratio=0.8,  # Placeholder
            breathiness_index=0.2,  # Placeholder
            loudness_curve=dynamics['loudness_curve'],
            intensity_zones=dynamics['intensity_zones'],
            dynamic_range_db=float(dynamics['dynamic_range_db']),
            compression_ratio=float(dynamics['compression_ratio']),
            emotional_arc=emotion['emotional_arc'],
            energy_curve=[e['energy'] for e in emotion['emotional_arc']],
            tension_curve=emotion['tension_curve'],
            onsets=onsets.tolist(),
            rhythm_pattern=rhythm_pattern,
            bpm=float(tempo),
            groove={},  # Placeholder
            pitch_accuracy=float(pitch_accuracy),
            timing_accuracy=0.9,  # Placeholder
            consistency_score=float(consistency_score),
            expressiveness_score=float(expressiveness_score)
        )
    
    def _calculate_vocal_range(self, pitch_points: List[PitchPoint]) -> Tuple[float, float, float, float]:
        """Calculate vocal range and tessitura"""
        valid_pitches = [p.frequency for p in pitch_points if p.is_voiced]
        
        if not valid_pitches:
            return 100.0, 500.0, 200.0, 400.0
        
        # Vocal range (5th to 95th percentile)
        range_low = np.percentile(valid_pitches, 5)
        range_high = np.percentile(valid_pitches, 95)
        
        # Tessitura (25th to 75th percentile - most comfortable range)
        tessitura_low = np.percentile(valid_pitches, 25)
        tessitura_high = np.percentile(valid_pitches, 75)
        
        return range_low, range_high, tessitura_low, tessitura_high
    
    def _extract_rhythm_pattern(self, beats: np.ndarray, sample_rate: int) -> List[float]:
        """Extract rhythmic pattern from beat positions"""
        if len(beats) < 4:
            return [0.25, 0.25, 0.25, 0.25]  # Default pattern
        
        # Calculate inter-beat intervals
        beat_times = librosa.frames_to_time(beats, sr=sample_rate)
        intervals = np.diff(beat_times)
        
        # Normalize to get rhythmic pattern
        if intervals.mean() > 0:
            pattern = intervals[:4] / intervals.mean()  # First 4 beats
            return pattern.tolist()
        
        return [0.25, 0.25, 0.25, 0.25]
    
    def _calculate_pitch_accuracy(self, pitch_points: List[PitchPoint]) -> float:
        """Calculate pitch accuracy based on note stability"""
        valid_points = [p for p in pitch_points if p.is_voiced]
        
        if len(valid_points) < 10:
            return 0.5
        
        # Calculate how close pitches are to tempered scale
        pitches = np.array([p.frequency for p in valid_points])
        midi_pitches = librosa.hz_to_midi(pitches)
        rounded_midi = np.round(midi_pitches)
        
        # Calculate deviation from tempered scale
        deviations = np.abs(midi_pitches - rounded_midi)
        
        # Accuracy: 1 - average deviation (max deviation = 0.5 semitone)
        avg_deviation = deviations.mean()
        accuracy = max(0, 1 - (avg_deviation / 0.5))
        
        return accuracy
    
    def _calculate_consistency(self, pitch_points: List[PitchPoint], phrases: List[VocalPhrase]) -> float:
        """Calculate consistency across phrases"""
        if len(phrases) < 2:
            return 0.5
        
        # Calculate pitch consistency across phrases
        phrase_pitches = []
        for phrase in phrases:
            phrase_pitch_points = [p for p in pitch_points 
                                  if phrase.start_time <= p.time <= phrase.end_time 
                                  and p.is_voiced]
            if phrase_pitch_points:
                phrase_pitches.append(np.mean([p.frequency for p in phrase_pitch_points]))
        
        if len(phrase_pitches) < 2:
            return 0.5
        
        # Consistency = 1 - coefficient of variation
        cv = np.std(phrase_pitches) / np.mean(phrase_pitches)
        consistency = max(0, 1 - cv)
        
        return consistency
    
    def _calculate_expressiveness(self, dynamics: Dict, emotion: Dict) -> float:
        """Calculate expressiveness score"""
        # Combine dynamics range and emotional variation
        dynamic_range_score = min(1.0, dynamics['dynamic_range_db'] / 30)  # Normalize
        emotional_variation = np.std(emotion['tension_curve']) if emotion['tension_curve'] else 0
        emotion_score = min(1.0, emotional_variation * 5)
        
        # Expressiveness = weighted combination
        expressiveness = dynamic_range_score * 0.6 + emotion_score * 0.4
        return expressiveness

# ============================================================================
# SONG ANALYSIS ENGINE
# ============================================================================

class SongAnalysisEngine:
    """Complete song analysis including structure, harmony, and instrumentation"""
    
    def __init__(self, sample_rate=44100):
        self.sample_rate = sample_rate
        self.vocal_engine = VocalAnalysisEngine(sample_rate)
    
    def analyze_song(self, audio: np.ndarray, song_id: str, has_vocals: bool = True) -> SongAnalysis:
        """Complete song analysis pipeline"""
        
        print(f"Analyzing song: {song_id}")
        
        # Convert to mono for analysis (keeping stereo for output)
        audio_mono = librosa.to_mono(audio) if audio.ndim > 1 else audio
        
        # 1. Basic song properties
        duration = len(audio_mono) / self.sample_rate
        
        # 2. Key detection
        print("  - Detecting key...")
        key = self._detect_key(audio_mono)
        
        # 3. Tempo detection
        print("  - Detecting tempo...")
        tempo, beats = librosa.beat.beat_track(y=audio_mono, sr=self.sample_rate)
        
        # 4. Time signature
        time_signature = self._detect_time_signature(audio_mono, beats)
        
        # 5. Structure analysis
        print("  - Analyzing structure...")
        sections = self._analyze_structure(audio_mono)
        
        # 6. Chord progression
        print("  - Extracting chords...")
        chord_progression = self._extract_chord_progression(audio_mono)
        
        # 7. Vocal analysis (if present)
        vocals = None
        if has_vocals:
            print("  - Analyzing vocals...")
            vocals = self.vocal_engine.analyze_vocals(audio_mono)
        
        # 8. Instrumentation analysis
        print("  - Analyzing instrumentation...")
        instruments = self._analyze_instrumentation(audio_mono)
        
        # 9. Energy analysis
        print("  - Analyzing energy...")
        energy_curve, energy_zones, climax = self._analyze_energy(audio_mono)
        
        # 10. Frequency spectrum
        frequency_spectrum = self._analyze_frequency_spectrum(audio)
        
        # 11. Stereo analysis
        stereo_width = self._calculate_stereo_width(audio) if audio.ndim > 1 else 0.0
        
        # 12. Dynamics
        dynamics = self._analyze_song_dynamics(audio_mono)
        
        # Calculate harmonic complexity
        harmonic_complexity = self._calculate_harmonic_complexity(chord_progression)
        
        # Extract mode
        mode = "major" if "major" in key.lower() else "minor"
        
        print(f"Analysis complete for {song_id}")
        
        return SongAnalysis(
            song_id=song_id,
            duration=float(duration),
            sample_rate=self.sample_rate,
            channels=audio.shape[0] if audio.ndim > 1 else 1,
            key=key,
            tempo=float(tempo),
            time_signature=time_signature,
            mode=mode,
            scale_type="diatonic",  # Placeholder
            sections=sections,
            chord_progression=chord_progression,
            harmonic_complexity=float(harmonic_complexity),
            vocals=vocals,
            instruments_detected=instruments,
            frequency_spectrum=frequency_spectrum,
            stereo_width=float(stereo_width),
            dynamics=dynamics,
            energy_curve=energy_curve,
            energy_zones=energy_zones,
            climax_point=float(climax)
        )
    
    def _detect_key(self, audio: np.ndarray) -> str:
        """Detect musical key using chromagram"""
        chroma = librosa.feature.chroma_cqt(y=audio, sr=self.sample_rate)
        chroma_avg = chroma.mean(axis=1)
        
        # Major and minor profiles
        major_profile = np.array([6.35, 2.23, 3.48, 2.33, 4.38, 4.09, 2.52, 5.19, 2.39, 3.66, 2.29, 2.88])
        minor_profile = np.array([6.33, 2.68, 3.52, 5.38, 2.60, 3.53, 2.54, 4.75, 3.98, 2.69, 3.34, 3.17])
        
        keys = ['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B']
        best_key = "C major"
        best_correlation = -1
        
        for i in range(12):
            # Rotate profiles
            major_rotated = np.roll(major_profile, i)
            minor_rotated = np.roll(minor_profile, i)
            
            # Calculate correlations
            major_corr = np.corrcoef(chroma_avg, major_rotated)[0, 1]
            minor_corr = np.corrcoef(chroma_avg, minor_rotated)[0, 1]
            
            if major_corr > best_correlation:
                best_correlation = major_corr
                best_key = f"{keys[i]} major"
            
            if minor_corr > best_correlation:
                best_correlation = minor_corr
                best_key = f"{keys[i]} minor"
        
        return best_key
    
    def _detect_time_signature(self, audio: np.ndarray, beats: np.ndarray) -> str:
        """Detect time signature from beat patterns"""
        if len(beats) < 8:
            return "4/4"  # Default
        
        beat_times = librosa.frames_to_time(beats, sr=self.sample_rate)
        beat_intervals = np.diff(beat_times)
        
        # Look for repeating patterns of 3 or 4 beats
        if len(beat_intervals) >= 12:
            # Check if first 4 beats are similar to next 4
            group1 = beat_intervals[:4].mean()
            group2 = beat_intervals[4:8].mean()
            
            if abs(group1 - group2) < 0.05:  # Similar groups suggest 4/4
                return "4/4"
            else:
                # Check for 3/4 pattern
                group1 = beat_intervals[:3].mean()
                group2 = beat_intervals[3:6].mean()
                
                if abs(group1 - group2) < 0.05:
                    return "3/4"
        
        return "4/4"
    
    def _analyze_structure(self, audio: np.ndarray) -> List[Dict]:
        """Analyze song structure using novelty detection"""
        
        # Compute novelty curve
        hop_length = 512
        novelty = librosa.segment.cross_similarity(audio, audio, 
                                                   mode='affinity',
                                                   k=5)
        
        # Use librosa's structural segmentation
        bound_frames = librosa.segment.agglomerative(novelty, k=7)
        bound_times = librosa.frames_to_time(bound_frames, sr=self.sample_rate, hop_length=hop_length)
        
        # Create sections
        sections = []
        section_types = [SongSection.INTRO, SongSection.VERSE, SongSection.CHORUS, 
                        SongSection.VERSE, SongSection.CHORUS, SongSection.BRIDGE, SongSection.OUTRO]
        
        for i in range(len(bound_times) - 1):
            start = bound_times[i]
            end = bound_times[i + 1]
            duration = end - start
            
            # Assign section type
            section_type = section_types[i % len(section_types)] if i < len(section_types) else SongSection.VERSE
            
            sections.append({
                'section_id': f"section_{i:03d}",
                'type': section_type.value,
                'start_time': float(start),
                'end_time': float(end),
                'duration': float(duration),
                'bars': int(np.ceil(duration * (self._get_bpm() / 60) / 4))  # Estimate bars
            })
        
        return sections
    
    def _get_bpm(self) -> float:
        """Get BPM for bar calculation"""
        return 120.0  # Default, should be calculated
    
    def _extract_chord_progression(self, audio: np.ndarray) -> List[str]:
        """Extract chord progression using chromagram"""
        chroma = librosa.feature.chroma_cqt(y=audio, sr=self.sample_rate)
        
        # Simple chord detection
        chords = []
        chord_names = ['C', 'Dm', 'Em', 'F', 'G', 'Am', 'Bdim']
        
        # Take strongest chroma at each time frame
        for i in range(min(16, chroma.shape[1])):  # First 16 time frames
            strongest = np.argmax(chroma[:, i])
            chords.append(chord_names[strongest % len(chord_names)])
        
        return chords
    
    def _analyze_instrumentation(self, audio: np.ndarray) -> List[str]:
        """Detect instruments based on spectral characteristics"""
        # Simplified instrument detection
        instruments = []
        
        # Analyze spectral characteristics
        spectral_centroid = librosa.feature.spectral_centroid(y=audio, sr=self.sample_rate)[0].mean()
        spectral_bandwidth = librosa.feature.spectral_bandwidth(y=audio, sr=self.sample_rate)[0].mean()
        zero_crossing = librosa.feature.zero_crossing_rate(audio)[0].mean()
        
        # Heuristic instrument detection
        if spectral_centroid > 2000 and zero_crossing > 0.1:
            instruments.append("drums")
        
        if spectral_centroid < 800 and spectral_bandwidth < 500:
            instruments.append("bass")
        
        if spectral_centroid > 1000 and spectral_centroid < 3000:
            instruments.append("guitar")
        
        if spectral_centroid > 1500 and spectral_bandwidth > 1000:
            instruments.append("synth")
        
        if spectral_centroid > 2000 and zero_crossing < 0.05:
            instruments.append("strings")
        
        return instruments
    
    def _analyze_energy(self, audio: np.ndarray) -> Tuple[List[float], Dict[str, List[float]], float]:
        """Analyze energy curve and find climax"""
        energy = librosa.feature.rms(y=audio)[0]
        energy_times = librosa.frames_to_time(range(len(energy)), sr=self.sample_rate)
        
        # Normalize energy
        energy_normalized = (energy - energy.min()) / (energy.max() - energy.min() + 1e-6)
        
        # Find energy zones
        zones = {
            'low': energy_normalized[energy_normalized < 0.3].tolist(),
            'medium': energy_normalized[(energy_normalized >= 0.3) & (energy_normalized < 0.7)].tolist(),
            'high': energy_normalized[energy_normalized >= 0.7].tolist()
        }
        
        # Find climax (highest energy point)
        climax_time = energy_times[np.argmax(energy_normalized)]
        
        return energy_normalized.tolist(), zones, float(climax_time)
    
    def _analyze_frequency_spectrum(self, audio: np.ndarray) -> Dict[str, float]:
        """Analyze frequency spectrum distribution"""
        if audio.ndim > 1:
            audio = librosa.to_mono(audio)
        
        fft = np.abs(np.fft.rfft(audio))
        freqs = np.fft.rfftfreq(len(audio), 1/self.sample_rate)
        
        # Calculate energy in different frequency bands
        bands = {
            'sub_bass': (20, 60),
            'bass': (60, 250),
            'low_mid': (250, 500),
            'mid': (500, 2000),
            'high_mid': (2000, 4000),
            'presence': (4000, 6000),
            'brilliance': (6000, 20000)
        }
        
        spectrum = {}
        for band_name, (low, high) in bands.items():
            mask = (freqs >= low) & (freqs <= high)
            if mask.any():
                band_energy = fft[mask].sum()
                total_energy = fft.sum()
                spectrum[band_name] = float(band_energy / total_energy) if total_energy > 0 else 0.0
        
        return spectrum
    
    def _calculate_stereo_width(self, audio: np.ndarray) -> float:
        """Calculate stereo width (0=mono, 1=full stereo)"""
        if audio.ndim == 1:
            return 0.0
        
        left = audio[0]
        right = audio[1]
        
        # Calculate correlation between channels
        if len(left) > 1 and len(right) > 1:
            correlation = np.corrcoef(left, right)[0, 1]
            # Convert to width: -1 = inverted (wide), 0 = mono, 1 = correlated (narrow)
            width = max(0, 1 - abs(correlation))
            return float(width)
        
        return 0.0
    
    def _analyze_song_dynamics(self, audio: np.ndarray) -> Dict[str, float]:
        """Analyze song-wide dynamics"""
        # Simplified dynamics analysis
        rms = librosa.feature.rms(y=audio)[0]
        
        return {
            'peak': float(rms.max()),
            'average': float(rms.mean()),
            'dynamic_range': float(rms.max() - rms.min()),
            'crest_factor': float(rms.max() / (rms.mean() + 1e-6))
        }
    
    def _calculate_harmonic_complexity(self, chords: List[str]) -> float:
        """Calculate harmonic complexity based on chord progression"""
        if len(chords) < 2:
            return 0.0
        
        # Simple complexity measure: number of unique chords
        unique_chords = len(set(chords))
        max_possible = 7  # Diatonic chords in a key
        
        return unique_chords / max_possible

# ============================================================================
# STEM SEPARATION ENGINE
# ============================================================================

class StemSeparationEngine:
    """Separate audio into stems using Demucs"""
    
    def __init__(self, model_name: str = "htdemucs"):
        self.model_name = model_name
    
    def separate(self, audio_path: Path, output_dir: Path) -> Dict[str, Path]:
        """Separate audio into stems"""
        
        print(f"Separating stems for {audio_path.name}...")
        
        try:
            # Try to import Demucs
            from demucs import pretrained
            from demucs.apply import apply_model
            from demucs.audio import AudioFile, save_audio
            
            # Load model
            model = pretrained.get_model(self.model_name)
            
            # Load audio
            audio = AudioFile(audio_path).read(streams=0, 
                                               samplerate=model.samplerate,
                                               channels=model.audio_channels)
            
            # Separate
            sources = apply_model(model, audio[None], device='cpu')[0]
            
            # Save stems
            stems = {}
            stem_names = ['drums', 'bass', 'other', 'vocals']
            
            for i, name in enumerate(stem_names):
                stem_path = output_dir / f"{name}.wav"
                save_audio(sources[i], str(stem_path), samplerate=model.samplerate)
                stems[name] = stem_path
            
            print("Stem separation complete!")
            return stems
            
        except ImportError:
            print("Demucs not available, using placeholder separation")
            return self._placeholder_separation(audio_path, output_dir)
    
    def _placeholder_separation(self, audio_path: Path, output_dir: Path) -> Dict[str, Path]:
        """Placeholder separation for testing"""
        y, sr = librosa.load(audio_path, sr=None, mono=False)
        
        if y.ndim > 1:
            y_mono = librosa.to_mono(y)
        else:
            y_mono = y
        
        # Create placeholder stems
        stems = {}
        stem_names = ['drums', 'bass', 'other', 'vocals']
        
        for name in stem_names:
            stem_path = output_dir / f"{name}.wav"
            
            if name == 'vocals':
                # Simple high-pass filter to simulate vocals
                from scipy import signal
                b, a = signal.butter(4, 300/(sr/2), btype='high')
                stem_audio = signal.filtfilt(b, a, y_mono)
            elif name == 'drums':
                # Simple transient detection
                stem_audio = y_mono * 0.3
            elif name == 'bass':
                # Simple low-pass filter
                from scipy import signal
                b, a = signal.butter(4, 250/(sr/2), btype='low')
                stem_audio = signal.filtfilt(b, a, y_mono)
            else:  # other
                stem_audio = y_mono * 0.4
            
            sf.write(stem_path, stem_audio, sr)
            stems[name] = stem_path
        
        return stems

# ============================================================================
# COMPATIBILITY ANALYSIS ENGINE
# ============================================================================

class CompatibilityEngine:
    """Analyze compatibility between two songs"""
    
    def __init__(self):
        self.music_theory = MusicTheoryHelper()
    
    def analyze_compatibility(self, song_a: SongAnalysis, song_b: SongAnalysis) -> CompatibilityScore:
        """Complete compatibility analysis between two songs"""
        
        print(f"Analyzing compatibility between {song_a.song_id} and {song_b.song_id}")
        
        # 1. Key compatibility
        print("  - Analyzing key compatibility...")
        key_comp, transposition, recommended_key = self._analyze_key_compatibility(song_a, song_b)
        
        # 2. Tempo compatibility
        print("  - Analyzing tempo compatibility...")
        tempo_comp, tempo_adjustment = self._analyze_tempo_compatibility(song_a, song_b)
        
        # 3. Range compatibility
        print("  - Analyzing vocal range compatibility...")
        range_comp = self._analyze_range_compatibility(song_a, song_b)
        
        # 4. Timbre compatibility
        print("  - Analyzing timbre compatibility...")
        timbre_comp = self._analyze_timbre_compatibility(song_a, song_b)
        
        # 5. Structure compatibility
        print("  - Analyzing structure compatibility...")
        structure_comp, suggested_structure = self._analyze_structure_compatibility(song_a, song_b)
        
        # 6. Emotional compatibility
        print("  - Analyzing emotional compatibility...")
        emotional_comp = self._analyze_emotional_compatibility(song_a, song_b)
        
        # 7. Harmonic compatibility
        print("  - Analyzing harmonic compatibility...")
        harmonic_comp = self._analyze_harmonic_compatibility(song_a, song_b)
        
        # 8. Vocal-specific compatibility
        print("  - Analyzing vocal blend...")
        vocal_blend, phrase_alignment, freq_separation = self._analyze_vocal_compatibility(song_a, song_b)
        
        # 9. Identify challenges and opportunities
        print("  - Identifying challenges and opportunities...")
        challenges = self._identify_challenges(song_a, song_b, key_comp, tempo_comp, range_comp)
        opportunities = self._identify_opportunities(song_a, song_b)
        
        # 10. Frequency collision analysis
        print("  - Analyzing frequency collisions...")
        collision_zones = self._analyze_frequency_collisions(song_a, song_b)
        
        # 11. Phrase overlap analysis
        print("  - Analyzing phrase overlaps...")
        phrase_overlap = self._analyze_phrase_overlaps(song_a, song_b)
        
        # 12. Arrangement strategies
        print("  - Generating arrangement strategies...")
        strategies = self._generate_arrangement_strategies(song_a, song_b, key_comp, tempo_comp)
        
        # 13. Calculate overall scores
        print("  - Calculating overall scores...")
        overall, difficulty, quality = self._calculate_overall_scores(
            key_comp, tempo_comp, range_comp, timbre_comp, 
            structure_comp, emotional_comp, harmonic_comp,
            vocal_blend, phrase_alignment, freq_separation
        )
        
        # 14. Harmonic tension points
        harmonic_tension = self._identify_harmonic_tension_points(song_a, song_b)
        
        print("Compatibility analysis complete!")
        
        return CompatibilityScore(
            song_a_id=song_a.song_id,
            song_b_id=song_b.song_id,
            key_compatibility=key_comp,
            tempo_compatibility=tempo_comp,
            range_compatibility=range_comp,
            timbre_compatibility=timbre_comp,
            structure_compatibility=structure_comp,
            emotional_compatibility=emotional_comp,
            harmonic_compatibility=harmonic_comp,
            vocal_blend_score=vocal_blend,
            phrase_alignment_score=phrase_alignment,
            frequency_separation_score=freq_separation,
            overall_score=overall,
            difficulty_score=difficulty,
            quality_potential=quality,
            recommended_transposition_semitones=transposition,
            recommended_tempo_adjustment_ratio=tempo_adjustment,
            recommended_key=recommended_key,
            arrangement_strategies=strategies,
            suggested_structure=suggested_structure,
            challenges=challenges,
            opportunities=opportunities,
            frequency_collision_zones=collision_zones,
            phrase_overlap_analysis=phrase_overlap,
            harmonic_tension_points=harmonic_tension
        )
    
    def _analyze_key_compatibility(self, song_a: SongAnalysis, song_b: SongAnalysis) -> Tuple[float, int, str]:
        """Analyze key compatibility and recommend transposition"""
        
        # Extract key information
        key_a = song_a.key.split()[0]
        key_b = song_b.key.split()[0]
        mode_a = "major" if "major" in song_a.key.lower() else "minor"
        mode_b = "major" if "major" in song_b.key.lower() else "minor"
        
        # Calculate key distance
        distance = self.music_theory.key_distance(key_a, key_b, mode_a, mode_b)
        
        # Calculate compatibility (1 - normalized distance)
        max_distance = 6  # Maximum distance in circle of fifths
        compatibility = 1.0 - (distance / max_distance)
        
        # Recommend transposition
        transposition, recommended_key = self.music_theory.recommend_transposition(
            key_a, mode_a, key_b, mode_b
        )
        
        return compatibility, transposition, recommended_key
    
    def _analyze_tempo_compatibility(self, song_a: SongAnalysis, song_b: SongAnalysis) -> Tuple[float, float]:
        """Analyze tempo compatibility and recommend adjustment"""
        
        tempo_a = song_a.tempo
        tempo_b = song_b.tempo
        
        # Check for simple ratios (1:1, 2:1, 3:2, 4:3)
        ratio = max(tempo_a, tempo_b) / min(tempo_a, tempo_b)
        
        # Calculate compatibility based on ratio closeness to simple ratios
        simple_ratios = [1.0, 2.0, 1.5, 1.333, 0.75, 0.667, 0.5]
        closest_ratio = min(simple_ratios, key=lambda x: abs(ratio - x))
        ratio_distance = abs(ratio - closest_ratio)
        
        # Compatibility: 1 if exact match, decreasing with distance
        compatibility = max(0, 1.0 - (ratio_distance * 5))
        
        # Recommend tempo adjustment
        if tempo_a > tempo_b:
            adjustment = closest_ratio if ratio > 1.1 else 1.0
        else:
            adjustment = 1.0 / closest_ratio if ratio > 1.1 else 1.0
        
        return compatibility, adjustment
    
    def _analyze_range_compatibility(self, song_a: SongAnalysis, song_b: SongAnalysis) -> float:
        """Analyze vocal range compatibility"""
        
        if not song_a.vocals or not song_b.vocals:
            return 0.5  # Neutral if no vocals
        
        # Get vocal ranges
        range_a_low = song_a.vocals.range_low_hz
        range_a_high = song_a.vocals.range_high_hz
        range_b_low = song_b.vocals.range_low_hz
        range_b_high = song_b.vocals.range_high_hz
        
        # Check for overlap
        overlap_low = max(range_a_low, range_b_low)
        overlap_high = min(range_a_high, range_b_high)
        
        if overlap_low < overlap_high:
            # Ranges overlap - calculate overlap percentage
            overlap_range = overlap_high - overlap_low
            min_range = min(range_a_high - range_a_low, range_b_high - range_b_low)
            overlap_percentage = overlap_range / min_range
            
            # Some overlap is good, too much can cause mud
            if overlap_percentage < 0.3:
                return 0.8  # Small overlap - good for harmony
            elif overlap_percentage < 0.7:
                return 0.5  # Moderate overlap - needs EQ carving
            else:
                return 0.3  # Large overlap - potential conflict
        else:
            # No overlap - could be good for octave separation
            gap = overlap_low - overlap_high
            total_span = max(range_a_high, range_b_high) - min(range_a_low, range_b_low)
            gap_percentage = gap / total_span
            
            if gap_percentage < 0.2:
                return 0.7  # Small gap - good for counterpoint
            elif gap_percentage < 0.5:
                return 0.9  # Medium gap - excellent for duet
            else:
                return 0.6  # Large gap - might sound disconnected
    
    def _analyze_timbre_compatibility(self, song_a: SongAnalysis, song_b: SongAnalysis) -> float:
        """Analyze timbre compatibility"""
        
        if not song_a.vocals or not song_b.vocals:
            return 0.5
        
        # Compare timbre profiles
        timbre_a = song_a.vocals.timbre
        timbre_b = song_b.vocals.timbre
        
        # Calculate similarity/difference scores
        spectral_diff = abs(timbre_a.spectral_centroid_mean - timbre_b.spectral_centroid_mean)
        brightness_diff = abs(timbre_a.brightness - timbre_b.brightness)
        warmth_diff = abs(timbre_a.warmth - timbre_b.warmth)
        
        # Normalize differences
        spectral_score = 1.0 - min(1.0, spectral_diff / 1000)
        brightness_score = 1.0 - brightness_diff
        warmth_score = 1.0 - warmth_diff
        
        # Calculate overall timbre compatibility
        # Different timbres can complement each other, so medium similarity is good
        similarity = (spectral_score + brightness_score + warmth_score) / 3
        
        if 0.4 <= similarity <= 0.7:
            return 0.8  # Good complementarity
        elif similarity > 0.7:
            return 0.6  # Too similar - might blend too much
        else:
            return 0.5  # Very different - might not blend well
    
    def _analyze_structure_compatibility(self, song_a: SongAnalysis, song_b: SongAnalysis) -> Tuple[float, List[Dict]]:
        """Analyze structure compatibility and suggest fusion structure"""
        
        sections_a = song_a.sections
        sections_b = song_b.sections
        
        if not sections_a or not sections_b:
            return 0.5, []
        
        # Calculate section type alignment
        section_types_a = [s['type'] for s in sections_a]
        section_types_b = [s['type'] for s in sections_b]
        
        # Find best alignment
        max_len = max(len(section_types_a), len(section_types_b))
        min_len = min(len(section_types_a), len(section_types_b))
        
        alignment_score = 0
        for i in range(min_len):
            if section_types_a[i] == section_types_b[i]:
                alignment_score += 1
        
        compatibility = alignment_score / max_len
        
        # Suggest fusion structure
        suggested_structure = self._suggest_fusion_structure(sections_a, sections_b)
        
        return compatibility, suggested_structure
    
    def _suggest_fusion_structure(self, sections_a: List[Dict], sections_b: List[Dict]) -> List[Dict]:
        """Suggest optimal fusion structure"""
        
        structure = []
        
        # Simple alternating structure for now
        max_sections = max(len(sections_a), len(sections_b))
        
        for i in range(max_sections):
            if i < len(sections_a) and i < len(sections_b):
                # Both songs have this section - alternate
                if i % 2 == 0:
                    structure.append({
                        'source': 'A',
                        'section': sections_a[i],
                        'duration': sections_a[i]['duration']
                    })
                else:
                    structure.append({
                        'source': 'B',
                        'section': sections_b[i],
                        'duration': sections_b[i]['duration']
                    })
            elif i < len(sections_a):
                structure.append({
                    'source': 'A',
                    'section': sections_a[i],
                    'duration': sections_a[i]['duration']
                })
            else:
                structure.append({
                    'source': 'B',
                    'section': sections_b[i],
                    'duration': sections_b[i]['duration']
                })
        
        return structure
    
    def _analyze_emotional_compatibility(self, song_a: SongAnalysis, song_b: SongAnalysis) -> float:
        """Analyze emotional compatibility"""
        
        # Compare energy curves
        energy_a = np.array(song_a.energy_curve)
        energy_b = np.array(song_b.energy_curve)
        
        # Resample to same length
        min_len = min(len(energy_a), len(energy_b))
        energy_a_resampled = energy_a[:min_len]
        energy_b_resampled = energy_b[:min_len]
        
        # Calculate correlation
        if min_len > 10:
            correlation = np.corrcoef(energy_a_resampled, energy_b_resampled)[0, 1]
            compatibility = (correlation + 1) / 2  # Convert from [-1, 1] to [0, 1]
        else:
            compatibility = 0.5
        
        return compatibility
    
    def _analyze_harmonic_compatibility(self, song_a: SongAnalysis, song_b: SongAnalysis) -> float:
        """Analyze harmonic compatibility"""
        
        chords_a = song_a.chord_progression
        chords_b = song_b.chord_progression
        
        if not chords_a or not chords_b:
            return 0.5
        
        # Simple compatibility based on chord complexity match
        complexity_a = song_a.harmonic_complexity
        complexity_b = song_b.harmonic_complexity
        
        # Similar complexity is good
        complexity_diff = abs(complexity_a - complexity_b)
        compatibility = 1.0 - complexity_diff
        
        return compatibility
    
    def _analyze_vocal_compatibility(self, song_a: SongAnalysis, song_b: SongAnalysis) -> Tuple[float, float, float]:
        """Analyze vocal-specific compatibility"""
        
        if not song_a.vocals or not song_b.vocals:
            return 0.5, 0.5, 0.5
        
        # 1. Vocal blend score
        blend_score = self._calculate_vocal_blend(song_a.vocals, song_b.vocals)
        
        # 2. Phrase alignment score
        alignment_score = self._calculate_phrase_alignment(song_a.vocals, song_b.vocals)
        
        # 3. Frequency separation score
        separation_score = self._calculate_frequency_separation(song_a.vocals, song_b.vocals)
        
        return blend_score, alignment_score, separation_score
    
    def _calculate_vocal_blend(self, vocals_a: VocalAnalysis, vocals_b: VocalAnalysis) -> float:
        """Calculate how well voices will blend"""
        
        # Consider voice type combination
        if vocals_a.voice_type != vocals_b.voice_type:
            # Different voice types usually blend well
            if (vocals_a.voice_type in [VoiceType.SOPRANO, VoiceType.MEZZO_SOPRANO, VoiceType.ALTO] and
                vocals_b.voice_type in [VoiceType.TENOR, VoiceType.BARITONE, VoiceType.BASS]):
                return 0.9  # Male-female duet - excellent blend
            else:
                return 0.7  # Different types but same gender
        else:
            # Same voice type - might blend too much or conflict
            return 0.5
    
    def _calculate_phrase_alignment(self, vocals_a: VocalAnalysis, vocals_b: VocalAnalysis) -> float:
        """Calculate how well phrases align temporally"""
        
        phrases_a = vocals_a.phrases
        phrases_b = vocals_b.phrases
        
        if not phrases_a or not phrases_b:
            return 0.5
        
        # Calculate phrase duration statistics
        durations_a = [p.duration for p in phrases_a]
        durations_b = [p.duration for p in phrases_b]
        
        avg_duration_a = np.mean(durations_a) if durations_a else 0
        avg_duration_b = np.mean(durations_b) if durations_b else 0
        
        # Similar average phrase duration is good for alignment
        if avg_duration_a > 0 and avg_duration_b > 0:
            ratio = min(avg_duration_a, avg_duration_b) / max(avg_duration_a, avg_duration_b)
            alignment = ratio
        else:
            alignment = 0.5
        
        return alignment
    
    def _calculate_frequency_separation(self, vocals_a: VocalAnalysis, vocals_b: VocalAnalysis) -> float:
        """Calculate natural frequency separation"""
        
        # Get tessitura ranges
        tess_a_low = vocals_a.tessitura_low_hz
        tess_a_high = vocals_a.tessitura_high_hz
        tess_b_low = vocals_b.tessitura_low_hz
        tess_b_high = vocals_b.tessitura_high_hz
        
        # Calculate overlap in tessitura
        overlap_low = max(tess_a_low, tess_b_low)
        overlap_high = min(tess_a_high, tess_b_high)
        
        if overlap_low < overlap_high:
            # Overlap exists
            overlap_range = overlap_high - overlap_low
            tess_a_range = tess_a_high - tess_a_low
            tess_b_range = tess_b_high - tess_b_low
            
            overlap_percentage = overlap_range / min(tess_a_range, tess_b_range)
            separation = 1.0 - overlap_percentage
        else:
            # No overlap - excellent separation
            separation = 1.0
        
        return separation
    
    def _identify_challenges(self, song_a: SongAnalysis, song_b: SongAnalysis, 
                           key_comp: float, tempo_comp: float, range_comp: float) -> List[str]:
        """Identify potential challenges in fusion"""
        
        challenges = []
        
        # Key challenges
        if key_comp < 0.3:
            challenges.append("keys_are_distant_requires_transposition")
        elif key_comp < 0.6:
            challenges.append("keys_differ_may_need_adjustment")
        
        # Tempo challenges
        if tempo_comp < 0.3:
            challenges.append("tempos_are_very_different")
        elif tempo_comp < 0.6:
            challenges.append("tempos_differ_need_time_stretch")
        
        # Range challenges
        if range_comp < 0.3:
            challenges.append("vocal_ranges_heavily_overlap")
        elif range_comp < 0.6:
            challenges.append("vocal_ranges_partially_overlap")
        
        # Structure challenges
        if len(song_a.sections) != len(song_b.sections):
            challenges.append("different_number_of_sections")
        
        # Emotional challenges
        if hasattr(song_a, 'vocals') and hasattr(song_b, 'vocals'):
            if song_a.vocals and song_b.vocals:
                if song_a.vocals.voice_type == song_b.vocals.voice_type:
                    challenges.append("same_voice_type_may_blend_too_much")
        
        return challenges
    
    def _identify_opportunities(self, song_a: SongAnalysis, song_b: SongAnalysis) -> List[str]:
        """Identify opportunities for great fusion"""
        
        opportunities = []
        
        # Check for male-female combination
        if hasattr(song_a, 'vocals') and hasattr(song_b, 'vocals'):
            if song_a.vocals and song_b.vocals:
                if (song_a.vocals.voice_type in [VoiceType.SOPRANO, VoiceType.MEZZO_SOPRANO, VoiceType.ALTO] and
                    song_b.vocals.voice_type in [VoiceType.TENOR, VoiceType.BARITONE, VoiceType.BASS]):
                    opportunities.append("male_female_duet_natural_harmony")
                elif (song_b.vocals.voice_type in [VoiceType.SOPRANO, VoiceType.MEZZO_SOPRANO, VoiceType.ALTO] and
                      song_a.vocals.voice_type in [VoiceType.TENOR, VoiceType.BARITONE, VoiceType.BASS]):
                    opportunities.append("male_female_duet_natural_harmony")
        
        # Check for complementary energy curves
        energy_a = np.array(song_a.energy_curve)
        energy_b = np.array(song_b.energy_curve)
        
        if len(energy_a) > 10 and len(energy_b) > 10:
            # Check if one song's low energy sections align with other's high energy
            min_len = min(len(energy_a), len(energy_b))
            correlation = np.corrcoef(energy_a[:min_len], energy_b[:min_len])[0, 1]
            
            if correlation < -0.3:
                opportunities.append("complementary_energy_curves")
        
        # Check for phrase gaps that can be filled
        if hasattr(song_a, 'vocals') and hasattr(song_b, 'vocals'):
            if song_a.vocals and song_b.vocals:
                silences_a = song_a.vocals.silences
                phrases_b = song_b.vocals.phrases
                
                if silences_a and phrases_b:
                    avg_silence_a = np.mean([s['duration'] for s in silences_a])
                    avg_phrase_b = np.mean([p.duration for p in phrases_b])
                    
                    if avg_silence_a > avg_phrase_b * 0.8:
                        opportunities.append("song_b_phrases_fit_in_song_a_gaps")
        
        return opportunities
    
    def _analyze_frequency_collisions(self, song_a: SongAnalysis, song_b: SongAnalysis) -> List[Tuple[float, float]]:
        """Identify frequency ranges where vocals might collide"""
        
        if not song_a.vocals or not song_b.vocals:
            return []
        
        # Get frequency spectra
        spectrum_a = song_a.frequency_spectrum
        spectrum_b = song_b.frequency_spectrum
        
        # Identify overlapping frequency bands with high energy
        collision_zones = []
        
        frequency_bands = [
            ('bass', 60, 250),
            ('low_mid', 250, 500),
            ('mid', 500, 2000),
            ('high_mid', 2000, 4000),
            ('presence', 4000, 6000)
        ]
        
        for band_name, low, high in frequency_bands:
            energy_a = spectrum_a.get(band_name, 0)
            energy_b = spectrum_b.get(band_name, 0)
            
            # Both have significant energy in this band
            if energy_a > 0.15 and energy_b > 0.15:
                collision_zones.append((float(low), float(high)))
        
        return collision_zones
    
    def _analyze_phrase_overlaps(self, song_a: SongAnalysis, song_b: SongAnalysis) -> Dict[str, Any]:
        """Analyze how phrases from each song overlap temporally"""
        
        if not song_a.vocals or not song_b.vocals:
            return {}
        
        phrases_a = song_a.vocals.phrases
        phrases_b = song_b.vocals.phrases
        
        if not phrases_a or not phrases_b:
            return {}
        
        # Calculate overlap statistics
        total_duration_a = sum(p.duration for p in phrases_a)
        total_duration_b = sum(p.duration for p in phrases_b)
        
        # Estimate potential overlaps if phrases were aligned
        overlap_potential = min(total_duration_a, total_duration_b) / max(total_duration_a, total_duration_b)
        
        # Analyze phrase length compatibility
        avg_phrase_a = np.mean([p.duration for p in phrases_a]) if phrases_a else 0
        avg_phrase_b = np.mean([p.duration for p in phrases_b]) if phrases_b else 0
        
        phrase_length_ratio = min(avg_phrase_a, avg_phrase_b) / max(avg_phrase_a, avg_phrase_b) if avg_phrase_a > 0 and avg_phrase_b > 0 else 0
        
        return {
            'total_vocal_duration_a': total_duration_a,
            'total_vocal_duration_b': total_duration_b,
            'overlap_potential': overlap_potential,
            'average_phrase_length_a': avg_phrase_a,
            'average_phrase_length_b': avg_phrase_b,
            'phrase_length_compatibility': phrase_length_ratio,
            'recommended_approach': 'call_and_response' if phrase_length_ratio > 0.8 else 'layered_harmony'
        }
    
    def _generate_arrangement_strategies(self, song_a: SongAnalysis, song_b: SongAnalysis,
                                       key_comp: float, tempo_comp: float) -> List[str]:
        """Generate arrangement strategies based on compatibility"""
        
        strategies = []
        
        # Key-based strategies
        if key_comp > 0.8:
            strategies.append("maintain_original_keys")
        elif key_comp > 0.5:
            strategies.append("transpose_one_to_match")
        else:
            strategies.append("find_common_key_meeting_in_middle")
        
        # Tempo-based strategies
        if tempo_comp > 0.8:
            strategies.append("maintain_original_tempos")
        elif tempo_comp > 0.6:
            strategies.append("slight_tempo_adjustment")
        else:
            strategies.append("significant_time_stretching")
        
        # Vocal arrangement strategies
        if hasattr(song_a, 'vocals') and hasattr(song_b, 'vocals'):
            if song_a.vocals and song_b.vocals:
                if song_a.vocals.voice_type != song_b.vocals.voice_type:
                    strategies.append("male_female_harmony")
                    strategies.append("octave_separation")
                else:
                    strategies.append("unison_singing")
                    strategies.append("close_harmony")
        
        # Structural strategies
        if len(song_a.sections) == len(song_b.sections):
            strategies.append("parallel_structure")
        else:
            strategies.append("interleaved_sections")
        
        # Additional strategies
        strategies.append("call_and_response")
        strategies.append("layered_harmonies")
        strategies.append("dynamic_ducking")
        
        return strategies
    
    def _calculate_overall_scores(self, *scores: float) -> Tuple[float, float, float]:
        """Calculate overall, difficulty, and quality scores"""
        
        # Weights for different compatibility aspects
        weights = {
            'key': 0.15,
            'tempo': 0.10,
            'range': 0.15,
            'timbre': 0.10,
            'structure': 0.10,
            'emotional': 0.10,
            'harmonic': 0.10,
            'vocal_blend': 0.08,
            'phrase_alignment': 0.07,
            'frequency_separation': 0.05
        }
        
        # Unpack scores
        (key_comp, tempo_comp, range_comp, timbre_comp, 
         structure_comp, emotional_comp, harmonic_comp,
         vocal_blend, phrase_alignment, freq_separation) = scores
        
        # Calculate weighted overall score
        overall = (
            key_comp * weights['key'] +
            tempo_comp * weights['tempo'] +
            range_comp * weights['range'] +
            timbre_comp * weights['timbre'] +
            structure_comp * weights['structure'] +
            emotional_comp * weights['emotional'] +
            harmonic_comp * weights['harmonic'] +
            vocal_blend * weights['vocal_blend'] +
            phrase_alignment * weights['phrase_alignment'] +
            freq_separation * weights['frequency_separation']
        )
        
        # Calculate difficulty score (inverse of compatibility)
        difficulty = 1.0 - overall
        
        # Calculate quality potential (overall with bonus for good vocal blend)
        quality = overall * 0.9 + vocal_blend * 0.1
        
        return overall, difficulty, quality
    
    def _identify_harmonic_tension_points(self, song_a: SongAnalysis, song_b: SongAnalysis) -> List[float]:
        """Identify points of harmonic tension that could be interesting"""
        
        # This is a simplified version
        # In a full implementation, would analyze chord progressions in detail
        
        tension_points = []
        
        # Add some placeholder tension points based on song duration
        if song_a.duration > 0:
            # Suggest tension points at 25%, 50%, 75% of song
            tension_points.append(float(song_a.duration * 0.25))
            tension_points.append(float(song_a.duration * 0.5))
            tension_points.append(float(song_a.duration * 0.75))
        
        return tension_points

class MusicTheoryHelper:
    """Music theory helper functions"""
    
    # Circle of fifths
    CIRCLE_OF_FIFTHS = ['C', 'G', 'D', 'A', 'E', 'B', 'F#', 'C#', 'G#', 'D#', 'A#', 'F']
    
    # Relative minor/major relationships
    RELATIVE_MINORS = {
        'C': 'Am', 'G': 'Em', 'D': 'Bm', 'A': 'F#m', 'E': 'C#m', 'B': 'G#m',
        'F#': 'D#m', 'C#': 'A#m', 'G#': 'Fm', 'D#': 'Cm', 'A#': 'Gm', 'F': 'Dm'
    }
    
    # Note to semitone mapping
    NOTE_TO_SEMITONE = {
        'C': 0, 'C#': 1, 'Db': 1, 'D': 2, 'D#': 3, 'Eb': 3,
        'E': 4, 'F': 5, 'F#': 6, 'Gb': 6, 'G': 7, 'G#': 8,
        'Ab': 8, 'A': 9, 'A#': 10, 'Bb': 10, 'B': 11
    }
    
    def key_distance(self, key1: str, key2: str, mode1: str, mode2: str) -> int:
        """Calculate distance between two keys in circle of fifths"""
        
        # Normalize keys (remove #/b for simplicity)
        key1_base = key1.replace('#', '').replace('b', '')
        key2_base = key2.replace('#', '').replace('b', '')
        
        # Find positions in circle of fifths
        try:
            pos1 = self.CIRCLE_OF_FIFTHS.index(key1_base)
            pos2 = self.CIRCLE_OF_FIFTHS.index(key2_base)
        except ValueError:
            return 6  # Maximum distance if not found
        
        # Calculate circular distance
        distance = min(abs(pos1 - pos2), len(self.CIRCLE_OF_FIFTHS) - abs(pos1 - pos2))
        
        # Adjust for relative major/minor relationship
        if mode1 != mode2:
            # Check if they're relative
            if mode1 == 'major' and self.RELATIVE_MINORS.get(key1_base) == f"{key2_base}m":
                distance = 0  # Relative keys are very close
            elif mode2 == 'major' and self.RELATIVE_MINORS.get(key2_base) == f"{key1_base}m":
                distance = 0
        
        return distance
    
    def recommend_transposition(self, key1: str, mode1: str, key2: str, mode2: str) -> Tuple[int, str]:
        """Recommend transposition to match keys"""
        
        # Convert keys to semitones from C
        semitone1 = self.NOTE_TO_SEMITONE.get(key1, 0)
        semitone2 = self.NOTE_TO_SEMITONE.get(key2, 0)
        
        # Calculate semitone difference
        difference = semitone2 - semitone1
        
        # Find closest match (considering circular nature)
        if difference > 6:
            difference -= 12
        elif difference < -6:
            difference += 12
        
        # Calculate target key
        target_semitone = (semitone1 + difference) % 12
        
        # Find key name
        target_key = [k for k, v in self.NOTE_TO_SEMITONE.items() if v == target_semitone][0]
        
        # Recommend mode (usually keep original mode unless specified)
        target_mode = mode1 if mode1 == mode2 else mode1  # Prefer first song's mode
        
        return difference, f"{target_key} {target_mode}"

# ============================================================================
# ARRANGEMENT ENGINE
# ============================================================================

class ArrangementEngine:
    """Intelligent arrangement engine for vocal fusion"""
    
    def __init__(self):
        self.compatibility_engine = CompatibilityEngine()
    
    def create_arrangement_plan(self, song_a: SongAnalysis, song_b: SongAnalysis, 
                               compatibility: CompatibilityScore) -> Dict[str, Any]:
        """Create detailed arrangement plan for vocal fusion"""
        
        print(f"Creating arrangement plan for {song_a.song_id} and {song_b.song_id}")
        
        # 1. Determine arrangement strategy
        print("  - Determining arrangement strategy...")
        strategy = self._determine_arrangement_strategy(song_a, song_b, compatibility)
        
        # 2. Create timeline
        print("  - Creating timeline...")
        timeline = self._create_timeline(song_a, song_b, strategy)
        
        # 3. Plan vocal interactions
        print("  - Planning vocal interactions...")
        vocal_plan = self._plan_vocal_interactions(song_a, song_b, compatibility, timeline)
        
        # 4. Plan instrumental fusion
        print("  - Planning instrumental fusion...")
        instrumental_plan = self._plan_instrumental_fusion(song_a, song_b, timeline)
        
        # 5. Create mixing plan
        print("  - Creating mixing plan...")
        mixing_plan = self._create_mixing_plan(song_a, song_b, compatibility, vocal_plan)
        
        # 6. Calculate arrangement difficulty
        print("  - Calculating arrangement difficulty...")
        difficulty = self._calculate_arrangement_difficulty(song_a, song_b, compatibility, strategy)
        
        print("Arrangement planning complete!")
        
        return {
            'strategy': strategy,
            'timeline': timeline,
            'vocal_plan': vocal_plan,
            'instrumental_plan': instrumental_plan,
            'mixing_plan': mixing_plan,
            'difficulty_score': difficulty,
            'estimated_processing_time': self._estimate_processing_time(song_a, song_b, difficulty),
            'recommended_tools': self._recommend_tools(strategy, difficulty)
        }
    
    def _determine_arrangement_strategy(self, song_a: SongAnalysis, song_b: SongAnalysis,
                                      compatibility: CompatibilityScore) -> str:
        """Determine the best arrangement strategy"""
        
        # Use compatibility scores to choose strategy
        if compatibility.vocal_blend_score > 0.8:
            if compatibility.phrase_alignment_score > 0.7:
                return "harmony_focused"  # Good for singing together
            else:
                return "call_and_response"  # Alternate phrases
        elif compatibility.frequency_separation_score > 0.8:
            return "counterpoint"  # One high, one low
        else:
            return "layered"  # Simple layering
    
    def _create_timeline(self, song_a: SongAnalysis, song_b: SongAnalysis, strategy: str) -> List[Dict]:
        """Create arrangement timeline"""
        
        timeline = []
        current_time = 0.0
        
        # Get song durations
        duration_a = song_a.duration
        duration_b = song_b.duration
        target_duration = max(duration_a, duration_b) * 1.2  # Allow for overlap
        
        # Create sections based on strategy
        if strategy == "harmony_focused":
            # Interleave verses, harmonize on choruses
            sections = [
                {'type': 'intro', 'duration': 15, 'source': 'A'},
                {'type': 'verse', 'duration': 30, 'source': 'A', 'vocals': 'A'},
                {'type': 'verse', 'duration': 30, 'source': 'B', 'vocals': 'B'},
                {'type': 'chorus', 'duration': 30, 'source': 'both', 'vocals': 'both'},
                {'type': 'verse', 'duration': 30, 'source': 'A', 'vocals': 'A'},
                {'type': 'bridge', 'duration': 20, 'source': 'B', 'vocals': 'B'},
                {'type': 'chorus', 'duration': 30, 'source': 'both', 'vocals': 'both'},
                {'type': 'outro', 'duration': 15, 'source': 'A'}
            ]
        elif strategy == "call_and_response":
            # Alternate between songs
            sections = [
                {'type': 'intro', 'duration': 10, 'source': 'A'},
                {'type': 'call', 'duration': 8, 'source': 'A', 'vocals': 'A'},
                {'type': 'response', 'duration': 8, 'source': 'B', 'vocals': 'B'},
                {'type': 'call', 'duration': 8, 'source': 'A', 'vocals': 'A'},
                {'type': 'response', 'duration': 8, 'source': 'B', 'vocals': 'B'},
                {'type': 'chorus', 'duration': 20, 'source': 'both', 'vocals': 'both'},
                {'type': 'bridge', 'duration': 16, 'source': 'alternating', 'vocals': 'alternating'},
                {'type': 'chorus', 'duration': 20, 'source': 'both', 'vocals': 'both'},
                {'type': 'outro', 'duration': 10, 'source': 'B'}
            ]
        else:  # layered
            # Both songs play together with one dominant
            sections = [
                {'type': 'intro', 'duration': 15, 'source': 'A'},
                {'type': 'verse', 'duration': 30, 'source': 'both', 'vocals': 'A'},  # A dominant
                {'type': 'chorus', 'duration': 30, 'source': 'both', 'vocals': 'both'},
                {'type': 'verse', 'duration': 30, 'source': 'both', 'vocals': 'B'},  # B dominant
                {'type': 'chorus', 'duration': 30, 'source': 'both', 'vocals': 'both'},
                {'type': 'outro', 'duration': 15, 'source': 'both'}
            ]
        
        # Build timeline with actual times
        for section in sections:
            timeline.append({
                'start_time': current_time,
                'end_time': current_time + section['duration'],
                'type': section['type'],
                'source': section['source'],
                'vocals': section.get('vocals', 'none'),
                'duration': section['duration']
            })
            current_time += section['duration']
        
        return timeline
    
    def _plan_vocal_interactions(self, song_a: SongAnalysis, song_b: SongAnalysis,
                                compatibility: CompatibilityScore, timeline: List[Dict]) -> Dict[str, Any]:
        """Plan detailed vocal interactions"""
        
        vocal_plan = {
            'transposition': {
                'song_a': compatibility.recommended_transposition_semitones,
                'song_b': -compatibility.recommended_transposition_semitones  # Opposite direction
            },
            'tempo_adjustment': {
                'song_a': 1.0,  # Reference
                'song_b': compatibility.recommended_tempo_adjustment_ratio
            },
            'harmony_rules': [],
            'phrasing_rules': [],
            'eq_carving': [],
            'spatial_placement': {}
        }
        
        # Generate harmony rules based on voice types
        if song_a.vocals and song_b.vocals:
            voice_combo = f"{song_a.vocals.voice_type.value}_{song_b.vocals.voice_type.value}"
            
            if (song_a.vocals.voice_type in [VoiceType.SOPRANO, VoiceType.MEZZO_SOPRANO, VoiceType.ALTO] and
                song_b.vocals.voice_type in [VoiceType.TENOR, VoiceType.BARITONE, VoiceType.BASS]):
                # Male-female: use 3rds and 6ths
                vocal_plan['harmony_rules'].append({
                    'type': 'parallel_thirds',
                    'distance_semitones': 3,
                    'applicability': 'chorus_sections'
                })
                vocal_plan['harmony_rules'].append({
                    'type': 'parallel_sixths',
                    'distance_semitones': 8,  # 6th = 8 semitones
                    'applicability': 'verse_sections'
                })
            elif song_a.vocals.voice_type == song_b.vocals.voice_type:
                # Same voice type: use unison or octaves
                vocal_plan['harmony_rules'].append({
                    'type': 'unison',
                    'distance_semitones': 0,
                    'applicability': 'all_sections'
                })
                vocal_plan['harmony_rules'].append({
                    'type': 'octave',
                    'distance_semitones': 12,
                    'applicability': 'chorus_sections'
                })
        
        # Generate phrasing rules based on phrase analysis
        if song_a.vocals and song_b.vocals:
            phrases_a = song_a.vocals.phrases
            phrases_b = song_b.vocals.phrases
            
            if phrases_a and phrases_b:
                # Find complementary phrases
                avg_duration_a = np.mean([p.duration for p in phrases_a])
                avg_duration_b = np.mean([p.duration for p in phrases_b])
                
                if avg_duration_a > avg_duration_b * 1.5:
                    # A has longer phrases, B can fill gaps
                    vocal_plan['phrasing_rules'].append({
                        'type': 'fill_gaps',
                        'source': 'B',
                        'target': 'A_gaps'
                    })
                elif avg_duration_b > avg_duration_a * 1.5:
                    vocal_plan['phrasing_rules'].append({
                        'type': 'fill_gaps',
                        'source': 'A',
                        'target': 'B_gaps'
                    })
        
        # Generate EQ carving plan based on frequency collisions
        for collision in compatibility.frequency_collision_zones:
            vocal_plan['eq_carving'].append({
                'frequency_range': collision,
                'action': 'carve',
                'strategy': 'dynamic_eq',
                'priority': 'alternating'  # Carve from whichever is less important at that moment
            })
        
        # Spatial placement
        vocal_plan['spatial_placement'] = {
            'song_a': {'pan': -0.3, 'width': 0.7, 'depth': 0.8},
            'song_b': {'pan': 0.3, 'width': 0.7, 'depth': 0.8},
            'both': {'pan': 0.0, 'width': 1.0, 'depth': 1.0}
        }
        
        return vocal_plan
    
    def _plan_instrumental_fusion(self, song_a: SongAnalysis, song_b: SongAnalysis,
                                 timeline: List[Dict]) -> Dict[str, Any]:
        """Plan instrumental fusion"""
        
        instrumental_plan = {
            'rhythm_section': {},
            'melodic_elements': [],
            'harmonic_background': [],
            'arrangement_rules': []
        }
        
        # Rhythm section plan
        # Usually pick the stronger drums, layer percussion
        instrumental_plan['rhythm_section'] = {
            'primary_drums': 'A' if song_a.dynamics.get('peak', 0) > song_b.dynamics.get('peak', 0) else 'B',
            'secondary_percussion': 'B' if song_a.dynamics.get('peak', 0) > song_b.dynamics.get('peak', 0) else 'A',
            'bass': 'both',  # Layer both basslines
            'kick_drum': 'primary_only',  # Avoid phase issues
            'snare': 'layer_both'
        }
        
        # Melodic elements - avoid conflicts
        instrumental_plan['melodic_elements'] = [
            {
                'source': 'A',
                'elements': ['lead_guitar', 'synth_lead'],
                'applicability': 'verse_a'
            },
            {
                'source': 'B',
                'elements': ['piano', 'strings'],
                'applicability': 'verse_b'
            },
            {
                'source': 'both',
                'elements': ['all_melodic'],
                'applicability': 'chorus'
            }
        ]
        
        # Harmonic background
        instrumental_plan['harmonic_background'] = [
            {
                'type': 'pads',
                'source': 'A',
                'role': 'foundation'
            },
            {
                'type': 'arpeggios',
                'source': 'B',
                'role': 'movement'
            }
        ]
        
        # Arrangement rules
        instrumental_plan['arrangement_rules'] = [
            'duck_instruments_when_vocals_present',
            'highlight_melodic_hooks',
            'build_intensity_toward_chorus',
            'create_space_for_vocals'
        ]
        
        return instrumental_plan
    
    def _create_mixing_plan(self, song_a: SongAnalysis, song_b: SongAnalysis,
                           compatibility: CompatibilityScore, vocal_plan: Dict) -> Dict[str, Any]:
        """Create detailed mixing plan"""
        
        mixing_plan = {
            'vocal_processing': {},
            'instrumental_balance': {},
            'dynamic_processing': {},
            'effects': {},
            'mastering': {}
        }
        
        # Vocal processing
        mixing_plan['vocal_processing'] = {
            'compression': {
                'ratio': 3.0,
                'threshold': -20,
                'attack': 10,
                'release': 100,
                'makeup_gain': 3
            },
            'eq': {
                'high_pass': 80,
                'presence_boost': {'freq': 3000, 'gain': 2, 'q': 1.0},
                'air_boost': {'freq': 12000, 'gain': 1, 'q': 0.7}
            },
            'de_essing': {'threshold': -30, 'ratio': 3.0, 'frequency': 6000},
            'reverb': {'type': 'hall', 'time': 2.0, 'mix': 0.15},
            'delay': {'type': 'slap', 'time': 0.25, 'feedback': 0.3, 'mix': 0.1}
        }
        
        # Adjust EQ based on frequency collisions
        for i, collision in enumerate(compatibility.frequency_collision_zones):
            center_freq = (collision[0] + collision[1]) / 2
            mixing_plan['vocal_processing']['eq'][f'carve_{i+1}'] = {
                'freq': center_freq,
                'gain': -3,
                'q': 1.5,
                'dynamic': True  # Only apply when both vocals are present
            }
        
        # Instrumental balance
        mixing_plan['instrumental_balance'] = {
            'drums': -6,
            'bass': -8,
            'melodic': -12,
            'harmonic': -15,
            'vocals': 0  # Reference
        }
        
        # Dynamic processing
        mixing_plan['dynamic_processing'] = {
            'sidechain': {
                'source': 'vocals',
                'target': ['instruments', 'bass'],
                'ratio': 4.0,
                'threshold': -25,
                'attack': 5,
                'release': 200
            },
            'multiband_compression': {
                'low': {'ratio': 2.0, 'threshold': -20},
                'mid': {'ratio': 1.5, 'threshold': -15},
                'high': {'ratio': 1.2, 'threshold': -10}
            }
        }
        
        # Effects
        mixing_plan['effects'] = {
            'master_reverb': {'type': 'plate', 'time': 1.5, 'mix': 0.08},
            'master_delay': {'type': 'ping_pong', 'time': 0.5, 'mix': 0.05},
            'excitement': {'type': 'harmonic', 'amount': 0.3}
        }
        
        # Mastering
        mixing_plan['mastering'] = {
            'limiter': {'threshold': -1.0, 'ceiling': -0.3},
            'loudness_target': -14,  # LUFS
            'true_peak': -1.0,
            'stereo_enhancement': 0.1
        }
        
        return mixing_plan
    
    def _calculate_arrangement_difficulty(self, song_a: SongAnalysis, song_b: SongAnalysis,
                                        compatibility: CompatibilityScore, strategy: str) -> float:
        """Calculate arrangement difficulty score (0=easy, 1=difficult)"""
        
        difficulty = compatibility.difficulty_score  # Start with compatibility difficulty
        
        # Adjust based on strategy
        strategy_difficulty = {
            'harmony_focused': 0.7,
            'call_and_response': 0.5,
            'counterpoint': 0.8,
            'layered': 0.3
        }
        
        strategy_multiplier = strategy_difficulty.get(strategy, 0.5)
        difficulty *= strategy_multiplier
        
        # Adjust based on song complexity
        complexity_a = song_a.harmonic_complexity
        complexity_b = song_b.harmonic_complexity
        
        complexity_difficulty = (complexity_a + complexity_b) / 2
        difficulty = (difficulty + complexity_difficulty) / 2
        
        return min(1.0, difficulty)
    
    def _estimate_processing_time(self, song_a: SongAnalysis, song_b: SongAnalysis,
                                 difficulty: float) -> str:
        """Estimate processing time needed"""
        
        base_time = (song_a.duration + song_b.duration) / 60  # Base in minutes
        
        # Adjust for difficulty
        adjusted_time = base_time * (1 + difficulty)
        
        if adjusted_time < 30:
            return "30 minutes"
        elif adjusted_time < 60:
            return "1 hour"
        elif adjusted_time < 120:
            return "2 hours"
        else:
            return f"{int(np.ceil(adjusted_time / 60))} hours"
    
    def _recommend_tools(self, strategy: str, difficulty: float) -> List[str]:
        """Recommend tools for the arrangement"""
        
        tools = [
            'digital_audio_workstation',
            'melodyne_or_autotune',
            'eq_plugins',
            'compressor_plugins',
            'reverb_delay_plugins'
        ]
        
        if difficulty > 0.7:
            tools.append('harmonic_editing_tools')
            tools.append('vocal_alignment_tools')
        
        if strategy == 'harmony_focused':
            tools.append('harmony_generator')
        
        if strategy == 'counterpoint':
            tools.append('midi_editor')
            tools.append('notation_software')
        
        return tools

# ============================================================================
# MIXING ENGINE
# ============================================================================

class MixingEngine:
    """AI-powered mixing engine for vocal fusion"""
    
    def __init__(self, sample_rate=44100):
        self.sample_rate = sample_rate
    
    def create_mix(self, stems_a: Dict[str, np.ndarray], stems_b: Dict[str, np.ndarray],
                  arrangement_plan: Dict, mixing_plan: Dict) -> Dict[str, np.ndarray]:
        """Create mixed output based on arrangement and mixing plans"""
        
        print("Creating mix...")
        
        # Initialize output stems
        output_stems = {
            'vocals': None,
            'drums': None,
            'bass': None,
            'other': None,
            'full_mix': None
        }
        
        # 1. Process vocals
        print("  - Processing vocals...")
        mixed_vocals = self._mix_vocals(stems_a.get('vocals'), stems_b.get('vocals'),
                                       arrangement_plan['vocal_plan'],
                                       mixing_plan['vocal_processing'])
        
        # 2. Process rhythm section
        print("  - Processing rhythm section...")
        mixed_drums, mixed_bass = self._mix_rhythm_section(
            stems_a.get('drums'), stems_b.get('drums'),
            stems_a.get('bass'), stems_b.get('bass'),
            arrangement_plan['instrumental_plan']['rhythm_section']
        )
        
        # 3. Process other instruments
        print("  - Processing other instruments...")
        mixed_other = self._mix_other_instruments(
            stems_a.get('other'), stems_b.get('other'),
            arrangement_plan['instrumental_plan']['melodic_elements']
        )
        
        # 4. Apply dynamic processing
        print("  - Applying dynamic processing...")
        processed_stems = self._apply_dynamic_processing(
            mixed_vocals, mixed_drums, mixed_bass, mixed_other,
            mixing_plan['dynamic_processing']
        )
        
        # 5. Create final mix
        print("  - Creating final mix...")
        full_mix = self._create_final_mix(processed_stems, mixing_plan['instrumental_balance'])
        
        # 6. Apply mastering
        print("  - Applying mastering...")
        mastered_mix = self._apply_mastering(full_mix, mixing_plan['mastering'])
        
        output_stems.update({
            'vocals': mixed_vocals,
            'drums': mixed_drums,
            'bass': mixed_bass,
            'other': mixed_other,
            'full_mix': mastered_mix
        })
        
        print("Mix complete!")
        return output_stems
    
    def _mix_vocals(self, vocals_a: Optional[np.ndarray], vocals_b: Optional[np.ndarray],
                   vocal_plan: Dict, processing: Dict) -> np.ndarray:
        """Mix and process vocals"""
        
        # Handle missing vocals
        if vocals_a is None and vocals_b is None:
            return np.zeros(44100)  # 1 second of silence
        
        # Apply transposition if needed
        if vocal_plan['transposition']['song_a'] != 0 and vocals_a is not None:
            vocals_a = self._pitch_shift(vocals_a, vocal_plan['transposition']['song_a'])
        
        if vocal_plan['transposition']['song_b'] != 0 and vocals_b is not None:
            vocals_b = self._pitch_shift(vocals_b, vocal_plan['transposition']['song_b'])
        
        # Apply tempo adjustment
        if vocal_plan['tempo_adjustment']['song_b'] != 1.0 and vocals_b is not None:
            vocals_b = self._time_stretch(vocals_b, vocal_plan['tempo_adjustment']['song_b'])
        
        # Mix vocals based on arrangement
        if vocals_a is not None and vocals_b is not None:
            # Simple mixing for now - equal blend
            mixed = (vocals_a + vocals_b) / 2
        elif vocals_a is not None:
            mixed = vocals_a
        else:
            mixed = vocals_b
        
        # Apply vocal processing
        mixed = self._apply_vocal_processing(mixed, processing)
        
        return mixed
    
    def _pitch_shift(self, audio: np.ndarray, semitones: int) -> np.ndarray:
        """Pitch shift audio by semitones"""
        try:
            from librosa import effects
            return effects.pitch_shift(audio, sr=self.sample_rate, n_steps=semitones)
        except:
            # Simple resampling for placeholder
            ratio = 2 ** (semitones / 12)
            new_length = int(len(audio) / ratio)
            indices = np.linspace(0, len(audio), new_length).astype(int)
            return audio[indices]
    
    def _time_stretch(self, audio: np.ndarray, ratio: float) -> np.ndarray:
        """Time stretch audio by ratio"""
        try:
            from librosa import effects
            return effects.time_stretch(audio, rate=1/ratio)
        except:
            # Simple resampling for placeholder
            new_length = int(len(audio) * ratio)
            indices = np.linspace(0, len(audio), new_length).astype(int)
            return audio[indices]
    
    def _apply_vocal_processing(self, audio: np.ndarray, processing: Dict) -> np.ndarray:
        """Apply vocal processing chain"""
        processed = audio.copy()
        
        # Apply EQ
        if 'eq' in processing:
            processed = self._apply_eq(processed, processing['eq'])
        
        # Apply compression
        if 'compression' in processing:
            processed = self._apply_compression(processed, processing['compression'])
        
        # Apply reverb
        if 'reverb' in processing:
            processed = self._apply_reverb(processed, processing['reverb'])
        
        return processed
    
    def _apply_eq(self, audio: np.ndarray, eq_settings: Dict) -> np.ndarray:
        """Apply EQ settings"""
        # Simplified EQ implementation
        from scipy import signal
        
        # High-pass filter
        if 'high_pass' in eq_settings:
            freq = eq_settings['high_pass']
            b, a = signal.butter(4, freq/(self.sample_rate/2), btype='high')
            audio = signal.filtfilt(b, a, audio)
        
        # Presence boost
        if 'presence_boost' in eq_settings:
            boost = eq_settings['presence_boost']
            # Simplified bell filter
            # In production, use proper parametric EQ
        
        return audio
    
    def _apply_compression(self, audio: np.ndarray, compression: Dict) -> np.ndarray:
        """Apply compression"""
        # Simplified compression implementation
        threshold_db = compression.get('threshold', -20)
        ratio = compression.get('ratio', 3.0)
        attack_ms = compression.get('attack', 10)
        release_ms = compression.get('release', 100)
        
        # Convert to linear
        threshold = 10 ** (threshold_db / 20)
        
        # Simple compression algorithm
        # In production, use proper compressor implementation
        envelope = np.abs(audio)
        
        # Apply compression curve
        gain_reduction = np.where(envelope > threshold,
                                  (envelope - threshold) / ratio,
                                  0)
        compressed = audio * (1 - gain_reduction / (envelope + 1e-6))
        
        return compressed
    
    def _apply_reverb(self, audio: np.ndarray, reverb_settings: Dict) -> np.ndarray:
        """Apply reverb"""
        # Simplified reverb implementation
        reverb_time = reverb_settings.get('time', 2.0)
        mix = reverb_settings.get('mix', 0.15)
        
        # Simple impulse response
        impulse_length = int(reverb_time * self.sample_rate)
        impulse = np.exp(-np.linspace(0, 5, impulse_length))
        impulse *= np.random.randn(impulse_length) * 0.1
        
        # Convolve with impulse response
        wet = np.convolve(audio, impulse, mode='same')
        
        # Mix dry and wet
        result = (1 - mix) * audio + mix * wet
        
        return result
    
    def _mix_rhythm_section(self, drums_a: Optional[np.ndarray], drums_b: Optional[np.ndarray],
                           bass_a: Optional[np.ndarray], bass_b: Optional[np.ndarray],
                           rhythm_plan: Dict) -> Tuple[np.ndarray, np.ndarray]:
        """Mix rhythm section"""
        
        # Mix drums
        if rhythm_plan['primary_drums'] == 'A' and drums_a is not None:
            mixed_drums = drums_a
            if drums_b is not None and rhythm_plan.get('snare') == 'layer_both':
                # Layer snare from B
                mixed_drums = mixed_drums + drums_b * 0.3
        elif drums_b is not None:
            mixed_drums = drums_b
            if drums_a is not None and rhythm_plan.get('snare') == 'layer_both':
                mixed_drums = mixed_drums + drums_a * 0.3
        elif drums_a is not None:
            mixed_drums = drums_a
        else:
            mixed_drums = np.zeros(44100)
        
        # Mix bass
        if bass_a is not None and bass_b is not None and rhythm_plan.get('bass') == 'both':
            # Layer both basslines
            mixed_bass = (bass_a + bass_b) / 2
        elif bass_a is not None:
            mixed_bass = bass_a
        elif bass_b is not None:
            mixed_bass = bass_b
        else:
            mixed_bass = np.zeros(44100)
        
        return mixed_drums, mixed_bass
    
    def _mix_other_instruments(self, other_a: Optional[np.ndarray], other_b: Optional[np.ndarray],
                              melodic_plan: List[Dict]) -> np.ndarray:
        """Mix other instruments"""
        
        if other_a is not None and other_b is not None:
            # Simple mix for now
            mixed = (other_a + other_b) / 2
        elif other_a is not None:
            mixed = other_a
        elif other_b is not None:
            mixed = other_b
        else:
            mixed = np.zeros(44100)
        
        return mixed
    
    def _apply_dynamic_processing(self, vocals: np.ndarray, drums: np.ndarray,
                                 bass: np.ndarray, other: np.ndarray,
                                 processing: Dict) -> Dict[str, np.ndarray]:
        """Apply dynamic processing to stems"""
        
        processed = {
            'vocals': vocals,
            'drums': drums,
            'bass': bass,
            'other': other
        }
        
        # Apply sidechain compression if specified
        if 'sidechain' in processing:
            sidechain = processing['sidechain']
            
            if 'vocals' in sidechain['source']:
                # Use vocals to sidechain other elements
                if 'instruments' in sidechain['target']:
                    # Simple sidechain ducking
                    vocal_envelope = np.abs(vocals)
                    duck_amount = np.clip(vocal_envelope * 2, 0, 0.5)
                    
                    processed['drums'] = drums * (1 - duck_amount)
                    processed['other'] = other * (1 - duck_amount * 0.7)
                
                if 'bass' in sidechain['target']:
                    processed['bass'] = bass * 0.8  # Simple ducking
        
        return processed
    
    def _create_final_mix(self, stems: Dict[str, np.ndarray], balance: Dict) -> np.ndarray:
        """Create final mix from processed stems"""
        
        # Align lengths
        max_length = max(len(stem) for stem in stems.values())
        
        # Apply balance and mix
        mix = np.zeros(max_length)
        
        for stem_name, stem_audio in stems.items():
            if len(stem_audio) < max_length:
                # Pad with zeros
                padded = np.zeros(max_length)
                padded[:len(stem_audio)] = stem_audio
            else:
                padded = stem_audio[:max_length]
            
            # Apply gain from balance settings
            gain_db = balance.get(stem_name, 0)
            gain_linear = 10 ** (gain_db / 20)
            
            mix += padded * gain_linear
        
        # Normalize to prevent clipping
        peak = np.max(np.abs(mix))
        if peak > 1.0:
            mix = mix / peak * 0.95
        
        return mix
    
    def _apply_mastering(self, mix: np.ndarray, mastering: Dict) -> np.ndarray:
        """Apply mastering processing"""
        
        mastered = mix.copy()
        
        # Apply limiter
        if 'limiter' in mastering:
            limiter = mastering['limiter']
            threshold = 10 ** (limiter.get('threshold', -1.0) / 20)
            ceiling = 10 ** (limiter.get('ceiling', -0.3) / 20)
            
            # Simple limiter
            mastered = np.where(np.abs(mastered) > threshold,
                               np.sign(mastered) * threshold,
                               mastered)
            mastered = mastered * (ceiling / threshold)
        
        # Apply loudness normalization
        if 'loudness_target' in mastering:
            target_lufs = mastering['loudness_target']
            # Simplified loudness calculation
            current_loudness = np.mean(mastered ** 2)
            target_linear = 10 ** (target_lufs / 20)
            gain = target_linear / (current_loudness + 1e-6)
            mastered = mastered * min(gain, 10)  # Cap gain at 10x
        
        return mastered

# ============================================================================
# MAIN VOCALFUSION ENGINE
# ============================================================================

class VocalFusionEngine:
    """Main VocalFusion engine orchestrating the entire process"""
    
    def __init__(self, base_dir: str = "VocalFusion"):
        self.base_dir = Path(base_dir)
        self.sample_rate = 44100
        
        # Initialize engines
        self.stem_engine = StemSeparationEngine()
        self.song_engine = SongAnalysisEngine(self.sample_rate)
        self.compatibility_engine = CompatibilityEngine()
        self.arrangement_engine = ArrangementEngine()
        self.mixing_engine = MixingEngine(self.sample_rate)
        
        # Setup directory structure
        self._setup_directories()
    
    def _setup_directories(self):
        """Create directory structure"""
        directories = [
            'raw',
            'stems',
            'analysis',
            'compatibility',
            'arrangements',
            'mixes',
            'exports',
            'cache',
            'logs',
            'visualizations'
        ]
        
        for dir_name in directories:
            dir_path = self.base_dir / dir_name
            dir_path.mkdir(parents=True, exist_ok=True)
    
    def process_single_song(self, audio_path: Path, song_id: Optional[str] = None) -> Dict[str, Any]:
        """Process a single song through the full pipeline"""
        
        if song_id is None:
            song_id = f"song_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        
        print(f"\n{'='*60}")
        print(f"PROCESSING SONG: {song_id}")
        print(f"{'='*60}\n")
        
        # 1. Create song directory
        song_dir = self.base_dir / "raw" / song_id
        song_dir.mkdir(parents=True, exist_ok=True)
        
        # 2. Load and convert audio
        print("Step 1: Loading audio...")
        audio, sr = librosa.load(audio_path, sr=self.sample_rate, mono=False)
        
        # Save original
        original_path = song_dir / "original.wav"
        sf.write(original_path, audio.T if audio.ndim > 1 else audio, sr)
        
        # 3. Separate stems
        print("\nStep 2: Separating stems...")
        stems_dir = self.base_dir / "stems" / song_id
        stems_dir.mkdir(parents=True, exist_ok=True)
        
        stems = self.stem_engine.separate(original_path, stems_dir)
        
        # 4. Analyze song
        print("\nStep 3: Analyzing song...")
        analysis_dir = self.base_dir / "analysis" / song_id
        analysis_dir.mkdir(parents=True, exist_ok=True)
        
        # Load vocal stem for analysis
        vocal_path = stems.get('vocals')
        if vocal_path and vocal_path.exists():
            vocal_audio, _ = librosa.load(vocal_path, sr=self.sample_rate, mono=False)
            has_vocals = True
        else:
            vocal_audio = audio
            has_vocals = False
        
        # Run analysis
        song_analysis = self.song_engine.analyze_song(vocal_audio, song_id, has_vocals)
        
        # Save analysis
        analysis_path = analysis_dir / "analysis.json"
        with open(analysis_path, 'w') as f:
            json.dump(song_analysis.to_dict(), f, indent=2, default=str)
        
        # 5. Create visualizations
        print("\nStep 4: Creating visualizations...")
        self._create_visualizations(song_analysis, analysis_dir)
        
        print(f"\n✓ Song processing complete: {song_id}")
        
        return {
            'song_id': song_id,
            'analysis': song_analysis,
            'paths': {
                'original': str(original_path),
                'stems': {k: str(v) for k, v in stems.items()},
                'analysis': str(analysis_path)
            }
        }
    
    def fuse_two_songs(self, song_a_id: str, song_b_id: str, 
                      arrangement_strategy: Optional[str] = None) -> Dict[str, Any]:
        """Fuse two songs together"""
        
        print(f"\n{'='*60}")
        print(f"FUSING SONGS: {song_a_id} + {song_b_id}")
        print(f"{'='*60}\n")
        
        # 1. Load analyses
        print("Step 1: Loading song analyses...")
        analysis_a = self._load_song_analysis(song_a_id)
        analysis_b = self._load_song_analysis(song_b_id)
        
        # 2. Analyze compatibility
        print("\nStep 2: Analyzing compatibility...")
        compatibility_dir = self.base_dir / "compatibility" / f"{song_a_id}_{song_b_id}"
        compatibility_dir.mkdir(parents=True, exist_ok=True)
        
        compatibility = self.compatibility_engine.analyze_compatibility(analysis_a, analysis_b)
        
        # Save compatibility report
        compatibility_path = compatibility_dir / "compatibility_report.json"
        with open(compatibility_path, 'w') as f:
            json.dump(asdict(compatibility), f, indent=2, default=str)
        
        # 3. Create arrangement plan
        print("\nStep 3: Creating arrangement plan...")
        if arrangement_strategy:
            # Override automatic strategy
            compatibility.arrangement_strategies = [arrangement_strategy]
        
        arrangement_dir = self.base_dir / "arrangements" / f"{song_a_id}_{song_b_id}"
        arrangement_dir.mkdir(parents=True, exist_ok=True)
        
        arrangement_plan = self.arrangement_engine.create_arrangement_plan(
            analysis_a, analysis_b, compatibility
        )
        
        # Save arrangement plan
        arrangement_path = arrangement_dir / "arrangement_plan.json"
        with open(arrangement_path, 'w') as f:
            json.dump(arrangement_plan, f, indent=2, default=str)
        
        # 4. Load stems for mixing
        print("\nStep 4: Loading stems for mixing...")
        stems_a = self._load_stems(song_a_id)
        stems_b = self._load_stems(song_b_id)
        
        # 5. Create mix
        print("\nStep 5: Creating mix...")
        mix_dir = self.base_dir / "mixes" / f"{song_a_id}_{song_b_id}"
        mix_dir.mkdir(parents=True, exist_ok=True)
        
        # Get mixing plan from arrangement
        mixing_plan = arrangement_plan['mixing_plan']
        
        # Create mix
        mix_result = self.mixing_engine.create_mix(
            stems_a, stems_b, arrangement_plan, mixing_plan
        )
        
        # 6. Save mixes
        print("\nStep 6: Saving mixes...")
        for stem_name, audio in mix_result.items():
            if audio is not None:
                mix_path = mix_dir / f"{stem_name}.wav"
                sf.write(mix_path, audio, self.sample_rate)
        
        # 7. Create fusion report
        print("\nStep 7: Creating fusion report...")
        fusion_report = self._create_fusion_report(
            analysis_a, analysis_b, compatibility, arrangement_plan
        )
        
        report_path = mix_dir / "fusion_report.json"
        with open(report_path, 'w') as f:
            json.dump(fusion_report, f, indent=2, default=str)
        
        print(f"\n✓ Fusion complete: {song_a_id} + {song_b_id}")
        
        return {
            'song_a_id': song_a_id,
            'song_b_id': song_b_id,
            'compatibility': compatibility,
            'arrangement_plan': arrangement_plan,
            'fusion_report': fusion_report,
            'paths': {
                'compatibility': str(compatibility_path),
                'arrangement': str(arrangement_path),
                'mixes': str(mix_dir),
                'report': str(report_path)
            }
        }
    
    def _load_song_analysis(self, song_id: str) -> SongAnalysis:
        """Load song analysis from file"""
        analysis_path = self.base_dir / "analysis" / song_id / "analysis.json"
        
        if not analysis_path.exists():
            raise FileNotFoundError(f"Analysis not found for song: {song_id}")
        
        with open(analysis_path, 'r') as f:
            data = json.load(f)
        
        # Reconstruct SongAnalysis object
        # This is simplified - in production would need proper deserialization
        return SongAnalysis(**data)
    
    def _load_stems(self, song_id: str) -> Dict[str, np.ndarray]:
        """Load stems for a song"""
        stems_dir = self.base_dir / "stems" / song_id
        
        stems = {}
        stem_names = ['vocals', 'drums', 'bass', 'other']
        
        for name in stem_names:
            stem_path = stems_dir / f"{name}.wav"
            if stem_path.exists():
                audio, _ = librosa.load(stem_path, sr=self.sample_rate, mono=False)
                stems[name] = audio
        
        return stems
    
    def _create_visualizations(self, analysis: SongAnalysis, output_dir: Path):
        """Create visualizations for analysis"""
        try:
            import matplotlib.pyplot as plt
            import matplotlib
            
            # Set style
            plt.style.use('seaborn-v0_8-darkgrid')
            matplotlib.rcParams['figure.figsize'] = [12, 8]
            matplotlib.rcParams['font.size'] = 10
            
            # 1. Pitch contour plot
            if analysis.vocals and analysis.vocals.pitch_contour:
                fig, axes = plt.subplots(2, 1, figsize=(14, 10))
                
                # Pitch contour
                ax1 = axes[0]
                times = [p.time for p in analysis.vocals.pitch_contour if p.is_voiced]
                pitches = [p.frequency for p in analysis.vocals.pitch_contour if p.is_voiced]
                
                ax1.plot(times, pitches, 'b-', alpha=0.7, linewidth=1)
                ax1.set_xlabel('Time (s)')
                ax1.set_ylabel('Pitch (Hz)')
                ax1.set_title('Vocal Pitch Contour')
                ax1.grid(True, alpha=0.3)
                
                # Range visualization
                ax2 = axes[1]
                ax2.axhspan(analysis.vocals.range_low_hz, analysis.vocals.range_high_hz, 
                           alpha=0.3, color='gray', label='Full Range')
                ax2.axhspan(analysis.vocals.tessitura_low_hz, analysis.vocals.tessitura_high_hz,
                           alpha=0.5, color='blue', label='Tessitura')
                
                ax2.set_xlim(0, 1)
                ax2.set_ylim(50, 1000)
                ax2.set_ylabel('Frequency (Hz)')
                ax2.set_title(f'Vocal Range - {analysis.vocals.voice_type.value}')
                ax2.legend()
                ax2.grid(True, alpha=0.3)
                
                plt.tight_layout()
                plt.savefig(output_dir / "pitch_analysis.png", dpi=150, bbox_inches='tight')
                plt.close()
            
            # 2. Energy and emotion plot
            fig, axes = plt.subplots(3, 1, figsize=(14, 12))
            
            # Energy curve
            ax1 = axes[0]
            if analysis.energy_curve:
                times = np.linspace(0, analysis.duration, len(analysis.energy_curve))
                ax1.plot(times, analysis.energy_curve, 'g-', linewidth=2)
                ax1.set_xlabel('Time (s)')
                ax1.set_ylabel('Energy')
                ax1.set_title('Song Energy Curve')
                ax1.grid(True, alpha=0.3)
            
            # Section structure
            ax2 = axes[1]
            if analysis.sections:
                colors = {'intro': 'blue', 'verse': 'green', 'chorus': 'red', 
                         'bridge': 'orange', 'outro': 'purple'}
                
                for section in analysis.sections:
                    color = colors.get(section['type'], 'gray')
                    ax2.axvspan(section['start_time'], section['end_time'], 
                               alpha=0.3, color=color, label=section['type'])
                
                # Remove duplicate labels
                handles, labels = ax2.get_legend_handles_labels()
                by_label = dict(zip(labels, handles))
                ax2.legend(by_label.values(), by_label.keys())
                
                ax2.set_xlabel('Time (s)')
                ax2.set_title('Song Structure')
                ax2.grid(True, alpha=0.3)
            
            # Frequency spectrum
            ax3 = axes[2]
            if analysis.frequency_spectrum:
                bands = list(analysis.frequency_spectrum.keys())
                values = list(analysis.frequency_spectrum.values())
                
                colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b']
                ax3.bar(range(len(bands)), values, color=colors[:len(bands)])
                ax3.set_xticks(range(len(bands)))
                ax3.set_xticklabels(bands, rotation=45)
                ax3.set_ylabel('Energy Ratio')
                ax3.set_title('Frequency Spectrum Distribution')
                ax3.grid(True, alpha=0.3, axis='y')
            
            plt.tight_layout()
            plt.savefig(output_dir / "song_analysis.png", dpi=150, bbox_inches='tight')
            plt.close()
            
        except ImportError:
            print("Matplotlib not available, skipping visualizations")
    
    def _create_fusion_report(self, song_a: SongAnalysis, song_b: SongAnalysis,
                            compatibility: CompatibilityScore, 
                            arrangement_plan: Dict) -> Dict[str, Any]:
        """Create comprehensive fusion report"""
        
        report = {
            'fusion_summary': {
                'song_a': song_a.song_id,
                'song_b': song_b.song_id,
                'overall_score': compatibility.overall_score,
                'difficulty': arrangement_plan['difficulty_score'],
                'estimated_processing_time': arrangement_plan['estimated_processing_time'],
                'recommended_strategy': arrangement_plan['strategy']
            },
            'key_adjustments': {
                'original_key_a': song_a.key,
                'original_key_b': song_b.key,
                'recommended_key': compatibility.recommended_key,
                'transposition_a': compatibility.recommended_transposition_semitones,
                'transposition_b': -compatibility.recommended_transposition_semitones
            },
            'tempo_adjustments': {
                'original_tempo_a': song_a.tempo,
                'original_tempo_b': song_b.tempo,
                'tempo_adjustment_ratio': compatibility.recommended_tempo_adjustment_ratio
            },
            'vocal_analysis': {
                'voice_type_a': song_a.vocals.voice_type.value if song_a.vocals else 'none',
                'voice_type_b': song_b.vocals.voice_type.value if song_b.vocals else 'none',
                'blend_score': compatibility.vocal_blend_score,
                'range_compatibility': compatibility.range_compatibility
            },
            'arrangement_details': {
                'strategy': arrangement_plan['strategy'],
                'timeline_length': len(arrangement_plan['timeline']),
                'vocal_interactions': len(arrangement_plan['vocal_plan'].get('harmony_rules', [])),
                'instrumental_layers': len(arrangement_plan['instrumental_plan'].get('melodic_elements', []))
            },
            'mixing_recommendations': {
                'vocal_processing_steps': len(arrangement_plan['mixing_plan'].get('vocal_processing', {})),
                'dynamic_processing': arrangement_plan['mixing_plan'].get('dynamic_processing', {}),
                'mastering_targets': arrangement_plan['mixing_plan'].get('mastering', {})
            },
            'challenges': compatibility.challenges,
            'opportunities': compatibility.opportunities,
            'success_probability': self._calculate_success_probability(compatibility, arrangement_plan),
            'next_steps': self._suggest_next_steps(compatibility, arrangement_plan)
        }
        
        return report
    
    def _calculate_success_probability(self, compatibility: CompatibilityScore, 
                                     arrangement_plan: Dict) -> float:
        """Calculate probability of successful fusion"""
        
        # Base success probability from overall score
        success = compatibility.overall_score
        
        # Adjust for arrangement difficulty
        difficulty = arrangement_plan['difficulty_score']
        success *= (1 - difficulty * 0.3)  # Difficulty reduces success
        
        # Boost for good vocal blend
        if compatibility.vocal_blend_score > 0.8:
            success *= 1.2
        
        # Reduce for key challenges
        if compatibility.key_compatibility < 0.4:
            success *= 0.8
        
        return min(1.0, success)
    
    def _suggest_next_steps(self, compatibility: CompatibilityScore, 
                           arrangement_plan: Dict) -> List[str]:
        """Suggest next steps for the fusion"""
        
        steps = []
        
        if compatibility.key_compatibility < 0.6:
            steps.append(f"Transpose one song by {compatibility.recommended_transposition_semitones} semitones")
        
        if compatibility.tempo_compatibility < 0.7:
            steps.append(f"Adjust tempo by factor {compatibility.recommended_tempo_adjustment_ratio:.2f}")
        
        if compatibility.frequency_collision_zones:
            steps.append("Apply EQ carving in collision frequency zones")
        
        if arrangement_plan['difficulty_score'] > 0.7:
            steps.append("Consider simplifying the arrangement")
        
        steps.append("Review vocal timing alignment")
        steps.append("Adjust levels for optimal balance")
        steps.append("Apply recommended mixing processing")
        
        return steps

# ============================================================================
# WEB INTERFACE
# ============================================================================

from flask import Flask, request, jsonify, send_from_directory, render_template_string
import threading
import uuid

app = Flask(__name__)

# HTML template for web interface
HTML_TEMPLATE = '''
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>VocalFusion AI - Vocal Fusion System</title>
    <style>
        * {
            margin: 0;
            padding: 0;
            box-sizing: border-box;
        }
        
        body {
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, Oxygen, Ubuntu, sans-serif;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            min-height: 100vh;
            padding: 20px;
        }
        
        .container {
            max-width: 1200px;
            margin: 0 auto;
            background: white;
            border-radius: 20px;
            box-shadow: 0 20px 60px rgba(0,0,0,0.3);
            overflow: hidden;
        }
        
        header {
            background: linear-gradient(135deg, #4f46e5 0%, #7c3aed 100%);
            color: white;
            padding: 40px;
            text-align: center;
        }
        
        h1 {
            font-size: 3rem;
            margin-bottom: 10px;
            font-weight: 800;
        }
        
        .subtitle {
            font-size: 1.2rem;
            opacity: 0.9;
            margin-bottom: 30px;
        }
        
        .main-content {
            display: grid;
            grid-template-columns: 1fr 1fr;
            gap: 30px;
            padding: 40px;
        }
        
        @media (max-width: 768px) {
            .main-content {
                grid-template-columns: 1fr;
            }
        }
        
        .panel {
            background: #f8fafc;
            border-radius: 15px;
            padding: 30px;
            box-shadow: 0 10px 30px rgba(0,0,0,0.1);
        }
        
        h2 {
            color: #334155;
            margin-bottom: 20px;
            font-size: 1.8rem;
            border-bottom: 3px solid #4f46e5;
            padding-bottom: 10px;
        }
        
        .upload-area {
            border: 3px dashed #cbd5e1;
            border-radius: 10px;
            padding: 40px;
            text-align: center;
            cursor: pointer;
            transition: all 0.3s ease;
            margin-bottom: 20px;
        }
        
        .upload-area:hover {
            border-color: #4f46e5;
            background: #f1f5f9;
        }
        
        .upload-icon {
            font-size: 3rem;
            color: #64748b;
            margin-bottom: 15px;
        }
        
        .file-input {
            display: none;
        }
        
        .button {
            background: linear-gradient(135deg, #4f46e5 0%, #7c3aed 100%);
            color: white;
            border: none;
            padding: 15px 30px;
            border-radius: 10px;
            font-size: 1rem;
            font-weight: 600;
            cursor: pointer;
            transition: transform 0.2s ease, box-shadow 0.2s ease;
            width: 100%;
            margin-top: 10px;
        }
        
        .button:hover {
            transform: translateY(-2px);
            box-shadow: 0 10px 20px rgba(79, 70, 229, 0.3);
        }
        
        .button:active {
            transform: translateY(0);
        }
        
        .song-list {
            list-style: none;
            max-height: 300px;
            overflow-y: auto;
            margin-top: 20px;
        }
        
        .song-item {
            background: white;
            padding: 15px;
            border-radius: 8px;
            margin-bottom: 10px;
            border-left: 4px solid #4f46e5;
            display: flex;
            justify-content: space-between;
            align-items: center;
        }
        
        .song-info {
            flex: 1;
        }
        
        .song-name {
            font-weight: 600;
            color: #334155;
        }
        
        .song-details {
            font-size: 0.9rem;
            color: #64748b;
            margin-top: 5px;
        }
        
        .select-button {
            background: #10b981;
            color: white;
            border: none;
            padding: 8px 15px;
            border-radius: 6px;
            cursor: pointer;
            font-size: 0.9rem;
        }
        
        .progress-container {
            margin-top: 20px;
            background: #e2e8f0;
            border-radius: 10px;
            overflow: hidden;
            height: 20px;
        }
        
        .progress-bar {
            height: 100%;
            background: linear-gradient(90deg, #4f46e5, #7c3aed);
            width: 0%;
            transition: width 0.3s ease;
        }
        
        .status-message {
            text-align: center;
            padding: 20px;
            background: #f1f5f9;
            border-radius: 10px;
            margin-top: 20px;
            color: #475569;
        }
        
        .compatibility-score {
            font-size: 3rem;
            font-weight: 800;
            text-align: center;
            color: #4f46e5;
            margin: 20px 0;
        }
        
        .score-label {
            text-align: center;
            color: #64748b;
            margin-bottom: 30px;
        }
        
        .metric {
            display: flex;
            justify-content: space-between;
            margin-bottom: 15px;
            padding-bottom: 15px;
            border-bottom: 1px solid #e2e8f0;
        }
        
        .metric-name {
            color: #475569;
            font-weight: 500;
        }
        
        .metric-value {
            color: #4f46e5;
            font-weight: 600;
        }
        
        .recommendations {
            background: #f0f9ff;
            border-radius: 10px;
            padding: 20px;
            margin-top: 20px;
        }
        
        .recommendation-item {
            padding: 10px;
            background: white;
            border-radius: 6px;
            margin-bottom: 10px;
            border-left: 3px solid #0ea5e9;
        }
        
        footer {
            text-align: center;
            padding: 20px;
            color: #64748b;
            border-top: 1px solid #e2e8f0;
            font-size: 0.9rem;
        }
    </style>
</head>
<body>
    <div class="container">
        <header>
            <h1>VocalFusion AI</h1>
            <div class="subtitle">AI-Powered Vocal Analysis and Fusion System</div>
        </header>
        
        <div class="main-content">
            <!-- Left Panel: Song Upload and Analysis -->
            <div class="panel">
                <h2>Analyze Single Song</h2>
                
                <div class="upload-area" onclick="document.getElementById('fileInput').click()">
                    <div class="upload-icon">🎵</div>
                    <div style="font-size: 1.2rem; margin-bottom: 10px; color: #334155;">
                        Click to upload audio file
                    </div>
                    <div style="color: #64748b; font-size: 0.9rem;">
                        Supports WAV, MP3, FLAC, M4A
                    </div>
                </div>
                
                <input type="file" id="fileInput" class="file-input" accept="audio/*" onchange="handleFileUpload()">
                
                <button class="button" onclick="analyzeSong()" id="analyzeButton" disabled>
                    Analyze Song
                </button>
                
                <div id="progressContainer" class="progress-container" style="display: none;">
                    <div class="progress-bar" id="progressBar"></div>
                </div>
                
                <div id="statusMessage" class="status-message"></div>
                
                <h2 style="margin-top: 40px;">Analyzed Songs</h2>
                <ul class="song-list" id="songList">
                    <!-- Songs will be populated here -->
                </ul>
            </div>
            
            <!-- Right Panel: Song Fusion -->
            <div class="panel">
                <h2>Fuse Two Songs</h2>
                
                <div class="song-selection">
                    <div style="margin-bottom: 20px;">
                        <div style="color: #475569; margin-bottom: 10px; font-weight: 500;">Select Song A:</div>
                        <select class="button" id="songASelect" style="padding: 12px; background: white; color: #334155; border: 2px solid #cbd5e1;">
                            <option value="">Choose a song...</option>
                        </select>
                    </div>
                    
                    <div style="margin-bottom: 20px;">
                        <div style="color: #475569; margin-bottom: 10px; font-weight: 500;">Select Song B:</div>
                        <select class="button" id="songBSelect" style="padding: 12px; background: white; color: #334155; border: 2px solid #cbd5e1;">
                            <option value="">Choose a song...</option>
                        </select>
                    </div>
                </div>
                
                <button class="button" onclick="analyzeCompatibility()" id="compatibilityButton" disabled>
                    Analyze Compatibility
                </button>
                
                <div id="compatibilityResults" style="display: none;">
                    <div class="compatibility-score" id="overallScore">0.0</div>
                    <div class="score-label">Overall Compatibility Score</div>
                    
                    <div id="compatibilityMetrics">
                        <!-- Metrics will be populated here -->
                    </div>
                    
                    <div class="recommendations">
                        <h3>Recommended Actions</h3>
                        <div id="recommendationsList">
                            <!-- Recommendations will be populated here -->
                        </div>
                    </div>
                    
                    <button class="button" onclick="createFusion()" id="fusionButton" style="margin-top: 20px;">
                        Create Fusion
                    </button>
                </div>
                
                <div id="fusionProgress" style="display: none;">
                    <div class="progress-container" style="margin-top: 20px;">
                        <div class="progress-bar" id="fusionProgressBar"></div>
                    </div>
                    <div id="fusionStatus" class="status-message"></div>
                </div>
            </div>
        </div>
        
        <footer>
            VocalFusion AI v1.0 | Advanced AI-Powered Vocal Analysis and Fusion System
        </footer>
    </div>

    <script>
        let selectedFile = null;
        let analyzedSongs = [];
        
        // Load analyzed songs on page load
        window.onload = function() {
            loadAnalyzedSongs();
        };
        
        function handleFileUpload() {
            const fileInput = document.getElementById('fileInput');
            const analyzeButton = document.getElementById('analyzeButton');
            
            if (fileInput.files.length > 0) {
                selectedFile = fileInput.files[0];
                analyzeButton.disabled = false;
                analyzeButton.innerHTML = `Analyze: ${selectedFile.name}`;
            }
        }
        
        async function analyzeSong() {
            if (!selectedFile) return;
            
            const formData = new FormData();
            formData.append('file', selectedFile);
            formData.append('name', selectedFile.name.replace(/\.[^/.]+$/, ""));
            
            // Show progress
            document.getElementById('progressContainer').style.display = 'block';
            document.getElementById('statusMessage').innerHTML = 'Uploading and analyzing...';
            
            try {
                const response = await fetch('/api/upload', {
                    method: 'POST',
                    body: formData
                });
                
                const result = await response.json();
                
                if (result.success) {
                    // Update progress
                    updateProgress(100);
                    document.getElementById('statusMessage').innerHTML = 
                        `✓ Analysis complete! Song ID: ${result.song_id}`;
                    
                    // Reload song list
                    loadAnalyzedSongs();
                    
                    // Reset file input
                    document.getElementById('fileInput').value = '';
                    document.getElementById('analyzeButton').disabled = true;
                    document.getElementById('analyzeButton').innerHTML = 'Analyze Song';
                    selectedFile = null;
                } else {
                    document.getElementById('statusMessage').innerHTML = 
                        `Error: ${result.error}`;
                }
            } catch (error) {
                document.getElementById('statusMessage').innerHTML = 
                    `Error: ${error.message}`;
            }
        }
        
        async function loadAnalyzedSongs() {
            try {
                const response = await fetch('/api/songs');
                const result = await response.json();
                
                if (result.success) {
                    analyzedSongs = result.songs;
                    updateSongLists();
                }
            } catch (error) {
                console.error('Error loading songs:', error);
            }
        }
        
        function updateSongLists() {
            const songList = document.getElementById('songList');
            const songASelect = document.getElementById('songASelect');
            const songBSelect = document.getElementById('songBSelect');
            
            // Clear existing options
            songList.innerHTML = '';
            songASelect.innerHTML = '<option value="">Choose a song...</option>';
            songBSelect.innerHTML = '<option value="">Choose a song...</option>';
            
            analyzedSongs.forEach(song => {
                // Add to song list
                const listItem = document.createElement('li');
                listItem.className = 'song-item';
                listItem.innerHTML = `
                    <div class="song-info">
                        <div class="song-name">${song.song_id}</div>
                        <div class="song-details">
                            ${song.duration ? Math.round(song.duration) + 's' : ''} | 
                            ${song.key || 'Key: N/A'} | 
                            ${song.tempo ? Math.round(song.tempo) + ' BPM' : ''}
                        </div>
                    </div>
                    <button class="select-button" onclick="selectSong('${song.song_id}')">
                        Select
                    </button>
                `;
                songList.appendChild(listItem);
                
                // Add to dropdowns
                const optionA = document.createElement('option');
                optionA.value = song.song_id;
                optionA.textContent = `${song.song_id} (${song.key || 'N/A'})`;
                songASelect.appendChild(optionA.cloneNode(true));
                songBSelect.appendChild(optionA);
            });
            
            // Update compatibility button state
            updateCompatibilityButton();
        }
        
        function selectSong(songId) {
            // Auto-select in dropdowns
            document.getElementById('songASelect').value = songId;
            updateCompatibilityButton();
        }
        
        function updateCompatibilityButton() {
            const songA = document.getElementById('songASelect').value;
            const songB = document.getElementById('songBSelect').value;
            const button = document.getElementById('compatibilityButton');
            
            button.disabled = !(songA && songB && songA !== songB);
        }
        
        async function analyzeCompatibility() {
            const songA = document.getElementById('songASelect').value;
            const songB = document.getElementById('songBSelect').value;
            
            if (!songA || !songB || songA === songB) return;
            
            document.getElementById('statusMessage').innerHTML = 
                `Analyzing compatibility between ${songA} and ${songB}...`;
            
            try {
                const response = await fetch(`/api/compatibility/${songA}/${songB}`);
                const result = await response.json();
                
                if (result.success) {
                    displayCompatibilityResults(result.compatibility);
                } else {
                    document.getElementById('statusMessage').innerHTML = 
                        `Error: ${result.error}`;
                }
            } catch (error) {
                document.getElementById('statusMessage').innerHTML = 
                    `Error: ${error.message}`;
            }
        }
        
        function displayCompatibilityResults(compatibility) {
            // Show results section
            document.getElementById('compatibilityResults').style.display = 'block';
            
            // Update overall score
            document.getElementById('overallScore').textContent = 
                compatibility.overall_score.toFixed(2);
            
            // Update metrics
            const metricsContainer = document.getElementById('compatibilityMetrics');
            metricsContainer.innerHTML = '';
            
            const metrics = [
                ['Key Compatibility', compatibility.key_compatibility],
                ['Tempo Compatibility', compatibility.tempo_compatibility],
                ['Vocal Range Compatibility', compatibility.range_compatibility],
                ['Timbre Compatibility', compatibility.timbre_compatibility],
                ['Structure Compatibility', compatibility.structure_compatibility],
                ['Vocal Blend Score', compatibility.vocal_blend_score]
            ];
            
            metrics.forEach(([name, value]) => {
                const metricDiv = document.createElement('div');
                metricDiv.className = 'metric';
                metricDiv.innerHTML = `
                    <span class="metric-name">${name}</span>
                    <span class="metric-value">${value.toFixed(2)}</span>
                `;
                metricsContainer.appendChild(metricDiv);
            });
            
            // Update recommendations
            const recContainer = document.getElementById('recommendationsList');
            recContainer.innerHTML = '';
            
            if (compatibility.recommended_transposition_semitones !== 0) {
                const rec = document.createElement('div');
                rec.className = 'recommendation-item';
                rec.textContent = `Transpose by ${compatibility.recommended_transposition_semitones} semitones`;
                recContainer.appendChild(rec);
            }
            
            if (compatibility.recommended_tempo_adjustment_ratio !== 1.0) {
                const rec = document.createElement('div');
                rec.className = 'recommendation-item';
                rec.textContent = `Adjust tempo by factor ${compatibility.recommended_tempo_adjustment_ratio.toFixed(2)}`;
                recContainer.appendChild(rec);
            }
            
            compatibility.arrangement_strategies.forEach(strategy => {
                const rec = document.createElement('div');
                rec.className = 'recommendation-item';
                rec.textContent = `Use ${strategy} arrangement strategy`;
                recContainer.appendChild(rec);
            });
            
            // Scroll to results
            document.getElementById('compatibilityResults').scrollIntoView({ behavior: 'smooth' });
        }
        
        async function createFusion() {
            const songA = document.getElementById('songASelect').value;
            const songB = document.getElementById('songBSelect').value;
            
            if (!songA || !songB || songA === songB) return;
            
            // Show fusion progress
            document.getElementById('fusionProgress').style.display = 'block';
            document.getElementById('fusionStatus').innerHTML = 
                `Creating fusion between ${songA} and ${songB}...`;
            
            try {
                const response = await fetch(`/api/fuse/${songA}/${songB}`, {
                    method: 'POST'
                });
                
                const result = await response.json();
                
                if (result.success) {
                    updateFusionProgress(100);
                    document.getElementById('fusionStatus').innerHTML = 
                        `✓ Fusion complete! Download link: <a href="${result.download_url}">Click here</a>`;
                } else {
                    document.getElementById('fusionStatus').innerHTML = 
                        `Error: ${result.error}`;
                }
            } catch (error) {
                document.getElementById('fusionStatus').innerHTML = 
                    `Error: ${error.message}`;
            }
        }
        
        function updateProgress(percentage) {
            const progressBar = document.getElementById('progressBar');
            progressBar.style.width = `${percentage}%`;
        }
        
        function updateFusionProgress(percentage) {
            const progressBar = document.getElementById('fusionProgressBar');
            progressBar.style.width = `${percentage}%`;
        }
        
        // Set up WebSocket for real-time updates
        const protocol = window.location.protocol === 'https:' ? 'wss:' : 'ws:';
        const ws = new WebSocket(`${protocol}//${window.location.host}/ws`);
        
        ws.onmessage = function(event) {
            const data = JSON.parse(event.data);
            
            if (data.type === 'progress') {
                updateProgress(data.percentage);
            } else if (data.type === 'complete') {
                document.getElementById('statusMessage').innerHTML = 
                    `✓ ${data.message}`;
                loadAnalyzedSongs();
            }
        };
    </script>
</body>
</html>
'''

# Global engine instance
engine = VocalFusionEngine()

# Store active jobs
active_jobs = {}

@app.route('/')
def index():
    return render_template_string(HTML_TEMPLATE)

@app.route('/api/upload', methods=['POST'])
def upload_song():
    """API endpoint to upload and analyze a song"""
    if 'file' not in request.files:
        return jsonify({'success': False, 'error': 'No file provided'}), 400
    
    file = request.files['file']
    song_name = request.form.get('name', file.filename)
    
    # Generate unique ID
    song_id = f"{song_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    
    # Save to temporary file
    import tempfile
    temp_path = Path(tempfile.mktemp(suffix='.wav'))
    file.save(temp_path)
    
    # Create job
    job_id = str(uuid.uuid4())
    active_jobs[job_id] = {
        'status': 'processing',
        'song_id': song_id,
        'progress': 0
    }
    
    # Process in background
    def process_background():
        try:
            # Process song
            result = engine.process_single_song(temp_path, song_id)
            
            # Update job
            active_jobs[job_id].update({
                'status': 'completed',
                'progress': 100,
                'result': result
            })
            
        except Exception as e:
            active_jobs[job_id].update({
                'status': 'failed',
                'error': str(e)
            })
        finally:
            # Clean up temp file
            temp_path.unlink(missing_ok=True)
    
    # Start thread
    thread = threading.Thread(target=process_background)
    thread.start()
    
    return jsonify({
        'success': True,
        'job_id': job_id,
        'song_id': song_id,
        'message': f'Processing {song_name}'
    })

@app.route('/api/songs')
def list_songs():
    """API endpoint to list analyzed songs"""
    analysis_dir = Path("VocalFusion") / "analysis"
    
    songs = []
    
    if analysis_dir.exists():
        for song_dir in analysis_dir.iterdir():
            if song_dir.is_dir():
                analysis_file = song_dir / "analysis.json"
                if analysis_file.exists():
                    try:
                        with open(analysis_file, 'r') as f:
                            analysis = json.load(f)
                            songs.append({
                                'song_id': analysis.get('song_id', song_dir.name),
                                'duration': analysis.get('duration', 0),
                                'key': analysis.get('key', 'Unknown'),
                                'tempo': analysis.get('tempo', 0),
                                'voice_type': analysis.get('vocals', {}).get('voice_type', 'unknown')
                            })
                    except:
                        continue
    
    return jsonify({'success': True, 'songs': songs})

@app.route('/api/compatibility/<song_a>/<song_b>')
def get_compatibility(song_a, song_b):
    """API endpoint to analyze compatibility between two songs"""
    try:
        # Load analyses
        analysis_a = engine._load_song_analysis(song_a)
        analysis_b = engine._load_song_analysis(song_b)
        
        # Analyze compatibility
        compatibility = engine.compatibility_engine.analyze_compatibility(analysis_a, analysis_b)
        
        return jsonify({
            'success': True,
            'compatibility': asdict(compatibility)
        })
        
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500

@app.route('/api/fuse/<song_a>/<song_b>', methods=['POST'])
def fuse_songs(song_a, song_b):
    """API endpoint to fuse two songs"""
    try:
        # Fuse songs
        result = engine.fuse_two_songs(song_a, song_b)
        
        # Create download link
        mix_dir = Path("VocalFusion") / "mixes" / f"{song_a}_{song_b}"
        mix_file = mix_dir / "full_mix.wav"
        
        if mix_file.exists():
            download_url = f"/download/{song_a}_{song_b}"
        else:
            download_url = None
        
        return jsonify({
            'success': True,
            'fusion_id': f"{song_a}_{song_b}",
            'download_url': download_url,
            'message': 'Fusion created successfully'
        })
        
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500

@app.route('/download/<fusion_id>')
def download_fusion(fusion_id):
    """Download fusion result"""
    mix_file = Path("VocalFusion") / "mixes" / fusion_id / "full_mix.wav"
    
    if mix_file.exists():
        return send_from_directory(mix_file.parent, mix_file.name, as_attachment=True)
    else:
        return "File not found", 404

@app.route('/api/job/<job_id>')
def get_job_status(job_id):
    """Get job status"""
    if job_id not in active_jobs:
        return jsonify({'success': False, 'error': 'Job not found'}), 404
    
    return jsonify({'success': True, 'job': active_jobs[job_id]})

# ============================================================================
# COMMAND LINE INTERFACE
# ============================================================================

import argparse
import sys

def main():
    parser = argparse.ArgumentParser(
        description='VocalFusion AI - Advanced Vocal Analysis and Fusion System',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog='''
Examples:
  %(prog)s analyze song.mp3 --name "My Song"
  %(prog)s list
  %(prog)s compatibility song1 song2
  %(prog)s fuse song1 song2 --strategy harmony_focused
  %(prog)s serve --port 8080
        '''
    )
    
    subparsers = parser.add_subparsers(dest='command', help='Commands')
    
    # Analyze command
    analyze_parser = subparsers.add_parser('analyze', help='Analyze a single song')
    analyze_parser.add_argument('file', help='Audio file to analyze')
    analyze_parser.add_argument('--name', help='Song name (optional)')
    
    # List command
    list_parser = subparsers.add_parser('list', help='List analyzed songs')
    
    # Compatibility command
    comp_parser = subparsers.add_parser('compatibility', help='Analyze compatibility between two songs')
    comp_parser.add_argument('song_a', help='First song ID')
    comp_parser.add_argument('song_b', help='Second song ID')
    
    # Fuse command
    fuse_parser = subparsers.add_parser('fuse', help='Fuse two songs together')
    fuse_parser.add_argument('song_a', help='First song ID')
    fuse_parser.add_argument('song_b', help='Second song ID')
    fuse_parser.add_argument('--strategy', help='Arrangement strategy', 
                           choices=['harmony_focused', 'call_and_response', 'counterpoint', 'layered'])
    
    # Serve command
    serve_parser = subparsers.add_parser('serve', help='Start web server')
    serve_parser.add_argument('--port', type=int, default=5000, help='Port to listen on')
    serve_parser.add_argument('--host', default='127.0.0.1', help='Host to bind to')
    
    args = parser.parse_args()
    
    if args.command == 'analyze':
        engine = VocalFusionEngine()
        result = engine.process_single_song(Path(args.file), args.name)
        
        print(f"\n{'='*60}")
        print(f"ANALYSIS COMPLETE")
        print(f"{'='*60}")
        print(f"Song ID: {result['song_id']}")
        print(f"Duration: {result['analysis'].duration:.1f}s")
        print(f"Key: {result['analysis'].key}")
        print(f"Tempo: {result['analysis'].tempo:.1f} BPM")
        
        if result['analysis'].vocals:
            print(f"Voice Type: {result['analysis'].vocals.voice_type.value}")
            print(f"Vocal Range: {result['analysis'].vocals.range_low_hz:.1f} - {result['analysis'].vocals.range_high_hz:.1f} Hz")
        
        print(f"\nAnalysis saved to: {result['paths']['analysis']}")
        print(f"Stems saved to: {result['paths']['stems']['vocals']}")
        
    elif args.command == 'list':
        engine = VocalFusionEngine()
        analysis_dir = Path("VocalFusion") / "analysis"
        
        if not analysis_dir.exists():
            print("No songs analyzed yet.")
            return
        
        print(f"\n{'='*60}")
        print(f"ANALYZED SONGS")
        print(f"{'='*60}")
        
        for song_dir in sorted(analysis_dir.iterdir()):
            if song_dir.is_dir():
                analysis_file = song_dir / "analysis.json"
                if analysis_file.exists():
                    try:
                        with open(analysis_file, 'r') as f:
                            analysis = json.load(f)
                            print(f"\n{song_dir.name}:")
                            print(f"  Duration: {analysis.get('duration', 0):.1f}s")
                            print(f"  Key: {analysis.get('key', 'Unknown')}")
                            print(f"  Tempo: {analysis.get('tempo', 0):.1f} BPM")
                            
                            if 'vocals' in analysis and analysis['vocals']:
                                print(f"  Voice Type: {analysis['vocals'].get('voice_type', 'unknown')}")
                    except:
                        print(f"\n{song_dir.name}: (Error reading analysis)")
        
        print(f"\n{'='*60}")
        
    elif args.command == 'compatibility':
        engine = VocalFusionEngine()
        
        try:
            analysis_a = engine._load_song_analysis(args.song_a)
            analysis_b = engine._load_song_analysis(args.song_b)
            
            compatibility = engine.compatibility_engine.analyze_compatibility(analysis_a, analysis_b)
            
            print(f"\n{'='*60}")
            print(f"COMPATIBILITY ANALYSIS: {args.song_a} + {args.song_b}")
            print(f"{'='*60}")
            print(f"\nOverall Score: {compatibility.overall_score:.2f}/1.0")
            print(f"Difficulty: {compatibility.difficulty_score:.2f}/1.0")
            
            print(f"\nKey Compatibility: {compatibility.key_compatibility:.2f}")
            print(f"  Original Keys: {analysis_a.key} / {analysis_b.key}")
            print(f"  Recommended: {compatibility.recommended_key}")
            print(f"  Transposition: {compatibility.recommended_transposition_semitones} semitones")
            
            print(f"\nTempo Compatibility: {compatibility.tempo_compatibility:.2f}")
            print(f"  Original Tempos: {analysis_a.tempo:.1f} / {analysis_b.tempo:.1f} BPM")
            print(f"  Adjustment Ratio: {compatibility.recommended_tempo_adjustment_ratio:.2f}")
            
            print(f"\nVocal Compatibility:")
            print(f"  Range: {compatibility.range_compatibility:.2f}")
            print(f"  Timbre: {compatibility.timbre_compatibility:.2f}")
            print(f"  Blend: {compatibility.vocal_blend_score:.2f}")
            
            print(f"\nArrangement Strategies:")
            for strategy in compatibility.arrangement_strategies[:3]:
                print(f"  • {strategy}")
            
            print(f"\nChallenges:")
            for challenge in compatibility.challenges[:3]:
                print(f"  • {challenge}")
            
            print(f"\nOpportunities:")
            for opportunity in compatibility.opportunities[:3]:
                print(f"  • {opportunity}")
            
            print(f"\n{'='*60}")
            
        except Exception as e:
            print(f"Error: {e}")
            
    elif args.command == 'fuse':
        engine = VocalFusionEngine()
        
        try:
            result = engine.fuse_two_songs(args.song_a, args.song_b, args.strategy)
            
            print(f"\n{'='*60}")
            print(f"FUSION COMPLETE: {args.song_a} + {args.song_b}")
            print(f"{'='*60}")
            print(f"\nOverall Score: {result['compatibility'].overall_score:.2f}/1.0")
            print(f"Difficulty: {result['arrangement_plan']['difficulty_score']:.2f}/1.0")
            print(f"Strategy: {result['arrangement_plan']['strategy']}")
            
            print(f"\nFiles Created:")
            for stem in ['vocals', 'drums', 'bass', 'other', 'full_mix']:
                mix_path = Path(result['paths']['mixes']) / f"{stem}.wav"
                if mix_path.exists():
                    print(f"  • {stem}.wav")
            
            print(f"\nReport: {result['paths']['report']}")
            print(f"\nEstimated Processing Time: {result['arrangement_plan']['estimated_processing_time']}")
            
            print(f"\nNext Steps:")
            for step in result['fusion_report']['next_steps'][:5]:
                print(f"  • {step}")
            
            print(f"\n{'='*60}")
            
        except Exception as e:
            print(f"Error: {e}")
            
    elif args.command == 'serve':
        print(f"\n{'='*60}")
        print(f"VocalFusion AI Web Server")
        print(f"{'='*60}")
        print(f"\nStarting server on http://{args.host}:{args.port}")
        print(f"Press Ctrl+C to stop\n")
        
        # Disable Flask's debug mode for production
        app.run(host=args.host, port=args.port, debug=False)
        
    else:
        parser.print_help()

if __name__ == "__main__":
    # Create VocalFusion directory
    engine = VocalFusionEngine()
    
    # Run CLI if arguments provided, otherwise start web server
    if len(sys.argv) > 1:
        main()
    else:
        print(f"\n{'='*60}")
        print(f"VocalFusion AI - Advanced Vocal Analysis and Fusion System")
        print(f"{'='*60}")
        print(f"\nNo command specified. Starting web server...")
        print(f"Open http://localhost:5000 to access the web interface")
        print(f"Or use 'python vocalfusion.py --help' for CLI options\n")
        
        app.run(host='127.0.0.1', port=5000, debug=False)
