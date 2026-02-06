"""
VocalFusion - Phase 1: Foundation
A simplified but functional version to validate the core pipeline.
"""

import os
import json
import numpy as np
import librosa
import soundfile as sf
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import tempfile
import shutil
from datetime import datetime

# Data models
from dataclasses import dataclass, asdict
from enum import Enum
import hashlib

# -------------------------------------------------------------------
# DATA MODELS (Following your architecture)
# -------------------------------------------------------------------

class VoiceType(str, Enum):
    SOPRANO = "soprano"
    MEZZO_SOPRANO = "mezzo-soprano"
    ALTO = "alto"
    TENOR = "tenor"
    BARITONE = "baritone"
    BASS = "bass"
    UNKNOWN = "unknown"

class SongSection(str, Enum):
    INTRO = "intro"
    VERSE = "verse"
    PRE_CHORUS = "pre_chorus"
    CHORUS = "chorus"
    BRIDGE = "bridge"
    OUTRO = "outro"
    INSTRUMENTAL = "instrumental"

@dataclass
class AudioMetadata:
    """Metadata for an audio file"""
    filename: str
    duration: float
    sample_rate: int
    channels: int
    bit_depth: Optional[int]
    format: str
    file_size: int
    md5_hash: str
    
    @classmethod
    def from_file(cls, filepath: Path):
        """Create metadata from audio file"""
        # Get basic file info
        stat = filepath.stat()
        with open(filepath, 'rb') as f:
            md5 = hashlib.md5(f.read()).hexdigest()
        
        # Load audio to get technical details
        y, sr = librosa.load(filepath, sr=None, mono=False)
        duration = len(y) / sr
        
        # Try to get bit depth from soundfile
        import soundfile as sf
        info = sf.info(str(filepath))
        
        return cls(
            filename=filepath.name,
            duration=duration,
            sample_rate=sr,
            channels=info.channels,
            bit_depth=info.subtype,
            format=filepath.suffix[1:],
            file_size=stat.st_size,
            md5_hash=md5
        )

@dataclass
class VocalAnalysis:
    """Core vocal analysis results"""
    pitch_contour: List[float]
    pitch_times: List[float]
    pitch_confidence: List[float]
    
    notes: List[Dict]  # Note events
    phrases: List[Dict]  # Vocal phrases
    silences: List[Dict]  # Silent sections
    
    # Statistics
    range_low: float  # Hz
    range_high: float
    tessitura_low: float
    tessitura_high: float
    voice_type: VoiceType
    pitch_accuracy: float
    
    # Timbre
    mfcc_mean: List[float]
    mfcc_std: List[float]
    spectral_centroid: float
    brightness: float
    
    # Timing
    onsets: List[float]
    rhythm_pattern: List[float]
    bpm: float

@dataclass
class SongAnalysis:
    """Complete analysis of a song"""
    song_id: str
    metadata: AudioMetadata
    structure: List[Dict]
    vocals: VocalAnalysis
    key: str
    tempo: float
    time_signature: str
    chord_progression: List[str]
    energy_curve: List[float]
    
    @classmethod
    def from_dict(cls, data: Dict):
        return cls(**data)
    
    def to_dict(self):
        return asdict(self)

@dataclass 
class CompatibilityScore:
    """Compatibility analysis between two songs"""
    song_a_id: str
    song_b_id: str
    
    # Individual scores (0-1)
    key_compatibility: float
    tempo_compatibility: float
    range_compatibility: float
    timbre_compatibility: float
    structure_compatibility: float
    
    # Overall
    overall_score: float
    recommended_transposition: int  # semitones
    recommended_tempo_adjustment: float  # ratio
    
    # Recommendations
    arrangement_strategies: List[str]
    challenges: List[str]
    opportunities: List[str]

# -------------------------------------------------------------------
# CORE PROCESSING ENGINE
# -------------------------------------------------------------------

class VocalFusionEngine:
    """Main processing engine following your architecture"""
    
    def __init__(self, base_dir: str = "VocalFusion"):
        self.base_dir = Path(base_dir)
        self._setup_directories()
        
    def _setup_directories(self):
        """Create directory structure from your design"""
        dirs = [
            "raw", "stems", "analysis", "compatibility",
            "arrangements", "renders", "cache", "logs"
        ]
        
        for dir_name in dirs:
            (self.base_dir / dir_name).mkdir(parents=True, exist_ok=True)
    
    def process_song(self, audio_path: Path, song_id: Optional[str] = None) -> Dict:
        """
        Full processing pipeline for a single song
        Returns: Dictionary with paths to all processed files
        """
        if song_id is None:
            song_id = f"song_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        
        # 1. Store in raw/ (preserve original)
        raw_dir = self.base_dir / "raw" / song_id
        raw_dir.mkdir(parents=True, exist_ok=True)
        
        original_path = raw_dir / "original.wav"
        self._convert_to_wav(audio_path, original_path)
        
        # 2. Extract metadata
        metadata = AudioMetadata.from_file(original_path)
        with open(raw_dir / "metadata.json", 'w') as f:
            json.dump(asdict(metadata), f, indent=2)
        
        # 3. Separate stems
        stems = self._separate_stems(original_path, song_id)
        
        # 4. Analyze vocals
        vocal_analysis = self._analyze_vocals(stems['vocals'], song_id)
        
        # 5. Analyze full song
        full_analysis = self._analyze_song(original_path, vocal_analysis, song_id)
        
        return {
            'song_id': song_id,
            'paths': {
                'raw': str(raw_dir),
                'stems': stems,
                'analysis': str(self.base_dir / "analysis" / song_id)
            },
            'metadata': asdict(metadata),
            'analysis': full_analysis
        }
    
    def _convert_to_wav(self, source: Path, target: Path):
        """Convert any audio format to WAV for processing"""
        if source.suffix.lower() == '.wav':
            shutil.copy2(source, target)
        else:
            # Use pydub for conversion
            from pydub import AudioSegment
            audio = AudioSegment.from_file(source)
            audio.export(target, format='wav')
    
    def _separate_stems(self, audio_path: Path, song_id: str) -> Dict[str, str]:
        """
        Separate audio into stems using Demucs
        Returns: Dict with paths to stems
        """
        stems_dir = self.base_dir / "stems" / song_id
        stems_dir.mkdir(parents=True, exist_ok=True)
        
        print(f"Separating stems for {song_id}...")
        
        # Use Demucs (simplified for now)
        # In production, you'd use the full Demucs pipeline
        temp_output = stems_dir / "demucs_output"
        
        # For MVP, we'll create placeholder stems by processing the original
        # In Phase 2, we'll integrate actual Demucs
        stems = {
            'vocals': stems_dir / "vocals.wav",
            'drums': stems_dir / "drums.wav", 
            'bass': stems_dir / "bass.wav",
            'other': stems_dir / "other.wav"
        }
        
        # Create placeholder stems for now
        y, sr = librosa.load(audio_path, sr=None, mono=False)
        
        # Simple band separation for MVP
        if y.ndim > 1:
            y = librosa.to_mono(y)
        
        # Create mock stems (will be replaced with real Demucs)
        sf.write(stems['vocals'], y * 0.7, sr)  # Mock vocals
        sf.write(stems['drums'], y * 0.3, sr)   # Mock drums
        sf.write(stems['bass'], y * 0.2, sr)    # Mock bass
        sf.write(stems['other'], y * 0.4, sr)   # Mock other
        
        # Save separation metadata
        metadata = {
            'separation_model': 'demucs_placeholder',
            'timestamp': datetime.now().isoformat(),
            'stems_generated': list(stems.keys())
        }
        
        with open(stems_dir / "separation_metadata.json", 'w') as f:
            json.dump(metadata, f, indent=2)
        
        return {k: str(v) for k, v in stems.items()}
    
    def _analyze_vocals(self, vocal_path: Path, song_id: str) -> VocalAnalysis:
        """
        Deep vocal analysis using Parselmouth (Praat) for accurate pitch tracking
        """
        analysis_dir = self.base_dir / "analysis" / song_id / "vocals"
        analysis_dir.mkdir(parents=True, exist_ok=True)
        
        print(f"Analyzing vocals for {song_id}...")
        
        # Load vocal stem
        y, sr = librosa.load(vocal_path, sr=None)
        
        # 1. Pitch analysis using Parselmouth (more accurate than librosa)
        pitch_contour, pitch_times, pitch_confidence = self._extract_pitch_parselmouth(vocal_path)
        
        # 2. Note detection
        notes = self._extract_notes(pitch_contour, pitch_times)
        
        # 3. Phrase detection
        phrases = self._extract_phrases(y, sr, pitch_contour)
        
        # 4. Silence detection
        silences = self._extract_silences(y, sr)
        
        # 5. Range analysis
        range_low, range_high = self._calculate_vocal_range(pitch_contour)
        tessitura_low, tessitura_high = self._calculate_tessitura(pitch_contour)
        
        # 6. Timbre analysis
        mfcc_mean, mfcc_std = self._extract_mfcc(y, sr)
        spectral_centroid = librosa.feature.spectral_centroid(y=y, sr=sr).mean()
        brightness = self._calculate_brightness(y, sr)
        
        # 7. Timing analysis
        onsets = librosa.onset.onset_detect(y=y, sr=sr, units='time')
        tempo, _ = librosa.beat.beat_track(y=y, sr=sr)
        
        # 8. Voice type classification
        voice_type = self._classify_voice_type(range_low, range_high)
        
        analysis = VocalAnalysis(
            pitch_contour=pitch_contour.tolist() if hasattr(pitch_contour, 'tolist') else pitch_contour,
            pitch_times=pitch_times.tolist() if hasattr(pitch_times, 'tolist') else pitch_times,
            pitch_confidence=pitch_confidence.tolist() if hasattr(pitch_confidence, 'tolist') else pitch_confidence,
            notes=notes,
            phrases=phrases,
            silences=silences,
            range_low=float(range_low),
            range_high=float(range_high),
            tessitura_low=float(tessitura_low),
            tessitura_high=float(tessitura_high),
            voice_type=voice_type,
            pitch_accuracy=0.9,  # Placeholder
            mfcc_mean=mfcc_mean,
            mfcc_std=mfcc_std,
            spectral_centroid=float(spectral_centroid),
            brightness=float(brightness),
            onsets=onsets.tolist(),
            rhythm_pattern=[0.5, 0.25, 0.25],  # Placeholder
            bpm=float(tempo)
        )
        
        # Save vocal analysis
        with open(analysis_dir / "vocal_analysis.json", 'w') as f:
            json.dump(asdict(analysis), f, indent=2)
        
        # Create visualizations
        self._create_vocal_visualizations(analysis, analysis_dir)
        
        return analysis
    
    def _extract_pitch_parselmouth(self, audio_path: Path) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Extract pitch using Parselmouth (Praat) - more accurate for vocals
        """
        try:
            import parselmouth
            snd = parselmouth.Sound(str(audio_path))
            pitch = snd.to_pitch()
            
            # Get pitch values
            pitch_times = pitch.xs()
            pitch_values = pitch.selected_array['frequency']
            pitch_confidence = pitch.selected_array['strength']
            
            # Convert 0 to NaN for unvoiced regions
            pitch_values[pitch_values == 0] = np.nan
            
            return pitch_values, pitch_times, pitch_confidence
            
        except ImportError:
            # Fallback to librosa if parselmouth not available
            print("Parselmouth not available, using librosa fallback")
            y, sr = librosa.load(audio_path, sr=None)
            
            # Use librosa's pitch tracking
            pitches, magnitudes = librosa.piptrack(y=y, sr=sr)
            
            # Get max pitch at each time frame
            pitch_contour = []
            for i in range(pitches.shape[1]):
                index = magnitudes[:, i].argmax()
                pitch = pitches[index, i]
                pitch_contour.append(pitch if pitch > 0 else np.nan)
            
            pitch_times = librosa.frames_to_time(range(pitches.shape[1]), sr=sr)
            pitch_confidence = np.ones_like(pitch_times) * 0.8  # Placeholder
            
            return np.array(pitch_contour), pitch_times, pitch_confidence
    
    def _extract_notes(self, pitch_contour: np.ndarray, times: np.ndarray) -> List[Dict]:
        """Extract discrete notes from pitch contour"""
        notes = []
        
        # Simple note extraction (will be improved)
        valid_pitches = pitch_contour[~np.isnan(pitch_contour)]
        if len(valid_pitches) > 0:
            # Convert pitch to MIDI note numbers
            midi_notes = librosa.hz_to_midi(valid_pitches)
            
            # Round to nearest note
            rounded_notes = np.round(midi_notes)
            
            # Group consecutive same notes
            current_note = None
            start_time = 0
            for i, (note, time) in enumerate(zip(rounded_notes, times[~np.isnan(pitch_contour)])):
                if current_note is None:
                    current_note = note
                    start_time = time
                elif note != current_note or i == len(rounded_notes) - 1:
                    notes.append({
                        'pitch_midi': float(current_note),
                        'pitch_hz': float(librosa.midi_to_hz(current_note)),
                        'pitch_name': librosa.midi_to_note(current_note),
                        'start_time': float(start_time),
                        'end_time': float(time),
                        'duration': float(time - start_time)
                    })
                    current_note = note
                    start_time = time
        
        return notes
    
    def _extract_phrases(self, audio: np.ndarray, sr: int, pitch_contour: np.ndarray) -> List[Dict]:
        """Extract vocal phrases"""
        phrases = []
        
        # Simple energy-based phrase detection
        energy = librosa.feature.rms(y=audio)[0]
        energy_times = librosa.frames_to_time(range(len(energy)), sr=sr, hop_length=512)
        
        # Threshold for vocal activity
        threshold = np.percentile(energy, 30)
        vocal_activity = energy > threshold
        
        # Find continuous vocal regions
        in_phrase = False
        phrase_start = 0
        
        for i, (active, time) in enumerate(zip(vocal_activity, energy_times)):
            if active and not in_phrase:
                in_phrase = True
                phrase_start = time
            elif not active and in_phrase:
                in_phrase = False
                if time - phrase_start > 0.5:  # Minimum phrase length
                    phrases.append({
                        'start_time': float(phrase_start),
                        'end_time': float(time),
                        'duration': float(time - phrase_start)
                    })
        
        return phrases
    
    def _extract_silences(self, audio: np.ndarray, sr: int) -> List[Dict]:
        """Extract silent sections"""
        silences = []
        
        # Simple silence detection
        energy = librosa.feature.rms(y=audio)[0]
        energy_times = librosa.frames_to_time(range(len(energy)), sr=sr, hop_length=512)
        
        threshold = np.percentile(energy, 10)
        silent = energy < threshold
        
        # Find continuous silent regions
        in_silence = False
        silence_start = 0
        
        for i, (is_silent, time) in enumerate(zip(silent, energy_times)):
            if is_silent and not in_silence:
                in_silence = True
                silence_start = time
            elif not is_silent and in_silence:
                in_silence = False
                if time - silence_start > 0.2:  # Minimum silence length
                    silences.append({
                        'start_time': float(silence_start),
                        'end_time': float(time),
                        'duration': float(time - silence_start),
                        'type': 'breath' if time - silence_start < 1.0 else 'phrase_gap'
                    })
        
        return silences
    
    def _calculate_vocal_range(self, pitch_contour: np.ndarray) -> Tuple[float, float]:
        """Calculate vocal range from pitch contour"""
        valid_pitches = pitch_contour[~np.isnan(pitch_contour)]
        if len(valid_pitches) == 0:
            return 100.0, 500.0  # Default range
        
        # Get lowest and highest sung pitches
        range_low = np.percentile(valid_pitches, 5)
        range_high = np.percentile(valid_pitches, 95)
        
        return float(range_low), float(range_high)
    
    def _calculate_tessitura(self, pitch_contour: np.ndarray) -> Tuple[float, float]:
        """Calculate tessitura (most comfortable range)"""
        valid_pitches = pitch_contour[~np.isnan(pitch_contour)]
        if len(valid_pitches) == 0:
            return 150.0, 350.0
        
        # Tessitura is middle 50% of pitches
        tessitura_low = np.percentile(valid_pitches, 25)
        tessitura_high = np.percentile(valid_pitches, 75)
        
        return float(tessitura_low), float(tessitura_high)
    
    def _extract_mfcc(self, audio: np.ndarray, sr: int, n_mfcc: int = 13) -> Tuple[List[float], List[float]]:
        """Extract MFCC features"""
        mfccs = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=n_mfcc)
        return mfccs.mean(axis=1).tolist(), mfccs.std(axis=1).tolist()
    
    def _calculate_brightness(self, audio: np.ndarray, sr: int) -> float:
        """Calculate spectral brightness"""
        spectral_centroid = librosa.feature.spectral_centroid(y=audio, sr=sr)
        return float(spectral_centroid.mean() / (sr / 2))  # Normalize to 0-1
    
    def _classify_voice_type(self, range_low: float, range_high: float) -> VoiceType:
        """Classify voice type based on range"""
        # Convert to MIDI for classification
        low_midi = librosa.hz_to_midi(range_low)
        high_midi = librosa.hz_to_midi(range_high)
        avg_midi = (low_midi + high_midi) / 2
        
        if avg_midi > 72:  # Above C5
            return VoiceType.SOPRANO
        elif avg_midi > 66:  # Above F4
            return VoiceType.MEZZO_SOPRANO
        elif avg_midi > 60:  # Above C4
            return VoiceType.ALTO
        elif avg_midi > 55:  # Above G3
            return VoiceType.TENOR
        elif avg_midi > 50:  # Above D3
            return VoiceType.BARITONE
        else:
            return VoiceType.BASS
    
    def _create_vocal_visualizations(self, analysis: VocalAnalysis, output_dir: Path):
        """Create visualization plots for vocal analysis"""
        import matplotlib.pyplot as plt
        
        # 1. Pitch contour plot
        plt.figure(figsize=(12, 4))
        valid_indices = ~np.isnan(analysis.pitch_contour)
        plt.plot(np.array(analysis.pitch_times)[valid_indices], 
                np.array(analysis.pitch_contour)[valid_indices], 
                'b-', alpha=0.7, linewidth=1)
        plt.xlabel('Time (s)')
        plt.ylabel('Pitch (Hz)')
        plt.title('Vocal Pitch Contour')
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(output_dir / "pitch_contour.png", dpi=150)
        plt.close()
        
        # 2. Range visualization
        plt.figure(figsize=(8, 6))
        # Create piano keyboard visualization
        # Simplified for now
        plt.text(0.5, 0.5, f"Range: {analysis.range_low:.1f} - {analysis.range_high:.1f} Hz\n"
                          f"Voice Type: {analysis.voice_type.value}\n"
                          f"Tessitura: {analysis.tessitura_low:.1f} - {analysis.tessitura_high:.1f} Hz",
                 ha='center', va='center', fontsize=12)
        plt.axis('off')
        plt.savefig(output_dir / "vocal_range.png", dpi=150)
        plt.close()
    
    def _analyze_song(self, audio_path: Path, vocal_analysis: VocalAnalysis, song_id: str) -> SongAnalysis:
        """Complete song analysis"""
        analysis_dir = self.base_dir / "analysis" / song_id
        analysis_dir.mkdir(parents=True, exist_ok=True)
        
        # Load audio
        y, sr = librosa.load(audio_path, sr=None)
        
        # Analyze song structure
        structure = self._analyze_structure(y, sr)
        
        # Detect key
        key = self._detect_key(y, sr)
        
        # Detect tempo
        tempo, _ = librosa.beat.beat_track(y=y, sr=sr)
        
        # Detect time signature (simplified)
        time_sig = self._detect_time_signature(y, sr)
        
        # Extract chord progression (simplified)
        chords = self._extract_chords(y, sr)
        
        # Calculate energy curve
        energy = librosa.feature.rms(y=y)[0]
        energy_times = librosa.frames_to_time(range(len(energy)), sr=sr, hop_length=512)
        energy_curve = [{'time': float(t), 'energy': float(e)} 
                       for t, e in zip(energy_times, energy)]
        
        # Create metadata
        metadata = AudioMetadata.from_file(audio_path)
        
        analysis = SongAnalysis(
            song_id=song_id,
            metadata=metadata,
            structure=structure,
            vocals=vocal_analysis,
            key=key,
            tempo=float(tempo),
            time_signature=time_sig,
            chord_progression=chords,
            energy_curve=energy_curve[:100]  # Limit for storage
        )
        
        # Save analysis
        with open(analysis_dir / "full_analysis.json", 'w') as f:
            json.dump(asdict(analysis), f, indent=2)
        
        # Create summary
        summary = {
            'song_id': song_id,
            'duration': analysis.metadata.duration,
            'key': analysis.key,
            'tempo': analysis.tempo,
            'vocal_range': f"{analysis.vocals.range_low:.1f}-{analysis.vocals.range_high:.1f}",
            'voice_type': analysis.vocals.voice_type.value,
            'sections': len(analysis.structure)
        }
        
        with open(analysis_dir / "summary.json", 'w') as f:
            json.dump(summary, f, indent=2)
        
        return analysis
    
    def _analyze_structure(self, audio: np.ndarray, sr: int) -> List[Dict]:
        """Analyze song structure"""
        # Simplified structure analysis
        # In production, use MSAF or similar
        sections = []
        
        # Use novelty curve for section detection
        novelty = librosa.segment.cross_similarity(audio, audio)
        
        # Simple beat-synced sections for now
        tempo, beats = librosa.beat.beat_track(y=audio, sr=sr)
        beat_times = librosa.frames_to_time(beats, sr=sr)
        
        # Create 4-bar sections
        bars_per_section = 4
        beats_per_bar = 4  # Assuming 4/4
        beats_per_section = bars_per_section * beats_per_bar
        
        for i in range(0, len(beat_times), beats_per_section):
            if i + beats_per_section < len(beat_times):
                start = beat_times[i]
                end = beat_times[min(i + beats_per_section, len(beat_times) - 1)]
                
                section_type = self._classify_section_type(i // beats_per_section)
                
                sections.append({
                    'section_id': f"section_{len(sections)+1:03d}",
                    'type': section_type.value,
                    'start_time': float(start),
                    'end_time': float(end),
                    'duration': float(end - start),
                    'bars': bars_per_section
                })
        
        return sections
    
    def _classify_section_type(self, section_index: int) -> SongSection:
        """Classify section type based on position"""
        if section_index == 0:
            return SongSection.INTRO
        elif section_index % 3 == 2:  # Every 3rd section starting from 2
            return SongSection.CHORUS
        elif section_index % 3 == 1:
            return SongSection.VERSE
        else:
            return SongSection.BRIDGE
    
    def _detect_key(self, audio: np.ndarray, sr: int) -> str:
        """Detect musical key"""
        # Use librosa's key detection
        chroma = librosa.feature.chroma_cqt(y=audio, sr=sr)
        chroma_mean = chroma.mean(axis=1)
        
        # Major and minor profiles
        major_profile = [6.35, 2.23, 3.48, 2.33, 4.38, 4.09, 2.52, 5.19, 2.39, 3.66, 2.29, 2.88]
        minor_profile = [6.33, 2.68, 3.52, 5.38, 2.60, 3.53, 2.54, 4.75, 3.98, 2.69, 3.34, 3.17]
        
        # Find best correlation
        keys = ['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B']
        best_corr = -1
        best_key = "C"
        
        for i in range(12):
            # Rotate profiles
            major_rotated = np.roll(major_profile, i)
            minor_rotated = np.roll(minor_profile, i)
            
            major_corr = np.corrcoef(chroma_mean, major_rotated)[0,1]
            minor_corr = np.corrcoef(chroma_mean, minor_rotated)[0,1]
            
            if major_corr > best_corr:
                best_corr = major_corr
                best_key = f"{keys[i]} major"
            if minor_corr > best_corr:
                best_corr = minor_corr
                best_key = f"{keys[i]} minor"
        
        return best_key
    
    def _detect_time_signature(self, audio: np.ndarray, sr: int) -> str:
        """Detect time signature (simplified)"""
        # For MVP, assume 4/4
        return "4/4"
    
    def _extract_chords(self, audio: np.ndarray, sr: int) -> List[str]:
        """Extract chord progression (simplified)"""
        # Simple chord extraction for MVP
        chords = ['C', 'G', 'Am', 'F']  # Common progression
        return chords

# -------------------------------------------------------------------
# COMPATIBILITY ENGINE
# -------------------------------------------------------------------

class CompatibilityEngine:
    """Analyze compatibility between two songs"""
    
    def __init__(self, base_dir: str = "VocalFusion"):
        self.base_dir = Path(base_dir)
    
    def analyze_compatibility(self, song_a_id: str, song_b_id: str) -> CompatibilityScore:
        """Analyze compatibility between two songs"""
        
        # Load analyses
        analysis_a = self._load_analysis(song_a_id)
        analysis_b = self._load_analysis(song_b_id)
        
        # Calculate compatibility scores
        key_comp = self._calculate_key_compatibility(analysis_a.key, analysis_b.key)
        tempo_comp = self._calculate_tempo_compatibility(analysis_a.tempo, analysis_b.tempo)
        range_comp = self._calculate_range_compatibility(analysis_a.vocals, analysis_b.vocals)
        timbre_comp = self._calculate_timbre_compatibility(analysis_a.vocals, analysis_b.vocals)
        structure_comp = self._calculate_structure_compatibility(analysis_a.structure, analysis_b.structure)
        
        # Overall score (weighted average)
        weights = {'key': 0.3, 'tempo': 0.2, 'range': 0.25, 'timbre': 0.15, 'structure': 0.1}
        overall = (key_comp * weights['key'] + 
                  tempo_comp * weights['tempo'] + 
                  range_comp * weights['range'] + 
                  timbre_comp * weights['timbre'] + 
                  structure_comp * weights['structure'])
        
        # Recommendations
        transposition = self._recommend_transposition(analysis_a, analysis_b)
        tempo_adjustment = self._recommend_tempo_adjustment(analysis_a, analysis_b)
        
        score = CompatibilityScore(
            song_a_id=song_a_id,
            song_b_id=song_b_id,
            key_compatibility=key_comp,
            tempo_compatibility=tempo_comp,
            range_compatibility=range_comp,
            timbre_compatibility=timbre_comp,
            structure_compatibility=structure_comp,
            overall_score=overall,
            recommended_transposition=transposition,
            recommended_tempo_adjustment=tempo_adjustment,
            arrangement_strategies=self._suggest_arrangements(analysis_a, analysis_b),
            challenges=self._identify_challenges(analysis_a, analysis_b),
            opportunities=self._identify_opportunities(analysis_a, analysis_b)
        )
        
        # Save compatibility analysis
        comp_dir = self.base_dir / "compatibility" / f"{song_a_id}__{song_b_id}"
        comp_dir.mkdir(parents=True, exist_ok=True)
        
        with open(comp_dir / "compatibility_report.json", 'w') as f:
            json.dump(asdict(score), f, indent=2)
        
        return score
    
    def _load_analysis(self, song_id: str) -> SongAnalysis:
        """Load song analysis from disk"""
        analysis_path = self.base_dir / "analysis" / song_id / "full_analysis.json"
        with open(analysis_path, 'r') as f:
            data = json.load(f)
        return SongAnalysis.from_dict(data)
    
    def _calculate_key_compatibility(self, key_a: str, key_b: str) -> float:
        """Calculate key compatibility (0-1)"""
        # Simplified key compatibility
        key_to_number = {
            'C': 0, 'C#': 1, 'D': 2, 'D#': 3, 'E': 4, 'F': 5,
            'F#': 6, 'G': 7, 'G#': 8, 'A': 9, 'A#': 10, 'B': 11
        }
        
        # Extract root note
        root_a = key_a.split()[0]
        root_b = key_b.split()[0]
        
        if root_a not in key_to_number or root_b not in key_to_number:
            return 0.5  # Default if key unknown
        
        # Calculate distance in circle of fifths
        dist = abs(key_to_number[root_a] - key_to_number[root_b])
        dist = min(dist, 12 - dist)  # Wrap around
        
        # Convert to compatibility score (closer = better)
        if dist == 0:  # Same key
            return 1.0
        elif dist == 7:  # Tritone - least compatible
            return 0.1
        elif dist == 5:  # Perfect fourth/fifth - very compatible
            return 0.9
        elif dist <= 2:  # Close keys
            return 0.8
        else:
            return 0.5
    
    def _calculate_tempo_compatibility(self, tempo_a: float, tempo_b: float) -> float:
        """Calculate tempo compatibility (0-1)"""
        ratio = min(tempo_a, tempo_b) / max(tempo_a, tempo_b)
        
        if ratio > 0.95:  # Within 5%
            return 1.0
        elif ratio > 0.9:  # Within 10%
            return 0.8
        elif ratio > 0.8:  # Within 20%
            return 0.6
        else:
            return 0.3
    
    def _calculate_range_compatibility(self, vocals_a: VocalAnalysis, vocals_b: VocalAnalysis) -> float:
        """Calculate vocal range compatibility (0-1)"""
        # Check for overlap
        overlap_min = max(vocals_a.range_low, vocals_b.range_low)
        overlap_max = min(vocals_a.range_high, vocals_b.range_high)
        
        if overlap_min < overlap_max:  # Ranges overlap
            overlap_ratio = (overlap_max - overlap_min) / min(
                vocals_a.range_high - vocals_a.range_low,
                vocals_b.range_high - vocals_b.range_low
            )
            return min(overlap_ratio * 2, 1.0)  # Scale up, cap at 1.0
        else:
            # Ranges don't overlap - could be good for harmony
            gap = overlap_min - overlap_max
            max_range = max(vocals_a.range_high, vocals_b.range_high) - min(vocals_a.range_low, vocals_b.range_low)
            
            if gap < max_range * 0.3:  # Small gap
                return 0.7
            else:  # Large gap
                return 0.4
    
    def _calculate_timbre_compatibility(self, vocals_a: VocalAnalysis, vocals_b: VocalAnalysis) -> float:
        """Calculate timbre compatibility (0-1)"""
        # Compare MFCCs (cosine similarity)
        mfcc_a = np.array(vocals_a.mfcc_mean)
        mfcc_b = np.array(vocals_b.mfcc_mean)
        
        # Cosine similarity
        similarity = np.dot(mfcc_a, mfcc_b) / (np.linalg.norm(mfcc_a) * np.linalg.norm(mfcc_b))
        
        # Convert to compatibility score
        # Different timbres can complement each other, so medium similarity is good
        if 0.3 < similarity < 0.7:
            return 0.8
        elif similarity > 0.8:  # Too similar
            return 0.6
        else:  # Too different
            return 0.5
    
    def _calculate_structure_compatibility(self, structure_a: List, structure_b: List) -> float:
        """Calculate structure compatibility (0-1)"""
        # Compare section counts and types
        sections_a = [s['type'] for s in structure_a]
        sections_b = [s['type'] for s in structure_b]
        
        # Simple comparison
        if len(sections_a) == len(sections_b):
            # Check if sections match
            matches = sum(1 for a, b in zip(sections_a, sections_b) if a == b)
            return matches / len(sections_a)
        else:
            # Different number of sections
            return 0.5
    
    def _recommend_transposition(self, analysis_a: SongAnalysis, analysis_b: SongAnalysis) -> int:
        """Recommend transposition in semitones to match keys"""
        # Simple: transpose to same key
        # In production, use music theory to find best key
        return 0  # Placeholder
    
    def _recommend_tempo_adjustment(self, analysis_a: SongAnalysis, analysis_b: SongAnalysis) -> float:
        """Recommend tempo adjustment ratio"""
        # Adjust to average tempo
        return (analysis_a.tempo + analysis_b.tempo) / (2 * analysis_b.tempo)
    
    def _suggest_arrangements(self, analysis_a: SongAnalysis, analysis_b: SongAnalysis) -> List[str]:
        """Suggest arrangement strategies"""
        strategies = []
        
        # Based on compatibility scores
        if analysis_a.vocals.voice_type != analysis_b.vocals.voice_type:
            strategies.append("male_female_duet")
        
        if len(analysis_a.structure) == len(analysis_b.structure):
            strategies.append("parallel_structure")
        else:
            strategies.append("interleaved_sections")
        
        # Add more based on analysis
        strategies.append("call_and_response")
        strategies.append("harmony_on_chorus")
        
        return strategies
    
    def _identify_challenges(self, analysis_a: SongAnalysis, analysis_b: SongAnalysis) -> List[str]:
        """Identify potential challenges"""
        challenges = []
        
        if abs(analysis_a.tempo - analysis_b.tempo) > 20:
            challenges.append("significant_tempo_difference")
        
        if analysis_a.key != analysis_b.key:
            challenges.append("different_keys")
        
        return challenges
    
    def _identify_opportunities(self, analysis_a: SongAnalysis, analysis_b: SongAnalysis) -> List[str]:
        """Identify opportunities for great fusion"""
        opportunities = []
        
        if analysis_a.vocals.voice_type != analysis_b.vocals.voice_type:
            opportunities.append("natural_frequency_separation")
        
        if len(analysis_a.vocals.phrases) > 0 and len(analysis_b.vocals.phrases) > 0:
            opportunities.append("good_phrase_alignment")
        
        return opportunities

# -------------------------------------------------------------------
# WEB API
# -------------------------------------------------------------------

from flask import Flask, request, jsonify, send_from_directory
from flask_socketio import SocketIO, emit
import threading
import time

app = Flask(__name__)
app.config['SECRET_KEY'] = 'vocalfusion-secret-key'
socketio = SocketIO(app, cors_allowed_origins="*")

engine = VocalFusionEngine()
comp_engine = CompatibilityEngine()

# Store active jobs
active_jobs = {}

@app.route('/')
def index():
    return jsonify({
        'status': 'VocalFusion API',
        'version': '0.1.0',
        'endpoints': {
            '/upload': 'POST - Upload and process song',
            '/analyze/<song_id>': 'GET - Get analysis results',
            '/compatibility/<song_a>/<song_b>': 'GET - Get compatibility',
            '/status/<job_id>': 'GET - Get job status'
        }
    })

@app.route('/upload', methods=['POST'])
def upload_song():
    """Upload and process a song"""
    if 'file' not in request.files:
        return jsonify({'error': 'No file provided'}), 400
    
    file = request.files['file']
    song_name = request.form.get('name', file.filename)
    
    # Save to temp file
    import tempfile
    temp_path = Path(tempfile.mktemp(suffix='.mp3'))
    file.save(temp_path)
    
    # Create job
    job_id = f"job_{int(time.time())}_{hash(song_name) % 10000}"
    active_jobs[job_id] = {
        'status': 'processing',
        'song_name': song_name,
        'progress': 0,
        'result': None
    }
    
    # Process in background
    def process_background():
        try:
            # Update progress
            active_jobs[job_id]['progress'] = 10
            socketio.emit('job_progress', {'job_id': job_id, 'progress': 10})
            
            # Process song
            result = engine.process_song(temp_path, song_name)
            
            # Update job
            active_jobs[job_id].update({
                'status': 'completed',
                'progress': 100,
                'result': result
            })
            
            socketio.emit('job_complete', {
                'job_id': job_id,
                'result': result
            })
            
        except Exception as e:
            active_jobs[job_id].update({
                'status': 'failed',
                'error': str(e)
            })
            socketio.emit('job_failed', {
                'job_id': job_id,
                'error': str(e)
            })
        finally:
            # Clean up temp file
            temp_path.unlink(missing_ok=True)
    
    # Start background thread
    thread = threading.Thread(target=process_background)
    thread.start()
    
    return jsonify({
        'job_id': job_id,
        'status': 'processing_started',
        'message': f'Processing {song_name}'
    })

@app.route('/analyze/<song_id>')
def get_analysis(song_id):
    """Get analysis results for a song"""
    analysis_path = Path("VocalFusion") / "analysis" / song_id / "full_analysis.json"
    
    if not analysis_path.exists():
        return jsonify({'error': 'Analysis not found'}), 404
    
    with open(analysis_path, 'r') as f:
        analysis = json.load(f)
    
    return jsonify(analysis)

@app.route('/compatibility/<song_a>/<song_b>')
def get_compatibility(song_a, song_b):
    """Get compatibility analysis between two songs"""
    comp = comp_engine.analyze_compatibility(song_a, song_b)
    return jsonify(asdict(comp))

@app.route('/status/<job_id>')
def get_status(job_id):
    """Get job status"""
    if job_id not in active_jobs:
        return jsonify({'error': 'Job not found'}), 404
    
    return jsonify(active_jobs[job_id])

@app.route('/songs')
def list_songs():
    """List all processed songs"""
    analysis_dir = Path("VocalFusion") / "analysis"
    
    if not analysis_dir.exists():
        return jsonify({'songs': []})
    
    songs = []
    for song_dir in analysis_dir.iterdir():
        if song_dir.is_dir():
            summary_path = song_dir / "summary.json"
            if summary_path.exists():
                with open(summary_path, 'r') as f:
                    songs.append(json.load(f))
    
    return jsonify({'songs': songs})

# WebSocket events
@socketio.on('connect')
def handle_connect():
    print('Client connected')
    emit('connected', {'message': 'Connected to VocalFusion'})

@socketio.on('disconnect')
def handle_disconnect():
    print('Client disconnected')

# -------------------------------------------------------------------
# COMMAND LINE INTERFACE
# -------------------------------------------------------------------

import argparse

def main():
    parser = argparse.ArgumentParser(description='VocalFusion CLI')
    subparsers = parser.add_subparsers(dest='command', help='Commands')
    
    # Process command
    process_parser = subparsers.add_parser('process', help='Process a song')
    process_parser.add_argument('file', help='Audio file to process')
    process_parser.add_argument('--name', help='Song name (optional)')
    
    # Compatibility command
    comp_parser = subparsers.add_parser('compatibility', help='Analyze compatibility')
    comp_parser.add_argument('song_a', help='First song ID')
    comp_parser.add_argument('song_b', help='Second song ID')
    
    # List command
    list_parser = subparsers.add_parser('list', help='List processed songs')
    
    # Serve command
    serve_parser = subparsers.add_parser('serve', help='Start web server')
    serve_parser.add_argument('--port', type=int, default=5000, help='Port to listen on')
    
    args = parser.parse_args()
    
    if args.command == 'process':
        engine = VocalFusionEngine()
        result = engine.process_song(Path(args.file), args.name)
        print(f"Processed song: {result['song_id']}")
        print(f"Analysis saved to: {result['paths']['analysis']}")
        
    elif args.command == 'compatibility':
        comp_engine = CompatibilityEngine()
        result = comp_engine.analyze_compatibility(args.song_a, args.song_b)
        print(f"Compatibility score: {result.overall_score:.2f}")
        print(f"Key compatibility: {result.key_compatibility:.2f}")
        print(f"Tempo compatibility: {result.tempo_compatibility:.2f}")
        print(f"Suggested arrangements: {', '.join(result.arrangement_strategies)}")
        
    elif args.command == 'list':
        analysis_dir = Path("VocalFusion") / "analysis"
        
        if not analysis_dir.exists():
            print("No songs processed yet")
            return
        
        for song_dir in analysis_dir.iterdir():
            if song_dir.is_dir():
                summary_path = song_dir / "summary.json"
                if summary_path.exists():
                    with open(summary_path, 'r') as f:
                        summary = json.load(f)
                        print(f"{summary['song_id']}: {summary.get('duration', 0):.1f}s, "
                              f"Key: {summary.get('key', 'Unknown')}, "
                              f"Tempo: {summary.get('tempo', 0):.1f} BPM")
    
    elif args.command == 'serve':
        print(f"Starting VocalFusion server on port {args.port}...")
        print(f"Open http://localhost:{args.port} to access the API")
        socketio.run(app, port=args.port, debug=True, allow_unsafe_werkzeug=True)
    
    else:
        parser.print_help()

if __name__ == "__main__":
    # Create directory structure
    engine = VocalFusionEngine()
    
    # Run CLI or web server
    import sys
    if len(sys.argv) > 1:
        main()
    else:
        # Start web server by default
        print("Starting VocalFusion web server...")
        print("Open http://localhost:5000 to access the API")
        print("Use --help for CLI options")
        socketio.run(app, port=5000, debug=True, allow_unsafe_werkzeug=True)
