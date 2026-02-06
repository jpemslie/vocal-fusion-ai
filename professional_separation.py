"""
Professional stem separation using Demucs with proper GPU support,
caching, and quality optimization.
"""

import os
import sys
import json
import shutil
import tempfile
import subprocess
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union
import numpy as np
import soundfile as sf
import librosa
import torch
import torchaudio
from dataclasses import dataclass, asdict, field
from datetime import datetime
import hashlib
import warnings
warnings.filterwarnings('ignore')

# -------------------------------------------------------------------
# PROFESSIONAL DEMUCS INTEGRATION
# -------------------------------------------------------------------

@dataclass
class SeparationConfig:
    """Configuration for professional stem separation"""
    model_name: str = "htdemucs_ft"
    model_variant: str = "6s"  # "4s" (4 stems) or "6s" (6 stems)
    shifts: int = 1  # Number of random shifts for equivariance
    overlap: float = 0.25  # Overlap between prediction windows
    segment: Optional[int] = 7  # Segment length in seconds
    split: bool = True  # Split into segments for less memory usage
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    workers: int = 1  # Number of parallel workers
    progress: bool = True
    callback = None
    verbose: bool = False
    
    # Quality settings
    quality_preset: str = "balanced"  # "fast", "balanced", "high_quality"
    
    def __post_init__(self):
        """Set parameters based on quality preset"""
        if self.quality_preset == "fast":
            self.shifts = 1
            self.overlap = 0.25
            self.segment = 6
        elif self.quality_preset == "balanced":
            self.shifts = 1
            self.overlap = 0.25
            self.segment = 7
        elif self.quality_preset == "high_quality":
            self.shifts = 5
            self.overlap = 0.5
            self.segment = 10
        elif self.quality_preset == "professional":
            self.shifts = 10
            self.overlap = 0.75
            self.segment = None  # No segmentation for max quality
    
    def to_dict(self) -> Dict:
        return asdict(self)

class ProfessionalSeparator:
    """Professional-grade stem separator using Demucs"""
    
    def __init__(self, config: Optional[SeparationConfig] = None):
        self.config = config or SeparationConfig()
        self._model = None
        self._device = torch.device(self.config.device)
        
        # Cache for model loading
        self.cache_dir = Path.home() / ".cache" / "vocalfusion" / "models"
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        
        print(f"Initializing ProfessionalSeparator on {self.config.device.upper()}")
        print(f"Using model: {self.config.model_name} ({self.config.model_variant})")
        print(f"Quality preset: {self.config.quality_preset}")
    
    def load_model(self):
        """Load Demucs model (lazy loading)"""
        if self._model is not None:
            return self._model
        
        try:
            from demucs import pretrained
            from demucs.apply import apply_model
            
            print(f"Loading Demucs model: {self.config.model_name}...")
            
            # Try to load with caching
            model_path = self.cache_dir / f"{self.config.model_name}.pt"
            
            if model_path.exists():
                print(f"Loading model from cache: {model_path}")
                # Load from cache
                model = torch.load(model_path, map_location=self._device)
            else:
                # Download and cache
                print(f"Downloading model...")
                model = pretrained.get_model(self.config.model_name)
                torch.save(model, model_path)
                print(f"Model cached at: {model_path}")
            
            model.to(self._device)
            model.eval()
            
            self._model = model
            self._apply_model = apply_model
            
            print(f"Model loaded successfully")
            return model
            
        except ImportError as e:
            print(f"Demucs not available, using fallback method")
            print(f"Install with: pip install 'demucs @ git+https://github.com/facebookresearch/demucs.git'")
            return None
        except Exception as e:
            print(f"Error loading model: {e}")
            return None
    
    def separate_file(self, input_path: Union[str, Path], 
                     output_dir: Optional[Union[str, Path]] = None,
                     stems: Optional[List[str]] = None) -> Dict[str, Path]:
        """
        Separate audio file into stems using Demucs
        
        Args:
            input_path: Path to input audio file
            output_dir: Directory to save stems (default: same as input)
            stems: Which stems to extract (default: all)
            
        Returns:
            Dict mapping stem names to output file paths
        """
        input_path = Path(input_path)
        
        if not input_path.exists():
            raise FileNotFoundError(f"Input file not found: {input_path}")
        
        # Set output directory
        if output_dir is None:
            output_dir = input_path.parent / "stems"
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        print(f"Separating: {input_path.name}")
        print(f"Output directory: {output_dir}")
        
        # Try to use Demucs
        model = self.load_model()
        if model is not None:
            return self._separate_with_demucs(input_path, output_dir, stems)
        else:
            # Fallback to simpler method
            print("Using fallback separation method")
            return self._separate_fallback(input_path, output_dir, stems)
    
    def _separate_with_demucs(self, input_path: Path, 
                             output_dir: Path,
                             stems: Optional[List[str]]) -> Dict[str, Path]:
        """Separate using actual Demucs model"""
        import torch
        from demucs.apply import apply_model
        
        # Load audio
        print(f"Loading audio: {input_path}")
        wav, sr = torchaudio.load(str(input_path))
        
        # Convert to model's expected format
        if wav.dim() == 1:
            wav = wav.unsqueeze(0)  # Add channel dimension
        if wav.shape[0] > 2:  # If more than stereo, take first two channels
            wav = wav[:2, :]
        elif wav.shape[0] == 1:  # Mono to stereo
            wav = torch.cat([wav, wav], dim=0)
        
        # Move to device
        wav = wav.to(self._device)
        
        # Normalize
        wav = wav / max(1.0, wav.abs().max())
        
        print(f"Separating with Demucs (this may take a while)...")
        
        # Apply model
        with torch.no_grad():
            # Demucs returns sources in order: ['drums', 'bass', 'other', 'vocals']
            # For 6-stem models: ['drums', 'bass', 'other', 'vocals', 'guitar', 'piano']
            sources = apply_model(
                self._model, 
                wav.unsqueeze(0),  # Add batch dimension
                shifts=self.config.shifts,
                overlap=self.config.overlap,
                segment=self.config.segment,
                device=self._device
            )
        
        # Remove batch dimension
        sources = sources.squeeze(0)
        
        # Get stem names from model
        stem_names = self._model.sources
        
        # Save each stem
        output_paths = {}
        for i, stem_name in enumerate(stem_names):
            if stems is not None and stem_name not in stems:
                continue
            
            stem_audio = sources[i].cpu().numpy()
            
            # Convert to mono if needed
            if stem_audio.shape[0] > 1:
                stem_audio = stem_audio.mean(axis=0)
            
            # Save as WAV
            output_path = output_dir / f"{stem_name}.wav"
            sf.write(output_path, stem_audio.T, sr)
            
            output_paths[stem_name] = output_path
            print(f"  Saved {stem_name}: {output_path}")
        
        # Save separation metadata
        metadata = {
            'input_file': str(input_path),
            'output_dir': str(output_dir),
            'model_used': self.config.model_name,
            'model_variant': self.config.model_variant,
            'quality_preset': self.config.quality_preset,
            'sample_rate': int(sr),
            'channels': wav.shape[0],
            'duration': wav.shape[1] / sr,
            'stems_extracted': list(output_paths.keys()),
            'separation_time': datetime.now().isoformat(),
            'device_used': str(self._device),
            'parameters': self.config.to_dict()
        }
        
        metadata_path = output_dir / "separation_metadata.json"
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2)
        
        print(f"Separation complete! Metadata saved to: {metadata_path}")
        
        return output_paths
    
    def _separate_fallback(self, input_path: Path, 
                          output_dir: Path,
                          stems: Optional[List[str]]) -> Dict[str, Path]:
        """Fallback separation using simpler methods"""
        import numpy as np
        import librosa
        
        print("WARNING: Using fallback separation (lower quality)")
        
        # Load audio
        y, sr = librosa.load(input_path, sr=None, mono=False)
        
        if y.ndim == 1:
            y = np.array([y, y])  # Convert to stereo
        
        # Simple frequency-based separation (for demonstration)
        # In production, you should always use Demucs
        
        stems_to_extract = stems or ['vocals', 'drums', 'bass', 'other']
        output_paths = {}
        
        for stem in stems_to_extract:
            if stem == 'vocals':
                # Simple HPF to approximate vocals
                import scipy.signal as signal
                b, a = signal.butter(4, 100/(sr/2), btype='high')
                stem_audio = signal.filtfilt(b, a, y.mean(axis=0))
            elif stem == 'drums':
                # Simple LPF + transient detection
                import scipy.signal as signal
                b, a = signal.butter(4, 200/(sr/2), btype='low')
                stem_audio = signal.filtfilt(b, a, y.mean(axis=0))
            elif stem == 'bass':
                # Very low frequencies
                import scipy.signal as signal
                b, a = signal.butter(4, [40/(sr/2), 200/(sr/2)], btype='band')
                stem_audio = signal.filtfilt(b, a, y.mean(axis=0))
            else:  # 'other'
                # Mid frequencies
                import scipy.signal as signal
                b, a = signal.butter(4, [200/(sr/2), 3000/(sr/2)], btype='band')
                stem_audio = signal.filtfilt(b, a, y.mean(axis=0))
            
            output_path = output_dir / f"{stem}.wav"
            sf.write(output_path, stem_audio, sr)
            output_paths[stem] = output_path
        
        return output_paths
    
    def separate_directory(self, input_dir: Union[str, Path],
                          output_base_dir: Optional[Union[str, Path]] = None,
                          file_pattern: str = "*.wav"):
        """Separate all audio files in a directory"""
        input_dir = Path(input_dir)
        
        if not input_dir.exists():
            raise FileNotFoundError(f"Directory not found: {input_dir}")
        
        if output_base_dir is None:
            output_base_dir = input_dir.parent / "separated"
        output_base_dir = Path(output_base_dir)
        
        results = {}
        
        for audio_file in input_dir.glob(file_pattern):
            if audio_file.is_file():
                print(f"\nProcessing: {audio_file.name}")
                song_name = audio_file.stem
                output_dir = output_base_dir / song_name
                
                try:
                    stems = self.separate_file(audio_file, output_dir)
                    results[song_name] = {
                        'stems': stems,
                        'output_dir': output_dir
                    }
                    print(f"Successfully separated {audio_file.name}")
                except Exception as e:
                    print(f"Failed to separate {audio_file.name}: {e}")
        
        return results

# -------------------------------------------------------------------
# PROFESSIONAL VOCAL ANALYSIS WITH PYWORLD
# -------------------------------------------------------------------

@dataclass
class VocalAnalysisResult:
    """Professional vocal analysis results using PyWorld"""
    pitch_contour: np.ndarray  # Fundamental frequency (F0)
    spectral_envelope: np.ndarray  # Spectral envelope
    aperiodicity: np.ndarray  # Aperiodicity
    f0_times: np.ndarray  # Time points for F0
    voiced_flags: np.ndarray  # Voiced/unvoiced flags
    
    # Derived features
    notes: List[Dict]  # Extracted notes with timing
    phrases: List[Dict]  # Vocal phrases
    vibrato_analysis: Dict  # Vibrato detection
    voice_quality: Dict  # Voice quality metrics
    
    # Statistical features
    mfcc: np.ndarray  # MFCC coefficients
    spectral_centroid: float
    spectral_bandwidth: float
    harmonic_ratio: float
    voice_type: str
    
    def to_dict(self) -> Dict:
        """Convert to JSON-serializable dict"""
        return {
            'pitch_contour': self.pitch_contour.tolist() if hasattr(self.pitch_contour, 'tolist') else self.pitch_contour,
            'f0_times': self.f0_times.tolist() if hasattr(self.f0_times, 'tolist') else self.f0_times,
            'voiced_flags': self.voiced_flags.tolist() if hasattr(self.voiced_flags, 'tolist') else self.voiced_flags,
            'notes': self.notes,
            'phrases': self.phrases,
            'vibrato_analysis': self.vibrato_analysis,
            'voice_quality': self.voice_quality,
            'mfcc': self.mfcc.tolist() if hasattr(self.mfcc, 'tolist') else self.mfcc,
            'spectral_centroid': self.spectral_centroid,
            'spectral_bandwidth': self.spectral_bandwidth,
            'harmonic_ratio': self.harmonic_ratio,
            'voice_type': self.voice_type
        }

class ProfessionalVocalAnalyzer:
    """Professional vocal analyzer using PyWorld (WORLD vocoder)"""
    
    def __init__(self, sample_rate: int = 44100, frame_period: float = 5.0):
        self.sr = sample_rate
        self.frame_period = frame_period  # ms
        
    def analyze(self, audio_path: Union[str, Path], 
                is_vocal_stem: bool = True) -> VocalAnalysisResult:
        """
        Perform professional vocal analysis using PyWorld
        
        Args:
            audio_path: Path to audio file (preferably vocal stem)
            is_vocal_stem: Whether this is an isolated vocal stem
            
        Returns:
            VocalAnalysisResult with detailed vocal features
        """
        # Load audio
        x, sr = librosa.load(audio_path, sr=self.sr, mono=True)
        
        # Ensure proper sample rate
        if sr != self.sr:
            import resampy
            x = resampy.resample(x, sr, self.sr)
            sr = self.sr
        
        # Use PyWorld for professional vocal analysis
        try:
            import pyworld as pw
            
            print(f"Analyzing vocals with PyWorld (WORLD vocoder)...")
            
            # 1. Extract F0 (fundamental frequency) using DIO
            f0, timeaxis = pw.dio(x.astype(np.float64), sr, frame_period=self.frame_period)
            
            # 2. Refine F0 using StoneMask
            f0 = pw.stonemask(x.astype(np.float64), f0, timeaxis, sr)
            
            # 3. Extract spectral envelope
            sp = pw.cheaptrick(x.astype(np.float64), f0, timeaxis, sr)
            
            # 4. Extract aperiodicity
            ap = pw.d4c(x.astype(np.float64), f0, timeaxis, sr)
            
            # 5. Detect voiced/unvoiced regions
            voiced_flags = f0 > 0
            
            # 6. Extract notes from F0 contour
            notes = self._extract_notes(f0, timeaxis, sr)
            
            # 7. Detect vocal phrases
            phrases = self._detect_phrases(x, sr, voiced_flags, timeaxis)
            
            # 8. Analyze vibrato
            vibrato = self._analyze_vibrato(f0[voiced_flags], timeaxis[voiced_flags])
            
            # 9. Extract voice quality features
            voice_quality = self._analyze_voice_quality(x, sr, f0, sp, ap)
            
            # 10. Extract MFCC features
            mfcc = librosa.feature.mfcc(y=x, sr=sr, n_mfcc=13)
            
            # 11. Calculate spectral features
            spectral_centroid = librosa.feature.spectral_centroid(y=x, sr=sr).mean()
            spectral_bandwidth = librosa.feature.spectral_bandwidth(y=x, sr=sr).mean()
            harmonic_ratio = self._calculate_harmonic_ratio(x, sr)
            
            # 12. Classify voice type
            voice_type = self._classify_voice_type(f0[voiced_flags])
            
            result = VocalAnalysisResult(
                pitch_contour=f0,
                spectral_envelope=sp,
                aperiodicity=ap,
                f0_times=timeaxis,
                voiced_flags=voiced_flags,
                notes=notes,
                phrases=phrases,
                vibrato_analysis=vibrato,
                voice_quality=voice_quality,
                mfcc=mfcc.mean(axis=1),
                spectral_centroid=float(spectral_centroid),
                spectral_bandwidth=float(spectral_bandwidth),
                harmonic_ratio=float(harmonic_ratio),
                voice_type=voice_type
            )
            
            return result
            
        except ImportError:
            print("PyWorld not available, using simplified analysis")
            return self._analyze_simplified(x, sr)
    
    def _extract_notes(self, f0: np.ndarray, timeaxis: np.ndarray, sr: int) -> List[Dict]:
        """Extract discrete notes from F0 contour"""
        notes = []
        
        # Find continuous voiced regions
        voiced = f0 > 0
        if not np.any(voiced):
            return notes
        
        # Group continuous voiced regions
        from scipy.ndimage import find_objects, label
        labeled, num_features = label(voiced)
        
        for i in range(1, num_features + 1):
            indices = np.where(labeled == i)[0]
            if len(indices) > 0:
                start_idx = indices[0]
                end_idx = indices[-1]
                
                # Get note properties
                note_f0 = f0[indices].mean()
                note_f0_std = f0[indices].std()
                
                # Convert to MIDI note number
                if note_f0 > 0:
                    midi_note = 69 + 12 * np.log2(note_f0 / 440.0)
                    
                    notes.append({
                        'note_id': f'note_{len(notes)+1:03d}',
                        'midi_number': float(midi_note),
                        'frequency': float(note_f0),
                        'frequency_std': float(note_f0_std),
                        'start_time': float(timeaxis[start_idx]),
                        'end_time': float(timeaxis[end_idx]),
                        'duration': float(timeaxis[end_idx] - timeaxis[start_idx]),
                        'confidence': float(1.0 - (note_f0_std / note_f0) if note_f0 > 0 else 0.5)
                    })
        
        return notes
    
    def _detect_phrases(self, audio: np.ndarray, sr: int, 
                       voiced_flags: np.ndarray, timeaxis: np.ndarray) -> List[Dict]:
        """Detect vocal phrases"""
        phrases = []
        
        # Simple energy-based phrase detection
        energy = librosa.feature.rms(y=audio)[0]
        energy_times = librosa.frames_to_time(range(len(energy)), sr=sr, hop_length=512)
        
        # Threshold for vocal activity
        threshold = np.percentile(energy, 40)
        vocal_activity = energy > threshold
        
        # Find continuous vocal regions
        from scipy.ndimage import label
        labeled, num_features = label(vocal_activity)
        
        for i in range(1, num_features + 1):
            indices = np.where(labeled == i)[0]
            if len(indices) > 2:  # Minimum 3 frames
                start_time = energy_times[indices[0]]
                end_time = energy_times[indices[-1]]
                duration = end_time - start_time
                
                if duration > 0.5:  # Minimum phrase length
                    phrases.append({
                        'phrase_id': f'phrase_{len(phrases)+1:03d}',
                        'start_time': float(start_time),
                        'end_time': float(end_time),
                        'duration': float(duration),
                        'energy_mean': float(energy[indices].mean()),
                        'energy_std': float(energy[indices].std())
                    })
        
        return phrases
    
    def _analyze_vibrato(self, f0: np.ndarray, times: np.ndarray) -> Dict:
        """Analyze vibrato in vocal signal"""
        if len(f0) < 10:
            return {'detected': False, 'rate_hz': 0.0, 'depth_cents': 0.0}
        
        # Convert to cents for vibrato analysis
        f0_cents = 1200 * np.log2(f0 / f0.mean())
        
        # Simple vibrato detection (looking for oscillations around 4-8 Hz)
        from scipy.signal import find_peaks, welch
        
        # Find peaks in F0 contour
        peaks, properties = find_peaks(f0_cents, prominence=10)  # 10 cents prominence
        
        if len(peaks) < 2:
            return {'detected': False, 'rate_hz': 0.0, 'depth_cents': 0.0}
        
        # Calculate vibrato rate (frequency)
        peak_times = times[peaks]
        if len(peak_times) > 1:
            intervals = np.diff(peak_times)
            vibrato_rate = 1.0 / intervals.mean() if intervals.mean() > 0 else 0
        else:
            vibrato_rate = 0
        
        # Calculate vibrato depth (extent in cents)
        vibrato_depth = np.std(f0_cents) * 2  # Approximate peak-to-peak
        
        # Check if within typical vibrato range
        is_vibrato = (4.0 <= vibrato_rate <= 8.0) and (vibrato_depth > 20)
        
        return {
            'detected': bool(is_vibrato),
            'rate_hz': float(vibrato_rate),
            'depth_cents': float(vibrato_depth),
            'regularity': float(1.0 - (np.std(intervals) / intervals.mean()) if intervals.mean() > 0 else 0)
        }
    
    def _analyze_voice_quality(self, audio: np.ndarray, sr: int, 
                              f0: np.ndarray, sp: np.ndarray, ap: np.ndarray) -> Dict:
        """Analyze voice quality characteristics"""
        voiced = f0 > 0
        
        if not np.any(voiced):
            return {
                'breathiness': 0.0,
                'nasality': 0.0,
                'tension': 0.0,
                'richness': 0.0,
                'clarity': 0.0
            }
        
        # Calculate breathiness (related to aperiodicity)
        breathiness = np.mean(ap[voiced, :]) if np.any(voiced) else 0.0
        
        # Calculate spectral tilt (related to nasality/tension)
        # Higher frequencies relative to lower frequencies
        spectral_tilt = np.mean(sp[voiced, 20:40]) / np.mean(sp[voiced, 0:20]) if np.any(voiced) else 1.0
        
        # Calculate harmonic richness (HNR - Harmonics-to-Noise Ratio)
        try:
            import pyworld as pw
            hnr = pw.hnr(f0[voiced], sr) if np.any(voiced) else 0.0
            hnr_mean = np.mean(hnr) if len(hnr) > 0 else 0.0
        except:
            hnr_mean = 0.0
        
        return {
            'breathiness': float(breathiness),
            'nasality': float(spectral_tilt),
            'tension': float(spectral_tilt),  # Simplified
            'richness': float(hnr_mean),
            'clarity': float(1.0 - breathiness)  # Inverse of breathiness
        }
    
    def _calculate_harmonic_ratio(self, audio: np.ndarray, sr: int) -> float:
        """Calculate harmonic ratio of audio signal"""
        # Simplified harmonic ratio calculation
        import scipy.signal as signal
        
        # Compute FFT
        fft = np.fft.rfft(audio)
        freqs = np.fft.rfftfreq(len(audio), 1/sr)
        
        # Find peaks in spectrum (potential harmonics)
        magnitude = np.abs(fft)
        
        # Simple harmonic ratio: energy in harmonic regions vs total energy
        # This is a simplified version
        total_energy = np.sum(magnitude ** 2)
        
        # Assume harmonics are at integer multiples of fundamental
        # Find fundamental (first significant peak)
        peaks, _ = signal.find_peaks(magnitude[:len(magnitude)//2], height=np.max(magnitude)*0.1)
        
        if len(peaks) == 0:
            return 0.0
        
        fundamental_idx = peaks[0]
        fundamental_freq = freqs[fundamental_idx]
        
        # Calculate energy at harmonic frequencies
        harmonic_energy = 0
        for harmonic in range(1, 10):  # First 10 harmonics
            harmonic_freq = fundamental_freq * harmonic
            if harmonic_freq < sr/2:
                idx = np.argmin(np.abs(freqs - harmonic_freq))
                harmonic_energy += magnitude[idx] ** 2
        
        harmonic_ratio = harmonic_energy / total_energy if total_energy > 0 else 0
        return float(harmonic_ratio)
    
    def _classify_voice_type(self, f0_voiced: np.ndarray) -> str:
        """Classify voice type based on F0 distribution"""
        if len(f0_voiced) == 0:
            return "unknown"
        
        mean_f0 = np.mean(f0_voiced)
        
        if mean_f0 > 300:  # Hz
            return "soprano"
        elif mean_f0 > 220:
            return "mezzo-soprano"
        elif mean_f0 > 180:
            return "alto"
        elif mean_f0 > 130:
            return "tenor"
        elif mean_f0 > 100:
            return "baritone"
        else:
            return "bass"
    
    def _analyze_simplified(self, audio: np.ndarray, sr: int) -> VocalAnalysisResult:
        """Simplified analysis if PyWorld is not available"""
        print("Using simplified vocal analysis")
        
        # Extract pitch using librosa
        f0, voiced_flag, voiced_probs = librosa.pyin(
            audio,
            fmin=librosa.note_to_hz('C2'),
            fmax=librosa.note_to_hz('C7'),
            sr=sr
        )
        
        # Create time axis
        timeaxis = librosa.frames_to_time(range(len(f0)), sr=sr, hop_length=512)
        
        # Create dummy arrays for PyWorld outputs
        sp = np.zeros((len(f0), 1025))
        ap = np.zeros((len(f0), 1025))
        
        # Extract notes
        notes = self._extract_notes(f0, timeaxis, sr)
        
        # Detect phrases
        phrases = self._detect_phrases(audio, sr, ~np.isnan(f0), timeaxis)
        
        # Analyze vibrato
        vibrato = self._analyze_vibrato(f0[~np.isnan(f0)], timeaxis[~np.isnan(f0)])
        
        # Voice quality (simplified)
        voice_quality = {
            'breathiness': 0.3,
            'nasality': 0.5,
            'tension': 0.5,
            'richness': 0.7,
            'clarity': 0.8
        }
        
        # MFCC
        mfcc = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=13)
        
        # Spectral features
        spectral_centroid = librosa.feature.spectral_centroid(y=audio, sr=sr).mean()
        spectral_bandwidth = librosa.feature.spectral_bandwidth(y=audio, sr=sr).mean()
        harmonic_ratio = 0.7  # Placeholder
        
        # Voice type
        voice_type = self._classify_voice_type(f0[~np.isnan(f0)])
        
        return VocalAnalysisResult(
            pitch_contour=f0,
            spectral_envelope=sp,
            aperiodicity=ap,
            f0_times=timeaxis,
            voiced_flags=~np.isnan(f0),
            notes=notes,
            phrases=phrases,
            vibrato_analysis=vibrato,
            voice_quality=voice_quality,
            mfcc=mfcc.mean(axis=1),
            spectral_centroid=float(spectral_centroid),
            spectral_bandwidth=float(spectral_bandwidth),
            harmonic_ratio=float(harmonic_ratio),
            voice_type=voice_type
        )

# -------------------------------------------------------------------
# MUSIC INFORMATION RETRIEVAL WITH ESSENTIA
# -------------------------------------------------------------------

class ProfessionalMusicAnalyzer:
    """Professional music analysis using Essentia (Spotify's MIR library)"""
    
    def __init__(self):
        try:
            import essentia
            import essentia.standard as es
            self.es = es
            self.essentia_available = True
            print("Essentia loaded successfully")
        except ImportError:
            print("Essentia not available, using fallback methods")
            self.essentia_available = False
    
    def analyze_song(self, audio_path: Union[str, Path]) -> Dict:
        """Comprehensive song analysis using Essentia"""
        if not self.essentia_available:
            return self._analyze_fallback(audio_path)
        
        try:
            # Load audio with Essentia
            loader = self.es.MonoLoader(filename=str(audio_path))
            audio = loader()
            
            # 1. Extract key and scale
            key, scale, strength = self.es.KeyExtractor()(audio)
            
            # 2. Extract tempo
            rhythm_extractor = self.es.RhythmExtractor2013()
            tempo, beats, beats_confidence, _, _ = rhythm_extractor(audio)
            
            # 3. Extract beats positions
            beat_tracker = self.es.BeatTrackerDegara()
            beat_positions = beat_tracker(audio)
            
            # 4. Extract loudness
            loudness = self.es.Loudness()(audio)
            
            # 5. Extract dynamic complexity
            dynamic_complexity = self.es.DynamicComplexity()(audio)
            
            # 6. Extract MFCC
            mfcc = self.es.MFCC()(audio)
            
            # 7. Extract spectral features
            spectral_centroid = self.es.Centroid()(audio)
            spectral_contrast = self.es.SpectralContrast()(audio)
            
            # 8. Extract pitch salience
            pitch_salience = self.es.PitchSalience()(audio)
            
            # 9. Extract chords
            chords = self.es.ChordsDetection()(audio)
            
            # 10. Extract onset rate
            onset_rate = self.es.OnsetRate()(audio)
            
            # 11. Extract danceability
            danceability = self.es.Danceability()(audio)
            
            return {
                'key': key,
                'scale': scale,
                'key_strength': float(strength),
                'tempo': float(tempo),
                'beats': beats.tolist(),
                'beat_positions': beat_positions.tolist(),
                'loudness': float(loudness),
                'dynamic_complexity': float(dynamic_complexity[0]),
                'mfcc': mfcc[0].tolist(),  # MFCC coefficients
                'spectral_centroid': float(spectral_centroid),
                'danceability': float(danceability[0]),
                'onset_rate': float(onset_rate),
                'analysis_success': True
            }
            
        except Exception as e:
            print(f"Essentia analysis failed: {e}")
            return self._analyze_fallback(audio_path)
    
    def _analyze_fallback(self, audio_path: Union[str, Path]) -> Dict:
        """Fallback analysis using librosa"""
        import librosa
        
        y, sr = librosa.load(audio_path, sr=None)
        
        # Key detection (simplified)
        chroma = librosa.feature.chroma_cqt(y=y, sr=sr)
        chroma_mean = chroma.mean(axis=1)
        key_idx = np.argmax(chroma_mean)
        keys = ['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B']
        key = keys[key_idx]
        
        # Tempo and beats
        tempo, beats = librosa.beat.beat_track(y=y, sr=sr)
        beat_times = librosa.frames_to_time(beats, sr=sr)
        
        # Loudness
        rms = librosa.feature.rms(y=y)
        loudness = float(rms.mean())
        
        # MFCC
        mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
        
        # Spectral centroid
        spectral_centroid = librosa.feature.spectral_centroid(y=y, sr=sr).mean()
        
        # Danceability (simplified)
        onset_env = librosa.onset.onset_strength(y=y, sr=sr)
        danceability = float(np.mean(onset_env))
        
        return {
            'key': key,
            'scale': 'major',  # Simplified
            'key_strength': 0.8,
            'tempo': float(tempo),
            'beats': beat_times.tolist(),
            'beat_positions': beat_times.tolist(),
            'loudness': loudness,
            'dynamic_complexity': 0.5,
            'mfcc': mfcc.mean(axis=1).tolist(),
            'spectral_centroid': float(spectral_centroid),
            'danceability': danceability,
            'onset_rate': len(beat_times) / (len(y) / sr),
            'analysis_success': True
        }

# -------------------------------------------------------------------
# MAIN EXECUTION
# -------------------------------------------------------------------

def main():
    """Test the professional separation and analysis"""
    import argparse
    
    parser = argparse.ArgumentParser(description='Professional VocalFusion Tools')
    subparsers = parser.add_subparsers(dest='command', help='Commands')
    
    # Separate command
    sep_parser = subparsers.add_parser('separate', help='Separate audio into stems')
    sep_parser.add_argument('input', help='Input audio file or directory')
    sep_parser.add_argument('--output', help='Output directory')
    sep_parser.add_argument('--model', default='htdemucs_ft', help='Demucs model to use')
    sep_parser.add_argument('--quality', default='balanced', 
                          choices=['fast', 'balanced', 'high_quality', 'professional'],
                          help='Separation quality preset')
    
    # Analyze command
    analyze_parser = subparsers.add_parser('analyze', help='Analyze vocal characteristics')
    analyze_parser.add_argument('input', help='Input audio file (preferably vocal stem)')
    analyze_parser.add_argument('--output', help='Output JSON file for results')
    
    args = parser.parse_args()
    
    if args.command == 'separate':
        print("=== Professional Stem Separation ===")
        
        config = SeparationConfig(
            model_name=args.model,
            quality_preset=args.quality
        )
        
        separator = ProfessionalSeparator(config)
        
        input_path = Path(args.input)
        if input_path.is_file():
            stems = separator.separate_file(input_path, args.output)
            print(f"\nSeparated {len(stems)} stems:")
            for stem_name, stem_path in stems.items():
                print(f"  {stem_name}: {stem_path}")
        elif input_path.is_dir():
            results = separator.separate_directory(input_path, args.output)
            print(f"\nSeparated {len(results)} songs")
        else:
            print(f"Input not found: {input_path}")
    
    elif args.command == 'analyze':
        print("=== Professional Vocal Analysis ===")
        
        analyzer = ProfessionalVocalAnalyzer()
        result = analyzer.analyze(args.input)
        
        # Save results
        if args.output:
            output_path = Path(args.output)
            with open(output_path, 'w') as f:
                json.dump(result.to_dict(), f, indent=2)
            print(f"Analysis saved to: {output_path}")
        else:
            # Print summary
            print(f"\nVocal Analysis Summary:")
            print(f"  Voice Type: {result.voice_type}")
            print(f"  Notes Detected: {len(result.notes)}")
            print(f"  Phrases Detected: {len(result.phrases)}")
            print(f"  Vibrato: {'Yes' if result.vibrato_analysis['detected'] else 'No'}")
            print(f"  Breathiness: {result.voice_quality['breathiness']:.2f}")
            print(f"  Clarity: {result.voice_quality['clarity']:.2f}")
        
        # Also run music analysis
        print("\n=== Music Analysis ===")
        music_analyzer = ProfessionalMusicAnalyzer()
        music_result = music_analyzer.analyze_song(args.input)
        
        if music_result.get('analysis_success'):
            print(f"  Key: {music_result['key']} {music_result['scale']}")
            print(f"  Tempo: {music_result['tempo']:.1f} BPM")
            print(f"  Danceability: {music_result['danceability']:.2f}")
    
    else:
        parser.print_help()

if __name__ == "__main__":
    main()
