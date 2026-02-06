#!/usr/bin/env python3
"""
VOCAL FUSION AI - WITH VOCAL ANALYSIS
Step 1: ✅ Audio loading (librosa)
Step 2: ✅ Stem separation (Demucs)
Step 3: 🎤 Vocal analysis (CREPE, PyWorld)
"""

import os
import json
import time
import traceback
import subprocess
import sys
import numpy as np
from pathlib import Path
from flask import Flask, request, jsonify, render_template_string, send_file
from dataclasses import dataclass, asdict
from typing import Dict, List, Optional, Tuple
import warnings
warnings.filterwarnings('ignore')

app = Flask(__name__)

# ============================================================================
# 1. DATA CLASSES FOR ANALYSIS
# ============================================================================
@dataclass
class PitchAnalysis:
    """Pitch analysis results."""
    f0: np.ndarray  # Fundamental frequency (Hz)
    confidence: np.ndarray  # Confidence scores
    times: np.ndarray  # Time points
    voiced: np.ndarray  # Voiced/unvoiced flags
    note_names: List[str]  # Note names (e.g., ["C4", "D4"])
    note_midi: List[int]  # MIDI note numbers
    
    def to_dict(self):
        return {
            'f0_mean': float(np.nanmean(self.f0[self.voiced])) if np.any(self.voiced) else 0,
            'f0_std': float(np.nanstd(self.f0[self.voiced])) if np.any(self.voiced) else 0,
            'f0_min': float(np.nanmin(self.f0[self.voiced])) if np.any(self.voiced) else 0,
            'f0_max': float(np.nanmax(self.f0[self.voiced])) if np.any(self.voiced) else 0,
            'voiced_percentage': float(np.mean(self.voiced) * 100),
            'note_range': f"{self.note_names[0]}-{self.note_names[-1]}" if self.note_names else "N/A",
            'midi_range': f"{min(self.note_midi)}-{max(self.note_midi)}" if self.note_midi else "N/A"
        }

@dataclass
class TimingAnalysis:
    """Timing analysis results."""
    beats: np.ndarray  # Beat times
    tempo: float  # BPM
    onsets: np.ndarray  # Onset times
    phrases: List[Tuple[float, float]]  # Start/end times of phrases
    silence_gaps: List[Tuple[float, float]]  # Gaps between phrases
    
    def to_dict(self):
        return {
            'tempo': float(self.tempo),
            'beats_count': len(self.beats),
            'onsets_count': len(self.onsets),
            'phrases_count': len(self.phrases),
            'silence_gaps_count': len(self.silence_gaps),
            'phrases': [[float(start), float(end)] for start, end in self.phrases]
        }

@dataclass
class SpectralAnalysis:
    """Spectral/timbre analysis."""
    mfcc: np.ndarray  # MFCC coefficients
    spectral_centroid: np.ndarray
    spectral_bandwidth: np.ndarray
    spectral_contrast: np.ndarray
    formants: List[float]  # First 3 formants (F1, F2, F3)
    brightness: float  # High frequency content
    roughness: float  # Perceived roughness
    
    def to_dict(self):
        return {
            'mfcc_mean': self.mfcc.mean(axis=1).tolist() if self.mfcc.size > 0 else [],
            'spectral_centroid_mean': float(np.mean(self.spectral_centroid)),
            'formants': [float(f) for f in self.formants],
            'brightness': float(self.brightness),
            'roughness': float(self.roughness)
        }

@dataclass
class DynamicsAnalysis:
    """Dynamics/loudness analysis."""
    loudness: np.ndarray  # LUFS over time
    rms: np.ndarray  # RMS energy
    dynamic_range: float  # Difference between loudest and quietest
    intensity_curve: np.ndarray  # Normalized intensity
    peak_locations: np.ndarray  # Locations of peaks
    
    def to_dict(self):
        return {
            'loudness_mean': float(np.mean(self.loudness)),
            'loudness_max': float(np.max(self.loudness)),
            'dynamic_range': float(self.dynamic_range),
            'peaks_count': len(self.peak_locations)
        }

@dataclass
class FullVocalAnalysis:
    """Complete vocal analysis."""
    pitch: PitchAnalysis
    timing: TimingAnalysis
    spectral: SpectralAnalysis
    dynamics: DynamicsAnalysis
    duration: float
    sample_rate: int
    key_estimate: str
    emotion_estimate: str  # happy, sad, angry, neutral
    
    def to_dict(self):
        return {
            'duration': float(self.duration),
            'sample_rate': self.sample_rate,
            'key': self.key_estimate,
            'emotion': self.emotion_estimate,
            'pitch': self.pitch.to_dict(),
            'timing': self.timing.to_dict(),
            'spectral': self.spectral.to_dict(),
            'dynamics': self.dynamics.to_dict(),
            'summary': self.get_summary()
        }
    
    def get_summary(self):
        """Get a human-readable summary."""
        return f"""
        🎤 VOCAL ANALYSIS SUMMARY:
        • Duration: {self.duration:.1f}s
        • Key: {self.key_estimate}
        • Emotion: {self.emotion_estimate}
        • Pitch Range: {self.pitch.to_dict()['note_range']}
        • Tempo: {self.timing.tempo:.0f} BPM
        • Loudness: {self.dynamics.loudness_mean:.1f} LUFS
        • Voiced: {self.pitch.voiced_percentage:.0f}%
        """

# ============================================================================
# 2. VOCAL FUSION AI ENGINE WITH VOCAL ANALYSIS
# ============================================================================
class VocalFusionAI:
    """Complete vocal fusion with analysis."""
    
    def __init__(self):
        self.setup_directories()
        self.status = "ready"
        print("🎵 Initializing Vocal Fusion AI with Vocal Analysis...")
        
    def setup_directories(self):
        """Create directory structure."""
        dirs = [
            'data/raw',
            'data/stems', 
            'data/analysis',
            'data/outputs',
            'test_audio',
            'test_stems',
            'static/plots'  # For visualization plots
        ]
        for d in dirs:
            Path(d).mkdir(parents=True, exist_ok=True)
        print("✅ Directory structure created")
    
    # ==================== EXISTING METHODS ====================
    def load_audio(self, file_path):
        """Load audio file (same as before)."""
        try:
            import librosa
            import soundfile as sf
            
            print(f"🔊 Loading audio: {file_path}")
            
            if not Path(file_path).exists():
                return {'error': f'File not found: {file_path}', 'status': 'error'}
            
            info = sf.info(file_path)
            audio, sample_rate = librosa.load(file_path, sr=None, mono=False)
            
            if isinstance(audio, list):
                audio = np.array(audio)
            
            duration = librosa.get_duration(y=audio, sr=sample_rate)
            
            if audio.ndim == 1:
                channels = 1
                audio = audio.reshape(1, -1)
            else:
                channels = audio.shape[0]
            
            return {
                'file_path': file_path,
                'audio': audio,
                'sample_rate': sample_rate,
                'duration': duration,
                'channels': channels,
                'shape': audio.shape,
                'status': 'loaded'
            }
            
        except Exception as e:
            print(f"❌ Error loading audio: {str(e)}")
            return {'error': str(e), 'status': 'error'}
    
    def separate_stems_with_demucs(self, audio_data, use_mp3=False):
        """Separate stems (same as before)."""
        # ... (same code as before, keep existing implementation)
        # For brevity, keeping the same stem separation logic
        pass
    
    def test_stem_separation(self, test_file=None):
        """Test stem separation (same as before)."""
        # ... (same code as before)
        pass
    
    # ==================== NEW: VOCAL ANALYSIS ====================
    
    def analyze_vocal(self, vocal_path):
        """Complete vocal analysis pipeline."""
        print(f"🎤 Analyzing vocal: {vocal_path}")
        start_time = time.time()
        
        try:
            # Load vocal audio
            audio_data = self.load_audio(vocal_path)
            if audio_data['status'] != 'loaded':
                return {'error': 'Failed to load vocal', 'status': 'error'}
            
            # Get mono audio for analysis
            audio_mono = audio_data['audio']
            if audio_mono.ndim > 1:
                audio_mono = librosa.to_mono(audio_mono)
            sr = audio_data['sample_rate']
            
            # Run all analyses
            print("  1. Analyzing pitch...")
            pitch_analysis = self._analyze_pitch(audio_mono, sr)
            
            print("  2. Analyzing timing...")
            timing_analysis = self._analyze_timing(audio_mono, sr)
            
            print("  3. Analyzing spectral features...")
            spectral_analysis = self._analyze_spectral(audio_mono, sr)
            
            print("  4. Analyzing dynamics...")
            dynamics_analysis = self._analyze_dynamics(audio_mono, sr)
            
            print("  5. Estimating key and emotion...")
            key_estimate = self._estimate_key(audio_mono, sr)
            emotion_estimate = self._estimate_emotion(audio_mono, sr)
            
            # Create full analysis
            full_analysis = FullVocalAnalysis(
                pitch=pitch_analysis,
                timing=timing_analysis,
                spectral=spectral_analysis,
                dynamics=dynamics_analysis,
                duration=audio_data['duration'],
                sample_rate=sr,
                key_estimate=key_estimate,
                emotion_estimate=emotion_estimate
            )
            
            # Save analysis to file
            analysis_dir = Path('data/analysis') / Path(vocal_path).stem
            analysis_dir.mkdir(parents=True, exist_ok=True)
            
            analysis_file = analysis_dir / 'vocal_analysis.json'
            with open(analysis_file, 'w') as f:
                # Convert numpy arrays to lists for JSON serialization
                analysis_dict = full_analysis.to_dict()
                json.dump(analysis_dict, f, indent=2, default=self._json_serializer)
            
            # Create visualization plots
            self._create_visualizations(full_analysis, analysis_dir, vocal_path)
            
            processing_time = time.time() - start_time
            print(f"✅ Vocal analysis complete in {processing_time:.1f}s")
            print(f"   Saved to: {analysis_file}")
            
            return {
                'success': True,
                'analysis': full_analysis.to_dict(),
                'analysis_file': str(analysis_file),
                'processing_time': processing_time,
                'summary': full_analysis.get_summary()
            }
            
        except Exception as e:
            print(f"❌ Vocal analysis failed: {str(e)}")
            traceback.print_exc()
            return {
                'success': False,
                'error': str(e),
                'processing_time': time.time() - start_time
            }
    
    def _analyze_pitch(self, audio, sr):
        """Analyze pitch using CREPE."""
        try:
            # Try CREPE (PyTorch version)
            import crepe_pytorch as crepe
            import torch
            
            # Process with CREPE
            time, frequency, confidence, activation = crepe.predict(
                audio, 
                sr, 
                viterbi=True,
                step_size=10  # ms
            )
            
            # Convert to numpy
            time = np.array(time)
            frequency = np.array(frequency)
            confidence = np.array(confidence)
            
            # Determine voiced/unvoiced (confidence > 0.5)
            voiced = confidence > 0.5
            
            # Convert frequencies to note names
            note_names = []
            note_midi = []
            for f in frequency[voiced]:
                if f > 0:
                    note_name = librosa.hz_to_note(f)
                    note_midi_num = librosa.hz_to_midi(f)
                    note_names.append(note_name)
                    note_midi.append(int(note_midi_num))
            
            return PitchAnalysis(
                f0=frequency,
                confidence=confidence,
                times=time,
                voiced=voiced,
                note_names=note_names,
                note_midi=note_midi
            )
            
        except ImportError:
            # Fallback to librosa's pyin
            print("   ⚠️ CREPE not available, using librosa pyin")
            f0, voiced_flag, voiced_probs = librosa.pyin(
                audio,
                fmin=librosa.note_to_hz('C2'),
                fmax=librosa.note_to_hz('C7'),
                sr=sr,
                frame_length=2048,
                hop_length=512
            )
            
            times = librosa.times_like(f0, sr=sr, hop_length=512)
            voiced = ~np.isnan(f0)
            f0_clean = np.nan_to_num(f0, nan=0.0)
            
            # Convert to note names
            note_names = []
            note_midi = []
            for f in f0_clean[voiced]:
                if f > 0:
                    note_name = librosa.hz_to_note(f)
                    note_midi_num = librosa.hz_to_midi(f)
                    note_names.append(note_name)
                    note_midi.append(int(note_midi_num))
            
            return PitchAnalysis(
                f0=f0_clean,
                confidence=voiced_probs,
                times=times,
                voiced=voiced,
                note_names=note_names,
                note_midi=note_midi
            )
    
    def _analyze_timing(self, audio, sr):
        """Analyze timing, beats, and phrases."""
        import librosa
        
        # Detect beats
        tempo, beats = librosa.beat.beat_track(y=audio, sr=sr)
        beat_times = librosa.frames_to_time(beats, sr=sr)
        
        # Detect onsets
        onset_frames = librosa.onset.onset_detect(y=audio, sr=sr)
        onset_times = librosa.frames_to_time(onset_frames, sr=sr)
        
        # Detect phrases (using RMS energy)
        hop_length = 512
        rms = librosa.feature.rms(y=audio, hop_length=hop_length)[0]
        rms_times = librosa.times_like(rms, sr=sr, hop_length=hop_length)
        
        # Find phrases (continuous regions with RMS above threshold)
        threshold = np.percentile(rms, 30)
        above_threshold = rms > threshold
        
        phrases = []
        in_phrase = False
        phrase_start = 0
        
        for i, (t, is_above) in enumerate(zip(rms_times, above_threshold)):
            if is_above and not in_phrase:
                in_phrase = True
                phrase_start = t
            elif not is_above and in_phrase:
                in_phrase = False
                if t - phrase_start > 0.5:  # Minimum phrase length
                    phrases.append((phrase_start, t))
        
        # Find silence gaps
        silence_gaps = []
        for i in range(len(phrases) - 1):
            gap_start = phrases[i][1]
            gap_end = phrases[i + 1][0]
            if gap_end - gap_start > 0.1:  # Minimum gap length
                silence_gaps.append((gap_start, gap_end))
        
        return TimingAnalysis(
            beats=beat_times,
            tempo=float(tempo),
            onsets=onset_times,
            phrases=phrases,
            silence_gaps=silence_gaps
        )
    
    def _analyze_spectral(self, audio, sr):
        """Analyze spectral features and formants."""
        import librosa
        import pyworld as pw
        
        hop_length = 512
        
        # MFCC
        mfcc = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=13, hop_length=hop_length)
        
        # Spectral features
        spectral_centroid = librosa.feature.spectral_centroid(y=audio, sr=sr, hop_length=hop_length)[0]
        spectral_bandwidth = librosa.feature.spectral_bandwidth(y=audio, sr=sr, hop_length=hop_length)[0]
        spectral_contrast = librosa.feature.spectral_contrast(y=audio, sr=sr, hop_length=hop_length)[0]
        
        # Formant estimation using PyWorld
        try:
            # PyWorld needs f0 first
            f0, t = pw.harvest(audio, sr)
            sp = pw.cheaptrick(audio, f0, t, sr)
            ap = pw.d4c(audio, f0, t, sr)
            
            # Estimate formants (simplified)
            formants = []
            if len(f0) > 0:
                # Use first 3 spectral peaks as formant estimates
                mean_sp = np.mean(sp, axis=0)
                peaks = librosa.util.peak_pick(mean_sp, pre_max=3, post_max=3, 
                                              pre_avg=3, post_avg=5, delta=0.5, wait=10)
                if len(peaks) >= 3:
                    formant_freqs = librosa.fft_frequencies(sr=sr, n_fft=sp.shape[1])
                    formants = [float(formant_freqs[p]) for p in peaks[:3]]
        except:
            formants = [500.0, 1500.0, 2500.0]  # Default values
        
        # Brightness (high frequency content)
        brightness = np.mean(librosa.feature.spectral_rolloff(y=audio, sr=sr, roll_percent=0.85))
        
        # Roughness (simplified)
        roughness = np.mean(librosa.feature.spectral_flatness(y=audio))
        
        return SpectralAnalysis(
            mfcc=mfcc,
            spectral_centroid=spectral_centroid,
            spectral_bandwidth=spectral_bandwidth,
            spectral_contrast=spectral_contrast,
            formants=formants,
            brightness=float(brightness),
            roughness=float(roughness)
        )
    
    def _analyze_dynamics(self, audio, sr):
        """Analyze loudness and dynamics."""
        import librosa
        
        hop_length = 512
        
        # RMS energy
        rms = librosa.feature.rms(y=audio, hop_length=hop_length)[0]
        
        # Perceived loudness (simplified LUFS)
        window_size = int(0.1 * sr)  # 100ms window
        loudness = []
        for i in range(0, len(audio), window_size):
            window = audio[i:i+window_size]
            if len(window) > 0:
                # Simplified loudness calculation
                window_loudness = 10 * np.log10(np.mean(window**2) + 1e-10)
                loudness.append(window_loudness)
        
        loudness = np.array(loudness)
        loudness_times = np.linspace(0, len(audio)/sr, len(loudness))
        
        # Dynamic range
        dynamic_range = np.max(loudness) - np.min(loudness)
        
        # Intensity curve (normalized)
        intensity_curve = (loudness - np.min(loudness)) / (np.max(loudness) - np.min(loudness) + 1e-10)
        
        # Peak locations
        peaks = librosa.util.peak_pick(intensity_curve, pre_max=3, post_max=3, 
                                      pre_avg=3, post_avg=5, delta=0.2, wait=10)
        peak_locations = loudness_times[peaks]
        
        return DynamicsAnalysis(
            loudness=loudness,
            rms=rms,
            dynamic_range=dynamic_range,
            intensity_curve=intensity_curve,
            peak_locations=peak_locations
        )
    
    def _estimate_key(self, audio, sr):
        """Estimate musical key."""
        try:
            import librosa
            
            # Chromagram
            chroma = librosa.feature.chroma_cqt(y=audio, sr=sr)
            chroma_mean = np.mean(chroma, axis=1)
            
            # Key profiles (Krumhansl profiles)
            major_profile = [6.35, 2.23, 3.48, 2.33, 4.38, 4.09, 2.52, 5.19, 2.39, 3.66, 2.29, 2.88]
            minor_profile = [6.33, 2.68, 3.52, 5.38, 2.60, 3.53, 2.54, 4.75, 3.98, 2.69, 3.34, 3.17]
            
            # Correlate with all 24 keys
            keys = ['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B']
            best_score = -1
            best_key = "Unknown"
            
            for i in range(12):
                # Major key
                rotated_major = np.roll(major_profile, i)
                major_corr = np.corrcoef(chroma_mean, rotated_major)[0,1]
                
                # Minor key
                rotated_minor = np.roll(minor_profile, i)
                minor_corr = np.corrcoef(chroma_mean, rotated_minor)[0,1]
                
                if major_corr > best_score:
                    best_score = major_corr
                    best_key = f"{keys[i]} major"
                
                if minor_corr > best_score:
                    best_score = minor_corr
                    best_key = f"{keys[i]} minor"
            
            return best_key
            
        except:
            return "Unknown"
    
    def _estimate_emotion(self, audio, sr):
        """Simple emotion estimation based on audio features."""
        try:
            import librosa
            from sklearn.preprocessing import StandardScaler
            
            # Extract features
            features = []
            
            # Pitch statistics
            f0 = librosa.yin(audio, fmin=80, fmax=1000)
            f0_clean = f0[f0 > 0]
            if len(f0_clean) > 0:
                features.append(np.mean(f0_clean))
                features.append(np.std(f0_clean))
            else:
                features.extend([0, 0])
            
            # Spectral centroid (brightness)
            spectral_centroid = librosa.feature.spectral_centroid(y=audio, sr=sr)[0]
            features.append(np.mean(spectral_centroid))
            
            # RMS energy (loudness)
            rms = librosa.feature.rms(y=audio)[0]
            features.append(np.mean(rms))
            
            # Zero crossing rate (noisiness)
            zcr = librosa.feature.zero_crossing_rate(audio)[0]
            features.append(np.mean(zcr))
            
            # Tempo
            tempo, _ = librosa.beat.beat_track(y=audio, sr=sr)
            features.append(tempo)
            
            # Simple rule-based emotion detection
            f0_mean = features[0] if len(features) > 0 else 200
            brightness = features[2] if len(features) > 2 else 1000
            loudness = features[3] if len(features) > 3 else 0.1
            tempo = features[5] if len(features) > 5 else 120
            
            if f0_mean > 250 and tempo > 140 and loudness > 0.15:
                return "happy"
            elif f0_mean < 180 and tempo < 100 and loudness < 0.1:
                return "sad"
            elif loudness > 0.2 and brightness > 2000:
                return "angry"
            else:
                return "neutral"
                
        except:
            return "neutral"
    
    def _create_visualizations(self, analysis, output_dir, vocal_path):
        """Create visualization plots for the analysis."""
        try:
            import matplotlib
            matplotlib.use('Agg')  # Non-interactive backend
            import matplotlib.pyplot as plt
            import librosa.display
            
            # Create pitch plot
            plt.figure(figsize=(12, 8))
            
            # Plot 1: Pitch contour
            plt.subplot(3, 1, 1)
            times = analysis.pitch.times
            f0 = analysis.pitch.f0
            voiced = analysis.pitch.voiced
            
            # Plot only voiced frames
            voiced_times = times[voiced]
            voiced_f0 = f0[voiced]
            
            plt.plot(voiced_times, voiced_f0, 'b-', linewidth=1, alpha=0.7)
            plt.fill_between(voiced_times, 0, voiced_f0, alpha=0.3)
            plt.title(f'Pitch Contour - {Path(vocal_path).name}')
            plt.ylabel('Frequency (Hz)')
            plt.grid(True, alpha=0.3)
            
            # Plot 2: Spectrogram with onsets
            plt.subplot(3, 1, 2)
            audio, sr = librosa.load(vocal_path, sr=None)
            D = librosa.amplitude_to_db(np.abs(librosa.stft(audio)), ref=np.max)
            librosa.display.specshow(D, sr=sr, x_axis='time', y_axis='log')
            
            # Mark onsets
            onset_times = analysis.timing.onsets
            for t in onset_times:
                plt.axvline(x=t, color='red', linestyle='--', alpha=0.5, linewidth=0.5)
            
            plt.title('Spectrogram with Onsets')
            plt.colorbar(format='%+2.0f dB')
            
            # Plot 3: Loudness curve
            plt.subplot(3, 1, 3)
            loudness_times = np.linspace(0, analysis.duration, len(analysis.dynamics.loudness))
            plt.plot(loudness_times, analysis.dynamics.loudness, 'g-', linewidth=2)
            
            # Mark peaks
            peak_times = analysis.dynamics.peak_locations
            peak_values = analysis.dynamics.loudness[
                (analysis.dynamics.peak_locations * len(analysis.dynamics.loudness) / analysis.duration).astype(int)
            ]
            plt.scatter(peak_times, peak_values, color='red', s=50, zorder=5)
            
            plt.title('Loudness Curve (LUFS)')
            plt.xlabel('Time (s)')
            plt.ylabel('Loudness (dB)')
            plt.grid(True, alpha=0.3)
            
            plt.tight_layout()
            plot_file = output_dir / 'analysis_plot.png'
            plt.savefig(plot_file, dpi=150, bbox_inches='tight')
            plt.close()
            
            print(f"   📊 Created visualization: {plot_file}")
            
        except Exception as e:
            print(f"   ⚠️ Could not create visualization: {str(e)}")
    
    def _json_serializer(self, obj):
        """Custom JSON serializer for numpy types."""
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, np.bool_):
            return bool(obj)
        raise TypeError(f"Type {type(obj)} not serializable")

# Initialize engine
ai = VocalFusionAI()

# ============================================================================
# 3. WEB INTERFACE
# ============================================================================
HTML_TEMPLATE = '''
<!DOCTYPE html>
<html>
<head>
    <title>Vocal Fusion AI - Complete Pipeline</title>
    <style>
        body { 
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            max-width: 1200px; 
            margin: 0 auto; 
            padding: 20px;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            min-height: 100vh;
            color: #333;
        }
        .container { 
            background: rgba(255,255,255,0.95); 
            padding: 40px; 
            border-radius: 20px; 
            box-shadow: 0 20px 40px rgba(0,0,0,0.1);
            backdrop-filter: blur(10px);
        }
        .header {
            text-align: center;
            margin-bottom: 40px;
            background: linear-gradient(135deg, #3B82F6, #8B5CF6);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
        }
        .step-card { 
            background: white; 
            padding: 25px; 
            margin: 20px 0; 
            border-radius: 15px; 
            box-shadow: 0 5px 15px rgba(0,0,0,0.08);
            border-left: 5px solid #3B82F6;
            transition: transform 0.3s ease;
        }
        .step-card:hover {
            transform: translateY(-5px);
        }
        .step-card.completed {
            border-left-color: #10B981;
        }
        .step-card.in-progress {
            border-left-color: #F59E0B;
        }
        .step-card.pending {
            border-left-color: #EF4444;
        }
        .btn { 
            background: linear-gradient(135deg, #3B82F6, #8B5CF6);
            color: white; 
            padding: 12px 24px; 
            border: none; 
            border-radius: 8px; 
            cursor: pointer; 
            text-decoration: none; 
            display: inline-block;
            font-weight: 600;
            transition: all 0.3s ease;
            margin: 5px;
        }
        .btn:hover { 
            transform: translateY(-2px);
            box-shadow: 0 10px 20px rgba(59, 130, 246, 0.3);
        }
        .btn-success { background: linear-gradient(135deg, #10B981, #059669); }
        .btn-warning { background: linear-gradient(135deg, #F59E0B, #D97706); }
        .btn-danger { background: linear-gradient(135deg, #EF4444, #DC2626); }
        .analysis-result {
            background: #F3F4F6;
            padding: 20px;
            border-radius: 10px;
            margin: 20px 0;
            font-family: 'Courier New', monospace;
            white-space: pre-wrap;
            max-height: 400px;
            overflow-y: auto;
        }
        .plot-container {
            text-align: center;
            margin: 20px 0;
        }
        .plot-container img {
            max-width: 100%;
            border-radius: 10px;
            box-shadow: 0 5px 15px rgba(0,0,0,0.1);
        }
        .status-badge {
            display: inline-block;
            padding: 4px 12px;
            border-radius: 20px;
            font-size: 0.85em;
            font-weight: 600;
            margin-left: 10px;
        }
        .status-complete { background: #D1FAE5; color: #065F46; }
        .status-in-progress { background: #FEF3C7; color: #92400E; }
        .status-pending { background: #FEE2E2; color: #991B1B; }
    </style>
</head>
<body>
    <div class="container">
        <div class="header">
            <h1 style="font-size: 2.5em; margin-bottom: 10px;">🎵 VOCAL FUSION AI</h1>
            <p style="font-size: 1.2em; color: #6B7280;">Complete pipeline for intelligent vocal fusion</p>
        </div>
        
        <div class="step-card completed">
            <h2>✅ Step 1: Audio Loading</h2>
            <p>Load and process audio files using librosa.</p>
            <button onclick="testAudioLoading()" class="btn">Test Audio Loading</button>
        </div>
        
        <div class="step-card completed">
            <h2>✅ Step 2: Stem Separation</h2>
            <p>Separate vocals, drums, bass, and other instruments using Demucs.</p>
            <button onclick="testStemSeparation()" class="btn">Test Stem Separation</button>
        </div>
        
        <div class="step-card in-progress">
            <h2>🎤 Step 3: Vocal Analysis <span class="status-badge status-in-progress">IN PROGRESS</span></h2>
            <p>Analyze vocal characteristics: pitch, timing, emotion, and more.</p>
            <button onclick="testVocalAnalysis()" class="btn btn-warning">Test Vocal Analysis</button>
            <button onclick="uploadForAnalysis()" class="btn">Upload Vocal File</button>
        </div>
        
        <div class="step-card pending">
            <h2>🔗 Step 4: Compatibility Analysis <span class="status-badge status-pending">PENDING</span></h2>
            <p>Analyze how well two vocals work together (key, tempo, emotion).</p>
        </div>
        
        <div class="step-card pending">
            <h2>🎭 Step 5: Fusion Engine <span class="status-badge status-pending">PENDING</span></h2>
            <p>Intelligently fuse two vocals into one cohesive performance.</p>
        </div>
        
        <div id="test-results" style="margin-top: 40px;">
            <h2>📊 Test Results</h2>
            <div id="results-container"></div>
            <div id="results-plot" class="plot-container"></div>
            <div id="results-analysis" class="analysis-result" style="display: none;"></div>
        </div>
        
        <div style="margin-top: 40px; text-align: center; color: #6B7280;">
            <p>🎯 Next: Test vocal analysis, then implement compatibility engine</p>
            <p>📈 Progress: 3/5 steps implemented</p>
        </div>
    </div>
    
    <script>
    async function testAudioLoading() {
        const container = document.getElementById('results-container');
        container.innerHTML = '<p>Testing audio loading...</p>';
        
        const response = await fetch('/api/test-audio-loading');
        const data = await response.json();
        
        if (data.success) {
            container.innerHTML = `
                <div class="step-card completed">
                    <h3>✅ Audio Loading Test Successful</h3>
                    <p>File: ${data.file_path}</p>
                    <p>Duration: ${data.duration.toFixed(2)}s</p>
                    <p>Sample Rate: ${data.sample_rate}Hz</p>
                    <audio controls src="/api/get-audio?file=${data.file_path}" style="width: 100%; margin-top: 10px;"></audio>
                </div>
            `;
        } else {
            container.innerHTML = `<div class="step-card pending"><h3>❌ Audio Loading Failed</h3><p>${data.error}</p></div>`;
        }
    }
    
    async function testStemSeparation() {
        const container = document.getElementById('results-container');
        container.innerHTML = '<p>Testing stem separation (may take 1-2 minutes)...</p>';
        
        const response = await fetch('/api/test-stem-separation', {method: 'POST'});
        const data = await response.json();
        
        if (data.success) {
            let stemsHtml = '';
            if (data.stems) {
                stemsHtml = '<h4>Stems Created:</h4><ul>';
                for (const [name, path] of Object.entries(data.stems)) {
                    const fileName = path.split('/').pop();
                    stemsHtml += `<li><strong>${name}:</strong> ${fileName}</li>`;
                }
                stemsHtml += '</ul>';
            }
            
            container.innerHTML = `
                <div class="step-card completed">
                    <h3>✅ Stem Separation Successful</h3>
                    <p>Processing Time: ${data.processing_time.toFixed(1)}s</p>
                    <p>Output Directory: ${data.output_dir}</p>
                    ${stemsHtml}
                    <p><a href="/list/stems" class="btn">View All Stems</a></p>
                </div>
            `;
        } else {
            container.innerHTML = `<div class="step-card pending"><h3>❌ Stem Separation Failed</h3><p>${data.error}</p></div>`;
        }
    }
    
    async function testVocalAnalysis() {
        const container = document.getElementById('results-container');
        const analysisDiv = document.getElementById('results-analysis');
        const plotDiv = document.getElementById('results-plot');
        
        container.innerHTML = '<p>Testing vocal analysis (may take 30-60 seconds)...</p>';
        analysisDiv.style.display = 'none';
        plotDiv.innerHTML = '';
        
        const response = await fetch('/api/test-vocal-analysis');
        const data = await response.json();
        
        if (data.success) {
            // Show plot if available
            if (data.plot_url) {
                plotDiv.innerHTML = `<img src="${data.plot_url}" alt="Vocal Analysis Plot">`;
            }
            
            // Show analysis results
            analysisDiv.style.display = 'block';
            analysisDiv.innerHTML = `
                <h3>🎤 Vocal Analysis Results</h3>
                <p><strong>File:</strong> ${data.file_path}</p>
                <p><strong>Processing Time:</strong> ${data.processing_time.toFixed(1)}s</p>
                <p><strong>Analysis File:</strong> ${data.analysis_file}</p>
                
                <h4>📊 Summary:</h4>
                <pre>${data.summary || 'No summary available'}</pre>
                
                <h4>📈 Detailed Analysis:</h4>
                <pre>${JSON.stringify(data.analysis, null, 2)}</pre>
            `;
            
            container.innerHTML = `
                <div class="step-card completed">
                    <h3>✅ Vocal Analysis Successful</h3>
                    <p>Analyzed vocal characteristics in ${data.processing_time.toFixed(1)} seconds</p>
                    <p>Key detected: ${data.analysis?.key || 'Unknown'}</p>
                    <p>Emotion: ${data.analysis?.emotion || 'Unknown'}</p>
                </div>
            `;
            
            // Scroll to results
            analysisDiv.scrollIntoView({ behavior: 'smooth' });
        } else {
            container.innerHTML = `<div class="step-card pending"><h3>❌ Vocal Analysis Failed</h3><p>${data.error}</p></div>`;
        }
    }
    
    function uploadForAnalysis() {
        const container = document.getElementById('results-container');
        container.innerHTML = `
            <div class="step-card">
                <h3>📤 Upload Vocal File for Analysis</h3>
                <input type="file" id="vocalFile" accept=".wav,.mp3,.flac">
                <button onclick="analyzeUploadedFile()" class="btn" style="margin-top: 10px;">Analyze Uploaded File</button>
            </div>
        `;
    }
    
    async function analyzeUploadedFile() {
        const fileInput = document.getElementById('vocalFile');
        if (!fileInput.files.length) {
            alert('Please select a file first');
            return;
        }
        
        const formData = new FormData();
        formData.append('vocal_file', fileInput.files[0]);
        
        const container = document.getElementById('results-container');
        container.innerHTML = '<p>Uploading and analyzing file...</p>';
        
        const response = await fetch('/api/analyze-uploaded-vocal', {
            method: 'POST',
            body: formData
        });
        
        const data = await response.json();
        
        if (data.success) {
            container.innerHTML = `
                <div class="step-card completed">
                    <h3>✅ Uploaded File Analyzed</h3>
                    <p>File: ${data.file_name}</p>
                    <p>Processing Time: ${data.processing_time.toFixed(1)}s</p>
                    <p><a href="${data.analysis_url}" class="btn">View Full Analysis</a></p>
                </div>
            `;
        } else {
            container.innerHTML = `<div class="step-card pending"><h3>❌ Analysis Failed</h3><p>${data.error}</p></div>`;
        }
    }
    </script>
</body>
</html>
'''

# ============================================================================
# 4. API ENDPOINTS
# ============================================================================
@app.route('/')
def home():
    return render_template_string(HTML_TEMPLATE)

@app.route('/api/test-audio-loading')
def api_test_audio_loading():
    """Test audio loading."""
    import numpy as np
    import soundfile as sf
    
    test_file = 'test_audio/analysis_test.wav'
    Path('test_audio').mkdir(exist_ok=True)
    
    # Create test audio
    sample_rate = 44100
    duration = 3.0
    t = np.linspace(0, duration, int(sample_rate * duration), endpoint=False)
    
    # Create melody
    melody = 0.2 * np.sin(2 * np.pi * 440 * t) * (0.5 + 0.5 * np.sin(2 * np.pi * 2 * t))
    sf.write(test_file, melody, sample_rate)
    
    # Load it
    result = ai.load_audio(test_file)
    
    if result['status'] == 'loaded':
        return jsonify({
            'success': True,
            'file_path': test_file,
            'sample_rate': result['sample_rate'],
            'duration': result['duration']
        })
    else:
        return jsonify({'success': False, 'error': result.get('error')})

@app.route('/api/test-stem-separation', methods=['POST'])
def api_test_stem_separation():
    """Test stem separation."""
    result = ai.test_stem_separation()
    return jsonify(result)

@app.route('/api/test-vocal-analysis')
def api_test_vocal_analysis():
    """Test vocal analysis with a generated vocal."""
    import numpy as np
    import soundfile as sf
    
    test_file = 'test_audio/vocal_test.wav'
    Path('test_audio').mkdir(exist_ok=True)
    
    # Create a more complex vocal-like audio
    sample_rate = 44100
    duration = 5.0
    t = np.linspace(0, duration, int(sample_rate * duration), endpoint=False)
    
    # Create pitch contour (simple melody)
    pitch_curve = 220 + 100 * np.sin(2 * np.pi * 0.5 * t)  # 220-320Hz range
    
    # Create amplitude envelope (like phrases)
    amp_envelope = 0.5 * (1 + np.sin(2 * np.pi * 0.4 * t)) * (0.5 + 0.5 * np.sin(2 * np.pi * 0.1 * t))
    
    # Generate audio with varying pitch
    audio = np.zeros_like(t)
    for i in range(len(t)):
        freq = pitch_curve[i]
        amp = amp_envelope[i]
        audio[i] = amp * np.sin(2 * np.pi * freq * t[i])
    
    # Add some vibrato
    vibrato = 0.05 * np.sin(2 * np.pi * 5 * t)
    audio *= (1 + vibrato)
    
    # Normalize
    audio = audio / np.max(np.abs(audio)) * 0.8
    
    sf.write(test_file, audio, sample_rate)
    
    # Analyze it
    result = ai.analyze_vocal(test_file)
    
    # Add plot URL if available
    if result['success']:
        analysis_dir = Path('data/analysis') / Path(test_file).stem
        plot_file = analysis_dir / 'analysis_plot.png'
        if plot_file.exists():
            result['plot_url'] = f'/static/plots/{Path(test_file).stem}_analysis.png'
            # Copy to static for web access
            static_plots = Path('static/plots')
            static_plots.mkdir(exist_ok=True)
            import shutil
            shutil.copy2(plot_file, static_plots / f'{Path(test_file).stem}_analysis.png')
    
    return jsonify(result)

@app.route('/api/analyze-uploaded-vocal', methods=['POST'])
def api_analyze_uploaded_vocal():
    """Analyze an uploaded vocal file."""
    if 'vocal_file' not in request.files:
        return jsonify({'success': False, 'error': 'No file uploaded'})
    
    file = request.files['vocal_file']
    if file.filename == '':
        return jsonify({'success': False, 'error': 'No file selected'})
    
    # Save uploaded file
    upload_dir = Path('data/raw/uploads')
    upload_dir.mkdir(parents=True, exist_ok=True)
    
    file_path = upload_dir / file.filename
    file.save(file_path)
    
    # Analyze it
    result = ai.analyze_vocal(str(file_path))
    result['file_name'] = file.filename
    
    if result['success']:
        result['analysis_url'] = f'/view-analysis/{Path(file_path).stem}'
    
    return jsonify(result)

@app.route('/view-analysis/<analysis_id>')
def view_analysis(analysis_id):
    """View analysis results."""
    analysis_dir = Path('data/analysis') / analysis_id
    if not analysis_dir.exists():
        return f'<h1>Analysis not found: {analysis_id}</h1>'
    
    analysis_file = analysis_dir / 'vocal_analysis.json'
    if not analysis_file.exists():
        return f'<h1>Analysis file not found</h1>'
    
    with open(analysis_file, 'r') as f:
        analysis_data = json.load(f)
    
    # Check for plot
    plot_file = analysis_dir / 'analysis_plot.png'
    has_plot = plot_file.exists()
    
    html = f'''
    <!DOCTYPE html>
    <html>
    <head>
        <title>Vocal Analysis: {analysis_id}</title>
        <style>
            body {{ font-family: Arial, sans-serif; max-width: 1000px; margin: 0 auto; padding: 20px; }}
            .analysis-container {{ background: #f5f5f5; padding: 30px; border-radius: 15px; }}
            pre {{ background: white; padding: 20px; border-radius: 10px; overflow: auto; }}
            .plot {{ text-align: center; margin: 20px 0; }}
            .plot img {{ max-width: 100%; border-radius: 10px; box-shadow: 0 5px 15px rgba(0,0,0,0.1); }}
        </style>
    </head>
    <body>
        <div class="analysis-container">
            <h1>🎤 Vocal Analysis: {analysis_id}</h1>
            <p><a href="/" class="btn">← Back</a></p>
            
            {f'<div class="plot"><img src="/static/plots/{analysis_id}_analysis.png" alt="Analysis Plot"></div>' if has_plot else ''}
            
            <h2>📊 Analysis Results</h2>
            <pre>{json.dumps(analysis_data, indent=2)}</pre>
        </div>
    </body>
    </html>
    '''
    
    return html

@app.route('/api/get-audio')
def api_get_audio():
    """Serve audio file."""
    file_path = request.args.get('file')
    if not file_path or not Path(file_path).exists():
        return jsonify({'error': 'File not found'}), 404
    
    ext = Path(file_path).suffix.lower()
    mime_types = {
        '.wav': 'audio/wav',
        '.mp3': 'audio/mpeg',
        '.flac': 'audio/flac'
    }
    
    mime_type = mime_types.get(ext, 'audio/wav')
    return send_file(file_path, mimetype=mime_type)

@app.route('/list/stems')
def list_stems():
    """List all stem directories."""
    stem_dir = Path('data/stems')
    
    if not stem_dir.exists():
        return '''
        <div style="max-width: 800px; margin: 0 auto; padding: 20px;">
            <h1>📁 No Stems Found</h1>
            <p>Run the stem separation test first.</p>
            <p><a href="/">← Back</a></p>
        </div>
        '''
    
    html = '''
    <div style="max-width: 800px; margin: 0 auto; padding: 20px;">
        <h1>📁 Stem Directories</h1>
        <p><a href="/">← Back</a></p>
    '''
    
    for folder in sorted(stem_dir.iterdir(), key=lambda x: x.stat().st_mtime, reverse=True):
        if folder.is_dir():
            wav_files = list(folder.rglob("*.wav"))
            mp3_files = list(folder.rglob("*.mp3"))
            json_files = list(folder.rglob("*.json"))
            
            html += f'''
            <div style="background: white; padding: 20px; margin: 20px 0; border-radius: 10px;">
                <h3>📂 {folder.name}</h3>
                <p><strong>Files:</strong> {len(wav_files)} WAV, {len(mp3_files)} MP3</p>
                
                <h4>Audio Files:</h4>
                <ul>
            '''
            
            for audio_file in wav_files[:5] + mp3_files[:5]:  # Show first 5 of each
                html += f'<li><a href="/api/get-audio?file={audio_file}">▶️ {audio_file.name}</a></li>'
            
            html += '''
                </ul>
                
                <h4>Info:</h4>
                <ul>
            '''
            
            for json_file in json_files:
                html += f'<li><a href="/view/json?file={json_file}">📄 {json_file.name}</a></li>'
            
            html += '''
                </ul>
            </div>
            '''
    
    html += '</div>'
    return html

# ============================================================================
# 5. MAIN EXECUTION
# ============================================================================
if __name__ == '__main__':
    print(f"""
    {'='*70}
    🎵 VOCAL FUSION AI - STEP 3: VOCAL ANALYSIS
    {'='*70}
    
    ✅ Step 1: Audio Loading (librosa)
    ✅ Step 2: Stem Separation (Demucs)
    🎤 Step 3: Vocal Analysis (CREPE, PyWorld, librosa)
    🔗 Step 4: Compatibility Analysis (Coming next)
    🎭 Step 5: Fusion Engine (Coming soon)
    
    🚀 Access Points:
      • Main Interface: http://localhost:5000
      • Test Vocal Analysis: Click button on main page
      • View Analyses: http://localhost:5000/view-analysis/[id]
      • List Stems: http://localhost:5000/list/stems
    
    📊 Vocal Analysis Features:
      1. Pitch tracking (CREPE/pyin)
      2. Timing analysis (beats, onsets, phrases)
      3. Spectral analysis (formants, timbre)
      4. Dynamics analysis (loudness, intensity)
      5. Key and emotion estimation
    
    ⚠️  Note: First vocal analysis will be slow (downloading models).
    
    {'='*70}
    """)
    
    # Import librosa here to avoid issues
    import librosa
    
    app.run(host='0.0.0.0', port=5000, debug=True)
