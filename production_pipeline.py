"""
Production-grade VocalFusion pipeline integrating professional tools
"""

import os
import sys
import json
import tempfile
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass, asdict
from datetime import datetime
import numpy as np
import soundfile as sf
import librosa
import torch
import warnings
warnings.filterwarnings('ignore')

# Import our professional modules
from professional_separation import (
    ProfessionalSeparator, SeparationConfig, 
    ProfessionalVocalAnalyzer, ProfessionalMusicAnalyzer
)

# -------------------------------------------------------------------
# PRODUCTION PIPELINE CONFIGURATION
# -------------------------------------------------------------------

@dataclass
class PipelineConfig:
    """Configuration for the production pipeline"""
    
    # Directories
    base_dir: str = "VocalFusion"
    cache_dir: str = ".cache/vocalfusion"
    
    # Processing settings
    sample_rate: int = 44100
    bit_depth: int = 24
    normalize_audio: bool = True
    target_lufs: float = -14.0
    
    # Stem separation
    separation_quality: str = "balanced"  # fast, balanced, high_quality, professional
    separation_model: str = "htdemucs_ft"
    keep_stems: bool = True
    
    # Analysis settings
    analyze_vocals: bool = True
    analyze_music: bool = True
    analyze_structure: bool = True
    extract_lyrics: bool = False  # Requires Whisper
    
    # Cache settings
    use_cache: bool = True
    cache_expiry_days: int = 30
    
    # GPU settings
    use_gpu: bool = True
    gpu_memory_limit: Optional[int] = None
    
    # Output settings
    save_intermediate: bool = True
    save_visualizations: bool = True
    save_reports: bool = True
    
    def __post_init__(self):
        """Create directories"""
        Path(self.base_dir).mkdir(parents=True, exist_ok=True)
        Path(self.cache_dir).mkdir(parents=True, exist_ok=True)

# -------------------------------------------------------------------
# PRODUCTION PIPELINE
# -------------------------------------------------------------------

class VocalFusionPipeline:
    """Production-grade vocal fusion pipeline"""
    
    def __init__(self, config: Optional[PipelineConfig] = None):
        self.config = config or PipelineConfig()
        
        # Initialize professional components
        self.separator = ProfessionalSeparator(
            SeparationConfig(
                model_name=self.config.separation_model,
                quality_preset=self.config.separation_quality,
                device="cuda" if self.config.use_gpu and torch.cuda.is_available() else "cpu"
            )
        )
        
        self.vocal_analyzer = ProfessionalVocalAnalyzer(
            sample_rate=self.config.sample_rate
        )
        
        self.music_analyzer = ProfessionalMusicAnalyzer()
        
        # Create pipeline directories
        self._setup_directories()
        
        print(f"=== VocalFusion Production Pipeline ===")
        print(f"Base Directory: {self.config.base_dir}")
        print(f"Sample Rate: {self.config.sample_rate} Hz")
        print(f"Separation Quality: {self.config.separation_quality}")
        print(f"GPU Enabled: {self.config.use_gpu and torch.cuda.is_available()}")
        print("=" * 50)
    
    def _setup_directories(self):
        """Create the full directory structure"""
        dirs = [
            "raw", "stems", "analysis", "compatibility",
            "arrangements", "renders", "exports", "cache",
            "logs", "visualizations", "reports"
        ]
        
        for dir_name in dirs:
            (Path(self.config.base_dir) / dir_name).mkdir(parents=True, exist_ok=True)
    
    def process_song(self, audio_path: Path, 
                    song_id: Optional[str] = None,
                    metadata: Optional[Dict] = None) -> Dict[str, Any]:
        """
        Full processing pipeline for a single song
        
        Returns:
            Dictionary with all processing results and paths
        """
        start_time = datetime.now()
        
        # Generate song ID if not provided
        if song_id is None:
            song_id = self._generate_song_id(audio_path, metadata)
        
        print(f"\n🎵 Processing Song: {song_id}")
        print(f"   Source: {audio_path.name}")
        
        # 1. Prepare raw directory
        raw_dir = Path(self.config.base_dir) / "raw" / song_id
        raw_dir.mkdir(parents=True, exist_ok=True)
        
        # Save original file
        original_path = raw_dir / "original.wav"
        self._convert_to_wav(audio_path, original_path)
        
        # Save metadata
        if metadata:
            with open(raw_dir / "metadata.json", 'w') as f:
                json.dump(metadata, f, indent=2)
        
        # 2. Separate stems using professional separator
        print(f"\n🔧 Step 1: Stem Separation")
        stems_dir = Path(self.config.base_dir) / "stems" / song_id
        stems = self.separator.separate_file(original_path, stems_dir)
        
        # 3. Analyze vocals from vocal stem
        print(f"\n🎤 Step 2: Vocal Analysis")
        vocal_results = {}
        if 'vocals' in stems and self.config.analyze_vocals:
            vocal_path = stems['vocals']
            vocal_results = self.vocal_analyzer.analyze(vocal_path)
            
            # Save vocal analysis
            vocal_analysis_dir = Path(self.config.base_dir) / "analysis" / song_id / "vocals"
            vocal_analysis_dir.mkdir(parents=True, exist_ok=True)
            
            with open(vocal_analysis_dir / "vocal_analysis.json", 'w') as f:
                json.dump(vocal_results.to_dict(), f, indent=2)
        
        # 4. Analyze full song (music characteristics)
        print(f"\n🎼 Step 3: Music Analysis")
        music_results = {}
        if self.config.analyze_music:
            music_results = self.music_analyzer.analyze_song(original_path)
            
            # Save music analysis
            music_analysis_dir = Path(self.config.base_dir) / "analysis" / song_id / "music"
            music_analysis_dir.mkdir(parents=True, exist_ok=True)
            
            with open(music_analysis_dir / "music_analysis.json", 'w') as f:
                json.dump(music_results, f, indent=2)
        
        # 5. Create comprehensive analysis summary
        print(f"\n📊 Step 4: Creating Analysis Summary")
        summary = self._create_analysis_summary(
            song_id, original_path, stems, vocal_results, music_results
        )
        
        analysis_dir = Path(self.config.base_dir) / "analysis" / song_id
        with open(analysis_dir / "summary.json", 'w') as f:
            json.dump(summary, f, indent=2)
        
        # 6. Create visualizations
        if self.config.save_visualizations:
            print(f"\n📈 Step 5: Creating Visualizations")
            self._create_visualizations(song_id, vocal_results, music_results)
        
        # Calculate processing time
        processing_time = (datetime.now() - start_time).total_seconds()
        summary['processing_time_seconds'] = processing_time
        
        print(f"\n✅ Processing Complete!")
        print(f"   Total Time: {processing_time:.1f} seconds")
        print(f"   Stems: {len(stems)}")
        print(f"   Analysis saved to: {analysis_dir}")
        
        return {
            'song_id': song_id,
            'paths': {
                'raw': str(raw_dir),
                'stems': {k: str(v) for k, v in stems.items()},
                'analysis': str(analysis_dir)
            },
            'analysis': summary,
            'processing_time': processing_time
        }
    
    def _generate_song_id(self, audio_path: Path, metadata: Optional[Dict]) -> str:
        """Generate unique song ID"""
        import hashlib
        
        # Use metadata if available
        if metadata and 'title' in metadata and 'artist' in metadata:
            base = f"{metadata['artist']}_{metadata['title']}"
        else:
            base = audio_path.stem
        
        # Add timestamp and hash for uniqueness
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        content_hash = hashlib.md5(audio_path.read_bytes()[:1000]).hexdigest()[:8]
        
        # Clean string for filesystem
        clean_base = ''.join(c if c.isalnum() or c in '_-' else '_' for c in base)
        return f"{clean_base}_{timestamp}_{content_hash}"
    
    def _convert_to_wav(self, source: Path, target: Path):
        """Convert any audio format to high-quality WAV"""
        if source.suffix.lower() == '.wav':
            # Just copy if already WAV
            import shutil
            shutil.copy2(source, target)
        else:
            # Use pydub for conversion
            try:
                from pydub import AudioSegment
                audio = AudioSegment.from_file(source)
                
                # Set parameters for high quality
                audio = audio.set_frame_rate(self.config.sample_rate)
                if self.config.bit_depth == 24:
                    audio = audio.set_sample_width(3)  # 24-bit
                else:
                    audio = audio.set_sample_width(2)  # 16-bit
                
                # Export
                audio.export(target, format='wav', parameters=[
                    '-acodec', 'pcm_s24le' if self.config.bit_depth == 24 else 'pcm_s16le'
                ])
                
            except ImportError:
                # Fallback to librosa
                y, sr = librosa.load(source, sr=self.config.sample_rate, mono=False)
                if y.ndim == 1:
                    y = y.reshape(1, -1)  # Make it 2D for soundfile
                sf.write(target, y.T, sr, subtype='PCM_24' if self.config.bit_depth == 24 else 'PCM_16')
    
    def _create_analysis_summary(self, song_id: str, audio_path: Path, 
                               stems: Dict, vocal_results: Any, 
                               music_results: Dict) -> Dict:
        """Create comprehensive analysis summary"""
        
        # Get audio duration
        import soundfile as sf
        info = sf.info(str(audio_path))
        
        summary = {
            'song_id': song_id,
            'timestamp': datetime.now().isoformat(),
            'audio_info': {
                'duration': info.duration,
                'sample_rate': info.samplerate,
                'channels': info.channels,
                'format': info.format
            },
            'stems_available': list(stems.keys()),
            'vocal_analysis': {
                'voice_type': getattr(vocal_results, 'voice_type', 'unknown'),
                'note_count': len(getattr(vocal_results, 'notes', [])),
                'phrase_count': len(getattr(vocal_results, 'phrases', [])),
                'has_vibrato': getattr(getattr(vocal_results, 'vibrato_analysis', {}), 'detected', False)
            } if vocal_results else None,
            'music_analysis': {
                'key': music_results.get('key', 'unknown'),
                'tempo': music_results.get('tempo', 0),
                'danceability': music_results.get('danceability', 0),
                'loudness': music_results.get('loudness', 0)
            } if music_results else None,
            'pipeline_version': '1.0.0',
            'config': asdict(self.config)
        }
        
        return summary
    
    def _create_visualizations(self, song_id: str, vocal_results: Any, music_results: Dict):
        """Create visualization plots"""
        try:
            import matplotlib.pyplot as plt
            import seaborn as sns
            
            viz_dir = Path(self.config.base_dir) / "visualizations" / song_id
            viz_dir.mkdir(parents=True, exist_ok=True)
            
            # 1. Pitch contour plot
            if hasattr(vocal_results, 'pitch_contour'):
                plt.figure(figsize=(12, 4))
                f0 = vocal_results.pitch_contour
                times = vocal_results.f0_times
                voiced = vocal_results.voiced_flags
                
                if np.any(voiced):
                    plt.plot(times[voiced], f0[voiced], 'b-', alpha=0.7, linewidth=1)
                    plt.fill_between(times[voiced], 0, f0[voiced], alpha=0.2)
                
                plt.xlabel('Time (s)')
                plt.ylabel('Frequency (Hz)')
                plt.title('Vocal Pitch Contour')
                plt.grid(True, alpha=0.3)
                plt.tight_layout()
                plt.savefig(viz_dir / "pitch_contour.png", dpi=150)
                plt.close()
            
            # 2. Voice type visualization
            plt.figure(figsize=(8, 6))
            voice_type = getattr(vocal_results, 'voice_type', 'unknown')
            
            # Simple visualization
            voice_ranges = {
                'soprano': (260, 1047),
                'mezzo-soprano': (220, 880),
                'alto': (175, 700),
                'tenor': (130, 523),
                'baritone': (110, 392),
                'bass': (82, 330)
            }
            
            if voice_type in voice_ranges:
                low, high = voice_ranges[voice_type]
                plt.barh(['Range'], [high - low], left=low)
                plt.xlim(50, 1200)
                plt.xlabel('Frequency (Hz)')
                plt.title(f'Voice Type: {voice_type.title()}')
            
            plt.tight_layout()
            plt.savefig(viz_dir / "voice_type.png", dpi=150)
            plt.close()
            
            print(f"   Visualizations saved to: {viz_dir}")
            
        except ImportError:
            print("   Visualization skipped (matplotlib not available)")

# -------------------------------------------------------------------
# COMPATIBILITY ANALYSIS
# -------------------------------------------------------------------

class ProfessionalCompatibilityAnalyzer:
    """Professional compatibility analysis between songs"""
    
    def __init__(self, base_dir: str = "VocalFusion"):
        self.base_dir = Path(base_dir)
        
    def analyze_compatibility(self, song_a_id: str, song_b_id: str) -> Dict:
        """Analyze compatibility between two songs"""
        
        # Load analyses
        analysis_a = self._load_analysis(song_a_id)
        analysis_b = self._load_analysis(song_b_id)
        
        # 1. Key compatibility
        key_comp = self._analyze_key_compatibility(analysis_a, analysis_b)
        
        # 2. Tempo compatibility
        tempo_comp = self._analyze_tempo_compatibility(analysis_a, analysis_b)
        
        # 3. Vocal range compatibility
        range_comp = self._analyze_vocal_range_compatibility(analysis_a, analysis_b)
        
        # 4. Structure compatibility
        structure_comp = self._analyze_structure_compatibility(analysis_a, analysis_b)
        
        # 5. Energy compatibility
        energy_comp = self._analyze_energy_compatibility(analysis_a, analysis_b)
        
        # Calculate overall score
        weights = {
            'key': 0.25,
            'tempo': 0.20,
            'range': 0.25,
            'structure': 0.15,
            'energy': 0.15
        }
        
        overall = (
            key_comp['score'] * weights['key'] +
            tempo_comp['score'] * weights['tempo'] +
            range_comp['score'] * weights['range'] +
            structure_comp['score'] * weights['structure'] +
            energy_comp['score'] * weights['energy']
        )
        
        # Generate recommendations
        recommendations = self._generate_recommendations(
            analysis_a, analysis_b, key_comp, tempo_comp, range_comp
        )
        
        result = {
            'song_a': song_a_id,
            'song_b': song_b_id,
            'overall_score': float(overall),
            'category_scores': {
                'key_compatibility': key_comp,
                'tempo_compatibility': tempo_comp,
                'vocal_range_compatibility': range_comp,
                'structure_compatibility': structure_comp,
                'energy_compatibility': energy_comp
            },
            'recommendations': recommendations,
            'fusion_strategies': self._suggest_fusion_strategies(analysis_a, analysis_b, overall),
            'analysis_timestamp': datetime.now().isoformat()
        }
        
        # Save compatibility analysis
        comp_dir = self.base_dir / "compatibility" / f"{song_a_id}__{song_b_id}"
        comp_dir.mkdir(parents=True, exist_ok=True)
        
        with open(comp_dir / "compatibility_report.json", 'w') as f:
            json.dump(result, f, indent=2)
        
        return result
    
    def _load_analysis(self, song_id: str) -> Dict:
        """Load song analysis"""
        summary_path = self.base_dir / "analysis" / song_id / "summary.json"
        with open(summary_path, 'r') as f:
            return json.load(f)
    
    def _analyze_key_compatibility(self, a: Dict, b: Dict) -> Dict:
        """Analyze key compatibility using music theory"""
        key_a = a.get('music_analysis', {}).get('key', 'C')
        key_b = b.get('music_analysis', {}).get('key', 'C')
        
        # Simple compatibility based on circle of fifths
        circle_of_fifths = ['C', 'G', 'D', 'A', 'E', 'B', 'F#', 'C#', 'G#', 'D#', 'A#', 'F']
        
        if key_a == key_b:
            score = 1.0
            relationship = "same_key"
        elif key_a in circle_of_fifths and key_b in circle_of_fifths:
            idx_a = circle_of_fifths.index(key_a)
            idx_b = circle_of_fifths.index(key_b)
            distance = min(abs(idx_a - idx_b), len(circle_of_fifths) - abs(idx_a - idx_b))
            
            if distance == 1:  # Perfect fifth/fourth
                score = 0.9
                relationship = "perfect_fifth"
            elif distance == 2:  # Major second
                score = 0.7
                relationship = "major_second"
            elif distance == 6:  # Tritone
                score = 0.3
                relationship = "tritone"
            else:
                score = 0.5
                relationship = "distant"
        else:
            score = 0.5
            relationship = "unknown"
        
        return {
            'score': score,
            'relationship': relationship,
            'key_a': key_a,
            'key_b': key_b,
            'recommendation': "Keep original keys" if score > 0.7 else "Consider transposition"
        }
    
    def _analyze_tempo_compatibility(self, a: Dict, b: Dict) -> Dict:
        """Analyze tempo compatibility"""
        tempo_a = a.get('music_analysis', {}).get('tempo', 120)
        tempo_b = b.get('music_analysis', {}).get('tempo', 120)
        
        ratio = min(tempo_a, tempo_b) / max(tempo_a, tempo_b)
        
        if ratio > 0.95:
            score = 1.0
            adjustment = 1.0
            recommendation = "Tempos match perfectly"
        elif ratio > 0.9:
            score = 0.8
            adjustment = tempo_a / tempo_b
            recommendation = "Minor tempo adjustment needed"
        elif ratio > 0.8:
            score = 0.6
            adjustment = tempo_a / tempo_b
            recommendation = "Moderate tempo adjustment recommended"
        else:
            score = 0.3
            adjustment = (tempo_a + tempo_b) / (2 * tempo_b)
            recommendation = "Significant tempo adjustment required"
        
        return {
            'score': score,
            'tempo_a': tempo_a,
            'tempo_b': tempo_b,
            'adjustment_ratio': adjustment,
            'recommendation': recommendation
        }
    
    def _analyze_vocal_range_compatibility(self, a: Dict, b: Dict) -> Dict:
        """Analyze vocal range compatibility"""
        # This would require detailed vocal analysis data
        # For now, use placeholder
        
        voice_a = a.get('vocal_analysis', {}).get('voice_type', 'unknown')
        voice_b = b.get('vocal_analysis', {}).get('voice_type', 'unknown')
        
        # Simple compatibility based on voice types
        compatible_pairs = [
            ('soprano', 'alto'),
            ('tenor', 'baritone'),
            ('soprano', 'tenor'),
            ('alto', 'bass')
        ]
        
        pair = (voice_a, voice_b)
        if voice_a == voice_b:
            score = 0.6  # Same voice type can work but may lack contrast
        elif pair in compatible_pairs or (pair[1], pair[0]) in compatible_pairs:
            score = 0.9  # Complementary voice types
        else:
            score = 0.5  # Unknown compatibility
        
        return {
            'score': score,
            'voice_a': voice_a,
            'voice_b': voice_b,
            'recommendation': "Good vocal blend expected" if score > 0.7 else "May need EQ adjustment"
        }
    
    def _analyze_structure_compatibility(self, a: Dict, b: Dict) -> Dict:
        """Analyze song structure compatibility"""
        duration_a = a.get('audio_info', {}).get('duration', 180)
        duration_b = b.get('audio_info', {}).get('duration', 180)
        
        # Compare durations
        ratio = min(duration_a, duration_b) / max(duration_a, duration_b)
        
        if ratio > 0.8:
            score = 0.9
            recommendation = "Similar durations, easy to align"
        elif ratio > 0.6:
            score = 0.7
            recommendation = "Moderate duration difference"
        else:
            score = 0.4
            recommendation = "Significant duration difference, may need editing"
        
        return {
            'score': score,
            'duration_a': duration_a,
            'duration_b': duration_b,
            'recommendation': recommendation
        }
    
    def _analyze_energy_compatibility(self, a: Dict, b: Dict) -> Dict:
        """Analyze energy/dynamic compatibility"""
        loudness_a = a.get('music_analysis', {}).get('loudness', 0)
        loudness_b = b.get('music_analysis', {}).get('loudness', 0)
        
        diff = abs(loudness_a - loudness_b)
        
        if diff < 3:
            score = 0.9
            recommendation = "Similar energy levels"
        elif diff < 6:
            score = 0.7
            recommendation = "Moderate energy difference"
        else:
            score = 0.5
            recommendation = "Significant energy difference, may need level adjustment"
        
        return {
            'score': score,
            'loudness_a': loudness_a,
            'loudness_b': loudness_b,
            'recommendation': recommendation
        }
    
    def _generate_recommendations(self, a: Dict, b: Dict, 
                                 key_comp: Dict, tempo_comp: Dict, 
                                 range_comp: Dict) -> List[str]:
        """Generate specific recommendations for fusion"""
        recommendations = []
        
        # Key recommendations
        if key_comp['score'] < 0.7:
            recommendations.append(f"Transpose one song to match keys: {key_comp['key_a']} → {key_comp['key_b']}")
        
        # Tempo recommendations
        if tempo_comp['score'] < 0.8:
            recommendations.append(f"Adjust tempo by {tempo_comp['adjustment_ratio']:.2f}x")
        
        # Vocal range recommendations
        if range_comp['score'] < 0.7:
            recommendations.append(f"Consider octave shifting for better vocal blend")
        
        # General recommendations
        recommendations.append("Use call-and-response arrangement for contrasting vocals")
        recommendations.append("Create instrumental breaks between vocal sections")
        
        return recommendations
    
    def _suggest_fusion_strategies(self, a: Dict, b: Dict, overall_score: float) -> List[Dict]:
        """Suggest fusion strategies based on compatibility"""
        
        strategies = []
        
        # Based on overall score
        if overall_score > 0.8:
            strategies.append({
                'name': 'seamless_blend',
                'description': 'Blend both songs seamlessly with overlapping vocals',
                'difficulty': 'medium',
                'expected_quality': 'high'
            })
        
        if overall_score > 0.6:
            strategies.append({
                'name': 'call_and_response',
                'description': 'Alternate between songs in call-and-response pattern',
                'difficulty': 'easy',
                'expected_quality': 'medium'
            })
        
        # Always include basic strategies
        strategies.append({
            'name': 'medley',
            'description': 'Play songs back-to-back with smooth transitions',
            'difficulty': 'easy',
            'expected_quality': 'medium'
        })
        
        strategies.append({
            'name': 'instrumental_mashup',
            'description': 'Use instrumentals from both songs with vocals from one',
            'difficulty': 'medium',
            'expected_quality': 'high'
        })
        
        return strategies

# -------------------------------------------------------------------
# COMMAND LINE INTERFACE
# -------------------------------------------------------------------

def main():
    """Command line interface for production pipeline"""
    import argparse
    
    parser = argparse.ArgumentParser(description='VocalFusion Production Pipeline')
    subparsers = parser.add_subparsers(dest='command', help='Commands')
    
    # Process command
    process_parser = subparsers.add_parser('process', help='Process a song through the pipeline')
    process_parser.add_argument('input', help='Input audio file')
    process_parser.add_argument('--name', help='Song name (optional)')
    process_parser.add_argument('--artist', help='Artist name (optional)')
    process_parser.add_argument('--output-dir', help='Output directory (default: VocalFusion)')
    
    # Compatibility command
    comp_parser = subparsers.add_parser('compatibility', help='Analyze compatibility between songs')
    comp_parser.add_argument('song_a', help='First song ID')
    comp_parser.add_argument('song_b', help='Second song ID')
    
    # Batch command
    batch_parser = subparsers.add_parser('batch', help='Process multiple songs')
    batch_parser.add_argument('input_dir', help='Directory containing audio files')
    batch_parser.add_argument('--pattern', default='*.mp3', help='File pattern (default: *.mp3)')
    
    args = parser.parse_args()
    
    if args.command == 'process':
        # Process single song
        config = PipelineConfig()
        if args.output_dir:
            config.base_dir = args.output_dir
        
        pipeline = VocalFusionPipeline(config)
        
        metadata = {}
        if args.name:
            metadata['title'] = args.name
        if args.artist:
            metadata['artist'] = args.artist
        
        result = pipeline.process_song(Path(args.input), metadata=metadata)
        
        print(f"\n🎉 Processing Complete!")
        print(f"Song ID: {result['song_id']}")
        print(f"Stems extracted: {len(result['paths']['stems'])}")
        print(f"Analysis saved to: {result['paths']['analysis']}")
        
    elif args.command == 'compatibility':
        # Analyze compatibility
        analyzer = ProfessionalCompatibilityAnalyzer()
        result = analyzer.analyze_compatibility(args.song_a, args.song_b)
        
        print(f"\n🔗 Compatibility Analysis: {args.song_a} ↔ {args.song_b}")
        print(f"Overall Score: {result['overall_score']:.2f}/1.0")
        
        print("\nCategory Scores:")
        for category, scores in result['category_scores'].items():
            print(f"  {category.replace('_', ' ').title()}: {scores['score']:.2f}")
        
        print("\nTop Recommendations:")
        for i, rec in enumerate(result['recommendations'][:3], 1):
            print(f"  {i}. {rec}")
        
        print("\nSuggested Fusion Strategies:")
        for strategy in result['fusion_strategies'][:2]:
            print(f"  • {strategy['name']}: {strategy['description']}")
        
    elif args.command == 'batch':
        # Process batch of songs
        config = PipelineConfig()
        pipeline = VocalFusionPipeline(config)
        
        input_dir = Path(args.input_dir)
        if not input_dir.exists():
            print(f"Directory not found: {input_dir}")
            return
        
        audio_files = list(input_dir.glob(args.pattern))
        print(f"Found {len(audio_files)} audio files to process")
        
        results = []
        for audio_file in audio_files:
            print(f"\nProcessing: {audio_file.name}")
            try:
                result = pipeline.process_song(audio_file)
                results.append(result['song_id'])
            except Exception as e:
                print(f"  Failed: {e}")
        
        print(f"\n✅ Batch processing complete!")
        print(f"Successfully processed {len(results)} songs:")
        for song_id in results:
            print(f"  • {song_id}")
        
    else:
        parser.print_help()

if __name__ == "__main__":
    main()
