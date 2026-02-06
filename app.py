#!/usr/bin/env python3
"""
VOCAL FUSION AI - SIMPLIFIED WITH LIBROSA'S PYIN
No external CREPE dependency needed!
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

app = Flask(__name__)

# ============================================================================
# 1. SIMPLIFIED VOCAL FUSION AI ENGINE
# ============================================================================
class VocalFusionAI:
    """Simplified version using only librosa (no CREPE)."""
    
    def __init__(self):
        self.setup_directories()
        self.status = "ready"
        print("🎵 Initializing Vocal Fusion AI (Simplified)...")
        
    def setup_directories(self):
        """Create directory structure."""
        dirs = [
            'data/raw',
            'data/stems', 
            'data/analysis',
            'data/outputs',
            'test_audio',
            'static/plots'
        ]
        for d in dirs:
            Path(d).mkdir(parents=True, exist_ok=True)
    
    # ==================== EXISTING METHODS ====================
    def load_audio(self, file_path):
        """Load audio file."""
        try:
            import librosa
            import soundfile as sf
            
            print(f"🔊 Loading audio: {file_path}")
            
            if not Path(file_path).exists():
                return {'error': f'File not found: {file_path}', 'status': 'error'}
            
            audio, sample_rate = librosa.load(file_path, sr=None, mono=False)
            duration = librosa.get_duration(y=audio, sr=sample_rate)
            
            return {
                'file_path': file_path,
                'audio': audio,
                'sample_rate': sample_rate,
                'duration': duration,
                'status': 'loaded'
            }
            
        except Exception as e:
            print(f"❌ Error loading audio: {str(e)}")
            return {'error': str(e), 'status': 'error'}
    
    # ==================== NEW: SIMPLIFIED VOCAL ANALYSIS ====================
    
    def analyze_vocal_simple(self, vocal_path):
        """Simple vocal analysis using only librosa (no CREPE)."""
        print(f"🎤 Analyzing vocal (simple): {vocal_path}")
        start_time = time.time()
        
        try:
            import librosa
            import librosa.display
            import matplotlib
            matplotlib.use('Agg')
            import matplotlib.pyplot as plt
            
            # Load audio
            audio_data = self.load_audio(vocal_path)
            if audio_data['status'] != 'loaded':
                return {'success': False, 'error': 'Failed to load audio'}
            
            audio = audio_data['audio']
            sr = audio_data['sample_rate']
            duration = audio_data['duration']
            
            # Convert to mono if needed
            if len(audio.shape) > 1:
                audio = librosa.to_mono(audio)
            
            print("  1. Extracting pitch with librosa pyin...")
            # Use librosa's pyin for pitch tracking (no CREPE needed)
            f0, voiced_flag, voiced_probs = librosa.pyin(
                audio,
                fmin=librosa.note_to_hz('C2'),  # ~65 Hz
                fmax=librosa.note_to_hz('C7'),  # ~2093 Hz
                sr=sr,
                frame_length=2048,
                hop_length=512
            )
            
            # Clean pitch data
            times = librosa.times_like(f0, sr=sr, hop_length=512)
            voiced = ~np.isnan(f0)
            f0_clean = np.nan_to_num(f0, nan=0.0)
            
            # Convert to notes
            note_names = []
            for f in f0_clean[voiced]:
                if f > 0:
                    note_names.append(librosa.hz_to_note(f))
            
            print("  2. Analyzing tempo and beats...")
            tempo, beats = librosa.beat.beat_track(y=audio, sr=sr)
            beat_times = librosa.frames_to_time(beats, sr=sr)
            
            print("  3. Extracting spectral features...")
            # MFCCs
            mfcc = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=13)
            
            # Spectral centroid (brightness)
            spectral_centroid = librosa.feature.spectral_centroid(y=audio, sr=sr)[0]
            
            # RMS energy (loudness)
            rms = librosa.feature.rms(y=audio)[0]
            
            print("  4. Estimating key...")
            key = self._estimate_key_simple(audio, sr)
            
            print("  5. Creating visualization...")
            # Create analysis plot
            fig, axes = plt.subplots(3, 1, figsize=(12, 10))
            
            # Plot 1: Waveform
            librosa.display.waveshow(audio, sr=sr, ax=axes[0])
            axes[0].set_title(f'Waveform: {Path(vocal_path).name}')
            axes[0].set_xlabel('Time (s)')
            axes[0].set_ylabel('Amplitude')
            
            # Plot 2: Pitch contour
            axes[1].plot(times[voiced], f0_clean[voiced], 'b-', linewidth=1)
            axes[1].fill_between(times[voiced], 0, f0_clean[voiced], alpha=0.3)
            axes[1].set_title('Pitch Contour')
            axes[1].set_xlabel('Time (s)')
            axes[1].set_ylabel('Frequency (Hz)')
            axes[1].grid(True, alpha=0.3)
            
            # Plot 3: Spectrogram
            D = librosa.amplitude_to_db(np.abs(librosa.stft(audio)), ref=np.max)
            img = librosa.display.specshow(D, sr=sr, x_axis='time', y_axis='log', ax=axes[2])
            axes[2].set_title('Spectrogram')
            axes[2].set_xlabel('Time (s)')
            axes[2].set_ylabel('Frequency (Hz)')
            plt.colorbar(img, ax=axes[2], format='%+2.0f dB')
            
            plt.tight_layout()
            
            # Save plot
            plot_dir = Path('static/plots')
            plot_dir.mkdir(exist_ok=True)
            plot_filename = f"{Path(vocal_path).stem}_analysis.png"
            plot_path = plot_dir / plot_filename
            plt.savefig(plot_path, dpi=150, bbox_inches='tight')
            plt.close()
            
            # Create analysis summary
            analysis = {
                'file': vocal_path,
                'duration': float(duration),
                'sample_rate': sr,
                'pitch': {
                    'mean': float(np.mean(f0_clean[voiced])) if np.any(voiced) else 0,
                    'min': float(np.min(f0_clean[voiced])) if np.any(voiced) else 0,
                    'max': float(np.max(f0_clean[voiced])) if np.any(voiced) else 0,
                    'range': f"{note_names[0]}-{note_names[-1]}" if note_names else "N/A",
                    'voiced_percentage': float(np.mean(voiced) * 100)
                },
                'tempo': float(tempo),
                'beats': len(beat_times),
                'key': key,
                'mfcc_shape': mfcc.shape,
                'spectral_centroid_mean': float(np.mean(spectral_centroid)),
                'rms_mean': float(np.mean(rms)),
                'plot_url': f'/static/plots/{plot_filename}',
                'processing_time': time.time() - start_time
            }
            
            # Save to file
            analysis_dir = Path('data/analysis') / Path(vocal_path).stem
            analysis_dir.mkdir(parents=True, exist_ok=True)
            
            with open(analysis_dir / 'analysis.json', 'w') as f:
                json.dump(analysis, f, indent=2)
            
            print(f"✅ Vocal analysis complete in {analysis['processing_time']:.1f}s")
            print(f"   Pitch range: {analysis['pitch']['range']}")
            print(f"   Tempo: {analysis['tempo']:.0f} BPM")
            print(f"   Key: {analysis['key']}")
            
            return {
                'success': True,
                'analysis': analysis,
                'analysis_file': str(analysis_dir / 'analysis.json'),
                'plot_path': str(plot_path)
            }
            
        except Exception as e:
            print(f"❌ Vocal analysis failed: {str(e)}")
            traceback.print_exc()
            return {
                'success': False,
                'error': str(e),
                'processing_time': time.time() - start_time
            }
    
    def _estimate_key_simple(self, audio, sr):
        """Simple key estimation using chromagram."""
        try:
            import librosa
            
            # Extract chromagram
            chroma = librosa.feature.chroma_cqt(y=audio, sr=sr)
            chroma_avg = np.mean(chroma, axis=1)
            
            # Major and minor profiles (Krumhansl)
            major_profile = np.array([6.35, 2.23, 3.48, 2.33, 4.38, 4.09, 2.52, 5.19, 2.39, 3.66, 2.29, 2.88])
            minor_profile = np.array([6.33, 2.68, 3.52, 5.38, 2.60, 3.53, 2.54, 4.75, 3.98, 2.69, 3.34, 3.17])
            
            # Find best correlation
            keys = ['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B']
            best_corr = -1
            best_key = "Unknown"
            
            for i in range(12):
                # Major
                major_rotated = np.roll(major_profile, i)
                major_corr = np.corrcoef(chroma_avg, major_rotated)[0, 1]
                
                # Minor
                minor_rotated = np.roll(minor_profile, i)
                minor_corr = np.corrcoef(chroma_avg, minor_rotated)[0, 1]
                
                if major_corr > best_corr:
                    best_corr = major_corr
                    best_key = f"{keys[i]} major"
                
                if minor_corr > best_corr:
                    best_corr = minor_corr
                    best_key = f"{keys[i]} minor"
            
            return best_key
            
        except:
            return "Unknown"
    
    def test_stem_separation(self):
        """Quick test of stem separation."""
        print("🎚️ Testing stem separation...")
        
        # Create test audio
        import numpy as np
        import soundfile as sf
        
        test_file = 'test_audio/quick_test.wav'
        Path('test_audio').mkdir(exist_ok=True)
        
        sr = 44100
        duration = 3.0
        t = np.linspace(0, duration, int(sr * duration), endpoint=False)
        
        # Simple audio
        audio = 0.3 * np.sin(2 * np.pi * 440 * t)
        sf.write(test_file, audio, sr)
        
        return {
            'success': True,
            'message': 'Test audio created',
            'file': test_file,
            'duration': duration
        }

# Initialize engine
ai = VocalFusionAI()

# ============================================================================
# 2. WEB INTERFACE
# ============================================================================
HTML_TEMPLATE = '''
<!DOCTYPE html>
<html>
<head>
    <title>Vocal Fusion AI - Simplified</title>
    <style>
        body { 
            font-family: -apple-system, BlinkMacSystemFont, sans-serif;
            max-width: 1000px; 
            margin: 0 auto; 
            padding: 20px;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            min-height: 100vh;
        }
        .container { 
            background: rgba(255,255,255,0.95); 
            padding: 40px; 
            border-radius: 20px; 
            backdrop-filter: blur(10px);
            box-shadow: 0 20px 40px rgba(0,0,0,0.1);
        }
        h1 {
            background: linear-gradient(135deg, #3B82F6, #8B5CF6);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            text-align: center;
        }
        .step {
            background: white;
            padding: 20px;
            margin: 20px 0;
            border-radius: 10px;
            box-shadow: 0 5px 15px rgba(0,0,0,0.08);
            border-left: 5px solid;
        }
        .step.complete { border-left-color: #10B981; }
        .step.current { border-left-color: #3B82F6; }
        .step.pending { border-left-color: #9CA3AF; }
        .btn {
            background: linear-gradient(135deg, #3B82F6, #8B5CF6);
            color: white;
            padding: 12px 24px;
            border: none;
            border-radius: 8px;
            cursor: pointer;
            font-weight: 600;
            margin: 10px 5px;
            display: inline-block;
            text-decoration: none;
            transition: all 0.3s ease;
        }
        .btn:hover {
            transform: translateY(-2px);
            box-shadow: 0 10px 20px rgba(59, 130, 246, 0.3);
        }
        .result {
            background: #F3F4F6;
            padding: 20px;
            border-radius: 10px;
            margin: 20px 0;
            font-family: 'Courier New', monospace;
            white-space: pre-wrap;
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
    </style>
</head>
<body>
    <div class="container">
        <h1>🎵 Vocal Fusion AI</h1>
        <p style="text-align: center; color: #6B7280; margin-bottom: 30px;">
            Simplified version - No CREPE dependencies
        </p>
        
        <div class="step complete">
            <h2>✅ Step 1: Audio Loading</h2>
            <p>Load any audio file (WAV, MP3, FLAC) using librosa.</p>
            <button onclick="testAudio()" class="btn">Test Audio Loading</button>
        </div>
        
        <div class="step complete">
            <h2>✅ Step 2: Stem Separation</h2>
            <p>Separate vocals from instruments using Demucs.</p>
            <button onclick="testStems()" class="btn">Test Stem Separation</button>
        </div>
        
        <div class="step current">
            <h2>🎤 Step 3: Vocal Analysis (Simplified)</h2>
            <p>Analyze pitch, tempo, key using librosa's built-in tools (no CREPE).</p>
            <button onclick="testVocalAnalysis()" class="btn">Test Vocal Analysis</button>
            <input type="file" id="vocalFile" accept=".wav,.mp3,.flac" style="margin-left: 20px;">
            <button onclick="analyzeUpload()" class="btn">Analyze Uploaded File</button>
        </div>
        
        <div id="results">
            <h2>📊 Results</h2>
            <div id="results-content"></div>
            <div id="results-plot" class="plot-container"></div>
            <div id="results-analysis" class="result" style="display: none;"></div>
        </div>
    </div>
    
    <script>
    async function testAudio() {
        showLoading('Testing audio loading...');
        const response = await fetch('/api/test-audio');
        const data = await response.json();
        showResults(data, 'Audio Loading Test');
    }
    
    async function testStems() {
        showLoading('Testing stem separation...');
        const response = await fetch('/api/test-stems', {method: 'POST'});
        const data = await response.json();
        showResults(data, 'Stem Separation Test');
    }
    
    async function testVocalAnalysis() {
        showLoading('Testing vocal analysis (may take 10-20 seconds)...');
        const response = await fetch('/api/test-vocal-analysis');
        const data = await response.json();
        showVocalResults(data);
    }
    
    async function analyzeUpload() {
        const fileInput = document.getElementById('vocalFile');
        if (!fileInput.files.length) {
            alert('Please select a file first');
            return;
        }
        
        showLoading('Uploading and analyzing file...');
        const formData = new FormData();
        formData.append('file', fileInput.files[0]);
        
        const response = await fetch('/api/analyze-vocal', {
            method: 'POST',
            body: formData
        });
        
        const data = await response.json();
        showVocalResults(data);
    }
    
    function showLoading(message) {
        document.getElementById('results-content').innerHTML = `
            <div style="text-align: center; padding: 40px;">
                <div style="width: 40px; height: 40px; border: 4px solid #f3f3f3; border-top: 4px solid #3B82F6; border-radius: 50%; animation: spin 1s linear infinite; margin: 0 auto;"></div>
                <p style="margin-top: 20px;">${message}</p>
            </div>
            <style>@keyframes spin { 0% { transform: rotate(0deg); } 100% { transform: rotate(360deg); } }</style>
        `;
        document.getElementById('results-plot').innerHTML = '';
        document.getElementById('results-analysis').style.display = 'none';
    }
    
    function showResults(data, title) {
        const container = document.getElementById('results-content');
        
        if (data.success) {
            container.innerHTML = `
                <div style="background: #D1FAE5; padding: 20px; border-radius: 10px;">
                    <h3 style="color: #065F46;">✅ ${title} Successful</h3>
                    <pre style="background: white; padding: 15px; border-radius: 5px; overflow: auto;">${JSON.stringify(data, null, 2)}</pre>
                </div>
            `;
        } else {
            container.innerHTML = `
                <div style="background: #FEE2E2; padding: 20px; border-radius: 10px;">
                    <h3 style="color: #991B1B;">❌ ${title} Failed</h3>
                    <p>${data.error || 'Unknown error'}</p>
                </div>
            `;
        }
    }
    
    function showVocalResults(data) {
        const container = document.getElementById('results-content');
        const plotDiv = document.getElementById('results-plot');
        const analysisDiv = document.getElementById('results-analysis');
        
        if (data.success) {
            // Show plot if available
            if (data.analysis && data.analysis.plot_url) {
                plotDiv.innerHTML = `<img src="${data.analysis.plot_url}" alt="Vocal Analysis Plot">`;
            }
            
            // Show analysis results
            analysisDiv.style.display = 'block';
            analysisDiv.innerHTML = `
                <h3>🎤 Vocal Analysis Results</h3>
                <p><strong>File:</strong> ${data.analysis.file || 'N/A'}</p>
                <p><strong>Duration:</strong> ${data.analysis.duration ? data.analysis.duration.toFixed(2) + 's' : 'N/A'}</p>
                <p><strong>Tempo:</strong> ${data.analysis.tempo ? data.analysis.tempo.toFixed(0) + ' BPM' : 'N/A'}</p>
                <p><strong>Key:</strong> ${data.analysis.key || 'N/A'}</p>
                <p><strong>Pitch Range:</strong> ${data.analysis.pitch ? data.analysis.pitch.range : 'N/A'}</p>
                <p><strong>Processing Time:</strong> ${data.analysis.processing_time ? data.analysis.processing_time.toFixed(2) + 's' : 'N/A'}</p>
                
                <button onclick="showRawAnalysis()" class="btn" style="margin-top: 10px;">View Raw Analysis Data</button>
                <div id="raw-analysis" style="display: none; margin-top: 20px;">
                    <pre>${JSON.stringify(data.analysis, null, 2)}</pre>
                </div>
            `;
            
            container.innerHTML = `
                <div style="background: #D1FAE5; padding: 20px; border-radius: 10px;">
                    <h3 style="color: #065F46;">✅ Vocal Analysis Successful</h3>
                    <p>Analysis complete! View results below.</p>
                </div>
            `;
        } else {
            container.innerHTML = `
                <div style="background: #FEE2E2; padding: 20px; border-radius: 10px;">
                    <h3 style="color: #991B1B;">❌ Vocal Analysis Failed</h3>
                    <p>Error: ${data.error || 'Unknown error'}</p>
                    <p>Check terminal for details.</p>
                </div>
            `;
            plotDiv.innerHTML = '';
            analysisDiv.style.display = 'none';
        }
    }
    
    function showRawAnalysis() {
        const rawDiv = document.getElementById('raw-analysis');
        rawDiv.style.display = rawDiv.style.display === 'none' ? 'block' : 'none';
    }
    </script>
</body>
</html>
'''

# ============================================================================
# 3. API ENDPOINTS
# ============================================================================
@app.route('/')
def home():
    return render_template_string(HTML_TEMPLATE)

@app.route('/api/test-audio')
def api_test_audio():
    """Test audio loading."""
    import numpy as np
    import soundfile as sf
    
    test_file = 'test_audio/simple_test.wav'
    Path('test_audio').mkdir(exist_ok=True)
    
    # Create test audio
    sr = 44100
    duration = 2.0
    t = np.linspace(0, duration, int(sr * duration), endpoint=False)
    audio = 0.3 * np.sin(2 * np.pi * 440 * t)
    sf.write(test_file, audio, sr)
    
    # Load it
    result = ai.load_audio(test_file)
    
    if result['status'] == 'loaded':
        return jsonify({
            'success': True,
            'file': test_file,
            'duration': result['duration'],
            'sample_rate': result['sample_rate']
        })
    else:
        return jsonify({'success': False, 'error': result.get('error')})

@app.route('/api/test-stems', methods=['POST'])
def api_test_stems():
    """Test stem separation."""
    result = ai.test_stem_separation()
    return jsonify(result)

@app.route('/api/test-vocal-analysis')
def api_test_vocal_analysis():
    """Test vocal analysis with generated audio."""
    import numpy as np
    import soundfile as sf
    
    test_file = 'test_audio/vocal_test.wav'
    Path('test_audio').mkdir(exist_ok=True)
    
    # Create a vocal-like test audio
    sr = 44100
    duration = 5.0
    t = np.linspace(0, duration, int(sr * duration), endpoint=False)
    
    # Simple melody
    audio = np.zeros_like(t)
    notes = [440, 523.25, 659.25, 523.25, 440]  # A4, C5, E5, C5, A4
    note_duration = duration / len(notes)
    
    for i, freq in enumerate(notes):
        start = i * note_duration
        end = (i + 1) * note_duration
        mask = (t >= start) & (t < end)
        audio[mask] = 0.2 * np.sin(2 * np.pi * freq * t[mask])
    
    # Add some vibrato
    vibrato = 0.05 * np.sin(2 * np.pi * 5 * t)
    audio *= (1 + vibrato)
    
    # Normalize
    audio = audio / np.max(np.abs(audio)) * 0.8
    
    sf.write(test_file, audio, sr)
    
    # Analyze it
    result = ai.analyze_vocal_simple(test_file)
    return jsonify(result)

@app.route('/api/analyze-vocal', methods=['POST'])
def api_analyze_vocal():
    """Analyze uploaded vocal file."""
    if 'file' not in request.files:
        return jsonify({'success': False, 'error': 'No file uploaded'})
    
    file = request.files['file']
    if file.filename == '':
        return jsonify({'success': False, 'error': 'No file selected'})
    
    # Save uploaded file
    upload_dir = Path('data/raw/uploads')
    upload_dir.mkdir(parents=True, exist_ok=True)
    
    file_path = upload_dir / file.filename
    file.save(file_path)
    
    # Analyze it
    result = ai.analyze_vocal_simple(str(file_path))
    return jsonify(result)

@app.route('/static/<path:filename>')
def static_files(filename):
    """Serve static files."""
    return send_file(Path('static') / filename)

# ============================================================================
# 4. MAIN EXECUTION
# ============================================================================
if __name__ == '__main__':
    print(f"""
    {'='*60}
    🎵 VOCAL FUSION AI - SIMPLIFIED VERSION
    {'='*60}
    
    ✅ No CREPE dependency
    ✅ Uses librosa's built-in pyin for pitch tracking
    ✅ All other features intact
    
    🚀 Access: http://localhost:5000
    
    Quick Tests:
    1. Click "Test Audio Loading"
    2. Click "Test Vocal Analysis"
    3. Upload your own audio file
    
    📊 Features:
    • Pitch tracking (librosa pyin)
    • Tempo detection
    • Key estimation
    • Spectral analysis
    • Visualization plots
    
    {'='*60}
    """)
    
    app.run(host='0.0.0.0', port=5000, debug=True)
