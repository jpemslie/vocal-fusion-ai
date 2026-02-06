#!/usr/bin/env python3
"""
VOCAL FUSION AI - WITH REAL AUDIO LOADING
"""

import os
import json
import traceback
from pathlib import Path
from flask import Flask, request, jsonify, render_template_string, send_file

app = Flask(__name__)

# ============================================================================
# 1. VOCAL FUSION AI ENGINE
# ============================================================================
class VocalFusionAI:
    """Complete vocal fusion system with real audio loading."""
    
    def __init__(self):
        self.setup_directories()
        self.status = "ready"
        
    def setup_directories(self):
        """Create directory structure."""
        dirs = [
            'data/raw',
            'data/stems',
            'data/analysis',
            'data/outputs',
            'test_audio'
        ]
        for d in dirs:
            Path(d).mkdir(parents=True, exist_ok=True)
    
    # REAL AUDIO LOADING FUNCTION
    def load_audio(self, file_path):
        """Load audio file using librosa with proper error handling."""
        try:
            import librosa
            import soundfile as sf
            import numpy as np
            
            print(f"🔊 Loading audio: {file_path}")
            
            # Check if file exists
            if not Path(file_path).exists():
                raise FileNotFoundError(f"Audio file not found: {file_path}")
            
            # Get file info
            info = sf.info(file_path)
            
            # Load audio
            audio, sample_rate = librosa.load(
                file_path,
                sr=None,
                mono=False,
                duration=None
            )
            
            # Convert to numpy array
            if isinstance(audio, list):
                audio = np.array(audio)
            
            # Get duration
            duration = librosa.get_duration(y=audio, sr=sample_rate)
            
            # Get channels
            if audio.ndim == 1:
                channels = 1
                audio = audio.reshape(1, -1)
            else:
                channels = audio.shape[0]
            
            # Calculate statistics
            rms = np.sqrt(np.mean(audio**2))
            peak = np.max(np.abs(audio))
            
            print(f"   ✓ Sample rate: {sample_rate} Hz")
            print(f"   ✓ Duration: {duration:.2f} seconds")
            print(f"   ✓ Channels: {channels}")
            print(f"   ✓ Shape: {audio.shape}")
            
            return {
                'file_path': file_path,
                'audio': audio,
                'sample_rate': sample_rate,
                'duration': duration,
                'channels': channels,
                'shape': audio.shape,
                'rms': float(rms),
                'peak': float(peak),
                'original_info': {
                    'samplerate': info.samplerate,
                    'frames': info.frames,
                    'sections': info.sections,
                    'format': str(info.format)
                },
                'status': 'loaded'
            }
            
        except Exception as e:
            print(f"❌ Error: {str(e)}")
            traceback.print_exc()
            return {
                'file_path': file_path,
                'error': str(e),
                'status': 'error'
            }
    
    # Other functions remain as placeholders for now
    def separate_stems(self, audio_data):
        """TODO: Replace with Demucs."""
        print("🎚️ [PLACEHOLDER] Stem separation placeholder")
        return {'status': 'placeholder'}
    
    def analyze_vocal(self, vocal_path):
        """TODO: Replace with real analysis."""
        print("🎤 [PLACEHOLDER] Vocal analysis placeholder")
        return {'status': 'placeholder'}
    
    def fuse_songs(self, song1_path, song2_path):
        """Main fusion pipeline."""
        print(f"\n🎭 Fusing: {song1_path} + {song2_path}")
        
        # Load both songs
        song1 = self.load_audio(song1_path)
        song2 = self.load_audio(song2_path)
        
        if song1['status'] != 'loaded' or song2['status'] != 'loaded':
            return {
                'success': False,
                'error': 'Failed to load audio files',
                'song1_status': song1.get('status'),
                'song2_status': song2.get('status')
            }
        
        return {
            'success': True,
            'song1': {
                'duration': song1['duration'],
                'sample_rate': song1['sample_rate'],
                'channels': song1['channels']
            },
            'song2': {
                'duration': song2['duration'],
                'sample_rate': song2['sample_rate'],
                'channels': song2['channels']
            },
            'message': 'Audio loaded successfully! Next: stem separation'
        }

# Initialize engine
ai_engine = VocalFusionAI()

# ============================================================================
# 2. WEB INTERFACE
# ============================================================================
HTML_TEMPLATE = '''
<!DOCTYPE html>
<html>
<head>
    <title>Vocal Fusion AI</title>
    <style>
        body { font-family: Arial, sans-serif; max-width: 1000px; margin: 0 auto; padding: 20px; }
        .container { background: #f5f5f5; padding: 30px; border-radius: 15px; }
        .card { background: white; padding: 20px; margin: 20px 0; border-radius: 10px; box-shadow: 0 2px 5px rgba(0,0,0,0.1); }
        .success { color: #10B981; }
        .error { color: #EF4444; }
        .warning { color: #F59E0B; }
        .step { display: flex; align-items: center; margin: 10px 0; }
        .step-number { background: #3B82F6; color: white; width: 30px; height: 30px; border-radius: 50%; display: flex; align-items: center; justify-content: center; margin-right: 15px; }
        .test-link { display: inline-block; background: #3B82F6; color: white; padding: 10px 20px; border-radius: 5px; text-decoration: none; margin: 10px 0; }
        .test-link:hover { background: #2563EB; }
    </style>
</head>
<body>
    <div class="container">
        <h1>🎵 Vocal Fusion AI</h1>
        <p>Step 1: Audio Loading ✓ Implemented</p>
        
        <div class="card">
            <h2>📊 System Status</h2>
            <div class="step">
                <div class="step-number">1</div>
                <div><strong>Audio Loading (librosa):</strong> <span class="success">✅ IMPLEMENTED</span></div>
            </div>
            <div class="step">
                <div class="step-number">2</div>
                <div><strong>Stem Separation (Demucs):</strong> <span class="warning">⚠️ PENDING</span></div>
            </div>
            <div class="step">
                <div class="step-number">3</div>
                <div><strong>Vocal Analysis:</strong> <span class="warning">⚠️ PENDING</span></div>
            </div>
            <div class="step">
                <div class="step-number">4</div>
                <div><strong>Fusion Engine:</strong> <span class="warning">⚠️ PENDING</span></div>
            </div>
        </div>
        
        <div class="card">
            <h2>🧪 Test Audio Loading</h2>
            <p>Test the implemented audio loading functionality:</p>
            <a class="test-link" href="/test/audio">Test Audio Loading</a>
            <a class="test-link" href="/create/test/audio">Create Test Tone</a>
            <a class="test-link" href="/test/upload">Test File Upload</a>
        </div>
        
        <div class="card">
            <h2>🎵 Upload Songs</h2>
            <form action="/upload" method="post" enctype="multipart/form-data">
                <p><strong>Song 1:</strong> <input type="file" name="song1" accept=".mp3,.wav,.flac"></p>
                <p><strong>Song 2:</strong> <input type="file" name="song2" accept=".mp3,.wav,.flac"></p>
                <button type="submit" style="background: #10B981; color: white; padding: 10px 20px; border: none; border-radius: 5px; cursor: pointer;">
                    🎭 Start Fusion
                </button>
            </form>
        </div>
        
        <div class="card">
            <h2>📝 Next Steps</h2>
            <ol>
                <li><strong>Done:</strong> Install librosa and soundfile</li>
                <li><strong>Done:</strong> Implement load_audio() function</li>
                <li><strong>Next:</strong> Install Demucs for stem separation</li>
                <li><strong>Then:</strong> Implement separate_stems() function</li>
            </ol>
        </div>
    </div>
</body>
</html>
'''

@app.route('/')
def home():
    return render_template_string(HTML_TEMPLATE)

@app.route('/test/audio')
def test_audio():
    """Test audio loading functionality."""
    test_file = 'test_audio/test_tone.wav'
    
    if not Path(test_file).exists():
        return '''
        <h1>Test Audio Not Found</h1>
        <p>Create a test tone first:</p>
        <a href="/create/test/audio">Create Test Tone</a>
        '''
    
    # Load the audio
    result = ai_engine.load_audio(test_file)
    
    # Format as HTML
    html = f'''
    <div style="max-width: 800px; margin: 0 auto; padding: 20px;">
        <h1>🎵 Audio Loading Test</h1>
        <p><a href="/">← Back</a></p>
        
        <div style="background: white; padding: 20px; border-radius: 10px; margin: 20px 0;">
            <h2>File: {result['file_path']}</h2>
            <p><strong>Status:</strong> <span style="color: {'green' if result['status'] == 'loaded' else 'red'}">{result['status']}</span></p>
            
            {f'''
            <h3>Audio Properties:</h3>
            <ul>
                <li><strong>Sample Rate:</strong> {result['sample_rate']} Hz</li>
                <li><strong>Duration:</strong> {result['duration']:.2f} seconds</li>
                <li><strong>Channels:</strong> {result['channels']}</li>
                <li><strong>Shape:</strong> {result['shape']}</li>
                <li><strong>RMS Level:</strong> {result['rms']:.4f}</li>
                <li><strong>Peak Level:</strong> {result['peak']:.4f}</li>
            </ul>
            
            <h3>Original File Info:</h3>
            <pre style="background: #f5f5f5; padding: 10px; border-radius: 5px;">
            Sample Rate: {result['original_info']['samplerate']} Hz
            Frames: {result['original_info']['frames']}
            Format: {result['original_info']['format']}
            </pre>
            ''' if result['status'] == 'loaded' else f'''
            <h3>Error:</h3>
            <p style="color: red;">{result.get('error', 'Unknown error')}</p>
            '''}
        </div>
        
        <p><a href="/play/test/audio">▶️ Play Test Audio</a></p>
    </div>
    '''
    
    return html

@app.route('/create/test/audio')
def create_test_audio():
    """Create a test audio file."""
    import numpy as np
    import soundfile as sf
    
    # Create test tone
    sample_rate = 44100
    duration = 3.0
    t = np.linspace(0, duration, int(sample_rate * duration), endpoint=False)
    
    # Create a melody
    freqs = [440, 523.25, 659.25, 783.99]  # A4, C5, E5, G5
    audio = np.zeros_like(t)
    
    for i, freq in enumerate(freqs):
        start = i * duration / len(freqs)
        end = (i + 1) * duration / len(freqs)
        mask = (t >= start) & (t < end)
        audio[mask] = 0.2 * np.sin(2 * np.pi * freq * t[mask])
    
    # Make stereo
    audio_stereo = np.vstack([audio, audio * 0.8]).T
    
    # Save
    sf.write('test_audio/test_tone.wav', audio_stereo, sample_rate)
    
    return '''
    <h1>✅ Test Audio Created!</h1>
    <p>Created: test_audio/test_tone.wav</p>
    <p><a href="/test/audio">Test Audio Loading</a> | <a href="/">← Back</a></p>
    '''

@app.route('/play/test/audio')
def play_test_audio():
    """Play the test audio file."""
    return '''
    <h1>🎵 Play Test Audio</h1>
    <audio controls autoplay>
        <source src="/static/test_tone.wav" type="audio/wav">
        Your browser does not support the audio element.
    </audio>
    <p><a href="/test/audio">← Back to Test</a></p>
    '''

@app.route('/test/upload')
def test_upload():
    """Test file upload page."""
    return '''
    <h1>Test File Upload</h1>
    <form action="/upload/test" method="post" enctype="multipart/form-data">
        <input type="file" name="file" accept=".mp3,.wav,.flac">
        <button type="submit">Upload Test</button>
    </form>
    <p><a href="/">← Back</a></p>
    '''

@app.route('/upload', methods=['POST'])
def upload_songs():
    """Handle song uploads."""
    try:
        song1 = request.files.get('song1')
        song2 = request.files.get('song2')
        
        if not song1 or not song2:
            return jsonify({'error': 'Please upload both songs'}), 400
        
        # Save files
        song1_path = f"data/raw/song1_{song1.filename}"
        song2_path = f"data/raw/song2_{song2.filename}"
        
        song1.save(song1_path)
        song2.save(song2_path)
        
        # Start fusion process
        result = ai_engine.fuse_songs(song1_path, song2_path)
        
        return jsonify(result)
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/health')
def health():
    return jsonify({
        'status': 'healthy',
        'service': 'vocal-fusion-ai',
        'audio_loading': 'implemented',
        'next_step': 'install_demucs'
    })

# ============================================================================
# 3. MAIN EXECUTION
# ============================================================================
if __name__ == '__main__':
    print(f"""
    {'='*60}
    🎵 VOCAL FUSION AI - STEP 1 COMPLETE
    {'='*60}
    
    ✅ Audio loading implemented with librosa
    ✅ Web interface running
    
    Next Steps:
    1. Test audio loading: http://localhost:5000/test/audio
    2. Create test tone: http://localhost:5000/create/test/audio
    3. Install Demucs for stem separation
    
    To install Demucs (next step):
    pip install demucs==4.0.0
    pip install torch==2.1.0
    
    {'='*60}
    """)
    
    app.run(host='0.0.0.0', port=5000, debug=True)
