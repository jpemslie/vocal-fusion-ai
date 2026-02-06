#!/usr/bin/env python3
"""
VOCAL FUSION AI - MINIMAL WORKING VERSION
Compatible with your installed packages (PyTorch 2.10.0)
"""

import os
import json
import time
import traceback
from pathlib import Path
from flask import Flask, request, jsonify, render_template_string, send_file

app = Flask(__name__)

# ============================================================================
# 1. SIMPLE VOCAL FUSION AI ENGINE
# ============================================================================
class SimpleVocalFusionAI:
    """Simplified version that works with current packages."""
    
    def __init__(self):
        self.setup_directories()
        self.status = "ready"
        
    def setup_directories(self):
        """Create minimal directory structure."""
        dirs = ['data/raw', 'data/stems', 'data/outputs', 'test_audio']
        for d in dirs:
            Path(d).mkdir(parents=True, exist_ok=True)
    
    def load_audio(self, file_path):
        """Load audio file."""
        try:
            import librosa
            import soundfile as sf
            import numpy as np
            
            print(f"🔊 Loading audio: {file_path}")
            
            if not Path(file_path).exists():
                return {'error': f'File not found: {file_path}', 'status': 'error'}
            
            # Load audio
            audio, sample_rate = librosa.load(file_path, sr=None, mono=False)
            
            if isinstance(audio, list):
                audio = np.array(audio)
            
            duration = librosa.get_duration(y=audio, sr=sample_rate)
            
            # Handle mono/stereo
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
    
    def test_stem_separation_simple(self, audio_data):
        """Test stem separation without complex dependencies."""
        print("🎚️ Testing stem separation capability...")
        
        try:
            # Check if demucs is available
            import demucs
            print(f"✅ Demucs version: {demucs.__version__}")
            
            # Create a simple test
            from pathlib import Path
            import tempfile
            import shutil
            
            # Create a temporary copy for testing
            input_path = audio_data['file_path']
            test_dir = Path('data/stems') / f"test_{int(time.time())}"
            test_dir.mkdir(parents=True, exist_ok=True)
            
            print(f"   Test directory: {test_dir}")
            
            # Try to import and run demucs
            try:
                # Method 1: Try command line
                import subprocess
                import sys
                
                cmd = [
                    sys.executable, "-m", "demucs",
                    "--out", str(test_dir),
                    "--name", "htdemucs",
                    "--two-stems", "vocals",  # Just separate vocals for testing
                    input_path
                ]
                
                print(f"   Running: {' '.join(cmd)}")
                
                # Run with timeout
                result = subprocess.run(
                    cmd,
                    capture_output=True,
                    text=True,
                    timeout=120  # 2 minute timeout
                )
                
                if result.returncode == 0:
                    print("✅ Demucs command succeeded!")
                    
                    # Find output files
                    output_files = list(test_dir.rglob("*.wav"))
                    stems = {}
                    
                    for f in output_files:
                        stem_name = f.stem
                        stems[stem_name] = str(f)
                        print(f"   Found stem: {stem_name} at {f}")
                    
                    return {
                        'success': True,
                        'stems': stems,
                        'output_dir': str(test_dir),
                        'method': 'demucs_cli'
                    }
                else:
                    print(f"⚠️ Demucs command failed: {result.stderr}")
                    
            except Exception as e:
                print(f"⚠️ Command line method failed: {str(e)}")
            
            # Try Python API as fallback
            try:
                print("   Trying Python API...")
                import torch
                import torchaudio
                from demucs.pretrained import get_model
                from demucs.apply import apply_model
                
                # Load model
                device = torch.device('cpu')
                model = get_model('htdemucs')
                model.to(device)
                model.eval()
                
                # Load audio
                wav, sr = torchaudio.load(input_path)
                
                # Convert to model format
                from demucs.audio import convert_audio
                wav = convert_audio(wav, sr, model.samplerate, model.audio_channels)
                wav = wav.to(device)
                
                # Separate
                with torch.no_grad():
                    sources = apply_model(model, wav[None], device=device, progress=True)[0]
                
                # Save stems
                stem_names = ['drums', 'bass', 'other', 'vocals']
                stems = {}
                
                for idx, name in enumerate(stem_names):
                    if idx < len(sources):
                        stem = sources[idx].cpu()
                        output_path = test_dir / f"{name}.wav"
                        torchaudio.save(str(output_path), stem, model.samplerate)
                        stems[name] = str(output_path)
                        print(f"   ✓ Saved {name}")
                
                return {
                    'success': True,
                    'stems': stems,
                    'output_dir': str(test_dir),
                    'method': 'demucs_python'
                }
                
            except Exception as e:
                print(f"⚠️ Python API failed: {str(e)}")
                
            return {
                'success': False,
                'error': 'All stem separation methods failed',
                'output_dir': str(test_dir)
            }
            
        except ImportError as e:
            print(f"❌ Demucs not properly installed: {str(e)}")
            return {
                'success': False,
                'error': f'Demucs import error: {str(e)}'
            }
        except Exception as e:
            print(f"❌ Unexpected error: {str(e)}")
            traceback.print_exc()
            return {
                'success': False,
                'error': str(e)
            }
    
    def test_system(self):
        """Test all system components."""
        tests = {
            'flask': False,
            'numpy': False,
            'librosa': False,
            'soundfile': False,
            'torch': False,
            'demucs': False
        }
        
        try:
            import flask
            tests['flask'] = True
        except: pass
        
        try:
            import numpy
            tests['numpy'] = True
        except: pass
        
        try:
            import librosa
            tests['librosa'] = True
        except: pass
        
        try:
            import soundfile
            tests['soundfile'] = True
        except: pass
        
        try:
            import torch
            tests['torch'] = True
        except: pass
        
        try:
            import demucs
            tests['demucs'] = True
        except: pass
        
        return tests

# Initialize engine
ai = SimpleVocalFusionAI()

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
        .test-btn { background: #3B82F6; color: white; padding: 10px 20px; border: none; border-radius: 5px; cursor: pointer; margin: 5px; }
        .test-btn:hover { background: #2563EB; }
        .success { color: #10B981; }
        .error { color: #EF4444; }
        .warning { color: #F59E0B; }
        .status-dot { display: inline-block; width: 10px; height: 10px; border-radius: 50%; margin-right: 10px; }
        .status-ok { background: #10B981; }
        .status-fail { background: #EF4444; }
        .status-warn { background: #F59E0B; }
    </style>
</head>
<body>
    <div class="container">
        <h1>🎵 Vocal Fusion AI - Compatibility Test</h1>
        <p>Testing your current package installations...</p>
        
        <div class="card">
            <h2>📊 System Status</h2>
            <div id="system-status">
                <p>Loading system status...</p>
            </div>
            <button class="test-btn" onclick="testSystem()">Refresh Status</button>
        </div>
        
        <div class="card">
            <h2>🧪 Quick Tests</h2>
            <button class="test-btn" onclick="testAudioLoading()">Test Audio Loading</button>
            <button class="test-btn" onclick="testStemSeparation()">Test Stem Separation</button>
            <button class="test-btn" onclick="createTestAudio()">Create Test Audio</button>
        </div>
        
        <div class="card">
            <h2>📝 Test Results</h2>
            <div id="test-results">
                <p>No tests run yet.</p>
            </div>
        </div>
        
        <div class="card">
            <h2>🚀 Next Steps</h2>
            <ol>
                <li>Test audio loading (librosa + soundfile)</li>
                <li>Test stem separation (Demucs + PyTorch)</li>
                <li>Create a simple fusion pipeline</li>
            </ol>
        </div>
    </div>
    
    <script>
    // Load system status on page load
    window.addEventListener('load', () => {
        testSystem();
    });
    
    async function testSystem() {
        const statusDiv = document.getElementById('system-status');
        statusDiv.innerHTML = '<p>Testing packages...</p>';
        
        const response = await fetch('/api/system-test');
        const data = await response.json();
        
        let html = '<h3>Package Status:</h3><ul>';
        for (const [pkg, installed] of Object.entries(data.tests)) {
            const statusClass = installed ? 'status-ok' : 'status-fail';
            const statusText = installed ? '✅ Installed' : '❌ Missing';
            html += `<li><span class="status-dot ${statusClass}"></span> ${pkg}: ${statusText}</li>`;
        }
        html += '</ul>';
        
        if (data.versions) {
            html += '<h3>Versions:</h3><ul>';
            for (const [pkg, version] of Object.entries(data.versions)) {
                html += `<li>${pkg}: ${version}</li>`;
            }
            html += '</ul>';
        }
        
        statusDiv.innerHTML = html;
    }
    
    async function testAudioLoading() {
        const resultsDiv = document.getElementById('test-results');
        resultsDiv.innerHTML = '<p>Testing audio loading...</p>';
        
        const response = await fetch('/api/test-audio-loading');
        const data = await response.json();
        
        if (data.success) {
            resultsDiv.innerHTML = `
                <h3 class="success">✅ Audio Loading Test Successful!</h3>
                <p>File: ${data.file_path}</p>
                <p>Sample Rate: ${data.sample_rate} Hz</p>
                <p>Duration: ${data.duration.toFixed(2)} seconds</p>
                <p>Channels: ${data.channels}</p>
                <audio controls src="/api/get-test-audio" style="width: 100%; margin: 10px 0;"></audio>
            `;
        } else {
            resultsDiv.innerHTML = `
                <h3 class="error">❌ Audio Loading Failed</h3>
                <p>Error: ${data.error}</p>
            `;
        }
    }
    
    async function testStemSeparation() {
        const resultsDiv = document.getElementById('test-results');
        resultsDiv.innerHTML = '<p>Testing stem separation (may take 1-2 minutes)...</p>';
        
        const response = await fetch('/api/test-stem-separation', {
            method: 'POST',
            headers: {'Content-Type': 'application/json'},
            body: JSON.stringify({test: true})
        });
        
        const data = await response.json();
        
        if (data.success) {
            let stemsHtml = '';
            if (data.stems) {
                stemsHtml = '<h4>Stems Created:</h4><ul>';
                for (const [name, path] of Object.entries(data.stems)) {
                    stemsHtml += `<li>${name}: ${Path(path).name}</li>`;
                }
                stemsHtml += '</ul>';
            }
            
            resultsDiv.innerHTML = `
                <h3 class="success">✅ Stem Separation Successful!</h3>
                <p>Method: ${data.method}</p>
                <p>Output Directory: ${data.output_dir}</p>
                ${stemsHtml}
                <p><a href="/list-test-stems">View Stem Files</a></p>
            `;
        } else {
            resultsDiv.innerHTML = `
                <h3 class="error">❌ Stem Separation Failed</h3>
                <p>Error: ${data.error}</p>
                <p>This is normal if Demucs needs to download models first.</p>
                <p>Try running in terminal: python -m demucs --help</p>
            `;
        }
    }
    
    async function createTestAudio() {
        const resultsDiv = document.getElementById('test-results');
        resultsDiv.innerHTML = '<p>Creating test audio...</p>';
        
        const response = await fetch('/api/create-test-audio');
        const data = await response.json();
        
        if (data.success) {
            resultsDiv.innerHTML = `
                <h3 class="success">✅ Test Audio Created!</h3>
                <p>File: ${data.file_path}</p>
                <p>Duration: ${data.duration} seconds</p>
                <p>Sample Rate: ${data.sample_rate} Hz</p>
                <audio controls src="${data.play_url}" style="width: 100%; margin: 10px 0;"></audio>
            `;
        } else {
            resultsDiv.innerHTML = `
                <h3 class="error">❌ Failed to create test audio</h3>
                <p>Error: ${data.error}</p>
            `;
        }
    }
    
    // Helper to extract filename from path
    function Path(path) {
        return {
            name: path.split('/').pop().split('\\\\').pop()
        };
    }
    </script>
</body>
</html>
'''

@app.route('/')
def home():
    return render_template_string(HTML_TEMPLATE)

@app.route('/api/system-test')
def api_system_test():
    """Test all system packages."""
    tests = ai.test_system()
    
    # Get versions for installed packages
    versions = {}
    try:
        import flask
        versions['flask'] = flask.__version__
    except: pass
    
    try:
        import numpy
        versions['numpy'] = numpy.__version__
    except: pass
    
    try:
        import librosa
        versions['librosa'] = librosa.__version__
    except: pass
    
    try:
        import soundfile
        versions['soundfile'] = soundfile.__version__
    except: pass
    
    try:
        import torch
        versions['torch'] = torch.__version__
    except: pass
    
    try:
        import demucs
        versions['demucs'] = demucs.__version__
    except: pass
    
    return jsonify({
        'tests': tests,
        'versions': versions,
        'python': f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}"
    })

@app.route('/api/test-audio-loading')
def api_test_audio_loading():
    """Test audio loading with a generated test file."""
    # Create a test audio file
    import numpy as np
    import soundfile as sf
    
    test_file = 'test_audio/simple_test.wav'
    Path('test_audio').mkdir(exist_ok=True)
    
    # Create simple audio
    sample_rate = 44100
    duration = 2.0
    t = np.linspace(0, duration, int(sample_rate * duration), endpoint=False)
    audio = 0.3 * np.sin(2 * np.pi * 440 * t)
    sf.write(test_file, audio, sample_rate)
    
    # Load it
    result = ai.load_audio(test_file)
    
    if result['status'] == 'loaded':
        return jsonify({
            'success': True,
            'file_path': test_file,
            'sample_rate': result['sample_rate'],
            'duration': result['duration'],
            'channels': result['channels']
        })
    else:
        return jsonify({
            'success': False,
            'error': result.get('error', 'Unknown error')
        })

@app.route('/api/get-test-audio')
def api_get_test_audio():
    """Serve the test audio file."""
    test_file = 'test_audio/simple_test.wav'
    if Path(test_file).exists():
        return send_file(test_file, mimetype='audio/wav')
    return jsonify({'error': 'Test audio not found'}), 404

@app.route('/api/test-stem-separation', methods=['POST'])
def api_test_stem_separation():
    """Test stem separation."""
    # First create a test file
    import numpy as np
    import soundfile as sf
    
    test_file = 'test_audio/stem_test.wav'
    Path('test_audio').mkdir(exist_ok=True)
    
    # Create audio with different frequency components
    sample_rate = 44100
    duration = 5.0  # 5 seconds for testing
    t = np.linspace(0, duration, int(sample_rate * duration), endpoint=False)
    
    # Create components
    bass = 0.3 * np.sin(2 * np.pi * 100 * t)
    melody = 0.2 * np.sin(2 * np.pi * 440 * t)
    drums = 0.15 * np.sin(2 * np.pi * 200 * t) * (np.sin(2 * np.pi * 2 * t) > 0)
    vocals = 0.25 * np.sin(2 * np.pi * 330 * t) * (0.5 + 0.5 * np.sin(2 * np.pi * 0.5 * t))
    
    # Combine
    audio = bass + melody + drums + vocals
    audio = audio / np.max(np.abs(audio))
    
    # Make stereo
    audio_stereo = np.vstack([audio, audio * 0.8]).T
    sf.write(test_file, audio_stereo, sample_rate)
    
    # Load audio
    audio_data = ai.load_audio(test_file)
    
    if audio_data['status'] != 'loaded':
        return jsonify({
            'success': False,
            'error': f'Failed to load audio: {audio_data.get("error")}'
        })
    
    # Try stem separation
    result = ai.test_stem_separation_simple(audio_data)
    
    return jsonify(result)

@app.route('/api/create-test-audio')
def api_create_test_audio():
    """Create a test audio file."""
    import numpy as np
    import soundfile as sf
    
    test_file = 'test_audio/generated.wav'
    Path('test_audio').mkdir(exist_ok=True)
    
    # Create a musical pattern
    sample_rate = 44100
    duration = 4.0
    t = np.linspace(0, duration, int(sample_rate * duration), endpoint=False)
    
    # Create a simple melody
    melody_freqs = [440, 523.25, 659.25, 523.25, 440, 392, 440]  # A4, C5, E5, C5, A4, G4, A4
    note_duration = duration / len(melody_freqs)
    
    audio = np.zeros_like(t)
    for i, freq in enumerate(melody_freqs):
        start = i * note_duration
        end = (i + 1) * note_duration
        mask = (t >= start) & (t < end)
        audio[mask] = 0.2 * np.sin(2 * np.pi * freq * t[mask])
    
    # Add some rhythm
    kick = 0.1 * np.sin(2 * np.pi * 60 * t) * (np.sin(2 * np.pi * 2 * t) > 0.5)
    audio += kick
    
    # Normalize
    audio = audio / np.max(np.abs(audio))
    
    # Make stereo
    audio_stereo = np.vstack([audio, audio * 0.8]).T
    sf.write(test_file, audio_stereo, sample_rate)
    
    return jsonify({
        'success': True,
        'file_path': test_file,
        'duration': duration,
        'sample_rate': sample_rate,
        'play_url': '/api/get-audio?file=' + test_file
    })

@app.route('/api/get-audio')
def api_get_audio():
    """Serve an audio file."""
    file_path = request.args.get('file')
    if file_path and Path(file_path).exists():
        return send_file(file_path, mimetype='audio/wav')
    return jsonify({'error': 'File not found'}), 404

@app.route('/list-test-stems')
def list_test_stems():
    """List test stem files."""
    import os
    from pathlib import Path
    
    stem_dir = Path('data/stems')
    
    if not stem_dir.exists():
        return '<h1>No stems found</h1><p>Run stem separation test first.</p>'
    
    html = '<div style="max-width: 800px; margin: 0 auto; padding: 20px;"><h1>📁 Test Stems</h1>'
    
    for folder in stem_dir.iterdir():
        if folder.is_dir() and 'test_' in folder.name:
            html += f'<h2>📂 {folder.name}</h2><ul>'
            for file in folder.rglob('*.wav'):
                html += f'<li><a href="/api/get-audio?file={file}">▶️ {file.name}</a></li>'
            html += '</ul>'
    
    html += '<p><a href="/">← Back</a></p></div>'
    return html

# ============================================================================
# 3. MAIN EXECUTION
# ============================================================================
import sys

if __name__ == '__main__':
    print(f"""
    {'='*60}
    🎵 VOCAL FUSION AI - COMPATIBILITY TEST
    {'='*60}
    
    Your environment:
    • PyTorch 2.10.0 installed ✓
    • Demucs 4.0.0 installed ✓
    • Flask 3.0.0 installed ✓
    
    Access: http://localhost:5000
    
    Quick Tests:
    1. Click "Refresh Status" to check packages
    2. Click "Test Audio Loading" 
    3. Click "Test Stem Separation" (may take 1-2 min)
    
    If stem separation fails, it might be downloading models.
    Check terminal for progress.
    
    {'='*60}
    """)
    
    app.run(host='0.0.0.0', port=5000, debug=True)
