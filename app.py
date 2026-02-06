#!/usr/bin/env python3
"""
VOCAL FUSION AI - MINIMAL WORKING VERSION
Skip complex dependencies until basic audio loading works.
"""

from flask import Flask, render_template_string, jsonify
import os

app = Flask(__name__)

# Simple HTML interface
HTML = '''
<!DOCTYPE html>
<html>
<head>
    <title>Vocal Fusion AI</title>
    <style>
        body { font-family: Arial, sans-serif; max-width: 800px; margin: 0 auto; padding: 20px; }
        .container { background: #f5f5f5; padding: 30px; border-radius: 15px; }
        .card { background: white; padding: 20px; margin: 20px 0; border-radius: 10px; }
        .status { padding: 10px; border-radius: 5px; }
        .success { background: #d4edda; }
        .warning { background: #fff3cd; }
        .error { background: #f8d7da; }
    </style>
</head>
<body>
    <div class="container">
        <h1>🎵 Vocal Fusion AI</h1>
        
        <div class="card">
            <h2>System Status</h2>
            <div class="status success">
                ✅ Flask server running on http://localhost:5000
            </div>
            <div class="status warning">
                ⚠️ Audio processing: Testing dependencies...
            </div>
        </div>
        
        <div class="card">
            <h2>Test Audio Loading</h2>
            <p><a href="/test/audio">Click here to test audio loading</a></p>
            <p>This will test if librosa and soundfile are working.</p>
        </div>
        
        <div class="card">
            <h2>Dependency Status</h2>
            <ul>
                <li>Flask: <span id="flask">Testing...</span></li>
                <li>NumPy: <span id="numpy">Testing...</span></li>
                <li>librosa: <span id="librosa">Testing...</span></li>
                <li>soundfile: <span id="soundfile">Testing...</span></li>
            </ul>
        </div>
    </div>
    
    <script>
    // Test dependencies via AJAX
    fetch('/test/dependencies')
        .then(r => r.json())
        .then(data => {
            for (const [lib, status] of Object.entries(data)) {
                const elem = document.getElementById(lib);
                if (elem) {
                    elem.textContent = status ? '✅ Installed' : '❌ Missing';
                    elem.style.color = status ? 'green' : 'red';
                }
            }
        });
    </script>
</body>
</html>
'''

@app.route('/')
def home():
    return render_template_string(HTML)

@app.route('/test/dependencies')
def test_dependencies():
    """Test if required dependencies are installed."""
    dependencies = {
        'flask': False,
        'numpy': False,
        'librosa': False,
        'soundfile': False
    }
    
    try:
        import flask
        dependencies['flask'] = True
    except ImportError:
        pass
    
    try:
        import numpy
        dependencies['numpy'] = True
    except ImportError:
        pass
    
    try:
        import librosa
        dependencies['librosa'] = True
    except ImportError:
        pass
    
    try:
        import soundfile
        dependencies['soundfile'] = True
    except ImportError:
        pass
    
    return jsonify(dependencies)

@app.route('/test/audio')
def test_audio():
    """Simple audio test that doesn't crash if dependencies missing."""
    try:
        import librosa
        import soundfile as sf
        import numpy as np
        
        # Create a simple test tone
        sample_rate = 44100
        duration = 1.0
        t = np.linspace(0, duration, int(sample_rate * duration), endpoint=False)
        audio = 0.3 * np.sin(2 * np.pi * 440 * t)
        
        # Save and load
        temp_file = 'temp_test.wav'
        sf.write(temp_file, audio, sample_rate)
        audio_loaded, sr = librosa.load(temp_file, sr=None)
        
        # Clean up
        import os
        os.remove(temp_file)
        
        return f'''
        <h1>✅ Audio Test Successful!</h1>
        <p>librosa and soundfile are working correctly.</p>
        <p>Loaded audio: shape={audio_loaded.shape}, sample_rate={sr}</p>
        <p>Duration: {len(audio_loaded)/sr:.2f} seconds</p>
        <p><a href="/">← Back</a></p>
        '''
        
    except ImportError as e:
        return f'''
        <h1>❌ Missing Dependencies</h1>
        <p>Error: {str(e)}</p>
        <p>Run these commands in terminal:</p>
        <pre>
        pip install librosa==0.10.1
        pip install soundfile==0.12.1
        </pre>
        <p><a href="/">← Back</a></p>
        '''
    except Exception as e:
        return f'''
        <h1>❌ Test Failed</h1>
        <p>Error: {str(e)}</p>
        <p><a href="/">← Back</a></p>
        '''

if __name__ == '__main__':
    print("""
    ============================================
    🎵 VOCAL FUSION AI - MINIMAL VERSION
    ============================================
    
    Access at: http://localhost:5000
    
    If audio dependencies are missing:
    1. Stop server (CTRL+C)
    2. Run: pip install librosa soundfile
    3. Restart: python app.py
    
    ============================================
    """)
    app.run(host='0.0.0.0', port=5000, debug=True)
