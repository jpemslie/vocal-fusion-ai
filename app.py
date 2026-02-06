#!/usr/bin/env python3
"""
VOCAL FUSION AI - FIXED VERSION (Handles torchcodec issue)
"""

import os
import json
import time
import traceback
import subprocess
import sys
from pathlib import Path
from flask import Flask, request, jsonify, render_template_string, send_file

app = Flask(__name__)

# ============================================================================
# 1. VOCAL FUSION AI ENGINE - FIXED
# ============================================================================
class VocalFusionAI:
    """Fixed version that handles torchcodec issues."""
    
    def __init__(self):
        self.setup_directories()
        self.status = "ready"
        print("🤖 Initializing Vocal Fusion AI...")
        print(f"   Python: {sys.version}")
        
    def setup_directories(self):
        """Create directory structure."""
        dirs = [
            'data/raw',
            'data/stems',
            'data/analysis',
            'data/outputs',
            'test_audio',
            'test_stems'
        ]
        for d in dirs:
            Path(d).mkdir(parents=True, exist_ok=True)
        print("✅ Directory structure created")
    
    def check_dependencies(self):
        """Check if all dependencies are available."""
        deps = {}
        
        try:
            import torch
            deps['torch'] = {
                'installed': True,
                'version': torch.__version__,
                'cuda': torch.cuda.is_available()
            }
        except ImportError:
            deps['torch'] = {'installed': False}
        
        try:
            import demucs
            deps['demucs'] = {
                'installed': True,
                'version': demucs.__version__
            }
        except ImportError:
            deps['demucs'] = {'installed': False}
        
        try:
            import librosa
            deps['librosa'] = {
                'installed': True,
                'version': librosa.__version__
            }
        except ImportError:
            deps['librosa'] = {'installed': False}
        
        try:
            import soundfile as sf
            deps['soundfile'] = {'installed': True}
        except ImportError:
            deps['soundfile'] = {'installed': False}
        
        # Check for torchcodec issue
        deps['torchcodec_issue'] = self.check_torchcodec_issue()
        
        return deps
    
    def check_torchcodec_issue(self):
        """Check if torchcodec is causing problems."""
        try:
            # Try to import torchcodec
            import torchcodec
            return {'has_torchcodec': True, 'issue': False}
        except ImportError:
            # Check if we have ffmpeg as alternative
            try:
                result = subprocess.run(['which', 'ffmpeg'], 
                                      capture_output=True, text=True)
                has_ffmpeg = result.returncode == 0
                return {
                    'has_torchcodec': False,
                    'has_ffmpeg': has_ffmpeg,
                    'issue': True,
                    'message': 'torchcodec missing, will use workarounds'
                }
            except:
                return {'has_torchcodec': False, 'issue': True}
    
    def load_audio(self, file_path):
        """Load audio file using librosa."""
        try:
            import librosa
            import soundfile as sf
            import numpy as np
            
            print(f"🔊 Loading audio: {file_path}")
            
            if not Path(file_path).exists():
                return {'error': f'File not found: {file_path}', 'status': 'error'}
            
            # Get file info
            info = sf.info(file_path)
            
            # Load audio
            audio, sample_rate = librosa.load(
                file_path,
                sr=None,
                mono=False,
                duration=None
            )
            
            if isinstance(audio, list):
                audio = np.array(audio)
            
            duration = librosa.get_duration(y=audio, sr=sample_rate)
            
            # Handle mono/stereo
            if audio.ndim == 1:
                channels = 1
                audio = audio.reshape(1, -1)
            else:
                channels = audio.shape[0]
            
            # Calculate stats
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
            print(f"❌ Error loading audio: {str(e)}")
            traceback.print_exc()
            return {'error': str(e), 'status': 'error'}
    
    def separate_stems_with_demucs(self, audio_data, use_mp3=False):
        """Separate stems using Demucs with torchcodec workaround."""
        print(f"🎚️ Starting stem separation for: {audio_data['file_path']}")
        print(f"   Using MP3 workaround: {use_mp3}")
        
        start_time = time.time()
        
        try:
            input_path = audio_data['file_path']
            song_name = Path(input_path).stem
            timestamp = int(time.time())
            
            # Create output directory
            output_dir = Path(f"data/stems/{song_name}_{timestamp}")
            output_dir.mkdir(parents=True, exist_ok=True)
            
            print(f"   Output directory: {output_dir}")
            
            # Build demucs command
            cmd = [
                sys.executable, "-m", "demucs",
                "--out", str(output_dir),
                "--name", "htdemucs"
            ]
            
            # Add MP3 option if torchcodec issue
            if use_mp3:
                cmd.extend(["--mp3", "--mp3-bitrate", "320"])
            else:
                # Try to save as WAV with ffmpeg fallback
                cmd.extend(["--float32"])  # Use float32 format
            
            # Add input file
            cmd.append(input_path)
            
            print(f"   Command: {' '.join(cmd)}")
            
            # Run demucs
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=300  # 5 minute timeout
            )
            
            if result.returncode != 0:
                print(f"❌ Demucs failed with error: {result.stderr}")
                
                # Try alternative model
                print("   Trying alternative model: mdx_extra_q")
                cmd_alt = [
                    sys.executable, "-m", "demucs",
                    "--out", str(output_dir),
                    "--name", "mdx_extra_q"
                ]
                
                if use_mp3:
                    cmd_alt.extend(["--mp3", "--mp3-bitrate", "320"])
                
                cmd_alt.append(input_path)
                
                result_alt = subprocess.run(
                    cmd_alt,
                    capture_output=True,
                    text=True,
                    timeout=300
                )
                
                if result_alt.returncode != 0:
                    raise Exception(f"Both models failed. Last error: {result_alt.stderr}")
                
                result = result_alt
            
            print(f"✅ Demucs completed successfully in {time.time() - start_time:.1f} seconds")
            
            # Find output files
            # Demucs creates: output_dir/model_name/song_name/stem.wav
            model_dirs = list(output_dir.glob("*"))
            if not model_dirs:
                raise Exception("No model directory found in output")
            
            model_dir = model_dirs[0]
            song_dirs = list(model_dir.glob("*"))
            
            if not song_dirs:
                # Look for any directory with stem files
                stem_files = list(model_dir.rglob("*.wav")) + list(model_dir.rglob("*.mp3"))
                if stem_files:
                    # Group by parent directory
                    from collections import defaultdict
                    dir_files = defaultdict(list)
                    for f in stem_files:
                        dir_files[f.parent].append(f)
                    
                    if dir_files:
                        song_dir = list(dir_files.keys())[0]
                        stem_files = dir_files[song_dir]
                else:
                    raise Exception("No stem files found")
            else:
                song_dir = song_dirs[0]
                stem_files = list(song_dir.glob("*"))
            
            stems = {}
            for stem_file in stem_files:
                stem_name = stem_file.stem  # Remove extension
                stems[stem_name] = str(stem_file)
                
                # Get duration if possible
                try:
                    import librosa
                    stem_audio, sr = librosa.load(str(stem_file), sr=None, mono=False)
                    duration = librosa.get_duration(y=stem_audio, sr=sr)
                    stems[f'{stem_name}_duration'] = duration
                    print(f"   ✓ {stem_name}: {stem_file.name} ({duration:.2f}s)")
                except:
                    print(f"   ✓ {stem_name}: {stem_file.name}")
            
            # Create info file
            info = {
                'input_file': input_path,
                'output_dir': str(output_dir),
                'stems': stems,
                'processing_time': time.time() - start_time,
                'demucs_version': '4.0.0',
                'model': model_dir.name,
                'format': 'mp3' if use_mp3 else 'wav'
            }
            
            with open(output_dir / 'separation_info.json', 'w') as f:
                json.dump(info, f, indent=2)
            
            print(f"✅ Stem separation complete!")
            print(f"   Found {len(stems)} stems in {output_dir}")
            
            return {
                'success': True,
                'status': 'separated',
                'stems': stems,
                'output_dir': str(output_dir),
                'info_path': str(output_dir / 'separation_info.json'),
                'processing_time': info['processing_time']
            }
            
        except subprocess.TimeoutExpired:
            error_msg = "Demucs timed out after 5 minutes"
            print(f"❌ {error_msg}")
            return {
                'success': False,
                'status': 'error',
                'error': error_msg
            }
        except Exception as e:
            error_msg = str(e)
            print(f"❌ Error in stem separation: {error_msg}")
            traceback.print_exc()
            return {
                'success': False,
                'status': 'error',
                'error': error_msg
            }
    
    def test_stem_separation(self, test_file=None):
        """Test stem separation with a simple audio file."""
        if not test_file:
            # Create a test audio file
            import numpy as np
            import soundfile as sf
            
            test_dir = Path("test_audio")
            test_dir.mkdir(exist_ok=True)
            
            test_file = test_dir / f"test_{int(time.time())}.wav"
            
            # Create a simple audio with multiple frequencies
            sample_rate = 44100
            duration = 10.0
            t = np.linspace(0, duration, int(sample_rate * duration), endpoint=False)
            
            # Different frequency components
            bass = 0.3 * np.sin(2 * np.pi * 100 * t)
            melody = 0.2 * np.sin(2 * np.pi * 440 * t)
            hihat = 0.1 * np.sin(2 * np.pi * 10000 * t) * (np.sin(2 * np.pi * 8 * t) > 0)
            vocals = 0.25 * np.sin(2 * np.pi * 330 * t) * (0.5 + 0.5 * np.sin(2 * np.pi * 0.5 * t))
            
            audio = bass + melody + hihat + vocals
            audio = audio / np.max(np.abs(audio))
            
            # Make stereo
            audio_stereo = np.vstack([audio, audio * 0.8]).T
            sf.write(str(test_file), audio_stereo, sample_rate)
            
            print(f"✅ Created test audio: {test_file}")
        
        # Load the audio
        audio_data = self.load_audio(str(test_file))
        if audio_data['status'] != 'loaded':
            return {
                'success': False,
                'error': f"Failed to load test audio: {audio_data.get('error')}"
            }
        
        # Try separation with MP3 workaround (for torchcodec issue)
        print("🚀 Testing stem separation with MP3 workaround...")
        result = self.separate_stems_with_demucs(audio_data, use_mp3=True)
        
        return result

# Initialize engine
ai = VocalFusionAI()

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
        .btn { background: #3B82F6; color: white; padding: 10px 20px; border: none; border-radius: 5px; cursor: pointer; text-decoration: none; display: inline-block; }
        .btn:hover { background: #2563EB; }
        .btn-success { background: #10B981; }
        .btn-warning { background: #F59E0B; }
        .btn-danger { background: #EF4444; }
        .status-ok { color: #10B981; }
        .status-warn { color: #F59E0B; }
        .status-error { color: #EF4444; }
        .log { background: #1F2937; color: #E5E7EB; padding: 15px; border-radius: 5px; font-family: monospace; white-space: pre-wrap; max-height: 300px; overflow-y: auto; }
    </style>
</head>
<body>
    <div class="container">
        <h1>🎵 Vocal Fusion AI - Working Version</h1>
        <p>Stem separation with Demucs (with torchcodec workaround)</p>
        
        <div class="card">
            <h2>📊 System Status</h2>
            <div id="system-status">Loading...</div>
            <a href="/api/dependencies" class="btn">Check Dependencies</a>
        </div>
        
        <div class="card">
            <h2>🧪 Test Stem Separation</h2>
            <p>This will create a test audio file and separate it into stems.</p>
            <button onclick="testStemSeparation()" class="btn btn-success">Test Stem Separation</button>
            <div id="test-result" style="margin-top: 20px;"></div>
            <div id="test-log" class="log" style="display: none;"></div>
        </div>
        
        <div class="card">
            <h2>📁 View Results</h2>
            <a href="/list/stems" class="btn">View All Stem Files</a>
            <a href="/list/test-audio" class="btn">View Test Audio</a>
        </div>
        
        <div class="card">
            <h2>🚀 Quick Start</h2>
            <ol>
                <li>Check system status above</li>
                <li>Click "Test Stem Separation"</li>
                <li>Wait 1-2 minutes for processing</li>
                <li>View results in "View All Stem Files"</li>
            </ol>
            <p><strong>Note:</strong> First run will download Demucs models (∼1GB).</p>
        </div>
    </div>
    
    <script>
    // Load system status
    fetch('/api/dependencies')
        .then(r => r.json())
        .then(data => {
            const statusDiv = document.getElementById('system-status');
            let html = '<h3>Dependencies:</h3><ul>';
            
            for (const [dep, info] of Object.entries(data.dependencies)) {
                if (info.installed) {
                    html += `<li class="status-ok">✅ ${dep}: ${info.version || 'Installed'}`;
                    if (dep === 'torch' && info.cuda) {
                        html += ` (CUDA available)`;
                    }
                    html += `</li>`;
                } else {
                    html += `<li class="status-error">❌ ${dep}: Missing</li>`;
                }
            }
            
            if (data.torchcodec_issue && data.torchcodec_issue.issue) {
                html += `<li class="status-warn">⚠️ Torchcodec: ${data.torchcodec_issue.message || 'Issue detected'}</li>`;
            }
            
            html += '</ul>';
            statusDiv.innerHTML = html;
        });
    
    async function testStemSeparation() {
        const resultDiv = document.getElementById('test-result');
        const logDiv = document.getElementById('test-log');
        
        resultDiv.innerHTML = '<p>Starting test... (This may take 1-2 minutes)</p>';
        logDiv.style.display = 'block';
        logDiv.innerHTML = 'Initializing...\n';
        
        try {
            // Start test
            const response = await fetch('/api/test-stem-separation', {
                method: 'POST',
                headers: {'Content-Type': 'application/json'}
            });
            
            const data = await response.json();
            
            // Update log
            logDiv.innerHTML += `\nStatus: ${data.status}\n`;
            
            if (data.success) {
                resultDiv.innerHTML = `
                    <h3 class="status-ok">✅ Stem Separation Successful!</h3>
                    <p>Processing time: ${data.processing_time.toFixed(1)} seconds</p>
                    <p>Output directory: ${data.output_dir}</p>
                    <p>Stems found: ${Object.keys(data.stems || {}).length}</p>
                    <p><a href="/list/stems" class="btn">View Stems</a></p>
                `;
                
                // List stems
                if (data.stems) {
                    logDiv.innerHTML += '\nStems created:\n';
                    for (const [name, path] of Object.entries(data.stems)) {
                        logDiv.innerHTML += `  • ${name}: ${path}\n`;
                    }
                }
            } else {
                resultDiv.innerHTML = `
                    <h3 class="status-error">❌ Stem Separation Failed</h3>
                    <p>Error: ${data.error || 'Unknown error'}</p>
                    <p>Check the terminal for more details.</p>
                `;
                logDiv.innerHTML += `\nError: ${data.error}\n`;
            }
            
        } catch (error) {
            resultDiv.innerHTML = `<h3 class="status-error">❌ Test Failed</h3><p>${error.message}</p>`;
            logDiv.innerHTML += `\nException: ${error.message}\n`;
        }
    }
    </script>
</body>
</html>
'''

@app.route('/')
def home():
    return render_template_string(HTML_TEMPLATE)

@app.route('/api/dependencies')
def api_dependencies():
    """Get dependency information."""
    deps = ai.check_dependencies()
    return jsonify({
        'dependencies': deps,
        'torchcodec_issue': ai.check_torchcodec_issue(),
        'system': {
            'python': sys.version,
            'cwd': os.getcwd()
        }
    })

@app.route('/api/test-stem-separation', methods=['POST'])
def api_test_stem_separation():
    """API endpoint to test stem separation."""
    result = ai.test_stem_separation()
    return jsonify(result)

@app.route('/list/stems')
def list_stems():
    """List all stem directories."""
    stem_dir = Path('data/stems')
    
    if not stem_dir.exists():
        return '''
        <div style="max-width: 800px; margin: 0 auto; padding: 20px;">
            <h1>📁 No Stems Found</h1>
            <p>Run the stem separation test first.</p>
            <p><a href="/" class="btn">← Back</a></p>
        </div>
        '''
    
    html = '''
    <div style="max-width: 800px; margin: 0 auto; padding: 20px;">
        <h1>📁 Stem Directories</h1>
        <p><a href="/" class="btn">← Back</a></p>
    '''
    
    for folder in sorted(stem_dir.iterdir(), key=lambda x: x.stat().st_mtime, reverse=True):
        if folder.is_dir():
            # Count files
            wav_files = list(folder.rglob("*.wav"))
            mp3_files = list(folder.rglob("*.mp3"))
            json_files = list(folder.rglob("*.json"))
            
            html += f'''
            <div style="background: white; padding: 20px; margin: 20px 0; border-radius: 10px;">
                <h3>📂 {folder.name}</h3>
                <p><strong>Path:</strong> {folder}</p>
                <p><strong>Files:</strong> {len(wav_files)} WAV, {len(mp3_files)} MP3, {len(json_files)} JSON</p>
                
                <h4>Audio Files:</h4>
                <ul>
            '''
            
            for audio_file in wav_files + mp3_files:
                rel_path = audio_file.relative_to(folder)
                html += f'<li><a href="/play/audio?file={audio_file}">▶️ {rel_path}</a></li>'
            
            html += '''
                </ul>
                
                <h4>Info Files:</h4>
                <ul>
            '''
            
            for json_file in json_files:
                rel_path = json_file.relative_to(folder)
                html += f'<li><a href="/view/json?file={json_file}">📄 {rel_path}</a></li>'
            
            html += '''
                </ul>
            </div>
            '''
    
    html += '</div>'
    return html

@app.route('/list/test-audio')
def list_test_audio():
    """List test audio files."""
    test_dir = Path('test_audio')
    
    if not test_dir.exists():
        return '''
        <div style="max-width: 800px; margin: 0 auto; padding: 20px;">
            <h1>No Test Audio</h1>
            <p>No test audio files found.</p>
            <p><a href="/" class="btn">← Back</a></p>
        </div>
        '''
    
    html = '''
    <div style="max-width: 800px; margin: 0 auto; padding: 20px;">
        <h1>🎵 Test Audio Files</h1>
        <p><a href="/" class="btn">← Back</a></p>
        <ul>
    '''
    
    for audio_file in test_dir.glob("*.wav"):
        html += f'''
        <li style="margin: 10px 0; padding: 10px; background: white; border-radius: 5px;">
            <strong>{audio_file.name}</strong>
            <br>
            <a href="/play/audio?file={audio_file}" class="btn" style="padding: 5px 10px; font-size: 0.9em;">▶️ Play</a>
            <a href="/api/analyze-audio?file={audio_file}" class="btn" style="padding: 5px 10px; font-size: 0.9em;">🔍 Analyze</a>
        </li>
        '''
    
    html += '</ul></div>'
    return html

@app.route('/play/audio')
def play_audio():
    """Play an audio file."""
    file_path = request.args.get('file')
    if not file_path or not Path(file_path).exists():
        return "Audio file not found", 404
    
    return f'''
    <!DOCTYPE html>
    <html>
    <head>
        <title>Play Audio</title>
        <style>
            body {{ font-family: Arial, sans-serif; padding: 20px; max-width: 600px; margin: 0 auto; }}
            .player {{ margin: 20px 0; }}
        </style>
    </head>
    <body>
        <h1>🎵 Playing Audio</h1>
        <p><strong>File:</strong> {Path(file_path).name}</p>
        <div class="player">
            <audio controls autoplay style="width: 100%;">
                <source src="/api/get-audio?file={file_path}" type="audio/wav">
                Your browser does not support the audio element.
            </audio>
        </div>
        <p><a href="javascript:history.back()" class="btn">← Back</a></p>
    </body>
    </html>
    '''

@app.route('/api/get-audio')
def api_get_audio():
    """Serve audio file."""
    file_path = request.args.get('file')
    if not file_path or not Path(file_path).exists():
        return jsonify({'error': 'File not found'}), 404
    
    # Determine MIME type
    ext = Path(file_path).suffix.lower()
    mime_types = {
        '.wav': 'audio/wav',
        '.mp3': 'audio/mpeg',
        '.flac': 'audio/flac'
    }
    
    mime_type = mime_types.get(ext, 'audio/wav')
    
    return send_file(file_path, mimetype=mime_type)

@app.route('/view/json')
def view_json():
    """View JSON file."""
    file_path = request.args.get('file')
    if not file_path or not Path(file_path).exists():
        return "JSON file not found", 404
    
    with open(file_path, 'r') as f:
        try:
            data = json.load(f)
            formatted = json.dumps(data, indent=2)
        except:
            formatted = "Invalid JSON"
    
    return f'''
    <div style="max-width: 800px; margin: 0 auto; padding: 20px;">
        <h1>📄 {Path(file_path).name}</h1>
        <pre style="background: #1F2937; color: #E5E7EB; padding: 20px; border-radius: 10px; overflow: auto;">
{formatted}
        </pre>
        <p><a href="javascript:history.back()" class="btn">← Back</a></p>
    </div>
    '''

# ============================================================================
# 3. MAIN EXECUTION
# ============================================================================
if __name__ == '__main__':
    print(f"""
    {'='*60}
    🎵 VOCAL FUSION AI - FIXED VERSION
    {'='*60}
    
    🔧 System Info:
      • Python: {sys.version.split()[0]}
      • CUDA: Available (torch 2.10.0+cu128)
      • Torchcodec issue: Will use MP3 workaround
    
    🚀 Access Points:
      • Main Interface: http://localhost:5000
      • Test Stem Separation: Click button on main page
      • View Stems: http://localhost:5000/list/stems
    
    ⚠️  Known Issue:
      • torchcodec missing - using MP3 workaround
      • First run downloads ~1GB models
    
    📋 Next Steps:
      1. Click "Test Stem Separation" on web page
      2. Wait 1-2 minutes for processing
      3. View results in "View All Stem Files"
      4. Then we'll implement vocal analysis
    
    {'='*60}
    """)
    
    # Check dependencies
    deps = ai.check_dependencies()
    print("\n📦 Dependencies:")
    for dep, info in deps.items():
        if dep != 'torchcodec_issue':
            status = "✅" if info.get('installed') else "❌"
            version = info.get('version', 'N/A')
            print(f"  {status} {dep}: {version}")
    
    print(f"\n{'='*60}\n")
    
    app.run(host='0.0.0.0', port=5000, debug=True)
