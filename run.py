"""
VocalFusion AI — Single Entry Point
=====================================
python run.py serve [--port N]          # Web UI at localhost:5000
python run.py analyze <file> [--name x] # Analyze a song
python run.py list                      # List analyzed songs
python run.py fuse <id_a> <id_b>        # Fuse two songs
"""

import os
import sys
import json
import argparse
import threading
import uuid
import numpy as np
import librosa
import soundfile as sf
from pathlib import Path
from datetime import datetime
from flask import Flask, request, jsonify, send_from_directory, render_template_string

from vocalfusion.professional_separation import ProfessionalSeparator
from vocalfusion.song_dna import SongAnalyzer
from vocalfusion.mixing_v2 import MixingEngineV2
from vocalfusion.mix_intelligence import MixIntelligence

DATA_DIR = Path("vf_data")
SAMPLE_RATE = 44100


# ============================================================================
# SAFE SERIALIZATION
# ============================================================================

def safe_serialize(obj):
    """Convert any object to a JSON-safe value"""
    if obj is None:
        return None
    if isinstance(obj, (str, int, float, bool)):
        return obj
    if isinstance(obj, np.integer):
        return int(obj)
    if isinstance(obj, np.floating):
        return float(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, dict):
        return {k: safe_serialize(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [safe_serialize(v) for v in obj]
    if hasattr(obj, 'value'):  # Enum
        return obj.value
    if hasattr(obj, '__dict__'):
        return {k: safe_serialize(v) for k, v in obj.__dict__.items()
                if not k.startswith('_')}
    return str(obj)


# ============================================================================
# ENGINE
# ============================================================================

class VocalFusion:
    """Main VocalFusion AI engine — no legacy dependencies"""

    def __init__(self, base_dir=DATA_DIR, ir_path=None):
        self.base_dir = Path(base_dir)
        self.sr = SAMPLE_RATE
        self.separator = ProfessionalSeparator()
        self.analyzer = SongAnalyzer(self.sr)
        self.mixing_engine = MixingEngineV2(sample_rate=self.sr, ir_path=ir_path)
        self.intelligence = MixIntelligence(self.sr)
        print("VocalFusion AI initialized (v10.0 — AI DJ)")

    # ------------------------------------------------------------------
    # PROCESS (separate + analyze + save)
    # ------------------------------------------------------------------

    def process_single_song(self, audio_path, song_id=None):
        """Separate stems, analyze, save results to vf_data/"""
        audio_path = Path(audio_path)
        if song_id is None:
            song_id = audio_path.stem

        print(f"\n{'='*60}")
        print(f"PROCESSING: {audio_path.name} → {song_id}")
        print(f"{'='*60}\n")

        stems_dir = self.base_dir / "stems" / song_id
        analysis_dir = self.base_dir / "analysis" / song_id
        stems_dir.mkdir(parents=True, exist_ok=True)
        analysis_dir.mkdir(parents=True, exist_ok=True)

        # Step 1: Stem separation
        print("Step 1: Separating stems (Demucs)...")
        stem_paths = self.separator.separate_file(audio_path, stems_dir)

        # Step 2: Load stems as numpy arrays
        print("Step 2: Loading stems...")
        stems = self._stems_from_paths(stem_paths)

        # Step 3: Deep analysis
        print("Step 3: Analyzing song DNA...")
        dna = self.analyzer.analyze(stems)

        # Step 4: Save analysis to JSON
        voice_type = self._estimate_voice_type(stems.get('vocals'))
        analysis_data = {
            'song_id': song_id,
            'duration': dna.duration,
            'key': dna.key,
            'tempo': dna.beat_grid.tempo,
            'bars': dna.beat_grid.bars,
            'has_strong_vocals': dna.has_strong_vocals,
            'has_strong_drums': dna.has_strong_drums,
            'overall_energy': dna.overall_energy,
            'vocals': {'voice_type': voice_type},
            'sections': [
                {
                    'start_time': s.start_time,
                    'end_time': s.end_time,
                    'duration': s.duration,
                    'classification': s.classification,
                    'energy': s.energy,
                    'has_vocals': s.has_vocals,
                }
                for s in dna.sections
            ],
            'key_chroma': dna.key_chroma.tolist() if len(dna.key_chroma) > 0 else [],
            'processed_at': datetime.now().isoformat(),
        }
        with open(analysis_dir / "analysis.json", 'w') as f:
            json.dump(safe_serialize(analysis_data), f, indent=2)

        print(f"\nDone! ID: {song_id}")
        print(f"  {dna.duration:.1f}s | {dna.key} | {dna.beat_grid.tempo:.0f} BPM")
        return {'song_id': song_id, 'analysis': dna, 'analysis_data': analysis_data}

    def _stems_from_paths(self, stem_paths):
        """Load stem WAV files into numpy arrays"""
        stems = {}
        for name, path in stem_paths.items():
            try:
                audio, _ = librosa.load(str(path), sr=self.sr, mono=True)
                stems[name] = audio
            except Exception as e:
                print(f"  Warning: Could not load {name}: {e}")
        return stems

    def _estimate_voice_type(self, vocals):
        """Estimate voice type from vocal spectral centroid"""
        if vocals is None or len(vocals) == 0:
            return "unknown"
        try:
            sc = librosa.feature.spectral_centroid(y=vocals, sr=self.sr)[0]
            mean_centroid = float(np.mean(sc))
            if mean_centroid > 3000:
                return "soprano"
            elif mean_centroid > 2400:
                return "alto"
            elif mean_centroid > 1800:
                return "tenor"
            return "baritone"
        except Exception:
            return "unknown"

    # ------------------------------------------------------------------
    # LOAD
    # ------------------------------------------------------------------

    def _load_song_analysis(self, song_id):
        """Load analysis JSON for a song"""
        path = self.base_dir / "analysis" / song_id / "analysis.json"
        if not path.exists():
            raise FileNotFoundError(f"No analysis found for '{song_id}'. Run 'analyze' first.")
        with open(path) as f:
            return json.load(f)

    def _load_stems(self, song_id):
        """Load stems for a song from disk"""
        stems_dir = self.base_dir / "stems" / song_id
        stems = {}
        for name in ['vocals', 'drums', 'bass', 'other']:
            path = stems_dir / f"{name}.wav"
            if path.exists():
                try:
                    audio, _ = librosa.load(str(path), sr=self.sr, mono=True)
                    stems[name] = audio
                except Exception as e:
                    print(f"  Warning: Could not load {name}: {e}")
        return stems

    # ------------------------------------------------------------------
    # COMPATIBILITY
    # ------------------------------------------------------------------

    def analyze_compatibility(self, song_a_id, song_b_id):
        """Compute compatibility scores between two analyzed songs"""
        data_a = self._load_song_analysis(song_a_id)
        data_b = self._load_song_analysis(song_b_id)

        prediction = self.intelligence.predict_mashup_quality(data_a, data_b)
        tempo_compat = float(prediction.get('tempo_score', 0.5))
        key_compat = float(prediction.get('key_score', 0.5))

        # Refine key compatibility with chroma correlation
        chroma_a = np.array(data_a.get('key_chroma', []))
        chroma_b = np.array(data_b.get('key_chroma', []))
        if len(chroma_a) == 12 and len(chroma_b) == 12:
            best_corr = max(
                float(np.corrcoef(chroma_a, np.roll(chroma_b, s))[0, 1])
                for s in range(12)
            )
            key_compat = float(np.clip((best_corr + 0.5) / 1.5, 0, 1))

        overall = float(np.clip(tempo_compat * 0.4 + key_compat * 0.4 + 0.1, 0, 1))
        blend = float(np.clip(key_compat * 0.6 + tempo_compat * 0.4, 0, 1))

        return {
            'overall_score': overall,
            'key_compatibility': key_compat,
            'tempo_compatibility': tempo_compat,
            'range_compatibility': 0.7,
            'timbre_compatibility': 0.6,
            'structure_compatibility': 0.65,
            'vocal_blend_score': blend,
        }

    # ------------------------------------------------------------------
    # FUSE
    # ------------------------------------------------------------------

    def fuse_two_songs(self, song_a_id, song_b_id):
        """Create an AI-mixed fusion of two analyzed songs"""
        print(f"\n{'='*60}")
        print(f"FUSING: {song_a_id} + {song_b_id}")
        print(f"{'='*60}\n")

        print("Step 1: Loading stems...")
        stems_a = self._load_stems(song_a_id)
        stems_b = self._load_stems(song_b_id)
        if not stems_a:
            raise RuntimeError(f"No stems for '{song_a_id}'. Run 'analyze' first.")
        if not stems_b:
            raise RuntimeError(f"No stems for '{song_b_id}'. Run 'analyze' first.")

        mix_dir = self.base_dir / "mixes" / f"{song_a_id}_{song_b_id}"
        mix_dir.mkdir(parents=True, exist_ok=True)

        print("Step 2: AI mixing (v10.0)...")
        mix_result = self.mixing_engine.create_mix(stems_a, stems_b, {}, {})

        print("Step 3: Saving output...")
        quality_scores = mix_result.pop('quality_scores', None)
        for name, audio in mix_result.items():
            if audio is not None and isinstance(audio, np.ndarray) and len(audio) > 0:
                path = mix_dir / f"{name}.wav"
                sf.write(str(path), audio, self.sr)
                print(f"  {name}.wav ({len(audio)/self.sr:.1f}s)")

        report = {
            'song_a_id': song_a_id,
            'song_b_id': song_b_id,
            'quality_scores': safe_serialize(quality_scores),
            'created_at': datetime.now().isoformat(),
        }
        with open(mix_dir / "fusion_report.json", 'w') as f:
            json.dump(report, f, indent=2, default=str)

        print(f"\nDone! Output: {mix_dir}")
        return {
            'song_a_id': song_a_id,
            'song_b_id': song_b_id,
            'quality_scores': safe_serialize(quality_scores),
            'paths': {'mixes': str(mix_dir)},
        }


# ============================================================================
# WEB APP
# ============================================================================

web_app = Flask(__name__)
engine = None
active_jobs = {}

HTML = '''<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>VocalFusion AI v10.0</title>
<style>
*{margin:0;padding:0;box-sizing:border-box}
body{font-family:-apple-system,BlinkMacSystemFont,'Segoe UI',Roboto,sans-serif;background:linear-gradient(135deg,#667eea 0%,#764ba2 100%);min-height:100vh;padding:20px}
.container{max-width:1200px;margin:0 auto;background:#fff;border-radius:20px;box-shadow:0 20px 60px rgba(0,0,0,.3);overflow:hidden}
header{background:linear-gradient(135deg,#4f46e5,#7c3aed);color:#fff;padding:40px;text-align:center}
h1{font-size:2.5rem;font-weight:800;margin-bottom:8px}
.sub{opacity:.9;font-size:1.1rem}
.badge{display:inline-block;background:rgba(255,255,255,.2);padding:4px 12px;border-radius:20px;font-size:.8rem;margin-top:10px}
.grid{display:grid;grid-template-columns:1fr 1fr;gap:30px;padding:30px}
@media(max-width:768px){.grid{grid-template-columns:1fr}}
.panel{background:#f8fafc;border-radius:15px;padding:25px}
h2{color:#334155;margin-bottom:15px;font-size:1.4rem;border-bottom:3px solid #4f46e5;padding-bottom:8px}
.drop{border:3px dashed #cbd5e1;border-radius:10px;padding:30px;text-align:center;cursor:pointer;transition:.3s}
.drop:hover{border-color:#4f46e5;background:#eef2ff}
.btn{background:linear-gradient(135deg,#4f46e5,#7c3aed);color:#fff;border:none;padding:12px 24px;border-radius:10px;font-size:1rem;font-weight:600;cursor:pointer;width:100%;margin-top:8px;transition:.2s}
.btn:hover{transform:translateY(-2px);box-shadow:0 8px 16px rgba(79,70,229,.3)}
.btn:disabled{opacity:.5;cursor:not-allowed;transform:none}
.songs{list-style:none;max-height:300px;overflow-y:auto;margin-top:15px}
.song{background:#fff;padding:12px;border-radius:8px;margin-bottom:8px;border-left:4px solid #4f46e5}
.song b{color:#334155;font-size:.95rem}
.song small{display:block;color:#64748b;margin-top:4px}
.bar-wrap{margin-top:15px;background:#e2e8f0;border-radius:10px;overflow:hidden;height:16px;display:none}
.bar{height:100%;background:linear-gradient(90deg,#4f46e5,#7c3aed);width:0%;transition:width .5s}
.status{text-align:center;padding:15px;background:#f1f5f9;border-radius:10px;margin-top:15px;color:#475569;display:none}
select{width:100%;padding:10px;border:2px solid #cbd5e1;border-radius:8px;font-size:.95rem;margin-bottom:10px;background:#fff}
.score{font-size:2.5rem;font-weight:800;text-align:center;color:#4f46e5;margin:15px 0}
.metric{display:flex;justify-content:space-between;padding:10px 0;border-bottom:1px solid #e2e8f0}
.metric span:first-child{color:#475569}
.metric span:last-child{color:#4f46e5;font-weight:600}
.ai-scores{margin-top:20px;padding:15px;background:#f0fdf4;border-radius:10px;border:2px solid #86efac;display:none}
.ai-scores h3{color:#166534;margin-bottom:10px;font-size:1.1rem}
.ai-bar{height:8px;background:#e2e8f0;border-radius:4px;margin:4px 0 10px 0;overflow:hidden}
.ai-fill{height:100%;border-radius:4px;transition:width .5s}
footer{text-align:center;padding:15px;color:#94a3b8;font-size:.85rem;border-top:1px solid #e2e8f0}
</style>
</head>
<body>
<div class="container">
<header>
<h1>VocalFusion AI</h1>
<div class="sub">AI-Powered Vocal Fusion</div>
<div class="badge">v10.0 — AI DJ</div>
</header>
<div class="grid">
<div class="panel">
<h2>Upload &amp; Analyze</h2>
<div class="drop" id="drop" onclick="document.getElementById('fi').click()">
<div style="font-size:2.5rem;color:#94a3b8;margin-bottom:10px">&#127925;</div>
<div style="color:#334155">Click to upload audio</div>
<div style="color:#94a3b8;font-size:.85rem">WAV, MP3, FLAC, M4A</div>
</div>
<input type="file" id="fi" accept="audio/*" style="display:none">
<div id="fn" style="text-align:center;color:#4f46e5;font-weight:600;margin-top:8px"></div>
<button class="btn" id="goBtn" disabled>Analyze Song</button>
<div class="bar-wrap" id="aBar"><div class="bar" id="aFill"></div></div>
<div class="status" id="aStat"></div>
<h2 style="margin-top:30px">Analyzed Songs</h2>
<ul class="songs" id="songList"><li style="color:#94a3b8;text-align:center;padding:20px">Loading...</li></ul>
</div>
<div class="panel">
<h2>Fuse Two Songs</h2>
<div style="margin-bottom:5px;color:#475569;font-weight:500;font-size:.9rem">Song A:</div>
<select id="sA"><option value="">Choose...</option></select>
<div style="margin-bottom:5px;color:#475569;font-weight:500;font-size:.9rem">Song B:</div>
<select id="sB"><option value="">Choose...</option></select>
<button class="btn" id="compBtn" disabled>Check Compatibility</button>
<div id="compRes" style="display:none">
<div class="score" id="oScore">-</div>
<div style="text-align:center;color:#64748b;margin-bottom:20px">Overall Compatibility</div>
<div id="mets"></div>
<button class="btn" id="fuseBtn" style="margin-top:15px">Create Fusion</button>
</div>
<div class="bar-wrap" id="fBar"><div class="bar" id="fFill"></div></div>
<div class="status" id="fStat"></div>
<div class="ai-scores" id="aiScores">
<h3>AI Mix Quality Analysis</h3>
<div id="aiMetrics"></div>
</div>
</div>
</div>
<footer>VocalFusion AI v10.0 &mdash; AI DJ &bull; Experiment &rarr; Score &rarr; Decide</footer>
</div>
<script>
var file=null,fi=document.getElementById("fi"),goBtn=document.getElementById("goBtn");
var sA=document.getElementById("sA"),sB=document.getElementById("sB"),compBtn=document.getElementById("compBtn");

fi.onchange=function(){if(fi.files.length){file=fi.files[0];document.getElementById("fn").textContent=file.name;goBtn.disabled=false}};
sA.onchange=sB.onchange=function(){compBtn.disabled=!(sA.value&&sB.value&&sA.value!==sB.value)};

function prog(id,pct){var w=document.getElementById(id);w.parentElement.style.display="block";w.style.width=pct+"%"}
function stat(id,html){var e=document.getElementById(id);e.style.display="block";e.innerHTML=html}

goBtn.onclick=function(){
  if(!file)return;
  var fd=new FormData();fd.append("file",file);fd.append("name",file.name.replace(/\.[^/.]+$/,""));
  prog("aFill",10);stat("aStat","Uploading...");goBtn.disabled=true;
  fetch("/api/upload",{method:"POST",body:fd}).then(function(r){return r.json()}).then(function(d){
    if(d.success){stat("aStat","Processing... (takes a few minutes)");prog("aFill",30);poll(d.job_id)}
    else{stat("aStat","Error: "+d.error);goBtn.disabled=false}
  }).catch(function(e){stat("aStat","Error: "+e.message);goBtn.disabled=false});
};

function poll(jid){
  var iv=setInterval(function(){
    fetch("/api/job/"+jid).then(function(r){return r.json()}).then(function(d){
      if(!d.success)return;var j=d.job;prog("aFill",j.progress||50);
      if(j.status==="completed"){clearInterval(iv);prog("aFill",100);stat("aStat","Done!");
        file=null;fi.value="";document.getElementById("fn").textContent="";goBtn.disabled=true;loadSongs()}
      else if(j.status==="failed"){clearInterval(iv);stat("aStat","Error: "+(j.error||"Unknown"));goBtn.disabled=false}
    }).catch(function(){});
  },3000);
}

compBtn.onclick=function(){
  var a=sA.value,b=sB.value;if(!a||!b||a===b)return;
  compBtn.disabled=true;stat("fStat","Checking...");
  fetch("/api/compatibility/"+encodeURIComponent(a)+"/"+encodeURIComponent(b))
  .then(function(r){return r.json()}).then(function(d){
    compBtn.disabled=false;
    if(d.success){showComp(d.compatibility);document.getElementById("fStat").style.display="none"}
    else stat("fStat","Error: "+d.error);
  }).catch(function(e){compBtn.disabled=false;stat("fStat","Error: "+e.message)});
};

function showComp(c){
  document.getElementById("compRes").style.display="block";
  document.getElementById("oScore").textContent=(c.overall_score||0).toFixed(2);
  var m=document.getElementById("mets");m.innerHTML="";
  [["Key",c.key_compatibility],["Tempo",c.tempo_compatibility],
   ["Range",c.range_compatibility],["Timbre",c.timbre_compatibility],
   ["Structure",c.structure_compatibility],["Blend",c.vocal_blend_score]].forEach(function(p){
    if(p[1]!==undefined){var d=document.createElement("div");d.className="metric";
    d.innerHTML="<span>"+p[0]+"</span><span>"+(p[1]||0).toFixed(2)+"</span>";m.appendChild(d)}});
}

function showAIScores(q){
  if(!q)return;
  var box=document.getElementById("aiScores");box.style.display="block";
  var m=document.getElementById("aiMetrics");m.innerHTML="";
  var dims=[["Beat Coherence","beat_coherence"],["Spectral Balance","spectral_balance"],
    ["Harmonic Clarity","harmonic_clarity"],["Vocal Clarity","vocal_clarity"],
    ["Dynamic Range","dynamic_range"],["Phase Coherence","phase_coherence"],
    ["Separation","spectral_separation"],["Energy","energy_consistency"]];
  dims.forEach(function(d){
    var val=q[d[1]]||0;
    var color=val>=0.7?"#22c55e":val>=0.4?"#eab308":"#ef4444";
    m.innerHTML+="<div style='display:flex;justify-content:space-between;font-size:.9rem'><span>"+d[0]+"</span><span style='color:"+color+";font-weight:600'>"+(val).toFixed(2)+"</span></div><div class='ai-bar'><div class='ai-fill' style='width:"+(val*100)+"%;background:"+color+"'></div></div>";
  });
  var overall=q.overall||0;
  var ocolor=overall>=0.7?"#22c55e":overall>=0.4?"#eab308":"#ef4444";
  m.innerHTML+="<div style='margin-top:10px;font-size:1.1rem;font-weight:700;text-align:center;color:"+ocolor+"'>Overall: "+(overall).toFixed(2)+"</div>";
}

document.getElementById("fuseBtn").onclick=function(){
  var a=sA.value,b=sB.value;if(!a||!b||a===b)return;
  prog("fFill",10);stat("fStat","AI analyzing and mixing... (several minutes)");
  document.getElementById("aiScores").style.display="none";
  fetch("/api/fuse/"+encodeURIComponent(a)+"/"+encodeURIComponent(b),{method:"POST"})
  .then(function(r){return r.json()}).then(function(d){
    prog("fFill",100);
    if(d.success){
      var m="Fusion complete!";
      if(d.download_url)m+=" <a href='"+d.download_url+"' style='color:#4f46e5;font-weight:600'>Download</a>";
      stat("fStat",m);
      if(d.quality_scores)showAIScores(d.quality_scores);
    }else stat("fStat","Error: "+d.error);
  }).catch(function(e){stat("fStat","Error: "+e.message)});
};

function loadSongs(){
  fetch("/api/songs").then(function(r){return r.json()}).then(function(d){
    if(!d.success)return;
    var list=document.getElementById("songList");list.innerHTML="";
    sA.innerHTML="<option value=''>Choose...</option>";sB.innerHTML="<option value=''>Choose...</option>";
    if(!d.songs.length){list.innerHTML="<li style='color:#94a3b8;text-align:center;padding:20px'>No songs yet</li>";return}
    d.songs.forEach(function(s){
      var li=document.createElement("li");li.className="song";
      li.innerHTML="<b>"+s.song_id+"</b><small>"+Math.round(s.duration)+"s | "+
        (s.key||"?")+" | "+Math.round(s.tempo)+" BPM | "+(s.voice_type||"?")+"</small>";
      list.appendChild(li);
      var o1=document.createElement("option");o1.value=s.song_id;o1.textContent=s.song_id;sA.appendChild(o1);
      var o2=document.createElement("option");o2.value=s.song_id;o2.textContent=s.song_id;sB.appendChild(o2);
    });
  }).catch(function(){document.getElementById("songList").innerHTML="<li style='color:#94a3b8;text-align:center;padding:20px'>Error loading</li>"});
}
loadSongs();
</script>
</body>
</html>'''


@web_app.route('/')
def index():
    return render_template_string(HTML)


@web_app.route('/api/upload', methods=['POST'])
def upload_song():
    global engine
    if 'file' not in request.files:
        return jsonify({'success': False, 'error': 'No file provided'}), 400
    file = request.files['file']
    song_name = request.form.get('name', file.filename)
    song_name_clean = os.path.splitext(song_name)[0]
    song_id = f"{song_name_clean}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

    temp_dir = Path(DATA_DIR) / "temp"
    temp_dir.mkdir(parents=True, exist_ok=True)
    ext = os.path.splitext(file.filename)[1] or '.mp3'
    temp_path = temp_dir / f"{song_id}{ext}"
    file.save(str(temp_path))

    job_id = str(uuid.uuid4())
    active_jobs[job_id] = {'status': 'processing', 'song_id': song_id, 'progress': 0}

    def bg():
        try:
            active_jobs[job_id]['progress'] = 10
            result = engine.process_single_song(temp_path, song_id)
            active_jobs[job_id].update({
                'status': 'completed', 'progress': 100,
                'result': {'song_id': result['song_id']}
            })
        except Exception as e:
            import traceback
            active_jobs[job_id].update({'status': 'failed', 'error': str(e)})
            print(traceback.format_exc())
        finally:
            try:
                temp_path.unlink(missing_ok=True)
            except Exception:
                pass

    threading.Thread(target=bg, daemon=True).start()
    return jsonify({'success': True, 'job_id': job_id, 'song_id': song_id})


@web_app.route('/api/songs')
def list_songs():
    analysis_dir = Path(DATA_DIR) / "analysis"
    songs = []
    if analysis_dir.exists():
        for d in sorted(analysis_dir.iterdir()):
            if d.is_dir() and (d / "analysis.json").exists():
                try:
                    with open(d / "analysis.json") as f:
                        data = json.load(f)
                    vt = 'none'
                    if isinstance(data.get('vocals'), dict):
                        vt = data['vocals'].get('voice_type', 'none')
                    songs.append({
                        'song_id': d.name,
                        'duration': data.get('duration', 0),
                        'key': data.get('key', '?'),
                        'tempo': data.get('tempo', 0),
                        'voice_type': vt,
                    })
                except Exception:
                    songs.append({'song_id': d.name, 'duration': 0, 'key': '?',
                                  'tempo': 0, 'voice_type': '?'})
    return jsonify({'success': True, 'songs': songs})


@web_app.route('/api/job/<job_id>')
def get_job(job_id):
    if job_id not in active_jobs:
        return jsonify({'success': False, 'error': 'Job not found'}), 404
    return jsonify({'success': True, 'job': active_jobs[job_id]})


@web_app.route('/api/compatibility/<path:song_a>/<path:song_b>')
def get_compatibility(song_a, song_b):
    try:
        c = engine.analyze_compatibility(song_a, song_b)
        return jsonify({'success': True, 'compatibility': c})
    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({'success': False, 'error': str(e)}), 500


@web_app.route('/api/fuse/<path:song_a>/<path:song_b>', methods=['POST'])
def fuse_songs(song_a, song_b):
    try:
        result = engine.fuse_two_songs(song_a, song_b)
        mix_file = Path(DATA_DIR) / "mixes" / f"{song_a}_{song_b}" / "full_mix.wav"
        url = f"/download/{song_a}_{song_b}" if mix_file.exists() else None
        return jsonify({
            'success': True,
            'download_url': url,
            'quality_scores': result.get('quality_scores'),
        })
    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({'success': False, 'error': str(e)}), 500


@web_app.route('/api/download/<path:fusion_id>')
@web_app.route('/download/<path:fusion_id>')
def download_mix(fusion_id):
    p = Path(DATA_DIR) / "mixes" / fusion_id / "full_mix.wav"
    if p.exists():
        return send_from_directory(str(p.parent), p.name, as_attachment=True)
    return "Not found", 404


# ============================================================================
# CLI
# ============================================================================

def main():
    global engine
    parser = argparse.ArgumentParser(description='VocalFusion AI v10.0')
    parser.add_argument('--ir', help='Reverb impulse response WAV')
    sub = parser.add_subparsers(dest='cmd')

    p = sub.add_parser('analyze', help='Analyze a song')
    p.add_argument('file')
    p.add_argument('--name', help='Song ID (default: filename stem)')

    sub.add_parser('list', help='List analyzed songs')

    p = sub.add_parser('compatibility', help='Check song compatibility')
    p.add_argument('song_a')
    p.add_argument('song_b')

    p = sub.add_parser('fuse', help='Fuse two songs')
    p.add_argument('song_a')
    p.add_argument('song_b')

    p = sub.add_parser('serve', help='Start web UI')
    p.add_argument('--port', type=int, default=5000)
    p.add_argument('--host', default='127.0.0.1')

    args = parser.parse_args()
    ir = getattr(args, 'ir', None)

    if args.cmd == 'serve':
        engine = VocalFusion(ir_path=ir)
        print(f"\nWeb UI: http://{args.host}:{args.port}\n")
        web_app.run(host=args.host, port=args.port, debug=False)

    elif args.cmd == 'analyze':
        engine = VocalFusion(ir_path=ir)
        result = engine.process_single_song(Path(args.file), args.name)
        a = result['analysis_data']
        print(f"\n  song_id:  {result['song_id']}")
        print(f"  duration: {a['duration']:.1f}s")
        print(f"  key:      {a['key']}")
        print(f"  tempo:    {a['tempo']:.0f} BPM")

    elif args.cmd == 'list':
        ad = Path(DATA_DIR) / "analysis"
        if not ad.exists() or not any(ad.iterdir()):
            print("No songs analyzed yet.")
            return
        print("\nAnalyzed songs:")
        for d in sorted(ad.iterdir()):
            if d.is_dir() and (d / "analysis.json").exists():
                try:
                    with open(d / "analysis.json") as f:
                        data = json.load(f)
                    print(f"  {d.name}: {data.get('duration', 0):.0f}s | "
                          f"{data.get('key', '?')} | {data.get('tempo', 0):.0f} BPM")
                except Exception:
                    print(f"  {d.name}: (error reading analysis)")

    elif args.cmd == 'compatibility':
        engine = VocalFusion(ir_path=ir)
        c = engine.analyze_compatibility(args.song_a, args.song_b)
        print(f"\nCompatibility: {args.song_a} + {args.song_b}")
        print(f"  Overall:   {c['overall_score']:.2f}")
        print(f"  Key:       {c['key_compatibility']:.2f}")
        print(f"  Tempo:     {c['tempo_compatibility']:.2f}")
        print(f"  Blend:     {c['vocal_blend_score']:.2f}")

    elif args.cmd == 'fuse':
        engine = VocalFusion(ir_path=ir)
        engine.fuse_two_songs(args.song_a, args.song_b)

    else:
        parser.print_help()


if __name__ == "__main__":
    main()
