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
import queue
import uuid
from typing import Dict
import numpy as np
import librosa
import soundfile as sf
from pathlib import Path
from datetime import datetime
from flask import Flask, request, jsonify, send_from_directory, render_template

from audio.professional_separation import ProfessionalSeparator
from analysis.song_dna import SongAnalyzer
from core.mixing_v2 import MixingEngineV2
from analysis.mix_intelligence import MixIntelligence
from ai.mert_embedder import MertEmbedder
from ai.mix_predictor import MixPredictor

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
        d = {k: safe_serialize(v) for k, v in obj.__dict__.items()
             if not k.startswith('_')}
        # Include @property values that aren't stored in __dict__ (e.g. MixScore.overall)
        for prop in ('overall',):
            if prop not in d and hasattr(type(obj), prop):
                try:
                    d[prop] = safe_serialize(getattr(obj, prop))
                except Exception:
                    pass
        return d
    return str(obj)


# ============================================================================
# LOG BROADCASTER — captures stdout and fans out to SSE subscribers
# ============================================================================

class _LogBroadcaster:
    """Wraps sys.stdout; every print() is mirrored to all SSE subscribers."""

    def __init__(self):
        self._queues = []
        self._lock = threading.Lock()
        self._real = sys.stdout

    def install(self):
        sys.stdout = self

    def write(self, text):
        self._real.write(text)
        self._real.flush()
        if text.strip():
            with self._lock:
                dead = []
                for q in self._queues:
                    try:
                        q.put_nowait(text.rstrip('\n'))
                    except queue.Full:
                        dead.append(q)
                for q in dead:
                    self._queues.remove(q)

    def flush(self):
        self._real.flush()

    def subscribe(self):
        q = queue.Queue(maxsize=500)
        with self._lock:
            self._queues.append(q)
        return q

    def unsubscribe(self, q):
        with self._lock:
            try:
                self._queues.remove(q)
            except ValueError:
                pass


_broadcaster = _LogBroadcaster()


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
        self.embedder = MertEmbedder(
            cache_dir=self.base_dir / "embeddings",
            sample_rate=self.sr)
        self.predictor = MixPredictor(data_dir=self.base_dir)
        print("VocalFusion AI initialized (v10.0 — AI DJ + Learning)")

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

        # Step 3: Generate audio embedding (cached)
        print("Step 3: Generating song embedding...")
        try:
            self.embedder.embed_song(song_id, stems)
        except Exception as e:
            print(f"  Warning: Embedding failed: {e}")

        # Step 4: Deep analysis
        print("Step 4: Analyzing song DNA...")
        dna = self.analyzer.analyze(stems)

        # Step 5: Save analysis to JSON
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

        # Timbre compatibility: cosine similarity of MFCC-based spectral centroid
        mfcc_a = np.array(data_a.get('key_chroma', []))
        mfcc_b = np.array(data_b.get('key_chroma', []))
        if len(mfcc_a) > 0 and len(mfcc_b) > 0 and len(mfcc_a) == len(mfcc_b):
            dot = float(np.dot(mfcc_a, mfcc_b))
            norm = float(np.linalg.norm(mfcc_a) * np.linalg.norm(mfcc_b) + 1e-10)
            timbre_compat = float(np.clip((dot / norm + 1) / 2, 0, 1))
        else:
            timbre_compat = 0.6

        # Range compatibility: RMS energy similarity
        energy_a = float(data_a.get('overall_energy', 0.5))
        energy_b = float(data_b.get('overall_energy', 0.5))
        range_compat = float(np.clip(1.0 - abs(energy_a - energy_b), 0, 1))

        # Structure compatibility: section count similarity
        secs_a = data_a.get('sections', [])
        secs_b = data_b.get('sections', [])
        n_a, n_b = max(len(secs_a), 1), max(len(secs_b), 1)
        structure_compat = float(np.clip(1.0 - abs(n_a - n_b) / max(n_a, n_b), 0, 1))

        overall = float(np.clip(
            tempo_compat * 0.35 + key_compat * 0.35 +
            timbre_compat * 0.15 + range_compat * 0.10 + 0.05,
            0, 1))
        blend = float(np.clip(key_compat * 0.6 + tempo_compat * 0.4, 0, 1))

        return {
            'overall_score': overall,
            'key_compatibility': key_compat,
            'tempo_compatibility': tempo_compat,
            'range_compatibility': range_compat,
            'timbre_compatibility': timbre_compat,
            'structure_compatibility': structure_compat,
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

        mix_id = str(uuid.uuid4())
        mix_dir = self.base_dir / "mixes" / f"{song_a_id}_{song_b_id}"
        mix_dir.mkdir(parents=True, exist_ok=True)

        # Load song embeddings for learning — compute from stems if not cached yet
        emb_a = self.embedder.load_cached(song_a_id)
        if emb_a is None:
            emb_a = self.embedder.embed_song(song_a_id, stems_a)
        emb_b = self.embedder.load_cached(song_b_id)
        if emb_b is None:
            emb_b = self.embedder.embed_song(song_b_id, stems_b)

        # Ask predictor for full parameter suggestions
        predicted_params = self.predictor.suggest_params(
            song_a_id, song_b_id, emb_a=emb_a, emb_b=emb_b)

        print("Step 2: AI mixing (v10.0)...")
        mix_result = self.mixing_engine.create_mix(
            stems_a, stems_b, {}, {}, predicted_params=predicted_params)

        print("Step 3: Saving output...")
        quality_scores = mix_result.pop('quality_scores', None)
        params_used = mix_result.pop('params_used', {})

        # Save mix params so the predictor can learn from user rating
        if emb_a is not None and emb_b is not None:
            try:
                self.predictor.save_mix_params(
                    mix_id=mix_id,
                    song_a=song_a_id,
                    song_b=song_b_id,
                    direction=params_used.get('direction', 'a_vocals'),
                    key_shift=params_used.get('key_shift', 0),
                    vox_rms=params_used.get('vox_rms', 0.14),
                    inst_rms=params_used.get('inst_rms', 0.05),
                    emb_a=emb_a,
                    emb_b=emb_b,
                    duck_depth=predicted_params.get('duck_depth', 0.3),
                    comp_threshold=predicted_params.get('comp_threshold', -18.0),
                )
                # Auto-rate using the AI quality score so the predictor learns
                # from every mix, not just the ones the user manually rates.
                # User ratings (1-5 stars) will override this if they rate later.
                if quality_scores is not None:
                    qs = safe_serialize(quality_scores)
                    overall = qs.get('overall') if isinstance(qs, dict) else None
                    if overall is not None:
                        auto_rating = max(1, min(5, round(float(overall) * 4 + 1)))
                        self.predictor.add_rating(mix_id, auto_rating)
                        print(f"  Auto-rated mix: {auto_rating}/5 (quality={float(overall):.2f})")
            except Exception as e:
                print(f"  Warning: Could not save mix params: {e}")

        for name, audio in mix_result.items():
            if audio is not None and isinstance(audio, np.ndarray) and len(audio) > 0:
                path = mix_dir / f"{name}.wav"
                sf.write(str(path), audio, self.sr)
                print(f"  {name}.wav ({len(audio)/self.sr:.1f}s)")

        report = {
            'mix_id': mix_id,
            'song_a_id': song_a_id,
            'song_b_id': song_b_id,
            'params_used': params_used,
            'quality_scores': safe_serialize(quality_scores),
            'created_at': datetime.now().isoformat(),
        }
        with open(mix_dir / "fusion_report.json", 'w') as f:
            json.dump(report, f, indent=2, default=str)

        print(f"\nDone! Output: {mix_dir}")
        return {
            'mix_id': mix_id,
            'song_a_id': song_a_id,
            'song_b_id': song_b_id,
            'quality_scores': safe_serialize(quality_scores),
            'paths': {'mixes': str(mix_dir)},
        }

    # ------------------------------------------------------------------
    # AUTOMATION HELPERS
    # ------------------------------------------------------------------

    def _list_analyzed_ids(self):
        """Return all song_ids that have been fully analyzed."""
        analysis_dir = self.base_dir / "analysis"
        if not analysis_dir.exists():
            return []
        return [d.name for d in sorted(analysis_dir.iterdir())
                if d.is_dir() and (d / "analysis.json").exists()]

    def find_best_match(self, song_id):
        """Return the song_id whose MERT embedding is most similar to this one."""
        emb = self.embedder.load_cached(song_id)
        if emb is None:
            return None
        best_id, best_sim = None, -1.0
        for sid in self._list_analyzed_ids():
            if sid == song_id:
                continue
            other = self.embedder.load_cached(sid)
            if other is None:
                continue
            sim = float(np.dot(emb, other))   # both L2-normalised
            if sim > best_sim:
                best_sim, best_id = sim, sid
        if best_id:
            print(f"  Best match for {song_id}: {best_id} (similarity={best_sim:.3f})")
        return best_id

    def select_next_song(self, current_id: str):
        """
        Return the most compatible song ID for current_id.
        Score = 0.5 * embedding_similarity + 0.3 * key_compat + 0.2 * tempo_compat.
        """
        candidates = [sid for sid in self._list_analyzed_ids() if sid != current_id]
        if not candidates:
            return None

        emb_cur = self.embedder.load_cached(current_id)
        best_id, best_score = None, -1.0

        for sid in candidates:
            score = 0.0
            if emb_cur is not None:
                emb_sid = self.embedder.load_cached(sid)
                if emb_sid is not None:
                    score += 0.5 * float(np.dot(emb_cur, emb_sid))
            try:
                compat = self.analyze_compatibility(current_id, sid)
                score += 0.3 * compat['key_compatibility']
                score += 0.2 * compat['tempo_compatibility']
            except Exception:
                score += 0.25
            if score > best_score:
                best_score, best_id = score, sid

        if best_id:
            print(f"  Best next song: {best_id} (score={best_score:.3f})")
        return best_id

    def dj_session(self, start_id: str, count: int = 5, learn: bool = False):
        """
        Autonomous DJ loop: select next song → fuse → prompt for rating → repeat.
        With --learn, retrains the predictor after each rating.
        """
        import time
        current_id = start_id
        print(f"\n{'='*60}")
        print(f"  AUTO DJ SESSION — {count} mix(es) from: {start_id}")
        print(f"  Learning mode: {'ON' if learn else 'OFF'}")
        print(f"{'='*60}")

        for i in range(count):
            print(f"\n[Mix {i+1}/{count}] Current: {current_id}")
            next_id = self.select_next_song(current_id)
            if next_id is None:
                print("  No compatible songs found. Session complete.")
                break

            print(f"  Next: {next_id}")
            try:
                result = self.fuse_two_songs(current_id, next_id)
                mix_dir = self.base_dir / "mixes" / f"{current_id}_{next_id}"
                print(f"\n  Output: {mix_dir / 'full_mix.wav'}")

                # Rating prompt
                try:
                    raw = input("\n  Rate this mix 1-5 (Enter to skip): ").strip()
                    if raw.isdigit():
                        rating = max(1, min(5, int(raw)))
                        self.predictor.add_rating(result['mix_id'], rating)
                        if learn and self.predictor.get_ratings_count() >= 5:
                            print("  Retraining predictor on new rating...")
                            self.predictor.train()
                except (EOFError, KeyboardInterrupt):
                    pass

            except Exception as e:
                import traceback
                print(f"  Error: {e}")
                traceback.print_exc()

            current_id = next_id

        print(f"\n{'='*60}")
        print(f"  DJ session complete ({i+1} mix(es)).")
        print(f"{'='*60}\n")

    def _find_top_pairs(self, song_ids, top_n=3):
        """Return top_n (a, b, similarity) pairs by MERT cosine similarity."""
        embeddings = {}
        for sid in song_ids:
            emb = self.embedder.load_cached(sid)
            if emb is not None:
                embeddings[sid] = emb
        ids = list(embeddings.keys())
        pairs = []
        for i in range(len(ids)):
            for j in range(i + 1, len(ids)):
                a, b = ids[i], ids[j]
                sim = float(np.dot(embeddings[a], embeddings[b]))
                pairs.append((a, b, sim))
        pairs.sort(key=lambda x: x[2], reverse=True)
        return pairs[:top_n]

    def batch_process(self, folder, auto_fuse=True, top_n=3):
        """
        Analyze every audio file in folder, then fuse the top_n most
        MERT-compatible pairs.  Already-analyzed songs are skipped.
        """
        import time
        folder = Path(folder)
        AUDIO_EXTS = {'.mp3', '.wav', '.flac', '.m4a', '.aac', '.ogg'}
        files = sorted(f for f in folder.iterdir()
                       if f.suffix.lower() in AUDIO_EXTS)
        if not files:
            print(f"No audio files found in {folder}")
            return

        print(f"\nBatch: found {len(files)} audio file(s) in {folder}")

        analyzed = []
        for f in files:
            song_id = f.stem
            if (self.base_dir / "analysis" / song_id / "analysis.json").exists():
                print(f"  Skipping {song_id} (already analyzed)")
                analyzed.append(song_id)
                continue
            try:
                self.process_single_song(f, song_id)
                analyzed.append(song_id)
            except Exception as e:
                print(f"  Error analyzing {f.name}: {e}")

        if not auto_fuse or len(analyzed) < 2:
            print(f"\nBatch analysis complete ({len(analyzed)} songs).")
            return

        print(f"\nFinding top {top_n} compatible pairs from {len(analyzed)} songs...")
        pairs = self._find_top_pairs(analyzed, top_n)
        if not pairs:
            print("  Not enough MERT embeddings to rank pairs yet.")
            return

        for a, b, sim in pairs:
            print(f"\nAuto-fusing: {a} + {b}  (MERT similarity={sim:.3f})")
            try:
                self.fuse_two_songs(a, b)
            except Exception as e:
                print(f"  Error: {e}")

        print(f"\nBatch complete. Mixes saved to {self.base_dir / 'mixes'}/")

    def watch_folder(self, watch_dir, auto_fuse=True, interval=5,
                     stop_event=None):
        """
        Watch a folder for new audio files.  Each new file is automatically
        analyzed, and (if auto_fuse) fused with its best MERT match.

        Runs until Ctrl+C or stop_event is set.
        """
        import time
        watch_dir = Path(watch_dir)
        watch_dir.mkdir(parents=True, exist_ok=True)
        AUDIO_EXTS = {'.mp3', '.wav', '.flac', '.m4a', '.aac', '.ogg'}

        # Treat already-analyzed songs as "seen" so they're not re-processed
        seen = set(self._list_analyzed_ids())
        # Also track filenames already picked up this session
        seen_files = set(f.name for f in watch_dir.iterdir()
                         if f.suffix.lower() in AUDIO_EXTS
                         and f.stem in seen)

        print(f"\nWatch mode active — drop audio files into:")
        print(f"  {watch_dir.resolve()}")
        print(f"Auto-analyze: ON")
        print(f"Auto-fuse with best match: {'ON' if auto_fuse else 'OFF'}")
        if stop_event is None:
            print("Press Ctrl+C to stop.\n")

        while True:
            if stop_event is not None and stop_event.is_set():
                print("[watch] Stopped.")
                break
            try:
                for f in sorted(watch_dir.iterdir()):
                    if f.suffix.lower() not in AUDIO_EXTS:
                        continue
                    if f.name in seen_files:
                        continue
                    seen_files.add(f.name)
                    song_id = f.stem

                    print(f"\n[watch] New file: {f.name}")
                    try:
                        self.process_single_song(f, song_id)
                        seen.add(song_id)

                        if auto_fuse:
                            match = self.find_best_match(song_id)
                            if match:
                                print(f"[watch] Auto-fusing {song_id} + {match}...")
                                self.fuse_two_songs(song_id, match)
                            else:
                                print(f"[watch] No match yet — need at least 2 songs.")
                    except Exception as e:
                        import traceback
                        print(f"[watch] Error: {e}")
                        traceback.print_exc()

                time.sleep(interval)

            except KeyboardInterrupt:
                print("\n[watch] Stopped.")
                break


# ============================================================================
# DJ SESSION STATE
# ============================================================================

class DJSession:
    """State machine for an auto-DJ session running in a background thread."""

    def __init__(self, session_id, start_song, count, learn):
        self.session_id = session_id
        self.start_song = start_song
        self.count = count
        self.learn = learn
        self.mix_n = 0
        self.status = 'starting'   # starting|selecting|mixing|waiting_rating|complete|stopped|error
        self.current_song = start_song
        self.next_song = None
        self.mix_id = None
        self.download_url = None
        self.quality_scores = None
        self.error = None
        self.stop_event = threading.Event()
        self._rating_event = threading.Event()
        self._pending_rating = None

    def to_dict(self):
        return {
            'session_id':   self.session_id,
            'status':       self.status,
            'mix_n':        self.mix_n,
            'total':        self.count,
            'current_song': self.current_song,
            'next_song':    self.next_song,
            'mix_id':       self.mix_id,
            'download_url': self.download_url,
            'quality_scores': safe_serialize(self.quality_scores),
            'error':        self.error,
        }


dj_sessions: Dict[str, DJSession] = {}


def _run_dj_session(vf_engine, session: DJSession):
    """Background thread — drives the DJ loop."""
    try:
        for i in range(session.count):
            if session.stop_event.is_set():
                break

            session.mix_n = i + 1
            session.status = 'selecting'
            print(f"\n[DJ] Mix {i+1}/{session.count}: selecting next song...")

            next_id = vf_engine.select_next_song(session.current_song)
            if next_id is None:
                session.error = 'No more compatible songs found.'
                break

            session.next_song = next_id
            session.status = 'mixing'
            print(f"[DJ] Mixing: {session.current_song} → {next_id}")

            try:
                result = vf_engine.fuse_two_songs(session.current_song, next_id)
                mix_dir = vf_engine.base_dir / 'mixes' / f'{session.current_song}_{next_id}'
                session.mix_id = result.get('mix_id')
                session.download_url = (
                    f'/download/{session.current_song}_{next_id}'
                    if (mix_dir / 'full_mix.wav').exists() else None)
                session.quality_scores = result.get('quality_scores')
            except Exception as e:
                import traceback; traceback.print_exc()
                session.status = 'error'
                session.error = str(e)
                break

            session.status = 'waiting_rating'
            print(f"[DJ] Mix ready — waiting for rating (5-min window)...")
            session._rating_event.clear()
            session._rating_event.wait(timeout=300)

            if session._pending_rating is not None:
                try:
                    vf_engine.predictor.add_rating(session.mix_id, session._pending_rating)
                    if session.learn and vf_engine.predictor.get_ratings_count() >= 5:
                        print("[DJ] Retraining predictor...")
                        vf_engine.predictor.train()
                except Exception:
                    pass
                session._pending_rating = None

            if session.stop_event.is_set():
                break

            session.current_song = next_id

        if not session.stop_event.is_set() and session.status not in ('error',):
            session.status = 'complete'
            print(f"\n[DJ] Session complete — {session.mix_n} mix(es) created.")
    except Exception as e:
        import traceback; traceback.print_exc()
        session.status = 'error'
        session.error = str(e)


# ============================================================================
# WEB APP
# ============================================================================

web_app = Flask(__name__, template_folder='templates')
engine = None
active_jobs = {}
_watch_thread = None
_watch_stop = None


def _cleanup_old_jobs():
    """Remove completed/failed jobs older than 10 minutes to prevent memory leak"""
    import time
    cutoff = time.time() - 600
    stale = [jid for jid, j in active_jobs.items()
             if j.get('status') in ('completed', 'failed')
             and j.get('completed_at', float('inf')) < cutoff]
    for jid in stale:
        del active_jobs[jid]



@web_app.route('/')
def index():
    return render_template('index.html')


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
    _cleanup_old_jobs()

    def bg():
        import time
        try:
            active_jobs[job_id]['progress'] = 10
            result = engine.process_single_song(temp_path, song_id)
            active_jobs[job_id].update({
                'status': 'completed', 'progress': 100,
                'result': {'song_id': result['song_id']},
                'completed_at': time.time(),
            })
        except Exception as e:
            import traceback
            active_jobs[job_id].update({'status': 'failed', 'error': str(e),
                                        'completed_at': time.time()})
            print(traceback.format_exc())
        finally:
            try:
                temp_path.unlink(missing_ok=True)
            except Exception:
                pass

    threading.Thread(target=bg, daemon=True).start()
    return jsonify({'success': True, 'job_id': job_id, 'song_id': song_id})


@web_app.route('/api/songs/<path:song_id>', methods=['DELETE'])
def delete_song(song_id):
    import shutil
    deleted = []
    errors = []
    for subdir in ['stems', 'analysis']:
        p = Path(DATA_DIR) / subdir / song_id
        if p.exists():
            try:
                shutil.rmtree(str(p))
                deleted.append(str(p))
            except Exception as e:
                errors.append(str(e))
    # Remove embedding files (both old .npy and new _mert.npy)
    emb_dir = Path(DATA_DIR) / 'embeddings'
    for emb_file in [emb_dir / f"{song_id}.npy",
                     emb_dir / f"{song_id}_mert.npy"]:
        if emb_file.exists():
            try:
                emb_file.unlink()
                deleted.append(str(emb_file))
            except Exception as e:
                errors.append(str(e))
    if errors:
        return jsonify({'success': False, 'error': '; '.join(errors)}), 500
    return jsonify({'success': True, 'deleted': deleted})


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
    _cleanup_old_jobs()
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
    """Start a fuse job in the background and return job_id immediately."""
    import time
    job_id = str(uuid.uuid4())
    active_jobs[job_id] = {'status': 'processing', 'progress': 10,
                            'type': 'fuse', 'song_a': song_a, 'song_b': song_b}
    _cleanup_old_jobs()

    def bg():
        try:
            result = engine.fuse_two_songs(song_a, song_b)
            mix_file = Path(DATA_DIR) / "mixes" / f"{song_a}_{song_b}" / "full_mix.wav"
            url = f"/download/{song_a}_{song_b}" if mix_file.exists() else None
            active_jobs[job_id].update({
                'status': 'completed', 'progress': 100,
                'completed_at': time.time(),
                'result': {
                    'download_url': url,
                    'quality_scores': result.get('quality_scores'),
                    'mix_id': result.get('mix_id'),
                    'ratings_count': engine.predictor.get_ratings_count(),
                },
            })
        except Exception as e:
            import traceback
            traceback.print_exc()
            active_jobs[job_id].update({'status': 'failed', 'error': str(e),
                                        'completed_at': time.time()})

    threading.Thread(target=bg, daemon=True).start()
    return jsonify({'success': True, 'job_id': job_id})


@web_app.route('/api/rate', methods=['POST'])
def rate_mix():
    try:
        data = request.get_json()
        if not data:
            return jsonify({'success': False, 'error': 'No JSON body'}), 400
        mix_id = data.get('mix_id')
        rating = data.get('rating')
        if not mix_id or rating is None:
            return jsonify({'success': False, 'error': 'mix_id and rating required'}), 400
        ok = engine.predictor.add_rating(mix_id, int(rating))
        count = engine.predictor.get_ratings_count()
        return jsonify({
            'success': ok,
            'ratings_count': count,
            'message': f'Rating saved. {count} total ratings.',
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


@web_app.route('/api/stream')
def stream_logs():
    """SSE endpoint: streams all print() output to the browser in real time."""
    from flask import Response, stream_with_context

    sub_queue = _broadcaster.subscribe()

    def generate():
        try:
            yield "data: Connected to VocalFusion AI log stream\n\n"
            while True:
                try:
                    line = sub_queue.get(timeout=15)
                    if line:
                        yield f"data: {line}\n\n"
                except queue.Empty:
                    yield ": keepalive\n\n"   # SSE comment keeps connection alive
        except GeneratorExit:
            pass
        finally:
            _broadcaster.unsubscribe(sub_queue)

    return Response(
        stream_with_context(generate()),
        mimetype='text/event-stream',
        headers={
            'Cache-Control': 'no-cache',
            'X-Accel-Buffering': 'no',
        },
    )


@web_app.route('/api/mixes')
def list_mixes():
    """Return all completed mixes with quality scores."""
    mixes_dir = Path(DATA_DIR) / "mixes"
    mixes = []
    if mixes_dir.exists():
        for d in sorted(mixes_dir.iterdir(), reverse=True):
            report = d / "fusion_report.json"
            if not (d.is_dir() and report.exists()):
                continue
            try:
                with open(report) as f:
                    r = json.load(f)
                qs = r.get('quality_scores') or {}
                overall = qs.get('overall', 0) if isinstance(qs, dict) else 0
                mix_file = d / "full_mix.wav"
                mixes.append({
                    'mix_id': r.get('mix_id'),
                    'song_a': r.get('song_a_id'),
                    'song_b': r.get('song_b_id'),
                    'overall': float(overall),
                    'created_at': r.get('created_at'),
                    'download_url': f"/download/{d.name}" if mix_file.exists() else None,
                })
            except Exception:
                pass
    return jsonify({'success': True, 'mixes': mixes})


@web_app.route('/api/watch/status')
def watch_status():
    active = _watch_thread is not None and _watch_thread.is_alive()
    folder = str(Path(DATA_DIR) / "watch")
    return jsonify({'active': active, 'folder': folder})


@web_app.route('/api/watch/toggle', methods=['POST'])
def watch_toggle():
    global _watch_thread, _watch_stop
    if _watch_thread is not None and _watch_thread.is_alive():
        _watch_stop.set()
        _watch_thread.join(timeout=3)
        _watch_thread = None
        _watch_stop = None
        return jsonify({'active': False})
    else:
        _watch_stop = threading.Event()
        watch_dir = Path(DATA_DIR) / "watch"

        def _run():
            engine.watch_folder(watch_dir, stop_event=_watch_stop)

        _watch_thread = threading.Thread(target=_run, daemon=True)
        _watch_thread.start()
        return jsonify({'active': True, 'folder': str(watch_dir.resolve())})


# ============================================================================
# DJ SESSION ROUTES
# ============================================================================

@web_app.route('/api/dj/start', methods=['POST'])
def dj_start():
    global engine
    data = request.get_json() or {}
    start_song = data.get('start_song')
    if not start_song:
        return jsonify({'success': False, 'error': 'start_song required'}), 400
    count = max(1, min(20, int(data.get('count', 5))))
    learn = bool(data.get('learn', False))
    session_id = str(uuid.uuid4())
    session = DJSession(session_id, start_song, count, learn)
    dj_sessions[session_id] = session
    threading.Thread(target=_run_dj_session, args=(engine, session),
                     daemon=True).start()
    return jsonify({'success': True, 'session_id': session_id})


@web_app.route('/api/dj/<session_id>')
def dj_status(session_id):
    session = dj_sessions.get(session_id)
    if not session:
        return jsonify({'success': False, 'error': 'Session not found'}), 404
    return jsonify({'success': True, 'session': session.to_dict()})


@web_app.route('/api/dj/<session_id>/rate', methods=['POST'])
def dj_rate_mix(session_id):
    session = dj_sessions.get(session_id)
    if not session:
        return jsonify({'success': False, 'error': 'Session not found'}), 404
    data = request.get_json() or {}
    rating = data.get('rating')
    if rating is not None:
        session._pending_rating = max(1, min(5, int(rating)))
    session._rating_event.set()
    return jsonify({'success': True})


@web_app.route('/api/dj/<session_id>/skip', methods=['POST'])
def dj_skip_mix(session_id):
    session = dj_sessions.get(session_id)
    if not session:
        return jsonify({'success': False, 'error': 'Session not found'}), 404
    session._rating_event.set()
    return jsonify({'success': True})


@web_app.route('/api/dj/<session_id>/stop', methods=['POST'])
def dj_stop_session(session_id):
    session = dj_sessions.get(session_id)
    if not session:
        return jsonify({'success': False, 'error': 'Session not found'}), 404
    session.stop_event.set()
    session._rating_event.set()
    session.status = 'stopped'
    return jsonify({'success': True})


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

    p = sub.add_parser('watch',
        help='Watch a folder — auto-analyze new files and fuse best pairs')
    p.add_argument('folder', nargs='?', default='vf_data/watch',
                   help='Folder to watch (default: vf_data/watch)')
    p.add_argument('--no-fuse', action='store_true',
                   help='Analyze only, do not auto-fuse')
    p.add_argument('--interval', type=int, default=5,
                   help='Polling interval in seconds (default: 5)')

    p = sub.add_parser('batch',
        help='Analyze all songs in a folder, then fuse the most compatible pairs')
    p.add_argument('folder', help='Folder containing audio files')
    p.add_argument('--top', type=int, default=3,
                   help='How many pairs to fuse (default: 3)')
    p.add_argument('--no-fuse', action='store_true',
                   help='Analyze only, do not auto-fuse')

    p = sub.add_parser('dj',
        help='Autonomous DJ session: auto-select songs, fuse, rate, repeat')
    p.add_argument('start_song', help='Song ID to start the session from')
    p.add_argument('--count', type=int, default=5,
                   help='Number of mixes to create (default: 5)')
    p.add_argument('--learn', action='store_true',
                   help='Retrain predictor in real-time after each rating')

    args = parser.parse_args()
    ir = getattr(args, 'ir', None)

    if args.cmd == 'serve':
        engine = VocalFusion(ir_path=ir)
        _broadcaster.install()   # capture all print() → SSE stream
        print(f"\nWeb UI: http://{args.host}:{args.port}\n")
        web_app.run(host=args.host, port=args.port, debug=False,
                    threaded=True)

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

    elif args.cmd == 'watch':
        engine = VocalFusion(ir_path=ir)
        engine.watch_folder(
            watch_dir=args.folder,
            auto_fuse=not args.no_fuse,
            interval=args.interval,
        )

    elif args.cmd == 'batch':
        engine = VocalFusion(ir_path=ir)
        engine.batch_process(
            folder=args.folder,
            auto_fuse=not args.no_fuse,
            top_n=args.top,
        )

    elif args.cmd == 'dj':
        engine = VocalFusion(ir_path=ir)
        engine.dj_session(
            start_id=args.start_song,
            count=args.count,
            learn=args.learn,
        )

    else:
        parser.print_help()


if __name__ == "__main__":
    main()
