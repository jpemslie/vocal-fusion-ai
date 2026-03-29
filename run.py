"""
VocalFusion — Flask Server
==========================
GET  /              → UI
POST /fuse          → multipart: song_a, song_b → {job_id}
GET  /status/<id>   → {status, progress, message, output_url?, variants?, score?}
GET  /output/<file> → serve completed WAV
"""

import os
import threading
import uuid
from pathlib import Path

from flask import Flask, jsonify, render_template, request, send_from_directory

from fuser import fuse

app = Flask(__name__)

UPLOAD_DIR = Path("vf_data/uploads")
OUTPUT_DIR = Path("vf_data/mixes")
STEMS_DIR  = Path("vf_data/stems")

for _d in (UPLOAD_DIR, OUTPUT_DIR, STEMS_DIR):
    _d.mkdir(parents=True, exist_ok=True)

_jobs: dict = {}
_lock = threading.Lock()


def _run_fuse(job_id: str, path_a: str, path_b: str, out_path: str,
              direct_vocal: bool = False) -> None:
    import time

    def on_progress(step, total, msg):
        with _lock:
            _jobs[job_id]["progress"] = int(step / total * 100)
            _jobs[job_id]["message"] = msg
            # Mark previous step done, current step running
            steps = _jobs[job_id]["steps"]
            for s in steps:
                if s["n"] < step and s["status"] == "running":
                    s["status"] = "done"
                    s["elapsed"] = round(time.time() - s.get("_started", time.time()), 1)
            # Find or create current step
            existing = next((s for s in steps if s["n"] == step), None)
            if not existing:
                steps.append({"n": step, "status": "running", "title": msg,
                               "details": {}, "_started": time.time(), "elapsed": None})
            else:
                existing["status"] = "running"
                existing["title"] = msg
                existing.setdefault("_started", time.time())

    def on_step_details(step_n, data):
        with _lock:
            steps = _jobs[job_id]["steps"]
            existing = next((s for s in steps if s["n"] == step_n), None)
            if existing:
                existing["details"] = data
            else:
                steps.append({"n": step_n, "status": "running", "title": "",
                               "details": data, "_started": __import__('time').time(), "elapsed": None})

    try:
        result = fuse(path_a, path_b, out_path,
                      stems_cache=str(STEMS_DIR),
                      progress_cb=on_progress,
                      step_details_cb=on_step_details,
                      direct_vocal=direct_vocal)

        # result is a dict {"radio": path, "club": path, "intimate": path}
        # or a plain string (backward compat)
        if isinstance(result, dict):
            radio_path    = result.get("radio", out_path)
            club_path     = result.get("club")
            intimate_path = result.get("intimate")
            score         = result.get("score")
        else:
            radio_path    = result or out_path
            club_path     = None
            intimate_path = None
            score         = None

        variants = {"radio": f"/output/{Path(radio_path).name}"}
        if club_path and Path(club_path).exists():
            variants["club"] = f"/output/{Path(club_path).name}"
        if intimate_path and Path(intimate_path).exists():
            variants["intimate"] = f"/output/{Path(intimate_path).name}"

        with _lock:
            # Mark all steps done on completion
            for s in _jobs[job_id]["steps"]:
                if s["status"] == "running":
                    s["status"] = "done"
            _jobs[job_id].update(
                status="done",
                progress=100,
                message="Complete",
                output_url=variants["radio"],   # default player src
                variants=variants,
                score=score,
                share_url=f"/share/{job_id}",
            )
    except Exception as exc:
        with _lock:
            _jobs[job_id].update(status="error", message=str(exc))


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/fuse", methods=["POST"])
def fuse_route():
    file_a = request.files.get("song_a")
    file_b = request.files.get("song_b")
    if not file_a or not file_b:
        return jsonify(error="Upload both song_a and song_b"), 400

    # direct_vocal=true means song_b is a clean phone recording (M4A/WAV/etc.)
    # — skips Demucs stem separation on the vocal track.
    direct_vocal = request.form.get("direct_vocal", "false").lower() in ("1", "true", "yes")

    job_id = uuid.uuid4().hex[:8]
    path_a = str(UPLOAD_DIR / f"{job_id}_a{Path(file_a.filename).suffix}")
    path_b = str(UPLOAD_DIR / f"{job_id}_b{Path(file_b.filename).suffix}")
    file_a.save(path_a)
    file_b.save(path_b)

    out_path = str(OUTPUT_DIR / f"{job_id}_fusion.wav")

    with _lock:
        _jobs[job_id] = {
            "status": "running", "progress": 0, "message": "Starting…",
            "name_a": Path(file_a.filename).stem,
            "name_b": Path(file_b.filename).stem,
            "direct_vocal": direct_vocal,
            "steps": [],
        }

    t = threading.Thread(
        target=_run_fuse, args=(job_id, path_a, path_b, out_path, direct_vocal), daemon=True
    )
    t.start()
    return jsonify(job_id=job_id)


@app.route("/status/<job_id>")
def status(job_id: str):
    with _lock:
        job = _jobs.get(job_id)
    if not job:
        return jsonify(error="Unknown job"), 404
    # Strip internal timing keys before sending to client
    import copy
    clean = copy.deepcopy(job)
    for s in clean.get("steps", []):
        s.pop("_started", None)
    return jsonify(clean), 200


@app.route("/share/<job_id>")
def share(job_id: str):
    with _lock:
        job = _jobs.get(job_id)
    if not job or job.get("status") != "done":
        return "Share link not ready or invalid.", 404
    return render_template("share.html", job_id=job_id, job=job)


@app.route("/output/<filename>")
def output_file(filename: str):
    return send_from_directory(str(OUTPUT_DIR.resolve()), filename)


from api import api_bp
app.register_blueprint(api_bp)


if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--port", type=int, default=5000)
    args = p.parse_args()
    print(f"API: http://0.0.0.0:{args.port}/api/v1/  (see /api/v1/keys for auth)")
    app.run(host="0.0.0.0", port=args.port, debug=False)
