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


def _run_fuse(job_id: str, path_a: str, path_b: str, out_path: str) -> None:
    def on_progress(step, total, msg):
        with _lock:
            _jobs[job_id]["progress"] = int(step / total * 100)
            _jobs[job_id]["message"] = msg

    try:
        result = fuse(path_a, path_b, out_path,
                      stems_cache=str(STEMS_DIR),
                      progress_cb=on_progress)

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
        }

    t = threading.Thread(
        target=_run_fuse, args=(job_id, path_a, path_b, out_path), daemon=True
    )
    t.start()
    return jsonify(job_id=job_id)


@app.route("/status/<job_id>")
def status(job_id: str):
    with _lock:
        job = _jobs.get(job_id)
    return (jsonify(job), 200) if job else (jsonify(error="Unknown job"), 404)


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
