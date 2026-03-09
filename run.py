"""
VocalFusion — Flask Server
==========================
GET  /              → UI
POST /fuse          → multipart: song_a, song_b → {job_id}
GET  /status/<id>   → {status, progress, message, output_url?}
GET  /output/<file> → serve completed WAV
"""

import builtins
import os
import re
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

_STEP_RE = re.compile(r"\[(\d+)/8\]")


def _run_fuse(job_id: str, path_a: str, path_b: str, out_path: str) -> None:
    orig_print = builtins.print

    def _capture(*args, **kwargs):
        orig_print(*args, **kwargs)
        msg = " ".join(str(a) for a in args)
        m = _STEP_RE.search(msg)
        with _lock:
            _jobs[job_id]["message"] = msg.strip()
            if m:
                _jobs[job_id]["progress"] = int(int(m.group(1)) / 8 * 100)

    builtins.print = _capture
    try:
        fuse(path_a, path_b, out_path, stems_cache=str(STEMS_DIR))
        with _lock:
            _jobs[job_id].update(
                status="done", progress=100, message="Complete",
                output_url=f"/output/{Path(out_path).name}"
            )
    except Exception as exc:
        with _lock:
            _jobs[job_id].update(status="error", message=str(exc))
    finally:
        builtins.print = orig_print


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
        _jobs[job_id] = {"status": "running", "progress": 0, "message": "Starting…"}

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


@app.route("/output/<filename>")
def output_file(filename: str):
    return send_from_directory(str(OUTPUT_DIR.resolve()), filename)


if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--port", type=int, default=5000)
    args = p.parse_args()
    app.run(host="0.0.0.0", port=args.port, debug=False)
