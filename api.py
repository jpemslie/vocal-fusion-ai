"""
VocalFusion REST API v1
=======================
All endpoints require: Authorization: Bearer <api_key>

POST /api/v1/fuse          — multipart: song_a, song_b → {job_id, status_url, share_url}
GET  /api/v1/status/<id>   — {status, progress, message, variants?, score?, share_url?}
GET  /api/v1/keys          — list your key's metadata {key, name, created_at, usage_count}
POST /api/v1/keys          — generate new key: body {name: str} → {key, name, created_at}
DELETE /api/v1/keys/<key>  — revoke a key
"""

import json
import secrets
import threading
import uuid
from datetime import datetime
from pathlib import Path

from flask import Blueprint, g, jsonify, render_template, request

# Shared state from run.py — imported at request time to avoid circular import issues
# (run.py imports api_bp from this module, so we do lazy imports inside handlers)

KEYS_FILE = Path("vf_data/api_keys.json")
_keys_lock = threading.Lock()

api_bp = Blueprint("api_v1", __name__, url_prefix="/api/v1")

# ---------------------------------------------------------------------------
# Key file helpers
# ---------------------------------------------------------------------------

def _load_keys() -> dict:
    """Read keys file. Must be called with _keys_lock held."""
    if not KEYS_FILE.exists():
        return {}
    with open(KEYS_FILE, "r") as fh:
        return json.load(fh)


def _save_keys(keys: dict) -> None:
    """Write keys file. Must be called with _keys_lock held."""
    KEYS_FILE.parent.mkdir(parents=True, exist_ok=True)
    with open(KEYS_FILE, "w") as fh:
        json.dump(keys, fh, indent=2)


def _ensure_default_key() -> None:
    """Create keys file with a default key if it doesn't exist."""
    with _keys_lock:
        if KEYS_FILE.exists():
            return
        key = "vf_" + secrets.token_hex(16)
        keys = {
            key: {
                "name": "default",
                "created_at": datetime.utcnow().isoformat() + "Z",
                "usage_count": 0,
                "active": True,
            }
        }
        _save_keys(keys)
        print(f"[VocalFusion API] Default API key created: {key}")
        print("[VocalFusion API] Store this key — it will not be shown again.")


# Call on module load so the key exists before the first request
_ensure_default_key()


# ---------------------------------------------------------------------------
# Auth middleware
# ---------------------------------------------------------------------------

def _error(message: str, code: str, status: int):
    return jsonify({"error": message, "code": code}), status


@api_bp.before_request
def _authenticate():
    """Require Bearer token on all routes except the docs root."""
    # Skip auth for the docs page
    if request.endpoint == "api_v1.docs":
        return None

    auth_header = request.headers.get("Authorization", "")
    if not auth_header.startswith("Bearer "):
        return _error("Missing or malformed Authorization header", "UNAUTHORIZED", 401)

    provided_key = auth_header[len("Bearer "):].strip()

    with _keys_lock:
        keys = _load_keys()
        key_data = keys.get(provided_key)
        if not key_data or not key_data.get("active", False):
            return _error("Invalid or revoked API key", "UNAUTHORIZED", 401)
        # Increment usage count
        key_data["usage_count"] = key_data.get("usage_count", 0) + 1
        keys[provided_key] = key_data
        _save_keys(keys)

    # Attach key info to request context
    g.api_key = provided_key
    g.api_key_data = dict(key_data)
    return None


# ---------------------------------------------------------------------------
# Docs (no auth)
# ---------------------------------------------------------------------------

@api_bp.route("/", methods=["GET"])
def docs():
    return render_template("api_docs.html")


# ---------------------------------------------------------------------------
# Key management endpoints
# ---------------------------------------------------------------------------

@api_bp.route("/keys", methods=["GET"])
def get_key():
    """Return the authenticated key's own metadata."""
    return jsonify({
        "key": g.api_key,
        "name": g.api_key_data.get("name"),
        "created_at": g.api_key_data.get("created_at"),
        "usage_count": g.api_key_data.get("usage_count"),
        "active": g.api_key_data.get("active"),
    })


@api_bp.route("/keys", methods=["POST"])
def create_key():
    """Generate a new API key. Requires an existing valid key."""
    body = request.get_json(silent=True) or {}
    name = body.get("name", "").strip()
    if not name:
        return _error("Request body must include a non-empty 'name' field", "BAD_REQUEST", 400)

    new_key = "vf_" + secrets.token_hex(16)
    created_at = datetime.utcnow().isoformat() + "Z"
    new_key_data = {
        "name": name,
        "created_at": created_at,
        "usage_count": 0,
        "active": True,
    }

    with _keys_lock:
        keys = _load_keys()
        keys[new_key] = new_key_data
        _save_keys(keys)

    return jsonify({
        "key": new_key,
        "name": name,
        "created_at": created_at,
    }), 201


@api_bp.route("/keys/<key_to_revoke>", methods=["DELETE"])
def revoke_key(key_to_revoke: str):
    """Revoke a key. Users can only revoke their own key."""
    if key_to_revoke != g.api_key:
        return _error("You can only revoke your own API key", "UNAUTHORIZED", 401)

    with _keys_lock:
        keys = _load_keys()
        if key_to_revoke not in keys:
            return _error("Key not found", "NOT_FOUND", 404)
        keys[key_to_revoke]["active"] = False
        _save_keys(keys)

    return jsonify({"revoked": True, "key": key_to_revoke})


# ---------------------------------------------------------------------------
# Fuse endpoint
# ---------------------------------------------------------------------------

@api_bp.route("/fuse", methods=["POST"])
def api_fuse():
    from run import _jobs, _lock, _run_fuse, UPLOAD_DIR, OUTPUT_DIR

    file_a = request.files.get("song_a")
    file_b = request.files.get("song_b")
    if not file_a or not file_b:
        return _error("Upload both song_a and song_b", "BAD_REQUEST", 400)

    # Rate limit: max 3 concurrent running jobs per API key
    api_key = g.api_key
    with _lock:
        running_for_key = sum(
            1 for job in _jobs.values()
            if job.get("status") == "running" and job.get("api_key") == api_key
        )
    if running_for_key >= 3:
        return _error(
            "Too many concurrent jobs for this API key (max 3)",
            "RATE_LIMITED",
            429,
        )

    job_id = uuid.uuid4().hex[:8]
    path_a = str(UPLOAD_DIR / f"{job_id}_a{Path(file_a.filename).suffix}")
    path_b = str(UPLOAD_DIR / f"{job_id}_b{Path(file_b.filename).suffix}")
    file_a.save(path_a)
    file_b.save(path_b)

    out_path = str(OUTPUT_DIR / f"{job_id}_fusion.wav")

    with _lock:
        _jobs[job_id] = {
            "status": "running",
            "progress": 0,
            "message": "Starting…",
            "name_a": Path(file_a.filename).stem,
            "name_b": Path(file_b.filename).stem,
            "api_key": api_key,
        }

    import threading as _threading
    t = _threading.Thread(
        target=_run_fuse, args=(job_id, path_a, path_b, out_path), daemon=True
    )
    t.start()

    return jsonify({
        "job_id": job_id,
        "status_url": f"/api/v1/status/{job_id}",
        "share_url": f"/share/{job_id}",
    }), 202


# ---------------------------------------------------------------------------
# Status endpoint
# ---------------------------------------------------------------------------

@api_bp.route("/status/<job_id>", methods=["GET"])
def api_status(job_id: str):
    from run import _jobs, _lock

    with _lock:
        job = _jobs.get(job_id)

    if job is None:
        return _error("Unknown job_id", "NOT_FOUND", 404)

    response = dict(job)
    # Remove internal field not meant for API consumers
    response.pop("api_key", None)
    response["share_url"] = f"/share/{job_id}"
    response["api_version"] = "v1"

    return jsonify(response), 200
