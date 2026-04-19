#!/usr/bin/env python3
"""
test_professional_fuse.py
Assert that the professional optimization loop produces chart_score >= 72.

Usage:
  python test_professional_fuse.py <vocals.wav> <beat.wav>
"""
import sys, time, requests

BASE = "http://localhost:8000"


def main():
    if len(sys.argv) < 3:
        print("Usage: python test_professional_fuse.py <vocals.wav> <beat.wav>")
        sys.exit(1)

    vocal_path = sys.argv[1]
    beat_path  = sys.argv[2]

    print(f"[test] uploading: {vocal_path}  +  {beat_path}")
    with open(vocal_path, "rb") as va, open(beat_path, "rb") as vb:
        r = requests.post(
            f"{BASE}/fuse",
            files={"song_a": ("vocals.wav", va), "song_b": ("beat.wav", vb)},
            data={"pro_mode": "true", "seed": "42"},
            timeout=30,
        )
    r.raise_for_status()
    job_id = r.json()["job_id"]
    print(f"[test] job_id = {job_id}")

    # Poll until done (max 10 minutes)
    deadline = time.time() + 600
    while time.time() < deadline:
        status = requests.get(f"{BASE}/status/{job_id}", timeout=10).json()
        pct = status.get("progress", 0)
        msg = status.get("message", "")
        print(f"  [{status['status']}] {pct:.0f}%  {msg}", end="\r", flush=True)

        if status["status"] == "done":
            print()
            chart_score = status.get("chart_score", 0)
            chart_grade = status.get("chart_grade", "?")
            attempts    = status.get("attempts", [])
            print(f"\n[test] Final chart score: {chart_score}/100 ({chart_grade})")
            if attempts:
                print(f"[test] Attempts: {len(attempts)}")
                for a in attempts:
                    marker = " <-- best" if a["chart_score"] == chart_score else ""
                    overrides = ", ".join(f"{k}={v:.2f}" for k, v in a["overrides"].items()) or "baseline"
                    print(f"       #{a['n']:2d}  {a['chart_score']:3d}/100 {a['chart_grade']}  [{overrides}]{marker}")

            if chart_score >= 72:
                print(f"\n[PASS] chart_score {chart_score} >= 72")
                sys.exit(0)
            else:
                print(f"\n[FAIL] chart_score {chart_score} < 72  (grade {chart_grade})")
                sys.exit(1)

        if status["status"] == "error":
            print(f"\n[FAIL] job error: {status.get('message', 'unknown')}")
            sys.exit(1)

        time.sleep(2)

    print("\n[FAIL] timeout waiting for job")
    sys.exit(1)


if __name__ == "__main__":
    main()
