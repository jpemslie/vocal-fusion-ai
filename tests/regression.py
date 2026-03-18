"""
VocalFusion Regression Test Suite — System 6
=============================================
Runs known-good song pairs and verifies they still score ≥ MIN_SCORE.
Fail fast on any regression. Run before committing fuser.py changes.

Usage:
    python tests/regression.py              # full run (uses cached stems)
    python tests/regression.py --fast       # score only (skip re-fuse, just re-score last output)
    python tests/regression.py --list       # list test cases
"""

import argparse
import os
import sys
import time
from pathlib import Path

# Make sure the project root is on the path
sys.path.insert(0, str(Path(__file__).parent.parent))

# ── Gold-standard test cases ──────────────────────────────────────────────────
# Add a new entry whenever you lock in a known-good combination.
# min_score: score that this pair must still reach after any code change.
TESTS = [
    {
        "name":      "John Summit + Travis Scott (v27 baseline)",
        "song_a":    "/Users/jpemslie/Downloads/John Summit - crystallized (feat. Inéz) Official Lyric Visualizer.mp3",
        "song_b":    "/Users/jpemslie/Downloads/Travis Scott - FE_N ft Playboi Carti.mp3",
        "out":       "vf_data/mixes/regression_john_travis.wav",
        "min_score": 95,
    },
]

STEMS_CACHE = "vf_data/stems"


def run_tests(fast: bool = False) -> bool:
    from fuser import fuse
    from listen import score_file, auto_score

    all_pass = True
    results   = []

    for test in TESTS:
        name = test["name"]

        # Check source files exist (skip gracefully if not)
        if not Path(test["song_a"]).exists() or not Path(test["song_b"]).exists():
            print(f"  SKIP  {name}: source files not on disk")
            continue

        print(f"\n{'─'*60}")
        print(f"  TEST: {name}")
        print(f"{'─'*60}")

        out = test["out"]

        if fast and Path(out).exists():
            print(f"  --fast: skipping fuse, re-scoring {out}")
        else:
            t0 = time.time()
            try:
                fuse(test["song_a"], test["song_b"], out, stems_cache=STEMS_CACHE)
                elapsed = time.time() - t0
                print(f"  Fuse completed in {elapsed/60:.1f} min")
            except Exception as e:
                print(f"  FAIL  {name}: fuse raised {e}")
                all_pass = False
                results.append({"name": name, "status": "ERROR", "score": 0,
                                 "min": test["min_score"], "error": str(e)})
                continue

        try:
            passed, score, summary = auto_score(out)[:3]
        except Exception as e:
            _, score, _ = score_file(out, print_report=True)
            passed = score >= test["min_score"]

        status = "PASS" if score >= test["min_score"] else "FAIL"
        print(f"  {status}  score={score}/100  min={test['min_score']}")
        if score < test["min_score"]:
            all_pass = False

        results.append({"name": name, "status": status,
                         "score": score, "min": test["min_score"]})

    # Summary table
    print(f"\n{'═'*60}")
    print(f"  REGRESSION RESULTS")
    print(f"{'═'*60}")
    for r in results:
        icon = "✓" if r["status"] == "PASS" else "✗"
        print(f"  {icon} {r['name']}: {r['score']}/100  (min {r['min']})")
    print(f"{'═'*60}")
    print(f"  {'ALL PASSED' if all_pass else 'REGRESSION DETECTED — do not merge'}")
    print()

    return all_pass


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--fast",  action="store_true",
                        help="Score existing output without re-fusing")
    parser.add_argument("--list",  action="store_true",
                        help="List test cases and exit")
    args = parser.parse_args()

    if args.list:
        for t in TESTS:
            print(f"  {t['name']}  (min {t['min_score']}/100)")
        sys.exit(0)

    ok = run_tests(fast=args.fast)
    sys.exit(0 if ok else 1)
