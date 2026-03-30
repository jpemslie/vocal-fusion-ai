"""
VocalFusion Regression Test Suite — System 6
=============================================
Runs known-good song pairs and verifies they still score ≥ MIN_SCORE.
Fail fast on any regression. Run before committing fuser.py changes.

Usage:
    python tests/regression.py              # full run (uses cached stems)
    python tests/regression.py --fast       # score only (skip re-fuse, just re-score last output)
    python tests/regression.py --list       # list test cases

Source files are resolved via environment variables so this test is portable
across machines. Set VF_SONG_A and VF_SONG_B (or VF_SONG_<N>_A / _B for
multiple pairs) to point at local MP3/WAV files before running.

Example:
    export VF_SONG_A="$HOME/Downloads/john_summit.mp3"
    export VF_SONG_B="$HOME/Downloads/travis_scott.mp3"
    python tests/regression.py
"""

import argparse
import os
import sys
import time
from pathlib import Path

# Make sure the project root is on the path
sys.path.insert(0, str(Path(__file__).parent.parent))

_ROOT = Path(__file__).parent.parent
_OUTPUT_DIR = Path(os.environ.get("VF_OUTPUT_DIR", _ROOT / "vf_data" / "mixes"))
STEMS_CACHE = str(Path(os.environ.get("VF_STEMS_DIR", _ROOT / "vf_data" / "stems")))


def _env_tests() -> list:
    """Build test list from VF_SONG_* environment variables.

    Simple pair: VF_SONG_A + VF_SONG_B (name defaults to "env pair")
    Multiple pairs: VF_SONG_1_A + VF_SONG_1_B, VF_SONG_2_A + VF_SONG_2_B, ...
    """
    tests = []

    # Single pair shorthand
    if os.environ.get("VF_SONG_A") and os.environ.get("VF_SONG_B"):
        tests.append({
            "name": os.environ.get("VF_REGRESSION_NAME", "env pair"),
            "song_a": os.environ["VF_SONG_A"],
            "song_b": os.environ["VF_SONG_B"],
            "out": str(_OUTPUT_DIR / "regression_env.wav"),
            "min_score": int(os.environ.get("VF_MIN_SCORE", "55")),
        })

    # Numbered pairs
    for i in range(1, 20):
        a = os.environ.get(f"VF_SONG_{i}_A")
        b = os.environ.get(f"VF_SONG_{i}_B")
        if not (a and b):
            break
        tests.append({
            "name": os.environ.get(f"VF_REGRESSION_NAME_{i}", f"pair {i}"),
            "song_a": a,
            "song_b": b,
            "out": str(_OUTPUT_DIR / f"regression_{i:02d}.wav"),
            "min_score": int(os.environ.get(f"VF_MIN_SCORE_{i}",
                              os.environ.get("VF_MIN_SCORE", "55"))),
        })

    return tests


# ── Gold-standard test cases ──────────────────────────────────────────────────
# DO NOT hard-code absolute paths here. Use environment variables (see above)
# or add entries relative to the project root with Path(__file__).parent.parent.
#
# Example — uncomment and set real relative paths if you have fixture files:
# TESTS = [
#     {
#         "name": "fixture pair",
#         "song_a": str(_ROOT / "tests" / "fixtures" / "song_a.mp3"),
#         "song_b": str(_ROOT / "tests" / "fixtures" / "song_b.mp3"),
#         "out":    str(_OUTPUT_DIR / "regression_fixture.wav"),
#         "min_score": 50,
#     },
# ]

TESTS: list = _env_tests()


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
