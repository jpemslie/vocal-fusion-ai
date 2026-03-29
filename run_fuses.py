#!/usr/bin/env python3
"""
Direct fuse runner — bypasses Flask server entirely.
Runs fuse() calls sequentially and scores each output.
"""
import os, sys, time
os.chdir("/Users/jpemslie/Desktop/VocalFusion")
sys.path.insert(0, ".")

BEATS = {
    "rhyme_dust":    "/Users/jpemslie/Downloads/MK Dom Dolla - Rhyme Dust Official Visualiser.mp3",
    "crystallized":  "/Users/jpemslie/Downloads/John Summit - crystallized (feat. Inéz) Official Lyric Visualizer.mp3",
    "test_hiphop":   "/Users/jpemslie/Desktop/VocalFusion/vf_data/test_hiphop.mp3",
    "test_deephouse":"/Users/jpemslie/Desktop/VocalFusion/vf_data/test_deephouse2.mp3",
}
VOCALS = {
    "don_toliver":   "/Users/jpemslie/Downloads/Don Toliver - E85 _ Lyrics.mp3",
    "travis_fein":   "/Users/jpemslie/Downloads/Travis Scott - FE_N ft Playboi Carti.mp3",
    "rap_acapella":  "/Users/jpemslie/Desktop/VocalFusion/vf_data/test_rap_acapella.mp3",
}

COMBOS = [
    ("rhyme_dust",    "travis_fein",  "run_travis_rhyme"),
    ("rhyme_dust",    "don_toliver",  "run_dontoliver_rhyme"),
    ("crystallized",  "don_toliver",  "run_dontoliver_crystal"),
    ("test_hiphop",   "travis_fein",  "run_travis_hiphop"),
    ("test_deephouse","don_toliver",  "run_dontoliver_house"),
]

import fuser, listen

results = []
for beat_key, vox_key, label in COMBOS:
    beat = BEATS[beat_key]
    vox  = VOCALS[vox_key]
    out  = f"vf_data/mixes/{label}_fusion.wav"
    if os.path.exists(out):
        print(f"\n[SKIP] {label} — output already exists")
    else:
        print(f"\n{'='*60}")
        print(f"FUSING: {vox_key} over {beat_key}")
        print(f"{'='*60}")
        t0 = time.time()
        try:
            result = fuser.fuse(beat, vox, out)
            elapsed = time.time() - t0
            print(f"Done in {elapsed/60:.1f} min → {out}")
        except Exception as e:
            print(f"FAILED: {e}")
            continue

    # Score the radio variant
    if os.path.exists(out):
        try:
            r = listen.score_file(out)
            score = r["score"]
            grade = r["grade"]
            problems = r["problems"][:3]
            lufs = r["measurements"].get("lufs", 0)
            hnr  = r["measurements"].get("hnr_db", 0)
            vp   = r["measurements"].get("vocal_presence", 0)
            print(f"\n  SCORE {score}/100 {grade}  LUFS={lufs:.1f}  HNR={hnr:.1f}  VP={vp:.2f}")
            print(f"  Top issues: {problems}")
            results.append((score, label, out, grade))
        except Exception as e:
            print(f"  Score failed: {e}")
            results.append((0, label, out, "?"))

print("\n" + "="*60)
print("RESULTS RANKED:")
for score, label, path, grade in sorted(results, reverse=True):
    print(f"  {score:3d}/100 {grade}  {label}")
    print(f"         {path}")
best = sorted(results, reverse=True)
if best:
    print(f"\nBEST: {best[0][1]}  ({best[0][0]}/100)")
    print(f"      {best[0][2]}")
