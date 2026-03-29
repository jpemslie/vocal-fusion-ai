#!/usr/bin/env python3
"""
Run user's recorded vocals through VocalFusion.
direct_vocal=True: skips Demucs, handles M4A, applies auto-tune + effects.
"""
import os, sys, time
os.chdir("/Users/jpemslie/Desktop/VocalFusion")
sys.path.insert(0, ".")

DL = "/Users/jpemslie/Downloads"

BEATS = {
    "rhyme_dust":   f"{DL}/MK Dom Dolla - Rhyme Dust Official Visualiser.mp3",
    "crystallized": f"{DL}/John Summit - crystallized (feat. Inéz) Official Lyric Visualizer.mp3",
    "hiphop":       "vf_data/test_hiphop.mp3",
    "deephouse":    "vf_data/test_deephouse2.mp3",
}

# S Church Ave = house vibes (recorded together, likely the melodic lines)
# After Hours Pediatrics = hip-hop lines
VOCALS = {
    "house_v1":   f"{DL}/S Church Ave.m4a",
    "house_v2":   f"{DL}/S Church Ave 2.m4a",
    "hiphop_v1":  f"{DL}/After Hours Pediatrics Urgent Care - South Tampa.m4a",
    "hiphop_v2":  f"{DL}/After Hours Pediatrics Urgent Care - South Tampa 2.m4a",
}

COMBOS = [
    ("rhyme_dust",  "house_v1",  "my_rhymedust_v1"),
    ("rhyme_dust",  "house_v2",  "my_rhymedust_v2"),
    ("crystallized","house_v1",  "my_crystal_v1"),
    ("hiphop",      "hiphop_v1", "my_hiphop_v1"),
    ("hiphop",      "hiphop_v2", "my_hiphop_v2"),
]

import fuser, listen

results = []
for beat_key, vox_key, label in COMBOS:
    beat = BEATS[beat_key]
    vox  = VOCALS[vox_key]
    out  = f"vf_data/mixes/{label}_fusion.wav"

    if os.path.exists(out):
        print(f"\n[SKIP] {label} — already exists")
    else:
        print(f"\n{'='*60}")
        print(f"FUSING: {vox_key} over {beat_key}")
        print(f"{'='*60}")
        t0 = time.time()
        try:
            result = fuser.fuse(beat, vox, out, direct_vocal=True)
            elapsed = time.time() - t0
            print(f"Done in {elapsed/60:.1f} min → {out}")
        except Exception as e:
            print(f"FAILED: {e}")
            import traceback; traceback.print_exc()
            continue

    if os.path.exists(out):
        try:
            r = listen.score_file(out)
            score = r["score"]
            grade = r["grade"]
            lufs  = r["measurements"].get("lufs", 0)
            hnr   = r["measurements"].get("hnr_db", 0)
            vp    = r["measurements"].get("vocal_presence", 0)
            print(f"\n  SCORE {score}/100 {grade}  LUFS={lufs:.1f}  HNR={hnr:.1f}  VP={vp:.2f}")
            results.append((score, label, out, grade))
        except Exception as e:
            print(f"  Score failed: {e}")
            results.append((0, label, out, "?"))

print("\n" + "="*60)
print("RESULTS RANKED:")
for score, label, path, grade in sorted(results, reverse=True):
    print(f"  {score:3d}/100 {grade}  {label}")
    print(f"         {path}")
if results:
    best = sorted(results, reverse=True)[0]
    print(f"\nBEST: {best[1]}  ({best[0]}/100)")
    print(f"      {best[2]}")
