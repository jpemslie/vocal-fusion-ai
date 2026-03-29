"""Quick fuse + score test — professional source files."""
import time, sys, os

print("Starting fuse test (pro sources)...", flush=True)
t0 = time.time()

import fuser

DL = "/Users/jpemslie/Downloads"
BEAT  = f"{DL}/MK Dom Dolla - Rhyme Dust Official Visualiser.mp3"
VOCAL = f"{DL}/Don Toliver - E85 _ Lyrics.mp3"
OUT   = "vf_data/mixes/toliver_rhymedust_radio.wav"

result = fuser.fuse(BEAT, VOCAL, OUT, stems_cache='vf_data/stems')
print(f"\nFuse done in {time.time()-t0:.0f}s", flush=True)

# Score it
from ml.scorer import score, print_report
MODEL = 'ml/ref_model.npz'
if os.path.exists(MODEL):
    out_path = result.get('radio') if isinstance(result, dict) else OUT
    print(f"\nScoring: {out_path}", flush=True)
    r = score(out_path, MODEL, duration=60.0)
    print_report(r)
    print("\nKey metrics:")
    for k in ['lra','plr','crest_db','bass_mid_ratio','band_bass','band_mid',
              'stereo_width_mid','stereo_width_high','spectral_slope','vocal_presence','hnr_db']:
        if k in r['features']:
            print(f"  {k:<25} val={r['features'][k]:+.4f}  z={r['z_scores'][k]:+.2f}")
else:
    print("No ref model found")
