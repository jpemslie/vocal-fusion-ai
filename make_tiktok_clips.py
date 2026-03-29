#!/usr/bin/env python3
"""
Wraps fused vocal clips into proper 30-second TikTok clips:
  - 4s beat-only intro (with fade-in)
  - fused vocal section (10s)
  - 14s beat-only outro (with fade-out)
Total: ~28 seconds
"""
import os, sys
os.chdir("/Users/jpemslie/Desktop/VocalFusion")
sys.path.insert(0, ".")

import numpy as np
import soundfile as sf
import librosa

SR = 44100
INTRO_S  = 4.0
OUTRO_S  = 14.0

DL = "/Users/jpemslie/Downloads"
BEAT_FILES = {
    "rhyme_dust":   f"{DL}/MK Dom Dolla - Rhyme Dust Official Visualiser.mp3",
    "crystallized": f"{DL}/John Summit - crystallized (feat. Inéz) Official Lyric Visualizer.mp3",
    "hiphop":       "vf_data/test_hiphop.mp3",
}

CLIPS = [
    ("my_rhymedust_v1_fusion.wav", "rhyme_dust"),
    ("my_rhymedust_v2_fusion.wav", "rhyme_dust"),
    ("my_crystal_v2_fusion.wav",   "crystallized"),
    ("my_hiphop_v1_fusion.wav",    "hiphop"),
    ("my_hiphop_v2_fusion.wav",    "hiphop"),
]

def make_stereo(y):
    if y.ndim == 1:
        return np.stack([y, y], axis=1)
    if y.shape[1] == 1:
        return np.concatenate([y, y], axis=1)
    return y

def fade(y, fade_s, mode="in"):
    n = int(fade_s * SR)
    n = min(n, y.shape[0])
    curve = np.linspace(0, 1, n) if mode == "in" else np.linspace(1, 0, n)
    out = y.copy()
    if mode == "in":
        out[:n] *= curve[:, None]
    else:
        out[-n:] *= curve[:, None]
    return out

for fname, beat_key in CLIPS:
    fused_path = f"vf_data/mixes/{fname}"
    if not os.path.exists(fused_path):
        print(f"[SKIP] {fname} — not yet fused")
        continue

    out_path = fused_path.replace("_fusion.wav", "_tiktok.wav")
    if os.path.exists(out_path):
        print(f"[SKIP] {out_path} — already exists")
        continue

    # Load fused vocal clip
    fused, _ = sf.read(fused_path, always_2d=True)
    fused = fused.astype(np.float32)
    fused_len = fused.shape[0]

    # Load beat, find the best drop point to start from
    beat_mono = librosa.load(BEAT_FILES[beat_key], sr=SR, mono=True)[0]
    beat_stereo = np.stack([beat_mono, beat_mono], axis=1).astype(np.float32)

    # Find the same downbeat the fuser used (first strong beat after ~0.5s)
    # Simple approach: RMS envelope, find first peak after 0.3s
    hop = 512
    rms = librosa.feature.rms(y=beat_mono, frame_length=2048, hop_length=hop)[0]
    start_frame = int(0.3 * SR / hop)
    peak_frame  = np.argmax(rms[start_frame:start_frame + int(4*SR/hop)]) + start_frame
    beat_start  = max(0, int(peak_frame * hop) - int(0.5 * SR))  # half a beat before peak

    # Normalise beat to match fused clip's overall level
    fused_rms = float(np.sqrt(np.mean(fused**2)) + 1e-9)
    beat_section = beat_stereo[beat_start:]
    beat_rms  = float(np.sqrt(np.mean(beat_section[:fused_len]**2)) + 1e-9)
    beat_gain = fused_rms / beat_rms * 0.75  # slightly quieter than fused mix
    beat_norm = (beat_section * beat_gain).astype(np.float32)

    intro_n  = int(INTRO_S * SR)
    outro_n  = int(OUTRO_S * SR)

    # Build: intro | fused | outro
    intro = beat_norm[:intro_n]
    if intro.shape[0] < intro_n:
        intro = np.pad(intro, ((0, intro_n - intro.shape[0]), (0, 0)))
    intro = fade(intro, 0.5, "in")

    outro_start = fused_len
    outro = beat_norm[outro_start:outro_start + outro_n]
    if outro.shape[0] < outro_n:
        outro = np.pad(outro, ((0, outro_n - outro.shape[0]), (0, 0)))
    outro = fade(outro, 2.5, "out")

    clip = np.concatenate([intro, fused, outro], axis=0).astype(np.float32)

    # Final peak normalise to -1 dBFS
    peak = np.max(np.abs(clip))
    if peak > 1e-6:
        clip = clip / peak * 0.891

    sf.write(out_path, clip, SR, subtype="PCM_24")
    dur = clip.shape[0] / SR
    print(f"[OK] {os.path.basename(out_path)}  ({dur:.1f}s)")

print("\nDone. TikTok clips in vf_data/mixes/")
