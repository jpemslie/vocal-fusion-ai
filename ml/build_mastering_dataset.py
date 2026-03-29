"""
Build training pairs for MasteringNetwork using matchering.

For each reference professional track, we create a "degraded" version
(volume normalization only, no mastering EQ/compression) and use
matchering to process it to match the reference. This gives us:
  - input:  degraded/unmastered-sounding mix
  - target: matchering-processed (professional-sounding) version

Usage:
    python -m ml.build_mastering_dataset \
        --ref-dir /path/to/reference_tracks \
        --mix-dir /path/to/raw_mixes \
        --out-dir ml/cache/mastering_pairs \
        --genre hiphop

Reference tracks: professional MP3/WAV releases (same style as your targets)
Raw mixes: any stems mix that hasn't been mastered yet
"""

import argparse
import random
import sys
from pathlib import Path

import numpy as np
import soundfile as sf


SR = 44100


def load_audio(path: str | Path, target_sr: int = SR) -> np.ndarray:
    """Load audio as (2, T) float32, resampling if needed."""
    import librosa
    y, sr = librosa.load(str(path), sr=target_sr, mono=False)
    if y.ndim == 1:
        y = np.stack([y, y])
    return y.astype(np.float32)


def normalize_lufs(audio: np.ndarray, target_lufs: float = -23.0,
                   sr: int = SR) -> np.ndarray:
    """Rough LUFS normalization (ITU-R BS.1770 approximation)."""
    try:
        import pyloudnorm as pyln
        meter = pyln.Meter(sr)
        loudness = meter.integrated_loudness(audio.T)
        if np.isfinite(loudness):
            gain_db = target_lufs - loudness
            audio = audio * (10 ** (gain_db / 20))
    except ImportError:
        # Fallback: RMS normalization
        rms = np.sqrt(np.mean(audio ** 2) + 1e-12)
        target_rms = 10 ** (target_lufs / 20)
        audio = audio * (target_rms / rms)
    return audio


def degrade(audio: np.ndarray) -> np.ndarray:
    """
    Simulate an 'unmastered' mix by:
    1. LUFS normalize to -23 (flat, no mastering compression)
    2. Add slight low-end muddiness (boost 100-300 Hz)
    3. Add slight high-frequency harshness (boost 3-6 kHz)
    4. Slight stereo imbalance
    This teaches the model to undo these artifacts.
    """
    try:
        from pedalboard import Pedalboard, HighpassFilter, LowShelfFilter, PeakFilter
        board = Pedalboard([
            LowShelfFilter(cutoff_frequency_hz=200.0,
                           gain_db=random.uniform(1.5, 3.0)),
            PeakFilter(cutoff_frequency_hz=random.uniform(3000, 5000),
                       gain_db=random.uniform(1.0, 2.5),
                       q=1.5),
        ])
        audio = board(audio, SR)
    except Exception:
        pass
    audio = normalize_lufs(audio, target_lufs=-23.0)
    return audio


def match_with_matchering(input_audio: np.ndarray, ref_audio: np.ndarray,
                          sr: int = SR) -> np.ndarray | None:
    """Use matchering to process input to sound like ref."""
    import tempfile, os
    try:
        import matchering as mg
    except ImportError:
        print("  WARNING: matchering not installed. pip install matchering")
        return None

    with tempfile.TemporaryDirectory() as tmp:
        inp_path  = os.path.join(tmp, "input.wav")
        ref_path  = os.path.join(tmp, "ref.wav")
        out_path  = os.path.join(tmp, "output.wav")
        sf.write(inp_path, input_audio.T, sr, subtype="PCM_24")
        sf.write(ref_path, ref_audio.T,   sr, subtype="PCM_24")
        try:
            mg.process(
                target=inp_path,
                reference=ref_path,
                results=[mg.pcm24(out_path)],
            )
            out, _ = sf.read(out_path, dtype="float32", always_2d=True)
            return out.T
        except Exception as e:
            print(f"  matchering error: {e}")
            return None


def build(args):
    ref_dir = Path(args.ref_dir)
    mix_dir = Path(args.mix_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    refs = list(ref_dir.glob("*.wav")) + list(ref_dir.glob("*.mp3"))
    mixes = list(mix_dir.glob("*.wav")) + list(mix_dir.glob("*.mp3"))

    if not refs:
        print(f"No reference tracks found in {ref_dir}")
        sys.exit(1)
    if not mixes:
        print(f"No mix files found in {mix_dir}")
        sys.exit(1)

    print(f"Found {len(refs)} references, {len(mixes)} mixes")
    print(f"Building {args.n_pairs} pairs → {out_dir}")

    pair_id = 0
    attempts = 0
    random.shuffle(mixes)
    random.shuffle(refs)

    while pair_id < args.n_pairs and attempts < args.n_pairs * 3:
        attempts += 1
        mix_path = mixes[pair_id % len(mixes)]
        ref_path = refs[pair_id % len(refs)]

        inp_path = out_dir / f"{pair_id:05d}_{args.genre}_input.npy"
        tgt_path = out_dir / f"{pair_id:05d}_{args.genre}_target.npy"
        if inp_path.exists() and tgt_path.exists():
            pair_id += 1
            continue

        print(f"  [{pair_id+1}/{args.n_pairs}] {mix_path.name} × {ref_path.name}", flush=True)

        try:
            mix_audio = load_audio(mix_path)
            ref_audio = load_audio(ref_path)
        except Exception as e:
            print(f"  Load error: {e}")
            continue

        # Trim to same length (use min, max 5 min)
        max_samples = 5 * 60 * SR
        L = min(mix_audio.shape[-1], ref_audio.shape[-1], max_samples)
        mix_audio = mix_audio[:, :L]
        ref_audio = ref_audio[:, :L]

        degraded = degrade(mix_audio.copy())
        mastered = match_with_matchering(degraded, ref_audio)
        if mastered is None:
            continue

        np.save(inp_path, degraded)
        np.save(tgt_path, mastered)
        pair_id += 1

    print(f"\nBuilt {pair_id} pairs in {out_dir}")


def main():
    p = argparse.ArgumentParser(description="Build MasteringNetwork training data")
    p.add_argument("--ref-dir", required=True,
                   help="Directory of professional reference tracks")
    p.add_argument("--mix-dir", required=True,
                   help="Directory of raw/unmastered mixes")
    p.add_argument("--out-dir", default="ml/cache/mastering_pairs",
                   help="Output directory for .npy pairs")
    p.add_argument("--genre", default="hiphop", help="Genre label")
    p.add_argument("--n-pairs", type=int, default=500,
                   help="Number of training pairs to generate")
    args = p.parse_args()
    build(args)


if __name__ == "__main__":
    main()
