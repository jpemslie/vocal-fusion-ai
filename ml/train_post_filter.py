"""
Train StemPostFilter on MUSDB18-HQ.

Pipeline:
  1. Run Demucs on MUSDB18-HQ mixture → get noisy_stem
  2. Use original stem as target
  3. Train Conv-TasNet to map noisy_stem → original_stem

Loss: SI-SDR (scale-invariant SDR) + spectral convergence

Usage:
    # One-time dataset prep (runs Demucs on MUSDB18-HQ)
    python -m ml.train_post_filter --prep-data \
        --musdb-dir /path/to/musdb18hq \
        --cache-dir ml/cache/post_filter_pairs

    # Training
    python -m ml.train_post_filter \
        --cache-dir ml/cache/post_filter_pairs \
        --stem vocals \
        --out ml/post_filter_vocals.pt \
        --epochs 100 --batch 8 --lr 1e-3
"""

import argparse
import random
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset

from ml.post_filter import build_model


# ------------------------------------------------------------------ #
#  SI-SDR loss                                                         #
# ------------------------------------------------------------------ #

def si_sdr(estimate: torch.Tensor, target: torch.Tensor,
           eps: float = 1e-8) -> torch.Tensor:
    """Scale-invariant SDR. Both (B, T). Returns scalar."""
    t = target - target.mean(dim=-1, keepdim=True)
    e = estimate - estimate.mean(dim=-1, keepdim=True)
    dot = (e * t).sum(dim=-1, keepdim=True)
    s_target = dot / (t.pow(2).sum(dim=-1, keepdim=True) + eps) * t
    noise = e - s_target
    sdr = 10 * torch.log10(
        (s_target.pow(2).sum(dim=-1) + eps) /
        (noise.pow(2).sum(dim=-1) + eps)
    )
    return -sdr.mean()


def spectral_convergence(estimate: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """Multi-scale spectral convergence loss."""
    total = torch.tensor(0.0, device=estimate.device)
    for n_fft in [512, 1024, 2048]:
        e_mag = torch.stft(estimate, n_fft, return_complex=True).abs()
        t_mag = torch.stft(target,   n_fft, return_complex=True).abs()
        total = total + (e_mag - t_mag).norm() / (t_mag.norm() + 1e-8)
    return total / 3.0


# ------------------------------------------------------------------ #
#  Dataset                                                             #
# ------------------------------------------------------------------ #

class StemPairDataset(Dataset):
    """
    Loads (noisy_stem, clean_stem) pairs from a cache directory.
    Each pair is stored as two .npy files: <id>_noisy.npy, <id>_clean.npy
    """

    def __init__(self, cache_dir: str, clip_s: float = 4.0, sr: int = 44100,
                 augment: bool = True):
        self.cache_dir = Path(cache_dir)
        self.clip = int(clip_s * sr)
        self.sr = sr
        self.augment = augment
        self.pairs = sorted(self.cache_dir.glob("*_noisy.npy"))
        if not self.pairs:
            raise FileNotFoundError(
                f"No *_noisy.npy files found in {cache_dir}. "
                "Run with --prep-data first."
            )

    def __len__(self):
        return len(self.pairs)

    def _load(self, path: Path) -> np.ndarray:
        a = np.load(path).astype(np.float32)
        if a.ndim == 2:
            a = a.mean(axis=0)  # stereo → mono
        return a

    def __getitem__(self, idx: int):
        noisy_path = self.pairs[idx]
        clean_path = Path(str(noisy_path).replace("_noisy.npy", "_clean.npy"))
        noisy = self._load(noisy_path)
        clean = self._load(clean_path)

        # Random crop
        L = min(len(noisy), len(clean))
        if L > self.clip:
            start = random.randint(0, L - self.clip)
            noisy = noisy[start:start + self.clip]
            clean = clean[start:start + self.clip]
        else:
            noisy = np.pad(noisy, (0, self.clip - len(noisy)))
            clean = np.pad(clean, (0, self.clip - len(clean)))

        # Augmentation: random gain
        if self.augment:
            gain = 10 ** (random.uniform(-6, 6) / 20)
            noisy = noisy * gain
            clean = clean * gain

        return torch.from_numpy(noisy), torch.from_numpy(clean)


# ------------------------------------------------------------------ #
#  Data preparation                                                    #
# ------------------------------------------------------------------ #

def prep_data(musdb_dir: str, cache_dir: str, stem: str = "vocals"):
    """Run Demucs on MUSDB18-HQ and cache (noisy, clean) pairs."""
    try:
        import musdb
        import demucs.separate
    except ImportError:
        raise ImportError("pip install musdb demucs")

    cache = Path(cache_dir)
    cache.mkdir(parents=True, exist_ok=True)

    db = musdb.DB(root=musdb_dir, is_wav=True)
    print(f"Found {len(db)} MUSDB18-HQ tracks")

    for i, track in enumerate(db):
        out_id = f"{i:04d}"
        noisy_out = cache / f"{out_id}_noisy.npy"
        clean_out = cache / f"{out_id}_clean.npy"
        if noisy_out.exists() and clean_out.exists():
            print(f"  [{i+1}/{len(db)}] {track.name} — cached, skipping")
            continue

        print(f"  [{i+1}/{len(db)}] {track.name} — separating…", flush=True)
        mixture_np = track.audio.T.astype(np.float32)  # (2, T)

        # Run Demucs
        import torch
        from demucs.pretrained import get_model
        from demucs.apply import apply_model

        model = get_model("htdemucs")
        model.eval()
        wav = torch.from_numpy(mixture_np[None])  # (1, 2, T)
        with torch.no_grad():
            sources = apply_model(model, wav, device="cpu")[0]  # (n_src, 2, T)
        src_names = model.sources
        stem_idx = src_names.index(stem)
        noisy_stem = sources[stem_idx].numpy()  # (2, T)

        # Clean stem: use original multi-track stem
        clean_stem = getattr(track.targets[stem], "audio", None)
        if clean_stem is None:
            print(f"    WARNING: no clean {stem} stem, skipping")
            continue
        clean_stem = clean_stem.T.astype(np.float32)  # (2, T)

        np.save(noisy_out, noisy_stem)
        np.save(clean_out, clean_stem)

    print(f"\nDone. Cached {len(list(cache.glob('*_noisy.npy')))} pairs → {cache_dir}")


# ------------------------------------------------------------------ #
#  Training loop                                                       #
# ------------------------------------------------------------------ #

def train(args):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Training on {device}")

    train_ds = StemPairDataset(args.cache_dir, clip_s=4.0, augment=True)
    val_size = max(1, len(train_ds) // 10)
    train_ds, val_ds = torch.utils.data.random_split(
        train_ds, [len(train_ds) - val_size, val_size]
    )
    train_dl = DataLoader(train_ds, batch_size=args.batch,
                          shuffle=True, num_workers=4, pin_memory=True)
    val_dl   = DataLoader(val_ds,   batch_size=args.batch,
                          shuffle=False, num_workers=2)

    model = build_model(args.stem).to(device)
    print(f"Model params: {sum(p.numel() for p in model.parameters()):,}")

    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=5,
                                                     factor=0.5, min_lr=1e-5)

    best_val = float("inf")
    for epoch in range(1, args.epochs + 1):
        # --- train ---
        model.train()
        total_loss = 0.0
        for noisy, clean in train_dl:
            noisy = noisy.unsqueeze(1).to(device)
            clean = clean.to(device)
            est = model(noisy).squeeze(1)
            loss = si_sdr(est, clean) + 0.1 * spectral_convergence(est, clean)
            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()
            total_loss += loss.item()

        train_loss = total_loss / len(train_dl)

        # --- val ---
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for noisy, clean in val_dl:
                noisy = noisy.unsqueeze(1).to(device)
                clean = clean.to(device)
                est = model(noisy).squeeze(1)
                val_loss += si_sdr(est, clean).item()
        val_loss /= len(val_dl)

        scheduler.step(val_loss)
        print(f"Epoch {epoch:3d}/{args.epochs}  train={train_loss:.3f}  val={val_loss:.3f}",
              flush=True)

        if val_loss < best_val:
            best_val = val_loss
            model.save(args.out, config={
                "n_filters": 512, "bottleneck": 128, "hidden": 512,
                "n_blocks": 8, "n_repeats": 3,
            })
            print(f"  ✓ Saved best model → {args.out}")

    print(f"\nTraining complete. Best val SI-SDR loss: {best_val:.3f}")


# ------------------------------------------------------------------ #
#  CLI                                                                 #
# ------------------------------------------------------------------ #

def main():
    p = argparse.ArgumentParser(description="Train StemPostFilter")
    p.add_argument("--prep-data", action="store_true",
                   help="Prepare dataset (run Demucs on MUSDB18-HQ)")
    p.add_argument("--musdb-dir", default="/path/to/musdb18hq",
                   help="MUSDB18-HQ root directory")
    p.add_argument("--cache-dir", default="ml/cache/post_filter_pairs",
                   help="Directory to cache (noisy, clean) pairs")
    p.add_argument("--stem", default="vocals",
                   choices=["vocals", "bass", "drums", "other"])
    p.add_argument("--out", default="ml/post_filter_vocals.pt",
                   help="Output model path")
    p.add_argument("--epochs", type=int, default=100)
    p.add_argument("--batch", type=int, default=8)
    p.add_argument("--lr", type=float, default=1e-3)
    args = p.parse_args()

    if args.prep_data:
        prep_data(args.musdb_dir, args.cache_dir, args.stem)
    else:
        train(args)


if __name__ == "__main__":
    main()
