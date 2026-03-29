"""
Train MasteringNetwork on (unmastered, matchering-processed) pairs.

Loss:
  - Multi-scale STFT (spectral convergence + log-magnitude L1)
  - Multi-resolution STFT discriminator (adversarial)
  - LUFS matching loss

Usage:
    python -m ml.train_mastering \
        --data-dir ml/cache/mastering_pairs \
        --genre hiphop \
        --out ml/mastering_net.pt \
        --epochs 200 --batch 4 --lr 3e-4
"""

import argparse
import random
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset

from ml.mastering_net import MasteringNetwork, GENRE2IDX


SR = 44100


# ------------------------------------------------------------------ #
#  Multi-scale STFT loss                                              #
# ------------------------------------------------------------------ #

class MultiScaleSTFTLoss(nn.Module):
    def __init__(self, fft_sizes=(512, 1024, 2048),
                 hop_ratios=(0.25, 0.25, 0.25)):
        super().__init__()
        self.configs = list(zip(fft_sizes, hop_ratios))

    def forward(self, est: torch.Tensor, tgt: torch.Tensor) -> torch.Tensor:
        """Both (B, 2, T) → scalar loss."""
        # Mix to mono for STFT
        e = est.mean(dim=1)
        t = tgt.mean(dim=1)
        total = torch.tensor(0.0, device=est.device)
        for n_fft, hop_r in self.configs:
            hop = int(n_fft * hop_r)
            e_s = torch.stft(e, n_fft, hop_length=hop,
                             return_complex=True)
            t_s = torch.stft(t, n_fft, hop_length=hop,
                             return_complex=True)
            e_mag = e_s.abs().clamp(min=1e-7)
            t_mag = t_s.abs().clamp(min=1e-7)
            # Spectral convergence
            sc = (t_mag - e_mag).norm() / (t_mag.norm() + 1e-8)
            # Log-magnitude L1
            lm = (t_mag.log() - e_mag.log()).abs().mean()
            total = total + sc + lm
        return total / len(self.configs)


# ------------------------------------------------------------------ #
#  LUFS matching loss                                                  #
# ------------------------------------------------------------------ #

def lufs_loss(est: torch.Tensor, tgt: torch.Tensor) -> torch.Tensor:
    """Approximate LUFS matching via RMS."""
    e_rms = (est ** 2).mean().sqrt().clamp(min=1e-8)
    t_rms = (tgt ** 2).mean().sqrt().clamp(min=1e-8)
    return (e_rms.log() - t_rms.log()).abs()


# ------------------------------------------------------------------ #
#  Simple multi-resolution discriminator                              #
# ------------------------------------------------------------------ #

class _SpectralDiscriminator(nn.Module):
    """Lightweight discriminator operating on STFT magnitude."""

    def __init__(self, n_fft: int = 1024):
        super().__init__()
        self.n_fft = n_fft
        n_bins = n_fft // 2 + 1
        self.net = nn.Sequential(
            nn.Conv2d(1, 32, (3, 3), stride=(2, 2), padding=1),
            nn.LeakyReLU(0.2),
            nn.Conv2d(32, 64, (3, 3), stride=(2, 2), padding=1),
            nn.LeakyReLU(0.2),
            nn.Conv2d(64, 128, (3, 3), stride=(2, 2), padding=1),
            nn.LeakyReLU(0.2),
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(128, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, 2, T) → mono
        m = x.mean(dim=1)
        spec = torch.stft(m, self.n_fft, return_complex=True).abs()
        spec = spec.unsqueeze(1)  # (B, 1, F, T)
        return self.net(spec)


class MultiResDiscriminator(nn.Module):
    def __init__(self, fft_sizes=(512, 1024, 2048)):
        super().__init__()
        self.discs = nn.ModuleList([_SpectralDiscriminator(f) for f in fft_sizes])

    def forward(self, x: torch.Tensor):
        return [d(x) for d in self.discs]


# ------------------------------------------------------------------ #
#  Dataset                                                             #
# ------------------------------------------------------------------ #

class MasteringPairDataset(Dataset):
    def __init__(self, data_dir: str, genre: str = "hiphop",
                 clip_s: float = 6.0, sr: int = SR, augment: bool = True):
        self.clip = int(clip_s * sr)
        self.sr = sr
        self.augment = augment
        self.genre_idx = GENRE2IDX.get(genre, GENRE2IDX["other"])

        d = Path(data_dir)
        self.inputs = sorted(d.glob(f"*_{genre}_input.npy"))
        if not self.inputs:
            # fallback: any input
            self.inputs = sorted(d.glob("*_input.npy"))
        if not self.inputs:
            raise FileNotFoundError(f"No *_input.npy in {data_dir}")

    def __len__(self):
        return len(self.inputs)

    def _load(self, path: Path) -> np.ndarray:
        a = np.load(path).astype(np.float32)
        if a.ndim == 1:
            a = np.stack([a, a])
        return a

    def __getitem__(self, idx: int):
        inp_path = self.inputs[idx]
        tgt_path = Path(str(inp_path).replace("_input.npy", "_target.npy"))
        inp = self._load(inp_path)
        tgt = self._load(tgt_path)

        L = min(inp.shape[-1], tgt.shape[-1])
        if L > self.clip:
            start = random.randint(0, L - self.clip)
            inp = inp[:, start:start + self.clip]
            tgt = tgt[:, start:start + self.clip]
        else:
            inp = np.pad(inp, ((0, 0), (0, self.clip - L)))
            tgt = np.pad(tgt, ((0, 0), (0, self.clip - L)))

        if self.augment:
            gain = 10 ** (random.uniform(-3, 3) / 20)
            inp = inp * gain
            tgt = tgt * gain

        return (torch.from_numpy(inp),
                torch.from_numpy(tgt),
                torch.tensor(self.genre_idx, dtype=torch.long))


# ------------------------------------------------------------------ #
#  Training loop                                                       #
# ------------------------------------------------------------------ #

def train(args):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Training on {device}")

    ds = MasteringPairDataset(args.data_dir, args.genre, augment=True)
    val_n = max(1, len(ds) // 10)
    train_ds, val_ds = torch.utils.data.random_split(ds, [len(ds) - val_n, val_n])
    train_dl = DataLoader(train_ds, batch_size=args.batch,
                          shuffle=True, num_workers=4, pin_memory=True)
    val_dl   = DataLoader(val_ds, batch_size=args.batch,
                          shuffle=False, num_workers=2)

    G = MasteringNetwork().to(device)
    D = MultiResDiscriminator().to(device)
    print(f"Generator params: {sum(p.numel() for p in G.parameters()):,}")
    print(f"Discriminator params: {sum(p.numel() for p in D.parameters()):,}")

    g_opt = optim.AdamW(G.parameters(), lr=args.lr, betas=(0.8, 0.99))
    d_opt = optim.AdamW(D.parameters(), lr=args.lr * 2, betas=(0.8, 0.99))
    g_sched = optim.lr_scheduler.CosineAnnealingLR(g_opt, args.epochs)
    d_sched = optim.lr_scheduler.CosineAnnealingLR(d_opt, args.epochs)

    stft_loss_fn = MultiScaleSTFTLoss().to(device)
    best_val = float("inf")

    for epoch in range(1, args.epochs + 1):
        G.train(); D.train()
        g_total = d_total = 0.0

        for inp, tgt, genre_idx in train_dl:
            inp  = inp.to(device)
            tgt  = tgt.to(device)
            genre_idx = genre_idx.to(device)

            # --- Discriminator step ---
            with torch.no_grad():
                fake = G(inp, genre_idx)
            d_real = D(tgt)
            d_fake = D(fake.detach())
            d_loss = sum(
                F.mse_loss(r, torch.ones_like(r)) +
                F.mse_loss(f, torch.zeros_like(f))
                for r, f in zip(d_real, d_fake)
            ) / len(d_real)
            d_opt.zero_grad()
            d_loss.backward()
            nn.utils.clip_grad_norm_(D.parameters(), 5.0)
            d_opt.step()

            # --- Generator step ---
            fake = G(inp, genre_idx)
            d_fake_g = D(fake)
            adv_loss = sum(F.mse_loss(f, torch.ones_like(f)) for f in d_fake_g)
            rec_loss = stft_loss_fn(fake, tgt)
            l_loss   = lufs_loss(fake, tgt)
            g_loss   = rec_loss + 0.05 * adv_loss + 0.1 * l_loss
            g_opt.zero_grad()
            g_loss.backward()
            nn.utils.clip_grad_norm_(G.parameters(), 5.0)
            g_opt.step()

            g_total += g_loss.item()
            d_total += d_loss.item()

        g_sched.step(); d_sched.step()

        # --- Validation ---
        G.eval()
        val_loss = 0.0
        with torch.no_grad():
            for inp, tgt, genre_idx in val_dl:
                inp  = inp.to(device)
                tgt  = tgt.to(device)
                genre_idx = genre_idx.to(device)
                fake = G(inp, genre_idx)
                val_loss += stft_loss_fn(fake, tgt).item()
        val_loss /= max(1, len(val_dl))

        print(f"Epoch {epoch:3d}/{args.epochs}  "
              f"G={g_total/len(train_dl):.3f}  "
              f"D={d_total/len(train_dl):.3f}  "
              f"val_stft={val_loss:.3f}", flush=True)

        if val_loss < best_val:
            best_val = val_loss
            G.save(args.out, config={"cond_dim": 128, "n_stages": 4})
            print(f"  ✓ Saved → {args.out}")

    print(f"\nDone. Best val STFT loss: {best_val:.3f}")


def main():
    p = argparse.ArgumentParser(description="Train MasteringNetwork")
    p.add_argument("--data-dir", default="ml/cache/mastering_pairs")
    p.add_argument("--genre", default="hiphop")
    p.add_argument("--out", default="ml/mastering_net.pt")
    p.add_argument("--epochs", type=int, default=200)
    p.add_argument("--batch", type=int, default=4)
    p.add_argument("--lr", type=float, default=3e-4)
    args = p.parse_args()
    train(args)


if __name__ == "__main__":
    main()
