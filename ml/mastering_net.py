"""
MasteringNetwork — Neural mastering chain.

Replaces the manual EQ/compression mastering chain with a learned 1D U-Net
conditioned on genre embedding. Trained on (unmastered_mix, matchering_target)
pairs from professional releases.

Architecture: ~8M parameters
  - Genre encoder: embedding → 128-dim condition vector
  - Encoder path: 4 downsampling stages (stride-2 conv + DualPathTransformer)
  - Bottleneck: 512 channels, DualPathTransformer
  - Decoder path: 4 upsampling stages (transposed conv + skip connection)
  - Output: 1×1 conv → waveform residual added to input

DualPathTransformer:
  - Intra-chunk self-attention (time domain, chunk_size=256)
  - Inter-chunk self-attention (frequency-like, across chunks)
  - Helps model capture both short-term transients and long-range tonal balance

Usage:
    from ml.mastering_net import MasteringNetwork
    model = MasteringNetwork.load("ml/mastering_net.pt")
    mastered = model.process(unmastered_np, genre="hiphop", sr=44100)
"""

import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from pathlib import Path


GENRES = ["hiphop", "pop", "rnb", "trap", "edm", "rock", "classical", "jazz", "other"]
GENRE2IDX = {g: i for i, g in enumerate(GENRES)}


# ------------------------------------------------------------------ #
#  DualPathTransformer                                                 #
# ------------------------------------------------------------------ #

class _MultiheadSelfAttn(nn.Module):
    def __init__(self, d_model: int, n_heads: int, dropout: float = 0.1):
        super().__init__()
        self.attn = nn.MultiheadAttention(d_model, n_heads,
                                          dropout=dropout, batch_first=True)
        self.norm = nn.LayerNorm(d_model)
        self.ffn  = nn.Sequential(
            nn.Linear(d_model, d_model * 4),
            nn.GELU(),
            nn.Linear(d_model * 4, d_model),
            nn.Dropout(dropout),
        )
        self.norm2 = nn.LayerNorm(d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        attn_out, _ = self.attn(x, x, x)
        x = self.norm(x + attn_out)
        x = self.norm2(x + self.ffn(x))
        return x


class DualPathTransformer(nn.Module):
    """
    Intra + inter chunk attention.
    Input: (B, C, T)
    """

    def __init__(self, channels: int, n_heads: int = 8,
                 chunk_size: int = 256, dropout: float = 0.1):
        super().__init__()
        self.chunk_size = chunk_size
        self.intra = _MultiheadSelfAttn(channels, n_heads, dropout)
        self.inter = _MultiheadSelfAttn(channels, n_heads, dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, C, T = x.shape
        cs = self.chunk_size
        # pad to multiple of cs
        pad = (cs - T % cs) % cs
        if pad:
            x = F.pad(x, (0, pad))
        n_chunks = x.shape[-1] // cs

        # Intra-chunk: (B * n_chunks, cs, C)
        xc = x.reshape(B, C, n_chunks, cs).permute(0, 2, 3, 1)  # (B, n, cs, C)
        xc = xc.reshape(B * n_chunks, cs, C)
        xc = self.intra(xc)
        xc = xc.reshape(B, n_chunks, cs, C).permute(0, 3, 1, 2)  # (B, C, n, cs)

        # Inter-chunk: (B * cs, n_chunks, C)
        xi = xc.permute(0, 3, 2, 1).reshape(B * cs, n_chunks, C)
        xi = self.inter(xi)
        xi = xi.reshape(B, cs, n_chunks, C).permute(0, 3, 2, 1)  # (B, C, n, cs)
        out = xi.reshape(B, C, -1)

        if pad:
            out = out[..., :T]
        return out


# ------------------------------------------------------------------ #
#  U-Net building blocks                                               #
# ------------------------------------------------------------------ #

class _CondConvBlock(nn.Module):
    """Conv block with FiLM conditioning from genre embedding."""

    def __init__(self, in_ch: int, out_ch: int, cond_dim: int,
                 stride: int = 1, n_heads: int = 8):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv1d(in_ch, out_ch, 3, stride=stride, padding=1),
            nn.GroupNorm(min(out_ch, 32), out_ch),
            nn.GELU(),
            nn.Conv1d(out_ch, out_ch, 3, padding=1),
            nn.GroupNorm(min(out_ch, 32), out_ch),
        )
        self.film_scale = nn.Linear(cond_dim, out_ch)
        self.film_shift = nn.Linear(cond_dim, out_ch)
        self.act = nn.GELU()
        self.dpt = DualPathTransformer(out_ch, n_heads=n_heads)

    def forward(self, x: torch.Tensor, cond: torch.Tensor) -> torch.Tensor:
        h = self.conv(x)
        scale = self.film_scale(cond).unsqueeze(-1)  # (B, out_ch, 1)
        shift = self.film_shift(cond).unsqueeze(-1)
        h = h * (1 + scale) + shift
        h = self.act(h)
        h = self.dpt(h)
        return h


# ------------------------------------------------------------------ #
#  MasteringNetwork                                                    #
# ------------------------------------------------------------------ #

class MasteringNetwork(nn.Module):
    """
    Genre-conditioned 1D U-Net for neural mastering.

    Input:  (B, 2, T) stereo waveform  [-1, 1]
    Output: (B, 2, T) mastered waveform (residual added to input)
    """

    N_GENRES = len(GENRES)
    BASE_CH  = 64

    def __init__(self, cond_dim: int = 128, n_stages: int = 4):
        super().__init__()
        self.genre_emb = nn.Embedding(self.N_GENRES, cond_dim)
        self.cond_proj = nn.Sequential(
            nn.Linear(cond_dim, cond_dim * 4),
            nn.GELU(),
            nn.Linear(cond_dim * 4, cond_dim),
        )

        # Encoder
        ch_seq = [self.BASE_CH * (2 ** i) for i in range(n_stages + 1)]
        self.enc_in = nn.Conv1d(2, ch_seq[0], 7, padding=3)
        self.enc_blocks = nn.ModuleList()
        self.enc_downs   = nn.ModuleList()
        for i in range(n_stages):
            self.enc_blocks.append(_CondConvBlock(ch_seq[i], ch_seq[i], cond_dim,
                                                  n_heads=max(1, ch_seq[i] // 64)))
            self.enc_downs.append(nn.Conv1d(ch_seq[i], ch_seq[i + 1], 4,
                                            stride=2, padding=1))

        # Bottleneck
        self.bottleneck = _CondConvBlock(ch_seq[-1], ch_seq[-1], cond_dim,
                                         n_heads=max(1, ch_seq[-1] // 64))

        # Decoder
        self.dec_ups    = nn.ModuleList()
        self.dec_blocks = nn.ModuleList()
        for i in reversed(range(n_stages)):
            self.dec_ups.append(nn.ConvTranspose1d(ch_seq[i + 1], ch_seq[i],
                                                   4, stride=2, padding=1))
            self.dec_blocks.append(_CondConvBlock(ch_seq[i] * 2, ch_seq[i], cond_dim,
                                                  n_heads=max(1, ch_seq[i] // 64)))

        self.out_conv = nn.Conv1d(ch_seq[0], 2, 1)

    def forward(self, x: torch.Tensor, genre_idx: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x:         (B, 2, T) input waveform
            genre_idx: (B,) integer genre indices
        Returns:
            (B, 2, T) mastered waveform
        """
        cond = self.cond_proj(self.genre_emb(genre_idx))  # (B, cond_dim)
        T = x.shape[-1]

        h = self.enc_in(x)
        skips = []
        for blk, down in zip(self.enc_blocks, self.enc_downs):
            h = blk(h, cond)
            skips.append(h)
            h = down(h)

        h = self.bottleneck(h, cond)

        for up, blk, skip in zip(self.dec_ups, self.dec_blocks, reversed(skips)):
            h = up(h)
            # align lengths for skip connection
            if h.shape[-1] != skip.shape[-1]:
                diff = skip.shape[-1] - h.shape[-1]
                if diff > 0:
                    h = F.pad(h, (0, diff))
                else:
                    h = h[..., :skip.shape[-1]]
            h = blk(torch.cat([h, skip], dim=1), cond)

        residual = self.out_conv(h)
        if residual.shape[-1] != T:
            residual = F.interpolate(residual, size=T, mode="linear",
                                     align_corners=False)
        return torch.tanh(x + residual)

    # ------------------------------------------------------------------ #
    #  Convenience helpers                                                 #
    # ------------------------------------------------------------------ #

    @classmethod
    def load(cls, path: str | Path, device: str = "cpu") -> "MasteringNetwork":
        ckpt = torch.load(path, map_location=device)
        model = cls(**ckpt.get("config", {}))
        model.load_state_dict(ckpt["state_dict"])
        model.eval()
        return model

    def save(self, path: str | Path, config: dict | None = None):
        torch.save({
            "state_dict": self.state_dict(),
            "config": config or {"cond_dim": 128, "n_stages": 4},
        }, path)

    @torch.no_grad()
    def process(self, audio: np.ndarray, genre: str = "hiphop",
                sr: int = 44100, chunk_s: float = 30.0,
                overlap_s: float = 2.0) -> np.ndarray:
        """
        Process a stereo numpy array through the mastering network.

        Args:
            audio:    (2, T) float32 stereo waveform
            genre:    genre string (see GENRES list)
            sr:       sample rate
            chunk_s:  processing chunk in seconds
            overlap_s: overlap/crossfade in seconds
        Returns:
            (2, T) mastered waveform
        """
        device = next(self.parameters()).device
        genre_idx = torch.tensor([GENRE2IDX.get(genre, GENRE2IDX["other"])],
                                 dtype=torch.long, device=device)

        if audio.ndim == 1:
            audio = np.stack([audio, audio])

        T = audio.shape[-1]
        chunk = int(chunk_s * sr)
        hop   = int((chunk_s - overlap_s) * sr)
        fade  = int(overlap_s * sr)

        if T <= chunk:
            t = torch.from_numpy(audio[None].astype(np.float32)).to(device)
            return self(t, genre_idx).squeeze(0).cpu().numpy()

        out  = np.zeros_like(audio)
        norm = np.zeros(T, dtype=np.float32)
        win  = np.ones(chunk, dtype=np.float32)
        win[:fade]  = np.linspace(0, 1, fade)
        win[-fade:] = np.linspace(1, 0, fade)

        pos = 0
        while pos < T:
            end = min(pos + chunk, T)
            seg = audio[:, pos:end]
            pad = chunk - seg.shape[-1]
            if pad:
                seg = np.pad(seg, ((0, 0), (0, pad)))
            t = torch.from_numpy(seg[None].astype(np.float32)).to(device)
            est = self(t, genre_idx).squeeze(0).cpu().numpy()
            L = end - pos
            w = win[:L]
            out[:, pos:end]  += est[:, :L] * w
            norm[pos:end]    += w
            pos += hop

        norm = np.maximum(norm, 1e-8)
        out /= norm[None, :]
        return np.clip(out, -1.0, 1.0).astype(np.float32)
