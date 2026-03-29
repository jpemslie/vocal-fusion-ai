"""
StemPostFilter — Conv-TasNet-based stem artifact removal.

Trained on (Demucs_output, original_stem) pairs from MUSDB18-HQ.
Removes separation bleed, reduces HNR artifacts from stem separation noise.

Architecture: ~2M parameter Conv-TasNet
  - 1D encoder: kernel=16, stride=8, N=512 filters
  - TCN: 8 blocks × 3 repeats, kernel=3, dilation doubling
  - 1D decoder: transposed conv back to waveform
  - Residual skip connections

Usage:
    from ml.post_filter import StemPostFilter
    model = StemPostFilter.load("ml/post_filter_vocals.pt")
    clean = model.enhance(noisy_stem_np, sr=44100)
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from pathlib import Path


class _ChannelNorm(nn.Module):
    def __init__(self, n_feats: int, eps: float = 1e-8):
        super().__init__()
        self.eps = eps
        self.gain = nn.Parameter(torch.ones(1, n_feats, 1))
        self.bias = nn.Parameter(torch.zeros(1, n_feats, 1))

    def forward(self, x):
        mean = x.mean(dim=1, keepdim=True)
        var = ((x - mean) ** 2).mean(dim=1, keepdim=True)
        x = (x - mean) / (var + self.eps).sqrt()
        return x * self.gain + self.bias


class _DepthwiseSeparable(nn.Module):
    """Depthwise-separable dilated conv block (TCN building block)."""

    def __init__(self, in_ch: int, h_ch: int, kernel: int, dilation: int):
        super().__init__()
        pad = (kernel - 1) * dilation // 2
        self.net = nn.Sequential(
            nn.Conv1d(in_ch, h_ch, 1),
            nn.PReLU(),
            _ChannelNorm(h_ch),
            nn.Conv1d(h_ch, h_ch, kernel, padding=pad,
                      dilation=dilation, groups=h_ch),
            nn.PReLU(),
            _ChannelNorm(h_ch),
            nn.Conv1d(h_ch, in_ch, 1),
        )
        self.skip = nn.Conv1d(in_ch, in_ch, 1)

    def forward(self, x):
        return self.net(x) + self.skip(x)


class StemPostFilter(nn.Module):
    """
    Conv-TasNet stem artifact post-filter.

    Args:
        n_filters:  Encoder output channels (N)
        bottleneck: Bottleneck channels (B)
        hidden:     TCN hidden channels (H)
        skip_ch:    Skip channels (Sc)
        kernel:     TCN kernel size
        n_blocks:   TCN blocks per repeat
        n_repeats:  Number of TCN repeats
        enc_kernel: Encoder/decoder kernel size
        enc_stride: Encoder/decoder stride
    """

    def __init__(
        self,
        n_filters: int = 512,
        bottleneck: int = 128,
        hidden: int = 512,
        skip_ch: int = 128,
        kernel: int = 3,
        n_blocks: int = 8,
        n_repeats: int = 3,
        enc_kernel: int = 16,
        enc_stride: int = 8,
    ):
        super().__init__()
        self.enc_stride = enc_stride

        # Encoder: waveform → latent
        self.encoder = nn.Conv1d(1, n_filters, enc_kernel, stride=enc_stride,
                                 padding=enc_kernel // 2 - enc_stride // 2, bias=False)

        # Separator: TCN mask estimator
        self.ln = _ChannelNorm(n_filters)
        self.bottleneck_conv = nn.Conv1d(n_filters, bottleneck, 1)

        self.tcn = nn.ModuleList()
        for r in range(n_repeats):
            for b in range(n_blocks):
                dil = 2 ** b
                self.tcn.append(_DepthwiseSeparable(bottleneck, hidden, kernel, dil))

        self.skip_conv = nn.Conv1d(bottleneck, skip_ch, 1)
        self.mask_conv = nn.Sequential(
            nn.Conv1d(skip_ch, n_filters, 1),
            nn.ReLU(),
        )

        # Decoder: masked latent → waveform
        self.decoder = nn.ConvTranspose1d(
            n_filters, 1, enc_kernel, stride=enc_stride,
            padding=enc_kernel // 2 - enc_stride // 2, bias=False,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, 1, T) waveform
        Returns:
            (B, 1, T) enhanced waveform
        """
        T = x.shape[-1]
        e = F.relu(self.encoder(x))          # (B, N, L)
        n = self.ln(e)
        h = self.bottleneck_conv(n)          # (B, B, L)

        for block in self.tcn:
            h = block(h)

        skip = self.skip_conv(h)
        mask = self.mask_conv(skip)          # (B, N, L)
        out = self.decoder(e * mask)         # (B, 1, T')
        # trim/pad back to input length
        if out.shape[-1] > T:
            out = out[..., :T]
        elif out.shape[-1] < T:
            out = F.pad(out, (0, T - out.shape[-1]))
        return out

    # ------------------------------------------------------------------ #
    #  Convenience helpers                                                 #
    # ------------------------------------------------------------------ #

    @classmethod
    def load(cls, path: str | Path, device: str = "cpu") -> "StemPostFilter":
        """Load a saved checkpoint."""
        ckpt = torch.load(path, map_location=device)
        model = cls(**ckpt.get("config", {}))
        model.load_state_dict(ckpt["state_dict"])
        model.eval()
        return model

    def save(self, path: str | Path, config: dict | None = None):
        torch.save({
            "state_dict": self.state_dict(),
            "config": config or {},
        }, path)

    @torch.no_grad()
    def enhance(self, audio: np.ndarray, sr: int = 44100,
                chunk_s: float = 4.0, overlap_s: float = 0.5) -> np.ndarray:
        """
        Enhance a numpy waveform (mono or stereo).

        Processes in overlapping chunks to handle arbitrary length.
        Returns same shape as input.
        """
        device = next(self.parameters()).device
        stereo = audio.ndim == 2

        if stereo:
            left  = self.enhance(audio[0], sr, chunk_s, overlap_s)
            right = self.enhance(audio[1], sr, chunk_s, overlap_s)
            return np.stack([left, right])

        chunk = int(chunk_s * sr)
        hop   = int((chunk_s - overlap_s) * sr)
        fade  = int(overlap_s * sr)

        if len(audio) <= chunk:
            t = torch.from_numpy(audio[None, None, :].astype(np.float32)).to(device)
            return self(t).squeeze().cpu().numpy()

        out = np.zeros_like(audio)
        norm = np.zeros_like(audio)
        win = np.ones(chunk, dtype=np.float32)
        win[:fade] = np.linspace(0, 1, fade)
        win[-fade:] = np.linspace(1, 0, fade)

        pos = 0
        while pos < len(audio):
            end = min(pos + chunk, len(audio))
            seg = audio[pos:end]
            pad = chunk - len(seg)
            if pad:
                seg = np.pad(seg, (0, pad))
            t = torch.from_numpy(seg[None, None, :].astype(np.float32)).to(device)
            enhanced = self(t).squeeze().cpu().numpy()
            w = win if not pad else win[:len(audio) - pos]
            out[pos:end]  += enhanced[:end - pos] * w[:end - pos]
            norm[pos:end] += w[:end - pos]
            pos += hop

        out /= np.maximum(norm, 1e-8)
        return out


def build_model(stem: str = "vocals") -> StemPostFilter:
    """Build a fresh (untrained) model for a given stem type."""
    # Larger capacity for vocals (more complex signal)
    if stem == "vocals":
        return StemPostFilter(n_filters=512, bottleneck=128, hidden=512,
                              n_blocks=8, n_repeats=3)
    # Lighter model for bass/drums (lower complexity)
    return StemPostFilter(n_filters=256, bottleneck=64, hidden=256,
                          n_blocks=6, n_repeats=2)
