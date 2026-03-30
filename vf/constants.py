"""
VocalFusion shared constants.

Centralises SR and model names so they can be imported by both
fuser.py and vf.stem_separation without circular dependency.
"""

import os

# ---------------------------------------------------------------------------
# Audio sample rate — everything in the engine runs at 44.1 kHz
# ---------------------------------------------------------------------------
SR: int = 44100

# ---------------------------------------------------------------------------
# Stem separation model names (audio-separator checkpoint filenames)
# ---------------------------------------------------------------------------

# Primary vocal separation (Mel-Band RoFormer, SDR ~11.4)
MBR_VOCAL      = "model_mel_band_roformer_ep_3005_sdr_11.4360.ckpt"

# Secondary vocal fallback (BS-RoFormer, SDR ~12.97)
BS_ROFORMER    = "model_bs_roformer_ep_317_sdr_12.9755.ckpt"

# MDX-Net models for CPU-path ensemble
MDX_VOCAL      = "Kim_Vocal_2.onnx"
MDX_VOCAL_2    = "UVR-MDX-NET-Voc_FT.onnx"
MDX23C_VOCAL   = "MDX23C-8KFFT-InstVoc_HQ.ckpt"

# Post-separation neural cleaning
DENOISE_MODEL  = "denoise_mel_band_roformer_aufr33_sdr_27.9959.ckpt"   # SDR 27.99
DEVERB_MODEL   = "deverb_bs_roformer_8_384dim_10depth.ckpt"

# ---------------------------------------------------------------------------
# Quality scorer grade thresholds (shared by listen.py, ml/scorer.py, etc.)
# ---------------------------------------------------------------------------
GRADES = [
    (85, "A"),
    (70, "B"),
    (55, "C"),
    (35, "D"),
    ( 0, "F"),
]


def grade(score: float) -> str:
    """Return letter grade for a 0-100 score."""
    for threshold, label in GRADES:
        if score >= threshold:
            return label
    return "F"


# ---------------------------------------------------------------------------
# Default directory paths (overridable via environment variables)
# ---------------------------------------------------------------------------
STEMS_DIR  = os.environ.get("VF_STEMS_DIR",  "vf_data/stems")
OUTPUT_DIR = os.environ.get("VF_OUTPUT_DIR", "vf_data/mixes")
UPLOAD_DIR = os.environ.get("VF_UPLOAD_DIR", "vf_data/uploads")
