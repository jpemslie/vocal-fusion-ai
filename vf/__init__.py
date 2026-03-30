"""
VocalFusion internal package.

Modules extracted from fuser.py to improve testability and maintainability:
  vf.constants       — SR, model names, grade thresholds
  vf.utils           — Pure-numpy audio utilities (_rms, _to_mono, M/S, file ID)
  vf.stem_separation — Stem separation (Demucs / BSRoformer / MDX-Net)

fuser.py remains the orchestration entry point and imports from these modules.
"""
