"""
vf/types.py — Typed dataclasses for all VocalFusion parameter sets.

Every numeric field includes its unit and valid range in the docstring.
Ranges are intentionally conservative — the optimizer is free to push
outside them but fuser.py clips before use.
"""
from __future__ import annotations
from dataclasses import dataclass, field
from typing import Optional


# ---------------------------------------------------------------------------
# Vocal-chain parameters
# ---------------------------------------------------------------------------

@dataclass
class VocalChainParams:
    """
    DSP parameters for the per-vocal processing chain inside fuser.py.

    All *_ms values are in milliseconds.
    All *_db values are in decibels.
    Frequencies are in Hz.
    """

    # De-esser
    de_ess_strength: float = 1.0
    """Scales max_reduction_db for the split-band de-esser.
    Range [0.5, 2.0].  1.0 = 7 dB max reduction.
    2.0 = 14 dB (very aggressive). 0.5 = 3.5 dB (gentle)."""

    # FET compressor (Stage 4)
    comp_attack_ms: float = 5.0
    """FET compressor attack. Range [1.0, 15.0] ms.
    Shorter = more transient squash. Longer = more punch preserved."""

    comp_release_ms: float = 100.0
    """FET compressor release. Range [40.0, 180.0] ms.
    Shorter = faster recovery, more pumping risk.
    Longer = more glued, less responsive."""

    # Exciter (Stage 5b)
    exciter_drive: float = 1.4
    """Harmonic exciter drive scale (tanh waveshaper).
    Range [0.5, 2.5].  1.4 = gentle warmth. 2.5 = obvious saturation."""

    # Post EQ (Stage 6)
    bass_shelf_gain: float = 0.0
    """Low-shelf gain at 100 Hz. Range [-3.0, +6.0] dB.
    Positive = more body/weight. Negative = leaner/tighter."""

    air_boost: float = 0.0
    """Additive gain to the style's air_db (high shelf at 12 kHz).
    Range [-2.0, +4.0] dB."""

    char_tilt_db: float = 0.0
    """Spectral tilt: +tilt → brighter (cuts 800 Hz, boosts 6 kHz).
    -tilt → darker (boosts 800 Hz, cuts 6 kHz). Range [-3.0, +3.0] dB."""


# ---------------------------------------------------------------------------
# Mix parameters
# ---------------------------------------------------------------------------

@dataclass
class MixParams:
    """Parameters controlling how vocal and instrumental are balanced."""

    vocal_level_mult: float = 1.0
    """Multiplier on the adaptive vocal_level derived from content analysis.
    Range [0.70, 1.30].  <1 = softer vocal, >1 = louder vocal."""

    carve_db: float = 3.0
    """Spectral carve depth: how many dB to cut the beat in the vocal's
    frequency zone (Wiener soft-mask). Range [1.0, 6.0] dB."""

    sidechain_depth: float = 0.40
    """Beat ducking depth when vocal peaks arrive.
    Range [0.20, 0.70].  0.2 = subtle, 0.7 = dramatic duck."""


# ---------------------------------------------------------------------------
# Mastering parameters
# ---------------------------------------------------------------------------

@dataclass
class MasteringParams:
    """Final bus parameters."""

    lufs_target: float = -12.0
    """Integrated loudness target. Range [-16.0, -8.0] LUFS.
    Spotify normalises to -14 LUFS; -12 leaves headroom."""

    true_peak_dbtp: float = -1.0
    """True-peak ceiling (ITU-R BS.1770-4). Range [-3.0, -0.3] dBTP."""

    limiter_release_ms: float = 100.0
    """Brickwall limiter release. Range [50.0, 300.0] ms."""


# ---------------------------------------------------------------------------
# Optimizer parameters
# ---------------------------------------------------------------------------

@dataclass
class OptimParams:
    """Configuration for the Optuna-based parameter search."""

    n_trials: int = 8
    """Maximum number of fuse+score attempts. Range [1, 20]."""

    target_score: int = 85
    """Stop early if chart_score reaches this. Range [72, 100]."""

    early_stop_score: int = 90
    """If this score is reached in the first 3 trials, stop immediately."""

    early_stop_after_n: int = 3
    """Number of trials after which early_stop_score is checked."""

    timeout_s: float = 600.0
    """Hard wall-clock timeout for the entire optimization (seconds)."""

    seed: int = 42
    """Random seed for reproducible Optuna sampling."""


# ---------------------------------------------------------------------------
# Full parameter set — union of all above
# ---------------------------------------------------------------------------

@dataclass
class FuseParams:
    """Complete parameter set used by the optimizer.

    Flat dict representation compatible with fuser._PARAM_OVERRIDE.
    """
    # Mix
    vocal_level_mult: float = 1.0
    carve_db:         float = 3.0
    sidechain_depth:  float = 0.40
    # Vocal chain
    de_ess_strength:  float = 1.0
    bass_shelf_gain:  float = 0.0
    air_boost:        float = 0.0
    comp_attack_ms:   float = 5.0
    comp_release_ms:  float = 100.0
    exciter_drive:    float = 1.4
    char_tilt_db:     float = 0.0

    def to_dict(self) -> dict[str, float]:
        """Return as plain dict for fuser._PARAM_OVERRIDE."""
        return {
            "vocal_level_mult": self.vocal_level_mult,
            "carve_db":         self.carve_db,
            "sidechain_depth":  self.sidechain_depth,
            "de_ess_strength":  self.de_ess_strength,
            "bass_shelf_gain":  self.bass_shelf_gain,
            "air_boost":        self.air_boost,
            "comp_attack_ms":   self.comp_attack_ms,
            "comp_release_ms":  self.comp_release_ms,
            "exciter_drive":    self.exciter_drive,
            "char_tilt_db":     self.char_tilt_db,
        }

    @classmethod
    def from_dict(cls, d: dict) -> "FuseParams":
        return cls(**{k: float(v) for k, v in d.items() if k in cls.__dataclass_fields__})

    @classmethod
    def defaults(cls) -> "FuseParams":
        return cls()


# ---------------------------------------------------------------------------
# Result
# ---------------------------------------------------------------------------

@dataclass
class FuseResult:
    """Structured result returned from one optimization attempt."""

    attempt_n: int
    chart_score: int
    chart_grade: str
    mix_score: int
    mix_grade: str
    params: FuseParams
    radio_path: Optional[str] = None
    club_path:  Optional[str] = None
    intimate_path: Optional[str] = None
    metadata: dict = field(default_factory=dict)
    reward: float = 0.0   # weighted multi-objective reward used by optimizer


@dataclass
class OptimResult:
    """Final result of the optimization loop."""

    best_attempt: FuseResult
    all_attempts: list[FuseResult] = field(default_factory=list)
    n_trials_run: int = 0
    elapsed_s: float = 0.0
    stopped_early: bool = False
