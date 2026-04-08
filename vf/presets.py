"""
vf/presets.py — Genre-tuned parameter presets for the VocalFusion Optuna optimizer.

Each preset provides starting values within the bounds defined in PARAM_SPACE
(vf/optimizer.py).  The optimizer is free to explore beyond these values; presets
are used to seed the first trial and to warm-start the search in a musically
sensible region.

PARAM_SPACE bounds (for reference / validation):
    vocal_level_mult : (0.5,  2.0)
    carve_db         : (0.0,  8.0)
    sidechain_depth  : (0.20, 0.70)
    de_ess_strength  : (0.5,  2.0)
    bass_shelf_gain  : (-4.0, 4.0)
    air_boost        : (0.0,  6.0)
    comp_attack_ms   : (0.5,  20.0)
    comp_release_ms  : (30.0, 200.0)
    exciter_drive    : (0.5,  3.0)
    char_tilt_db     : (-3.0, 3.0)
"""

# ---------------------------------------------------------------------------
# Genre labels
# ---------------------------------------------------------------------------

GENRE_LABELS: list[str] = ["pop", "hiphop", "rnb", "edm", "rock", "acoustic"]

# ---------------------------------------------------------------------------
# Presets
# ---------------------------------------------------------------------------

GENRE_PRESETS: dict[str, dict[str, float]] = {
    # ------------------------------------------------------------------
    # Pop — balanced mix with vocals slightly forward.
    # Moderate compression, light air, subtle brightness.
    # ------------------------------------------------------------------
    "pop": {
        "vocal_level_mult": 1.1,
        "carve_db":         3.0,
        "sidechain_depth":  0.4,
        "de_ess_strength":  1.0,
        "bass_shelf_gain":  0.0,
        "air_boost":        2.0,
        "comp_attack_ms":   4.0,
        "comp_release_ms":  100.0,
        "exciter_drive":    1.4,
        "char_tilt_db":     0.5,
    },

    # ------------------------------------------------------------------
    # Hip-hop — punchy low end, heavy sidechain pumping, glued vocals.
    # Slower attack lets transients breathe; de-esser eased off for
    # consonant clarity in rap delivery.
    # ------------------------------------------------------------------
    "hiphop": {
        "vocal_level_mult": 0.9,
        "carve_db":         3.5,
        "sidechain_depth":  0.65,
        "de_ess_strength":  0.8,
        "bass_shelf_gain":  2.5,
        "air_boost":        1.0,
        "comp_attack_ms":   8.0,
        "comp_release_ms":  120.0,
        "exciter_drive":    1.6,
        "char_tilt_db":     0.0,
    },

    # ------------------------------------------------------------------
    # R&B — warm, intimate tone. Smooth, transparent compression with a
    # long release keeps phrasing natural.  Mild bass warmth; slight
    # dark tilt for a velvety character.
    # ------------------------------------------------------------------
    "rnb": {
        "vocal_level_mult": 1.15,
        "carve_db":         2.5,
        "sidechain_depth":  0.35,
        "de_ess_strength":  1.1,
        "bass_shelf_gain":  1.5,
        "air_boost":        1.5,
        "comp_attack_ms":   10.0,
        "comp_release_ms":  150.0,
        "exciter_drive":    1.2,
        "char_tilt_db":     -1.0,
    },

    # ------------------------------------------------------------------
    # EDM — aggressive sidechain ducking, heavy sub presence, fast
    # compression.  Extra air and exciter drive cut through dense synths.
    # ------------------------------------------------------------------
    "edm": {
        "vocal_level_mult": 1.05,
        "carve_db":         4.0,
        "sidechain_depth":  0.70,
        "de_ess_strength":  1.2,
        "bass_shelf_gain":  3.0,
        "air_boost":        3.0,
        "comp_attack_ms":   1.5,
        "comp_release_ms":  60.0,
        "exciter_drive":    2.0,
        "char_tilt_db":     0.8,
    },

    # ------------------------------------------------------------------
    # Rock — aggressive mid carve to seat vocals in a dense guitar mix.
    # Bright tilt and strong de-esser manage harshness from loud sources.
    # Fast attack clamps peaks quickly.
    # ------------------------------------------------------------------
    "rock": {
        "vocal_level_mult": 1.0,
        "carve_db":         5.0,
        "sidechain_depth":  0.45,
        "de_ess_strength":  1.5,
        "bass_shelf_gain":  0.5,
        "air_boost":        2.5,
        "comp_attack_ms":   2.0,
        "comp_release_ms":  80.0,
        "exciter_drive":    2.2,
        "char_tilt_db":     1.5,
    },

    # ------------------------------------------------------------------
    # Acoustic — gentle, natural-sounding treatment.  High vocal level
    # lets the performance breathe; minimal carve and sidechain preserve
    # dynamics; low exciter keeps tone organic; slight warmth tilt.
    # ------------------------------------------------------------------
    "acoustic": {
        "vocal_level_mult": 1.3,
        "carve_db":         1.5,
        "sidechain_depth":  0.2,
        "de_ess_strength":  0.9,
        "bass_shelf_gain":  0.5,
        "air_boost":        1.0,
        "comp_attack_ms":   12.0,
        "comp_release_ms":  180.0,
        "exciter_drive":    0.8,
        "char_tilt_db":     -0.5,
    },
}

# ---------------------------------------------------------------------------
# Accessor
# ---------------------------------------------------------------------------

def get_preset(genre: str) -> dict[str, float]:
    """Return a copy of the preset for *genre*.

    Falls back to ``'pop'`` if *genre* is not a recognised label so that the
    optimizer always receives a valid starting point.

    Args:
        genre: One of the strings in ``GENRE_LABELS``.

    Returns:
        A shallow copy of the matching preset dict so callers cannot mutate
        the module-level data.
    """
    return dict(GENRE_PRESETS.get(genre, GENRE_PRESETS["pop"]))
