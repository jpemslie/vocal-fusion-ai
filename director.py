"""
VocalFusion AI Director
=======================
Uses Claude API to reason about mix quality and decide parameter adjustments.

Replaces the static CORRECTIONS lookup table with genuine AI reasoning:
- Reads all 23 quality metrics and understands causal relationships
- Recognizes when the same symptom has different root causes
- Tracks history across correction iterations to avoid repeating failed fixes
- Can adjust parameters that rule-based code can't diagnose correctly

Usage (from fuser.py auto-correction loop):
    from director import get_corrections
    adj = get_corrections(metrics, issues, current_params, attempt_num, history)
    # adj = {"carve_db": 8.5, "vocal_level": 2.3, ...} absolute values
"""

import json
import os

# ── Tunable parameter bounds ───────────────────────────────────────────────────
PARAM_BOUNDS = {
    "carve_db":              (3.0,  12.0),
    "presence_db":           (0.0,   4.0),
    "air_db":                (0.0,   5.0),
    "vocal_level":           (1.0,   4.0),
    "lufs_target":          (-16.0,  -9.0),
    "harmonic_mix_dry":      (0.30,   0.70),
    "wiener_mask_floor":     (0.04,   0.20),
    "drum_weight_hh":        (1.0,    5.0),
    "hihat_alpha":           (0.50,   0.95),
    "noisereduce_strength":  (0.05,   0.50),
}

# ── System prompt ──────────────────────────────────────────────────────────────
_SYSTEM = """You are the AI audio director for VocalFusion, a professional AI mashup tool.
Your job: read mix quality metrics, diagnose root causes of problems, and output specific
parameter adjustments to make the mix sound more professional.

═══ SIGNAL CHAIN (in order) ═══

1. STEM SEPARATION (htdemucs_ft 4-stem): Song B → vocals / drums / bass / other
2. ORACLE WIENER MASK (_oracle_wiener_clean):
   - Uses all 4 stems as oracle interference: mask = V² / (V² + D_weighted² + B² + O²)
   - drum_weight_hh: drums are weighted N× heavier in 4-16kHz hi-hat band
   - wiener_mask_floor: minimum mask value (floor prevents complete suppression)
   - Lower floor + higher drum_weight → more aggressive bleed removal
   - Risk: too aggressive removes sibilants (S/SH/CH), makes vocal thin
3. TARGETED HI-HAT SUBTRACTION (_targeted_hihat_suppression):
   - Spectral subtraction in 6-16kHz: mag_out = max(|V| - α·|D|, β·|V|)
   - hihat_alpha: subtraction strength (0.90 = strong, 0.50 = gentle)
   - Higher alpha → more hi-hat removed, but risk of punching holes in sibilants
4. HARMONIC RESYNTHESIS (_harmonic_vocal_process):
   - PYIN tracks vocal F0 → keeps only harmonic bins, suppresses inter-harmonic above 4kHz
   - harmonic_mix_dry: dry/wet blend (0.50 = 50% processed + 50% original)
   - Lower harmonic_mix_dry → more harmonic-only → cleaner but can sound synthetic
   - Higher harmonic_mix_dry → more original → natural but passes more bleed
   - Suppression only above 4kHz — below 4kHz is voice character (formants, vowels)
5. REFERENCE NOISEREDUCE (noisereduce with song B instrumental as noise profile):
   - noisereduce_strength: prop_decrease (0.15 = light touch, 0.50 = heavy)
   - Heavier NR can introduce artifacts on complex music stems
6. VOCAL PROCESSING CHAIN (_process_vocals, 8 stages):
   - HPF 80Hz, subtractive EQ (mud/boxy notches)
   - De-esser (before compression, prevents sibilance pumping)
   - FET compressor (ratio: 2.2:1 non-rap → 6:1 rap)
   - Noise gate
   - Parallel tape saturation (20% wet tanh)
   - Presence peak (presence_db dB @ 3kHz) + air shelf (air_db @ 10kHz)
   - Short reverb (5-8% wet, HPF'd return)
7. MIXING (_iterative_mix, 2 iterations):
   - Energy match: vocal vs inst mid-RMS → target 45-70% presence
   - vocal_level: level multiplier (1.0-4.0), sets how loud vocal sits in mix
   - Spectral carve (carve_db): Wiener mask cuts beat spectrum in vocal range
   - M/S encode: vocal → Mid only
   - Sidechain compression on beat when vocal is present
8. MASTERING (_master):
   - lufs_target: mastering loudness (typically -12 LUFS)
   - Brick-wall limiter at -2.0 dBFS

═══ DIAGNOSTIC DECISION TREES ═══

VOCAL_BLEED_SCORE too high (> 0.40):
  IF vocal_spectral_crest is also low (<4.0): root cause = bleed surviving harmonic resynth
    → lower harmonic_mix_dry (more harmonic skeleton, less original bleed)
    → raise drum_weight_hh (stronger Wiener suppression in hi-hat band)
    → lower wiener_mask_floor (more aggressive Wiener)
  IF vocal_spectral_crest is in range: root cause = broadband noise, not hi-hat
    → raise noisereduce_strength (NR targets broadband noise)
    → lower hihat_alpha slightly (over-subtraction creating rough artifacts)
  NEVER raise carve_db for bleed — carve only affects the beat, not the vocal stem

VOCAL_SPECTRAL_CREST too low (< 4.0):
  IF bleed_score is also high: see above (bleed cause)
  IF bleed_score is OK but crest is low: compression squashing harmonic structure
    → raise presence_db (boost harmonic presence zone 3kHz)
    → raise air_db (open up harmonic overtones)
  IF both bleed and compression look OK: harmonic_mix_dry too high
    → lower harmonic_mix_dry (lean more on PYIN skeleton)

VOCAL_MODULATION_INDEX too low (< 0.20):
  Syllable rhythm is lost — vocal sounds like noise, not speech
  → raise vocal_level (vocal buried in mix, rhythm masked)
  → lower carve_db (over-carving is removing vocal body from the beat's perspective)
  DO NOT raise noisereduce_strength — NR can smear syllable onsets

VOCAL_MODULATION_INDEX too high (> 0.65):
  Vocal envelope is choppy/gated — gate or sidechain too aggressive
  → this is a symptom of the noise gate opening/closing rapidly
  → raise wiener_mask_floor (prevents total suppression between syllables)
  → lower noisereduce_strength

VOCAL_CLARITY_INDEX too low (< -5.0):
  Bass masking the voice — vocal intelligibility zone (1-4kHz) buried under low end
  → raise carve_db (cut more from beat in vocal range)
  → raise presence_db
  → lower lufs_target (louder = more bass energy overwhelming voice)

LRA too low (< 3.5 LU):
  Dynamics squashed — sounds over-compressed
  → lower lufs_target slightly (more headroom, less limiting compression)
  → this is NOT fixable via mix params — it's a mastering chain property

LRA too high (> 14 LU):
  Too dynamic — quiet sections too quiet
  → raise lufs_target

LUFS out of range:
  → adjust lufs_target directly

BEAT_SYNC_SCORE too low (< 0.35):
  Beat and vocal not aligning — onset timing mismatch
  → raise vocal_level (if vocal is too quiet, its onsets don't show in the cross-correlation)
  → this is mostly an alignment issue not fixable via EQ/level

═══ KEY RULES ═══
1. Output ABSOLUTE parameter values, not deltas. The loop applies your values directly.
2. Only output parameters that need changing. Don't touch working parameters.
3. Consider interaction effects: raising presence_db + carve_db simultaneously can over-
   brighten the mix. Raising vocal_level + lowering carve_db can make the vocal dominate.
4. Check the history — if a change was tried and score went down, don't repeat it.
5. If score is already ≥ 82/100, be conservative — small adjustments only.
6. If multiple metrics fail, fix the most impactful first. Don't over-correct.
7. Bounds are enforced by the caller — you can request values at the boundary safely.
"""

_USER_TMPL = """
CURRENT MIX METRICS (from listen.py — 23 metrics):
{metrics_block}

CURRENT PARAMETER STATE:
{params_block}

ISSUES DETECTED (severity / metric / value / target range / description):
{issues_block}

CORRECTION HISTORY (this fuse session):
{history_block}

Attempt {attempt_num} of 3. Score must reach 82/100 to pass.

Diagnose the root cause(s) of the failing metrics and output parameter adjustments.
Respond ONLY with a JSON object in this exact format:
{{
  "reasoning": "your concise diagnosis and rationale (2-4 sentences)",
  "adjustments": {{
    "param_name": absolute_value,
    ...
  }}
}}
"""


def get_corrections(
    metrics: dict,
    issues: list,
    current_params: dict,
    attempt_num: int,
    history: list,
) -> dict:
    """
    Call Claude API to diagnose mix issues and return parameter adjustments.

    Parameters
    ----------
    metrics : dict
        Full metrics dict from listen._measure() — all 23 keys.
    issues : list
        List of (sev, key, val, lo, hi, desc) tuples from listen._score().
    current_params : dict
        Current values of all tunable parameters (see PARAM_BOUNDS).
    attempt_num : int
        Which iteration this is (0-indexed). Passed to the model for context.
    history : list
        List of {"attempt": int, "score": int, "params": dict, "issues": [str]}
        from prior correction attempts in this fuse() call.

    Returns
    -------
    dict
        Mapping of param_name → absolute value. Empty dict if no changes needed
        or if the API call fails (caller falls back to lookup table).
    """
    api_key = os.environ.get("ANTHROPIC_API_KEY", "")
    if not api_key:
        # No API key — use built-in causal decision tree (understands root causes,
        # not just metric→delta mapping like the simple lookup table)
        return _causal_corrections(metrics, issues, current_params, history)

    try:
        import anthropic
    except ImportError:
        return {}

    # ── Format prompt sections ─────────────────────────────────────────────────
    metrics_block = _format_metrics(metrics)
    params_block  = _format_params(current_params)
    issues_block  = _format_issues(issues)
    history_block = _format_history(history)

    user_msg = _USER_TMPL.format(
        metrics_block=metrics_block,
        params_block=params_block,
        issues_block=issues_block,
        history_block=history_block,
        attempt_num=attempt_num + 1,
    )

    # ── API call ───────────────────────────────────────────────────────────────
    try:
        client   = anthropic.Anthropic(api_key=api_key)
        response = client.messages.create(
            model="claude-sonnet-4-6",
            max_tokens=512,
            system=_SYSTEM,
            messages=[{"role": "user", "content": user_msg}],
        )
        raw = response.content[0].text.strip()

        # Strip markdown code fence if present
        if raw.startswith("```"):
            raw = raw.split("```")[1]
            if raw.startswith("json"):
                raw = raw[4:]
        raw = raw.strip()

        parsed = json.loads(raw)
        reasoning = parsed.get("reasoning", "")
        adjustments = parsed.get("adjustments", {})

        if reasoning:
            print(f"\n  [AI Director] {reasoning}", flush=True)

        # ── Clamp to safe bounds ───────────────────────────────────────────────
        clamped = {}
        for k, v in adjustments.items():
            if k not in PARAM_BOUNDS:
                print(f"  [AI Director] Unknown param '{k}' — skipping.", flush=True)
                continue
            lo, hi = PARAM_BOUNDS[k]
            clamped[k] = float(max(lo, min(hi, float(v))))

        return clamped

    except (json.JSONDecodeError, KeyError, IndexError) as e:
        print(f"  [AI Director] Parse error ({e}) — falling back to lookup table.",
              flush=True)
        return {}
    except Exception as e:
        print(f"  [AI Director] API error ({type(e).__name__}: {e}) — "
              f"falling back to lookup table.", flush=True)
        return {}


# ── Formatting helpers ─────────────────────────────────────────────────────────

def _format_metrics(m: dict) -> str:
    lines = []
    fields = [
        ("lufs_integrated",       "LUFS integrated",          "dB",   "target -14 to -7"),
        ("true_peak_dbfs",        "True peak",                "dBFS", "must be < -0.5"),
        ("lra_lu",                "LRA",                      "LU",   "target 3.5-14"),
        ("crest_factor_db",       "Crest factor",             "dB",   "> 8 = punchy"),
        ("stereo_correlation",    "Stereo correlation",        "",     "0.4-0.99"),
        ("transient_clarity",     "Transient clarity",         "dB",   "target 8-28 dB"),
        ("kick_headroom_db",      "Kick headroom",             "dB",   "target 8-32 dB"),
        ("section_consistency_lu","Section consistency",       "LU",   "target < 5 LU"),
        ("spectral_slope_db_oct", "Spectral slope",            "dB/oct","target -12 to -2"),
        ("beat_sync_score",       "Beat sync score",           "",     "target 0.35-1.0"),
        ("vocal_clarity_index",   "Vocal clarity index",       "dB",   "target -5 to +20"),
        ("tempo_stability",       "Tempo stability",           "",     "target 0.6-1.0"),
        ("click_artifact_score",  "Click artifact score",      "",     "target < 0.005"),
        ("vocal_bleed_score",     "Vocal bleed score",         "",     "target < 0.40"),
        ("vocal_spectral_crest",  "Vocal spectral crest",      "",     "target 4-30"),
        ("vocal_modulation_index","Vocal modulation index",    "",     "target 0.20-0.65"),
        ("bass_vs_mid",           "Bass vs Mid ratio",         "dB",   "target +2 to +15"),
        ("lomid_vs_mid",          "Lo-Mid vs Mid ratio",       "dB",   "target -3 to +8"),
        ("himid_vs_mid",          "Hi-Mid vs Mid ratio",       "dB",   "target -20 to -3"),
        ("high_vs_mid",           "High vs Mid ratio",         "dB",   "target -32 to -8"),
        ("mud_index",             "Mud index",                 "",     "target 1.0-5.5"),
    ]
    for key, label, unit, note in fields:
        if key in m:
            val = m[key]
            lines.append(f"  {label:<30} {val:>+9.3f} {unit:<5}  ({note})")
    return "\n".join(lines)


def _format_params(p: dict) -> str:
    lines = []
    for k, v in p.items():
        lo, hi = PARAM_BOUNDS.get(k, (None, None))
        bound_str = f"  [bounds: {lo} – {hi}]" if lo is not None else ""
        lines.append(f"  {k:<28} = {v:.3f}{bound_str}")
    return "\n".join(lines)


def _format_issues(issues: list) -> str:
    if not issues:
        return "  (none — all metrics in range)"
    lines = []
    for sev, key, val, lo, hi, desc in issues:
        lines.append(f"  [{sev:<8}] {key:<30} val={val:.3f}  target=[{lo:.2f}, {hi:.2f}]  {desc}")
    return "\n".join(lines)


def _format_history(history: list) -> str:
    if not history:
        return "  (first attempt — no prior history)"
    lines = []
    for h in history:
        p_str = ", ".join(f"{k}={v:.3f}" for k, v in h.get("params", {}).items())
        issues_str = ", ".join(h.get("issues", [])) or "none"
        lines.append(
            f"  Attempt {h['attempt']+1}: score={h['score']}/100  "
            f"params=[{p_str}]  failing=[{issues_str}]"
        )
    return "\n".join(lines)


def _causal_corrections(metrics: dict, issues: list, current_params: dict,
                         history: list) -> dict:
    """
    Causal decision tree — works without Claude API key.

    Encodes the same diagnostic logic as the Claude system prompt:
    identifies ROOT CAUSES rather than blindly mapping metric→delta.
    Checks history to avoid repeating failed adjustments.

    Returns absolute param values (same format as Claude director output).
    """
    adj = {}
    issue_keys = {i[1] for i in issues}
    m = metrics
    p = current_params

    # Helper: get previously tried values for a param, to avoid repeating failures
    def _tried(param):
        return [h["params"].get(param) for h in history if param in h.get("params", {})]

    def _set(param, val):
        lo, hi = PARAM_BOUNDS.get(param, (None, None))
        if lo is not None:
            val = max(lo, min(hi, val))
        # Don't repeat a value we already tried (within 5% tolerance)
        for tried_val in _tried(param):
            if tried_val is not None and abs(tried_val - val) / (abs(tried_val) + 1e-6) < 0.05:
                return  # already tried this — skip
        adj[param] = val

    bleed  = m.get("vocal_bleed_score", 0.0)
    crest  = m.get("vocal_spectral_crest", 10.0)
    modul  = m.get("vocal_modulation_index", 0.35)
    clarity = m.get("vocal_clarity_index", 5.0)
    lra    = m.get("lra_lu", 8.0)
    lufs   = m.get("lufs_integrated", -12.0)

    # ── Vocal bleed too high ────────────────────────────────────────────────
    if "vocal_bleed_score" in issue_keys:
        if crest < 4.0:
            # Bleed survived harmonic resynth — tighten the source (bleed cause)
            _set("harmonic_mix_dry",   max(0.30, p.get("harmonic_mix_dry", 0.50) - 0.10))
            _set("drum_weight_hh",     min(5.0,  p.get("drum_weight_hh", 3.5) + 0.7))
            _set("wiener_mask_floor",  max(0.04, p.get("wiener_mask_floor", 0.08) - 0.02))
        else:
            # Spectral crest OK → broadband noise, not hi-hat bleed
            _set("noisereduce_strength", min(0.40, p.get("noisereduce_strength", 0.15) + 0.10))
            _set("hihat_alpha",          min(0.95, p.get("hihat_alpha", 0.90) + 0.03))

    # ── Vocal spectral crest too low (vocal sounds flat/noisy) ─────────────
    if "vocal_spectral_crest" in issue_keys and bleed <= 0.40:
        # Crest low but bleed OK → compression squashing harmonic peaks
        _set("presence_db", min(4.0, p.get("presence_db", 2.0) + 0.8))
        _set("air_db",      min(5.0, p.get("air_db", 2.5) + 0.5))

    # ── Vocal modulation too low (rhythm/syllables lost) ───────────────────
    if "vocal_modulation_index" in issue_keys and modul < 0.20:
        _set("vocal_level", min(4.0, p.get("vocal_level", 2.1) + 0.20))
        _set("carve_db",    max(3.0, p.get("carve_db", 10.0) - 1.0))

    # ── Vocal modulation too high (choppy/gated sound) ─────────────────────
    if "vocal_modulation_index" in issue_keys and modul > 0.65:
        _set("wiener_mask_floor",  min(0.15, p.get("wiener_mask_floor", 0.08) + 0.03))
        _set("noisereduce_strength", max(0.05, p.get("noisereduce_strength", 0.15) - 0.05))

    # ── Vocal clarity too low (bass masking voice) ──────────────────────────
    if "vocal_clarity_index" in issue_keys and clarity < -5.0:
        _set("carve_db",    min(12.0, p.get("carve_db", 10.0) + 1.5))
        _set("presence_db", min(4.0,  p.get("presence_db", 2.0) + 0.5))

    # ── LUFS out of range ───────────────────────────────────────────────────
    if "lufs_integrated" in issue_keys:
        if lufs < -14.0:
            _set("lufs_target", min(-9.0, p.get("lufs_target", -12.0) + 1.5))
        elif lufs > -7.0:
            _set("lufs_target", max(-16.0, p.get("lufs_target", -12.0) - 1.5))

    # ── LRA too low (over-compressed) ───────────────────────────────────────
    if "lra_lu" in issue_keys and lra < 3.5:
        _set("lufs_target", max(-16.0, p.get("lufs_target", -12.0) - 1.0))

    # ── LRA too high (too dynamic) ──────────────────────────────────────────
    if "lra_lu" in issue_keys and lra > 14.0:
        _set("lufs_target", min(-9.0, p.get("lufs_target", -12.0) + 1.0))

    if adj:
        print(f"  [Built-in director] Causal corrections: {adj}", flush=True)
    return adj
