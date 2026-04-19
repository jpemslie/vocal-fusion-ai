"""
autonomous_songwriter.py
VocalFusion — Autonomous Song Generator  v1
============================================
Generates complete original songs from scratch using pre-trained AI models.
No manual input required.

Pipeline:
  1. Generate prompt  (random or user-supplied)
  2. Generate instrumental  (MusicGen)
  3. Detect key + tempo from instrumental  (librosa)
  4. Generate lyrics  (template system, key-aware)
  5. Generate vocals  (Bark TTS with singing prompts)
  6. Fuse with fuser.py  (professional mixing chain)
  7. Score with listen.py  (chart readiness)
  8. Self-evaluate: if chart_score < 85, adjust params and retry
  9. Save best result to vf_data/autogen/

Usage:
  python autonomous_songwriter.py                     # generate 1 song
  python autonomous_songwriter.py --generate-n 5     # generate 5 songs
  python autonomous_songwriter.py --prompt "dark trap beat with heavy 808" --n-attempts 8
  python autonomous_songwriter.py --genre pop         # force genre
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import random
import sys
import time
import uuid
from pathlib import Path
from typing import Optional

import librosa
import numpy as np
import soundfile as sf
from scipy.signal import resample_poly

# ── Paths ─────────────────────────────────────────────────────────────────────
ROOT       = Path(__file__).parent
AUTOGEN    = ROOT / "vf_data" / "autogen"
STEMS_DIR  = ROOT / "vf_data" / "stems"
AUTOGEN.mkdir(parents=True, exist_ok=True)
STEMS_DIR.mkdir(parents=True, exist_ok=True)

SR_TARGET = 44100   # fuser.py expects 44.1 kHz

# ── Logging ───────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("autosong")

# ---------------------------------------------------------------------------
# Prompt library
# ---------------------------------------------------------------------------
GENRE_PROMPTS: dict[str, list[str]] = {
    "trap": [
        "dark trap beat, heavy 808 bass, hi-hat rolls, minor key, 140 bpm",
        "melodic trap instrumental, distorted 808, piano melody, atmospheric pads, 145 bpm",
        "hard trap beat, punchy snare, rolling hi-hats, heavy sub bass, 140 bpm",
        "dark drill beat, sliding 808, sparse hi-hats, ominous strings, 145 bpm",
        "trap soul beat, emotional piano, 808 bass, atmospheric reverb, 140 bpm",
    ],
    "pop": [
        "upbeat pop instrumental, catchy synth melody, four-on-the-floor kick, bright pads, 120 bpm",
        "modern pop beat, punchy 808, synth arpeggios, clap on 2 and 4, energetic, 128 bpm",
        "dream pop instrumental, lush reverb, clean electric piano, gentle kick, 110 bpm",
        "electro pop beat, bright synth lead, four-on-the-floor, vocal chops, 125 bpm",
        "indie pop instrumental, acoustic guitar, electronic drums, warm bass, 118 bpm",
    ],
    "hiphop": [
        "boom bap hip hop beat, classic drum break, jazz sample chops, deep bass, 90 bpm",
        "lo-fi hip hop instrumental, dusty drum loop, warm piano, vinyl crackle, 85 bpm",
        "hard hip hop beat, heavy snare, punchy kick, dark sample, 95 bpm",
        "west coast hip hop instrumental, smooth G-funk synth, laid-back groove, 95 bpm",
        "east coast boom bap, jazzy horn sample, crisp snare, deep bass, 90 bpm",
    ],
    "rnb": [
        "smooth R&B instrumental, warm bass, soulful chords, gentle hi-hats, 90 bpm",
        "contemporary R&B beat, 808 bass, lush pads, melodic piano, 95 bpm",
        "neo soul instrumental, warm Rhodes, live-sounding drums, jazzy bass, 88 bpm",
        "chill R&B beat, trap hi-hats, warm sub, emotional strings, 100 bpm",
        "alternative R&B instrumental, layered synths, off-beat percussion, 93 bpm",
    ],
    "edm": [
        "progressive house instrumental, building synths, four-on-the-floor kick, emotional drop, 128 bpm",
        "future bass drop, heavy kick, lush chords, saw synths, energetic, 140 bpm",
        "deep house groove, warm bass, rolling kick, hypnotic synth loop, 124 bpm",
        "melodic dubstep instrumental, emotional lead, heavy bass wobble, 140 bpm",
        "tropical house instrumental, steel drums, clean kick, bright atmosphere, 115 bpm",
    ],
}

ALL_GENRES = list(GENRE_PROMPTS.keys())

# Lyric building blocks per genre
LYRIC_HOOKS: dict[str, list[str]] = {
    "trap": [
        "I came from the bottom, never looked back",
        "Money on my mind, grinding every night",
        "Streets made me who I am today",
        "They doubted me but I kept on rising",
        "Drip too hard, they can't deny it",
    ],
    "pop": [
        "I found the light when everything went dark",
        "Dancing through the night with you",
        "We don't need to slow down, keep going",
        "Every time you call my name I feel alive",
        "We're burning bright like stars tonight",
    ],
    "hiphop": [
        "Real recognize real, you know the deal",
        "From the ground up, built this with my hands",
        "Watch me level up, they can't hold me down",
        "I speak my truth through every single verse",
        "Knowledge is power, I keep on learning",
    ],
    "rnb": [
        "You got me feeling like I'm flying",
        "Every time you touch me I lose my mind",
        "Baby stay right here beside me tonight",
        "Your love is like a melody I can't shake",
        "I never knew what love was till I found you",
    ],
    "edm": [
        "Lose yourself in the sound tonight",
        "We are alive, feel it in your bones",
        "Let the music take control of your soul",
        "Rise up, feel the energy around us",
        "Chase the light until the morning comes",
    ],
}

LYRIC_VERSE_LINES: dict[str, list[str]] = {
    "trap": [
        "Started from the bottom, worked my way to the top",
        "All these fake people, I just had to stop",
        "Count my blessings every single day",
        "Staying focused, never losing my way",
        "Built my empire from the ground up",
        "They said I couldn't do it, now they shut up",
        "Every setback was a setup for success",
        "Moving silent, never stopping to rest",
    ],
    "pop": [
        "I remember when you said we'd make it through",
        "Now the world is ours and the sky is blue",
        "Every moment feels like a dream come true",
        "Nothing's gonna stop us, me and you",
        "We were made for moments just like this",
        "Feeling every heartbeat in your kiss",
        "Time is slipping but we're holding on",
        "Every day with you is a brand new song",
    ],
    "hiphop": [
        "Been through the struggle, came out stronger",
        "Every day I grind a little longer",
        "My words are weapons, my mind is my armor",
        "Never compromise on my true character",
        "I walk a different path than most people see",
        "Authentic to the core, staying true to me",
        "The system tried to break me but I rose above",
        "Everything I do is done with passion and love",
    ],
    "rnb": [
        "When I see you my whole world slows down",
        "You're the only one to lift me when I'm down",
        "Every word you say feels like a song",
        "With you beside me I know I belong",
        "You came into my life and changed it all",
        "Now I catch myself before I fall",
        "You're my remedy when everything's wrong",
        "Our story keeps on going, going strong",
    ],
    "edm": [
        "Feel the bass line hit your chest right now",
        "Lose the world and just forget somehow",
        "Every beat drop takes us higher",
        "Dancing til the morning, never tire",
        "Hands up to the sky, feel the vibe",
        "In this moment we are all alive",
        "The music takes us somewhere far away",
        "We'll be here dancing until the day",
    ],
}


def _build_lyrics(genre: str, key_note: str, tempo: float) -> str:
    """Generate lyrics for the given genre/key/tempo."""
    rng = random.Random()
    hook_lines  = LYRIC_HOOKS.get(genre, LYRIC_HOOKS["pop"])
    verse_lines = LYRIC_VERSE_LINES.get(genre, LYRIC_VERSE_LINES["pop"])

    hook  = rng.choice(hook_lines)
    verse = rng.sample(verse_lines, min(4, len(verse_lines)))

    # Build structure: [Intro hook] + [Verse] + [Hook x2]
    structure = [
        "♪ " + hook + " ♪",
        verse[0],
        verse[1],
        verse[2] if len(verse) > 2 else verse[0],
        verse[3] if len(verse) > 3 else verse[1],
        "♪ " + hook + " ♪",
        "♪ " + hook + " ♪",
    ]
    return "\n".join(structure)


# ---------------------------------------------------------------------------
# Model loading (lazy, cached)
# ---------------------------------------------------------------------------
_musicgen_model    = None
_musicgen_processor = None
_bark_processor    = None
_bark_model        = None


def _load_musicgen(device: str = "cpu"):
    global _musicgen_model, _musicgen_processor
    if _musicgen_model is not None:
        return _musicgen_model, _musicgen_processor

    log.info("Loading MusicGen (facebook/musicgen-small)…")
    from transformers import MusicgenForConditionalGeneration, AutoProcessor
    import torch

    model_id = "facebook/musicgen-small"
    _musicgen_processor = AutoProcessor.from_pretrained(model_id)
    _musicgen_model = MusicgenForConditionalGeneration.from_pretrained(model_id)
    _musicgen_model.to(device)
    _musicgen_model.eval()
    log.info(f"MusicGen loaded on {device}")
    return _musicgen_model, _musicgen_processor


def _load_bark(device: str = "cpu"):
    global _bark_processor, _bark_model
    if _bark_model is not None:
        return _bark_model, _bark_processor

    log.info("Loading Bark (suno/bark)…")
    from transformers import BarkModel, AutoProcessor
    import torch

    model_id = "suno/bark-small"
    _bark_processor = AutoProcessor.from_pretrained(model_id)
    _bark_model = BarkModel.from_pretrained(model_id)
    _bark_model.to(device)
    _bark_model.eval()
    log.info(f"Bark loaded on {device}")
    return _bark_model, _bark_processor


def _detect_device() -> str:
    try:
        import torch
        if torch.cuda.is_available():
            return "cuda"
        if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            return "mps"
    except ImportError:
        pass
    return "cpu"


# ---------------------------------------------------------------------------
# Step 1: Generate instrumental with MusicGen
# ---------------------------------------------------------------------------
def generate_instrumental(
    prompt: str,
    duration_s: float = 30.0,
    device: str = "cpu",
    seed: int = 42,
) -> np.ndarray:
    """
    Generate a mono instrumental at MusicGen's native SR (32 kHz).
    Returns float32 numpy array. Caller resamples to 44.1 kHz.
    """
    import torch

    model, processor = _load_musicgen(device)
    log.info(f"Generating instrumental: '{prompt[:80]}…' ({duration_s:.0f}s)")

    torch.manual_seed(seed)
    if device == "cuda":
        torch.cuda.manual_seed(seed)

    inputs = processor(text=[prompt], return_tensors="pt")
    inputs = {k: v.to(device) for k, v in inputs.items()}

    # MusicGen generates at 32 kHz; max_new_tokens ≈ duration_s * 50
    max_tokens = int(duration_s * 50)

    with torch.no_grad():
        audio_values = model.generate(
            **inputs,
            max_new_tokens=max_tokens,
            do_sample=True,
            guidance_scale=3.0,
        )

    # audio_values shape: (batch, channels, samples)
    audio = audio_values[0, 0].cpu().float().numpy()
    log.info(f"Instrumental generated: {len(audio)/32000:.1f}s at 32 kHz")
    return audio, 32000  # (samples, sr)


# ---------------------------------------------------------------------------
# Step 2: Detect key and tempo
# ---------------------------------------------------------------------------
def detect_key_tempo(audio_44k: np.ndarray) -> tuple[str, str, float]:
    """Returns (root_note, mode, bpm) from a 44.1 kHz mono array."""
    # Tempo
    tempo, _ = librosa.beat.beat_track(y=audio_44k, sr=SR_TARGET)
    bpm = float(tempo) if not np.isscalar(tempo) else float(tempo)

    # Key via chroma + Krumhansl-Schmuckler profiles
    chroma = librosa.feature.chroma_cqt(y=audio_44k, sr=SR_TARGET)
    chroma_mean = chroma.mean(axis=1)

    KS_MAJOR = np.array([6.35, 2.23, 3.48, 2.33, 4.38, 4.09,
                          2.52, 5.19, 2.39, 3.66, 2.29, 2.88])
    KS_MINOR = np.array([6.33, 2.68, 3.52, 5.38, 2.60, 3.53,
                          2.54, 4.75, 3.98, 2.69, 3.34, 3.17])
    NOTES     = ["C", "C#", "D", "D#", "E", "F",
                  "F#", "G", "G#", "A", "A#", "B"]

    best_score = -np.inf
    best_root  = "C"
    best_mode  = "major"
    for i in range(12):
        rolled = np.roll(chroma_mean, -i)
        for profile, mode in [(KS_MAJOR, "major"), (KS_MINOR, "minor")]:
            score = np.corrcoef(rolled, profile)[0, 1]
            if score > best_score:
                best_score = score
                best_root  = NOTES[i]
                best_mode  = mode

    log.info(f"Detected key: {best_root} {best_mode}  tempo: {bpm:.1f} BPM")
    return best_root, best_mode, bpm


# ---------------------------------------------------------------------------
# Step 3: Generate vocals with Bark
# ---------------------------------------------------------------------------
BARK_VOICE_PRESETS_VOCAL = [
    "v2/en_speaker_6",   # male, expressive
    "v2/en_speaker_9",   # female, clear
    "v2/en_speaker_3",   # male, deeper
    "v2/en_speaker_5",   # female, slightly husky
]


def generate_vocals(
    lyrics: str,
    genre: str,
    device: str = "cpu",
    seed: int = 42,
) -> tuple[np.ndarray, int]:
    """
    Generate vocals from lyrics using Bark.
    Returns (samples float32, sample_rate).
    Bark outputs at 24 kHz.
    """
    import torch
    from transformers import BarkModel, AutoProcessor

    model, processor = _load_bark(device)

    rng = random.Random(seed)
    voice_preset = rng.choice(BARK_VOICE_PRESETS_VOCAL)
    log.info(f"Generating vocals with Bark preset {voice_preset}")

    torch.manual_seed(seed)

    # Bark has a 13s hard limit per call — split lyrics into chunks
    lines = [l.strip() for l in lyrics.split("\n") if l.strip()]
    chunks = []
    current = ""
    for line in lines:
        candidate = (current + " " + line).strip() if current else line
        # Rough token estimate: ~4 chars per token, Bark limit ~250 tokens
        if len(candidate) > 200 and current:
            chunks.append(current)
            current = line
        else:
            current = candidate
    if current:
        chunks.append(current)

    BARK_SR = 24000
    audio_parts = []

    for i, chunk in enumerate(chunks):
        log.info(f"  Bark chunk {i+1}/{len(chunks)}: '{chunk[:60]}…'")
        inputs = processor(chunk, voice_preset=voice_preset, return_tensors="pt")
        inputs = {k: v.to(device) if hasattr(v, 'to') else v for k, v in inputs.items()}

        with torch.no_grad():
            speech_output = model.generate(**inputs, do_sample=True, fine_temperature=0.4, coarse_temperature=0.5)

        arr = speech_output[0].cpu().float().numpy()
        audio_parts.append(arr)

        # Brief silence between chunks (0.3s)
        silence = np.zeros(int(BARK_SR * 0.3), dtype=np.float32)
        audio_parts.append(silence)

    if not audio_parts:
        log.warning("Bark produced no audio — generating silence placeholder")
        return np.zeros(BARK_SR * 10, dtype=np.float32), BARK_SR

    audio = np.concatenate(audio_parts).astype(np.float32)
    log.info(f"Vocals generated: {len(audio)/BARK_SR:.1f}s at {BARK_SR} Hz")
    return audio, BARK_SR


# ---------------------------------------------------------------------------
# Audio utilities
# ---------------------------------------------------------------------------

def _resample(audio: np.ndarray, src_sr: int, dst_sr: int = SR_TARGET) -> np.ndarray:
    """Resample audio array from src_sr to dst_sr using polyphase filter."""
    if src_sr == dst_sr:
        return audio
    from math import gcd
    g = gcd(src_sr, dst_sr)
    up, down = dst_sr // g, src_sr // g
    resampled = resample_poly(audio.astype(np.float64), up, down)
    return resampled.astype(np.float32)


def _to_stereo(audio: np.ndarray) -> np.ndarray:
    """Convert mono (N,) or (2,N) or (N,2) to (N,2) stereo."""
    if audio.ndim == 1:
        return np.stack([audio, audio], axis=1)
    if audio.shape[0] == 2 and audio.ndim == 2:
        return audio.T
    if audio.shape[1] == 2:
        return audio
    return np.stack([audio[:, 0], audio[:, 0]], axis=1)


def _normalize_peak(audio: np.ndarray, target_db: float = -1.0) -> np.ndarray:
    peak = float(np.max(np.abs(audio)) + 1e-9)
    target_lin = 10 ** (target_db / 20.0)
    return (audio * (target_lin / peak)).astype(np.float32)


def _match_length(a: np.ndarray, b: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Pad the shorter array with zeros to match the longer one."""
    if len(a) < len(b):
        a = np.pad(a, (0, len(b) - len(a)))
    elif len(b) < len(a):
        b = np.pad(b, (0, len(a) - len(b)))
    return a, b


def save_wav(path: str, audio: np.ndarray, sr: int = SR_TARGET) -> None:
    sf.write(path, audio, sr, subtype="PCM_16")


# ---------------------------------------------------------------------------
# Step 4: Fuse with fuser.py + optimization loop
# ---------------------------------------------------------------------------
PARAM_DEFAULTS: dict[str, float] = {
    "vocal_level_mult": 1.0,
    "carve_db":         3.0,
    "sidechain_depth":  0.40,
    "de_ess_strength":  1.0,
    "bass_shelf_gain":  0.0,
    "air_boost":        0.0,
}

PARAM_SPACE: dict[str, tuple[float, float]] = {
    "vocal_level_mult": (0.70, 1.30),
    "carve_db":         (1.0,  6.0),
    "sidechain_depth":  (0.20, 0.70),
    "de_ess_strength":  (0.50, 2.00),
    "bass_shelf_gain":  (-3.0, 6.0),
    "air_boost":        (-2.0, 4.0),
}

TARGET_SCORE  = 85
MAX_FUSE_ATTEMPTS = 8


def _fuse_once(
    inst_path: str,
    vocal_path: str,
    out_path: str,
    seed: int,
    params: dict,
    stems_cache: str,
) -> tuple[dict | None, str | None]:
    """Run one fuse with given params. Returns (result_dict, radio_path)."""
    try:
        import fuser as _fuser
        _fuser._PARAM_OVERRIDE.clear()
        _fuser._PARAM_OVERRIDE.update(params)

        result = _fuser.fuse(
            song_a=inst_path,
            song_b=vocal_path,
            out_path=out_path,
            stems_cache=stems_cache,
            direct_vocal=True,   # Bark output is already a clean vocal signal
            seed=seed,
        )
        _fuser._PARAM_OVERRIDE.clear()

        if isinstance(result, dict):
            radio = result.get("radio", out_path)
            return result, radio
        radio = result or out_path
        return {"radio": radio}, radio

    except Exception as e:
        log.error(f"fuse() error: {e}")
        try:
            import fuser as _f
            _f._PARAM_OVERRIDE.clear()
        except Exception:
            pass
        return None, None


def _score_file(path: str) -> tuple[int, str, dict]:
    """Returns (chart_score, chart_grade, details_dict)."""
    try:
        from listen import score_any as _sa
        cs, cg, details = _sa(path)
        return int(round(cs)), cg, details or {}
    except Exception as e:
        log.warning(f"Scoring error: {e}")
        return 0, "?", {}


def _clip_params(p: dict) -> dict:
    return {k: float(np.clip(v, *PARAM_SPACE[k])) for k, v in p.items()}


def _adjust_params_from_score(
    params: dict,
    chart_score: int,
    details: dict,
    rng: random.Random,
) -> dict:
    """
    Heuristic parameter adjustment based on which score dimensions are failing.
    Maps listen.py detail keys to fuser param tweaks.
    """
    p = dict(params)

    # Extract dimension subscores if available
    dim_scores: dict = {}
    if isinstance(details, dict):
        dim_scores = details

    # Helpers
    def get_dim(key: str, default: float = 50.0) -> float:
        return float(dim_scores.get(key, default))

    spectral_ok   = get_dim("spectral",   50) >= 55
    loudness_ok   = get_dim("loudness",   50) >= 55
    dynamics_ok   = get_dim("dynamics",   50) >= 55
    perceptual_ok = get_dim("perceptual", 50) >= 55

    # Vocal too quiet → boost level
    if not spectral_ok:
        shift = rng.uniform(0.05, 0.15)
        p["vocal_level_mult"] = min(p["vocal_level_mult"] + shift, 1.3)
        p["carve_db"] = min(p["carve_db"] + rng.uniform(0.5, 1.5), 6.0)

    # Perceptual (beat sync, clarity) → more sidechain clarity
    if not perceptual_ok:
        p["sidechain_depth"] = min(p["sidechain_depth"] + rng.uniform(0.05, 0.10), 0.70)
        p["de_ess_strength"] = max(p["de_ess_strength"] - rng.uniform(0.1, 0.3), 0.5)

    # Dynamics → reduce de-essing (was over-processing)
    if not dynamics_ok:
        p["de_ess_strength"] = max(p["de_ess_strength"] - 0.2, 0.5)

    # Loudness (mastering issue) → boost air, slight bass
    if not loudness_ok:
        p["air_boost"] = min(p["air_boost"] + rng.uniform(0.5, 1.0), 4.0)
        p["bass_shelf_gain"] = min(p["bass_shelf_gain"] + rng.uniform(0.5, 1.5), 6.0)

    return _clip_params(p)


def _adjust_prompts_from_score(
    inst_prompt: str,
    vocal_prompt_suffix: str,
    chart_score: int,
    details: dict,
    attempt: int,
) -> tuple[str, str]:
    """Modify generation prompts based on weak score dimensions."""
    new_inst   = inst_prompt
    new_vocal  = vocal_prompt_suffix

    dim_scores: dict = details if isinstance(details, dict) else {}

    def get_dim(key: str, default: float = 50.0) -> float:
        return float(dim_scores.get(key, default))

    spectral_score   = get_dim("spectral",   50)
    loudness_score   = get_dim("loudness",   50)
    dynamics_score   = get_dim("dynamics",   50)
    perceptual_score = get_dim("perceptual", 50)

    if spectral_score < 50:
        if "balanced" not in new_inst:
            new_inst += ", balanced mix, clear midrange"
    if dynamics_score < 50:
        if "dynamic" not in new_inst:
            new_inst += ", punchy transients, dynamic range"
    if loudness_score < 50:
        if "loud" not in new_inst:
            new_inst += ", loud, energetic"
    if perceptual_score < 50:
        new_vocal = "clear diction, on-beat, rhythmic"

    # Every other attempt: also add genre-specific brightness modifier
    if attempt % 2 == 0 and "crisp" not in new_inst:
        new_inst += ", crisp hi-hats, clear separation"

    return new_inst, new_vocal


# ---------------------------------------------------------------------------
# Core generation + eval loop
# ---------------------------------------------------------------------------

def generate_song(
    genre: str = "auto",
    inst_prompt: str = "",
    n_attempts: int = 10,
    seed: int = 42,
    device: str = "auto",
    output_dir: Path = AUTOGEN,
    duration_s: float = 30.0,
) -> dict:
    """
    Full autonomous song generation pipeline.
    Returns dict with final paths, score, metadata.
    """
    if device == "auto":
        device = _detect_device()
    log.info(f"Device: {device}")

    song_id  = uuid.uuid4().hex[:8]
    song_dir = output_dir / song_id
    song_dir.mkdir(parents=True, exist_ok=True)

    log_path = song_dir / "generation.log"
    fh = logging.FileHandler(log_path)
    fh.setFormatter(logging.Formatter("%(asctime)s %(message)s"))
    log.addHandler(fh)

    # ── Pick genre + prompt ────────────────────────────────────────────────
    if genre == "auto":
        genre = random.choice(ALL_GENRES)
    log.info(f"Genre: {genre}  ID: {song_id}")

    if not inst_prompt:
        inst_prompt = random.choice(GENRE_PROMPTS.get(genre, GENRE_PROMPTS["pop"]))
    log.info(f"Instrumental prompt: {inst_prompt}")

    vocal_prompt_suffix = ""

    # ── Generate instrumental once (expensive — reuse across fuse attempts) ─
    inst_audio_raw, inst_sr = generate_instrumental(
        prompt=inst_prompt,
        duration_s=duration_s,
        device=device,
        seed=seed,
    )
    inst_mono_44k = _resample(inst_audio_raw, inst_sr, SR_TARGET)
    inst_stereo   = _to_stereo(inst_mono_44k)
    inst_stereo   = _normalize_peak(inst_stereo, -1.0)

    inst_path = str(song_dir / "instrumental.wav")
    save_wav(inst_path, inst_stereo)
    log.info(f"Instrumental saved: {inst_path}")

    # ── Detect key + tempo ────────────────────────────────────────────────
    key_root, key_mode, bpm = detect_key_tempo(inst_mono_44k)
    log.info(f"Key: {key_root} {key_mode}  BPM: {bpm:.1f}")

    # ── Generate lyrics ────────────────────────────────────────────────────
    lyrics = _build_lyrics(genre, key_root, bpm)
    log.info(f"Lyrics generated ({len(lyrics.split(chr(10)))} lines)")

    # ── Generate vocals once initially ────────────────────────────────────
    vocal_audio_raw, vocal_sr = generate_vocals(
        lyrics=lyrics,
        genre=genre,
        device=device,
        seed=seed,
    )
    vocal_mono_44k = _resample(vocal_audio_raw, vocal_sr, SR_TARGET)
    vocal_stereo   = _to_stereo(vocal_mono_44k)
    vocal_stereo   = _normalize_peak(vocal_stereo, -1.0)

    vocal_path = str(song_dir / "vocals.wav")
    save_wav(vocal_path, vocal_stereo)
    log.info(f"Vocals saved: {vocal_path}")

    # ── Optimization loop ──────────────────────────────────────────────────
    best_score   = 0
    best_grade   = "?"
    best_path    = None
    best_meta    = None
    best_params  = dict(PARAM_DEFAULTS)
    params       = dict(PARAM_DEFAULTS)
    attempts_log = []
    rng_adj      = random.Random(seed + 999)

    for attempt in range(1, n_attempts + 1):
        log.info(f"\n── Attempt {attempt}/{n_attempts} ──────────────────────────")

        # Regenerate instrumental on even attempts after first if score is low
        if attempt > 2 and attempt % 3 == 0 and best_score < 60:
            log.info("Score still low — regenerating instrumental with adjusted prompt")
            inst_audio_raw, inst_sr = generate_instrumental(
                prompt=inst_prompt,
                duration_s=duration_s,
                device=device,
                seed=seed + attempt * 17,
            )
            inst_mono_44k = _resample(inst_audio_raw, inst_sr, SR_TARGET)
            inst_stereo   = _normalize_peak(_to_stereo(inst_mono_44k), -1.0)
            save_wav(inst_path, inst_stereo)

        # Regenerate vocals on odd attempts after first if score is low
        if attempt > 1 and attempt % 3 == 1 and best_score < 72:
            log.info("Regenerating vocals with adjusted parameters")
            new_lyrics = _build_lyrics(genre, key_root, bpm)
            vocal_audio_raw, vocal_sr = generate_vocals(
                lyrics=new_lyrics,
                genre=genre,
                device=device,
                seed=seed + attempt * 31,
            )
            vocal_mono_44k = _resample(vocal_audio_raw, vocal_sr, SR_TARGET)
            vocal_stereo   = _normalize_peak(_to_stereo(vocal_mono_44k), -1.0)
            save_wav(vocal_path, vocal_stereo)

        att_dir  = song_dir / f"attempt_{attempt}"
        att_dir.mkdir(parents=True, exist_ok=True)
        out_path = str(att_dir / "mix.wav")

        log.info(f"Fusing with params: {json.dumps({k: round(v, 3) for k, v in params.items()})}")
        result, radio_path = _fuse_once(
            inst_path=inst_path,
            vocal_path=vocal_path,
            out_path=out_path,
            seed=seed + attempt,
            params=params,
            stems_cache=str(STEMS_DIR),
        )

        if radio_path is None or not Path(radio_path).exists():
            log.warning(f"Attempt {attempt}: fuse produced no output — skipping")
            attempts_log.append({"n": attempt, "chart_score": 0, "error": "no output"})
            continue

        # Score
        chart_score, chart_grade, details = _score_file(radio_path)
        log.info(f"Attempt {attempt} → chart_score={chart_score} ({chart_grade})")

        # Save metadata
        meta = {
            "attempt": attempt,
            "params": params,
            "chart_score": chart_score,
            "chart_grade": chart_grade,
            "details": details,
            "inst_prompt": inst_prompt,
            "genre": genre,
            "key": f"{key_root} {key_mode}",
            "bpm": bpm,
            "radio": radio_path,
        }
        (att_dir / "metadata.json").write_text(json.dumps(meta, indent=2, default=str))
        attempts_log.append({"n": attempt, "chart_score": chart_score, "chart_grade": chart_grade})

        if chart_score > best_score:
            best_score  = chart_score
            best_grade  = chart_grade
            best_path   = radio_path
            best_meta   = meta
            best_params = dict(params)

        if chart_score >= TARGET_SCORE:
            log.info(f"Target reached: {chart_score} >= {TARGET_SCORE}")
            break

        # Adjust for next attempt
        params = _adjust_params_from_score(params, chart_score, details, rng_adj)
        inst_prompt, vocal_prompt_suffix = _adjust_prompts_from_score(
            inst_prompt, vocal_prompt_suffix, chart_score, details, attempt
        )

    # ── Copy best to song root ─────────────────────────────────────────────
    if best_path and Path(best_path).exists():
        import shutil
        final_path = str(song_dir / "best_mix.wav")
        shutil.copy2(best_path, final_path)
        log.info(f"\nBest mix saved: {final_path}  [{best_score}/100 {best_grade}]")
    else:
        final_path = ""
        log.warning("No successful mix produced.")

    # ── Save summary ───────────────────────────────────────────────────────
    summary = {
        "song_id":      song_id,
        "genre":        genre,
        "inst_prompt":  inst_prompt,
        "key":          f"{key_root} {key_mode}",
        "bpm":          bpm,
        "best_score":   best_score,
        "best_grade":   best_grade,
        "best_path":    final_path,
        "best_params":  best_params,
        "attempts":     attempts_log,
        "n_attempts":   len(attempts_log),
        "device":       device,
        "seed":         seed,
    }
    (song_dir / "summary.json").write_text(json.dumps(summary, indent=2, default=str))

    log.removeHandler(fh)
    fh.close()

    _print_result_banner(song_id, best_score, best_grade, final_path, attempts_log)
    return summary


def _print_result_banner(song_id: str, score: int, grade: str, path: str, attempts: list) -> None:
    bar = "█" * (score // 5) + "░" * (20 - score // 5)
    grade_color = {
        "S": "\033[93m",  # yellow
        "A": "\033[95m",  # magenta
        "B": "\033[94m",  # blue
        "C": "\033[96m",  # cyan
        "D": "\033[90m",  # grey
        "F": "\033[91m",  # red
    }.get(grade[0] if grade else "?", "")
    reset = "\033[0m"

    print(f"\n{'='*60}")
    print(f"  Song ID : {song_id}")
    print(f"  Score   : {grade_color}{score:3d}/100  {grade}{reset}  [{bar}]")
    print(f"  Attempts: {len(attempts)}")
    if path:
        print(f"  Output  : {path}")
    if attempts:
        for a in attempts:
            marker = " ← best" if a.get("chart_score") == score else ""
            print(f"    #{a['n']:2d}  {a.get('chart_score', 0):3d}/100  {a.get('chart_grade','?')}{marker}")
    print(f"{'='*60}\n")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="VocalFusion Autonomous Song Generator",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python autonomous_songwriter.py
  python autonomous_songwriter.py --generate-n 5
  python autonomous_songwriter.py --genre trap --seed 123
  python autonomous_songwriter.py --prompt "dark lo-fi hip hop beat" --n-attempts 8
  python autonomous_songwriter.py --duration 45 --device cuda
        """,
    )
    parser.add_argument("--generate-n",   type=int,   default=1,      help="Number of songs to generate sequentially")
    parser.add_argument("--genre",        type=str,   default="auto",  choices=ALL_GENRES + ["auto"], help="Genre (default: random)")
    parser.add_argument("--prompt",       type=str,   default="",     help="Custom instrumental prompt (overrides random)")
    parser.add_argument("--n-attempts",   type=int,   default=10,     help="Max fuse+score attempts per song (default: 10)")
    parser.add_argument("--seed",         type=int,   default=None,   help="Random seed (default: random per song)")
    parser.add_argument("--device",       type=str,   default="auto", choices=["auto", "cpu", "cuda", "mps"], help="Inference device")
    parser.add_argument("--duration",     type=float, default=30.0,   help="Instrumental duration in seconds (default: 30)")
    parser.add_argument("--output-dir",   type=str,   default=str(AUTOGEN), help="Output directory")
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    summaries = []
    for i in range(args.generate_n):
        song_seed = args.seed if args.seed is not None else random.randint(0, 2**31)
        if args.generate_n > 1:
            log.info(f"\n{'='*60}")
            log.info(f"  Song {i+1} of {args.generate_n}  (seed={song_seed})")
            log.info(f"{'='*60}")

        summary = generate_song(
            genre=args.genre,
            inst_prompt=args.prompt,
            n_attempts=args.n_attempts,
            seed=song_seed,
            device=args.device,
            output_dir=output_dir,
            duration_s=args.duration,
        )
        summaries.append(summary)

    if args.generate_n > 1:
        scores = [s["best_score"] for s in summaries]
        log.info(f"\nGenerated {len(summaries)} songs.")
        log.info(f"Scores: {scores}")
        log.info(f"Average: {sum(scores)/len(scores):.1f}/100")
        log.info(f"Best: {max(scores)}/100")

        report_path = output_dir / f"batch_{uuid.uuid4().hex[:6]}.json"
        report_path.write_text(json.dumps(summaries, indent=2, default=str))
        log.info(f"Batch report: {report_path}")


if __name__ == "__main__":
    main()
