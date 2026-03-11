# VocalFusion — Project Context Backup
*Paste this file into a new Claude conversation to restore full project context.*

---

## What This Project Is
VocalFusion is an AI mashup engine that takes any two songs, separates stems, and produces a professional-quality mix. The goal: ANY two songs in → top-chart quality output, every time.

**Location:** `/Users/jpemslie/Desktop/VocalFusion/`
**GitHub:** `https://github.com/jpemslie/vocal-fusion-ai`
**Python env:** `conda env vocal-fusion` at `/usr/local/Caskroom/miniconda/base/envs/vocal-fusion/`
**Run server:** `/usr/local/Caskroom/miniconda/base/envs/vocal-fusion/bin/python run.py --port 5000`

---

## Key Files
| File | Purpose |
|------|---------|
| `fuser.py` | Core engine (~2100 lines). All AI analysis, stem separation, DSP chain. |
| `run.py` | Flask server: `POST /fuse`, `GET /status/<id>`, `GET /output/<file>` |
| `templates/index.html` | Minimal web UI: 2 file inputs, fuse button, audio player |
| `compare.py` | Spectral analysis: measures 5 frequency bands, generates mel spectrogram PNG |
| `qa_audio.py` | QA tool: HF ratio, spike detection, waveform PNG |
| `requirements.txt` | Python dependencies |

---

## Architecture: How fuse() Works

### Stem Separation
- **GPU:** BS-Roformer (`model_bs_roformer_ep_317_sdr_12.9755.ckpt`) via audio-separator — SDR ~13 dB
- **CPU:** MDX-Net Kim Vocal 2 (`Kim_Vocal_2.onnx`) via audio-separator — SDR ~9.5 dB
- **CPU fallback:** Demucs `htdemucs_ft` — SDR ~8.5 dB
- Stems are cached by file fingerprint in `vf_data/stems/`

### Step-by-Step Pipeline (fuse() function)
1. Load both songs mono at 44100 Hz
2. Detect BPM (librosa beat_track + onset-strength tempo, hip-hop prior, half-tempo correction)
3. Detect key (Krumhansl-Schmuckler chromagram profile)
4. Separate stems — Song A → instrumental, Song B → vocals
5. **HPSS Wiener bleed cleanup** on vocal stem (margin=3.0, removes hi-hat/drum bleed)
6. Compute BPM ratio with octave correction (tries half/double BPM before clamping)
7. Compute key shift with Camelot-wheel alternatives (tries relative mode + adjacent keys)
8. AI content analysis → all DSP params driven by audio content:
   - `_analyze_beat_character()`: aggressiveness, bass_weight, brightness
   - `_analyze_vocal_character()`: rap_score from ZCR, flatness, onset rate, pitch range, gender
   - `_style_params()`: maps scores to all DSP parameters
9. **`_process_vocals()`** — full vocal processing chain (see below)
10. `_iterative_mix()` — 3-iteration level-matching + spectral carve
11. `_master()` — full mastering chain (see below)
12. 7-point auto quality evaluation (beat sync, vocal presence, spectral balance, clipping, LUFS, phase, stereo correlation)

### Vocal Processing Chain (_process_vocals)
1. Breath reduction (spectral flatness detection, 4 dB attenuation)
2. **[ONLY when pitch-shifting]** 8kHz LPF (removes hi-hat bleed before rubberband shift)
3. pyrubberband R3 time-stretch + pitch-shift (formant-preserving; flag `{'-3': ''}`)
4. HPF 80 Hz + subtractive EQ (-3 dB@300 Hz mud, -2 dB@500 Hz boxy, +1.5 dB@250 Hz Demucs restore)
5. De-esser (split-band, before compression)
6. Soft-knee FET compressor → Opto compressor (style-adaptive ratios/times)
7. Noise gate (threshold from stem noise floor analysis)
8. 4-band multiband compression (80/400/2500/8000 Hz crossovers; 400-2500 Hz gentlest)
9. Presence boost + air shelf (style-adaptive, +1-2 dB presence, +2-3 dB air)
9b. Consonant enhancement (4-9 kHz transient boost, NOT sustained)
9c. Safe mid-band harmonic exciter (400-2500 Hz only, 15% parallel, tanh 0.12, Mid channel only)
10. Early reflections (7/14/21 ms) + reverb tail (20 ms pre-delay, HPF'd at 500 Hz)
11. Stereo ADT (±4-5 cents L/R, 22/27 ms delay, LFO 0.3 Hz)

### Mixing (_iterative_mix)
- Energy-envelope matching per 2s window (gains capped 0.5-1.5×)
- Spectral carve (Wiener mask, 300-5kHz weighted, 2-4kHz peak 1.5×)
- SPL-style transient shaping (+4 dB attack, -2 dB sustain, 80 ms slow window)
- Kick→sub sidechain (20-80 Hz ducked, depth 0.20)
- NY parallel compression (30% wet)
- Multiband sidechain (mids/highs only, 200 Hz crossover, depth 0.07-0.15)
- M/S encode: vocal → Mid only, beat Sides preserved
- Groove quantization (8th-note grid, 35-45% strength, 3 ms crossfades)
- Chorus-to-chorus structural alignment + beat-grid fine alignment

### Mastering (_master)
M/S EQ → mastering EQ → harmonic exciter (5kHz+, 7%) → Chebyshev soft-clip → sub-bass limiter (20-80 Hz, -3 dBFS) → LUFS normalize (-10 LUFS) → post-normalize EQ (-3 dB@6kHz shelf) → brick-wall limiter (-2 dBTP) → PCM_24

### Auto Quality Checks (7 points)
1. Beat sync — cross-correlation of onset envelopes, want >45%
2. Vocal presence — 40-70% of combined stem energy
3. Spectral balance — 5 bands vs commercial reference ranges
4. Clipping — peak > 1.001
5. Integrated LUFS — -15 to -8 range
6. Phase cancellation — per-band, >110° mean diff = warning
7. Stereo correlation — >0.7 mono-safe, >0.5 pass

---

## Critical Bugs Fixed (NEVER REVERT)

### Imports and subprocess
- Use `sys.executable -m demucs` NOT `"python"` in subprocess
- Demucs flag: `-n htdemucs_ft` NOT `--model htdemucs_ft`
- `import librosa.feature.rhythm` explicitly at top — then call as `librosa.feature.rhythm.tempo(...)`
- pyrubberband R3 flag: `rbargs={'-3': ''}` (empty string, NOT `{'--engine': '3'}`)

### Scope bugs
- **NEVER** import `butter` or `sosfilt` inside a try-block in `_process_vocals`. Python function-scope rules treat any variable assigned anywhere in a function as local. Since butter/sosfilt are used before Step 9c, they must be available as module-level imports (line ~47).

### DSP calibration
- Transient shaper slow_ms = 80 ms (not 20 ms — 20 ms too fast, catches attacks instead of sustain)
- Multiband crossovers: 80/400/2500/8000 Hz (NOT 80/250/2000/8000)
- Output format: PCM_24 (not PCM_16)
- LUFS normalize goes LAST in mastering (after glue comp + sub limiter, BEFORE final brick-wall)
- vocal_level: `np.interp(rap, [0,1], [1.0, 1.3])` (NOT [1.5, 2.0] — was burying the beat)
- Energy-match-envelope gains capped 0.5-1.5× (NOT 0.4-4.0×)

### Removed stages (caused static/artifacts)
- **DeepFilterNet** — removed entirely. Trained on speech+stationary noise; applied to music stems it creates metallic static.
- **PYIN pitch correction** — removed. Wrong pitch readings on bleed-contaminated stems → random pitch jumps.
- **Vocal waveshaper/asymmetric saturation** — removed. Saturation on noisy stems amplifies bleed.
- **Aphex-style harmonic exciter** — removed (old version). Replaced with safe mid-band-only exciter (400-2500 Hz).

---

## Current State & Known Issues
- **CPU only** (no GPU) — stem separation uses MDX-Net → Demucs fallback
- `_has_gpu()` returns `False`
- Outputs go to `vf_data/mixes/`
- Stems cached in `vf_data/stems/` by SHA256 fingerprint

### Known remaining issues to investigate
- BPM detection biased toward hip-hop (130 BPM lognormal prior) — may misdetect ballads
- Spectral carve hardcoded to 300-5kHz — misses deep bass vocals (<300 Hz) and high sopranos (>5kHz)
- Chorus detection uses k=7 fixed segments — fails on ambient/unusual structures
- Iterative mix only 3 iterations — can oscillate and miss 40-70% vocal presence target
- Reverb wet% driven entirely by rap_score — wrong for opera/classical/metal

---

## User Preferences
- Always push to GitHub after every commit
- Keep improving indefinitely — target is professional/top-chart quality
- Compare with ChatGPT's system as quality benchmark
- Any two songs must produce clean output — bulletproof for all genres and BPM/key combos

---

## Run Commands
```bash
# Start web server
/usr/local/Caskroom/miniconda/base/envs/vocal-fusion/bin/python run.py --port 5000

# Quick spectral comparison
/usr/local/Caskroom/miniconda/base/envs/vocal-fusion/bin/python compare.py song_a.mp3 output.wav

# QA artifact check
/usr/local/Caskroom/miniconda/base/envs/vocal-fusion/bin/python qa_audio.py output.wav

# Test import
/usr/local/Caskroom/miniconda/base/envs/vocal-fusion/bin/python -c "import fuser; print('OK')"
```

---

*Last updated: 2026-03-11. See MEMORY.md for compressed notes.*
