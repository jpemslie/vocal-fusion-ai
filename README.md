# VocalFusion

AI-powered mashup engine. Give it two songs — it separates stems, aligns BPM/key, processes vocals, and produces a professional-sounding mashup.

**Current best score:** 67/100 [C] (Mahalanobis/Drake reference) | Target: 85+ [A]

---

## Quick Start

```bash
# Install dependencies
pip install -r requirements.txt

# Fuse two songs (CLI)
python -c "from fuser import fuse; fuse('song_a.mp3', 'song_b.mp3', 'output.wav')"

# Or run the web server
python run.py
# → open http://localhost:5000
```

## Requirements

- Python 3.10+
- `conda activate vocal-fusion` (or equivalent venv)
- ~4 GB RAM minimum; 8 GB recommended for Mel-Roformer denoiser
- GPU optional (MPS/CUDA speeds up stem separation ~4×)

## Architecture

```
fuser.py          — Orchestration layer (fuse() entry point, correction loop)
├── stem_separation — Demucs / BSRoformer / Mel-Roformer denoiser + CPU-aware watchdog
├── vocal_chain     — HPSS, EQ, pitch correction, presence, breath reduction
├── arrangement     — BPM/key detection, section detection, beat alignment
├── mixing          — Iterative mix loop, spectral carve, sidechain
├── mastering       — Radio/club/intimate masters, LUFS normalization
└── scoring         — listen.py quality gate, ML scorer (Mahalanobis/PCA)

api.py            — REST API (Flask blueprint)
run.py            — Flask server entry point
listen.py         — Audio quality scorer (tonal/dynamic/stereo/HNR/vocal/mashup)
director.py       — AI Director (Claude API — reasons about mix quality)
ml/scorer.py      — ML scorer: Mahalanobis distance to Drake reference
ml/clap_scorer.py — CLAP-based perceptual scorer (genre-agnostic, in progress)
```

## Scoring

Two independent scorers run on every output:

| Scorer | What it measures | Range |
|--------|-----------------|-------|
| `listen.py` | Tonal balance, dynamics, stereo, HNR, vocal quality | 0–100 |
| `ml/scorer.py` | Proximity to Drake reference tracks (PCA/Mahalanobis) | 0–100 |

Grade thresholds: A ≥85, B ≥70, C ≥55, D ≥35, F <35.

Run both scorers on a file:
```bash
python score_test.py path/to/output.wav
```

## Tests

```bash
# Unit tests (no audio files required)
python -m pytest tests/ -v

# Regression test (requires audio files — see tests/regression.py)
python tests/regression.py --fast   # re-score last output only
python tests/regression.py          # full re-fuse + score
```

## Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `ANTHROPIC_API_KEY` | — | Required for AI Director |
| `VF_STEMS_DIR` | `vf_data/stems` | Stem cache directory |
| `VF_OUTPUT_DIR` | `vf_data/mixes` | Output directory |
| `VF_NEURAL_MASTER` | `0` | Set to `1` to enable neural mastering (Phase 3) |
| `VF_NO_DENOISER` | `0` | Set to `1` to skip Mel-Roformer denoiser |

## Roadmap

See [ROADMAP.md](ROADMAP.md) for the full plan toward 85+ quality.

Key milestones:
- **Phase 1A** — BSRoformer stems (~12.9 dB SDR vs Demucs ~8.4 dB) → est. +5–8 pts
- **Phase 2A** — VoiceFixer neural vocal enhancement → est. +3–5 pts
- **Phase 3** — Neural mastering chain (genre-conditioned) → est. +5–8 pts
- **Phase 4** — CLAP perceptual scorer (genre-agnostic)

## Known Quality Ceilings

These are source-material limitations, not code bugs:

- **Sub vs Mid ratio**: Rhyme Dust beat has structural heavy sub — cutting it makes the mix thin
- **HNR 0.3–0.6 dB**: Demucs bleed is signal-correlated, denoiser cannot remove it
- **Groove timing**: AI-quantized trap beats score 0 (no micro-timing variation by design)
- **Dynamic complexity**: Modern trap mastering = flat RMS (intentional, not a defect)

## License

MIT — see [LICENSE](LICENSE).
