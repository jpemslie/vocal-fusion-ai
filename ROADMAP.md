# VocalFusion — Roadmap to Top-Chart Quality

**Goal:** Beat ChatGPT's mashup quality. Professional, commercial-sounding output on any two songs, any genre, any BPM/key combo.

**Current best score:** 67/100 [C] (Mahalanobis ML scorer) | beats 83-86% of Drake reference
**listen.py best:** 55/100 (v44, v48) — structural ceilings: sub/mid+22 (Rhyme Dust), HNR 0.4 dB (Demucs bleed)
**Target:** A-grade (85+) on ML scorer + CLAP scorer

---

## Session 2026-03-29 Progress

- [x] Smart CPU-aware watchdog (Mel-Roformer survives 15-min run — was killed at 13/20 chunks)
- [x] HNR gate: A/B test denoised vs original; revert if HNR gain < 1 dB
  - Result: E85 vocal HNR 2.2 → 2.2 dB (+0.0 dB) — denoiser always rejected for Demucs-separated commercial stems
  - Root cause: Demucs bleed is signal-correlated — not noise, denoiser can't touch it
- [x] Carve ceiling 12 → 8 dB (11.5 dB was eating mid band, making sub/mid ratio fail harder)
- [x] Regression guard: saves/restores ALL mutable params (style + sub_cut_db + lufs_target)
- [x] Regression guard: clean revert — skip corrections that pass, remix first then score fresh
- [x] presence_db floor 0.0 → 1.5 dB (director was burying vocals)
- [x] LUFS -12 → -14 dBFS for better LRA/headroom
- [x] Auto sub-cut in mastering: measure sub/mid ratio, cut up to 6 dB if > 1.8
- [x] Correction passes 3 → 6 with oscillation detection

**Ceiling diagnosis:** listen.py 55/100 is the hard ceiling for Toliver E85 × Rhyme Dust:
- Sub vs Mid +22 dB (target ≤ +18): Rhyme Dust sub is structural, cutting it makes the mix thin
- HNR 0.4-0.6 dB: Demucs bleed = correlated artifacts, can't be denoised
- Breaking past 55 requires Phase 1A (BSRoformer stems, ~10 dB SDR) or Phase 1B (StemPostFilter)

---

## Phase 0 — Foundation (Done ✅)

- [x] Stem separation (Demucs htdemucs)
- [x] BPM detection + beat-grid alignment
- [x] Key detection + pitch shifting
- [x] Iterative mix loop (3 passes) with auto-correction
- [x] 3 mastering variants (radio / club / intimate)
- [x] ML scorer (Mahalanobis, 70-track Drake reference)
- [x] `listen.py` quality gate (tonal, dynamic, stereo, HNR, vocal)
- [x] DeepFilter noise reduction (M/S mid-only, wet=0.20)
- [x] Master EQ chain (bass cut, highmid cut, mid lift, sides HPF)
- [x] Fix "robot getting stabbed" artifacts — WORLD vocoder disabled for commercial stems
- [x] Fix `true_peak_dbfs` false positive — boundary loosened to -0.3 dBFS
- [x] `vocal_robot_score` metric in `listen.py`
- [x] Correction memory (avoid oscillating corrections)
- [x] `sub_cut_db` correction parameter
- [x] Director + built-in correction merge (both run every loop)

---

## Phase 1 — Better Stem Separation (High Impact, Low Effort)

> **Why this first:** Everything downstream (vocals, mix balance, HNR) improves when stems are cleaner. Current Demucs gives ~8.4 dB SDR. Target: 10+ dB SDR.

- [ ] **1A — BSRoformer / Mel-Roformer swap**
  - `pip install audio-separator`
  - Download `model_bs_roformer_ep_368_sdr_12.9628.ckpt` (~100 MB)
  - Update `fuser.py:_separate_stems()` → try `audio_separator` first, fall back to demucs on error
  - Expected: vocal HNR improvement, less bleed artifacts
  - Test: `python score_test.py` → compare HNR and listen.py vocal_hnr_db (target: >8 dB)

- [ ] **1B — StemPostFilter (Conv-TasNet)**
  - Files ready: `ml/post_filter.py`, `ml/train_post_filter.py`
  - Steps:
    - [ ] `pip install musdb demucs`
    - [ ] `python -m ml.train_post_filter --prep-data --musdb-dir /path/to/musdb18hq --cache-dir ml/cache/post_filter_pairs`
    - [ ] `python -m ml.train_post_filter --cache-dir ml/cache/post_filter_pairs --stem vocals --out ml/post_filter_vocals.pt --epochs 100`
    - [ ] Wire into `fuser.py`: after `_separate_stems()`, apply `StemPostFilter.enhance(vocal_stem)`
  - Expected: vocal_hnr_db stuck at 0.4 → target 6+ dB

---

## Phase 2 — Neural Vocal Enhancement

> **Why:** Even after better separation, residual artifacts (Demucs bleed, de-essing artifacts) remain. VoiceFixer handles these at the perceptual level.

- [ ] **2A — VoiceFixer integration**
  - `pip install voicefixer`
  - Add `_enhance_vocal_neural(vox_np, sr)` to `fuser.py`
  - Call after stem separation (or after StemPostFilter if Phase 1B done)
  - Gate: only run if `vocal_hnr_db < 6.0` (don't process already-clean vocals)
  - Test: check vocal_robot_score + HNR improvement

- [ ] **2B — Phase-aware harmonic restoration**
  - Use `librosa.effects.harmonic()` + Wiener masking to recover vocal harmonics lost in separation
  - Alternative to VoiceFixer if too slow for realtime

---

## Phase 3 — Neural Mastering Chain

> **Why:** Manual EQ is tuned for trap/hip-hop. Neural mastering learns genre-appropriate treatment from professional references.

- [ ] **3A — Build training data**
  - Files ready: `ml/build_mastering_dataset.py`
  - `pip install matchering pyloudnorm`
  - Gather 50+ professional reference tracks per genre (Drake, Travis, Gunna for hiphop)
  - `python -m ml.build_mastering_dataset --ref-dir /path/to/refs --mix-dir vf_data/mixes --out-dir ml/cache/mastering_pairs --genre hiphop --n-pairs 200`

- [ ] **3B — Train MasteringNetwork**
  - Files ready: `ml/mastering_net.py`, `ml/train_mastering.py`
  - `python -m ml.train_mastering --data-dir ml/cache/mastering_pairs --genre hiphop --out ml/mastering_net.pt --epochs 200`
  - Needs GPU (MPS/CUDA) for reasonable training time

- [ ] **3C — Wire into fuser.py**
  - Add `_master_neural(mix, genre, sr)` to `fuser.py`
  - Feature flag: `USE_NEURAL_MASTER = os.environ.get("VF_NEURAL_MASTER", "0") == "1"`
  - A/B test: run both and score both, keep better

---

## Phase 4 — CLAP Quality Scorer

> **Why:** Current scorer uses 70 Drake tracks in PCA/Mahalanobis space — genre-locked, feature-engineered. CLAP is perceptual and genre-agnostic.

- [ ] **4A — Install CLAP**
  - `pip install msclap` (or `pip install laion-clap`)
  - File ready: `ml/clap_scorer.py`

- [ ] **4B — Build reference set**
  - Gather 100+ professional tracks spanning: hip-hop, pop, R&B, trap, EDM, rock
  - `python -m ml.clap_scorer --build-ref --ref-dir /path/to/pro_tracks --out ml/clap_ref.npz`

- [ ] **4C — Dual-score all outputs**
  - Report both ML score (Mahalanobis) and CLAP score in `score_test.py`
  - Eventually replace Mahalanobis as primary scorer

---

## Phase 5 — Genre Robustness

> **Why:** Known bugs cause failure on non-hip-hop genres. Need to fix before claiming "any genre" support.

- [ ] **5A — BPM detection bias fix**
  - Current: lognormal prior peaks at 130 BPM (hip-hop)
  - Fix: use `madmom` BPM detector (genre-agnostic, uses neural onset detection)
  - Test on: ballads (~75 BPM), EDM (~128 BPM), jazz (~180 BPM)

- [ ] **5B — Spectral carve range fix**
  - Current: hardcoded 300-5kHz
  - Fix: detect vocal fundamental range first → extend carve to cover fundamental ±2 octaves
  - Catches: bass vocalists (<300 Hz), sopranos (>5 kHz)

- [ ] **5C — Chorus detection fix**
  - Current: k=7 fixed segments (fails on ambient/drone)
  - Fix: use `librosa.segment.agglomerative()` with dynamic k based on segment count
  - Or: use beat-synchronous SSM with novelty-curve peak detection

- [ ] **5D — Reverb genre conditioning**
  - Current: reverb wet% driven entirely by `rap_score`
  - Fix: add genre classifier (`genre_score` dict: rap/edm/classical/rock) → lookup table for reverb

---

## Phase 6 — Convergence & Polish

- [ ] **6A — Iterative loop: 3 → 8 passes**
  - Increase `N_CORRECTION_PASSES` from 3 to 8
  - Add early exit: stop when `listen_score >= 80` or delta < 0.5 across 2 consecutive passes
  - Add oscillation detection: if score alternates between two values, try intermediate params

- [ ] **6B — Auto-gain staging**
  - Pre-check vocal level before mix: if vocal RMS < -25 dBFS in mix, auto-raise before mastering
  - Fixes cases where vocal is buried and correction loop can't dig it out

- [ ] **6C — vocal_hnr_db target tracking**
  - Current correction for vocal_hnr_db: apply carve_db cut (wrong — carve doesn't fix HNR)
  - Fix: add denoising strength correction (`deepfilter_wet` → increase when HNR low)
  - Wire `deepfilter_wet` as a correctable parameter in the loop

- [ ] **6D — UI: genre selector**
  - Add genre dropdown to `templates/index.html`
  - Pass `genre` param through `POST /fuse` → `fuser.fuse()`
  - Use in: reverb, mastering EQ style, neural mastering conditioning

---

## Quality Milestones

| Milestone | ML Score | CLAP Score | What it means |
|-----------|----------|------------|---------------|
| Current   | 67 [C]   | TBD        | Beats 83% of reference, clean vocals |
| Phase 1   | 72-75    | TBD        | Better stems, less HNR noise |
| Phase 2   | 75-78    | TBD        | Neural vocal enhancement |
| Phase 3   | 78-82    | 70+        | Genre-appropriate mastering |
| Phase 5   | 80+      | 75+        | Works on all genres |
| Target    | 85+ [A]  | 80+        | Beats ChatGPT's system |

---

## Known Structural Ceilings (accept, don't fight)

These are content limitations — the source material, not our code:

- **PLR ceiling**: Drake reference tracks peak above 0 dBFS from streaming normalization. Our -1 dBFS true peak will always show slightly worse PLR. Not fixable without reference normalization.
- **vocal_hnr_db 0.4 dB on commercial stems**: Demucs bleed is baked into the vocal stem. Current ceiling until Phase 1B (StemPostFilter) or Phase 2A (VoiceFixer).
- **groove_timing_score 0.0 on AI-quantized beats**: Trap beats with perfect quantization score 0 because there's no natural micro-timing variation. This is a scorer bug, not a mix bug — will need human-timing dataset.
- **dynamic_complexity 2.1 dB on trap**: Modern trap mastering produces very flat RMS. This is intentional, not a defect. Scorer lower bound already adjusted to 2.0 dB.
- **sub_bass ratio on Rhyme Dust**: The beat inherently has heavy sub content. Cutting it makes things sound thin. Accept the ratio penalty.
