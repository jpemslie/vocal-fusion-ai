"""
VocalFusion AI — Intelligent Orchestrator v10.0
=================================================

Uses the AI DJ: every decision is a scored experiment.
No hardcoded templates. The system tries things, listens,
and keeps what sounds best.

Pipeline:
  1. SongAnalyzer  — Deep analysis of both songs
  2. MatchFinder   — Find compatible sections (for direction decision)
  3. AI DJ         — Makes all mixing decisions by experimenting + scoring
  4. MixRefiner    — Final polish (iterate on weak dimensions)
"""

import numpy as np
from typing import Dict, Optional


from analysis.song_dna import SongAnalyzer
from analysis.match_finder import MatchFinder
from core.ai_dj import AIDJ


class MixingEngineV2:
    """Intelligent mashup engine — powered by AI DJ"""

    def __init__(self, sample_rate: int = 44100, ir_path: Optional[str] = None):
        self.sr = sample_rate
        self.analyzer = SongAnalyzer(sample_rate)
        self.matcher = MatchFinder(sample_rate)
        self.dj = AIDJ(sample_rate)

    def create_mix(self, stems_a: Dict[str, np.ndarray],
                   stems_b: Dict[str, np.ndarray],
                   arrangement_plan: Dict,
                   mixing_plan: Dict,
                   predicted_params: Optional[Dict] = None) -> Dict[str, np.ndarray]:
        """
        Create an AI-driven mashup.
        Same interface as always — drop-in replacement for run.py.
        """
        print(f"\n{'='*60}")
        print(f"  VOCALFUSION v10.0 — AI DJ")
        print(f"{'='*60}")

        # ============================================================
        # PHASE 1: DEEP ANALYSIS
        # ============================================================
        print(f"\n  PHASE 1: Analyzing songs...")
        print(f"  {'─'*50}")

        print(f"    Analyzing Song A...")
        dna_a = self.analyzer.analyze(stems_a)

        print(f"\n    Analyzing Song B...")
        dna_b = self.analyzer.analyze(stems_b)

        print(f"\n    A: {dna_a.beat_grid.tempo:.0f} BPM, {dna_a.key}")
        print(f"    B: {dna_b.beat_grid.tempo:.0f} BPM, {dna_b.key}")

        # ============================================================
        # PHASE 2: AI DJ — Experiment → Score → Decide
        # ============================================================
        print(f"\n  PHASE 2: AI DJ")
        print(f"  {'─'*50}")

        result = self.dj.create_mashup(stems_a, stems_b, dna_a, dna_b,
                                       predicted_params=predicted_params)

        final_score = result.get('quality_scores')

        dur = len(result['full_mix']) / self.sr
        print(f"\n{'='*60}")
        score_str = f"{final_score.overall:.3f}" if final_score else "n/a"
        print(f"  COMPLETE: {dur:.1f}s | Score: {score_str}")
        print(f"{'='*60}\n")

        return result
