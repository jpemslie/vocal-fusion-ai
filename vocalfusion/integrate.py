"""
VocalFusion AI - Integration Patch (v2.1)

This file patches the new upgraded modules into your existing vocalfusion monolith.
It replaces the MixingEngine with MixingEngineV2 and wires in EnhancedDSP.

HOW TO USE:
===========
1. Put these files in your project:
      vocal-fusion-ai/
      ├── fusionnew.py              (your existing monolith)
      ├── vocalfusion/
      │   ├── __init__.py
      │   ├── dsp.py                (enhanced DSP - reverb, beat align, smooth gains)
      │   ├── mixing_v2.py          (upgraded mixer using new DSP)
      │   └── integrate.py          (THIS FILE - wires it all together)

2. Run with the upgraded engine:
      python -m vocalfusion.integrate analyze /path/to/song.mp3 --name "test"
      python -m vocalfusion.integrate serve
      python -m vocalfusion.integrate fuse song_a song_b

   OR import in your own code:
      from vocalfusion.integrate import VocalFusionV2
      engine = VocalFusionV2()
      engine.process_single_song(Path("song.mp3"), "my_song")
      engine.fuse_two_songs("song_a_id", "song_b_id")

WHAT'S DIFFERENT:
=================
  - Beat grids are aligned before mixing (fixes sloppy rhythm)
  - Vocal switching uses 300ms windows + 50ms ramps (fixes choppy switching)
  - Reverb uses impulse response convolution (fixes metallic sound)
  - Equal-power crossfades at section boundaries
  - Phase-coherent drum layering
  - Frequency-split bass (no phase cancellation in sub)
  - All the analysis, arrangement, compatibility, and web UI stay the same
"""

import sys
import os
import json
import argparse
import numpy as np
import librosa
import soundfile as sf
from pathlib import Path
from dataclasses import asdict
from datetime import datetime

# Add parent directory to path so we can import the monolith
parent_dir = str(Path(__file__).resolve().parent.parent)
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)

# Import everything from the monolith EXCEPT the old MixingEngine
# We'll use our own
try:
    from fusionnew import (
        VocalFusionEngine, StemSeparationEngine, SongAnalysisEngine,
        CompatibilityEngine, ArrangementEngine,
        SongAnalysis, VocalAnalysis, CompatibilityScore,
        app, HTML_TEMPLATE
    )
    MONOLITH_AVAILABLE = True
except ImportError:
    print("WARNING: Could not import fusionnew.py (the monolith).")
    print("Make sure fusionnew.py is in the parent directory of vocalfusion/")
    MONOLITH_AVAILABLE = False

# Import our upgraded modules
from vocalfusion.dsp import EnhancedDSP
from vocalfusion.mixing_v2 import MixingEngineV2


class VocalFusionV2(VocalFusionEngine if MONOLITH_AVAILABLE else object):
    """
    VocalFusion with upgraded mixing engine.

    Everything is the same as the monolith EXCEPT:
      - self.mixing_engine is MixingEngineV2 (not the old MixingEngine)
      - fuse_two_songs uses the new beat alignment + smooth mixing
    """

    def __init__(self, base_dir: str = "vf_data", ir_path: str = None):
        """
        Args:
            base_dir: Working directory for all VocalFusion data
            ir_path: Optional path to a reverb impulse response WAV file.
                     If provided, vocal reverb will sound dramatically better.
                     Free IRs at: https://www.voxengo.com/impulses/
        """
        if not MONOLITH_AVAILABLE:
            raise RuntimeError(
                "Cannot find fusionnew.py. Make sure it's in the parent directory.\n"
                "Expected layout:\n"
                "  vocal-fusion-ai/\n"
                "    fusionnew.py\n"
                "    vocalfusion/\n"
                "      integrate.py (this file)\n"
                "      dsp.py\n"
                "      mixing_v2.py"
            )

        # Initialize the parent (monolith) engine
        super().__init__(base_dir)

        # OVERRIDE: Replace the old mixer with our upgraded one
        self.mixing_engine = MixingEngineV2(
            sample_rate=self.sample_rate,
            ir_path=ir_path
        )
        self.ir_path = ir_path

        print(f"VocalFusion v2.1 initialized (upgraded mixing engine)")
        if ir_path:
            print(f"  Using impulse response: {ir_path}")
        else:
            print(f"  Using synthetic IR reverb (for better quality, provide an IR WAV file)")

    def fuse_two_songs(self, song_a_id, song_b_id, arrangement_strategy=None):
        """
        Fuse two songs using the upgraded mixing engine.

        This overrides the monolith's method to add:
          - Beat grid alignment
          - Smooth vocal mixing
          - Better reverb
        """
        print(f"\n{'=' * 60}")
        print(f"FUSING SONGS (v2.1): {song_a_id} + {song_b_id}")
        print(f"{'=' * 60}\n")

        # Steps 1-3 are the same as the monolith
        print("Step 1: Loading song analyses...")
        analysis_a = self._load_song_analysis(song_a_id)
        analysis_b = self._load_song_analysis(song_b_id)

        print("\nStep 2: Analyzing compatibility...")
        compat_dir = self.base_dir / "compatibility" / f"{song_a_id}_{song_b_id}"
        compat_dir.mkdir(parents=True, exist_ok=True)
        compatibility = self.compatibility_engine.analyze_compatibility(analysis_a, analysis_b)
        with open(compat_dir / "compatibility_report.json", 'w') as f:
            json.dump(asdict(compatibility), f, indent=2, default=str)

        print("\nStep 3: Creating arrangement plan...")
        if arrangement_strategy:
            compatibility.arrangement_strategies = [arrangement_strategy]
        arr_dir = self.base_dir / "arrangements" / f"{song_a_id}_{song_b_id}"
        arr_dir.mkdir(parents=True, exist_ok=True)
        arrangement_plan = self.arrangement_engine.create_arrangement_plan(
            analysis_a, analysis_b, compatibility)

        # Inject tempo info for beat alignment
        arrangement_plan['vocal_plan']['tempo_adjustment']['song_a_tempo'] = analysis_a.tempo
        arrangement_plan['vocal_plan']['tempo_adjustment']['song_b_tempo'] = analysis_b.tempo

        with open(arr_dir / "arrangement_plan.json", 'w') as f:
            json.dump(arrangement_plan, f, indent=2, default=str)

        # Step 4: Load stems
        print("\nStep 4: Loading stems...")
        stems_a = self._load_stems(song_a_id)
        stems_b = self._load_stems(song_b_id)

        if not stems_a:
            raise RuntimeError(f"No stems found for {song_a_id}. Run analysis first.")
        if not stems_b:
            raise RuntimeError(f"No stems found for {song_b_id}. Run analysis first.")

        # Step 5: Create mix with UPGRADED engine
        print("\nStep 5: Creating mix (v2.1 upgraded engine)...")
        mix_dir = self.base_dir / "mixes" / f"{song_a_id}_{song_b_id}"
        mix_dir.mkdir(parents=True, exist_ok=True)

        mix_result = self.mixing_engine.create_mix(
            stems_a, stems_b, arrangement_plan,
            arrangement_plan['mixing_plan'])

        # Step 6: Save
        print("\nStep 6: Saving mixes...")
        for stem_name, audio in mix_result.items():
            if audio is not None and len(audio) > 0:
                output_path = mix_dir / f"{stem_name}.wav"
                sf.write(str(output_path), audio, self.sample_rate)
                duration = len(audio) / self.sample_rate
                print(f"  Saved {stem_name}.wav ({duration:.1f}s)")

        # Step 7: Report
        print("\nStep 7: Creating fusion report...")
        fusion_report = self._create_fusion_report(
            analysis_a, analysis_b, compatibility, arrangement_plan)
        report_path = mix_dir / "fusion_report.json"
        with open(report_path, 'w') as f:
            json.dump(fusion_report, f, indent=2, default=str)

        print(f"\n{'=' * 60}")
        print(f"FUSION COMPLETE: {song_a_id} + {song_b_id}")
        print(f"Output: {mix_dir}")
        print(f"{'=' * 60}\n")

        return {
            'song_a_id': song_a_id,
            'song_b_id': song_b_id,
            'compatibility': compatibility,
            'arrangement_plan': arrangement_plan,
            'fusion_report': fusion_report,
            'paths': {
                'compatibility': str(compat_dir / "compatibility_report.json"),
                'arrangement': str(arr_dir / "arrangement_plan.json"),
                'mixes': str(mix_dir),
                'report': str(report_path)
            }
        }


# ============================================================================
# CLI
# ============================================================================

def main():
    parser = argparse.ArgumentParser(
        description='VocalFusion AI v2.1 - Upgraded Mixing Engine',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog='''
Examples:
  python -m vocalfusion.integrate analyze song.mp3 --name "My Song"
  python -m vocalfusion.integrate list
  python -m vocalfusion.integrate compatibility song1 song2
  python -m vocalfusion.integrate fuse song1 song2
  python -m vocalfusion.integrate fuse song1 song2 --ir reverb.wav
  python -m vocalfusion.integrate serve
        ''')

    parser.add_argument('--ir', help='Path to reverb impulse response WAV file')

    subparsers = parser.add_subparsers(dest='command', help='Commands')

    # analyze
    analyze_p = subparsers.add_parser('analyze', help='Analyze a single song')
    analyze_p.add_argument('file', help='Audio file to analyze')
    analyze_p.add_argument('--name', help='Song name')

    # list
    subparsers.add_parser('list', help='List analyzed songs')

    # compatibility
    comp_p = subparsers.add_parser('compatibility', help='Check compatibility')
    comp_p.add_argument('song_a', help='First song ID')
    comp_p.add_argument('song_b', help='Second song ID')

    # fuse
    fuse_p = subparsers.add_parser('fuse', help='Fuse two songs')
    fuse_p.add_argument('song_a', help='First song ID')
    fuse_p.add_argument('song_b', help='Second song ID')
    fuse_p.add_argument('--strategy', choices=[
        'harmony_focused', 'call_and_response', 'counterpoint', 'layered'])
    fuse_p.add_argument('--ir', dest='fuse_ir', help='Impulse response WAV for this fusion')

    # serve
    serve_p = subparsers.add_parser('serve', help='Start web server')
    serve_p.add_argument('--port', type=int, default=5000)
    serve_p.add_argument('--host', default='127.0.0.1')

    args = parser.parse_args()
    ir_path = args.ir if hasattr(args, 'ir') else None

    if args.command == 'analyze':
        engine = VocalFusionV2(ir_path=ir_path)
        result = engine.process_single_song(Path(args.file), args.name)
        print(f"\nSong ID: {result['song_id']}")
        a = result['analysis']
        print(f"Duration: {a.duration:.1f}s | Key: {a.key} | Tempo: {a.tempo:.0f} BPM")
        if a.vocals:
            print(f"Voice: {a.vocals.voice_type.value} | "
                  f"Range: {a.vocals.range_low_hz:.0f}-{a.vocals.range_high_hz:.0f} Hz")
        print(f"Saved to: {result['paths']['analysis']}")

    elif args.command == 'list':
        analysis_dir = Path("vf_data") / "analysis"
        if not analysis_dir.exists():
            print("No songs analyzed yet.")
            return
        print(f"\n{'Song ID':<40} {'Duration':>8} {'Key':>10} {'Tempo':>8} {'Voice':>12}")
        print("-" * 82)
        for d in sorted(analysis_dir.iterdir()):
            if d.is_dir() and (d / "analysis.json").exists():
                try:
                    with open(d / "analysis.json") as f:
                        data = json.load(f)
                    voice = data.get('vocals', {}).get('voice_type', '-') if data.get('vocals') else '-'
                    print(f"{d.name:<40} {data.get('duration', 0):>7.1f}s "
                          f"{data.get('key', '?'):>10} {data.get('tempo', 0):>7.0f} {voice:>12}")
                except Exception:
                    print(f"{d.name:<40} {'(error reading)':>40}")

    elif args.command == 'compatibility':
        engine = VocalFusionV2(ir_path=ir_path)
        aa = engine._load_song_analysis(args.song_a)
        ab = engine._load_song_analysis(args.song_b)
        c = engine.compatibility_engine.analyze_compatibility(aa, ab)
        print(f"\nOverall: {c.overall_score:.2f} | Difficulty: {c.difficulty_score:.2f}")
        print(f"Key: {c.key_compatibility:.2f} ({aa.key} / {ab.key})")
        print(f"  → Transpose B by {c.recommended_transposition_semitones} semitones to {c.recommended_key}")
        print(f"Tempo: {c.tempo_compatibility:.2f} ({aa.tempo:.0f} / {ab.tempo:.0f} BPM)")
        print(f"  → Adjust by {c.recommended_tempo_adjustment_ratio:.3f}x")
        print(f"Vocal Blend: {c.vocal_blend_score:.2f} | Range: {c.range_compatibility:.2f}")
        if c.challenges:
            print(f"\nChallenges: {', '.join(c.challenges[:3])}")
        if c.opportunities:
            print(f"Opportunities: {', '.join(c.opportunities[:3])}")

    elif args.command == 'fuse':
        fuse_ir = args.fuse_ir if hasattr(args, 'fuse_ir') and args.fuse_ir else ir_path
        engine = VocalFusionV2(ir_path=fuse_ir)
        result = engine.fuse_two_songs(args.song_a, args.song_b, args.strategy)
        print(f"Strategy: {result['arrangement_plan']['strategy']}")
        print(f"Output: {result['paths']['mixes']}")

    elif args.command == 'serve':
        # Patch the Flask app to use our upgraded engine
        import vocalfusion.integrate as mod
        mod._global_engine = VocalFusionV2(ir_path=ir_path)

        # Override the Flask routes to use our engine
        _patch_flask_app(mod._global_engine)

        print(f"\nVocalFusion v2.1 Web Server")
        print(f"http://{args.host}:{args.port}")
        print(f"Press Ctrl+C to stop\n")
        app.run(host=args.host, port=args.port, debug=False)

    else:
        parser.print_help()


def _patch_flask_app(engine):
    """Replace the monolith's global engine with our upgraded one"""
    try:
        import fusionnew
        fusionnew.engine = engine
    except Exception as e:
        print(f"Warning: Could not patch Flask app: {e}")


# Allow running as module: python -m vocalfusion.integrate
if __name__ == "__main__":
    main()


# Patch: make dict-based vocals work with attribute access
class DictAsObj:
    """Wraps a dict so you can access keys as attributes"""
    def __init__(self, d):
        for k, v in d.items():
            if isinstance(v, dict):
                setattr(self, k, DictAsObj(v))
            elif isinstance(v, list):
                setattr(self, k, [DictAsObj(i) if isinstance(i, dict) else i for i in v])
            else:
                setattr(self, k, v)
    def __getattr__(self, name):
        return None
    def get(self, key, default=None):
        return getattr(self, key, default)

# Override _load_song_analysis to fix dict-vs-object issue
_original_load = VocalFusionV2._load_song_analysis

def _patched_load(self, song_id):
    result = _original_load(self, song_id)
    if isinstance(result.vocals, dict):
        result.vocals = DictAsObj(result.vocals)
    return result

VocalFusionV2._load_song_analysis = _patched_load
