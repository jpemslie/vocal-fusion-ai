"""
Audio analysis + spectrogram generator for visual comparison.
Generates a side-by-side panel Claude can read as an image.
"""
import sys
import numpy as np
import librosa
import librosa.display
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import pyloudnorm as pyln
import soundfile as sf
from pathlib import Path

SR = 44100

def analyze(path):
    y, sr = librosa.load(path, sr=SR, mono=True, duration=90)  # first 90s
    meter = pyln.Meter(SR)
    lufs = meter.integrated_loudness(y)
    lra = meter.loudness_range(y) if hasattr(meter, 'loudness_range') else 0

    # Frequency band energy (dB)
    S = np.abs(librosa.stft(y, n_fft=2048))
    freqs = librosa.fft_frequencies(sr=SR)
    def band_db(lo, hi):
        mask = (freqs >= lo) & (freqs < hi)
        return float(librosa.amplitude_to_db(S[mask].mean() + 1e-9, ref=1.0))

    return {
        'y': y,
        'lufs': lufs,
        'bass_db':    band_db(20, 250),
        'lowmid_db':  band_db(250, 800),
        'mid_db':     band_db(800, 2500),
        'highmid_db': band_db(2500, 6000),
        'high_db':    band_db(6000, 20000),
        'dynamic_range': float(np.percentile(librosa.amplitude_to_db(np.abs(y)+1e-9), 99) -
                               np.percentile(librosa.amplitude_to_db(np.abs(y)+1e-9), 10)),
    }

def make_report(files, out_png="vf_data/compare.png"):
    data = {Path(f).stem[:30]: analyze(f) for f in files}
    names = list(data.keys())
    n = len(names)

    fig, axes = plt.subplots(2, n, figsize=(7*n, 10))
    if n == 1:
        axes = axes.reshape(2, 1)

    fig.suptitle("Audio Comparison", fontsize=14, fontweight='bold')

    for i, name in enumerate(names):
        d = data[name]
        # Row 0: mel spectrogram
        ax = axes[0, i]
        S_mel = librosa.feature.melspectrogram(y=d['y'], sr=SR, n_mels=128, fmax=16000)
        S_db  = librosa.power_to_db(S_mel, ref=np.max)
        img = librosa.display.specshow(S_db, sr=SR, x_axis='time', y_axis='mel',
                                       fmax=16000, ax=ax, cmap='magma')
        ax.set_title(f"{name}\nLUFS: {d['lufs']:.1f}  Dyn: {d['dynamic_range']:.1f} dB",
                     fontsize=9)
        fig.colorbar(img, ax=ax, format='%+2.0f dB')

        # Row 1: frequency balance bar chart
        ax2 = axes[1, i]
        bands = ['Bass\n20-250', 'Lo-Mid\n250-800', 'Mid\n800-2.5k',
                 'Hi-Mid\n2.5-6k', 'High\n6-20k']
        vals  = [d['bass_db'], d['lowmid_db'], d['mid_db'],
                 d['highmid_db'], d['high_db']]
        colors = ['#e74c3c','#e67e22','#f1c40f','#2ecc71','#3498db']
        bars = ax2.bar(bands, vals, color=colors, edgecolor='black', linewidth=0.5)
        ax2.set_ylabel('Avg Energy (dB)')
        ax2.set_title(f"Frequency Balance", fontsize=9)
        ax2.set_ylim(min(vals)-5, max(vals)+5)
        for bar, val in zip(bars, vals):
            ax2.text(bar.get_x() + bar.get_width()/2, val + 0.5,
                     f'{val:.0f}', ha='center', va='bottom', fontsize=8)

    plt.tight_layout()
    Path(out_png).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_png, dpi=120, bbox_inches='tight')
    plt.close()
    print(f"Saved → {out_png}")

    # Print text summary
    print("\n── Text Summary ──────────────────────────────────")
    for name, d in data.items():
        print(f"\n{name}")
        print(f"  LUFS:          {d['lufs']:.1f} dB")
        print(f"  Dynamic range: {d['dynamic_range']:.1f} dB")
        print(f"  Bass:          {d['bass_db']:.1f} dB")
        print(f"  Lo-Mid:        {d['lowmid_db']:.1f} dB")
        print(f"  Mid:           {d['mid_db']:.1f} dB")
        print(f"  Hi-Mid:        {d['highmid_db']:.1f} dB")
        print(f"  High:          {d['high_db']:.1f} dB")

if __name__ == "__main__":
    files = sys.argv[1:] if len(sys.argv) > 1 else []
    if not files:
        print("Usage: python compare.py file1.mp3 file2.wav ...")
        sys.exit(1)
    make_report(files)
