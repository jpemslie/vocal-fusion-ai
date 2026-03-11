"""Audio artifact detector — generates images for visual inspection."""
import numpy as np
import librosa
import soundfile as sf
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from pathlib import Path

OUT = "/Users/jpemslie/Desktop/VocalFusion/vf_data/mixes/qa_check.wav"
OUTDIR = Path("/Users/jpemslie/Desktop/VocalFusion/vf_data/qa")
OUTDIR.mkdir(parents=True, exist_ok=True)
SR = 44100

y, sr = sf.read(OUT)
if y.ndim == 2:
    mono = y.mean(axis=1).astype(np.float32)
else:
    mono = y.astype(np.float32)

print(f"Duration: {len(mono)/SR:.1f}s  Peak: {np.max(np.abs(mono)):.4f}  RMS: {np.sqrt(np.mean(mono**2)):.4f}")

# 1. Full waveform — look for clipping (flat tops), clicks (spikes), static (dense noise)
fig, ax = plt.subplots(figsize=(20, 4))
t = np.arange(len(mono)) / SR
ax.plot(t, mono, linewidth=0.3, alpha=0.8)
ax.axhline(0.99, color='red', linewidth=1, linestyle='--', label='clip zone')
ax.axhline(-0.99, color='red', linewidth=1, linestyle='--')
ax.set_xlabel('Time (s)'); ax.set_ylabel('Amplitude')
ax.set_title('Waveform — check for flat-top clipping, random spikes (clicks), dense noise floor')
ax.legend()
plt.tight_layout()
plt.savefig(str(OUTDIR / '1_waveform.png'), dpi=120)
plt.close()
print(f"Saved: {OUTDIR}/1_waveform.png")

# 2. Full spectrogram — static appears as horizontal broadband noise floor
fig, ax = plt.subplots(figsize=(20, 6))
D = librosa.amplitude_to_db(np.abs(librosa.stft(mono, n_fft=2048)), ref=np.max)
librosa.display.specshow(D, sr=SR, x_axis='time', y_axis='log', ax=ax, cmap='magma')
ax.set_title('Full Spectrogram (log freq) — static = raised noise floor across all freqs')
plt.colorbar(ax.collections[0], ax=ax, format='%+2.0f dB')
plt.tight_layout()
plt.savefig(str(OUTDIR / '2_spectrogram_full.png'), dpi=100)
plt.close()
print(f"Saved: {OUTDIR}/2_spectrogram_full.png")

# 3. Zoom into first 30s where issues most noticeable
clip30 = mono[:SR*30]
fig, axes = plt.subplots(2, 1, figsize=(20, 8))
t30 = np.arange(len(clip30)) / SR
axes[0].plot(t30, clip30, linewidth=0.3)
axes[0].set_title('Waveform: first 30s'); axes[0].set_ylabel('Amp')
D30 = librosa.amplitude_to_db(np.abs(librosa.stft(clip30)), ref=np.max)
librosa.display.specshow(D30, sr=SR, x_axis='time', y_axis='log', ax=axes[1], cmap='magma')
axes[1].set_title('Spectrogram: first 30s')
plt.tight_layout()
plt.savefig(str(OUTDIR / '3_first30s.png'), dpi=120)
plt.close()
print(f"Saved: {OUTDIR}/3_first30s.png")

# 4. Click / transient detection — look for abnormal spikes
from scipy.signal import find_peaks
abs_mono = np.abs(mono)
rms_global = float(np.sqrt(np.mean(mono**2)))
# A "click" is any sample > 6x global RMS that is isolated (not part of a musical transient)
peaks, props = find_peaks(abs_mono, height=rms_global * 6, distance=100)
print(f"\nArtifact analysis:")
print(f"  Global RMS: {rms_global:.4f} ({20*np.log10(rms_global+1e-9):.1f} dBFS)")
print(f"  Peak amplitude: {np.max(abs_mono):.4f} ({20*np.log10(np.max(abs_mono)+1e-9):.1f} dBFS)")
print(f"  Samples clipped (>0.99): {np.sum(abs_mono > 0.99)}")
print(f"  High-amplitude spikes (>6x RMS): {len(peaks)}")
if len(peaks) > 0:
    print(f"  Spike times (s): {[f'{p/SR:.2f}' for p in peaks[:20]]}")

# 5. Noise floor check — measure RMS during quiet sections
# Split into 1s windows, find 10th percentile = noise floor
win = SR
n_win = len(mono) // win
window_rms = [np.sqrt(np.mean(mono[i*win:(i+1)*win]**2)) for i in range(n_win)]
noise_floor_rms = float(np.percentile(window_rms, 10))
loudest_rms = float(np.percentile(window_rms, 90))
print(f"  Noise floor (10th pct window RMS): {20*np.log10(noise_floor_rms+1e-9):.1f} dBFS")
print(f"  Loud section RMS (90th pct): {20*np.log10(loudest_rms+1e-9):.1f} dBFS")
print(f"  Dynamic range (loud-floor): {20*np.log10((loudest_rms+1e-9)/(noise_floor_rms+1e-9)):.1f} dB")

# 6. High-frequency noise check (static = elevated 8-20kHz noise)
from scipy.signal import butter, sosfilt
sos_hf = butter(4, 8000/(SR/2), btype='high', output='sos')
hf_content = sosfilt(sos_hf, mono)
hf_rms = float(np.sqrt(np.mean(hf_content**2)))
total_rms = float(np.sqrt(np.mean(mono**2)))
hf_ratio = hf_rms / (total_rms + 1e-9)
print(f"  HF content ratio (8kHz+): {hf_ratio:.3f} ({hf_ratio*100:.1f}% of total RMS)")
print(f"  [Normal: 5-15%. >25% = too bright/harsh. >40% = static/noise probable]")
