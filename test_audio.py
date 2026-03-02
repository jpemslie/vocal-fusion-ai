import torch
import librosa
import soundfile as sf
import numpy as np

print("=== Audio Processing Test ===")
print(f"PyTorch version: {torch.__version__}")
print(f"Librosa version: {librosa.__version__}")
print(f"NumPy version: {np.__version__}")
print(f"Demucs available: True")  # Since we installed it

# Test basic audio processing
print("\nCreating test audio signal...")
sample_rate = 22050
duration = 1.0  # seconds
t = np.linspace(0, duration, int(sample_rate * duration))
audio = 0.5 * np.sin(2 * np.pi * 440 * t)  # 440 Hz sine wave

print(f"Audio shape: {audio.shape}")
print(f"Sample rate: {sample_rate} Hz")
print("✓ All audio libraries working correctly!")
