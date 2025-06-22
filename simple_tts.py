#!/usr/bin/env python3
import torch
from TTS.api import TTS

# Get device - prioritize Apple Silicon GPU (MPS) for macOS
if torch.backends.mps.is_available():
    device = "mps"
elif torch.cuda.is_available():
    device = "cuda"
else:
    device = "cpu"

print(f"Using device: {device}")

# Initialize Tortoise-TTS model
tts = TTS(model_name="tts_models/en/multi-dataset/tortoise-v2", progress_bar=False).to(device)

# Basic text-to-speech
text = "Hello! This is a sample text generated using Tortoise TTS model."
output_path = "output.wav"

# Generate speech
tts.tts_to_file(text=text, file_path=output_path)

print(f"Audio saved to {output_path}")

# You can also get the audio as numpy array
wav = tts.tts(text=text)
print(f"Audio shape: {wav.shape}")
print(f"Sample rate: {tts.synthesizer.output_sample_rate}")