import torch
import torchaudio
import librosa
import numpy as np
import matplotlib.pyplot as plt
import os
import sys

# Add the parent directory of the rmvpe_model_code to the path
# This allows the relative import inside rmvpe.py to work.
script_dir = os.path.dirname(os.path.abspath(__file__))
rmvpe_parent_dir = os.path.join(script_dir, 'pretrained_models')
sys.path.insert(0, rmvpe_parent_dir)

from rmvpe_model_code.rmvpe import RMVPE

def visualize_pitch(audio_path, model):
    """
    Loads an audio file, runs RMVPE pitch detection, and creates a plot
    of the waveform and the resulting pitch curve.
    """
    try:
        # Load and resample audio
        audio, sr = torchaudio.load(audio_path)
        if sr != 16000:
            resampler = torchaudio.transforms.Resample(sr, 16000)
            audio = resampler(audio)
        
        if audio.ndim > 1 and audio.shape[0] > 1:
            audio = torch.mean(audio, dim=0, keepdim=True)
            
        audio = audio.squeeze(0).cpu().numpy()

        # Run RMVPE
        pitch = model.infer_from_audio(audio, thred=0.03)

        # Create plot
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8), sharex=True)

        # Plot waveform
        time = np.linspace(0, len(audio) / 16000, num=len(audio))
        ax1.plot(time, audio)
        ax1.set_title("Audio Waveform")
        ax1.set_ylabel("Amplitude")
        ax1.grid(True)

        # Plot pitch curve
        pitch_time = np.linspace(0, len(audio) / 16000, num=len(pitch))
        ax2.plot(pitch_time, pitch)
        ax2.set_title("RMVPE Pitch Curve")
        ax2.set_xlabel("Time (s)")
        ax2.set_ylabel("Frequency (Hz)")
        ax2.grid(True)

        plt.tight_layout()
        plt.savefig("pitch_visualization.png")
        print("Successfully generated pitch_visualization.png")

    except Exception as e:
        print(f"Error during visualization: {e}")

if __name__ == '__main__':
    model_path = 'pretrained_models/rmvpe_safe-models/model.safetensors'
    audio_file = '/media/k4/storage2/Datasets/Emilia/EN_EXTRACTED/EN_B00002_S06974_W000043.mp3'

    print("--- Visualizing Pitch Curve ---")
    
    # Initialize the model
    # Note: is_half=False because we are running on CPU for this test.
    # The safetensors model is float32.
    rmvpe_model = RMVPE(model_path, is_half=False, device='cpu')

    visualize_pitch(audio_file, rmvpe_model)
    print("-" * 20) 