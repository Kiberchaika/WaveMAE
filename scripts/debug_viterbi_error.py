import torch
import torchaudio
import torchcrepe
import librosa
import numpy as np
import os
import json
import torch.nn.functional as F
import torchaudio.transforms as T
import logging

LOGGER = logging.getLogger(__name__)

def get_emilia_data_path(file_id, index_path="precomputed_data/emilia_file_index.json"):
    """
    Finds the full path to an EMILIA data file by looking it up in the dataset index.
    """
    if not os.path.exists(index_path):
        raise FileNotFoundError(
            f"Dataset index not found at {index_path}. "
            "Please run the dataset scanning/pre-computation first, or provide the correct path."
        )

    with open(index_path, 'r') as f:
        data_splits = json.load(f)
    
    for split in data_splits.values():
        for item in split:
            if item['id'] == file_id:
                return item['path']
    
    raise ValueError(f"File ID '{file_id}' not found in the dataset index.")

# --- Copied directly from src/wavemae/data/emilia_dataset.py ---

def _compute_pitch_worker_safe(waveform: torch.Tensor, sr: int) -> torch.Tensor:
    """
    Computes pitch using a CREPE model in a worker-safe manner.
    This method loads its own model instance to avoid multiprocessing issues
    with torchcrepe's cached model.
    It returns the raw logits from the model.
    """
    target_sr_16k = 16000
    model_size = 'tiny'
    # Manually construct path to avoid file lock in crepe.load.model_path
    filename = f'{model_size}.pth'
    try:
        import torchcrepe as crepe
        crepe_dir = os.path.dirname(crepe.__file__)
        model_path = os.path.join(crepe_dir, 'assets', filename)
    except (ImportError, AttributeError):
        crepe_repo_path = os.path.join(os.path.dirname(__file__), '..', '..', '..', 'torchcrepe')
        if os.path.exists(crepe_repo_path):
             model_path = os.path.join(crepe_repo_path, 'torchcrepe', 'assets', filename)
        else:
            model_path = os.path.join('torchcrepe', 'assets', filename)

    if not os.path.exists(model_path):
         import torchcrepe as crepe
         crepe.load.model(model_size, 'cpu')

    import torchcrepe as crepe
    model = crepe.Crepe(model_size)
    model.load_state_dict(torch.load(model_path, map_location=waveform.device))
    model.to(waveform.device)
    model.eval()

    if waveform.ndim > 1:
        waveform = torch.mean(waveform, dim=0, keepdim=True)

    resampler = T.Resample(orig_freq=sr, new_freq=target_sr_16k).to(waveform.device)
    resampled_waveform = resampler(waveform)

    hop_length = int(target_sr_16k / 100)
    
    frame_generator = crepe.preprocess(
        resampled_waveform,
        sample_rate=target_sr_16k,
        hop_length=hop_length,
        batch_size=2048, 
        device=str(waveform.device),
        pad=True
    )

    all_logits = []
    for frames in frame_generator:
        with torch.no_grad():
            logits = model(frames)
            all_logits.append(logits)
    
    if not all_logits:
        return torch.empty((1, 360, 0), device=waveform.device)

    all_logits = torch.cat(all_logits, dim=0)
    all_logits = all_logits.permute(1, 0)
    all_logits = all_logits.unsqueeze(0)
    
    return all_logits

# --- End of copied code ---

def main():
    problematic_file_id = "EN_B00002_S04234_W000002"
    
    try:
        audio_path = get_emilia_data_path(problematic_file_id)
    except (FileNotFoundError, ValueError) as e:
        print(e)
        return
        
    print(f"Loading problematic file: {audio_path}")
    
    try:
        waveform, sr = torchaudio.load(audio_path)
    except Exception as e:
        print(f"Error loading audio file: {e}")
        return

    print(f"Original waveform shape: {waveform.shape}, Sample rate: {sr}")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    waveform = waveform.to(device)
    
    print("\n--- Replicating EmiliaDataset.__getitem__ logic ---")

    try:
        print("1. Computing pitch logits with _compute_pitch_worker_safe...")
        logits = _compute_pitch_worker_safe(waveform, sr)
        print(f"Logits shape: {logits.shape}")

        if logits.shape[-1] == 0:
            print("Pitch extraction resulted in empty logits. Exiting.")
            return

        print("2. Applying sigmoid to get emission probabilities...")
        emission_probabilities = torch.sigmoid(logits).squeeze(0).cpu().numpy()
        print(f"Emission probabilities shape: {emission_probabilities.shape}")
        
        if emission_probabilities.shape[1] < 2:
            print(f"WARNING: Viterbi input has fewer than 2 steps ({emission_probabilities.shape[1]}). This will likely fail.")

        print("3. Creating transition matrix...")
        transition_matrix = librosa.sequence.transition_loop(360, 0.9)
        print(f"Transition matrix shape: {transition_matrix.shape}")
        
        print("4. Calling librosa.sequence.viterbi...")
        pitch_midi_viterbi = librosa.sequence.viterbi(emission_probabilities, transition_matrix)
        print("Viterbi decoding successful!")
        print(f"Decoded path shape: {pitch_midi_viterbi.shape}")

    except Exception as e:
        import traceback
        tb_str = traceback.format_exc()
        print(f"\n--- PROCESSING FAILED ---")
        print(f"Error: {e}\n{tb_str}")


if __name__ == "__main__":
    main() 