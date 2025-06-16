import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parents[1]))

import hydra
import torch
from omegaconf import DictConfig, OmegaConf
from src.wavemae.data.emilia_dataset import EmiliaDataset

def get_device():
    if torch.cuda.is_available():
        return torch.device("cuda")
    elif torch.backends.mps.is_available():
        return torch.device("mps")
    else:
        return torch.device("cpu")

@hydra.main(config_path="../conf", config_name="config", version_base=None)
def main(cfg: DictConfig):
    print("--- Testing On-the-Fly Data Processing ---")
    print(OmegaConf.to_yaml(cfg.data))
    
    device = get_device()
    print(f"Using device: {device}")

    try:
        # Use dev split for testing
        dataset = EmiliaDataset(cfg=cfg, split="dev", device=device)
        print(f"Dataset for 'dev' split loaded successfully. Number of samples: {len(dataset)}")
    except Exception as e:
        print(f"Failed to instantiate dataset: {e}")
        return

    if len(dataset) == 0:
        print("Dataset is empty. Cannot perform test.")
        return

    # --- Test a few samples ---
    num_samples_to_test = min(3, len(dataset))
    print(f"\\nFetching and processing {num_samples_to_test} samples...")

    for i in range(num_samples_to_test):
        print(f"\\n--- Sample {i} ---")
        try:
            sample = dataset[i]
            
            stft = sample['stft']
            pitch = sample['pitch']

            print(f"Item ID: {sample['id']}")
            
            # Check STFT
            print(f"STFT shape: {stft.shape}")
            assert len(stft.shape) == 2, "STFT tensor should be 2D"
            assert stft.dtype == torch.float32, "STFT tensor should be float32"
            print("STFT check: PASS")

            # Check Pitch
            print(f"Pitch shape: {pitch.shape}")
            assert len(pitch.shape) == 1, "Pitch tensor should be 1D"
            assert pitch.shape[0] == stft.shape[1], "Pitch length must match STFT length"
            assert pitch.dtype == torch.float32, "Pitch tensor should be float32"
            print("Pitch check: PASS")

        except Exception as e:
            print(f"Failed to process sample {i}: {e}")
            raise

    print("\\n---------------------------------")
    print("On-the-fly processing test completed successfully.")

if __name__ == "__main__":
    main() 