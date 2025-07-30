import sys
from pathlib import Path

import hydra
import torch
from omegaconf import DictConfig
from torch.utils.data import DataLoader

# Add project root to path
sys.path.append(str(Path(__file__).resolve().parents[1]))

from src.wavemae.data.emilia_dataset import EmiliaDataset
from scripts.train import collate_fn

@hydra.main(config_path="../conf", config_name="config", version_base=None)
def main(cfg: DictConfig):
    print("--- Starting DataLoader Debug Script ---")
    
    # Use a single device for simplicity
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # 1. Instantiate the Dataset
    try:
        dataset = EmiliaDataset(cfg=cfg, split="dev")
        print(f"Dataset for 'dev' split loaded successfully. Number of samples: {len(dataset)}")
    except Exception as e:
        print(f"Failed to instantiate dataset: {e}")
        return

    # 2. Instantiate the DataLoader with num_workers=0
    print("Instantiating DataLoader with num_workers=0...")
    data_loader = DataLoader(
        dataset,
        batch_size=4,
        shuffle=True,
        collate_fn=collate_fn,
        num_workers=0  # This is the important part for debugging
    )
    print("DataLoader instantiated.")

    # 3. Try to fetch one batch
    try:
        print("\\nAttempting to fetch one batch...")
        batch = next(iter(data_loader))
        print("--- Batch Fetched Successfully! ---")
        
        # 4. Inspect the batch
        print("\\nInspecting batch contents:")
        for key, value in batch.items():
            if isinstance(value, torch.Tensor):
                print(f"  - Key: '{key}', Type: Tensor, Shape: {value.shape}, Dtype: {value.dtype}")
            else:
                print(f"  - Key: '{key}', Type: {type(value)}, Length: {len(value)}")
        
        print("\\nDebug script finished successfully.")

    except Exception as e:
        print(f"\\n--- FAILED to fetch a batch ---")
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main() 