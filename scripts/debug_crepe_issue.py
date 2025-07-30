import sys
from pathlib import Path
# Add project root to sys.path
sys.path.append(str(Path(__file__).resolve().parents[1]))

import hydra
import torch
from tqdm import tqdm
import logging
from src.wavemae.data.emilia_dataset import EmiliaDataset

# Configure logging
logging.basicConfig(level=logging.INFO)
LOGGER = logging.getLogger(__name__)

@hydra.main(config_path="../conf", config_name="config", version_base=None)
def main(cfg):
    """
    Debug script to find the problematic file causing the CREPE ValueError.
    """
    LOGGER.info("Initializing dataset for debugging...")
    # We check the 'train' split, as that's what the training script uses.
    dataset = EmiliaDataset(cfg, split="train")
    
    LOGGER.info(f"Found {len(dataset)} items in the dataset.")
    LOGGER.info("Starting iteration to find faulting item...")

    for i in tqdm(range(len(dataset)), desc="Debugging dataset"):
        try:
            # We access the item directly. If it fails, the exception will be caught.
            item = dataset[i]
            # Optionally, do a basic check on the output
            if 'pitch' not in item or not isinstance(item['pitch'], torch.Tensor):
                 LOGGER.warning(f"Item {i} ({dataset.items[i]['path']}) produced a malformed output.")

        except Exception as e:
            problem_item = dataset.items[i]
            LOGGER.error(f"!!!!!!!!!!!!! CRASH DETECTED !!!!!!!!!!!!!")
            LOGGER.error(f"Error occurred at index: {i}")
            LOGGER.error(f"File path: {problem_item['path']}")
            LOGGER.error(f"Error: {e}", exc_info=True) # Print full traceback
            LOGGER.error(f"!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
            # Stop after the first error to keep the log clean.
            break

    LOGGER.info("Debug script finished.")

if __name__ == "__main__":
    main() 