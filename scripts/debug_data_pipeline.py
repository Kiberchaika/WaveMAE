import logging
import hydra
from omegaconf import DictConfig
import sys
from pathlib import Path
from tqdm import tqdm

# Add project root to path
sys.path.append(str(Path(__file__).resolve().parents[1]))

from src.wavemae.data.emilia_dataset import EmiliaDataset

# Configure logging to show info level messages
logging.basicConfig(level=logging.INFO, format='[%(asctime)s][%(name)s][%(levelname)s] - %(message)s')
LOGGER = logging.getLogger(__name__)

@hydra.main(config_path="../conf", config_name="config", version_base=None)
def main(cfg: DictConfig):
    """
    Initializes the dataset and iterates through it to identify problematic files.
    """
    LOGGER.info("Creating dataset for debugging...")
    # We use the 'train' split as it's the one causing issues.
    dataset = EmiliaDataset(config=cfg, split="train") 
    
    LOGGER.info(f"Dataset size: {len(dataset)}")
    LOGGER.info("Starting iteration to find the hanging file...")

    try:
        for i in tqdm(range(len(dataset)), desc="Processing files"):
            try:
                # We call __getitem__ directly, simulating a single worker.
                item = dataset[i]
                if item is None:
                    item_info = dataset.file_index[dataset.split][i]
                    file_path = item_info.get("path", "N/A")
                    LOGGER.warning(f"Item {i} ({file_path}) failed to load (returned None).")
            except Exception as e:
                item_info = dataset.file_index[dataset.split][i]
                file_path = item_info["path"]
                LOGGER.critical(f"A critical error occurred while processing file {i} ({file_path}).", exc_info=True)
                LOGGER.critical("Stopping debugging session.")
                break
    finally:
        LOGGER.info(f"Finished checking files.")

if __name__ == "__main__":
    main() 