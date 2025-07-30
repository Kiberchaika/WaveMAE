import hydra
import torch
import torchaudio
import numpy as np
import logging
from torch.utils.data import Dataset, DataLoader
from pathlib import Path
import random
import sys
import os
import torchcrepe
from omegaconf import DictConfig, OmegaConf
from tqdm import tqdm
import json

# Add project root to path
sys.path.append(str(Path(__file__).resolve().parents[1]))

LOGGER = logging.getLogger(__name__)

# A self-contained dataset that only returns raw audio and its original path
class AudioOnlyDataset(Dataset):
    def __init__(self, config, split='train'):
        self.config = config
        self.split = split
        self.data_path = Path(config.data.path)
        self.file_index = self._build_index(config.data.cache_path, self.data_path)[split]
        
        self.sample_rate = config.data.sampling_rate
        self.segment_length = int(config.data.segment_duration_secs * self.sample_rate)

    def _build_index(self, index_path, data_path):
        cache_path = Path(index_path)
        if cache_path.exists():
            LOGGER.info(f"Loading cached file index from {cache_path}")
            with open(cache_path, 'r') as f:
                return json.load(f)

        LOGGER.info(f"Cache not found. Scanning dataset at {data_path}. This may take a while...")
        all_audio_files = list(data_path.rglob("*.mp3"))
        
        items_full = [{"path": str(p), "id": p.stem} for p in all_audio_files]
        
        random.seed(42)
        random.shuffle(items_full)

        num_files = len(items_full)
        train_end = int(num_files * 0.8)
        dev_end = int(num_files * 0.9)

        data_splits = {
            "train": items_full[:train_end],
            "dev": items_full[train_end:dev_end],
            "test": items_full[dev_end:]
        }
        
        LOGGER.info(f"Found {len(data_splits['train'])} train, {len(data_splits['dev'])} dev, and {len(data_splits['test'])} test samples.")
        LOGGER.info(f"Saving file index to {cache_path}...")
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        with open(cache_path, 'w') as f:
            json.dump(data_splits, f)

        return data_splits

    def __len__(self):
        return len(self.file_index)

    def __getitem__(self, idx):
        item_info = self.file_index[idx]
        file_path = item_info["path"]
        try:
            audio, sr = torchaudio.load(file_path)
            if sr != self.sample_rate:
                resampler = torchaudio.transforms.Resample(sr, self.sample_rate)
                audio = resampler(audio)

            if audio.ndim > 1 and audio.shape[0] > 1:
                audio = torch.mean(audio, dim=0, keepdim=True)

            if audio.shape[-1] < self.segment_length:
                pad_amount = self.segment_length - audio.shape[-1]
                audio = torch.nn.functional.pad(audio, (0, pad_amount))
            elif audio.shape[-1] > self.segment_length:
                start = random.randint(0, audio.shape[-1] - self.segment_length)
                audio = audio[:, start:start + self.segment_length]
            
            return {"audio": audio.squeeze(0), "path": file_path}
        except Exception as e:
            LOGGER.error(f"Error loading {file_path}: {e}")
            return None

def collate_fn_audio_only(batch):
    batch = [item for item in batch if item is not None]
    if not batch:
        return {}
    
    audios = torch.stack([item['audio'] for item in batch])
    paths = [item['path'] for item in batch]
    return {"audio": audios, "paths": paths}

def run_crepe_on_batch(audio_chunk, crepe_model, hop_length):
    """Runs the full CREPE pipeline on a batch of audio."""
    with torch.no_grad():
        # 1. Pad audio
        padded_audio = torch.nn.functional.pad(audio_chunk, (512, 512), 'reflect')

        # 2. Create frames
        frame_size = 1024
        frames = padded_audio.unfold(1, frame_size, hop_length) # (B, N_frames, frame_size)
        
        batch_size, n_frames, _ = frames.shape

        # 3. Reshape and normalize for model
        frames = frames.reshape(-1, frame_size) # (B * N_frames, frame_size)
        frames -= frames.mean(dim=1, keepdim=True)
        frames /= frames.std(dim=1, keepdim=True)
        
        # 4. Infer
        salience = crepe_model(frames) # (B * N_frames, 360)
        
        # 5. Reshape back for Viterbi
        salience = salience.reshape(batch_size, n_frames, 360)
        salience_for_viterbi = salience.permute(0, 2, 1) # (B, 360, N_frames)

        # 6. Decode
        pitch = torchcrepe.decode.viterbi(salience_for_viterbi)[0] # (B, N_frames)
        return pitch

@hydra.main(config_path="../conf", config_name="config", version_base=None)
def main(cfg: DictConfig):
    if not torch.cuda.is_available():
        LOGGER.error("CUDA not available. This script requires a GPU.")
        return

    device = torch.device("cuda")
    
    # --- Model ---
    LOGGER.info(f"Loading CREPE model: {cfg.model.crepe_model_name}...")
    crepe_model = torchcrepe.Crepe(cfg.model.crepe_model_name)
    crepe_model.to(device)
    crepe_model.eval()
    LOGGER.info("CREPE model loaded.")

    # --- Paths ---
    source_data_root = Path(cfg.data.path)
    crepe_output_root = Path(cfg.data.precomputed_crepe_path)
    LOGGER.info(f"Source data path: {source_data_root}")
    LOGGER.info(f"Output path for CREPE .npy files: {crepe_output_root}")

    # Resampler
    resampler = torchaudio.transforms.Resample(
        orig_freq=cfg.data.sampling_rate, 
        new_freq=cfg.data.aux_models.target_sr
    ).to(device)
    
    # --- Setup Dataloaders and Global Progress Bar ---
    splits = ["train", "dev", "test"]
    dataloaders = {}
    total_batches = 0
    LOGGER.info("Calculating total number of batches...")
    for split in splits:
        dataset = AudioOnlyDataset(config=cfg, split=split)
        dataloader = DataLoader(
            dataset,
            batch_size=cfg.train.batch_size,
            shuffle=False,
            collate_fn=collate_fn_audio_only,
            num_workers=cfg.train.num_workers
        )
        dataloaders[split] = dataloader
        total_batches += len(dataloader)
    LOGGER.info(f"Total batches to process across all splits: {total_batches}")

    with tqdm(total=total_batches) as progress_bar:
        for split in splits:
            progress_bar.set_description(f"Precomputing CREPE for {split}")
            dataloader = dataloaders[split]
            
            for batch in dataloader:
                if not batch:
                    progress_bar.update(1) # Still update progress for empty batches
                    continue

                audio_batch = batch['audio'].to(device)
                paths = batch['paths']
                
                resampled_audio_batch = resampler(audio_batch)

                time_splits = 4 # A sensible default based on our benchmark
                hop_length = cfg.data.stft.hop_length
                
                audio_chunks = torch.chunk(resampled_audio_batch, chunks=time_splits, dim=1)
                pitch_chunks = []
                for audio_chunk in audio_chunks:
                    pitch_chunk = run_crepe_on_batch(audio_chunk, crepe_model, hop_length)
                    pitch_chunks.append(pitch_chunk)
                
                pitch_full = torch.cat(pitch_chunks, dim=1).cpu().numpy()

                for i in range(pitch_full.shape[0]):
                    original_path = Path(paths[i])
                    relative_path = original_path.relative_to(source_data_root)
                    output_path = crepe_output_root / relative_path
                    output_path.parent.mkdir(parents=True, exist_ok=True)
                    final_npy_path = output_path.with_suffix('.crepe.npy')
                    np.save(final_npy_path, pitch_full[i])

                progress_bar.update(1)

    LOGGER.info("CREPE pre-computation finished for all splits.")

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
    main() 