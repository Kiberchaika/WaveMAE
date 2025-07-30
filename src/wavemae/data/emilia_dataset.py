import logging
import os
import json
from pathlib import Path
import torch
from torch.utils.data import Dataset
import torchaudio
import torchaudio.transforms as T
import random
import numpy as np

LOGGER = logging.getLogger(__name__)

class EmiliaDataset(Dataset):
    def __init__(self, config, split='train'):
        self.config = config
        self.split = split
        self.source_data_root = Path(config.data.path)
        self.precomputed_crepe_root = Path(config.data.precomputed_crepe_path)
        self.file_index = self._build_index(config.data.cache_path, self.source_data_root)
        
        self.sample_rate = config.data.sampling_rate
        self.segment_length = int(config.data.segment_duration_secs * self.sample_rate)
        
        self.n_fft = config.data.stft.n_fft
        self.hop_length = config.data.stft.hop_length
        self.win_length = config.data.stft.win_length

    def _build_index(self, index_path, data_path):
        cache_path = Path(index_path)
        if cache_path.exists():
            LOGGER.info(f"Loading cached file index from {cache_path}")
            with open(cache_path, 'r') as f:
                return json.load(f)[self.split]

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

        return data_splits[self.split]

    def __len__(self):
        return len(self.file_index)

    def __getitem__(self, idx):
        item_info = self.file_index[idx]
        file_path = Path(item_info["path"])
        
        try:
            # 1. Load Audio
            audio, sr = torchaudio.load(file_path)
            if sr != self.sample_rate:
                resampler = torchaudio.transforms.Resample(sr, self.sample_rate)
                audio = resampler(audio)
            
            if audio.ndim > 1 and audio.shape[0] > 1:
                audio = torch.mean(audio, dim=0, keepdim=True)

            # --- Pad or crop to segment length ---
            if audio.shape[-1] < self.segment_length:
                pad_amount = self.segment_length - audio.shape[-1]
                audio = torch.nn.functional.pad(audio, (0, pad_amount))
            elif audio.shape[-1] > self.segment_length:
                start = random.randint(0, audio.shape[-1] - self.segment_length)
                audio = audio[:, start:start + self.segment_length]

            # 2. Load pre-computed CREPE pitch
            relative_path = file_path.relative_to(self.source_data_root)
            crepe_path = (self.precomputed_crepe_root / relative_path).with_suffix('.crepe.npy')

            if not crepe_path.exists():
                LOGGER.warning(f"Precomputed CREPE file not found for {file_path}, skipping. Expected at: {crepe_path}")
                return None
            
            pitch = torch.from_numpy(np.load(crepe_path))

            # 3. Compute STFT
            stft_transform = T.Spectrogram(
                n_fft=self.n_fft,
                hop_length=self.hop_length,
                win_length=self.win_length,
                power=None,
            )
            stft = stft_transform(audio).squeeze(0)

            # 4. Align pitch and STFT tensors
            stft_len = stft.shape[-1]
            # Unsqueeze to (N, C, L) for interpolate
            pitch = pitch.unsqueeze(0).unsqueeze(0).float()
            aligned_pitch = torch.nn.functional.interpolate(pitch, size=stft_len, mode='nearest')
            aligned_pitch = aligned_pitch.squeeze(0).squeeze(0).long()

            return {"pitch": aligned_pitch, "stft": stft, "id": item_info["id"]}

        except Exception as e:
            LOGGER.error(f"Error processing {file_path}: {e}", exc_info=True)
            return None 