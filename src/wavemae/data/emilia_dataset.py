import json
from pathlib import Path

import torch
import torchcrepe
import torchaudio
import torchaudio.transforms as T
from torch.nn import functional as F
from torch.utils.data import Dataset


class EmiliaDataset(Dataset):
    def __init__(self, cfg, split, device='cpu'):
        self.cfg_data = cfg.data
        self.split = split
        self.device = device
        self.data_path = Path(self.cfg_data.path)
        
        self.items = self._scan_dataset()
        
        self.resampler_stft = None
        self.resampler_16k = None
        self.target_sr_stft = self.cfg_data.sampling_rate
        self.target_sr_16k = self.cfg_data.aux_models.target_sr

    def _scan_dataset(self):
        items = []
        # In the absence of a split key, we will use a simple heuristic for train/dev/test.
        # This is a placeholder and should be replaced with a more robust method.
        all_json_files = sorted(list(self.data_path.rglob("*.json")))
        
        # Simple split: 80% train, 10% dev, 10% test
        num_files = len(all_json_files)
        train_end = int(num_files * 0.8)
        dev_end = int(num_files * 0.9)

        if self.split == 'train':
            split_files = all_json_files[:train_end]
        elif self.split == 'dev':
            split_files = all_json_files[train_end:dev_end]
        else: # test
            split_files = all_json_files[dev_end:]

        for json_path in split_files:
            mp3_path = json_path.with_suffix(".mp3")
            if mp3_path.exists():
                items.append({
                    "path": str(mp3_path),
                    "id": mp3_path.stem
                })
        return items

    def __len__(self):
        return len(self.items)

    def _get_resampler(self, resampler, orig_freq, target_freq):
        if resampler is None or resampler.orig_freq != orig_freq:
            return T.Resample(orig_freq=orig_freq, new_freq=target_freq).to(self.device)
        return resampler

    def _compute_stft(self, waveform, sr):
        resampler = self._get_resampler(self.resampler_stft, sr, self.target_sr_stft)
        waveform_resampled = resampler(waveform)

        stft_params = self.cfg_data.stft
        stft = torch.stft(
            waveform_resampled.squeeze(0),
            n_fft=stft_params.n_fft,
            hop_length=stft_params.hop_length,
            win_length=stft_params.win_length,
            window=torch.hann_window(stft_params.win_length).to(self.device),
            return_complex=True,
        )
        return torch.abs(stft)

    def _compute_pitch(self, waveform, sr):
        resampler = self._get_resampler(self.resampler_16k, sr, self.target_sr_16k)
        waveform_16k = resampler(waveform)
        
        pitch, periodicity = torchcrepe.predict(
            waveform_16k,
            self.target_sr_16k,
            hop_length=160, # Corresponds to 10ms hop at 16kHz
            fmin=50.0,
            fmax=1500.0,
            model='full',
            batch_size=512,
            device=self.device,
            return_periodicity=True,
        )
        return pitch.squeeze(0)

    def _align_features(self, target_len, pitch):
        # Align CREPE pitch
        pitch = pitch.unsqueeze(0).unsqueeze(0)
        aligned_pitch = F.interpolate(pitch, size=target_len, mode='linear', align_corners=False)
        return aligned_pitch.squeeze()

    def __getitem__(self, idx):
        item_info = self.items[idx]
        
        waveform, sr = torchaudio.load(item_info["path"])
        waveform = waveform.to(self.device)
        
        if waveform.ndim > 1:
            waveform = torch.mean(waveform, dim=0, keepdim=True)

        stft_mag = self._compute_stft(waveform, sr)
        pitch = self._compute_pitch(waveform, sr)
        
        aligned_pitch = self._align_features(stft_mag.shape[1], pitch)

        return {
            "id": item_info["id"],
            "stft": stft_mag.cpu(),
            "pitch": aligned_pitch.cpu(),
            "w2v_bert": None, # Placeholder for now
        } 