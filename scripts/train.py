import argparse
import logging
import os
import sys
from pathlib import Path

import hydra
import torch
import torchaudio
from accelerate import Accelerator
from omegaconf import DictConfig, OmegaConf
from torch.optim import AdamW
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

import auraloss

# Add project root to path
sys.path.append(str(Path(__file__).resolve().parents[1]))

from models import WaveMAE
from src.wavemae.data.emilia_dataset import EmiliaDataset

# Suppress torchaudio git version warnings
torchaudio.utils.sox_utils.set_verbosity(0)

LOGGER = logging.getLogger(__name__)


def collate_fn(batch):
    """
    Pads STFT and pitch tensors to the maximum length in a batch.
    """
    max_len = max(item['stft'].shape[1] for item in batch)
    
    stfts, pitches, ids = [], [], []

    for item in batch:
        stft = item['stft']
        pitch = item['pitch']
        
        len_diff = max_len - stft.shape[1]
        
        stfts.append(torch.nn.functional.pad(stft, (0, len_diff)))
        pitches.append(torch.nn.functional.pad(pitch, (0, len_diff)))
        ids.append(item['id'])

    return {
        'stft': torch.stack(stfts),
        'pitch': torch.stack(pitches),
        'id': ids
    }


def log_audio_to_tensorboard(writer, model, val_batch, global_step, cfg):
    LOGGER.info(f"Step {global_step}: Logging audio samples to TensorBoard...")
    model.eval()

    stfts = val_batch['stft']
    
    with torch.no_grad():
        reconstructed_stft, _, _, _ = model(stfts, None)
    
    # Use only a few examples to avoid clutter
    num_to_log = min(stfts.size(0), 4)

    # Inverse STFT
    istft_transform = torchaudio.transforms.InverseSpectrogram(
        n_fft=cfg.data.stft.n_fft,
        hop_length=cfg.data.stft.hop_length,
        win_length=cfg.data.stft.win_length,
        window_fn=torch.hann_window
    ).to(stfts.device)

    for i in range(num_to_log):
        original_audio = istft_transform(stfts[i].unsqueeze(0).cpu()).squeeze()
        reconstructed_audio = istft_transform(reconstructed_stft[i].unsqueeze(0).cpu()).squeeze()
        
        # Normalize for visualization
        original_audio /= original_audio.abs().max()
        reconstructed_audio /= reconstructed_audio.abs().max()

        writer.add_audio(f'audio/original_{i}', original_audio.unsqueeze(0), global_step, sample_rate=cfg.data.sampling_rate)
        writer.add_audio(f'audio/reconstructed_{i}', reconstructed_audio.unsqueeze(0), global_step, sample_rate=cfg.data.sampling_rate)

    model.train()
    LOGGER.info("Finished logging audio samples.")


def set_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    # torch.backends.cudnn.deterministic = True
    # torch.backends.cudnn.benchmark = False
    

@hydra.main(config_path="../conf", config_name="config", version_base=None)
def main(cfg: DictConfig):
    # --- Setup ---
    set_seed(cfg.train.seed)
    run_name = os.path.basename(os.getcwd())
    LOGGER.info(f"Starting run: {run_name}")
    LOGGER.info(f"Config:\\n{OmegaConf.to_yaml(cfg)}")
    
    # Check for resume argument
    parser = argparse.ArgumentParser()
    parser.add_argument('--resume_from_checkpoint', type=str, default=None)
    args, _ = parser.parse_known_args()

    accelerator = Accelerator()
    device = accelerator.device
    
    output_dir = Path(os.getcwd())
    writer = SummaryWriter(log_dir=str(output_dir / "tensorboard"))

    # --- Data ---
    train_dataset = EmiliaDataset(cfg=cfg, split="train", device=device)
    val_dataset = EmiliaDataset(cfg=cfg, split="dev", device=device)
    
    train_loader = DataLoader(train_dataset, batch_size=cfg.train.batch_size, shuffle=True, collate_fn=collate_fn, num_workers=cfg.train.num_workers)
    val_loader = DataLoader(val_dataset, batch_size=cfg.train.batch_size, shuffle=False, collate_fn=collate_fn, num_workers=cfg.train.num_workers)

    # --- Model & Optimizer ---
    model = WaveMAE(
        in_channels=cfg.model.in_channels,
        encoder_dims=cfg.model.encoder.dims,
        encoder_depths=cfg.model.encoder.depths,
        decoder_depth=cfg.model.decoder.depth,
        mask_ratio=cfg.model.mask_ratio
    )
    optimizer = AdamW(model.parameters(), lr=cfg.train.learning_rate)
    
    # --- Loss Functions ---
    l1_loss = torch.nn.L1Loss()
    perceptual_loss = auraloss.freq.MultiResolutionSTFTLoss(
        fft_sizes=[1024, 2048, 512],
        hop_sizes=[256, 512, 128],
        win_lengths=[1024, 2048, 512],
        device=device
    )

    # --- Prepare with Accelerate ---
    model, optimizer, train_loader, val_loader = accelerator.prepare(
        model, optimizer, train_loader, val_loader
    )

    # --- Resuming from checkpoint ---
    if args.resume_from_checkpoint:
        LOGGER.info(f"Resuming from checkpoint: {args.resume_from_checkpoint}")
        accelerator.load_state(args.resume_from_checkpoint)
        # The step needs to be recovered from the checkpoint path
        try:
            global_step = int(args.resume_from_checkpoint.split('_')[-1])
            start_epoch = global_step // len(train_loader)
        except (ValueError, IndexError):
            LOGGER.warning("Could not parse step from checkpoint path. Starting from step 0.")
            global_step = 0
            start_epoch = 0
    else:
        global_step = 0
        start_epoch = 0

    # --- Training Loop ---
    LOGGER.info("Starting training loop...")
    val_batch_for_logging = next(iter(val_loader))

    for epoch in range(start_epoch, cfg.train.epochs):
        model.train()
        progress_bar = tqdm(train_loader, desc=f"Epoch {epoch}", disable=not accelerator.is_main_process)
        
        for batch in progress_bar:
            if cfg.train.max_steps > 0 and global_step >= cfg.train.max_steps:
                LOGGER.info(f"Reached max_steps ({cfg.train.max_steps}). Stopping training.")
                break

            optimizer.zero_grad()

            stfts = batch['stft']
            pitches = batch['pitch']
            
            reconstructed_stft, predicted_pitch, _, inverse_mask = model(stfts, None)
            
            # --- Loss Calculation ---
            loss_recon = l1_loss(reconstructed_stft, stfts)
            
            # Perceptual loss needs audio, so we need to do iSTFT
            istft_transform = torchaudio.transforms.InverseSpectrogram(
                n_fft=cfg.data.stft.n_fft, hop_length=cfg.data.stft.hop_length, win_length=cfg.data.stft.win_length, window_fn=torch.hann_window
            ).to(device)
            original_audio = istft_transform(stfts.cpu()).to(device)
            reconstructed_audio = istft_transform(reconstructed_stft.cpu()).to(device)
            loss_percep = perceptual_loss(reconstructed_audio, original_audio)
            
            # Gather ground truth pitch at masked locations
            masked_pitch = torch.stack([
                pitches[i][inverse_mask[i]] for i in range(pitches.size(0))
            ])
            loss_crepe = l1_loss(predicted_pitch.squeeze(-1), masked_pitch)
            
            total_loss = (cfg.loss.weights.recon * loss_recon +
                          cfg.loss.weights.percep * loss_percep +
                          cfg.loss.weights.crepe * loss_crepe)

            accelerator.backward(total_loss)
            optimizer.step()

            # --- Logging ---
            if accelerator.is_main_process:
                writer.add_scalar("loss/total", total_loss.item(), global_step)
                writer.add_scalar("loss/reconstruction", loss_recon.item(), global_step)
                writer.add_scalar("loss/perceptual", loss_percep.item(), global_step)
                writer.add_scalar("loss/crepe_pitch", loss_crepe.item(), global_step)
                progress_bar.set_postfix({"loss": total_loss.item()})

                # Log audio and save checkpoint
                if global_step > 0 and global_step % cfg.train.log_audio_steps == 0:
                    log_audio_to_tensorboard(writer, accelerator.unwrap_model(), val_batch_for_logging, global_step, cfg)

                if global_step > 0 and global_step % cfg.train.save_checkpoint_steps == 0:
                    checkpoint_path = output_dir / f"checkpoint_{global_step}"
                    LOGGER.info(f"Saving checkpoint to {checkpoint_path}")
                    accelerator.save_state(str(checkpoint_path))

            global_step += 1
        
        if cfg.train.max_steps > 0 and global_step >= cfg.train.max_steps:
            break

    LOGGER.info("Training finished.")
    # Final checkpoint
    if accelerator.is_main_process:
        final_checkpoint_path = output_dir / f"checkpoint_{global_step}"
        LOGGER.info(f"Saving final checkpoint to {final_checkpoint_path}")
        accelerator.save_state(str(final_checkpoint_path))
        writer.close()


if __name__ == "__main__":
    main() 