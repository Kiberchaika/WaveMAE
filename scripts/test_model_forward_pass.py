from models import WaveMAE
import hydra
from omegaconf import DictConfig, OmegaConf
import torch
import os


# Create dummy yaml files for testing
os.makedirs("../conf/model", exist_ok=True)

with open("../conf/config.yaml", "w") as f:
    f.write("""
defaults:
  - model: wavemae
  - _self_

hydra:
  run:
    dir: .
""")

with open("../conf/model/wavemae.yaml", "w") as f:
    f.write("""
_target_: models.WaveMAE
in_channels: 513
encoder_dims: [96, 192, 384, 768]
encoder_depths: [3, 3, 9, 3]
decoder_depth: 1
mask_ratio: 0.75
""")

@hydra.main(config_path="../conf", config_name="config")
def test_forward_pass(cfg: DictConfig) -> None:
    print("--- Running Forward Pass Test ---")
    print(OmegaConf.to_yaml(cfg))

    model = hydra.utils.instantiate(cfg.model)
    print("\n--- Model Architecture ---")
    print(model)
    print("-" * 26)

    # Create a dummy batch of tensors
    # Shapes from Task 0.1
    batch_size = 4
    num_freq_bins = 513
    num_frames = 1024 # example length
    stft_frames = torch.randn(batch_size, num_freq_bins, num_frames)
    crepe_pitch = torch.randn(batch_size, num_frames)

    print(f"\n--- Input Shapes ---")
    print(f"STFT frames: {stft_frames.shape}")
    print(f"CREPE pitch: {crepe_pitch.shape}")
    print("-" * 20)

    # Run forward pass
    try:
        reconstructed_stft, predicted_pitch, predicted_w2v_bert, mask = model(stft_frames, crepe_pitch)
        print("\n--- Forward Pass SUCCESS ---")
        print("\n--- Output Shapes ---")
        print(f"Reconstructed STFT: {reconstructed_stft.shape}")
        print(f"Predicted Pitch: {predicted_pitch.shape}")
        if predicted_w2v_bert is not None:
            print(f"Predicted W2V-BERT: {predicted_w2v_bert.shape}")
        else:
            print("Predicted W2V-BERT: N/A (disabled)")
        print(f"Mask: {mask.shape}")
        print("-" * 21)

        # Check shapes
        assert stft_frames.shape == reconstructed_stft.shape, "Reconstructed STFT shape mismatch"
        
        num_masked = int(cfg.model.mask_ratio * num_frames)
        assert predicted_pitch.shape == (batch_size, num_masked, 1), f"Predicted pitch shape mismatch. Expected {(batch_size, num_masked, 1)}, got {predicted_pitch.shape}"

        print("\n--- Test 1: Forward Pass Shapes Test: PASS ---")

    except Exception as e:
        print(f"\n--- Forward Pass FAILED ---")
        print(e)
        raise

    print("\n--- Running Hydra Configuration Test ---")
    # Change a model parameter and re-instantiate
    cfg.model.encoder_depths = [2, 2, 6, 2]
    new_model = hydra.utils.instantiate(cfg.model)
    print("\n--- New Model Architecture (encoder_depths=[2, 2, 6, 2]) ---")
    print(new_model)
    print("-" * 60)
    
    original_depths = cfg.model.encoder_depths
    new_depths = [len(s) for s in new_model.encoder.stages]
    
    # This is a bit of a hack to check. We are checking the number of blocks in each stage.
    # The config depths are [2, 2, 6, 2], so the stages should have these many blocks.
    # The check below should pass if the model is instantiated correctly.
    assert new_depths == [2, 2, 6, 2], f"Model depths not updated correctly. Expected [2, 2, 6, 2], got {new_depths}"

    print("\n--- Test 2: Hydra Configuration Test: PASS ---")


if __name__ == "__main__":
    test_forward_pass() 