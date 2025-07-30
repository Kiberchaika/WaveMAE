import sys
import torch
import os

# Add the submodule to the python path
# This is a bit of a hack, but it's the easiest way to get the imports to work
# without installing the submodule as a package.
script_dir = os.path.dirname(os.path.abspath(__file__))
rmvpe_dir = os.path.join(script_dir, 'RMVPE')
sys.path.append(rmvpe_dir)

from src.model import E2E0
from src import constants

def inspect_model(model_path, model):
    """Loads a checkpoint and inspects the dtype of its parameters."""
    try:
        checkpoint = torch.load(model_path, map_location='cpu')
        
        # The checkpoint might be the state_dict itself or a model
        if isinstance(checkpoint, dict) and 'model' in checkpoint:
            # Handle the case where the state_dict is nested under a 'model' key
            state_dict = checkpoint['model']
        elif isinstance(checkpoint, dict):
            # This handles checkpoints that are just the state dict
            state_dict = checkpoint
        else:
            # This handles checkpoints that are the entire model
            model = checkpoint
            state_dict = model.state_dict()

        # Try to load the state_dict with strict=False to see what matches
        model.load_state_dict(state_dict, strict=False)
        
        # Get the dtype of the first parameter
        first_param_dtype = next(model.parameters()).dtype
        print(f"Model: {os.path.basename(model_path)}")
        print(f"  - Parameter dtype: {first_param_dtype}")

    except Exception as e:
        print(f"Could not inspect model {model_path}: {e}")


if __name__ == '__main__':
    # Hyperparameters from train.py
    hop_length = 20
    hop_samples = int(hop_length / 1000 * constants.SAMPLE_RATE)
    n_blocks = 4
    n_gru = 1
    kernel_size = (2, 2)

    # Hypothesis: The pretrained models were trained with N_MELS = 128
    # Temporarily override the constant to test this.
    original_n_mels = constants.N_MELS
    constants.N_MELS = 128

    # Instantiate the model
    model = E2E0(hop_samples, n_blocks, n_gru, kernel_size, en_de_layers=5, inter_layers=4, in_channels=1, en_out_channels=16)

    # Restore the original value
    constants.N_MELS = original_n_mels

    # Paths to the downloaded models
    model1_path = 'pretrained_models/rmvpe/rmvpe_richardgr.pt'
    model2_path = 'pretrained_models/rmvpe/rmvpe_lj1995.pt'

    print("--- Inspecting Pretrained Models ---")
    inspect_model(model1_path, model)
    print("-" * 20)
    inspect_model(model2_path, model)
    print("-" * 20) 