import sys
import torch
import os
import json
from safetensors.torch import load_file

# Add the submodule to the python path
script_dir = os.path.dirname(os.path.abspath(__file__))
rmvpe_dir = os.path.join(script_dir, 'RMVPE')
sys.path.append(rmvpe_dir)

from src.model import E2E0
from src import constants

def load_model(model_path, config_path):
    """Loads a safetensors model and inspects it."""
    try:
        with open(config_path, 'r') as f:
            config = json.load(f)

        # Hyperparameters from config and constants
        hop_length = 20
        hop_samples = int(hop_length / 1000 * constants.SAMPLE_RATE)
        
        # Instantiate the model with config params
        model = E2E0(
            hop_length=hop_samples,
            n_blocks=config['n_blocks'],
            n_gru=config['n_gru'],
            kernel_size=tuple(config['kernel_size']),
            en_de_layers=config['en_de_layers'],
            inter_layers=config['inter_layers'],
            in_channels=config['in_channels'],
            en_out_channels=config['en_out_channels']
        )
        
        # Load the state dict from safetensors
        state_dict = load_file(model_path, device='cpu')
        model.load_state_dict(state_dict)
        
        # Get the dtype of the first parameter
        first_param_dtype = next(model.parameters()).dtype
        print(f"Successfully loaded model from: {os.path.basename(model_path)}")
        print(f"  - Parameter dtype: {first_param_dtype}")
        print(f"  - Model Config: {config}")

    except Exception as e:
        print(f"Could not load model {model_path}: {e}")

if __name__ == '__main__':
    model_path = 'pretrained_models/rmvpe_safe-models/model.safetensors'
    config_path = 'pretrained_models/rmvpe_safe-models/config.json'

    print("--- Loading SafeTensor Model ---")
    load_model(model_path, config_path)
    print("-" * 20) 