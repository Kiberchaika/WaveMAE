import sys
import torch
import os
import json
from safetensors.torch import load_file

# Add the JingleGenerator RVC project to the python path
script_dir = os.path.dirname(os.path.abspath(__file__))
rvc_project_dir = os.path.join(script_dir, '../JingleGenerator/Retrieval-based-Voice-Conversion-WebUI')
sys.path.insert(0, rvc_project_dir)

# Now we can import the correct E2E model from the JingleGenerator project
from infer.lib.rmvpe import E2E

def load_model(model_path, config_path):
    """Loads a safetensors model and inspects it using the JingleGenerator E2E model."""
    try:
        with open(config_path, 'r') as f:
            config = json.load(f)

        # Instantiate the model with config params from JingleGenerator's E2E
        model = E2E(
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
    # We are in the WaveMAE project root, so we need to go up one level
    # to get to the JingleGenerator project.
    model_path = 'pretrained_models/rmvpe_safe-models/model.safetensors'
    config_path = 'pretrained_models/rmvpe_safe-models/config.json'

    print("--- Loading SafeTensor Model with JingleGenerator code ---")
    load_model(model_path, config_path)
    print("-" * 20) 