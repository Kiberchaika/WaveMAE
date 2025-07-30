# WaveMAE: A Masked Autoencoder for Audio Representation Learning

## Overview

WaveMAE is a research project focused on developing a state-of-the-art, general-purpose audio representation model. It uses a masked autoencoder (MAE) architecture to learn a semantically rich latent space from raw audio waveforms. This latent space is designed to be highly descriptive for downstream tasks, particularly generative modeling.

The core of the project involves training an autoencoder to reconstruct masked portions of an audio signal's STFT representation. To enrich the learned latent space, the model uses auxiliary decoders aligned with pre-trained models, including:

*   **Wav2Vec2-BERT:** For general audio features.
*   **RMVPE:** For pitch information.

The project is structured into several phases, from initial scaffolding and data pipeline implementation to systematic experiments and model analysis.

## Project Structure

*   `AIDocs/`: Contains the project plan and technical specifications.
*   `conf/`: Hydra configuration files for the model, training, and experiments.
*   `data/`: Data loading and preprocessing scripts.
*   `models/`: PyTorch model definitions for the WaveMAE architecture.
*   `scripts/`: Training, evaluation, and utility scripts.
*   `pretrained_models/`: Contains the pre-trained RMVPE model and its source code.

## Setup

1.  **Create and activate the virtual environment:**
    ```bash
    bash setup.sh
    source .venv/bin/activate
    ```
2.  **Run the training script:**
    ```bash
    python scripts/train.py
    ```
