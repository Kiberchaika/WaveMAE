### **Project Specification: WaveMAE**

**To:** Lead ML Architect
**From:** Project Lead
**Date:** October 7, 2024
**Subject:** Technical Specification and Plan for WaveMAE Stage 1 - Audio Tokenizer Pre-training

#### **1. Project Philosophy & Concept**

**WaveMAE** is a research project to develop a state-of-the-art, general-purpose audio representation model. The core hypothesis is that by decoupling the tasks of **representation learning** and **reconstruction**, we can create a latent space that is both highly **descriptive** of semantic content and highly **modelable** for downstream generative tasks.

This project is the first of a two-stage process. The objective of this stage is to train a powerful **tokenizer** (an autoencoder) that converts raw audio into a compact, structured latent representation. A subsequent project will use this frozen representation to train a generative model (e.g., RFWave), but that is **out of scope for this plan.**

Our approach is guided by three key sources:

1.  **[MAETok (Chen et al., 2024)](https://arxiv.org/abs/2402.03444):** We adopt the principle of using an asymmetric Masked Autoencoder (MAE) with multiple auxiliary decoders to force the encoder to learn a semantically rich latent space.
2.  **[FCMAE (Woo et al., 2023)](https://arxiv.org/abs/2301.00808):** We will use a fully convolutional architecture for its efficiency and scalability, specifically adapting the ConvNeXt V2 block for both the encoder and the shallow decoder.
3.  **[Generative modelling in latent space (Dieleman, 2025)](https://sander.ai/2025/04/15/latents.html):** We embrace the philosophy that representation and reconstruction are separate tasks. Our shallow decoder and auxiliary losses are designed to "curate" a modelable latent space, abstracting away high-entropy noise while preserving semantic structure.

The final output of this project will be a series of pre-trained, frozen audio encoders, validated for their descriptive power.

#### **2. Core Architectural Components**

The model is an autoencoder operating on STFT frames.

*   **Input Processing:**
    *   Single-channel audio from the Emilia dataset.
    *   Audio is converted to a sequence of STFT frames.
    *   A configurable percentage of time-axis frames are masked before being passed to the encoder.

*   **Encoder:**
    *   A **ConvNeXt V2** backbone, adapted for 1D sequences of STFT frames.
    *   The architecture must be configurable via Hydra to control depth, kernel sizes, and channel dimensions, allowing for explicit control over the **receptive field**.
    *   Operates only on the unmasked STFT frames for computational efficiency.

*   **Latent Space:**
    *   The encoder outputs a sequence of latent vectors, which form a compressed representation of the input.
    *   The **Tensor Size Reduction (TSR)** ratio is a key experimental variable. It is defined as the ratio of total elements in the input STFT frames to the total elements in the latent representation. This will be controlled by varying the encoder's downsampling factor and the latent channel dimension.

*   **Main Decoder (Shallow):**
    *   A single **ConvNeXt V2 block**, as per the FCMAE architecture.
    *   **Task:** Reconstruct the *full* sequence of STFT frames (both masked and unmasked) from the latent representation.

*   **Auxiliary Decoders:**
    *   These are shallow MLP or Transformer-based decoders that operate on the same latent representation. Their gradients flow back to the encoder.
    *   **Wav2Vec2-BERT Decoder:** Predicts the hidden-state features from a frozen [`wav2vec2-bert-base`](https://huggingface.co/docs/transformers/en/model_doc/wav2vec2-bert) model for the *masked* input positions.
    *   **RMVPE Pitch Decoder:** Predicts the full F0 pitch curve (as derived by RMVPE) for the *masked* input positions. This decoder will take the latent representation and the *unmasked* portion of the pitch curve as input to perform in-painting.

#### **3. Data & Loss Formulation**

*   **Dataset:** [Emilia Dataset](https://huggingface.co/datasets/speech-io/emilia) (Speech & Text pairs). We will use the `train`, `dev`, and `test` splits accordingly.
*   **Loss Functions:** The total loss is a weighted sum of the following components, with weights controlled via Hydra.
    *   `L_main_recon`: Reconstruction loss (L1/MSE) on the output of the main shallow decoder against the original STFT frames.
    *   `L_main_percep`: Perceptual loss (e.g., Multi-Resolution STFT Loss) on the output of the main shallow decoder.
    *   `L_aux_w2v`: L2 loss between the Wav2Vec2-BERT decoder output and the ground-truth features.
    *   `L_aux_rmvpe`: L1 loss between the RMVPE decoder output and the ground-truth pitch curve.

#### **4. Key Research Questions & Experiments**

This project will systematically answer the following:

1.  **TSR Optimization:** What is the optimal trade-off between temporal downsampling and latent channel dimension for maximizing descriptiveness? We will train several models with a fixed TSR but different shapes.
2.  **Receptive Field Analysis:** How does the encoder's receptive field size affect the quality of the learned representation? This will be controlled by varying the ConvNeXt V2 configuration.
3.  **Auxiliary Loss Impact:** Quantify the contribution of each auxiliary loss to the final descriptiveness of the latent space via ablation studies.

#### **5. Technical & Evaluation Specification**

*   **Frameworks:**
    *   **Core:** PyTorch
    *   **Training:** Hugging Face `accelerate` for seamless multi-GPU/distributed training.
    *   **Configuration:** Hydra for modular and reproducible experiment configuration.
    *   **Logging:** TensorBoard.

*   **Evaluation Protocol:**
    *   **Primary Metric:** Descriptiveness, measured by **linear probing**. A linear classifier will be trained on top of the frozen encoder's output latents every 2-3 epochs on a downstream task (e.g., language identification using Emilia's metadata). The validation accuracy will be logged to track how descriptiveness evolves.
    *   **Secondary Metrics:** Reconstruction and perceptual loss values on the dev set.

*   **Logging & Checkpointing:**
    *   **Audio Logging:** A utility function must be implemented to log a few validation samples (original, reconstructed) to TensorBoard as audio every `n=100` steps.
    *   **Checkpointing:** The training loop must support saving a full training state (model, optimizer, `accelerate` state) every `k=1000` steps and must be able to resume seamlessly.

*   **Pre-trained Models:**
    *   The `safetensors` model weights and `config.json` for the RMVPE model are located in `pretrained_models/rmvpe_safe-models/`.
    *   The compatible Python source code for this model, sourced from the `Retrieval-based-Voice-Conversion-WebUI` project, is located in `pretrained_models/rmvpe_model_code/`.

---

### **Actionable Plan Outline**

This plan is structured in phases to de-risk the project and ensure foundational components are robust before committing to large-scale training.

**Phase 0: Scaffolding & Sanity Checks (Sprint 1-2)**

*   **Goal:** Build a complete, testable pipeline. No long training runs.
*   **Tasks:**
    1.  **Data Pipeline:** Implement the Emilia `Dataset` and `DataLoader`. The pipeline must handle loading audio, resampling, STFT, and pre-computing targets for auxiliary losses (pitch via RMVPE, features via Wav2Vec2-BERT).
    2.  **Model Scaffolding:** Implement the full WaveMAE architecture in PyTorch, with all components (Encoder, Decoders) wired together. Make all key parameters (TSR, receptive field, loss weights) configurable via Hydra.
    3.  **Core Utilities:**
        *   **Implement and test checkpoint saving/loading** with `accelerate`. Verify that a resumed run produces bit-for-bit identical results to a continuous run.
        *   **Implement and test TensorBoard audio logging.** Run for 200 steps and confirm audio samples appear correctly in the UI.
    4.  **Overfit on a Single Batch:** Run the training loop on a single, fixed batch of data. The goal is to see the loss drop to near-zero, confirming that the architecture can learn and gradients are flowing correctly.

**Phase 1: Baseline Model Training (Sprint 3)**

*   **Goal:** Train a single, reasonably configured model to establish a performance baseline and validate the full training loop.
*   **Tasks:**
    1.  Define a "default" configuration in Hydra (e.g., TSR=16x, medium receptive field).
    2.  Launch a full training run on the Emilia dataset.
    3.  Monitor training, paying close attention to the linear probing accuracy curve in TensorBoard. Ensure it is trending upwards.

**Phase 2: Systematic Experiments (Sprints 4-5)**

*   **Goal:** Execute the core research experiments.
*   **Tasks:**
    1.  **TSR Sweep:** Using Hydra's multi-run capabilities, launch a sweep of experiments varying the latent space shape while keeping TSR constant.
    2.  **Receptive Field Sweep:** Launch a sweep varying the encoder's kernel sizes and depth.
    3.  **Ablation Study:** Launch runs with one or more auxiliary losses disabled to measure their impact.

**Phase 3: Analysis & Final Model Selection (Sprint 6)**

*   **Goal:** Analyze results and deliver the final, pre-trained tokenizer models.
*   **Tasks:**
    1.  Aggregate all linear probing results and loss curves from the experimental runs.
    2.  Create visualizations comparing the descriptiveness of different model configurations.
    3.  Select the top 1-3 best-performing encoder models based on the evaluation protocol.
    4.  Package the frozen encoder weights with clear documentation for use in the next stage of the project.