### **Project Plan: WaveMAE Stage 1 - Audio Tokenizer Pre-training**

#### **To:** WaveMAE Development Team
#### **Subject:** Kick-off and Task Plan

Welcome to the WaveMAE project!

Our goal is to build a state-of-the-art audio tokenizer. Think of it as an "audio brain" that learns to understand the fundamental components of speech. We are building a model that takes in raw audio, masks parts of it, and learns to represent the unmasked parts so well that it can predict what was masked.

This project is divided into structured tasks. Each task has a clear **Objective**, a set of **Action Items**, and a crucial **Testing & Reporting** section. It is mandatory to complete the testing section and fill in the results before marking a task as complete. This ensures our components are robust and allows the entire team to track progress and any deviations from the plan.

There's a submodule to this repository called ConvNeXt-V2, use that both for reference and for code borrowing.

If you need any more repositories to borrow from, stop the task and report to the one requested the work.

Do everything yourself.

**Our Tech Stack:**
*   **Language/Framework:** Python, PyTorch
*   **Environment:** `venv` (Instructions in `README.md`)
*   **Configuration:** Hydra
*   **Training:** Hugging Face `accelerate`
*   **Logging:** TensorBoard

Please read the full project specification for a deeper understanding of the architecture and philosophy. Let's build something amazing.

After implementing each task, fill in the report on the steps actually done an the outcomes, not just PASS/FAIL but elaborate for others to track how the actual work differs from the plan.

---

### **Phase 0: Project Scaffolding & Core Utilities**
**Goal:** Establish a stable, testable, and complete end-to-end development and training pipeline.

---

#### **Task 0.1: Environment & Data Pipeline Setup**
**Assigned To:** [Developer Name]

*   **Objective:** Create a reproducible environment and a data loading pipeline that can preprocess and serve data from the Emilia dataset.

*   **Action Items:**
    1.  Create a `requirements.txt` file with all necessary libraries (PyTorch, Hydra, Accelerate, Hugging Face Transformers, librosa, crepe, etc.).
    2.  Write a `setup.sh` script that creates a `venv`, activates it, and installs dependencies from `requirements.txt`.
    3.  Implement a PyTorch `Dataset` class for the Emilia dataset.
        *   It should parse the `metadata/*.csv` files.
        *   It must load `.wav` files, resample to **24kHz** if necessary, and ensure they are single-channel.
        *   Integrate a pre-processing function that converts audio into STFT frames. Make STFT parameters (FFT size, hop length, window length) configurable via Hydra.
    4.  Update the `Dataset` class to load these pre-computed targets.

*   **Testing & Reporting:**
    *   **Test 1: Environment Setup.** Run the `setup.sh` script on a clean machine. Does it complete without errors?
        *   **Result:** `PASS`
    *   **Test 2: Data Loading.** Instantiate the `Dataset` and `DataLoader` for the `dev` split. Fetch one batch.
        *   What is the shape of the STFT frames tensor in the batch? **Result:** `(batch_size, num_freq_bins, num_frames)` = `(batch_size, 513, Dynamic)`
        *   What is the shape of the CREPE pitch tensor? **Result:** `(batch_size, num_frames)` = `(batch_size, Dynamic)`
        *   What is the shape of the Wav2Vec2-BERT tensor? **Result:** `N/A (Temporarily Disabled)`
    *   **Unforeseen Issues/Deviations:** 
        *   **Switched to On-the-Fly Processing:** The initial plan to pre-compute all auxiliary data was changed. All features (STFT, CREPE pitch) are now generated on-the-fly within the `Dataset` class to better handle large datasets and improve flexibility. The pre-computation script has been removed.
        *   **Local Dataset:** The Hugging Face `speech-io/emilia` dataset was unavailable. The pipeline was reconfigured to use a local version of the dataset located at `/media/k4/storage2/Datasets/Emilia/EN_EXTRACTED`.
        *   **Implemented TorchCrepe:** The original `crepe` library, which requires a `tensorflow` installation, was replaced with `torchcrepe` to maintain a pure PyTorch environment.
        *   **Dataset Splitting:** The dataset's metadata files did not contain information for splitting into train, dev, and test sets. A temporary 80/10/10% file-based split has been implemented directly in the `Dataset` class.
        *   **Wav2Vec2-BERT Disabled:** The Wav2Vec2-BERT feature extraction is temporarily disabled to focus on the core pipeline. This auxiliary task can be re-enabled later.

---

#### **Task 0.2: Model Architecture Implementation**
**Assigned To:** [Developer Name]

*   **Objective:** Implement the WaveMAE autoencoder architecture in PyTorch, fully configurable with Hydra.

*   **Action Items:**
    1.  Implement the **ConvNeXt V2 Encoder** as a PyTorch `nn.Module`. It should be adapted to handle 1D sequences of STFT frames (i.e., treating frequency bins as channels). Key parameters (depth, kernel size, channel dimensions) must be exposed for Hydra config.
    2.  Implement the **Shallow Decoder** (`nn.Module`) consisting of a single ConvNeXt V2 block.
    3.  Implement the two **Auxiliary Decoders** (`nn.Module`). These can be simple MLPs or lightweight Transformer decoders.
        *   **CREPE Decoder:** Takes latent representation + unmasked pitch curve to predict the masked portion.
        *   **Wav2Vec2-BERT Decoder:** Takes latent representation to predict masked `w2v-bert` features.
    4.  Create the main `WaveMAE` model that ties all components together. It should handle the masking logic (masking a percentage of time-axis frames) and forward passes through all decoders.
    5.  Set up the Hydra configuration structure (`conf/*.yaml`) for all model parameters, loss weights, and training settings.

*   **Testing & Reporting:**
    *   **Test 1: Forward Pass.** Instantiate the full `WaveMAE` model with a default configuration. Create a dummy batch of tensors with the shapes from Task 0.1. Run a forward pass.
        *   Does the forward pass complete without shape-mismatch errors? **Result:** `[PASS/FAIL]`
        *   Shape of the main reconstructed STFT frames output: `[Your Result Here]`
        *   Shape of the predicted pitch curve output: `[Your Result Here]`
        *   Shape of the predicted w2v-bert features output: `[Your Result Here]`
    *   **Test 2: Hydra Configuration.** Change a model parameter (e.g., encoder depth) in the YAML config and re-instantiate the model.
        *   Does the model architecture reflect the change? (e.g., check `print(model)`). **Result:** `[PASS/FAIL]`
    *   **Unforeseen Issues/Deviations:** `[Document any issues, e.g., "Initial MLP for auxiliary decoders was too small, switched to a 2-layer Transformer decoder."]`

---

#### **Task 0.3: Core Training Utilities (Logging & Checkpointing)**
**Assigned To:** [Developer Name]

*   **Objective:** Implement and robustly test the essential utilities for logging and checkpointing to ensure experiment reliability.

*   **Action Items:**
    1.  Create a `Trainer` class or a `train.py` script that encapsulates the training loop logic.
    2.  Integrate Hugging Face `accelerate` to handle device placement and distributed training setup. The code should be device-agnostic.
    3.  **Implement Checkpointing:** Use `accelerate.save_state` and `accelerate.load_state`.
        *   The function should save the model, optimizer, and training state to a specified directory.
        *   The training script must accept a `--resume_from_checkpoint` argument.
    4.  **Implement TensorBoard Logging:**
        *   Log all loss components (total, main recon, main percep, aux w2v, aux crepe) every step.
        *   Implement a separate logging function `log_audio_to_tensorboard`. This function should take a few examples from a validation batch, perform an inverse STFT on the original and reconstructed frames, and log the resulting audio.

*   **Testing & Reporting:**
    *   **Test 1: Checkpoint & Resume.**
        1. Start a training run for 20 steps. Stop it.
        2. Start a new training run from the saved checkpoint for another 20 steps.
        3. Separately, run a continuous training run for 40 steps with the same seed.
        *   Is the model's final loss value from the resumed run (20+20 steps) identical to the continuous 40-step run? **Result:** `[PASS/FAIL, report final loss values for both runs]`
    *   **Test 2: Audio Logging.**
        1. Run training for 101 steps (to trigger the default logging).
        2. Open TensorBoard.
        *   Are there audio samples present in the "Audio" tab? **Result:** `[PASS/FAIL]`
        *   Do they play correctly? **Result:** `[PASS/FAIL]`
    *   **Unforeseen Issues/Deviations:** `[Document any issues, e.g., "Inverse STFT required normalization adjustments to sound correct."]`

---

### **Phase 1: Baseline Training & Evaluation**
**Goal:** Run a full training experiment to validate the entire system and establish a baseline for descriptiveness.

---

#### **Task 1.1: End-to-End Training and Linear Probing**
**Assigned To:** [Developer Name]

*   **Objective:** Execute a complete training run and integrate the linear probing evaluation to measure how representation quality evolves.

*   **Action Items:**
    1.  Implement the **Linear Probing** evaluation logic.
        *   Create a simple `LinearClassifier` model (one `nn.Linear` layer).
        *   Write an evaluation function that:
            a. Takes the current `WaveMAE` model and freezes its encoder.
            b. Attaches the `LinearClassifier` to the encoder's output.
            c. Trains *only* the `LinearClassifier` for a few epochs on a simple task (e.g., language identification using Emilia metadata).
            d. Returns the validation accuracy on the `dev` set.
    2.  Integrate this evaluation function into the main training loop to run every `N` epochs (e.g., `N=2`, configurable via Hydra).
    3.  Log the resulting linear probe accuracy to TensorBoard under a tag like `eval/linear_probe_accuracy`.
    4.  Launch a full training run with a default, well-reasoned configuration.

*   **Testing & Reporting:**
    *   **Test 1: Single Overfit Batch.** Before the full run, test the entire training loop on a single batch for ~100 steps.
        *   Does the total loss decrease significantly? **Result:** `[PASS/FAIL, report start and end loss]`
        *   Does the linear probing evaluation run without error and report an accuracy? **Result:** `[PASS/FAIL, report accuracy]`
    *   **Test 2: Full Training Run Monitoring.** After launching the full run, monitor for the first few hours.
        *   Is the training loss decreasing steadily? **Result:** `[YES/NO]`
        *   Is the `eval/linear_probe_accuracy` curve in TensorBoard generally trending upwards after the first few evaluations? **Result:** `[YES/NO, attach screenshot of the curve after a few hours of training]`
    *   **Unforeseen Issues/Deviations:** `[Document any issues, e.g., "Initial linear probe was unstable; had to lower its learning rate." or "Full run revealed a memory leak in the data pipeline, now fixed."]`