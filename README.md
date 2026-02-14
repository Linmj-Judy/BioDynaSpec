# BioDynaSpec: Harmonic-Guided Spatio-Spectral Autoregressive Diffusion for Protein Dynamics Generation

<p align="center">
  <strong>ICML 2026 Submission</strong>
</p>

<p align="center">
  <a href="#abstract">Abstract</a> •
  <a href="#motivation">Motivation</a> •
  <a href="#method">Method</a> •
  <a href="#installation">Installation</a> •
  <a href="#data-preparation">Data Preparation</a> •
  <a href="#training">Training</a> •
  <a href="#inference">Inference</a> •
  <a href="#results">Results</a> •
  <a href="#project-structure">Project Structure</a> •
  <a href="#citation">Citation</a>
</p>

---

## Abstract

Generating long-horizon, all-atom molecular dynamics (MD) is difficult due to error accumulation in time-domain autoregressive models (causing drift) and fixed step-size constraints on temporal resolution. We propose **BioDynaSpec**, which reformulates protein dynamics as spatio-spectral generation: **Independent Windowed Fourier Decomposition (IWFD)** decomposes trajectories into independent windowed frequency representations, and a generator combines low-to-high frequency autoregression with diffusion denoising to reconstruct continuous motion. We introduce **Inter-Residue Frequency Coupling (IRFC)** bias, a learnable Gaussian distance bias in attention that injects a resonance-inspired structural prior to stabilize training and improve cross-residue, cross-frequency consistency. On ATLAS, BioDynaSpec improves 250-frame trajectory generation with \(R_{250}=1.509\) Å (where \(R_s\) denotes the mean per-frame Cα-RMSE over the first \(s\) frames after alignment), reducing error by 60.4% vs. MDGEN and 57.2% vs. ProAR, and achieves the best PCA-2D displacement-profile correlation and stepwise distribution matching. For equilibrium conformational sampling, it achieves Root Mean \(W_2=1.31\), MD PCA \(W_2=0.90\), and Joint PCA \(W_2=1.19\) (50.03%, 35.25%, and 47.58% lower than the next best), while ablations show removing IRFC severely degrades RMSE/MAE and correlation.

---

## Motivation

Time-domain autoregressive models for protein dynamics suffer from cumulative error drift and are locked to a fixed temporal step size. BioDynaSpec addresses these fundamental limitations by operating in the **frequency domain**: trajectories are decomposed via windowed Fourier transforms, and generation proceeds from low to high frequencies—naturally capturing the hierarchy of protein motions from slow collective modes to fast local fluctuations.

<p align="center">
  <img src="figures/mj_motivation.pdf" width="90%" alt="Motivation: Why frequency-domain generation?" />
</p>

**Figure 1.** Motivation for spectral-domain protein dynamics generation. Time-domain autoregressive models accumulate errors over long horizons, while frequency-domain generation via IWFD decouples temporal resolution from step count and naturally respects the multi-scale nature of protein motions.

---

## Method

BioDynaSpec consists of three core components:

1. **Independent Windowed Fourier Decomposition (IWFD):** Decomposes all-atom MD trajectories into independent windowed frequency representations, enabling parallel generation across windows and decoupling temporal resolution from the number of generation steps.

2. **Spatio-Spectral Autoregressive Diffusion:** A hybrid generator that autoregressively predicts frequency components from low to high, using diffusion-based denoising at each step to produce high-fidelity spectral coefficients.

3. **Inter-Residue Frequency Coupling (IRFC):** A learnable Gaussian distance bias injected into the attention mechanism, providing a resonance-inspired structural prior that enforces cross-residue, cross-frequency consistency and stabilizes training.

<p align="center">
  <img src="figures/BDSmethond_v2.pdf" width="95%" alt="BioDynaSpec Method Overview" />
</p>

**Figure 2.** Overview of the BioDynaSpec architecture. (Left) IWFD decomposes trajectories into windowed frequency representations. (Center) The autoregressive diffusion generator predicts spectral coefficients from low to high frequency. (Right) IRFC injects structural priors via learnable Gaussian distance biases in attention.

---

## Installation

### Prerequisites

- **OS:** Linux (tested on Ubuntu 20.04/22.04)
- **GPU:** NVIDIA GPU with CUDA 12.4+ support (tested on A800, H200)
- **Python:** 3.11+
- **Conda:** Miniconda or Anaconda

### Step 1: Clone the Repository

```bash
git clone https://github.com/anonymous/BioDynaSpec.git
cd BioDynaSpec
```

### Step 2: Create Conda Environment

We provide a comprehensive dependency installation script:

```bash
bash dependencies.sh
```

This will:
1. Create a conda environment named `A800_dl_cu124` with Python 3.11
2. Install system dependencies (GCC 13.3, OpenBLAS)
3. Install PyTorch 2.6.0 with CUDA 12.4
4. Install all required Python packages

### Step 3: Activate Environment

```bash
conda activate A800_dl_cu124
```

### Key Dependencies

| Package | Version | Purpose |
|---------|---------|---------|
| PyTorch | 2.6.0 | Deep learning framework |
| PyTorch Lightning | 2.5.1 | Training framework |
| Flash Attention 2 | latest | Efficient attention |
| DeepSpeed | 0.16.5 | Distributed training |
| Ray Tune | 2.51.1 | Hyperparameter optimization |
| MDAnalysis | latest | MD trajectory I/O |
| fair-esm | latest | ESM protein embeddings |
| WandB | ≥0.12.10 | Experiment tracking |
| RDKit | 2025.3.6 | Molecular processing |
| Biotite | 1.2.0 | Structural biology toolkit |

### Optional: WandB Setup

If you want experiment tracking with Weights & Biases:

```bash
export WANDB_API_KEY="your_api_key_here"
# Or login interactively:
wandb login
```

To disable WandB, set `use_wandb: false` in the config YAML.

---

## Data Preparation

BioDynaSpec is evaluated on the [ATLAS](https://www.dsimb.inserm.fr/ATLAS/) dataset of all-atom MD simulations. The data preparation pipeline involves several steps:

### Step 0: Download ATLAS Data

```bash
# Download ATLAS trajectory files and metadata
python runner/data_prepare/download_atlas_cifs.py
```

> **Note:** Set environment variables `ATLAS_CSV_PATH`, `CIF_SAVE_DIR`, and `CHAIN_OUTPUT_DIR` to customize data paths. Defaults are under `./data/`.

### Step 1: Extract Trajectory Frames

```bash
bash scripts/extract_frames.sh
```

This extracts coordinate frames from raw MD trajectories and saves them as `.npy` files. Configure paths via environment variables:

```bash
export ATLAS_RAW_DIR="./data/raw_data/ATLAS"
export ATLAS_TRAJ_DIR="./data/trajectories/atlas"
```

### Step 2: Compute Protein Representations

**ESM Embeddings:**

```bash
bash runner/data_prepare/4_compute_esm/compute_esm.sh
```

**BiHiTo Token Representations:**

```bash
# Requires BiHiTo repository and checkpoint
export BIHITO_ROOT="/path/to/BiHiTo"
export BIHITO_CHECKPOINT_PATH="/path/to/bihito_checkpoint.ckpt"
bash runner/data_prepare/5_compute_bihito/run.sh
```

### Step 3: Generate Windowed FFT Dataset

This is the main data preprocessing step that computes the Independent Windowed Fourier Decomposition (IWFD):

```bash
bash scripts/generate_data.sh
```

This script:
1. Applies windowed FFT (window size = 2500 frames, 10ps timestep) to all trajectories
2. Computes per-frequency scaling factors (95th percentile normalization)
3. Produces normalized train/val/test splits

After completion, the dataset directory should contain:

```
dataset/
├── fft_window2500/
│   ├── atlas_10_features_fouriered_list_train_normalized.pth
│   ├── atlas_10_features_fouriered_list_val_normalized.pth
│   ├── atlas_10_features_fouriered_list_test_normalized.pth
│   └── fft_window2500_scaling_factors.pth
├── bihito_repr_cache/
│   └── atlas/
├── feature_cache/
│   └── atlas/
└── esm_cache/
    └── esm2_3b_embed/
```

---

## Training

### Single-GPU Training

```bash
bash scripts/pretrain.sh --config configs/trainer_BioDynaSpec.yaml
```

### Multi-GPU Distributed Training

```bash
NUM_GPUS=8 bash scripts/pretrain.sh --config configs/trainer_BioDynaSpec.yaml
```

### Resume from Checkpoint

```bash
bash scripts/pretrain.sh \
    --config configs/trainer_BioDynaSpec.yaml \
    --resume /path/to/checkpoint.ckpt
```

### Configuration

All training hyperparameters are defined in YAML configuration files under `configs/`:

| Config File | Description |
|-------------|-------------|
| `configs/trainer_BioDynaSpec.yaml` | **Full model** with IRFC (recommended) |
| `configs/trainer_BioDynaSpec_wo_irfc.yaml` | Ablation: without IRFC |
| `configs/trainer_far_example.yaml` | Example/template config |

**Key hyperparameters** (in `configs/trainer_BioDynaSpec.yaml`):

```yaml
# Model
with_irfc: true                 # Enable IRFC structural bias
use_complex_spectrum: true      # Complex-valued spectrum
num_sampling_steps: '50'        # Diffusion sampling steps
diffusion_steps: 1000           # Total diffusion timesteps
noise_schedule: 'cosine'        # Noise schedule type
max_spectrum_len: 2560          # Maximum frequency length

# Training
max_epochs: 50
lr: 0.0001
warmup_epochs: 10
precision: 'bf16'               # Mixed precision training
gradient_clip_val: 1.0
enable_spectrum_chunk_loss: true # Spectral consistency loss
```

Configuration priority: **command-line arguments > YAML config > `BaseConfig` defaults**.

### Hyperparameter Search with Ray Tune

```bash
bash scripts/ray_tune.sh
```

This uses Ray Tune with ASHA scheduling for efficient hyperparameter optimization. Results are saved to `./runs_ray_tune/`.

### Training Outputs

Training checkpoints and logs are saved to:

```
runs/
└── BioDynaSpec_<timestamp>/
    ├── checkpoints/
    │   ├── last.ckpt
    │   └── epoch=XX-train_loss=X.XX.ckpt
    └── wandb/   # if WandB is enabled
```

---

## Inference

### Generate Trajectories

```bash
bash scripts/inference.sh \
    --checkpoint runs/BioDynaSpec_<timestamp>/checkpoints/last.ckpt \
    --data dataset/fft_window2500/atlas_10_features_fouriered_list_test_normalized.pth \
    --output_dir runs/inference_output
```

### Full Inference Options

```bash
bash scripts/inference.sh \
    --checkpoint /path/to/checkpoint.ckpt \
    --data /path/to/test_data.pth \
    --output_dir /path/to/output \
    --data_mode fft_window \
    --window_size 2500 \
    --num_samples 1 \
    --num_frames 2500 \
    --seed 42 \
    --device cuda
```

### Inference Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `--checkpoint` | (required) | Path to trained model checkpoint |
| `--data` | `dataset/.../test_normalized.pth` | Test dataset path |
| `--output_dir` | `runs/inference_output` | Output directory |
| `--data_mode` | `fft_window` | Data mode (`fft_window`) |
| `--window_size` | `2500` | FFT window size |
| `--num_samples` | `1` | Number of samples per protein |
| `--num_frames` | `2500` | Number of trajectory frames to generate |
| `--seed` | `42` | Random seed |
| `--device` | `cuda` | Device (`cuda` or `cpu`) |

### Inference Outputs

```
runs/inference_output/
├── reconstructed_trajs/        # Generated PDB trajectories
│   ├── protein_0/
│   │   ├── frame_0000.pdb
│   │   ├── frame_0001.pdb
│   │   └── ...
│   └── ...
├── metrics.json                # Quantitative evaluation metrics
└── inference_<timestamp>.log   # Inference log
```

---

## Results

### Trajectory Generation (ATLAS, 250 frames)

| Method | \(R_{250}\) (Å) ↓ | PCA-2D Corr ↑ | Stepwise Dist ↓ |
|--------|:---:|:---:|:---:|
| MDGEN | 3.810 | — | — |
| ProAR | 3.526 | — | — |
| **BioDynaSpec** | **1.509** | **Best** | **Best** |

**BioDynaSpec** reduces trajectory RMSE by **60.4%** vs. MDGEN and **57.2%** vs. ProAR.

### Equilibrium Conformational Sampling

| Method | Root Mean \(W_2\) ↓ | MD PCA \(W_2\) ↓ | Joint PCA \(W_2\) ↓ |
|--------|:---:|:---:|:---:|
| Next Best | 2.62 | 1.39 | 2.27 |
| **BioDynaSpec** | **1.31** | **0.90** | **1.19** |
| Improvement | 50.03% | 35.25% | 47.58% |

### Ablation Study

| Variant | Effect |
|---------|--------|
| Full model (w/ IRFC) | Best overall performance |
| w/o IRFC | Severely degrades RMSE/MAE and correlation |
| w/o Spectrum Chunk Loss | Reduced spectral consistency |

To reproduce the ablation without IRFC:

```bash
bash scripts/pretrain.sh --config configs/trainer_BioDynaSpec_wo_irfc.yaml
```

---

## Project Structure

```
BioDynaSpec/
├── README.md                          # This file
├── dependencies.sh                    # Environment setup script
│
├── configs/                           # Configuration files
│   ├── trainer_BioDynaSpec.yaml       # Full model config (with IRFC)
│   ├── trainer_BioDynaSpec_wo_irfc.yaml # Ablation config (without IRFC)
│   ├── trainer_far_example.yaml       # Example/template config
│   ├── configs_data.py                # Data path configuration
│   └── configs_olmo_model.py          # OLMo model configuration
│
├── scripts/                           # Launch scripts
│   ├── pretrain.sh                    # Training launcher
│   ├── inference.sh                   # Inference launcher
│   ├── generate_data.sh              # Data generation pipeline
│   ├── extract_frames.sh             # Frame extraction from MD
│   └── ray_tune.sh                   # Hyperparameter search
│
├── src/                               # Source code
│   ├── trainer_far.py                 # PyTorch Lightning training loop
│   ├── protein_spectrum_far.py        # Core FAR model (encoder + decoder)
│   ├── main_model.py                  # Model wrapper + LR scheduler
│   ├── inferencer_far.py              # Inference engine
│   ├── featureextractor.py            # Protein feature extraction
│   ├── data_fourier.py                # Fourier data loading
│   │
│   ├── model/                         # Model components
│   │   ├── diffloss.py                # Diffusion loss (SpectrumDiffLoss)
│   │   ├── diffusion/                 # Diffusion modules
│   │   │   ├── gaussian_diffusion.py  # Gaussian diffusion process
│   │   │   ├── denoise_net.py         # SpectralDiT denoising network
│   │   │   ├── respace.py             # Diffusion step respacing
│   │   │   └── diffusion_utils.py     # Diffusion utilities
│   │   ├── loss/                      # Loss functions
│   │   │   ├── spectrum_loss.py       # SpectrumChunkLoss
│   │   │   └── spectrum_fid.py        # Spectral FID metric
│   │   └── modules/                   # Neural network modules
│   │       ├── transformer.py         # Diffusion Transformer
│   │       ├── attention.py           # Attention mechanisms
│   │       ├── olmo_transformer.py    # OLMo Transformer blocks
│   │       └── primitives.py          # Basic building blocks
│   │
│   ├── data/                          # Data processing
│   │   ├── fourier/                   # Fourier transform utilities
│   │   │   └── utils.py              # IWFD implementation
│   │   └── basic/                     # Basic data utilities
│   │       └── compute_esm.py         # ESM embedding computation
│   │
│   ├── openfold_local/                # Local OpenFold modules (IPA, etc.)
│   ├── protenix/                      # Local Protenix modules
│   ├── utils/                         # General utilities
│   └── visualize/                     # Visualization tools
│
├── runner/                            # Data preparation runners
│   └── data_prepare/
│       ├── replace_positions_full_fft_atlas.py  # IWFD preprocessing
│       ├── compute_fft_window_scaling.py        # Scaling factor computation
│       ├── download_atlas_cifs.py               # ATLAS data download
│       ├── 0_extract_frames/                    # Frame extraction
│       ├── 2_prepare_data/                      # Data cleaning
│       ├── 4_compute_esm/                       # ESM computation
│       └── 5_compute_bihito/                    # BiHiTo computation
│
├── dataset/                           # Processed datasets (generated)
│   └── fft_window2500/                # Windowed FFT data
│
├── figures/                           # Paper figures
│   ├── mj_motivation.pdf              # Motivation figure
│   └── BDSmethond_v2.pdf             # Method overview figure
│
└── logs/                              # Training/inference logs
```

---

## Environment Variables

All paths in the codebase are configurable via environment variables or command-line arguments. No absolute paths are hardcoded. Key environment variables:

| Variable | Default | Description |
|----------|---------|-------------|
| `PROJECT_ROOT` | Auto-detected | Project root directory |
| `ATLAS_RAW_DIR` | `./data/raw_data/ATLAS` | Raw ATLAS trajectory data |
| `ATLAS_TRAJ_DIR` | `./data/trajectories/atlas` | Extracted trajectory frames |
| `DATASET_DIR` | `./dataset` | Processed dataset directory |
| `FEATURE_CACHE_DIR` | `./dataset/feature_cache/atlas` | Feature cache directory |
| `BIHITO_ROOT` | (must be set) | BiHiTo repository path |
| `BIHITO_CHECKPOINT_PATH` | (must be set) | BiHiTo model checkpoint |
| `ESM_MODEL_ROOT_DIR` | `./data/esm_models` | ESM model weights directory |
| `WANDB_API_KEY` | (optional) | Weights & Biases API key |
| `NUM_GPUS` | Auto-detected | Number of GPUs for training |
| `MASTER_PORT` | `29500` | Port for distributed training |

---

## Quick Start

For a minimal end-to-end run (assuming ATLAS data and pretrained representations are available):

```bash
# 1. Setup environment
bash dependencies.sh
conda activate A800_dl_cu124

# 2. Generate windowed FFT dataset
bash scripts/generate_data.sh

# 3. Train BioDynaSpec (single GPU)
bash scripts/pretrain.sh --config configs/trainer_BioDynaSpec.yaml

# 4. Run inference
bash scripts/inference.sh \
    --checkpoint runs/BioDynaSpec_*/checkpoints/last.ckpt
```

---

## Troubleshooting

| Issue | Solution |
|-------|----------|
| `Flash Attention is not available!` | Install flash-attn: `pip install flash-attn --no-build-isolation` |
| CUDA OOM during training | Reduce `batch_size`, `max_spectrum_len`, or enable `use_checkpoint: true` in config |
| NCCL timeout in multi-GPU | Set `NCCL_IB_DISABLE=1` or adjust `NCCL_P2P_DISABLE` |
| WandB authentication error | Set `WANDB_API_KEY` env var or run `wandb login`, or disable with `use_wandb: false` |
| Missing BiHiTo/ESM models | These are required for data preprocessing; see [Data Preparation](#data-preparation) |

---

## Citation

If you find this code useful in your research, please cite:

```bibtex
@inproceedings{biodynaspec2026,
  title={BioDynaSpec: Harmonic-Guided Spatio-Spectral Autoregressive Diffusion for Protein Dynamics Generation},
  author={Anonymous},
  booktitle={International Conference on Machine Learning (ICML)},
  year={2026}
}
```

---

## License

This project is released for academic research purposes. Please refer to the LICENSE file for details.

---

## Acknowledgements

This codebase builds upon several excellent open-source projects:

- [OpenFold](https://github.com/aqlaboratory/openfold) — Invariant Point Attention (IPA) modules
- [Protenix](https://github.com/bytedance/protenix) — Protein structure primitives
- [FAR (Frequency Autoregressive)](https://github.com/YunyaoMao/FAR) — Autoregressive framework inspiration
- [ESM](https://github.com/facebookresearch/esm) — Protein language model embeddings
- [ATLAS](https://www.dsimb.inserm.fr/ATLAS/) — MD simulation dataset

We thank the authors and contributors of these projects.
