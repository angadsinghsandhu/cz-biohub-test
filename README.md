# cz-biohub-test

This project trains a multimodal deep learning model to integrate single-cell omics embeddings (from Geneformer) and textual embeddings (derived from cell type labels) for donor ID prediction. The model also includes adversarial training to mitigate sex confounding. An optional contrastive pretraining phase for the Perceiver Resampler module and hyperparameter tuning using Optuna are provided.

## Table of Contents

- [Overview](#overview)
- [Directory Structure](#directory-structure)
- [Installation and Requirements](#installation-and-requirements)
- [Usage](#usage)
  - [Command-Line Arguments](#command-line-arguments)
  - [Training the Model](#training-the-model)
  - [Hyperparameter Search](#hyperparameter-search)
  - [Pretraining the Resampler](#pretraining-the-resampler)
- [Project Workflow](#project-workflow)
- [Results and Outputs](#results-and-outputs)
- [License](#license)
- [Contact](#contact)

## Overview

This project implements a multimodal deep learning architecture that combines:
- **Omics Data:** Geneformer-generated single-cell embeddings.
- **Textual Data:** Text embeddings for cell type labels using a pretrained BioBERT model.
- **Fusion Mechanism:** Cross-attention between omics and text tokens, followed by additional refinement via a transformer encoder.
- **Latent Aggregation:** A Perceiver Resampler aggregates fused features into a latent representation.
- **Adversarial Branch:** Controls for sex confounding by applying gradient reversal during training.

The pipeline includes options for:
- Hyperparameter optimization with Optuna.
- Contrastive pretraining for the Perceiver Resampler.
- Modality ablation (disabling textual inputs) to assess the impact of cell type label information.

## Directory Structure

```
ROOT
├── uv.lock
├── report.pdf
├── main.py
├── out
│   └── ... (training artifacts, checkpoints, logs)
├── wandb
│   └── ... (Weights & Biases logs)
├── configs
├── sh
├── output
│   ├── Hyperparameter Search with Baseline Multimodal Model
│   ├── Baseline Multimodal Model (Text + Omics, No Pretraining)
│   ├── ...
│   └── temp
├── utils
│   └── print.py
└── models
    └── ... (model architectures and related files)
```

- **ROOT:** Contains the main script (`main.py`), a report, and project lock files.
- **out:** Directory for intermediate output and checkpoint files during training.
- **wandb:** Contains Weights & Biases logs if enabled.
- **configs:** Store configuration files.
- **sh:** Shell scripts for running the project.
- **output:** Final outputs including UMAP visualizations, training curves, and cross-attention heatmaps.
- **utils:** Utility scripts (e.g., printing functions).
- **models:** Custom model modules and architectures.

## Installation and Requirements

### Requirements

- **Python:** 3.8+
- **PyTorch:** 1.8+
- **Transformers:** For loading the pretrained BioBERT model.
- **Optuna:** For hyperparameter optimization.
- **Accelerate:** For distributed/accelerated training.
- **Scanpy:** For single-cell data processing and UMAP visualization.
- **cellxgene_census:** To load single-cell datasets.
- **Other Libraries:** NumPy, pandas, matplotlib, tqdm, and more.

### Installation

1. **Clone the repository:**
   ```bash
   git clone https://github.com/angadsinghsandhu/cz-biohub-test
   cd cz-biohub-test
   ```

2. **Create a virtual environment:**
   ```bash
   pip install uv
   uv init
   source venv/bin/activate  # On Windows use: venv\Scripts\activate
   ```

3. **Install dependencies:**
   ```bash
   uv sync
   ```

> *Ensure that you have installed the specific versions of the libraries as required by your project.*

## Usage

### Command-Line Arguments

The main script `main.py` accepts various command-line arguments. Here are some key options:

- `--census-version`: Specify the CELLxGENE census version (default: "2023-12-15").
- `--organism`: Set the organism name (default: "homo_sapiens").
- `--measurement-name`: Define the measurement type (e.g., "RNA").
- `--output-dir`: Directory for saving outputs and checkpoints.
- `--batch-size`: Batch size for training (default: 64).
- `--num-epochs`: Number of training epochs (default: 50).
- `--local-model-path`: Local path for the pretrained BioBERT model.
- `--seed`: Random seed for reproducibility.
- `--use-wandb`: Enable logging to Weights & Biases.
- `--search-hparams`: Enable hyperparameter search with Optuna.
- `--train-model`: Flag to train the model; if not set, the script loads `final_model.pt` for evaluation.
- `--pretrain-resampler`: Enable contrastive pretraining for the Perceiver Resampler.
- `--disable-text-modality`: Disable using textual inputs (cell type labels) for ablation studies.

### Training the Model

To train the model, run:

```bash
python main.py --train-model --use-wandb
```

This will:
1. Load and preprocess data.
2. Extract (or dummy) textual embeddings.
3. Build and split the dataset.
4. (Optionally) Run hyperparameter search.
5. Train the final model.
6. Save training checkpoints, UMAP plots, attention heatmaps, and final results.

### Hyperparameter Search

To perform hyperparameter optimization with Optuna, use:

```bash
python main.py --search-hparams
```

This will execute a series of trials and store the best hyperparameters in `best_hyperparameters.json`.

### Pretraining the Resampler

To enable contrastive pretraining for the Perceiver Resampler, add the flag:

```bash
python main.py --pretrain-resampler
```

This phase helps refine the projection modules before the final model training.

To run the project on an HPC, you can also use the provided shell scripts in the `sh` directory.

```bash
sbatch sh/cz.sh
```

## Project Workflow

1. **Data Loading and UMAP Generation:**  
   Loads single-cell data and generates a UMAP plot for visual inspection.

2. **Textual Embedding Extraction:**  
   Uses a pretrained BioBERT model (or dummy embeddings when disabled) to extract embeddings from cell type labels.

3. **Dataset Preparation:**  
   Creates a unified dataset that includes omics data, textual embeddings, and labels (donor and sex).

4. **Model Building:**  
   Constructs a multimodal model with cross-attention, transformer encoder, Perceiver Resampler, and adversarial branches.

5. **Training & Evaluation:**  
   Trains the model while logging training curves and saving model checkpoints. Evaluates model performance on validation and test splits.

6. **Hyperparameter Optimization:**  
   Optionally optimizes model hyperparameters using Optuna.

7. **Visualization:**  
   Outputs UMAP plots, training curves, and a heatmap of the Perceiver Resampler’s cross-attention weights.

## Results and Outputs

After training, you can expect to see:
- **UMAP Plot:** `umap_geneformer.png` in the output directory.
- **Training Curves:** `training_curves.png` showing train, validation, and test accuracy.
- **Attention Heatmap:** `perceiver_attention_heatmap.png` visualizing the average cross-attention weights.
- **Final Model Checkpoint:** `final_model.pt` for evaluation or future inference.
- **Hyperparameters and Final Metrics:** Saved in `best_hyperparameters.json`.

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.

## Contact

For questions or contributions, please contact:
- **Author:** Angad Sandhu
- **Email:** angadsandhuworkmail@gmail.com
- **GitHub:** [@angadsinghsandhu](https://github.com/angadsinghsandhu)