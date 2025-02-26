#!/usr/bin/env python3
"""
main.py

Script to train a multimodal model that integrates Geneformer single-cell 
embeddings (omics modality) and textual embeddings (from cell type labels) 
to predict donor IDs. This updated version uses a shared Optuna storage backend 
to ensure that each trial is only run once across multiple GPUs.
"""

import os
import json
import argparse
import numpy as np
import scanpy as sc
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, AutoModel, get_cosine_schedule_with_warmup
import optuna
import matplotlib.pyplot as plt
import warnings
import pandas as pd
import cellxgene_census
from tqdm import tqdm
import wandb
from accelerate import Accelerator

# Global placeholders for shared variables
textual_embeddings = None
accelerator = None
omics_data = None
dataloader = None
donor_map = None

##############################################################################
#                           Argument Parsing
##############################################################################
def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train multimodal model with shared Optuna hyperparameter search using Accelerate and DeepSpeed"
    )
    parser.add_argument("--census-version", type=str, default="2023-12-15",
                        help="CELLxGENE census version to use.")
    parser.add_argument("--organism", type=str, default="homo_sapiens",
                        help="Organism name.")
    parser.add_argument("--measurement-name", type=str, default="RNA",
                        help="Measurement name (e.g., RNA).")
    parser.add_argument("--output-dir", type=str, default="output",
                        help="Directory to store output artifacts and checkpoints.")
    parser.add_argument("--batch-size", type=int, default=64,
                        help="Batch size for training.")
    parser.add_argument("--num-epochs", type=int, default=50,
                        help="Number of training epochs.")
    parser.add_argument("--local-model-path", type=str,
                        default="/home/asandhu9/cz-biohub-test/models/dmis-lab/biobert/1.1",
                        help="Path to pretrained BioBERT model for textual embeddings.")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed for reproducibility.")
    parser.add_argument("--use-wandb", action="store_true",
                        help="Enable logging to Weights & Biases.")
    return parser.parse_args()

##############################################################################
#                     Data Loading and UMAP Visualization
##############################################################################
def load_data(census_version: str, organism: str, measurement_name: str, output_dir: str):
    warnings.filterwarnings("ignore")
    emb_names = ["geneformer"]
    with cellxgene_census.open_soma(census_version=census_version) as census:
        adata = cellxgene_census.get_anndata(
            census,
            organism=organism,
            measurement_name=measurement_name,
            obs_value_filter="tissue_general == 'central nervous system'",
            obs_column_names=["cell_type", "donor_id", "sex"],
            obs_embeddings=emb_names,
        )
    print(f"[Data] Loaded {adata.n_obs} cells.")

    sc.pp.neighbors(adata, use_rep="geneformer")
    sc.tl.umap(adata)
    umap_path = os.path.join(output_dir, "umap_geneformer.png")
    sc.pl.umap(adata, color="cell_type", title="UMAP of Geneformer Embeddings", show=False)
    plt.savefig(umap_path, bbox_inches="tight")
    plt.close()
    print(f"[Data] UMAP plot saved to {umap_path}")
    return adata, umap_path

##############################################################################
#                     Textual Embedding Extraction
##############################################################################
def extract_textual_embeddings(adata, model_path: str, device: torch.device):
    tokenizer = AutoTokenizer.from_pretrained(model_path, local_files_only=True)
    text_model = AutoModel.from_pretrained(model_path, local_files_only=True).to(device)
    text_model.eval()

    def get_text_embedding(label: str) -> np.ndarray:
        inputs = tokenizer(label, return_tensors="pt")
        inputs = {k: v.to(device) for k, v in inputs.items()}
        with torch.no_grad():
            outputs = text_model(**inputs)
        # Use the [CLS] token embedding
        return outputs.last_hidden_state[:, 0, :].squeeze(0).cpu().numpy()

    embeddings = [get_text_embedding(label) for label in adata.obs["cell_type"]]
    textual_embeddings_local = np.stack(embeddings, axis=0)
    print(f"[Text] Textual embeddings shape: {textual_embeddings_local.shape}")
    return textual_embeddings_local

##############################################################################
#                           Model Architecture
##############################################################################
class GradientReversal(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, lambda_):
        ctx.lambda_ = lambda_
        return x.view_as(x)
    @staticmethod
    def backward(ctx, grad_output):
        return -ctx.lambda_ * grad_output, None

def grad_reverse(x, lambda_=1.0):
    return GradientReversal.apply(x, lambda_)

class CrossAttentionBlock(nn.Module):
    def __init__(self, d_model: int, n_heads: int, dropout: float = 0.1, batch_first: bool = True):
        super(CrossAttentionBlock, self).__init__()
        print(f"[Model] Initializing CrossAttentionBlock (d_model={d_model}, n_heads={n_heads})")
        self.attention = nn.MultiheadAttention(embed_dim=d_model, num_heads=n_heads,
                                                dropout=dropout, batch_first=batch_first)
        self.norm = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
    def forward(self, query, key, value):
        attn_output, attn_weights = self.attention(query, key, value)
        output = self.norm(query + self.dropout(attn_output))
        return output, attn_weights

class PerceiverResampler(nn.Module):
    def __init__(self, input_dim: int, num_latents: int, latent_dim: int, n_heads: int,
                 dropout: float = 0.1, batch_first: bool = True):
        super(PerceiverResampler, self).__init__()
        self.latents = nn.Parameter(torch.randn(num_latents, latent_dim))
        self.cross_attn = CrossAttentionBlock(d_model=latent_dim, n_heads=n_heads,
                                                dropout=dropout, batch_first=batch_first)
        self.input_proj = nn.Linear(input_dim, latent_dim) if input_dim != latent_dim else nn.Identity()
    def forward(self, x):
        x_proj = self.input_proj(x)
        batch = x.size(0)
        latents_expanded = self.latents.unsqueeze(0).expand(batch, -1, -1)
        x_proj_unsq = x_proj.unsqueeze(1)
        out, attn_weights = self.cross_attn(latents_expanded, x_proj_unsq, x_proj_unsq)
        out = out.mean(dim=1)
        return out, attn_weights

class MultiModalModel(nn.Module):
    def __init__(self, omics_dim: int = 512, text_dim: int = 768, fusion_dim: int = 256,
                 n_heads: int = 8, num_tokens: int = 8, num_latents: int = 32,
                 latent_dim: int = 128, dropout: float = 0.1, adv_lambda: float = 0.1,
                 num_donors: int = 10):
        super(MultiModalModel, self).__init__()
        self.num_tokens = num_tokens
        self.omics_token_dim = omics_dim // num_tokens  # e.g., 64
        self.text_token_dim = text_dim // num_tokens      # e.g., 96
        self.omics_token_proj = nn.Linear(self.omics_token_dim, fusion_dim)
        self.text_token_proj = nn.Linear(self.text_token_dim, fusion_dim)
        self.cross_attn = CrossAttentionBlock(d_model=fusion_dim, n_heads=n_heads,
                                                dropout=dropout, batch_first=True)
        self.ffn = nn.Sequential(
            nn.Linear(fusion_dim, fusion_dim),
            nn.ReLU(),
            nn.Linear(fusion_dim, fusion_dim)
        )
        # Additional transformer encoder layer to refine fused tokens.
        self.transformer_encoder = nn.TransformerEncoderLayer(d_model=fusion_dim, nhead=n_heads,
                                                               dropout=dropout, batch_first=True)
        self.resampler = PerceiverResampler(input_dim=fusion_dim, num_latents=num_latents,
                                             latent_dim=latent_dim, n_heads=n_heads,
                                             dropout=dropout, batch_first=True)
        self.classifier = nn.Sequential(
            nn.Linear(latent_dim, latent_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(latent_dim, latent_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(latent_dim // 2, num_donors)
        )
        self.adv_classifier = nn.Sequential(
            nn.Linear(latent_dim, latent_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(latent_dim, latent_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(latent_dim // 2, 2)
        )
        self.adv_lambda = adv_lambda
    def forward(self, omics, text):
        batch = omics.size(0)
        omics_tokens = omics.view(batch, self.num_tokens, self.omics_token_dim)
        text_tokens = text.view(batch, self.num_tokens, self.text_token_dim)
        omics_tokens = self.omics_token_proj(omics_tokens)
        text_tokens = self.text_token_proj(text_tokens)
        fused_tokens, attn_weights = self.cross_attn(omics_tokens, text_tokens, text_tokens)
        fused_tokens = self.ffn(fused_tokens)
        fused_tokens = self.transformer_encoder(fused_tokens)
        fused_feat = fused_tokens.mean(dim=1)
        latent, attn_resampler = self.resampler(fused_feat)
        donor_logits = self.classifier(latent)
        adv_input = grad_reverse(latent, self.adv_lambda)
        sex_logits = self.adv_classifier(adv_input)
        return donor_logits, sex_logits, attn_weights, attn_resampler

##############################################################################
#                           Dataset Preparation
##############################################################################
class SingleCellDataset(Dataset):
    def __init__(self, omics_data: np.ndarray, text_data: np.ndarray, donor_labels: list, sex_labels: list):
        self.omics = torch.tensor(omics_data, dtype=torch.float32)
        self.text = torch.tensor(text_data, dtype=torch.float32)
        self.donor_labels = torch.tensor(donor_labels, dtype=torch.long)
        self.sex_labels = torch.tensor(sex_labels, dtype=torch.long)
    def __len__(self):
        return self.omics.shape[0]
    def __getitem__(self, idx):
        return self.omics[idx], self.text[idx], self.donor_labels[idx], self.sex_labels[idx]

##############################################################################
#                              Training Loop
##############################################################################
def train_model(model: nn.Module, dataloader: DataLoader, optimizer, scheduler,
                num_epochs: int, accelerator: Accelerator, log_to_wandb: bool = False):
    model.train()
    donor_criterion = nn.CrossEntropyLoss()
    adv_criterion = nn.CrossEntropyLoss()
    donor_losses, adv_losses, accuracies = [], [], []

    for epoch in range(num_epochs):
        running_donor_loss = 0.0
        running_adv_loss = 0.0
        correct = 0
        total = 0
        for omics, text, donor_labels, sex_labels in tqdm(dataloader, desc=f"Epoch {epoch+1}/{num_epochs}"):
            omics = omics.to(torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu"))
            text = text.to(torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu"))
            donor_labels = donor_labels.to(torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu"))
            sex_labels = sex_labels.to(torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu"))
            optimizer.zero_grad()
            donor_logits, sex_logits, attn_weights, attn_resampler = model(omics, text)
            loss_donor = donor_criterion(donor_logits, donor_labels)
            loss_adv = adv_criterion(sex_logits, sex_labels)
            loss = loss_donor + loss_adv
            accelerator.backward(loss)
            optimizer.step()
            scheduler.step()
            running_donor_loss += loss_donor.item() * omics.size(0)
            running_adv_loss += loss_adv.item() * omics.size(0)
            _, predicted = torch.max(donor_logits, 1)
            total += donor_labels.size(0)
            correct += (predicted == donor_labels).sum().item()
        epoch_donor_loss = running_donor_loss / total
        epoch_adv_loss = running_adv_loss / total
        epoch_acc = correct / total
        donor_losses.append(epoch_donor_loss)
        adv_losses.append(epoch_adv_loss)
        accuracies.append(epoch_acc)
        print(f"[Train] Epoch {epoch+1}/{num_epochs} - Donor Loss: {epoch_donor_loss:.4f} - "
              f"Adv Loss: {epoch_adv_loss:.4f} - Accuracy: {epoch_acc:.4f}")
        if log_to_wandb:
            wandb.log({"epoch": epoch+1, "donor_loss": epoch_donor_loss, "adv_loss": epoch_adv_loss, "accuracy": epoch_acc})
    return donor_losses, adv_losses, accuracies

##############################################################################
#                              Model Saving
##############################################################################
def save_model(model: nn.Module, output_dir: str, accelerator: Accelerator):
    os.makedirs(output_dir, exist_ok=True)
    model_to_save = accelerator.unwrap_model(model)
    save_path = os.path.join(output_dir, "final_model.pt")
    torch.save(model_to_save.state_dict(), save_path)
    print(f"[Save] Model checkpoint saved to {save_path}")

##############################################################################
#                        Hyperparameter Optimization Objective
##############################################################################
def objective(trial):
    # Suggest hyperparameters
    lr = trial.suggest_loguniform("lr", 1e-5, 1e-2)
    dropout = trial.suggest_uniform("dropout", 0.1, 0.5)
    num_latents = trial.suggest_int("num_latents", 16, 64)
    latent_dim = trial.suggest_int("latent_dim", 64, 256, step=8)
    adv_lambda = trial.suggest_uniform("adv_lambda", 0.01, 0.5)
    num_donors = len(donor_map)
    
    model_instance = MultiModalModel(omics_dim=omics_data.shape[1],
                                     text_dim=textual_embeddings.shape[1],
                                     fusion_dim=256,
                                     n_heads=8,
                                     num_tokens=8,
                                     num_latents=num_latents,
                                     latent_dim=latent_dim,
                                     dropout=dropout,
                                     adv_lambda=adv_lambda,
                                     num_donors=num_donors).to(accelerator.device)
    optimizer_instance = optim.Adam(model_instance.parameters(), lr=lr, weight_decay=1e-4)
    total_steps = len(dataloader) * 5  # Use 5 epochs for tuning
    warmup_steps = int(0.1 * total_steps)
    scheduler_instance = get_cosine_schedule_with_warmup(optimizer_instance,
                                                         num_warmup_steps=warmup_steps,
                                                         num_training_steps=total_steps)
    # Run training for 5 epochs
    _, _, accuracies_trial = train_model(model_instance, dataloader, optimizer_instance, scheduler_instance,
                                         num_epochs=5, accelerator=accelerator, log_to_wandb=False)
    final_acc = accuracies_trial[-1]
    return -final_acc  # Minimization objective

##############################################################################
#                                  Main
##############################################################################
def main():
    global textual_embeddings, accelerator, omics_data, dataloader, donor_map

    args = parse_args()
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    # Initialize Accelerator (DeepSpeed data parallelism via Accelerate)
    accelerator = Accelerator()
    print(f"[Main] Accelerator initialized on device: {accelerator.device}")

    os.makedirs(args.output_dir, exist_ok=True)

    # Load data and generate UMAP plot.
    adata, umap_path = load_data(args.census_version, args.organism, args.measurement_name, args.output_dir)

    # Extract textual embeddings.
    print("Extracting Textual Embeddings...")
    textual_embeddings = extract_textual_embeddings(adata, args.local_model_path, accelerator.device)
    print("Textual Embeddings Extracted!!!")

    # Build label mappings.
    donor_map = {donor: i for i, donor in enumerate(pd.unique(adata.obs["donor_id"]))}
    sex_map = {"male": 0, "female": 1}
    donor_labels = [donor_map[d] for d in adata.obs["donor_id"]]
    sex_labels = [sex_map[s.lower()] if s.lower() in sex_map else 0 for s in adata.obs["sex"]]

    # Build dataset and dataloader.
    omics_data = adata.obsm["geneformer"]
    dataset = SingleCellDataset(omics_data, textual_embeddings, donor_labels, sex_labels)
    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, num_workers=4)

    # ----------------------------
    # Hyperparameter Optimization with Shared Storage
    # ----------------------------
    storage_name = "sqlite:///optuna_study.db"
    study = optuna.create_study(direction="minimize", storage=storage_name,
                                study_name="multimodal_study", load_if_exists=True)
    study.optimize(objective, n_trials=10)
    accelerator.wait_for_everyone()  # Ensure all processes have finished optimization.
    best_params = study.best_params
    current_best_acc = -study.best_value
    print(f"[Optuna] Best hyperparameters from Optuna: {best_params} with accuracy {current_best_acc:.4f}")

    # Save best hyperparameters.
    best_hp_path = os.path.join(args.output_dir, "best_hyperparameters.json")
    final_hp_results = {"best_params": best_params, "final_accuracy": current_best_acc}
    with open(best_hp_path, "w") as f:
        json.dump(final_hp_results, f, indent=4)
    print(f"[Main] Best hyperparameters saved to {best_hp_path}")

    if args.use_wandb:
        wandb.init(project="cz-biohub-test", config=best_params, reinit=True)

    num_donors = len(donor_map)
    model = MultiModalModel(
        omics_dim=omics_data.shape[1],
        text_dim=textual_embeddings.shape[1],
        fusion_dim=256,
        n_heads=8,
        num_tokens=8,
        num_latents=best_params["num_latents"],
        latent_dim=best_params["latent_dim"],
        dropout=best_params["dropout"],
        adv_lambda=best_params["adv_lambda"],
        num_donors=num_donors
    )

    optimizer = optim.Adam(model.parameters(), lr=best_params["lr"], weight_decay=1e-4)
    total_steps = len(dataloader) * args.num_epochs
    warmup_steps = int(0.1 * total_steps)
    scheduler = get_cosine_schedule_with_warmup(optimizer, num_warmup_steps=warmup_steps, num_training_steps=total_steps)

    model, optimizer, dataloader, scheduler = accelerator.prepare(model, optimizer, dataloader, scheduler)

    donor_losses, adv_losses, accuracies = train_model(model, dataloader, optimizer, scheduler,
                                                      args.num_epochs, accelerator, log_to_wandb=args.use_wandb)

    epochs = range(1, args.num_epochs + 1)
    training_curves_path = os.path.join(args.output_dir, "training_curves.png")
    plt.figure(figsize=(18, 5))
    plt.subplot(1, 3, 1)
    plt.plot(epochs, donor_losses, label="Donor Loss")
    plt.xlabel("Epoch"); plt.ylabel("Loss"); plt.title("Donor Loss"); plt.legend()
    plt.subplot(1, 3, 2)
    plt.plot(epochs, adv_losses, label="Adversarial Loss", color="orange")
    plt.xlabel("Epoch"); plt.ylabel("Loss"); plt.title("Adversarial Loss"); plt.legend()
    plt.subplot(1, 3, 3)
    plt.plot(epochs, accuracies, label="Accuracy", color="green")
    plt.xlabel("Epoch"); plt.ylabel("Accuracy"); plt.title("Donor ID Prediction Accuracy"); plt.legend()
    plt.tight_layout()
    plt.savefig(training_curves_path)
    plt.close()
    print(f"[Main] Training curves saved to {training_curves_path}")

    model.eval()
    sample_omics, sample_text, _, _ = next(iter(dataloader))
    sample_omics = sample_omics.to(accelerator.device)
    sample_text = sample_text.to(accelerator.device)
    _, _, attn_weights, _ = model(sample_omics, sample_text)
    if attn_weights.dim() == 3:
        avg_attn = attn_weights.mean(dim=0).detach().cpu().numpy()
    elif attn_weights.dim() == 4:
        avg_attn = attn_weights.mean(dim=(0, 1)).detach().cpu().numpy()
    else:
        raise ValueError("Unexpected attention weights dimensions.")
    cross_attn_path = os.path.join(args.output_dir, "cross_attention_heatmap.png")
    plt.figure(figsize=(6, 4))
    plt.imshow(avg_attn, cmap="viridis", aspect="auto")
    plt.colorbar()
    plt.title("Average Cross-Attention Weights")
    plt.xlabel("Key Token"); plt.ylabel("Query Token")
    plt.savefig(cross_attn_path)
    plt.close()
    print(f"[Main] Cross-attention heatmap saved to {cross_attn_path}")

    final_results = {"best_params": best_params, "final_accuracy": accuracies[-1]}
    with open(best_hp_path, "w") as f:
        json.dump(final_results, f, indent=4)
    print(f"[Main] Best hyperparameters and final accuracy saved to {best_hp_path}")

    save_model(model, args.output_dir, accelerator)

    if args.use_wandb:
        wandb.log({"final_accuracy": accuracies[-1]})
        wandb.finish()

    print("\n[Main] Final Results Summary:")
    print(f" - Number of cells loaded: {adata.n_obs}")
    print(f" - UMAP plot saved at: {umap_path}")
    print(f" - Training curves saved at: {training_curves_path}")
    print(f" - Cross-attention heatmap saved at: {cross_attn_path}")
    print(f" - Best hyperparameters saved at: {best_hp_path}")
    print(f" - Final donor ID prediction accuracy: {accuracies[-1]:.4f}")

if __name__ == "__main__":
    main()
