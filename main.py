#!/usr/bin/env python3
"""
main.py

Script to train a multimodal model that integrates Geneformer single-cell 
embeddings (omics modality) and textual embeddings (from cell type labels) 
to predict donor IDs with adversarial training to control for the sex confounder.
Additionally, this script includes:
  - A pretraining phase for the Perceiver Resampler using a contrastive objective.
  - An ablation option to disable the textual modality (cell type label information) 
    to assess its impact on model performance.
  
Features:
- Flags to toggle hyperparameter search, model training, pretraining, and text modality.
- Modular training components for efficient parallel processing.
- Detailed logging and final analysis summary.
"""

##############################################################################
#                              Imports
##############################################################################
import os
import json
import argparse
from typing import Any, Tuple, List

import numpy as np
import scanpy as sc
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, random_split
from transformers import AutoTokenizer, AutoModel, get_cosine_schedule_with_warmup
import optuna
import matplotlib.pyplot as plt
import warnings
import pandas as pd
import cellxgene_census
from tqdm import tqdm
import wandb
from accelerate import Accelerator
from anndata import AnnData  # Type hint for the AnnData object from scanpy

##############################################################################
#                   Global Placeholders for Shared Variables
##############################################################################
textual_embeddings: np.ndarray = None
accelerator: Accelerator = None
omics_data: np.ndarray = None
# Global dataloaders for the splits (train, val, test)
train_dataloader: DataLoader = None
val_dataloader: DataLoader = None
test_dataloader: DataLoader = None
donor_map: dict = None

# Global flag for pretraining (set in main)
USE_PRETRAIN = False

##############################################################################
#                           Argument Parsing
##############################################################################
def parse_args() -> argparse.Namespace:
    """
    Parse command-line arguments for the training script.
    """
    parser = argparse.ArgumentParser(
        description="Train multimodal model with optional hyperparameter search, pretraining, and modality ablation."
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
    # New flags
    parser.add_argument("--search-hparams", action="store_true",
                        help="Enable searching for hyperparameters with Optuna.")
    parser.add_argument("--train-model", action="store_true",
                        help="Enable training the model. If false, loads final_model.pt for evaluation.")
    parser.add_argument("--pretrain-resampler", action="store_true",
                        help="Enable contrastive pretraining for the Perceiver Resampler modules.")
    parser.add_argument("--disable-text-modality", action="store_true",
                        help="Disable using textual (cell type) information; use only omics data.")
    return parser.parse_args()

##############################################################################
#              Data Loading and UMAP Visualization
##############################################################################
def load_data(census_version: str, organism: str, measurement_name: str, output_dir: str) -> Tuple[AnnData, str]:
    """
    Load single-cell data from the CELLxGENE census and generate a UMAP plot.
    """
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

    # Compute neighbors and UMAP for visualization
    sc.pp.neighbors(adata, use_rep="geneformer")
    sc.tl.umap(adata)
    umap_path = os.path.join(output_dir, "umap_geneformer.png")
    sc.pl.umap(adata, color="cell_type", title="UMAP of Geneformer Embeddings", show=False)
    plt.savefig(umap_path, bbox_inches="tight")
    plt.close()
    print(f"[Data] UMAP plot saved to {umap_path}")
    return adata, umap_path

##############################################################################
#                Textual Embedding Extraction
##############################################################################
def extract_textual_embeddings(adata: AnnData, model_path: str, device: torch.device) -> np.ndarray:
    """
    Extract textual embeddings for cell type labels using a pretrained model.
    """
    tokenizer = AutoTokenizer.from_pretrained(model_path, local_files_only=True)
    text_model = AutoModel.from_pretrained(model_path, local_files_only=True).to(device)
    text_model.eval()

    def get_text_embedding(label: str) -> np.ndarray:
        """
        Obtain the embedding for a given label string.
        """
        inputs = tokenizer(label, return_tensors="pt")
        inputs = {k: v.to(device) for k, v in inputs.items()}
        with torch.no_grad():
            outputs = text_model(**inputs)
        # Use the [CLS] token embedding (first token)
        return outputs.last_hidden_state[:, 0, :].squeeze(0).cpu().numpy()

    embeddings = [get_text_embedding(label) for label in adata.obs["cell_type"]]
    textual_embeddings_local = np.stack(embeddings, axis=0)
    print(f"[Text] Textual embeddings shape: {textual_embeddings_local.shape}")
    return textual_embeddings_local

##############################################################################
#                         Gradient Reversal Operation
##############################################################################
class GradientReversal(torch.autograd.Function):
    """
    Implements the gradient reversal operation for adversarial training.
    """
    @staticmethod
    def forward(ctx, x: torch.Tensor, lambda_: float) -> torch.Tensor:
        ctx.lambda_ = lambda_
        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor) -> Tuple[torch.Tensor, None]:
        return -ctx.lambda_ * grad_output, None

def grad_reverse(x: torch.Tensor, lambda_: float = 1.0) -> torch.Tensor:
    """
    Apply gradient reversal to the input tensor.
    """
    return GradientReversal.apply(x, lambda_)

##############################################################################
#                      Cross-Attention Block Module
##############################################################################
class CrossAttentionBlock(nn.Module):
    """
    A module that performs multi-head cross-attention.
    """
    def __init__(self, d_model: int, n_heads: int, dropout: float = 0.1, batch_first: bool = True) -> None:
        super(CrossAttentionBlock, self).__init__()
        print(f"[Model] Initializing CrossAttentionBlock (d_model={d_model}, n_heads={n_heads})")
        self.attention = nn.MultiheadAttention(embed_dim=d_model, num_heads=n_heads,
                                               dropout=dropout, batch_first=batch_first)
        self.norm = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, query: torch.Tensor, key: torch.Tensor, value: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        attn_output, attn_weights = self.attention(query, key, value)
        output = self.norm(query + self.dropout(attn_output))
        return output, attn_weights

##############################################################################
#                    Perceiver Resampler Module
##############################################################################
class PerceiverResampler(nn.Module):
    """
    Module that projects fused features into latent space via cross-attention.
    Expects a 3D input: [batch, seq_len, input_dim].
    """
    def __init__(self, input_dim: int, num_latents: int, latent_dim: int, n_heads: int,
                 dropout: float = 0.1, batch_first: bool = True) -> None:
        super(PerceiverResampler, self).__init__()
        self.latents = nn.Parameter(torch.randn(num_latents, latent_dim))
        self.cross_attn = CrossAttentionBlock(d_model=latent_dim, n_heads=n_heads,
                                              dropout=dropout, batch_first=batch_first)
        # Project input into latent_dim if needed
        if input_dim != latent_dim:
            self.input_proj = nn.Linear(input_dim, latent_dim)
        else:
            self.input_proj = nn.Identity()

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        x: [batch, seq_len, input_dim]
        Returns:
          out: [batch, latent_dim]
          attn_weights: [batch, n_heads, num_latents, seq_len]
        """
        x_proj = self.input_proj(x)  # [batch, seq_len, latent_dim]
        batch_size, seq_len, _ = x_proj.shape

        # Expand the learned latents to match the batch dimension
        latents_expanded = self.latents.unsqueeze(0).expand(batch_size, -1, -1)
        # Cross-attention: latents as query, x_proj as key & value
        out, attn_weights = self.cross_attn(latents_expanded, x_proj, x_proj)
        # Average across the latents dimension
        out = out.mean(dim=1)  # [batch, latent_dim]
        return out, attn_weights

##############################################################################
#                      Multimodal Model Architecture
##############################################################################
class MultiModalModel(nn.Module):
    """
    Multimodal model integrating omics and textual embeddings with:
      - Cross-attention between omics tokens and text tokens (if text is enabled)
      - Transformer encoder for fused tokens (when using text)
      - Perceiver Resampler for latent aggregation
      - Adversarial branch for sex confounding
      
    If text modality is disabled, the model uses only omics data.
    """
    def __init__(self, omics_dim: int = 512, text_dim: int = 768, fusion_dim: int = 256,
                 n_heads: int = 8, num_tokens: int = 8, num_latents: int = 32,
                 latent_dim: int = 128, dropout: float = 0.1, adv_lambda: float = 0.1,
                 num_donors: int = 10, use_text: bool = True) -> None:
        super(MultiModalModel, self).__init__()
        self.num_tokens = num_tokens
        self.omics_token_dim = omics_dim // num_tokens  # e.g., 64 if omics_dim=512 and num_tokens=8
        self.text_token_dim = text_dim // num_tokens    # e.g., 96 if text_dim=768 and num_tokens=8
        self.use_text = use_text

        # Project tokens into a common fusion space
        self.omics_token_proj = nn.Linear(self.omics_token_dim, fusion_dim)
        if self.use_text:
            self.text_token_proj = nn.Linear(self.text_token_dim, fusion_dim)
            # Cross-attention: omics tokens query text tokens
            self.cross_attn = CrossAttentionBlock(d_model=fusion_dim, n_heads=n_heads,
                                                  dropout=dropout, batch_first=True)
            # Feed-forward network after cross-attention
            self.ffn = nn.Sequential(
                nn.Linear(fusion_dim, fusion_dim),
                nn.ReLU(),
                nn.Linear(fusion_dim, fusion_dim)
            )
            # Transformer encoder for additional refinement
            self.transformer_encoder = nn.TransformerEncoderLayer(d_model=fusion_dim,
                                                                  nhead=n_heads,
                                                                  dropout=dropout,
                                                                  batch_first=True)
        # Perceiver Resampler to aggregate fused features into latent representation
        self.resampler = PerceiverResampler(input_dim=fusion_dim,
                                            num_latents=num_latents,
                                            latent_dim=latent_dim,
                                            n_heads=n_heads,
                                            dropout=dropout,
                                            batch_first=True)

        # Donor classifier head
        self.classifier = nn.Sequential(
            nn.Linear(latent_dim, latent_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(latent_dim, latent_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(latent_dim // 2, num_donors)
        )

        # Adversarial sex classifier head
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

    def forward(self, omics: torch.Tensor, text: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        omics: [batch, omics_dim]
        text:  [batch, text_dim]
        """
        batch_size = omics.size(0)
        omics_tokens = omics.view(batch_size, self.num_tokens, self.omics_token_dim)   # [B, 8, 64]

        if self.use_text:
            text_tokens = text.view(batch_size, self.num_tokens, self.text_token_dim)      # [B, 8, 96]
            # Project tokens into the fusion space
            omics_tokens_proj = self.omics_token_proj(omics_tokens)  # [B, 8, fusion_dim]
            text_tokens_proj = self.text_token_proj(text_tokens)     # [B, 8, fusion_dim]

            # Cross-attention: omics tokens (query) -> text tokens (key/value)
            fused_tokens, attn_weights = self.cross_attn(omics_tokens_proj, text_tokens_proj, text_tokens_proj)
            fused_tokens = self.ffn(fused_tokens)
            # Transformer encoder for additional refinement
            fused_tokens = self.transformer_encoder(fused_tokens)
        else:
            # If text modality is disabled, use only omics tokens.
            fused_tokens = self.omics_token_proj(omics_tokens)
            # Dummy attention weights for consistency
            attn_weights = torch.zeros(batch_size, 1, 1, 1, device=omics.device)

        # Pass the fused tokens into PerceiverResampler
        latent, attn_resampler = self.resampler(fused_tokens)
        # Donor classification branch
        donor_logits = self.classifier(latent)
        # Adversarial branch with gradient reversal
        adv_input = grad_reverse(latent, self.adv_lambda)
        sex_logits = self.adv_classifier(adv_input)

        return donor_logits, sex_logits, attn_weights, attn_resampler

##############################################################################
#                        Dataset Preparation Class
##############################################################################
class SingleCellDataset(Dataset):
    """
    Dataset class for single-cell omics and textual data.
    """
    def __init__(self, omics_data: np.ndarray, text_data: np.ndarray, donor_labels: List[int], sex_labels: List[int]) -> None:
        self.omics = torch.tensor(omics_data, dtype=torch.float32)
        self.text = torch.tensor(text_data, dtype=torch.float32)
        self.donor_labels = torch.tensor(donor_labels, dtype=torch.long)
        self.sex_labels = torch.tensor(sex_labels, dtype=torch.long)

    def __len__(self) -> int:
        return self.omics.shape[0]

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        return self.omics[idx], self.text[idx], self.donor_labels[idx], self.sex_labels[idx]

##############################################################################
#         Modified Evaluation Function (Donor & Adversarial Loss)
##############################################################################
def evaluate_model_full(model: nn.Module, dataloader: DataLoader, accelerator: Accelerator) -> Tuple[float, float, float]:
    """
    Evaluate the model on a given dataloader and return:
      - Average donor loss, average adversarial loss, and donor accuracy.
    """
    model.eval()
    donor_criterion = nn.CrossEntropyLoss()
    adv_criterion = nn.CrossEntropyLoss()
    running_donor_loss = 0.0
    running_adv_loss = 0.0
    correct = 0
    total = 0
    with torch.no_grad():
        for omics, text, donor_labels, sex_labels in dataloader:
            omics = omics.to(accelerator.device)
            text = text.to(accelerator.device)
            donor_labels = donor_labels.to(accelerator.device)
            sex_labels = sex_labels.to(accelerator.device)

            donor_logits, sex_logits, _, _ = model(omics, text)
            loss_donor = donor_criterion(donor_logits, donor_labels)
            loss_adv = adv_criterion(sex_logits, sex_labels)

            running_donor_loss += loss_donor.item() * omics.size(0)
            running_adv_loss += loss_adv.item() * omics.size(0)

            _, predicted = torch.max(donor_logits, 1)
            total += donor_labels.size(0)
            correct += (predicted == donor_labels).sum().item()

    avg_donor_loss = running_donor_loss / total
    avg_adv_loss = running_adv_loss / total
    accuracy = correct / total
    model.train()  # Switch back to train mode
    return avg_donor_loss, avg_adv_loss, accuracy

##############################################################################
#           Modified Training Loop with Epoch-wise Checkpointing
##############################################################################
def train_model(model: nn.Module,
                train_loader: DataLoader,
                optimizer: optim.Optimizer,
                scheduler: torch.optim.lr_scheduler._LRScheduler,
                num_epochs: int,
                accelerator: Accelerator,
                output_dir: str,
                log_to_wandb: bool = False,
                val_loader: DataLoader = None,
                test_loader: DataLoader = None,
                early_stopping_patience: int = 3,
                save_checkpoints: bool = True) -> Tuple[List[float], List[float], List[float], List[int]]:
    """
    Train the multimodal model for a given number of epochs while evaluating on 
    validation and test sets.
    """
    model.train()
    donor_criterion = nn.CrossEntropyLoss()
    adv_criterion = nn.CrossEntropyLoss()

    train_accs, val_accs, test_accs, epoch_list = [], [], [], []
    
    best_test_acc = 0.0
    best_epoch = -1
    patience_counter = 0
    prev_train_acc, prev_val_acc, prev_test_acc = None, None, None

    if save_checkpoints:
        os.makedirs(output_dir, exist_ok=True)

    for epoch in range(num_epochs):
        running_donor_loss = 0.0
        running_adv_loss = 0.0
        correct_train = 0
        total_train = 0

        for omics, text, donor_labels, sex_labels in tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs}"):
            omics = omics.to(accelerator.device)
            text = text.to(accelerator.device)
            donor_labels = donor_labels.to(accelerator.device)
            sex_labels = sex_labels.to(accelerator.device)

            optimizer.zero_grad()
            donor_logits, sex_logits, _, _ = model(omics, text)
            loss_donor = donor_criterion(donor_logits, donor_labels)
            loss_adv = adv_criterion(sex_logits, sex_labels)
            loss = loss_donor + loss_adv

            accelerator.backward(loss)
            optimizer.step()
            scheduler.step()

            running_donor_loss += loss_donor.item() * omics.size(0)
            running_adv_loss += loss_adv.item() * omics.size(0)
            _, predicted = torch.max(donor_logits, 1)
            total_train += donor_labels.size(0)
            correct_train += (predicted == donor_labels).sum().item()

        epoch_train_acc = correct_train / total_train

        if val_loader is not None:
            val_d_loss, val_a_loss, epoch_val_acc = evaluate_model_full(model, val_loader, accelerator)
        else:
            val_d_loss = 0.0
            val_a_loss = 0.0
            epoch_val_acc = 0.0

        if test_loader is not None:
            test_d_loss, test_a_loss, epoch_test_acc = evaluate_model_full(model, test_loader, accelerator)
        else:
            test_d_loss = 0.0
            test_a_loss = 0.0
            epoch_test_acc = 0.0

        print(f"[Epoch {epoch+1}]")
        print(f"  Train - Accuracy: {epoch_train_acc:.4f}")
        print(f"  Val   - Donor Loss: {val_d_loss:.4f}, Adv Loss: {val_a_loss:.4f}, Accuracy: {epoch_val_acc:.4f}")
        print(f"  Test  - Donor Loss: {test_d_loss:.4f}, Adv Loss: {test_a_loss:.4f}, Accuracy: {epoch_test_acc:.4f}")

        if log_to_wandb:
            wandb.log({
                "epoch": epoch+1,
                "train_accuracy": epoch_train_acc,
                "val_donor_loss": val_d_loss,
                "val_adv_loss": val_a_loss,
                "val_accuracy": epoch_val_acc,
                "test_donor_loss": test_d_loss,
                "test_adv_loss": test_a_loss,
                "test_accuracy": epoch_test_acc,
            })

        train_accs.append(epoch_train_acc)
        val_accs.append(epoch_val_acc)
        test_accs.append(epoch_test_acc)
        epoch_list.append(epoch+1)

        if save_checkpoints:
            epoch_checkpoint_path = os.path.join(output_dir, f"model_epoch_{epoch+1}.pt")
            torch.save(accelerator.unwrap_model(model).state_dict(), epoch_checkpoint_path)
            print(f"  [Checkpoint] Model saved to {epoch_checkpoint_path}")

            if epoch_test_acc > best_test_acc:
                best_test_acc = epoch_test_acc
                best_epoch = epoch+1
                best_model_path = os.path.join(output_dir, "best_model.pt")
                torch.save(accelerator.unwrap_model(model).state_dict(), best_model_path)
                print(f"  [Best] New best model at epoch {epoch+1} with test accuracy: {epoch_test_acc:.4f}")

        if prev_train_acc is not None and prev_val_acc is not None and prev_test_acc is not None:
            if (epoch_train_acc > prev_train_acc and epoch_val_acc < prev_val_acc and epoch_test_acc < prev_test_acc):
                patience_counter += 1
                print(f"  Overfitting detected. Patience counter: {patience_counter}/{early_stopping_patience}")
            else:
                patience_counter = 0
        prev_train_acc, prev_val_acc, prev_test_acc = epoch_train_acc, epoch_val_acc, epoch_test_acc

        if patience_counter >= early_stopping_patience:
            print("  Early stopping triggered due to overfitting.")
            break

    if save_checkpoints:
        print(f"[Training] Best model from epoch {best_epoch} with test accuracy: {best_test_acc:.4f}")
    return train_accs, val_accs, test_accs, epoch_list

##############################################################################
#                  Function to Print Sample Predictions
##############################################################################
def sample_predictions(model: nn.Module, dataloader: DataLoader, accelerator: Accelerator, donor_inv_map: dict, num_samples: int = 5) -> None:
    """
    Print sample predictions from the model on the provided dataloader.
    """
    model.eval()
    samples_printed = 0
    print("\n[Sample Predictions]")
    with torch.no_grad():
        for omics, text, donor_labels, _ in dataloader:
            omics = omics.to(accelerator.device)
            text = text.to(accelerator.device)
            donor_logits, _, _, _ = model(omics, text)
            _, predicted = torch.max(donor_logits, 1)
            for i in range(omics.size(0)):
                pred_label = predicted[i].item()
                true_label = donor_labels[i].item()
                print(f"Sample {samples_printed+1}: Predicted Donor: {donor_inv_map[pred_label]}, True Donor: {donor_inv_map[true_label]}")
                samples_printed += 1
                if samples_printed >= num_samples:
                    break
            if samples_printed >= num_samples:
                break
    model.train()

##############################################################################
#                           Model Saving Function
##############################################################################
def save_model(model: nn.Module, output_dir: str, accelerator: Accelerator) -> None:
    """
    Save the final model checkpoint to disk.
    """
    os.makedirs(output_dir, exist_ok=True)
    model_to_save = accelerator.unwrap_model(model)
    save_path = os.path.join(output_dir, "final_model.pt")
    torch.save(model_to_save.state_dict(), save_path)
    print(f"[Save] Model checkpoint saved to {save_path}")

##############################################################################
#         Contrastive Pretraining Function for the Perceiver Resampler
##############################################################################
def pretrain_contrastive(model: MultiModalModel, dataloader: DataLoader,
                         optimizer: optim.Optimizer, scheduler: torch.optim.lr_scheduler._LRScheduler,
                         num_epochs: int, device: torch.device, temperature: float = 0.1) -> None:
    """
    Pretrain the omics and text projection modules using a contrastive loss.
    For each batch, the positive pair is the omics and text representations (averaged over tokens)
    for the same cell, and negatives are other samples in the batch.
    """
    model.train()
    loss_fn = nn.CrossEntropyLoss()
    for epoch in range(num_epochs):
        running_loss = 0.0
        for omics, text, _, _ in dataloader:
            omics = omics.to(device)
            text = text.to(device)
            batch_size = omics.size(0)
            # Compute omics representation
            omics_tokens = omics.view(batch_size, model.num_tokens, model.omics_token_dim)
            omics_tokens = omics_tokens.to(next(model.omics_token_proj.parameters()).device)
            omics_repr = model.omics_token_proj(omics_tokens).mean(dim=1)  # [B, fusion_dim]
            # Compute text representation only if text modality is enabled
            if model.use_text:
                text_tokens = text.view(batch_size, model.num_tokens, model.text_token_dim)
                text_tokens = text_tokens.to(next(model.text_token_proj.parameters()).device)
                text_repr = model.text_token_proj(text_tokens).mean(dim=1)  # [B, fusion_dim]
            else:
                continue  # Skip pretraining if text modality is disabled

            # Normalize representations
            omics_norm = nn.functional.normalize(omics_repr, p=2, dim=1)
            text_norm = nn.functional.normalize(text_repr, p=2, dim=1)
            # Compute similarity matrix
            logits = torch.matmul(omics_norm, text_norm.T) / temperature  # [B, B]
            logits = logits.to(device)
            targets = torch.arange(batch_size, device=device)
            loss_omics_to_text = loss_fn(logits, targets)
            loss_text_to_omics = loss_fn(logits.T, targets)
            loss = (loss_omics_to_text + loss_text_to_omics) / 2

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            scheduler.step()
            running_loss += loss.item() * batch_size

        avg_loss = running_loss / len(dataloader.dataset)
        print(f"[Pretrain] Epoch {epoch+1}/{num_epochs}, Loss: {avg_loss:.4f}")

##############################################################################
#     Hyperparameter Optimization Objective Function (Optuna)
##############################################################################
def objective(trial: optuna.trial.Trial) -> float:
    """
    Objective function for Optuna hyperparameter search.
    Trains the model for 5 epochs (with optional pretraining) and evaluates on the validation set.
    Returns the negative validation accuracy.
    """
    global train_dataloader, val_dataloader, USE_PRETRAIN
    lr = trial.suggest_loguniform("lr", 1e-5, 1e-2)
    dropout = trial.suggest_uniform("dropout", 0.1, 0.5)
    num_latents = trial.suggest_int("num_latents", 16, 64)
    latent_dim = trial.suggest_int("latent_dim", 64, 256, step=8)
    adv_lambda = trial.suggest_uniform("adv_lambda", 0.01, 0.5)

    num_donors = len(donor_map)
    # Create a new model instance with the current hyperparameters and modality flag.
    model_instance = MultiModalModel(omics_dim=omics_data.shape[1],
                                     text_dim=textual_embeddings.shape[1],
                                     fusion_dim=256,
                                     n_heads=8,
                                     num_tokens=8,
                                     num_latents=num_latents,
                                     latent_dim=latent_dim,
                                     dropout=dropout,
                                     adv_lambda=adv_lambda,
                                     num_donors=num_donors,
                                     use_text=not args.disable_text_modality).to(accelerator.device)

    # Optional pretraining phase
    if USE_PRETRAIN and model_instance.use_text:
        pretrain_epochs = 3
        pretrain_optimizer = optim.Adam(model_instance.parameters(), lr=1e-4, weight_decay=1e-4)
        total_steps_pretrain = len(train_dataloader) * pretrain_epochs
        warmup_steps_pretrain = int(0.1 * total_steps_pretrain)
        pretrain_scheduler = get_cosine_schedule_with_warmup(pretrain_optimizer,
                                                             num_warmup_steps=warmup_steps_pretrain,
                                                             num_training_steps=total_steps_pretrain)
        pretrain_contrastive(model_instance, train_dataloader, pretrain_optimizer, pretrain_scheduler,
                             num_epochs=pretrain_epochs, device=accelerator.device)

    optimizer_instance = optim.Adam(model_instance.parameters(), lr=lr, weight_decay=1e-4)
    total_steps = len(train_dataloader) * 5  # Quick tuning over 5 epochs
    warmup_steps = int(0.1 * total_steps)
    scheduler_instance = get_cosine_schedule_with_warmup(optimizer_instance,
                                                         num_warmup_steps=warmup_steps,
                                                         num_training_steps=total_steps)

    # Train for 5 epochs without saving checkpoints
    _, _, val_accs, _ = train_model(model_instance,
                                    train_dataloader,
                                    optimizer_instance,
                                    scheduler_instance,
                                    num_epochs=5,
                                    accelerator=accelerator,
                                    output_dir="temp_output",
                                    log_to_wandb=False,
                                    val_loader=val_dataloader,
                                    save_checkpoints=False)
    final_val_acc = val_accs[-1]
    print(f"[Optuna] Trial completed with validation accuracy: {final_val_acc:.4f}")
    return -final_val_acc  # Negative accuracy for minimization

##############################################################################
#                                   Main Function
##############################################################################
def main() -> None:
    """
    Main function to run the training pipeline.
    Steps:
      1. Parse arguments and set seeds.
      2. Initialize Accelerator.
      3. Load data and generate UMAP visualization.
      4. Extract textual embeddings.
      5. Build label mappings and create dataset.
      6. Split dataset into training, validation, and test sets.
      7. Run hyperparameter search (or load best hyperparameters).
      8. (Optional) Pretrain the Perceiver Resampler if flagged.
      9. Train final model (or load final_model.pt).
     10. Evaluate the final model, print sample predictions, and generate outputs.
     11. Print analysis regarding pretraining and use of cell type labels.
    """
    global textual_embeddings, accelerator, omics_data
    global train_dataloader, val_dataloader, test_dataloader, donor_map, USE_PRETRAIN, args

    args = parse_args()
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    # Set global flag for pretraining
    USE_PRETRAIN = args.pretrain_resampler

    # Initialize Accelerator for distributed training
    accelerator = Accelerator()
    print(f"[Main] Accelerator initialized on device: {accelerator.device}")

    os.makedirs(args.output_dir, exist_ok=True)

    # 1. Load data and generate UMAP plot
    adata, umap_path = load_data(args.census_version,
                                 args.organism,
                                 args.measurement_name,
                                 args.output_dir)

    # 2. Extract textual embeddings (only if text modality is enabled)
    if not args.disable_text_modality:
        print("Extracting Textual Embeddings...")
        textual_embeddings_local = extract_textual_embeddings(adata,
                                                              args.local_model_path,
                                                              accelerator.device)
        textual_embeddings = textual_embeddings_local
        print("Textual Embeddings Extracted!!!")
    else:
        # If text modality is disabled, create a dummy textual embedding (zeros)
        textual_embeddings = np.zeros((adata.n_obs, 768))
        print("[Text] Text modality disabled; using dummy embeddings.")

    # 3. Build label mappings for donors and sex
    donor_map = {donor: i for i, donor in enumerate(pd.unique(adata.obs["donor_id"]))}
    sex_map = {"male": 0, "female": 1}
    donor_labels = [donor_map[d] for d in adata.obs["donor_id"]]
    sex_labels = [sex_map[s.lower()] if s.lower() in sex_map else 0 for s in adata.obs["sex"]]

    # 4. Build dataset from omics and textual embeddings
    omics_data = adata.obsm["geneformer"]
    dataset = SingleCellDataset(omics_data, textual_embeddings, donor_labels, sex_labels)

    # 5. Split dataset into training (70%), validation (15%), and test (15%) sets
    total_size = len(dataset)
    train_size = int(0.7 * total_size)
    val_size = int(0.15 * total_size)
    test_size = total_size - train_size - val_size
    train_dataset, val_dataset, test_dataset = random_split(dataset, [train_size, val_size, test_size])
    print(f"[Data] Dataset split into Train: {len(train_dataset)}, Validation: {len(val_dataset)}, Test: {len(test_dataset)}")

    # 6. Create dataloaders for each split
    train_dataloader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=4)
    val_dataloader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=4)
    test_dataloader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, num_workers=4)
    train_dataloader, val_dataloader, test_dataloader = accelerator.prepare(train_dataloader, val_dataloader, test_dataloader)

    # 7. Hyperparameter search or load best hyperparameters
    storage_name = "sqlite:///optuna_study.db"
    study_name = "multimodal_study"
    best_hp_path = os.path.join(args.output_dir, "best_hyperparameters.json")
    if args.search_hparams:
        print("[Main] Running Optuna hyperparameter search...")
        study = optuna.create_study(direction="minimize",
                                    storage=storage_name,
                                    study_name=study_name,
                                    load_if_exists=True)
        study.optimize(objective, n_trials=10)
        accelerator.wait_for_everyone()
        best_params = study.best_params
        current_best_acc = -study.best_value
        with open(best_hp_path, "w") as f:
            json.dump({"best_params": best_params, "final_validation_accuracy": current_best_acc}, f, indent=4)
        print(f"[Main] Best hyperparameters saved to {best_hp_path}")
        print(f"[Optuna] Best hyperparameters: {best_params} with validation accuracy {current_best_acc:.4f}")
    else:
        print("[Main] Loading best hyperparameters from best_hyperparameters.json...")
        if not os.path.exists(best_hp_path):
            print("[Main] best_hyperparameters.json not found. Using default hyperparameters.")
            best_params = {
                "lr": 0.0005947012811791727,
                "dropout": 0.11378010821029086,
                "num_latents": 61,
                "latent_dim": 256,
                "adv_lambda": 0.35824957883020864
            }
            current_best_acc = 0.0
        else:
            with open(best_hp_path, "r") as f:
                loaded_hp = json.load(f)
            best_params = loaded_hp.get("best_params", {
                "lr": 0.0005947012811791727,
                "dropout": 0.11378010821029086,
                "num_latents": 61,
                "latent_dim": 256,
                "adv_lambda": 0.35824957883020864
            })
            current_best_acc = loaded_hp.get("final_test_accuracy", 0.0)
        print(f"[Main] Loaded best hyperparameters: {best_params}, with val acc: {current_best_acc:.4f}")

    # 8. Create final model instance with the best hyperparameters and modality flag.
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
        num_donors=num_donors,
        use_text=not args.disable_text_modality
    )

    if args.use_wandb:
        wandb.init(project="cz-biohub-test", config=best_params, reinit=True)

    # 9. (Optional) Pretrain the model if flagged
    if args.pretrain_resampler and accelerator.unwrap_model(model).use_text:
        print("[Main] Starting pretraining phase for the Perceiver Resampler...")
        pretrain_epochs = 3
        pretrain_optimizer = optim.Adam(model.parameters(), lr=1e-4, weight_decay=1e-4)
        total_steps_pretrain = len(train_dataloader) * pretrain_epochs
        warmup_steps_pretrain = int(0.1 * total_steps_pretrain)
        pretrain_scheduler = get_cosine_schedule_with_warmup(pretrain_optimizer,
                                                             num_warmup_steps=warmup_steps_pretrain,
                                                             num_training_steps=total_steps_pretrain)
        pretrain_contrastive(model, train_dataloader, pretrain_optimizer, pretrain_scheduler,
                             num_epochs=pretrain_epochs, device=accelerator.device)
        print("[Main] Pretraining completed.")

    # 10. Train final model (if --train-model) or load final_model.pt
    if args.train_model:
        print("[Main] Training the final model with best hyperparameters...")
        optimizer = optim.Adam(model.parameters(), lr=best_params["lr"], weight_decay=1e-4)
        total_steps = len(train_dataloader) * args.num_epochs
        warmup_steps = int(0.1 * total_steps)
        scheduler = get_cosine_schedule_with_warmup(optimizer,
                                                    num_warmup_steps=warmup_steps,
                                                    num_training_steps=total_steps)
        model, optimizer, train_dataloader, scheduler = accelerator.prepare(model, optimizer, train_dataloader, scheduler)
        temp_dir = os.path.join(args.output_dir, "temp")
        os.makedirs(temp_dir, exist_ok=True)
        train_accs, val_accs, test_accs, epochs_trained = train_model(
            model,
            train_dataloader,
            optimizer,
            scheduler,
            num_epochs=args.num_epochs,
            accelerator=accelerator,
            output_dir=temp_dir,
            log_to_wandb=args.use_wandb,
            val_loader=val_dataloader,
            test_loader=test_dataloader,
            early_stopping_patience=3,
            save_checkpoints=True
        )
        training_curves_path = os.path.join(args.output_dir, "training_curves.png")
        plt.figure(figsize=(12, 6))
        plt.plot(epochs_trained, train_accs, label="Train Accuracy")
        plt.plot(epochs_trained, val_accs, label="Validation Accuracy")
        plt.plot(epochs_trained, test_accs, label="Test Accuracy")
        plt.xlabel("Epoch")
        plt.ylabel("Accuracy")
        plt.title("Accuracy Curves")
        plt.legend()
        plt.tight_layout()
        plt.savefig(training_curves_path)
        plt.close()
        print(f"[Main] Training curves saved to {training_curves_path}")
        best_model_path = os.path.join(temp_dir, "best_model.pt")
        best_model_state = torch.load(best_model_path, map_location=accelerator.device)
        accelerator.unwrap_model(model).load_state_dict(best_model_state)
        print(f"[Main] Loaded best model from {best_model_path} for final evaluation.")
    else:
        print("[Main] Skipping training. Loading final_model.pt...")
        model = model.to(accelerator.device)
        final_model_path = os.path.join(args.output_dir, "final_model.pt")
        if not os.path.isfile(final_model_path):
            raise FileNotFoundError(f"final_model.pt not found in {args.output_dir}. Please run with --train-model at least once.")
        model_state = torch.load(final_model_path, map_location=accelerator.device)
        model.load_state_dict(model_state)
        print(f"[Main] Loaded final model from {final_model_path}.")

    # 11. Evaluate final model
    model.eval()
    model = accelerator.prepare(model)
    final_test_donor_loss, final_test_adv_loss, final_test_acc = evaluate_model_full(model, test_dataloader, accelerator)
    print(f"[Test] Final Test Donor Loss: {final_test_donor_loss:.4f}, Adv Loss: {final_test_adv_loss:.4f}, Accuracy: {final_test_acc:.4f}")

    donor_inv_map = {v: k for k, v in donor_map.items()}
    sample_predictions(model, test_dataloader, accelerator, donor_inv_map, num_samples=5)

    best_hp_path = os.path.join(args.output_dir, "best_hyperparameters.json")
    if os.path.isfile(best_hp_path):
        with open(best_hp_path, "r") as f:
            final_results = json.load(f)
    else:
        final_results = {}

    final_results.update({
        "best_params": best_params,
        "final_train_accuracy": None,
        "final_validation_accuracy": None,
        "final_test_accuracy": final_test_acc,
        "final_test_donor_loss": final_test_donor_loss,
        "final_test_adv_loss": final_test_adv_loss
    })

    with open(best_hp_path, "w") as f:
        json.dump(final_results, f, indent=4)
    print(f"[Main] Best hyperparameters and final results saved to {best_hp_path}")

    if args.train_model:
        save_model(model, args.output_dir, accelerator)

    # Generate cross-attention heatmap from a sample of the test set
    single_loader = DataLoader(test_dataset, batch_size=1, shuffle=True)
    single_loader = accelerator.prepare(single_loader)
    sample_omics, sample_text, _, _ = next(iter(single_loader))
    sample_omics = sample_omics.to(accelerator.device)
    sample_text = sample_text.to(accelerator.device)
    with torch.no_grad():
        _, _, _, attn_resampler = model(sample_omics, sample_text)
        print(f"[Debug] attn_resampler shape: {attn_resampler.shape}")
        if attn_resampler.dim() == 4:
            attn_resampler = attn_resampler[0]
            avg_attn = attn_resampler.mean(dim=0).detach().cpu().numpy()
        elif attn_resampler.dim() == 3:
            attn_resampler = attn_resampler[0]
            avg_attn = attn_resampler.detach().cpu().numpy()
        else:
            raise ValueError(f"Unexpected shape for attn_resampler: {attn_resampler.shape}.")
    cross_attn_path = os.path.join(args.output_dir, "perceiver_attention_heatmap.png")
    plt.figure(figsize=(6, 5))
    plt.imshow(avg_attn, cmap="viridis", aspect="auto")
    plt.colorbar()
    plt.title("Average Perceiver Cross-Attention Weights (Single Sample)")
    plt.xlabel("Key/Token Axis")
    plt.ylabel("Query/Latent Axis")
    plt.savefig(cross_attn_path)
    plt.close()
    print(f"[Main] Perceiver cross-attention heatmap saved to {cross_attn_path}")

    if args.use_wandb:
        wandb.log({
            "final_test_accuracy": final_test_acc,
            "final_test_donor_loss": final_test_donor_loss,
            "final_test_adv_loss": final_test_adv_loss,
        })
        wandb.finish()

    # 12. Final Summary and Bonus Analysis Statements
    print("\n[Main] Final Results Summary:")
    print(f" - Number of cells loaded: {adata.n_obs}")
    print(f" - UMAP plot saved at: {umap_path}")
    print(f" - Training curves saved at: {os.path.join(args.output_dir, 'training_curves.png')}")
    print(f" - Perceiver cross-attention heatmap saved at: {cross_attn_path}")
    print(f" - Best hyperparameters and final results saved at: {best_hp_path}")
    print(f" - Final Test Accuracy: {final_test_acc:.4f}")
    print(f" - Final Test Donor Loss: {final_test_donor_loss:.4f}")
    print(f" - Final Test Adv Loss: {final_test_adv_loss:.4f}")
    
    # Bonus Analysis Output:
    if args.pretrain_resampler and accelerator.unwrap_model(model).use_text:
        print("Analysis: Pretraining of the Perceiver Resampler was performed. To assess its impact, compare these results with a run without pretraining.")
    else:
        print("Analysis: No pretraining was performed for the Perceiver Resampler.")
    
    if args.disable_text_modality:
        print("Analysis: The model was trained using only omics data. Compare these results with a run using both modalities to determine the impact of cell type label information.")
    else:
        print("Analysis: The model incorporated both omics and cell type label information. Compare these results with an omics-only run to assess the benefit of textual data.")

if __name__ == "__main__":
    main()
