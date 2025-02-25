#!/usr/bin/env python3
import os
import numpy as np
import scanpy as sc
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, AutoModel
import optuna
import matplotlib.pyplot as plt
import warnings
import pandas as pd
import cellxgene_census

# Set device to GPU if available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Create output directory if not exists
os.makedirs("output", exist_ok=True)

# ------------------------------
# 1. Data Loading and UMAP Visualization
# ------------------------------
warnings.filterwarnings("ignore")

# Configuration for Census query
census_version = "2023-12-15"
organism = "homo_sapiens"
measurement_name = "RNA"
emb_names = ["geneformer"]

# Open Census and query for CNS cells; we load the cell type, donor_id, and sex information
with cellxgene_census.open_soma(census_version=census_version) as census:
    adata = cellxgene_census.get_anndata(
        census,
        organism=organism,
        measurement_name=measurement_name,
        obs_value_filter="tissue_general == 'central nervous system'",
        obs_column_names=["cell_type", "donor_id", "sex"],
        obs_embeddings=emb_names,
    )
print("Number of cells loaded:", adata.n_obs)

# Generate UMAP plot of geneformer embeddings
sc.pp.neighbors(adata, use_rep="geneformer")
sc.tl.umap(adata)
# Instead of showing interactively, save the UMAP plot to the output directory.
sc.pl.umap(adata, color="cell_type", title="UMAP of Geneformer Embeddings", show=False)
plt.savefig(os.path.join("output", "umap_geneformer.png"))
plt.close()

# ------------------------------
# 2. Textual Embedding Extraction
# ------------------------------
# Load pretrained tokenizer and model locally from the "models" directory.
model_path = "/home/asandhu9/cz-biohub-test/models/dmis-lab/biobert/1.1"
tokenizer = AutoTokenizer.from_pretrained(model_path, local_files_only=True)
text_model = AutoModel.from_pretrained(model_path, local_files_only=True).to(device)
text_model.eval()

def get_text_embedding(cell_type_label):
    """
    Given a cell type label, tokenize and extract the embedding for the [CLS] token.
    """
    inputs = tokenizer(cell_type_label, return_tensors="pt")
    inputs = {k: v.to(device) for k, v in inputs.items()}
    with torch.no_grad():
        outputs = text_model(**inputs)
    # Extract [CLS] token embedding (first token)
    return outputs.last_hidden_state[:, 0, :].squeeze(0).cpu().numpy()

# Create a mapping from cell index to textual embedding
textual_embeddings = []
for label in adata.obs["cell_type"]:
    emb = get_text_embedding(label)
    textual_embeddings.append(emb)
textual_embeddings = np.stack(textual_embeddings, axis=0)
print("Textual embeddings shape:", textual_embeddings.shape)

# ------------------------------
# 3. Model Architecture
# ------------------------------
class GradientReversal(torch.autograd.Function):
    """
    Implements a gradient reversal layer for adversarial training.
    """
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
    """
    Multi-head cross-attention block.
    Query: geneformer embeddings.
    Key/Value: textual embeddings.
    """
    def __init__(self, d_model, n_heads, dropout=0.1):
        super(CrossAttentionBlock, self).__init__()
        self.attention = nn.MultiheadAttention(embed_dim=d_model, num_heads=n_heads, dropout=dropout)
        self.norm = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, query, key, value):
        # query shape: (seq_len, batch, d_model)
        # key, value shape: (seq_len, batch, d_model)
        attn_output, attn_weights = self.attention(query, key, value)
        output = self.norm(query + self.dropout(attn_output))
        return output, attn_weights

class PerceiverResampler(nn.Module):
    """
    Perceiver Resampler that uses latent vectors to summarize high-dimensional input.
    """
    def __init__(self, input_dim, num_latents, latent_dim, n_heads, dropout=0.1):
        super(PerceiverResampler, self).__init__()
        self.latents = nn.Parameter(torch.randn(num_latents, latent_dim))
        self.cross_attn = CrossAttentionBlock(d_model=latent_dim, n_heads=n_heads, dropout=dropout)
        self.input_proj = nn.Linear(input_dim, latent_dim) if input_dim != latent_dim else nn.Identity()
    
    def forward(self, x):
        # x: (batch, input_dim)
        x_proj = self.input_proj(x)  # (batch, latent_dim)
        x_proj = x_proj.unsqueeze(0)  # shape: (1, batch, latent_dim)
        batch_size = x.size(0)
        latents_expanded = self.latents.unsqueeze(1).expand(-1, batch_size, -1)
        out, attn_weights = self.cross_attn(latents_expanded, x_proj, x_proj)
        out = out.mean(dim=0)
        return out, attn_weights

class MultiModalModel(nn.Module):
    """
    Multimodal model integrating geneformer (omics) embeddings and textual embeddings.
    Also includes an adversarial branch to control for sex.
    """
    def __init__(self, 
                 omics_dim=512, 
                 text_dim=768, 
                 fusion_dim=256, 
                 n_heads=8, 
                 num_latents=32, 
                 latent_dim=128, 
                 dropout=0.1,
                 adv_lambda=0.1,
                 num_donors=10):
        super(MultiModalModel, self).__init__()
        self.omics_proj = nn.Linear(omics_dim, fusion_dim)
        self.text_proj = nn.Linear(text_dim, fusion_dim)
        self.cross_attn = CrossAttentionBlock(d_model=fusion_dim, n_heads=n_heads, dropout=dropout)
        self.resampler = PerceiverResampler(input_dim=fusion_dim, 
                                            num_latents=num_latents, 
                                            latent_dim=latent_dim, 
                                            n_heads=n_heads, 
                                            dropout=dropout)
        self.classifier = nn.Sequential(
            nn.Linear(latent_dim, latent_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(latent_dim // 2, num_donors)
        )
        self.adv_classifier = nn.Sequential(
            nn.Linear(latent_dim, latent_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(latent_dim // 2, 2)
        )
        self.adv_lambda = adv_lambda
        
    def forward(self, omics, text):
        omics_feat = self.omics_proj(omics)  # (batch, fusion_dim)
        text_feat = self.text_proj(text)     # (batch, fusion_dim)
        omics_feat = omics_feat.unsqueeze(0)   # (1, batch, fusion_dim)
        text_feat = text_feat.unsqueeze(0)     # (1, batch, fusion_dim)
        fused_feat, attn_weights = self.cross_attn(omics_feat, text_feat, text_feat)
        fused_feat = fused_feat.squeeze(0)
        latent, attn_resampler = self.resampler(fused_feat)
        donor_logits = self.classifier(latent)
        adv_input = grad_reverse(latent, self.adv_lambda)
        sex_logits = self.adv_classifier(adv_input)
        return donor_logits, sex_logits, attn_weights, attn_resampler

# ------------------------------
# 4. Dataset and DataLoader Preparation
# ------------------------------
class SingleCellDataset(Dataset):
    def __init__(self, omics_data, text_data, donor_labels, sex_labels):
        self.omics = torch.tensor(omics_data, dtype=torch.float32)
        self.text = torch.tensor(text_data, dtype=torch.float32)
        self.donor_labels = torch.tensor(donor_labels, dtype=torch.long)
        self.sex_labels = torch.tensor(sex_labels, dtype=torch.long)
        
    def __len__(self):
        return self.omics.shape[0]
    
    def __getitem__(self, idx):
        return self.omics[idx], self.text[idx], self.donor_labels[idx], self.sex_labels[idx]

# Create donor and sex label mappings from adata.obs.
donor_map = {donor: i for i, donor in enumerate(pd.unique(adata.obs["donor_id"]))}
sex_map = {"male": 0, "female": 1}
donor_labels = [donor_map[d] for d in adata.obs["donor_id"]]
sex_labels = [sex_map[s.lower()] if s.lower() in sex_map else 0 for s in adata.obs["sex"]]

omics_data = adata.obsm["geneformer"]
# textual_embeddings is already computed

dataset = SingleCellDataset(omics_data, textual_embeddings, donor_labels, sex_labels)
batch_size = 64
dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=4)

# ------------------------------
# 5. Training Loop with Hyperparameter Optimization
# ------------------------------
def train_model(model, dataloader, optimizer, scheduler, num_epochs=20):
    model.train()
    donor_criterion = nn.CrossEntropyLoss()
    adv_criterion = nn.CrossEntropyLoss()
    
    donor_losses = []
    adv_losses = []
    accuracies = []
    
    for epoch in range(num_epochs):
        running_donor_loss = 0.0
        running_adv_loss = 0.0
        correct = 0
        total = 0
        
        for omics, text, donor_labels, sex_labels in dataloader:
            omics = omics.to(device)
            text = text.to(device)
            donor_labels = donor_labels.to(device)
            sex_labels = sex_labels.to(device)
            
            optimizer.zero_grad()
            donor_logits, sex_logits, attn_weights, attn_resampler = model(omics, text)
            loss_donor = donor_criterion(donor_logits, donor_labels)
            loss_adv = adv_criterion(sex_logits, sex_labels)
            loss = loss_donor + loss_adv
            loss.backward()
            optimizer.step()
            
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
        
        scheduler.step()
        print(f"Epoch {epoch+1}/{num_epochs} - Donor Loss: {epoch_donor_loss:.4f} - Adv Loss: {epoch_adv_loss:.4f} - Accuracy: {epoch_acc:.4f}")
    
    return donor_losses, adv_losses, accuracies

# ------------------------------
# 6. Hyperparameter Optimization with Optuna
# ------------------------------
def objective(trial):
    lr = trial.suggest_loguniform("lr", 1e-5, 1e-2)
    dropout = trial.suggest_uniform("dropout", 0.1, 0.5)
    num_latents = trial.suggest_int("num_latents", 16, 64)
    latent_dim = trial.suggest_int("latent_dim", 64, 256)
    adv_lambda = trial.suggest_uniform("adv_lambda", 0.01, 0.5)
    
    num_donors = len(donor_map)
    model_instance = MultiModalModel(omics_dim=omics_data.shape[1],
                                     text_dim=textual_embeddings.shape[1],
                                     fusion_dim=256,
                                     n_heads=8,
                                     num_latents=num_latents,
                                     latent_dim=latent_dim,
                                     dropout=dropout,
                                     adv_lambda=adv_lambda,
                                     num_donors=num_donors).to(device)
    
    optimizer_instance = optim.Adam(model_instance.parameters(), lr=lr)
    scheduler_instance = optim.lr_scheduler.CosineAnnealingLR(optimizer_instance, T_max=20)
    
    _, _, accuracies = train_model(model_instance, dataloader, optimizer_instance, scheduler_instance, num_epochs=5)
    final_acc = accuracies[-1]
    return -final_acc  # Optuna minimizes the objective

study = optuna.create_study()
study.optimize(objective, n_trials=10)
print("Best hyperparameters:", study.best_params)

# ------------------------------
# 7. Final Training and Visualization
# ------------------------------
best_params = study.best_params
num_donors = len(donor_map)
model_instance = MultiModalModel(omics_dim=omics_data.shape[1],
                                 text_dim=textual_embeddings.shape[1],
                                 fusion_dim=256,
                                 n_heads=8,
                                 num_latents=best_params["num_latents"],
                                 latent_dim=best_params["latent_dim"],
                                 dropout=best_params["dropout"],
                                 adv_lambda=best_params["adv_lambda"],
                                 num_donors=num_donors).to(device)
optimizer_instance = optim.Adam(model_instance.parameters(), lr=best_params["lr"])
scheduler_instance = optim.lr_scheduler.CosineAnnealingLR(optimizer_instance, T_max=20)

donor_losses, adv_losses, accuracies = train_model(model_instance, dataloader, optimizer_instance, scheduler_instance, num_epochs=20)

# Plot loss curves and accuracy
epochs = range(1, 21)
plt.figure(figsize=(12, 4))
plt.subplot(1, 3, 1)
plt.plot(epochs, donor_losses, label="Donor Loss")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.title("Donor Loss")
plt.legend()

plt.subplot(1, 3, 2)
plt.plot(epochs, adv_losses, label="Adversarial Loss", color="orange")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.title("Adversarial Loss")
plt.legend()

plt.subplot(1, 3, 3)
plt.plot(epochs, accuracies, label="Accuracy", color="green")
plt.xlabel("Epoch")
plt.ylabel("Accuracy")
plt.title("Donor ID Prediction Accuracy")
plt.legend()
plt.tight_layout()
plt.savefig(os.path.join("output", "training_curves.png"))
plt.close()

# Bonus: Visualize cross-attention weights from a sample batch
omics_sample, text_sample, _, _ = next(iter(dataloader))
omics_sample = omics_sample.to(device)
text_sample = text_sample.to(device)
_, _, attn_weights, _ = model_instance(omics_sample, text_sample)
attn_weights_np = attn_weights.detach().cpu().numpy().squeeze(0)
plt.figure(figsize=(6, 4))
plt.imshow(attn_weights_np, cmap="viridis", aspect="auto")
plt.colorbar()
plt.title("Cross-Attention Weights Heatmap")
plt.xlabel("Key Position")
plt.ylabel("Query Position")
plt.savefig(os.path.join("output", "cross_attention_heatmap.png"))
plt.close()

# ------------------------------
# 8. Results Summary and Bonus Analysis
# ------------------------------
print("\nResults Summary:")
print("- UMAP plot saved as 'output/umap_geneformer.png'")
print("- Training curves saved as 'output/training_curves.png'")
print("- Cross-attention heatmap saved as 'output/cross_attention_heatmap.png'")
print("- Final donor ID prediction accuracy: {:.4f}".format(accuracies[-1]))

# End of script.
