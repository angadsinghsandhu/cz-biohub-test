#!/usr/bin/env python3
import os
import json
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

# Open Census and query for CNS cells; load cell_type, donor_id, and sex information.
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

# Generate UMAP plot of Geneformer embeddings.
sc.pp.neighbors(adata, use_rep="geneformer")
sc.tl.umap(adata)
sc.pl.umap(adata, color="cell_type", title="UMAP of Geneformer Embeddings", show=False)
umap_path = os.path.join("output", "umap_geneformer.png")
plt.savefig(umap_path)
plt.close()

# ------------------------------
# 2. Textual Embedding Extraction
# ------------------------------
# Load pretrained tokenizer and model (using BioBERT) from local path.
# (Alternatively, you could experiment with a deeper model like SciBERT)
model_path = "/home/asandhu9/cz-biohub-test/models/dmis-lab/biobert/1.1"
tokenizer = AutoTokenizer.from_pretrained(model_path, local_files_only=True)
text_model = AutoModel.from_pretrained(model_path, local_files_only=True).to(device)
text_model.eval()

def get_text_embedding(cell_type_label):
    """
    Given a cell type label, tokenize and extract the [CLS] token embedding.
    """
    inputs = tokenizer(cell_type_label, return_tensors="pt")
    inputs = {k: v.to(device) for k, v in inputs.items()}
    with torch.no_grad():
        outputs = text_model(**inputs)
    # Return the [CLS] token embedding (first token)
    return outputs.last_hidden_state[:, 0, :].squeeze(0).cpu().numpy()

# Compute textual embeddings for each cell type label.
textual_embeddings = []
for label in adata.obs["cell_type"]:
    emb = get_text_embedding(label)
    textual_embeddings.append(emb)
textual_embeddings = np.stack(textual_embeddings, axis=0)
print("Textual embeddings shape:", textual_embeddings.shape)

# ------------------------------
# 3. Model Architecture with Tokenization & Deeper Fusion
# ------------------------------
class GradientReversal(torch.autograd.Function):
    """
    Gradient reversal layer for adversarial training.
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
    Multi-head cross-attention block using batch_first mode.
    Query: tokens from one modality.
    Key/Value: tokens from the other modality.
    """
    def __init__(self, d_model, n_heads, dropout=0.1, batch_first=True):
        super(CrossAttentionBlock, self).__init__()
        print(f"Initializing CrossAttentionBlock with embed_dim: {d_model}, num_heads: {n_heads}")
        self.attention = nn.MultiheadAttention(embed_dim=d_model, num_heads=n_heads, dropout=dropout, batch_first=batch_first)
        self.norm = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, query, key, value):
        attn_output, attn_weights = self.attention(query, key, value)
        output = self.norm(query + self.dropout(attn_output))
        return output, attn_weights

class PerceiverResampler(nn.Module):
    """
    Perceiver Resampler that uses latent vectors to summarize input features.
    Now adapted to batch_first inputs.
    """
    def __init__(self, input_dim, num_latents, latent_dim, n_heads, dropout=0.1, batch_first=True):
        super(PerceiverResampler, self).__init__()
        self.latents = nn.Parameter(torch.randn(num_latents, latent_dim))
        self.cross_attn = CrossAttentionBlock(d_model=latent_dim, n_heads=n_heads, dropout=dropout, batch_first=batch_first)
        self.input_proj = nn.Linear(input_dim, latent_dim) if input_dim != latent_dim else nn.Identity()
    
    def forward(self, x):
        # x: (batch, input_dim)
        x_proj = self.input_proj(x)  # (batch, latent_dim)
        # Expand latents to match batch size: (batch, num_latents, latent_dim)
        batch = x.size(0)
        latents_expanded = self.latents.unsqueeze(0).expand(batch, -1, -1)
        # Use the projected x as key/value by unsqueezing to (batch, 1, latent_dim)
        x_proj_unsq = x_proj.unsqueeze(1)
        out, attn_weights = self.cross_attn(latents_expanded, x_proj_unsq, x_proj_unsq)
        # Aggregate latent tokens by mean pooling
        out = out.mean(dim=1)
        return out, attn_weights

class MultiModalModel(nn.Module):
    """
    Multimodal model integrating Geneformer (omics) and textual embeddings.
    Uses tokenization of the flat embeddings to yield multiple tokens per cell.
    Also includes a feedforward block after cross-attention and an adversarial branch.
    """
    def __init__(self, 
                 omics_dim=512, 
                 text_dim=768, 
                 fusion_dim=256, 
                 n_heads=8, 
                 num_tokens=8, 
                 num_latents=32, 
                 latent_dim=128, 
                 dropout=0.1,
                 adv_lambda=0.1,
                 num_donors=10):
        super(MultiModalModel, self).__init__()
        self.num_tokens = num_tokens
        # Determine token dimensions (ensure divisibility)
        self.omics_token_dim = omics_dim // num_tokens  # 512/8 = 64
        self.text_token_dim = text_dim // num_tokens      # 768/8 = 96
        # Project each token into the shared fusion space
        self.omics_token_proj = nn.Linear(self.omics_token_dim, fusion_dim)
        self.text_token_proj = nn.Linear(self.text_token_dim, fusion_dim)
        
        # Cross-attention block: let omics tokens attend to text tokens.
        self.cross_attn = CrossAttentionBlock(d_model=fusion_dim, n_heads=n_heads, dropout=dropout, batch_first=True)
        # Add an extra feedforward network for further processing.
        self.ffn = nn.Sequential(
            nn.Linear(fusion_dim, fusion_dim),
            nn.ReLU(),
            nn.Linear(fusion_dim, fusion_dim)
        )
        # After fusion, aggregate tokens by mean pooling.
        # Pass aggregated representation to the Perceiver Resampler.
        self.resampler = PerceiverResampler(input_dim=fusion_dim, 
                                            num_latents=num_latents, 
                                            latent_dim=latent_dim, 
                                            n_heads=n_heads, 
                                            dropout=dropout,
                                            batch_first=True)
        # Classifier for donor ID
        self.classifier = nn.Sequential(
            nn.Linear(latent_dim, latent_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(latent_dim, latent_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(latent_dim // 2, num_donors)
        )
        # Adversarial classifier for sex variable
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
        # omics: (batch, 512), text: (batch, 768)
        batch = omics.size(0)
        # Reshape into tokens
        omics_tokens = omics.view(batch, self.num_tokens, self.omics_token_dim)   # (batch, num_tokens, 64)
        text_tokens = text.view(batch, self.num_tokens, self.text_token_dim)      # (batch, num_tokens, 96)
        # Project tokens into fusion space
        omics_tokens = self.omics_token_proj(omics_tokens)  # (batch, num_tokens, fusion_dim)
        text_tokens = self.text_token_proj(text_tokens)     # (batch, num_tokens, fusion_dim)
        # Cross-attention: omics tokens attend to text tokens
        fused_tokens, attn_weights = self.cross_attn(omics_tokens, text_tokens, text_tokens)
        # Process with feedforward network
        fused_tokens = self.ffn(fused_tokens)
        # Aggregate tokens by averaging over the token dimension
        fused_feat = fused_tokens.mean(dim=1)  # (batch, fusion_dim)
        # Pass through the Perceiver Resampler
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
# textual_embeddings already computed above.

dataset = SingleCellDataset(omics_data, textual_embeddings, donor_labels, sex_labels)
batch_size = 64
dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=4)

# ------------------------------
# 5. Training Loop with Enhanced Training Strategy
# ------------------------------
def train_model(model, dataloader, optimizer, scheduler, num_epochs=50, log_to_wandb=False):
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
        
        for omics, text, donor_labels, sex_labels in tqdm(dataloader, desc=f"Epoch {epoch+1}/{num_epochs}"):
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
            scheduler.step()  # Step the LR scheduler after each batch
            
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
        
        print(f"Epoch {epoch+1}/{num_epochs} - Donor Loss: {epoch_donor_loss:.4f} - Adv Loss: {epoch_adv_loss:.4f} - Accuracy: {epoch_acc:.4f}")
        
        if log_to_wandb:
            wandb.log({"epoch": epoch+1,
                       "donor_loss": epoch_donor_loss,
                       "adv_loss": epoch_adv_loss,
                       "accuracy": epoch_acc})
    
    return donor_losses, adv_losses, accuracies

# ------------------------------
# 6. Hyperparameter Optimization with Optuna
# ------------------------------
def objective(trial):
    # Use best hyperparameter search ranges; these will later be updated only if better ones are found.
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
                                     num_donors=num_donors).to(device)
    
    optimizer_instance = optim.Adam(model_instance.parameters(), lr=lr, weight_decay=1e-4)
    total_steps = len(dataloader) * 5  # 5 epochs for each trial
    warmup_steps = int(0.1 * total_steps)
    scheduler_instance = get_cosine_schedule_with_warmup(optimizer_instance,
                                                         num_warmup_steps=warmup_steps,
                                                         num_training_steps=total_steps)
    
    _, _, accuracies = train_model(model_instance, dataloader, optimizer_instance, scheduler_instance, num_epochs=5, log_to_wandb=False)
    final_acc = accuracies[-1]
    return -final_acc  # Minimization objective

study = optuna.create_study()
study.optimize(objective, n_trials=10)
print("Best hyperparameters from Optuna:", study.best_params)

# ------------------------------
# 7. Final Training and Logging with wandb and Saving Hyperparameters
# ------------------------------
# Initialize wandb for the final run.
best_params = study.best_params
wandb.init(project="cz-biohub-test", config=best_params, reinit=True)

num_donors = len(donor_map)
model_instance = MultiModalModel(omics_dim=omics_data.shape[1],
                                 text_dim=textual_embeddings.shape[1],
                                 fusion_dim=256,
                                 n_heads=8,
                                 num_tokens=8,
                                 num_latents=best_params["num_latents"],
                                 latent_dim=best_params["latent_dim"],
                                 dropout=best_params["dropout"],
                                 adv_lambda=best_params["adv_lambda"],
                                 num_donors=num_donors).to(device)
optimizer_instance = optim.Adam(model_instance.parameters(), lr=best_params["lr"], weight_decay=1e-4)
total_steps = len(dataloader) * 50  # 50 epochs final training
warmup_steps = int(0.1 * total_steps)
scheduler_instance = get_cosine_schedule_with_warmup(optimizer_instance,
                                                     num_warmup_steps=warmup_steps,
                                                     num_training_steps=total_steps)

donor_losses, adv_losses, accuracies = train_model(model_instance, dataloader, optimizer_instance, scheduler_instance, num_epochs=50, log_to_wandb=True)

# Save training curves and final accuracy figures.
epochs = range(1, 51)
plt.figure(figsize=(18, 5))
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
training_curves_path = os.path.join("output", "training_curves.png")
plt.savefig(training_curves_path)
plt.close()

# Bonus: Visualize cross-attention weights from a sample batch.
# Now that we have multiple tokens, average the attention weights over the batch to produce a heatmap.
omics_sample, text_sample, _, _ = next(iter(dataloader))
omics_sample = omics_sample.to(device)
text_sample = text_sample.to(device)
_, _, attn_weights, _ = model_instance(omics_sample, text_sample)
# attn_weights shape: (batch, query_tokens, key_tokens) from the first cross-attn block
avg_attn = attn_weights.mean(dim=0).detach().cpu().numpy()  # shape: (num_tokens, num_tokens)
plt.figure(figsize=(6, 4))
plt.imshow(avg_attn, cmap="viridis", aspect="auto")
plt.colorbar()
plt.title("Average Cross-Attention Weights")
plt.xlabel("Key Token")
plt.ylabel("Query Token")
cross_attn_path = os.path.join("output", "cross_attention_heatmap.png")
plt.savefig(cross_attn_path)
plt.close()

# ------------------------------
# 8. Save Best Hyperparameters and Final Accuracy to JSON (only update if improved)
# ------------------------------
best_hyperparams_file = os.path.join("output", "best_hyperparameters.json")
final_results = {
    "best_params": best_params,
    "final_accuracy": accuracies[-1]
}
if os.path.exists(best_hyperparams_file):
    with open(best_hyperparams_file, "r") as f:
        existing = json.load(f)
    if final_results["final_accuracy"] > existing.get("final_accuracy", 0):
        with open(best_hyperparams_file, "w") as f:
            json.dump(final_results, f, indent=4)
else:
    with open(best_hyperparams_file, "w") as f:
        json.dump(final_results, f, indent=4)

# ------------------------------
# 9. Final Deliverable Print Statements
# ------------------------------
final_accuracy = accuracies[-1]
print("\nFinal Results Summary:")
print(f"- Number of cells loaded: {adata.n_obs}")
print(f"- UMAP plot saved at: {umap_path}")
print(f"- Training curves saved at: {training_curves_path}")
print(f"- Cross-attention heatmap saved at: {cross_attn_path}")
print(f"- Best hyperparameters saved at: {best_hyperparams_file}")
print(f"- Final donor ID prediction accuracy: {final_accuracy:.4f}")

wandb.log({"final_accuracy": final_accuracy})
wandb.finish()
