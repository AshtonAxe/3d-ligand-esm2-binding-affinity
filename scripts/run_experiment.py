import pandas as pd
import numpy as np
import torch
import torch_geometric
from torch_geometric.data import Data
from torch.utils.data import Dataset
from torch_geometric.loader import DataLoader as PyGDataLoader
import torch.utils.data
import ast
import rdkit.Geometry as rdGeom
from sklearn.model_selection import train_test_split
from torch_geometric.nn import MessagePassing, global_mean_pool
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt

# Load DataFrame
df = pd.read_csv('bindingdb_esm2_embedded_50k.csv')
invalid_indices = [5479] + list(range(19487, 19999))
df = df.drop(index=invalid_indices).reset_index(drop=True)

# Simplified type mapping for metals
type_to_onehot = {
    "nonmetal": [1, 0, 0, 0],
    "metal": [0, 1, 0, 0],
    "metalloid": [0, 0, 1, 0],
    "noble_gas": [0, 0, 0, 1]
}
atomic_df = pd.read_csv('Atomic_Property_Lookup_Table.csv')
# Build lookup dict from DataFrame (assuming atomic_df exists)
property_lookup = {}
for row in atomic_df.itertuples():
    z = row.Z
    metal_type = row.metal_type

    # Coalesce complex metal types to a single "metal" category
    if metal_type in ["alkali", "alkaline", "transition", "post_transition"]:
        metal_type = "metal"

    property_lookup[z] = {
        "electronegativity": row.electronegativity if pd.notnull(row.electronegativity) else 0.0,
        "valence_electrons": row.valence_electrons,
        "atomic_mass": row.atomic_mass,
        "covalent_radius": row.covalent_radius,
        "metal_onehot": type_to_onehot.get(metal_type, [0, 0, 0, 0])
    }

def ligand_to_pyg_data(z, pos, in_channels=100):
    # Convert inputs to tensors
    try:
        z = torch.tensor(z, dtype=torch.long)
        pos = torch.tensor(pos, dtype=torch.float32)
    except Exception as e:
        print(f"ERROR converting z or pos to tensor: {e}")
        raise

    # Validate shapes
    if len(z.shape) != 1:
        raise ValueError(f"z should be 1D, got shape {z.shape}")
    if pos.shape[0] != z.shape[0]:
        raise ValueError(f"pos and z length mismatch: pos.shape={pos.shape}, z.shape={z.shape}")
    if pos.shape[1] != 3:
        raise ValueError(f"pos should be N x 3, got {pos.shape}")

    # Enhanced atom-level features using atomic number + property lookup
    atom_features = []
    for i, atomic_num in enumerate(z):
        atomic_num = int(atomic_num)
        atomic_onehot = [0] * in_channels
        if 1 <= atomic_num <= in_channels:
            atomic_onehot[atomic_num - 1] = 1
        else:
            print(f"[WARNING] atomic_num {atomic_num} at index {i} is out of range (1-{in_channels})")

        props = property_lookup.get(atomic_num, {
            "electronegativity": 0.0,
            "valence_electrons": 0,
            "atomic_mass": 0.0,
            "covalent_radius": 0.0,
            "metal_onehot": [0, 0, 0, 0]
        })

        feat = (
            atomic_onehot +
            [props["electronegativity"]] +
            [props["valence_electrons"]] +
            [props["atomic_mass"]] +
            [props["covalent_radius"]] +
            props["metal_onehot"]
        )
        atom_features.append(feat)

    x = torch.tensor(atom_features, dtype=torch.float32)

    # Edge computation (all atom pairs)
    try:
        edge_index = torch.combinations(torch.arange(len(z)), r=2).t().to(torch.long)
        edge_attr = torch.norm(pos[edge_index[0]] - pos[edge_index[1]], dim=-1).unsqueeze(-1)
    except Exception as e:
        print(f"ERROR computing edge_index or edge_attr: {e}")
        raise

    # Create PyG Data object
    try:
        data = Data(x=x, pos=pos, edge_index=edge_index, edge_attr=edge_attr)
    except Exception as e:
        print(f"ERROR creating PyG Data object: {e}")
        raise

    return data

# Custom dataset class
class ProteinLigandDataset(Dataset):
    def __init__(self, df):
        self.df = df
        self.in_channels = 100
        valid_indices = []

        for idx in range(len(df)):
            row = df.iloc[idx]

            # Parse z, pos, esm2 if needed
            z = ast.literal_eval(row['z']) if isinstance(row['z'], str) else row['z']
            pos = ast.literal_eval(row['pos']) if isinstance(row['pos'], str) else row['pos']
            esm2 = ast.literal_eval(row['esm2']) if isinstance(row['esm2'], str) else row['esm2']
            esm2 = np.array(esm2) if isinstance(esm2, list) else esm2

            # Validity check
            if (
                isinstance(esm2, np.ndarray)
                and esm2.shape == (1280,)
                and np.all(np.isfinite(esm2))
            ):
                valid_indices.append(idx)
                max_atomic_num = max(z) if z else 1
                self.in_channels = min(max(max_atomic_num, self.in_channels), 100)
            else:
                print(f"Invalid esm2 at index {idx}: shape={getattr(esm2, 'shape', 'N/A')}")

        self.valid_df = df.iloc[valid_indices].reset_index(drop=True)
        print(f"Valid samples: {len(self.valid_df)}/{len(df)}")

    def __len__(self):
        return len(self.valid_df)

    def __getitem__(self, idx):
        row = self.valid_df.iloc[idx]
        z = ast.literal_eval(row['z']) if isinstance(row['z'], str) else row['z']
        pos = ast.literal_eval(row['pos']) if isinstance(row['pos'], str) else row['pos']
        esm2 = ast.literal_eval(row['esm2']) if isinstance(row['esm2'], str) else row['esm2']
        esm2 = np.array(esm2) if isinstance(esm2, list) else esm2
        # print(f"[DEBUG] calling ligand_to_pyg_data for idx={idx}")
        ligand_data = ligand_to_pyg_data(z, pos, in_channels=100)
        protein_emb = torch.tensor(esm2, dtype=torch.float32)
        pKi = torch.tensor(row['pKi'], dtype=torch.float32)
        return {
            'ligand': ligand_data,
            'protein_emb': protein_emb,
            'pKi': pKi
        }

# Custom collate function
def custom_collate(batch):
    ligand_data = [item['ligand'] for item in batch]
    protein_emb_list = []
    for item in batch:
        if item['protein_emb'].shape[0] == 1280:
            protein_emb_list.append(item['protein_emb'])
        else:
            print(f"Skipping invalid protein_emb: shape={item['protein_emb'].shape}")
    if not protein_emb_list:
        raise ValueError("No valid protein_emb in batch")
    protein_emb = torch.stack(protein_emb_list)
    pKi = torch.stack([item['pKi'] for item in batch])
    ligand_batch = torch_geometric.data.Batch.from_data_list(ligand_data)
    # print(f"custom_collate: ligand_batch.x.shape={ligand_batch.x.shape}, protein_emb.shape={protein_emb.shape}")
    return {
        'ligand': ligand_batch,
        'protein_emb': protein_emb,
        'pKi': pKi
    }

valid_df = df

# Randomly shuffle and split into train/val/test (80/10/10)
train_df, temp_df = train_test_split(valid_df, test_size=0.2, random_state=42, shuffle=True)
val_df, test_df = train_test_split(temp_df, test_size=0.5, random_state=42, shuffle=True)

# Compute normalization from train only
mean_pKi = train_df['pKi'].mean()
std_pKi = train_df['pKi'].std()

# Normalize all sets using training stats
train_df['pKi'] = (train_df['pKi'] - mean_pKi) / std_pKi
val_df['pKi'] = (val_df['pKi'] - mean_pKi) / std_pKi
test_df['pKi'] = (test_df['pKi'] - mean_pKi) / std_pKi

train_dataset = ProteinLigandDataset(train_df)
val_dataset = ProteinLigandDataset(val_df)
test_dataset = ProteinLigandDataset(test_df)

# Create data loaders
train_loader = torch.utils.data.DataLoader(
    train_dataset, batch_size=16, shuffle=True, collate_fn=custom_collate
)
val_loader = torch.utils.data.DataLoader(
    val_dataset, batch_size=16, collate_fn=custom_collate
)
test_loader = torch.utils.data.DataLoader(
    test_dataset, batch_size=16, collate_fn=custom_collate
)

class GNN(MessagePassing):
    def __init__(self, in_channels, out_channels, num_rbf=32, cutoff=10.0, gamma=10.0):
        super().__init__(aggr='add')

        self.cutoff = cutoff
        self.gamma = gamma

        # Define RBF centers
        self.mu = nn.Parameter(torch.linspace(0, cutoff, num_rbf), requires_grad=False)
        self.lin_edge = nn.Linear(num_rbf, in_channels)

        # MLP for message update
        self.lin = nn.Linear(in_channels, out_channels)

    def rbf(self, dist):
        # dist: [num_edges, 1]
        return torch.exp(-self.gamma * (dist - self.mu) ** 2)  # shape: [num_edges, num_rbf]

    def forward(self, x, pos, edge_index, edge_attr):
        row, col = edge_index
        dist = torch.norm(pos[row] - pos[col], dim=1, keepdim=True)  # shape: [num_edges, 1]

        # RBF expansion of distance
        rbf_feat = self.rbf(dist)  # shape: [num_edges, num_rbf]
        edge_feat = self.lin_edge(rbf_feat)  # shape: [num_edges, in_channels]

        return self.propagate(edge_index, x=x, edge_attr=edge_feat)

    def message(self, x_j, edge_attr):
        return x_j * edge_attr  # shape: [num_edges, in_channels]

    def update(self, aggr_out):
        return F.relu(self.lin(aggr_out))

class LigandGNN(nn.Module):
    def __init__(self, in_channels, hidden_channels=128, out_channels=128):
        super().__init__()
        self.conv1 = GNN(hidden_channels, hidden_channels)
        self.conv2 = GNN(hidden_channels, hidden_channels)
        self.conv3 = GNN(hidden_channels, out_channels)
        self.lin = nn.Linear(in_channels, hidden_channels)

    def forward(self, data):
        x, pos, edge_index, edge_attr, batch = data.x, data.pos, data.edge_index, data.edge_attr, data.batch
        # print(f"LigandGNN input: x.shape={x.shape}") #LigandGNN input: x.shape=torch.Size([405, 100])
        x = self.lin(x)
        x = self.conv1(x, pos, edge_index, edge_attr)
        x = self.conv2(x, pos, edge_index, edge_attr)
        x = self.conv3(x, pos, edge_index, edge_attr)
        x = global_mean_pool(x, batch)
        return x

class ProteinLigandAffinityModel(nn.Module):
    def __init__(self, esm2_dim=1280, ligand_in_channels=100, embedding_dim=128, n_attention_heads=8):
        super().__init__()
        self.ligand_gnn = LigandGNN(ligand_in_channels, embedding_dim, embedding_dim)
        self.protein_projection = nn.Linear(esm2_dim, embedding_dim)
        self.cross_attention = nn.MultiheadAttention(
            embed_dim=embedding_dim,
            num_heads=n_attention_heads,
            batch_first=True
        )
        self.mlp = nn.Sequential(
            nn.Linear(embedding_dim * 2, 256),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(128, 1)
        )

    def forward(self, batch):
        ligand_batch = batch['ligand']  # PyG batch object
        ligand_emb = self.ligand_gnn(ligand_batch)  # shape: [num_nodes_total, 128]

        # Project protein to same dimension
        protein_emb = self.protein_projection(batch['protein_emb'])

        # Cross attention expects 3D tensors: [batch_size, seq_len, dim]
        protein_emb = protein_emb.unsqueeze(1)  # [batch_size, 1, 128]
        ligand_emb = ligand_emb.unsqueeze(1)    # [batch_size, 1, 128]

        attn_output, _ = self.cross_attention(
            query=protein_emb,
            key=ligand_emb,
            value=ligand_emb
        )

        combined_emb = torch.cat([protein_emb, attn_output], dim=-1).squeeze(1)  # [batch_size, 256]
        return self.mlp(combined_emb).squeeze(-1)  # [batch_size]

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = ProteinLigandAffinityModel(
    esm2_dim=1280, ligand_in_channels=108, embedding_dim=128
).to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
loss_fn = nn.MSELoss()

best_val_loss = float('inf')
patience = 7        # Number of epochs to wait after no improvement
patience_counter = 0
min_delta = 1e-4    # Minimum change in val loss to be considered an improvement
save_path = 'new_best_model.pt'

train_losses = []
val_losses = []
val_rmses = []
val_r2s = []

def compute_rmse(pred, target):
    return torch.sqrt(torch.mean((pred - target) ** 2)).item()

def compute_r2(pred, target):
    ss_res = torch.sum((target - pred) ** 2)
    ss_tot = torch.sum((target - torch.mean(target)) ** 2)
    r2 = 1 - ss_res / ss_tot
    return r2.item()

num_epochs = 150
for epoch in range(num_epochs):
    model.train()
    train_loss = 0
    train_batches = 0

    train_bar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs} [Train]", leave=False)

    for batch in train_bar:
        try:
            batch = {k: v.to(device) if isinstance(v, torch.Tensor) or isinstance(v, torch_geometric.data.Batch) else v for k, v in batch.items()}
            batch['ligand'] = batch['ligand'].to(device)
            optimizer.zero_grad()
            pred = model(batch)
            loss = loss_fn(pred, batch['pKi'])
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
            train_batches += 1
            train_bar.set_postfix(batch_loss=loss.item())
        except Exception as e:
            print(f"[SKIPPED BATCH] due to error: {e}")
            continue

    avg_train_loss = train_loss / max(train_batches, 1)
    train_bar.set_postfix(avg_train_loss=avg_train_loss)

    model.eval()
    val_loss = 0
    val_batches = 0

    val_bar = tqdm(val_loader, desc=f"Epoch {epoch+1}/{num_epochs} [Val]", leave=False)

    all_preds = []
    all_targets = []
    with torch.no_grad():
        for batch in val_bar:
            try:
                batch = {k: v.to(device) if isinstance(v, torch.Tensor) or isinstance(v, torch_geometric.data.Batch) else v for k, v in batch.items()}
                batch['ligand'] = batch['ligand'].to(device)
                pred = model(batch)
                pred_actual = pred * std_pKi + mean_pKi
                target_actual = batch['pKi'] * std_pKi + mean_pKi
                loss = loss_fn(pred_actual, target_actual)
                val_loss += loss.item()
                val_batches += 1
                val_bar.set_postfix(batch_loss=loss.item())

                all_preds.append(pred_actual.cpu())
                all_targets.append(target_actual.cpu())

            except Exception as e:
                print(f"[SKIPPED VAL BATCH] due to error: {e}")
                continue

    avg_val_loss = val_loss / max(val_batches, 1)
    val_bar.set_postfix(avg_val_loss=avg_val_loss)


    # End of epoch validation loss check
    if avg_val_loss < best_val_loss - min_delta:
        best_val_loss = avg_val_loss
        patience_counter = 0
        torch.save(model.state_dict(), save_path)
        print(f"Saved new best model with val_loss = {best_val_loss:.4f}")
    else:
        patience_counter += 1
        print(f"No improvement for {patience_counter} epochs")

    # Check if early stopping condition met
    if patience_counter >= patience:
        print("Early stopping triggered")
        break

    all_preds = torch.cat(all_preds, dim=0)
    all_targets = torch.cat(all_targets, dim=0)

    # Record values for plots
    train_losses.append(avg_train_loss)
    val_losses.append(avg_val_loss)

    # Compute RMSE and R² from unstandardized values
    val_rmse = compute_rmse(all_preds, all_targets)
    val_r2 = compute_r2(all_preds, all_targets)

    val_rmses.append(val_rmse)
    val_r2s.append(val_r2)

    print(f"Epoch {epoch+1}/{num_epochs}, Train Loss: {avg_train_loss:.4f}, Val Loss: {avg_val_loss:.4f}")

torch.save(model.state_dict(), "best_model.pt")
torch.save(optimizer.state_dict(), "optimizer.pt")
torch.save({
    'epoch': epoch,
    'train_losses': train_losses,
    'val_losses': val_losses,
    'val_rmses': val_rmses,
    'val_r2s': val_r2s,
    'best_val_loss': best_val_loss,
    'patience_counter': patience_counter,
    'mean_pKi': mean_pKi,
    'std_pKi': std_pKi,
    'min_delta': min_delta,
    'patience': patience,
    'save_path': save_path
}, "training_state.pt")

# Training & Validation Loss over Epochs
plt.plot(train_losses, label='Train Loss')
plt.plot(val_losses, label='Validation Loss')
plt.xlabel("Epoch")
plt.ylabel("MSE Loss (pKi)")
plt.legend()
plt.title("Training and Validation Loss")
plt.show()

# Validation RMSE over Epochs
plt.plot(val_rmses, label='Validation RMSE')
plt.xlabel("Epoch")
plt.ylabel("RMSE (pKi)")
plt.title("Validation RMSE")
plt.legend()
plt.show()

# Validation R² Score over Epochs
plt.plot(val_r2s, label='Validation R²')
plt.xlabel("Epoch")
plt.ylabel("R² Score")
plt.title("Validation R² Score")
plt.legend()
plt.show()

# Predicted vs Actual pKi on Test Set
model.eval()

all_preds = []
all_targets = []

with torch.no_grad():
    for batch in test_loader:
        batch = {k: v.to(device) if isinstance(v, torch.Tensor) or isinstance(v, torch_geometric.data.Batch) else v for k, v in batch.items()}
        batch['ligand'] = batch['ligand'].to(device)
        pred = model(batch)
        pred_actual = pred * std_pKi + mean_pKi
        target_actual = batch['pKi'] * std_pKi + mean_pKi
        all_preds.append(pred_actual.cpu())
        all_targets.append(target_actual.cpu())

# Combine and convert to lists for plotting
all_preds = torch.cat(all_preds).tolist()
all_targets = torch.cat(all_targets).tolist()

# Plot
plt.scatter(all_targets, all_preds, alpha=0.5)
plt.plot([min(all_targets), max(all_targets)], [min(all_targets), max(all_targets)], color='red', linestyle='--')
plt.xlabel("Actual pKi")
plt.ylabel("Predicted pKi")
plt.title("Predicted vs. Actual pKi on Test Set")
plt.show()

# Convert list of floats to tensors
all_preds_tensor = torch.tensor(all_preds)
all_targets_tensor = torch.tensor(all_targets)

# Compute residuals
residuals = all_preds_tensor - all_targets_tensor

# Plot histogram
plt.hist(residuals.tolist(), bins=50, alpha=0.7)
plt.xlabel("Prediction Error (pKi)")
plt.ylabel("Frequency")
plt.title("Residuals Histogram")
plt.show()

# Concatenate all predictions and targets
all_preds_tensor = torch.cat(all_preds)
all_targets_tensor = torch.cat(all_targets)

# Compute RMSE
test_rmse = torch.sqrt(torch.mean((all_preds_tensor - all_targets_tensor) ** 2)).item()

# Compute R²
ss_res = torch.sum((all_targets_tensor - all_preds_tensor) ** 2)
ss_tot = torch.sum((all_targets_tensor - torch.mean(all_targets_tensor)) ** 2)
test_r2 = (1 - ss_res / ss_tot).item()

print(f"Test RMSE: {test_rmse:.4f}")
print(f"Test R²: {test_r2:.4f}")

# Concatenate all predictions and targets
all_preds_tensor = torch.cat(all_preds)
all_targets_tensor = torch.cat(all_targets)
