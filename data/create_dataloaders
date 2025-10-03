import pandas as pd
import numpy as np

# Load DataFrame
df = pd.read_csv('bindingdb_esm2_embedded_50k.csv')
invalid_indices = [5479] + list(range(19487, 19999))
df = df.drop(index=invalid_indices).reset_index(drop=True)

# Install dependencies
!pip install torch==2.0.1 torchvision==0.15.2 torchaudio==2.0.2 --index-url https://download.pytorch.org/whl/cu118
!pip install torch-geometric==2.5.3
!pip install torch-scatter torch-sparse -f https://data.pyg.org/whl/torch-2.0.1+cu118.html
!pip install numpy
!pip install pandas
!pip install rdkit

# Verify installations
import torch
import torch_geometric
import numpy as np
import pandas as pd
print("PyTorch version:", torch.__version__)
print("PyTorch Geometric version:", torch_geometric.__version__)
print("NumPy version:", np.__version__)
print("Pandas version:", pd.__version__)

# Simplified type mapping for metals
type_to_onehot = {
    "nonmetal": [1, 0, 0, 0],
    "metal": [0, 1, 0, 0],
    "metalloid": [0, 0, 1, 0],
    "noble_gas": [0, 0, 0, 1]
}
atomic_df = pd.read_csv('Atomic_Property_Lookup_Table.csv')

property_lookup = {}
for row in atomic_df.itertuples():
    z = row.Z
    metal_type = row.metal_type

    if metal_type in ["alkali", "alkaline", "transition", "post_transition"]:
        metal_type = "metal"

    property_lookup[z] = {
        "electronegativity": row.electronegativity if pd.notnull(row.electronegativity) else 0.0,
        "valence_electrons": row.valence_electrons,
        "atomic_mass": row.atomic_mass,
        "covalent_radius": row.covalent_radius,
        "metal_onehot": type_to_onehot.get(metal_type, [0, 0, 0, 0])
    }


from torch_geometric.data import Data
from torch.utils.data import Dataset
from torch_geometric.loader import DataLoader as PyGDataLoader
import torch.utils.data
import ast
import rdkit.Geometry as rdGeom
from sklearn.model_selection import train_test_split


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

    # Atom-level features using atomic number + property lookup
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

            # Parse z, pos, esm2
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
