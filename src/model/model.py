from torch_geometric.nn import MessagePassing, global_mean_pool
import torch.nn as nn
import torch.nn.functional as F
import torch


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
        ligand_batch = batch['ligand']
        ligand_emb = self.ligand_gnn(ligand_batch)  # shape: [num_nodes_total, 128]

        # Project protein to same dimension
        protein_emb = self.protein_projection(batch['protein_emb'])

        # Cross attention expects: [batch_size, seq_len, dim]
        protein_emb = protein_emb.unsqueeze(1)  # [batch_size, 1, 128]
        ligand_emb = ligand_emb.unsqueeze(1)    # [batch_size, 1, 128]

        attn_output, _ = self.cross_attention(
            query=protein_emb,
            key=ligand_emb,
            value=ligand_emb
        )

        combined_emb = torch.cat([protein_emb, attn_output], dim=-1).squeeze(1)  # [batch_size, 256]
        return self.mlp(combined_emb).squeeze(-1)  # [batch_size]
