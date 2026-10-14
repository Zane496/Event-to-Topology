"""Full E2T model (GRU + shared static/dynamic R-GCN + node-gated feature fusion)."""
import torch
import torch.nn as nn
from .layers import StandardRGCNLayer
HIDDEN_DIM = 64
FEATURE_MIX = 1.0
TOPOLOGY_EPS = 1e-8

class E2TFullModel(nn.Module):
    """Full Model: GRU, tied static/dynamic R-GCN, binary node-change gate, feature fusion."""
    def __init__(self, feature_dim):
        super().__init__()
        self.gru = nn.GRU(feature_dim, HIDDEN_DIM, batch_first=True)
        self.temporal_ln = nn.LayerNorm(HIDDEN_DIM)
        self.rgcn = StandardRGCNLayer(HIDDEN_DIM, HIDDEN_DIM, 4)
        self.head = nn.Sequential(nn.Linear(HIDDEN_DIM * 2, HIDDEN_DIM),
                                  nn.LeakyReLU(0.01), nn.Dropout(0.1), nn.Linear(HIDDEN_DIM, 1))

    @staticmethod
    def node_gate(static_adj, dynamic_adj):
        changed = (dynamic_adj - static_adj).abs() > TOPOLOGY_EPS
        return (changed.any(dim=0).any(dim=1) | changed.any(dim=0).any(dim=0)).float().unsqueeze(-1)

    def forward(self, x_seq, static_adj, dynamic_adj):
        if x_seq.dim() == 4: x_seq = x_seq.squeeze(0)
        if static_adj.dim() == 4: static_adj = static_adj.squeeze(0)
        if dynamic_adj.dim() == 4: dynamic_adj = dynamic_adj.squeeze(0)
        x_seq = x_seq.reshape(x_seq.size(0), x_seq.size(1), -1)
        temporal = self.temporal_ln(self.gru(x_seq)[0][:, -1, :])
        s = self.rgcn(temporal, static_adj)
        d = self.rgcn(temporal, dynamic_adj)
        g = self.node_gate(static_adj, dynamic_adj).to(device=temporal.device, dtype=temporal.dtype)
        z = s + FEATURE_MIX * g * (d - s)
        return self.head(torch.cat([temporal, z], dim=-1)).squeeze(-1)

