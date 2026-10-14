"""Standard relation-specific GCN layer for E2T Full Model."""
import torch
import torch.nn as nn
import torch.nn.functional as F

class StandardRGCNLayer(nn.Module):
    def __init__(self, in_features, out_features, num_relations=4):
        super().__init__()
        self.num_relations = num_relations
        self.out_features = out_features
        self.weight_rel = nn.Parameter(torch.empty(num_relations, in_features, out_features))
        self.weight_self = nn.Parameter(torch.empty(in_features, out_features))
        self.bias = nn.Parameter(torch.zeros(out_features))
        self.ln = nn.LayerNorm(out_features)
        nn.init.xavier_uniform_(self.weight_rel)
        nn.init.xavier_uniform_(self.weight_self)

    def forward(self, x, adj):
        out = torch.mm(x, self.weight_self)
        for r in range(self.num_relations):
            out = out + torch.mm(adj[r], torch.mm(x, self.weight_rel[r]))
        return F.leaky_relu(self.ln(out + self.bias), negative_slope=0.01)

