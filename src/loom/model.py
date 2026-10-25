"""Graph embedding, persistent wire states, and pre-normalized Loom blocks."""

import torch
from torch import nn
from .butterfly import ButterflyMixer


def normalized_neighbors(x, edge_index, directed=False):
    """Zero-self-loop one-step normalization; no hidden-feature refresh."""
    src, dst = edge_index
    keep = src != dst
    src, dst = src[keep], dst[keep]
    n = len(x)
    out = torch.bincount(src, minlength=n).to(x.dtype).clamp_min(1.0)
    incoming = torch.bincount(dst, minlength=n).to(x.dtype).clamp_min(1.0)
    weight = (out[src] * incoming[dst]).rsqrt()
    a = torch.sparse_coo_tensor(torch.stack((dst, src)), weight, (n, n)).coalesce()
    return torch.sparse.mm(a, x)


class LoomBlock(nn.Module):
    def __init__(self, dim: int, heads: int, stages: int, dropout: float, angle: float):
        super().__init__()
        self.norm_mix = nn.LayerNorm(dim)
        self.mixer = ButterflyMixer(dim, heads, stages, angle)
        self.projection = nn.Linear(dim, dim, bias=False)
        self.norm_ff = nn.LayerNorm(dim)
        self.ff = nn.Sequential(
            nn.Linear(dim, 2 * dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(2 * dim, dim),
        )
        self.drop = nn.Dropout(dropout)

    def forward(self, states: torch.Tensor):
        states = states + self.drop(self.projection(self.mixer(self.norm_mix(states))))
        return states + self.drop(self.ff(self.norm_ff(states)))


class Loom(nn.Module):
    """Node classification or regression with one graph embedding.

    `neighbors` is normalized_neighbors(x, edge_index), computed once for
    static inputs. `mapping[i]` gives node i's wire address. Outputs are logits
    for classification; apply softmax only when probabilities are required.
    Batched graphs use local addresses and a common power-of-two capacity.
    """

    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        max_nodes: int,
        dim: int = 64,
        heads: int = 4,
        blocks: int = 4,
        dropout: float = 0.1,
        angle: float = 0.2,
    ):
        super().__init__()
        if max_nodes < 1 or blocks < 1 or heads < 1 or dim % (2 * heads):
            raise ValueError("Use positive sizes and an even feature width per head")
        self.dim = dim
        self.stages = (max_nodes - 1).bit_length()
        self.self_projection = nn.Linear(input_dim, dim)
        self.graph_projection = nn.Linear(input_dim, dim, bias=False)
        self.embed_norm = nn.LayerNorm(dim)
        self.blocks = nn.ModuleList(
            LoomBlock(dim, heads, self.stages, dropout, angle) for _ in range(blocks)
        )
        self.head_norm = nn.LayerNorm(dim)
        self.head = nn.Linear(dim, output_dim)

    def encode(self, x, neighbors, mapping, graph_ids=None, padded=None):
        """Place once, run every block, then restore real-node order."""
        h = self.embed_norm(self.self_projection(x) + self.graph_projection(neighbors))
        if graph_ids is None:
            padded = padded or 1 << max(0, (len(mapping) - 1).bit_length())
            states = h.new_zeros(padded, self.dim).index_copy(0, mapping, h)
            for block in self.blocks:
                states = block(states)
            return states[mapping]
        if padded is None:
            raise ValueError("Batched graphs require a common padded capacity")
        graphs = int(graph_ids.max()) + 1
        addresses = graph_ids * padded + mapping
        states = h.new_zeros(graphs * padded, self.dim).index_copy(0, addresses, h)
        states = states.reshape(graphs, padded, self.dim)
        for block in self.blocks:
            states = block(states)
        return states.reshape(-1, self.dim)[addresses]

    def forward(self, x, neighbors, mapping, graph_ids=None, padded=None):
        h = self.encode(x, neighbors, mapping, graph_ids, padded)
        return self.head(self.head_norm(h))

    def forward_graph(self, x, neighbors, mapping, graph_ids, padded):
        """Mean-pool real nodes after mixing for graph-level prediction."""
        h = self.encode(x, neighbors, mapping, graph_ids, padded)
        graphs = int(graph_ids.max()) + 1
        pooled = h.new_zeros(graphs, self.dim).index_add(0, graph_ids, h)
        counts = torch.bincount(graph_ids, minlength=graphs).to(h.dtype).unsqueeze(-1)
        return self.head(self.head_norm(pooled / counts.clamp_min(1)))
