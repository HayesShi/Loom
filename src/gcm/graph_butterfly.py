"""Graph-supported butterfly aggregation.

The compiler assigns each directed graph edge to the one dyadic level where
its endpoint wires first separate. The level controls a *real edge*, rather
than sending a message through an arbitrary butterfly partner. This operator
is initially exactly the supplied weighted adjacency, including self-loops.
"""

import torch
from torch import nn


class GraphButterflyAggregate(nn.Module):
    """Apply ``sum_s gain[block, s] A_s x`` over the compiled edge list.

    ``edge_index`` uses PyG's source-to-target convention; ``edge_weight`` is
    already normalized by the caller. Self-loops retain their supplied weight
    and are not assigned a butterfly level. Duplicate directed edges add.
    The coalesced edge list and level assignment are compiled once for all
    blocks. Each forward computes one edge-weighted gather and index-add;
    PyTorch 2.5 CUDA's dynamic sparse-value backward is much slower and uses
    more memory for this operator. The compiled indices stay shared by blocks.
    With ``row_preserve=True``, each receiver's original total edge weight is
    held constant while the gains redistribute that weight across levels.
    """

    def __init__(self, edge_index, node_to_wire, edge_weight, num_stages=None, blocks=1,
                 row_preserve=False):
        super().__init__()
        if edge_index.ndim != 2 or edge_index.shape[0] != 2:
            raise ValueError('edge_index must have shape [2, E]')
        if node_to_wire.ndim != 1:
            raise ValueError('node_to_wire must have shape [N]')
        if edge_weight.ndim != 1 or edge_weight.numel() != edge_index.shape[1]:
            raise ValueError('edge_weight must have shape [E]')
        if edge_index.dtype != torch.long or node_to_wire.dtype != torch.long:
            raise TypeError('edge_index and node_to_wire must be int64 tensors')
        if not edge_weight.is_floating_point():
            raise TypeError('edge_weight must be floating point')
        if not bool(torch.isfinite(edge_weight).all()):
            raise ValueError('edge_weight must be finite')
        if row_preserve and bool((edge_weight < 0).any()):
            raise ValueError('row_preserve requires nonnegative edge_weight')

        # Compilation is a one-time CPU operation. Buffers follow model.to().
        edge_index = edge_index.detach().cpu()
        node_to_wire = node_to_wire.detach().cpu()
        edge_weight = edge_weight.detach().cpu()
        nodes = node_to_wire.numel()
        if nodes and (bool((node_to_wire < 0).any()) or node_to_wire.unique().numel() != nodes):
            raise ValueError('node_to_wire must be a nonnegative injection')
        inferred_stages = int(node_to_wire.max()).bit_length() if nodes else 0
        if num_stages is None:
            num_stages = inferred_stages
        if isinstance(num_stages, bool) or not isinstance(num_stages, int) or num_stages < inferred_stages:
            raise ValueError('num_stages must cover all real wire IDs')
        if isinstance(blocks, bool) or not isinstance(blocks, int) or blocks < 1:
            raise ValueError('blocks must be a positive integer')
        if edge_index.numel() and (bool((edge_index < 0).any()) or bool((edge_index >= nodes).any())):
            raise ValueError('edge_index contains an invalid node ID')

        # row=receiver, col=sender. Coalescing preserves duplicate-edge sums.
        src, dst = edge_index
        adjacency = torch.sparse_coo_tensor(
            torch.stack((dst, src)), edge_weight, (nodes, nodes)
        ).coalesce()
        indices = adjacency.indices()
        rows, col = indices
        crossing = node_to_wire[col] ^ node_to_wire[rows]
        stage_key = torch.zeros_like(crossing)
        nonself = crossing != 0
        if bool(nonself.any()):
            powers = 1 << torch.arange(num_stages, dtype=torch.long)
            # 0 is reserved for self-loops. Non-self edge with highest
            # differing bit s is controlled by gain[s], at key s+1.
            stage_key[nonself] = torch.searchsorted(
                powers, crossing[nonself].contiguous(), right=True
            )
        self.num_nodes = nodes
        self.num_stages = num_stages
        self.blocks = blocks
        self.row_preserve = bool(row_preserve)
        self.register_buffer('indices', indices, persistent=False)
        self.register_buffer('base_weight', adjacency.values(), persistent=False)
        self.register_buffer('stage_key', stage_key, persistent=False)
        if self.row_preserve:
            # The graph is fixed: compile the row mass of each dyadic edge
            # group once, avoiding an edge-wise reduction every forward.
            mass = torch.zeros(nodes * (num_stages + 1), dtype=edge_weight.dtype)
            mass.index_add_(0, rows * (num_stages + 1) + stage_key, adjacency.values())
            mass = mass.reshape(nodes, num_stages + 1)
            self.register_buffer('mass_by_stage', mass, persistent=False)
            self.register_buffer('row_total', mass.sum(-1), persistent=False)
        else:
            self.register_buffer('mass_by_stage', None, persistent=False)
            self.register_buffer('row_total', None, persistent=False)
        # Positive, bounded stage gains avoid reversing a graph edge. At zero
        # the operator is exactly the supplied normalized adjacency.
        self.raw_gain = nn.Parameter(torch.zeros((blocks, num_stages), dtype=edge_weight.dtype))

    @property
    def gain(self):
        return 1 + self.raw_gain.tanh()

    def forward(self, x, block_index=0):
        if x.ndim != 2 or x.shape[0] != self.num_nodes:
            raise ValueError(f'x must have shape [{self.num_nodes}, D]')
        if isinstance(block_index, bool) or not isinstance(block_index, int) or not 0 <= block_index < self.blocks:
            raise ValueError('block_index must identify a compiled block')
        scale = torch.cat((self.raw_gain.new_ones(1), self.gain[block_index]))
        values = self.base_weight * scale[self.stage_key]
        rows, columns = self.indices
        messages = x.index_select(0, columns) * values.unsqueeze(-1)
        result = torch.zeros_like(x).index_add_(0, rows, messages)
        if self.row_preserve:
            denominator = self.mass_by_stage @ scale
            factor = torch.where(
                self.row_total > 0,
                self.row_total / denominator.clamp_min(torch.finfo(denominator.dtype).tiny),
                torch.ones_like(self.row_total),
            )
            result = result * factor.unsqueeze(-1)
        return result
