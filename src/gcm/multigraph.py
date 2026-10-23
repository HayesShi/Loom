"""Shared-parameter GCM for batches of graphs with different node counts.

Each graph is compiled to a node-to-wire permutation.  The trainable pair
switches and all surrounding projections are shared across graphs.  No graph
is pooled until its full node-state line has passed through every block.
"""

from __future__ import annotations

import math
import numbers
from collections import deque
from collections.abc import Sequence

import torch
from torch import nn
from torch.nn import functional as F
from torch_geometric.nn.conv.gcn_conv import gcn_norm

from .architecture import compile_interfaces
from .pure import PhasePair


def compile_pair_distance_angle_scales(edge_index: torch.Tensor,
                                       node_to_wire: torch.Tensor) -> torch.Tensor:
    """Compute 1 / graph-hop distance for every regular butterfly pair.

    This offline compiler uses only adjacency and a full node-to-wire
    permutation. Its output has shape [stages, nodes // 2], with pairs in the
    same order as the regular butterfly at each stage. It adds no model state.
    """
    mapping = torch.as_tensor(node_to_wire, dtype=torch.long).detach().cpu()
    nodes = mapping.numel()
    if mapping.ndim != 1 or nodes < 2 or nodes & (nodes - 1):
        raise ValueError("Pair distances require a full power-of-two wire line")
    if not torch.equal(torch.sort(mapping).values, torch.arange(nodes)):
        raise ValueError("node_to_wire must be a permutation of all wires")
    edges = torch.as_tensor(edge_index, dtype=torch.long).detach().cpu()
    if edges.ndim != 2 or edges.shape[0] != 2:
        raise ValueError("edge_index must have shape [2, num_edges]")
    if edges.numel() and (bool((edges < 0).any()) or bool((edges >= nodes).any())):
        raise ValueError("edge_index contains an out-of-range node")
    neighbors = [set() for _ in range(nodes)]
    for source, target in edges.T.tolist():
        if source != target:
            neighbors[source].add(target)
            neighbors[target].add(source)

    wire_to_node = torch.empty_like(mapping)
    wire_to_node[mapping] = torch.arange(nodes)
    stages = nodes.bit_length() - 1
    requests: list[list[tuple[int, int, int]]] = [[] for _ in range(nodes)]
    for level in range(stages):
        bit = 1 << level
        left_wires = [wire for wire in range(nodes) if not wire & bit]
        for pair_index, left_wire in enumerate(left_wires):
            right_node = int(wire_to_node[left_wire ^ bit])
            requests[left_wire].append((level, pair_index, right_node))

    distances = torch.empty((stages, nodes // 2), dtype=torch.float32)
    for source_wire, destinations in enumerate(requests):
        if not destinations:
            continue
        source_node = int(wire_to_node[source_wire])
        hops = [-1] * nodes
        hops[source_node] = 0
        pending = deque([source_node])
        while pending:
            current = pending.popleft()
            for neighbor in neighbors[current]:
                if hops[neighbor] < 0:
                    hops[neighbor] = hops[current] + 1
                    pending.append(neighbor)
        for level, pair_index, destination in destinations:
            distance = hops[destination]
            if distance < 1:
                raise ValueError("A butterfly pair is disconnected in the graph")
            distances[level, pair_index] = distance
    return distances.reciprocal()


def compile_headwise_pair_angle_scales(distance_scales: torch.Tensor,
                                       heads: int,
                                       global_angle_heads: int,
                                       distance_powers: Sequence[float] | None = None
                                       ) -> torch.Tensor:
    """Assign existing J heads a fixed graph-distance mixing range.

    Zero global heads returns the original two-dimensional table unchanged,
    preserving the legacy all-head distance path exactly.
    """
    if distance_scales.ndim != 2:
        raise ValueError("distance_scales must have shape [stages, pairs]")
    if heads < 1 or not 0 <= global_angle_heads <= heads:
        raise ValueError("global_angle_heads must be between zero and heads")
    if distance_powers is not None:
        if global_angle_heads or len(distance_powers) != heads or any(
                not math.isfinite(power) or power < 0
                for power in distance_powers):
            raise ValueError("distance_powers must have one finite nonnegative "
                             "exponent per head and no global_angle_heads")
        powers = distance_scales.new_tensor(distance_powers)
        return distance_scales.unsqueeze(-1).pow(powers)
    if global_angle_heads == 0:
        return distance_scales
    scales = distance_scales.unsqueeze(-1).expand(-1, -1, heads).clone()
    scales[..., :global_angle_heads] = 1.
    return scales


class _VariableSizeCircuit(nn.Module):
    """PureCircuit's regular butterfly and PhasePair, with a runtime dummy mask.

    ``PureCircuit`` stores one graph's padded-wire mask.  A mini-batch needs a
    different mask for each graph, so this class keeps its switch equations and
    stage parameters but receives the mask in ``forward``.  A zero pair scale
    removes only partner transport; stage feed-forward terms, if requested,
    are still run on the zero-exchange control.
    """

    def __init__(self, dim: int, heads: int, max_stages: int, mode: str,
                 angle: float, stage_ff: bool, tie_pair_gate: bool = False,
                 prior_only_angle_scale: bool = False,
                 reverse_stages: bool = False,
                 interstage_head_shuffle: bool = False,
                 repeat_pair_switch: bool = False,
                 gate_anchor: float = 0.,
                 shuffle_exempt_heads: int = 0):
        super().__init__()
        if (isinstance(gate_anchor, bool) or not isinstance(gate_anchor, numbers.Real)
                or not math.isfinite(gate_anchor) or not 0. <= gate_anchor <= 1.):
            raise ValueError("gate_anchor must be finite and in [0, 1]")
        self.gate_anchor = float(gate_anchor)
        if interstage_head_shuffle and stage_ff:
            raise ValueError("interstage_head_shuffle requires stage_ff=False")
        if (isinstance(shuffle_exempt_heads, bool)
                or not isinstance(shuffle_exempt_heads, numbers.Integral)
                or not 0 <= shuffle_exempt_heads < heads):
            raise ValueError("shuffle_exempt_heads must be in [0, heads)")
        if shuffle_exempt_heads and not interstage_head_shuffle:
            raise ValueError("shuffle_exempt_heads requires interstage_head_shuffle=True")
        self.repeat_pair_switch = bool(repeat_pair_switch)
        # Keep every checkpoint key and the initialization stream unchanged.
        # Repeat mode routes all stages through switch 0; the remaining
        # registered switches are dormant so paired runs start identically.
        self.switches = nn.ModuleList(
            PhasePair(dim, heads, mode=mode, angle=angle)
            for _ in range(max_stages)
        )
        if tie_pair_gate:
            # A single content rule is applied at every resolution; each
            # resolution still has its own learned base angle.
            shared_gate = self.switches[0].gate
            for switch in self.switches[1:]:
                switch.gate = shared_gate
        self.stage_ff = bool(stage_ff)
        self.prior_only_angle_scale = bool(prior_only_angle_scale)
        self.reverse_stages = bool(reverse_stages)
        self.interstage_head_shuffle = bool(interstage_head_shuffle)
        self.shuffle_exempt_heads = int(shuffle_exempt_heads)
        self.max_stages = max_stages
        if stage_ff:
            self.gains = nn.Parameter(torch.full((max_stages, dim), .1))
            self.biases = nn.Parameter(torch.zeros(max_stages, dim))
        self.register_buffer(
            "shuffle", torch.arange(dim).reshape(heads, -1).T.flatten(),
            persistent=False,
        )
        # Transpose complete real/imaginary pairs across movable heads.
        # Exempt carrier heads retain their complex channel coordinates.
        pairs_per_head = dim // (2 * heads)
        if shuffle_exempt_heads:
            fixed_pairs = torch.arange(shuffle_exempt_heads * pairs_per_head)
            movable_pairs = torch.arange(shuffle_exempt_heads * pairs_per_head,
                                         heads * pairs_per_head).reshape(
                heads - shuffle_exempt_heads, pairs_per_head).T.flatten()
            if heads - shuffle_exempt_heads > 1:
                movable_pairs = movable_pairs.roll(1)
            pair_order = torch.cat((fixed_pairs, movable_pairs))
        else:
            # Keep the original permutation bitwise identical by default.
            pair_order = torch.arange(heads * pairs_per_head).reshape(
                heads, pairs_per_head).T.flatten()
            if heads > 1:
                pair_order = pair_order.roll(1)
        pair_channels = 2 * pair_order[:, None] + torch.arange(2)
        permutation = pair_channels.flatten()
        self.register_buffer("interstage_shuffle", permutation,
                             persistent=False)
        # Undo the cumulative wiring after the final J stage. This fixes the
        # output channel coordinates and makes a bypassed circuit exactly the
        # identity, including for graphs with fewer active stages in a batch.
        powers = []
        composite = torch.arange(dim)
        for _ in range(max_stages):
            powers.append(torch.argsort(composite))
            composite = composite[permutation]
        self.register_buffer("interstage_unshuffle", torch.stack(powers),
                             persistent=False)

    def forward(self, z: torch.Tensor, real_mask: torch.Tensor,
                pair_scale: float = 1.,
                active_stages: torch.Tensor | None = None,
                pair_angle_scales: torch.Tensor | None = None,
                relay_mask: torch.Tensor | None = None) -> torch.Tensor:
        if z.ndim != 3:
            raise ValueError("Circuit input must have shape [graphs, padded wires, dim]")
        graphs, padded, dim = z.shape
        if padded < 1 or padded & (padded - 1):
            raise ValueError("Each graph needs a power-of-two padded wire line")
        if real_mask.shape != (graphs, padded) or real_mask.dtype != torch.bool:
            raise ValueError("real_mask must have one boolean value per wire")
        if relay_mask is not None:
            if (relay_mask.shape != (graphs, padded)
                    or relay_mask.dtype != torch.bool
                    or bool((real_mask & ~relay_mask).any())):
                raise ValueError("relay_mask must include every real wire")
        if not 0. <= pair_scale <= 1.:
            raise ValueError("pair_scale must be in [0, 1]")
        stages = min(padded.bit_length() - 1, self.max_stages)
        if pair_angle_scales is not None and (
                pair_angle_scales.ndim not in (2, 3)
                or pair_angle_scales.shape[0] < stages
                or pair_angle_scales.shape[1] != padded // 2
                or (pair_angle_scales.ndim == 3
                    and pair_angle_scales.shape[2] != self.switches[0].heads)):
            raise ValueError("pair_angle_scales must have shape "
                             "[stages, padded // 2] or "
                             "[stages, padded // 2, heads]")
        participating_mask = real_mask if relay_mask is None else relay_mask
        if self.interstage_head_shuffle and active_stages is not None and (
                active_stages.shape != (graphs,) or
                bool((active_stages < 0).any())):
            raise ValueError("active_stages must contain one nonnegative count per graph")
        levels = (range(stages - 1, -1, -1) if self.reverse_stages
                  else range(stages))
        entry = z if self.gate_anchor else None
        for level in levels:
            stride = 1 << level
            shape = (graphs, padded // (2 * stride), 2, stride, dim)
            grouped = z.reshape(shape)
            left, right = grouped[:, :, 0], grouped[:, :, 1]
            mask = participating_mask.reshape(
                graphs, padded // (2 * stride), 2, stride)
            valid = mask[:, :, 0] & mask[:, :, 1]
            if pair_scale:
                switch = self.switches[0 if self.repeat_pair_switch else level]
                if pair_angle_scales is None:
                    if entry is None:
                        new_left, new_right = switch(left, right)
                    else:
                        anchor = entry.reshape(shape)
                        gate_u = ((1. - self.gate_anchor) * left
                                  + self.gate_anchor * anchor[:, :, 0])
                        gate_v = ((1. - self.gate_anchor) * right
                                  + self.gate_anchor * anchor[:, :, 1])
                        new_left, new_right = switch(
                            left, right, gate_u=gate_u, gate_v=gate_v)
                else:
                    head_columns = (1 if pair_angle_scales.ndim == 2
                                    else switch.heads)
                    angle_scale = pair_angle_scales[level].reshape(
                        1, padded // (2 * stride), stride, head_columns)
                    if entry is None:
                        new_left, new_right = switch(
                            left, right, angle_scale=angle_scale,
                            prior_only_scale=self.prior_only_angle_scale)
                    else:
                        anchor = entry.reshape(shape)
                        gate_u = ((1. - self.gate_anchor) * left
                                  + self.gate_anchor * anchor[:, :, 0])
                        gate_v = ((1. - self.gate_anchor) * right
                                  + self.gate_anchor * anchor[:, :, 1])
                        new_left, new_right = switch(
                            left, right, angle_scale=angle_scale,
                            prior_only_scale=self.prior_only_angle_scale,
                            gate_u=gate_u, gate_v=gate_v)
                if pair_scale != 1.:
                    new_left = left + pair_scale * (new_left - left)
                    new_right = right + pair_scale * (new_right - right)
                left = torch.where(valid.unsqueeze(-1), new_left, left)
                right = torch.where(valid.unsqueeze(-1), new_right, right)
                z = torch.stack((left, right), dim=2).reshape_as(z)
            if self.stage_ff:
                transformed = z + self.gains[level] * F.gelu(z + self.biases[level])
                transformed = transformed.index_select(-1, self.shuffle)
                if active_stages is not None:
                    z = torch.where((active_stages > level)[:, None, None], transformed, z)
                else:
                    z = transformed
            if self.interstage_head_shuffle and (
                    level > 0 if self.reverse_stages else level + 1 < stages):
                transformed = z.index_select(-1, self.interstage_shuffle)
                if active_stages is None:
                    z = transformed
                else:
                    has_next_stage = (active_stages > level if self.reverse_stages
                                      else active_stages > level + 1)
                    z = torch.where(has_next_stage[:, None, None], transformed, z)
        if self.interstage_head_shuffle:
            if active_stages is None:
                z = z.index_select(-1, self.interstage_unshuffle[max(stages - 1, 0)])
            else:
                undo_count = active_stages.clamp(max=stages).sub(1).clamp_min(0)
                undo = self.interstage_unshuffle[undo_count]
                z = z.gather(-1, undo[:, None, :].expand(graphs, padded, dim))
        return z


def _aggregate(node_values: torch.Tensor, edge_index: torch.Tensor,
               edge_weight: torch.Tensor, edge_values: torch.Tensor | None,
               num_nodes: int) -> torch.Tensor:
    out = node_values.new_zeros((num_nodes, node_values.shape[-1]))
    if edge_index.numel() == 0:
        return out
    source, target = edge_index
    message = node_values.index_select(0, source)
    if edge_values is not None:
        message = message + edge_values
    out.index_add_(0, target, message * edge_weight.unsqueeze(-1))
    return out


def _edge_weights(edge_index: torch.Tensor, num_nodes: int,
                  mode: str) -> torch.Tensor:
    """Normalize incoming messages without erasing directed source nodes."""
    if mode == "symmetric":
        normalized_edges, weights = gcn_norm(
            edge_index, num_nodes=num_nodes, add_self_loops=False,
        )
        if normalized_edges.shape[1] != edge_index.shape[1]:
            raise RuntimeError("Unexpected edge reordering during normalization")
        return weights
    if mode == "directed_symmetric":
        if edge_index.numel() == 0:
            return torch.empty(0, dtype=torch.float32,
                               device=edge_index.device)
        source, target = edge_index
        outdegree = torch.bincount(source, minlength=num_nodes)
        indegree = torch.bincount(target, minlength=num_nodes)
        return outdegree[source].rsqrt() * indegree[target].rsqrt()
    if mode == "target_mean":
        if edge_index.numel() == 0:
            return torch.empty(0, dtype=torch.float32,
                               device=edge_index.device)
        target = edge_index[1]
        indegree = torch.bincount(target, minlength=num_nodes)
        return indegree[target].reciprocal()
    raise ValueError(f"Unsupported edge-weight mode: {mode}")


class _MultiGraphBlock(nn.Module):
    def __init__(self, dim: int, heads: int, max_stages: int, mode: str,
                 dropout: float, angle: float, stage_ff: bool,
                 tie_pair_gate: bool = False,
                 prior_only_angle_scale: bool = False,
                 reverse_stages: bool = False,
                 interstage_head_shuffle: bool = False,
                 repeat_pair_switch: bool = False,
                 gate_anchor: float = 0.,
                 shuffle_exempt_heads: int = 0):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.refresh = nn.Linear(dim, dim, bias=False)
        self.circuit = _VariableSizeCircuit(
            dim, heads, max_stages, mode, angle, stage_ff, tie_pair_gate,
            prior_only_angle_scale, reverse_stages, interstage_head_shuffle,
            repeat_pair_switch, gate_anchor, shuffle_exempt_heads,
        )
        self.output = nn.Linear(dim, dim, bias=False)
        self.ffn = nn.Sequential(
            nn.LayerNorm(dim), nn.Linear(dim, 2 * dim), nn.GELU(),
            nn.Dropout(dropout), nn.Linear(2 * dim, dim),
        )
        self.dropout = dropout

    def forward(self, z: torch.Tensor, wire_ids: torch.Tensor,
                real_mask: torch.Tensor, active_stages: torch.Tensor,
                edge_index: torch.Tensor, edge_weight: torch.Tensor,
                edge_values: torch.Tensor | None, exchange_scale: float,
                graph_refresh: bool,
                pair_angle_scales: torch.Tensor | None = None,
                relay_mask: torch.Tensor | None = None,
                ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        normalized = self.norm(z)
        real = normalized.reshape(-1, z.shape[-1]).index_select(0, wire_ids)
        if graph_refresh:
            local = _aggregate(self.refresh(real), edge_index, edge_weight,
                               edge_values, len(wire_ids))
            local_wires = z.new_zeros(z.numel() // z.shape[-1], z.shape[-1])
            local_wires = local_wires.index_copy(0, wire_ids, local).reshape_as(z)
            value = normalized + local_wires
        else:
            value = normalized
        if relay_mask is not None:
            # Empty wires are transport workspace, not persistent node states.
            # Reset them at each block's circuit entry; subsequent J stages may
            # fill and reuse them within this block.
            value = value.masked_fill(~real_mask.unsqueeze(-1), 0.)
        if self.circuit.stage_ff or self.circuit.interstage_head_shuffle:
            # The zero-exchange model retains every local stage transform.
            baseline = self.circuit(value, real_mask, pair_scale=0.,
                                    active_stages=active_stages,
                                    relay_mask=relay_mask)
        else:
            baseline = value
        if exchange_scale:
            if pair_angle_scales is None:
                mixed = self.circuit(value, real_mask,
                                     active_stages=active_stages,
                                     relay_mask=relay_mask)
            else:
                mixed = self.circuit(
                    value, real_mask, active_stages=active_stages,
                    pair_angle_scales=pair_angle_scales,
                    relay_mask=relay_mask)
            exchanged = baseline + exchange_scale * (mixed - baseline)
        else:
            exchanged = baseline
        z = z + F.dropout(self.output(exchanged), self.dropout, self.training)
        z = z + F.dropout(self.ffn(z), self.dropout, self.training)
        return (z,
                baseline.reshape(-1, z.shape[-1]).index_select(0, wire_ids),
                exchanged.reshape(-1, z.shape[-1]).index_select(0, wire_ids))


class MultiGraphGCM(nn.Module):
    """Graph-level GCM with shared switches and one compiled mapping per graph.

    Parameters mirror the node-level PureGCM where possible.  ``input_dim``
    and ``edge_dim`` are column counts.  For 9-column integer atom features and
    3-column integer bond features, the official OGB molecular encoders are
    selected automatically; other features go through linear projections.
    ``mappings`` may supply precompiled local node-to-wire vectors to avoid
    first-epoch compilation cost.  Otherwise graph-component mappings are
    cached by graph topology for this model instance's lifetime.
    ``relay_padding`` lets zero-initialized empty wires carry signals between
    regular J stages within each graph's own power-of-two circuit.
    """

    def __init__(self, input_dim: int, edge_dim: int | None, output_dim: int,
                 dim: int = 64, heads: int = 4, blocks: int = 2,
                 max_stages: int = 3, mode: str = "phase",
                 exchange_scale: float = .25, dropout: float = .2,
                 angle: float = .2, stage_ff: bool = False,
                 graph_embed: bool = True, graph_refresh: bool = True,
                 pool: str = "mean", layout: str = "graph_components",
                 seed: int = 0, tie_pair_gate: bool = False,
                 relay_padding: bool = False,
                 prior_only_angle_scale: bool = False,
                 edge_weight_mode: str = "symmetric",
                 reverse_odd_blocks: bool = False,
                 interstage_head_shuffle: bool = False,
                 repeat_pair_switch: bool = False,
                 gate_anchor: float = 0.,
                 shuffle_exempt_heads: int = 0):
        super().__init__()
        if dim <= 0 or heads <= 0 or dim % (2 * heads):
            raise ValueError("dim must be divisible by twice the head count")
        if blocks < 0 or max_stages < 1:
            raise ValueError("blocks must be nonnegative and max_stages positive")
        if not math.isfinite(exchange_scale) or not 0. <= exchange_scale <= 1.:
            raise ValueError("exchange_scale must be finite and in [0, 1]")
        if (isinstance(gate_anchor, bool) or not isinstance(gate_anchor, numbers.Real)
                or not math.isfinite(gate_anchor) or not 0. <= gate_anchor <= 1.):
            raise ValueError("gate_anchor must be finite and in [0, 1]")
        if (isinstance(shuffle_exempt_heads, bool)
                or not isinstance(shuffle_exempt_heads, numbers.Integral)
                or not 0 <= shuffle_exempt_heads < heads):
            raise ValueError("shuffle_exempt_heads must be in [0, heads)")
        if shuffle_exempt_heads and not interstage_head_shuffle:
            raise ValueError("shuffle_exempt_heads requires interstage_head_shuffle=True")
        if pool not in ("mean", "sum"):
            raise ValueError("pool must be 'mean' or 'sum'")
        if layout not in ("graph_components", "graph", "random", "input_order",
                          "graph_spectral", "graph_spectral_2opt",
                          "graph_spectral_hamming"):
            raise ValueError(f"Unsupported graph layout: {layout}")
        if edge_weight_mode not in ("symmetric", "directed_symmetric",
                                    "target_mean"):
            raise ValueError(f"Unsupported edge-weight mode: {edge_weight_mode}")
        self.dim = dim
        self.output_dim = output_dim
        self.dropout = dropout
        self.exchange_scale = float(exchange_scale)
        self.graph_embed = bool(graph_embed)
        self.graph_refresh = bool(graph_refresh)
        self.pool = pool
        self.layout = layout
        self.seed = seed
        self.max_stages = max_stages
        self.tie_pair_gate = bool(tie_pair_gate)
        self.relay_padding = bool(relay_padding)
        self.prior_only_angle_scale = bool(prior_only_angle_scale)
        self.edge_weight_mode = edge_weight_mode
        self.reverse_odd_blocks = bool(reverse_odd_blocks)
        self.interstage_head_shuffle = bool(interstage_head_shuffle)
        self.shuffle_exempt_heads = int(shuffle_exempt_heads)
        self.repeat_pair_switch = bool(repeat_pair_switch)
        self.gate_anchor = float(gate_anchor)
        self.node_encoder = nn.Linear(input_dim, dim)
        self.edge_encoder = nn.Linear(edge_dim, dim) if edge_dim else None
        self.atom_encoder = None
        self.bond_encoder = None
        if input_dim == 9 or edge_dim == 3:
            try:
                from ogb.graphproppred.mol_encoder import AtomEncoder, BondEncoder
            except ImportError:
                pass
            else:
                if input_dim == 9:
                    self.atom_encoder = AtomEncoder(dim)
                if edge_dim == 3:
                    self.bond_encoder = BondEncoder(dim)
        self.neighbor_encoder = nn.Linear(dim, dim, bias=False) if graph_embed else None
        self.input_norm = nn.LayerNorm(dim)
        self.blocks = nn.ModuleList(
            _MultiGraphBlock(dim, heads, max_stages, mode, dropout, angle,
                             stage_ff, self.tie_pair_gate,
                             self.prior_only_angle_scale,
                             self.reverse_odd_blocks and block_index % 2 == 1,
                             self.interstage_head_shuffle,
                             self.repeat_pair_switch, gate_anchor,
                             self.shuffle_exempt_heads)
            for block_index in range(blocks)
        )
        # A shared nodewise nonlinearity lets the graph readout observe
        # redistribution among node states.  Both exchange settings use it.
        self.node_readout = nn.Sequential(
            nn.LayerNorm(dim), nn.Linear(dim, dim), nn.GELU(),
            nn.Linear(dim, dim),
        )
        self.head = nn.Sequential(nn.LayerNorm(dim), nn.Linear(dim, output_dim))
        self.node_head = nn.Sequential(nn.LayerNorm(dim), nn.Linear(dim, output_dim))
        self._mapping_cache: dict[tuple[int, bytes], torch.Tensor] = {}

    @property
    def compiled_graphs(self) -> int:
        return len(self._mapping_cache)

    def _node_features(self, x: torch.Tensor) -> torch.Tensor:
        if not x.is_floating_point() and x.ndim == 2 and x.shape[1] == 9 \
                and self.atom_encoder is not None:
            return self.atom_encoder(x.long())
        return self.node_encoder(x.float())

    def _edge_features(self, edge_attr: torch.Tensor | None) -> torch.Tensor | None:
        if edge_attr is None or self.edge_encoder is None:
            return None
        if not edge_attr.is_floating_point() and edge_attr.ndim == 2 \
                and edge_attr.shape[1] == 3 and self.bond_encoder is not None:
            return self.bond_encoder(edge_attr.long())
        return self.edge_encoder(edge_attr.float())

    def _compile_mapping(self, edge_index: torch.Tensor,
                         nodes: int) -> torch.Tensor:
        if nodes == 1:
            return torch.zeros(1, dtype=torch.long)
        cpu_edges = edge_index.contiguous()
        key = (nodes, cpu_edges.numpy().tobytes())
        mapping = self._mapping_cache.get(key)
        if mapping is None:
            mapping, _, _ = compile_interfaces(
                cpu_edges, nodes, layout=self.layout, alignment=False,
                seed=self.seed,
            )
            self._mapping_cache[key] = mapping
        return mapping

    @staticmethod
    def _validate_mapping(mapping: torch.Tensor, nodes: int,
                          padded: int, device: torch.device) -> torch.Tensor:
        mapping = torch.as_tensor(mapping, dtype=torch.long, device=device)
        if mapping.ndim != 1 or len(mapping) != nodes:
            raise ValueError("Each mapping must contain one wire per graph node")
        if nodes and (bool((mapping < 0).any()) or bool((mapping >= padded).any())
                      or mapping.unique().numel() != nodes):
            raise ValueError("Mapping must be injective and within the padded wires")
        return mapping

    def _pool_groups(self, values: torch.Tensor, graph_ids: torch.Tensor,
                     graphs: int) -> torch.Tensor:
        pooled = values.new_zeros((graphs, values.shape[-1]))
        pooled.index_add_(0, graph_ids, values)
        if self.pool == "mean":
            counts = torch.bincount(graph_ids, minlength=graphs).clamp_min(1)
            pooled = pooled / counts.unsqueeze(-1)
        return pooled

    def encode_nodes(self, batch, mappings: Sequence[torch.Tensor] | None = None,
                     return_diagnostics: bool = False,
                     pair_angle_scales: torch.Tensor | None = None):
        """Return all final real-node states in the incoming PyG node order."""
        ptr = getattr(batch, "ptr", None)
        if ptr is None:
            ptr = torch.tensor([0, batch.x.shape[0]], device=batch.x.device)
        graph_count = ptr.numel() - 1
        ptr_values = ptr.detach().cpu().tolist()
        if mappings is not None and len(mappings) != graph_count:
            raise ValueError("mappings must have one vector per graph")
        if batch.x.shape[0] == 0 or any(
                ptr_values[g + 1] == ptr_values[g] for g in range(graph_count)):
            raise ValueError("Empty graphs are unsupported")
        edge_index = batch.edge_index
        if edge_index.ndim != 2 or edge_index.shape[0] != 2:
            raise ValueError("edge_index must have shape [2, num_edges]")

        # PyG concatenates each graph's edges.  Copy them once to the CPU for
        # cached topology compilation; all trainable computations stay batched
        # on the original device.
        cpu_edges = edge_index.detach().cpu()
        cpu_ptr = ptr.detach().cpu()
        if cpu_edges.numel():
            edge_graph = torch.bucketize(cpu_edges[0], cpu_ptr[1:], right=True)
            target_graph = torch.bucketize(cpu_edges[1], cpu_ptr[1:], right=True)
            if (not torch.equal(edge_graph, target_graph)
                    or not bool((edge_graph[1:] >= edge_graph[:-1]).all())):
                raise ValueError("PyG Batch edges must stay within and be grouped by graph")
            edge_counts = torch.bincount(edge_graph, minlength=graph_count).tolist()
        else:
            edge_counts = [0] * graph_count
        edge_ptr = [0]
        for count in edge_counts:
            edge_ptr.append(edge_ptr[-1] + count)

        padded_sizes = [
            1 << (ptr_values[g + 1] - ptr_values[g] - 1).bit_length()
            for g in range(graph_count)
        ]
        padded = max(padded_sizes)
        local_mappings = []
        for graph in range(graph_count):
            nodes = ptr_values[graph + 1] - ptr_values[graph]
            edge_start, edge_end = edge_ptr[graph], edge_ptr[graph + 1]
            local_edges = cpu_edges[:, edge_start:edge_end] - ptr_values[graph]
            mapping = (self._compile_mapping(local_edges, nodes)
                       if mappings is None else mappings[graph])
            local_mappings.append(self._validate_mapping(
                mapping, nodes, padded_sizes[graph], torch.device("cpu"),
            ))
        wire_ids = torch.cat([
            local_mappings[g] + g * padded for g in range(graph_count)
        ]).to(device=batch.x.device, non_blocking=True)
        real_mask = torch.zeros((graph_count * padded,), dtype=torch.bool,
                                device=batch.x.device)
        real_mask[wire_ids] = True
        real_mask = real_mask.reshape(graph_count, padded)
        relay_mask = None
        if self.relay_padding:
            # Batch-wide padding beyond each graph's own power-of-two line is
            # not part of that graph's compiled circuit.
            local_sizes = torch.tensor(padded_sizes, device=batch.x.device)
            relay_mask = (torch.arange(padded, device=batch.x.device)[None, :]
                          < local_sizes[:, None])
        active_stages = torch.tensor(
            [size.bit_length() - 1 for size in padded_sizes],
            dtype=torch.long, device=batch.x.device,
        )
        if pair_angle_scales is not None:
            pair_angle_scales = torch.as_tensor(
                pair_angle_scales, dtype=batch.x.dtype, device=batch.x.device)
        graph_ids = torch.repeat_interleave(
            torch.arange(graph_count, device=batch.x.device),
            torch.tensor([ptr_values[g + 1] - ptr_values[g]
                          for g in range(graph_count)], device=batch.x.device),
            output_size=batch.x.shape[0],
        )

        # Static directed graphs may cache this input-only normalization when
        # preparing each graph. It is identical to recomputing the degrees on
        # every forward pass, but avoids repeated GPU degree reductions.
        precomputed = (getattr(batch, "directed_edge_weight", None)
                       if self.edge_weight_mode == "directed_symmetric" else None)
        if precomputed is not None:
            if precomputed.shape != (edge_index.shape[1],):
                raise ValueError("Invalid precomputed directed edge weights")
            weights = precomputed
        else:
            weights = _edge_weights(edge_index, batch.x.shape[0],
                                    self.edge_weight_mode)
        x = F.dropout(self._node_features(batch.x), self.dropout, self.training)
        edge_values = self._edge_features(getattr(batch, "edge_attr", None))
        if edge_values is not None:
            edge_values = F.dropout(edge_values, self.dropout, self.training)
        if self.graph_embed:
            x = x + _aggregate(self.neighbor_encoder(x), edge_index, weights,
                               edge_values, x.shape[0])
        x = self.input_norm(x)
        z = x.new_zeros((graph_count * padded, self.dim))
        z = z.index_copy(0, wire_ids, x).reshape(graph_count, padded, self.dim)

        before_all = []
        after_all = []
        read_before_all = []
        read_after_all = []
        for block in self.blocks:
            z, before, after = block(
                z, wire_ids, real_mask, active_stages, edge_index, weights,
                edge_values, self.exchange_scale, self.graph_refresh,
                pair_angle_scales, relay_mask,
            )
            if return_diagnostics:
                before_all.append(self._pool_groups(before, graph_ids, graph_count).detach())
                after_all.append(self._pool_groups(after, graph_ids, graph_count).detach())
                read_before_all.append(self._pool_groups(
                    self.node_readout(before), graph_ids, graph_count).detach())
                read_after_all.append(self._pool_groups(
                    self.node_readout(after), graph_ids, graph_count).detach())
        states = z.reshape(graph_count * padded, self.dim).index_select(0, wire_ids)
        if not return_diagnostics:
            return states

        pair_counts = x.new_zeros((graph_count, self.max_stages))
        participating_mask = real_mask if relay_mask is None else relay_mask
        for level in range(min(padded.bit_length() - 1, self.max_stages)):
            stride = 1 << level
            mask = participating_mask.reshape(
                graph_count, padded // (2 * stride), 2, stride)
            pair_counts[:, level] = (mask[:, :, 0] & mask[:, :, 1]).sum(dim=(1, 2))
        empty = x.new_empty((graph_count, 0, self.dim))
        diagnostics = {
            "pooled_before": torch.stack(before_all, dim=1) if before_all else empty,
            "pooled_after": torch.stack(after_all, dim=1) if after_all else empty,
            "readout_pooled_before": (torch.stack(read_before_all, dim=1)
                                      if read_before_all else empty),
            "readout_pooled_after": (torch.stack(read_after_all, dim=1)
                                     if read_after_all else empty),
            "pooled_final": self._pool_groups(states, graph_ids, graph_count).detach(),
            "active_pairs_per_stage": pair_counts,
        }
        return states, diagnostics

    def forward(self, batch, mappings: Sequence[torch.Tensor] | None = None,
                return_diagnostics: bool = False,
                pair_angle_scales: torch.Tensor | None = None):
        encoded = self.encode_nodes(batch, mappings, return_diagnostics,
                                    pair_angle_scales)
        if return_diagnostics:
            states, diagnostics = encoded
        else:
            states = encoded
        ptr = getattr(batch, "ptr", None)
        if ptr is None:
            ptr = torch.tensor([0, states.shape[0]], device=states.device)
        graph_count = ptr.numel() - 1
        graph_ids = getattr(batch, "batch", None)
        if graph_ids is None:
            graph_ids = torch.zeros(states.shape[0], dtype=torch.long,
                                    device=states.device)
        pooled = self._pool_groups(self.node_readout(states),
                                   graph_ids, graph_count)
        prediction = self.head(pooled)
        return (prediction, diagnostics) if return_diagnostics else prediction

    def predict_nodes(self, batch, mappings: Sequence[torch.Tensor] | None = None,
                      pair_angle_scales: torch.Tensor | None = None):
        return self.node_head(self.encode_nodes(
            batch, mappings, pair_angle_scales=pair_angle_scales))
