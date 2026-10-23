"""Loom: complete persistent-wire architecture from the manuscript."""
from __future__ import annotations

import math
import warnings
import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as sla
import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.checkpoint import checkpoint

from .architecture import _spectral_order
from .hamming_compiler import compiler_adjacency, refine_loom_mapping
from .pure import PhasePair
from .layers import rotate90


def compile_loom(edge_index, nodes, seed=0):
    """Fiedler ranks -> Gray addresses -> swaps and vacant-wire moves."""
    edges = edge_index.detach().cpu().numpy()
    a = compiler_adjacency(sp.csr_matrix(
        (np.ones(edges.shape[1]), (edges[0], edges[1])), shape=(nodes, nodes)))
    count, labels = sp.csgraph.connected_components(a, directed=False)
    components = [np.flatnonzero(labels == i) for i in range(count)]
    components.sort(key=lambda ids: (-len(ids), int(ids[0])))
    orders = []
    for ids in components:
        sub = a[ids][:, ids].astype(float)
        if len(ids) <= 12000:
            order = _spectral_order(sub, allow_symmetric=True)
        else:
            # Iterative Fiedler solve avoids the fill-in of a large sparse LU.
            lap = sp.csgraph.laplacian(sub).tocsr()
            degree = np.asarray(sub.sum(1)).ravel()
            rng = np.random.default_rng(seed)
            x = rng.standard_normal((len(ids), 1))
            pre = sp.diags(1. / np.maximum(degree, 1.))
            with warnings.catch_warnings():
                warnings.simplefilter('ignore', UserWarning)
                values, vectors, residuals = sla.lobpcg(
                    lap, x, Y=np.ones((len(ids), 1)), M=pre,
                    largest=False, tol=1e-5, maxiter=1000,
                    retResidualNormsHistory=True)
            axis = vectors[:, 0]
            if axis @ (degree - degree.mean()) > 0:
                axis = -axis
            order = np.argsort(axis, kind='stable')
        orders.append(ids[order])
    order = np.concatenate(orders)
    ranks = np.arange(nodes, dtype=np.int64)
    mapping = np.empty(nodes, dtype=np.int64)
    mapping[order] = ranks ^ (ranks >> 1)
    padded = 1 << max(0, (nodes - 1).bit_length())
    mapping, audit = refine_loom_mapping(a, mapping, padded, seed=seed)
    return torch.from_numpy(mapping), audit


def normalized_neighbors(x, edge_index, directed=False):
    """Zero-self-loop one-step normalization; no hidden-feature refresh."""
    src, dst = edge_index
    keep = src != dst
    src, dst = src[keep], dst[keep]
    n = len(x)
    out = torch.bincount(src, minlength=n).to(x.dtype).clamp_min(1.)
    incoming = torch.bincount(dst, minlength=n).to(x.dtype).clamp_min(1.)
    weight = (out[src] * incoming[dst]).rsqrt()
    a = torch.sparse_coo_tensor(torch.stack((dst, src)), weight, (n, n)).coalesce()
    return torch.sparse.mm(a, x)


def controller_linear(layer,value):
    """Split rows so controller gradients use more GPU blocks."""
    rows=value.numel()//value.shape[-1]
    if value.is_cuda and torch.is_grad_enabled() and 4096<=rows<=32768 and rows%4096==0:
        matrix=value.reshape(-1,4096,value.shape[-1])
        weight=layer.weight.T.unsqueeze(0).expand(len(matrix),-1,-1)
        output=torch.bmm(matrix,weight)
        if layer.bias is not None:
            output=output+layer.bias
        return output.reshape(*value.shape[:-1],layer.out_features)
    return layer(value)


class LoomPair(PhasePair):
    """Algebraically fuse the three lifting updates with the same coefficients."""
    def __init__(self, dim, heads, angle):
        super().__init__(dim, heads, 'lifting', angle, gate_normalize=True)

    def forward(self, u, v):
        shape=u.shape
        difference=u-v
        score=(controller_linear(self.gate,torch.cat((u+v,difference.abs()),-1))+
               controller_linear(self.direction_gate,difference))/self.gate_scale
        theta=self.base+.5*score.tanh()
        t=(.5*theta).clamp(-1.25,1.25).tan().unsqueeze(-1)
        s=(theta.sin()+self.middle_delta.tanh()).unsqueeze(-1)
        u=u.reshape(*shape[:-1],self.heads,self.width)
        v=v.reshape_as(u)
        retained=1.-t*s
        return ((retained*u+(2.*t-t*t*s)*rotate90(v)).reshape(shape),
                (retained*v+s*rotate90(u)).reshape(shape))


class LoomBlock(nn.Module):
    def __init__(self, dim, heads, stages, dropout, angle):
        super().__init__()
        self.norm_mix = nn.LayerNorm(dim)
        self.checkpoint_pairs=False
        self.indexed_execution=False
        self.cuda_stage_execution=False
        self.pairs = nn.ModuleList(LoomPair(dim, heads, angle) for _ in range(stages))
        self.projection = nn.Linear(dim, dim, bias=False)
        self.norm_ff = nn.LayerNorm(dim)
        self.ff = nn.Sequential(nn.Linear(dim, 2 * dim), nn.GELU(),
                                nn.Dropout(dropout), nn.Linear(2 * dim, dim))
        self.drop = nn.Dropout(dropout)

    def mix(self, z, indexed=None):
        if indexed is None:
            indexed=self.indexed_execution
        n, d = z.shape[-2:]
        for stage in range(n.bit_length() - 1):
            stride = 1 << stage
            if self.cuda_stage_execution and not indexed:
                from .loom_cuda import supports, stage as cuda_stage
                if supports(z,self.pairs[stage]):
                    if self.checkpoint_pairs and self.training:
                        z=checkpoint(cuda_stage,z,self.pairs[stage],stride,use_reentrant=False,preserve_rng_state=False)
                    else:
                        z=cuda_stage(z,self.pairs[stage],stride)
                    continue
            if indexed:
                ids = torch.arange(n // 2, device=z.device)
                left = (ids // stride) * (2 * stride) + ids % stride
                right = left + stride
                u,v=z.index_select(-2,left),z.index_select(-2,right)
                if self.checkpoint_pairs and self.training:
                    u,v=checkpoint(self.pairs[stage],u,v,use_reentrant=False,preserve_rng_state=False)
                else:
                    u,v=self.pairs[stage](u,v)
                output = torch.empty_like(z)
                output.index_copy_(-2, left, u)
                output.index_copy_(-2, right, v)
                z = output
            else:
                grouped = z.reshape(*z.shape[:-2], n // (2 * stride), 2, stride, d)
                if self.checkpoint_pairs and self.training:
                    u,v=checkpoint(self.pairs[stage],grouped[...,0,:,:],grouped[...,1,:,:],use_reentrant=False,preserve_rng_state=False)
                else:
                    u, v = self.pairs[stage](grouped[..., 0, :, :], grouped[..., 1, :, :])
                z = torch.stack((u, v), dim=-3).reshape_as(z)
        return z

    def forward(self, z):
        z = z + self.drop(self.projection(self.mix(self.norm_mix(z))))
        return z + self.drop(self.ff(self.norm_ff(z)))


class Loom(nn.Module):
    """All stages and padded states persist across all blocks.

    Preaggregating raw X is exactly A(X Wg) by associativity; it saves the
    graph sparse multiply on every optimizer step when inputs are static.
    Batched graphs supply a flat node-to-wire map and graph IDs.
    """
    def __init__(self, input_dim, output_dim, max_nodes, dim=64, heads=4,
                 blocks=4, dropout=.1, angle=.2, checkpoint_blocks=False):
        super().__init__()
        self.dim = dim
        self.stages = max(0, (max_nodes - 1).bit_length())
        self.checkpoint_blocks = checkpoint_blocks
        self.self_projection = nn.Linear(input_dim, dim)
        self.graph_projection = nn.Linear(input_dim, dim, bias=False)
        self.embed_norm = nn.LayerNorm(dim)
        self.blocks = nn.ModuleList(LoomBlock(dim, heads, self.stages, dropout, angle)
                                    for _ in range(blocks))
        for block in self.blocks:
            block.checkpoint_pairs=checkpoint_blocks
        self.head_norm = nn.LayerNorm(dim)
        self.head = nn.Linear(dim, output_dim)

    @staticmethod
    def project(layer, value):
        if isinstance(value, tuple):
            matrix, offset, scale = value
            return layer(matrix) + scale * F.linear(offset, layer.weight)
        return layer(value)

    def enable_cuda_stages(self,enabled=True):
        """Opt in to exact FP32 stage kernels; install CuPy for this CUDA path."""
        for block in self.blocks:
            block.cuda_stage_execution=enabled
        return self

    def forward(self, x, neighbors, mapping, graph_ids=None, padded=None):
        h = self.embed_norm(self.project(self.self_projection,x) + self.project(self.graph_projection,neighbors))
        if graph_ids is None:
            padded = padded or 1 << max(0, (len(mapping) - 1).bit_length())
            z = h.new_zeros(padded, self.dim).index_copy(0, mapping, h)
            for block in self.blocks:
                z = checkpoint(block, z, use_reentrant=False,preserve_rng_state=block.drop.p>0) if self.checkpoint_blocks and self.training else block(z)
            restored = z[mapping]
        else:
            if padded is None:
                raise ValueError('Batched graphs need their padded size')
            graphs = int(graph_ids[-1]) + 1
            flat_map = graph_ids * padded + mapping
            z = h.new_zeros(graphs * padded, self.dim).index_copy(0, flat_map, h)
            z = z.reshape(graphs, padded, self.dim)
            for block in self.blocks:
                z = checkpoint(block, z, use_reentrant=False,preserve_rng_state=block.drop.p>0) if self.checkpoint_blocks and self.training else block(z)
            restored = z.reshape(-1, self.dim)[flat_map]
        return self.head(self.head_norm(restored))

    def encode_packed(self, x, neighbors, real_mask):
        h = self.embed_norm(self.self_projection(x) + self.graph_projection(neighbors))
        z = h * real_mask.unsqueeze(-1)
        for block in self.blocks:
            z = checkpoint(block, z, use_reentrant=False,preserve_rng_state=block.drop.p>0) if self.checkpoint_blocks and self.training else block(z)
        return z

    def forward_packed(self, x, neighbors, real_mask):
        """Preplaced batches; zero padding only at initial placement."""
        return self.head(self.head_norm(self.encode_packed(x,neighbors,real_mask)))
