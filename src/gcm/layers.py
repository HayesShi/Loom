from __future__ import annotations

import math

import torch
from torch import nn
from torch.nn import functional as F


def rotate90(x: torch.Tensor) -> torch.Tensor:
    if x.shape[-1] % 2:
        raise ValueError("feature dimension must be even")
    pairs = x.reshape(*x.shape[:-1], -1, 2)
    return torch.stack((-pairs[..., 1], pairs[..., 0]), dim=-1).flatten(-2)


class ContentControlledPairRotation(nn.Module):
    def __init__(
        self,
        dim: int,
        heads: int = 1,
        content_gate: bool = True,
        angle_range: float = 0.5,
    ) -> None:
        super().__init__()
        if dim % heads or (dim // heads) % 2:
            raise ValueError("each head dimension must be even")
        self.dim = dim
        self.heads = heads
        self.head_dim = dim // heads
        self.content_gate = content_gate
        self.angle_range = angle_range
        self.base_angle = nn.Parameter(torch.zeros(heads))
        if content_gate:
            self.sum_weight = nn.Parameter(torch.empty(heads, self.head_dim))
            self.diff_weight = nn.Parameter(torch.empty(heads, self.head_dim))
            nn.init.normal_(self.sum_weight, std=0.02)
            nn.init.normal_(self.diff_weight, std=0.02)

    def angles(self, u: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
        leading = u.shape[:-1]
        uh = u.reshape(*leading, self.heads, self.head_dim)
        vh = v.reshape(*leading, self.heads, self.head_dim)
        theta = self.base_angle
        if self.content_gate:
            score = ((uh + vh) * self.sum_weight).sum(-1)
            score = score + ((uh - vh).abs() * self.diff_weight).sum(-1)
            theta = theta + self.angle_range * torch.tanh(score / math.sqrt(self.head_dim))
        return theta

    def forward(self, u: torch.Tensor, v: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        leading = u.shape[:-1]
        uh = u.reshape(*leading, self.heads, self.head_dim)
        vh = v.reshape(*leading, self.heads, self.head_dim)
        theta = self.angles(u, v)
        c, s = theta.cos().unsqueeze(-1), theta.sin().unsqueeze(-1)
        out_u = c * uh + s * rotate90(vh)
        out_v = c * vh + s * rotate90(uh)
        return out_u.flatten(-2), out_v.flatten(-2)


class GatedPairMixer(nn.Module):
    """A generic two-output matched control for the rotation primitive."""

    def __init__(self, dim: int) -> None:
        super().__init__()
        self.gate = nn.Sequential(nn.Linear(2 * dim, dim), nn.GELU(), nn.Linear(dim, dim))
        nn.init.zeros_(self.gate[-1].weight)
        nn.init.zeros_(self.gate[-1].bias)

    def forward(self, u: torch.Tensor, v: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        gate = torch.sigmoid(self.gate(torch.cat((u + v, (u - v).abs()), dim=-1)))
        return gate * u + (1 - gate) * v, gate * v + (1 - gate) * u


class MeanPairMixer(nn.Module):
    def forward(self, u: torch.Tensor, v: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        mean = 0.5 * (u + v)
        return mean, mean


class IdentityPairMixer(nn.Module):
    def forward(self, u: torch.Tensor, v: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        return u, v


class ButterflyMixer(nn.Module):
    def __init__(
        self,
        dim: int,
        num_nodes: int,
        heads: int = 1,
        operator: str = "rotation",
        angle_range: float = 0.5,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        self.num_nodes = num_nodes
        self.padded_nodes = 1 << (num_nodes - 1).bit_length()
        self.num_stages = int(math.log2(self.padded_nodes))
        self.dropout = dropout
        stages: list[nn.Module] = []
        for _ in range(self.num_stages):
            if operator == "rotation":
                stage = ContentControlledPairRotation(dim, heads, True, angle_range)
            elif operator == "fixed_rotation":
                stage = ContentControlledPairRotation(dim, heads, False, angle_range)
            elif operator == "gated":
                stage = GatedPairMixer(dim)
            elif operator == "mean":
                stage = MeanPairMixer()
            elif operator == "identity":
                stage = IdentityPairMixer()
            else:
                raise ValueError(f"unknown pair operator: {operator}")
            stages.append(stage)
        self.stages = nn.ModuleList(stages)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.shape[-2] != self.num_nodes:
            raise ValueError(f"expected {self.num_nodes} nodes, got {x.shape[-2]}")
        if self.padded_nodes > self.num_nodes:
            x = F.pad(x, (0, 0, 0, self.padded_nodes - self.num_nodes))
        dim = x.shape[-1]
        batch_shape = x.shape[:-2]
        for stage_idx, stage in enumerate(self.stages):
            stride = 1 << stage_idx
            blocks = self.padded_nodes // (2 * stride)
            grouped = x.reshape(*batch_shape, blocks, 2, stride, dim)
            u, v = grouped[..., 0, :, :], grouped[..., 1, :, :]
            out_u, out_v = stage(u, v)
            x = torch.stack((out_u, out_v), dim=-3).reshape(*batch_shape, self.padded_nodes, dim)
            x = F.dropout(x, p=self.dropout, training=self.training)
        return x[..., : self.num_nodes, :]
