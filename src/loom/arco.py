"""Adaptive rotary coupling with two persistent output states."""

import math
import torch
from torch import nn


def rotate90(x: torch.Tensor) -> torch.Tensor:
    """Apply J to adjacent feature pairs: (a, b) -> (-b, a)."""
    pairs = x.reshape(*x.shape[:-1], -1, 2)
    return torch.stack((-pairs[..., 1], pairs[..., 0]), dim=-1).flatten(-2)


class ARCO(nn.Module):
    """Content-controlled three-update coupling (Section III-E).

    Inputs and outputs have shape [..., dim]. Each head has even width.
    The fused expression equals u1=u+tJv; v'=v+sJu1; u'=u1+tJv'.
    """

    def __init__(self, dim: int, heads: int, angle: float = 0.2):
        super().__init__()
        if heads < 1 or dim < 1 or dim % (2 * heads):
            raise ValueError("dim must be divisible by twice the number of heads")
        self.heads, self.width = heads, dim // heads
        self.gate_scale = math.sqrt(dim)
        self.base = nn.Parameter(torch.full((heads,), angle))
        self.gate = nn.Linear(2 * dim, heads)
        nn.init.zeros_(self.gate.weight)
        nn.init.zeros_(self.gate.bias)
        with torch.random.fork_rng(devices=[]):
            self.direction_gate = nn.Linear(dim, heads, bias=False)
        nn.init.zeros_(self.direction_gate.weight)
        self.middle_delta = nn.Parameter(torch.zeros(heads))

    def coefficients(self, u: torch.Tensor, v: torch.Tensor):
        difference = u - v
        shared = torch.cat((u + v, difference.abs()), dim=-1)
        score = (self.gate(shared) + self.direction_gate(difference)) / self.gate_scale
        theta = self.base + 0.5 * score.tanh()
        t = (0.5 * theta).clamp(-1.25, 1.25).tan().unsqueeze(-1)
        s = (theta.sin() + self.middle_delta.tanh()).unsqueeze(-1)
        return t, s

    def forward(self, u: torch.Tensor, v: torch.Tensor):
        shape = u.shape
        t, s = self.coefficients(u, v)
        u = u.reshape(*shape[:-1], self.heads, self.width)
        v = v.reshape_as(u)
        retained = 1.0 - t * s
        left = retained * u + (2.0 * t - t * t * s) * rotate90(v)
        right = retained * v + s * rotate90(u)
        return left.reshape(shape), right.reshape(shape)
