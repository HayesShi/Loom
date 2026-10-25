"""Parallel XOR pairings on a fixed power-of-two wire layout."""

import torch
from torch import nn
from .arco import ARCO


def butterfly_pairs(num_wires: int, stage: int, device=None):
    """Enumerate each pair once, with bit `stage` zero at the left port."""
    if num_wires < 1 or num_wires & (num_wires - 1):
        raise ValueError("num_wires must be a power of two")
    if not 0 <= stage < num_wires.bit_length() - 1:
        raise ValueError("stage is outside the butterfly schedule")
    stride = 1 << stage
    ids = torch.arange(num_wires // 2, device=device)
    left = (ids // stride) * (2 * stride) + ids % stride
    return left, left ^ stride


class ButterflyMixer(nn.Module):
    """Separate ARCO parameters at each stage, shared across all its pairs."""

    def __init__(self, dim: int, heads: int, stages: int, angle: float = 0.2):
        super().__init__()
        self.pairs = nn.ModuleList(ARCO(dim, heads, angle) for _ in range(stages))

    def forward(self, states: torch.Tensor):
        num_wires, dim = states.shape[-2:]
        if num_wires < 1 or num_wires & (num_wires - 1):
            raise ValueError("The wire dimension must be a power of two")
        stages = num_wires.bit_length() - 1
        if stages > len(self.pairs):
            raise ValueError("The model was initialized for fewer wires")
        for stage in range(stages):
            stride = 1 << stage
            groups = states.reshape(
                *states.shape[:-2], num_wires // (2 * stride), 2, stride, dim
            )
            u, v = self.pairs[stage](groups[..., 0, :, :], groups[..., 1, :, :])
            states = torch.stack((u, v), dim=-3).reshape_as(states)
        return states
