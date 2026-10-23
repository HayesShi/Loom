"""Feature-canonical Gray wiring for periodic square-grid graphs.

The graph supplies the grid but cannot choose a unique origin or orientation:
a torus has translations, reflections, and an axis swap. For LRIM's observed
binary spins, local color refinement usually yields a unique feature anchor.
The eight frames around it are then ordered by the full observed spin pattern.
No node IDs, target labels, or supplied coordinates enter that choice.

If the observed features retain a symmetry, an equivariant bijection to fixed
wires cannot be selected. This compiler raises instead of using node IDs as a
hidden tie breaker. It is specific to even power-of-two square tori and binary
spin features, and leaves the downstream butterfly and J switches unchanged.
"""

from __future__ import annotations

import math
from collections import Counter

import torch


class TorusFeatureAmbiguity(ValueError):
    """Observed features do not determine a unique torus chart."""


class TorusFeatureGrayCompiler:
    """Compile many feature instances on one fixed torus topology."""

    def __init__(self, edge_index: torch.Tensor, num_nodes: int,
                 max_refinement: int = 8):
        side = math.isqrt(num_nodes)
        if side * side != num_nodes or side < 8 or side & (side - 1):
            raise ValueError("Expected a square torus with power-of-two side >= 8")
        if max_refinement < 0:
            raise ValueError("max_refinement must be nonnegative")
        edges = torch.as_tensor(edge_index, dtype=torch.long).detach().cpu()
        if edges.ndim != 2 or edges.shape[0] != 2:
            raise ValueError("edge_index must have shape [2, edges]")
        neighbors = [set() for _ in range(num_nodes)]
        for source, target in edges.T.tolist():
            if source < 0 or source >= num_nodes or target < 0 or target >= num_nodes:
                raise ValueError("edge_index contains an out-of-range node")
            if source == target:
                raise ValueError("A periodic grid cannot have self-loops")
            neighbors[source].add(target)
            neighbors[target].add(source)
        if any(len(group) != 4 for group in neighbors):
            raise ValueError("Expected degree four at every torus node")
        self.side = side
        self.num_nodes = num_nodes
        self.max_refinement = max_refinement
        self.neighbors = tuple(neighbors)
        self._orientation_positions: list[tuple[int, ...]] = []
        self._orientation_wires: list[tuple[int, ...]] = []
        gray = tuple(value ^ (value >> 1) for value in range(side))
        for swap in (False, True):
            for sign_x in (1, -1):
                for sign_y in (1, -1):
                    positions = []
                    wires = []
                    for slot in range(num_nodes):
                        row, col = divmod(slot, side)
                        first, second = (col, row) if swap else (row, col)
                        first = (sign_x * first) % side
                        second = (sign_y * second) % side
                        positions.append(first * side + second)
                        wires.append(side * gray[first] + gray[second])
                    self._orientation_positions.append(tuple(positions))
                    self._orientation_wires.append(tuple(wires))

        # Validate the topology once. The instance can then reuse its
        # adjacency across all 10,000 LRIM feature instances.
        origin = 0
        first = min(self.neighbors[origin])
        perpendicular = self._perpendicular(origin, first)
        self._chart(origin, first, min(perpendicular), validate=True)

    def _perpendicular(self, origin: int, first: int) -> list[int]:
        candidates = [
            node for node in self.neighbors[origin]
            if node != first and len(self.neighbors[first] & self.neighbors[node]) == 2
        ]
        if len(candidates) != 2:
            raise ValueError("Grid axes cannot be identified from four-cycles")
        return candidates

    def _straight(self, previous: int, current: int) -> int:
        candidates = [
            node for node in self.neighbors[current]
            if node != previous
            and len(self.neighbors[previous] & self.neighbors[node]) == 1
        ]
        if len(candidates) != 1:
            raise ValueError("Grid direction cannot be propagated uniquely")
        return candidates[0]

    def _chart(self, origin: int, first: int, perpendicular: int,
               validate: bool = False) -> list[int]:
        side = self.side
        x_line = [origin, first]
        while len(x_line) < side:
            x_line.append(self._straight(x_line[-2], x_line[-1]))
        if self._straight(x_line[-2], x_line[-1]) != origin:
            raise ValueError("First torus axis does not close")

        y_starts = [perpendicular]
        for index in range(1, side):
            completion = (self.neighbors[x_line[index]]
                          & self.neighbors[y_starts[-1]]) - {x_line[index - 1]}
            if len(completion) != 1:
                raise ValueError("Grid square cannot be completed uniquely")
            y_starts.append(next(iter(completion)))

        chart = []
        for index in range(side):
            y_line = [x_line[index], y_starts[index]]
            while len(y_line) < side:
                y_line.append(self._straight(y_line[-2], y_line[-1]))
            if self._straight(y_line[-2], y_line[-1]) != x_line[index]:
                raise ValueError("Second torus axis does not close")
            chart.extend(y_line)
        if len(set(chart)) != self.num_nodes:
            raise ValueError("Reconstructed chart is not a node permutation")
        if validate:
            coordinates = {node: divmod(slot, side) for slot, node in enumerate(chart)}
            for node in range(self.num_nodes):
                row, col = coordinates[node]
                expected = {
                    ((row + 1) % side, col), ((row - 1) % side, col),
                    (row, (col + 1) % side), (row, (col - 1) % side),
                }
                if {coordinates[neighbor] for neighbor in self.neighbors[node]} != expected:
                    raise ValueError("Graph is not a periodic square grid")
        return chart

    def _spins(self, x: torch.Tensor) -> list[int]:
        observed = torch.as_tensor(x).detach().cpu()
        if observed.shape != (self.num_nodes, 1):
            raise ValueError("Expected one observed spin per node, shape [nodes, 1]")
        values = observed.reshape(-1).tolist()
        if any(value not in (-1, 1) for value in values):
            raise ValueError("This compiler expects binary spins encoded as -1 and +1")
        return [int(value > 0) for value in values]

    def _feature_anchor(self, spins: list[int]) -> tuple[int, int]:
        colors = spins
        for radius in range(self.max_refinement + 1):
            sizes = Counter(colors)
            unique_colors = [color for color, size in sizes.items() if size == 1]
            if unique_colors:
                return colors.index(min(unique_colors)), radius
            signatures = [
                (colors[node], tuple(sorted(colors[neighbor]
                                            for neighbor in self.neighbors[node])))
                for node in range(self.num_nodes)
            ]
            vocabulary = {signature: index for index, signature
                          in enumerate(sorted(set(signatures)))}
            colors = [vocabulary[signature] for signature in signatures]
        raise TorusFeatureAmbiguity(
            f"No unique feature anchor within {self.max_refinement} refinements")

    def compile_with_radius(self, x: torch.Tensor) -> tuple[torch.Tensor, int]:
        """Return node-to-wire mapping and feature-anchor refinement radius."""
        spins = self._spins(x)
        anchor, radius = self._feature_anchor(spins)
        first = min(self.neighbors[anchor])
        perpendicular = min(self._perpendicular(anchor, first))
        chart = self._chart(anchor, first, perpendicular)

        best_score: bytes | None = None
        best_orientation = -1
        tie = False
        for orientation, positions in enumerate(self._orientation_positions):
            ordered = bytearray(self.num_nodes)
            for slot, node in enumerate(chart):
                ordered[positions[slot]] = spins[node]
            score = bytes(ordered)
            if best_score is None or score < best_score:
                best_score = score
                best_orientation = orientation
                tie = False
            elif score == best_score:
                tie = True
        if tie:
            raise TorusFeatureAmbiguity(
                "Observed spins do not distinguish the eight torus orientations")

        mapping = torch.empty(self.num_nodes, dtype=torch.long)
        wires = self._orientation_wires[best_orientation]
        for slot, node in enumerate(chart):
            mapping[node] = wires[slot]
        return mapping, radius

    def compile(self, x: torch.Tensor) -> torch.Tensor:
        return self.compile_with_radius(x)[0]

    def _wire_spin_bytes(self, x: torch.Tensor,
                         mapping: torch.Tensor) -> bytes:
        """Encode the observed spins in the compiled Gray wire order."""
        spins = self._spins(x)
        ordered = bytearray(self.num_nodes)
        for node, wire in enumerate(mapping.tolist()):
            ordered[wire] = spins[node]
        return bytes(ordered)

    def compile_spin_flip_gauge_with_radius(
            self, x: torch.Tensor) -> tuple[torch.Tensor, int, bool]:
        """Choose one chart and spin sign for the global-flip orbit.

        The two candidate charts are compiled independently from observed
        features. Comparing their complete wire states, rather than a node ID
        or a graph-level summary, preserves both graph relabeling equivariance
        and the exact x -> -x gauge symmetry. A tied canonical state cannot
        select a sign equivariantly, so it is rejected explicitly.
        """
        positive_mapping, positive_radius = self.compile_with_radius(x)
        negative_x = -torch.as_tensor(x)
        negative_mapping, negative_radius = self.compile_with_radius(negative_x)
        positive_score = self._wire_spin_bytes(x, positive_mapping)
        negative_score = self._wire_spin_bytes(negative_x, negative_mapping)
        if positive_score == negative_score:
            raise TorusFeatureAmbiguity(
                "Spin-flip candidates have the same canonical wire state")
        if negative_score < positive_score:
            return negative_mapping, negative_radius, True
        return positive_mapping, positive_radius, False


def compile_torus_feature_gray(edge_index: torch.Tensor,
                               x: torch.Tensor) -> torch.Tensor:
    """One-shot feature-canonical torus compilation."""
    return TorusFeatureGrayCompiler(edge_index, x.shape[0]).compile(x)
