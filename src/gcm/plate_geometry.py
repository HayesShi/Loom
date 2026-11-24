"""Label-free wire compilation for the fixed Plate Deformation mesh.

The official plate has a 29 by 29 grid with five-unit spacing.  Interleaving
the two five-bit Gray coordinates places every mesh point at the same wire in
all simulations, independently of node order. An optional spaced Gray gauge
uses the full 32 by 32 address range, while an ordinary-binary gauge makes
each circuit stage a fixed physical scale. Force nodes
occupy distinct unused wires chosen by their initial positions, optionally
with force-to-mesh contacts in the placement cost.
The returned vector is local node index -> wire index; the model still runs
its ordinary 1024-wire, ten-stage pair circuit.
"""

from __future__ import annotations

from functools import lru_cache

import numpy as np
import torch
from scipy.optimize import linear_sum_assignment

from .architecture import compile_interfaces


GRID_SIDE = 29
GRID_SPACING = 5.0
WIRE_SIDE = 32
WIRES = WIRE_SIDE * WIRE_SIDE
_POPCOUNT = np.asarray([wire.bit_count() for wire in range(WIRES)],
                       dtype=np.float64)


def _gray(value: int) -> int:
    return value ^ (value >> 1)


def _interleaved_gray(x: int, y: int) -> int:
    gx, gy = _gray(x), _gray(y)
    return sum(((gx >> bit) & 1) << (2 * bit) |
               ((gy >> bit) & 1) << (2 * bit + 1)
               for bit in range(5))


def _spaced_coordinate(index: int) -> int:
    """Embed a 29-point physical axis monotonically in 32 Gray addresses."""
    return int(np.rint(index * (WIRE_SIDE - 1) / (GRID_SIDE - 1)))


def _interleaved_spaced_gray(x: int, y: int) -> int:
    return _interleaved_gray(_spaced_coordinate(x), _spaced_coordinate(y))


def _interleaved_binary(x: int, y: int) -> int:
    return sum(((x >> bit) & 1) << (2 * bit) |
               ((y >> bit) & 1) << (2 * bit + 1)
               for bit in range(5))


def _inverse_gray(value: int) -> int:
    result = value
    while value := value >> 1:
        result ^= value
    return result


def _wire_grid_coordinates(wire: int) -> tuple[int, int]:
    gx = sum(((wire >> (2 * bit)) & 1) << bit for bit in range(5))
    gy = sum(((wire >> (2 * bit + 1)) & 1) << bit for bit in range(5))
    return _inverse_gray(gx), _inverse_gray(gy)


def _binary_wire_grid_coordinates(wire: int) -> tuple[int, int]:
    x = sum(((wire >> (2 * bit)) & 1) << bit for bit in range(5))
    y = sum(((wire >> (2 * bit + 1)) & 1) << bit for bit in range(5))
    return x, y


def compile_plate_gray_distance_angle_scales(
        heads: int, carrier_heads: int = 0) -> torch.Tensor:
    """Fixed geometric scales in the existing ten-stage J pair order.

    Only pairs of actual 29x29 mesh wires receive a distance-dependent scale.
    The first ``carrier_heads`` and every pair involving an unused or force
    wire retain the original angle.  No graph target or forcing is consulted.
    """
    if heads < 1 or not 0 <= carrier_heads <= heads:
        raise ValueError("carrier_heads must be between zero and heads")
    coordinates = np.asarray([_wire_grid_coordinates(wire)
                              for wire in range(WIRES)], dtype=np.float64)
    mesh_wire = (coordinates < GRID_SIDE).all(axis=1)
    scales = torch.ones((WIRES.bit_length() - 1, WIRES // 2, heads),
                        dtype=torch.float32)
    for level in range(scales.shape[0]):
        stride = 1 << level
        pair_index = np.arange(WIRES // 2)
        left = (pair_index // stride) * (2 * stride) + pair_index % stride
        right = left + stride
        mesh_pair = mesh_wire[left] & mesh_wire[right]
        distances = np.linalg.norm(
            coordinates[left[mesh_pair]] - coordinates[right[mesh_pair]],
            axis=1) * GRID_SPACING
        median = float(np.median(distances))
        if carrier_heads < heads:
            scales[level, torch.from_numpy(np.flatnonzero(mesh_pair)),
                   carrier_heads:] = torch.from_numpy(
                       (2. * median / (median + distances)).astype(np.float32)
                   ).unsqueeze(-1)
    return scales


@lru_cache(maxsize=4)
def _scaffold_component_wires(edge_bytes: bytes) -> np.ndarray:
    """Compile the canonical 841-node mesh once, without variable force edges."""
    edges = np.frombuffer(edge_bytes, dtype=np.int64).reshape(2, -1).copy()
    mapping, _, _ = compile_interfaces(
        torch.from_numpy(edges), GRID_SIDE ** 2,
        layout="graph_components", alignment=False, seed=0,
    )
    return mapping.numpy()


@lru_cache(maxsize=1)
def _gray_reference():
    mesh_wires = np.asarray([_interleaved_gray(x, y)
                             for x in range(GRID_SIDE) for y in range(GRID_SIDE)])
    free_wires = np.setdiff1d(np.arange(WIRES, dtype=np.int64), mesh_wires, assume_unique=True)
    xy = np.asarray([_wire_grid_coordinates(int(wire)) for wire in free_wires], dtype=np.float64)
    return mesh_wires, free_wires, xy


def compile_plate_reference(graph) -> torch.Tensor:
    """Reuse the canonical Gray mesh and candidate wire coordinates across cases."""
    return compile_plate_gray_contact(graph, reuse_reference=True)


def compile_plate_gray_contact(graph, mesh_layout: str = "gray",
                               force_layout: str = "contact", *,
                               reuse_reference: bool = False) -> torch.Tensor:
    """Compile input-only, injective node-to-wire IDs.

    Each mesh vertex receives a canonical Gray, spaced Gray, binary, or
    scaffold wire from its initial xy position and the selected mesh layout.
    The small force set is assigned jointly to free wires.  ``contact`` uses
    mean hypercube distance to contacted mesh vertices, with initial xy as a
    geometric tiebreak.  ``nearest`` minimizes the total initial xy distance
    alone, retaining the selected mesh wires. Neither mode uses targets
    or split membership.  Forces with identical positions, directions, and
    contacts cannot be distinguished under node relabeling, so their mutual
    ordering is deliberately unspecified.
    """
    node_type = torch.as_tensor(graph.node_type).detach().cpu().numpy()
    pos = torch.as_tensor(graph.pos).detach().cpu().numpy().astype(np.float64)
    edges = torch.as_tensor(graph.edge_index).detach().cpu().numpy()
    if node_type.ndim != 1 or pos.shape != (len(node_type), 3):
        raise ValueError("Plate layout requires node_type [N] and initial pos [N,3]")
    if edges.ndim != 2 or edges.shape[0] != 2:
        raise ValueError("Plate layout requires edge_index [2,E]")
    if not np.isfinite(pos).all() or not np.isin(node_type, [0, 1, 2]).all():
        raise ValueError("Plate layout received nonfinite positions or node types")
    if edges.size and (edges.min() < 0 or edges.max() >= len(node_type)):
        raise ValueError("Plate layout edge index is out of range")

    mesh = np.flatnonzero(node_type != 1)
    forces = np.flatnonzero(node_type == 1)
    if len(mesh) != GRID_SIDE ** 2 or len(forces) < 1:
        raise ValueError("Plate layout requires 841 mesh vertices and force nodes")
    if len(node_type) > WIRES:
        raise ValueError("Plate graph exceeds its 1024-wire circuit")
    if hasattr(graph, "mesh_node_count") and int(graph.mesh_node_count) != len(mesh):
        raise ValueError("Plate mesh_node_count disagrees with node types")

    grid_float = pos[mesh, :2] / GRID_SPACING
    grid = np.rint(grid_float).astype(np.int64)
    if (np.abs(grid_float - grid).max() > 1e-3 or
            (grid < 0).any() or (grid >= GRID_SIDE).any() or
            np.abs(pos[mesh, 2]).max() > 1e-3):
        raise ValueError("Plate mesh is not the fixed 29x29 initial xy grid")
    flat_grid = grid[:, 0] * GRID_SIDE + grid[:, 1]
    if len(np.unique(flat_grid)) != GRID_SIDE ** 2:
        raise ValueError("Plate grid coordinates must be unique and complete")

    if mesh_layout not in ("gray", "gray_spaced", "binary",
                           "scaffold_components"):
        raise ValueError("Unknown Plate mesh layout")
    if force_layout not in ("contact", "nearest"):
        raise ValueError("Unknown Plate force layout")
    if reuse_reference and mesh_layout != "gray":
        raise ValueError("The reusable Plate reference requires the Gray mesh layout")
    mapping = np.full(len(node_type), -1, dtype=np.int64)
    if reuse_reference:
        reference, free_wires, candidate_xy = _gray_reference()
        mapping[mesh] = reference[flat_grid]
    elif mesh_layout == "gray":
        mapping[mesh] = [_interleaved_gray(int(x), int(y)) for x, y in grid]
    elif mesh_layout == "gray_spaced":
        mapping[mesh] = [_interleaved_spaced_gray(int(x), int(y))
                         for x, y in grid]
    elif mesh_layout == "binary":
        mapping[mesh] = [_interleaved_binary(int(x), int(y)) for x, y in grid]
    else:
        canonical = np.full(len(node_type), -1, dtype=np.int64)
        canonical[mesh] = flat_grid
        mesh_edges = (node_type[edges[0]] != 1) & (node_type[edges[1]] != 1)
        scaffold_edges = canonical[edges[:, mesh_edges]]
        # Canonical order makes the cache and compiler independent of the
        # incoming PyG node order and the order in which edges were stored.
        scaffold_edges = np.unique(scaffold_edges.T, axis=0).T.copy()
        mapping[mesh] = _scaffold_component_wires(
            scaffold_edges.tobytes())[flat_grid]
    if not reuse_reference:
        free_wires = np.setdiff1d(np.arange(WIRES, dtype=np.int64), mapping[mesh],
                                  assume_unique=True)
    if len(forces) > len(free_wires):
        raise ValueError("Too many force nodes for free plate wires")

    # Official edges point from each force to its contacted mesh vertices.
    # Accept the reverse direction too, so compilation depends on incidence
    # rather than the edge-storage convention.
    is_mesh = node_type != 1
    is_force = node_type == 1
    source, target = edges
    directed = is_force[source] & is_mesh[target]
    reverse = is_mesh[source] & is_force[target]
    contact: dict[int, set[int]] = {int(force): set() for force in forces}
    for force, vertex in zip(source[directed], target[directed]):
        contact[int(force)].add(int(vertex))
    for vertex, force in zip(source[reverse], target[reverse]):
        contact[int(force)].add(int(vertex))
    if any(not incident for incident in contact.values()):
        raise ValueError("Each force needs a contact edge to the plate mesh")

    properties = getattr(graph, "properties", None)
    directions = (np.asarray(torch.as_tensor(properties).detach().cpu())[:, 3:6]
                  if properties is not None else np.zeros((len(node_type), 3)))
    if directions.shape != (len(node_type), 3) or not np.isfinite(directions).all():
        raise ValueError("Plate force properties must have six finite channels")
    # Canonical row order is independent of PyG node order whenever forces
    # can be distinguished by input geometry, direction, or contacts.
    ordered_forces = sorted((int(force) for force in forces), key=lambda force: (
        *pos[force].tolist(), *directions[force].tolist(),
        tuple(sorted(int(mapping[vertex]) for vertex in contact[force])),
    ))
    if not reuse_reference:
        wire_coordinates = (_binary_wire_grid_coordinates if mesh_layout == "binary"
                            else _wire_grid_coordinates)
        candidate_xy = np.asarray([wire_coordinates(int(wire))
                                   for wire in free_wires], dtype=np.float64)
    cost = np.empty((len(forces), len(free_wires)), dtype=np.float64)
    for row, force in enumerate(ordered_forces):
        if mesh_layout == "gray_spaced":
            # The unused Gray addresses are on a 32-point axis whose endpoints
            # coincide with the 0 and 140 physical mesh coordinates.
            physical_distance = np.linalg.norm(
                candidate_xy * ((GRID_SIDE - 1) * GRID_SPACING /
                                (WIRE_SIDE - 1)) - pos[force, :2],
                axis=1) / GRID_SPACING
        else:
            physical_distance = np.linalg.norm(
                candidate_xy - pos[force, :2] / GRID_SPACING, axis=1)
        if force_layout == "nearest":
            cost[row] = physical_distance
        else:
            contact_wires = mapping[np.asarray(sorted(contact[force]))]
            # A ten-bit xor counts the J stages needed to transport a component.
            contact_distance = _POPCOUNT[
                free_wires[:, None] ^ contact_wires[None, :]].mean(axis=1)
            # The contact cost governs placement; this bounded secondary term
            # selects spatially nearer free wires among almost-equivalent choices.
            cost[row] = contact_distance + 0.01 * physical_distance / WIRE_SIDE
    rows, columns = linear_sum_assignment(cost)
    for row, column in zip(rows, columns):
        mapping[ordered_forces[row]] = int(free_wires[column])
    if (mapping < 0).any() or len(np.unique(mapping)) != len(mapping):
        raise AssertionError("Plate compiler failed to assign injective wires")
    return torch.from_numpy(mapping)


def compile_plate_binary_contact(graph) -> torch.Tensor:
    """Map the fixed mesh to interleaved integer bits and forces by contact."""
    return compile_plate_gray_contact(graph, mesh_layout="binary",
                                      force_layout="contact")


def compile_plate_gray_spaced_contact(graph) -> torch.Tensor:
    """Map the fixed mesh to spaced Gray wires and forces by contact."""
    return compile_plate_gray_contact(graph, mesh_layout="gray_spaced",
                                      force_layout="contact")
