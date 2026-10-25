"""Stable Gray addresses for reference meshes and variable contact nodes."""

import numpy as np
import torch
from scipy.optimize import linear_sum_assignment

GRID_SIDE = 29
GRID_SPACING = 5.0
WIRE_SIDE = 32
WIRES = WIRE_SIDE * WIRE_SIDE
_POPCOUNT = np.asarray([i.bit_count() for i in range(WIRES)], dtype=np.float64)


def _gray(value: int) -> int:
    return value ^ (value >> 1)


def _interleaved_gray(x: int, y: int) -> int:
    gx, gy = _gray(x), _gray(y)
    return sum(
        ((gx >> bit) & 1) << (2 * bit) | ((gy >> bit) & 1) << (2 * bit + 1)
        for bit in range(5)
    )


def _inverse_gray(value: int) -> int:
    result = value
    while value := value >> 1:
        result ^= value
    return result


def _wire_grid_coordinates(wire: int) -> tuple[int, int]:
    gx = sum(((wire >> (2 * bit)) & 1) << bit for bit in range(5))
    gy = sum(((wire >> (2 * bit + 1)) & 1) << bit for bit in range(5))
    return _inverse_gray(gx), _inverse_gray(gy)


def compile_plate(graph) -> torch.Tensor:
    """Assign the fixed Plate mesh by Gray coordinates and sources by contact cost."""
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
    if len(mesh) != GRID_SIDE**2 or len(forces) < 1:
        raise ValueError("Plate layout requires 841 mesh vertices and force nodes")
    if len(node_type) > WIRES:
        raise ValueError("Plate graph exceeds its 1024-wire circuit")
    if hasattr(graph, "mesh_node_count") and int(graph.mesh_node_count) != len(mesh):
        raise ValueError("Plate mesh_node_count disagrees with node types")
    grid_float = pos[mesh, :2] / GRID_SPACING
    grid = np.rint(grid_float).astype(np.int64)
    if (
        np.abs(grid_float - grid).max() > 0.001
        or (grid < 0).any()
        or (grid >= GRID_SIDE).any()
        or (np.abs(pos[mesh, 2]).max() > 0.001)
    ):
        raise ValueError("Plate mesh is not the fixed 29x29 initial xy grid")
    flat_grid = grid[:, 0] * GRID_SIDE + grid[:, 1]
    if len(np.unique(flat_grid)) != GRID_SIDE**2:
        raise ValueError("Plate grid coordinates must be unique and complete")
    mapping = np.full(len(node_type), -1, dtype=np.int64)
    mapping[mesh] = [_interleaved_gray(int(x), int(y)) for x, y in grid]
    free_wires = np.setdiff1d(
        np.arange(WIRES, dtype=np.int64), mapping[mesh], assume_unique=True
    )
    if len(forces) > len(free_wires):
        raise ValueError("Too many force nodes for free plate wires")
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
    if any((not incident for incident in contact.values())):
        raise ValueError("Each force needs a contact edge to the plate mesh")
    properties = getattr(graph, "properties", None)
    directions = (
        np.asarray(torch.as_tensor(properties).detach().cpu())[:, 3:6]
        if properties is not None
        else np.zeros((len(node_type), 3))
    )
    if directions.shape != (len(node_type), 3) or not np.isfinite(directions).all():
        raise ValueError("Plate force properties must have six finite channels")
    ordered_forces = sorted(
        (int(force) for force in forces),
        key=lambda force: (
            *pos[force].tolist(),
            *directions[force].tolist(),
            tuple(sorted((int(mapping[vertex]) for vertex in contact[force]))),
        ),
    )
    wire_coordinates = _wire_grid_coordinates
    candidate_xy = np.asarray(
        [wire_coordinates(int(wire)) for wire in free_wires], dtype=np.float64
    )
    cost = np.empty((len(forces), len(free_wires)), dtype=np.float64)
    for row, force in enumerate(ordered_forces):
        physical_distance = np.linalg.norm(
            candidate_xy - pos[force, :2] / GRID_SPACING, axis=1
        )
        contact_wires = mapping[np.asarray(sorted(contact[force]))]
        contact_distance = _POPCOUNT[free_wires[:, None] ^ contact_wires[None, :]].mean(
            axis=1
        )
        cost[row] = contact_distance + 0.01 * physical_distance / WIRE_SIDE
    rows, columns = linear_sum_assignment(cost)
    for row, column in zip(rows, columns):
        mapping[ordered_forces[row]] = int(free_wires[column])
    if (mapping < 0).any() or len(np.unique(mapping)) != len(mapping):
        raise AssertionError("Plate compiler failed to assign injective wires")
    return torch.from_numpy(mapping)


def gray_grid(coordinates: torch.Tensor, bits: int):
    """Interleave Gray-coded nonnegative integer coordinates (Appendix A).

    coordinates: [num_nodes, dimensions], each entry smaller than 2**bits.
    """
    coordinates = torch.as_tensor(coordinates, dtype=torch.long)
    if coordinates.ndim != 2 or bits < 1:
        raise ValueError(
            "Expected coordinates [num_nodes, dimensions] and positive bits"
        )
    if (coordinates < 0).any() or (coordinates >= 1 << bits).any():
        raise ValueError("Coordinates are outside the reference address range")
    gray = coordinates ^ (coordinates >> 1)
    address = torch.zeros(
        coordinates.shape[0], dtype=torch.long, device=coordinates.device
    )
    for bit in range(bits):
        for axis in range(coordinates.shape[1]):
            address |= ((gray[:, axis] >> bit) & 1) << (
                coordinates.shape[1] * bit + axis
            )
    return address
