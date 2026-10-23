"""GCM v2: graph-compiled interfaces and persistent content control."""
import math
import networkx as nx
import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla
import torch
from torch import nn
from torch.nn import functional as F
from torch_geometric.nn import APPNP
from .layers import rotate90


def _refined_graph_split(graph, nodes, seed):
    """Find a balanced topology cut from the current order and three restarts."""
    region = graph.subgraph(nodes)
    target = (len(nodes) + 1) // 2
    initial_left = set(map(int, nodes[:target]))
    initial_right = set(map(int, nodes[target:]))
    best = None
    best_cut = math.inf
    for restart in range(4):
        initial = (initial_left, initial_right) if restart == 0 else None
        left, right = nx.algorithms.community.kernighan_lin_bisection(
            region, partition=initial, max_iter=20,
            seed=(seed * 10000 + restart) % (2**32))
        cut = nx.cut_size(region, left, right)
        if cut < best_cut:
            best, best_cut = set(left), cut
    mask = np.fromiter((int(node) in best for node in nodes), dtype=bool,
                       count=len(nodes))
    return nodes[mask], nodes[~mask]


def _spectral_order(adjacency, allow_symmetric=False):
    """Seriate a connected graph; optionally resolve symmetric signs by index."""
    n = adjacency.shape[0]
    if n < 3:
        return np.arange(n, dtype=np.int64)
    count, _ = sp.csgraph.connected_components(adjacency, directed=False)
    if count != 1:
        raise ValueError("graph_spectral requires a connected graph")
    degree = np.asarray(adjacency.sum(1)).ravel()
    laplacian = sp.csgraph.laplacian(adjacency, normed=False).tocsr()
    initial = degree - degree.mean() + .01
    _, vectors = spla.eigsh(laplacian + 1e-3 * sp.eye(n, format="csr"),
                            k=2, sigma=0., which="LM", tol=1e-6,
                            maxiter=5000, v0=initial)
    axis = vectors[:, 1]
    # The eigenvector sign is arbitrary. A topology-only moment fixes the
    # direction whenever the graph distinguishes its two ends.
    for anchor in (degree, adjacency @ degree, adjacency @ (adjacency @ degree)):
        centered = np.asarray(anchor).ravel() - np.mean(anchor)
        moment = float(axis @ centered)
        if abs(moment) > 1e-6 * np.linalg.norm(centered):
            if moment > 0:
                axis = -axis
            break
    else:
        if not allow_symmetric:
            raise ValueError("graph_spectral cannot orient this symmetric graph")
        # New layout's deterministic fallback; unresolved symmetries may
        # depend on input numbering, as recorded in the compiler specification.
        if axis[int(np.argmax(np.abs(axis)))] < 0:
            axis = -axis
    return np.argsort(axis, kind="stable")


def _refine_spectral_order(adjacency, order, window=32):
    """Short 2-opt reversals remove missing graph edges in a spectral order."""
    order = order.copy()
    n = len(order)
    indptr, indices = adjacency.indptr, adjacency.indices

    def linked(u, v):
        return v in indices[indptr[u]:indptr[u + 1]]

    gaps = np.array([not linked(int(order[i]), int(order[i + 1]))
                     for i in range(n - 1)], dtype=bool)
    while True:
        changed = False
        for i in range(n - 2):
            a, b = int(order[i]), int(order[i + 1])
            upper = min(i + window, n - 1)
            candidates = (np.arange(i + 2, upper + 1) if gaps[i] else
                          np.flatnonzero(gaps[i + 2:upper + 1]) + i + 2)
            best = None
            for j in candidates:
                j = int(j)
                c = int(order[j])
                d = int(order[j + 1]) if j + 1 < n else None
                old = int(not gaps[i]) + int(d is not None and not gaps[j])
                new = int(linked(a, c)) + int(d is not None and linked(b, d))
                gain = new - old
                if gain > 0 and (best is None or (gain, -j) > (best[0], -best[1])):
                    best = (gain, j)
            if best is not None:
                j = best[1]
                order[i + 1:j + 1] = order[i + 1:j + 1][::-1]
                gaps[i + 1:j] = gaps[i + 1:j][::-1]
                gaps[i] = not linked(int(order[i]), int(order[i + 1]))
                if j < n - 1:
                    gaps[j] = not linked(int(order[j]), int(order[j + 1]))
                changed = True
        if not changed:
            return order


def compile_interfaces(edge_index, n, layout="graph", alignment=True, seed=0,
                       node_features=None):
    """Graph ordering, then role-aligned bijections at every level.

    Every wire occurs in exactly one pair per level. Bijections connect sibling
    regions, preserving inductive dependency coverage even with padded leaves.
    ``graph_components`` keeps disconnected induced components in dyadic regions
    when they fit, then uses the same diffusion order inside each component.
    ``graph_refined`` adds balanced sparse graph cuts to connected regions.
    ``graph_spectral`` preserves a connected graph's one-dimensional spectral
    order; its direction is fixed by graph structure, not node numbering.
    ``graph_spectral_2opt`` retains the legacy linear-adjacency refinement.
    ``graph_spectral_hamming`` refines the Fiedler placement by exact
    descending Hamming-cost address swaps; it uses the fixed XOR schedule.
    ``graph_feature_anchor`` initializes each recursive diffusion from a fixed
    projection of observed node features.  When these features distinguish
    nodes, the resulting order does not depend on their input numbering.
    ``graph_feature_anchor_components`` uses the same feature-anchored order
    while packing disconnected induced components into dyadic regions.
    Symmetric graphs with identical features still have unresolved ties.
    Other layouts may use node ids for tie breaking and are not canonical.
    """
    src, dst = edge_index.cpu().numpy()
    a = sp.csr_matrix((np.ones(len(src)), (src, dst)), shape=(n, n))
    a = a.maximum(a.T); a.setdiag(0); a.eliminate_zeros()
    degree = np.asarray(a.sum(1)).ravel()
    neighbor = (a @ degree) / np.maximum(degree, 1)
    features = np.stack([np.log1p(degree), np.log1p(neighbor),
                         np.log1p((a @ neighbor) / np.maximum(degree, 1))], axis=1)
    features = (features - features.mean(0)) / np.maximum(features.std(0), 1e-6)
    feature_anchor = None
    if layout in ("graph_feature_anchor", "graph_feature_anchor_components"):
        if node_features is None:
            raise ValueError(f"{layout} requires observed node_features")
        observed = torch.as_tensor(node_features).detach().cpu().numpy()
        if observed.ndim != 2 or observed.shape[0] != n or not observed.shape[1]:
            raise ValueError("node_features must have shape [num_nodes, num_features]")
        observed = np.asarray(observed, dtype=np.float64)
        observed = np.nan_to_num(observed, nan=0., posinf=0., neginf=0.)
        observed -= observed.mean(0, keepdims=True)
        observed /= np.maximum(observed.std(0, keepdims=True), 1e-6)
        columns = np.arange(1, observed.shape[1] + 1, dtype=np.float64)
        weights = np.sin(columns * np.sqrt(2.)) + np.cos(columns * np.sqrt(3.))
        weights /= np.linalg.norm(weights)
        feature_anchor = observed @ weights
    padded = 1 << (n-1).bit_length()
    rng = np.random.default_rng(seed)
    graph = nx.from_scipy_sparse_array(a) if layout == "graph_refined" else None
    baseline_mapping = (compile_interfaces(edge_index, n, layout="graph_components",
                                           alignment=False, seed=seed)[0].numpy()
                        if layout == "graph_refined" else None)
    split_index = 0

    def split_components(nodes, capacity, labels, count):
        """Pack whole components into the two wire regions when possible."""
        half = capacity // 2
        order = np.argsort(labels, kind="stable")
        offsets = np.concatenate(([0], np.cumsum(np.bincount(labels, minlength=count))))
        components = [nodes[order[offsets[index]:offsets[index + 1]]]
                      for index in range(count)]
        if feature_anchor is None:
            components.sort(key=lambda component: (-len(component), int(component.min())))
        else:
            components.sort(key=lambda component: (-len(component),
                                                  float(feature_anchor[component].mean()),
                                                  float(feature_anchor[component].min())))

        # Subset sums use one bit per possible occupancy. Record only newly
        # reached sums so reconstruction does not depend on later components.
        reachable = 1
        limit = (1 << (half + 1)) - 1
        predecessor = np.full(half + 1, -1, dtype=np.int32)
        previous_sum = np.full(half + 1, -1, dtype=np.int32)
        for index, component in enumerate(components):
            size = len(component)
            if size > half:
                continue
            shifted = (reachable << size) & limit
            new = shifted & ~reachable
            while new:
                bit = new & -new
                total = bit.bit_length() - 1
                predecessor[total] = index
                previous_sum[total] = total - size
                new -= bit
            reachable |= shifted

        target = (len(nodes) + 1) // 2
        minimum = max(0, len(nodes) - half)
        chosen = None
        for distance in range(half + 1):
            for total in (target + distance, target - distance):
                if minimum <= total <= half and (reachable >> total) & 1:
                    chosen = total
                    break
            if chosen is not None:
                break

        if chosen is not None:
            selected = np.zeros(count, dtype=bool)
            while chosen:
                index = predecessor[chosen]
                selected[index] = True
                chosen = previous_sum[chosen]
            left_parts = [component for index, component in enumerate(components)
                          if selected[index]]
            right_parts = [component for index, component in enumerate(components)
                           if not selected[index]]
            left = np.concatenate(left_parts) if left_parts else np.empty(0, dtype=nodes.dtype)
            right = np.concatenate(right_parts) if right_parts else np.empty(0, dtype=nodes.dtype)
            return left, right

        # At least one component must cross the boundary. Keep the other
        # components intact and let recursion resolve the split component.
        ordered = np.concatenate(components)
        return ordered[:target], ordered[target:]

    def partition(nodes, capacity):
        nonlocal split_index
        if capacity == 1:
            return nodes.tolist() if len(nodes) else [-1]
        if layout == "graph_refined" and len(nodes) > 2 and capacity >= 64:
            # The component-aware layout supplies a deterministic initial cut.
            # Refine even disconnected regions: KL can move entire components
            # between sides when the earlier cut fragmented a community.
            nodes = nodes[np.argsort(baseline_mapping[nodes], kind="stable")]
            left, right = _refined_graph_split(graph, nodes, seed + split_index)
            split_index += 1
            return partition(left, capacity // 2) + partition(right, capacity // 2)
        sub = None
        if layout in ("graph_components", "graph_refined",
                      "graph_feature_anchor_components") and len(nodes) > 1:
            sub = a[nodes][:, nodes]
            count, labels = sp.csgraph.connected_components(sub, directed=False)
            if count > 1:
                left, right = split_components(nodes, capacity, labels, count)
                return partition(left, capacity // 2) + partition(right, capacity // 2)
        if layout == "graph_feature_anchor_components" and len(nodes) == 2:
            # A two-node connected region has no diffusion step below. The
            # anchor must still orient its leaves independently of node IDs.
            nodes = nodes[np.argsort(feature_anchor[nodes], kind="stable")]
        if len(nodes) > 2 and layout in ["graph","graph_random","xor","structural",
                                          "graph_components", "graph_refined",
                                          "graph_feature_anchor",
                                          "graph_feature_anchor_components"]:
            if sub is None:
                sub = a[nodes][:, nodes]
            d = np.asarray(sub.sum(1)).ravel() + 1
            inv = 1 / np.sqrt(d)
            stationary = np.sqrt(d); stationary /= np.linalg.norm(stationary)
            if layout == 'structural':
                # Topology-derived initial signal; exact ties still need a
                # symmetry-aware treatment, so this is not canonical labeling.
                vector = np.sin(features[nodes] @ np.array([1.61803398875,2.41421356237,.73205080757]) + .5)
            elif layout == 'graph_random':
                # A topology-smoothed random probe gives each compiler head a
                # different balanced cut while remaining graph-aware.
                vector = rng.standard_normal(len(nodes))
            elif layout in ('graph_feature_anchor', 'graph_feature_anchor_components'):
                vector = feature_anchor[nodes].copy()
            else:
                vector = np.sin(nodes * 1.61803398875 + 0.5)
            for _ in range(16):
                vector -= stationary * np.dot(stationary, vector)
                vector /= max(np.linalg.norm(vector), 1e-12)
                vector = .5 * vector + .5 * inv * ((sub @ (inv * vector)) + inv * vector)
            if layout in ('graph_feature_anchor', 'graph_feature_anchor_components'):
                vector = vector + 1e-8 * feature_anchor[nodes]
            nodes = nodes[np.argsort(vector, kind="stable")]
        split = (len(nodes)+1)//2
        return partition(nodes[:split], capacity//2) + partition(nodes[split:], capacity//2)

    nodes = rng.permutation(n) if layout == "random" else np.arange(n)
    if layout in ("graph_spectral", "graph_spectral_2opt", "graph_spectral_hamming"):
        if layout == "graph_spectral_hamming":
            from .hamming_compiler import compiler_adjacency
            simple = compiler_adjacency(a).astype(float)
            count, labels = sp.csgraph.connected_components(simple, directed=False)
            components = [np.flatnonzero(labels == c) for c in range(count)]
            components.sort(key=lambda nodes: (-len(nodes), int(nodes[0])))
            order = np.concatenate([
                nodes[_spectral_order(simple[nodes][:, nodes], allow_symmetric=True)]
                for nodes in components])
        else:
            order = _spectral_order(a)
        if layout == "graph_spectral_2opt":
            order = _refine_spectral_order(a, order)
        wires = np.concatenate((order, np.full(padded - n, -1, dtype=order.dtype)))
        alignment = False
    elif layout == "input_order":
        # Diagnostic layout for datasets whose node order is meaningful.
        # It intentionally uses input IDs and is not permutation equivariant.
        wires = np.concatenate((nodes, np.full(padded - n, -1, dtype=nodes.dtype)))
        alignment = False
    else:
        wires = np.asarray(partition(nodes, padded))
    real_wires = np.flatnonzero(wires >= 0)
    node_to_wire = np.empty(n, dtype=np.int64)
    node_to_wire[wires[real_wires]] = real_wires
    if layout == 'graph_spectral_hamming':
        from .hamming_compiler import refine_hamming_mapping
        node_to_wire, _ = refine_hamming_mapping(a, node_to_wire, padded, seed=seed)
        wires[:] = -1
        wires[node_to_wire] = np.arange(n)
    if layout == 'xor':
        from .circuit_spectrum import refine_xor_mapping
        node_to_wire=refine_xor_mapping(a,node_to_wire,padded,seed)
        wires[:]=-1;wires[node_to_wire]=np.arange(n)
    roles = np.zeros((padded, 3), dtype=np.float32)
    roles[real_wires] = features[wires[real_wires]]
    pairs = []
    for level in range(padded.bit_length()-1):
        stride = 1 << level
        left, right = [], []
        for start in range(0, padded, 2*stride):
            l = np.arange(start, start+stride); r = l+stride
            if alignment:
                l = l[np.argsort(roles[l, 0] + .3*roles[l, 1], kind="stable")]
                r = r[np.argsort(roles[r, 0] + .3*roles[r, 1], kind="stable")]
            left.extend(l); right.extend(r)
        pairs.append(torch.tensor([left, right], dtype=torch.long))
    return torch.from_numpy(node_to_wire), torch.from_numpy(features.astype(np.float32)), torch.stack(pairs)


class ControlledCircuit(nn.Module):
    def __init__(self, dim, heads, node_to_wire, pairs, checkpoint=False, persistent=True, reversible=False):
        super().__init__()
        if dim % (2*heads):
            raise ValueError("Each rotation head must have even dimension")
        self.heads, self.width = heads, dim//heads
        self.register_buffer("node_to_wire", node_to_wire)
        self.register_buffer("pairs", pairs)
        self.padded = pairs.shape[-1]*2
        self.checkpoint = checkpoint
        self.persistent = persistent
        self.reversible = reversible
        if reversible and not persistent:
            raise ValueError("Reconstruction requires persistent control")
        self.control = nn.ModuleList(nn.Linear(2*dim, heads) for _ in pairs)
        self.payload = nn.Parameter(torch.zeros(len(pairs), heads, self.width))
        for gate in self.control:
            nn.init.normal_(gate.weight, std=.01); nn.init.constant_(gate.bias, .15)

    def stage(self, z, c, index, inverse=False):
        left, right = self.pairs[index]
        u, v = z[left], z[right]
        a, b = (c[left], c[right]) if self.persistent else (u, v)
        angle = self.control[index](torch.cat((a+b, (a-b).abs()), -1))
        uh, vh = u.reshape(-1,self.heads,self.width), v.reshape(-1,self.heads,self.width)
        energy = ((uh+vh).square().mean(-1)+1e-6).sqrt()
        angle = .5 * torch.tanh(angle + energy * self.payload[index].mean(-1))
        if inverse:
            angle=-angle
        co, si = angle.cos().unsqueeze(-1), angle.sin().unsqueeze(-1)
        out = torch.empty_like(z)
        out = out.index_copy(0, left, (co*uh+si*rotate90(vh)).flatten(1))
        return out.index_copy(0, right, (co*vh+si*rotate90(uh)).flatten(1))

    def forward(self, x, control):
        from torch.utils.checkpoint import checkpoint
        z = x.new_zeros(self.padded, x.shape[-1]).index_copy(0,self.node_to_wire,x)
        c = control.new_zeros(self.padded, control.shape[-1]).index_copy(0,self.node_to_wire,control)
        if self.reversible and self.training and torch.is_grad_enabled():
            from .reversible import ReconstructCircuit
            return ReconstructCircuit.apply(z,c,self,*tuple(self.parameters()))[self.node_to_wire]
        for index in range(len(self.pairs)):
            if self.checkpoint and self.training:
                z = checkpoint(lambda value, context, i=index: self.stage(value,context,i), z,c,use_reentrant=False)
            else:
                z = self.stage(z,c,index)
        return z[self.node_to_wire]


class InterfaceGCM(nn.Module):
    def __init__(self, data, args):
        super().__init__()
        dim=args.hidden
        mapping, roles, pairs = compile_interfaces(data.edge_index.detach().cpu(),data.num_nodes,
                                                   args.layout,not args.no_alignment,args.seed)
        self.encoder=nn.Sequential(nn.Linear(data.num_features,dim),nn.ReLU(),nn.Dropout(args.dropout))
        self.readout=nn.Linear(dim,int(data.y.max())+1)
        self.register_buffer("roles", roles)
        self.structure=nn.Linear(3,dim,bias=False)
        self.norm=nn.LayerNorm(dim)
        self.circuit=ControlledCircuit(dim,args.heads,mapping,pairs,args.checkpoint_stages,not args.no_control,
                                       getattr(args,'reversible',False))
        self.decode=nn.Linear(dim,dim,bias=False)
        self.scale=nn.Parameter(torch.full((dim,),.01))
        self.propagate=APPNP(K=args.propagation_steps,alpha=args.teleport,cached=True)
        self.disable_circuit=args.disable_circuit
        self.bounded_readout=getattr(args,"bounded_readout",False)
        self.input_dropout=getattr(args,"input_dropout",False)
        self.dropout=args.dropout
        self.last_ratio=0.

    def forward(self,x,edge_index):
        if self.input_dropout:
            x=F.dropout(x,self.dropout,self.training)
        h=self.encoder(x)
        local=self.propagate(h,edge_index)
        if not self.disable_circuit:
            payload=self.norm(local)
            control=self.norm(h+self.structure(self.roles))
            delta=self.decode(self.circuit(payload,control)-payload)
            if self.bounded_readout:
                delta=delta/delta.square().mean(-1,keepdim=True).add(1.).sqrt()
            delta=delta*local.square().mean(-1,keepdim=True).add(1e-12).sqrt()
            delta=.1*torch.tanh(self.scale)*delta
            if not self.training:
                self.last_ratio=float(delta.norm().detach()/local.norm().detach().clamp_min(1e-8))
            local=local+delta
        if self.bounded_readout:
            base=self.propagate(self.readout(h),edge_index)
            return base if self.disable_circuit else base+F.linear(delta,self.readout.weight)
        return self.readout(local)


class APPNPBaseline(nn.Module):
    def __init__(self, data,args):
        super().__init__()
        self.encoder=nn.Sequential(nn.Linear(data.num_features,args.hidden),nn.ReLU(),nn.Dropout(args.dropout))
        self.readout=nn.Linear(args.hidden,int(data.y.max())+1)
        self.propagate=APPNP(K=args.propagation_steps,alpha=args.teleport,cached=True)
        self.input_dropout=getattr(args,"input_dropout",False)
        self.dropout=args.dropout

    def forward(self,x,edge_index):
        if self.input_dropout:
            x=F.dropout(x,self.dropout,self.training)
        return self.propagate(self.readout(self.encoder(x)),edge_index)
