# Loom VLDB methods

## Reference layout reuse

`gcm.plate_geometry.compile_plate_reference(graph)` reuses the canonical Gray
addresses of the fixed Plate mesh and the coordinates of unoccupied wires.
The per-request compiler validates the mesh coordinates, reads the current
force contacts, and solves an injective assignment for the forces. The cost is
the mean Hamming distance to contacted mesh wires plus the existing geometric
tie-break term. Selecting the Gray reference preserves the ordinary Gray
compiler's assignment for the same input.

## Maintenance and reads

For a simple undirected graph, the normalized neighbor row for node $u$
aggregates $x_v / \sqrt{d_ud_v}$ over each neighbor $v$. After an update,
the row can change only if its neighbor set changed, a current neighbor's
degree changed, or a current neighbor's features changed. A change in the
row's own degree already implies a change in its neighbor set. This gives the
refresh rule implemented in `LoomSession.publish`. It refreshes whole affected
rows with the same normalization as `normalized_neighbors`, including
zero rows for isolated nodes, up to floating-point rounding.

```python
import torch
from gcm import Loom, LoomSession

x = torch.randn(4, 8)
edges = torch.tensor([[0, 1, 2], [1, 2, 3]])
model = Loom(8, 3, 4, dim=16, heads=2, blocks=2, dropout=0.)
session = LoomSession(model, x, edges)
session.publish(delete=[(1, 2)], insert=[(0, 3)], features={3: -x[3]})
version, logits = session.read()
```

`retain=False` uses the same API and recomputes normalized inputs on each read.
Both modes run the full predictor at read time with fixed node IDs, layout,
and weights. An update publishes a new current graph version. The session
does not keep older snapshots. The Python lock serializes calls. CPU replay
is supported out of the box, and CUDA callers should use one CUDA stream
for each session.

## Lifecycle accounting

`gcm.PolicyCost` charges one build plus one publish cost per version and one
read cost per request. For $V$ versions with $Q$ reads each, total cost is
$B+V(P+QR)$. For a policy whose read cost is lower by $s>0$, the policy
beats the baseline once $Q s > \Delta P+\Delta B/V$.
`crossover` returns the first positive integer read count at which it wins,
or `None` if there is no winning count. Measure costs for the graph and
update distribution under consideration before applying the threshold.

## Replay a workload

Provide an NPZ with arrays `x` and `edge_index` in shapes `[N, F]` and
`[2, E]`. The trace is a JSON array. Each batch has `edits` containing
`[u, v, sign]` triples and `features` listing rows whose sign to flip.
The replay applies deletions before insertions. All selected policies
receive the same batches, share one mapping and predictor, and check logits
against one another. By default the predictor uses seeded random weights.
Supply a local checkpoint with `--weights` and set the model dimensions
to match the checkpoint for trained inference.

```sh
python scripts/replay_updates.py graph.npz trace.json --output results/replay.csv
```

To include the ElectricSQL differential-dataflow adapter, install Node.js
and run `npm ci --prefix scripts/d2ts` before adding `--d2ts`. The CSV
repeats build and setup time per policy and publish time per version. Charge
each once when aggregating a stream.
