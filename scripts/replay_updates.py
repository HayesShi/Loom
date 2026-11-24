"""Compare fixed-layout update policies on a user-supplied graph and edit stream."""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
import subprocess
from time import perf_counter

import numpy as np
import torch

from gcm import Loom, LoomSession, compile_loom, normalized_neighbors


class DifferentialView:
    """Materialize D2TS's row patches for the same Loom predictor."""

    def __init__(self, process, model, x, edge_index, mapping):
        self.process = process
        self.model = model
        self.mapping = mapping
        self.x = x.clone()
        self.cache = normalized_neighbors(x, edge_index)
        edges = sorted({tuple(sorted((u, v))) for u, v in edge_index.T.tolist()})
        self._exchange({"n": len(x), "edits": [[u, v, 1] for u, v in edges], "features": []})

    def _exchange(self, batch):
        self.process.stdin.write(json.dumps(batch) + "\n")
        self.process.stdin.flush()
        response = self.process.stdout.readline()
        if not response:
            raise RuntimeError("D2TS adapter closed without responding")
        return json.loads(response)

    def publish(self, *, insert, delete, features):
        patches = self._exchange({
            "edits": [[u, v, -1] for u, v in delete] + [[u, v, 1] for u, v in insert],
            "features": list(features),
        })
        for node, value in features.items():
            self.x[node] = value
        rows, dst, src, products = [], [], [], []
        for i, (node, terms) in enumerate(patches):
            rows.append(node)
            for neighbor, degree_product in terms:
                dst.append(i)
                src.append(neighbor)
                products.append(degree_product)
        if rows:
            index = torch.tensor([dst, src], dtype=torch.long)
            matrix = torch.sparse_coo_tensor(index, self.x.new_tensor(products).rsqrt(),
                                             (len(rows), len(self.x))).coalesce()
            self.cache[rows] = torch.sparse.mm(matrix, self.x)
        return len(rows)

    def predict(self):
        return self.model(self.x, self.cache, self.mapping)


@torch.inference_mode()
def replay(args):
    with np.load(args.graph) as graph:
        x = torch.tensor(graph["x"], dtype=torch.float32)
        edges = set()
        for u, v in graph["edge_index"].T.tolist():
            if u != v:
                edges.add(tuple(sorted((u, v))))
        directed = sorted(edges | {(v, u) for u, v in edges})
        edge_index = torch.tensor(directed, dtype=torch.long).reshape(-1, 2).T.contiguous()
    with args.trace.open(encoding="utf-8") as stream:
        trace = json.load(stream)

    torch.manual_seed(args.seed)
    model = Loom(x.shape[1], args.outputs, len(x), dim=args.dim, heads=args.heads,
                 blocks=args.blocks, dropout=0.).eval()
    if args.weights:
        state = torch.load(args.weights, map_location="cpu", weights_only=True)
        model.load_state_dict(state.get("model", state))
    t0 = perf_counter()
    mapping, _ = compile_loom(edge_index, len(x), seed=args.seed)
    build_s = perf_counter() - t0
    process = None
    results = []
    try:
        policies = {}
        for label, retain in (("recompute", False), ("retain", True)):
            t0 = perf_counter()
            session = LoomSession(model, x, edge_index, mapping, retain=retain)
            setup = perf_counter() - t0
            policies[label] = (session.publish, lambda s=session: s.read()[1], setup)
        if args.d2ts:
            adapter = Path(__file__).resolve().parent / "d2ts" / "adapter.mjs"
            t0 = perf_counter()
            process = subprocess.Popen(["node", str(adapter), "d2ts"], encoding="utf-8",
                                       stdin=subprocess.PIPE, stdout=subprocess.PIPE)
            control = DifferentialView(process, model, x, edge_index, mapping)
            policies["d2ts"] = (control.publish, control.predict, perf_counter() - t0)
        for _, predict, _ in policies.values():
            predict()
        for version, edit in enumerate(trace, 1):
            updates = {
                "insert": [(u, v) for u, v, sign in edit["edits"] if sign == 1],
                "delete": [(u, v) for u, v, sign in edit["edits"] if sign == -1],
                "features": {node: -x[node] for node in edit["features"]},
            }
            for node, value in updates["features"].items():
                x[node] = value
            oracle = None
            order = list(policies)
            order = order[version % len(order):] + order[:version % len(order)]
            for label in order:
                publish, predict, setup = policies[label]
                t0 = perf_counter()
                affected = publish(**updates)
                publish_s = perf_counter() - t0
                for read in range(1, args.reads + 1):
                    t0 = perf_counter()
                    logits = predict()
                    read_s = perf_counter() - t0
                    if oracle is None:
                        oracle = logits
                    torch.testing.assert_close(logits, oracle)
                    results.append(dict(version=version, mode=label, read=read,
                                        build_s=build_s, setup_s=setup, publish_s=publish_s,
                                        read_s=read_s, affected_rows=affected))
    finally:
        if process is not None:
            process.stdin.close()
            process.wait()
            process.stdout.close()
    if not results:
        raise ValueError("trace must contain at least one batch")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(results[0]))
        writer.writeheader()
        writer.writerows(results)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("graph", type=Path, help="NPZ containing x and edge_index")
    parser.add_argument("trace", type=Path, help="JSON array of edit batches")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--weights", type=Path, help="optional local Loom checkpoint")
    parser.add_argument("--reads", type=int, default=4)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--outputs", type=int, default=4)
    parser.add_argument("--dim", type=int, default=32)
    parser.add_argument("--heads", type=int, default=4)
    parser.add_argument("--blocks", type=int, default=2)
    parser.add_argument("--threads", type=int, default=1)
    parser.add_argument("--d2ts", action="store_true")
    args = parser.parse_args()
    if args.reads < 1 or args.threads < 1:
        parser.error("--reads and --threads must be positive")
    torch.set_num_threads(args.threads)
    replay(args)


if __name__ == "__main__":
    main()
