"""Run a Loom forward/backward pass on a small graph, without downloading data."""
import torch
from gcm import Loom, compile_loom, normalized_neighbors


def main():
    torch.manual_seed(0)
    nodes, input_dim, classes = 17, 8, 3
    source = torch.arange(nodes)
    target = (source + 1) % nodes
    edges = torch.stack((torch.cat((source, target)), torch.cat((target, source))))
    x = torch.randn(nodes, input_dim)
    labels = torch.arange(nodes) % classes
    mapping, audit = compile_loom(edges, nodes, seed=0)
    model = Loom(input_dim, classes, nodes, dim=16, heads=2, blocks=2, dropout=0.)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    logits = model(x, normalized_neighbors(x, edges), mapping)
    loss = torch.nn.functional.cross_entropy(logits, labels)
    loss.backward()
    optimizer.step()
    assert logits.shape == (nodes, classes)
    assert torch.isfinite(logits).all()
    print(f"Loom output: {tuple(logits.shape)}, loss: {loss.item():.4f}")
    print(f"Layout cost: {audit['initial_cost']} -> {audit['final_cost']}")


if __name__ == '__main__':
    main()
