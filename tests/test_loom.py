import numpy as np
import scipy.sparse as sp
import torch
from gcm.loom import Loom, LoomBlock, LoomPair, normalized_neighbors
from gcm.pure import PhasePair
from gcm.hamming_compiler import refine_loom_mapping, hamming_cost


def test_grouped_circuit_matches_indexed_outputs_and_gradients():
    torch.manual_seed(7)
    block = LoomBlock(16, 2, 4, 0., .2).double()
    for pair in block.pairs:
        torch.nn.init.normal_(pair.gate.weight, std=.1)
        torch.nn.init.normal_(pair.direction_gate.weight, std=.1)
        pair.middle_delta.data.fill_(.12)
    x = torch.randn(2, 16, 16, dtype=torch.double, requires_grad=True)
    a = block.mix(x)
    b = block.mix(x, indexed=True)
    torch.testing.assert_close(a, b, atol=1e-12, rtol=1e-12)
    params = [x] + list(block.pairs.parameters())
    ga = torch.autograd.grad(a.square().sum(), params, retain_graph=True)
    gb = torch.autograd.grad(b.square().sum(), params)
    for aa, bb in zip(ga, gb):
        torch.testing.assert_close(aa, bb, atol=1e-10, rtol=1e-10)


def test_embedding_and_padding_persistence_match_explicit_equations():
    torch.manual_seed(3)
    x = torch.randn(5, 3)
    edges = torch.tensor([[0, 1, 1, 2, 3, 4], [1, 0, 2, 1, 4, 3]])
    mapping = torch.tensor([0, 3, 7, 2, 4])
    model = Loom(3, 2, 5, dim=8, heads=2, blocks=2, dropout=0.)
    nei = normalized_neighbors(x, edges)
    d = torch.bincount(edges[0], minlength=5).float()
    adjacency = torch.zeros(5, 5)
    adjacency[edges[1], edges[0]] = (d[edges[0]] * d[edges[1]]).rsqrt()
    h = model.embed_norm(model.self_projection(x) + adjacency @ model.graph_projection(x))
    z = torch.zeros(8, 8).index_copy(0, mapping, h)
    for block in model.blocks:
        z = block(z)
    expected = model.head(model.head_norm(z[mapping]))
    torch.testing.assert_close(model(x, nei, mapping), expected)
    # Batched execution keeps samples independent, including their padding.
    ids = torch.arange(2).repeat_interleave(5)
    out = model(x.repeat(2, 1), nei.repeat(2, 1), mapping.repeat(2), ids, 8)
    torch.testing.assert_close(out, expected.repeat(2, 1))


def test_refinement_includes_exact_vacant_wire_moves():
    rng = np.random.default_rng(4)
    a = sp.csr_matrix(rng.random((13, 13)) < .12)
    initial = rng.choice(16, 13, replace=False)
    refined, audit = refine_loom_mapping(a, initial, 16, seed=3)
    assert len(np.unique(refined)) == 13
    assert audit['accepted_moves'] > 0
    assert hamming_cost(a, refined) == audit['final_cost'] <= audit['initial_cost']
    assert all(b < aa for aa, b in zip(audit['cost_history'], audit['cost_history'][1:]))


def test_fused_lifting_matches_three_updates_and_gradients():
    torch.manual_seed(17)
    pair=LoomPair(16,2,.2).double()
    original=PhasePair(16,2,'lifting',.2,True).double()
    for parameter in pair.parameters():
        torch.nn.init.normal_(parameter, std=.1)
    original.load_state_dict(pair.state_dict())
    u=torch.randn(11,16,dtype=torch.double,requires_grad=True)
    v=torch.randn_like(u,requires_grad=True)
    a=pair(u,v);b=original(u,v)
    for aa,bb in zip(a,b):
        torch.testing.assert_close(aa,bb,atol=1e-12,rtol=1e-12)
    ga=torch.autograd.grad(sum(z.square().sum() for z in a),[u,v]+list(pair.parameters()),retain_graph=True)
    gb=torch.autograd.grad(sum(z.square().sum() for z in b),[u,v]+list(original.parameters()))
    for aa,bb in zip(ga,gb):
        torch.testing.assert_close(aa,bb,atol=1e-10,rtol=1e-10)


def test_sparse_offset_embedding_is_exact_and_keeps_weight_gradients():
    torch.manual_seed(23)
    layer=torch.nn.Linear(7,5).double()
    matrix=torch.randn(11,7,dtype=torch.double)
    matrix[matrix.abs()<.6]=0
    offset=torch.randn(1,7,dtype=torch.double)
    scale=torch.rand(11,1,dtype=torch.double)
    actual=Loom.project(layer,(matrix.to_sparse(),offset,scale))
    expected=layer(matrix+scale*offset)
    torch.testing.assert_close(actual,expected,atol=1e-12,rtol=1e-12)
    a=torch.autograd.grad(actual.square().sum(),list(layer.parameters()),retain_graph=True)
    b=torch.autograd.grad(expected.square().sum(),list(layer.parameters()))
    for aa,bb in zip(a,b):
        torch.testing.assert_close(aa,bb,atol=1e-10,rtol=1e-10)


def test_nested_recomputation_preserves_outputs_and_gradients():
    torch.manual_seed(2)
    original=Loom(3,2,16,dim=8,heads=2,blocks=2,dropout=0.).double()
    recompute=Loom(3,2,16,dim=8,heads=2,blocks=2,dropout=0.,checkpoint_blocks=True).double()
    recompute.load_state_dict(original.state_dict())
    x=torch.randn(2,16,3,dtype=torch.double)
    nei=torch.randn_like(x)
    mask=torch.ones(2,16,dtype=torch.bool)
    a=original.forward_packed(x,nei,mask);b=recompute.forward_packed(x,nei,mask)
    torch.testing.assert_close(a,b,rtol=1e-12,atol=1e-12)
    a.square().sum().backward();b.square().sum().backward()
    for aa,bb in zip(original.parameters(),recompute.parameters()):
        torch.testing.assert_close(aa.grad,bb.grad,rtol=1e-10,atol=1e-10)
