"""Exact-stage CUDA parity, including nonzero controllers and clamp branches."""
import copy
import pytest
import torch
from gcm.loom import Loom, LoomPair

pytestmark=pytest.mark.skipif(not torch.cuda.is_available(),reason='CUDA required')


@pytest.mark.parametrize('batch,n,dim,heads,stride,angle',[
    (1,32,16,2,1,.2),(2,128,64,4,32,.4),
    (1,64,64,4,32,2.8),(1,64,64,4,4,-2.8),
    # 384 pair rows exercise the specialized 256-row reduction's final tile.
    (3,256,64,4,128,.4),
    # Six pair rows also exercise the eight-pair forward tile's final warps.
    (3,4,64,4,1,.4),
    # Generic path at the supported head-width / dimension / head-count limits.
    (1,32,256,8,16,.4),(1,32,64,32,4,.4)])
def test_cuda_stage_output_and_all_first_gradients(batch,n,dim,heads,stride,angle):
    pytest.importorskip('cupy')
    from gcm.loom_cuda import stage
    torch.manual_seed(32)
    torch.backends.cuda.matmul.allow_tf32=False
    pair=LoomPair(dim,heads,angle).cuda()
    torch.nn.init.normal_(pair.gate.weight,std=.08)
    torch.nn.init.normal_(pair.direction_gate.weight,std=.08)
    pair.middle_delta.data.fill_(.13)
    z=torch.randn(batch,n,dim,device='cuda',requires_grad=True)
    grouped=z.reshape(batch,n//(2*stride),2,stride,dim)
    u,v=pair(grouped[:,:,0],grouped[:,:,1])
    expected=torch.stack((u,v),dim=-3).reshape_as(z)
    actual=stage(z,pair,stride)
    torch.testing.assert_close(actual,expected,rtol=3e-5,atol=3e-6)
    direction=torch.randn_like(z)
    inputs=[z,*pair.parameters()]
    ga=torch.autograd.grad((actual*direction).sum(),inputs,retain_graph=True)
    ge=torch.autograd.grad((expected*direction).sum(),inputs)
    for a,e in zip(ga,ge):
        torch.testing.assert_close(a,e,rtol=3e-4,atol=3e-4)
    # A nonzero controller must receive a training signal on both paths.
    gate_index=next(i for i,p in enumerate(inputs) if p is pair.gate.weight)
    assert bool(ga[gate_index].abs().max()>0)


def test_full_cuda_model_outputs_gradients_and_optimizer_step():
    pytest.importorskip('cupy')
    torch.manual_seed(51)
    torch.backends.cuda.matmul.allow_tf32=False
    ref=Loom(7,3,100,dim=64,heads=4,blocks=2,dropout=0.).cuda()
    for block in ref.blocks:
        for pair in block.pairs:
            torch.nn.init.normal_(pair.gate.weight,std=.04)
            torch.nn.init.normal_(pair.direction_gate.weight,std=.04)
    fast=copy.deepcopy(ref).enable_cuda_stages()
    x=torch.randn(100,7,device='cuda');nei=torch.randn_like(x)
    mapping=torch.randperm(128,device='cuda')[:100]
    y=torch.randint(3,(100,),device='cuda')
    optimizers=[torch.optim.SGD(m.parameters(),lr=.001) for m in [ref,fast]]
    for _ in range(2):
        outputs=[]
        for model,opt in zip([ref,fast],optimizers):
            opt.zero_grad();out=model(x,nei,mapping);outputs.append(out)
            torch.nn.functional.cross_entropy(out,y).backward()
        torch.testing.assert_close(outputs[0],outputs[1],rtol=1e-4,atol=1e-5)
        for a,b in zip(ref.parameters(),fast.parameters()):
            torch.testing.assert_close(a.grad,b.grad,rtol=3e-4,atol=3e-5)
        for opt in optimizers:opt.step()
    for a,b in zip(ref.parameters(),fast.parameters()):
        torch.testing.assert_close(a,b,rtol=1e-4,atol=1e-6)


@pytest.mark.parametrize('dim,heads',[(16,2),(64,4)])
def test_cuda_stage_graph_replay_matches_eager_optimizer_updates(dim,heads):
    pytest.importorskip('cupy')
    torch.manual_seed(62)
    eager=Loom(5,3,32,dim=dim,heads=heads,blocks=2,dropout=0.).cuda()
    captured=copy.deepcopy(eager).enable_cuda_stages()
    x=torch.randn(32,5,device='cuda');neighbors=torch.randn_like(x)
    mapping=torch.arange(32,device='cuda');target=torch.randint(3,(32,),device='cuda')
    optimizers=[torch.optim.AdamW(m.parameters(),lr=.001,fused=True,capturable=True) for m in [eager,captured]]
    def step(model,opt):
        opt.zero_grad(set_to_none=False)
        loss=torch.nn.functional.cross_entropy(model(x,neighbors,mapping),target)
        loss.backward();opt.step();return loss
    stream=torch.cuda.Stream();stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        for _ in range(3):step(captured,optimizers[1])
    torch.cuda.current_stream().wait_stream(stream)
    for _ in range(3):step(eager,optimizers[0])
    graph=torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):loss=step(captured,optimizers[1])
    # Capture records launches; it does not execute an optimizer update.
    for _ in range(2):graph.replay();step(eager,optimizers[0])
    torch.cuda.synchronize()
    assert bool(loss.isfinite())
    for a,b in zip(eager.parameters(),captured.parameters()):
        torch.testing.assert_close(a,b,rtol=3e-4,atol=5e-5)
