"""CUDA optimizer parity when batches use different padded graph sizes."""
import copy
from pathlib import Path
import sys

import pytest
import torch

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
sys.path.insert(0,str(ROOT/'scripts'))
from gcm.loom import Loom,controller_linear
from loom_graph_campaign import CapturedStep,loss_value


@pytest.mark.skipif(not torch.cuda.is_available(),reason='CUDA required')
def test_controller_parallel_gradient_matches_linear():
    torch.manual_seed(31)
    value=torch.randn(2,4096,128,device='cuda',requires_grad=True)
    layer=torch.nn.Linear(128,4).cuda()
    expected=layer(value)
    actual=controller_linear(layer,value)
    torch.testing.assert_close(actual,expected,rtol=2e-5,atol=2e-6)
    parameters=(value,*layer.parameters())
    expected_grad=torch.autograd.grad(expected.square().mean(),parameters)
    actual_grad=torch.autograd.grad(actual.square().mean(),parameters)
    for actual_value,expected_value in zip(actual_grad,expected_grad):
        torch.testing.assert_close(actual_value,expected_value,rtol=2e-5,atol=2e-6)


@pytest.mark.skipif(not torch.cuda.is_available(),reason='CUDA required')
@pytest.mark.parametrize('sizes',[[8,16,8,16,8],[1024,8192,1024,8192,1024]])
def test_captured_adamw_across_padded_sizes(sizes):
    torch.manual_seed(23)
    captured_model=Loom(3,2,max(sizes),dim=16,heads=2,blocks=2,dropout=0.).cuda()
    eager_model=copy.deepcopy(captured_model)
    def optimizer(model):
        return torch.optim.AdamW(model.parameters(),lr=torch.tensor(.002,device='cuda'),
                                 weight_decay=.01,fused=True,capturable=True)
    captured_optimizer=optimizer(captured_model)
    eager_optimizer=optimizer(eager_model)
    info={'task':'capture_check','mean':torch.zeros(2,device='cuda'),
          'std':torch.ones(2,device='cuda')}
    graphs={}
    for padded in sizes:
        batch={'x':torch.randn(2,padded,3,device='cuda'),
               'neighbors':torch.randn(2,padded,3,device='cuda'),
               'real':torch.ones(2,padded,dtype=torch.bool,device='cuda'),
               'mask':torch.ones(2,padded,dtype=torch.bool,device='cuda'),
               'y':10*torch.randn(2,padded,2,device='cuda')}
        repetitions=1
        if padded not in graphs:
            graphs[padded]=CapturedStep(captured_model,captured_optimizer,batch,info)
            graphs[padded].capture(batch)
            repetitions+=3
        graphs[padded].replay(batch)
        for _ in range(repetitions):
            eager_optimizer.zero_grad(set_to_none=True)
            loss=loss_value(eager_model.forward_packed(batch['x'],batch['neighbors'],batch['real']),batch,info)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(eager_model.parameters(),1.,foreach=True)
            eager_optimizer.step()
        torch.cuda.synchronize()
        for left,right in zip(captured_model.parameters(),eager_model.parameters()):
            torch.testing.assert_close(left,right,rtol=2e-5,atol=2e-6)
            left_state=captured_optimizer.state.get(left,{})
            right_state=eager_optimizer.state.get(right,{})
            assert left_state.keys()==right_state.keys()
            for key in left_state:
                torch.testing.assert_close(left_state[key],right_state[key],rtol=2e-5,atol=2e-6)
