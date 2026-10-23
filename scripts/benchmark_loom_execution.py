"""Measure equivalent Loom execution paths using one Roman checkpoint."""
import argparse
import csv
import gc
import json
from pathlib import Path
import sys
import time
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
REPO=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(REPO/'src'))
sys.path.insert(0,str(REPO/'scripts'))
from gcm.loom import Loom, normalized_neighbors
from gcm.pure import PhasePair
from gcm.diverse_data import load_diverse
from run_loom_campaign import benchmark_inference,save,write_json,sync,seed_all,node_recomputation


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--output',type=Path,default=REPO/'results/loom')
    args=parser.parse_args()
    torch.set_num_threads(4);torch.set_float32_matmul_precision('high')
    root=args.output/'roman_empire'
    result=json.loads((root/'result.json').read_text())
    candidates=[json.loads(p.read_text()) for p in root.glob('config_*.json')]
    winner=max(candidates,key=lambda row:row['validation'])
    cfg=winner['config']
    data=load_diverse('roman_empire')
    transform=cfg.get('feature_transform','none')
    if transform=='scale20':
        data.x=data.x*20.
    elif transform=='train_zscore_clip5':
        train=data.x[data.train_mask].double()
        mean=train.mean(0);std=train.std(0,unbiased=False).clamp_min(1e-6)
        data.x=((data.x.double()-mean)/std).clamp(-5.,5.).float()
    x=data.x.cuda();neighbors=normalized_neighbors(x,data.edge_index.cuda())
    mapping=torch.load(root/'mapping.pt',weights_only=False)['mapping'].cuda()
    checkpoint=torch.load(root/f"config_{winner['config_id']}_best.pt",weights_only=False)
    train_idx=torch.where(data.train_mask)[0].cuda();targets=data.y.cuda()
    rows=[];reference=None
    for execution in ['Indexed pairs','Grouped pairs','Fused grouped pairs']:
        seed_all(0)
        model=Loom(x.shape[1],int(data.y.max())+1,len(x),
                   **{k:cfg[k] for k in ['dim','heads','blocks','dropout','angle']},
                   checkpoint_blocks=node_recomputation(len(x),cfg)).cuda()
        model.load_state_dict(checkpoint['model'])
        for block in model.blocks:
            block.indexed_execution=execution=='Indexed pairs'
            block.checkpoint_pairs=model.checkpoint_blocks
            if execution!='Fused grouped pairs':
                pairs=nn.ModuleList()
                for pair in block.pairs:
                    original=PhasePair(cfg['dim'],cfg['heads'],'lifting',cfg['angle'],True).cuda()
                    original.load_state_dict(pair.state_dict());pairs.append(original)
                block.pairs=pairs
        model.eval()
        with torch.no_grad():
            prediction=model(x,neighbors,mapping).cpu()
        if reference is None:
            reference=prediction
        # Algebraic fusion changes float32 rounding across sixty pair stages.
        relative_rms=float((prediction-reference).double().norm()/reference.double().norm().clamp_min(1e-12))
        torch.testing.assert_close(prediction,reference,rtol=1e-4,atol=1e-3)
        if relative_rms>1e-4:
            raise RuntimeError('Execution variants differ beyond float32 rounding')
        torch.testing.assert_close(prediction[data.test_mask].argmax(1),
                                   reference[data.test_mask].argmax(1))
        accuracy=100.*float((prediction[data.test_mask].argmax(1)==data.y[data.test_mask]).float().mean())
        if abs(accuracy-result['metrics']['Accuracy'])>1e-4:
            raise RuntimeError('Benchmark prediction differs from the selected campaign checkpoint')
        infer=benchmark_inference(lambda:model(x,neighbors,mapping))
        model.train()
        lr=torch.tensor(cfg['lr'],device='cuda')
        optimizer=torch.optim.AdamW(model.parameters(),lr=lr,weight_decay=cfg['weight_decay'],fused=True,capturable=True)
        torch.cuda.empty_cache();torch.cuda.reset_peak_memory_stats()
        def step():
            optimizer.zero_grad(set_to_none=False)
            output=model(x,neighbors,mapping)
            loss=F.cross_entropy(output.index_select(0,train_idx),targets.index_select(0,train_idx))
            loss.backward();nn.utils.clip_grad_norm_(model.parameters(),1.,foreach=True);optimizer.step()
            return loss
        stream=torch.cuda.Stream();stream.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(stream):
            for _ in range(3):
                step()
        torch.cuda.current_stream().wait_stream(stream)
        sync()
        torch.cuda.empty_cache()
        graph=torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            captured_loss=step()
        times=[]
        for _ in range(20):
            begin=torch.cuda.Event(enable_timing=True);end=torch.cuda.Event(enable_timing=True)
            begin.record();graph.replay();end.record();sync();times.append(begin.elapsed_time(end))
        if not bool(torch.isfinite(captured_loss)):
            raise FloatingPointError(f'Nonfinite training loss: {execution}')
        row=dict(Execution=execution,Accuracy=accuracy,Train_step_ms=float(np.median(times)),
                 Inference_ms=infer['inference_ms'],Eager_inference_ms=infer['inference_eager_ms'],
                 Peak_training_MiB=torch.cuda.max_memory_allocated()/2**20,
                 Peak_inference_MiB=infer['peak_inference_MiB'],
                 Activation_recomputation=model.checkpoint_blocks,
                 Relative_RMS_output_difference=relative_rms,
                 Max_output_difference=float((prediction-reference).abs().max()))
        rows.append(row);print(row,flush=True)
        del graph,captured_loss,model,optimizer,prediction
        gc.collect();torch.cuda.empty_cache()
    write_json(args.output/'execution_benchmark.json',dict(hardware=torch.cuda.get_device_name(),
                config=cfg,seed=0,rows=rows,precision='float32'))
    with (args.output/'execution_benchmark.csv').open('w',newline='',encoding='utf-8-sig') as file:
        writer=csv.DictWriter(file,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)


if __name__=='__main__':
    main()
