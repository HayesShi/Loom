"""Resumable one-seed Loom campaign; validation selects configs and epochs."""
from __future__ import annotations
import argparse
import csv
import gc
import importlib
import json
import math
import os
from pathlib import Path
import random
import sys
import time
import traceback
from types import SimpleNamespace

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / 'src'))
sys.path.insert(0, str(REPO / 'scripts'))
import numpy as np
import scipy.sparse as sp
import torch
from torch import nn
from torch.nn import functional as F
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score, average_precision_score, r2_score
from gcm.paths import data_path
from gcm.loom import Loom, compile_loom, normalized_neighbors
from gcm.diverse_data import load_diverse

NODE_TASKS = ['cora', 'citeseer', 'pubmed', 'dblp', 'wikics', 'coauthor_cs',
              'coauthor_physics', 'roman_empire', 'amazon_ratings',
              'chameleon_filtered', 'minesweeper', 'tolokers', 'questions', 'ogbn_arxiv']
LAND_TASKS = ['city-roads-M', 'tolokers-2', 'artnet-exp', 'artnet-views',
              'city-reviews', 'twitch-views']
GRAPH_TASKS = ['lrim_16_0.6_10k', 'lrim_32_0.6_10k', 'lrim_16_1.5_10k',
               'lrim_32_1.5_10k', 'echo_sssp', 'plate', '2dele', 'hodge', 'laplace']
SYNTHETIC_TASKS=['reuse_1','reuse_4','reuse_16','transfer_1','transfer_8','transfer_32',
                 'scale_511','scale_512','scale_513','scale_8192']
GRAPH_TASKS+=SYNTHETIC_TASKS
ALL_TASKS = NODE_TASKS + LAND_TASKS + ['paris'] + GRAPH_TASKS


def write_json(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix('.tmp')
    tmp.write_text(json.dumps(data, ensure_ascii=False, indent=2, allow_nan=False), encoding='utf-8')
    os.replace(tmp, path)


def save(path, data):
    tmp = path.with_suffix('.tmp')
    torch.save(data, tmp)
    os.replace(tmp, path)


def append(path, data):
    with path.open('a', encoding='utf-8') as f:
        f.write(json.dumps(data, allow_nan=False) + '\n')


def seed_all(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def sync():
    torch.cuda.synchronize()


def state_cpu(model):
    return {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}


def static_inputs(raw):
    if raw.x.shape[1] > 2000:
        source,target=raw.edge_index.numpy()
        keep=source!=target
        source,target=source[keep],target[keep]
        out=np.bincount(source,minlength=len(raw.x)).clip(1)
        incoming=np.bincount(target,minlength=len(raw.x)).clip(1)
        a=sp.csr_matrix((1./np.sqrt(out[source]*incoming[target]),(target,source)),
                        shape=(len(raw.x),len(raw.x)),dtype=np.float32)
        baseline=raw.x.amin(0)
        features=sp.csr_matrix((raw.x-baseline).numpy())
        def tensor(matrix):
            coo=matrix.tocoo()
            indices=torch.from_numpy(np.stack((coo.row,coo.col)).astype(np.int64))
            return torch.sparse_coo_tensor(indices,torch.from_numpy(coo.data),coo.shape).coalesce().cuda()
        offset=baseline.cuda().unsqueeze(0)
        return ((tensor(features),offset,torch.ones(len(raw.x),1,device='cuda')),
                (tensor(a@features),offset,torch.from_numpy(np.asarray(a.sum(1))).cuda()))
    x=raw.x.float().cuda()
    return x,normalized_neighbors(x,raw.edge_index.cuda())


@torch.no_grad()
def benchmark_inference(forward, capture=True):
    torch.cuda.reset_peak_memory_stats()
    for _ in range(5):
        forward()
    sync()
    def measure(call):
        values=[]
        for _ in range(20):
            begin=torch.cuda.Event(enable_timing=True);end=torch.cuda.Event(enable_timing=True)
            begin.record();call();end.record();sync()
            values.append(begin.elapsed_time(end))
        return float(np.median(values))
    eager=measure(forward)
    result=dict(inference_eager_ms=eager,inference_ms=eager,inference_cuda_graph=False)
    if capture:
        torch.cuda.empty_cache()
        graph=torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            prediction=forward()
        graph.replay();sync()
        torch.testing.assert_close(prediction,forward(),rtol=1e-5,atol=1e-5)
        result.update(inference_ms=measure(graph.replay),inference_cuda_graph=True)
    result['peak_inference_MiB']=torch.cuda.max_memory_allocated()/2**20
    return result


def node_metrics(pred, target, kind):
    p = pred.detach().float().cpu().numpy()
    y = target.detach().cpu().numpy()
    if kind == 'classification':
        label = p.argmax(1)
        return dict(Accuracy=100. * accuracy_score(y, label),
                    Macro_F1=100. * f1_score(y, label, average='macro', zero_division=0))
    if kind == 'binary':
        return dict(ROC_AUC=100. * roc_auc_score(y, p.reshape(-1)),
                    AP=100. * average_precision_score(y, p.reshape(-1)))
    p = p.reshape(-1)
    y = y.reshape(-1)
    return dict(R2=100. * r2_score(y, p), MSE=float(np.mean((p-y)**2)))


def load_node(task):
    if task in LAND_TASKS:
        runner = importlib.import_module('run_graphland')
        d = runner._official_dataset_class()(name=task, device='cpu', split='RL', use_pyg=True)
        kind = 'regression' if d.task == 'regression' else 'binary'
        raw = type('NodeData', (), {})()
        raw.x, raw.y, raw.edge_index = d.features, d.targets, d.graph
        for key in ['train', 'val', 'test']:
            setattr(raw, key+'_mask', getattr(d, key+'_mask'))
        primary = 'R2' if kind == 'regression' else 'AP'
        def metric(pred, split):
            out = runner._score(d, pred, split)
            return {k: v*100. if k in ('AP', 'R2', 'ROC_AUC', 'accuracy', 'F1') else v
                    for k, v in out.items() if v is not None}
        return raw, kind, primary, metric
    if task == 'paris':
        from torch_geometric.datasets import CityNetwork
        raw = CityNetwork(root=str(data_path('city_networks')), name='paris', augmented=True)[0]
        for key in ['train', 'val', 'test']:
            mask = getattr(raw, key+'_mask')
            setattr(raw, key+'_mask', mask[:, 0] if mask.ndim == 2 else mask)
    else:
        raw = load_diverse(task, split=0)
    raw.y=raw.y.reshape(-1)
    kind = 'binary' if task in ['minesweeper', 'tolokers', 'questions'] else 'classification'
    primary = 'ROC_AUC' if kind == 'binary' else 'Accuracy'
    def metric(pred, split):
        idx = getattr(raw, split+'_mask')
        return node_metrics(pred[idx], raw.y[idx], kind)
    return raw, kind, primary, metric


def configs(task, kind):
    # Same complete architecture; task-specific optimizer and regularization.
    if task in ['cora', 'citeseer', 'pubmed']:
        opts = [(64, .5, .005, .005), (64, .35, .003, .001)]
    elif task in ['coauthor_cs', 'coauthor_physics', 'dblp', 'wikics']:
        opts = [(64, .2, .003, .001), (96, .3, .002, .001)]
    elif task in ['roman_empire', 'amazon_ratings', 'chameleon_filtered']:
        opts = [(64, .1, .003, .001), (96, .2, .002, .0001)]
    elif task == 'ogbn_arxiv':
        opts = [(64, .3, .002, .0001), (64, .15, .001, .0001)]
    elif task == 'paris':
        opts = [(64, .15, .002, .001), (96, .25, .001, .001)]
    elif task in LAND_TASKS:
        opts = [(64, .1, .002, .001), (64, .2, .001, .0001)]
    else:
        opts = [(64, .1, .002, .001), (96, .2, .001, .0001)]
    result = [dict(dim=d, heads=4, blocks=4, dropout=drop, lr=lr,
                 weight_decay=wd, angle=.2, epochs=600, patience=100)
            for d, drop, lr, wd in opts]
    if task in ['cora','citeseer','pubmed']:
        result += [dict(dim=dim,heads=4,blocks=4,dropout=drop,lr=lr,
                        weight_decay=wd,angle=angle,epochs=800,patience=150)
                   for dim,drop,lr,wd,angle in [(32,.6,.001,.1,.02),
                        (64,.5,.0005,.05,0.),(32,.3,.0005,.01,.02)]]
        result += [dict(dim=64,heads=4,blocks=4,dropout=.5,lr=.001,
                        weight_decay=.005,angle=.2,epochs=800,patience=150,
                        feature_transform=transform)
                   for transform in ['scale20','train_zscore_clip5']]
        result += [dict(dim=64,heads=4,blocks=4,dropout=.5,lr=.001,
                        weight_decay=.0001,angle=angle,epochs=400,patience=80,
                        feature_transform='scale20',input_l2=.0005)
                   for angle in [.02,math.pi/4]]
    elif task != 'ogbn_arxiv':
        result += [dict(dim=dim,heads=4,blocks=4,dropout=drop,lr=lr,
                        weight_decay=wd,angle=angle,epochs=800,patience=150)
                   for dim,drop,lr,wd,angle in [(128,.5,.0008,.01,.02),
                                              (64,.5,.0005,.05,0.)]]
        if task=='paris':
            for configuration in result[2:]:
                configuration['feature_transform']='train_zscore_clip5'
        if task in ['paris','city-reviews','twitch-views']:
            result[2]['dim']=64
        if kind=='binary':
            result[2]['class_weight']='sqrt_balance'
            result[3]['class_weight']='balance'
        if task in NODE_TASKS:
            result[2]['feature_transform']='train_zscore_clip5'
            result[3]['feature_transform']='scale20'
        if task in ['roman_empire','amazon_ratings','chameleon_filtered']:
            learning_rate={'roman_empire':.002,'amazon_ratings':.001,
                           'chameleon_filtered':.0005}[task]
            result.append(dict(dim=64,heads=4,blocks=4,dropout=.3 if task=='chameleon_filtered' else .1,
                               lr=learning_rate,weight_decay=.001,angle=math.pi/4,
                               epochs=400,patience=80))
    else:
        result.append(dict(dim=64,heads=4,blocks=4,dropout=.5,lr=.0005,
                      weight_decay=.01,angle=.02,epochs=800,patience=150,
                      feature_transform='scale20'))
    return result


def make_loss(kind, y, indices, class_weight=None):
    if kind == 'classification':
        return lambda p: F.cross_entropy(p.index_select(0, indices), y.index_select(0, indices).long())
    if kind == 'binary':
        pos_weight=None
        if class_weight is not None:
            rate=y.index_select(0,indices).float().mean().clamp(1e-4,1-1e-4)
            pos_weight=(1-rate)/rate
            if class_weight=='sqrt_balance':
                pos_weight=pos_weight.sqrt()
        return lambda p: F.binary_cross_entropy_with_logits(p.index_select(0, indices).reshape(-1),
                                                          y.index_select(0, indices).float(),pos_weight=pos_weight)
    return lambda p: F.mse_loss(p.index_select(0, indices).reshape(-1), y.index_select(0, indices).float())


def input_regularization(model, cfg):
    coefficient=cfg.get('input_l2',0.)
    if not coefficient:
        return 0.
    return .5*coefficient*(model.self_projection.weight.square().sum()+
                           model.graph_projection.weight.square().sum())


def node_recomputation(nodes,cfg):
    stages=max(0,(nodes-1).bit_length())
    padded=1<<stages
    return padded*cfg['dim']*cfg['blocks']*stages>128*1024*1024


def benchmark_node_training(model,x,neighbors,mapping,y,train_indices,kind,cfg):
    model.checkpoint_blocks=node_recomputation(len(mapping),cfg)
    for block in model.blocks:
        block.checkpoint_pairs=model.checkpoint_blocks
    model.train()
    lr=torch.tensor(cfg['lr'],device='cuda')
    optimizer=torch.optim.AdamW(model.parameters(),lr=lr,weight_decay=cfg['weight_decay'],
                                fused=True,capturable=True)
    loss_fn=make_loss(kind,y,train_indices,cfg.get('class_weight'))
    capture=not model.checkpoint_blocks and isinstance(x,torch.Tensor)
    def step():
        optimizer.zero_grad(set_to_none=not capture)
        loss=loss_fn(model(x,neighbors,mapping))+input_regularization(model,cfg)
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(),1.,foreach=True)
        optimizer.step()
        return loss
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    graph=None
    if capture:
        stream=torch.cuda.Stream();stream.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(stream):
            for _ in range(3):
                step()
        torch.cuda.current_stream().wait_stream(stream)
        sync();torch.cuda.empty_cache()
        graph=torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            loss=step()
    else:
        for _ in range(3):
            loss=step()
    times=[]
    for _ in range(10):
        sync();start=time.perf_counter()
        if graph is None:
            loss=step()
        else:
            graph.replay()
        sync();times.append(1000*(time.perf_counter()-start))
    if not bool(torch.isfinite(loss)):
        raise FloatingPointError('Nonfinite loss during node training benchmark')
    return dict(train_step_ms=float(np.median(times)),
                peak_training_MiB=torch.cuda.max_memory_allocated()/2**20,
                training_cuda_graph=capture,training_speed_measurement=2,
                activation_recomputation=model.checkpoint_blocks)


def train_node(task, args):
    out = args.output/task
    out.mkdir(parents=True, exist_ok=True)
    if (out/'result.json').is_file() and not args.retune:
        return json.loads((out/'result.json').read_text(encoding='utf-8'))
    raw, kind, primary, metric = load_node(task)
    n = len(raw.x)
    start = time.perf_counter()
    cache = out/'mapping.pt'
    if cache.is_file():
        c = torch.load(cache, weights_only=False)
        mapping = c['mapping']; compile_seconds = c['seconds']
    else:
        print(task, 'compiling', n, 'nodes', flush=True)
        mapping, audit = compile_loom(raw.edge_index, n)
        compile_seconds = time.perf_counter()-start
        save(cache, dict(mapping=mapping, seconds=compile_seconds, audit=audit))
    # Sparse multiply of static raw features is shared by every config/epoch.
    feature_start=time.perf_counter()
    x,neighbors = static_inputs(raw)
    sync()
    feature_preprocess_seconds=time.perf_counter()-feature_start
    input_cache={'none':(x,neighbors,feature_preprocess_seconds)}
    def inputs_for(configuration):
        transform=configuration.get('feature_transform','none')
        if transform not in input_cache:
            start=time.perf_counter()
            if transform=='scale20':
                transformed=raw.x*20.
            elif transform=='train_zscore_clip5':
                train=raw.x[raw.train_mask].double()
                mean=train.mean(0);std=train.std(0,unbiased=False).clamp_min(1e-6)
                transformed=((raw.x.double()-mean)/std).clamp(-5.,5.).float()
            else:
                raise ValueError(transform)
            value=SimpleNamespace(x=transformed,edge_index=raw.edge_index)
            a,b=static_inputs(value);sync()
            input_cache[transform]=(a,b,time.perf_counter()-start)
        return input_cache[transform]
    y = raw.y.cuda()
    mapping = mapping.cuda()
    train_idx = torch.where(raw.train_mask)[0].cuda()
    loss_fn = make_loss(kind, y, train_idx)
    output_dim = int(raw.y.max())+1 if kind == 'classification' else 1
    winner = None
    for config_id, cfg in enumerate(configs(task, kind)):
        if args.configs and config_id >= args.configs:
            break
        x,neighbors,feature_preprocess_seconds=inputs_for(cfg)
        loss_fn=make_loss(kind,y,train_idx,cfg.get('class_weight'))
        record_path = out/f'config_{config_id}.json'
        if record_path.is_file():
            record = json.loads(record_path.read_text())
        else:
            seed_all(args.seed)
            model = Loom(raw.x.shape[1], output_dim, n, **{k:cfg[k] for k in
                         ['dim','heads','blocks','dropout','angle']},
                         checkpoint_blocks=node_recomputation(n,cfg)).cuda()
            lr_tensor = torch.tensor(cfg['lr'], device='cuda')
            optimizer = torch.optim.AdamW(model.parameters(), lr=lr_tensor,
                                           weight_decay=cfg['weight_decay'], fused=True, capturable=True)
            def step():
                optimizer.zero_grad(set_to_none=True)
                prediction = model(x, neighbors, mapping)
                loss = loss_fn(prediction)+input_regularization(model,cfg)
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), 1., foreach=True)
                optimizer.step()
                return loss
            model.train()
            sync(); train_start = time.perf_counter()
            graph = None
            if not model.checkpoint_blocks and isinstance(x,torch.Tensor) and not args.no_cuda_graph:
                stream = torch.cuda.Stream()
                stream.wait_stream(torch.cuda.current_stream())
                with torch.cuda.stream(stream):
                    for _ in range(3):
                        step()
                torch.cuda.current_stream().wait_stream(stream)
                sync();torch.cuda.empty_cache()
                graph = torch.cuda.CUDAGraph()
                with torch.cuda.graph(graph):
                    captured_loss = step()
            best_val, best_epoch, stale, best_state = -float('inf'), 0, 0, None
            train_times = []
            torch.cuda.reset_peak_memory_stats()
            for epoch in range(1, cfg['epochs']+1):
                model.train()
                # Smooth decay retains useful late training after the fast start.
                lr_tensor.fill_(cfg['lr']*(.1 + .9*.5*(1+math.cos(math.pi*epoch/cfg['epochs']))))
                sync(); t = time.perf_counter()
                if graph is None:
                    loss = step()
                else:
                    graph.replay(); loss = captured_loss
                sync(); train_times.append((time.perf_counter()-t)*1000)
                if epoch % 5 == 0 or epoch == 1:
                    model.eval()
                    with torch.no_grad():
                        pred = model(x, neighbors, mapping).cpu()
                    value = metric(pred, 'val')[primary]
                    if value > best_val:
                        best_val, best_epoch, stale = value, epoch, 0
                        best_state = state_cpu(model)
                        save(out/f'config_{config_id}_best.pt',dict(model=best_state,config=cfg,
                             seed=args.seed,best_epoch=best_epoch,val=best_val))
                    else:
                        stale += 5
                    append(out/'epochs.jsonl', dict(config=config_id, epoch=epoch,
                           loss=float(loss), val=value, best_val=best_val))
                    if epoch % 50 == 0:
                        print(task, config_id, epoch, primary, round(value,4),
                              'best', round(best_val,4), flush=True)
                    if stale >= cfg['patience']:
                        break
            sync()
            peak = torch.cuda.max_memory_allocated()/2**20
            seconds = time.perf_counter()-train_start
            checkpoint_path = out/f'config_{config_id}_best.pt'
            save(checkpoint_path, dict(model=best_state, config=cfg, seed=args.seed,
                                      best_epoch=best_epoch, val=best_val))
            record = dict(config_id=config_id, config=cfg, validation=best_val,
                          best_epoch=best_epoch, epochs=epoch, train_seconds=seconds,
                          train_step_ms=float(np.median(train_times[10:] or train_times)),
                          peak_training_MiB=peak, cuda_graph=graph is not None)
            write_json(record_path, record)
            del graph, model, optimizer, best_state
            del loss
            if 'captured_loss' in locals():
                del captured_loss
            gc.collect(); torch.cuda.empty_cache()
        if winner is None or record['validation'] > winner['validation']:
            winner = record
    cfg = winner['config']
    x,neighbors,feature_preprocess_seconds=inputs_for(cfg)
    input_cache={cfg.get('feature_transform','none'):(x,neighbors,feature_preprocess_seconds)}
    model = Loom(raw.x.shape[1], output_dim, n, **{k:cfg[k] for k in
                 ['dim','heads','blocks','dropout','angle']}).cuda()
    checkpoint_record = torch.load(out/f"config_{winner['config_id']}_best.pt", weights_only=False)
    model.load_state_dict(checkpoint_record['model']); model.eval()
    with torch.no_grad():
        pred = model(x, neighbors, mapping).cpu()
        latency=benchmark_inference(lambda:model(x,neighbors,mapping),capture=isinstance(x,torch.Tensor))
    save(out/'test_predictions.pt', dict(prediction=pred, targets=raw.y,
                                         test_mask=raw.test_mask, seed=args.seed))
    training_speed=benchmark_node_training(model,x,neighbors,mapping,y,train_idx,kind,cfg)
    result = dict(dataset=task, seed=args.seed, primary=primary, metrics=metric(pred,'test'),
                  validation=winner['validation'], config=cfg, best_epoch=winner['best_epoch'],
                  compile_seconds=compile_seconds, **latency,
                  feature_preprocess_seconds=feature_preprocess_seconds,
                  **training_speed,
                  total_train_seconds=sum(json.loads(p.read_text())['train_seconds'] for p in out.glob('config_*.json')),
                  nodes=n, implementation='gcm.loom.Loom')
    write_json(out/'result.json', result)
    print('DONE', task, result['metrics'], flush=True)
    del model, x, neighbors, y, mapping
    gc.collect(); torch.cuda.empty_cache()
    return result


def report(output):
    rows = [json.loads(p.read_text(encoding='utf-8')) for p in output.glob('*/result.json')
            if p.parent.name!='laplace']
    expected=[t for t in ALL_TASKS if t!='laplace']+['laplace_A','laplace_B','laplace_C']
    order={task:i for i,task in enumerate(expected)}
    rows.sort(key=lambda row:order.get(row['dataset'],len(order)))
    with (output/'results.csv').open('w', newline='', encoding='utf-8-sig') as f:
        writer = csv.DictWriter(f, fieldnames=['Dataset','Seed','Metric','Result','Best epoch','Validation'])
        writer.writeheader()
        for row in rows:
            for key,value in row['metrics'].items():
                writer.writerow({'Dataset':row['dataset'],'Seed':row['seed'],'Metric':key,
                                 'Result':value,'Best epoch':row['best_epoch'],'Validation':row['validation']})
    with (output/'speed.csv').open('w', newline='', encoding='utf-8-sig') as f:
        names=['Dataset','Compile s','Feature preprocessing s','Train step ms','Inference ms',
               'Eager inference ms','Peak training MiB','Peak inference MiB','Total train s']
        writer=csv.writer(f); writer.writerow(names)
        for row in rows:
            writer.writerow([row['dataset'],row.get('compile_seconds'),row.get('feature_preprocess_seconds'),
                             row.get('train_step_ms'),row.get('inference_ms'),row.get('inference_eager_ms'),
                             row.get('peak_training_MiB'),row.get('peak_inference_MiB'),row.get('total_train_seconds')])
    lines=['# Loom 本轮运行结果','', '| 数据集 | 指标 | 结果 | 种子 |','|---|---|---:|---:|']
    for row in rows:
        key=row['primary']
        value=row['metrics'][key]
        formatted=f'{value:.6f}' if 0<abs(value)<.01 else f'{value:.4f}'
        lines.append(f"| {row['dataset']} | {key} | {formatted} | {row['seed']} |")
    plate=next((row for row in rows if row['dataset']=='plate'),None)
    if plate and 'Hamming_distance' in plate['metrics']:
        keys=['Hamming_distance','Path_agreement','Relabel_drift','Near_MSE','Far_MSE']
        lines.extend(['','## Plate 地址一致性','',
                      '| 放置方式 | Hamming 距离 | 路径一致性 | 重标号偏差 | 近区域 MSE | 远区域 MSE |',
                      '|---|---:|---:|---:|---:|---:|',
                      '| Stable reference | '+' | '.join(f"{plate['metrics'][key]:.6f}" for key in keys)+' |'])
    (output/'RESULTS.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    speed=['# Loom 本轮速度结果','',
           '| 数据集 | 编译 s | 特征预处理 s | 训练步 ms | 推理 ms | 训练峰值 MiB | 推理峰值 MiB |',
           '|---|---:|---:|---:|---:|---:|---:|']
    for row in rows:
        values=[row.get(key) for key in ['compile_seconds','feature_preprocess_seconds','train_step_ms',
                                         'inference_ms','peak_training_MiB','peak_inference_MiB']]
        speed.append('| '+row['dataset']+' | '+' | '.join('—' if v is None else f'{v:.3f}' for v in values)+' |')
    execution_path=output/'execution_benchmark.json'
    if execution_path.is_file():
        executions=json.loads(execution_path.read_text(encoding='utf-8'))
        speed.extend(['','## Roman Empire 执行速度','',
                      '| 执行方式 | 准确率 % | 训练步 ms | 推理 ms | 训练峰值 MiB | 推理峰值 MiB |',
                      '|---|---:|---:|---:|---:|---:|'])
        for row in executions['rows']:
            speed.append(f"| {row['Execution']} | {row['Accuracy']:.4f} | {row['Train_step_ms']:.3f} | "
                         f"{row['Inference_ms']:.3f} | {row['Peak_training_MiB']:.3f} | "
                         f"{row['Peak_inference_MiB']:.3f} |")
    (output/'SPEED.md').write_text('\n'.join(speed)+'\n',encoding='utf-8')
    finished={r['dataset'] for r in rows}
    write_json(output/'campaign_status.json',dict(completed=[t for t in expected if t in finished],
               pending=[t for t in expected if t not in finished],total=len(expected)))


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--tasks',nargs='+',default=ALL_TASKS)
    parser.add_argument('--seed',type=int,default=0)
    parser.add_argument('--configs',type=int,default=0)
    parser.add_argument('--no-cuda-graph',action='store_true')
    parser.add_argument('--retune',action='store_true')
    parser.add_argument('--output',type=Path,default=REPO/'results/loom')
    args=parser.parse_args()
    torch.set_num_threads(4)
    torch.set_float32_matmul_precision('high')
    args.output.mkdir(parents=True,exist_ok=True)
    from loom_graph_campaign import configurations as graph_configurations
    write_json(args.output/'plan.json',dict(tasks=args.tasks,seed=args.seed,model='Loom',
               hardware=torch.cuda.get_device_name(),tuning='validation',
               configs_per_task={task:len(configs(task,'')) if task not in GRAPH_TASKS else len(graph_configurations(task))
                                 for task in args.tasks}))
    failed=[]
    for task in args.tasks:
        write_json(args.output/'status.json',dict(current=task,failed=failed,state='running'))
        try:
            if task in GRAPH_TASKS:
                from loom_graph_campaign import train_graph
                train_graph(task,args)
            else:
                train_node(task,args)
            (args.output/task/'error.json').unlink(missing_ok=True)
        except Exception:
            error=traceback.format_exc()
            print(error,flush=True)
            write_json(args.output/task/'error.json',dict(error=error))
            failed.append(task)
            gc.collect(); torch.cuda.empty_cache()
        report(args.output)
    write_json(args.output/'status.json',dict(failed=failed,state='failed' if failed else 'complete'))


if __name__=='__main__':
    main()
