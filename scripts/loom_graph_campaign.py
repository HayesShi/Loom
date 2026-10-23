"""Full-split graph-task adapters and batched Loom training."""
from __future__ import annotations
import gc
import importlib
import math
from pathlib import Path
import time
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from gcm.paths import data_path
from gcm.loom import Loom, compile_loom, normalized_neighbors
from run_loom_campaign import write_json, save, append, seed_all, sync, state_cpu, benchmark_inference


def gray_grid(nodes):
    side = math.isqrt(nodes)
    ids = torch.arange(nodes)
    x, y = ids // side, ids % side
    x, y = x ^ (x >> 1), y ^ (y >> 1)
    result = torch.zeros(nodes, dtype=torch.long)
    for bit in range((side-1).bit_length()):
        result |= ((x >> bit) & 1) << (2*bit)
        result |= ((y >> bit) & 1) << (2*bit+1)
    return result


def load_graphs(task):
    if task.startswith(('reuse_','transfer_','scale_')):
        from loom_synthetic_data import build_dataset
        root=Path('results/loom')/task
        root.mkdir(parents=True,exist_ok=True)
        return build_dataset(task,root/'synthetic_raw.pt'),dict(task=task,primary='Reader_MSE',
                                            output_dim=8,normalizer=False)
    if task.startswith('lrim_'):
        module = importlib.import_module('run_lrim')
        dataset, indices, nodes, root = module._load_dataset(data_path('lrim'), task)
        datasets = {key:dataset[idx] for key,idx in indices.items()}
        return datasets, dict(task=task, primary='logMSE', output_dim=1,
                             mapping=gray_grid(nodes), normalizer=True)
    if task == 'echo_sssp':
        module = importlib.import_module('run_echo')
        datasets, info = module._load_task(data_path('echo_benchmark/official'), 'sssp', module._loader_class())
        return datasets, dict(task=task, primary='MAE', output_dim=1, raw_scale=40., normalizer=True)
    if task == 'plate':
        module = importlib.import_module('run_plate_deformation')
        datasets, info = module._load_data(data_path('plate_deformation/processed_static'),
                                          layout='plate_gray_contact')
        return datasets, dict(task=task, primary='state_MSE', output_dim=3, normalizer=True)
    if task == '2dele':
        module = importlib.import_module('run_2dele')
        datasets, info, mapping = module._load_data(module.OFFICIAL_ROOT/'data/2Dele', 'primal_dual')
        return datasets, dict(task=task, primary='Field_score', output_dim=4, normalizer=True)
    if task == 'hodge':
        module = importlib.import_module('run_hodge_magnetostatics')
        root = data_path('hodge_magnetostatics/official_cpu_seed42_3000')
        datasets, info, mapping = module.load_official(root/'flux_field_dataset.pkl',
                            root/'split_indices_public_code_seed42.npz')
        return datasets, dict(task=task, primary='MSE', output_dim=3, normalizer=True)
    if task == 'laplace':
        module = importlib.import_module('run_sni_laplace')
        datasets={key:module.SniDataset(data_path('sni_laplace')/filename,
                    key, 1.3, 'spatial_morton') for key,filename in module.FILES.items()}
        return datasets, dict(task=task, primary='Relative_L2_percent', output_dim=1, normalizer=False)
    raise ValueError(task)


def prepare(task, out):
    cache = out/'packed.pt'
    if cache.is_file():
        print(task,'loading cached complete data',flush=True)
        groups,info=torch.load(cache,weights_only=False)
        if task=='plate' and 'force_rms' not in info:
            energy=count=0.
            for group in groups['train'].values():
                force=group['x'][group['real']&~group['mask']][:,3:6].double()
                energy+=float(force.square().sum());count+=len(force)
            info['force_rms']=max(math.sqrt(energy/count),1e-6)
            save(cache,(groups,info))
        return groups,info
    datasets, info = load_graphs(task)
    topology_cache = {}
    groups = {}
    start = time.perf_counter()
    for split,dataset in datasets.items():
        split_groups={}
        for i in range(len(dataset)):
            g=dataset[i]
            n=len(g.x); padded=1 << (n-1).bit_length()
            if 'mapping' in info:
                mapping=info['mapping']
            elif task == 'plate':
                mapping=g.plate_node_to_wire
            else:
                key=(n,g.edge_index.numpy().tobytes())
                mapping=topology_cache.get(key)
                if mapping is None:
                    mapping,_=compile_loom(g.edge_index,n)
                    topology_cache[key]=mapping
            y=g.y.float().reshape(n,-1)
            score=getattr(g,'score_mask',torch.ones(n,dtype=torch.bool))
            active=torch.ones(n,dtype=torch.bool)
            boundary=torch.zeros(n,dtype=torch.bool)
            raw_target=None
            if task == 'plate':
                raw_target=(g.y-g.u0).float()
                y=raw_target[:,3:6]
                score=g.loss_mask
                active=g.node_type == 0
            elif task == 'laplace':
                boundary=g.boundary_mask
            x=torch.zeros(padded,g.x.shape[1])
            nei=torch.zeros_like(x)
            target=torch.zeros(padded,y.shape[1])
            real=torch.zeros(padded,dtype=torch.bool)
            mask=torch.zeros_like(real); act=torch.zeros_like(real); bdry=torch.zeros_like(real)
            x[mapping]=g.x.float()
            nei[mapping]=normalized_neighbors(g.x.float(),g.edge_index)
            target[mapping]=y
            real[mapping]=True; mask[mapping]=score; act[mapping]=active; bdry[mapping]=boundary
            row=dict(x=x,neighbors=nei,y=target,real=real,mask=mask,active=act,boundary=bdry)
            if task.startswith(('reuse_','transfer_','scale_')):
                source=torch.zeros_like(real);source[mapping]=g.x[:,-2]==1
                row['source']=source
                if task.startswith('transfer_') and split=='test' and i==0:
                    info['jacobian_example']=dict(x=x,edge_index=g.edge_index,mapping=mapping,
                                                real=real,readers=mask,sources=source)
            if raw_target is not None:
                raw=torch.zeros(padded,6); raw[mapping]=raw_target; row['raw_target']=raw
            split_groups.setdefault(padded,[]).append(row)
            if (i+1)%1000==0:
                print(task,'preparing',split,i+1,'/',len(dataset),flush=True)
        groups[split]={padded:{k:torch.stack([r[k] for r in rows]) for k in rows[0]}
                       for padded,rows in split_groups.items()}
    targets=torch.cat([g['y'][g['mask']] for g in groups['train'].values()])
    info['mean']=targets.mean(0) if info['normalizer'] else torch.zeros(info['output_dim'])
    info['std']=targets.std(0).clamp_min(1e-6) if info['normalizer'] else torch.ones(info['output_dim'])
    info['compile_seconds']=time.perf_counter()-start
    info['max_nodes']=max(max(g) for g in groups.values())
    info['train_max_nodes']=max(groups['train'])
    info['input_dim']=next(iter(groups['train'].values()))['x'].shape[-1]
    info.pop('mapping',None)
    if task=='plate':
        forces=torch.cat([group['x'][group['real']&~group['mask']][:,3:6]
                          for group in groups['train'].values()]).double()
        info['force_rms']=max(float(forces.square().sum(-1).mean().sqrt()),1e-6)
    save(cache,(groups,info))
    return groups,info


def configurations(task):
    if task.startswith(('reuse_','transfer_','scale_')):
        base=dict(dim=64,heads=4,blocks=4 if task.startswith('scale_') else 1,
                  dropout=0.,weight_decay=.0001,batch=4 if task=='scale_8192' else 64)
        return [dict(base,angle=.2,lr=.002,epochs=150,patience=35),
                dict(base,angle=math.pi/4,lr=.001,
                     epochs=60 if task=='scale_8192' else 150,
                     patience=20 if task=='scale_8192' else 35)]
    if task.startswith('lrim_'):
        lr=.002; epochs=150; patience=30; batch=32
    elif task == 'echo_sssp':
        lr=.002; epochs=120; patience=25; batch=32
    elif task == 'plate':
        lr=.002; epochs=500; patience=100; batch=16
    elif task == '2dele':
        lr=.001; epochs=800; patience=150; batch=16
    elif task == 'hodge':
        lr=.002; epochs=80; patience=25; batch=8
    else:
        lr=.002; epochs=100; patience=20; batch=64
    fast_batch=128 if task.startswith('lrim_16_') else 32 if task.startswith('lrim_32_') or task=='plate' else 512 if task=='laplace' else batch
    candidates=[dict(dim=64,heads=4,blocks=4,dropout=drop,angle=angle,lr=lr*factor,
                     weight_decay=.0001,epochs=epochs,patience=patience,batch=bs)
                for drop,angle,factor,bs in [(0.,.2,1.,fast_batch),(.05,.05,.5,batch)]]
    if task=='echo_sssp':
        candidates.append(dict(dim=64,heads=4,blocks=4,dropout=0.,angle=math.pi/4,
                               lr=.001,weight_decay=.0001,epochs=300,patience=50,batch=64))
    elif task=='laplace':
        candidates.append(dict(dim=64,heads=4,blocks=4,dropout=0.,angle=math.pi/4,
                               lr=.001,weight_decay=.0001,epochs=200,patience=40,batch=128))
    elif task=='plate':
        candidates.append(dict(dim=64,heads=4,blocks=4,dropout=.05,angle=.05,
                               lr=.001,weight_decay=.0001,epochs=200,patience=50,batch=32,
                               force_feature_scale='train_rms'))
    return candidates


def scale_batch_features(batch,info,configuration):
    if not configuration or not configuration.get('force_feature_scale'):
        return batch
    output=dict(batch)
    for key in ['x','neighbors']:
        value=batch[key].clone()
        value[...,3:6].div_(info['force_rms'])
        output[key]=value
    return output


def decode(pred,batch,info):
    pred=pred*info['std'].to(pred.device)+info['mean'].to(pred.device)
    if info['task']=='plate':
        pred=pred*batch['active'].unsqueeze(-1)
    elif info['task']=='laplace':
        pred=torch.where(batch['boundary'].unsqueeze(-1),batch['y'],pred)
    return pred


def loss_value(pred,batch,info):
    prediction=decode(pred,batch,info)
    mask=batch['mask'].unsqueeze(-1)
    residual=prediction-batch['y']
    if info['task']=='laplace':
        error=(residual.square()*mask).sum((1,2))
        denominator=(batch['y'].square()*mask).sum((1,2)).clamp_min(1e-12)
        return (error/denominator+1e-12).sqrt().mean()
    error=residual/info['std'].to(pred.device)
    if info['task']=='echo_sssp':
        return (error.abs()*mask).sum()/(mask.sum()*error.shape[-1])
    return (error.square()*mask).sum()/(mask.sum()*error.shape[-1])


@torch.no_grad()
def evaluate(model,groups,info,batch_size,prediction_path=None,configuration=None):
    model.eval()
    sse=sae=sumy=sumy2=0.; count=nodes=0
    relative=[]; plate_scores=[]; field_scores=[]
    saved={}
    for padded,g in groups.items():
        predictions=[]
        for first in range(0,len(g['x']),batch_size):
            batch={k:v[first:first+batch_size].cuda() for k,v in g.items()}
            batch=scale_batch_features(batch,info,configuration)
            pred=decode(model.forward_packed(batch['x'],batch['neighbors'],batch['real']),batch,info)
            if prediction_path is not None:
                predictions.append(pred.cpu())
            mask=batch['mask']
            error=(pred-batch['y'])[mask].double()
            y=batch['y'][mask].double()
            sse+=float(error.square().sum()); sae+=float(error.abs().sum())
            sumy+=float(y.sum()); sumy2+=float(y.square().sum())
            count+=y.numel(); nodes+=len(y)
            if info['task']=='plate':
                full=torch.cat((pred,pred),-1)-batch['raw_target']
                plate_scores.append(float(full[mask].double().square().sum()/mask.sum()))
            if info['task']=='2dele':
                field_scores.append(float(error.square().sum()/mask.sum()))
            if info['task']=='laplace':
                error=(pred-batch['y']).square().sum((1,2))
                den=batch['y'].square().sum((1,2)).clamp_min(1e-12)
                relative.extend((100*(error/den).sqrt()).cpu().tolist())
        if prediction_path is not None:
            saved[padded]=dict(prediction=torch.cat(predictions),target=g['y'],mask=g['mask'],real=g['real'])
    if prediction_path is not None:
        save(prediction_path,saved)
    mse=sse/count
    scale=info.get('raw_scale',1.)
    metrics=dict(MSE=mse*scale**2,MAE=sae/count*scale,RMSE=math.sqrt(mse)*scale,
                 R2=1-sse/max(sumy2-sumy**2/count,1e-12),logMSE=math.log10(max(mse*scale**2,1e-30)))
    if info['task'].startswith(('reuse_','transfer_','scale_')):
        metrics['Reader_MSE']=mse
    if plate_scores:
        metrics['state_MSE']=float(np.mean(plate_scores))
    if field_scores:
        metrics['Field_score']=float(np.mean(field_scores))
    if relative:
        metrics['Relative_L2_percent']=float(np.mean(relative))
    return metrics


@torch.no_grad()
def source_probe(model,groups,info):
    from sklearn.linear_model import Ridge
    arrays={};ratios=[]
    for split in ['train','test']:
        features=[];payloads=[]
        for padded,g in groups[split].items():
            for first in range(0,len(g['x']),64):
                b={k:v[first:first+64].cuda() for k,v in g.items()}
                z=model.encode_packed(b['x'],b['neighbors'],b['real'])
                features.append(model.head_norm(z)[b['source']].cpu().numpy())
                payloads.append(b['y'][b['source']].cpu().numpy())
                if split=='test':
                    h=model.embed_norm(model.self_projection(b['x'])+model.graph_projection(b['neighbors']))
                    h=h*b['real'].unsqueeze(-1)
                    entry=model.blocks[0].norm_mix(h)
                    mixed=model.blocks[0].mix(entry)
                    ratio=mixed.norm(dim=(1,2))/entry.norm(dim=(1,2)).clamp_min(1e-12)
                    ratios.extend(ratio.cpu().tolist())
        arrays[split]=(np.concatenate(features),np.concatenate(payloads))
    probe=Ridge(alpha=.01).fit(*arrays['train'])
    error=probe.predict(arrays['test'][0])-arrays['test'][1]
    return dict(Source_MSE=float(np.mean(error**2)),State_norm_ratio=float(np.mean(ratios)))


def effective_rank(model,info):
    example=info['jacobian_example']
    x=example['x'].cuda();mapping=example['mapping'].cuda()
    source=torch.where(example['sources'])[0].cuda()
    readers=torch.where(example['readers'])[0].cuda()
    edges=example['edge_index'].cuda();src,dst=edges
    n=len(mapping);padded=len(x)
    out=torch.bincount(src,minlength=n).float().clamp_min(1)
    incoming=torch.bincount(dst,minlength=n).float().clamp_min(1)
    a=torch.zeros(padded,padded,device='cuda')
    a[mapping[dst],mapping[src]]=(out[src]*incoming[dst]).rsqrt()
    real=example['real'].cuda().unsqueeze(0)
    model.eval()
    def prediction(payload):
        rows=torch.cat((payload,x[source,8:]),1)
        value=x.index_copy(0,source,rows)
        output=model.forward_packed(value.unsqueeze(0),(a@value).unsqueeze(0),real)
        return output[0,readers]
    jacobian=torch.autograd.functional.jacobian(prediction,x[source,:8],vectorize=False)
    matrix=jacobian.reshape(len(readers)*8,len(source)*8)
    sigma=torch.linalg.svdvals(matrix.double())
    if not bool(sigma.sum()>0):
        return 0.
    p=sigma/sigma.sum().clamp_min(1e-30)
    return float(torch.exp(-(p*p.clamp_min(1e-30).log()).sum()))


def benchmark_preprocessing(task,force_scale=None):
    datasets,info=load_graphs(task)
    dataset=datasets['test'] if 'test' in datasets else datasets['A']
    times=[];feature_times=[]
    for index in range(min(3,len(dataset))):
        graph=dataset[index]
        start=time.perf_counter()
        if task.startswith('lrim_'):
            gray_grid(graph.num_nodes)
        elif task=='plate':
            from gcm.plate_geometry import compile_plate_gray_contact
            compile_plate_gray_contact(graph)
        else:
            compile_loom(graph.edge_index,graph.num_nodes)
        times.append(time.perf_counter()-start)
        start=time.perf_counter()
        value=graph.x.float()
        if force_scale is not None:
            value=value.clone();value[:,3:6].div_(force_scale)
        normalized_neighbors(value,graph.edge_index)
        feature_times.append(time.perf_counter()-start)
    return dict(compile_seconds=float(np.median(times)),
                feature_preprocess_seconds=float(np.median(feature_times)))


def active_parameters(model,padded):
    stages=padded.bit_length()-1
    return [p for name,p in model.named_parameters()
            if '.pairs.' not in name or int(name.split('.pairs.')[1].split('.')[0])<stages]


class CapturedStep:
    def __init__(self,model,optimizer,example,info):
        self.batch={k:v[:1].repeat((1,)+(1,)*(v.ndim-1)).cuda() for k,v in example.items()}
        self.model=model; self.optimizer=optimizer; self.info=info
        self.graph=None

    def capture(self,batch):
        self.batch={k:v.cuda().clone() for k,v in batch.items()}
        model,optimizer,info=self.model,self.optimizer,self.info
        parameters=active_parameters(model,self.batch['x'].shape[-2])
        original_parameters=optimizer.param_groups[0]['params']
        for parameter in parameters:
            if parameter.grad is None:
                parameter.grad=torch.zeros_like(parameter)
        def step():
            optimizer.zero_grad(set_to_none=False)
            pred=model.forward_packed(self.batch['x'],self.batch['neighbors'],self.batch['real'])
            loss=loss_value(pred,self.batch,info)
            loss.backward()
            nn.utils.clip_grad_norm_(parameters,1.,foreach=True)
            optimizer.step()
            return loss
        optimizer.param_groups[0]['params']=parameters
        try:
            stream=torch.cuda.Stream(); stream.wait_stream(torch.cuda.current_stream())
            with torch.cuda.stream(stream):
                for _ in range(3):
                    step()
            torch.cuda.current_stream().wait_stream(stream)
            sync()
            torch.cuda.empty_cache()
            self.graph=torch.cuda.CUDAGraph()
            with torch.cuda.graph(self.graph):
                self.loss=step()
        finally:
            optimizer.param_groups[0]['params']=original_parameters

    def replay(self,batch):
        for k,v in batch.items():
            self.batch[k].copy_(v,non_blocking=True)
        self.graph.replay()
        return self.loss


def benchmark_graph_training(model,groups,info,cfg):
    train=groups['train']
    train_bytes=sum(v.numel()*v.element_size() for g in train.values() for v in g.values())
    if train_bytes<750*2**20:
        train={size:{k:v.cuda() for k,v in g.items()} for size,g in train.items()}
    padded=max(train,key=lambda size:len(train[size]['x']))
    example=train[padded]
    batch={k:v[:cfg['batch']].cuda() for k,v in example.items()}
    batch=scale_batch_features(batch,info,cfg)
    model.checkpoint_blocks=cfg['batch']*max(train)>65536
    for block in model.blocks:
        block.checkpoint_pairs=model.checkpoint_blocks
    model.train()
    optimizer=torch.optim.AdamW(model.parameters(),lr=torch.tensor(cfg['lr'],device='cuda'),
                                weight_decay=cfg['weight_decay'],fused=True,capturable=True)
    capture=not (model.checkpoint_blocks and cfg['dropout']>0)
    torch.cuda.empty_cache();torch.cuda.reset_peak_memory_stats()
    captured=None
    if capture:
        captured=CapturedStep(model,optimizer,example,info)
        captured.capture(batch)
        step=lambda:captured.replay(batch)
    else:
        def step():
            optimizer.zero_grad(set_to_none=True)
            loss=loss_value(model.forward_packed(batch['x'],batch['neighbors'],batch['real']),batch,info)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(),1.,foreach=True)
            optimizer.step()
            return loss
        for _ in range(3):
            step()
    times=[]
    for _ in range(10):
        sync();start=time.perf_counter();loss=step();sync()
        times.append(1000*(time.perf_counter()-start))
    if not bool(torch.isfinite(loss)):
        raise FloatingPointError('Nonfinite loss during graph training benchmark')
    return dict(train_step_ms=float(np.median(times)),
                peak_training_MiB=torch.cuda.max_memory_allocated()/2**20,
                training_speed_measurement=1,training_cuda_graph=capture,
                training_benchmark_padded=padded,training_benchmark_batch=len(batch['x']))


def train_graph(task,args):
    out=args.output/task; out.mkdir(parents=True,exist_ok=True)
    if (out/'result.json').is_file():
        count=len(configurations(task))
        if args.configs:
            count=min(count,args.configs)
        if not args.retune or all((out/f'config_{i}.json').is_file() for i in range(count)):
            return
    groups,info=prepare(task,out)
    info['mean']=info['mean'].cuda()
    info['std']=info['std'].cuda()
    train_bytes=sum(v.numel()*v.element_size() for g in groups['train'].values() for v in g.values())
    if train_bytes<750*2**20:
        groups['train']={padded:{k:v.cuda() for k,v in g.items()} for padded,g in groups['train'].items()}
    print(task,'full split sizes',{k:sum(len(v['x']) for v in g.values()) for k,g in groups.items()},flush=True)
    configs=configurations(task)
    winner=None
    for config_id,cfg in enumerate(configs):
        if args.configs and config_id>=args.configs:
            break
        record_path=out/f'config_{config_id}.json'
        if record_path.is_file():
            import json
            record=json.loads(record_path.read_text())
        else:
            seed_all(args.seed)
            model=Loom(info['input_dim'],info['output_dim'],info['max_nodes'],
                       **{k:cfg[k] for k in ['dim','heads','blocks','dropout','angle']},
                       checkpoint_blocks=(cfg['batch']*max(groups['train'])>65536)).cuda()
            lr=torch.tensor(cfg['lr'],device='cuda')
            optimizer=torch.optim.AdamW(model.parameters(),lr=lr,weight_decay=cfg['weight_decay'],
                                         fused=True,capturable=True)
            best=math.inf; stale=0; best_state=None; step_ms=[]; captured={}
            resume_epoch=0
            checkpoint_path=out/f'config_{config_id}_best.pt'
            if checkpoint_path.is_file():
                previous=torch.load(checkpoint_path,weights_only=False)
                comparable=['dim','heads','blocks','dropout','angle','lr','weight_decay']
                if previous['seed']!=args.seed or any(previous['config'][k]!=cfg[k] for k in comparable):
                    raise ValueError(f'{task}: saved checkpoint configuration differs')
                model.load_state_dict(previous['model'])
                best=float(previous['val']);best_epoch=int(previous['epoch'])
                resume_epoch=best_epoch;best_state=state_cpu(model)
                print(task,config_id,'continuing best checkpoint at epoch',resume_epoch,flush=True)
                del previous
            shuffler=torch.Generator(device='cuda').manual_seed(args.seed+17)
            torch.cuda.reset_peak_memory_stats(); sync(); start=time.perf_counter()
            epoch=resume_epoch
            for epoch in range(resume_epoch+1,cfg['epochs']+1):
                model.train()
                lr.fill_(cfg['lr']*(.05+.95*.5*(1+math.cos(math.pi*epoch/cfg['epochs']))))
                losses=[]
                for padded,g in groups['train'].items():
                    order=torch.randperm(len(g['x']),device='cuda',generator=shuffler)
                    for first in range(0,len(order),cfg['batch']):
                        ids=order[first:first+cfg['batch']]
                        batch={k:v[ids].cuda() for k,v in g.items()}
                        batch=scale_batch_features(batch,info,cfg)
                        key=(padded,len(ids))
                        sync(); t=time.perf_counter()
                        if not args.no_cuda_graph and not (model.checkpoint_blocks and cfg['dropout']>0):
                            if key not in captured:
                                # Each captured backward keeps its own gradient buffers.
                                # Sharing an optimizer across different graphs needs
                                # fixed gradient tensors, hence set_to_none=False below.
                                captured[key]=CapturedStep(model,optimizer,g,info)
                                captured[key].capture(batch)
                            loss=captured[key].replay(batch)
                        else:
                            optimizer.zero_grad(set_to_none=True)
                            pred=model.forward_packed(batch['x'],batch['neighbors'],batch['real'])
                            loss=loss_value(pred,batch,info)
                            loss.backward(); nn.utils.clip_grad_norm_(model.parameters(),1.,foreach=True)
                            optimizer.step()
                        sync(); step_ms.append(1000*(time.perf_counter()-t)); losses.append(float(loss))
                if epoch%5==0 or epoch==1:
                    metrics=evaluate(model,groups['val'],info,cfg['batch'],configuration=cfg)
                    value=metrics[info['primary']]
                    if not math.isfinite(value):
                        raise FloatingPointError(f'{task}: nonfinite validation')
                    if value<best:
                        best=value; best_epoch=epoch; best_state=state_cpu(model); stale=0
                        save(out/f'config_{config_id}_best.pt',dict(model=best_state,config=cfg,
                                     val=value,epoch=epoch,seed=args.seed))
                    else:
                        stale+=5
                    append(out/'epochs.jsonl',dict(config=config_id,epoch=epoch,
                           loss=float(np.mean(losses)),validation=value,best=best))
                    if epoch%10==0 or epoch==1:
                        print(task,config_id,epoch,info['primary'],round(value,6),'best',round(best,6),flush=True)
                    if stale>=cfg['patience']:
                        break
            sync()
            record=dict(config_id=config_id,config=cfg,validation=best,best_epoch=best_epoch,
                        epochs=epoch,train_seconds=time.perf_counter()-start,
                        train_step_ms=float(np.median(step_ms[10:] or step_ms)) if step_ms else None,
                        peak_training_MiB=torch.cuda.max_memory_allocated()/2**20,
                        training_execution=2,resumed_from_epoch=resume_epoch)
            write_json(record_path,record)
            del captured,model,optimizer,best_state
            if step_ms:
                del loss,batch
            gc.collect(); torch.cuda.empty_cache()
        if winner is None or record['validation']<winner['validation']:
            winner=record
    cfg=winner['config']
    model=Loom(info['input_dim'],info['output_dim'],info['max_nodes'],
               **{k:cfg[k] for k in ['dim','heads','blocks','dropout','angle']}).cuda()
    checkpoint=torch.load(out/f"config_{winner['config_id']}_best.pt",weights_only=False)
    model.load_state_dict(checkpoint['model']); model.eval()
    splits=['A','B','C'] if task=='laplace' else ['test']
    results={split:evaluate(model,groups[split],info,cfg['batch'],out/f'{split}_predictions.pt',cfg) for split in splits}
    if task.startswith('reuse_'):
        results['test'].update(source_probe(model,groups,info))
    if task=='transfer_32':
        results['test']['Effective_rank']=effective_rank(model,info)
    example=next(iter(groups[splits[0]].values()))
    batch={k:v[:1].cuda() for k,v in example.items()}
    batch=scale_batch_features(batch,info,cfg)
    latency=benchmark_inference(lambda:model.forward_packed(batch['x'],batch['neighbors'],batch['real']))
    import json
    records=[json.loads(p.read_text()) for p in out.glob('config_*.json')]
    seconds=None if any(record.get('resumed_from_epoch',0) for record in records) else sum(record['train_seconds'] for record in records)
    preprocessing=benchmark_preprocessing(task,info['force_rms'] if cfg.get('force_feature_scale') else None)
    base=dict(seed=args.seed,primary=info['primary'],validation=winner['validation'],
              config=cfg,best_epoch=winner['best_epoch'],**preprocessing,
              dataset_preparation_seconds=info['compile_seconds'],
              **latency,train_step_ms=winner['train_step_ms'],
              peak_training_MiB=winner['peak_training_MiB'],total_train_seconds=seconds,
              implementation='gcm.loom.Loom',training_execution=2,preprocessing_measurement=1)
    if task=='laplace':
        for split,metrics in results.items():
            example=next(iter(groups[split].values()))
            domain_batch={k:v[:1].cuda() for k,v in example.items()}
            domain_latency=benchmark_inference(lambda:model.forward_packed(domain_batch['x'],domain_batch['neighbors'],domain_batch['real']))
            write_json(args.output/f'laplace_{split}'/'result.json',dict(base,**domain_latency,dataset=f'laplace_{split}',metrics=metrics))
        write_json(out/'result.json',dict(base,dataset=task,metrics=results['A'],domains=results))
    else:
        write_json(out/'result.json',dict(base,dataset=task,metrics=results['test']))
    print('DONE',task,results,flush=True)
    del model,groups
    gc.collect();torch.cuda.empty_cache()
