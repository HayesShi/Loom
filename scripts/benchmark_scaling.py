"""Full-model large-graph systems experiment; no synthetic accuracy claims.
Run with a CUDA-enabled PyTorch environment. Each case is separately resumable.
"""
import argparse, gc, json, sys, time, statistics, traceback
from pathlib import Path
from types import SimpleNamespace
ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'results/scaling'
SGSOURCE=ROOT/'external/SGFormer/medium'
sys.path[:0]=[str(ROOT/'src'),str(ROOT/'scripts'),str(SGSOURCE)]
import numpy as np
import torch
from gcm.loom import Loom, compile_loom, normalized_neighbors
TRACE=None

def make_sgformer(din):
    from ours import SGFormer
    from models import GCN
    gnn=GCN(din,64,64,num_layers=4,dropout=.1,save_mem=True,use_bn=True)
    model=SGFormer(din,64,40,num_layers=1,num_heads=1,alpha=.5,dropout=.1,
        use_bn=True,use_residual=True,use_weight=True,use_graph=True,use_act=False,
        graph_weight=.8,gnn=gnn,aggregate='add').cuda().eval()
    cfg=dict(hidden_width=64,dropout=.1,attention_layers=1,attention_heads=1,
        gnn_layers=4,graph_weight=.8,alpha=.5,gnn_normalization_cached=False,
        implementation='official qitianwu/SGFormer medium/ours.py SGFormer + models.py GCN')
    return model,cfg
def sparse_graph(edge,n):
    from torch_sparse import SparseTensor
    # Explicit unit values let coalescing retain duplicate-edge multiplicity.
    adj=SparseTensor(row=edge[1],col=edge[0],value=torch.ones(edge.shape[1],device=edge.device),
                     sparse_sizes=(n,n)).coalesce()
    adj.csr()
    return adj

def sync(): torch.cuda.synchronize()
def timed(call):
    sync(); t=time.perf_counter(); value=call(); sync()
    return value,(time.perf_counter()-t)*1000
def measure(call,repeats,scope='full'):
    for _ in range(3): call()
    sync(); torch.cuda.reset_peak_memory_stats(); samples=[]
    for i in range(repeats):
        value,ms=timed(call); samples.append(ms)
        if TRACE:
            with TRACE.open('a',encoding='utf8') as trace:
                trace.write(json.dumps(dict(scope=scope,repeat=i,wall_ms=ms,
                    peak_allocated_MiB=torch.cuda.max_memory_allocated()/2**20,
                    peak_reserved_MiB=torch.cuda.max_memory_reserved()/2**20))+'\n')
        if not bool(torch.isfinite(value).all()): raise RuntimeError('Nonfinite output')
        del value
    return dict(raw_wall_ms=samples,median_ms=statistics.median(samples),
                mean_ms=statistics.mean(samples),timing_sd_ms=statistics.stdev(samples),
                peak_allocated_MiB=torch.cuda.max_memory_allocated()/2**20,
                peak_reserved_MiB=torch.cuda.max_memory_reserved()/2**20)
def synthetic(n):
    # Connected sparse small-world family: undirected ring + 3 random offsets per node.
    rng=np.random.default_rng(0); u=np.repeat(np.arange(n,dtype=np.int64),4)
    v=np.empty((n,4),dtype=np.int64); v[:,0]=(np.arange(n)+1)%n
    v[:,1:]=(np.arange(n)[:,None]+rng.integers(1,n,size=(n,3)))%n
    v=v.reshape(-1); edge=torch.from_numpy(np.stack([np.concatenate([u,v]),np.concatenate([v,u])]))
    gen=torch.Generator().manual_seed(0)
    return SimpleNamespace(x=torch.randn(n,128,generator=gen),edge_index=edge,y=torch.arange(n)%40)
def main():
    global TRACE
    p=argparse.ArgumentParser(); p.add_argument('--nodes',type=int,default=4096)
    p.add_argument('--dataset',default='synthetic'); p.add_argument('--method',default='loom',choices=['loom','loom_cuda','loom_cuda_gc','sgformer','sgformer_sparse'])
    p.add_argument('--repeats',type=int,default=10);p.add_argument('--rerun',action='store_true');p.add_argument('--skip-cached',action='store_true');p.add_argument('--compile-only',action='store_true'); a=p.parse_args(); OUT.mkdir(parents=True,exist_ok=True)
    key=f'{a.dataset}_{a.nodes}_{a.method}'; path=OUT/f'{key}.json'
    if a.method=='sgformer_sparse' and (OUT/'stop_sgformer_sparse.txt').exists():
        print('Sparse execution expansion cancelled by user; existing results retained',flush=True);return
    if a.method=='loom_cuda_gc' and (OUT/'stop_loom_cuda_gc.txt').exists():
        print('Managed-GC expansion stopped after ineffective memory experiment',flush=True);return
    prior=None
    if path.exists() and a.rerun:
        prior=json.loads(path.read_text())
        archive=OUT/'prior_observations';archive.mkdir(exist_ok=True)
        path.rename(archive/f'{key}_{int(time.time())}.json')
    if path.exists(): print('Already measured',key); return
    TRACE=OUT/f'{key}_timing_progress.jsonl'
    if TRACE.exists() and a.rerun:
        TRACE.rename(OUT/'prior_observations'/f'{key}_timing_progress_{int(time.time())}.jsonl')
    torch.set_num_threads(4); torch.manual_seed(0)
    torch.backends.cuda.matmul.allow_tf32=False; torch.backends.cudnn.allow_tf32=False
    row=dict(case=key,dataset=a.dataset,method=a.method,hardware=torch.cuda.get_device_name(),
             torch=torch.__version__,cuda=torch.version.cuda,precision='float32',tf32=False,
             warmups=3,repeats=a.repeats,seed=0,status='started',accuracy=None,
             timing='CUDA synchronized wall clock; eager; excludes loading and transfer')
    row['device_total_MiB']=torch.cuda.get_device_properties(0).total_memory/2**20
    try:
        if a.dataset=='ogbn_arxiv':
            from gcm.diverse_data import load_diverse
            raw=load_diverse(a.dataset); campaign=ROOT/'results/loom'/a.dataset
        elif a.dataset=='city-roads-L':
            from run_graphland import _official_dataset_class
            d=_official_dataset_class()(name=a.dataset,device='cpu',split='RL',use_pyg=True)
            raw=SimpleNamespace(x=d.features,edge_index=d.graph,y=torch.zeros(len(d.features),dtype=torch.long))
        else: raw=synthetic(a.nodes)
        n,din=raw.x.shape; row.update(nodes=n,edges=int(raw.edge_index.shape[1]),input_dim=din,padded=1<<(n-1).bit_length())
        if a.compile_only:
            del raw.x
            start=time.perf_counter();mapping,audit=compile_loom(raw.edge_index,n,seed=0)
            comp=dict(nodes=n,edges=int(raw.edge_index.shape[1]),seed=0,compile_ms=(time.perf_counter()-start)*1000,
                compiler_audit={k:v for k,v in audit.items() if k!='cost_history'},status='compiled',
                mapping_file=f'synthetic_{n}_loom_mapping.pt',implementation='Unchanged complete compile_loom; CPU only')
            torch.save(dict(mapping=mapping,audit=audit),OUT/comp['mapping_file'])
            folder=OUT/'compilation';folder.mkdir(exist_ok=True)
            (folder/f'synthetic_{n}_loom.json').write_text(json.dumps(comp,indent=2));print(json.dumps(comp),flush=True);return
        x=raw.x.cuda(); edge=raw.edge_index.cuda()
        if a.method in ['loom','loom_cuda','loom_cuda_gc']:
            cfg=dict(dim=64,heads=4,blocks=4,dropout=.3,angle=.2)
            if a.dataset=='ogbn_arxiv':
                mapping=torch.load(campaign/'mapping.pt',weights_only=False)['mapping']
                row['mapping_source']=str(campaign/'mapping.pt'); row['compile_ms']=None
            elif (OUT/'compilation'/f'synthetic_{n}_loom.json').exists():
                comp=json.loads((OUT/'compilation'/f'synthetic_{n}_loom.json').read_text())
                mapping=torch.load(OUT/comp['mapping_file'],weights_only=False)['mapping']
                row['compile_ms']=comp['compile_ms'];row['compiler_audit']=comp['compiler_audit']
                row['mapping_source']=comp['mapping_file'];row['compile_observation']='Complete CPU-only compiler measurement reused'
            elif (OUT/f"synthetic_{n}_{'loom' if a.method!='loom' else 'loom_cuda_gc'}_mapping.pt").exists():
                peer='loom' if a.method!='loom' else 'loom_cuda_gc'
                base=json.loads((OUT/f'synthetic_{n}_{peer}.json').read_text())
                mapping=torch.load(OUT/f'synthetic_{n}_{peer}_mapping.pt',weights_only=False)['mapping']
                row['compile_ms']=base.get('compile_ms');row['compiler_audit']=base.get('compiler_audit')
                row['mapping_source']=f'synthetic_{n}_{peer}_mapping.pt'
                row['compile_observation']='Same full compiler observation and mapping as original execution'
            elif prior is not None and (OUT/f'{key}_mapping.pt').exists():
                saved=torch.load(OUT/f'{key}_mapping.pt',weights_only=False)
                mapping=saved['mapping'];row['compile_ms']=prior['compile_ms'];row['compiler_audit']=prior.get('compiler_audit')
                row['compile_observation']='Original full compiler measurement retained; mapping reused for inference remeasurement'
            else:
                start=time.perf_counter(); mapping,audit=compile_loom(raw.edge_index,n,seed=0)
                row['compile_ms']=(time.perf_counter()-start)*1000
                row['compiler_audit']={k:v for k,v in audit.items() if k!='cost_history'}
                torch.save(dict(mapping=mapping,audit=audit),OUT/f'{key}_mapping.pt')
            model=Loom(din,1 if a.dataset=='city-roads-L' else 40,n,**cfg).cuda().eval(); mapping=mapping.cuda()
            if a.method in ['loom_cuda','loom_cuda_gc']:model.enable_cuda_stages()
            if a.method=='loom_cuda_gc':
                import gcm.loom_cuda as kernel
                original_stage=kernel.stage
                def managed_stage(*args,**kwargs):
                    value=original_stage(*args,**kwargs)
                    gc.collect()
                    return value
                kernel.stage=managed_stage
            if a.dataset=='ogbn_arxiv':
                checkpoint=torch.load(campaign/'config_0_best.pt',weights_only=False)
                model.load_state_dict(checkpoint['model']); row['checkpoint']=str(campaign/'config_0_best.pt')
            row.update(config=cfg,implementation='src/gcm/loom.py Loom + compile_loom',
                       weights='trained checkpoint' if a.dataset=='ogbn_arxiv' else 'fixed random initialization; systems only')
            row['execution']='Full FP32 fused CUDA stages with per-stage Python ownership collection; unchanged embedding/FFN/head' if a.method=='loom_cuda_gc' else ('Full FP32 fused CUDA mixing stages; ordinary eager embedding/FFN/head' if a.method=='loom_cuda' else 'Full grouped-pair PyTorch eager execution')
            neighbors,prep_ms=timed(lambda:normalized_neighbors(x,edge)); row['feature_preprocess_ms']=prep_ms
            del neighbors
            full=lambda:model(x,normalized_neighbors(x,edge),mapping)
            with torch.inference_mode():
                pred,first=timed(full); row['first_full_inference_ms']=first
                # This experiment measures systems costs only. The historic
                # checkpoint is independent of the latest manuscript accuracy.
                del pred; row['end_to_end']=measure(full,a.repeats)
                if not a.skip_cached:
                    neighbors=normalized_neighbors(x,edge)
                    cached=lambda:model(x,neighbors,mapping)
                    row['cached']=measure(cached,a.repeats,scope='cached')
                else:row['cached_scope']='Not repeated at capacity extension; full inference is measured unchanged'
        else:
            # construct owns its device inputs; avoid retaining a duplicate
            # device graph/features from the Loom setup path.
            del x,edge
            gc.collect();torch.cuda.empty_cache()
            x=raw.x.cuda();edge=raw.edge_index.cuda()
            model,cfg=make_sgformer(din)
            if a.method=='sgformer_sparse':
                torch.cuda.reset_peak_memory_stats()
                adjacency,conversion_ms=timed(lambda:sparse_graph(edge,n))
                row['graph_preprocess_ms']=conversion_ms
                row['graph_preprocess_peak_allocated_MiB']=torch.cuda.max_memory_allocated()/2**20
                row['execution']='Unweighted transposed SparseTensor graph; native GCNConv CSR SpMM; normalization uncached per GCN layer'
                del edge;edge=adjacency
            inputs=SimpleNamespace(graph={'node_feat':x,'edge_index':edge})
            forward=lambda:model(inputs)
            prep=dict(graph_preprocess_seconds=0.,excludes='dataset loading, model initialization, device transfer')
            row.update(config=cfg,preprocessing=prep,weights='fixed random initialization; systems only',compile_ms=0.)
            with torch.inference_mode():
                pred,first=timed(forward);del pred
                row['first_full_inference_ms']=first;row['end_to_end']=measure(forward,a.repeats)
                row['first_with_graph_preprocess_ms']=first+row.get('graph_preprocess_ms',0.)
        row['parameters']=sum(v.numel() for v in model.parameters());row['status']='ok'
    except Exception as e:
        row.update(status='oom' if isinstance(e,torch.cuda.OutOfMemoryError) else 'error',error=str(e),traceback=traceback.format_exc())
        if torch.cuda.is_initialized():
            row['failure_peak_allocated_MiB']=torch.cuda.max_memory_allocated()/2**20
            row['failure_peak_reserved_MiB']=torch.cuda.max_memory_reserved()/2**20
    path.write_text(json.dumps(row,indent=2),encoding='utf8')
    with (OUT/'raw_runs.jsonl').open('a',encoding='utf8') as f: f.write(json.dumps(row)+'\n')
    print(json.dumps({k:v for k,v in row.items() if k not in ['traceback','compiler_audit']}),flush=True)
if __name__=='__main__': main()
