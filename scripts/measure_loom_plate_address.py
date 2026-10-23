"""Measure the trained Loom reference placement using the paper's Plate bins."""
import argparse
import json
from pathlib import Path
import sys

REPO=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(REPO/'src'))
sys.path.insert(0,str(REPO/'scripts'))
import numpy as np
import scipy.sparse as sp
from scipy.sparse.csgraph import dijkstra
import torch
from torch_geometric.data import Data
from gcm.loom import Loom,normalized_neighbors
from gcm.plate_geometry import compile_plate_gray_contact
from loom_graph_campaign import load_graphs
from run_loom_campaign import write_json


def source_distances(graph):
    edges=graph.edge_index.numpy()
    adjacency=sp.csr_matrix((np.ones(edges.shape[1]),edges),shape=(len(graph.x),len(graph.x)))
    sources=np.flatnonzero(graph.node_type.numpy()==1)
    distance=dijkstra(adjacency,directed=False,indices=sources,unweighted=True,min_only=True)
    if not np.isfinite(distance[graph.loss_mask.numpy()]).all():
        raise ValueError('Scored Plate nodes must be reachable from a source')
    return distance


def path_agreement(reference,current):
    cross=reference[:,None]^reference[None,:]
    changed=current[:,None]^current[None,:]
    same=(cross==changed)&(((reference^current)[:,None]&cross)==0)
    np.fill_diagonal(same,False)
    return float(same.sum()/(len(reference)*(len(reference)-1)))


@torch.no_grad()
def predict_state(model,graph,mapping,mean,std,force_scale=None):
    x=graph.x.cuda()
    if force_scale is not None:
        x=x.clone();x[:,3:6].div_(force_scale)
    neighbors=normalized_neighbors(x,graph.edge_index.cuda())
    delta=model(x,neighbors,mapping.cuda())*std+mean
    delta=delta*(graph.node_type.cuda()==0).unsqueeze(-1)
    return (graph.u0.cuda()+torch.cat((delta,delta),-1)).cpu()


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--output',type=Path,default=REPO/'results/loom')
    args=parser.parse_args()
    torch.set_num_threads(4);torch.set_float32_matmul_precision('high')
    root=args.output/'plate'
    result=json.loads((root/'result.json').read_text(encoding='utf-8'))
    records=[json.loads(path.read_text(encoding='utf-8')) for path in root.glob('config_*.json')]
    winner=min(records,key=lambda record:record['validation'])
    identity=[winner['config_id'],winner['best_epoch']]
    if result.get('address_consistency_checkpoint')==identity and (root/'address_consistency.json').is_file():
        return
    datasets,_=load_graphs('plate')
    training_distances=np.concatenate([source_distances(graph)[graph.loss_mask.numpy()]
                                      for graph in datasets['train']])
    lower,upper=np.quantile(training_distances,[.25,.75])
    predictions=torch.load(root/'test_predictions.pt',weights_only=False)
    offsets={padded:0 for padded in predictions}
    bitcount=np.array([wire.bit_count() for wire in range(1024)],dtype=np.uint8)
    near_sse=far_sse=0.;near_count=far_count=0
    hamming=[];agreements=[];scores={};reference=None
    test=datasets['test']
    chosen=set(np.random.default_rng(0).choice(len(test),min(8,len(test)),replace=False).tolist())
    examples=[]
    for index in range(len(test)):
        graph=test[index]
        mapping=compile_plate_gray_contact(graph)
        padded=1<<(len(graph.x)-1).bit_length()
        delta=predictions[padded]['prediction'][offsets[padded]][mapping]
        offsets[padded]+=1
        error=(torch.cat((delta,delta),-1)-(graph.y-graph.u0).float()).double().square().sum(-1).numpy()
        score=graph.loss_mask.numpy();distances=source_distances(graph)
        near=score&(distances<=lower);far=score&(distances>=upper)
        near_sse+=float(error[near].sum());far_sse+=float(error[far].sum())
        near_count+=int(near.sum());far_count+=int(far.sum())
        scores.setdefault(padded,[]).append((float(error[score].sum()),int(score.sum())))
        edges=graph.edge_index.numpy().T
        edges=np.unique(np.sort(edges,axis=1),axis=0)
        edges=edges[edges[:,0]!=edges[:,1]]
        addresses=mapping.numpy()
        hamming.append(float(bitcount[addresses[edges[:,0]]^addresses[edges[:,1]]].mean()))
        persistent=np.flatnonzero(graph.node_type.numpy()!=1)
        positions=graph.pos[persistent].numpy()
        order=np.lexsort((positions[:,2],positions[:,1],positions[:,0]))
        addresses=addresses[persistent[order]];positions=positions[order]
        if reference is None:
            reference=(positions.copy(),addresses.copy())
        np.testing.assert_allclose(positions,reference[0],rtol=0,atol=1e-3)
        agreements.append(path_agreement(reference[1],addresses))
        if index in chosen:
            examples.append((index,graph,mapping))
    batch_scores=[]
    for values in scores.values():
        for first in range(0,len(values),winner['config']['batch']):
            batch=values[first:first+winner['config']['batch']]
            batch_scores.append(sum(value[0] for value in batch)/sum(value[1] for value in batch))
    np.testing.assert_allclose(np.mean(batch_scores),result['metrics']['state_MSE'],rtol=1e-6,atol=1e-6)
    groups,info=torch.load(root/'packed.pt',weights_only=False)
    del groups
    cfg=winner['config']
    model=Loom(info['input_dim'],info['output_dim'],info['max_nodes'],
               **{key:cfg[key] for key in ['dim','heads','blocks','dropout','angle']}).cuda().eval()
    checkpoint=torch.load(root/f"config_{winner['config_id']}_best.pt",weights_only=False)
    model.load_state_dict(checkpoint['model'])
    mean=info['mean'].cuda();std=info['std'].cuda()
    force_scale=info['force_rms'] if cfg.get('force_feature_scale') else None
    generator=torch.Generator().manual_seed(0)
    drifts=[];mapping_agreements=[]
    for index,graph,mapping in examples:
        original=predict_state(model,graph,mapping,mean,std,force_scale)
        perm=torch.randperm(len(graph.x),generator=generator)
        inverse=torch.empty_like(perm);inverse[perm]=torch.arange(len(perm))
        relabeled=Data(x=graph.x[perm],pos=graph.pos[perm],properties=graph.properties[perm],
                       node_type=graph.node_type[perm],u0=graph.u0[perm],
                       edge_index=inverse[graph.edge_index])
        remapping=compile_plate_gray_contact(relabeled)
        transformed=predict_state(model,relabeled,remapping,mean,std,force_scale)[inverse]
        drifts.append(float((transformed-original).double().norm()/(original.double().norm()+1e-12)))
        mapping_agreements.append(float((remapping[inverse]==mapping).float().mean()))
    metrics=dict(Hamming_distance=float(np.mean(hamming)),Path_agreement=float(np.mean(agreements)),
                 Relabel_drift=float(np.mean(drifts)),Near_MSE=near_sse/near_count,Far_MSE=far_sse/far_count)
    payload=dict(seed=0,placement='Stable reference',checkpoint=identity,metrics=metrics,
                 train_distance_quartiles=[float(lower),float(upper)],
                 near_nodes=near_count,far_nodes=far_count,test_graphs=len(test),
                 relabel_graph_indices=sorted(chosen),relabel_mapping_agreement=mapping_agreements)
    write_json(root/'address_consistency.json',payload)
    result['metrics'].update(metrics);result['address_consistency_checkpoint']=identity
    write_json(root/'result.json',result)
    print('Plate address consistency:',metrics,flush=True)


if __name__=='__main__':
    main()
