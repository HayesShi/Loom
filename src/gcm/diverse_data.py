from pathlib import Path
import numpy as np
import torch
from torch_geometric.data import Data
from torch_geometric.utils import to_undirected,coalesce
from gcm.paths import data_path
from .data import load_planetoid


ROOT=data_path('node_benchmarks')


def load_diverse(name,split=0):
    if name in ['cora','citeseer','pubmed']: return load_planetoid(name)[1]
    path=ROOT/name/'raw'/f'{name}.npz'
    with np.load(path,allow_pickle=False) as values:
        x=torch.from_numpy(values['node_features'].copy()).float()
        y=torch.from_numpy(values['node_labels'].copy()).long()
        edges=torch.from_numpy(values['edges'].copy()).long()
        if edges.shape[0]!=2: edges=edges.T
        data=Data(x=x,y=y,edge_index=coalesce(to_undirected(edges,num_nodes=len(x))))
        for key in ['train','val','test']:
            mask=torch.from_numpy(values[key+'_masks'].copy()).bool()
            if mask.ndim==2:
                if mask.shape[0]==len(x): mask=mask.T
                if split>=len(mask): raise ValueError(f'{name} has only {len(mask)} splits')
                mask=mask[split]
            setattr(data,key+'_mask',mask)
    train=data.x[data.train_mask]
    mean=train.mean(0); std=train.std(0).clamp_min(1.)
    data.x=(data.x-mean)/std
    data.source=str(path); data.split_id=split
    assert not (data.train_mask & data.val_mask).any()
    assert not (data.train_mask & data.test_mask).any()
    assert not (data.val_mask & data.test_mask).any()
    return data
