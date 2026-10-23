"""Source reuse, independent payload transfer, and padding workloads."""
from pathlib import Path
import math
import networkx as nx
import numpy as np
import scipy.linalg
import torch
from torch_geometric.data import Data


def connected_regular(nodes, seed):
    attempt=0
    while True:
        graph=nx.random_regular_graph(4,nodes,seed=seed+attempt*100000)
        if nx.is_connected(graph):
            return graph
        attempt+=1


def build_dataset(task,cache):
    if cache.is_file():
        return torch.load(cache,weights_only=False)
    transfer=task.startswith('transfer_')
    readers=int(task.split('_')[1]) if transfer or task.startswith('reuse_') else 16
    nodes=int(task.split('_')[1]) if task.startswith('scale_') else 256
    # Multiple independent payloads on each graph; graph seeds are disjoint.
    graphs_per_split={'train':64,'val':16,'test':16}
    payloads=8
    if nodes==8192:
        graphs_per_split={'train':32,'val':8,'test':8}
        payloads=16
    codes=torch.from_numpy(scipy.linalg.hadamard(32).astype(np.float32))/math.sqrt(32)
    datasets={}
    for split_id,(split,count) in enumerate(graphs_per_split.items()):
        result=[]
        for graph_id in range(count):
            seed=1000000+split_id*10000+graph_id
            if transfer:
                first=connected_regular(128,seed)
                second=nx.relabel_nodes(connected_regular(128,seed+5000),lambda i:i+128)
                graph=nx.compose(first,second)
                rng=np.random.default_rng(seed)
                left=rng.choice(128,4,replace=False)
                right=rng.choice(128,4,replace=False)+128
                graph.add_edges_from(zip(left.tolist(),right.tolist()))
                source_pool=np.setdiff1d(np.arange(128),left)
                reader_pool=np.arange(128,256)
            else:
                graph=connected_regular(nodes,seed)
            edges=torch.tensor(list(graph.edges()),dtype=torch.long).T
            edges=torch.cat((edges,edges.flip(0)),1)
            for realization in range(payloads):
                rng=np.random.default_rng(seed*100+realization)
                x=torch.zeros(nodes,42 if transfer else 10)
                y=torch.zeros(nodes,8)
                if transfer:
                    sources=rng.choice(source_pool,readers,replace=False)
                    requested=rng.choice(reader_pool,readers,replace=False)
                    vectors=torch.from_numpy(rng.standard_normal((readers,8)).astype(np.float32))
                    x[sources,:8]=vectors
                    x[sources,8:40]=codes[:readers]
                    x[requested,8:40]=codes[:readers]
                    y[sources]=vectors;y[requested]=vectors
                else:
                    selected=rng.choice(nodes,readers+1,replace=False)
                    sources=selected[:1];requested=selected[1:]
                    vector=torch.from_numpy(rng.standard_normal(8).astype(np.float32))
                    x[sources,:8]=vector
                    y[sources]=vector;y[requested]=vector
                x[sources,-2]=1.;x[requested,-1]=1.
                mask=torch.zeros(nodes,dtype=torch.bool);mask[requested]=True
                result.append(Data(x=x,y=y,edge_index=edges,score_mask=mask,
                                   graph_seed=seed,payload_seed=seed*100+realization))
        datasets[split]=result
    torch.save(datasets,cache)
    return datasets
