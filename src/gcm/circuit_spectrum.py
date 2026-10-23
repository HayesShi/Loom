"""Exact diagonal of a graph Laplacian in the initial Walsh circuit basis.

For a symmetric directed edge list, lambda[k]=(E-FWHT(hist(xor(i,j)))[k])/N.
Graph compilation uses O(E+N log N) work and O(N) temporary memory.
This is a Rayleigh quotient, not an eigendecomposition of the graph.
"""
import torch
import numpy as np
from collections import Counter


def walsh_transform(x):
    n=len(x);z=x.clone()
    for s in range(n.bit_length()-1):
        v=z.reshape(-1,2,1<<s)
        z=torch.stack([v[:,0]+v[:,1],v[:,0]-v[:,1]],1).reshape(n)
    return z


def graph_rayleigh(edge_index,mapping,padded,normalize=True):
    left,right=mapping[edge_index[0]],mapping[edge_index[1]]
    histogram=torch.bincount(left^right,minlength=padded).double()
    energy=(histogram.sum()-walsh_transform(histogram))/padded
    if normalize:energy=energy/energy.mean().clamp_min(1e-12)
    return energy.clamp_min(0).float()


def refine_xor_mapping(adjacency,mapping,padded,seed=0,passes=2):
    """Greedy wire swaps increase XOR-edge histogram concentration.

    For an unweighted undirected graph without loops and a Walsh basis this
    maximizes the squared norm of diag(W L W^T), equivalently reducing its
    off-diagonal Frobenius energy. It is a local search, not a global optimum.
    """
    mapping=mapping.copy();n=len(mapping)
    inverse=np.full(padded,-1,dtype=np.int64);inverse[mapping]=np.arange(n)
    source,target=adjacency.nonzero();keep=source<target
    histogram=np.bincount(mapping[source[keep]]^mapping[target[keep]],minlength=padded).astype(np.int64)
    rng=np.random.default_rng(seed)
    for _ in range(passes):
        for a in rng.permutation(n):
            level=int(rng.integers(1,padded.bit_length()))
            partner=int(mapping[a])^int(rng.integers(1,1<<level))
            b=int(inverse[partner])
            if b<0:continue
            delta=Counter()
            for node,other in [(a,b),(b,a)]:
                neighbors=adjacency.indices[adjacency.indptr[node]:adjacency.indptr[node+1]]
                for neighbor in neighbors:
                    if neighbor==other:continue
                    delta[int(mapping[node]^mapping[neighbor])]-=1
                    delta[int(mapping[other]^mapping[neighbor])]+=1
            improvement=sum(2*int(histogram[key])*change+change*change for key,change in delta.items())
            if improvement>0:
                for key,change in delta.items():histogram[key]+=change
                wa,wb=mapping[a],mapping[b]
                mapping[a],mapping[b]=wb,wa;inverse[wa],inverse[wb]=b,a
    return mapping
