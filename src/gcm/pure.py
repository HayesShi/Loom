"""Pure GCM: one graph embedding, followed exclusively by pair circuits."""
import math
import numbers
import torch
from torch import nn
from torch.nn import functional as F
from torch_geometric.nn.conv.gcn_conv import gcn_norm
from .layers import rotate90
from .architecture import compile_interfaces
from .graph_butterfly import GraphButterflyAggregate
from .feature_compilation import compile_feature_wire_order


def _compile_edge_pair_mask(edge_index, mapping, pairs, max_stages=None):
    """Mark butterfly pairs that correspond to an original graph edge.

    Work in node IDs (not padded wire IDs), and treat either edge direction as
    adjacency. Sorted int64 keys avoid constructing an N-by-N adjacency matrix.
    """
    nodes=mapping.numel()
    stages=len(pairs) if max_stages is None else min(len(pairs),max_stages)
    mask=torch.zeros((stages,pairs.shape[-1]),dtype=torch.bool)
    if not nodes or not edge_index.numel(): return mask
    wire_to_node=torch.full((2*pairs.shape[-1],),-1,dtype=torch.long)
    wire_to_node[mapping]=torch.arange(nodes)
    source,target=edge_index.long()
    keep=source!=target
    source,target=source[keep],target[keep]
    if not source.numel(): return mask
    edge_keys=torch.sort(torch.minimum(source,target)*nodes+torch.maximum(source,target)).values
    for stage in range(stages):
        endpoints=wire_to_node[pairs[stage]]
        real=(endpoints>=0).all(dim=0)
        if not real.any(): continue
        left,right=endpoints[:,real]
        keys=torch.minimum(left,right)*nodes+torch.maximum(left,right)
        positions=torch.searchsorted(edge_keys,keys)
        found=(positions<edge_keys.numel()) & (edge_keys[positions.clamp(max=edge_keys.numel()-1)]==keys)
        mask[stage,real]=found
    return mask


def _stage_window_permutation(padded, stages_per_block, block_index):
    """Move one block's assigned butterfly bits into the circuit's low bits.

    The two bit windows are swapped, so this permutation is its own inverse.
    It is a fixed reindexing of wires, not a learned map or a state transform.
    """
    shift=block_index*stages_per_block
    if not shift: return None
    mask=(1<<stages_per_block)-1
    ids=torch.arange(padded)
    low=ids&mask
    assigned=(ids>>shift)&mask
    return (ids&~(mask|(mask<<shift)))|(low<<shift)|assigned


class PhasePair(nn.Module):
    """Two-port coupler; the default orthogonal mode recovers original GCM.

    [u';v']=[cos(theta)I, sin(theta)R(phi); -sin(theta)R(-phi),cos(theta)I][u;v].
    Content dependence preserves forward energy in the rotational modes,
    not general invertibility.
    mode='plain' is a deliberately non-orthogonal, parameter-matched ablation.
    mode='shear' adds tanh(angle)-bounded partner information without
    attenuating the self signal.
    mode='lifting' decouples the middle coefficient of the three-shear
    factorization of the original J rotation. At zero offset it starts at
    exactly the original rotation for the initialized angle range.
    mode='directed' keeps the J carrier but lets each ordered port add the
    other without attenuating its own state. The same gate is reused with
    reversed inputs, so directional transport adds no parameters.
    mode='centered_j' preserves the pair mean and the original J carrier's
    rotation of its difference.
    """
    def __init__(self, dim, heads, mode='phase', angle=.2,gate_normalize=False):
        super().__init__()
        if dim % (2*heads): raise ValueError('Even head width required')
        self.heads,self.width=heads,dim//heads
        self.mode=mode
        self.gate_scale=math.sqrt(dim) if gate_normalize else 1.
        self.base=nn.Parameter(torch.full((heads,),angle))
        if mode=='phase': self.phase=nn.Parameter(torch.zeros(heads))
        else: self.register_buffer('phase',torch.full((heads,),math.pi/2))
        self.gate=nn.Linear(2*dim,heads)
        nn.init.zeros_(self.gate.weight); nn.init.zeros_(self.gate.bias)
        if mode=='lifting':
            # The old gate still spans content-controlled rotations. This
            # extra signed term can distinguish the two ordered pair ports.
            # Keep its checkpoint key without shifting all later random
            # initializations relative to the original mode.
            with torch.random.fork_rng(devices=[]):
                self.direction_gate=nn.Linear(dim,heads,bias=False)
            nn.init.zeros_(self.direction_gate.weight)
            self.middle_delta=nn.Parameter(torch.zeros(heads))
        if mode=='correlation':
            # Q/K starts as cosine relation matching, then specializes.
            self.query=nn.Linear(dim,dim,bias=False)
            self.key=nn.Linear(dim,dim,bias=False)
            nn.init.eye_(self.query.weight);nn.init.eye_(self.key.weight)
            self.correlation_gain=nn.Parameter(torch.full((heads,),.01))

    def forward(self,u,v,angle_scale=1.,prior_only_scale=False,
                gate_u=None,gate_v=None):
        if self.mode=='identity': return u,v
        shape=u.shape
        flat_u,flat_v=u,v
        if (gate_u is None)!=(gate_v is None):
            raise ValueError('gate_u and gate_v must be supplied together')
        control_u,control_v=(flat_u,flat_v) if gate_u is None else (gate_u,gate_v)
        u=u.reshape(*shape[:-1],self.heads,self.width);v=v.reshape_as(u)
        if self.mode=='directed':
            forward=self.gate(torch.cat([control_u,control_v],-1))/self.gate_scale
            reverse=self.gate(torch.cat([control_v,control_u],-1))/self.gate_scale
            into_u=((self.base+forward)*angle_scale).tanh().unsqueeze(-1)
            into_v=((self.base+reverse)*angle_scale).tanh().unsqueeze(-1)
            return ((u+into_u*rotate90(v)).reshape(shape),
                    (v+into_v*rotate90(u)).reshape(shape))
        dynamic_phase=None
        if self.mode=='correlation':
            qu=F.normalize(self.query(control_u).reshape_as(u),dim=-1)
            qv=F.normalize(self.query(control_v).reshape_as(v),dim=-1)
            ku=F.normalize(self.key(control_u).reshape_as(u),dim=-1)
            kv=F.normalize(self.key(control_v).reshape_as(v),dim=-1)
            uv=(qu*kv).sum(-1);vu=(qv*ku).sum(-1)
            gain=self.correlation_gain
            score=gain*.5*(uv+vu)
            # Antisymmetric QK affinity becomes a directional transport phase.
            dynamic_phase=math.pi/2+(math.pi/2)*(gain*.5*(uv-vu)).tanh()
        else:
            score=self.gate(torch.cat([control_u+control_v,(control_u-control_v).abs()],-1))/self.gate_scale
            if self.mode=='lifting':
                score=score+self.direction_gate(control_u-control_v)/self.gate_scale
        # A compiled geometric prior may initialize the base rotation without
        # also capping the content gate at distant pairs.
        angle=(self.base*angle_scale+.5*score.tanh() if prior_only_scale else
               (self.base+.5*score.tanh())*angle_scale)
        if self.mode=='centered_j':
            mean=.5*(u+v)
            difference=.5*(u-v)
            rotated=angle.cos().unsqueeze(-1)*difference-angle.sin().unsqueeze(-1)*rotate90(difference)
            return (mean+rotated).reshape(shape),(mean-rotated).reshape(shape)
        if self.mode=='lifting':
            # tan(theta/2) has a pole at theta=pi. The useful initialized
            # range is far inside this cap; it bounds the three shears if
            # training drives an angle toward the pole.
            half_angle=(.5*angle).clamp(-1.25,1.25)
            t=half_angle.tan().unsqueeze(-1)
            s=(angle.sin()+self.middle_delta.tanh()).unsqueeze(-1)
            u1=u+t*rotate90(v)
            v1=v+s*rotate90(u1)
            u1=u1+t*rotate90(v1)
            return u1.reshape(shape),v1.reshape(shape)
        if self.mode=='shear':
            strength=angle.tanh().unsqueeze(-1)
            return (u+strength*rotate90(v)).reshape(shape),(v+strength*rotate90(u)).reshape(shape)
        co,si=angle.cos().unsqueeze(-1),angle.sin().unsqueeze(-1)
        if self.mode=='plain':
            return (co*u+si*v).reshape(shape),(co*v+si*u).reshape(shape)
        phase=self.phase if dynamic_phase is None else dynamic_phase
        cp,sp=phase.cos().unsqueeze(-1),phase.sin().unsqueeze(-1)
        rv=cp*v+sp*rotate90(v)
        ru=cp*u-sp*rotate90(u)
        return (co*u+si*rv).reshape(shape),(co*v-si*ru).reshape(shape)


class PureCircuit(nn.Module):
    def __init__(self,dim,heads,pairs,mode='phase',stage_ff=True,angle=.2,scale_readout=False,gate_normalize=False,max_stages=None,edge_pair_mask=None,skip_padding_pairs=False,real_wire_mask=None,stage_angle_decay=False,multiscale_reach=False,tie_pair_gate=False,gate_anchor=0.):
        super().__init__()
        if (isinstance(gate_anchor,bool) or not isinstance(gate_anchor,numbers.Real)
                or not math.isfinite(gate_anchor) or not 0.<=gate_anchor<=1.):
            raise ValueError('gate_anchor must be finite and in [0, 1]')
        self.gate_anchor=float(gate_anchor)
        if max_stages is not None and (isinstance(max_stages,bool) or not isinstance(max_stages,int) or max_stages<1):
            raise ValueError('max_stages must be a positive integer or None')
        if not isinstance(skip_padding_pairs,bool):
            raise ValueError('skip_padding_pairs must be boolean')
        if not isinstance(stage_angle_decay,bool):
            raise ValueError('stage_angle_decay must be boolean')
        if not isinstance(multiscale_reach,bool):
            raise ValueError('multiscale_reach must be boolean')
        if not isinstance(tie_pair_gate,bool):
            raise ValueError('tie_pair_gate must be boolean')
        if tie_pair_gate and mode!='centered_j':
            raise ValueError("tie_pair_gate requires mode='centered_j'")
        if multiscale_reach:
            if mode!='centered_j':
                raise ValueError("multiscale_reach requires mode='centered_j'")
            if stage_angle_decay:
                raise ValueError('multiscale_reach requires stage_angle_decay=False')
            active_stages=min(len(pairs),max_stages) if max_stages is not None else len(pairs)
            # Head zero retains the supplied angle. Other heads target
            # expected Hamming shells 1, 2, 4, ... in the active circuit.
            # K>L would imply a transfer probability greater than one.
            largest_shell=1<<(heads-2) if heads>1 else 0
            if active_stages<largest_shell:
                raise ValueError(f'multiscale_reach requires at least {largest_shell} active stages for {heads} heads')
        if real_wire_mask is not None and (real_wire_mask.dtype!=torch.bool
                                           or tuple(real_wire_mask.shape)!=(2*pairs.shape[-1],)):
            raise ValueError('real_wire_mask must be boolean with one entry per compiled wire')
        self.skip_padding_pairs=skip_padding_pairs
        self.register_buffer('real_wire_mask',real_wire_mask,persistent=False)
        self.max_stages=max_stages
        self.stage_angle_decay=stage_angle_decay
        self.multiscale_reach=multiscale_reach
        self.tie_pair_gate=tie_pair_gate
        self.register_buffer('pairs',pairs)
        if edge_pair_mask is not None:
            if mode=='gauge': raise ValueError('edge-only pairs do not support gauge mode')
            expected=(min(len(pairs),max_stages) if max_stages is not None else len(pairs),pairs.shape[-1])
            if edge_pair_mask.dtype!=torch.bool or tuple(edge_pair_mask.shape)!=expected:
                raise ValueError(f'edge_pair_mask must be a boolean tensor of shape {expected}')
            # Non-persistent buffers keep old checkpoint keys unchanged while
            # moving the precompiled mask and active indices with model.to().
            self.register_buffer('edge_pair_mask',edge_pair_mask,persistent=False)
            self.edge_pair_names=[]
            for stage in range(len(edge_pair_mask)):
                name=f'edge_pairs_{stage}'
                self.register_buffer(name,pairs[stage,:,edge_pair_mask[stage]].contiguous(),persistent=False)
                self.edge_pair_names.append(name)
        else:
            self.edge_pair_mask=None
            self.edge_pair_names=None
        ids=torch.arange(2*pairs.shape[-1],device=pairs.device)
        self.regular=all(torch.equal(pair[0],ids[(ids&(1<<s))==0]) and
                         torch.equal(pair[1],ids[(ids&(1<<s))==0]^(1<<s)) for s,pair in enumerate(pairs))
        self.gauge=mode=='gauge'
        self.switches=nn.ModuleList(PhasePair(dim,heads,'original' if self.gauge else mode,angle,gate_normalize) for _ in pairs)
        if tie_pair_gate:
            # Share only the content rule. Every stage keeps its own base angle.
            # Construct all switches first so other parameters retain their
            # original initialization and the zero-exchange path is unchanged.
            shared_gate=self.switches[0].gate
            for switch in self.switches[1:]:
                switch.gate=shared_gate
        if multiscale_reach:
            # For centered J, a pair transfers the fraction sin²(theta/2)
            # of each port's energy. With L independent butterfly bits,
            # theta=2 asin(sqrt(K/L)) targets mean shell distance K.
            head_angles=[angle]+[2.*math.asin(math.sqrt((1<<(head-1))/active_stages))
                                 for head in range(1,heads)]
            with torch.no_grad():
                for switch in self.switches:
                    switch.base.copy_(switch.base.new_tensor(head_angles))
        self.stage_ff=stage_ff
        self.scale_readout=scale_readout
        self.heads=heads
        if scale_readout:
            self.scale_score=nn.Linear(2*dim,heads)
            nn.init.zeros_(self.scale_score.weight);nn.init.zeros_(self.scale_score.bias)
            self.scale_prior=nn.Parameter(torch.linspace(0.,-2.,len(pairs)+1)[:,None].repeat(1,heads))
        if stage_ff:
            self.gains=nn.Parameter(torch.full((len(pairs),dim),.1))
            self.biases=nn.Parameter(torch.zeros(len(pairs),dim))
        # Channel permutation costs O(Nd) and exchanges feature heads.
        self.register_buffer('shuffle',torch.arange(dim).reshape(heads,-1).T.flatten())
        self.register_buffer('parity',torch.tensor([i.bit_count()%2 for i in range(2*pairs.shape[-1])]).bool())

    def forward(self,z):
        context=z
        def encode(value):return torch.where(self.parity[:,None],rotate90(value),value)
        def decode(value):return torch.where(self.parity[:,None],-rotate90(value),value)
        def contribution(value,level):
            score=self.scale_score(torch.cat([context,value],-1))
            weight=(3*score.tanh()+self.scale_prior[level].clamp(-6,6)).exp().unsqueeze(-1)
            shaped=value.reshape(*value.shape[:-1],self.heads,-1)
            return weight*shaped,weight
        if self.scale_readout:
            total,normalizer=contribution(z,0)
        if self.gauge:
            if not self.regular:raise ValueError('Gauge mode requires the regular butterfly schedule')
            z=encode(z)
        entry=z if self.gate_anchor else None
        pairs=self.pairs
        n=z.shape[-2]
        stages=n.bit_length()-1
        if n<2 or n&(n-1) or stages>len(self.switches):
            raise ValueError('Circuit requires a power-of-two size within its stage budget')
        active_stages=stages if self.max_stages is None else min(stages,self.max_stages)
        if self.edge_pair_mask is not None and n!=2*self.pairs.shape[-1]:
            raise ValueError('Edge-only pairs require the compiled graph size')
        if self.regular:
            pairs=[None]*active_stages
        elif z.shape[-2] != 2*pairs.shape[-1]:
            n=z.shape[-2]
            if n<2 or n&(n-1) or n>2*pairs.shape[-1]:
                raise ValueError('Circuit requires a power-of-two size within its stage budget')
            ids=torch.arange(n,device=z.device)
            pairs=[torch.stack([ids[(ids&(1<<s))==0],ids[(ids&(1<<s))==0]^(1<<s)])
                   for s in range(active_stages)]
        for s in range(active_stages):
            switch,pair=self.switches[s],pairs[s]
            angle_scale=1./math.sqrt(s+1) if self.stage_angle_decay else 1.
            if self.edge_pair_mask is not None:
                pair=getattr(self,self.edge_pair_names[s])
                if pair.shape[-1]:
                    left,right=pair
                    old_u,old_v=z.index_select(-2,left),z.index_select(-2,right)
                    if entry is None:
                        u,v=switch(old_u,old_v,angle_scale=angle_scale)
                    else:
                        gate_u=(1.-self.gate_anchor)*old_u+self.gate_anchor*entry.index_select(-2,left)
                        gate_v=(1.-self.gate_anchor)*old_v+self.gate_anchor*entry.index_select(-2,right)
                        u,v=switch(old_u,old_v,angle_scale=angle_scale,gate_u=gate_u,gate_v=gate_v)
                    if self.skip_padding_pairs and self.real_wire_mask is not None:
                        valid=self.real_wire_mask[left]&self.real_wire_mask[right]
                        u=torch.where(valid[...,None],u,old_u)
                        v=torch.where(valid[...,None],v,old_v)
                    out=z.clone()
                    out.index_copy_(-2,left,u)
                    out.index_copy_(-2,right,v)
                else:
                    # Keep per-stage FFN/readout semantics, but skip the switch.
                    out=z
            elif self.regular:
                stride=1<<s
                grouped=z.reshape(*z.shape[:-2],n//(2*stride),2,stride,z.shape[-1])
                u,v=grouped[...,0,:,:],grouped[...,1,:,:]
                if entry is not None:
                    anchor=entry.reshape_as(grouped)
                    gate_u=(1.-self.gate_anchor)*u+self.gate_anchor*anchor[...,0,:,:]
                    gate_v=(1.-self.gate_anchor)*v+self.gate_anchor*anchor[...,1,:,:]
                if self.skip_padding_pairs and self.real_wire_mask is not None:
                    old_u,old_v=u,v
                if self.gauge:
                    parity=self.parity.reshape(n//(2*stride),2,stride)
                    pu,pv=parity[:,0],parity[:,1]
                    source_u,source_v=(u,v) if entry is None else (gate_u,gate_v)
                    du=torch.where(pu[...,None],-rotate90(source_u),source_u)
                    dv=torch.where(pv[...,None],-rotate90(source_v),source_v)
                    score=switch.gate(torch.cat([du+dv,(du-dv).abs()],-1))/switch.gate_scale
                    angle=(switch.base+.5*score.tanh())*angle_scale*(2*pu.float()-1)[...,None]
                    shape=u.shape
                    uh=u.reshape(*shape[:-1],switch.heads,switch.width)
                    vh=v.reshape_as(uh)
                    co,si=angle.cos().unsqueeze(-1),angle.sin().unsqueeze(-1)
                    u=(co*uh+si*rotate90(vh)).reshape(shape)
                    v=(co*vh+si*rotate90(uh)).reshape(shape)
                else:
                    if entry is None:
                        u,v=switch(u,v,angle_scale=angle_scale)
                    else:
                        u,v=switch(u,v,angle_scale=angle_scale,gate_u=gate_u,gate_v=gate_v)
                if self.skip_padding_pairs and self.real_wire_mask is not None:
                    wire_groups=self.real_wire_mask[:n].reshape(n//(2*stride),2,stride)
                    valid=wire_groups[:,0]&wire_groups[:,1]
                    u=torch.where(valid[...,None],u,old_u)
                    v=torch.where(valid[...,None],v,old_v)
                out=torch.stack([u,v],dim=-3).reshape_as(z)
            else:
                left,right=pair
                old_u,old_v=z.index_select(-2,left),z.index_select(-2,right)
                if entry is None:
                    u,v=switch(old_u,old_v,angle_scale=angle_scale)
                else:
                    gate_u=(1.-self.gate_anchor)*old_u+self.gate_anchor*entry.index_select(-2,left)
                    gate_v=(1.-self.gate_anchor)*old_v+self.gate_anchor*entry.index_select(-2,right)
                    u,v=switch(old_u,old_v,angle_scale=angle_scale,gate_u=gate_u,gate_v=gate_v)
                if self.skip_padding_pairs and self.real_wire_mask is not None:
                    valid=self.real_wire_mask[left]&self.real_wire_mask[right]
                    u=torch.where(valid[...,None],u,old_u)
                    v=torch.where(valid[...,None],v,old_v)
                out=torch.empty_like(z).index_copy(-2,left,u).index_copy(-2,right,v)
            if self.stage_ff:
                if self.gauge:out=decode(out)
                out=out+self.gains[s]*F.gelu(out+self.biases[s]);out=out.index_select(-1,self.shuffle)
                if self.gauge:out=encode(out)
            z=out
            if self.scale_readout:
                value,weight=contribution(decode(z) if self.gauge else z,s+1)
                total=total+value;normalizer=normalizer+weight
        return (total/normalizer).reshape_as(z) if self.scale_readout else (decode(z) if self.gauge else z)


class PureBlock(nn.Module):
    def __init__(self,dim,heads,pairs,mode,dropout,stage_ff,angle,scale_readout=False,gate_normalize=False,circuit_scale=1.,learnable_circuit_scale=False,max_stages=None,edge_pair_mask=None,exchange_scale=1.,skip_padding_pairs=False,real_wire_mask=None,wire_permutation=None,refresh_after_circuit=False,checkpoint_circuit=False,stage_angle_decay=False,multiscale_reach=False,tie_pair_gate=False,relay_padding=False,gate_anchor=0.):
        super().__init__()
        if (isinstance(exchange_scale,bool) or not isinstance(exchange_scale,numbers.Real)
                or not math.isfinite(exchange_scale) or not 0.<=exchange_scale<=1.):
            raise ValueError('exchange_scale must be finite and in [0, 1]')
        self.exchange_scale=float(exchange_scale)
        if not isinstance(refresh_after_circuit,bool):
            raise ValueError('refresh_after_circuit must be boolean')
        self.refresh_after_circuit=refresh_after_circuit
        if not isinstance(relay_padding,bool):
            raise ValueError('relay_padding must be boolean')
        if relay_padding and (skip_padding_pairs or real_wire_mask is None):
            raise ValueError('relay_padding requires a real-wire mask and active padding pairs')
        self.relay_padding=relay_padding
        self.checkpoint_circuit=bool(checkpoint_circuit)
        self.circuit_scale=(nn.Parameter(torch.tensor(float(circuit_scale)))
                            if learnable_circuit_scale else float(circuit_scale))
        self.norm=nn.LayerNorm(dim)
        self.circuit=PureCircuit(dim,heads,pairs,mode,stage_ff,angle,scale_readout,gate_normalize,max_stages,edge_pair_mask,
                                 skip_padding_pairs,real_wire_mask,stage_angle_decay,multiscale_reach,tie_pair_gate,gate_anchor)
        if wire_permutation is not None:
            padded=2*pairs.shape[-1]
            if (wire_permutation.dtype!=torch.long or tuple(wire_permutation.shape)!=(padded,)
                    or not torch.equal(wire_permutation.sort().values,torch.arange(padded,device=wire_permutation.device))):
                raise ValueError('wire_permutation must permute all compiled wires')
        self.register_buffer('wire_permutation',wire_permutation,persistent=False)
        self.register_buffer('wire_inverse',wire_permutation.argsort() if wire_permutation is not None else None,
                             persistent=False)
        self.output=nn.Linear(dim,dim,bias=False)
        self.ffn=nn.Sequential(nn.LayerNorm(dim),nn.Linear(dim,2*dim),nn.GELU(),
                               nn.Dropout(dropout),nn.Linear(2*dim,dim))
        self.dropout=dropout

    def forward(self,x,local=None,normalized=None,circuit_input_override=None):
        # Empty wires start every block at zero, then remain free to relay
        # signals between the block's regular J stages.  The zero-exchange
        # fast path runs on real nodes only and needs no padding mask.
        relay_mask=(self.circuit.real_wire_mask if self.relay_padding
                    and x.shape[-2]==self.circuit.real_wire_mask.numel() else None)
        if relay_mask is not None:
            empty=~relay_mask[:,None]
            x=x.masked_fill(empty,0.)
        normalized=self.norm(x) if normalized is None else normalized
        if relay_mask is not None:
            # LayerNorm's learned bias would otherwise seed empty wires.
            normalized=normalized.masked_fill(empty,0.)
            if local is not None: local=local.masked_fill(empty,0.)
            if circuit_input_override is not None:
                circuit_input_override=circuit_input_override.masked_fill(empty,0.)
        value=normalized
        if local is not None: value=value+local
        if self.exchange_scale==0.:
            exchanged=value
        else:
            def run_circuit(signal):
                if self.checkpoint_circuit and self.training and torch.is_grad_enabled():
                    from torch.utils.checkpoint import checkpoint
                    return checkpoint(self.circuit,signal,use_reentrant=False)
                return self.circuit(signal)
            circuit_input=(circuit_input_override if circuit_input_override is not None else
                           normalized if self.refresh_after_circuit else value)
            if self.wire_permutation is None:
                circuit_output=run_circuit(circuit_input)
            else:
                # Local refresh and external readout keep the original wire IDs.
                circuit_output=run_circuit(circuit_input.index_select(-2,self.wire_permutation))
                circuit_output=circuit_output.index_select(-2,self.wire_inverse)
            if self.refresh_after_circuit:
                # The graph message stays at its destination wire; the circuit
                # transports only the persistent node state.
                exchanged=value+self.exchange_scale*(circuit_output-circuit_input)
            else:
                exchanged=(circuit_output if self.exchange_scale==1. else
                           value+self.exchange_scale*(circuit_output-value))
        x=x+self.circuit_scale*F.dropout(self.output(exchanged),self.dropout,self.training)
        return x+F.dropout(self.ffn(x),self.dropout,self.training)


class PureGCM(nn.Module):
    def __init__(self,data,dim=64,heads=4,blocks=2,mode='phase',dropout=.2,
                 layout='graph',stage_ff=True,angle=.2,graph_embed=True,seed=0,graph_refresh=False,scale_readout=False,gate_normalize=False,circuit_scale=1.,learnable_circuit_scale=False,max_stages=None,edge_only_pairs=False,exchange_scale=1.,group_read_size=None,group_read_scale=0.,refresh_mode='linear',graph_butterfly=False,graph_butterfly_row_preserve=False,skip_padding_pairs=False,progressive_pairing=False,feature_pairing=False,refresh_after_circuit=False,pair_alignment=False,graph_innovation=False,staggered_pairing=False,feature_first_stage=False,random_first_stage=False,checkpoint_circuit=False,stage_angle_decay=False,multiscale_reach=False,tie_pair_gate=False,relay_padding=False,gate_anchor=0.):
        super().__init__()
        if refresh_mode not in ('linear', 'scalar', 'diagonal'):
            raise ValueError("refresh_mode must be 'linear', 'scalar', or 'diagonal'")
        if refresh_mode!='linear' and not graph_refresh:
            raise ValueError(f"refresh_mode='{refresh_mode}' requires graph_refresh=True")
        if graph_butterfly and (not graph_refresh or refresh_mode!='linear'):
            raise ValueError("graph_butterfly requires graph_refresh=True and refresh_mode='linear'")
        if graph_butterfly_row_preserve and not graph_butterfly:
            raise ValueError('graph_butterfly_row_preserve requires graph_butterfly=True')
        if graph_innovation and (not graph_refresh or not refresh_after_circuit):
            raise ValueError('graph_innovation requires graph_refresh and refresh_after_circuit')
        if not isinstance(pair_alignment,bool):
            raise ValueError('pair_alignment must be boolean')
        if not isinstance(relay_padding,bool):
            raise ValueError('relay_padding must be boolean')
        if relay_padding and skip_padding_pairs:
            raise ValueError('relay_padding and skip_padding_pairs are mutually exclusive')
        self.relay_padding=relay_padding
        mapping,_,pairs=compile_interfaces(data.edge_index.cpu(),data.num_nodes,layout,
                                           pair_alignment,seed,node_features=data.x)
        padded=pairs.shape[-1]*2
        if not isinstance(progressive_pairing,bool):
            raise ValueError('progressive_pairing must be boolean')
        if progressive_pairing:
            if (max_stages is None or isinstance(max_stages,bool) or not isinstance(max_stages,int)
                    or max_stages<1):
                raise ValueError('progressive_pairing requires a positive integer max_stages')
            if max_stages*blocks>padded.bit_length()-1:
                raise ValueError('progressive_pairing requires max_stages * blocks <= log2(padded wires)')
            if edge_only_pairs or skip_padding_pairs:
                raise ValueError('progressive_pairing does not support edge_only_pairs or skip_padding_pairs')
        if not isinstance(feature_pairing,bool):
            raise ValueError('feature_pairing must be boolean')
        if feature_pairing:
            if blocks!=2 or max_stages is None or isinstance(max_stages,bool) or not isinstance(max_stages,int) or max_stages<1:
                raise ValueError('feature_pairing requires two blocks and a positive integer max_stages')
            if progressive_pairing or edge_only_pairs or skip_padding_pairs:
                raise ValueError('feature_pairing does not support progressive_pairing, edge_only_pairs or skip_padding_pairs')
        if staggered_pairing and (blocks != 2 or progressive_pairing or feature_pairing or edge_only_pairs):
            raise ValueError('staggered_pairing requires two blocks and regular butterfly pairs')
        if not isinstance(feature_first_stage,bool) or not isinstance(random_first_stage,bool):
            raise ValueError('first-stage pairing options must be boolean')
        if feature_first_stage and random_first_stage:
            raise ValueError('feature_first_stage and random_first_stage are mutually exclusive')
        if feature_first_stage or random_first_stage:
            if (blocks!=2 or max_stages is None or isinstance(max_stages,bool)
                    or not isinstance(max_stages,int) or max_stages<1):
                raise ValueError('first-stage pairing requires two blocks and a positive integer max_stages')
            if (progressive_pairing or feature_pairing or edge_only_pairs or skip_padding_pairs):
                raise ValueError('first-stage pairing does not support other pairing variants or skip_padding_pairs')
        feature_order=(compile_feature_wire_order(data.x,mapping,padded,1<<max_stages,seed)[0]
                       if feature_pairing else None)
        staggered_order=torch.arange(padded).roll(1) if staggered_pairing else None
        first_stage_pairs=None
        if feature_first_stage or random_first_stage:
            if feature_first_stage:
                order=compile_feature_wire_order(data.x,mapping,padded,1<<max_stages,seed)[0]
            else:
                # Match the feature compiler's real/dummy slot occupancy so
                # the random control has the same number of active node pairs.
                generator=torch.Generator().manual_seed(seed)
                real_mask=torch.zeros(padded,dtype=torch.bool)
                real_mask[mapping]=True
                dummy=torch.arange(padded)[~real_mask]
                order=torch.cat((mapping[torch.randperm(len(mapping),generator=generator)],
                                 dummy[torch.randperm(len(dummy),generator=generator)]))
            first_stage_pairs=pairs.clone()
            old_wire_pairs=order[pairs[0]]
            first_stage_pairs[0]=(staggered_order.argsort()[old_wire_pairs]
                                  if staggered_order is not None else old_wire_pairs)
        if group_read_size is not None and (isinstance(group_read_size,bool) or not isinstance(group_read_size,int)
                                            or group_read_size<2 or group_read_size&(group_read_size-1)
                                            or padded%group_read_size):
            raise ValueError('group_read_size must be a power of two >= 2 that divides padded size')
        if (isinstance(group_read_scale,bool) or not isinstance(group_read_scale,numbers.Real)
                or not math.isfinite(group_read_scale) or not 0.<=group_read_scale<=1.):
            raise ValueError('group_read_scale must be finite and in [0, 1]')
        self.group_read_size=group_read_size
        self.group_read_scale=float(group_read_scale)
        if group_read_size is not None and self.group_read_scale>0.:
            real_wire_mask=torch.zeros(padded,dtype=torch.bool)
            real_wire_mask[mapping]=True
            self.register_buffer('real_wire_mask',real_wire_mask,persistent=False)
            self.register_buffer('group_read_counts',real_wire_mask.reshape(-1,group_read_size).sum(1),persistent=False)
            self.register_buffer('group_read_ids',mapping//group_read_size,persistent=False)
        else:
            self.register_buffer('real_wire_mask',None,persistent=False)
            self.register_buffer('group_read_counts',None,persistent=False)
            self.register_buffer('group_read_ids',None,persistent=False)
        edge_pair_mask=(_compile_edge_pair_mask(data.edge_index.cpu(),mapping,pairs,max_stages)
                        if edge_only_pairs else None)
        circuit_real_wire_mask=None
        if skip_padding_pairs or relay_padding:
            circuit_real_wire_mask=torch.zeros(padded,dtype=torch.bool)
            circuit_real_wire_mask[mapping]=True
        edge,weight=gcn_norm(data.edge_index.cpu(),num_nodes=data.num_nodes,add_self_loops=False)
        adjacency=torch.sparse_coo_tensor(edge,weight,(data.num_nodes,data.num_nodes)).coalesce()
        self.register_buffer('adjacency',adjacency)
        self.register_buffer('mapping',mapping)
        # Preserve the legacy adjacency orientation: the existing sparse
        # tensor uses gcn_norm's edge[0] as its row. The new operator accepts
        # source->target edges, so reverse them here for exact initialization.
        self.graph_butterfly=(GraphButterflyAggregate(edge.flip(0).contiguous(),mapping,weight,
                                                     num_stages=len(pairs),blocks=blocks,
                                                     row_preserve=graph_butterfly_row_preserve)
                              if graph_butterfly else None)
        self.padded=padded
        self.graph_innovation=graph_innovation
        self.graph_embed=graph_embed
        self.encoder=nn.Linear(data.num_features,dim)
        self.neighbor_encoder=nn.Linear(data.num_features,dim,bias=False) if graph_embed else None
        self.input_norm=nn.LayerNorm(dim)
        self.blocks=nn.ModuleList(PureBlock(dim,heads,first_stage_pairs if first_stage_pairs is not None and block_index==1 else pairs,mode,dropout,stage_ff,angle,scale_readout,gate_normalize,circuit_scale,learnable_circuit_scale,max_stages,edge_pair_mask,exchange_scale,
                                            skip_padding_pairs,circuit_real_wire_mask,
                                            (_stage_window_permutation(padded,max_stages,block_index) if progressive_pairing
                                             else feature_order if feature_pairing and block_index==1
                                             else staggered_order if staggered_pairing and block_index==1
                                             else None),
                                            refresh_after_circuit,checkpoint_circuit,stage_angle_decay,multiscale_reach,tie_pair_gate,relay_padding,gate_anchor)
                                  for block_index in range(blocks))
        self.refresh_mode=refresh_mode
        if not graph_refresh:
            self.refresh=None
        elif refresh_mode=='scalar':
            # 2*sigmoid(0)=1 initially; one bounded gain replaces each d-by-d map.
            self.refresh=nn.ParameterList(nn.Parameter(torch.zeros(())) for _ in range(blocks))
        elif refresh_mode=='diagonal':
            # Signed per-channel gains retain one sparse aggregation per block.
            self.refresh=nn.ParameterList(nn.Parameter(torch.ones(dim)) for _ in range(blocks))
        else:
            self.refresh=nn.ModuleList(nn.Linear(dim,dim,bias=False) for _ in range(blocks))
        self.classifier=nn.Sequential(nn.LayerNorm(dim),nn.Linear(dim,int(data.y.max())+1))
        self.dropout=dropout

    @torch.no_grad()
    def project_circuit_scales(self):
        for block in self.blocks:
            if isinstance(block.circuit_scale, nn.Parameter):
                block.circuit_scale.clamp_(0., 1.)

    def circuit_scales(self):
        return [float(block.circuit_scale.detach()) if isinstance(block.circuit_scale, nn.Parameter)
                else float(block.circuit_scale) for block in self.blocks]

    def refresh_scales(self):
        if self.refresh_mode!='scalar' or self.refresh is None:
            return []
        return [float((2*torch.sigmoid(raw)).detach()) for raw in self.refresh]

    def refresh_diagonals(self):
        if self.refresh_mode!='diagonal' or self.refresh is None:
            return []
        return [gain.detach().cpu().tolist() for gain in self.refresh]

    def graph_butterfly_gains(self):
        if self.graph_butterfly is None:
            return []
        return self.graph_butterfly.gain.detach().cpu().tolist()

    def _refresh_local(self,block_index,value):
        if self.refresh_mode=='scalar':
            return (2*torch.sigmoid(self.refresh[block_index]))*torch.sparse.mm(self.adjacency,value)
        if self.refresh_mode=='diagonal':
            return torch.sparse.mm(self.adjacency,value)*self.refresh[block_index]
        message=self.refresh[block_index](value)
        if self.graph_butterfly is not None:
            return self.graph_butterfly(message,block_index)
        return torch.sparse.mm(self.adjacency,message)

    def _group_read(self,z):
        real=z.index_select(0,self.mapping)
        if self.group_read_ids is None:
            return real
        group_sum=z.new_zeros(self.padded//self.group_read_size,z.shape[-1])
        group_sum.index_add_(0,self.group_read_ids,real)
        group_mean=group_sum/self.group_read_counts.clamp_min(1).unsqueeze(-1)
        return real+self.group_read_scale*(group_mean.index_select(0,self.group_read_ids)-real)

    def forward(self,x,edge_index=None):
        x=F.dropout(x,self.dropout,self.training)
        h=self.encoder(x)
        if self.graph_embed:
            h=h+torch.sparse.mm(self.adjacency,self.neighbor_encoder(x))
        h=self.input_norm(h)
        if (self.refresh is not None and self.group_read_ids is None
                and all(block.exchange_scale==0. for block in self.blocks)):
            # With no exchange, padded wires cannot affect real nodes. Keep
            # only real nodes and share each block's norm with its refresh.
            for i,block in enumerate(self.blocks):
                normalized=block.norm(h)
                local=self._refresh_local(i,normalized)
                h=block(h,local,normalized=normalized)
            return self.classifier(h)
        z=h.new_zeros(self.padded,h.shape[-1]).index_copy(0,self.mapping,h)
        for i,block in enumerate(self.blocks):
            local=None
            circuit_input_override=None
            if self.refresh is not None:
                value=block.norm(z)[self.mapping]
                local_value=self._refresh_local(i,value)
                local=z.new_zeros(z.shape).index_copy(0,self.mapping,local_value)
                if self.graph_innovation and block.exchange_scale>0.:
                    # Route the node's innovation relative to its original
                    # graph neighborhood, while the local message stays put.
                    neighbor=torch.sparse.mm(self.adjacency,value)
                    circuit_input_override=z.new_zeros(z.shape).index_copy(
                        0,self.mapping,value-neighbor)
            z=block(z,local,circuit_input_override=circuit_input_override)
        return self.classifier(self._group_read(z))


class DensePureGCM(nn.Module):
    """Batched pure circuits for communication tasks; no graph propagation."""
    def __init__(self,input_dim,output_dim,nodes,dim=64,heads=4,blocks=2,mode='phase',stage_ff=True,skip_padding_pairs=False):
        super().__init__()
        idx=torch.arange(nodes)
        pairs=torch.stack([torch.stack([idx[(idx&(1<<s))==0],idx[(idx&(1<<s))==0]^(1<<s)])
                           for s in range(nodes.bit_length()-1)])
        self.encoder=nn.Linear(input_dim,dim)
        real_wire_mask=torch.ones(nodes,dtype=torch.bool) if skip_padding_pairs else None
        self.blocks=nn.ModuleList(PureBlock(dim,heads,pairs,mode,0.,stage_ff,.2,
                                            skip_padding_pairs=skip_padding_pairs,
                                            real_wire_mask=real_wire_mask) for _ in range(blocks))
        self.output=nn.Linear(dim,output_dim)

    def forward(self,x):
        x=self.encoder(x)
        for block in self.blocks: x=block(x)
        return self.output(x)
