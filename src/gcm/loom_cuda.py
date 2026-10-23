"""Optional FP32 CUDA execution of a complete Loom mixing stage.

Uses CuPy only to compile/launch kernels on PyTorch-owned storage and streams.
The checkpoint parameters and content-dependent lifting equations are unchanged.
"""
from functools import lru_cache
import numpy as np
import torch
from torch.autograd.function import once_differentiable


@lru_cache(None)
def _kernels():
    import cupy as cp
    code=r'''
    extern "C" __global__ void stage_forward(const float* z,const float* gate,
        const float* direction,const float* bias,const float* base,const float* delta,
        float* score,float* out,int M,int D,int H,int stride,float scale) {
        int r=blockIdx.x,h=threadIdx.x/32,lane=threadIdx.x%32;
        int left=(r/stride)*(2*stride)+r%stride;
        float raw=0.f;
        for(int c=lane;c<D;c+=32) {
            float u=z[left*D+c],v=z[(left+stride)*D+c],diff=u-v;
            raw+=(u+v)*gate[h*2*D+c]+fabsf(diff)*gate[h*2*D+D+c]+diff*direction[h*D+c];
        }
        for(int off=16;off>0;off/=2)raw+=__shfl_down_sync(0xffffffff,raw,off);
        raw=__shfl_sync(0xffffffff,raw,0)+bias[h];
        if(lane==0)score[r*H+h]=raw;
        int W=D/H;
        if(lane<W) {
            int c=h*W+lane,a=left*D+c,b=(left+stride)*D+c;
            float angle=base[h]+.5f*tanhf(raw/scale);
            float t=tanf(fminf(1.25f,fmaxf(-1.25f,.5f*angle)));
            float s=sinf(angle)+tanhf(delta[h]);
            float keep=1.f-t*s,cross=2.f*t-t*t*s,sign=(c&1)?1.f:-1.f;
            out[a]=keep*z[a]+cross*sign*z[b^1];
            out[b]=keep*z[b]+s*sign*z[a^1];
        }
    }
    extern "C" __global__ void stage_forward_64_4(const float* z,const float* gate,
        const float* direction,const float* bias,const float* base,const float* delta,
        float* score,float* out,int M,int stride,float scale) {
        int r=blockIdx.x*8+threadIdx.x/32,lane=threadIdx.x%32,h=lane/8,j=lane%8;
        if(r>=M)return;
        int left=(r/stride)*(2*stride)+r%stride;
        float raw=0.f;
        #pragma unroll
        for(int c=j;c<64;c+=8) {
            float u=z[left*64+c],v=z[(left+stride)*64+c],diff=u-v;
            raw+=(u+v)*gate[h*128+c]+fabsf(diff)*gate[h*128+64+c]+diff*direction[h*64+c];
        }
        for(int off=4;off>0;off/=2)raw+=__shfl_down_sync(0xffffffff,raw,off,8);
        raw=__shfl_sync(0xffffffff,raw,0,8)+bias[h];
        if(j==0)score[r*4+h]=raw;
        float angle=base[h]+.5f*tanhf(raw/scale);
        float t=tanf(fminf(1.25f,fmaxf(-1.25f,.5f*angle))),s=sinf(angle)+tanhf(delta[h]);
        float keep=1.f-t*s,cross=2.f*t-t*t*s;
        #pragma unroll
        for(int k=0;k<2;k++) {
            int c=h*16+j+k*8,a=left*64+c,b=(left+stride)*64+c;
            float sign=(c&1)?1.f:-1.f;
            out[a]=keep*z[a]+cross*sign*z[b^1];out[b]=keep*z[b]+s*sign*z[a^1];
        }
    }
    extern "C" __global__ void features(const float* z, float* f,
        int M, int D, int stride) {
        int q=blockIdx.x*blockDim.x+threadIdx.x;
        if(q>=M*D)return;
        int r=q/D,c=q%D,left=(r/stride)*(2*stride)+r%stride;
        float u=z[left*D+c],v=z[(left+stride)*D+c];
        f[r*3*D+c]=u+v;
        f[r*3*D+D+c]=fabsf(u-v);
        f[r*3*D+2*D+c]=u-v;
    }
    extern "C" __global__ void couple(const float* z,const float* score,
        const float* base,const float* delta,float* out,
        int M,int D,int H,int stride,float scale) {
        int q=blockIdx.x*blockDim.x+threadIdx.x;
        if(q>=M*D)return;
        int r=q/D,c=q%D,h=c/(D/H),left=(r/stride)*(2*stride)+r%stride;
        int a=left*D+c,b=(left+stride)*D+c;
        float angle=base[h]+0.5f*tanhf(score[r*H+h]/scale);
        float t=tanf(fminf(1.25f,fmaxf(-1.25f,0.5f*angle)));
        float s=sinf(angle)+tanhf(delta[h]);
        float keep=1.f-t*s,cross=2.f*t-t*t*s;
        float sign=(c&1)?1.f:-1.f;
        out[a]=keep*z[a]+cross*sign*z[b^1];
        out[b]=keep*z[b]+s*sign*z[a^1];
    }
    extern "C" __global__ void coefficients_backward(
        const float* z,const float* go,const float* score,
        const float* base,const float* delta,float* derivatives,
        int M,int D,int H,int stride,float scale) {
        int r=blockIdx.x,c=threadIdx.x,h=c/(D/H),W=D/H;
        int left=(r/stride)*(2*stride)+r%stride;
        int a=left*D+c,b=(left+stride)*D+c;
        float angle=base[h]+0.5f*tanhf(score[r*H+h]/scale);
        float t=tanf(fminf(1.25f,fmaxf(-1.25f,0.5f*angle)));
        float td=tanhf(delta[h]),s=sinf(angle)+td;
        float sign=(c&1)?1.f:-1.f;
        float ga=go[a]*z[a]+go[b]*z[b];
        float gb=go[a]*sign*z[b^1];
        float gs=go[b]*sign*z[a^1];
        unsigned mask=__activemask();
        for(int off=W/2;off>0;off/=2) {
            ga+=__shfl_down_sync(mask,ga,off,W);
            gb+=__shfl_down_sync(mask,gb,off,W);
            gs+=__shfl_down_sync(mask,gs,off,W);
        }
        if(c%W==0) {
            float dt=-s*ga+(2.f-2.f*t*s)*gb;
            float ds=-t*ga-t*t*gb+gs;
            float da=ds*cosf(angle);
            if(0.5f*angle>=-1.25f && 0.5f*angle<=1.25f)da+=dt*.5f*(1.f+t*t);
            float th=tanhf(score[r*H+h]/scale);
            derivatives[r*H+h]=da*.5f*(1.f-th*th)/scale;
            derivatives[M*H+r*H+h]=da;
            derivatives[2*M*H+r*H+h]=ds*(1.f-td*td);
        }
    }
    extern "C" __global__ void input_backward(
        const float* z,const float* go,const float* df,const float* score,
        const float* base,const float* delta,float* dz,
        int M,int D,int H,int stride,float scale) {
        int q=blockIdx.x*blockDim.x+threadIdx.x;
        if(q>=M*D)return;
        int r=q/D,c=q%D,h=c/(D/H),left=(r/stride)*(2*stride)+r%stride;
        int a=left*D+c,b=(left+stride)*D+c;
        float angle=base[h]+.5f*tanhf(score[r*H+h]/scale);
        float t=tanf(fminf(1.25f,fmaxf(-1.25f,.5f*angle)));
        float s=sinf(angle)+tanhf(delta[h]);
        float keep=1.f-t*s,cross=2.f*t-t*t*s,sign=(c&1)?1.f:-1.f;
        float diff=z[a]-z[b],sgn=(diff>0.f)?1.f:((diff<0.f)?-1.f:0.f);
        float fs=df[r*3*D+c],fd=sgn*df[r*3*D+D+c]+df[r*3*D+2*D+c];
        dz[a]=keep*go[a]-s*sign*go[b^1]+fs+fd;
        dz[b]=keep*go[b]-cross*sign*go[a^1]+fs-fd;
    }
    extern "C" __global__ void input_backward_direct(
        const float* z,const float* go,const float* dsraw,const float* gate,const float* direction,
        const float* score,const float* base,const float* delta,float* dz,
        int M,int D,int H,int stride,float scale) {
        int q=blockIdx.x*blockDim.x+threadIdx.x;
        if(q>=M*D)return;
        int r=q/D,c=q%D,h=c/(D/H),left=(r/stride)*(2*stride)+r%stride;
        int a=left*D+c,b=(left+stride)*D+c;
        float angle=base[h]+.5f*tanhf(score[r*H+h]/scale);
        float t=tanf(fminf(1.25f,fmaxf(-1.25f,.5f*angle)));
        float s=sinf(angle)+tanhf(delta[h]);
        float keep=1.f-t*s,cross=2.f*t-t*t*s,sign=(c&1)?1.f:-1.f;
        float diff=z[a]-z[b],sgn=(diff>0.f)?1.f:((diff<0.f)?-1.f:0.f);
        float fs=0.f,fd=0.f;
        for(int j=0;j<H;j++) {
            float g=dsraw[r*H+j];
            fs+=g*gate[j*2*D+c];
            fd+=g*(sgn*gate[j*2*D+D+c]+direction[j*D+c]);
        }
        dz[a]=keep*go[a]-s*sign*go[b^1]+fs+fd;
        dz[b]=keep*go[b]-cross*sign*go[a^1]+fs-fd;
    }
    extern "C" __global__ void stage_backward(
        const float* z,const float* go,const float* score,const float* gate,
        const float* direction,const float* base,const float* delta,
        float* derivatives,float* dz,int M,int D,int H,int stride,float scale) {
        __shared__ float raw_derivative[32];
        int r=blockIdx.x,c=threadIdx.x,W=D/H,h=c/W;
        int left=(r/stride)*(2*stride)+r%stride,a=left*D+c,b=(left+stride)*D+c;
        float angle=base[h]+.5f*tanhf(score[r*H+h]/scale);
        float t=tanf(fminf(1.25f,fmaxf(-1.25f,.5f*angle)));
        float td=tanhf(delta[h]),s=sinf(angle)+td,sign=(c&1)?1.f:-1.f;
        float ga=go[a]*z[a]+go[b]*z[b],gb=go[a]*sign*z[b^1],gs=go[b]*sign*z[a^1];
        unsigned mask=__activemask();
        for(int off=W/2;off>0;off/=2) {
            ga+=__shfl_down_sync(mask,ga,off,W);
            gb+=__shfl_down_sync(mask,gb,off,W);
            gs+=__shfl_down_sync(mask,gs,off,W);
        }
        if(c%W==0) {
            float dt=-s*ga+(2.f-2.f*t*s)*gb,ds=-t*ga-t*t*gb+gs;
            float da=ds*cosf(angle);
            if(.5f*angle>=-1.25f && .5f*angle<=1.25f)da+=dt*.5f*(1.f+t*t);
            float th=tanhf(score[r*H+h]/scale),dr=da*.5f*(1.f-th*th)/scale;
            raw_derivative[h]=dr;derivatives[r*H+h]=dr;
            derivatives[M*H+r*H+h]=da;derivatives[2*M*H+r*H+h]=ds*(1.f-td*td);
        }
        __syncthreads();
        float diff=z[a]-z[b],sgn=(diff>0.f)?1.f:((diff<0.f)?-1.f:0.f),fs=0.f,fd=0.f;
        for(int j=0;j<H;j++) {
            float g=raw_derivative[j];fs+=g*gate[j*2*D+c];
            fd+=g*(sgn*gate[j*2*D+D+c]+direction[j*D+c]);
        }
        float keep=1.f-t*s,cross=2.f*t-t*t*s;
        dz[a]=keep*go[a]-s*sign*go[b^1]+fs+fd;
        dz[b]=keep*go[b]-cross*sign*go[a^1]+fs-fd;
    }
    // Roman configuration: coalesced feature reads and no expanded feature tensor.
    extern "C" __global__ void parameter_partials_64_4(
        const float* z,const float* derivatives,float* partial,int M,int stride) {
        __shared__ float sums[8*4*3*32];
        int c=blockIdx.y*32+threadIdx.x%32,rl=threadIdx.x/32,chunk=blockIdx.x;
        float acc[4][3]={};
        for(int rr=rl;rr<256;rr+=8) {
            int r=chunk*256+rr;
            if(r<M) {
                int left=(r/stride)*(2*stride)+r%stride;
                float u=z[left*64+c],v=z[(left+stride)*64+c],diff=u-v;
                #pragma unroll
                for(int h=0;h<4;h++) {
                    float dr=derivatives[r*4+h];
                    acc[h][0]+=dr*(u+v);acc[h][1]+=dr*fabsf(diff);acc[h][2]+=dr*diff;
                }
            }
        }
        #pragma unroll
        for(int h=0;h<4;h++) {
            #pragma unroll
            for(int k=0;k<3;k++)sums[((rl*4+h)*3+k)*32+c%32]=acc[h][k];
        }
        __syncthreads();
        if(rl==0) {
            #pragma unroll
            for(int h=0;h<4;h++) {
                #pragma unroll
                for(int k=0;k<3;k++) {
                    float total=0.f;
                    #pragma unroll
                    for(int j=0;j<8;j++)total+=sums[((j*4+h)*3+k)*32+c%32];
                    partial[chunk*780+h*195+k*64+c]=total;
                }
            }
        }
        if(blockIdx.y==0 && threadIdx.x<4) {
            int h=threadIdx.x;float bias=0.f,base=0.f,delta=0.f;
            for(int rr=0;rr<256 && chunk*256+rr<M;rr++) {
                int r=chunk*256+rr;bias+=derivatives[r*4+h];
                base+=derivatives[M*4+r*4+h];delta+=derivatives[2*M*4+r*4+h];
            }
            partial[chunk*780+h*195+192]=bias;
            partial[chunk*780+h*195+193]=base;
            partial[chunk*780+h*195+194]=delta;
        }
    }
    extern "C" __global__ void parameter_reduce_64_4(
        const float* partial,float* gradient,int chunks) {
        int c=blockIdx.x*blockDim.x+threadIdx.x;
        if(c>=780)return;
        float total=0.f;
        for(int j=0;j<chunks;j++)total+=partial[j*780+c];
        gradient[c]=total;
    }
    '''
    module=cp.RawModule(code=code,options=('--std=c++11','--fmad=false'),
        name_expressions=['features','couple','coefficients_backward','input_backward','stage_forward','input_backward_direct',
                          'stage_backward','parameter_partials_64_4','parameter_reduce_64_4','stage_forward_64_4'])
    return cp,{name:module.get_function(name) for name in
        ['features','couple','coefficients_backward','input_backward','stage_forward','input_backward_direct',
         'stage_backward','parameter_partials_64_4','parameter_reduce_64_4','stage_forward_64_4']}


def _array(tensor,cp):
    # CuPy does not own the allocation; tensor ownership covers the async launch.
    memory=cp.cuda.UnownedMemory(tensor.data_ptr(),tensor.numel()*tensor.element_size(),tensor)
    return cp.ndarray(tensor.shape,dtype=cp.float32,memptr=cp.cuda.MemoryPointer(memory,0))


def _launch(name,grid,block,args,device):
    cp,kernels=_kernels()
    with cp.cuda.Device(device.index):
        stream=cp.cuda.ExternalStream(torch.cuda.current_stream(device).cuda_stream)
        converted=tuple(_array(v,cp) if isinstance(v,torch.Tensor) else v for v in args)
        kernels[name](grid,block,converted,stream=stream)


class _Stage(torch.autograd.Function):
    @staticmethod
    def forward(ctx,z,gate,direction,bias,base,delta,stride,scale):
        z=z.contiguous();D=z.shape[-1];M=z.numel()//(2*D);H=base.numel()
        args=(np.int32(M),np.int32(D),np.int32(H),np.int32(stride),np.float32(scale))
        score=z.new_empty((M,H))
        out=torch.empty_like(z)
        if D==64 and H==4:
            _launch('stage_forward_64_4',((M+7)//8,),(256,),
                (z,gate,direction,bias,base,delta,score,out,args[0],args[3],args[4]),z.device)
        else:
            _launch('stage_forward',(M,),(H*32,),
                (z,gate,direction,bias,base,delta,score,out,*args),z.device)
        ctx.save_for_backward(z,score,gate,direction,base,delta)
        ctx.args=args
        return out

    @staticmethod
    @once_differentiable
    def backward(ctx,go):
        z,score,gate,direction,base,delta=ctx.saved_tensors
        args=ctx.args;M,D,H=map(int,args[:3]);go=go.contiguous()
        derivatives=z.new_empty((3,M,H))
        dz=torch.empty_like(z)
        _launch('stage_backward',(M,),(D,),
            (z,go,score,gate,direction,base,delta,derivatives,dz,*args),z.device)
        if D==64 and H==4:
            chunks=(M+255)//256
            partial=z.new_empty((chunks,4,195));gradient=z.new_empty((4,195))
            _launch('parameter_partials_64_4',(chunks,2),(256,),
                (z,derivatives,partial,args[0],args[3]),z.device)
            _launch('parameter_reduce_64_4',(4,),(256,),
                (partial,gradient,np.int32(chunks)),z.device)
            return (dz,gradient[:,:128],gradient[:,128:192],gradient[:,192],
                    gradient[:,193],gradient[:,194],None,None)
        # Recompute inexpensive input features instead of retaining them at all stages.
        features=z.new_empty((M,3*D))
        _launch('features',((M*D+255)//256,),(256,),
            (z,features,args[0],args[1],args[3]),z.device)
        if M>=256 and M%256==0:
            dw=torch.bmm(derivatives[0].view(-1,256,H).transpose(1,2),features.view(-1,256,3*D)).sum(0)
        else:
            dw=derivatives[0].T@features
        return (dz,dw[:,:2*D],dw[:,2*D:],derivatives[0].sum(0),
                derivatives[1].sum(0),derivatives[2].sum(0),None,None)


def supports(z,pair):
    width=pair.width
    return (z.is_cuda and z.dtype==torch.float32 and z.shape[-1]<=256
            and not torch.is_autocast_enabled() and pair.gate.weight.dtype==torch.float32
            and pair.mode=='lifting' and pair.heads<=32
            and width<=32 and width>=2 and width&(width-1)==0)


def stage(z,pair,stride):
    return _Stage.apply(z,pair.gate.weight,pair.direction_gate.weight,pair.gate.bias,
                        pair.base,pair.middle_delta,stride,pair.gate_scale)
