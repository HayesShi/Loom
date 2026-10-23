"""First-order backward reconstruction for invariant-controlled circuits."""
import torch


class ReconstructCircuit(torch.autograd.Function):
    @staticmethod
    def forward(ctx,z,c,module,*parameters):
        for index in range(len(module.pairs)):
            z=module.stage(z,c,index)
        ctx.module=module
        ctx.save_for_backward(z,c,*parameters)
        return z

    @staticmethod
    def backward(ctx,gradient):
        current,control,*parameters=ctx.saved_tensors
        module=ctx.module
        control_gradient=torch.zeros_like(control)
        parameter_gradients=[None]*len(parameters)
        for index in reversed(range(len(module.pairs))):
            with torch.no_grad():
                previous=module.stage(current,control,index,inverse=True)
            with torch.enable_grad():
                x=previous.detach().requires_grad_(True)
                c=control.detach().requires_grad_(True)
                output=module.stage(x,c,index)
                grads=torch.autograd.grad(output,(x,c,*parameters),gradient,allow_unused=True)
            gradient=grads[0]
            if grads[1] is not None:
                control_gradient=control_gradient+grads[1]
            for k,g in enumerate(grads[2:]):
                if g is not None:
                    parameter_gradients[k]=g if parameter_gradients[k] is None else parameter_gradients[k]+g
            current=previous
        return (gradient,control_gradient,None,*parameter_gradients)
