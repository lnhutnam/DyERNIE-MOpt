from typing import Any, Tuple
import torch
import numpy as np

max_norm = 85
eps = 1e-8

def to_device(tensor):
    if torch.cuda.is_available():
        return tensor.cuda()
    else:
        return tensor

class LeakyClamp(torch.autograd.Function):
    @staticmethod
    def forward(ctx: Any, x: torch.Tensor, min: float, max: float) -> torch.Tensor:
        ctx.save_for_backward(x.ge(min) * x.le(max))
        return torch.clamp(x, min=min, max=max)

    @staticmethod
    def backward(ctx: Any, grad_output: torch.Tensor) -> Tuple[torch.Tensor, None, None]:
        mask, = ctx.saved_tensors
        mask = mask.type_as(grad_output)
        return grad_output * mask + grad_output * (1 - mask) * eps, None, None

def clamp(x: torch.Tensor, min: float = float("-inf"), max: float = float("+inf")) -> torch.Tensor:
    return LeakyClamp.apply(x, min, max)

def cosh(x: torch.Tensor) -> torch.Tensor:
    x = clamp(x, min=-max_norm, max=max_norm)
    return torch.cosh(x)

def save(filename, log_file, model, args, opts, epoch, entity_idxs, relation_idxs, timestamp_idxs, main_dirName):
    """Save current state to specified file"""
    log_file.write("Saving checkpoint to {}... \n".format(filename))
    model = [component.state_dict() for component in model]
    torch.save(
        {
            "type": "train",
            "epoch": epoch,
            "model": model,
            "optimizer_state_dict": [optimizer.state_dict() for optimizer in opts],
            "entity_idxs": entity_idxs,
            "relation_idxs" : relation_idxs,
            "timestamp_idxs" : timestamp_idxs,
            "learning_rate" : args.lr,
            "dim" : args.dim,
            "nneg" : args.nneg,
            "num_iterations" : args.num_iterations,
            "batch_size" : args.batch_size,
            "batch_size_eva" : args.batch_size_eva,
            "lr_cur" : args.lr_cur,
            "curvatures_fixed" : args.curvatures_fixed,
            "curvatures_trainable" : args.curvatures_trainable,
            # "tr_cur" : args.trainable_curvature,
            "main_dirName" : main_dirName,
            "dataset" : args.dataset,
            "time_rescale" : args.time_rescale
        },
        filename,
    )