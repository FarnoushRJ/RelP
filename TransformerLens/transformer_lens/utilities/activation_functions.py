"""Activation Functions.

Utilities for interacting with all supported activation functions.
"""
from typing import Callable, Dict, Optional, Union, Any

import torch
from torch import nn
import torch.nn.functional as F

from transformer_lens.utils import gelu_fast, gelu_new, solu
from transformer_lens.lrp_utils import stabilize

# Convenient type for the format of each activation function
ActivationFunction = Callable[..., torch.Tensor]

# All currently supported activation functions. To add a new function, simply
# put the name of the function as the key, and the value as the actual callable.
SUPPORTED_ACTIVATIONS: Dict[str, ActivationFunction] = {
    "solu": solu,
    "solu_ln": solu,
    "gelu_new": gelu_new,
    "gelu_fast": gelu_fast,
    "silu": F.silu,
    "relu": F.relu,
    "gelu": F.gelu,
    "gelu_pytorch_tanh": lambda tensor: F.gelu(tensor, approximate="tanh"),
}


# ----------- LRP ---------------
# activation function should be replaced with 'identity' in the backward path
class ModifiedAct(nn.Module):
    def __init__(
            self,
            act: Any,
            transform: Any
    ):
        """
       A wrapper to make activation layers such as torch.nn.SiLU or torch.nn.GELU explainable.
       -------------------

       :param act: an activation layer (torch.nn.SiLU or torch.nn.GELU).
       """
        super(ModifiedAct, self).__init__()
        self.modified_act = nn.Identity()
        self.act = act
        self.transform = transform

    def forward(
            self,
            x
    ):
        z = self.act(x)

        if self.transform is None or isinstance(self.act, nn.ReLU):
            return z
        elif self.transform == 'identity':
            zp = self.modified_act(x)
            zp = stabilize(zp)
            return zp * (z / zp).data
        else:
            raise NotImplementedError