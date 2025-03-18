from typing import Union

import torch
from torch import nn

import numpy as np
from gym.wrappers.frame_stack import LazyFrames

Activation = Union[str, nn.Module]

_str_to_activation = {
    'relu': nn.ReLU(),
    'tanh': nn.Tanh(),
    'leaky_relu': nn.LeakyReLU(),
    'sigmoid': nn.Sigmoid(),
    'selu': nn.SELU(),
    'softplus': nn.Softplus(),
    'identity': nn.Identity(),
}
    
def build_mlp(
        input_size: int,
        output_size: int,
        **kwargs
    ):
    """
    Builds a feedforward neural network

    arguments:
        n_layers: number of hidden layers
        size: dimension of each hidden layer
        activation: activation of each hidden layer
        input_size: size of the input layer
        output_size: size of the output layer
        output_activation: activation of the output layer

    returns:
        MLP (nn.Module)
    """
    try:
        params = kwargs["params"]
    except:
        params = kwargs

    # Extract parameters
    n_layers = params.get("n_layers", 2)  # Default to 2 hidden layers
    size = params.get("size", 64)         # Default size of hidden layers
    activation = params.get("activation", "relu")  # Default activation is ReLU
    output_activation = params.get("output_activation", "identity")  # Default output activation is identity
    
    # Retrieve activation functions
    activation_fn = _str_to_activation.get(activation, nn.Identity())
    if isinstance(params["output_activation"], str):
        output_activation_fn = _str_to_activation.get(params["output_activation"], nn.Identity())
    else:
        output_activation_fn = params["output_activation"]

    # Build layers
    layers = []
    in_dim = input_size

    for _ in range(n_layers):
        layers.append(nn.Linear(in_dim, size))
        layers.append(activation_fn)
        in_dim = size

    # Add output layer
    layers.append(nn.Linear(in_dim, output_size))
    layers.append(output_activation_fn)

    # Return as nn.Sequential
    return nn.Sequential(*layers)

device = None

def init_gpu(use_gpu=True, gpu_id=0):
    global device
    if torch.cuda.is_available() and use_gpu:
        device = torch.device("cuda:" + str(gpu_id))
        print("Using GPU id {}".format(gpu_id))
    else:
        device = torch.device("cpu")
        print("GPU not detected. Defaulting to CPU.")


def set_device(gpu_id):
    torch.cuda.set_device(gpu_id)

def from_numpy(*args, **kwargs):
    arr = args[0]

    # If arr is LazyFrames, convert to NumPy:
    if isinstance(arr, LazyFrames):
        arr = np.array(arr, copy=False)

    return torch.from_numpy(arr).float().to(device)

def to_numpy(tensor):
    return tensor.to('cpu').detach().numpy()
