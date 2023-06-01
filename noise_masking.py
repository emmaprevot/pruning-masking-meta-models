import torch
import numpy as np
import json

import haiku as hk
import jax
import functools
from jax import nn, random, image, jit, grad
import jax.numpy as jnp
from jax.flatten_util import ravel_pytree

from haiku import initializers
from haiku._src.basic import Linear
from haiku._src.conv import Conv2D
from haiku._src.batch_norm import BatchNorm
from haiku._src.module import Module
from typing import Callable, Tuple
from haiku.initializers import Initializer, Constant, RandomNormal, TruncatedNormal, VarianceScaling

from optax import adam, rmsprop, sgd, softmax_cross_entropy, apply_updates

from scipy.stats import skew

import datasets


optimizer_dict = {
    "Adam": adam,
    "RMSProp": rmsprop,
    "MomentumSGD": functools.partial(sgd, momentum=0.9),
}

activation_dict = {
    "ReLU": nn.relu,
    "ELU": nn.elu,
    "Sigmoid": nn.sigmoid,
    "Tanh": nn.tanh,
}

initializer_dict = {
    "Constant": Constant(0.1),
    "RandomNormal": RandomNormal(),
    "GlorotUniform": VarianceScaling(1.0, "fan_avg", "uniform"),
    "GlorotNormal": VarianceScaling(1.0, "fan_avg", "truncated_normal"),
}

"""
dataset_dict = {
    "MNIST": datasets.load_dataset("mnist", split="train").with_format("jax"),
    "CIFAR-10": datasets.load_dataset("cifar10", split="train").with_format("jax").rename_column('img', 'image'),
    "SVHN": datasets.load_dataset("svhn", "cropped_digits", split="train").with_format("jax"),
    "Fashion-MNIST": datasets.load_dataset("fashion_mnist", split="train").with_format("jax"),  
}
"""



class CTCNet(hk.Module):
    def __init__(self,
                 n_classes: int,
                 activation: Callable = nn.relu,
                 w_init: Initializer = TruncatedNormal(),
                 kernel_size: Tuple[int, int] = (5, 5),
                 n_conv_layers: int = 3,
                 n_filters: int = 32,
                 n_fc_layers: int = 3,
                 fc_width: int = 128,
                 dropout_rate: float = 0.5):
        super().__init__()
        self.n_classes = n_classes
        self.activation = activation
        self.w_init = w_init
        self.kernel_size = kernel_size
        self.n_conv_layers = n_conv_layers
        self.n_filters = n_filters
        self.n_fc_layers = n_fc_layers
        self.fc_width = fc_width
        self.dropout_rate = dropout_rate

    def __call__(self, x: jnp.ndarray, is_training: bool) -> jnp.ndarray:
        for _ in range(self.n_conv_layers):
            x = hk.Conv2D(output_channels=self.n_filters, kernel_shape=self.kernel_size,
                          padding="SAME", w_init=self.w_init)(x)
            x = hk.BatchNorm(create_scale=True, create_offset=True, decay_rate=0.9)(x, is_training)
            x = self.activation(x)

        x = hk.Flatten()(x)

        for _ in range(self.n_fc_layers - 1):
            x = hk.Linear(self.fc_width, w_init=self.w_init)(x)
            x = hk.BatchNorm(create_scale=True, create_offset=True, decay_rate=0.9)(x, is_training)
            x = self.activation(x)
            x = hk.dropout(hk.next_rng_key(), self.dropout_rate, x) if is_training else x

        x = hk.Linear(self.n_classes, w_init=self.w_init)(x)
        return x


# Load the hyperparameters from the JSON file
with open('./transfer/0/run_data.json', 'r') as f:
    data = json.load(f)
hyperparameters = data['hyperparameters']

def net_fn(x: jnp.ndarray, is_training: bool) -> jnp.ndarray:
    return CTCNet(n_classes=10,
                  activation=activation_dict[hyperparameters['activation']],
                  w_init=initializer_dict[hyperparameters['initialization']],
                  n_conv_layers=3,
                  n_filters=32,
                  n_fc_layers=3,
                  fc_width=128,
                  dropout_rate=0.5)(x, is_training)

# Transform it into a haiku model
net = hk.transform(net_fn)

params = jnp.load('./transfer/0/epoch_20.npy', allow_pickle=True).item()

print("NETWORK ARCHITECTURE")
for layer, param_dict in params.items():
    for param_name, array in param_dict.items():
        print(f"{layer} {param_name} shape: {array.shape}")


"""
Maskig approach to pruning. This should still be considered as pruning but applied with a mask.
Essentially, you create a mask of the same shape as your weights. The mask contains 1s for weights you want to keep and 0s for weights you want to prune. 
Then, you simply multiply the weights with the mask. This has the effect of "zeroing out" the pruned weights.

In this example, we're creating a binary mask for each layer of weights, where values in the mask are set to 1 
if the corresponding weight is greater than the pruning threshold and 0 otherwise. 
"""


def create_mask(params, prune_ratio):
    mask = {}
    for layer, param_dict in params.items():
        sub_mask = {}
        for k, v in param_dict.items():
            if 'w' in k:
                # Compute the threshold
                threshold = jnp.percentile(jnp.abs(v), prune_ratio * 100)
                # Create a mask that is 1 where weights are greater than the threshold and 0 otherwise
                sub_mask[k] = jnp.where(jnp.abs(v) > threshold, 1, 0)
            else:
                # For biases just create a mask of ones
                sub_mask[k] = jnp.ones_like(v)
        mask[layer] = sub_mask
    return mask

def apply_mask(params, mask):
    pruned_params = {}
    for layer, param_dict in params.items():
        layer_params = {}
        for k, v in param_dict.items():
            if 'w' in layer_name:
                layer_param[k] = v * mask[layer][k]
            else:
                layer_param[k] = v
        pruned_params[layer] = layer_param
    return pruned_params


"""
Adding noise to weights. This can be done by simply creating a noise tensor of the same shape as your weights and then adding it. 

"""


def add_noise(params, noise_type = 'normal', std_dev=0.05, noise_ratio=0.1, scale = 1.0, lam=10.0):

    """
    The function adds different types of noise which correspond to different probability distributions.
    jax.random.bernoulli is used to generate a binary mask indicating where to apply noise.

    The standard type of noise is (normally-distributed) GAUSSIAN NOISE. 
    The jax.random.normal function is used to generate noise values with a standard deviation specified by the std_dev parameter. 
    This controls the amount of noise added - a larger std_dev means more noise.

    UNIFORM NOISE: The uniform distribution has equal probability for all values in a given range. 
    This means that every value within the defined bounds (minval and maxval) has an equal chance of being picked.

    BERNOULLI NOISE: The Bernoulli distribution has only two possible outcomes with a probability 'p' and '1-p'. 

    LAPLACE AND CAUCHY NOISE: The Laplace and Cauchy distributions are similar to the Gaussian distribution, but with heavier tails. 
    This means it is more likely to produce values far from the mean. 
    In the context of adding noise, the choice between these two would depend on your specific needs. 
    For instance, if you wanted noise that can occasionally generate more extreme outliers, you might opt for Cauchy noise. 
    If you want noise that is less likely to generate extreme values but still heavier-tailed than Gaussian noise, you might opt for Laplace noise.

    POISSON NOISE: The Poisson distribution is defined over the integers and is typically used to model counts. 
    Poisson noise is integer-valued and thus may not be suitable for all tasks.
    
    """
    rng = jax.random.PRNGKey(0)
    params_noisy = {}
    
    for layer, param_dict in params.items():
        layer_params = {}
        for k, v in param_dict.items():
            if 'w' in k:
                weights = v
                rng, subkey = jax.random.split(rng)
                noise_mask = jax.random.bernoulli(subkey, noise_ratio, weights.shape)
                rng, subkey = jax.random.split(rng)

                if noise_type == "normal":
                    noise_values = jax.random.normal(subkey, shape = weights.shape) * std_dev
                elif noise_type == "uniform":
                    noise_values = jax.random.uniform(subkey, shape = weights.shape, minval=-scale, maxval=scale) 
                elif noise_type == "laplace":
                    noise_values = jax.random.laplace(subkey, shape = weights.shape) * scale
                elif noise_type == "bernoulli":
                    noise_values = jax.random.bernoulli(subkey, p=0.5, shape = weights.shape) * scale
                elif noise_type == "cauchy":
                    noise_values = jax.random.cauchy(subkey, shape = weights.shape) * scale
                elif noise_type == "poisson":
                    noise_values = jax.random.poisson(subkey, lam=lam, shape = weights.shape)
            
                weights_noisy = jnp.where(noise_mask, weights + noise_values, weights)
                layer_params[k] = weights_noisy
            else:
                layer_params[k] = v
        params_noisy[layer] = layer_params
    return params_noisy




def main():
    for i in range(1):
        params = jnp.load('./transfer/'+ str(i) + '/epoch_20.npy', allow_pickle=True).item()

        # Create a mask based on a threshold
        mask = create_mask(params, prune_ratio=0.5)
        # Apply the mask to prune the parameters
        pruned_params = apply_mask(params, mask)

        # Adding Gaussian noise
        noisy_params = add_noise(params)

        jnp.save('./transfer/'+ str(i) + '/epoch_20_pruned.npy', pruned_params)
        jnp.save('./transfer/'+ str(i) + '/epoch_20_noisy.npy', noisy_params)

main()