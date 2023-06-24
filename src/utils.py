import plotly.express as px
import plotly.io as pio
import plotly.graph_objects as go
pio.renderers.default = "notebook_connected" # or use "browser" if you want plots to open with browser
import torch as t
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import einops
from fancy_einsum import einsum
from typing import List, Optional, Callable, Tuple, Union
import functools
from tqdm import tqdm
from IPython.display import display
import random
import os
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from sklearn.utils import shuffle
import json


from transformer_lens.hook_points import HookPoint
from transformer_lens import utils, HookedTransformer, HookedTransformerConfig, FactoredMatrix, ActivationCache

# Saves computation time, since we don't need it for the contents of this notebook
t.set_grad_enabled(False)


def SVD(matrix: t.Tensor) -> Tuple[t.Tensor, t.Tensor, t.Tensor]:
    """
    Compute the SVD of a matrix.
    Returns the three associated matrices
    """
    U, S, V_H = t.linalg.svd(matrix)
    return U, S, V_H

def dataset_activations(
    model: HookedTransformer,
    dataset: List[str]
):
    """
    Run a dataset through the model, store all the activations.
    Note: this is the unoptimised version, which runs the entire dataset through the model at once.
    This returns a cache with all activations (not just the final ones).
    """
    # Tokenise the batch, form a batch tensor
    batch_tokens = model.to_tokens(dataset)
    # Feed the tensor through the model
    logits, cache = model.run_with_cache(batch_tokens, return_cache_object=True, remove_batch_dim=False)

    return logits, cache


def dataset_activations_optimised(
    model: HookedTransformer,
    dataset: List[str],
    location: str,
    max_batch_size: int
):
    """
    Runs a dataset through the model, stores the activations of the final token of each sequence.
    This is the optimised version, which runs the dataset in batches.
    It also only stores the final activations, and not the entire cache.
    """
    num_batches = (len(dataset) + max_batch_size - 1) // max_batch_size
    all_final_activations = []
    # Process each batch
    for batch_idx in range(num_batches):
        t.cuda.empty_cache()
        # print("batch_idx be: ", batch_idx)

        # Determine the start and end index for this batch
        start_idx = batch_idx * max_batch_size
        end_idx = min(start_idx + max_batch_size, len(dataset))

        # Extract the subset of the dataset for this batch
        batch_subset = dataset[start_idx:end_idx]

        # Tokenise the batch, form a batch tensor
        batch_tokens = model.to_tokens(batch_subset)

        mask = batch_tokens != 50256
        final_indices = ((mask.cumsum(dim=1) == mask.sum(dim=1).unsqueeze(1)).int()).argmax(dim=1)
        final_indices = final_indices.view(-1,1)

        # Feed the tensor through the model
        _, cache = model.run_with_cache(batch_tokens, return_cache_object=True, remove_batch_dim=False)
        activations = cache[location]

        # # Take the last activation
        index_expanded = final_indices.unsqueeze(-1).expand(-1, -1, activations.size(2))
        # print("index_expanded: ", index_expanded)
        final_activations = t.gather(activations, 1, index_expanded)
        # Move the activations to the CPU and store them
        final_activations = final_activations.cpu()
        final_activations = final_activations.squeeze()
        all_final_activations.append(final_activations)

    # # Concatenate all activation tensors into a single tensor
    all_final_activations = t.cat(all_final_activations, dim=0)

    return all_final_activations


def reshape_activations(
    batch_activations: t.Tensor
) -> t.Tensor:
    """
    Rearrange a pytorch tensor of shape (batch_size, num_tokens, num_features) 
    into a pytorch tensor of shape (batch_size * num_tokens, num_features).
    """
    squeezed_tensor = einops.rearrange(batch_activations, 'b tokens dim -> (b tokens) dim')
    return squeezed_tensor


def activation_SVD(
    model: HookedTransformer,
    dataset: List[str],
    location: str
) -> Tuple[t.Tensor, t.Tensor, t.Tensor]:
    """
    Given a model, a dataset, and a location in the model, 
    compute the SVD of the activations at that location.
    """
    _, cache = dataset_activations(model, dataset)
    activation_cache = cache[location]
    squeezed_activations = reshape_activations(activation_cache)
    U, S, V_H = SVD(squeezed_activations)
    return U, S, V_H


def dataset_activations_tokens(
    model: HookedTransformer,
    dataset_tokens: t.Tensor
):
    """
    Run a dataset which is already tokenised (and stored as a tensor)
    through the model, return the logits and all the activations.
    """
    # Tokenise the batch, form a batch tensor
    batch_tokens = dataset_tokens
    # Feed the tensor through the model
    logits, cache = model.run_with_cache(batch_tokens, return_cache_object=True, remove_batch_dim=False)
    return logits, cache


def activation_SVD_tokens(
    model: HookedTransformer,
    dataset_tokens: t.Tensor,
    location: str
) -> Tuple[t.Tensor, t.Tensor, t.Tensor]:
    """
    Given a model, a dataset (expressed as a tensor of tokens),
    and a location in the model, compute the SVD of the activations at that location.
    """
    _, cache = dataset_activations_tokens(model, dataset_tokens)
    activation_cache = cache[location]
    squeezed_activations = reshape_activations(activation_cache)
    U, S, V_H = SVD(squeezed_activations)
    return U, S, V_H


def activation_SVD_covariance(
    model: HookedTransformer,
    dataset_tokens: List[str],
    location: str
) -> Tuple[t.Tensor, t.Tensor, t.Tensor]:
    """
    Given a model, a dataset (in text), and a location in the model, 
    compute the SVD of the normalised covariance activation matrix at that location.
    """
    _, cache = dataset_activations_tokens(model, dataset_tokens)
    activation_cache = cache[location]
    squeezed_activations = reshape_activations(activation_cache)
    print(squeezed_activations.shape)
    mean_activation = squeezed_activations.mean(dim=0, keepdim=True)
    centred_activations = squeezed_activations - mean_activation
    covariance_matrix = centred_activations.T @ centred_activations
    print(covariance_matrix.shape)
    U, S, V_H = SVD(covariance_matrix)
    return U, S, V_H

# Using tokens as a starting point, and also using the covariance matrix
def activation_SVD_tokens_covariance(
    model: HookedTransformer,
    dataset_tokens: t.Tensor,
    location: str
) -> Tuple[t.Tensor, t.Tensor, t.Tensor]:
    """
    Given a model, a dataset (expressed as a tensor of tokens),
    and a location in the model, 
    compute the SVD of the normalised covariance activation matrix at that location.
    """
    _, cache = dataset_activations_tokens(model, dataset_tokens)
    activation_cache = cache[location]
    squeezed_activations = reshape_activations(activation_cache)
    mean_activation = squeezed_activations.mean(dim=0, keepdim=True)
    centred_activations = squeezed_activations - mean_activation
    covariance_matrix = centred_activations.T @ centred_activations
    U, S, V_H = SVD(covariance_matrix)
    return U, S, V_H

