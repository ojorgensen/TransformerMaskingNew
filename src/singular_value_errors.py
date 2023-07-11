# import plotly.express as px
# import plotly.io as pio
# import plotly.graph_objects as go
# pio.renderers.default = "notebook_connected" # or use "browser" if you want plots to open with browser
import torch as t
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import einops
from fancy_einsum import einsum
from typing import List, Optional, Callable, Tuple, Union, Dict
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

import src.utils

# Saves computation time, since we don't need it for the contents of this notebook
t.set_grad_enabled(False)



def approximation_error_experiment(
    model: HookedTransformer,
    location,
    target_datasets_lang,
    target_datasets_tokens,
    comparison_datasets_lang,
    comparison_datasets_tokens,
    n = None
):
    """
    target dataset: the dataset we will use to generate subspaces to project onto
    comparison dataset lang: the dataset we will project onto the subspaces, where the dataset is text

    n: number of points we want to use

    Perform the following experiment:
    1. Get the activations for the target dataset
    2. Do SVD for the target dataset. Get the first k singular values and the corresponding singular vectors
    3. Get the activations for the comparison dataset
    4. For each k less than the dimension of the activation space, project the comparison dataset onto the first k singular vectors.
    5. For each k, calculate the approximation error for the comparison dataset
    6. Plot the approximation error against k
    7. Repeat for each comparison dataset
    """
    # Get the activations for each dataset
    if target_datasets_lang != None:
        target_name = [key for key in target_datasets_lang][0]
        target_dataset_lang = target_datasets_lang[target_name]
        target_activations = src.utils.dataset_activations(model, target_dataset_lang)[1][location]
        target_X = src.utils.reshape_activations(target_activations)
        # Do SVD for target_X
        _, _, V_H = src.utils.activation_SVD(model, target_dataset_lang, location)

    else:
        target_name = [key for key in target_datasets_tokens][0]
        target_dataset_tokens = target_datasets_tokens[target_name]
        target_activations = src.utils.dataset_activations_tokens(model, target_dataset_tokens)[1][location]
        target_X = src.utils.reshape_activations(target_activations)
        # Do SVD for target_X
        _, _, V_H = src.utils.activation_SVD_tokens(model, target_dataset_tokens, location)


    comparison_X_dict = {}
    for key in comparison_datasets_lang:
        comparison_dataset = comparison_datasets_lang[key]
        comparison_activations = src.utils.dataset_activations(model, comparison_dataset)[1][location]
        comparison_X = src.utils.reshape_activations(comparison_activations)
        # Normalise rows of the comparison activations (to avoid penalising large vectors)
        norms = t.linalg.norm(comparison_X, dim=1, keepdim=True)
        comparison_X = comparison_X / norms
        comparison_X_dict[key] = comparison_X

    for key in comparison_datasets_tokens:
        comparison_dataset = comparison_datasets_tokens[key]
        comparison_activations = src.utils.dataset_activations_tokens(model, comparison_dataset)[1][location]
        comparison_X = src.utils.reshape_activations(comparison_activations)
        # Normalise rows of the comparison activations (to avoid penalising large vectors)
        norms = t.linalg.norm(comparison_X, dim=1, keepdim=True)
        comparison_X = comparison_X / norms
        comparison_X_dict[key] = comparison_X

    dim = comparison_X.shape[1]
    if n is None:
        n = dim

    # Form approximations for each dataset
    comp_errors_list = []
    for key in comparison_X_dict:
        comparison_X = comparison_X_dict[key]
        comp_errors = []
        target_errors = []
        for i in range(n):
            k = (i * dim) // n
            # Get errors for the comparison dataset
            approx_comparison_X = src.utils.top_k_projection(comparison_X, V_H, k)
            comp_errors.append(src.utils.matrix_error(approx_comparison_X, comparison_X).cpu())

            # Get errors for the target dataset
            approx_target_X = src.utils.top_k_projection(target_X, V_H, k)
            target_errors.append(src.utils.matrix_error(approx_target_X, target_X).cpu())

        comp_errors_list.append(comp_errors)
        plt.plot(comp_errors, label = key)

    # plt.yscale("log")
    plt.ylabel("Reconstruction error")
    plt.xlabel("Subspace dimension")
    plt.title(f"Dataset Reconstruction Errors at {location}. Target dataset {target_name} ")
    plt.legend()
    plt.show()

    return comp_errors_list

def pca_reconstruction_errors(
    model: HookedTransformer,
    target_dataset: List[str],
    comparison_datasets: Dict[str, List[str]],
    k: int,
    layers: int,
    location: str
):
    """
    Given a target dataset and a comparison dataset, with a transformer model,
    for each location in the model:
        1. Compute the singular value decomposition of activations of the final token
        of the model for the target dataset.
        2. Consider the subspace corresponding to the top k principle components. (determine k lmao)
        3. Project the activations corresponding to the comparison dataset onto this subspace.
        4. Look at the L_2 error of the reconstruction (as a fraction of the total length).

    Repeat this for every location in the model, construct a plot for the MLP layers, the attention layers, and
    the residual stream.
    """
    # Get the number of layers of the model
    # Shouldn't hardcode, but I know this for gpt2 xl is 48
    all_accuracies = {}
    all_comparison_activations_dict = {}
    target_activations_dict = src.utils.dataset_activations_optimised_locations(
        model,
        target_dataset,
        layers,
        location,
        2
    )
    for name, comparison_dataset in comparison_datasets.items():
        comparison_activations_dict = src.utils.dataset_activations_optimised_locations(
        model,
        comparison_dataset,
        layers,
        location,
        2
        )

        all_comparison_activations_dict[name] = comparison_activations_dict

        all_accuracies[name] = []

    for layer in range(layers):
        target_activations = target_activations_dict[layer]
        # Do SVD
        _, _, V_H = src.utils.SVD(target_activations)


        for name, comparison_dataset in comparison_datasets.items():
            comparison_activations_dict = all_comparison_activations_dict[name]
            comparison_activations = comparison_activations_dict[layer]
            # Project the comparison activations onto the top k vectors of V_H
            comparison_approx = top_k_projection(comparison_activations, V_H, k)

            # Get the errors
            error = matrix_error(comparison_approx, comparison_activations)
            total_l2_norms = t.sum(t.norm(comparison_activations, dim=1, p=2))
            error_fraction = error / total_l2_norms
            all_accuracies[name].append(1 - error_fraction)

    print(all_accuracies)

    for name, accuracies in all_accuracies.items():
        plt.plot(accuracies, label=name)
    plt.ylabel("Reconstruction accuracies (as fraction of original)")
    plt.xlabel("Layer")
    plt.title(f"Reconstruction accuracies of comparison projection on top {k} vectors of target activations at {location}")
    plt.legend()
    plt.show()


if __name__ == "__main__":
    # Load the model
    gpt2_small = src.utils.load_model("gpt2-small")

    # Load the datasets
    # TODO: load the datasets
    datasets_lang = {
        "addition": addition_dataset,
        "even addition": even_addition_dataset,
        "multiplication": multiplication_dataset,
        "odd multiplication": odd_multiplication_dataset,
        "shuffled addition": shuffled_addition_dataset,
        "combined addition multiplication": combined_add_mult_dataset
    }

    datasets_tokens = {
        "random token addition": random_token_addition_dataset,
        "random token": random_tokens,
        "training subset": training_subset_tokens
    }

    add_errors = approximation_error_experiment(
                    'blocks.9.hook_mlp_out', 
                    {"even addition":even_addition_dataset}, 
                    None, 
                    datasets_lang, 
                    datasets_tokens
                )
