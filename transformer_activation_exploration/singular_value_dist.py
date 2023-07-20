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


import transformer_activation_exploration.utils
# Saves computation time, since we don't need it for the contents of this notebook
t.set_grad_enabled(False)

def svd_distributions(
    location: str,
    model: HookedTransformer,
    text_datasets: dict[str, List[str]],
    token_datasets: dict[str, t.Tensor],
    covariance: bool = False
):
    """
    For a given transformer model, at a given locations,
    plot the singular value distributions of the activation matrices.
    If covariance is True, plot the singular value distributions of the covariance activation matrices.
    TODO: Log these results besides plotting them.
    """
    for dataset_name, dataset in text_datasets.items():
        if covariance:
            _, S, _ = transformer_activation_exploration.utils.activations_SVD_covariance(model, dataset, location)
        else:
            _, S, _ = transformer_activation_exploration.utils.activations_SVD(model, dataset, location)
        plt.plot(S.cpu(), label=dataset_name)
    
    for dataset_name, dataset in token_datasets.items():
        if covariance:
            _, S, _ = transformer_activation_exploration.utils.activations_SVD_covariance_tokens(model, dataset, location)
        else:
            _, S, _ = transformer_activation_exploration.utils.activations_SVD_tokens(model, dataset, location)
        plt.plot(S.cpu(), label=dataset_name)

    # Add labels and title
    plt.title(f'Singular values of dataset activations for {model.name} at {location}')
    plt.yscale('log')
    plt.ylabel("Singular value")
    plt.xlabel("Rank")

    # Add legend
    plt.legend()

    # Display the plot
    plt.show()


if __name__ == "__main__":
    # Load the model
    gpt2_small = gpt2_small = transformer_activation_exploration.utils.load_model('gpt2-small')
    svd_distributions('blocks.7.hook_attn_out', gpt2_small)