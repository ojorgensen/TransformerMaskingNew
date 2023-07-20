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

import transformer_activation_exploration.utils

import transformer_activation_exploration.comparing_centres

# Saves computation time, since we don't need it for the contents of this notebook
t.set_grad_enabled(False)

def feature_detection_experiment(
    model: HookedTransformer,
    location: str,
    feature_dataset: List[str],
    baseline_dataset: List[str],
    use_all_activations: bool,
    evaluation_dataset: Dict[str, List[str]],
    inner_product_type: str,
    threshold_type: str,
):
    """
    Performs an experiment to see how well inner products with the vector arising from the
    feature dataset can be used to predict the labels of the evaluation dataset.
    """

    # Get the feature vector from the feature dataset
    # By default, have this be the vector from the centre of the baseline
    # activations to the centre of the feature activations
    feature_vector = transformer_activation_exploration.comparing_centres.find_activations_centre_diff(
        model,
        feature_dataset,
        baseline_dataset,
        location,
        2,
        use_all_activations
    )
    print("part 1 done")

    # Get the activations for the evaluation dataset
    evaluation_activations = {} 
    for label, dataset in evaluation_dataset.items():
        # Note: we are only using the final activations here (regardless of arg!)
        evaluation_activations[label] = transformer_activation_exploration.utils.dataset_activations_optimised_new(
            model,
            dataset,
            location,
            2,
            False
        )
    print("part 2 done")

    # Calculate the inner products between the feature vector and the activations
    inner_products = {}
    for label, activations in evaluation_activations.items():
        if inner_product_type == 'dot':
            inner_products[label] = t.einsum(
                'ij,j->i',
                activations,
                feature_vector
            )
        else:
            raise NotImplementedError("Only dot product is currently supported")
    print("part 3 done")

    # Calculate the threshold for the inner products (maybe using the feature + baseline dataset?)
    if threshold_type == "zero":
        threshold = 0
    else:
        raise NotImplementedError("Only zero threshold is currently supported")

    print("part 4 done")
    # Do classification
    # Values larger than 0 are classified as 1, values smaller than 0 are classified as 0
    for label, inner_product in inner_products.items():
        classification = t.where(inner_product > threshold, t.tensor(1), t.tensor(0))
    
    print("part 5 done")


    # Get metrics / plot histograms / ROC curves
    # Plot histograms
    for label, inner_product in inner_products.items():
        plt.hist(inner_product.cpu().numpy(), bins=100, label=label)
    plt.legend()
    plt.show()

    return
