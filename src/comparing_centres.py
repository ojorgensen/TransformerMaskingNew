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


def find_activations_centre(
  model: HookedTransformer,
  dataset: List[str],
  location: str,
  max_batch_size: int,
  use_all_activations: bool = False
):
  """
  Find the centre of the activations of a dataset, at some
  layer of a certain model.
  """
  all_activations = src.utils.dataset_activations_optimised_new(
    model,
    dataset,
    location,
    max_batch_size,
    use_all_activations
  )

  # Find the mean
  mean = t.mean(all_activations, dim=0)


  return mean

def find_activations_centre_diff(
  model: HookedTransformer,
  target_dataset: List[str],
  baseline_dataset: List[str],
  location: str,
  max_batch_size: int,
  use_all_activations: bool = False
):
  """
  Find the centre of the activations of the baseline dataset,
  take this away from the centre of the activations of a second dataset.

  Return the resulting difference vector
  """

  baseline_centre = find_activations_centre(
    model,
    baseline_dataset,
    location,
    max_batch_size,
    use_all_activations
  )

  baseline_target = find_activations_centre(
    model,
    target_dataset,
    location,
    max_batch_size,
    use_all_activations
  )

  difference = baseline_target - baseline_centre
  return difference