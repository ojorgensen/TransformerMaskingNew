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

def activation_plot_final_acts_optimised(
  model: HookedTransformer,
  datasets_lang: List[str],
  datasets_tokens: List,
  plot_type: str,
  location: str,
  dimension: int,
  random: bool = False,
  centre: bool = True
):
  """
  Given a dataset, create a plot of the activations for the final token.
  Use gpt 2 small, look at any given location.
  Works for either numerical or categorical labels.
  Should support both t-SNE and PCA.
  TODO: include PCA support
  """
  t.cuda.empty_cache()
  activation_dict = {}
  # Do the forward pass for each dataset
  for name in datasets_lang:
    dataset_lang = datasets_lang[name]
    final_activations = transformer_activation_exploration.utils.dataset_activations_optimised(model, dataset_lang, location, 16)
    # Looking at the final activations! Might want to change this?

    # target_X = reshape_activations(activations)

    # Convert the tensor to a numpy array as scikit-learn works with numpy arrays
    data_numpy = final_activations.cpu().numpy()
    activation_dict[name] = data_numpy
    print("data numpy shape is: ", data_numpy.shape)

  for name in datasets_tokens:
    dataset_token = datasets_tokens[name]
    activations = dataset_activations_tokens(model, dataset_token)[1][location]
    # Looking at the final activations! Might want to change this?
    final_activations = activations[:,-1,:]

    # target_X = reshape_activations(activations)

    # Convert the tensor to a numpy array as scikit-learn works with numpy arrays
    data_numpy = final_activations.cpu().numpy()
    activation_dict[name] = data_numpy

  all_data = []
  all_labels = []
  for label, data in activation_dict.items():
      all_data.append(data)
      all_labels.extend([label] * len(data))

  all_data = np.concatenate(all_data, axis=0)

  print("all data shape is: ", all_data.shape)

  if random:

    # Determine the number of data points per label (assuming all labels have the same number of data points)
    num_points_per_label = len(activation_dict[list(activation_dict.keys())[0]])

    # Calculate the mean and variance of the entire dataset
    mean = np.mean(all_data, axis=0)
    variance = np.var(all_data, axis=0)
    # Generate some random data
    # Will use the same mean and variance as the data
    # Generate synthetic data with the same variance
    if centre:
      synthetic_data = np.random.normal(loc=mean, scale=np.sqrt(variance), size=(num_points_per_label, all_data.shape[1]))
    else:
      zeros = 0 * mean
      synthetic_data = np.random.normal(loc=zeros, scale=np.sqrt(variance), size=(num_points_per_label, all_data.shape[1]))
    synthetic_labels = ['Synthetic'] * len(synthetic_data)

    # Concatenate the synthetic data with the original data
    all_data = np.concatenate([all_data, synthetic_data], axis=0)
    all_labels.extend(synthetic_labels)



  # Initialize the t-SNE
  tsne = TSNE(n_components=2, random_state=21)

  # Fit and transform the data to 2D
  data_2d = tsne.fit_transform(all_data)

  data_2d_copy, all_labels_copy = data_2d, all_labels


  # Shuffle the data and labels
  data_2d, all_labels = shuffle(data_2d, all_labels, random_state=42)

  # Plot the transformed data
  # Create a colormap for labels
  unique_labels = list(set(all_labels))
  colors = plt.cm.get_cmap('viridis', len(unique_labels))

  # Plot the transformed data with labels
  plt.figure(figsize=(6, 5))
  markers = ['o', 's', '^', 'D', 'P']

  # Plot all points together, coloring them based on their labels
  for i, label in enumerate(all_labels):
      color_idx = unique_labels.index(label)
      marker = markers[color_idx % len(markers)]
      plt.scatter(data_2d[i, 0], data_2d[i, 1], color=colors(color_idx), s=20, alpha=0.6, label=label if color_idx not in [unique_labels.index(lbl) for lbl in all_labels[:i]] else "", marker=marker)

  # Add a legend
  handles, labels = plt.gca().get_legend_handles_labels()
  plt.legend(handles, labels, title="Labels")

  plt.xlabel('t-SNE feature 0')
  plt.ylabel('t-SNE feature 1')
  plt.title(f't-SNE visualization of the final activations of {model.name} at {location}')
  plt.show()
  return data_2d_copy, all_labels_copy