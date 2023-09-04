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

from sklearn.metrics import roc_curve, auc


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

    # Get the mean vector (to use for different types of inner products)
    # TODO: stop this from repeating computations
    mean_vector = transformer_activation_exploration.comparing_centres.find_activations_centre(
        model,
        baseline_dataset,
        location,
        2,
        use_all_activations
    )

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
        elif inner_product_type == 'centered_dot':
            inner_products[label] = t.einsum(
                'ij,j->i',
                activations - mean_vector,
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

    # Create ROC curve
    # Merge "positive" and "negative" examples
    y_true = np.concatenate(
        [
            np.ones_like(inner_products['positive'].cpu().numpy()),
            np.zeros_like(inner_products['negative'].cpu().numpy())
        ]
    )
    y_score = np.concatenate(
        [
            inner_products['positive'].cpu().numpy(),
            inner_products['negative'].cpu().numpy()
        ]
    )
    fpr, tpr, _ = roc_curve(y_true, y_score)
    roc_auc = auc(fpr, tpr)
    print(f"ROC AUC: {roc_auc:.2f}")
    # Plot ROC curve
    plt.plot(fpr, tpr, label=f'ROC curve (area = {roc_auc:.2f})')

    return fpr, tpr


def find_optimal_threshold(true_values, scores):
    """
    Find the threshold that maximizes classification accuracy.
    
    Parameters:
    - true_values (numpy array): Ground truth labels (1 for positive class, 0 for negative class).
    - scores (numpy array): Scores or probabilities for each datapoint.
    
    Returns:
    - optimal_threshold (float): The threshold value that maximizes accuracy.
    - max_accuracy (float): The maximum achieved accuracy.
    """
    
    # Ensure the arrays have the same shape
    assert true_values.shape == scores.shape, "Input arrays must have the same shape."

    # Get unique scores as potential thresholds
    potential_thresholds = np.unique(scores)

    max_accuracy = 0
    optimal_threshold = None

    for threshold in potential_thresholds:
        # Predict 1 if score is greater than or equal to threshold, else 0
        predictions = (scores >= threshold).astype(int)
        
        # Calculate accuracy for this threshold
        accuracy = np.mean(predictions == true_values)
        
        # Update max_accuracy and optimal_threshold if necessary
        if accuracy > max_accuracy:
            max_accuracy = accuracy
            optimal_threshold = threshold

    return optimal_threshold, max_accuracy



def feature_detection_classifier(
    model: HookedTransformer,
    location: str,
    training_dataset: Dict[str, List[str]],
    baseline_dataset: List[str],
    use_all_activations: bool,
    threshold_dataset: Dict[str, List[str]],
    evaluation_dataset: Dict[str, List[str]],
):
    """
    Use AtDotLLM to classify text based on inner product with a feature vector.
    Split the training dataset: use one half of the positive examples to find a feature vector;
    use the second half to determine the threshold value which maximises accuracy.
    Then evaluate the evaluation dataset using the feature vector
    """

    # Get the feature vector from the positive examples of the training dataset
    feature_vector = transformer_activation_exploration.comparing_centres.find_activations_centre_diff(
        model,
        training_dataset,
        baseline_dataset,
        location,
        2,
        use_all_activations
    )
    print("feature vector calculated")


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

    # Determine the threshold for the inner products
    # Use the threshold dataset to determine the threshold

    # Get the inner for the threshold dataset
    threshold_activations = {}
    inner_products = {}
    for label, dataset in threshold_dataset.items():
        # Note: we are only using the final activations here (regardless of arg!)
        threshold_activations[label] = transformer_activation_exploration.utils.dataset_activations_optimised_new(
            model,
            dataset,
            location,
            2,
            False
        )
        inner_products[label] = t.einsum(
            'ij,j->i',
            threshold_activations[label],
            feature_vector
        )
    print("part 3 done")
    # Determine the threshold which maximises accuracy

    y_true = np.concatenate(
        [
            np.ones_like(inner_products['positive'].cpu().numpy()),
            np.zeros_like(inner_products['negative'].cpu().numpy())
        ]
    )
    y_score = np.concatenate(
        [
            inner_products['positive'].cpu().numpy(),
            inner_products['negative'].cpu().numpy()
        ]
    )

    # Find the threshold which maximises accuracy

    threshold, max_accuracy = find_optimal_threshold(y_true, y_score)
    print("Optimal threshold:", threshold)
    print("Maximum accuracy:", max_accuracy)



    # Calculate the inner products between the feature vector and the activations
    inner_products_evaluations = {}
    for label, activations in evaluation_activations.items():
        inner_products_evaluations[label] = t.einsum(
            'ij,j->i',
            activations,
            feature_vector
        )



    print("part 4 done")
    # Do classification
    classification = {}
    # Values larger than 0 are classified as 1, values smaller than 0 are classified as 0
    for label, inner_product in inner_products_evaluations.items():
        classification[label] = t.where(inner_product > threshold, t.tensor(1), t.tensor(0))

    # Report accuracy
    # Get the true labels for the evaluation dataset
    true_labels = {}
    for label, dataset in evaluation_dataset.items():
        if label == "positive":
            true_labels[label] = np.ones(len(dataset))
        else:
            true_labels[label] = np.zeros(len(dataset))
    
    # Merge "positive" and "negative" examples
    y_true_eval = np.concatenate(
        [
            true_labels['positive'],
            true_labels['negative']
        ]
    )

    y_score_eval = np.concatenate(
        [
            classification['positive'].cpu().numpy(),
            classification['negative'].cpu().numpy()
        ]
    )

    # Calculate accuracy
    accuracy = np.mean(y_true_eval == y_score_eval)
    print(f"Accuracy: {accuracy:.2f}")

    return accuracy
    


    
