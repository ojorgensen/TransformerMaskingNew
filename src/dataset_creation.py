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
import random
import time
import pickle

from transformer_lens.hook_points import HookPoint
from transformer_lens import utils, HookedTransformer, HookedTransformerConfig, FactoredMatrix, ActivationCache

import src.utils

# Saves computation time, since we don't need it for the contents of this notebook
t.set_grad_enabled(False)


def story_creator(
    number: int,
    seed: int
) -> str:
    """
    Create a story, which follows a given template and varies the names and final question
    based on the seed. The number controls some aspects of the story.
    """
    random.seed(seed)
    family_names = {
        "husband": ["Robert", "John", "William", "James", "Richard", "Charles", "Joseph", "Michael", "David", "Thomas"],
        "matriarch": ["Margaret", "Elizabeth", "Susan", "Mary", "Patricia", "Linda", "Barbara", "Jennifer", "Jessica", "Sarah"],
        "child": ["Emily", "Sophia", "Olivia", "Emma", "Ava", "Isabella", "Mia", "Abigail", "Madison", "Charlotte"],
        "cousin": ["Ethan", "Noah", "Liam", "Alexander", "Aiden", "Lucas", "Benjamin", "Mason", "Elijah", "Jackson"],
        "brother": ["Daniel", "Christopher", "Matthew", "Andrew", "Nicholas", "Joshua", "Anthony", "Ryan", "Brandon", "Kevin"]
    }

    selected_names = {
        key: random.choice(value)
        for key, value in family_names.items()
    }

    template = f"""
    The Thompson family was renowned for their artistry.
    {{matriarch}} was the matriarch.
    Her husband was {{husband}}.
    They had one child, {{child}}.
    {{matriarch}} had a cousin, {{cousin}}
    {{matriarch}}'s brother was called {{brother}}.

    The name of {{husband}}'s wife was {{matriarch}}.
    The name of {{matriarch}}'s {{relation}} was"""

    if number == 2 or number == 3:
        # Add a relation to selected names
        selected_names["relation"] = "brother"
    else:
        relations = ["brother", "husband", "cousin", "child"]
        selected_names["relation"] = random.choice(relations)

    # Fix the output name
    if number == 1 or number == 3:
        relation = selected_names["relation"]
        selected_names[relation] = "Taylor"

    # Format the template string with the selected names
    story = template.format(**selected_names)

    # Return the generated story
    return story


def story_creator_dataset(
    number: int,
    seed: int,
    size: int = 500
) -> List[str]:
    """
    Create a dataset of stories, which follows a given template and varies the names and final question
    based on the seed. The number controls some aspects of the story.
    """
    # Create a list of stories
    stories = [
        story_creator(number, seed + i)
        for i in range(size)
    ]

    # Return the generated stories
    return stories


def arithmetic_datasets(
    start: int,
    n: int,
    progression: Callable[[int], int],
    symbol: str,
) -> List[str]:
    """
    Create a dataset of arithmetic problems, which follows a given template and varies the numbers
    based on the progression function. The symbol controls the operation.
    """
    # Create a list of arithmetic problems
    problems = [
        f"{start + progression(i)} {symbol} {start + progression(j)} ="
        for i in range(n) for j in range(n)
    ]

    # Return the generated problems
    return problems


def generate_random_addition_dataset(
    sqrt_n: int
) -> t.Tensor:
    """
    Generate a random addition dataset of size sqrt_n^2.
    This will consist of sequences of length 5, where the first token is the EOS token,
    the second and fourth tokens are random tokens, the third token is the addition token,
    and the fifth token is the equal token.
    """
    # Randomly sample 25 tokens
    random_token_choice = random.sample(list(range(50256)), sqrt_n)

    # Get the token for " +" (this is token 1343)
    add_token = 1343
    equal_token = 796
    EOS_token = 50256

    # Create the addition dataset using these random tokens
    random_token_addition_list = [t.Tensor([[EOS_token, a, add_token, b, equal_token]]) for a in random_token_choice for b in random_token_choice]
    dataset_tensor = t.cat(random_token_addition_list, dim=0)
    dataset_tensor = dataset_tensor.int()

    return dataset_tensor

# Shuffle the addition and multiplication datasets
def shuffle_string(s):
    # Split the string based on whitespace
    chars = s.split()

    # Shuffle the characters
    random.shuffle(chars)

    # Reintroduce whitespace between each character
    shuffled = ' '.join(chars)

    return shuffled


if __name__ == "__main__":
    # Create datasets for numbers 1, 2, 3
    # for number in [1, 2, 3]:
    #     stories = story_creator_dataset(number, 0)
    #     with open(f"datasets/story_{number}.json", "w") as f:
    #         json.dump(stories, f)

    # Create some arithmetic datasets
    arithmetic_datasets_dict = {}
    arithmetic_datasets_tokens_dict = {}
    n = 24
    arithmetic_datasets_dict["addition"] = arithmetic_datasets(0, n, lambda i: i, "+")
    arithmetic_datasets_dict["odd addition"] = arithmetic_datasets(0, n, lambda i: 2*i + 1, "+")
    arithmetic_datasets_dict["even addition"] = arithmetic_datasets(0, n, lambda i: 2*i, "+")
    arithmetic_datasets_dict["* multiplication"] = arithmetic_datasets(0, n, lambda i: i, "*")
    arithmetic_datasets_dict["odd multiplication"] = arithmetic_datasets(0, n, lambda i: 2*i + 1, "*")
    arithmetic_datasets_dict["even multiplication"] = arithmetic_datasets(0, n, lambda i: 2*i, "*")
    arithmetic_datasets_dict["x multiplication"] = arithmetic_datasets(0, n, lambda i: i, "x")

    # Create a combined addition and multiplication dataset
    addition_dataset = arithmetic_datasets_dict["addition"]
    multiplication_dataset = arithmetic_datasets_dict["* multiplication"]
    random.shuffle(addition_dataset)
    random.shuffle(multiplication_dataset)
    combined_add_mult_dataset = addition_dataset + multiplication_dataset
    random.shuffle(combined_add_mult_dataset)
    combined_add_mult_dataset = combined_add_mult_dataset[:n**2]
    arithmetic_datasets_dict["combined add mult"] = combined_add_mult_dataset

    # Create a shuffled addition dataset
    shuffled_addition_dataset = [shuffle_string(s) for s in arithmetic_datasets_dict["addition"]]

    # Create a random addition dataset
    arithmetic_datasets_tokens_dict["random addition"] = generate_random_addition_dataset(n)

    # Save the datasets
    with open("datasets/arithmetic_datasets.json", "w") as f:
        json.dump(arithmetic_datasets_dict, f)
    with open("datasets/arithmetic_datasets_tokens.pkl", "wb") as f:
        pickle.dump(arithmetic_datasets_tokens_dict, f)
    