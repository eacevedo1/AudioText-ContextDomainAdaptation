# Import libraries
import os
import sys

import git
import torch

# Get the root directory of the project
ROOT_DIR = git.Repo(".", search_parent_directories=True).working_tree_dir
sys.path.append(ROOT_DIR)

import src.get_models as get_models


def get_background_profile_text(bg_type, text_features):
    """
    Get the background profile embeddings for domain adaptation

    Args:
        bg_type: string containing the background type

    Returns:
        bg_embd: tensor containing the background profile embeddings
    """

    # Load the model
    get_model = getattr(get_models, "get_LAIONCLAP_model")
    model = get_model()

    # Create the prompt text
    prompts = [
        "This is a sound of " + bg_type,
        bg_type.capitalize() + " sounds in the background",
        "This is a sound of " + bg_type + " in the background",
    ]

    # Get the text embeddings
    bg_embd = model.get_text_embedding(prompts)

    # Get the background profile embeddings
    background_profiles = (torch.tensor(bg_embd) @ text_features.t()).detach().cpu()

    # Take the mean of the background profile embeddings
    background_profile = background_profiles.mean(dim=0)

    return background_profile
