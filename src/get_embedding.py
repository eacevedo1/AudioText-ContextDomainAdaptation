# Import Libraries
import sys

import git
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

# Get the root directory of the project
ROOT_DIR = git.Repo(".", search_parent_directories=True).working_tree_dir
sys.path.append(ROOT_DIR)

import src.get_models as get_models


def get_urbansound8k_embeddings(train_set, feat_data, args):
    """
    Returns the embeddings for the UrbanSound8k dataset
    """
    # Load the dataset
    train_dataloader = DataLoader(
        train_set, batch_size=64, shuffle=False, num_workers=args.num_workers
    )

    # Load the model
    get_model = getattr(get_models, f"get_LAIONCLAP_model")
    model = get_model()

    # Iterate over the dataset
    for paths in tqdm(train_dataloader):
        audio_embd = None

        # Get the embeddings of a batch of audio files
        audio_embd = model.get_audio_embedding_from_filelist(paths)

        # Store the embeddings and metadata
        for idx, embd in enumerate(audio_embd):
            # Get the file name, label and fold of each audio file of the batch
            path = paths[idx]
            file_name, label, fold = None, None, None

            file_name = path.split("/")[-1]
            label = int(file_name.split(".")[0].split("-")[1])
            fold = int(path.split("/")[-2].split("fold")[1])

            # Store the label, embeddings and fold
            feat_data[file_name] = {}
            feat_data[file_name]["class_gt"] = label
            feat_data[file_name]["embd"] = embd
            feat_data[file_name]["fold"] = fold

    return feat_data
