# Import libraries
import os
from glob import glob

import git
import pandas as pd
from torch.utils.data import Dataset

# Root directory of the project
ROOT_DIR = git.Repo(".", search_parent_directories=True).working_tree_dir


# UrbanSound8k Dataset Class
class urbansound8k_Dataset(Dataset):
    """
    Dataset class for the UrbanSound8k dataset.
    """

    def __init__(self, **kwargs):
        super(urbansound8k_Dataset, self).__init__()

        # See if the folder path is provided
        if kwargs["folder_path"] is not None:
            self.dataset_folder_path = self.dataset_folder_path = os.path.join(
                ROOT_DIR, "data", "input", kwargs["folder_path"]
            )
        else:
            self.dataset_folder_path = os.path.join(
                ROOT_DIR, "data", "input", "urbansound8k", "audio"
            )

        # Get all audio files paths
        self.paths = glob(os.path.join(self.dataset_folder_path, "fold*/*.wav"))

        # Get the label map
        self.label_map = self.get_label_map()

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx):
        path = self.paths[idx]
        return path

    def get_fold_count(self):
        """
        Returns the number of folds in the dataset
        """
        return len(glob(self.audio_base_path + "fold*"))

    def get_label_map(self):
        """
        Returns the label map for the dataset
        """
        # Load the metadata
        labels_path = os.path.join(
            ROOT_DIR, "data", "input", "urbansound8k", "metadata", "UrbanSound8K.csv"
        )
        df = pd.read_csv(labels_path)

        # Change _ to space in the class column
        df["class_"] = df["class"].apply(lambda x: " ".join(x.split("_")))

        # Create a dictionary with the classID as the key and the class as the value
        label_map = dict(zip(list(df.classID), list(df.class_)))

        # Sort the dictionary by the classID
        label_map = dict(sorted(label_map.items()))

        return label_map
