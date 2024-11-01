"""
    Script for the Data Loaders
    ---
        Function "get_data" read the data from the path.
"""

from torch.utils.data import DataLoader
from .data_class import CSVDataLoader, EvaluationDataLoader
import pandas as pd
import gzip
import numpy as np

import torch

def get_jepa_loaders(path,
                     batch_size,
                     ratio_patches=10,
                     series_split_size=320,
                     mask_ratio=0.9):
    """
        Load and prepare the data to be used with the TS-JEPA
    """

    dataloader = CSVDataLoader(path_data=path,
                               series_split_size=series_split_size,
                               patch_size=32,
                               mask_ratio=mask_ratio)
    dataloader = DataLoader(dataloader,
                            batch_size=batch_size,
                            shuffle=True)

    return dataloader


def get_evaluation_loaders(path,
                           batch_size,
                           ratio_patches=10,
                           mask_ratio=0.9):
    """
        Load and prepare the data to be used for the downstream tasks
    """

    dataloader = EvaluationDataLoader(path_data=path,
                                      patch_size=32,
                                      context_size=10)
    dataloader = DataLoader(dataloader, batch_size=batch_size, shuffle=False)

    return dataloader



if __name__ == "__main__":
    pass
