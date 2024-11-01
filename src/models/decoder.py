"""
    Script for the Decoder
    ---
        Class Decoder contains the decoder achitecture which is based on a
        simple Linear Layer.
"""

import torch
import torch.nn as nn
import numpy as np
from .utils.modules import *

class LinearDecoder(nn.Module):
    def __init__(self, emb_dim, patch_size):
        super(LinearDecoder, self).__init__()
        self.fc = nn.Linear(emb_dim, patch_size)

    def forward(self, encoded_patch):
        return self.fc(encoded_patch)
