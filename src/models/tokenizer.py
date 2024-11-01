"""
    Script for the Tokenizer.
    Takes a batch of patches and generates the tokens.
    ---
        Class TS_Tokenizer contains the tokenization scheme.
"""

import torch
import torch.nn as nn


class TS_Tokenizer(nn.Module):
    """
        Tokenizer class based on a Conv1D.
        ---
            dim_in (int): Input Dimension of the patch
            kernel_size (int): Kernel size to be used for the Conv
            embed_dim (int): Output of the Tokenizer size, which is the input to
                             the Encoder.
    """
    def __init__(self, dim_in, kernel_size, embed_dim, embed_bias, activation=nn.GELU()):
        super().__init__()

        self.embed_dim = embed_dim
        self.kernel_size = kernel_size
        self.proj = nn.Conv1d(in_channels=1,
                              out_channels=embed_dim,
                              kernel_size=self.kernel_size,
                              stride=self.kernel_size,
                              padding=0)

        # Calculate the output length after the first conv
        conv_output_length = dim_in
        conv_output_length = (conv_output_length - self.kernel_size) // self.kernel_size + 1

        # Linear layer to adapt the last flattened output to the embedding dimension
        self.fc = nn.Linear(embed_dim * conv_output_length, embed_dim)


    def forward(self,x):
        batch_size, num_patches, length_patch = x.shape
        x = x.view(-1, 1, length_patch)

        # Apply the Conv
        x = self.proj(x)

        # Flatten the output
        x = x.view(x.size(0), -1)

        # Linear layer to transform to embedding dimension needed for Encoder
        x = self.fc(x)
        x = x.view(batch_size, num_patches, -1)

        return x
