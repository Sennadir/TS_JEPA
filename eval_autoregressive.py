"""
    Script to run the long-term forecasting task.
    ---
        We consider the horizon and then predict the next value, increment the
        horizon, then predict the new value in a loop way.
"""

from config.config_downstream import config

import torch
import json
import copy
import logging
import argparse
import pickle

from main.utils import prepare_args
from main.utils import mse, mae, _reduce

import numpy as np
import random

from src.data_loaders.data_loader import get_jepa_loaders, get_evaluation_loaders
from src.models.encoder import Encoder
from src.models.decoder import LinearDecoder

import warnings

warnings.filterwarnings("ignore")


if __name__ == "__main__":
    # Parse the args and get the config setup
    config = prepare_args(config)

    # Define some parameters
    num_epochs = 500
    context = 32
    num_patches = 10

    # Load device
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # Init Encoder, Decoder, Optimizer

    # Load Data
    print("Load data")

    config["path_data"] = "./data/" + "weather" + "/" + "weather" + ".csv"

    loader = get_evaluation_loaders(
        config["path_data"],
        config["batch_size"],
        config["ratio_patches"],
        config["mask_ratio"],
    )

    input_dim = len(loader.dataset[0][0][0])
    # Encoder
    encoder = Encoder(
        num_patches=len(loader.dataset[0][0]),
        dim_in=input_dim,
        kernel_size=config["pretrain_encoder_kernel_size"],
        embed_dim=config["pretrain_encoder_embed_dim"],
        embed_bias=config["pretrain_encoder_embed_bias"],
        nhead=config["pretrain_encoder_nhead"],
        num_layers=config["pretrain_encoder_num_layers"],
        jepa=True,
    )
    # Load Decoder
    decoder = LinearDecoder(emb_dim=config["pretrain_encoder_embed_dim"], patch_size=32)

    path_name = (
        "/Auto_regressive_lr_"
        + str(config["lr_pretrain"])
        + "_encoder_"
        + str(config["pretrain_encoder_embed_dim"])
        + "_"
        + str(config["pretrain_encoder_nhead"])
        + "_"
        + str(config["pretrain_encoder_num_layers"])
        + "_epoch_"
        + str(config["checkpoint_to_use"])
    )

    # if config["model"] == "pre_train":
    name_loader = torch.load(
        config["path_save"] + path_name + ".pt", map_location=torch.device("cpu")
    )["encoder"]
    encoder.load_state_dict(name_loader)
    print("Model loaded")

    # We consider training only the decoder head
    param_groups = [{"params": (p for n, p in decoder.named_parameters())}]

    optimizer = torch.optim.AdamW(param_groups, lr=config["lr"])

    # We train the model on the train set
    print("start train")
    for epoch in range(num_epochs):
        encoder.eval()
        decoder.train()
        total_loss = 0
        for context_patches, target_patch in loader:
            optimizer.zero_grad()
            encoded_patches = encoder(context_patches)
            summed_embedding = torch.sum(encoded_patches, dim=1)
            predicted_next_patch = decoder(summed_embedding)
            # loss = criterion(predicted_next_patch, patches_tensor[i+1])
            loss = torch.nn.functional.mse_loss(
                predicted_next_patch, target_patch, reduction="mean"
            )

            loss.backward()
            optimizer.step()

            total_loss += loss / config["batch_size"]
        if epoch % 10 == 0:
            print("Epoch: {} - Total loss: {}".format(epoch, total_loss))

    # We test the model on the last prediction
    # We define the number of steps we will have
    num_steps = (len(loader.dataset.test_df) - context * num_patches) // context

    predictions = []
    current_context = (
        loader.dataset.test_df[: context * num_patches]
        .reshape(num_patches, context)
        .unsqueeze(0)
    )

    for step in range(num_steps):
        # Encode the current context patches
        encoded_patches = encoder(current_context)

        # Sum the embeddings of the context patches
        summed_embedding = torch.sum(encoded_patches, dim=1)

        # Predict the next patch using the decoder
        predicted_next_patch = decoder(summed_embedding)

        real_value = loader.dataset.test_df[
            context * num_patches
            + step * context : context * num_patches
            + (step + 1) * context
        ]
        # Store the prediction
        predictions.extend(predicted_next_patch.flatten())

        # Update the context: Remove the oldest patch and append the predicted patch
        current_context = torch.cat(
            [current_context[:, 1:], predicted_next_patch.unsqueeze(0)], dim=1
        )

    real_values = loader.dataset.test_df[context * num_patches :]
    len_test = len(real_values)
    predictions = torch.tensor(predictions[:len_test])

    loss_test = mse(real_values.numpy(), predictions.numpy())
    print("loss is: {}".format(loss_test))
