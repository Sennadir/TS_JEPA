"""
    Script to run the short-term forecasting task.
    ---
        We consider the horizon and then predict a single value.
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
    print('Load data')
    config["path_data"] = "./data/" + args.data + "/" + args.data + ".csv"

    loader = get_evaluation_loaders(config["path_data"],
                                        config["batch_size"],
                                        config["ratio_patches"],
                                        config["mask_ratio"])

    input_dim = len(loader.dataset[0][0][0])
    # Encoder
    encoder = Encoder(num_patches=len(loader.dataset[0][0]),
                        dim_in=input_dim,
                        kernel_size=config["pretrain_encoder_kernel_size"],
                        embed_dim=config["pretrain_encoder_embed_dim"],
                        embed_bias=config["pretrain_encoder_embed_bias"],
                        nhead=config["pretrain_encoder_nhead"],
                        num_layers=config["pretrain_encoder_num_layers"],
                        jepa=True
                     )

    decoder = LinearDecoder(emb_dim=config["pretrain_encoder_embed_dim"], patch_size=32)

    # Load the pretrained model
    path_name = "/Auto_regressive_lr_" + str(config["lr_pretrain"]) \
            + "_encoder_" + str(config["pretrain_encoder_embed_dim"]) + "_" \
            + str(config["pretrain_encoder_nhead"]) + "_" \
            + str(config["pretrain_encoder_num_layers"]) \
            + "_epoch_" + str(config["checkpoint_to_use"])

    name_loader = torch.load(config["path_save"] + path_name + ".pt",
                                map_location=torch.device('cpu'))["encoder"]
    encoder.load_state_dict(name_loader)
    print("Model loaded")

    # We consider training only the decoder head
    param_groups = [
        {'params' : (p for n, p in decoder.named_parameters())}]

    optimizer = torch.optim.AdamW(param_groups, lr=config["lr"])

    # We train the model on the train set
    print('start train')
    for epoch in range(num_epochs):
        encoder.eval()
        decoder.train()
        total_loss = 0
        for context_patches, target_patch in loader:
            optimizer.zero_grad()
            encoded_patches = encoder(context_patches)
            summed_embedding = torch.sum(encoded_patches, dim=1)
            predicted_next_patch = decoder(summed_embedding)
            loss = torch.nn.functional.mse_loss(predicted_next_patch,
                                            target_patch, reduction="mean")

            loss.backward()
            optimizer.step()

            total_loss += loss/config["batch_size"]
        if epoch % 10 == 0:
            print('Epoch: {} - Total loss: {}' .format(epoch, total_loss))


    # We test the model on the last prediction
    # We define the number of steps we will have
    num_steps = (len(loader.dataset.test_df[context * num_patches:])) // context

    predictions = []
    total_diff = 0
    l_val_mse = []
    l_val_mae = []
    for step in range(num_steps):
        # Encode the current context patches
        current_context = loader.dataset.test_df[context * step: context * num_patches + context * step].reshape(num_patches, context).unsqueeze(0)
        target_value = loader.dataset.test_df[context * num_patches + step * context: context * num_patches + (step + 1) * context]

        encoded_patches = encoder(current_context)

        # Sum the embeddings of the context patches
        summed_embedding = torch.sum(encoded_patches, dim=1)

        # Predict the next patch using the decoder
        predicted_next_patch = decoder(summed_embedding)

        # Compute the Loss
        val_mse = mse(predicted_next_patch.flatten().detach().numpy(), target_value.numpy())
        val_mae = mae(predicted_next_patch.flatten().detach().numpy(), target_value.numpy())

        l_val_mse.append(val_mse)
        l_val_mae.append(val_mae)


    print("MSE Loss is: {}" .format(np.mean(val_mse)))
    print("MAE Loss is: {}" .format(np.mean(l_val_mae)))
