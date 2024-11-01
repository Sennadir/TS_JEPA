"""
    Utils script for the eval of the model
    ---
        - Function to init Classifier/regressor in the case of pre-train eval
        - Function to init CNN
"""

import torch
from src.models.encoder import Encoder
from src.models.transformer_based import Transformer_based
# from src.models.attention_only import Attention_Classifier
from src.models.cnn import CNN_classifier

from src.data_loaders.data_loader_classification import get_eval_loaders

def init_eval_pretrain(config, regression=True):
    # Load Data
    print('Loading Data ..')
    train_loader, test_loader = get_eval_loaders(config["path_train"], 
                                                 config["path_test"],
                                                 config["batch_size"],
                                                 config["ratio_patches"],
                                                 ratio_supervision=config["ratio_supervision"],
                                                 mask=config["pre_train_mask"],
                                                 mask_ratio=config["mask_ratio"],
                                                 regression=config["regression"])

    input_dim = len(train_loader.dataset[0][0][0])
    num_patch = len(train_loader.dataset[0][0])

    if config["regression"]:
        n_classes = train_loader.dataset.label_length
    else:
        n_classes = train_loader.dataset.n_classes

    # Define the models (Encoder and Predictor)
    encoder = Encoder(num_patches=len(train_loader.dataset[0][0]),
                      dim_in=input_dim,
                      kernel_size=config["pretrain_encoder_kernel_size"],
                      embed_dim=config["pretrain_encoder_embed_dim"],
                      embed_bias=config["pretrain_encoder_embed_bias"],
                      nhead=config["pretrain_encoder_nhead"],
                      num_layers=config["pretrain_encoder_num_layers"])
    # Define the models (Encoder and Predictor)

    path_name = "batch_size_" + str(config["batch_size_pretrain"]) + "_lr_" + str(config["lr_pretrain"]) + "_ema_momentum_" + str(config["ema_pretrain"]) + "epoch_" + str(config["checkpoint_to_use"]) + "_ratio_mask_" + str(config["mask_ratio"])


    path_name = "/lr_" + str(config["lr_pretrain"]) \
        + "_ema_momentum_" + str(config["ema_pretrain"]) \
        + "_mask_ratio_" + str(config["mask_ratio"]) \
        + "_ratio_patches_" + str(config["ratio_patches"]) \
        + "_encoder_" + str(config["pretrain_encoder_embed_dim"]) + "_" + str(config["pretrain_encoder_nhead"]) + "_" + str(config["pretrain_encoder_num_layers"]) \
        + "_predictor_" + str(config["pretrain_decoder_embed_dim"]) + "_" + str(config["pretrain_decoder_nhead"]) + "_" + str(config["pretrain_decoder_num_layers"]) \
        + "_epoch_" + str(config["checkpoint_to_use"])
    
    if config["model"] == "pre_train":
        name_loader = torch.load(config["path_save"] + path_name + ".pt", map_location=torch.device('cpu'))["encoder"]
        encoder.load_state_dict(name_loader)

    classifier = Transformer_based(encoder,
                                embed_dim=config["pretrain_encoder_embed_dim"],
                                dense_dim=config["pretrain_transformer_dense_dim"],
                                patch_size=input_dim, 
                                num_patch=num_patch,
                                n_classes=n_classes,
                                # flatten=False,
                                pooling=config["pooling"],
                                regression=config["regression"],
                                pretrained=True)
    
    if config["pooling"] == "attention":
        param_groups = [
            {
                'params': (p for p in classifier.attention_pooling.parameters())
            }, {
                'params': (p for p in classifier.class_fc.parameters())
            }
        ]
    else:
        param_groups = [ {
                'params': (p for p in classifier.class_fc.parameters())
            }
        ]

    # optimizer = torch.optim.Adam(classifier.parameters(), lr=config["lr"])
    optimizer = torch.optim.Adam(param_groups, lr=config["lr"])
    
    #optimizer = torch.optim.Adam(classifier.class_fc.parameters(), lr=config["lr"])
    print('Start Training ..')
    return train_loader, test_loader, classifier, optimizer


def init_eval_transformer(config, regression=True, ratio_supervision=1):
    # Load Data
    train_loader, test_loader = get_eval_loaders(config["path_train"],
                                                config["path_test"],
                                                config["batch_size"], 
                                                config["ratio_patches"],
                                                ratio_supervision=ratio_supervision,
                                                regression=config["regression"])

    input_dim = len(train_loader.dataset[0][0][0])
    num_patch = len(train_loader.dataset[0][0])

    if config["regression"]:
        n_classes = train_loader.dataset.label_length
    else:
        n_classes = train_loader.dataset.n_classes

    # Define the models (Encoder and Predictor)
    encoder = Encoder(num_patches=len(train_loader.dataset[0][0]),
                      dim_in=input_dim,
                      kernel_size=config["kernel_size"],
                      embed_dim=config["embed_dim"],
                      embed_bias=config["embed_bias"],
                      nhead= config["nhead"],
                      num_layers=config["num_layers"])

    # Define the models
    classifier = Transformer_based(encoder, embed_dim=config["embed_dim"],
                                            dense_dim=config["transformer_dense_dim"],
                                            patch_size=input_dim,
                                            num_patch=num_patch,
                                            n_classes=n_classes,
                                            pooling=config["pooling"],
                                            regression=config["regression"])

    optimizer = torch.optim.Adam(classifier.parameters(), lr=config["lr"])

    return train_loader, test_loader, classifier, optimizer


def init_eval_cnn(config, ratio_supervision=1):
    # Load Data
    train_loader, test_loader = get_eval_loaders(config["path_train"],
                                                 config["path_test"],
                                                 config["batch_size"],
                                                 config["ratio_patches"],
                                                 ratio_supervision=ratio_supervision,
                                                 transform_to_patch=False,
                                                 regression=config["regression"])

    input_dim = 1
    input_length = len(train_loader.dataset[0][0])

    if config["regression"]:
        n_classes = train_loader.dataset.label_length
    else:
        n_classes = train_loader.dataset.n_classes

    # Define the model and optimizer
    classifier = CNN_classifier(input_dim,
                                out_channels=config["cnn_out_channels"],
                                kernel_size=config["cnn_kernel_size"],
                                output_dim=n_classes,
                                dense_dim=config["cnn_dense_dim"],
                                input_length=input_length,
                                regression=config["regression"])


    optimizer = torch.optim.Adam(classifier.parameters(), lr=config["lr"])
    return train_loader, test_loader, classifier, optimizer


def init_eval_finetune(config, regression=True, ratio_supervision=1):
    # Load Data
    print('Loading Data ..')
    train_loader, test_loader = get_eval_loaders(config["path_train"], 
                                                 config["path_test"],
                                                 config["batch_size"],
                                                 config["ratio_patches"],
                                                 ratio_supervision=ratio_supervision,
                                                 mask=config["pre_train_mask"],
                                                 mask_ratio=config["mask_ratio"],
                                                 regression=config["regression"])

    input_dim = len(train_loader.dataset[0][0][0])
    num_patch = len(train_loader.dataset[0][0])

    if config["regression"]:
        n_classes = train_loader.dataset.label_length
    else:
        n_classes = train_loader.dataset.n_classes

    # Define the models (Encoder and Predictor)
    encoder = Encoder(num_patches=len(train_loader.dataset[0][0]),
                      dim_in=input_dim,
                      kernel_size=config["pretrain_encoder_kernel_size"],
                      embed_dim=config["pretrain_encoder_embed_dim"],
                      embed_bias=config["pretrain_encoder_embed_bias"],
                      nhead=config["pretrain_encoder_nhead"],
                      num_layers=config["pretrain_encoder_num_layers"])
    # Define the models (Encoder and Predictor)

    path_name = "batch_size_" + str(config["batch_size_pretrain"]) + "_lr_" + str(config["lr_pretrain"]) + "_ema_momentum_" + str(config["ema_pretrain"]) + "epoch_" + str(config["checkpoint_to_use"]) + "_ratio_mask_" + str(config["mask_ratio"])


    path_name = "/lr_" + str(config["lr_pretrain"]) \
        + "_ema_momentum_" + str(config["ema_pretrain"]) \
        + "_mask_ratio_" + str(config["mask_ratio"]) \
        + "_ratio_patches_" + str(config["ratio_patches"]) \
        + "_encoder_" + str(config["pretrain_encoder_embed_dim"]) + "_" + str(config["pretrain_encoder_nhead"]) + "_" + str(config["pretrain_encoder_num_layers"]) \
        + "_predictor_" + str(config["pretrain_decoder_embed_dim"]) + "_" + str(config["pretrain_decoder_nhead"]) + "_" + str(config["pretrain_decoder_num_layers"]) \
        + "_epoch_" + str(config["checkpoint_to_use"])
    
    if config["model"] == "pre_train":
        name_loader = torch.load(config["path_save"] + path_name + ".pt", map_location=torch.device('cpu'))["encoder"]
        encoder.load_state_dict(name_loader)

    classifier = Transformer_based(encoder,
                                embed_dim=config["pretrain_encoder_embed_dim"],
                                dense_dim=config["pretrain_transformer_dense_dim"],
                                patch_size=input_dim, 
                                num_patch=num_patch,
                                n_classes=n_classes,
                                # flatten=False,
                                pooling=config["pooling"],
                                regression=config["regression"],
                                pretrained=False)
    

    optimizer = torch.optim.Adam(classifier.parameters(), lr=config["lr"])
    
    #optimizer = torch.optim.Adam(classifier.class_fc.parameters(), lr=config["lr"])
    print('Start Training ..')
    return train_loader, test_loader, classifier, optimizer
