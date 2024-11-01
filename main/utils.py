"""
    Utils script
    ---
        - Function to init Encoder/decoder
        - Function to init the loader
"""

import torch
import argparse
import math

import random
import numpy as np

def prepare_args(config):

    seed = random.randint(0, 100)
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)

    parser = argparse.ArgumentParser()
    parser.add_argument('--data', type=str, default="uniform_uni_data")
    parser.add_argument('--notes', type=str, default="")
    parser.add_argument('--name_model', type=str, default="pre_train")

    parser.add_argument('--lr', type=float, default=config["lr"])
    parser.add_argument('--lr_pretrain', type=float, default=config["lr_pretrain"])
    parser.add_argument('--ratio_patches', type=int, default=config["ratio_patches"])
    parser.add_argument('--mask_ratio', type=float, default=config["mask_ratio"])
    parser.add_argument('--pooling', type=str, default=config["pooling"])
    parser.add_argument('--pre_train_mask', type=bool, default=False)
    parser.add_argument('--batch_size', type=int, default=config["batch_size"])

    parser.add_argument('--checkpoint_to_use', type=int, default=config["checkpoint_to_use"])
    parser.add_argument('--ratio_supervision', type=float, default=config["ratio_supervision"])

    # Transformers parameters
    parser.add_argument('--embed_dim', type=int, default=config["embed_dim"])
    parser.add_argument('--nhead', type=int, default=config["nhead"])
    parser.add_argument('--num_layers', type=int, default=config["num_layers"])
    parser.add_argument('--kernel_size', type=int, default=config["kernel_size"])
    parser.add_argument('--transformer_dense_dim', type=int, default=config["transformer_dense_dim"])

    # CNN parameters
    parser.add_argument('--cnn_out_channels', type=int, default=config["cnn_out_channels"])
    parser.add_argument('--cnn_kernel_size', type=int, default=config["cnn_kernel_size"])
    parser.add_argument('--cnn_dense_dim', type=int, default=config["cnn_dense_dim"])

    # Pretrain parameters
    parser.add_argument('--pretrain_encoder_embed_dim', type=int, default=config["pretrain_encoder_embed_dim"])
    parser.add_argument('--pretrain_encoder_nhead', type=int, default=config["pretrain_encoder_nhead"])
    parser.add_argument('--pretrain_encoder_num_layers', type=int, default=config["pretrain_encoder_num_layers"])
    parser.add_argument('--pretrain_encoder_kernel_size', type=int, default=config["pretrain_encoder_kernel_size"])
    parser.add_argument('--pretrain_transformer_dense_dim', type=int, default=config["pretrain_transformer_dense_dim"])
    parser.add_argument('--ema_pretrain', type=float, default=config["ema_pretrain"])

    parser.add_argument('--pretrain_decoder_embed_dim', type=int, default=config["pretrain_decoder_embed_dim"])
    parser.add_argument('--pretrain_decoder_nhead', type=int, default=config["pretrain_decoder_nhead"])
    parser.add_argument('--pretrain_decoder_num_layers', type=int, default=config["pretrain_decoder_num_layers"])

    args = parser.parse_args()
    config["model"] = args.name_model.lower()
    config["data"] = args.data
    config["lr"] = args.lr
    config["batch_size"] = args.batch_size
    config["lr_pretrain"] = args.lr_pretrain
    config["seed"] = seed
    config["ratio_patches"] = args.ratio_patches
    config["checkpoint_to_use"] = args.checkpoint_to_use


    config["path_data"] = "./data/" + args.data + "/" + args.data + ".csv"


    config["mask_ratio"] = args.mask_ratio
    config["pre_train_mask"] = args.pre_train_mask
    config["ratio_supervision"] = args.ratio_supervision
    config["pooling"] = args.pooling

    config["pretrain_transformer_dense_dim"] = args.transformer_dense_dim

    # Transformer parameters
    config["embed_dim"] = args.embed_dim
    config["nhead"] = args.nhead
    config["num_layers"] = args.num_layers
    config["kernel_size"] = args.kernel_size
    config["transformer_dense_dim"] = args.transformer_dense_dim


    # CNN parameters
    config["cnn_out_channels"] = args.cnn_out_channels
    config["cnn_kernel_size"] = args.cnn_kernel_size
    config["cnn_dense_dim"] = args.cnn_dense_dim

    # Pretrained encoder
    config["pretrain_encoder_embed_dim"] = args.pretrain_encoder_embed_dim
    config["pretrain_encoder_nhead"] = args.pretrain_encoder_nhead
    config["pretrain_encoder_num_layers"] = args.pretrain_encoder_num_layers
    config["pretrain_encoder_kernel_size"] = args.pretrain_encoder_kernel_size
    config["pretrain_transformer_dense_dim"] = args.pretrain_transformer_dense_dim

    config["ema_pretrain"] = args.ema_pretrain

    # Pretrained decoder
    config["pretrain_decoder_embed_dim"] = args.pretrain_decoder_embed_dim
    config["pretrain_decoder_nhead"] = args.pretrain_decoder_nhead
    config["pretrain_decoder_num_layers"] = args.pretrain_decoder_num_layers

    config["notes"] = args.notes

    return config


def prepare_args_pretrain(config):

    seed = random.randint(0, 100)
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)

    parser = argparse.ArgumentParser()
    parser.add_argument('--data', type=str, default="weather")

    parser.add_argument('--mask_ratio', type=float, default=config["mask_ratio"])
    parser.add_argument('--batch_size', type=int, default=config["batch_size"])
    parser.add_argument('--lr', type=float, default=config["lr"])
    parser.add_argument('--ema_momentum', type=float, default=config["ema_momentum"])
    parser.add_argument('--ratio_patches', type=int, default=config["ratio_patches"])
    parser.add_argument('--notes', type=str, default="")

    # Encoder
    parser.add_argument('--encoder_embed_dim', type=int, default=config["encoder_embed_dim"])
    parser.add_argument('--encoder_nhead', type=int, default=config["encoder_nhead"])
    parser.add_argument('--encoder_num_layers', type=int, default=config["encoder_num_layers"])
    parser.add_argument('--encoder_kernel_size', type=int, default=config["encoder_kernel_size"])

    # predictor
    parser.add_argument('--predictor_embed', type=int, default=config["predictor_embed"])
    parser.add_argument('--predictor_nhead', type=int, default=config["predictor_nhead"])
    parser.add_argument('--predictor_num_layers', type=int, default=config["predictor_num_layers"])

    args = parser.parse_args()

    config["mask_ratio"] = args.mask_ratio
    config["lr"] = args.lr
    config["batch_size"] = args.batch_size
    config["ema_momentum"] = args.ema_momentum
    config["ratio_patches"] = args.ratio_patches
    config["data"] = args.data

    config["seed"] = seed

    config["path_data"] = "./data/" + args.data + "/" + args.data + ".csv"

    config["notes"] = args.notes

    config["wandb_project_name"] = args.data + "_pretrain"


    # Encoder
    config["encoder_embed_dim"] = args.encoder_embed_dim
    config["encoder_nhead"] = args.encoder_nhead
    config["encoder_num_layers"] = args.encoder_num_layers
    config["encoder_kernel_size"] = args.encoder_kernel_size

    # predictor
    config["predictor_embed"] = args.predictor_embed
    config["predictor_nhead"] = args.predictor_nhead
    config["predictor_num_layers"] = args.predictor_num_layers

    config["path_save"] = "./logs/output_model/" + args.data \
        + "/lr_" + str(config["lr"]) \
        + "_ema_momentum_" + str(config["ema_momentum"]) \
        + "_mask_ratio_" + str(config["mask_ratio"]) \
        + "_ratio_patches_" + str(config["ratio_patches"]) \
        + "_encoder_" + str(config["encoder_embed_dim"]) + "_" + str(config["encoder_nhead"]) + "_" + str(config["encoder_num_layers"]) \
        + "_predictor_" + str(config["predictor_embed"]) + "_" + str(config["predictor_nhead"]) + "_" + str(config["predictor_num_layers"])

    return config




# Init stuff

def _no_grad_trunc_normal_(tensor, mean, std, a, b):
    # Cut & paste from PyTorch official master until it's in a few official releases - RW
    # Method based on https://people.sc.fsu.edu/~jburkardt/presentations/truncated_normal.pdf
    def norm_cdf(x):
        # Computes standard normal cumulative distribution function
        return (1. + math.erf(x / math.sqrt(2.))) / 2.

    with torch.no_grad():
        # Values are generated by using a truncated uniform distribution and
        # then using the inverse CDF for the normal distribution.
        # Get upper and lower cdf values
        l = norm_cdf((a - mean) / std)
        u = norm_cdf((b - mean) / std)

        # Uniformly fill tensor with values from [l, u], then translate to
        # [2l-1, 2u-1].
        tensor.uniform_(2 * l - 1, 2 * u - 1)

        # Use inverse cdf transform for normal distribution to get truncated
        # standard normal
        tensor.erfinv_()

        # Transform to proper mean, std
        tensor.mul_(std * math.sqrt(2.))
        tensor.add_(mean)

        # Clamp to ensure it's in the proper range
        tensor.clamp_(min=a, max=b)
        return tensor


def trunc_normal_(tensor, mean=0., std=1., a=-2., b=2.):
    # type: (Tensor, float, float, float, float) -> Tensor
    return _no_grad_trunc_normal_(tensor, mean, std, a, b)

def init_weights(m):
    if isinstance(m, torch.nn.Linear):
        trunc_normal_(m.weight, std=0.02)
        if m.bias is not None:
            torch.nn.init.constant_(m.bias, 0)
    elif isinstance(m, torch.nn.LayerNorm):
        torch.nn.init.constant_(m.bias, 0)
        torch.nn.init.constant_(m.weight, 1.0)


def grad_logger(model):
    count_non_grad = 0
    count_grad = 0
    grad_norm = 0

    for n, p in model.named_parameters():
        if (p.grad is not None) and not (n.endswith('.bias')):
            grad_norm += float(torch.norm(p.grad.data))
            count_grad += 1
        if p.grad is None and not (n.endswith('.bias')):
            count_non_grad += 1


    return grad_norm / count_grad, count_grad, count_non_grad

def _reduce(metric, reduction="mean", axis=None):
    if reduction == "mean":
        return np.nanmean(metric, axis=axis)
    elif reduction == "sum":
        return np.nansum(metric, axis=axis)
    elif reduction == "none":
        return metric


def mse(
    y: np.ndarray,
    y_hat: np.ndarray,
    reduction: str = "mean",
    axis = None,):
    delta_y = np.square(y - y_hat)
    return _reduce(delta_y, reduction=reduction, axis=axis)


def mae(
    y: np.ndarray,
    y_hat: np.ndarray,
    reduction: str = "mean",
    axis= None):
    delta_y = np.abs(y - y_hat)
    return _reduce(delta_y, reduction=reduction, axis=axis)




if __name__ == "__main__":
    pass
