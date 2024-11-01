"""
    Script for the Encoder
    ---
        Class Encoder contains the encoder achitecture which is based on an
        attention-based Transformer.
"""
import torch
import torch.nn as nn
import numpy as np
import math

from .tokenizer import *
from .utils.mask_utils import apply_mask
from .utils.modules import *


class Encoder(nn.Module):
    def __init__(self, num_patches,
                       dim_in,
                       kernel_size,
                       embed_dim,
                       embed_bias,
                       nhead,
                       num_layers,
                       mlp_ratio=4.0,
                       qkv_bias=True,
                       qk_scale=None,
                       drop_rate=0.0,
                       attn_drop_rate=0.0,
                       norm_layer=nn.LayerNorm,
                       jepa=False,
                       embed_activation=nn.GELU()):

        super().__init__()

        # Parameters
        self.embed_dim = embed_dim
        self.num_patches = num_patches
        self.activation = embed_activation if embed_activation else nn.GELU()

        # Building the tokenizer
        self.tokenizer = TS_Tokenizer(dim_in=dim_in,
                                      kernel_size=kernel_size,
                                      embed_dim=embed_dim,
                                      embed_bias=embed_bias,
                                      activation = self.activation)

        # Positional Encoder -- We use a Sin-Cos one (not learnable)
        self.pos_embed = nn.Parameter(torch.zeros(1, self.num_patches,
                                         self.embed_dim), requires_grad=False)
        self.init_embed()

        # Transformer part of the encoder
        self.predictor_blocks = nn.ModuleList([
            Block(
                dim=embed_dim,
                num_heads=nhead,
                mlp_ratio=mlp_ratio,
                qkv_bias=qkv_bias,
                qk_scale=qk_scale,
                drop=drop_rate,
                attn_drop=attn_drop_rate,
                act_layer=nn.GELU,
                norm_layer=norm_layer
                )
            for i in range(num_layers)])

        self.encoder_norm = nn.LayerNorm(embed_dim)
        self.jepa = jepa

    def forward(self, x, mask=None):
        batch_size, num_patches, patch_length = x.size()

        # Embed the data using the Tokenizer
        x = self.tokenizer(x)

        # Add positional encoding
        pos_embs = self.pos_embed.repeat(batch_size, 1, 1)[:,:num_patches,:]
        x = x + pos_embs

        # Apply mask -- In the encoder, we keep only the unmasked part
        if mask is not None and self.jepa:
            x = apply_mask(x, mask)

        # Encode using Attention
        for blk in self.predictor_blocks:
            x = blk(x, mask=None)

        x = self.encoder_norm(x)
        return x

    def init_embed(self):
        """
            This function serves for the positional encoder which is based on a
            Sin-Cos Pos Encoder.
            ---
            Users can choose any other Positional Encoder that they may see fit.
        """
        assert self.embed_dim % 2 == 0

        omega = np.arange(self.embed_dim // 2, dtype=float)
        omega /= self.embed_dim / 2.
        omega = 1. / 10000**omega

        pos = np.arange(self.num_patches, dtype=float)
        pos = pos.reshape(-1)
        out = np.einsum('m,d->md', pos, omega)

        emb_sin = np.sin(out)
        emb_cos = np.cos(out)

        emb = np.concatenate([emb_sin, emb_cos], axis=1)

        self.pos_embed.data.copy_(torch.from_numpy(emb).float().unsqueeze(0))

        return emb
