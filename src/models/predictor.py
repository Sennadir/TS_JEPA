"""
    Script for the Predictor on the JEPA architecture
    ---
        Class Predictor contains the architecture which is adaptable to the output
        of the JEPA encoder.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from .utils.modules import *

from .utils.mask_utils import apply_mask
import math


class Predictor(nn.Module):
    def __init__(
        self,
        num_patches,
        encoder_embed_dim=128,
        predictor_embed_dim=128,
        nhead=2,
        num_layers=1,
        mlp_ratio=4.0,
        qkv_bias=True,
        qk_scale=None,
        drop_rate=0.0,
        attn_drop_rate=0.0,
        norm_layer=nn.LayerNorm,
        embed_activation=nn.GELU(),
    ):
        super(Predictor, self).__init__()

        # Model's parameters
        self.activation = embed_activation if embed_activation else nn.GELU()
        self.predictor_embed_dim = predictor_embed_dim
        self.num_patches = num_patches

        # Map the Encoder's embed dim to the predictor's embed dim
        self.predictor_embed = nn.Linear(
            encoder_embed_dim, predictor_embed_dim, bias=True
        )

        # Positional Encoder -- Note that we are using a Sin-Cos PE
        self.pos_embed = nn.Parameter(
            torch.zeros(1, self.num_patches, self.predictor_embed_dim),
            requires_grad=False,
        )
        self.init_embed()

        # Mask tokens
        self.mask_token = nn.Parameter(torch.zeros(1, 1, predictor_embed_dim))

        # Transformer part of the Decoder
        self.predictor_blocks = nn.ModuleList(
            [
                Block(
                    dim=predictor_embed_dim,
                    num_heads=nhead,
                    mlp_ratio=mlp_ratio,
                    qkv_bias=qkv_bias,
                    qk_scale=qk_scale,
                    drop=drop_rate,
                    attn_drop=attn_drop_rate,
                    act_layer=nn.GELU,
                    norm_layer=norm_layer,
                )
                for i in range(num_layers)
            ]
        )

        # To Normalize and map back to the encoder dimension (before applying
        # the loss function)
        self.predictor_norm = nn.LayerNorm(predictor_embed_dim)
        self.predictor_proj = nn.Linear(
            predictor_embed_dim, encoder_embed_dim, bias=True
        )

    def forward(self, encoded_vals, mask=None, non_masks=None):

        assert (mask is not None) and (encoded_vals is not None), "No input found"

        _, ctx_size, _ = encoded_vals.size()
        batch_size, masked_num = mask.size()

        # Map the output of the encoder to the Predictor's dimension
        x = self.predictor_embed(encoded_vals)
        # Add PE and apply mask to keep only the non-masked part
        cnt_pos_enc = self.pos_embed.repeat(batch_size, 1, 1)
        cnt_pos_enc = apply_mask(cnt_pos_enc, non_masks)
        x = x + cnt_pos_enc

        # Create the Target vectors and add PE
        target_pos_enc = self.pos_embed.repeat(batch_size, 1, 1)
        target_pos_enc = apply_mask(target_pos_enc, mask)
        pred_tokens = self.mask_token.repeat(batch_size, target_pos_enc.size(1), 1)
        pred_tokens = pred_tokens + target_pos_enc

        # Concat the context (from the encoder) and the mask tokens
        x = torch.cat([x, pred_tokens], dim=1)

        # Push through attention
        for blk in self.predictor_blocks:
            x = blk(x, mask=None)

        x = self.predictor_norm(x)

        # Output only the part related to the masked area and adapt the dim
        x = x[:, ctx_size:]
        x = self.predictor_proj(x)

        return x

    def init_embed(self):
        """
        This function serves for the positional encoder which is based on a
        Sin-Cos Pos Encoder.
        ---
        Users can choose any other Positional Encoder that they may see fit.
        """
        assert self.predictor_embed_dim % 2 == 0

        omega = np.arange(self.predictor_embed_dim // 2, dtype=float)
        omega /= self.predictor_embed_dim / 2.0
        omega = 1.0 / 10000**omega

        pos = np.arange(self.num_patches, dtype=float)
        pos = pos.reshape(-1)
        out = np.einsum("m,d->md", pos, omega)

        emb_sin = np.sin(out)
        emb_cos = np.cos(out)

        emb = np.concatenate([emb_sin, emb_cos], axis=1)

        self.pos_embed.data.copy_(torch.from_numpy(emb).float().unsqueeze(0))

        return emb
