"""
CSCD84 - Artificial Intelligence, Winter 2025, Assignment 4
B. Chan
"""


import inspect
import os
import sys

currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(currentdir)
sys.path.insert(0, parentdir)

from src.mlp import MLPDiscriminator

import numpy as np
import torch
import torch.nn as nn


IMG_DIM = (1, 28, 28)
PATCH_SIZE = 7
assert IMG_DIM[1] % PATCH_SIZE == 0, (
    "patch size {} should divide the image dimensionality {}".format(
        PATCH_SIZE, IMG_DIM[1],
    )
)
PATCH_PER_AXIS = IMG_DIM[1] // PATCH_SIZE
NUM_PATCHES = PATCH_PER_AXIS ** 2
FLATTENED_IMG_DIM = np.prod(IMG_DIM)


class ViTGAN(nn.Module):
    def __init__(self, embed_dim, discriminator="mlp"):
        super().__init__()
        self.embed_dim = embed_dim

        if discriminator == "MLP":
            self.discriminator = MLPDiscriminator()
        elif discriminator == "ViT":
            self.discriminator = ViTDiscriminator(
                embed_dim,
                num_blocks=2,
                num_heads=4,
                widening_factor=4,
            )
        else:
            raise NotImplementedError
        self.generator = ViTGenerator(
            embed_dim,
            num_blocks=2,
            num_heads=4,
            widening_factor=2,
        )

    def generate(self, noise):
        return self.generator(noise)
    
    def discriminate(self, sample):
        return self.discriminator(sample)
    
    def sample_noise(self, batch_size):
        return torch.randn(batch_size, NUM_PATCHES + 1, self.embed_dim)


class TransformerEncoder(nn.Module):
    """
    Transformer architecture that follows GPT-2 (Radford et al., 2019).
    Here, we only consider the encoder architecture where there is no causal masking.
    """
    def __init__(self, embed_dim, num_heads, widening_factor=4):
        super().__init__()
        self.ln_1 = nn.LayerNorm(embed_dim)
        self.attention = nn.MultiheadAttention(
            embed_dim,
            num_heads,
            dropout=0.0,
            batch_first=True,
        )
        self.ln_2 = nn.LayerNorm(embed_dim)
        self.mlp_1 = nn.Linear(embed_dim, embed_dim * widening_factor)
        self.mlp_2 = nn.Linear(embed_dim * widening_factor, embed_dim)

    def forward(self, x):
        """
        Applies the transformer encoder on the sequence of inputs
        See Figure 2 in the handout for the transformer encoder architecture.
        NOTE: The MLP has two layers and the non-linearity is a GELU.
        """

        init = x
        x = self.ln_1(x)
        x = self.attention(x, x, x)[0]
        x = x + init
        init = x
        x = self.ln_2(x)
        x = self.mlp_1(x)
        x = nn.functional.gelu(x)
        x = self.mlp_2(x)
        x = x + init
        return x


class VisionTransformer(nn.Module):
    def __init__(self, embed_dim, num_blocks, num_heads, widening_factor=4):
        super().__init__()

        # Here we have one extra token for the [CLS] embedding
        self.num_tokens = NUM_PATCHES + 1

        # NOTE: We can actually implement this and patchify altogether using a 2D convolution
        self.linear_proj = nn.Linear(
            IMG_DIM[0] * PATCH_SIZE ** 2, embed_dim 
        )

        self.cls_token = nn.Parameter(
            torch.randn(1, 1, embed_dim)
        )

        # NOTE: We use learnable encoding here, but one can replace this with
        # e.g.
        # - Sinusoidal encoding: https://arxiv.org/pdf/1706.03762
        # - Rotary encoding: https://arxiv.org/abs/2104.09864
        # - Many more: https://arxiv.org/pdf/2312.17044v4
        self.pos_encoding = nn.Parameter(
            torch.randn(1, self.num_tokens, embed_dim)
        )

        self.transformer_blocks = nn.Sequential(*[
            TransformerEncoder(embed_dim, num_heads, widening_factor)
            for _ in range(num_blocks)
        ])

    def patchify(self, x):
        """
        Converts a batch of images into a batch of image patches.
        """
        B, C, H, W = x.shape
        x = x.reshape(B, C, H // PATCH_SIZE, PATCH_SIZE, W // PATCH_SIZE, PATCH_SIZE)
        x = x.permute(0, 2, 4, 1, 3, 5)
        x = x.flatten(1, 2)
        return x

    def forward(self, x, cls_token=None):
        batch_size = len(x)

        if cls_token is None:
            x = self.patchify(x)
            x = x.flatten(2, 4)
            # NOTE: x is now shaped (batch_size, num_patches, patch_size ** 2)

            cls_token = self.cls_token.repeat(batch_size, 1, 1)
            tokens = self.linear_proj(x)
        else:
            tokens = x

        # ========================================================
        # TODO: Complete the forward call of ViT
        # The shapes of relevant variables:
        # - tokens: (batch_size, num_patches, embed_dim)
        # - cls_tokens: (batch_size, 1, embed_dim)
        # Concatenate CLS token with patch tokens
        tokens = torch.cat([cls_token, tokens], dim=1)

        # Add positional encoding
        tokens = tokens + self.pos_encoding

        # Pass through transformer blocks
        tokens = self.transformer_blocks(tokens)

        # ========================================================

        # NOTE: tokens should be shaped (batch_size, num_patches, 1 + patch_size ** 2)
        #       We will be using these tokens for image generation and classification later.
        return tokens


class ViTDiscriminator(nn.Module):
    def __init__(self, embed_dim, num_blocks, num_heads, widening_factor=4):
        super().__init__()

        self.vit = VisionTransformer(
            embed_dim,
            num_blocks,
            num_heads,
            widening_factor,
        )
        self.out = nn.Linear(
            in_features=embed_dim,
            out_features=1,
            bias=True,
        )

    def forward(self, x):
        """
        Classifies using only the first token outputted by ViT
        """
        x = self.vit(x)
        x = x[:, 0]
        x = self.out(x)

        return x


class ViTGenerator(nn.Module):
    def __init__(self, embed_dim, num_blocks, num_heads, widening_factor=4):
        super().__init__()

        self.embed_dim = embed_dim
        self.vit = VisionTransformer(
            embed_dim,
            num_blocks,
            num_heads,
            widening_factor,
        )
        self.noise_embed = nn.Linear(
            in_features=embed_dim,
            out_features=embed_dim,
        )

        self.out = nn.Sequential(
            nn.Linear(
                in_features=embed_dim,
                out_features=(IMG_DIM[0] * PATCH_SIZE ** 2) // 2,
            ),
            nn.LeakyReLU(),
            nn.Linear(
                in_features=(IMG_DIM[0] * PATCH_SIZE ** 2) // 2,
                out_features=IMG_DIM[0] * PATCH_SIZE ** 2,
            ),
        )

    def forward(self, z):
        """
        Generates an image from noise
        """
        batch_size = len(z)

        # Flatten the noise
        cls_token = self.noise_embed(z)

        tokens = self.vit(cls_token[:, 1:], cls_token[:, :1])
        tokens = self.out(tokens[:, 1:])

        # Merge image patches back
        img = tokens.reshape(
            batch_size,
            PATCH_PER_AXIS,
            PATCH_PER_AXIS,
            IMG_DIM[0],
            PATCH_SIZE,
            PATCH_SIZE,
        ).permute(0, 3, 1, 4, 2, 5).reshape(batch_size, *IMG_DIM)

        # Images are assumed to have range [-1, 1]
        img = nn.functional.tanh(img)

        return img
