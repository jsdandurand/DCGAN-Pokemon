"""
Vision Transformer GAN implementation with support for conditional generation
and multiple attention mechanisms.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from src.mlp import MLPDiscriminator

class ViTGAN(nn.Module):
    def __init__(
        self,
        embed_dim,
        img_size=64,
        patch_size=8,
        num_classes=18,  # Number of Pokemon types
        discriminator="ViT",
        channels=3,
    ):
        super().__init__()
        self.embed_dim = embed_dim
        self.img_size = img_size
        self.patch_size = patch_size
        self.num_patches = (img_size // patch_size) ** 2
        
        assert img_size % patch_size == 0, "Image size must be divisible by patch size"
        # Add embedding for type conditioning
        self.num_classes = num_classes
        self.class_embedding = nn.Embedding(num_classes, embed_dim)

        if discriminator == "CNN":
            self.discriminator = CNNDiscriminator(
                channels=channels,
                img_size=img_size,
            )
        elif discriminator == "ViT":
            self.discriminator = ViTDiscriminator(
                embed_dim,
                img_size=img_size,
                patch_size=patch_size,
                in_channels=channels,  # RGB images
                num_blocks=3,  # Increased for more complex images
                num_heads=8,
                widening_factor=4,
            )
        else:
            raise NotImplementedError
            
        self.generator = ViTGenerator(
            embed_dim,
            img_size=img_size,
            patch_size=patch_size,
            out_channels=channels,  # RGB images
            num_blocks=3,  # Increased for more complex images
            num_heads=8,
            widening_factor=4,
        )

    def generate(self, noise, labels):
        """
        Generate images conditioned on Pokemon type labels
        Args:
            noise: Input noise tensor of shape [batch_size, embed_dim]
            labels: Pokemon type labels tensor of shape [batch_size]
        Returns:
            Generated images of shape [batch_size, out_channels, img_size, img_size]
        """
    
        return self.generator(noise)

    def discriminate(self, x):
        return self.discriminator(x)

    def sample_noise(self, batch_size):
        return torch.randn(batch_size, self.num_patches + 1, self.embed_dim)


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
        x: (batch_size, num_patches, embed_dim)
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
    def __init__(
        self,
        embed_dim,
        img_size,
        patch_size,
        in_channels,
        num_blocks,
        num_heads,
        widening_factor=4,
    ):
        super().__init__()
        self.img_size = img_size
        self.patch_size = patch_size
        self.in_channels = in_channels
        
        # Calculate number of patches
        assert img_size % patch_size == 0, f"Image size {img_size} must be divisible by patch size {patch_size}"
        self.num_patches = (img_size // patch_size) ** 2
        self.num_tokens = self.num_patches + 1  # Add CLS token
        
        # Patch embedding projection
        patch_dim = in_channels * patch_size * patch_size
        self.linear_proj = nn.Linear(patch_dim, embed_dim)
        
        # CLS token
        self.cls_token = nn.Parameter(torch.randn(1, 1, embed_dim))
        
        # Positional encoding
        self.pos_encoding = nn.Parameter(torch.randn(1, self.num_tokens, embed_dim))
        
        # Transformer blocks
        self.transformer_blocks = nn.Sequential(*[
            TransformerEncoder(embed_dim, num_heads, widening_factor)
            for _ in range(num_blocks)
        ])

    def patchify(self, x):
        """
        Converts a batch of images into a batch of image patches.
        """
        B, C, H, W = x.shape
        x = x.reshape(B, C, H // self.patch_size, self.patch_size, W // self.patch_size, self.patch_size)
        x = x.permute(0, 2, 4, 1, 3, 5)
        x = x.flatten(1, 2)
        return x


    def forward(self, x, cls_token=None):
        """
        Forward pass of Vision Transformer
        x: Either image tensor [B, C, H, W] or patch embeddings [B, num_patches, embed_dim]
        cls_token: Optional pre-computed CLS token
        """
        batch_size = len(x)
        

        # cls token is None for Discriminator, and passed in for Generator
        # because Generator needs to pass in the type embedding and noise is already patched
        if cls_token is None:
            # If x is image, convert to patches
            if len(x.shape) == 4:
                x = self.patchify(x)
                x = x.flatten(2, 4)                
                x = self.linear_proj(x)
            cls_token = self.cls_token.expand(batch_size, -1, -1)
            
        # Concatenate CLS token
        x = torch.cat([cls_token, x], dim=1)
        
        # Add positional encoding
        x = x + self.pos_encoding
        
        # Pass through transformer blocks
        x = self.transformer_blocks(x)
        
        return x


class ViTDiscriminator(nn.Module):
    def __init__(
        self,
        embed_dim,
        img_size=64,
        patch_size=8,
        in_channels=3,
        num_blocks=4,
        num_heads=8,
        widening_factor=4,
    ):
        super().__init__()
        self.vit = VisionTransformer(
            embed_dim=embed_dim,
            img_size=img_size,
            patch_size=patch_size,
            in_channels=in_channels,
            num_blocks=num_blocks,
            num_heads=num_heads,
            widening_factor=widening_factor,
        )
        self.output_proj = nn.Linear(embed_dim, 1)

    def forward(self, x):
        """
        Classifies using only the CLS token output
        """
        x = self.vit(x)
        x = x[:, 0]  # Take CLS token
        x = self.output_proj(x)
        return x


class ViTGenerator(nn.Module):
    def __init__(
        self,
        embed_dim,
        img_size=64,
        patch_size=8,
        out_channels=3,
        num_blocks=4,
        num_heads=8,
        widening_factor=4,
    ):
        super().__init__()
        self.embed_dim = embed_dim
        self.img_size = img_size
        self.patch_size = patch_size
        self.out_channels = out_channels
        
        # Calculate number of patches and dimensions
        self.num_patches = (img_size // patch_size) ** 2
        self.patch_dim = patch_size * patch_size * out_channels
        
        # Project input noise to initial embeddings
        self.noise_embed = nn.Sequential(
            nn.Linear(embed_dim, embed_dim)
        )
        
        # Vision Transformer for processing
        self.vit = VisionTransformer(
            embed_dim=embed_dim,
            img_size=img_size,
            patch_size=patch_size,
            in_channels=out_channels,
            num_blocks=num_blocks,
            num_heads=num_heads,
            widening_factor=widening_factor,
        )
        
        # Progressive output projection
        self.out = nn.Sequential(
            nn.Linear(
                in_features=embed_dim,
                out_features=(self.out_channels * self.patch_size ** 2) // 2,
            ),
            nn.LeakyReLU(),
            nn.Linear(
                in_features=(self.out_channels * self.patch_size ** 2) // 2,
                out_features=self.out_channels * self.patch_size ** 2,
            ),
        )

            

    def forward(self, z):
        """
        Generates images from noise vectors
        Args:
            z: Input noise tensor of shape [batch_size, num_patches + 1, embed_dim]
        Returns:
            Generated images of shape [batch_size, out_channels, img_size, img_size]
        """
        batch_size = len(z)

        # Flatten the noise
        cls_token = self.noise_embed(z)

        tokens = self.vit(cls_token[:, 1:], cls_token[:, :1])
        tokens = self.out(tokens[:, 1:]) # [B, num_patches, patch_dim]

        # Merge image patches back
        images = tokens.reshape(
            batch_size,
            self.img_size // self.patch_size,  # H patches
            self.img_size // self.patch_size,  # W patches
            self.out_channels,                  # channels
            self.patch_size,                   # patch height
            self.patch_size,                   # patch width

        )
        
        # Rearrange dimensions to standard image format
        images = images.permute(0, 3, 1, 4, 2, 5)
        images = images.reshape(batch_size, self.out_channels, self.img_size, self.img_size)
        
        images = nn.functional.tanh(images)
        return images


class CNNDiscriminator(nn.Module):
    def __init__(self, channels=3, ndf=64, img_size=32):
        super(CNNDiscriminator, self).__init__()
        self.img_size = img_size
        self.ndf = ndf
        
        # Calculate number of downsampling layers needed
        # For 32x32: need 3 downsampling layers (32->16->8->4)
        # For 96x96: need 4 downsampling layers (96->48->24->12)
        self.num_downsamples = int(np.log2(img_size / (img_size // 16)))
        
        # Channel dimensions that will be shared between architectures
        self.channels = [
            channels,   # First layer
            ndf * 1,    # Second layer
            ndf * 2,    # Third layer
            ndf * 4,    # Fourth layer
            ndf * 8,    # Fifth layer
            ndf * 16,    # Sixth layer
        ]

        # Initial layers
        self.initial = nn.Sequential(
            nn.Conv2d(channels, self.channels[1], 3, 1, 1, bias=False),
            nn.LeakyReLU(0.2, inplace=True)
        )

        # Downsampling layers
        self.downsampling_layers = nn.ModuleList()
        for i in range(self.num_downsamples):
            in_ch = self.channels[i+1]
            out_ch = self.channels[i+2]
            self.downsampling_layers.append(nn.Sequential(
                nn.Conv2d(in_ch, out_ch, 4, 2, 1, bias=False),
                nn.BatchNorm2d(out_ch),
                nn.LeakyReLU(0.1, inplace=True)
            ))

        # Calculate final feature map size
        self.final_size = img_size // (2 ** self.num_downsamples)
        self.final_channels = self.channels[-1]
        
        # # Final layers
        # self.final = nn.Sequential(
        #     nn.Conv2d(self.final_channels, self.final_channels, 3, 1, 1, bias=False),
        #     nn.BatchNorm2d(self.final_channels),
        #     nn.LeakyReLU(0.1, inplace=True)
        # )
        
        # Calculate flattened size dynamically
        self.flattened_size = self.final_channels * self.final_size * self.final_size
        
        # MLP layers
        self.mlp = nn.Sequential(
            nn.Linear(self.flattened_size, 128),
            nn.LeakyReLU(0.2),
            nn.Dropout(0.3),
            nn.Linear(128, 1),
        )

    def forward(self, input):
        x = self.initial(input)
        for layer in self.downsampling_layers:
            x = layer(x)
        #x = self.final(x)
        x = x.view(-1, self.flattened_size)
        x = self.mlp(x)
        return x.view(-1, 1)

    def get_features(self, input):
        x = self.initial(input)
        for layer in self.downsampling_layers:
            x = layer(x)
        x = x.view(-1, self.flattened_size)
        return x


class ViTAEGAN(nn.Module):
    def __init__(
        self,
        latent_dim=100,
        num_classes=10,
        image_size=64,
        patch_size=16,
        embed_dim=256,
        channels=3,
    ):
        super().__init__()
        
        self.latent_dim = latent_dim
        self.num_classes = num_classes
        self.image_size = image_size
        self.patch_size = patch_size
        self.embed_dim = embed_dim
        self.channels = channels
        
        # Calculate number of patches
        self.num_patches = (image_size // patch_size) ** 2
        
        # Initialize encoder
        self.encoder = ViTEncoder(
            image_size=image_size,
            patch_size=patch_size,
            in_channels=channels,
            embed_dim=embed_dim,
            latent_dim=latent_dim
        )
        
        # Initialize generator
        self.generator = ViTGenerator(
            latent_dim=latent_dim,
            num_classes=num_classes,
            image_size=image_size,
            patch_size=patch_size,
            embed_dim=embed_dim,
            channels=channels
        )
        
        # Initialize discriminator
        self.discriminator = ViTDiscriminator(
            image_size=image_size,
            patch_size=patch_size,
            in_channels=channels,
            embed_dim=embed_dim
        )
        
        # Initialize weights
        self.apply(self._init_weights)
    
    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.xavier_uniform_(m.weight)
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, nn.LayerNorm):
            nn.init.ones_(m.weight)
            nn.init.zeros_(m.bias)
    
    def sample_noise(self, batch_size):
        return torch.randn(batch_size, self.latent_dim)
    
    def encode(self, x):
        return self.encoder(x)
    
    def decode(self, z, labels):
        return self.generator(z, labels)
    
    def generate(self, z, labels):
        return self.generator(z, labels)
    
    def discriminate(self, x):
        return self.discriminator(x)
    
    def reconstruct(self, x, labels):
        z = self.encode(x)
        return self.decode(z, labels)

class ViTEncoder(nn.Module):
    def __init__(
        self,
        image_size=64,
        patch_size=16,
        in_channels=3,
        embed_dim=256,
        latent_dim=100,
        depth=6,
        num_heads=8,
        mlp_ratio=4.0,
        dropout=0.0
    ):
        super().__init__()
        
        # Patch embedding
        self.patch_embed = PatchEmbed(
            image_size=image_size,
            patch_size=patch_size,
            in_channels=in_channels,
            embed_dim=embed_dim
        )
        
        # Position embedding
        num_patches = (image_size // patch_size) ** 2
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches, embed_dim))
        
        # Transformer blocks
        self.blocks = nn.Sequential(*[
            TransformerBlock(
                dim=embed_dim,
                num_heads=num_heads,
                mlp_ratio=mlp_ratio,
                dropout=dropout
            )
            for _ in range(depth)
        ])
        
        # Final layer norm
        self.norm = nn.LayerNorm(embed_dim)
        
        # Project to latent space
        self.proj = nn.Linear(embed_dim, latent_dim)
        
        # Initialize weights
        nn.init.trunc_normal_(self.pos_embed, std=0.02)
    
    def forward(self, x):
        # Patch embedding
        x = self.patch_embed(x)
        
        # Add position embedding
        x = x + self.pos_embed
        
        # Transformer blocks
        x = self.blocks(x)
        
        # Final layer norm
        x = self.norm(x)
        
        # Global average pooling
        x = x.mean(dim=1)
        
        # Project to latent space
        x = self.proj(x)
        
        return x

