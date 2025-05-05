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
        attention_type="normal"
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

        if discriminator == "MLP":
            self.discriminator = MLPDiscriminator(
                img_size=img_size,
                in_channels=3  # RGB images
            )
        elif discriminator == "ViT":
            self.discriminator = ViTDiscriminator(
                embed_dim,
                img_size=img_size,
                patch_size=patch_size,
                in_channels=3,  # RGB images
                num_blocks=4,  # Increased for more complex images
                num_heads=8,
                widening_factor=4,
                attention_type=attention_type
            )
        else:
            raise NotImplementedError
            
        self.generator = ViTGenerator(
            embed_dim,
            img_size=img_size,
            patch_size=patch_size,
            out_channels=3,  # RGB images
            num_blocks=4,  # Increased for more complex images
            num_heads=8,
            widening_factor=4,
            attention_type=attention_type
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
        # Get type embeddings
        type_embed = self.class_embedding(labels)
        type_embed = type_embed.unsqueeze(1)
        # Combine noise and type embedding
        combined_input = noise + type_embed
        return self.generator(combined_input)

    def discriminate(self, x):
        return self.discriminator(x)

    def sample_noise(self, batch_size):
        return torch.randn(batch_size, self.num_patches + 1, self.embed_dim)


class TransformerEncoder(nn.Module):
    """
    Transformer architecture that follows GPT-2 (Radford et al., 2019).
    Here, we only consider the encoder architecture where there is no causal masking.
    """
    def __init__(self, embed_dim, num_heads, widening_factor=4, attention_type="normal"):
        super().__init__()
        self.ln_1 = nn.LayerNorm(embed_dim)
        self.attention_type = attention_type
        
        if attention_type == "normal":
            self.attention = nn.MultiheadAttention(
                embed_dim,
                num_heads,
                dropout=0.0,
                batch_first=True,
            )
        elif attention_type == "spatial":
            self.attention = SpatialAttention(embed_dim)
        else:
            raise ValueError(f"Unknown attention type: {attention_type}")
            
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
        
        if self.attention_type == "normal":
            x = self.attention(x, x, x)[0]
        else:  # spatial attention
            # Reshape for spatial attention
            B, N, C = x.shape
            H = W = int(np.sqrt(N))
            x = x.reshape(B, H, W, C).permute(0, 3, 1, 2)  # B, C, H, W
            x = self.attention(x)
            x = x.permute(0, 2, 3, 1).reshape(B, N, C)  # Back to B, N, C
            
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
        attention_type="normal"
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
            TransformerEncoder(embed_dim, num_heads, widening_factor, attention_type)
            for _ in range(num_blocks)
        ])

    def patchify(self, x):
        """
        Converts a batch of images into a batch of patches.
        Input: (batch_size, channels, height, width)
        Output: (batch_size, num_patches, patch_dim)
        """
        B, C, H, W = x.shape
        assert H == W == self.img_size, f"Input image size ({H}x{W}) doesn't match expected size ({self.img_size}x{self.img_size})"
        
        # Reshape into patches
        x = x.reshape(B, C, H // self.patch_size, self.patch_size, W // self.patch_size, self.patch_size)
        x = x.permute(0, 2, 4, 1, 3, 5)  # [B, H', W', C, p, p]
        x = x.flatten(1, 2)  # [B, num_patches, C, p, p]
        x = x.reshape(B, self.num_patches, -1)  # [B, num_patches, C*p*p]
        
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
        attention_type="normal"
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
            attention_type=attention_type
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
        attention_type="normal"
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
            nn.Linear(embed_dim, embed_dim * 2),
            nn.LeakyReLU(0.2),
            nn.Linear(embed_dim * 2, embed_dim)
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
            attention_type=attention_type
        )
        
        # Progressive output projection
        self.out = nn.Sequential(
            nn.Linear(embed_dim, self.patch_dim // 2),
            nn.LeakyReLU(0.2),
            nn.Linear(self.patch_dim // 2, self.patch_dim),
            nn.Tanh()  # Ensures output is in [-1, 1] range
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
            self.patch_size,                   # patch height
            self.patch_size,                   # patch width
            self.out_channels                  # channels
        )
        
        # Rearrange dimensions to standard image format
        images = images.permute(0, 5, 1, 3, 2, 4).contiguous()
        images = images.view(batch_size, self.out_channels, self.img_size, self.img_size)
        
        return images


def interpolate_digits(model, start_digit, end_digit, steps=10):
    """Generate smooth transition between two digits."""
    # Create start and end labels
    start_label = torch.tensor([start_digit]).to(DEVICE)
    end_label = torch.tensor([end_digit]).to(DEVICE)
    
    # Generate fixed noise
    z = model.sample_noise(1).to(DEVICE)
    
    # Get embeddings
    start_embed = model.class_embedding(start_label)
    end_embed = model.class_embedding(end_label)
    
    images = []
    # Interpolate between embeddings
    for alpha in np.linspace(0, 1, steps):
        # Interpolate embeddings
        curr_embed = start_embed * (1 - alpha) + end_embed * alpha
        
        # Add to noise
        curr_z = z.clone()
        curr_z[:, 0, :] += curr_embed
        
        # Generate image
        with torch.no_grad():
            img = model.generate(curr_z)
            images.append(img)
    
    return torch.cat(images, dim=0)


class SpatialAttention(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.query = nn.Conv2d(channels, channels//8, 1)
        self.key = nn.Conv2d(channels, channels//8, 1)
        self.value = nn.Conv2d(channels, channels, 1)
        
    def forward(self, x):
        b, c, h, w = x.shape
        q = self.query(x).view(b, -1, h*w) # B, C/8, H*W (H*W = Number of patches)
        k = self.key(x).view(b, -1, h*w)
        v = self.value(x).view(b, -1, h*w)
        
        attention = torch.bmm(q.transpose(1,2), k)
        attention = F.softmax(attention, dim=-1)
        
        out = torch.bmm(v, attention.transpose(1,2))
        return out.view(b, c, h, w)
