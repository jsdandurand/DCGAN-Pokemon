"""
CSCD84 - Artificial Intelligence, Winter 2025, Assignment 4
B. Chan
"""


import numpy as np
import torch
import torch.nn as nn


IMG_DIM = (1, 28, 28)
FLATTENED_IMG_DIM = np.prod(IMG_DIM)


class MLPGAN(nn.Module):
    def __init__(self, noise_dim):
        super().__init__()
        self.noise_dim = noise_dim
        self.discriminator = MLPDiscriminator()
        self.generator = MLPGenerator(noise_dim)

    def generate(self, noise):
        return self.generator(noise)
    
    def discriminate(self, sample):
        return self.discriminator(sample)
    
    def sample_noise(self, batch_size):
        return torch.randn(batch_size, self.noise_dim)


class MLPDiscriminator(nn.Module):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.hidden_1 = nn.Linear(
            in_features=FLATTENED_IMG_DIM,
            out_features=FLATTENED_IMG_DIM,
            bias=True,
        )

        self.hidden_2 = nn.Linear(
            in_features=FLATTENED_IMG_DIM,
            out_features=FLATTENED_IMG_DIM,
            bias=True,
        )

        self.out = nn.Linear(
            in_features=FLATTENED_IMG_DIM,
            out_features=1,
            bias=True,
        )

    def forward(self, x):
        # Flatten the image
        x = x.reshape(len(x), -1)

        x = self.hidden_1(x)
        x = nn.functional.relu(x)
        x = self.hidden_2(x)
        x = nn.functional.relu(x)
        x = self.out(x)

        return x


class MLPGenerator(nn.Module):
    def __init__(self, noise_dim, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.hidden_1 = nn.Linear(
            in_features=noise_dim,
            out_features=FLATTENED_IMG_DIM // 4,
            bias=True,
        )

        self.hidden_2 = nn.Linear(
            in_features=FLATTENED_IMG_DIM // 4,
            out_features=FLATTENED_IMG_DIM // 2,
            bias=True,
        )

        self.out = nn.Linear(
            in_features=FLATTENED_IMG_DIM // 2,
            out_features=FLATTENED_IMG_DIM,
            bias=True,
        )

    def forward(self, z):
        # Flatten the noise
        x = z.reshape(len(z), -1)

        x = self.hidden_1(x)
        x = nn.functional.leaky_relu(x)
        x = self.hidden_2(x)
        x = nn.functional.leaky_relu(x)
        x = self.out(x)

        # Images are assumed to have range [-1, 1]
        x = nn.functional.tanh(x).reshape((-1, *IMG_DIM))

        return x
