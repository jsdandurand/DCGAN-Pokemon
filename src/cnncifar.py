import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class Generator(nn.Module):
    def __init__(self, channels, latent_dim, num_classes, ngf=64, img_size=32):
        super(Generator, self).__init__()
        self.latent_dim = latent_dim
        self.num_classes = num_classes
        self.img_size = img_size
        self.ngf = ngf
        self.initial_size = img_size // 8
        
        # Calculate channel dimensions for each layer
        self.channels = [
            ngf * 16,  # First layer
            ngf * 8,  # Second layer
            ngf * 4,  # Third layer
            ngf * 2,  # Fourth layer
            channels  # Final layer
        ]

        self.embed = nn.Embedding(num_classes, latent_dim)
        self.main = nn.Sequential(
            # latent_dim x 1 x 1 -> ngf * 8 x initial_size x initial_size
            nn.ConvTranspose2d(latent_dim, self.channels[0], self.initial_size, 1, 0, bias=False),
            nn.BatchNorm2d(self.channels[0]),
            nn.ReLU(True),

            # ngf * 8 x initial_size x initial_size -> ngf * 4 x initial_size*2 x initial_size*2
            nn.ConvTranspose2d(self.channels[0], self.channels[1], 4, 2, 1, bias=False),
            nn.BatchNorm2d(self.channels[1]),
            nn.ReLU(True),

            # ngf * 4 x initial_size*2 x initial_size*2 -> ngf * 2 x initial_size*4 x initial_size*4
            nn.ConvTranspose2d(self.channels[1], self.channels[2], 4, 2, 1, bias=False),
            nn.BatchNorm2d(self.channels[2]),
            nn.ReLU(True),

            # ngf * 2 x initial_size*4 x initial_size*4 -> ngf x initial_size*8 x initial_size*8
            nn.ConvTranspose2d(self.channels[2], self.channels[3], 4, 2, 1, bias=False),
            nn.BatchNorm2d(self.channels[3]),
            nn.ReLU(True),

            # ngf x initial_size*8 x initial_size*8 -> channels x img_size x img_size
            nn.Conv2d(self.channels[3], self.channels[4], kernel_size=3, padding=1, bias=False),
            nn.Tanh()
        )

    def forward(self, input, labels):
        # Input should already be in shape (batch_size, latent_dim, 1, 1)
        input = input + self.embed(labels)
        return self.main(input.unsqueeze(-1).unsqueeze(-1))


class Discriminator(nn.Module):
    def __init__(self, channels=3, ndf=64, img_size=32):
        super(Discriminator, self).__init__()
        self.img_size = img_size
        self.ndf = ndf
        
        # Calculate number of downsampling layers needed
        self.num_downsamples = int(np.log2(img_size // (img_size // 8)))
        
        # CNN layers for downsampling
        cnn_layers = []
        in_channels = channels
        out_channels = ndf
        
        # Add downsampling layers
        for i in range(self.num_downsamples):
            cnn_layers.extend([
                nn.Conv2d(in_channels, out_channels, 4, 2, 1, bias=False),
                nn.LeakyReLU(0.2, inplace=True)
            ])
            in_channels = out_channels
            out_channels = out_channels * 2
            
            if i > 0:  # Add batch norm after first layer
                cnn_layers.append(nn.BatchNorm2d(in_channels))
        
        self.cnn = nn.Sequential(*cnn_layers)
        
        # Calculate flattened size after CNN
        self.flattened_size = in_channels * (img_size // 8) * (img_size // 8)
        
        # MLP layers
        self.mlp = nn.Sequential(
            nn.Linear(self.flattened_size, 512),
            nn.LeakyReLU(0.2),
            nn.Linear(512, 1),
        )

    def forward(self, input):
        # Downsample with CNN
        x = self.cnn(input)
        # Flatten
        x = x.view(-1, self.flattened_size)
        # MLP classification
        x = self.mlp(x)
        return x.view(-1, 1)


class CifarCNNGAN(nn.Module):
    def __init__(self, latent_dim=100, num_classes=10, channels=3, img_size=32):
        super(CifarCNNGAN, self).__init__()
        self.latent_dim = latent_dim
        self.num_classes = num_classes
        self.img_size = img_size
        
        # Initialize generator and discriminator
        self.generator = Generator(channels, latent_dim, num_classes, img_size=img_size)
        self.discriminator = Discriminator(channels, num_classes, img_size=img_size)
        
        # Initialize weights
        self.apply(self._init_weights)
        
    def _init_weights(self, m):
        classname = m.__class__.__name__
        if classname.find('Conv') != -1:
            m.weight.data.normal_(0.0, 0.02)
        elif classname.find('BatchNorm') != -1:
            m.weight.data.normal_(1.0, 0.02)
            m.bias.data.fill_(0)
        
    def generate(self, z, labels):
        """Generate images from noise"""
        # z should already be in shape (batch_size, latent_dim, 1, 1)
        return self.generator(z, labels)
    
    def discriminate(self, images):
        """Discriminate images"""
        return self.discriminator(images)
    
    def sample_noise(self, batch_size):
        """Sample random noise for the generator"""
        return torch.randn(batch_size, self.latent_dim)
