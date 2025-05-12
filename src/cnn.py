import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch.nn.utils import spectral_norm

class Generator(nn.Module):
    def __init__(self, channels, latent_dim, num_classes, ngf=64, img_size=32):
        super(Generator, self).__init__()
        self.latent_dim = latent_dim
        self.img_size = img_size
        self.ngf = ngf

        #self.embed = nn.Embedding(num_classes, 16)
        
        # Calculate initial size based on target resolution
        # For 32x32: initial_size = 2 (32/16)
        # For 96x96: initial_size = 6 (96/16)
        print(img_size)
        self.initial_size = img_size // 16
        print(self.initial_size)
        
        # Calculate number of upsampling layers needed
        # For 32x32: need 3 upsampling layers (2->4->8->16)
        # For 96x96: need 4 upsampling layers (6->12->24->48)
        self.num_upsamples = int(np.log2(img_size / self.initial_size))
        
        # Channel dimensions that will be shared between architectures
        self.channels = [
            ngf * 8,  # First layer
            ngf * 8,   # Second layer
            ngf * 4,   # Third layer
            ngf * 2,   # Fourth layer
            ngf   # Final layer
        ]

        # Initial projection layer
        self.initial = nn.Sequential(
            nn.ConvTranspose2d(latent_dim, self.channels[0], self.initial_size, 1, 0, bias=False),
            nn.BatchNorm2d(self.channels[0]),
            nn.ReLU(True)
        )

        # Upsampling layers
        self.upsampling_layers = nn.ModuleList()
        for i in range(self.num_upsamples):
            in_ch = self.channels[i]
            out_ch = self.channels[i+1]
            self.upsampling_layers.append(nn.Sequential(
                nn.ConvTranspose2d(in_ch, out_ch, 4, 2, 1, bias=False),
                nn.BatchNorm2d(out_ch),
                nn.ReLU(True)
            ))

        self.final = nn.Sequential(
            nn.Conv2d(self.channels[-1], channels, 3, 1, 1, bias=False),
            nn.Tanh()
        )
        # # Final layers
        # self.residual = nn.Sequential(
        #     nn.Conv2d(self.channels[-2], self.channels[-2], 3, 1, 1, bias=False),
        #     nn.BatchNorm2d(self.channels[-2]),
        #     nn.ReLU(True),
        #     nn.Conv2d(self.channels[-2], self.channels[-2], 3, 1, 1, bias=False),
        #     )
        


    def forward(self, input, labels):
        #input = torch.cat([input, self.embed(labels)], dim=1)
        x = self.initial(input.unsqueeze(-1).unsqueeze(-1))
        for layer in self.upsampling_layers:
            x = layer(x)
        #x = x + self.residual(x)
        x = self.final(x)
        return x

class Discriminator(nn.Module):
    def __init__(self, channels=3, ndf=64, img_size=32):
        super(Discriminator, self).__init__()
        self.img_size = img_size
        self.ndf = ndf
        
        # Calculate number of downsampling layers needed
        # For 32x32: need 3 downsampling layers (32->16->8->4)
        # For 96x96: need 4 downsampling layers (96->48->24->12)
        self.num_downsamples = int(np.log2(img_size / (img_size // 16)))
        self.final_size = img_size // (2 ** self.num_downsamples)
        
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
            spectral_norm(nn.Conv2d(channels, self.channels[1], 3, 1, 1, bias=False)),
            nn.LeakyReLU(0.2, inplace=True)
        )

        # Downsampling layers
        self.downsampling_layers = nn.ModuleList()
        for i in range(self.num_downsamples):
            in_ch = self.channels[i+1]
            out_ch = self.channels[i+2]
            self.downsampling_layers.append(nn.Sequential(
                spectral_norm(nn.Conv2d(in_ch, out_ch, 4, 2, 1, bias=False)),
                nn.BatchNorm2d(out_ch),
                nn.LeakyReLU(0.1, inplace=True)
            ))

        # Calculate final feature map size
        self.final = spectral_norm(nn.Conv2d(self.channels[-1], 1, self.final_size, 1, 0, bias=False))
        
        # # Final layers
        # self.final = nn.Sequential(
        #     nn.Conv2d(self.final_channels, self.final_channels, 3, 1, 1, bias=False),
        #     nn.BatchNorm2d(self.final_channels),
        #     nn.LeakyReLU(0.1, inplace=True)
        # )
        
        # # Calculate flattened size dynamically
        # self.flattened_size = self.final_channels * self.final_size * self.final_size
        
        # # MLP layers
        # self.mlp = nn.Sequential(
        #     spectral_norm(nn.Linear(self.flattened_size, 128)),
        #     nn.LeakyReLU(0.2),
        #     nn.Dropout(0.3),
        #     spectral_norm(nn.Linear(128, 1)),
        # )

    def forward(self, input):
        x = self.initial(input)
        for layer in self.downsampling_layers:
            x = layer(x)
        #x = self.final(x)
        #x = x.view(-1, self.flattened_size)
        x = self.final(x)
        return x.squeeze(-1).squeeze(-1)

    def get_features(self, input):
        x = self.initial(input)
        for layer in self.downsampling_layers:
            x = layer(x)
        return x

class CNNGAN(nn.Module):
    def __init__(self, latent_dim=100, num_classes=10, channels=3, img_size=32):
        super(CNNGAN, self).__init__()
        self.latent_dim = latent_dim
        self.num_classes = num_classes
        self.img_size = img_size
        
        # Initialize generator and discriminator
        self.generator = Generator(channels, latent_dim, num_classes, img_size=img_size)
        self.discriminator = Discriminator(channels, img_size=img_size)
        
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

def transfer_weights(pretrained_model, new_model):
    """Transfer weights from a pretrained model's generator to a new model's generator"""
    pretrained_state = pretrained_model.state_dict()
    new_state = new_model.state_dict()
    
    # Define which layers to transfer (early layers only)
    transfer_layers = [
        'generator.initial',  # Initial projection
        'generator.upsampling_layers.0',  # First upsampling
        'generator.upsampling_layers.1',  # Second upsampling
        # Skip later layers to allow domain-specific learning
    ]
    
    # Transfer weights for matching layers in generator only
    for key in pretrained_state:
        if key in new_state and pretrained_state[key].shape == new_state[key].shape:
            # Only transfer if it's an early layer
            if any(layer in key for layer in transfer_layers):
                print("Transferred generator layer: {}".format(key))
                new_state[key] = pretrained_state[key]
            else:
                print("Skipping later layer: {}".format(key))
        else:
            print("Skipping incompatible layer: {}".format(key))
    
    new_model.load_state_dict(new_state)
    return new_model

def get_layer_specific_params(model, transferred_lr=1e-5, new_lr=1e-4):
    """Get layer-specific parameters for fine-tuning"""
    params = []
    
    # Add parameters for transferred layers with lower learning rate
    transferred_layers = [
        model.generator.initial,
        model.generator.upsampling_layers[0],
        model.generator.upsampling_layers[1]
    ]
    for layer in transferred_layers:
        params.append({
            'params': layer.parameters(),
            'lr': transferred_lr
        })
    
    # Add parameters for new layers with higher learning rate
    new_layers = [
        *model.generator.upsampling_layers[2:],
        model.generator.final
    ]
    for layer in new_layers:
        params.append({
            'params': layer.parameters(),
            'lr': new_lr
        })
    
    return params

# # Usage example:
# # 1. Train on CIFAR-10 (32x32)
# model_32 = CNNGAN(latent_dim=100, num_classes=10, channels=3, img_size=32)
# # ... train on CIFAR-10 ...

# # 2. Create new model for 96x96
# model_96 = CNNGAN(latent_dim=100, num_classes=10, channels=3, img_size=96)

# # 3. Transfer weights
# model_96 = transfer_weights(model_32, model_96)

# 4. Finetune on Pokémon
# ... finetune on Pokémon ...
