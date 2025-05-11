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

from datetime import datetime
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from cnn import transfer_weights
import argparse
import json
import numpy as np
import torch
import uuid
import torch.nn as nn
import torch.optim as optim
from pathlib import Path

from src.train import train, train_curriculum, train_stochastic_curriculum
from vit import ViTGAN
from cnn import CNNGAN
from cnncifar import CifarCNNGAN
#from hybrid_gan import HybridGAN
from pokemon_data import get_pokemon_dataloader, NUM_TYPES, POKEMON_TYPES
from train import train_step_gan

def get_cifar10_dataloader(batch_size, image_size, num_workers=2):
    """Get CIFAR10 dataloader with proper transforms"""
    transform = transforms.Compose([
        transforms.Resize(image_size),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])
    
    dataset = datasets.CIFAR10(
        root='./data',
        train=True,
        download=True,
        transform=transform
    )
    
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        # pin_memory=True,
        # drop_last=True
    )
    
    return dataloader

def get_celeba_dataloader(batch_size, image_size, num_workers=2):
    """Get CelebA dataloader with proper transforms (center crop and resize to 64x64)"""
    transform = transforms.Compose([
        transforms.CenterCrop(178),
        transforms.Resize(image_size),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])
    dataset = datasets.CelebA(
        root='./data',
        split='train',
        download=True,
        transform=transform
    )
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
    )
    return dataloader

def get_mnist_dataloader(batch_size, image_size, num_workers=2):
    """Get MNIST dataloader with proper transforms"""
    transform = transforms.Compose([
        transforms.Resize(image_size),
        transforms.RandomAffine(
            degrees=0,
            translate=(0.1, 0.1),
            scale=(1.0, 1.0),
        ),
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))  # MNIST is grayscale, so only one channel
    ])
    
    dataset = datasets.MNIST(
        root='./data',
        train=True,
        download=True,
        transform=transform
    )
    
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
    )
    
    return dataloader

def set_seed(seed=None):
    if seed is None:
        seed = np.random.randint(0, 2 ** 10)

    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True

def main(args):
    set_seed(args.seed)

    # Constants
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    BATCH_SIZE = 128  # Reduced batch size for better stability
    EMBED_DIM = 64  # Reduced embedding dimension for CIFAR-10
    if args.dataset == "pokemon":
        IMAGE_SIZE = 96
        PATCH_SIZE = 8
        CHANNELS = 3
    elif args.dataset == "mnist":
        IMAGE_SIZE = 28
        PATCH_SIZE = 7
        CHANNELS = 1
    elif args.dataset == "cifar10":
        IMAGE_SIZE = 32
        PATCH_SIZE = 4
        CHANNELS = 3
    elif args.dataset == "celeba":
        IMAGE_SIZE = 64
        PATCH_SIZE = 8
        CHANNELS = 3
    D_LEARNING_RATE = 2e-4  # Reduced learning rate
    G_LEARNING_RATE = 2e-4 # Reduced learning rate
    NUM_CLASSES = NUM_TYPES if args.dataset == "pokemon" else 10  # CIFAR-10 has 10 classes
    SAVE_DIR = Path(args.save_path)

    # Clear CUDA cache
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        
    # Create save directory
    SAVE_DIR.mkdir(exist_ok=True)
    
    # Get appropriate dataloader
    if args.dataset == "pokemon":
        dataloader = get_pokemon_dataloader(
            batch_size=BATCH_SIZE,
            image_size=IMAGE_SIZE,
            download=args.download,
            num_workers=2,
        )
    elif args.dataset == "celeba":
        dataloader = get_celeba_dataloader(
            batch_size=BATCH_SIZE,
            image_size=IMAGE_SIZE,
            num_workers=2
        )
    elif args.dataset == "mnist":
        dataloader = get_mnist_dataloader(
            batch_size=BATCH_SIZE,
            image_size=IMAGE_SIZE,
            num_workers=2
        )
    else:  # cifar10
        dataloader = get_cifar10_dataloader(
            batch_size=BATCH_SIZE,
            image_size=IMAGE_SIZE,
            num_workers=2
        )
    
    # # Initialize model based on architecture choice
    # if args.model == "CNN" and args.dataset == "cifar10":
    #     model = CifarCNNGAN(
    #         latent_dim=100,
    #         num_classes=NUM_CLASSES,
    #         channels=3,
    #         img_size=IMAGE_SIZE
    #     ).to(DEVICE)
    if args.model == "CNN":
        model = CNNGAN(
            latent_dim=100,
            num_classes=NUM_CLASSES,
            channels=3,
            img_size=IMAGE_SIZE
        ).to(DEVICE)
    # elif args.model == "Hybrid":
    #     model = HybridGAN(
    #         latent_dim=256,
    #         num_classes=NUM_TYPES,
    #         channels=3,
    #         img_size=IMAGE_SIZE
    #     ).to(DEVICE)
    else:  # ViT models
        model = ViTGAN(
            embed_dim=EMBED_DIM,
            #latent_dim=1024,
            img_size=IMAGE_SIZE,
            patch_size=PATCH_SIZE,
            num_classes=NUM_CLASSES,
            discriminator="ViT",
            channels=CHANNELS,
        ).to(DEVICE)

    if args.load_path and os.path.isfile(args.load_path):
        if args.model == "CNN":
            checkpoint = torch.load(args.load_path, weights_only=True)
            
            pretrained_model = CNNGAN(latent_dim=100, num_classes=NUM_CLASSES, channels=3, img_size=32)
            pretrained_model.load_state_dict(checkpoint["model_state_dict"])
            model = transfer_weights(pretrained_model, model)
            print("Transferred weights from {}".format(args.load_path))
        else:
            print("Loading parameters from {}".format(args.load_path))
            checkpoint = torch.load(args.load_path, weights_only=True)
            model.load_state_dict(checkpoint["model_state_dict"])
            optimizer.load_state_dict(checkpoint["optimizer_state_dict"])

    # Print model summary
    print(f"Selected model: {args.model}")
    print(f"Dataset: {args.dataset}")
    print(f"Image size: {IMAGE_SIZE}x{IMAGE_SIZE}")
    print(f"Patch size: {PATCH_SIZE}x{PATCH_SIZE}")
    print(f"Number of classes: {NUM_CLASSES}")
    print(model)
    print(f"Number of Generator Parameters: {sum(p.numel() for p in model.generator.parameters())}")
    print(f"Number of Discriminator Parameters: {sum(p.numel() for p in model.discriminator.parameters())}")
    print(f"Number of Total Parameters: {sum(p.numel() for p in model.parameters())}")

    time_tag = datetime.strftime(datetime.now(), "%m-%d-%y_%H_%M_%S")
    run_id = str(uuid.uuid4())
    run_name = os.path.join(
        args.save_path,
        "models",
        "{}-{}".format(time_tag, run_id),
    )

    print("Saving to {}".format(run_name))
    os.makedirs(run_name, exist_ok=True)
    # Save model architecture to file for logging
    model_arch_path = os.path.join(run_name, "model.txt")

    with open(model_arch_path, "w") as f:
        f.write(str(model))
        f.write("\n\nGenerator:\n")
        f.write(str(model.generator))
        f.write("\n\nDiscriminator:\n")
        f.write(str(model.discriminator))

    # Optimizers with better parameters for stability
    optimizer = {
        "generator": optim.Adam(
            model.generator.parameters(),
            lr=G_LEARNING_RATE,
            betas=(0.5, 0.999),
            # eps=1e-8,
            # weight_decay=1e-5
        ),
        "discriminator": optim.Adam(
            model.discriminator.parameters(),
            lr=D_LEARNING_RATE,
            betas=(0.5, 0.999),
            # eps=1e-8,
            # weight_decay=1e-5
        ),
    }




    json.dump(
        args,
        open(os.path.join(run_name, "config.json"), "w"),
        default=lambda s: vars(s),
    )

    if args.curriculum and args.dataset == "pokemon":
        print("Using curriculum learning...")
        train_curriculum(dataloader, model, optimizer, args, run_name)
    elif args.stochastic_curriculum and args.dataset == "pokemon":
        print("Using stochastic curriculum learning...")
        train_stochastic_curriculum(dataloader, model, optimizer, args, run_name)
    else:
        print("Using standard training...")
        train(dataloader, model, optimizer, args, run_name)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--dataset",
        choices=["pokemon", "cifar10", "celeba", "mnist"],
        default="pokemon",
        help="Dataset to train on"
    )

    parser.add_argument(
        "--num_iterations",
        type=int,
        default=100000,
        help="Number of times the data is iterated on"
    )

    parser.add_argument(
        "--model",
        choices=["MLP", "ViT", "ViT:MLP", "CNN", "Hybrid"],
        default="ViT",
        help="The model to use"
    )

    parser.add_argument(
        "--gradient_penalty",
        default=0,
        type=float,
        help="The gradient penalty coefficient"
    )

    parser.add_argument(
        "--save_path",
        type=str,
        default="./logs",
        help="The path to store any artifacts"
    )

    parser.add_argument(
        "--load_path",
        type=str,
        default=None,
        help="The path to restore an artifact"
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="The seed for randomness"
    )
    
    parser.add_argument(
        "--download",
        type=bool,
        default=True,
        help="Whether to download the dataset"
    )

    parser.add_argument(
        "--curriculum",
        action="store_true",
        help="Whether to use curriculum learning (train one type at a time)"
    )

    parser.add_argument(
        "--epochs_per_type",
        type=int,
        default=5,
        help="Number of epochs to train on each type when using curriculum learning"
    )

    parser.add_argument(
        "--iterations_per_type",
        type=int,
        default=1000,
        help="Number of iterations to train on each type before switching (for stochastic curriculum)"
    )

    parser.add_argument(
        "--stochastic_curriculum",
        action="store_true",
        help="Whether to use stochastic curriculum learning (randomly pick types)"
    )

    parser.add_argument(
        "--use_diffaug",
        action="store_true",
        help="Whether to use differentiable augmentation during training"
    )

    parser.add_argument(
        "--critic_steps",
        type=int,
        default=1,
        help="Number of critic (discriminator) steps per generator step"
    )

    parser.add_argument(
        "--feature_matching",
        action="store_true",
        help="Use feature matching loss for generator"
    )

    args = parser.parse_args()
    main(args)
