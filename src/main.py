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
from torchvision import datasets
from torchvision import transforms

import argparse
import json
import numpy as np
import torch
import uuid
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from pathlib import Path

from src.train import train
from vit import ViTGAN
from pokemon_data import get_pokemon_dataloader, NUM_TYPES, POKEMON_TYPES
from train import train_step_gan


def set_seed(seed=None):
    if seed is None:
        seed = np.random.randint(0, 2 ** 10)

    np.random.seed(seed)
    torch.manual_seed(seed)


def main(args):
    set_seed(args.seed)

    # Constants
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    BATCH_SIZE = 64
    EMBED_DIM = 512
    IMAGE_SIZE = 96 # Original image size
    PATCH_SIZE = 12 # 96/12 = 8 patches per side
    LEARNING_RATE = 1e-4
    SAVE_DIR = Path(args.save_path)

    # Clear CUDA cache
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        
    # Create save directory
    SAVE_DIR.mkdir(exist_ok=True)
    
    # Get data
    dataloader = get_pokemon_dataloader(
        batch_size=BATCH_SIZE,
        image_size=IMAGE_SIZE,
        download=args.download,
        num_workers=2,  # Limit number of workers
    )
    
    # Initialize model
    model = ViTGAN(
        embed_dim=EMBED_DIM,
        img_size=IMAGE_SIZE,
        patch_size=PATCH_SIZE,
        num_classes=NUM_TYPES,
        discriminator="ViT",
        attention_type="normal"
    ).to(DEVICE)

    # Print model summary
    print(model)
    # Print number of parameters
    print(f"Number of parameters: {sum(p.numel() for p in model.parameters())}")
    
    # Optimizers
    optimizer = {
        "generator": optim.Adam(
            model.generator.parameters(),
            lr=LEARNING_RATE,
            betas=(0.5, 0.999)
        ),
        "discriminator": optim.Adam(
            model.discriminator.parameters(),
            lr=LEARNING_RATE,
            betas=(0.5, 0.999)
        ),
    }

    if args.load_path and os.path.isfile(args.load_path):
        print("Loading parameters from {}".format(args.load_path))
        checkpoint = torch.load(args.load_path, weights_only=True)
        model.load_state_dict(checkpoint["model_state_dict"])
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])

    run_id = str(uuid.uuid4())
    time_tag = datetime.strftime(datetime.now(), "%m-%d-%y_%H_%M_%S")
    run_name = os.path.join(
        args.save_path,
        "models",
        "{}-{}".format(time_tag, run_id),
    )

    print("Saving to {}".format(run_name))

    os.makedirs(
        run_name,
        exist_ok=True,
    )

    json.dump(
        args,
        open(os.path.join(run_name, "config.json"), "w"),
        default=lambda s: vars(s),
    )

    print("Training...")
    train(dataloader, model, optimizer, args, run_name)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--num_iterations",
        type=int,
        default=40000,
        help="Number of times the data is iterated on"
    )

    parser.add_argument(
        "--model",
        choices=["MLP", "ViT", "ViT:MLP"],
        default="ViT",
        help="The model to use"
    )

    parser.add_argument(
        "--gradient_penalty",
        default=10,
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
    
    args = parser.parse_args()
    main(args)
