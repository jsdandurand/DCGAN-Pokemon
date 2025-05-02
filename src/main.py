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

from src.train import train


def set_seed(seed=None):
    if seed is None:
        seed = np.random.randint(0, 2 ** 10)

    np.random.seed(seed)
    torch.manual_seed(seed)


def main(args):
    set_seed(args.seed)

    transform = transforms.Compose([
        transforms.RandomAffine(
            degrees=0,
            translate=(0.1, 0.1),
            scale=(1.0, 1.0),
        ),
        transforms.ToTensor(), 
        transforms.Normalize((0.5,), (0.5,)),
    ])

    dataset = datasets.MNIST(
        os.path.join(args.save_path, "datasets"),
        train=True,
        download=True,
        transform=transform,
    )

    device = "cuda" if torch.cuda.is_available() else "cpu"

    if args.model == "MLP":
        import src.mlp as mlp
        noise_dim = 100
        model = mlp.MLPGAN(noise_dim)
    elif args.model.startswith("ViT"):
        import src.vit as vit
        embed_dim = 64
        model_name = args.model.split(":")
        if len(model_name) == 1:
            model_name = "ViT"
        else:
            model_name = model_name[1]
        model = vit.ViTGAN(embed_dim, model_name)
    else:
        raise NotImplementedError
    model.to(device)

    learning_rate_disc = 5e-4
    learning_rate_gen = 2e-4
    beta_1 = 0.5
    beta_2 = 0.999
    optimizer = {
        "discriminator": torch.optim.Adam(
            model.discriminator.parameters(),
            lr=learning_rate_disc,
            betas=(beta_1, beta_2),
        ),
        "generator": torch.optim.Adam(
            model.generator.parameters(),
            lr=learning_rate_gen,
            betas=(beta_1, beta_2),
        )
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
    train(dataset, model, optimizer, args, run_name)


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
    
    args = parser.parse_args()
    main(args)
