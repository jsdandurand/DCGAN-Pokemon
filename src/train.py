"""
CSCD84 - Artificial Intelligence, Winter 2025, Assignment 4
B. Chan
"""


import pickle
import math
import matplotlib.pyplot as plt
import numpy as np
import os
import timeit
import torch
import torch.nn as nn
import torch.nn.functional as F
from pathlib import Path

from functools import partial
from torch import autograd


LOG_INTERVAL = 1000
SPLIT_SEED = 0
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def denormalize_image(image):
    """
    Denormalize images from [-1, 1] range back to [0, 1]
    Inverse of transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    """
    return (image + 1) / 2


def train_step_gan(batch, model, optimizer):
    """
    Performs a step of batch update using the GAN objective with Pokemon type conditioning.
    """
    loss_fn = nn.BCEWithLogitsLoss()
    
    batch_size = len(batch["train_X"])
    labels = batch["train_y"]

    # Update discriminator
    noise = batch["noise"]
    fake_images = model.generate(noise, labels)  # Pass labels for conditional generation
    real_images = batch["train_X"]

    all_images = torch.cat((fake_images.detach(), real_images), dim=0)
    preds = model.discriminate(all_images)

    targs = torch.tensor(
        [0.0, 1.0], device=DEVICE
    ).tile(batch_size, 1).T.reshape(-1, 1)

    optimizer["discriminator"].zero_grad()
    disc_loss = loss_fn(preds, targs) * 2 # the mean is across 2N samples.
    disc_loss.backward()
    optimizer["discriminator"].step()

    # Update generator
    preds = model.discriminate(fake_images)
    optimizer["generator"].zero_grad()
    gen_loss = loss_fn(preds, targs[batch_size:])
    gen_loss.backward()
    optimizer["generator"].step()

    step_info = {
        "discriminator_loss": disc_loss.detach().cpu().item(),
        "generator_loss": gen_loss.detach().cpu().item(),
        "fake_images": fake_images.detach().cpu().numpy(),
        "labels": labels.detach().cpu().numpy(),  # Save labels for visualization
    }

    # Compute gradient norm
    for n, p in filter(
        lambda layer: layer[1].grad is not None,
        model.named_parameters()
    ):
        step_info["grad_norm/{}".format(n)] = p.grad.data.norm(2).item()
        step_info["param_norm/{}".format(n)] = p.norm(2).item()

    return step_info


def compute_gradient_penalty(model, real_images, fake_images, eps):
    """
    Computes the gradient penalty.

    Inputs' shape:
    - real_images: (batch_size, image_channel, image_height, image_width)
    - fake_images: (batch_size, image_channel, image_height, image_width)
    - eps: (batch_size, 1, 1, 1)

    NOTE: You should take the convex combination as described in the handout,
          i.e., eps * x_real + (1 - eps) * x_fake.
    NOTE: autograd.grad might be helpful when computing the gradient w.r.t. inputs.
    """

    batch_size = len(real_images)
    eps = eps.expand_as(real_images)
    gp_loss = None


    x_hat = eps * real_images + (1 - eps) * fake_images
    x_hat.requires_grad = True
    preds = model.discriminate(x_hat)
    grads = autograd.grad(outputs=preds, inputs=x_hat, grad_outputs=torch.ones_like(preds), create_graph=True, retain_graph=True, only_inputs=True)[0]
    gp_loss = ((grads.norm(2, dim=1) - 1) ** 2).mean()


    
    return gp_loss


def train_step_wgan(batch, model, optimizer, gp):
    """
    Performs a step of batch update using the Wasserstein GAN objective with gradient penalty.
    """

    batch_size = len(batch["train_X"])
    labels = batch["train_y"]

    # Update discriminator/critic
    noise = batch["noise"]
    fake_images = model.generate(noise, labels)  # Pass labels for conditional generation
    real_images = batch["train_X"]

    all_images = torch.cat((fake_images.detach(), real_images), dim=0)
    preds = model.discriminate(all_images)

    optimizer["discriminator"].zero_grad()
    
    disc_loss = None
    gen_loss = None

    disc_loss = -(preds[batch_size:].mean() - preds[:batch_size].mean())  # Negate to minimize

    eps = batch["eps"]
    gp_loss = compute_gradient_penalty(model, real_images, fake_images.detach(), eps)
    total_loss = disc_loss + gp * gp_loss
    total_loss.backward()
    optimizer["discriminator"].step()

    # Update generator
    preds = model.discriminate(fake_images)
    optimizer["generator"].zero_grad()
    
    gen_loss = -preds[:batch_size].mean()  # Negate to maximize

    gen_loss.backward()
    optimizer["generator"].step()

    step_info = {
        "discriminator_loss": disc_loss.detach().cpu().item(),
        "generator_loss": gen_loss.detach().cpu().item(),
        "gp_loss": gp_loss.detach().cpu().item(),
        "fake_images": fake_images.detach().cpu().numpy(),
        "labels": labels.detach().cpu().numpy(),  # Save labels for visualization
    }

    # Compute gradient norm
    for n, p in filter(
        lambda layer: layer[1].grad is not None,
        model.named_parameters()
    ):
        step_info["grad_norm/{}".format(n)] = p.grad.data.norm(2).item()
        step_info["param_norm/{}".format(n)] = p.norm(2).item()

    return step_info


def train(train_loader, model, optimizer, args, run_name):
    """
    Runs the training loop.
    """

    # Make sure to split the same way
    gen = torch.Generator()
    gen.manual_seed(SPLIT_SEED)

    batch_size = 64

    loader = iter(train_loader)

    gp = getattr(args, "gradient_penalty", None)
    train_step = train_step_gan if gp is None or gp == 0.0 else partial(train_step_wgan, gp=gp)

    logs = dict()
    step_infos = dict()
    tic = timeit.default_timer()
    for iter_i in range(args.num_iterations):
        try:
            (train_X, train_y) = next(loader)
        except StopIteration:
            loader = iter(train_loader)
            (train_X, train_y) = next(loader)

        step_info = train_step(
            {
                "train_X": train_X.to(DEVICE),
                "train_y": train_y.to(DEVICE),
                "noise": model.sample_noise(batch_size).to(DEVICE),
                "eps": torch.rand(batch_size, 1, 1, 1).to(DEVICE),
            },
            model,
            optimizer,
        )
        for k, v in step_info.items():
            step_infos.setdefault(k, [])
            step_infos[k].append(v)

        if (iter_i + 1) % LOG_INTERVAL == 0:
            step_infos = {k: np.mean(v) for k, v in step_infos.items()}
            toc = timeit.default_timer()

            print("Iteration {} ==========================================".format(iter_i + 1))
            print("Time taken for training: {:4f}s".format(toc - tic))
            for k, v in step_info.items():
                if k == "fake_images":
                    continue
                logs.setdefault(k, [])
                logs[k].append(v)
                print("Avg {}: {:4f}".format(k, v))

            # Save images sampled
            num_images = len(step_info["fake_images"])
            num_rows = num_cols = math.ceil(math.sqrt(num_images))
            num_axes = num_rows * num_cols
            fig, axes = plt.subplots(
                num_rows,
                num_cols,
                figsize=(num_rows * 1.5, num_cols * 1.5),
                layout="constrained",
            )
            axes = axes.flatten()
            for image_i in range(num_axes):
                if image_i < num_images:
                    image = step_info["fake_images"][image_i]
                    label = step_info["labels"][image_i]
                    axes[image_i].set_xticklabels([])
                    axes[image_i].set_yticklabels([])
                    axes[image_i].imshow(image.transpose((1, 2, 0)))
                    axes[image_i].set_title(f'Label: {label}')
                else:
                    axes[image_i].axis("off")

            fig.savefig(
                os.path.join(run_name, "{}.pdf".format(iter_i + 1)),
                dpi=600,
            )
            plt.close(fig)
            
            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": {
                        k: v.state_dict() for k, v in optimizer.items()
                    },
                },
                os.path.join(run_name, "latest.pt")
            )
            step_infos = dict()
            tic = timeit.default_timer()

        pickle.dump(logs, open(os.path.join(run_name, "logs.pkl"), "wb"))

    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": {
                k: v.state_dict() for k, v in optimizer.items()
            },
        },
        os.path.join(run_name, "final.pt")
    )
