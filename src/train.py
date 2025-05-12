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
from tqdm import tqdm
from PIL import Image, ImageDraw, ImageFont
import torchvision.utils as vutils

from functools import partial
from torch import autograd
from pokemon_data import POKEMON_TYPES
from src.diffaugment import DiffAugment


LOG_INTERVAL = 1000
SPLIT_SEED = 0
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# DiffAugment policies
CIFAR_POLICY = 'color,translation,cutout'
POKEMON_POLICY = 'color,translation, cutout' 
CELEBA_POLICY = 'color,translation,cutout'


def denormalize_image(image):
    """
    Denormalize images from [-1, 1] range back to [0, 1]
    Inverse of transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    """
    return (image + 1) / 2


def create_image_grid(images, grid_size):
    """Create a grid of images using PIL without class labels"""
    num_images = len(images)
    cell_size = images[0].shape[1]  # Assuming square images
    
    # Calculate grid dimensions
    grid_width = grid_size * cell_size
    grid_height = grid_size * cell_size
    
    # Create a white background image
    grid_image = Image.new('RGB', (grid_width, grid_height), 'white')
    
    # Ensure we have a perfect square grid
    num_cells = grid_size * grid_size
    for idx in range(min(num_images, num_cells)):
        row = idx // grid_size
        col = idx % grid_size
        
        # Convert numpy array to PIL Image
        img_data = images[idx]
        # Normalize from [-1, 1] to [0, 1] range
        img_data = (img_data + 1) / 2
        
        # Handle both single-channel and three-channel images
        if img_data.shape[0] == 1:  # Single channel (grayscale)
            # Convert to uint8 range [0, 255]
            img_data = (img_data[0] * 255).astype(np.uint8)
            cell_image = Image.fromarray(img_data, mode='L')
            # Convert to RGB for consistency
            cell_image = cell_image.convert('RGB')
        else:  # Three channels (RGB)
            # Convert to uint8 range [0, 255]
            img_data = (img_data.transpose(1, 2, 0) * 255).astype(np.uint8)
            cell_image = Image.fromarray(img_data)
        
        # Calculate position
        x_pos = col * cell_size
        y_pos = row * cell_size
        
        # Paste the image into the grid
        grid_image.paste(cell_image, (x_pos, y_pos))
    
    return grid_image


def train_step_gan(batch, model, optimizer, feature_matching=False, lambda_fm=1.0):
    """
    Performs a step of batch update using the GAN objective with class conditioning.
    """
    real_label = 0.9
    fake_label = 0.1
    loss_fn = nn.BCEWithLogitsLoss()
    # Get real images
    real_images = batch["train_X"]
    batch_size = len(real_images)
    
    # Train discriminator
    optimizer["discriminator"].zero_grad()
    label = torch.full((batch_size,), real_label, dtype=torch.float, device=real_images.device)
    
    # Train with real
    if batch.get("use_diffaug", False):
        real_images = DiffAugment(real_images, policy=CIFAR_POLICY)
    real_preds = model.discriminate(real_images).squeeze(-1)
    errD_real = loss_fn(real_preds, label)
    errD_real.backward()
    D_x = real_preds.mean().item()

    # Train with fake
    noise = batch["noise"]
    fake_images = model.generate(noise, label)
    if batch.get("use_diffaug", False):
        fake_images = DiffAugment(fake_images, policy=CIFAR_POLICY)
    label.fill_(fake_label)
    fake_preds = model.discriminate(fake_images.detach()).squeeze(-1)
    errD_fake = loss_fn(fake_preds, label)
    errD_fake.backward()
    D_G_z1 = fake_preds.mean().item()
    errD = errD_real + errD_fake
    torch.nn.utils.clip_grad_norm_(model.discriminator.parameters(), max_norm=1.0)
    optimizer["discriminator"].step()

    # Train generator
    optimizer["generator"].zero_grad()
    label.fill_(real_label)  # Fake labels are real for generator cost
    fake_preds = model.discriminate(fake_images).squeeze(-1)
    errG = loss_fn(fake_preds, label)

    # Feature matching loss
    fm_loss = torch.tensor(0.0, device=real_images.device)
    if feature_matching:
        real_features = model.discriminator.get_features(real_images)
        fake_features = model.discriminator.get_features(fake_images)
        fm_loss = F.mse_loss(fake_features.mean(dim=0), real_features.mean(dim=0))
        errG = errG + lambda_fm * fm_loss

    errG.backward()
    D_G_z2 = fake_preds.mean().item()
    torch.nn.utils.clip_grad_norm_(model.generator.parameters(), max_norm=1.0)
    optimizer["generator"].step()

    step_info = {
        "discriminator_loss": errD.item(),
        "generator_loss": errG.item(),
        "D_x": D_x,
        "D_G_z1": D_G_z1,
        "D_G_z2": D_G_z2,
        "fake_images": fake_images.detach().cpu().numpy(),
        "labels": batch["train_y"].detach().cpu().numpy(),
        "fm_loss": fm_loss.item() if feature_matching else 0.0,
    }

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


def train_step_wgan(batch, model, optimizer, gp, update_generator=True):
    """
    Performs a step of batch update using the Wasserstein GAN objective with gradient penalty.
    """
    # Handle different data formats

    images = batch["train_X"]
    labels = batch["train_y"]
    eps = batch["eps"]

    
    batch_size = len(images)

    # Update discriminator/critic
    noise = batch["noise"]
    fake_images = model.generate(noise, labels)

    # First compute critic scores
    fake_scores = model.discriminate(fake_images.detach())
    real_scores = model.discriminate(images)

    # Wasserstein loss for critic
    optimizer["discriminator"].zero_grad()
    
    # Critic tries to maximize: E[D(real)] - E[D(fake)]
    wasserstein_distance = real_scores.mean() - fake_scores.mean()
    disc_loss = -wasserstein_distance  # Negate because we minimize
    
    # Compute gradient penalty
    gp_loss = compute_gradient_penalty(model, images, fake_images.detach(), eps)
    
    # Total critic loss with gradient penalty
    total_loss = disc_loss + gp * gp_loss
    total_loss.backward()
    
    # Clip gradients for stability
    torch.nn.utils.clip_grad_norm_(model.discriminator.parameters(), max_norm=1.0)
    optimizer["discriminator"].step()
    gen_loss = torch.tensor(0.0, device=images.device)
    if update_generator:
        # Update generator with diversity-promoting techniques
        noise = model.sample_noise(batch_size).to(images.device)
                
        fake_images = model.generate(noise, labels)
        fake_scores = model.discriminate(fake_images)
        
        optimizer["generator"].zero_grad()
        
        # Basic generator loss
        gen_loss = -fake_scores.mean()
        
        gen_loss.backward()
        
        # Clip gradients for stability
        torch.nn.utils.clip_grad_norm_(model.generator.parameters(), max_norm=1.0)
        optimizer["generator"].step()

    step_info = {
        "discriminator_loss": disc_loss.detach().cpu().item(),
        "generator_loss": gen_loss.detach().cpu().item(),
        "wasserstein_distance": wasserstein_distance.detach().cpu().item(),
        "gp_loss": gp_loss.detach().cpu().item(),
        "fake_images": fake_images.detach().cpu().numpy(),
        "labels": labels.detach().cpu().numpy(),
    }

    return step_info


def train(train_loader, model, optimizer, args, run_name):
    """
    Runs the training loop.
    """
    # Make sure to split the same way
    gen = torch.Generator()
    gen.manual_seed(SPLIT_SEED)

    # Create necessary directories
    epoch_images_dir = os.path.join(run_name, "epoch_images")
    fixed_noise_dir = os.path.join(run_name, "fixed_noise")
    checkpoints_dir = os.path.join(run_name, "checkpoints")
    os.makedirs(epoch_images_dir, exist_ok=True)
    os.makedirs(fixed_noise_dir, exist_ok=True)
    os.makedirs(checkpoints_dir, exist_ok=True)

    batch_size = 64
    loader = iter(train_loader)

    gp = getattr(args, "gradient_penalty", None)
    feature_matching = getattr(args, "feature_matching", False)
    if gp is None or gp == 0.0:
        train_step = partial(train_step_gan, feature_matching=feature_matching)
    else:
        train_step = partial(train_step_wgan, gp=gp)

    logs = dict()
    step_infos = {
        "discriminator_loss": [],
        "generator_loss": [],
        "wasserstein_distance": [] if gp else None,
        "gp_loss": [] if gp else None,
        "fake_images": None,  # Store only the latest
        "labels": None,       # Store only the latest
    }

    # Only initialize weights if we use DC GAN
    if args.model == "CNN":
        def weights_init(m):
            classname = m.__class__.__name__
            if classname.find('Conv') != -1:
                m.weight.data.normal_(0.0, 0.02)
            elif classname.find('BatchNorm') != -1:
                m.weight.data.normal_(1.0, 0.02)
                m.bias.data.fill_(0)

        model.generator.apply(weights_init)
        model.discriminator.apply(weights_init)
        print("Initialized weights")
    
    # Create progress bar for epochs
    pbar = tqdm(range(args.num_epochs), desc="Training")
    
    critic_steps = getattr(args, "critic_steps", 1)
    total_steps = 0
    steps_per_epoch = 0  # Will be set after first epoch

    # Generate fixed noise for consistent image generation
    grid_size = 8  # Fixed grid size of 8x8
    num_grid_images = grid_size * grid_size  # 64 images for a perfect square grid
    fixed_noise = model.sample_noise(num_grid_images).to(DEVICE)
    fixed_labels = torch.randint(0, 10, (num_grid_images,), device=DEVICE)  # Assuming 10 classes, adjust if needed
    
    for epoch in pbar:
        # Reset the dataloader for each epoch
        loader = iter(train_loader)
        epoch_steps = 0
        
        while True:
            try:
                batch = next(loader)
                # Handle different data formats
                if isinstance(batch, dict):  # Pokemon data format
                    train_X = batch["image"]
                    train_y = batch["type"]
                else:  # CIFAR-10 format
                    train_X, train_y = batch
                    
            except StopIteration:
                    # Set steps_per_epoch after first epoch
                    if epoch == 0:
                        steps_per_epoch = epoch_steps
                    break  # End of epoch
                
            real_batch_size = train_X.shape[0]
            
            # Create consistent dictionary format for both datasets
            batch_dict = {
                "train_X": train_X.to(DEVICE),
                "train_y": train_y.to(DEVICE),
                "noise": model.sample_noise(real_batch_size).to(DEVICE),
                "eps": torch.rand(real_batch_size, 1, 1, 1).to(DEVICE),
                    "iteration": total_steps,  # Pass iteration count for generator update frequency
                "use_diffaug": getattr(args, "diffaug", False),  # Pass diffaug flag
            }
            
            # Only update generator every critic_steps
            if (total_steps + 1) % critic_steps == 0:
                # Standard GAN: update both D and G in train_step
                step_info = train_step(batch_dict, model, optimizer)
            else:
                # Only update discriminator/critic
                if hasattr(train_step, 'func') and train_step.func.__name__ == 'train_step_wgan':
                    # For WGAN, pass update_generator=False
                    step_info = train_step(batch_dict, model, optimizer, update_generator=False)
                else:
                    # For standard GAN, skip generator update (not typical, but for consistency)
                    step_info = train_step(batch_dict, model, optimizer)
            
            # Only store scalar values in lists
            step_infos["discriminator_loss"].append(step_info["discriminator_loss"])
            step_infos["generator_loss"].append(step_info["generator_loss"])
            if gp:
                step_infos["wasserstein_distance"].append(step_info["wasserstein_distance"])
                step_infos["gp_loss"].append(step_info["gp_loss"])
            
            # Store only latest images and labels
            step_infos["fake_images"] = step_info["fake_images"]
            step_infos["labels"] = step_info["labels"]

            # Update progress bar with losses and Wasserstein distance if using WGAN
            progress_info = {
                    'Iter': f"{epoch_steps + 1}/{steps_per_epoch if steps_per_epoch > 0 else '?'}",
                'D_loss': f"{step_info['discriminator_loss']:.4f}",
                'G_loss': f"{step_info['generator_loss']:.4f}"
            }
            if gp:
                progress_info['W_dist'] = f"{step_info['wasserstein_distance']:.4f}"
            pbar.set_postfix(progress_info)

            total_steps += 1
            epoch_steps += 1

        # End of epoch - save images and checkpoints
        # Calculate means for the epoch
        epoch_stats = {
                k: (np.mean(v) if v is not None and k not in ['fake_images', 'labels'] else v[-1] if v is not None else None) 
                for k, v in step_infos.items()
            }
            
        # Save epoch statistics
        for k, v in epoch_stats.items():
                if k == "fake_images" or k == "labels":
                    continue
                logs.setdefault(k, [])
                logs[k].append(v)

        # Log and save images every k epochs
        if (epoch + 1) % args.log_every == 0:
            # Generate a new batch of images for the grid
            with torch.no_grad():
                grid_noise = model.sample_noise(num_grid_images).to(DEVICE)
                grid_labels = torch.randint(0, 10, (num_grid_images,), device=DEVICE)
                grid_images = model.generate(grid_noise, grid_labels)
            
                # Save training images
                grid_image = create_image_grid(
                    grid_images.cpu().numpy(),
                    grid_size
                )
                grid_image.save(os.path.join(epoch_images_dir, f"epoch_{epoch + 1}.png"))

                # Generate and save fixed noise images
                fixed_images = model.generate(fixed_noise, fixed_labels)
                fixed_grid = create_image_grid(
                    fixed_images.cpu().numpy(),
                grid_size
            )
                fixed_grid.save(os.path.join(fixed_noise_dir, f"epoch_{epoch + 1}.png"))
            
        # Save checkpoint every 25 epochs
        if (epoch + 1) % 25 == 0:
                torch.save(
                    {
                        "model_state_dict": model.state_dict(),
                        "optimizer_state_dict": {
                            k: v.state_dict() for k, v in optimizer.items()
                        },
                    "epoch": epoch,
                    "total_steps": total_steps,
                    },
                os.path.join(checkpoints_dir, f"checkpoint_epoch_{epoch + 1}.pt")
                )
            
        # Clear step_infos for next epoch
        step_infos = {
                "discriminator_loss": [],
                "generator_loss": [],
                "wasserstein_distance": [] if gp else None,
                "gp_loss": [] if gp else None,
                "fake_images": None,  # Store only the latest
                "labels": None,       # Store only the latest
            }

        pickle.dump(logs, open(os.path.join(run_name, "logs.pkl"), "wb"))

    # Save final model
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": {
                k: v.state_dict() for k, v in optimizer.items()
            },
            "epoch": args.num_epochs,
            "total_steps": total_steps,
        },
        os.path.join(checkpoints_dir, "final.pt")
    )

    # Create GIFs from the saved images
    try:
        from PIL import Image
        import imageio

        # Create GIF from epoch images
        epoch_images = []
        for i in range(0, args.num_epochs, args.log_every):
            img_path = os.path.join(epoch_images_dir, f"epoch_{i + 1}.png")
            if os.path.exists(img_path):
                epoch_images.append(imageio.imread(img_path))
        imageio.mimsave(os.path.join(run_name, 'training_progress.gif'), epoch_images, fps=5)

        # Create GIF from fixed noise images
        fixed_images = []
        for i in range(0, args.num_epochs, args.log_every):
            img_path = os.path.join(fixed_noise_dir, f"epoch_{i + 1}.png")
            if os.path.exists(img_path):
                fixed_images.append(imageio.imread(img_path))
        imageio.mimsave(os.path.join(run_name, 'fixed_noise_progress.gif'), fixed_images, fps=5)
    except ImportError:
        print("Warning: imageio not installed. Could not create GIFs.")
