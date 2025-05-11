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
    
    for idx in range(min(num_images, grid_size * grid_size)):
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


# def train_step_gan(batch, model, optimizer, feature_matching=False, lambda_fm=10.0):
#     """
#     Performs a step of batch update using the GAN objective with class conditioning.
#     """
#     real_label = 1
#     fake_label = 0
#     loss_fn = nn.BCEWithLogitsLoss()
    
#     # Get real images
#     real_images = batch["train_X"]
#     batch_size = len(real_images)
    
#     # Train discriminator
#     optimizer["discriminator"].zero_grad()
#     label = torch.full((batch_size,), real_label, dtype=torch.float, device=real_images.device)
    
#     # Train with real
#     # if batch.get("use_diffaug", False):
#     #     real_images = DiffAugment(real_images, policy=CIFAR_POLICY)
#     real_preds = model.discriminate(real_images).squeeze(-1)
#     errD_real = loss_fn(real_preds, label)
#     errD_real.backward()
#     D_x = real_preds.mean().item()

#     # Train with fake
#     noise = batch["noise"]
#     fake_images = model.generate(noise, label)
#     # if batch.get("use_diffaug", False):
#     #     fake_images = DiffAugment(fake_images, policy=CIFAR_POLICY)
#     label.fill_(fake_label)
#     fake_preds = model.discriminate(fake_images.detach()).squeeze(-1)
#     errD_fake = loss_fn(fake_preds, label)
#     errD_fake.backward()
#     D_G_z1 = fake_preds.mean().item()
#     errD = errD_real + errD_fake
#     optimizer["discriminator"].step()

#     # Train generator
#     optimizer["generator"].zero_grad()
#     label.fill_(real_label)  # Fake labels are real for generator cost
#     fake_preds = model.discriminate(fake_images).squeeze(-1)
#     errG = loss_fn(fake_preds, label)

#     # Feature matching loss
#     fm_loss = torch.tensor(0.0, device=real_images.device)
#     if feature_matching:
#         real_features = model.discriminator.get_features(real_images)
#         fake_features = model.discriminator.get_features(fake_images)
#         fm_loss = F.mse_loss(fake_features.mean(dim=0), real_features.mean(dim=0))
#         errG = errG + lambda_fm * fm_loss

#     errG.backward()
#     D_G_z2 = fake_preds.mean().item()
#     optimizer["generator"].step()

#     step_info = {
#         "discriminator_loss": errD.item(),
#         "generator_loss": errG.item(),
#         "D_x": D_x,
#         "D_G_z1": D_G_z1,
#         "D_G_z2": D_G_z2,
#         "fake_images": fake_images.detach().cpu().numpy(),
#         "labels": batch["train_y"].detach().cpu().numpy(),
#         "fm_loss": fm_loss.item() if feature_matching else 0.0,
#     }

#     return step_info

def train_step_gan(batch, model, optimizer, feature_matching=False, lambda_fm=10.0):
    """
    Performs a step of batch update using the original GAN objective.
    """
    
    loss_fn = nn.BCEWithLogitsLoss()

    batch_size = len(batch["train_X"])

    # Update discriminator
    noise = batch["noise"]
    fake_images = model.generate(noise, batch["train_y"])
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
        "labels": batch["train_y"].detach().cpu().numpy(),
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
    def weights_init(m):
        classname = m.__class__.__name__
        if classname.find('Conv') != -1:
            m.weight.data.normal_(0.0, 0.02)
        elif classname.find('BatchNorm') != -1:
            m.weight.data.normal_(1.0, 0.02)
            m.bias.data.fill_(0)

    model.generator.apply(weights_init)
    model.discriminator.apply(weights_init)
    
    # Create progress bar
    pbar = tqdm(range(args.num_iterations), desc="Training")
    
    critic_steps = getattr(args, "critic_steps", 1)
    
    for iter_i in pbar:
        try:
            batch = next(loader)
            # Handle different data formats
            if isinstance(batch, dict):  # Pokemon data format
                train_X = batch["image"]
                train_y = batch["type"]
            else:  # CIFAR-10 format
                train_X, train_y = batch
                
        except StopIteration:
            loader = iter(train_loader)
            batch = next(loader)
            # Handle different data formats again
            if isinstance(batch, dict):  # Pokemon data format
                train_X = batch["image"]
                train_y = batch["type"]
            else:  # CIFAR-10 format
                train_X, train_y = batch
            
        real_batch_size = train_X.shape[0]
        
        # Create consistent dictionary format for both datasets
        batch_dict = {
            "train_X": train_X.to(DEVICE),
            "train_y": train_y.to(DEVICE),
            "noise": model.sample_noise(real_batch_size).to(DEVICE),
            "eps": torch.rand(real_batch_size, 1, 1, 1).to(DEVICE),
            "iteration": iter_i,  # Pass iteration count for generator update frequency
            "use_diffaug": getattr(args, "diffaug", False),  # Pass diffaug flag
        }
        
        # Only update generator every critic_steps
        if (iter_i + 1) % critic_steps == 0:
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
            'D_loss': f"{step_info['discriminator_loss']:.4f}",
            'G_loss': f"{step_info['generator_loss']:.4f}"
        }
        if gp:
            progress_info['W_dist'] = f"{step_info['wasserstein_distance']:.4f}"
        pbar.set_postfix(progress_info)

        if (iter_i + 1) % LOG_INTERVAL == 0 or (iter_i + 1) % 25000 == 0:
            # Calculate means only for scalar values
            step_infos = {
                k: (np.mean(v) if v is not None and k not in ['fake_images', 'labels'] else v[-1] if v is not None else None) 
                for k, v in step_infos.items()
            }
            
            for k, v in step_info.items():
                if k == "fake_images" or k == "labels":
                    continue
                logs.setdefault(k, [])
                logs[k].append(v)

            # Save images sampled
            num_images = len(step_info["fake_images"])
            grid_size = math.ceil(math.sqrt(num_images))
            
            # Create and save image grid
            grid_image = create_image_grid(
                step_info["fake_images"],
                grid_size
            )
            grid_image.save(os.path.join(run_name, f"{iter_i + 1}.png"))
            
            if (iter_i + 1) % 25000 == 0:
                # Save checkpoint
                torch.save(
                    {
                        "model_state_dict": model.state_dict(),
                        "optimizer_state_dict": {
                            k: v.state_dict() for k, v in optimizer.items()
                        },
                    },
                    os.path.join(run_name, f"checkpoint_{iter_i + 1}.pt")
                )
            
            # Clear step_infos
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
        },
        os.path.join(run_name, "final.pt")
    )


def train_curriculum(train_loader, model, optimizer, args, run_name):
    """
    Runs the training loop with curriculum learning - training one type at a time.
    """
    gen = torch.Generator()
    gen.manual_seed(SPLIT_SEED)

    batch_size = train_loader.batch_size  # Use the batch size from the dataloader
    epochs_per_type = args.epochs_per_type  # Number of epochs to train on each type
    
    logs = dict()
    step_infos = {
        "discriminator_loss": [],
        "generator_loss": [],
        "gp_loss": [] if getattr(args, "gradient_penalty", None) else None,
        "fake_images": None,  # Store only the latest
        "labels": None,       # Store only the latest
    }
    
    # Train on each type sequentially
    for type_idx, pokemon_type in enumerate(POKEMON_TYPES):
        print(f"\nTraining on type: {pokemon_type} ({type_idx + 1}/{len(POKEMON_TYPES)})")
        
        # Collect all images for current type
        type_images = []
        type_labels = []
        
        # Collect all samples for current type
        for batch in train_loader:
            mask = (batch['type'] == type_idx)
            if mask.sum() > 0:
                type_images.append(batch['image'][mask])
                type_labels.append(batch['type'][mask])
        
        if not type_images:
            print(f"No samples found for type {pokemon_type}, skipping...")
            continue
            
        # Concatenate all samples
        type_images = torch.cat(type_images, dim=0)
        type_labels = torch.cat(type_labels, dim=0)
        num_samples = len(type_images)
        
        if num_samples < batch_size:
            print(f"Warning: Only {num_samples} samples for type {pokemon_type}, which is less than batch size {batch_size}")
            continue
            
        # Calculate number of batches
        num_batches = num_samples // batch_size
        if num_batches == 0:
            continue
            
        print(f"Found {num_samples} samples for type {pokemon_type}, creating {num_batches} batches of size {batch_size}")
        
        # Train on current type for several epochs
        total_steps = epochs_per_type * num_batches
        pbar = tqdm(range(total_steps), desc=f"Training {pokemon_type}")
        
        for epoch in range(epochs_per_type):
            # Shuffle indices for this epoch
            indices = torch.randperm(num_samples)
            
            for batch_idx in range(num_batches):
                # Get batch indices
                start_idx = batch_idx * batch_size
                end_idx = start_idx + batch_size
                batch_indices = indices[start_idx:end_idx]
                
                # Create batch
                batch_images = type_images[batch_indices]
                batch_labels = type_labels[batch_indices]
                
                step_info = train_step_gan(
                    {
                        "train_X": batch_images.to(DEVICE),
                        "train_y": batch_labels.to(DEVICE),
                        "noise": model.sample_noise(batch_size).to(DEVICE),
                        "eps": torch.rand(batch_size, 1, 1, 1).to(DEVICE),
                        "use_diffaug": getattr(args, "use_diffaug", False),
                    },
                    model,
                    optimizer,
                )
                
                # Update progress bar
                pbar.set_postfix({
                    'D_loss': f"{step_info['discriminator_loss']:.4f}",
                    'G_loss': f"{step_info['generator_loss']:.4f}",
                    'Epoch': f"{epoch + 1}/{epochs_per_type}"
                })
                pbar.update(1)
                
                # Store metrics
                step_infos["discriminator_loss"].append(step_info["discriminator_loss"])
                step_infos["generator_loss"].append(step_info["generator_loss"])
                step_infos["fake_images"] = step_info["fake_images"]
                step_infos["labels"] = step_info["labels"]
                
                # Save progress periodically
                if len(step_infos["discriminator_loss"]) % LOG_INTERVAL == 0:
                    # Calculate means for scalar values
                    step_infos = {
                        k: (np.mean(v) if v is not None and k not in ['fake_images', 'labels'] else v)
                        for k, v in step_infos.items()
                    }
                    
                    for k, v in step_info.items():
                        if k == "fake_images" or k == "labels":
                            continue
                        logs.setdefault(k, [])
                        logs[k].append(v)
                    
                    # Save images
                    num_images = len(step_info["fake_images"])
                    grid_size = math.ceil(math.sqrt(num_images))
                    grid_image = create_image_grid(
                        step_info["fake_images"],
                        grid_size
                    )
                    grid_image.save(os.path.join(run_name, f"{pokemon_type}_{len(logs['discriminator_loss'])}.png"))
                    
                    # Save checkpoint
                    torch.save(
                        {
                            "model_state_dict": model.state_dict(),
                            "optimizer_state_dict": {
                                k: v.state_dict() for k, v in optimizer.items()
                            },
                            "current_type": type_idx,
                            "current_epoch": epoch,
                        },
                        os.path.join(run_name, f"latest_{pokemon_type}.pt")
                    )
                    
                    # Reset step_infos
                    step_infos = {
                        "discriminator_loss": [],
                        "generator_loss": [],
                        "gp_loss": [] if getattr(args, "gradient_penalty", None) else None,
                        "fake_images": None,
                        "labels": None,
                    }
        
        # Save final model for this type
        torch.save(
            {
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": {
                    k: v.state_dict() for k, v in optimizer.items()
                },
                "current_type": type_idx,
            },
            os.path.join(run_name, f"final_{pokemon_type}.pt")
        )
    
    # Save final model after all types
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": {
                k: v.state_dict() for k, v in optimizer.items()
            },
            "current_type": len(POKEMON_TYPES),
        },
        os.path.join(run_name, "final.pt")
    )
    
    pickle.dump(logs, open(os.path.join(run_name, "logs.pkl"), "wb"))


def train_stochastic_curriculum(train_loader, model, optimizer, args, run_name):
    """
    Runs the training loop with stochastic curriculum learning - randomly picking types to train on.
    """
    gen = torch.Generator()
    gen.manual_seed(SPLIT_SEED)

    batch_size = train_loader.batch_size
    iterations_per_type = args.iterations_per_type
    
    logs = dict()
    step_infos = {
        "discriminator_loss": [],
        "generator_loss": [],
        "gp_loss": [] if getattr(args, "gradient_penalty", None) else None,
        "fake_images": None,
        "labels": None,
    }

    # First, collect samples for each type
    type_data = {}
    for type_idx, pokemon_type in enumerate(POKEMON_TYPES):
        type_images = []
        type_labels = []
        
        for batch in train_loader:
            mask = (batch['type'] == type_idx)
            if mask.sum() > 0:
                type_images.append(batch['image'][mask])
                type_labels.append(batch['type'][mask])
        
        if type_images:
            type_images = torch.cat(type_images, dim=0)
            type_labels = torch.cat(type_labels, dim=0)
            num_samples = len(type_images)
            
            if num_samples >= batch_size:
                type_data[type_idx] = {
                    'images': type_images,
                    'labels': type_labels,
                    'num_samples': num_samples,
                    'name': pokemon_type
                }
                print(f"Type {pokemon_type}: {num_samples} samples")
    
    if not type_data:
        raise ValueError("No types have enough samples for training!")

    # Create progress bar for total iterations
    total_iterations = args.num_iterations
    pbar = tqdm(range(total_iterations), desc="Training")
    
    current_type_idx = None
    iterations_on_current_type = 0
    
    for iter_i in pbar:
        # Check if we need to switch to a new type
        if current_type_idx is None or iterations_on_current_type >= iterations_per_type:
            # Randomly select a new type from available types
            available_types = list(type_data.keys())
            # Don't pick the same type twice in a row if possible
            if current_type_idx is not None and len(available_types) > 1:
                available_types.remove(current_type_idx)
            current_type_idx = np.random.choice(available_types)
            current_type = type_data[current_type_idx]
            print(f"\nSwitching to type: {current_type['name']}")
            iterations_on_current_type = 0

        # Get random batch for current type
        num_samples = current_type['num_samples']
        indices = torch.randperm(num_samples)[:batch_size]
        batch_images = current_type['images'][indices]
        batch_labels = current_type['labels'][indices]
        
        step_info = train_step_gan(
            {
                "train_X": batch_images.to(DEVICE),
                "train_y": batch_labels.to(DEVICE),
                "noise": model.sample_noise(batch_size).to(DEVICE),
                "eps": torch.rand(batch_size, 1, 1, 1).to(DEVICE),
                "use_diffaug": getattr(args, "use_diffaug", False),
            },
            model,
            optimizer,
        )
        
        # Update progress information
        iterations_on_current_type += 1
        remaining_iterations = iterations_per_type - iterations_on_current_type
        
        # Update progress bar
        pbar.set_postfix({
            'D_loss': f"{step_info['discriminator_loss']:.4f}",
            'G_loss': f"{step_info['generator_loss']:.4f}",
            'Type': current_type['name'],
            'Remaining': remaining_iterations
        })
        
        # Store metrics
        step_infos["discriminator_loss"].append(step_info["discriminator_loss"])
        step_infos["generator_loss"].append(step_info["generator_loss"])
        step_infos["fake_images"] = step_info["fake_images"]
        step_infos["labels"] = step_info["labels"]
        
        # Save progress periodically
        if (iter_i + 1) % LOG_INTERVAL == 0:
            # Calculate means for scalar values
            step_infos = {
                k: (np.mean(v) if v is not None and k not in ['fake_images', 'labels'] else v)
                for k, v in step_infos.items()
            }
            
            for k, v in step_info.items():
                if k == "fake_images" or k == "labels":
                    continue
                logs.setdefault(k, [])
                logs[k].append(v)
            
            # Save images
            num_images = len(step_info["fake_images"])
            grid_size = math.ceil(math.sqrt(num_images))
            grid_image = create_image_grid(
                step_info["fake_images"],
                grid_size
            )
            grid_image.save(os.path.join(run_name, f"iter_{iter_i + 1}.png"))
            
            # Save checkpoint
            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": {
                        k: v.state_dict() for k, v in optimizer.items()
                    },
                    "current_type_idx": current_type_idx,
                    "iteration": iter_i,
                },
                os.path.join(run_name, "latest.pt")
            )
            
            # Reset step_infos
            step_infos = {
                "discriminator_loss": [],
                "generator_loss": [],
                "gp_loss": [] if getattr(args, "gradient_penalty", None) else None,
                "fake_images": None,
                "labels": None,
            }
    
    # Save final model
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": {
                k: v.state_dict() for k, v in optimizer.items()
            },
            "iteration": total_iterations,
        },
        os.path.join(run_name, "final.pt")
    )
    
    pickle.dump(logs, open(os.path.join(run_name, "logs.pkl"), "wb"))
