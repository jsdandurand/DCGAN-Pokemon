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

from src.train import compute_gradient_penalty, train_step_wgan

import numpy as np
import torch
import torch.nn as nn

from pprint import pprint

BATCH_SIZE = 3
IMG_DIM = (3, 10, 10)
FLATTENED_IMG_DIM = int(np.prod(IMG_DIM))
NOISE_DIM = 5


class MockGAN(nn.Module):
    def __init__(self):
        super().__init__()

        self.discriminator = nn.Sequential(
            nn.Linear(FLATTENED_IMG_DIM, FLATTENED_IMG_DIM),
            nn.Linear(FLATTENED_IMG_DIM, 2),
        )

        self.generator = nn.Sequential(
            nn.Linear(NOISE_DIM, FLATTENED_IMG_DIM),
            nn.Linear(FLATTENED_IMG_DIM, FLATTENED_IMG_DIM),
        )

    def generate(self, noise):
        return self.generator(noise).reshape(BATCH_SIZE, *IMG_DIM)
    
    def discriminate(self, sample):
        sample = sample.reshape(-1, FLATTENED_IMG_DIM)
        return self.discriminator(sample)
    
    def sample_noise(self, batch_size):
        return torch.randn(batch_size, NOISE_DIM)


def test_compute_gradient_penalty(test_data):
    model = MockGAN()
    model.load_state_dict(
        torch.load(
            os.path.join(currentdir, "mock_model.pt"),
            weights_only=True,
        )
    )

    loss = compute_gradient_penalty(
        model,
        test_data["train_X"],
        test_data["fake_samples"],
        test_data["eps"],
    )
    return loss

def test_train_step_wgan_no_gp(test_data):
    model = MockGAN()
    model.load_state_dict(
        torch.load(
            os.path.join(currentdir, "mock_model.pt"),
            weights_only=True,
        )
    )
    optimizer = {
        "discriminator": torch.optim.SGD(model.discriminator.parameters()),
        "generator": torch.optim.SGD(model.generator.parameters()),
    }
    step_info = train_step_wgan(test_data, model, optimizer, gp=0.0)
    return step_info


def test_train_step_wgan_with_gp(test_data):
    model = MockGAN()
    model.load_state_dict(
        torch.load(
            os.path.join(currentdir, "mock_model.pt"),
            weights_only=True,
        )
    )
    optimizer = {
        "discriminator": torch.optim.SGD(model.discriminator.parameters()),
        "generator": torch.optim.SGD(model.generator.parameters()),
    }
    step_info = train_step_wgan(test_data, model, optimizer, gp=1.0)
    test_data["wgan_without_gp"] = step_info
    return step_info


if __name__ == "__main__":
    test_data = torch.load(
        os.path.join(currentdir, "test_train_data.pt"),
        weights_only=False,
    )
    gp_result = test_compute_gradient_penalty(test_data)

    print("Test for compute_gradient_penalty")
    print("Correct: {}".format(
        torch.allclose(gp_result, test_data["compute_gradient_penalty"]))
    )

    wgan_result = test_train_step_wgan_no_gp(test_data)
    print("Test for test_train_step_wgan without gradient penalty")
    matches = {
        k: np.allclose(
            wgan_result[k], test_data["wgan_without_gp"][k]
        ) for k in wgan_result
    }
    print("Correct: {}".format(np.all(matches.values())))
    if np.mean(list(matches.values())) < 1:
        pprint(matches)

    wgan_gp_result = test_train_step_wgan_with_gp(test_data)
    print("Test for test_train_step_wgan with gradient penalty")
    matches = {
        k: np.allclose(
            wgan_gp_result[k], test_data["wgan_with_gp"][k]
        ) for k in wgan_gp_result
    }
    print("Correct: {}".format(np.all(matches.values())))
    if np.mean(list(matches.values())) < 1:
        pprint(matches)
