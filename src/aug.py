import random

import torch
import torch.nn.functional as F


# --------------------------------------------------------
# Basic spatial augmentation
# --------------------------------------------------------
# TODO: add rotation?
def _rand_trans(max_px):
    dx = random.randint(-max_px, max_px)
    dy = random.randint(-max_px, max_px)
    return dx, dy


def apply_translation(img, dx, dy):
    # img [1,1,H,W], pixels in [0,1]
    B, C, H, W = img.shape
    # normalize offsets to [-1,1] for grid_sample
    tx = 2.0 * dx / (W - 1)
    ty = 2.0 * dy / (H - 1)
    yy, xx = torch.meshgrid(
        torch.linspace(-1, 1, H, device=img.device),
        torch.linspace(-1, 1, W, device=img.device),
        indexing="ij",
    )
    grid = torch.stack([xx - tx, yy - ty], dim=-1).unsqueeze(
        0
    )  # sample from shifted coords
    aug = F.grid_sample(
        img, grid, mode="bilinear", align_corners=True, padding_mode="border"
    )  # Use edge pixels for padding
    return aug


# --------------------------------------------------------
# Noise augmentations
# --------------------------------------------------------
def add_gaussian(img, sigma, mean=0.0):
    """
    Additive Gaussian noise.
    Allows non-zero mean to simulate bias (e.g. electronic baseline drift).
    """
    if sigma <= 0:
        return img
    noise = torch.randn_like(img) * sigma + mean
    return torch.clamp(img + noise, 0.0, 1.0)


def add_speckle(img, sigma, mean=0.0):
    """
    Multiplicative (signal-dependent) noise.
    Bright areas fluctuate more strongly.
    """
    if sigma <= 0:
        return img
    noise = torch.randn_like(img) * sigma + mean  # non-zero mean possible
    return torch.clamp(img * (1.0 + noise), 0.0, 1.0)


def add_poisson(img, scale=1.0, bias=0.0):
    """
    Poisson (shot) noise with optional bias.
    The bias term simulates uneven illumination or detector offset.
    """
    if scale <= 0:
        return img
    noisy = torch.poisson(torch.clamp(img * scale, 0, None)) / scale
    return torch.clamp(noisy + bias, 0.0, 1.0)


def add_salt_pepper(img, prob=0.05, low_val=0.0, high_val=1.0):
    """
    Impulsive noise (salt & pepper).
    Can be asymmetric if low_val != 0 or high_val != 1.
    """
    if prob <= 0:
        return img
    mask = torch.rand_like(img)
    I_noisy = img.clone()
    I_noisy[mask < prob / 2] = low_val
    I_noisy[mask > 1 - prob / 2] = high_val
    return I_noisy


# --------------------------------------------------------
# Main augmentation sampler
# --------------------------------------------------------
def sample_augmentation(cfg, clean_img):
    """
    Returns: I_aug [1,1,H,W] in [0,1]
    """
    aug_cfg = cfg["augment"]
    new_img = clean_img

    # translation
    if random.random() < aug_cfg["prob_translation"]:
        dx, dy = _rand_trans(aug_cfg["translation_px_max"])
        new_img = apply_translation(new_img, dx, dy)

    # noises
    if random.random() < aug_cfg["prob_gaussian"]:
        new_img = add_gaussian(
            new_img,
            sigma=aug_cfg["gaussian_sigma"],
            mean=aug_cfg.get("gaussian_mean", 0.0),
        )
    if random.random() < aug_cfg["prob_speckle"]:
        new_img = add_speckle(
            new_img,
            sigma=aug_cfg["speckle_sigma"],
            mean=aug_cfg.get("speckle_mean", 0.0),
        )
    if random.random() < aug_cfg["prob_poisson"]:
        new_img = add_poisson(
            new_img,
            scale=aug_cfg["poisson_scale"],
            bias=aug_cfg.get("poisson_bias", 0.0),
        )
    if random.random() < aug_cfg["prob_saltpepper"]:
        new_img = add_salt_pepper(
            new_img,
            prob=aug_cfg["saltpepper_prob"],
            low_val=aug_cfg.get("saltpepper_low", 0.0),
            high_val=aug_cfg.get("saltpepper_high", 1.0),
        )

    return new_img
