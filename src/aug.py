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
def add_gaussian(img, sigma):
    if sigma <= 0:
        return img
    noise = torch.randn_like(img) * sigma
    return torch.clamp(img + noise, 0.0, 1.0)


def add_speckle(img, sigma):
    """
    Bright areas get stronger fluctuations, dark areas barely change.
    CT - how many X-ray photons make it through the body and reach the detector. -> Follow Poisson
    """
    if sigma <= 0:
        return img
    noise = torch.randn_like(img) * sigma
    return torch.clamp(img * (1.0 + noise), 0.0, 1.0)


def add_poisson(img, scale=1.0):
    """
    Simulate photon shot noise (Poisson).
    Physically realistic for CT / low-light photon counting.
    scale controls photon count range.
    """
    if scale <= 0:
        return img
    # Ensure image in [0,1], scale up to photon counts
    noisy = torch.poisson(img * scale) / scale
    return torch.clamp(noisy, 0.0, 1.0)


def add_salt_pepper(img, prob=0.05):
    """
    Randomly replaces pixels with 0 or 1.
    prob: probability of any pixel being corrupted.
    """
    if prob <= 0:
        return img
    mask = torch.rand_like(img)
    I_noisy = img.clone()
    I_noisy[mask < prob / 2] = 0.0  # pepper
    I_noisy[mask > 1 - prob / 2] = 1.0  # salt
    return I_noisy


# --------------------------------------------------------
# Main augmentation sampler
# --------------------------------------------------------
def sample_augmentation(cfg, clean_img):
    """
    Returns: I_aug [1,1,H,W] in [0,1]
    """
    pT = cfg["augment"]["prob_translation"]
    pG = cfg["augment"]["prob_gaussian"]
    pS = cfg["augment"]["prob_speckle"]
    pP = cfg["augment"]["prob_poisson"]
    pSP = cfg["augment"]["prob_saltpepper"]

    new_img = clean_img

    # translation
    if random.random() < pT:
        dx, dy = _rand_trans(cfg["augment"]["translation_px_max"])
        new_img = apply_translation(new_img, dx, dy)

    # noises
    if random.random() < pG:
        new_img = add_gaussian(new_img, cfg["augment"]["gaussian_sigma"])
    if random.random() < pS:
        new_img = add_speckle(new_img, cfg["augment"]["speckle_sigma"])
    if random.random() < pP:
        new_img = add_poisson(new_img, cfg["augment"]["poisson_scale"])
    if random.random() < pSP:
        new_img = add_salt_pepper(new_img, cfg["augment"]["saltpepper_prob"])

    return new_img
