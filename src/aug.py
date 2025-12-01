import random
import torch
import torch.nn as nn
import torch.nn.functional as F

# --------------------------------------------------------
# Grid Generation Helpers
# --------------------------------------------------------
def _get_identity_grid(H, W, device):
    yy, xx = torch.meshgrid(
        torch.linspace(-1, 1, H, device=device),
        torch.linspace(-1, 1, W, device=device),
        indexing="ij",
    )
    grid = torch.stack([xx, yy], dim=-1).unsqueeze(0)  # [1, H, W, 2]
    return grid

def _get_rotation_grid(grid, degrees, device):
    theta = torch.deg2rad(torch.tensor(degrees, device=device, dtype=torch.float32))
    cos_t, sin_t = torch.cos(theta), torch.sin(theta)
    rot_matrix = torch.tensor(
        [[cos_t, sin_t], [-sin_t, cos_t]], device=device, dtype=torch.float32
    )
    return grid @ rot_matrix

def _get_shear_grid(grid, shear_x, shear_y, device):
    shear_matrix = torch.tensor(
        [[1, shear_y], [shear_x, 1]], device=device, dtype=torch.float32
    )
    return grid @ shear_matrix

def _get_nonlinear_warp_grid(grid, warp_scale, warp_resolution, device):
    B, H, W, _ = grid.shape
    low_res_noise = (
        torch.rand(B, 2, warp_resolution, warp_resolution, device=device) - 0.5
    ) * 2 * warp_scale
    warp_field = F.interpolate(low_res_noise, (H, W), mode="bilinear", align_corners=True)
    warp_field = warp_field.permute(0, 2, 3, 1)
    return grid + warp_field

def _get_translation_grid(grid, dx, dy, device):
    _, H, W, _ = grid.shape
    tx = 2.0 * dx / (W - 1)
    ty = 2.0 * dy / (H - 1)
    trans_vec = torch.tensor([tx, ty], device=device).view(1, 1, 1, 2)
    return grid - trans_vec

# --------------------------------------------------------
# Noise augmentations
# --------------------------------------------------------
def add_gaussian(img, sigma, mean=0.0):
    if sigma <= 0: return img
    noise = torch.randn_like(img) * sigma + mean
    return torch.clamp(img + noise, 0.0, 1.0)

def add_speckle(img, sigma, mean=0.0):
    if sigma <= 0: return img
    noise = torch.randn_like(img) * sigma + mean
    return torch.clamp(img * (1.0 + noise), 0.0, 1.0)

def add_poisson(img, scale=1.0, bias=0.0):
    if scale <= 0: return img
    noisy = torch.poisson(torch.clamp(img * scale, 0, None)) / scale
    return torch.clamp(noisy + bias, 0.0, 1.0)

def add_salt_pepper(img, prob=0.05, low_val=0.0, high_val=1.0):
    if prob <= 0: return img
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
    Returns:
        noisy_img: Augmented image tensor
        aug_params: Dictionary containing applied parameters (dx, dy, etc.)
    """
    aug_cfg = cfg["augment"]
    device = clean_img.device
    B, C, H, W = clean_img.shape
    
    # Store parameters here
    aug_params = {
        "dx": 0, 
        "dy": 0, 
        "deg": 0.0,
        "shear_x": 0.0,
        "shear_y": 0.0
    }

    # --- 1. Spatial ---
    grid = _get_identity_grid(H, W, device)

    if random.random() < aug_cfg.get("prob_rotation", 0.0):
        deg = random.uniform(
            -aug_cfg.get("rotation_deg_max", 0.0), 
            aug_cfg.get("rotation_deg_max", 0.0)
        )
        aug_params["deg"] = deg
        grid = _get_rotation_grid(grid, deg, device)
        
    if random.random() < aug_cfg.get("prob_shear", 0.0):
        sx = random.uniform(-aug_cfg.get("shear_factor_max", 0.0), aug_cfg.get("shear_factor_max", 0.0))
        sy = random.uniform(-aug_cfg.get("shear_factor_max", 0.0), aug_cfg.get("shear_factor_max", 0.0))
        aug_params["shear_x"] = sx
        aug_params["shear_y"] = sy
        grid = _get_shear_grid(grid, sx, sy, device)

    if random.random() < aug_cfg.get("prob_warp", 0.0):
        grid = _get_nonlinear_warp_grid(
            grid,
            aug_cfg.get("warp_scale", 0.1),
            aug_cfg.get("warp_resolution", 16),
            device
        )
        
    if random.random() < aug_cfg.get("prob_translation", 0.0):
        dx = random.randint(-aug_cfg.get("translation_px_max", 0), aug_cfg.get("translation_px_max", 0))
        dy = random.randint(-aug_cfg.get("translation_px_max", 0), aug_cfg.get("translation_px_max", 0))
        aug_params["dx"] = dx
        aug_params["dy"] = dy
        grid = _get_translation_grid(grid, dx, dy, device)

    spatially_augmented_img = F.grid_sample(
        clean_img, grid, mode="bilinear", align_corners=True, padding_mode="zeros"
    )
    
    # --- 2. Noise ---
    noisy_img = spatially_augmented_img
    # (Noise logic omitted for brevity, same as before)
    if random.random() < aug_cfg.get("prob_gaussian", 0.0):
        noisy_img = add_gaussian(noisy_img, sigma=aug_cfg.get("gaussian_sigma", 0.0), mean=aug_cfg.get("gaussian_mean", 0.0))
    if random.random() < aug_cfg.get("prob_speckle", 0.0):
        noisy_img = add_speckle(noisy_img, sigma=aug_cfg.get("speckle_sigma", 0.0), mean=aug_cfg.get("speckle_mean", 0.0))
    if random.random() < aug_cfg.get("prob_poisson", 0.0):
        noisy_img = add_poisson(noisy_img, scale=aug_cfg.get("poisson_scale", 1.0), bias=aug_cfg.get("poisson_bias", 0.0))
    if random.random() < aug_cfg.get("prob_saltpepper", 0.0):
        noisy_img = add_salt_pepper(noisy_img, prob=aug_cfg.get("saltpepper_prob", 0.0), low_val=aug_cfg.get("saltpepper_low", 0.0), high_val=aug_cfg.get("saltpepper_high", 1.0))

    return noisy_img, aug_params