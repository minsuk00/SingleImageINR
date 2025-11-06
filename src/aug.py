import random
import torch
import torch.nn.functional as F


# --------------------------------------------------------
# Grid Generation Helpers
# --------------------------------------------------------

def _get_identity_grid(H, W, device):
    """Generates an identity grid [1, H, W, 2] for F.grid_sample"""
    yy, xx = torch.meshgrid(
        torch.linspace(-1, 1, H, device=device),
        torch.linspace(-1, 1, W, device=device),
        indexing="ij",
    )
    grid = torch.stack([xx, yy], dim=-1).unsqueeze(0)  # [1, H, W, 2]
    return grid


def _get_rotation_grid(grid, degrees, device):
    """Applies a rotation to a grid."""
    theta = torch.deg2rad(torch.tensor(degrees, device=device, dtype=torch.float32))
    cos_t, sin_t = torch.cos(theta), torch.sin(theta)
    
    # Rotation matrix (transposed for matrix multiplication)
    # [cos, -sin]
    # [sin,  cos]
    rot_matrix = torch.tensor(
        [[cos_t, sin_t], 
         [-sin_t, cos_t]], 
        device=device, dtype=torch.float32
    )
    
    # (grid @ rot_matrix) is equivalent to [x*cos-y*sin, x*sin+y*cos]
    rotated_grid = grid @ rot_matrix
    return rotated_grid


def _get_shear_grid(grid, shear_x, shear_y, device):
    """Applies shear (linear deformation) to a grid."""
    shear_matrix = torch.tensor(
        [[1, shear_y], 
         [shear_x, 1]], 
        device=device, dtype=torch.float32
    )
    sheared_grid = grid @ shear_matrix
    return sheared_grid


def _get_nonlinear_warp_grid(grid, warp_scale, warp_resolution, device):
    """Applies a smooth nonlinear deformation field to a grid."""
    B, H, W, _ = grid.shape
    
    # Create low-res random noise field
    low_res_noise = (
        torch.rand(B, 2, warp_resolution, warp_resolution, device=device) - 0.5
    ) * 2 * warp_scale
    
    # Upsample to full resolution
    warp_field = F.interpolate(
        low_res_noise, (H, W), mode="bilinear", align_corners=True
    )
    
    # Reshape and add to grid
    warp_field = warp_field.permute(0, 2, 3, 1)  # [B, H, W, 2]
    warped_grid = grid + warp_field
    return warped_grid


def _get_translation_grid(grid, dx, dy, device):
    """Applies translation to a grid."""
    B, H, W, _ = grid.shape
    # normalize pixel offsets to [-1,1]
    tx = 2.0 * dx / (W - 1)
    ty = 2.0 * dy / (H - 1)
    
    # Create translation vector [1, 1, 1, 2]
    trans_vec = torch.tensor([tx, ty], device=device).view(1, 1, 1, 2)
    
    # Subtract vector from grid
    translated_grid = grid - trans_vec
    return translated_grid


# --------------------------------------------------------
# Noise augmentations (Unchanged)
# --------------------------------------------------------
def add_gaussian(img, sigma, mean=0.0):
    if sigma <= 0:
        return img
    noise = torch.randn_like(img) * sigma + mean
    return torch.clamp(img + noise, 0.0, 1.0)

def add_speckle(img, sigma, mean=0.0):
    if sigma <= 0:
        return img
    noise = torch.randn_like(img) * sigma + mean
    return torch.clamp(img * (1.0 + noise), 0.0, 1.0)

def add_poisson(img, scale=1.0, bias=0.0):
    if scale <= 0:
        return img
    noisy = torch.poisson(torch.clamp(img * scale, 0, None)) / scale
    return torch.clamp(noisy + bias, 0.0, 1.0)

def add_salt_pepper(img, prob=0.05, low_val=0.0, high_val=1.0):
    if prob <= 0:
        return img
    mask = torch.rand_like(img)
    I_noisy = img.clone()
    I_noisy[mask < prob / 2] = low_val
    I_noisy[mask > 1 - prob / 2] = high_val
    return I_noisy


# --------------------------------------------------------
# Main augmentation sampler (Refactored)
# --------------------------------------------------------
def sample_augmentation(cfg, clean_img):
    """
    Applies a chain of spatial and noise augmentations.
    Spatial augmentations are combined and applied *once*.
    """
    aug_cfg = cfg["augment"]
    device = clean_img.device
    B, C, H, W = clean_img.shape

    # --- 1. Build Spatial Augmentation Grid ---
    grid = _get_identity_grid(H, W, device)

    # Rotation
    if random.random() < aug_cfg.get("prob_rotation", 0.0):
        deg = random.uniform(
            -aug_cfg.get("rotation_deg_max", 0.0), 
            aug_cfg.get("rotation_deg_max", 0.0)
        )
        grid = _get_rotation_grid(grid, deg, device)
        
    # Shear (Linear Deformation)
    if random.random() < aug_cfg.get("prob_shear", 0.0):
        sx = random.uniform(
            -aug_cfg.get("shear_factor_max", 0.0), 
            aug_cfg.get("shear_factor_max", 0.0)
        )
        sy = random.uniform(
            -aug_cfg.get("shear_factor_max", 0.0), 
            aug_cfg.get("shear_factor_max", 0.0)
        )
        grid = _get_shear_grid(grid, sx, sy, device)

    # Nonlinear Warping
    if random.random() < aug_cfg.get("prob_warp", 0.0):
        grid = _get_nonlinear_warp_grid(
            grid,
            aug_cfg.get("warp_scale", 0.1),
            aug_cfg.get("warp_resolution", 16),
            device
        )
        
    # Translation (applied last)
    if random.random() < aug_cfg.get("prob_translation", 0.0):
        dx = random.randint(
            -aug_cfg.get("translation_px_max", 0), 
            aug_cfg.get("translation_px_max", 0)
        )
        dy = random.randint(
            -aug_cfg.get("translation_px_max", 0), 
            aug_cfg.get("translation_px_max", 0)
        )
        grid = _get_translation_grid(grid, dx, dy, device)

    # --- 2. Apply Spatial Augmentation ---
    # Apply the combined grid transform once
    spatially_augmented_img = F.grid_sample(
        clean_img, 
        grid, 
        mode="bilinear", 
        align_corners=True, 
        padding_mode="border"
    )
    
    # --- 3. Apply Noise Augmentation ---
    noisy_img = spatially_augmented_img

    if random.random() < aug_cfg.get("prob_gaussian", 0.0):
        noisy_img = add_gaussian(
            noisy_img,
            sigma=aug_cfg.get("gaussian_sigma", 0.0),
            mean=aug_cfg.get("gaussian_mean", 0.0),
        )
    if random.random() < aug_cfg.get("prob_speckle", 0.0):
        noisy_img = add_speckle(
            noisy_img,
            sigma=aug_cfg.get("speckle_sigma", 0.0),
            mean=aug_cfg.get("speckle_mean", 0.0),
        )
    if random.random() < aug_cfg.get("prob_poisson", 0.0):
        noisy_img = add_poisson(
            noisy_img,
            scale=aug_cfg.get("poisson_scale", 1.0),
            bias=aug_cfg.get("poisson_bias", 0.0),
        )
    if random.random() < aug_cfg.get("prob_saltpepper", 0.0):
        noisy_img = add_salt_pepper(
            noisy_img,
            prob=aug_cfg.get("saltpepper_prob", 0.0),
            low_val=aug_cfg.get("saltpepper_low", 0.0),
            high_val=aug_cfg.get("saltpepper_high", 1.0),
        )

    return noisy_img
