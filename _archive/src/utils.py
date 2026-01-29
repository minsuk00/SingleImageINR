import os
import random
from typing import Union

import numpy as np
import SimpleITK as sitk
import torch
import torchvision.transforms.functional as TF
from PIL import Image
from torchvision.utils import save_image
import matplotlib.pyplot as plt
import io
import cv2

def set_seed(seed):
    random.seed(seed)
    torch.manual_seed(seed)
    np.random.seed(seed)


def load_gray_image(path, normalize_to="0_1", device: Union[str, torch.device] = "cpu"):
    if not os.path.exists(path):
        raise FileNotFoundError(f"Image not found at: {path}")
        
    img = Image.open(path).convert("L")
    img_t = TF.to_tensor(img).unsqueeze(0)  # [1,1,H,W], 0..1
    if normalize_to == "-1_1":
        img_t = img_t * 2 - 1
    return img_t.to(device), img.size[::-1]  # (H,W)


def grid_coords(H, W, device):
    ys = torch.linspace(0, 1, H, device=device)
    xs = torch.linspace(0, 1, W, device=device)
    # ys = torch.linspace(-1, 1, H, device=device)
    # xs = torch.linspace(-1, 1, W, device=device)
    yy, xx = torch.meshgrid(ys, xs, indexing="ij") # indexing='ij' means meshgrid returns (H, W)
    return torch.stack([xx, yy], dim=-1).view(-1, 2)  # [H*W,2]


def render_full(model, H, W, device, batch=262144):
    """
    Note: if model relies on gradients (training), do NOT use torch.no_grad() outside.
    """
    coords = grid_coords(H, W, device)
    out = []
    
    # Process in chunks to avoid OOM during forward pass
    for i in range(0, coords.shape[0], batch):
        chunk = coords[i : i + batch]
        # TCNN requires inputs on GPU, usually float or half
        pred = model(chunk)
        out.append(pred)
        
    # Concatenate all chunks
    pred = torch.cat(out, dim=0).view(1, 1, H, W)
    
    # Depending on output activation of TCNN, we might need clamping
    # If output is linear, clamp to [0,1] for image validity
    pred = torch.clamp(pred, 0.0, 1.0)
    return pred


def save_image01(tensor01, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    save_image(tensor01, path)  # expects [0,1]


def psnr(x, y, eps=1e-8):
    mse = torch.mean((x - y) ** 2)
    return -10.0 * torch.log10(mse + eps)


def load_nii_image(nii_path, resize=None, as_rgb=True):
    if not os.path.exists(nii_path):
        print(f"[load_nii_image] File not found: {nii_path}")
        return None

    img_nii = sitk.ReadImage(nii_path)
    img_np = sitk.GetArrayFromImage(img_nii)
    
    if img_np.ndim == 3:
        img_np = img_np[0]  # squeeze if shape is (1,H,W)

    img_norm = (img_np - img_np.min()) / (img_np.max() - img_np.min() + 1e-8)
    img_uint8 = (img_norm * 255).astype(np.uint8)

    if as_rgb:
        img_rgb = np.stack([img_uint8] * 3, axis=-1)
        image = Image.fromarray(img_rgb)
        mode = "RGB"
    else:
        image = Image.fromarray(img_uint8, mode="L")
        mode = "L"

    if resize is not None:
        image = image.resize(resize)

    return image


def tensor_to_cv2(tensor):
    """Converts [1,1,H,W] float tensor (0..1) to [H,W,3] uint8 numpy for OpenCV"""
    img = tensor.squeeze().cpu().detach().numpy()
    img = (np.clip(img, 0, 1) * 255).astype(np.uint8)
    img_bgr = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    return img_bgr

def draw_correspondence_lines(imgA_t, imgB_t, matches, WA, HA, WB, HB, num_samples=20):
    """
    Draws lines between matched patches with DYNAMIC scaling for high-res images.
    Includes a white separator line between images.
    """
    imgA = tensor_to_cv2(imgA_t)
    imgB = tensor_to_cv2(imgB_t)
    matches = matches.cpu().numpy()
    N_A = len(matches)

    # --- Dynamic Scaling ---
    img_w = imgA.shape[1]
    img_h = imgA.shape[0]
    radius = max(4, int(img_w * 0.008))
    thickness = max(1, int(img_w * 0.002))
    sep_width = 5

    # Sampling indices
    indices = np.linspace(0, N_A - 1, num_samples, dtype=int)

    # --- Create Separator & Canvas ---
    # White separator (H, sep_width, 3)
    separator = np.full((img_h, sep_width, 3), 255, dtype=np.uint8)
    
    # Concatenate: A | separator | B
    canvas = np.concatenate([imgA, separator, imgB], axis=1)

    # Offset for coordinates on the right image (width of A + width of separator)
    imgB_offset_x = imgA.shape[1] + sep_width

    stepA_x = imgA.shape[1] / WA
    stepA_y = imgA.shape[0] / HA
    stepB_x = imgB.shape[1] / WB
    stepB_y = imgB.shape[0] / HB

    for i in indices:
        b_idx = matches[i]
        
        # A patch center
        Ay = (i // WA) * stepA_y + stepA_y / 2
        Ax = (i % WA) * stepA_x + stepA_x / 2
        
        # B patch center (local coordinates + offset to right side of canvas)
        By = (b_idx // WB) * stepB_y + stepB_y / 2
        Bx = (b_idx % WB) * stepB_x + stepB_x / 2 + imgB_offset_x

        pA = (int(Ax), int(Ay))
        pB = (int(Bx), int(By))

        cv2.circle(canvas, pA, radius, (255, 0, 0), -1) # Blue on A
        cv2.circle(canvas, pB, radius, (0, 0, 255), -1) # Red on B
        cv2.line(canvas, pA, pB, (0, 255, 0), thickness)   # Green Line

    # Convert back to PIL for WandB
    canvas_rgb = cv2.cvtColor(canvas, cv2.COLOR_BGR2RGB)
    return Image.fromarray(canvas_rgb)

def compute_correspondence_error_map(matches, WA, HA, WB, HB, patch_size, true_dx, true_dy):
    """
    Computes heatmap of error vs known translation.
    Changed: Uses explicit patch_size to align perfectly with DINO grid logic.

    WA, HA, WB, HB: # of patches each side
    matches: An array of length (# of total patches)
    """
    matches = matches.cpu().numpy()
    
    # 1. Patch centers A
    # DINO centers are at (index * patch_size) + (patch_size / 2)
    # But we just need relative movement, so (index * patch_size) is fine for delta.
    xs_A = np.tile(np.arange(WA), HA) # [0, 1, 2] -> [0, 1, 2, 0, 1, 2]
    ys_A = np.repeat(np.arange(HA), WA) # [0, 1] -> [0, 0, 0, 1, 1, 1]
    
    # 2. Patch centers B (from matches)
    xs_B = matches % WB # X coord
    ys_B = matches // WB # Y coord
    
    # 3. Displacement (patch units)
    dx_patch = xs_B - xs_A
    dy_patch = ys_B - ys_A
    
    # 4. Displacement (pixels)
    dx_px = dx_patch * patch_size
    dy_px = dy_patch * patch_size
    
    # 5. Error
    error_x = dx_px - true_dx
    error_y = dy_px - true_dy
    
    error_mag = np.sqrt(error_x**2 + error_y**2)
    
    # 6. Reshape
    error_img = error_mag.reshape(HA, WA)
    return error_img


def render_heatmap_from_array(arr, vmin=None, vmax=None, cmap='inferno'):
    fig, ax = plt.subplots(figsize=(5, 5))
    im = ax.imshow(arr, cmap=cmap, vmin=vmin, vmax=vmax)
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    plt.axis('off')
    
    buf = io.BytesIO()
    plt.savefig(buf, format='png', bbox_inches='tight', pad_inches=0.05, dpi=100)
    plt.close(fig)
    buf.seek(0)
    return Image.open(buf)