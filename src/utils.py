import os
import random
from typing import Union

import numpy as np
import SimpleITK as sitk
import torch
import torchvision.transforms.functional as TF
from PIL import Image
from torchvision.utils import save_image


def set_seed(seed):
    random.seed(seed)
    torch.manual_seed(seed)
    np.random.seed(seed)


def load_gray_image(path, normalize_to="0_1", device: Union[str, torch.device] = "cpu"):
    img = Image.open(path).convert("L")
    img_t = TF.to_tensor(img).unsqueeze(0)  # [1,1,H,W], 0..1
    if normalize_to == "-1_1":
        img_t = img_t * 2 - 1
    return img_t.to(device), img.size[::-1]  # (H,W)


def grid_coords(H, W, device):
    ys = torch.linspace(-1, 1, H, device=device)
    xs = torch.linspace(-1, 1, W, device=device)
    yy, xx = torch.meshgrid(ys, xs, indexing="ij")
    return torch.stack([xx, yy], dim=-1).view(-1, 2)  # [H*W,2]


def render_full(model, H, W, device, batch=262144):
    coords = grid_coords(H, W, device)
    out = []
    for i in range(0, coords.shape[0], batch):
        pred = model(coords[i : i + batch])
        out.append(pred)
    pred = torch.cat(out, dim=0).view(1, 1, H, W)
    pred = torch.clamp(pred, 0.0, 1.0)
    return pred


def save_image01(tensor01, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    save_image(tensor01, path)  # expects [0,1]


def psnr(x, y, eps=1e-8):
    mse = torch.mean((x - y) ** 2)
    return -10.0 * torch.log10(mse + eps)


def load_nii_image(nii_path, resize=None, as_rgb=True):
    """
    Load a 2D .nii or .nii.gz medical projection image and convert it into a PIL Image.

    Args:
        nii_path (str): Path to the 2D NIfTI file (e.g., CT projection).
        resize (tuple[int, int], optional): Resize output (width, height).
                                            If None, keeps original size.
        as_rgb (bool, optional): If True, converts grayscale → 3-channel RGB.
                                 If False, keeps single-channel grayscale.

    Returns:
        image (PIL.Image): PIL image (RGB or grayscale) normalized to [0, 255].
    """
    img_nii = sitk.ReadImage(nii_path)
    img_np = sitk.GetArrayFromImage(img_nii)
    print(f"[load_nii_image] Loaded NIfTI shape: {img_np.shape}")

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

    print(f"[load_nii_image] Output image size: {image.size}, mode: {mode}")
    return image
