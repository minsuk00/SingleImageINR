import os
import random
from typing import Union

import numpy as np
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
