import torch
import torch.nn as nn
import torch.nn.functional as F
import tinycudann as tcnn
import numpy as np
import random
import os
from PIL import Image
import torchvision.transforms.functional as TF
import wandb
import argparse
import yaml
from datetime import datetime
from tqdm import tqdm

from utils import set_seed, load_gray_image, render_full, save_image01
from models import build_model

def generate_run_name(cfg):
    parts = ["SIO_10_augmentation_20_parameter"]
    parts.append(datetime.now().strftime("%H%M"))
    return "_".join(parts)

# --- The Augmentation Module ---
class LearnableAugmentation(nn.Module):
    def __init__(self, num_augmentations, device, init_scale=0.0):
        super().__init__()
        self.num_augmentations = num_augmentations
        self.device = device
        # Initialize 2 parameters (dx, dy) for each augmentation (use randn?)
        self.shifts = nn.Parameter(torch.zeros(num_augmentations, 2, device=device) * init_scale)

    def forward(self, image_batch, override_shifts=None):
        """
        image_batch: [1, 1, H, W] (Canonical)
        Returns: [N, 1, H, W] (Deformed copies)
        """
        x = image_batch.float()

        B, C, H, W = x.shape
        N = self.num_augmentations
        
        # 1. Expand canonical to N copies
        x_expanded = x.expand(N, -1, -1, -1)
        
        # 2. Prepare Affine Matrices
        # Identity: [[1, 0, 0], [0, 1, 0]]
        theta = torch.eye(2, 3, device=self.device).unsqueeze(0).repeat(N, 1, 1)
        
        # 3. Inject Learnable Shifts
        shifts_to_use = self.shifts if override_shifts is None else override_shifts
        theta[:, :, 2] += shifts_to_use
        
        # 4. Grid Sample
        grid = F.affine_grid(theta, x_expanded.size(), align_corners=False)
        x_shifted = F.grid_sample(x_expanded, grid, align_corners=False)
        
        return x_shifted

# --- 3. Main Loop ---

def main(cfg):
    num_augs = 10
    is_infinite = True
    aug_range_max = 0.1
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    set_seed(cfg["seed"])
    
    # Setup WandB
    run_name = generate_run_name(cfg)
    wandb.init(
        project=cfg.get("wandb_project", "joint-opt"),
        name=run_name,
        config=cfg,
        mode="online" if cfg.get("wandb_enabled", True) else "disabled"
    )
    
    # Directories
    out_dir = os.path.join(cfg["out_dir"], run_name)
    os.makedirs(out_dir, exist_ok=True)
    
    # Load Data
    clean_image, (H, W) = load_gray_image("/home/minsukc/SIO/data/ct_chest_2_resized_256x256.png" , normalize_to=cfg["normalize_to"], device=device)
    save_image01(clean_image, f"{out_dir}/target.png")

    print(f"--- Generating Synthetic Data ---")
    temp_aug = LearnableAugmentation(num_augs, device)
    def get_random_shifts(range_max = 0.2):
        # Range: -0.2 to 0.2
        return (torch.rand(num_augs, 2, device=device) * range_max* 2) - range_max

    gt_shifts = get_random_shifts(aug_range_max)
    with torch.no_grad():
        # [N, 1, H, W]
        gt_batch = temp_aug(clean_image, override_shifts=gt_shifts)
    save_image01(gt_batch[0], f"{out_dir}/target_example_0.png")
    print(f"Generated {num_augs} target images with known shifts.")

    log_dict = {
        "data/clean_gt_base": wandb.Image(clean_image.squeeze(0), caption="Base Clean Image")
    }
    # Log all 10 shifted targets
    for i in range(num_augs):
        img_tensor = gt_batch[i] 
        shift_vals = gt_shifts[i].tolist()
        
        # --- CONVERSION TO PIXELS ---
        dx_px = (shift_vals[0] / 2.0) * W
        dy_px = (shift_vals[1] / 2.0) * H
        
        caption = f"Target {i} (dx:{dx_px:.1f}px, dy:{dy_px:.1f}px)"
        log_dict[f"data/shifted_target_{i}"] = wandb.Image(img_tensor, caption=caption)
    wandb.log(log_dict, step=0)
    
    # Initialize Model & Augmentation
    model = build_model(cfg, device)
    
    aug_module = LearnableAugmentation(
        num_augmentations=num_augs, 
        device=device,
    )
    
    # Optimizer with separate groups
    optimizer = torch.optim.Adam([
        {'params': model.parameters(), 'lr': float(1e-3)},
        {'params': aug_module.parameters(), 'lr': float(1e-3)}
    ])
    
    print(f"--- Starting Joint Optimization: {run_name} ---")
    print(f"Model Params: {sum(p.numel() for p in model.parameters())}")
    print(f"Aug Params: {sum(p.numel() for p in aug_module.parameters())} (Should be {num_augs*2})")

    pbar = tqdm(range(cfg["epochs"]), desc="Training", ncols=100)
    for step in pbar:
        if is_infinite:
            with torch.no_grad():
                gt_shifts = get_random_shifts(aug_range_max) # Generate NEW random shifts
                gt_batch = temp_aug(clean_image, override_shifts=gt_shifts) # Generate NEW images

        optimizer.zero_grad()
        # 1. Render Canonical (1 image)
        pred_canonical = render_full(model, H, W, device)
        
        # 2. Apply Learnable Augmentations (1 -> 10 images)
        pred_deformed_batch = aug_module(pred_canonical)
        
        # 3. Compute Loss
        diff = pred_deformed_batch - gt_batch

        loss = (diff ** 2).mean() # Averages over batch (10) and pixels
        
        # 4. Backprop
        loss.backward()
        optimizer.step()
        
        # 5. Logging
        if (step + 1) % 10 == 0:
            # Calculate how close our learned parameters are to the fixed GT parameters
            with torch.no_grad():
                param_error = (aug_module.shifts - gt_shifts).abs().mean().item()
                param_error_px = (param_error / 2.0) * W

                # Get raw values
                learned_x0 = aug_module.shifts[0, 0].item()
                gt_x0 = gt_shifts[0, 0].item()
                
                # Convert specific params to pixels for charting
                learned_x0_px = (learned_x0 / 2.0) * W
                gt_x0_px = (gt_x0 / 2.0) * W
            
            pbar.set_postfix({"loss": f"{loss.item():.5f}", "err_px": f"{param_error_px:.2f}"})
            
            wandb.log({
                "loss/total": loss.item(),
                "metrics/param_error_norm": param_error,
                "metrics/avg_error_px": param_error_px,
                
                # Normalized values (original)
                "params/learned_x_0": learned_x0,
                "params/gt_x_0": gt_x0, 
                
                # Pixel values (new)
                "params/learned_x_0_px": learned_x0_px,
                "params/gt_x_0_px": gt_x0_px,
            }, step=step+1)
            
        # 6. Visualization
        viz_every = cfg["viz"]["viz_every"]
        if (step + 1) % viz_every == 0 or (step + 1) == cfg["epochs"]:
            with torch.no_grad():
                save_image01(pred_canonical, f"{out_dir}/step{step+1}_canonical.png")
                
                # Get learned shifts for caption
                learned_s = aug_module.shifts[0].tolist()
                dx_px = (learned_s[0] / 2.0) * W
                dy_px = (learned_s[1] / 2.0) * H
                
                # Get GT shifts for caption
                gt_s = gt_shifts[0].tolist()
                gt_dx_px = (gt_s[0] / 2.0) * W
                gt_dy_px = (gt_s[1] / 2.0) * H
                
                wandb.log({
                    "images/canonical": wandb.Image(pred_canonical.squeeze(0), caption="Learned Canonical"),
                    
                    "images/deformed_0": wandb.Image(
                        pred_deformed_batch[0], 
                        caption=f"Canonical + Learned (dx:{dx_px:.1f}px, dy:{dy_px:.1f}px)"
                    ),
                    
                    "images/gt_0": wandb.Image(
                        gt_batch[0], 
                        caption=f"GT Shift (dx:{gt_dx_px:.1f}px, dy:{gt_dy_px:.1f}px)"
                    )
                }, step=step+1)

    print(f"--- Finished run: {run_name} ---")
    print(f"Final Learned Shift 0: {aug_module.shifts[0].tolist()}")
    print(f"True GT Shift 0:     {gt_shifts[0].tolist()}")
    wandb.finish()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="/home/minsukc/SIO/config/config.yaml")
    parser.add_argument("--no-wandb", action="store_false", dest="wandb_enabled")
    args = parser.parse_args()

    with open(args.config, "r") as f:
        cfg = yaml.safe_load(f)

    # Allow CLI override for wandb
    if not args.wandb_enabled:
        cfg["wandb_enabled"] = False
        
    main(cfg)