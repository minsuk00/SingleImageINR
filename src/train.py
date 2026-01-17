import argparse
import os
from datetime import datetime

import lpips
import torch
import yaml
from skimage.metrics import structural_similarity as ssim
from PIL import Image
import torch.nn.functional as F

import wandb
from losses import LossBundle
from models import build_model
from aug import PrecomputedAugmentations, InfiniteAugmentations, SingleFixedAugmentation
from utils import (
    load_gray_image, psnr, render_full, save_image01, set_seed, 
    draw_correspondence_lines, compute_correspondence_error_map, 
    render_heatmap_from_array
)
from tqdm import tqdm

import warnings
warnings.filterwarnings("ignore")

def generate_run_name(cfg):
    parts = []
    img_path = cfg.get("image_path", "")
    img_name = os.path.splitext(os.path.basename(img_path))[0]
    parts.append(img_name)
    
    # Aug Mode Name
    aug = cfg.get("augment", {})
    mode = aug.get("mode", "fixed")
    count = aug.get("num_views", 10)
    parts.append(f"{mode.capitalize()}{count}")

    if cfg.get("loss", {}).get("use_alignment_correction", False):
        parts.append("AlignCorr")
    
    # Translation naming (Frac vs Px)
    if float(aug.get("prob_translation", 0)) > 0:
        if "translation_frac_max" in aug:
            val = aug["translation_frac_max"]
            parts.append(f"TrFrac{val}")
    else:
        parts.append("Static")

    loss = cfg.get("loss", {})
    loss_map = {
        "pixel_weight": "Pix",
        "corr_weight": "Corr",
        "corr_pixel_weight": "CPix",
        "ssim_weight": "SSIM",
        "lpips_weight": "LPIPS",
        "chess_weight": "CheSS",
        "dino_weight": "Dino",
        "sam_weight": "SAM"
    }
    
    for key, alias in loss_map.items():
        w = float(loss.get(key, 0))
        if w > 0:
            parts.append(f"{alias}{w}")
    
    parts.append(f"lr{cfg.get('lr', '?')}")
    parts.append(datetime.now().strftime("%H%M"))
    return f"{cfg.get('exp_name', 'exp')}_{'_'.join(parts)}"

def main(cfg):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    set_seed(cfg["seed"])

    # --- W&B Initialization ---
    use_wandb = cfg.get("wandb_enabled", True)
    run_name = generate_run_name(cfg)
    wandb.init(
        project=cfg.get("wandb_project", "inir-project"),
        name=run_name,
        config=cfg,
        group=cfg.get("sweep_group", "standalone_runs"),
        mode="online" if use_wandb else "disabled",
    )

    # --- Data Loading ---
    clean_img, (H, W) = load_gray_image(
        cfg["image_path"], normalize_to=cfg.get("normalize_to", "0_1"), device=device
    )
    wandb.config.update({"img_width": W, "img_height": H})

    # --- Setup Augmentation Strategy ---
    aug_cfg = cfg.get("augment", {})
    aug_mode = aug_cfg.get("mode", "fixed") # Default to fixed
    num_views = aug_cfg.get("num_views", 10) # Number of views to accumulate gradients over

    if aug_mode == "infinite":
        aug_set = InfiniteAugmentations(cfg, clean_img, num_views=1)
    elif aug_mode == "single_fixed":
        aug_set = SingleFixedAugmentation(cfg, clean_img) 
    else:
        aug_set = PrecomputedAugmentations(cfg, clean_img, num_views=num_views)

    
    # Log Setup to WandB
    init_log = {
        "images/clean_ground_truth": wandb.Image(clean_img.squeeze(0)),
        "images/average_augmentation": wandb.Image(aug_set.average_img.squeeze(0), caption="Average View")
    }
    # Log views
    for i in range(len(aug_set)):
        view = aug_set[i]
        params = view.get('params', {'dx': 0, 'dy': 0})
        caption = f"dx: {params['dx']:.2f}, dy: {params['dy']:.2f}"
        init_log[f"augmentations/view_{i}"] = wandb.Image(view['image'].squeeze(0), caption=caption)
    wandb.log(init_log)
    
    # --- Metric Helpers (LPIPS) ---
    # Keep LPIPS on a specific device to avoid thrashing
    metric_device = device 
    lpips_fn = lpips.LPIPS(net="vgg").to(metric_device)
    lpips_fn.eval()
    losses = LossBundle(cfg, device, lpips_fn=lpips_fn)

    # --- Model & Optimizer ---
    model = build_model(cfg, device)
    opt = torch.optim.Adam(model.parameters(), lr=float(cfg["lr"]))

    # --- Output Dir ---
    out_dir = os.path.join(cfg["out_dir"], run_name)
    os.makedirs(out_dir, exist_ok=True)
    save_image01(clean_img, f"{out_dir}/clean.png")

    viz_cfg = cfg.get("viz", {})
    viz_every = viz_cfg.get("viz_every", 50)
    show_corr = viz_cfg.get("correspondence", True)
    use_alignment = cfg.get("loss", {}).get("use_alignment_correction", False)

    
    print(f"--- Starting training: {run_name} ---")
    pbar = tqdm(range(cfg["epochs"]), desc="Training", ncols=100)
    
    for step in pbar:
        opt.zero_grad(set_to_none=True)
        accum_loss = 0.0
        accum_loss_dict = {} # Initialize dict for averaging components
        accum_shift_error = 0.0 # Track alignment accuracy

        # 1. Render Canonical
        pred_canonical = render_full(model, H, W, device)

        # Iterate through views (Fixed list OR infinite generator)
        for view_idx in range(len(aug_set)):
            view = aug_set[view_idx] # In infinite mode, this generates a fresh view
            
            target_img = view['image']
            # grid = view['grid']
            mask = view['mask']

            # Get Ground Truth Shift (pixels)
            gt_dx = view['params']['dx']
            gt_dy = view['params']['dy']
            
            if use_alignment:
                # 1. Predict Alignment (Detached)
                grid_pred, (pred_dx, pred_dy) = losses.compute_predicted_alignment(pred_canonical, target_img)
                # 2. Warp
                grid_pred = grid_pred.to(pred_canonical.dtype)
                pred_for_loss = F.grid_sample(pred_canonical, grid_pred, align_corners=True, padding_mode="zeros")
                accum_shift_error += (abs(pred_dx - gt_dx) + abs(pred_dy - gt_dy))

            else:
                # Blind: Compare directly
                pred_for_loss = pred_canonical
            
            # 2. Warp Canonical to View
            # pred_warped = F.grid_sample(pred_canonical, grid, align_corners=True, padding_mode="zeros")
            
            # 3. Calculate Loss
            # step_loss, _ = losses.total(pred_warped, target_img, mask=mask)
            step_loss, step_dict = losses.total(pred_canonical, target_img, mask=mask)
            # Scale loss by 1/N so the sum is an average
            step_loss = step_loss / len(aug_set)
            for k, v in step_dict.items():
                accum_loss_dict[k] = accum_loss_dict.get(k, 0.0) + (v / len(aug_set))

            # 4. Accumulate Gradient
            retain_graph = (view_idx < len(aug_set) - 1)
            step_loss.backward(retain_graph=retain_graph)
            accum_loss += step_loss.item()
            
        # 5. Update Weights
        opt.step()

        # 5) Logging
        if (step + 1) % 5 == 0:
            with torch.no_grad():
                psnr_val = psnr(pred_canonical, clean_img)
                
                pred_np = pred_canonical.squeeze().cpu().numpy()
                clean_np = clean_img.squeeze().cpu().numpy()
                ssim_val = ssim(pred_np, clean_np, data_range=1.0)

                # LPIPS
                pred_l = pred_canonical.repeat(1,3,1,1) * 2 - 1
                clean_l = clean_img.repeat(1,3,1,1) * 2 - 1
                
                if pred_l.device != metric_device:
                    pred_l = pred_l.to(metric_device)
                    clean_l = clean_l.to(metric_device)
                
                lpips_val = lpips_fn(pred_l, clean_l).item()

            pbar.set_postfix({
                "loss": f"{accum_loss:.4f}",
                "psnr": f"{psnr_val.item():.2f}"
            })

            log_data = {
                "loss/total_loss": accum_loss,
                "metrics/psnr_canonical": psnr_val.item(),
                "metrics/ssim_canonical": ssim_val,
                "metrics/lpips_canonical": lpips_val,
                # "learning_rate": opt.param_groups[0]["lr"],
            }
            log_data.update({f"loss/{k}": v for k, v in accum_loss_dict.items()})
            # Log Shift Error if applicable
            if use_alignment:
                log_data["metrics/shift_mae_px"] = accum_shift_error / len(aug_set)
            wandb.log(log_data, step=step + 1)

        # 7. Visualization
        if step == 0 or (step + 1) % viz_every == 0 or (step + 1) == cfg["epochs"]:
            with torch.no_grad():
                save_image01(pred_canonical, f"{out_dir}/recon_step{step+1}.png")
                
                diff_img = torch.abs(clean_img - pred_canonical)
                max_diff = diff_img.max().item()
                diff_np = diff_img.squeeze().cpu().detach().numpy()
                heatmap_pil = render_heatmap_from_array(diff_np, vmin=0.0, vmax=0.1, cmap='inferno')
                
                viz_dict = {
                    "images/reconstruction": wandb.Image(pred_canonical.squeeze(0), caption="Canonical Reconstruction"),
                    "images/diff_heatmap": wandb.Image(heatmap_pil, caption=f"Canonical Error | Max: {max_diff:.4f}"),
                }
                
                # --- CORRESPONDENCE VIZ ---
                if show_corr:
                    # Force calc if not done by loss
                    view0=aug_set[0]
                    target_0 = view0['image']

                    if use_alignment:
                        grid_pred_0, (dx_0, dy_0) = losses.compute_predicted_alignment(pred_canonical, target_0)
                        grid_pred_0 = grid_pred.to(pred_canonical.dtype)
                        # print(grid_pred_0)
                        pred_viz = F.grid_sample(pred_canonical, grid_pred_0, align_corners=True, padding_mode="zeros")
                        
                        gt_dx_0 = view0['params']['dx']
                        gt_dy_0 = view0['params']['dy']
                        if abs(dx_0) < 1.0 and abs(dy_0) < 1.0:
                            pred_viz = pred_canonical.clone() # Use clone to avoid referencing memory from the grid
                            caption_suffix = f" (Est: {dx_0:.1f},{dy_0:.1f} vs GT: {gt_dx_0:.1f},{gt_dy_0:.1f} - No Warp)"
                        else:
                            # 3. If shift is significant, apply warp (Casting done here)
                            grid_pred_0 = grid_pred_0.to(pred_canonical.dtype)
                            pred_viz = F.grid_sample(pred_canonical, grid_pred_0, align_corners=True, padding_mode="zeros")
                            caption_suffix = f" (Est: {dx_0:.1f},{dy_0:.1f} vs GT: {gt_dx_0:.1f},{gt_dy_0:.1f})"
                        
                        # caption_suffix = f" (Est: {dx_0:.1f},{dy_0:.1f} vs GT: {gt_dx_0:.1f},{gt_dy_0:.1f})"
                        
                        viz_dict["images/reconstruction_shifted"] = wandb.Image(
                            pred_viz.squeeze(0), 
                            caption=f"Recon Shifted to View 0{caption_suffix}"
                        )
                    else:
                        pred_viz = pred_canonical
                        caption_suffix = ""
                        
                    if losses.latest_corr_data is None:
                        losses.compute_for_visualization(pred_viz, target_0)
                    
                    if losses.latest_corr_data is not None:
                        cdata = losses.latest_corr_data
                        params0 = view0['params']
                        
                        # A. Lines
                        lines_img = draw_correspondence_lines(
                            pred_viz, target_0, 
                            cdata['match_B'], 
                            cdata['WA'], cdata['HA'], cdata['WB'], cdata['HB'],
                            num_samples=40
                        )
                        viz_dict["images/corr_lines"] = wandb.Image(lines_img, caption=f"Blue:Pred, Red:Aug (dx:{params0['dx']:.3f} dy:{params0['dy']:.3f})")

                        # B. Error Map
                        # if abs(params0['dx']) > 0 or abs(params0['dy']) > 0:
                        err_arr = compute_correspondence_error_map(
                            cdata['match_B'], 
                            cdata['WA'], cdata['HA'], cdata['WB'], cdata['HB'],
                            cdata['patch_size'], 
                            params0['dx'], params0['dy']
                        )
                        
                        err_max = err_arr.max()
                        err_mean = err_arr.mean()
                        
                        err_heatmap = render_heatmap_from_array(err_arr, vmax=50, cmap="inferno")
                        viz_dict["images/corr_error_map"] = wandb.Image(
                            err_heatmap, 
                            caption=f"Patch Error (px) | Max: {err_max:.1f}, Mean: {err_mean:.1f}"
                        )
                        
                        frank_img = losses.get_frankenstein_target(target_0)
                        viz_dict["images/frankenstein_target"] = wandb.Image(
                            frank_img.squeeze(0),
                            caption="Frankenstein Target (Matched Patches)"
                        )

                wandb.log(viz_dict, step=step+1)


    wandb.finish()
    print(f"--- Finished run: {run_name} ---")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="/home/minsukc/SIO/config/config.yaml")
    parser.add_argument("-W", "--no-wandb", action="store_false", dest="wandb_enabled")
    args = parser.parse_args()

    with open(args.config, "r") as f:
        cfg = yaml.safe_load(f)

    cfg["wandb_enabled"] = args.wandb_enabled
    
    # Default group if running standalone
    if "sweep_group" not in cfg:
        cfg["sweep_group"] = "standalone_runs"
    
    main(cfg)
