import argparse
import os
from datetime import datetime

import lpips
import torch
import yaml
from skimage.metrics import structural_similarity as ssim
from PIL import Image

import wandb
from aug import sample_augmentation
from losses import LossBundle
from models import build_model
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
    
    # 0. Image name
    img_path = cfg.get("image_path", "")
    img_name = os.path.splitext(os.path.basename(img_path))[0]
    parts.append(img_name)

    # 1. Translation augmentation
    aug = cfg.get("augment", {})
    if float(aug.get("prob_translation", 0)) > 0:
        val = aug.get("translation_px_max", 0)
        parts.append(f"Tr{val}")
    else:
        parts.append("Static")

    # 2. Active losses
    loss = cfg.get("loss", {})
    loss_map = {
        "pixel_weight": "Pix",
        "corr_weight": "Corr",
        "corr_pixel_weight": "CPix",
        "ssim_weight": "SSIM",
        "lpips_weight": "LPIPS",
        "dino_weight": "Dino",
        "sam_weight": "SAM"
    }
    
    for key, alias in loss_map.items():
        w = float(loss.get(key, 0))
        if w > 0:
            parts.append(f"{alias}{w}")

    # 3. Learning rate
    parts.append(f"lr{cfg.get('lr', '?')}")

    # 4. Timestamp
    parts.append(datetime.now().strftime("%H%M"))

    # Combine
    base = cfg.get("exp_name", "exp")
    return f"{base}_{'_'.join(parts)}"

def main(cfg):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    set_seed(cfg["seed"])

    # --- Config/Safety Checks ---
    # if cfg["loss"]["pixel_weight"] > 0 and cfg["augment"]["prob_translation"] > 0:
    #     print("\n" + "!"*60)
    #     print("WARNING: You are using Pixel Loss (L1/L2) with Translation Augmentation.")
    #     print("Unless you are using Correspondence Pixel Loss exclusively, this")
    #     print("will force the model to learn a blurry average.")
    #     print("!"*60 + "\n")

    # --- W&B Initialization ---
    use_wandb = cfg.get("wandb_enabled", True)
    
    # Generate the descriptive name here
    run_name = generate_run_name(cfg)

    wandb.init(
        project=cfg.get("wandb_project", "inir-project"),
        name=run_name,
        config=cfg,
        group=cfg.get("sweep_group", "standalone_runs"),
        mode="online" if use_wandb else "disabled",
    )

    # --- Data Loading ---
    # clean_img: [1, 1, H, W]
    clean_img, (H, W) = load_gray_image(
        cfg["image_path"], normalize_to=cfg.get("normalize_to", "0_1"), device=device
    )
    
    # Log basic image info to wandb config
    wandb.config.update({
        "img_width": W,
        "img_height": H
    })
    
    # print(f"\n[INFO] Loaded image: {cfg['image_path']}")
    # print(f"[INFO] Dimensions: {W}x{H} (Width x Height)\n")

    # --- Metric Helpers (LPIPS) ---
    # Keep LPIPS on a specific device to avoid thrashing
    metric_device = device 
    lpips_fn = lpips.LPIPS(net="vgg").to(metric_device)
    lpips_fn.eval()

    # --- Model & Optimizer ---
    model = build_model(cfg, device)
    opt = torch.optim.Adam(model.parameters(), lr=float(cfg["lr"]))

    # --- Losses ---
    losses = LossBundle(cfg, device, lpips_fn=lpips_fn)

    # --- Output Dir ---
    # Use the descriptive run_name for the folder too, so it's easier to find locally
    out_dir = os.path.join(cfg["out_dir"], run_name)
    os.makedirs(out_dir, exist_ok=True)
    
    save_image01(clean_img, f"{out_dir}/clean.png")
    wandb.log({"images/ground_truth": wandb.Image(clean_img.squeeze(0))})

    # Viz Config
    viz_cfg = cfg.get("viz", {})
    show_corr = viz_cfg.get("correspondence", True)
    
    print(f"--- Starting training: {run_name} ---")
    pbar = tqdm(range(cfg["epochs"]), desc="Training", ncols=100)
    
    for step in pbar:
        # 1) Augmentation
        I_aug, aug_params = sample_augmentation(cfg, clean_img)

        # 2) Render Canonical
        pred_img = render_full(model, H, W, device, batch=262144)

        # 3) Compute Loss
        total_loss, loss_dict = losses.total(pred_img, I_aug)

        # 4) Optimize
        opt.zero_grad(set_to_none=True)
        total_loss.backward()
        opt.step()

        # 5) Logging
        if (step + 1) % 5 == 0:
            with torch.no_grad():
                psnr_val = psnr(pred_img, clean_img)
                
                pred_np = pred_img.squeeze().cpu().numpy()
                clean_np = clean_img.squeeze().cpu().numpy()
                ssim_val = ssim(pred_np, clean_np, data_range=1.0)

                # LPIPS
                pred_l = pred_img.repeat(1,3,1,1) * 2 - 1
                clean_l = clean_img.repeat(1,3,1,1) * 2 - 1
                
                if pred_l.device != metric_device:
                    pred_l = pred_l.to(metric_device)
                    clean_l = clean_l.to(metric_device)
                
                lpips_val = lpips_fn(pred_l, clean_l).item()

            pbar.set_postfix({
                "loss": f"{total_loss.item():.4f}",
                "psnr": f"{psnr_val.item():.2f}"
            })

            log_data = {
                "total_loss": total_loss.item(),
                "metrics/psnr_clean": psnr_val.item(),
                "metrics/ssim_clean": ssim_val,
                "metrics/lpips_clean": lpips_val,
                "learning_rate": opt.param_groups[0]["lr"],
            }
            log_data.update({f"loss/{k}": v for k, v in loss_dict.items()})
            wandb.log(log_data, step=step + 1)

        # 6) Visualization
        if step == 0 or (step + 1) % viz_cfg["viz_every"] == 0 or (step + 1) == cfg["epochs"]:
            with torch.no_grad():
                save_image01(pred_img, f"{out_dir}/recon_step{step+1}.png")
                save_image01(I_aug, f"{out_dir}/aug_step{step+1}.png")
                
                # Diff Logic
                diff_img = torch.abs(clean_img - pred_img)
                max_diff = diff_img.max().item()
                
                # Save Raw Diff
                save_image01(diff_img, f"{out_dir}/diff_raw_step{step+1}.png")

                # Render Heatmap (Convert Tensor to Numpy first)
                diff_np = diff_img.squeeze().cpu().detach().numpy()
                heatmap_pil = render_heatmap_from_array(
                    diff_np, 
                    vmin=0.0, 
                    vmax=0.1,  # Highlights errors in the 0.0 - 0.1 range
                    cmap='inferno'
                )
                heatmap_pil.save(f"{out_dir}/diff_heatmap_step{step+1}.png")
                
                viz_dict = {
                    "images/reconstruction": wandb.Image(pred_img.squeeze(0)),
                    "images/last_augmentation": wandb.Image(I_aug.squeeze(0), caption=f"dx:{aug_params['dx']} dy:{aug_params['dy']}"),
                    "images/diff_heatmap": wandb.Image(heatmap_pil, caption=f"Diff Heatmap (Max Err: {max_diff:.4f})"),
                    "images/diff_raw": wandb.Image(diff_img.squeeze(0), caption=f"Diff Raw (Max: {max_diff:.4f})"),
                }

                # --- CORRESPONDENCE VIZ ---
                if show_corr:
                    # Force calc if not done by loss
                    if losses.latest_corr_data is None:
                        losses.compute_for_visualization(pred_img, I_aug)
                    
                    if losses.latest_corr_data is not None:
                        cdata = losses.latest_corr_data
                        
                        # A. Lines
                        lines_img = draw_correspondence_lines(
                            pred_img, I_aug, 
                            cdata['match_B'], 
                            cdata['WA'], cdata['HA'], cdata['WB'], cdata['HB'],
                            num_samples=40
                        )
                        viz_dict["images/corr_lines"] = wandb.Image(lines_img, caption="Blue:Pred, Red:Aug")

                        # B. Error Map
                        if abs(aug_params['dx']) > 0 or abs(aug_params['dy']) > 0:
                            err_arr = compute_correspondence_error_map(
                                cdata['match_B'], 
                                cdata['WA'], cdata['HA'], cdata['WB'], cdata['HB'],
                                cdata['patch_size'], 
                                aug_params['dx'], aug_params['dy']
                            )
                            
                            err_max = err_arr.max()
                            err_mean = err_arr.mean()
                            
                            err_heatmap = render_heatmap_from_array(err_arr, vmax=50, cmap="inferno")
                            viz_dict["images/corr_error_map"] = wandb.Image(
                                err_heatmap, 
                                caption=f"Patch Error (px) | Max: {err_max:.1f}, Mean: {err_mean:.1f}"
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