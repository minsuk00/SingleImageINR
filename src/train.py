import argparse
import os
from datetime import datetime

import torch
import yaml

import wandb
import lpips 
from aug import sample_augmentation
from losses import LossBundle
from models import build_model
from utils import load_gray_image, psnr, render_full, save_image01, set_seed
from skimage.metrics import structural_similarity as ssim

# NOTE: This main function will be imported and called by run_sweep.py
def main(cfg):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    set_seed(cfg["seed"])

    # --- W&B Initialization ---
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    run_name = f"{cfg['exp_name']}_{timestamp}"
    
    wandb.init(
        project=cfg.get("wandb_project", "inir-project"), 
        name=run_name, 
        config=cfg,
        group=cfg.get("sweep_group", "default-group")
    )

    # data [B,C,H,W]
    clean_img, (H, W) = load_gray_image(
        cfg["image_path"], normalize_to="0_1", device=device
    )  # [1,1,H,W]
    
    # --- Load LPIPS on CPU to share ---
    # We use 'vgg' to match the network you had in losses.py
    # It lives on the CPU to save VRAM.
    lpips_fn = lpips.LPIPS(net='vgg')

    # model
    model = build_model(cfg, device)
    opt = torch.optim.Adam(model.parameters(), lr=float(cfg["lr"]))
    
    # --- Pass the shared LPIPS model to the LossBundle ---
    losses = LossBundle(cfg, device, lpips_fn=lpips_fn)

    # --- Setup output directory ---
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    out_dir = os.path.join(cfg["out_dir"], timestamp)
    os.makedirs(out_dir, exist_ok=True)
    save_image01(clean_img, f"{out_dir}/clean.png")

    # --- Log initial ground truth image ---
    wandb.log({"images/ground_truth": wandb.Image(clean_img.squeeze(0))})

    print(f"--- Starting training for run: {run_name} ---")
    
    for step in range(cfg["epochs"]):
        # 1) sample augmentation
        I_aug = sample_augmentation(cfg, clean_img)  # [1,1,H,W]

        # 2) render model's current prediction (canonical)
        pred_img = render_full(model, H, W, device)  # [1,1,H,W]

        # 3) compute loss directly on full images
        total_loss, loss_dict = losses.total(pred_img, I_aug)

        # 4) optimize
        opt.zero_grad(set_to_none=True)
        total_loss.backward()
        opt.step()

        # 5) logging / visualization
        if (step + 1) % 50 == 0:
            psnr_val = psnr(pred_img, clean_img)
            ssim_val = ssim(pred_img.squeeze().cpu().detach().numpy(), clean_img.squeeze().cpu().detach().numpy(), data_range=1.0)
            
            # --- Calculate LPIPS metric (using the same shared model) ---
            lpips_fn.to(device) # Move to GPU
            pred_img_lpips = pred_img * 2.0 - 1.0
            clean_img_lpips = clean_img * 2.0 - 1.0
            with torch.no_grad():
                lpips_val = lpips_fn(pred_img_lpips, clean_img_lpips).item()
            lpips_fn.to('cpu') # Move back to CPU
            
            print(
                f"[{step+1}/{cfg['epochs']}] loss={total_loss.item():.4f} psnr_clean={psnr_val.item():.2f}dB ssim_clean={ssim_val:.4f} lpips_clean={lpips_val:.4f}"
            )

            # --- Log metrics to W&B ---
            log_data = {
                "total_loss": total_loss.item(),
                "psnr_clean": psnr_val.item(),
                "ssim_clean": ssim_val,
                "lpips_clean": lpips_val,
                "learning_rate": opt.param_groups[0]['lr']
            }
            log_data.update({f"loss/{k}": v for k, v in loss_dict.items()})
            wandb.log(log_data, step=step + 1)

        if (step + 1) % cfg["viz_every"] == 0 or (step + 1) == cfg["epochs"]:
            save_image01(pred_img, f"{out_dir}/recon_step{step+1}.png")
            save_image01(I_aug, f"{out_dir}/aug_step{step+1}.png")
            
            diff_img = torch.abs(clean_img - pred_img)
            save_image01(diff_img, f"{out_dir}/diff_step{step+1}.png")

            # --- Log images to W&B ---
            wandb.log(
                {
                    "images/reconstruction": wandb.Image(pred_img.squeeze(0)),
                    "images/last_augmentation": wandb.Image(I_aug.squeeze(0)),
                    "images/difference": wandb.Image(diff_img.squeeze(0)),
                },
                step=step + 1,
            )

    wandb.finish()
    print(f"--- Finished run: {run_name} ---")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="/home/sincheol/SingleImageINR/config/config.yaml")
    args = parser.parse_args()
    with open(args.config, "r") as f:
        cfg = yaml.safe_load(f)
    
    # Set a default group name for single runs
    cfg["sweep_group"] = "single_run"
    main(cfg)

