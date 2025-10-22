import argparse
import os
from datetime import datetime

import torch
import yaml

# import wandb
from aug import sample_augmentation
from losses import LossBundle
from models import build_model
from utils import load_gray_image, psnr, render_full, save_image01, set_seed
from skimage.metrics import structural_similarity as ssim


def main(cfg):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    set_seed(cfg["seed"])

    # --- W&B Initialization ---
    # timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    # run_name = f"{cfg['exp_name']}_{timestamp}"
    # wandb.init(
    #     project=cfg.get("wandb_project", "inir-project"), name=run_name, config=cfg
    # )

    # data [B,C,H,W]
    clean_img, (H, W) = load_gray_image(
        cfg["image_path"], normalize_to="0_1", device=device
    )  # [1,1,H,W]

    # model
    model = build_model(cfg, device)
    opt = torch.optim.Adam(model.parameters(), lr=float(cfg["lr"]))
    losses = LossBundle(cfg, device)

    # Automatically append datetime (e.g. runs/exp1_2025-10-13_23-45-02)
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    out_dir = os.path.join(cfg["out_dir"], timestamp)
    os.makedirs(out_dir, exist_ok=True)
    save_image01(clean_img, f"{out_dir}/clean.png")

    # --- Log initial ground truth image to W&B ---
    # wandb.log({"ground_truth": wandb.Image(clean_img)})

    # print(f"Starting training for run: {run_name}")
    for step in range(cfg["epochs"]):
        # 1) sample augmentation
        I_aug = sample_augmentation(cfg, clean_img)  # [1,1,H,W]

        # 2) render model's current prediction (canonical)
        pred_img = render_full(model, H, W, device)  # [1,1,H,W]

        # 3) compute loss directly on full images
        # total_loss = losses.total(pred_img, I_aug)
        total_loss, loss_dict = losses.total(pred_img, I_aug)

        # 4) optimize
        opt.zero_grad(set_to_none=True)
        total_loss.backward()
        opt.step()

        # 5) logging / visualization
        if (step + 1) % 50 == 0:
            psnr_val = psnr(pred_img, clean_img)
            ssim_val = ssim(pred_img.squeeze().cpu().detach().numpy(), clean_img.squeeze().cpu().detach().numpy(), data_range=1.0)
            print(
                f"[{step+1}/{cfg['epochs']}] loss={total_loss.item():.4f} psnr_clean={psnr_val.item():.2f}dB ssim_clean={ssim_val:.4f}"
            )

            # --- Log metrics to W&B ---
            log_data = {
                "total_loss": total_loss.item(),
                "psnr_clean": psnr_val.item(),
            }
            # Add individual losses from the dictionary
            log_data.update({f"loss/{k}": v for k, v in loss_dict.items()})
            # wandb.log(log_data, step=step + 1)

        if (step + 1) % cfg["viz_every"] == 0 or (step + 1) == cfg["epochs"]:
            save_image01(pred_img, f"{out_dir}/recon_step{step+1}.png")
            save_image01(I_aug, f"{out_dir}/aug_step{step+1}.png")

            # --- Log images to W&B ---
            # wandb.log(
            #     {
            #         "reconstruction": wandb.Image(pred_img),
            #         "last_augmentation": wandb.Image(I_aug),
            #     },
            #     step=step + 1,
            # )

    # wandb.finish()
    print("Training finished and W&B run completed.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="config/config.yaml")
    args = parser.parse_args()
    with open(args.config, "r") as f:
        cfg = yaml.safe_load(f)

    main(cfg)
