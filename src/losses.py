import lpips  # LPIPS (expects 3-ch)
import torch
import torch.nn.functional as F
from pytorch_msssim import ssim


class LossBundle:
    def __init__(self, cfg, device):
        # Pixel-level (L1/L2)
        self.pixel = cfg["loss"]["pixel"]

        # Structural (SSIM)
        self.ssim_w = float(cfg["loss"]["ssim_weight"])
        self.ssim_ks = int(cfg["loss"]["ssim_window"])

        # Perceptual (LPIPS / DINO / CLIP etc.)
        self.lpips_w = float(cfg["loss"]["lpips_weight"])
        self.lpips_model = None
        if self.lpips_w > 0.0:
            self.lpips_model = lpips.LPIPS(net="vgg").to(device).eval()

        self.device = device

    # --------------------------------------------------------
    # Pixel loss (L1 or L2)
    # --------------------------------------------------------
    def pixel_loss(self, pred_img, target_img):
        if self.pixel == "l1":
            return (pred_img - target_img).abs().mean()
        elif self.pixel == "l2":
            return F.mse_loss(pred_img, target_img)
        else:
            raise ValueError("pixel must be 'l1' or 'l2'")

    # --------------------------------------------------------
    # Structural loss (SSIM)
    # --------------------------------------------------------
    def structural_loss(self, pred_img, target_img):
        if self.ssim_w <= 0.0:
            return 0.0

        ssim_loss = 1.0 - ssim(
            pred_img.float(),
            target_img.float(),
            data_range=1.0,
            size_average=True,
            win_size=self.ssim_ks,
        )
        return self.ssim_w * ssim_loss

    # --------------------------------------------------------
    # Perceptual loss (LPIPS / feature-based)
    # --------------------------------------------------------
    @torch.no_grad()
    def _to_3ch(self, x):  # [B,1,H,W] -> [B,3,H,W]
        return x.repeat(1, 3, 1, 1)

    def perceptual_loss(self, pred_img, target_img):
        if self.lpips_w <= 0.0 or self.lpips_model is None:
            return 0.0
        # LPIPS expects [-1,1] RGB input
        pred = self._to_3ch(pred_img) * 2 - 1
        targ = self._to_3ch(target_img) * 2 - 1
        lp = self.lpips_model(pred, targ).mean()
        return self.lpips_w * lp

    # --------------------------------------------------------
    # Total loss (combined)
    # --------------------------------------------------------
    def total(self, pred_img, target_img):
        loss = self.pixel_loss(pred_img, target_img)
        loss += self.structural_loss(pred_img, target_img)
        loss += self.perceptual_loss(pred_img, target_img)
        return loss
