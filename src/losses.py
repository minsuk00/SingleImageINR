import lpips  # LPIPS (expects 3-ch)
import torch
import torch.nn as nn
import torch.nn.functional as F
from pytorch_msssim import ssim
from torchvision import models
from transformers import AutoImageProcessor, AutoModel


class LossBundle:
    def __init__(self, cfg, device):
        # Pixel-level (L1/L2)
        self.pixel = cfg["loss"]["pixel"]
        self.pixel_w = float(cfg["loss"]["pixel_weight"])

        # Structural (SSIM)
        self.ssim_w = float(cfg["loss"]["ssim_weight"])
        self.ssim_ks = int(cfg["loss"]["ssim_window"])

        # Perceptual (LPIPS / DINO / CLIP etc.)
        self.lpips_w = float(cfg["loss"]["lpips_weight"])
        self.lpips_model = None
        if self.lpips_w > 0.0:
            self.lpips_model = lpips.LPIPS(net="vgg").to(device).eval()

        # Medical perceptual (CheSS)
        self.chess_w = float(cfg["loss"].get("chess_weight", 0.0))
        self.chess_model = None
        if self.chess_w > 0.0:
            self.chess_model = self._load_chess_model(cfg["loss"]["chess_ckpt"], device)

        # DINOv3 (semantic perceptual)
        self.dino_w = float(cfg["loss"].get("dino_weight", 0.0))
        self.dino_model = None
        self.dino_processor = None
        self.dino_layer = cfg["loss"].get("dino_layer", "avg")

        if self.dino_w > 0.0:
            model_name = cfg["loss"].get(
                "dino_model", "facebook/dinov3-convnext-tiny-pretrain-lvd1689m"
            )
            self.dino_model, self.dino_processor = self._load_dino_model(
                model_name, device
            )
            print(f"[DINOv3] loaded {model_name} for perceptual loss")

        self.device = device

    # --------------------------------------------------------
    # Pixel loss (L1 or L2)
    # --------------------------------------------------------
    def pixel_loss(self, pred_img, target_img):
        if self.pixel == "l1":
            return self.pixel_w * (pred_img - target_img).abs().mean()
        elif self.pixel == "l2":
            return self.pixel_w * F.mse_loss(pred_img, target_img)
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
    # CheSS perceptual loss (grayscale)
    # --------------------------------------------------------
    def _load_chess_model(self, ckpt_path, device):
        model = models.resnet50(num_classes=1000)
        model.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
        checkpoint = torch.load(ckpt_path, map_location=device)
        state_dict = checkpoint["state_dict"]
        new_state = {
            k[len("module.encoder_q.") :]: v
            for k, v in state_dict.items()
            if k.startswith("module.encoder_q")
            and not k.startswith("module.encoder_q.fc")
        }
        model.load_state_dict(new_state, strict=False)
        model = model.to(device).eval()
        return model

    @torch.no_grad()
    def _extract_chess_feat(self, model, img):
        # img: [B,1,H,W] in [0,1]
        img = img.float()  # ensure FP32
        x = (img - 0.485) / 0.229
        x = F.interpolate(x, size=224, mode="bilinear", align_corners=False)
        with torch.cuda.amp.autocast(enabled=False):
            x = model.conv1(x)
            x = model.bn1(x)
            x = model.relu(x)
            x = model.maxpool(x)
            x = model.layer1(x)
            x = model.layer2(x)
            x = model.layer3(x)
            x = model.layer4(x)
            x = model.avgpool(x)
            x = torch.flatten(x, 1)
        return x

    def chess_loss(self, pred_img, target_img):
        if self.chess_w <= 0.0 or self.chess_model is None:
            return 0.0
        feat_pred = self._extract_chess_feat(self.chess_model, pred_img)
        feat_targ = self._extract_chess_feat(self.chess_model, target_img)
        loss = F.mse_loss(feat_pred, feat_targ)
        return self.chess_w * loss

    # --------------------------------------------------------
    # DINOv3 perceptual loss (semantic)
    # --------------------------------------------------------
    def _load_dino_model(self, model_name, device):
        processor = AutoImageProcessor.from_pretrained(model_name)
        model = AutoModel.from_pretrained(model_name).to(device).eval()
        return model, processor

    @torch.no_grad()
    def _extract_dino_feat(self, model, processor, img, layer="avg"):
        # img: [B, C, H, W] in [0,1]; DINO expects 3ch RGB
        if img.shape[1] == 1:
            img = self._to_3ch(img)
        img = img.clamp(0, 1)
        inputs = processor(images=[x for x in img], return_tensors="pt").to(img.device)
        outputs = model(**inputs, output_hidden_states=True)

        last_hidden = outputs.last_hidden_state  # [B, tokens, D]
        if layer == "cls":
            feat = last_hidden[:, 0, :]  # CLS token
        else:
            feat = last_hidden[:, 1:, :].mean(dim=1)  # mean of patch embeddings
        feat = F.normalize(feat, dim=-1)
        return feat

    def dino_loss(self, pred_img, target_img):
        if self.dino_w <= 0.0 or self.dino_model is None:
            return 0.0
        feat_pred = self._extract_dino_feat(
            self.dino_model, self.dino_processor, pred_img, self.dino_layer
        )
        feat_targ = self._extract_dino_feat(
            self.dino_model, self.dino_processor, target_img, self.dino_layer
        )
        # loss = (1 - (feat_pred * feat_targ).sum(dim=-1)).mean()  # cosine distance
        loss = F.mse_loss(feat_pred, feat_targ)
        return self.dino_w * loss

    # --------------------------------------------------------
    # Total loss (combined)
    # --------------------------------------------------------
    def total(self, pred_img, target_img):
        """
        Calculates the total loss and returns it, along with a dictionary
        of the individual, unweighted loss components for logging.
        """
        loss_dict = {}
        total_loss = torch.tensor(0.0, device=self.device)

        # Calculate each loss component
        l_pix = self.pixel_loss(pred_img, target_img)
        l_ssim = self.structural_loss(pred_img, target_img)
        l_lpips = self.perceptual_loss(pred_img, target_img)
        l_chess = self.chess_loss(pred_img, target_img)
        l_dino = self.dino_loss(pred_img, target_img)

        # Add to total loss and log dictionary if they are active
        if isinstance(l_pix, torch.Tensor):
            total_loss += l_pix
            loss_dict['pixel'] = l_pix.item()
        if isinstance(l_ssim, torch.Tensor):
            total_loss += l_ssim
            loss_dict['ssim'] = l_ssim.item()
        if isinstance(l_lpips, torch.Tensor):
            total_loss += l_lpips
            loss_dict['lpips'] = l_lpips.item()
        if isinstance(l_chess, torch.Tensor):
            total_loss += l_chess
            loss_dict['chess'] = l_chess.item()
        if isinstance(l_dino, torch.Tensor):
            total_loss += l_dino
            loss_dict['dino'] = l_dino.item()

        return total_loss, loss_dict


if __name__ == "__main__":
    import torch

    ckpt = torch.load("/home/minsukc/SIO/CheSS.pth.tar", map_location="cpu")
    state_dict = ckpt["state_dict"]

    # Find conv1 weight
    for k, v in state_dict.items():
        if "conv1.weight" in k:
            print(k, v.shape)
            break
