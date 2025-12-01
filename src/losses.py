import lpips
import torch
import torch.nn as nn
import torch.nn.functional as F
from pytorch_msssim import ssim
from torchvision import models
from transformers import AutoImageProcessor, AutoModel
from PIL import Image

class LossBundle:
    def __init__(self, cfg, device, lpips_fn=None):
        self.device = device
        loss_cfg = cfg["loss"]
        
        # --- State for Visualization ---
        self.has_printed_patch_info = False
        self.patch_info = {}
        
        # Stores the most recent correspondence calculation results
        # Keys: 'match_B', 'WA', 'HA', 'WB', 'HB'
        self.latest_corr_data = None 

        # --- Weights & Configs ---
        self.pixel = loss_cfg["pixel"]
        self.pixel_w = float(loss_cfg["pixel_weight"])

        self.ssim_w = float(loss_cfg["ssim_weight"])
        self.ssim_ks = int(loss_cfg["ssim_window"])

        self.lpips_w = float(loss_cfg["lpips_weight"])
        
        # --- Models Initialization ---
        self.lpips_model = lpips_fn if self.lpips_w > 0 else None

        # CheSS
        self.chess_w = float(loss_cfg.get("chess_weight", 0.0))
        self.chess_model = None
        if self.chess_w > 0.0:
            self.chess_model = self._load_chess_model(loss_cfg["chess_ckpt"], device)

        # DINOv3
        self.dino_w = float(loss_cfg.get("dino_weight", 0.0))
        self.dino_model = None
        self.dino_layer = loss_cfg.get("dino_layer", "avg")
        if self.dino_w > 0.0:
            self.dino_model, self.dino_processor = self._load_dino_model(
                loss_cfg.get("dino_model", "facebook/dinov3-convnext-tiny-pretrain-lvd1689m"), 
                device
            )

        # SAM2
        self.sam_w = float(loss_cfg.get("sam_weight", 0.0))
        self.sam_model = None
        self.sam_layer = loss_cfg.get("sam_layer", -3)
        if self.sam_w > 0.0:
            self._init_sam(loss_cfg.get("sam_model", "facebook/sam2.1-hiera-small"), device)

        # Correspondence
        self.corr_w = float(loss_cfg.get("corr_weight", 0.0))
        self.corr_pixel_w = float(loss_cfg.get("corr_pixel_weight", 0.0))
        self.corr_model = None

        viz_corr = cfg.get("viz", {}).get("correspondence", True)

        if self.corr_w > 0 or self.corr_pixel_w > 0 or viz_corr:
            self._init_correspondence(loss_cfg, device)

    def _init_sam(self, model_name, device):
        from transformers import Sam2Model, Sam2Processor
        print(f"[SAM2] Loading {model_name}...")
        self.sam_processor = Sam2Processor.from_pretrained(model_name)
        self.sam_model = Sam2Model.from_pretrained(model_name).vision_encoder.to(device).eval()

    def _init_correspondence(self, loss_cfg, device):
        model_name = loss_cfg.get("corr_model", "facebook/dinov3-vits16-pretrain-lvd1689m")
        print(f"[CORR] Loading Backbone: {model_name}")
        self.corr_processor = AutoImageProcessor.from_pretrained(model_name)
        model = AutoModel.from_pretrained(model_name, output_hidden_states=True).to(device).eval()

        for param in model.parameters():
            param.requires_grad = False
            
        if "sam" in model_name:
            self.corr_model = model.vision_encoder
            self.corr_patch_size = 16
            self.corr_num_registers = 0
        else:
            self.corr_model = model
            self.corr_patch_size = model.config.patch_size
            self.corr_num_registers = getattr(model.config, "num_register_tokens", 0)
        self.corr_patch_layer = loss_cfg.get("corr_layer", -1)
        self.corr_search_r = loss_cfg.get("corr_search_r", None)

    def _to_3ch(self, x):
        return x.repeat(1, 3, 1, 1) if x.shape[1] == 1 else x

    def pixel_loss(self, pred_img, target_img):
        if self.pixel_w <= 0: return 0.0
        return (pred_img - target_img).abs().mean() if self.pixel == "l1" else F.mse_loss(pred_img, target_img)

    def structural_loss(self, pred_img, target_img):
        if self.ssim_w <= 0: return 0.0
        return 1.0 - ssim(pred_img.float(), target_img.float(), data_range=1.0, size_average=True, win_size=self.ssim_ks)

    def perceptual_loss(self, pred_img, target_img):
        if self.lpips_w <= 0 or self.lpips_model is None: return 0.0
        return self.lpips_model(self._to_3ch(pred_img) * 2 - 1, self._to_3ch(target_img) * 2 - 1).mean()

    # --------------------------------------------------------
    # CheSS Loss
    # --------------------------------------------------------
    def _load_chess_model(self, ckpt_path, device):
        model = models.resnet50(num_classes=1000)
        model.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
        checkpoint = torch.load(ckpt_path, map_location=device)
        
        # Handle DataParallel prefix if present
        state_dict = {k.replace("module.encoder_q.", ""): v for k, v in checkpoint["state_dict"].items() 
                      if "encoder_q" in k and "fc" not in k}
        
        model.load_state_dict(state_dict, strict=False)
        return model.to(device).eval()

    @torch.no_grad()
    def _extract_chess_feat(self, model, img):
        # img: [B,1,H,W] in [0,1]
        img = img.float()
        # CheSS specific normalization
        x = (img - 0.485) / 0.229
        x = F.interpolate(x, size=224, mode="bilinear", align_corners=False)
        
        # Forward pass through ResNet parts
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
        if self.chess_w <= 0 or self.chess_model is None: return 0.0
        
        self.chess_model.to(pred_img.device)
        feat_pred = self._extract_chess_feat(self.chess_model, pred_img)
        feat_targ = self._extract_chess_feat(self.chess_model, target_img)
        # Avoid keeping model on GPU if memory is tight, but for now assume shared
        return F.mse_loss(feat_pred, feat_targ)

    # --------------------------------------------------------
    # DINO / SAM Losses
    # --------------------------------------------------------
    def _load_dino_model(self, model_name, device):
        processor = AutoImageProcessor.from_pretrained(model_name)
        model = AutoModel.from_pretrained(model_name).to(device).eval()
        return model, processor

    def _extract_dino_feat(self, model, processor, img, layer="avg"):
        # img: [B, C, H, W] in [0,1]; DINO expects 3ch RGB
        if img.shape[1] == 1:
            img = self._to_3ch(img)
        img = img.clamp(0, 1)

        # We can use processor, or manually normalize. 
        # Using processor ensures exact match with training, but slower.
        # Here we use processor for safety.
        inputs = processor(images=[x for x in img], return_tensors="pt").to(model.device)

        outputs = model(**inputs, output_hidden_states=True)
        last_hidden = outputs.last_hidden_state
        
        if layer == "cls":
            feat = last_hidden[:, 0, :]
        else:
            feat = last_hidden[:, 1:, :].mean(dim=1)
            
        feat = F.normalize(feat, dim=-1)
        return feat

    def dino_loss(self, pred_img, target_img):
        if self.dino_w <= 0 or self.dino_model is None: return 0.0
        
        self.dino_model.to(pred_img.device)
        feat_pred = self._extract_dino_feat(self.dino_model, self.dino_processor, pred_img, self.dino_layer)
        feat_targ = self._extract_dino_feat(self.dino_model, self.dino_processor, target_img, self.dino_layer)
        return F.mse_loss(feat_pred, feat_targ) 

    @torch.no_grad()
    def _extract_sam_feat(self, model, processor, img, layer_idx=-3):
        if img.shape[1] == 1:
            img = self._to_3ch(img)
        img = img.clamp(0, 1)
        
        device = model.device
        inputs = processor(images=[x for x in img], return_tensors="pt").to(device)
        outputs = model(**inputs, output_hidden_states=True)
        feats = outputs.hidden_states[layer_idx] # [B, Hf, Wf, C]
        B, Hf, Wf, C = feats.shape
        feats = feats.reshape(B, Hf * Wf, C)
        feats = F.normalize(feats, dim=-1)
        return feats

    def sam_loss(self, pred_img, target_img):
        if self.sam_w <= 0 or self.sam_model is None: return 0.0
        
        self.sam_model.to(pred_img.device)
        feat_pred = self._extract_sam_feat(self.sam_model, self.sam_processor, pred_img, self.sam_layer)
        feat_targ = self._extract_sam_feat(self.sam_model, self.sam_processor, target_img, self.sam_layer)
        
        # Cosine similarity loss
        sim = F.cosine_similarity(feat_pred, feat_targ, dim=-1).mean()
        return 1.0 - sim

    # --------------------------------------------------------
    # Correspondence Logic (VECTORIZED)
    # --------------------------------------------------------
    def _extract_dino_patches_corr(self, img):
        img_rgb = self._to_3ch(img)
        pixel_values = self.corr_processor(images=img_rgb, do_resize=False, do_center_crop=False, return_tensors="pt").pixel_values.to(self.device)
        out = self.corr_model(pixel_values=pixel_values, output_hidden_states=True)
        layer_feats = out.hidden_states[self.corr_patch_layer] 
        patch_feats = layer_feats[:, 1 + self.corr_num_registers:, :] 
        
        H_px, W_px = img.shape[2], img.shape[3]
        Hf = H_px // self.corr_patch_size
        Wf = W_px // self.corr_patch_size
        
        if not self.has_printed_patch_info:
            self.patch_info = {
                "dino_img_width": W_px, "dino_img_height": H_px,
                "dino_patch_size": self.corr_patch_size,
                "dino_grid_cols": Wf, "dino_grid_rows": Hf,
                "dino_total_patches": Wf*Hf
            }
            self.has_printed_patch_info = True
        
        return patch_feats, Hf, Wf

    def _compute_correspondence_vectorized(self, sim, WA, HA, WB, HB, r=None):
        if r is None: return sim.argmax(dim=1)
        device = sim.device
        yA = torch.arange(HA, device=device).repeat_interleave(WA) 
        xA = torch.arange(WA, device=device).repeat(HA)            
        yB = torch.arange(HB, device=device).repeat_interleave(WB) 
        xB = torch.arange(WB, device=device).repeat(HB)            
        dy = (yA[:, None] - yB[None, :]).abs()
        dx = (xA[:, None] - xB[None, :]).abs()
        mask = (dy <= r) & (dx <= r)
        sim_masked = sim.clone()
        sim_masked[~mask] = -float('inf')
        return sim_masked.argmax(dim=1)

    @torch.no_grad()
    def compute_for_visualization(self, pred_img, target_img):
        """
        Calculates correspondence data specifically for visualization,
        even if loss weights are 0.
        """
        if self.corr_model is None:
            return

        fA, HA, WA = self._extract_dino_patches_corr(pred_img)
        fB, HB, WB = self._extract_dino_patches_corr(target_img)
        
        fA = F.normalize(fA, dim=-1)
        fB = F.normalize(fB, dim=-1)
        
        sim = torch.matmul(fA, fB.transpose(1, 2)) 
        match_B = self._compute_correspondence_vectorized(sim[0], WA, HA, WB, HB, r=self.corr_search_r)
        
        self.latest_corr_data = {
            "match_B": match_B.detach(),
            "WA": WA, "HA": HA,
            "WB": WB, "HB": HB,
            "patch_size": self.corr_patch_size
        }
        
    def correspondence_loss(self, pred_img, target_img, search_r=None):
        if self.corr_w <= 0: return 0.0
        
        fA, HA, WA = self._extract_dino_patches_corr(pred_img)
        fB, HB, WB = self._extract_dino_patches_corr(target_img)
        
        fA = F.normalize(fA, dim=-1)
        fB = F.normalize(fB, dim=-1)
        
        sim = torch.matmul(fA, fB.transpose(1, 2)) 
        match_B = self._compute_correspondence_vectorized(sim[0], WA, HA, WB, HB, r=search_r)
        
        # --- CACHE FOR VISUALIZATION ---
        self.latest_corr_data = {
            "match_B": match_B.detach(), # Indices of B matching A
            "WA": WA, "HA": HA,
            "WB": WB, "HB": HB,
            "patch_size": self.corr_patch_size
        }

        fB_match = fB[0, match_B]
        # return F.mse_loss(fA[0], fB_match) * 100.0
        # return F.mse_loss(fA[0], fB_match) * 1000.0
        return F.mse_loss(fA[0], fB_match) * 10.0

    def correspondence_pixel_loss(self, pred_img, target_img, search_r=None):
        if self.corr_pixel_w <= 0: return 0.0
        # Re-uses extract, but visualization usually relies on the feature-based 'match_B'
        # If you only use pixel loss, you might want to cache matches here too.
        # For now, assuming you use at least corr_weight > 0 or that we cache anyway.
        
        fA, HA, WA = self._extract_dino_patches_corr(pred_img)
        fB, HB, WB = self._extract_dino_patches_corr(target_img)
        fA = F.normalize(fA, dim=-1)
        fB = F.normalize(fB, dim=-1)
        sim = torch.matmul(fA, fB.transpose(1, 2))
        match_B = self._compute_correspondence_vectorized(sim[0], WA, HA, WB, HB, r=search_r)
        
        self.latest_corr_data = {
            "match_B": match_B.detach(),
            "WA": WA, "HA": HA,
            "WB": WB, "HB": HB,
            "patch_size": self.corr_patch_size
        }

        patch = self.corr_patch_size
        B, C, H, W = pred_img.shape
        
        def make_patches(x):
            # unfold [B, C, H, W] -> [N, C, patch, patch]
            kc, kh, kw = C, patch, patch
            dc, dh, dw = C, patch, patch
            patches = x.unfold(2, kh, dh).unfold(3, kw, dw)
            patches = patches.permute(0, 2, 3, 1, 4, 5)
            patches = patches.reshape(-1, C, patch, patch)
            return patches

        patches_A = make_patches(pred_img)
        patches_B = make_patches(target_img) 
        matched_B_patches = patches_B[match_B]
        
        return F.l1_loss(patches_A, matched_B_patches)

    # --------------------------------------------------------
    # Total Logic
    # --------------------------------------------------------
    def total(self, pred_img, target_img):
        loss_dict = {}
        total_loss = torch.tensor(0.0, device=self.device)

        # Basic
        if self.pixel_w > 0:
            l_pix = self.pixel_loss(pred_img, target_img)
            loss_dict["pixel"] = l_pix.item()
            total_loss += l_pix * self.pixel_w
            
        if self.ssim_w > 0:
            l_ssim = self.structural_loss(pred_img, target_img)
            loss_dict["ssim"] = l_ssim.item()
            total_loss += l_ssim * self.ssim_w
            
        if self.lpips_w > 0:
            l_lpips = self.perceptual_loss(pred_img, target_img)
            loss_dict["lpips"] = l_lpips.item()
            total_loss += l_lpips * self.lpips_w

        # Advanced (CheSS, DINO, SAM)
        if self.chess_w > 0:
            l_chess = self.chess_loss(pred_img, target_img)
            loss_dict["chess"] = l_chess.item()
            total_loss += l_chess * self.chess_w
            
        if self.dino_w > 0:
            l_dino = self.dino_loss(pred_img, target_img)
            loss_dict["dino"] = l_dino.item()
            total_loss += l_dino * self.dino_w

        if self.sam_w > 0:
            l_sam = self.sam_loss(pred_img, target_img)
            loss_dict["sam"] = l_sam.item()
            total_loss += l_sam * self.sam_w

        # Correspondence
        if self.corr_w > 0:
            l_corr = self.correspondence_loss(pred_img, target_img, self.corr_search_r)
            loss_dict["corr"] = l_corr.item()
            total_loss += l_corr * self.corr_w
            
        if self.corr_pixel_w > 0:
            l_c_pix = self.correspondence_pixel_loss(pred_img, target_img, self.corr_search_r)
            loss_dict["corr_pixel"] = l_c_pix.item()
            total_loss += l_c_pix * self.corr_pixel_w

        return total_loss, loss_dict