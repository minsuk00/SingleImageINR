# ==========================================
# 1. Imports & Setup
# ==========================================
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from PIL import Image
from transformers import AutoModel
from tqdm.auto import tqdm
import os
import wandb
import gc
# import tinycudann as tcnn

def set_seed(seed=42):
    torch.manual_seed(seed)
    np.random.seed(seed)
    torch.backends.cudnn.deterministic = True

set_seed(42)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Running on device: {device}")

# ==========================================
# 2. Helper Functions
# ==========================================
def load_image_dynamic(base_path, res):
    """Loads image based on resolution string"""
    # Assumes file name format: ct_chest_2_resized_{RES}x{RES}.png
    # Adjust the path formatting if your file naming is different
    path = f"{base_path}/ct_chest_2_resized_{res}x{res}.png"
    
    try:
        img = Image.open(path).convert('L')
    except Exception as e:
        print(f"Error loading image at {path}: {e}")
        return None
        
    img = img.resize((res, res)) # Ensure exact resize
    img_tensor = torch.from_numpy(np.array(img)).float() / 255.0
    return img_tensor.unsqueeze(0).unsqueeze(0).to(device)
    
def get_grad_norm(model):
    total_norm = 0.0
    for p in model.parameters():
        if p.grad is not None:
            param_norm = p.grad.data.norm(2)
            total_norm += param_norm.item() ** 2
    return total_norm ** 0.5

# ==========================================
# 3. Models
# ==========================================
class SimpleINR(nn.Module):
    def __init__(self, hidden_dim=256, num_layers=4):
        super().__init__()
        layers = []
        # Input: (x, y) coords -> Output: (gray)
        layers.append(nn.Linear(2, hidden_dim))
        layers.append(nn.ReLU())
        for _ in range(num_layers - 2):
            layers.append(nn.Linear(hidden_dim, hidden_dim))
            layers.append(nn.ReLU())
        layers.append(nn.Linear(hidden_dim, 1))
        layers.append(nn.Sigmoid()) 
        self.net = nn.Sequential(*layers)

    def forward(self, H, W):
        ys = torch.linspace(-1, 1, H, device=device)
        xs = torch.linspace(-1, 1, W, device=device)
        yy, xx = torch.meshgrid(ys, xs, indexing='ij')
        coords = torch.stack([xx, yy], dim=-1).reshape(-1, 2)
        out = self.net(coords)
        return out.reshape(1, 1, H, W)

class FourierINR(nn.Module):
    def __init__(self, hidden_dim=256, num_layers=4, mapping_size=256, scale=10.0):
        super().__init__()
        self.mapping_size = mapping_size
        self.scale = scale
        
        self.register_buffer('B', torch.randn((2, mapping_size)) * scale)
        
        layers = []
        # Input: (sin, cos) features * mapping_size -> Output: (gray)
        # 입력 차원이 2 -> 2 * mapping_size로 뻥튀기됨
        layers.append(nn.Linear(2 * mapping_size, hidden_dim))
        layers.append(nn.ReLU())
        for _ in range(num_layers - 2):
            layers.append(nn.Linear(hidden_dim, hidden_dim))
            layers.append(nn.ReLU())
        layers.append(nn.Linear(hidden_dim, 1))
        layers.append(nn.Sigmoid())
        self.net = nn.Sequential(*layers)

    def forward(self, H, W):
        ys = torch.linspace(-1, 1, H, device=self.B.device)
        xs = torch.linspace(-1, 1, W, device=self.B.device)
        yy, xx = torch.meshgrid(ys, xs, indexing='ij')
        coords = torch.stack([xx, yy], dim=-1).reshape(-1, 2) # [HW, 2]
        
        # [Fourier Feature Mapping]
        # x -> [sin(2pi * B @ x), cos(2pi * B @ x)]
        x_proj = (2.0 * np.pi * coords) @ self.B # [HW, mapping_size]
        emb = torch.cat([torch.sin(x_proj), torch.cos(x_proj)], dim=-1) # [HW, 2*mapping_size]
        
        out = self.net(emb)
        return out.reshape(1, 1, H, W)

class TCNN_INR(nn.Module):
    def __init__(self, device):
        super().__init__()
        encoding_config = {
            "otype": "Grid",
            "type": "Hash",
            "n_levels": 12,
            "n_features_per_level": 4,
            "log2_hashmap_size": 20,
            "base_resolution": 16,
            "per_level_scale": 1.5,
            "interpolation": "Linear"
        }
        network_config = {
            "otype": "FullyFusedMLP",
            "activation": "ReLU",
            "output_activation": "None",
            # "output_activation": "Sigmoid",
            "n_neurons": 128,
            "n_hidden_layers": 2,
        }

        self.model = tcnn.NetworkWithInputEncoding(
            n_input_dims=2,
            n_output_dims=1,
            encoding_config=encoding_config,
            network_config=network_config,
        ).to(device)

    def forward(self, H, W):
        # TCNN expects inputs in [0, 1] range
        ys = torch.linspace(0, 1, H, device=device)
        xs = torch.linspace(0, 1, W, device=device)
        yy, xx = torch.meshgrid(ys, xs, indexing='ij')
        coords = torch.stack([xx, yy], dim=-1).reshape(-1, 2) # [HW, 2]
        
        out = self.model(coords)
        return out.reshape(1, 1, H, W)

# (2) MASt3R-Style Matching Head
class MatchingHead(nn.Module):
    def __init__(self, input_dim=384, output_dim=24, patch_size=16):
        super().__init__()
        self.patch_size = patch_size
        # MLP: DINO feat -> (24 * 16 * 16) expansion
        self.mlp = nn.Sequential(
            nn.Linear(input_dim, 512),
            nn.GELU(),
            nn.Linear(512, output_dim * (patch_size ** 2))
        )

    def forward(self, x, H, W):
        # x: DINO Patch Features [B, N, 384]
        B, N, C = x.shape
        x = self.mlp(x) # [B, N, 24*256]
        
        # Reshape to grid
        H_grid, W_grid = H // self.patch_size, W // self.patch_size
        x = x.transpose(1, 2).reshape(B, -1, H_grid, W_grid)
        
        # Pixel Shuffle [B, 24, H, W]
        x = F.pixel_shuffle(x, self.patch_size)
        
        # Cosine Similarity를 위해 Normalize 수행
        return F.normalize(x, dim=1) 

# Augmentation Manager 
class AugmentationManager:
    def __init__(self, infinite=False, num_fixed=10, max_shift=15.0, H=256, W=256, device="cuda", aug_mode="rigid", elastic_alpha=0.1):
        self.infinite = infinite
        self.max_shift = max_shift
        self.H = H
        self.W = W
        self.device = device
        self.aug_mode = aug_mode
        self.elastic_alpha = elastic_alpha  # Scale of deformation
        
        # 고정된 Augmentation 미리 생성
        self.fixed_transforms = []
        if not infinite:
            print(f"[Aug] Generating {num_fixed} fixed augmentations (Mode: {aug_mode})...")
            for _ in range(num_fixed):
                self.fixed_transforms.append(self._generate_single_transform())
    
    def _generate_single_transform(self):
        if self.aug_mode == "rigid":
            sx = np.random.uniform(-self.max_shift, self.max_shift)
            sy = np.random.uniform(-self.max_shift, self.max_shift)
            return {"type": "rigid", "val": (sx, sy)}

        elif self.aug_mode == "elastic":
            # 1. Create coarse random vectors (e.g. 4x4 or 8x8 grid)
            grid_h, grid_w = 6, 6
            # Random offsets in range [-1, 1] * alpha
            coarse_flow = (torch.rand(1, 2, grid_h, grid_w, device=self.device) - 0.5) * 2 * self.elastic_alpha

            # 2. Upsample to image size (H, W) using Bicubic
            # flow shape: [1, 2, H, W]
            flow = F.interpolate(coarse_flow, size=(self.H, self.W), mode="bicubic", align_corners=False)

            # 3. Permute for grid_sample: [1, H, W, 2]
            flow = flow.permute(0, 2, 3, 1)

            return {"type": "elastic", "val": flow}

    def get_transform(self):
        if self.infinite:
            return self._generate_single_transform()
        else:
            idx = np.random.randint(0, len(self.fixed_transforms))
            return self.fixed_transforms[idx]
            

# ==========================================
# 4. Joint Trainer
# ==========================================
class JointTrainer:
    def __init__(self, device, config):
        self.device = device
        self.cfg = config
        self.use_corr = config.get("use_correspondence", True)
        
        # 1. Models
        print("Loading Models...")
        model_type = config.get("model_type", "simple")
        if model_type == "tcnn":
            print("Using TCNN INR")
            self.inr = TCNN_INR(device)
        elif model_type == "fourier":
            print("Using Fourier INR")
            self.inr = FourierINR(hidden_dim=256).to(device)
        else:
            print("Using Simple INR")
            self.inr = SimpleINR().to(device)
        
        if self.use_corr:
            self.dino = AutoModel.from_pretrained('facebook/dinov3-vits16-pretrain-lvd1689m').to(device).eval()
            for p in self.dino.parameters(): p.requires_grad = False
            # self.head = MatchingHead().to(device)
            self.head_ref = MatchingHead().to(device)
            self.head_mov = MatchingHead().to(device)
            
            # Optimizer for Head + INR
            self.optimizer = torch.optim.Adam([
                {'params': self.inr.parameters(), 'lr': config["lr_inr"]},
                # {'params': self.head.parameters(), 'lr': config["lr_head"]}
                {'params': self.head_ref.parameters(), 'lr': config["lr_head"]},
                {'params': self.head_mov.parameters(), 'lr': config["lr_head"]}
            ])
        else:
            # Optimizer for INR only
            self.optimizer = torch.optim.Adam([
                {'params': self.inr.parameters(), 'lr': config["lr_inr"]}
            ])

        # Utils
        self.mean = torch.tensor([0.485, 0.456, 0.406], device=device).view(1, 3, 1, 1)
        self.std = torch.tensor([0.229, 0.224, 0.225], device=device).view(1, 3, 1, 1)
        self.ce_loss = nn.CrossEntropyLoss()
        
        # State
        self.augmentor = AugmentationManager(
            infinite=config["infinite_augmentation"],
            num_fixed=config["num_aug"],
            max_shift=config["max_shift_px"],
            H=config["res"],
            W=config["res"],
            device=device,
            aug_mode=config["aug_mode"], 
            elastic_alpha=config["elastic_alpha"],
        )
        self.latest_viz_data = {}

    def extract_desc(self, img, is_moving=False):
        if not self.use_corr: return None
        B, C, H, W = img.shape
        img_3ch = img.repeat(1, 3, 1, 1)
        img_norm = (img_3ch - self.mean) / self.std
        
        H_grid = H // 16; W_grid = W // 16
        num_patches = H_grid * W_grid
        
        with torch.no_grad():
            out = self.dino(pixel_values=img_norm, output_hidden_states=True)
            patch_feats = out.last_hidden_state[:, -num_patches:, :]
        
        # return self.head(patch_feats, H, W)
        if is_moving:
            return self.head_mov(patch_feats, H, W)
        else:
            return self.head_ref(patch_feats, H, W)
    
    def train_step(self, clean_img, step):
        self.optimizer.zero_grad()
        B, C, H, W = clean_img.shape
        
        # --- A. Augmentation ---
        aug_data = self.augmentor.get_transform()
        
        # Base Identity Grid
        theta_identity = torch.tensor([[1, 0, 0], [0, 1, 0]], dtype=torch.float, device=self.device).unsqueeze(0)
        base_grid = F.affine_grid(theta_identity, clean_img.size(), align_corners=False)
        
        # Variables to store pixel-wise shift for Loss calculation
        gt_flow_y = None # Shape (H, W) or Scalar
        gt_flow_x = None
        
        if aug_data['type'] == 'rigid':
            shift_x, shift_y = aug_data['val']
            # Convert px shift to normalized [-1, 1] shift for grid_sample
            # Note: grid_sample(x, y) samples from input at (x, y). 
            # To shift image RIGHT by 10px, we sample from LEFT (x-10).
            # So tx should be negative of desired visual shift.
            tx = -shift_x / (W / 2.0)
            ty = -shift_y / (H / 2.0)
            
            # Create Rigid Grid
            theta = torch.tensor([[1, 0, tx], [0, 1, ty]], dtype=torch.float, device=self.device).unsqueeze(0)
            grid = F.affine_grid(theta, clean_img.size(), align_corners=False)
            
            # Store scalar shifts for loss
            gt_flow_y = shift_y
            gt_flow_x = shift_x
            
        elif aug_data['type'] == 'elastic':
            # Elastic Logic
            # We generated a flow field representing Canonical -> Target displacement.
            # grid_sample is a "pull" operation (Target[x] = Clean[x + offset]).
            # To make the logic consistent (Target[x + flow] = Clean[x]), we use 
            # grid = identity - flow. This allows us to use straight addition 
            # for the loss coordinates later.
            flow = aug_data['val'] # [1, H, W, 2]
            grid = base_grid - flow
            
            # Save flow for loss (convert to pixels)
            # flow is (1, H, W, 2).
            gt_flow_y = flow[0, ..., 1] * (H / 2.0)
            gt_flow_x = flow[0, ..., 0] * (W / 2.0)
            
        target_img = F.grid_sample(clean_img, grid, align_corners=False) # Shifted GT
        
        # --- B. Rendering ---
        pred_img = self.inr(H, W)
        
        logs = {}
        total_loss = 0.0
        
        if self.use_corr:
            # --- C. InfoNCE & Alignment ---
            # Use separate heads for Pred (Ref) and Target (Mov)
            desc_pred = self.extract_desc(pred_img, is_moving=False)
            desc_target = self.extract_desc(target_img, is_moving=True)
            
            # (1) InfoNCE Loss
            # GT Transformation에 의해 매칭된 좌표만 짝꿍(Positive)이고 나머지는 전부 가짜(Negative)로 취급
            # num_samples = 4096
            num_samples = 8192
            flat_inds = torch.randperm(H * W, device=self.device)[:num_samples]
            ys = flat_inds // W
            xs = flat_inds % W
            
            feats_q = desc_pred[0, :, ys, xs].T
            
            # ys_tgt = torch.clamp(ys + int(round(shift_y)), 0, H-1)
            # xs_tgt = torch.clamp(xs + int(round(shift_x)), 0, W-1)
            # feats_k_pos = desc_target[0, :, ys_tgt, xs_tgt].T

            if aug_data['type'] == 'rigid':
                ys_tgt = torch.clamp(ys + int(round(gt_flow_y)), 0, H-1)
                xs_tgt = torch.clamp(xs + int(round(gt_flow_x)), 0, W-1)
            elif aug_data['type'] == 'elastic':
                # Retrieve flow at these specific coordinates
                # gt_flow_y is (H, W) tensor
                dy = gt_flow_y[ys, xs]
                dx = gt_flow_x[ys, xs]
                ys_tgt = torch.clamp(ys + torch.round(dy).long(), 0, H-1)
                xs_tgt = torch.clamp(xs + torch.round(dx).long(), 0, W-1)

            feats_k_pos = desc_target[0, :, ys_tgt, xs_tgt].T

            temp = 0.05
            logits = torch.matmul(feats_q, feats_k_pos.T) / temp
            labels = torch.arange(num_samples, device=self.device)
            
            # 두 가지 항 구현: Pred->Target & Target->Pred
            loss_nce_q2k = self.ce_loss(logits, labels)    # Row-wise Softmax
            loss_nce_k2q = self.ce_loss(logits.T, labels)  # Column-wise Softmax
            loss_nce = (loss_nce_q2k + loss_nce_k2q) / 2.0
            
            # loss_nce = self.ce_loss(logits, labels)
            
            # (2) Alignment Loss (Nearest Neighbor)
            # with torch.no_grad():
            flat_target = desc_target.view(24, -1)
            
            # Global Search for Pred Samples
            sim = torch.matmul(feats_q.detach(), flat_target)
            match_idx = sim.argmax(dim=1)
            
            # GT Shift-based Border Masking Logic
            if self.cfg.get("use_border_mask", False) and aug_data['type'] == 'rigid':
                match_ys = match_idx // W
                match_xs = match_idx % W
                
                # If shifted right (shift_x > 0), x must be >= shift_x
                # If shifted left (shift_x < 0), x must be < W + shift_x
                x_valid_min = max(0, shift_x)
                x_valid_max = min(W, W + shift_x)
                
                y_valid_min = max(0, shift_y)
                y_valid_max = min(H, H + shift_y)
                
                # Check if matched coordinates fall within the valid image region
                mask_x = (match_xs >= x_valid_min) & (match_xs < x_valid_max)
                mask_y = (match_ys >= y_valid_min) & (match_ys < y_valid_max)
                valid_mask = mask_x & mask_y
                
                if valid_mask.sum() > 0:
                    pred_pixels = pred_img.view(1, -1)[:, flat_inds][:, valid_mask].T
                    target_pixels = target_img.view(1, -1)[:, match_idx][:, valid_mask].T
                    loss_align = F.mse_loss(pred_pixels, target_pixels)
                else:
                    loss_align = torch.tensor(0.0, device=self.device, requires_grad=True)
            else:
                # Fallback to no mask
                pred_pixels = pred_img.view(1, -1)[:, flat_inds].T
                target_pixels = target_img.view(1, -1)[:, match_idx].T
                loss_align = F.mse_loss(pred_pixels, target_pixels)
            # pred_pixels = pred_img.view(1, -1)[:, flat_inds].T
            # target_pixels = target_img.view(1, -1)[:, match_idx].T
            # loss_align = F.mse_loss(pred_pixels, target_pixels)

            naive_loss = F.mse_loss(pred_img, target_img)
            w_align = self.cfg["w_align"]
            w_naive = self.cfg["w_naive"]
            
            # total_loss = loss_nce + (loss_align * w_align) + (naive_loss * w_naive) 
            total_loss = loss_nce + (loss_align * w_align)
            # total_loss = (loss_align * w_align) + (naive_loss * w_naive) 
            # total_loss = loss_nce
            
            # logs["Loss/InfoNCE"] = loss_nce.item()
            logs["Loss/InfoNCE"] = loss_nce.item()
            logs["Loss/InfoNCE_Q2K"] = loss_nce_q2k.item()
            logs["Loss/InfoNCE_K2Q"] = loss_nce_k2q.item()
            logs["Loss/Align_INR"] = loss_align.item()
            logs["Loss/Naive_MSE"] = naive_loss.item()
            
            # --- Calc Correspondence Error for Viz ---
            # Head가 찾은 match_idx가 실제 GT(ys_tgt, xs_tgt)와 얼마나 다른가?
            match_ys = match_idx // W
            match_xs = match_idx % W
            
            err_y = (match_ys - ys_tgt).float().abs().mean()
            err_x = (match_xs - xs_tgt).float().abs().mean()
            corr_err = (err_y + err_x) / 2.0
            logs["Metrics/Corr_Error_Px"] = corr_err.item()
            
        else:
            # --- Naive Baseline (Just L1 against Shifted Target) ---
            loss_naive = F.mse_loss(pred_img, target_img)
            total_loss = loss_naive
            logs["Loss/Naive_MSE"] = loss_naive.item()

        # --- D. Update & Logs ---
        total_loss.backward()
        
        # Log Gradients & LR
        logs["Train/GradNorm_INR"] = get_grad_norm(self.inr)
        logs["Train/LR_INR"] = self.optimizer.param_groups[0]['lr']
        if self.use_corr:
            logs["Train/LR_Head"] = self.optimizer.param_groups[1]['lr']
            # logs["Train/GradNorm_Head"] = get_grad_norm(self.head)
            logs["Train/GradNorm_HeadRef"] = get_grad_norm(self.head_ref)
            logs["Train/GradNorm_HeadMov"] = get_grad_norm(self.head_mov)
        
        self.optimizer.step()
        logs["Loss/Total"] = total_loss.item()
        
        # Save for visualization
        if step % 500 == 0:
            self.latest_viz_data = {
                    "pred": pred_img.detach(),
                    "target": target_img.detach(),
                    "clean": clean_img.detach(),
                    "gt_flow": (gt_flow_x, gt_flow_y), 
                    "aug_type": aug_data['type'],
                    "step": step,
                }
        
        if self.cfg.get("wandb", True) and step % 10 == 0:
            wandb.log(logs, step=step)
            
        return logs
        
    def log_fixed_augmentations(self, clean_img):
        """Generates and logs a gallery of the fixed distortions used in this run"""
        if not self.cfg.get("wandb", True):
            return
            
        print(f"[Debug] Logging {len(self.augmentor.fixed_transforms)} fixed augmentations to WandB...")
        gallery_images = []
        B, C, H, W = clean_img.shape
        
        # Identity for base
        theta_identity = torch.tensor([[1, 0, 0], [0, 1, 0]], dtype=torch.float, device=self.device).unsqueeze(0)
        base_grid = F.affine_grid(theta_identity, clean_img.size(), align_corners=False)

        for i, aug_data in enumerate(self.augmentor.fixed_transforms):
            if aug_data['type'] == 'rigid':
                shift_x, shift_y = aug_data['val']
                tx, ty = -shift_x / (W / 2.0), -shift_y / (H / 2.0)
                theta = torch.tensor([[1, 0, tx], [0, 1, ty]], dtype=torch.float, device=self.device).unsqueeze(0)
                grid = F.affine_grid(theta, clean_img.size(), align_corners=False)
            else:
                grid = base_grid - aug_data['val']
            
            warped = F.grid_sample(clean_img, grid, align_corners=False)
            img_np = warped[0, 0].cpu().numpy()
            gallery_images.append(wandb.Image(img_np, caption=f"Aug {i}: {aug_data['type']}"))
            
        wandb.log({"Train/Fixed_Augmentation_Gallery": gallery_images}, step=0)
    
    def compute_dense_error_map(self, pred, target, gt_flow, aug_type):
        """시각화를 위해 전체 픽셀에 대한 에러 맵을 계산 (Batch 처리)"""
        B, C, H, W = pred.shape
        # desc_pred = self.extract_desc(pred)   # [1, 24, H, W]
        # desc_target = self.extract_desc(target) # [1, 24, H, W]
        desc_pred = self.extract_desc(pred, is_moving=False)
        desc_target = self.extract_desc(target, is_moving=True)
        
        flat_pred = desc_pred.view(24, -1).T  # [HW, 24]
        flat_target = desc_target.view(24, -1).T # [HW, 24]
        
        gt_flow_x, gt_flow_y = gt_flow
        
        # Initialize maps for GT flow, Pred flow, and Error
        pred_flow_map_x = torch.zeros(H, W)
        pred_flow_map_y = torch.zeros(H, W)
        gt_flow_map_x = torch.zeros(H, W)
        gt_flow_map_y = torch.zeros(H, W)
        error_map_x = torch.zeros(H, W)
        error_map_y = torch.zeros(H, W)
        
        # 메모리 절약을 위해 Chunk 단위로 계산
        chunk_size = 4096 
        num_pixels = H * W

        for i in range(0, num_pixels, chunk_size):
            end = min(i + chunk_size, num_pixels)
            
            # 1. Query Chunk
            q_chunk = flat_pred[i:end] 
            
            # 2. Global Search
            sim = torch.matmul(q_chunk, flat_target.T)
            match_idx = sim.argmax(dim=1).cpu()
            
            # 3. Predicted Coords
            pred_y_idx = match_idx // W
            pred_x_idx = match_idx % W
            
            # 4. GT Expected Coords
            indices = torch.arange(i, end)
            ys = indices // W
            xs = indices % W
            
            if aug_type == 'rigid':
                actual_dy = torch.full_like(ys, float(gt_flow_y)).float()
                actual_dx = torch.full_like(xs, float(gt_flow_x)).float()
            else:
                actual_dy = gt_flow_y[ys, xs].cpu()
                actual_dx = gt_flow_x[ys, xs].cpu()
            
            # 5. Calculate Flows & Error
            # Flow = Target_Coord - Canonical_Coord
            # Pred Flow (Always integer because it's from argmax)
            current_pred_flow_y = (pred_y_idx - ys).float()
            current_pred_flow_x = (pred_x_idx - xs).float()
            
            # Use actual float Ground Truth for error calculation
            diff_y = current_pred_flow_y - actual_dy
            diff_x = current_pred_flow_x - actual_dx
            
            # Store in full maps
            gt_flow_map_y.view(-1)[i:end] = actual_dy
            gt_flow_map_x.view(-1)[i:end] = actual_dx
            
            pred_flow_map_y.view(-1)[i:end] = current_pred_flow_y
            pred_flow_map_x.view(-1)[i:end] = current_pred_flow_x

            error_map_y.view(-1)[i:end] = diff_y
            error_map_x.view(-1)[i:end] = diff_x
            
        return pred_flow_map_x, pred_flow_map_y, gt_flow_map_x, gt_flow_map_y, error_map_x, error_map_y
        
    def plot_quiver(self, ax, flow_x, flow_y, title, step=8):
        """
        Plots a downsampled vector field.
        step: stride for downsampling (e.g., show every 16th arrow)
        """
        H, W = flow_x.shape
        # Create grid for arrows
        y, x = np.mgrid[0:H:step, 0:W:step]
        u = flow_x[::step, ::step]
        v = flow_y[::step, ::step]
        
        # Flip v because image coordinates (y) increase downwards, 
        # but matplotlib quiver assumes y increases upwards by default unless on image.
        # However, we will invert axis to match image convention.
        
        ax.quiver(x, y, u, v, angles='xy', scale_units='xy', scale=0.5, color='r')
        ax.set_aspect('equal')
        ax.invert_yaxis() # Match image coordinate system (0,0 top-left)
        ax.set_title(title) 
    
    def get_jacobian_det(self, flow_x, flow_y):
        """
        Calculates the Jacobian Determinant of the flow field.
        det(J) = (1 + du/dx)(1 + dv/dy) - (du/dy)(dv/dx)
        """
        ux, uy = np.gradient(flow_x.numpy())
        vx, vy = np.gradient(flow_y.numpy())
        
        # Grid index increases: y is row (0th dim), x is col (1st dim)
        # np.gradient(arr) returns [grad_row, grad_col]
        du_dy, du_dx = ux, uy
        dv_dy, dv_dx = vx, vy
        
        det_j = (1 + du_dx) * (1 + dv_dy) - (du_dy * dv_dx)
        return det_j

    # Warped Grid Helper
    def get_warped_grid(self, H, W, flow_x, flow_y, step=16):
        """Creates a checkerboard and warps it with flow"""
        grid = np.zeros((H, W))
        # Create checkerboard
        for i in range(0, H, step*2):
            for j in range(0, W, step*2):
                grid[i:i+step, j:j+step] = 1.0
                grid[i+step:i+step*2, j+step:j+step*2] = 1.0
        
        # Warp with grid_sample logic
        # Canonical p corresponds to Target p + flow(p)
        grid_torch = torch.from_numpy(grid).float().to(device).unsqueeze(0).unsqueeze(0)
        
        # We want to show how Canonical space is warped to Target space
        # Target_Coord = Canonical_Coord + Flow
        # Create an identity grid and add the predicted flow to it.
        # Coordinates in [-1, 1]
        ys = torch.linspace(-1, 1, H, device=device)
        xs = torch.linspace(-1, 1, W, device=device)
        yy, xx = torch.meshgrid(ys, xs, indexing='ij')
        base = torch.stack([xx, yy], dim=-1).unsqueeze(0) # [1, H, W, 2]
        
        # Convert pixel flow to normalized flow
        norm_fx = flow_x.to(device).unsqueeze(0) / (W / 2.0)
        norm_fy = flow_y.to(device).unsqueeze(0) / (H / 2.0)
        norm_flow = torch.stack([norm_fx, norm_fy], dim=-1)
        
        # Warping: where did pixels move?
        # Sample from canonical grid at (p + flow)
        warped = F.grid_sample(grid_torch, base + norm_flow, align_corners=False)
        return warped[0, 0].cpu().numpy() 
    
    def visualize(self):
        if not self.latest_viz_data: return
        data = self.latest_viz_data
        
        pred = data["pred"]
        targ = data["target"]
        clean = data["clean"]
        step = data["step"]
        gt_flow = data["gt_flow"]
        aug_type = data["aug_type"]
        
        # 1. Image Residual
        res = (pred - clean).abs()
        
        # 2. Compute Correspondence Error Map (Only if use_corr is True)
        if self.use_corr:
            pred_fx, pred_fy, gt_fx, gt_fy, err_x, err_y = self.compute_dense_error_map(pred, targ, gt_flow, aug_type)
            pred_fx, pred_fy = pred_fx.cpu(), pred_fy.cpu()
            gt_fx, gt_fy = gt_fx.cpu(), gt_fy.cpu()
            err_x, err_y = err_x.cpu(), err_y.cpu()

            warped_grid = self.get_warped_grid(pred.shape[-2], pred.shape[-1], pred_fx, pred_fy)
            gt_warped_grid = self.get_warped_grid(pred.shape[-2], pred.shape[-1], gt_fx, gt_fy)

            pred_jac = self.get_jacobian_det(pred_fx, pred_fy)
            gt_jac = self.get_jacobian_det(gt_fx, gt_fy)

            num_pix = pred_fx.shape[0] * pred_fx.shape[1]
            pred_folds = (pred_jac <= 0).sum()
            gt_folds = (gt_jac <= 0).sum()
            pred_fold_perc = (pred_folds / num_pix) * 100
            gt_fold_perc = (gt_folds / num_pix) * 100
            
            mean_err_x = err_x.abs().mean().item()
            mean_err_y = err_y.abs().mean().item()
        else:
            # Correspondence를 안 쓰면 에러 맵을 계산할 수 없으므로 0으로 채움
            H, W = pred.shape[-2:]
            pred_fx = pred_fy = err_x = err_y = torch.zeros(H, W)
            pred_jac = gt_jac = np.ones((H, W))
            warped_grid = gt_warped_grid = np.zeros((H, W))
            mean_err_x = mean_err_y = 0.0
            pred_fold_perc = gt_fold_perc = 0.0 

            if aug_type == 'rigid':
                # gt_flow contains scalars (shift_x, shift_y)
                gt_fx = torch.full((H, W), gt_flow[0])
                gt_fy = torch.full((H, W), gt_flow[1])
            else:
                # gt_flow contains tensors (H, W)
                gt_fx = gt_flow[0].cpu()
                gt_fy = gt_flow[1].cpu()

        # Calculate Max Possible Shift for Caption
        if self.cfg.get("aug_mode") == "elastic":
             max_px = self.cfg['elastic_alpha'] * (self.cfg['res'] / 2.0)
             limit_str = f"Max Warp: {max_px:.1f}px"
        else:
             max_px = self.cfg['res'] * self.cfg['shift_ratio']
             limit_str = f"Max Shift: {max_px:.1f}px"
            
        fig, axes = plt.subplots(4, 4, figsize=(20, 20))
        fig.suptitle(f"Step {step} | {limit_str} | Aug: {aug_type}", fontsize=16)
        
        # --- Row 1: Images & PCA ---
        axes[0,0].imshow(clean[0].permute(1,2,0).squeeze().cpu(), cmap='gray', vmin=0, vmax=1)
        axes[0,0].set_title("GT Canonical")
        axes[0,1].imshow(targ[0].permute(1,2,0).squeeze().cpu(), cmap='gray', vmin=0, vmax=1)
        axes[0,1].set_title(f"Target (Distorted)")
        axes[0,2].imshow(pred[0].permute(1,2,0).squeeze().cpu(), cmap='gray', vmin=0, vmax=1)
        axes[0,2].set_title(f"Learned Canonical")
        im_res = axes[0,3].imshow(res[0].permute(1,2,0).squeeze().cpu(), cmap='inferno', vmin=0, vmax=0.1)
        axes[0,3].set_title("Recon Residual (L1)")
        plt.colorbar(im_res, ax=axes[0,3])

        # --- Row 2: Jacobian & Warped Grid ---
        # GT Jacobian (Grayscale)
        im_gt_jac = axes[1,0].imshow(gt_jac, cmap='gray', vmin=0, vmax=2)
        axes[1,0].set_title(f"GT Jac (Folds: {gt_fold_perc:.2f}%)")
        plt.colorbar(im_gt_jac, ax=axes[1,0], fraction=0.046, pad=0.04)
        # Pred Jacobian (Grayscale + Red Mask Overlay)
        im_pred_jac = axes[1,1].imshow(pred_jac, cmap='gray', vmin=0, vmax=2)
        fold_mask = (pred_jac <= 0).astype(float)
        masked_red = np.ma.masked_where(fold_mask == 0, fold_mask)
        axes[1,1].imshow(masked_red, cmap=mcolors.ListedColormap(['red']), alpha=1.0)
        axes[1,1].set_title(f"Pred Jac (Red = Folding) (Folds: {pred_fold_perc:.2f}%)")
        plt.colorbar(im_pred_jac, ax=axes[1,1], fraction=0.046, pad=0.04)
        axes[1,2].imshow(gt_warped_grid, cmap='gray')
        axes[1,2].set_title("GT Distortion Grid")
        axes[1,3].imshow(warped_grid, cmap='gray')
        axes[1,3].set_title("Pred Distortion Grid")
        
        v_min_max = 10
        # --- Row 3: X-Axis Flows & GT Quiver ---
        im_gx = axes[2,0].imshow(gt_fx, cmap='RdBu', vmin=-v_min_max, vmax=v_min_max)
        axes[2,0].set_title("GT Flow X (px)")
        plt.colorbar(im_gx, ax=axes[2,0])
        im_px = axes[2,1].imshow(pred_fx, cmap='RdBu', vmin=-v_min_max, vmax=v_min_max)
        axes[2,1].set_title("Pred Flow X (px)")
        plt.colorbar(im_px, ax=axes[2,1])
        im_ex = axes[2,2].imshow(err_x, cmap='RdBu', vmin=-v_min_max, vmax=v_min_max)
        axes[2,2].set_title(f"Error X (Pred - GT) (Avg: {mean_err_x:.2f}px)")
        plt.colorbar(im_ex, ax=axes[2,2])
        self.plot_quiver(axes[2,3], gt_fx, gt_fy, "GT Vector Field")

        # --- Row 4: Y-Axis Flows & Pred Quiver ---
        im_gy = axes[3,0].imshow(gt_fy, cmap='RdBu', vmin=-v_min_max, vmax=v_min_max)
        axes[3,0].set_title("GT Flow Y (px)")
        plt.colorbar(im_gy, ax=axes[3,0])
        im_py = axes[3,1].imshow(pred_fy, cmap='RdBu', vmin=-v_min_max, vmax=v_min_max)
        axes[3,1].set_title("Pred Flow Y (px)")
        plt.colorbar(im_py, ax=axes[3,1])
        im_ey = axes[3,2].imshow(err_y, cmap='RdBu', vmin=-v_min_max, vmax=v_min_max)
        axes[3,2].set_title(f"Error Y (Pred - GT) (Avg: {mean_err_y:.2f}px)")
        plt.colorbar(im_ey, ax=axes[3,2])
        self.plot_quiver(axes[3,3], pred_fx, pred_fy, "Pred Vector Field")
        
        plt.tight_layout()
        
        if self.cfg.get("wandb", True):
             wandb.log({
                "Visual/Result": wandb.Image(fig),
             }, step=step)
        
        plt.show()
        plt.close(fig)

def run_experiment(exp_config):
    set_seed(42)
    
    # 1. Setup WandB with unique name
    mask_str = "mask" if exp_config.get("use_border_mask", False) else "nomask"
    # NEW: Updated run name to include aug_mode
    aug_str = exp_config.get('aug_mode', 'rigid')
    run_name = f"res{exp_config['res']}_{exp_config['model_type']}_{'corr' if exp_config['use_correspondence'] else 'naive'}_{aug_str}_{mask_str}"
    
    if exp_config["wandb"]:
        wandb.init(project="Single-Image-Optimization", config=exp_config, name=run_name, reinit=True)

    print(f"--- Starting Experiment: {run_name} ---")
    
    # 2. Load Image dynamically based on resolution
    base_path = "/home/minsukc/SIO/data" 
    clean_image = load_image_dynamic(base_path, exp_config['res'])
    
    if clean_image is None:
        print("Skipping this experiment due to image load error.")
        wandb.finish()
        return

    # 3. Calculate max shift in pixels
    exp_config["max_shift_px"] = exp_config["res"] * exp_config["shift_ratio"]
    if exp_config.get("aug_mode") == "rigid":
        print(f"Max Shift: {exp_config['shift_ratio']*100}% = {exp_config['max_shift_px']} pixels")
    if exp_config.get("aug_mode") == "elastic":
        # Alpha is in normalized coords [-1, 1], where 1.0 = half image size
        max_elastic_px = exp_config['elastic_alpha'] * (exp_config['res'] / 2.0)
        print(f"Elastic Alpha: {exp_config['elastic_alpha']} (Approx. max distortion: {max_elastic_px:.1f} pixels)")

    # 4. Initialize Trainer
    trainer = JointTrainer(device, exp_config)

    if not exp_config["infinite_augmentation"]:
        trainer.log_fixed_augmentations(clean_image)

    # 5. Training Loop
    pbar = tqdm(range(exp_config["epochs"]), desc=run_name, leave=False)
    for step in pbar:
        logs = trainer.train_step(clean_image, step)
        
        if step % 100 == 0:
            desc = f"L: {logs['Loss/Total']:.4f}"
            if exp_config["use_correspondence"]:
                desc += f" | Err: {logs['Metrics/Corr_Error_Px']:.2f}px"
            pbar.set_postfix_str(desc)
            
        if step % 500 == 0:
            trainer.visualize()

    # 6. Cleanup
    if exp_config["wandb"]: wandb.finish()
    
    # Force delete to free GPU memory for next run
    del trainer
    del clean_image
    torch.cuda.empty_cache()
    gc.collect()
    print(f"--- Finished {run_name} ---\n")

# Common settings
common_config = {
    "wandb": True,
    # "wandb": False,
    # "epochs": 4001,
    # "epochs": 20001,
    "epochs": 50001,
    # "epochs": 10001,
    # "epochs": 501,
    # "lr_inr": 1e-3,
    "lr_inr": 5e-4,
    # "lr_head": 1e-4,
    "lr_head": 5e-4,
    # "infinite_augmentation": True,
    "infinite_augmentation": False,
    "num_aug": 10,
    "w_align": 1.0,
    "w_naive": 0.0,
    "elastic_alpha": 0.05,
}

# Variable settings
# resolutions = [256, 512, 1024]
# resolutions = [256, 512] # 1024 CUDA OOM
resolutions = [256]
model_types = ["fourier"] 
shift_ratios = [0.40] # 5% shift, 10% shift
# shift_ratios = [0.10, 0.40] # 5% shift, 10% shift
elastic_alpha = [0.1]
# elastic_alpha = [0.05, 0.1]
# modes = ["naive", "corr"] 
# modes = ["corr", "naive"] 
modes = ["corr"] 
# modes = ["naive"] 
# mask_options = [True, False] 
mask_options = [False] 
# aug_modes = ["elastic", "rigid"] 
aug_modes = ["elastic"] 


# Generate List of Experiments
experiment_queue = []

for res in resolutions:
    for shift_r in shift_ratios:
        for mode in modes:
            for model in model_types:
                for use_mask in mask_options:
                    for aug in aug_modes:
                        for alpha in elastic_alpha:
                            # Create specific config
                            cfg = common_config.copy()
                            cfg["res"] = res
                            cfg["model_type"] = model
                            cfg["shift_ratio"] = shift_r
                            cfg["use_correspondence"] = (mode == "corr")
                            cfg["use_border_mask"] = use_mask
                            cfg["aug_mode"] = aug 
                            cfg["elastic_alpha"] = alpha
        
                            experiment_queue.append(cfg)

print(f"Queued {len(experiment_queue)} experiments.")

# Run them all
for i, exp_cfg in enumerate(experiment_queue):
    print(f"Running {i+1}/{len(experiment_queue)}")
    run_experiment(exp_cfg)