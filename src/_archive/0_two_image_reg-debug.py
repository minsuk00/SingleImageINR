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
import copy
from skimage.transform import radon, iradon, resize
import SimpleITK as sitk

# from feature_communicator import FeatureCommunicator

def set_seed(seed=42):
    torch.manual_seed(seed)
    np.random.seed(seed)
    torch.backends.cudnn.deterministic = True

set_seed(42)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Running on device: {device}")

class FourierINR(nn.Module):
    def __init__(self, hidden_dim=256, num_layers=4, mapping_size=256, scale=10.0):
        super().__init__()
        self.mapping_size = mapping_size
        self.scale = scale
        
        self.register_buffer('B', torch.randn((2, mapping_size)) * scale)
        
        layers = []
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
        coords = torch.stack([xx, yy], dim=-1).reshape(-1, 2)
        
        x_proj = (2.0 * np.pi * coords) @ self.B 
        emb = torch.cat([torch.sin(x_proj), torch.cos(x_proj)], dim=-1)
        
        out = self.net(emb)
        return out.reshape(1, 1, H, W)

# ==========================================
# 2. Helper Functions
# ==========================================
def load_image_dynamic(base_path, res):
    """Loads image based on resolution string"""
    # Prioritize loading the pre-processed tensor directly
    # pt_path = f"{base_path}/L067.pt"
    # if os.path.exists(pt_path):
    #     print(f"[DEBUG] Loading tensor from: {pt_path}")
    #     return torch.load(pt_path, map_location=device)

    # Fallback to PNG logic if .pt not found (or if user wants png)
    # path = f"{base_path}/L067_preview.png"
    path = "/home/minsukc/SIO/data/ct_chest_2_resized_256x256.png"
    
    print(f"[DEBUG] .pt file not found. Falling back to PNG: {path}")
    
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

# CT Simulation Helpers
def add_poisson_noise(sinogram, I0=10000):
    if I0 is None: return sinogram
    intensity = I0 * np.exp(-sinogram)
    noisy_intensity = np.random.poisson(intensity)
    noisy_intensity[noisy_intensity == 0] = 1
    noisy_sinogram = -np.log(noisy_intensity / I0)
    return noisy_sinogram

def simulate_sparse_ct(image_tensor, num_views, noise_intensity=None):
    device = image_tensor.device
    img_np = image_tensor.squeeze().cpu().numpy()
    if len(img_np.shape) > 2: img_np = img_np[0] # Batch handling
    H, W = img_np.shape

    theta = np.linspace(0.0, 180.0, num_views, endpoint=False)
    # sinogram = radon(img_np, theta=theta, circle=True)
    sinogram = radon(img_np, theta=theta, circle=False)

    PHYSICAL_MAX_ATTENUATION = 4.0
    current_max = sinogram.max()
    scale_factor = PHYSICAL_MAX_ATTENUATION / max(current_max, 1e-6)

    if noise_intensity is not None:
        sinogram_physical = sinogram * scale_factor
        sinogram_noisy = add_poisson_noise(sinogram_physical, I0=noise_intensity)
        sinogram = sinogram_noisy / scale_factor

    # reconstruction = iradon(sinogram, theta=theta, filter_name="ramp", circle=True, output_size=max(H, W))
    reconstruction = iradon(sinogram, theta=theta, filter_name="ramp", circle=False, output_size=max(H, W))
    reconstruction = np.clip(reconstruction, 0, 1)

    recon_tensor = torch.from_numpy(reconstruction).float().unsqueeze(0).unsqueeze(0).to(device)
    return recon_tensor

# ==========================================
# 3. Models
# ==========================================

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
        # (B, N, C) -> (B, C, N) -> (B, C, H_grid, W_grid) is cleaner for spatial reshaping
        x = x.transpose(1, 2).reshape(B, -1, H_grid, W_grid)
        
        # Pixel Shuffle [B, 24, H, W]
        x = F.pixel_shuffle(x, self.patch_size)
        
        # Cosine Similarity를 위해 Normalize 수행
        return F.normalize(x, dim=1) 

# Augmentation Manager 
class AugmentationManager:
    def __init__(self, infinite=False, num_fixed=10, max_shift=15.0, H=256, W=256, device="cuda", aug_mode="rigid", elastic_alpha=0.1, elastic_grid_res = 6):
        self.infinite = infinite
        self.max_shift = max_shift
        self.H = H
        self.W = W
        self.device = device
        self.aug_mode = aug_mode
        self.elastic_alpha = elastic_alpha  # Scale of deformation
        self.elastic_grid_res = elastic_grid_res
        
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
            # grid_h, grid_w = 6, 6
            grid_h, grid_w = self.elastic_grid_res, self.elastic_grid_res
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
        self.use_flow_loss = config.get("use_flow_loss", False)
        
        # 1. Models
        print("Loading Models...")
        
        if self.use_corr:
            dim_enc = -1
            if config["encoder"] == "dinov3-vits":
                self.dino = AutoModel.from_pretrained('facebook/dinov3-vits16-pretrain-lvd1689m').to(device).eval()
                dim_enc = 384
            elif config["encoder"] == "dinov3-vitb":
                self.dino = AutoModel.from_pretrained('facebook/dinov3-vitb16-pretrain-lvd1689m').to(device).eval()
                dim_enc = 768
            else:
                raise Exception("encoder not found...")
            
            for p in self.dino.parameters(): p.requires_grad = False

            dim_dec = 768
            dim_total = dim_enc + dim_dec 
            
            # Initialize Communication Module
            # self.communicator = FeatureCommunicator(
            #     input_dim=dim_enc, 
            #     embed_dim=dim_dec, 
            #     grid_size=(config['res']//16, config['res']//16),
            #     depth=2, 
            #     num_heads=8
            # ).to(device)
            
            # Heads now take (Enc + Dec) dimension
            self.inr = FourierINR(hidden_dim=256).to(device)
            self.head_ref = MatchingHead(input_dim=dim_enc).to(device)
            self.head_mov = MatchingHead(input_dim=dim_enc).to(device)
            
            # Optimizer for Head + INR
            self.optimizer = torch.optim.Adam([
                # {'params': self.communicator.parameters(), 'lr': config["lr_comm"]}, 
                {'params': self.inr.parameters(), 'lr': 5e-4},
                {'params': self.head_ref.parameters(), 'lr': config["lr_head"]},
                {'params': self.head_mov.parameters(), 'lr': config["lr_head"]}
            ])

            # warmup_steps = config["lr_warmup_steps"]
            # def lr_lambda(current_step):
            #     if current_step < warmup_steps:
            #         return float(current_step) / float(max(1, warmup_steps))
            #     return 1.0
            # self.scheduler = torch.optim.lr_scheduler.LambdaLR(self.optimizer, lr_lambda)

            H, W = config["res"], config["res"]
            grid_y, grid_x = torch.meshgrid(
                torch.arange(H, device=self.device), 
                torch.arange(W, device=self.device), 
                indexing='ij'
            )
            self.cached_flat_y = grid_y.flatten().float()
            self.cached_flat_x = grid_x.flatten().float()

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
            elastic_grid_res = config["elastic_grid_res"],
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
        
        if is_moving:
            return self.head_mov(patch_feats, H, W)
        else:
            return self.head_ref(patch_feats, H, W)

    # Helper to get raw DINO tokens (Before Communication)
    def get_backbone_feats(self, img):
        B, C, H, W = img.shape
        img_3ch = img.repeat(1, 3, 1, 1)
        img_norm = (img_3ch - self.mean) / self.std
        
        H_grid = H // 16
        W_grid = W // 16
        num_patches = H_grid * W_grid
        
        with torch.no_grad():
            out = self.dino(pixel_values=img_norm, output_hidden_states=True)
            patch_feats = out.last_hidden_state[:, -num_patches:, :] # [B, N, 384]

        return patch_feats

    def get_tv_loss(self, pred_x, pred_y, patch_size):
        # 1. Reshape flat predictions to the patch grid
        # pred_x: [N_samples] -> [P, P]
        flow_x = pred_x.view(patch_size, patch_size)
        flow_y = pred_y.view(patch_size, patch_size)

        # 2. Calculate gradients (neighbors)
        # Difference between right neighbor and current pixel
        dx_x = flow_x[:, 1:] - flow_x[:, :-1]
        dy_x = flow_x[1:, :] - flow_x[:-1, :]
        dx_y = flow_y[:, 1:] - flow_y[:, :-1]
        dy_y = flow_y[1:, :] - flow_y[:-1, :]

        # 3. Mean Squared Error of gradients
        loss = (dx_x**2).mean() + (dy_x**2).mean() + (dx_y**2).mean() + (dy_y**2).mean()
        return loss
    
    # sample features at arbitrary (float) pixel coordinates
    def sample_at_coords(self, feature_map, x_px, y_px, H, W):
        # align_corners=False matches pixels to [-1 + 1/W, 1 - 1/W] range
        # Formula: ((x + 0.5) / W) * 2 - 1
        norm_x = ((x_px + 0.5) / W) * 2 - 1
        norm_y = ((y_px + 0.5) / H) * 2 - 1
        
        grid = torch.stack([norm_x, norm_y], dim=-1).unsqueeze(0).unsqueeze(0)
        return F.grid_sample(feature_map, grid, align_corners=False, mode='bilinear').squeeze(0).squeeze(1).T
        
    def compute_soft_argmax_flow(self, feats_q, feats_dense, H, W, temp=0.05):
        """
        feats_q: [N_samples, C]
        feats_dense: [B, C, H, W] -> We assume B=1 for simplicity here
        """
        # 1. Flatten Dense Features: [C, H*W]
        flat_dense = feats_dense.view(feats_dense.shape[1], -1) 
        
        # 2. Similarity Map: [N_samples, H*W]
        # Compare every query pixel to EVERY pixel in the image
        sim_map = torch.matmul(feats_q, flat_dense) / temp
        
        # 3. Spatial Softmax (Get probability distribution)
        # shape: [N_samples, H*W]
        prob_map = F.softmax(sim_map, dim=1)
        
        # 4. Expectation (Use Cached Grid)
        pred_y = torch.sum(prob_map * self.cached_flat_y, dim=1) # [N_samples]
        pred_x = torch.sum(prob_map * self.cached_flat_x, dim=1) # [N_samples]
        
        return pred_x, pred_y
        
    def train_step(self, clean_img, fixed_img_sparse_ignored, step, info_nce_temp=0.05):
        self.optimizer.zero_grad()
        B, C, H, W = clean_img.shape
        
        # --- A. Augmentation (Create the "Target" / Moving Image) ---
        aug_data = self.augmentor.get_transform()
        
        # Base Identity Grid
        theta_identity = torch.tensor([[1, 0, 0], [0, 1, 0]], dtype=torch.float, device=self.device).unsqueeze(0)
        base_grid = F.affine_grid(theta_identity, clean_img.size(), align_corners=False)
        
        # Store GT Flow
        gt_flow_y, gt_flow_x = None, None
        if aug_data['type'] == 'rigid':
            shift_x, shift_y = aug_data['val']
            tx, ty = -shift_x / (W / 2.0), -shift_y / (H / 2.0)
            theta = torch.tensor([[1, 0, tx], [0, 1, ty]], dtype=torch.float, device=self.device).unsqueeze(0)
            grid = F.affine_grid(theta, clean_img.size(), align_corners=False)
            gt_flow_y, gt_flow_x = shift_y, shift_x
        elif aug_data['type'] == 'elastic':
            flow = aug_data['val']
            grid = base_grid - flow
            gt_flow_y = flow[0, ..., 1] * (H / 2.0)
            gt_flow_x = flow[0, ..., 0] * (W / 2.0)
            
        # This is the "Observed" distorted image
        target_img = F.grid_sample(clean_img, grid, align_corners=False) 
        
        # --- B. Rendering (The INR generates the "Fixed" image) ---
        pred_img = self.inr(H, W)
        
        # --- C. Feature Extraction ---
        # Pred = Fixed (Ref), Target = Moving (Mov)
        desc_pred = self.extract_desc(pred_img, is_moving=False)   # [B, 24, H, W]
        desc_target = self.extract_desc(target_img, is_moving=True) # [B, 24, H, W]
        
        # --- D. InfoNCE Loss (Train the Heads) ---
        num_samples = 4096
        flat_inds = torch.randperm(H * W, device=self.device)[:num_samples]
        ys = flat_inds // W
        xs = flat_inds % W
        
        # Query Features at Random Locations
        feats_q = desc_pred[0, :, ys, xs].T # [N, C]
        
        # Calculate POSITIVE Key coordinates based on GT flow
        if aug_data['type'] == 'rigid':
            ys_tgt = torch.clamp(ys + int(round(gt_flow_y)), 0, H-1)
            xs_tgt = torch.clamp(xs + int(round(gt_flow_x)), 0, W-1)
        elif aug_data['type'] == 'elastic':
            dy = gt_flow_y[ys, xs]
            dx = gt_flow_x[ys, xs]
            ys_tgt = torch.clamp(ys + torch.round(dy).long(), 0, H-1)
            xs_tgt = torch.clamp(xs + torch.round(dx).long(), 0, W-1)
            
        feats_k_pos = desc_target[0, :, ys_tgt, xs_tgt].T

        # Contrastive Loss
        logits = torch.matmul(feats_q, feats_k_pos.T) / info_nce_temp
        labels = torch.arange(num_samples, device=self.device)
        loss_nce = (self.ce_loss(logits, labels) + self.ce_loss(logits.T, labels)) / 2.0
        
        # --- E. Alignment Loss (Train the INR) ---
        # 1. Flatten the target features for global search
        flat_target = desc_target.view(24, -1) # [24, HW]
        
        # 2. Find "Learned" Matches (Where does the Head THINK the match is?)
        # We detach feats_q so we don't backprop InfoNCE gradients through this Argmax path (optional but cleaner)
        sim = torch.matmul(feats_q.detach(), flat_target) # [N, HW]
        match_idx = sim.argmax(dim=1) # [N] - Indices in Target Image
        
        # 3. Gather Pixels
        # pred_pixels: Pixels at the query locations (ys, xs) in the INR image
        pred_pixels = pred_img.view(1, -1)[:, flat_inds].T
        
        # target_pixels: Pixels at the MATCHED locations in the Target image
        target_pixels = target_img.view(1, -1)[:, match_idx].T
        
        # 4. MSE Loss (Force INR to look like the matched target pixels)
        loss_align = F.mse_loss(pred_pixels, target_pixels)
        
        # Total Loss
        w_align = self.cfg.get("w_align", 1.0)
        total_loss = loss_nce + (loss_align * w_align)
        
        # --- F. Update ---
        total_loss.backward()
        self.optimizer.step()
        
        # Logs
        logs = {
            "Loss/Total": total_loss.item(), 
            "Loss/InfoNCE": loss_nce.item(),
            "Loss/Align_MSE": loss_align.item()
        }

        # --- G. Metrics (For Debugging/Logging) ---
        with torch.no_grad():
            # Calculate where the head *should* have pointed (Target GT) vs where it *did* point (match_idx)
            match_ys = match_idx // W
            match_xs = match_idx % W
            
            # Check L1 distance between Predicted Match and GT Target Location
            # ys_tgt, xs_tgt are the TRUE locations of the query pixels in the target image
            err_y = (match_ys - ys_tgt).float().abs().mean()
            err_x = (match_xs - xs_tgt).float().abs().mean()
            corr_err = (err_y + err_x) / 2.0
            
            logs["Metrics/Corr_Error_Px"] = corr_err.item()

        # Update visualizer data (already there, just ensuring flow)
        if step % 500 == 0:
             self.latest_viz_data = {
                    "fixed": pred_img.detach(),   
                    "moving": target_img.detach(),
                    "clean": clean_img.detach(),
                    "gt_flow": (gt_flow_x, gt_flow_y), 
                    "aug_type": aug_data['type'],
                    "step": step,
                }
        
        return logs
        
    def log_fixed_augmentations(self, clean_img):
        pass 

    def compute_dense_error_map(self, fixed, moving, gt_flow, aug_type):
        """
        fixed: The INR Output (Prediction)
        moving: The Augmented Input (Target)
        """
        B, C, H, W = fixed.shape
        
        # Use the updated extract_desc (Direct DINO + Head)
        desc_fixed = self.extract_desc(fixed, is_moving=False)   # [B, 24, H, W]
        desc_moving = self.extract_desc(moving, is_moving=True)  # [B, 24, H, W]
        
        flat_fixed = desc_fixed.view(24, -1).T  # [HW, 24]
        flat_moving = desc_moving.view(24, -1).T # [HW, 24]
        
        gt_flow_x, gt_flow_y = gt_flow
        
        # Initialize maps
        pred_flow_map_x = torch.zeros(H, W)
        pred_flow_map_y = torch.zeros(H, W)
        gt_flow_map_x = torch.zeros(H, W)
        gt_flow_map_y = torch.zeros(H, W)
        error_map_x = torch.zeros(H, W)
        error_map_y = torch.zeros(H, W)
        
        chunk_size = 4096 
        num_pixels = H * W

        for i in range(0, num_pixels, chunk_size):
            end = min(i + chunk_size, num_pixels)
            q_chunk = flat_fixed[i:end]
            
            # Global Search: Compare every INR pixel to every Target pixel
            sim = torch.matmul(q_chunk, flat_moving.T)
            match_idx = sim.argmax(dim=1).cpu()
            
            pred_y_idx = match_idx // W
            pred_x_idx = match_idx % W
            
            indices = torch.arange(i, end)
            ys = indices // W
            xs = indices % W
            
            if aug_type == 'rigid':
                actual_dy = torch.full_like(ys, float(gt_flow_y)).float()
                actual_dx = torch.full_like(xs, float(gt_flow_x)).float()
            else:
                actual_dy = gt_flow_y[ys, xs].cpu()
                actual_dx = gt_flow_x[ys, xs].cpu()
            
            # Flow = Target_Coord - Canonical_Coord
            current_pred_flow_y = (pred_y_idx - ys).float()
            current_pred_flow_x = (pred_x_idx - xs).float()
            
            diff_y = current_pred_flow_y - actual_dy
            diff_x = current_pred_flow_x - actual_dx
            
            gt_flow_map_y.view(-1)[i:end] = actual_dy
            gt_flow_map_x.view(-1)[i:end] = actual_dx
            pred_flow_map_y.view(-1)[i:end] = current_pred_flow_y
            pred_flow_map_x.view(-1)[i:end] = current_pred_flow_x
            error_map_y.view(-1)[i:end] = diff_y
            error_map_x.view(-1)[i:end] = diff_x
            
        return pred_flow_map_x, pred_flow_map_y, gt_flow_map_x, gt_flow_map_y, error_map_x, error_map_y
    def get_jacobian_det(self, flow_x, flow_y):
        """Calculates the Jacobian Determinant of the flow field."""
        ux, uy = np.gradient(flow_x.numpy())
        vx, vy = np.gradient(flow_y.numpy())
        # NOTE: rewrite using torch.gradient if i want to add jacobian loss
        
        du_dy, du_dx = ux, uy
        dv_dy, dv_dx = vx, vy
        
        det_j = (1 + du_dx) * (1 + dv_dy) - (du_dy * dv_dx)
        return det_j

    def get_warped_grid(self, H, W, flow_x, flow_y, step=16):
        """Creates a checkerboard and warps it with flow"""
        grid = np.zeros((H, W))
        for i in range(0, H, step*2):
            for j in range(0, W, step*2):
                grid[i:i+step, j:j+step] = 1.0
                grid[i+step:i+step*2, j+step:j+step*2] = 1.0
        
        grid_torch = torch.from_numpy(grid).float().to(self.device).unsqueeze(0).unsqueeze(0)
        
        ys = torch.linspace(-1, 1, H, device=self.device)
        xs = torch.linspace(-1, 1, W, device=self.device)
        yy, xx = torch.meshgrid(ys, xs, indexing='ij')
        base = torch.stack([xx, yy], dim=-1).unsqueeze(0) 
        
        norm_fx = flow_x.to(self.device).unsqueeze(0) / (W / 2.0)
        norm_fy = flow_y.to(self.device).unsqueeze(0) / (H / 2.0)
        norm_flow = torch.stack([norm_fx, norm_fy], dim=-1)
        
        warped = F.grid_sample(grid_torch, base + norm_flow, align_corners=False)
        return warped[0, 0].cpu().numpy()
        
    def plot_quiver(self, ax, flow_x, flow_y, title, step=8):
        H, W = flow_x.shape
        y, x = np.mgrid[0:H:step, 0:W:step]
        u = flow_x[::step, ::step]
        v = flow_y[::step, ::step]
        ax.quiver(x, y, u, v, angles='xy', scale_units='xy', scale=0.5, color='r')
        ax.set_aspect('equal')
        ax.invert_yaxis() 
        ax.set_title(title) 
    
    def visualize(self):
        if not self.latest_viz_data: return
        data = self.latest_viz_data
        
        # Mapping names from Code A's train_step to Code B's logic
        # fixed = INR Output (Prediction)
        # moving = Augmented Input (Target)
        pred = data["fixed"] 
        targ = data["moving"]
        clean = data["clean"]
        step = data["step"]
        gt_flow = data["gt_flow"]
        aug_type = data["aug_type"]
        
        # 1. Image Residual
        res = (pred - clean).abs()
        
        # 2. Compute Correspondence Error Map
        if self.use_corr:
            pred_fx, pred_fy, gt_fx, gt_fy, err_x, err_y = self.compute_dense_error_map(pred, targ, gt_flow, aug_type)
            pred_fx, pred_fy = pred_fx.cpu(), pred_fy.cpu()
            gt_fx, gt_fy = gt_fx.cpu(), gt_fy.cpu()
            err_x, err_y = err_x.cpu(), err_y.cpu()

            # Warped Grids
            H, W = pred.shape[-2], pred.shape[-1]
            warped_grid = self.get_warped_grid(H, W, pred_fx, pred_fy)
            gt_warped_grid = self.get_warped_grid(H, W, gt_fx, gt_fy)

            # Jacobians & Folding
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
            return

        v_min_max = 10
        
        # Plotting - 4 Rows, 4 Columns (Matching Code B)
        fig, axes = plt.subplots(4, 4, figsize=(20, 20))
        fig.suptitle(f"Step {step} | Aug: {aug_type}", fontsize=16)
        
        # --- Row 1: Images ---
        axes[0,0].imshow(clean[0].permute(1,2,0).squeeze().cpu(), cmap='gray', vmin=0, vmax=1)
        axes[0,0].set_title("GT Canonical")
        axes[0,1].imshow(targ[0].permute(1,2,0).squeeze().cpu(), cmap='gray', vmin=0, vmax=1)
        axes[0,1].set_title(f"Target (Distorted)")
        axes[0,2].imshow(pred[0].permute(1,2,0).squeeze().cpu(), cmap='gray', vmin=0, vmax=1)
        axes[0,2].set_title(f"Learned Canonical (INR)")
        im_res = axes[0,3].imshow(res[0].permute(1,2,0).squeeze().cpu(), cmap='inferno', vmin=0, vmax=0.1)
        axes[0,3].set_title("Recon Residual (L1)")
        plt.colorbar(im_res, ax=axes[0,3])

        # --- Row 2: Jacobian & Warped Grid ---
        im_gt_jac = axes[1,0].imshow(gt_jac, cmap='gray', vmin=0, vmax=2)
        axes[1,0].set_title(f"GT Jac (Folds: {gt_fold_perc:.2f}%)")
        plt.colorbar(im_gt_jac, ax=axes[1,0], fraction=0.046, pad=0.04)

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
        
        # --- Row 3: X-Axis Flows ---
        im_gx = axes[2,0].imshow(gt_fx, cmap='RdBu', vmin=-v_min_max, vmax=v_min_max)
        axes[2,0].set_title("GT Flow X (px)")
        plt.colorbar(im_gx, ax=axes[2,0])
        im_px = axes[2,1].imshow(pred_fx, cmap='RdBu', vmin=-v_min_max, vmax=v_min_max)
        axes[2,1].set_title("Pred Flow X (px)")
        plt.colorbar(im_px, ax=axes[2,1])
        im_ex = axes[2,2].imshow(err_x, cmap='RdBu', vmin=-v_min_max, vmax=v_min_max)
        axes[2,2].set_title(f"Error X (Avg: {mean_err_x:.2f}px)")
        plt.colorbar(im_ex, ax=axes[2,2])
        self.plot_quiver(axes[2,3], gt_fx, gt_fy, "GT Vector Field")

        # --- Row 4: Y-Axis Flows ---
        im_gy = axes[3,0].imshow(gt_fy, cmap='RdBu', vmin=-v_min_max, vmax=v_min_max)
        axes[3,0].set_title("GT Flow Y (px)")
        plt.colorbar(im_gy, ax=axes[3,0])
        im_py = axes[3,1].imshow(pred_fy, cmap='RdBu', vmin=-v_min_max, vmax=v_min_max)
        axes[3,1].set_title("Pred Flow Y (px)")
        plt.colorbar(im_py, ax=axes[3,1])
        im_ey = axes[3,2].imshow(err_y, cmap='RdBu', vmin=-v_min_max, vmax=v_min_max)
        axes[3,2].set_title(f"Error Y (Avg: {mean_err_y:.2f}px)")
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
    # Updated run name to include aug_mode
    aug_str = exp_config.get('aug_mode', 'rigid')
    run_name = f"Consensus_{exp_config['res']}_{aug_str}_views{exp_config['views']}"
    
    if exp_config["wandb"]:
        wandb.init(project="Single-Image-Optimization", config=exp_config, name=run_name, reinit=True)

    print(f"--- Starting Experiment: {run_name} ---")
    
    # 2. Load Image dynamically based on resolution
    base_path = "/home/minsukc/SIO/src_corrupted_ct" 
    clean_image = load_image_dynamic(base_path, exp_config['res'])
    
    if clean_image is None:
        print("Skipping this experiment due to image load error.")
        wandb.finish()
        return

    exp_config["max_shift_px"] = exp_config["res"] * exp_config["shift_ratio"]
    if exp_config.get("aug_mode") == "rigid":
        print(f"Max Shift: {exp_config['shift_ratio']*100}% = {exp_config['max_shift_px']} pixels")
        
    # 4. Initialize Trainer
    trainer = JointTrainer(device, exp_config)

    # print("[Perf] Pre-calculating fixed sparse image...")
    # with torch.no_grad():
    #     fixed_img_sparse_cache = simulate_sparse_ct(clean_image, exp_config['views'], exp_config['noise_i0'])

    print("[Perf] Pre-calculating fixed image...")
    if exp_config.get("use_sparse_view_simulation", True):
        print(f"[DEBUG] Simulation ON: Running Radon transform...")
        with torch.no_grad():
            fixed_img_sparse_cache = simulate_sparse_ct(clean_image, exp_config['views'], exp_config['noise_i0'])
    else:
        print(f"[DEBUG] Simulation OFF: Using clean image directly.")
        fixed_img_sparse_cache = clean_image.clone()

    # 5. Training Loop
    pbar = tqdm(range(exp_config["epochs"]), desc=run_name, leave=False)
    for step in pbar:
        logs = trainer.train_step(clean_image, fixed_img_sparse_cache, step, info_nce_temp = exp_config["info_nce_temp"])
        
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
    "epochs": 5001,
    # "lr_comm": 1e-3,
    "lr_comm": 3e-4,
    "lr_head": 3e-4,
    # "lr_head": 1e-3,
    # "lr_scheduler_epochs": 5000,
    # "lr_scheduler_min_lr": 1e-5,
    "lr_warmup_steps": 500,
    # "infinite_augmentation": True, # Infinite augs for consensus
    "infinite_augmentation": False, # Infinite augs for consensus
    "num_aug": 10,
    "w_align": 1.0,
    "w_naive": 0.0,
    "w_tv": 0.1, # 0.0 to disable
    # "w_cyc": 0.1, # 0.0 to disable
    "w_cyc": 0.0, # 0.0 to disable
    "use_flow_loss": False,
    # "info_nce_temp": 0.10,
    "info_nce_temp": 0.05,
    # "elastic_alpha": 0.05, 
    "elastic_alpha": 0.1, 
    "elastic_grid_res": 6,
    "res": 256,
    "model_type": "fourier",
    "shift_ratio": 0.10,
    "use_correspondence": True, 
    "use_border_mask": False,
    "aug_mode": "elastic", # "elastic", "rigid"
    # "aug_mode": "rigid",
    "encoder": "dinov3-vits", 
    "use_sparse_view_simulation": True,
    "views": 90, # Sparse views
    # "noise_i0": 5000 # Poisson noise (None for no noise)
    "noise_i0": None,
}

experiment_updates = [
    # {"views": 90,  "noise_i0": None, "aug_mode": "rigid", "shift_ratio": 0.05, "epochs": 20001},
    # {"views": 90,  "noise_i0": None, "aug_mode": "elastic", "epochs": 50001},
    # {"views": 90,  "noise_i0": None, "aug_mode": "elastic", "epochs": 50001, "use_flow_loss": True},
    # {"views": 90,  "noise_i0": None, "aug_mode": "elastic", "epochs": 50001, "use_sparse_view_simulation": False, "use_flow_loss": True, "info_nce_temp": 0.05, "w_cyc": 0.01},
    {"views": 90,  "noise_i0": None, "aug_mode": "elastic", "epochs": 50001, "use_sparse_view_simulation": False, "use_flow_loss": False, "info_nce_temp": 0.05, "w_cyc": 0.0},
]

print(f"Total experiments to run: {len(experiment_updates)}")
for i, update in enumerate(experiment_updates):
    exp_cfg = copy.deepcopy(common_config)
    exp_cfg.update(update) 
    
    print(f"\n--- Running Experiment {i+1}/{len(experiment_updates)} ---")
    run_experiment(exp_cfg)