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
import sys
import wandb
import gc
import copy
from skimage.transform import radon, iradon
# from skimage.transform import radon, iradon, resize

from feature_communicator import FeatureCommunicator

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
def load_image_dynamic(base_path, res, device):
    """Loads image based on resolution string"""
    # Prioritize loading the pre-processed tensor directly
    # pt_path = f"{base_path}/L067.pt"
    # if os.path.exists(pt_path):
    #     print(f"[DEBUG] Loading tensor from: {pt_path}")
    #     return torch.load(pt_path, map_location=device)

    # Fallback to PNG logic if .pt not found (or if user wants png)
    path = f"{base_path}/L067_preview.png"
    # path = "/home/minsukc/SIO/data/ct_chest_2_resized_256x256.png"
    
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
    def __init__(self, ref_image, config, device="cuda"):
        self.ref_image = ref_image
        self.cfg = config
        self.device = device
        self.H = config['res']
        self.W = config['res']
        
        self.infinite = config["infinite_augmentation"]
        self.aug_mode = config["aug_mode"]
        self.elastic_alpha = config["elastic_alpha"]
        self.elastic_grid_res = config["elastic_grid_res"]
        
        # Pre-calculate base grid for elastic aug
        theta_identity = torch.tensor([[1, 0, 0], [0, 1, 0]], dtype=torch.float, device=self.device).unsqueeze(0)
        self.base_grid = F.affine_grid(theta_identity, (1, 1, self.H, self.W), align_corners=False)

        # Cache
        self.cached_samples = []
        
        if not self.infinite:
            print(f"[Aug] Generating and Caching {config['num_aug']} samples (Mode: {self.aug_mode})...")
            # We use a simple loop, add tqdm if you want
            for _ in tqdm(range(config['num_aug'])):
                self.cached_samples.append(self._generate_single_sample())

    def _generate_params(self):
        if self.aug_mode == "rigid":
            max_shift = self.cfg["max_shift_px"]
            sx = np.random.uniform(-max_shift, max_shift)
            sy = np.random.uniform(-max_shift, max_shift)
            return {"type": "rigid", "val": (sx, sy)}

        elif self.aug_mode == "elastic":
            grid_h, grid_w = self.elastic_grid_res, self.elastic_grid_res
            # Random offsets in range [-1, 1] * alpha
            coarse_flow = (torch.rand(1, 2, grid_h, grid_w, device=self.device) - 0.5) * 2 * self.elastic_alpha
            flow = F.interpolate(coarse_flow, size=(self.H, self.W), mode="bicubic", align_corners=False)
            flow = flow.permute(0, 2, 3, 1) # [1, H, W, 2]
            return {"type": "elastic", "val": flow}

    def _generate_single_sample(self):
        # 1. Get Geometric Params
        aug_data = self._generate_params()
        
        # 2. Warp the Clean Image & Calculate GT Flow
        gt_flow_y = None
        gt_flow_x = None
        grid = None
        
        if aug_data['type'] == 'rigid':
            shift_x, shift_y = aug_data['val']
            # Affine Matrix Construction
            tx = -shift_x / (self.W / 2.0)
            ty = -shift_y / (self.H / 2.0)
            theta = torch.tensor([[1, 0, tx], [0, 1, ty]], dtype=torch.float, device=self.device).unsqueeze(0)
            grid = F.affine_grid(theta, self.ref_image.size(), align_corners=False)
            
            # Scalar flow for rigid
            gt_flow_y = shift_y
            gt_flow_x = shift_x
            
        elif aug_data['type'] == 'elastic':
            flow = aug_data['val'] # [1, H, W, 2]
            grid = self.base_grid - flow
            
            # Dense flow map for elastic
            gt_flow_y = flow[0, ..., 1] * (self.H / 2.0)
            gt_flow_x = flow[0, ..., 0] * (self.W / 2.0)

        # Apply Geometric Warp
        warped_clean_img = F.grid_sample(self.ref_image, grid, align_corners=False)

        # 3. Simulate Sparse CT (Radon/IRadon)
        if self.cfg.get("use_sparse_view_simulation", True):
            # simulate_sparse_ct is expensive; this is what we are caching
            moving_img_sparse = simulate_sparse_ct(warped_clean_img, self.cfg['views'], self.cfg['noise_i0'])
        else:
            moving_img_sparse = warped_clean_img

        return {
            "moving": moving_img_sparse.detach(), # Detach to save memory
            "warped_clean": warped_clean_img.detach(),
            "gt_flow": (gt_flow_x, gt_flow_y),
            "aug_type": aug_data['type'],
            "params": aug_data['val']
        }

    def get_sample(self):
        if self.infinite:
            return self._generate_single_sample()
        else:
            idx = np.random.randint(0, len(self.cached_samples))
            return self.cached_samples[idx]
            
class Visualizer:
    def __init__(self, device):
        self.device = device

    def get_jacobian_det(self, flow_x, flow_y):
        """Calculates the Jacobian Determinant of the flow field."""
        ux, uy = np.gradient(flow_x)
        vx, vy = np.gradient(flow_y)
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
        
        norm_fx = torch.from_numpy(flow_x).to(self.device).unsqueeze(0) / (W / 2.0)
        norm_fy = torch.from_numpy(flow_y).to(self.device).unsqueeze(0) / (H / 2.0)
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

    def plot(self, data, config=None):
        if not data or "res_dict" not in data: return
        
        res = data["res_dict"]
        gt_fx, gt_fy = data["gt_flow_map"]
        clean = data["clean"]
        fixed = data["fixed"]
        moving = data["moving"]
        step = data["step"]
        aug_type = data["aug_type"]

        main_mode, sec_mode = "Soft", "Hard"
        main_data = res.get("soft")
        sec_data = res.get("hard")

        if main_data is None and sec_data is not None:
            main_mode, sec_mode = "Hard", None
            main_data = sec_data
            sec_data = None
        
        if main_data is None: return 

        nrows = 8 if sec_data is not None else 5
        fig, axes = plt.subplots(nrows, 4, figsize=(20, 5 * nrows), layout="constrained")
        fig.suptitle(f"Step {step} | Aug: {aug_type} | Top: {main_mode} vs Bot: {sec_mode}", fontsize=16)

        def compute_metrics(flow_x, flow_y):
            H, W = fixed.shape[-2:]
            norm_fx = flow_x.to(self.device) / (W / 2.0)
            norm_fy = flow_y.to(self.device) / (H / 2.0)
            norm_flow = torch.stack([norm_fx, norm_fy], dim=-1).unsqueeze(0)
            
            ys = torch.linspace(-1, 1, H, device=self.device)
            xs = torch.linspace(-1, 1, W, device=self.device)
            yy, xx = torch.meshgrid(ys, xs, indexing='ij')
            base_grid = torch.stack([xx, yy], dim=-1).unsqueeze(0)
            
            sample_grid = base_grid + norm_flow
            unwarped = F.grid_sample(moving.to(self.device), sample_grid, align_corners=False, padding_mode="border")
            
            stack = torch.cat([fixed.to(self.device), unwarped], dim=0)
            cons_med = torch.median(stack, dim=0, keepdim=True).values
            cons_avg = torch.mean(stack, dim=0, keepdim=True)
            
            res_med = (cons_med - clean.to(self.device)).abs()
            res_avg = (cons_avg - clean.to(self.device)).abs()
            
            fx_np, fy_np = flow_x.cpu().numpy(), flow_y.cpu().numpy()
            jac = self.get_jacobian_det(fx_np, fy_np)
            grid_img = self.get_warped_grid(H, W, fx_np, fy_np)
            
            return unwarped.cpu(), cons_med.cpu(), cons_avg.cpu(), res_med.cpu(), res_avg.cpu(), jac, grid_img

        # --- PART 1: MAIN MODE ---
        fx, fy = main_data["flow_x"], main_data["flow_y"]
        unwarped, c_med, c_avg, r_med, r_avg, jac, grid_img = compute_metrics(fx, fy)
        
        gt_jac = self.get_jacobian_det(gt_fx.cpu().numpy(), gt_fy.cpu().numpy())
        gt_grid = self.get_warped_grid(fixed.shape[-2], fixed.shape[-1], gt_fx.cpu().numpy(), gt_fy.cpu().numpy())
        v_min, v_max = -10, 10

        # Row 0: Images
        axes[0,0].imshow(clean[0,0].cpu(), cmap='gray'); axes[0,0].set_title("GT Clean")
        axes[0,1].imshow(moving[0,0].cpu(), cmap='gray'); axes[0,1].set_title("Input (Moving)")
        axes[0,2].imshow(fixed[0,0].cpu(), cmap='gray'); axes[0,2].set_title("Target (Fixed)")
        axes[0,3].imshow(unwarped[0,0], cmap='gray'); axes[0,3].set_title(f"[{main_mode}] Unwarped")

        # Row 1: Consensus (Colorbars Added)
        axes[1,0].imshow(c_med[0,0], cmap='gray'); axes[1,0].set_title(f"[{main_mode}] Median Cons")
        im_r1 = axes[1,1].imshow(r_med[0,0], cmap='inferno', vmin=0, vmax=0.1); axes[1,1].set_title("Resid (Median)")
        plt.colorbar(im_r1, ax=axes[1,1])
        axes[1,2].imshow(c_avg[0,0], cmap='gray'); axes[1,2].set_title(f"[{main_mode}] Avg Cons")
        im_r2 = axes[1,3].imshow(r_avg[0,0], cmap='inferno', vmin=0, vmax=0.1); axes[1,3].set_title("Resid (Avg)")
        plt.colorbar(im_r2, ax=axes[1,3])

        # Row 2: Jacobian (Percentage & Colorbar Added)
        gt_perc = ((gt_jac <= 0).sum() / gt_jac.size) * 100
        im_j1 = axes[2,0].imshow(gt_jac, cmap='gray', vmin=0, vmax=2); axes[2,0].set_title(f"GT Jac (Folds: {gt_perc:.2f}%)")
        plt.colorbar(im_j1, ax=axes[2,0])
        
        pred_perc = ((jac <= 0).sum() / jac.size) * 100
        im_j2 = axes[2,1].imshow(jac, cmap='gray', vmin=0, vmax=2)
        fold_mask = (jac <= 0).astype(float)
        axes[2,1].imshow(np.ma.masked_where(fold_mask == 0, fold_mask), cmap=mcolors.ListedColormap(['red']), alpha=1.0)
        axes[2,1].set_title(f"[{main_mode}] Jac (Folds: {pred_perc:.2f}%)")
        plt.colorbar(im_j2, ax=axes[2,1])
        
        axes[2,2].imshow(gt_grid, cmap='gray'); axes[2,2].set_title("GT Grid")
        axes[2,3].imshow(grid_img, cmap='gray'); axes[2,3].set_title(f"[{main_mode}] Pred Grid")

        # Row 3: Flow X (Colorbars Added)
        im_fx1 = axes[3,0].imshow(gt_fx.cpu(), cmap='RdBu', vmin=v_min, vmax=v_max); axes[3,0].set_title("GT Flow X")
        plt.colorbar(im_fx1, ax=axes[3,0])
        im_fx2 = axes[3,1].imshow(fx.cpu(), cmap='RdBu', vmin=v_min, vmax=v_max); axes[3,1].set_title(f"[{main_mode}] Flow X")
        plt.colorbar(im_fx2, ax=axes[3,1])
        err_x = main_data["err_x"]
        im_ex = axes[3,2].imshow(err_x.cpu(), cmap='RdBu', vmin=v_min, vmax=v_max); axes[3,2].set_title(f"Err X (Avg: {err_x.abs().mean():.2f})")
        plt.colorbar(im_ex, ax=axes[3,2])
        self.plot_quiver(axes[3,3], gt_fx.cpu().numpy(), gt_fy.cpu().numpy(), "GT Vectors")

        # Row 4: Flow Y (Colorbars Added)
        im_fy1 = axes[4,0].imshow(gt_fy.cpu(), cmap='RdBu', vmin=v_min, vmax=v_max); axes[4,0].set_title("GT Flow Y")
        plt.colorbar(im_fy1, ax=axes[4,0])
        im_fy2 = axes[4,1].imshow(fy.cpu(), cmap='RdBu', vmin=v_min, vmax=v_max); axes[4,1].set_title(f"[{main_mode}] Flow Y")
        plt.colorbar(im_fy2, ax=axes[4,1])
        err_y = main_data["err_y"]
        im_ey = axes[4,2].imshow(err_y.cpu(), cmap='RdBu', vmin=v_min, vmax=v_max); axes[4,2].set_title(f"Err Y (Avg: {err_y.abs().mean():.2f})")
        plt.colorbar(im_ey, ax=axes[4,2])
        self.plot_quiver(axes[4,3], fx.cpu().numpy(), fy.cpu().numpy(), f"[{main_mode}] Vectors")

        # --- PART 2: SECONDARY MODE (If exists) ---
        if sec_data is not None:
            s_fx, s_fy = sec_data["flow_x"], sec_data["flow_y"]
            s_unw, s_cm, s_ca, s_rm, s_ra, s_jac, s_grid = compute_metrics(s_fx, s_fy)
            s_perc = ((s_jac <= 0).sum() / s_jac.size) * 100

            # Row 5: Structural Analysis
            axes[5,0].imshow(s_unw[0,0], cmap='gray'); axes[5,0].set_title(f"[{sec_mode}] Unwarped")
            im_sj = axes[5,1].imshow(s_jac, cmap='gray', vmin=0, vmax=2)
            s_fold_m = (s_jac <= 0).astype(float)
            axes[5,1].imshow(np.ma.masked_where(s_fold_m == 0, s_fold_m), cmap=mcolors.ListedColormap(['red']), alpha=1.0)
            axes[5,1].set_title(f"[{sec_mode}] Jac (Folds: {s_perc:.2f}%)")
            plt.colorbar(im_sj, ax=axes[5,1])
            axes[5,2].imshow(s_grid, cmap='gray'); axes[5,2].set_title(f"[{sec_mode}] Grid")
            self.plot_quiver(axes[5,3], s_fx.cpu().numpy(), s_fy.cpu().numpy(), f"[{sec_mode}] Vectors")

            # Row 6: Flow & Error Summary
            im_sfx = axes[6,0].imshow(s_fx.cpu(), cmap='RdBu', vmin=v_min, vmax=v_max); plt.colorbar(im_sfx, ax=axes[6,0])
            s_ex = sec_data["err_x"]
            im_sex = axes[6,1].imshow(s_ex.cpu(), cmap='RdBu', vmin=v_min, vmax=v_max); plt.colorbar(im_sex, ax=axes[6,1])
            im_sfy = axes[6,2].imshow(s_fy.cpu(), cmap='RdBu', vmin=v_min, vmax=v_max); plt.colorbar(im_sfy, ax=axes[6,2])
            s_ey = sec_data["err_y"]
            im_sey = axes[6,3].imshow(s_ey.cpu(), cmap='RdBu', vmin=v_min, vmax=v_max); plt.colorbar(im_sey, ax=axes[6,3])

            # Row 7: Consensus Summary
            axes[7,0].imshow(s_cm[0,0], cmap='gray'); axes[7,0].set_title(f"[{sec_mode}] Med Cons")
            im_srm = axes[7,1].imshow(s_rm[0,0], cmap='inferno', vmin=0, vmax=0.1); plt.colorbar(im_srm, ax=axes[7,1])
            axes[7,2].imshow(s_ca[0,0], cmap='gray'); axes[7,2].set_title(f"[{sec_mode}] Avg Cons")
            im_sra = axes[7,3].imshow(s_ra[0,0], cmap='inferno', vmin=0, vmax=0.1); plt.colorbar(im_sra, ax=axes[7,3])

        if config is not None and config.get("wandb", True) and wandb.run is not None:
             wandb.log({"Visual/Result": wandb.Image(fig), "step": step})
        
        plt.close(fig)

# ==========================================
# 4. Joint Trainer
# ==========================================
class JointTrainer:
    def __init__(self, device, config, clean_image, fixed_img_sparse):
        self.device = device
        self.cfg = config

        self.clean_image = clean_image
        self.fixed_img_sparse = fixed_img_sparse

        self.use_corr = config.get("use_correspondence", True)
        
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

            self.use_communicator = config["use_communicator"]
            dim_dec = 768

            if self.use_communicator:
                dim_total = dim_enc + dim_dec
            else:
                dim_total = dim_enc

            self.head_ref = MatchingHead(input_dim=dim_total).to(device)
            self.head_mov = MatchingHead(input_dim=dim_total).to(device)
            
            params_list = [
                {'params': self.head_ref.parameters(), 'lr': config["lr_head"]},
                {'params': self.head_mov.parameters(), 'lr': config["lr_head"]}
            ]

            if self.use_communicator:
                print("[Model] Initializing Feature Communicator...")
                self.communicator = FeatureCommunicator(
                    input_dim=dim_enc, 
                    embed_dim=dim_dec, 
                    grid_size=(config['res']//16, config['res']//16),
                    depth=config['communicator_depth'], 
                    num_heads=config['communicator_num_heads']
                ).to(device)
                params_list.append({'params': self.communicator.parameters(), 'lr': config["lr_comm"]})
            else:
                print("[Model] Skipping Feature Communicator (Direct DINO -> Head)...")
                self.communicator = None

            # 4. Initialize Optimizer
            self.optimizer = torch.optim.Adam(params_list)
            self.scaler = torch.amp.GradScaler("cuda")

            if config['use_scheduler']:
                warmup_steps = config["lr_warmup_steps"]
                def lr_lambda(current_step):
                    if current_step < warmup_steps:
                        return float(current_step) / float(max(1, warmup_steps))
                    return 1.0
                self.scheduler = torch.optim.lr_scheduler.LambdaLR(self.optimizer, lr_lambda)
            else:
                self.scheduler = None

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

        if self.use_corr:
            print("[Perf] Pre-computing Fixed Image Features (One-time cost)...")
            with torch.no_grad():
                # This runs the Heavy Vision Transformer ONE time.
                self.cached_enc_fixed = self.get_backbone_feats(self.fixed_img_sparse).detach()
                print(f"[Perf] Cached Fixed Features shape: {self.cached_enc_fixed.shape}")

                
        # State
        self.augmentor = AugmentationManager(
            ref_image=self.clean_image,
            config=config,
            device=device
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

    def sample_at_coords(self, feature_map, x_px, y_px, H, W):
        norm_x = ((x_px + 0.5) / W) * 2 - 1
        norm_y = ((y_px + 0.5) / H) * 2 - 1
        
        grid = torch.stack([norm_x, norm_y], dim=-1).unsqueeze(0).unsqueeze(0)
        
        mode = self.cfg["sampling_mode"]
        sampled = F.grid_sample(feature_map, grid, align_corners=False, mode=mode, padding_mode="border")
        return sampled.squeeze(0).squeeze(1).T
        
    def train_step(self, step, info_nce_temp=0.05):
        self.optimizer.zero_grad()
        B, C, H, W = self.clean_image.shape
        
        # --- A. Augmentation (Cached or Generated) ---
        sample = self.augmentor.get_sample()
        
        moving_img_sparse = sample["moving"]
        warped_clean_img = sample["warped_clean"] 
        gt_flow_x, gt_flow_y = sample["gt_flow"]
        aug_type = sample["aug_type"]

        logs = {}
        total_loss = 0.0

        with torch.amp.autocast("cuda"):
            if self.use_corr:
                # --- C. InfoNCE & Alignment ---
                # enc_fixed = self.get_backbone_feats(self.fixed_img_sparse)   
                enc_fixed = self.cached_enc_fixed
                enc_moving = self.get_backbone_feats(moving_img_sparse)   
    
                if self.use_communicator:
                    dec_fixed, dec_moving = self.communicator(enc_fixed, enc_moving) 
                    feats_fixed = torch.cat([enc_fixed, dec_fixed], dim=-1)
                    feats_moving = torch.cat([enc_moving, dec_moving], dim=-1)
                else:
                    feats_fixed = enc_fixed
                    feats_moving = enc_moving
                
                desc_fixed = self.head_ref(feats_fixed, H, W)
                desc_moving = self.head_mov(feats_moving, H, W)
                
                # (1) InfoNCE Loss - SOFT NEGATIVES
                num_samples = self.cfg["num_samples"] 
                flat_inds = torch.randperm(H * W, device=self.device)[:num_samples]
                # flat_inds = torch.randint(0, H * W, (num_samples,), device=self.device)
                ys = flat_inds // W
                xs = flat_inds % W
                
                feats_q = desc_fixed[0, :, ys, xs].T
                
                if aug_type == 'rigid':
                    target_y_float = ys.float() + gt_flow_y
                    target_x_float = xs.float() + gt_flow_x
                elif aug_type == 'elastic':
                    dy = gt_flow_y[ys, xs]
                    dx = gt_flow_x[ys, xs]
                    target_y_float = ys.float() + dy
                    target_x_float = xs.float() + dx

                # Sample Positive Keys via Bilinear Interpolation
                # This allows the model to see sub-pixel features
                feats_k_pos = self.sample_at_coords(desc_moving, target_x_float, target_y_float, H, W)
    
                # temp = 0.05
                logits = torch.matmul(feats_q, feats_k_pos.T) / info_nce_temp
                labels = torch.arange(num_samples, device=self.device)
                
                loss_nce_q2k = self.ce_loss(logits, labels)    
                loss_nce_k2q = self.ce_loss(logits.T, labels)  
                loss_nce = (loss_nce_q2k + loss_nce_k2q) / 2.0
                
                total_loss += loss_nce
                logs["Loss/InfoNCE"] = loss_nce.item()
    
                # Supervised Soft-Argmax Flow Loss
                w_flow = self.cfg.get("w_flow", 0.0)
                if w_flow > 0:
                    temp_soft = self.cfg.get("soft_argmax_temp", 0.05)
                    pred_x_f2m, pred_y_f2m = self.compute_soft_argmax_flow(feats_q, desc_moving, H, W, temp=temp_soft)
                    w_flow = self.cfg.get("w_flow", 0.1)
                    loss_flow = F.mse_loss(pred_x_f2m, target_x_float) + F.mse_loss(pred_y_f2m, target_y_float)
                    total_loss += w_flow * loss_flow
                    logs["Loss/Flow_MSE"] = loss_flow.item()
       
    
                # Cycle Consistency
                cycle_warmup_steps = 5000
                w_cyc=self.cfg.get("w_cyc", 0.0) > 0
                if w_cyc > 0 and step > cycle_warmup_steps:
                    temp_soft = self.cfg.get("soft_argmax_temp", 0.05) 
                    if w_flow <= 0:
                         pred_x_f2m, pred_y_f2m = self.compute_soft_argmax_flow(feats_q, desc_moving, H, W, temp=temp_soft)
                    
                    # Use bilinear sampling for cycle back-projection
                    feats_from_moving = self.sample_at_coords(desc_moving, pred_x_f2m, pred_y_f2m, H, W)
                    pred_x_m2f, pred_y_m2f = self.compute_soft_argmax_flow(feats_from_moving, desc_fixed, H, W, temp=temp_soft)
                    
                    loss_cyc = F.mse_loss(pred_x_m2f, xs.float()) + F.mse_loss(pred_y_m2f, ys.float())
                    total_loss += w_cyc * loss_cyc
                    logs["Loss/Cycle"] = loss_cyc.item()  
                    
                # --- Calc Correspondence Error for Viz ---
                if step % 100 == 0:
                    with torch.no_grad():
                        flat_mov = desc_moving.view(24, -1)
                        sim_matrix = torch.matmul(feats_q.detach(), flat_mov) # [N, HW]
                        
                        # 1. Hard Argmax Error (Nearest Neighbor - Integer)
                        match_idx = sim_matrix.argmax(dim=1)
                        match_ys = match_idx // W
                        match_xs = match_idx % W
                        err_y_hard = (match_ys - target_y_float).abs().mean()
                        err_x_hard = (match_xs - target_x_float).abs().mean()
                        
                        # 2. Soft Argmax Error (Sub-pixel - Float)
                        temp_soft = self.cfg.get("soft_argmax_temp", 0.05) 
                        prob = F.softmax(sim_matrix / temp_soft, dim=1)
                        pred_y_soft = torch.sum(prob * self.cached_flat_y, dim=1)
                        pred_x_soft = torch.sum(prob * self.cached_flat_x, dim=1)
                        err_y_soft = (pred_y_soft - target_y_float).abs().mean()
                        err_x_soft = (pred_x_soft - target_x_float).abs().mean()
    
                        logs["Metrics/Corr_Err_Px_Hard"] = ((err_y_hard + err_x_hard) / 2.0).item()
                        logs["Metrics/Corr_Err_Px_Soft"] = ((err_y_soft + err_x_soft) / 2.0).item()
            else:
                loss_naive = F.mse_loss(fixed_img_sparse, moving_img_sparse)
                total_loss = loss_naive
                logs["Loss/Naive_MSE"] = loss_naive.item()
        
        # --- D. Update & Logs ---
        self.scaler.scale(total_loss).backward()
        
        if self.use_corr:
            self.scaler.unscale_(self.optimizer) # Important for clipping
            logs["Train/LR_Head"] = self.optimizer.param_groups[0]['lr'] 
            
            norm_ref = torch.nn.utils.clip_grad_norm_(self.head_ref.parameters(), max_norm=1.0)
            norm_mov = torch.nn.utils.clip_grad_norm_(self.head_mov.parameters(), max_norm=1.0)
            logs["Train/GradNorm_HeadRef"] = norm_ref.item()
            logs["Train/GradNorm_HeadMov"] = norm_mov.item()

            if self.use_communicator:
                logs["Train/LR_Comm"] = self.optimizer.param_groups[2]['lr'] 
                norm_comm = torch.nn.utils.clip_grad_norm_(self.communicator.parameters(), max_norm=1.0)
                logs["Train/GradNorm_Comm"] = norm_comm.item()
        
        self.scaler.step(self.optimizer)
        self.scaler.update()

        if self.scheduler is not None:
            self.scheduler.step()
        
        logs["Loss/Total"] = total_loss.item()
        
        if step % 500 == 0:
            self.latest_viz_data = {
                    "fixed": self.fixed_img_sparse.detach(), 
                    "moving": moving_img_sparse.detach(),
                    "clean": self.clean_image.detach(),
                    "gt_flow": (gt_flow_x, gt_flow_y), 
                    "aug_type": aug_type,
                    "step": step,
                }
        
        if self.cfg.get("wandb", True) and step % 10 == 0:
            wandb.log(logs, step=step)
            
        return logs
        
    def log_fixed_augmentations(self):
        if not self.cfg.get("wandb", True): return
        
        samples = self.augmentor.cached_samples
        if not samples or len(samples) == 0:
            print("[Log] No fixed samples found (Infinite Aug mode). Skipping log.")
            return

        print(f"[Log] Logging augmentation samples to WandB...")
        
        # NOTE: Show up to 7 augmented samples + 1 slot for the Fixed Reference
        num_aug_to_show = min(len(samples), 7)
        total_cols = num_aug_to_show + 1
        
        fig, axes = plt.subplots(1, total_cols, figsize=(3 * total_cols, 4))
        
        # NOTE: Reference Image (The "Target" we want to align to)
        fixed_ref = self.clean_image.squeeze().cpu().float()
        axes[0].imshow(fixed_ref, cmap='gray', vmin=0, vmax=1)
        axes[0].set_title("FIXED (Ref)")
        axes[0].axis('off')

        # NOTE: Plot augmented samples
        for i in range(num_aug_to_show):
            ax = axes[i + 1]
            moving = samples[i]["moving"].squeeze().cpu().float()
            
            ax.imshow(moving, cmap='gray', vmin=0, vmax=1)
            ax.set_title(f"Augmented {i}")
            ax.axis('off')

        plt.suptitle(f"Augmentation Cache Preview (Total in Cache: {len(samples)})")
        plt.tight_layout()
        
        # NOTE: Added debug line to verify flow presence
        print(f"[DEBUG] Visualizing {num_aug_to_show} out of {len(samples)} samples.")
        
        wandb.log({"Train/Fixed_Augmentations": wandb.Image(fig)})
        plt.close(fig)

    def compute_dense_error_map(self, fixed, moving, gt_flow, aug_type):
        B, C, H, W = fixed.shape
        
        with torch.amp.autocast("cuda"):
            # enc_fixed = self.get_backbone_feats(fixed)
            enc_fixed = self.cached_enc_fixed
            enc_moving = self.get_backbone_feats(moving)
            
            if self.use_communicator:
                dec_fixed, dec_moving = self.communicator(enc_fixed, enc_moving)
                feats_fixed = torch.cat([enc_fixed, dec_fixed], dim=-1)
                feats_moving = torch.cat([enc_moving, dec_moving], dim=-1)
            else:
                feats_fixed = enc_fixed
                feats_moving = enc_moving
            
            desc_fixed = self.head_ref(feats_fixed, H, W)
            desc_moving = self.head_mov(feats_moving, H, W)
        
        flat_fixed = desc_fixed.view(24, -1).T  # [HW, 24]
        flat_moving = desc_moving.view(24, -1).T  # [HW, 24]
        
        gt_flow_x, gt_flow_y = gt_flow
        gt_flow_map_x = torch.zeros(H, W)
        gt_flow_map_y = torch.zeros(H, W)
        
        # chunk_size = 4096 
        chunk_size = self.cfg["chunk_size"]
        num_pixels = H * W
        # use_soft = self.cfg.get("use_soft_argmax", True) # Config에서 제어
        temp = self.cfg.get("soft_argmax_temp", 0.05)
        do_hard = self.cfg.get("viz_hard", True)
        do_soft = self.cfg.get("viz_soft", True)

        res = {
            "hard": None,
            "soft": None
        }
        if do_hard:
            res["hard"] = {"flow_x": torch.zeros(H, W), "flow_y": torch.zeros(H, W), "err_x": torch.zeros(H, W), "err_y": torch.zeros(H, W)}
        if do_soft:
            res["soft"] = {"flow_x": torch.zeros(H, W), "flow_y": torch.zeros(H, W), "err_x": torch.zeros(H, W), "err_y": torch.zeros(H, W)}
        
        for i in range(0, num_pixels, chunk_size):
            end = min(i + chunk_size, num_pixels)
            q_chunk = flat_fixed[i:end]
            sim = torch.matmul(q_chunk, flat_moving.T)
            
            indices = torch.arange(i, end)
            ys = indices // W
            xs = indices % W

            # GT Flow
            if aug_type == 'rigid':
                actual_dy = torch.full_like(ys, float(gt_flow_y)).float()
                actual_dx = torch.full_like(xs, float(gt_flow_x)).float()
            else:
                actual_dy = gt_flow_y[ys, xs].cpu()
                actual_dx = gt_flow_x[ys, xs].cpu()
            gt_flow_map_y.view(-1)[i:end] = actual_dy
            gt_flow_map_x.view(-1)[i:end] = actual_dx
            
            # ==========================================
            # [CASE 1] Hard Argmax (Conditional)
            # ==========================================
            if do_hard:
                match_idx = sim.argmax(dim=1).cpu()
                pred_y_hard = (match_idx // W).float()
                pred_x_hard = (match_idx % W).float()
                
                flow_y_hard = (pred_y_hard - ys).float()
                flow_x_hard = (pred_x_hard - xs).float()
                
                res["hard"]["flow_y"].view(-1)[i:end] = flow_y_hard
                res["hard"]["flow_x"].view(-1)[i:end] = flow_x_hard
                res["hard"]["err_y"].view(-1)[i:end] = flow_y_hard - actual_dy
                res["hard"]["err_x"].view(-1)[i:end] = flow_x_hard - actual_dx

            # ==========================================
            # [CASE 2] Soft Argmax (Conditional)
            # ==========================================
            if do_soft:
                prob = F.softmax(sim / temp, dim=1)
                pred_y_soft = torch.sum(prob * self.cached_flat_y.to(self.device), dim=1).cpu()
                pred_x_soft = torch.sum(prob * self.cached_flat_x.to(self.device), dim=1).cpu()
                
                flow_y_soft = (pred_y_soft - ys).float()
                flow_x_soft = (pred_x_soft - xs).float()

                res["soft"]["flow_y"].view(-1)[i:end] = flow_y_soft
                res["soft"]["flow_x"].view(-1)[i:end] = flow_x_soft
                res["soft"]["err_y"].view(-1)[i:end] = flow_y_soft - actual_dy
                res["soft"]["err_x"].view(-1)[i:end] = flow_x_soft - actual_dx
            
        return res, (gt_flow_map_x, gt_flow_map_y)

    @torch.no_grad()
    def get_viz_data(self, step):
        if not self.latest_viz_data: return None

        data = self.latest_viz_data
        
        # 2. Compute Dense Flow (moved logic here)
        if self.use_corr:
            res_dict, gt_flow_map = self.compute_dense_error_map(
                data["fixed"], data["moving"], data["gt_flow"], data["aug_type"]
            )
            # Add results to data dict
            data.update({
                "res_dict": res_dict,
                "gt_flow_map": gt_flow_map,
            })
            
        return data

        
def run_experiment(exp_config, update_dict=None):
    set_seed(42)

    # 1. Setup WandB
    notes_str = str(update_dict) if update_dict else "No specific updates"

    sim_prefix = f"SV_v{exp_config['views']}" if exp_config['use_sparse_view_simulation'] else "CLN"
    noise_info = f"_N{exp_config['noise_i0']}" if exp_config['noise_i0'] is not None else "_NoNoise"
    
    if exp_config['use_communicator']:
        comm_info = f"_Comm_D{exp_config['communicator_depth']}H{exp_config['communicator_num_heads']}"
    else:
        comm_info = "_NoComm"
    
    if exp_config['aug_mode'] == "elastic":
        aug_info = f"_Elas_a{exp_config['elastic_alpha']}"
    else:
        aug_info = f"_Rigid_s{exp_config['shift_ratio']}"
    run_name = f"{sim_prefix}{noise_info}{comm_info}{aug_info}_Res{exp_config['res']}"
    
    if exp_config["wandb"]:
        tags = exp_config.get("wandb_tags", [])
        
        wandb.init(
            project="Single-Image-Optimization", 
            config=exp_config, 
            name=run_name, 
            notes=notes_str, 
            tags=tags,
            save_code=True,   
            reinit=True
        )
        wandb.run.log_code(".", include_fn=lambda path: path.endswith(".py"))
        
    print(f"--- Starting Experiment: {run_name} ---")
    
    # 2. Load Image dynamically based on resolution
    base_path = "/home/minsukc/SIO/src_corrupted_ct" 
    clean_image = load_image_dynamic(base_path, exp_config['res'], device)
    
    if clean_image is None:
        print("Skipping this experiment due to image load error.")
        wandb.finish()
        return

    exp_config["max_shift_px"] = exp_config["res"] * exp_config["shift_ratio"]
    if exp_config.get("aug_mode") == "rigid":
        print(f"Max Shift: {exp_config['shift_ratio']*100}% = {exp_config['max_shift_px']} pixels")

    print("[Perf] Pre-calculating fixed image...")
    if exp_config.get("use_sparse_view_simulation", True):
        print(f"[DEBUG] Simulation ON: Running Radon transform...")
        with torch.no_grad():
            fixed_img_sparse_cache = simulate_sparse_ct(clean_image, exp_config['views'], exp_config['noise_i0'])
    else:
        print(f"[DEBUG] Simulation OFF: Using clean image directly.")
        fixed_img_sparse_cache = clean_image.clone()
    
    # Initialize Trainer
    trainer = JointTrainer(device, exp_config, clean_image, fixed_img_sparse_cache)
    visualizer = Visualizer(device) 
    trainer.log_fixed_augmentations()

    # 5. Training Loop
    pbar = tqdm(range(exp_config["epochs"]), desc=run_name, leave=False)
    for step in pbar:
        logs = trainer.train_step(step, info_nce_temp = exp_config["info_nce_temp"])
        
        if step % 100 == 0:
            desc = f"L: {logs['Loss/Total']:.4f}"
            if exp_config["use_correspondence"]:
                hard_err = logs.get("Metrics/Corr_Err_Px_Hard")
                soft_err = logs.get("Metrics/Corr_Err_Px_Soft")
                
                if hard_err is not None:
                    desc += f" | Hard: {hard_err:.2f}px"
                if soft_err is not None:
                    desc += f" | Soft: {soft_err:.2f}px"
            pbar.set_postfix_str(desc)
            
        if step % 500 == 0:
            viz_data = trainer.get_viz_data(step)
            if viz_data:
                visualizer.plot(viz_data, exp_config)

    # 6. Cleanup
    if exp_config["wandb"]: wandb.finish()
    
    # Force delete to free GPU memory for next run
    del trainer
    del clean_image
    torch.cuda.empty_cache()
    gc.collect()
    print(f"--- Finished {run_name} ---\n")

# ==========================================
# Configuration
# ==========================================
common_config = {
    # --- Experiment & Logging ---
    "wandb": True,
    "wandb_tags": [],
    "epochs": 5001,
    
    # --- Data & Resolution ---
    "res": 256,
    "views": 90, # Number of views for Sparse CT simulation
    "noise_i0": None, # Poisson noise intensity (None to disable)
    "use_sparse_view_simulation": True, # If False, performs direct clean-to-clean registration
    "use_border_mask": False,

    # --- Augmentation (Geometric) ---
    "aug_mode": "elastic", # Options: "elastic", "rigid"
    "num_aug": 10,              
    "infinite_augmentation": False,
    "shift_ratio": 0.10,          
    "elastic_alpha": 0.1,
    "elastic_grid_res": 6, 

    # --- Model Architecture ---
    "encoder": "dinov3-vits", # Options: "dinov3-vits", "dinov3-vitb"
    "use_correspondence": True, # Main flag to enable the feature matching pipeline

    # --- Feature Communicator ---
    "use_communicator": False, 
    "communicator_depth": 2,
    "communicator_num_heads": 8,

    # --- Optimization ---
    "lr_comm": 3e-4,            
    "lr_head": 3e-4,            
    "use_scheduler": False,     
    "lr_warmup_steps": 500,

    # --- Loss Weights & Hyperparams ---
    "info_nce_temp": 0.05,
    "w_align": 1.0, # Weight for alignment (InfoNCE)
    "w_naive": 0.0, # Weight for naive MSE (if not using correspondence)
    "w_tv": 0.0, # Total Variation regularization weight
    "w_cyc": 0.0, # Cycle Consistency weight
    "w_flow": 0.0, # Flow Loss weight. Uses soft-argmax. 0.0 to disable

    # Sub-pixel / Continuous Learning Options
    # "use_soft_argmax": True,    # True: DVF/Vis/Metric에 Soft-argmax 사용 (매끄러움), False: Hard-argmax # DEPRECATED
    "sampling_mode": "bilinear", # 'bilinear' (Continuous) or 'nearest' (Discrete)
    "soft_argmax_temp": 0.01,
    "viz_hard": True,
    "viz_soft": True,

    # --- Performance & Batching ---
    "num_samples": 4096, # Number of pixels sampled for InfoNCE
    "chunk_size": 4096, # Batch size for dense evaluation (inference)
}

experiment_updates = [
    # --- Experiment 1: Elastic Aug, No Simulation, With Communicator ---
    {
        "aug_mode": "elastic", 
        "epochs": 100001, 
        "use_sparse_view_simulation": False, 
        "use_communicator": False, 
        "communicator_depth": 2, 
        "communicator_num_heads": 8,
        "elastic_alpha": 0.05,
        
        # "viz_soft": False,
        # "soft_argmax_temp": 0.005,
        # "w_flow": 0.1,
        "sampling_mode": "nearest",
        "viz_soft": False,
        
        
        # "wandb_tags": ["bilinear", "temp 0.01"],
        # "wandb_tags": ["bilinear", "temp 0.01", "flow loss"],
    },
]

print(f"Total experiments to run: {len(experiment_updates)}")
try: 
    for i, update in enumerate(experiment_updates):
        exp_cfg = copy.deepcopy(common_config)
        exp_cfg.update(update) 
        
        print(f"\n--- Running Experiment {i+1}/{len(experiment_updates)} ---")
        run_experiment(exp_cfg, update_dict=update)
except KeyboardInterrupt: 
    print("\n[STOP] User interrupted the process. Cleaning up...")
    if wandb.run is not None:
        wandb.finish()
    print("[STOP] Exiting all experiments.")
    sys.exit(0) 