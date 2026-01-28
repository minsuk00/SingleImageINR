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

    def plot(self, data, config):
        if not data: return
        
        fixed = data["fixed"] 
        moving = data["moving"] 
        clean = data["clean"]
        step = data["step"]
        gt_flow = data["gt_flow"]
        aug_type = data["aug_type"]
        
        if "pred_flow" in data:
            pred_fx, pred_fy = data["pred_flow"]
            gt_fx, gt_fy = data["gt_flow_map"]
            err_x, err_y = data["error_map"]
            
            # --- Median & Average Consensus ---
            H, W = fixed.shape[-2:]
            norm_flow_x = pred_fx.to(self.device).unsqueeze(0).unsqueeze(0) / (W / 2.0)
            norm_flow_y = pred_fy.to(self.device).unsqueeze(0).unsqueeze(0) / (H / 2.0)
            norm_flow = torch.stack([norm_flow_x, norm_flow_y], dim=-1).squeeze(0)

            yy, xx = torch.meshgrid(torch.linspace(-1, 1, H, device=self.device), torch.linspace(-1, 1, W, device=self.device), indexing="ij")
            base_grid = torch.stack([xx, yy], dim=-1).unsqueeze(0)
            sample_grid = base_grid + norm_flow
            
            unwarped_moving = F.grid_sample(moving, sample_grid, align_corners=False, padding_mode="border")
            
            stack = torch.cat([fixed, unwarped_moving], dim=0)
            
            # Calculate both Median and Mean
            consensus_median = torch.median(stack, dim=0).values 
            consensus_avg = torch.mean(stack, dim=0)
            
            res_median = (consensus_median - clean).abs() 
            res_avg = (consensus_avg - clean).abs()
            
            # --- Move Tensors to CPU/Numpy ---
            pred_fx, pred_fy = pred_fx.cpu(), pred_fy.cpu()
            gt_fx, gt_fy = gt_fx.cpu(), gt_fy.cpu()
            err_x, err_y = err_x.cpu(), err_y.cpu()

            # --- Compute Grids & Jacobian ---
            # Jacobian Calculation
            pred_jac = self.get_jacobian_det(pred_fx, pred_fy)
            gt_jac = self.get_jacobian_det(gt_fx, gt_fy)
            
            # Folding Statistics
            num_pix = pred_fx.shape[0] * pred_fx.shape[1]
            pred_folds = (pred_jac <= 0).sum()
            gt_folds = (gt_jac <= 0).sum()
            pred_fold_perc = (pred_folds / num_pix) * 100
            gt_fold_perc = (gt_folds / num_pix) * 100

            warped_grid = self.get_warped_grid(H, W, pred_fx, pred_fy)
            gt_warped_grid = self.get_warped_grid(H, W, gt_fx, gt_fy)
            
            mean_err_x = err_x.abs().mean().item()
            mean_err_y = err_y.abs().mean().item()

        else:
            return

        v_min_max = 10
        
        fig, axes = plt.subplots(5, 4, figsize=(20, 25), layout="constrained")
        fig.suptitle(f"Step {step} | Aug: {aug_type}", fontsize=16)
        
        # --- Row 1: Images ---
        axes[0,0].imshow(clean[0].permute(1,2,0).squeeze().cpu(), cmap='gray', vmin=0, vmax=1)
        axes[0,0].set_title("GT Clean")
        axes[0,1].imshow(moving[0].permute(1,2,0).squeeze().cpu(), cmap='gray', vmin=0, vmax=1)
        axes[0,1].set_title(f"Moving (Distorted)")
        axes[0,2].imshow(fixed[0].permute(1,2,0).squeeze().cpu(), cmap='gray', vmin=0, vmax=1)
        axes[0,2].set_title(f"Fixed (Reference)")
        axes[0,3].imshow(unwarped_moving[0].permute(1,2,0).squeeze().detach().cpu(), cmap='gray', vmin=0, vmax=1)
        axes[0,3].set_title("Unwarped Moving")

        # --- Row 2: Consensus & Residuals ---
        axes[1,0].imshow(consensus_median.permute(1,2,0).squeeze().cpu(), cmap='gray', vmin=0, vmax=1)
        axes[1,0].set_title(f"Median Consensus")
        
        im_res_med = axes[1,1].imshow(res_median[0].permute(1,2,0).squeeze().cpu(), cmap='inferno', vmin=0, vmax=0.1)
        axes[1,1].set_title("Resid (Median)")
        plt.colorbar(im_res_med, ax=axes[1,1])
        
        axes[1,2].imshow(consensus_avg.permute(1,2,0).squeeze().cpu(), cmap='gray', vmin=0, vmax=1)
        axes[1,2].set_title(f"Average Consensus")
        
        im_res_avg = axes[1,3].imshow(res_avg[0].permute(1,2,0).squeeze().cpu(), cmap='inferno', vmin=0, vmax=0.1)
        axes[1,3].set_title("Resid (Average)")
        plt.colorbar(im_res_avg, ax=axes[1,3])

        # --- Row 3: Jacobian & Grids ---
        im_gt_jac = axes[2,0].imshow(gt_jac, cmap='gray', vmin=0, vmax=2)
        axes[2,0].set_title(f"GT Jac (Folds: {gt_fold_perc:.2f}%)")
        plt.colorbar(im_gt_jac, ax=axes[2,0])

        im_pred_jac = axes[2,1].imshow(pred_jac, cmap='gray', vmin=0, vmax=2)
        # Overlay red for folds
        fold_mask = (pred_jac <= 0).astype(float)
        masked_red = np.ma.masked_where(fold_mask == 0, fold_mask)
        axes[2,1].imshow(masked_red, cmap=mcolors.ListedColormap(['red']), alpha=1.0)
        axes[2,1].set_title(f"Pred Jac (Red=Fold) (Folds: {pred_fold_perc:.2f}%)")
        plt.colorbar(im_pred_jac, ax=axes[2,1])

        axes[2,2].imshow(gt_warped_grid, cmap='gray')
        axes[2,2].set_title("GT Grid")
        axes[2,3].imshow(warped_grid, cmap='gray')
        axes[2,3].set_title("Pred Grid")

        # --- Row 4: X-Axis Flows ---
        im_gx = axes[3,0].imshow(gt_fx, cmap='RdBu', vmin=-v_min_max, vmax=v_min_max)
        axes[3,0].set_title("GT Flow X")
        plt.colorbar(im_gx, ax=axes[3,0])
        im_px = axes[3,1].imshow(pred_fx, cmap='RdBu', vmin=-v_min_max, vmax=v_min_max)
        axes[3,1].set_title("Pred Flow X")
        plt.colorbar(im_px, ax=axes[3,1])
        im_ex = axes[3,2].imshow(err_x, cmap='RdBu', vmin=-v_min_max, vmax=v_min_max)
        axes[3,2].set_title(f"Error X (Avg: {mean_err_x:.2f})")
        plt.colorbar(im_ex, ax=axes[3,2])
        self.plot_quiver(axes[3,3], gt_fx, gt_fy, "GT Vectors")

        # --- Row 5: Y-Axis Flows ---
        im_gy = axes[4,0].imshow(gt_fy, cmap='RdBu', vmin=-v_min_max, vmax=v_min_max)
        axes[4,0].set_title("GT Flow Y")
        plt.colorbar(im_gy, ax=axes[4,0])
        im_py = axes[4,1].imshow(pred_fy, cmap='RdBu', vmin=-v_min_max, vmax=v_min_max)
        axes[4,1].set_title("Pred Flow Y")
        plt.colorbar(im_py, ax=axes[4,1])
        im_ey = axes[4,2].imshow(err_y, cmap='RdBu', vmin=-v_min_max, vmax=v_min_max)
        axes[4,2].set_title(f"Error Y (Avg: {mean_err_y:.2f})")
        plt.colorbar(im_ey, ax=axes[4,2])
        self.plot_quiver(axes[4,3], pred_fx, pred_fy, "Pred Vectors")
        
        if config.get("wandb", True):
             wandb.log({
                "Visual/Result": wandb.Image(fig),
             }, step=step)
        
        plt.show()
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
                    # gt_flow_x/y are scalars
                    ys_tgt = torch.clamp(ys + int(round(gt_flow_y)), 0, H-1)
                    xs_tgt = torch.clamp(xs + int(round(gt_flow_x)), 0, W-1)
                    
                    target_y_float = ys.float() + gt_flow_y
                    target_x_float = xs.float() + gt_flow_x
                    
                elif aug_type == 'elastic':
                    # gt_flow_x/y are tensors [H, W]
                    dy = gt_flow_y[ys, xs]
                    dx = gt_flow_x[ys, xs]
                    
                    ys_tgt = torch.clamp(ys + torch.round(dy).long(), 0, H-1)
                    xs_tgt = torch.clamp(xs + torch.round(dx).long(), 0, W-1)
                    
                    target_y_float = ys.float() + dy
                    target_x_float = xs.float() + dx
    
                feats_k_pos = desc_moving[0, :, ys_tgt, xs_tgt].T
    
                # temp = 0.05
                logits = torch.matmul(feats_q, feats_k_pos.T) / info_nce_temp
                labels = torch.arange(num_samples, device=self.device)
                
                loss_nce_q2k = self.ce_loss(logits, labels)    
                loss_nce_k2q = self.ce_loss(logits.T, labels)  
                loss_nce = (loss_nce_q2k + loss_nce_k2q) / 2.0
                
                total_loss += loss_nce
                logs["Loss/InfoNCE"] = loss_nce.item()
    
                # Supervised Soft-Argmax Flow Loss
                if self.use_flow_loss:
                    pred_x_f2m, pred_y_f2m = self.compute_soft_argmax_flow(feats_q, desc_moving, H, W, temp=info_nce_temp)
                
                    w_flow = self.cfg.get("w_flow", 0.1)
                    loss_flow = F.mse_loss(pred_x_f2m, target_x_float) + F.mse_loss(pred_y_f2m, target_y_float)
                    
                    total_loss += w_flow * loss_flow
                    logs["Loss/Flow_MSE"] = loss_flow.item()
       
    
                    with torch.no_grad():
                        avg_dist = (torch.abs(pred_x_f2m - target_x_float) + torch.abs(pred_y_f2m - target_y_float)).mean()
                        logs["Metrics/Train_Soft_Argmax_Err_Px"] = avg_dist.item()
    
                    # ==========================================
                    # 4. Cycle Consistency (Topology Regularization)
                    # ==========================================
                    # w_cyc = self.cfg.get("w_cyc", 0.1)
                    # if w_cyc > 0:
                    cycle_warmup_steps = 5000
                    if self.cfg.get("w_cyc", 0.0) > 0 and step > cycle_warmup_steps:
                        # A. Sample features at the PREDICTED locations
                        # Note: We use the sample_at_coords helper I gave you earlier
                        feats_from_moving = self.sample_at_coords(desc_moving, pred_x_f2m, pred_y_f2m, H, W)
                        
                        # B. Backward Prediction
                        pred_x_m2f, pred_y_m2f = self.compute_soft_argmax_flow(feats_from_moving, desc_fixed, H, W, temp=info_nce_temp)
                        
                        # C. Cycle Constraint (Should return to start)
                        loss_cyc = F.mse_loss(pred_x_m2f, xs.float()) + F.mse_loss(pred_y_m2f, ys.float())
                        
                        total_loss += w_cyc * loss_cyc
                        logs["Loss/Cycle"] = loss_cyc.item()
                
                # --- Calc Correspondence Error for Viz ---
                # with torch.no_grad():
                flat_mov = desc_moving.view(24, -1)
                sim = torch.matmul(feats_q.detach(), flat_mov)
                match_idx = sim.argmax(dim=1)
    
                match_ys = match_idx // W
                match_xs = match_idx % W
                
                err_y = (match_ys - ys_tgt).float().abs().mean()
                err_x = (match_xs - xs_tgt).float().abs().mean()
                corr_err = (err_y + err_x) / 2.0
                logs["Metrics/Corr_Error_Px"] = corr_err.item()
                
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

        print(f"[Log] Logging {len(samples)} fixed augmentations to WandB...")
        # Limit to 8 samples to keep the log clean
        num_show = min(len(samples), 8)
        
        # Create a figure: Top row = Moving (Input), Bottom row = Warped (GT)
        fig, axes = plt.subplots(2, num_show, figsize=(3 * num_show, 6))
        
        # Handle edge case if num_show=1
        if num_show == 1: 
            axes = axes.reshape(2, 1)

        for i in range(num_show):
            sample = samples[i]
            # Squeeze to [H, W] and move to CPU
            moving = sample["moving"].squeeze().cpu().float()
            warped = sample["warped_clean"].squeeze().cpu().float()
            
            # Row 0: The Input Image (Sparse/Noisy)
            axes[0, i].imshow(moving, cmap='gray', vmin=0, vmax=1)
            axes[0, i].axis('off')
            axes[0, i].set_title(f"Input {i}")
            
            # Row 1: The Target Geometry (Clean)
            axes[1, i].imshow(warped, cmap='gray', vmin=0, vmax=1)
            axes[1, i].axis('off')
            axes[1, i].set_title(f"GT Warped {i}")

        plt.suptitle(f"Fixed Cache Preview (Total: {len(samples)})")
        plt.tight_layout()
        
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
        
        pred_flow_map_x = torch.zeros(H, W)
        pred_flow_map_y = torch.zeros(H, W)
        gt_flow_map_x = torch.zeros(H, W)
        gt_flow_map_y = torch.zeros(H, W)
        error_map_x = torch.zeros(H, W)
        error_map_y = torch.zeros(H, W)
        
        # chunk_size = 4096 
        chunk_size = self.cfg["chunk_size"]
        num_pixels = H * W

        for i in range(0, num_pixels, chunk_size):
            end = min(i + chunk_size, num_pixels)
            q_chunk = flat_fixed[i:end]
            
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
            
            # Flow = Moving - Fixed
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

    @torch.no_grad()
    def get_viz_data(self, step):
        if not self.latest_viz_data: return None

        data = self.latest_viz_data
        
        # 2. Compute Dense Flow (moved logic here)
        if self.use_corr:
            pred_fx, pred_fy, gt_fx, gt_fy, err_x, err_y = self.compute_dense_error_map(
                data["fixed"], data["moving"], data["gt_flow"], data["aug_type"]
            )
            # Add results to data dict
            data.update({
                "pred_flow": (pred_fx, pred_fy),
                "gt_flow_map": (gt_fx, gt_fy),
                "error_map": (err_x, err_y)
            })
            
        return data

        
def run_experiment(exp_config, update_dict=None):
    set_seed(42)

    # 1. Setup WandB
    notes_str = str(update_dict) if update_dict else "No specific updates"

    aug_str = exp_config.get('aug_mode', 'rigid')
    run_name = f"Consensus_{exp_config['res']}_{aug_str}_views{exp_config['views']}"
    
    if exp_config["wandb"]:
        wandb.init(
            project="Single-Image-Optimization", 
            config=exp_config, 
            name=run_name, 
            notes=notes_str, 
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
                desc += f" | Err: {logs['Metrics/Corr_Error_Px']:.2f}px"
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
    # "elastic_alpha": 0.05,     
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
    "use_flow_loss": False, # Enable supervised flow MSE loss
    "w_align": 1.0, # Weight for alignment (InfoNCE)
    "w_naive": 0.0, # Weight for naive MSE (if not using correspondence)
    "w_tv": 0.1, # Total Variation regularization weight
    "w_cyc": 0.0, # Cycle Consistency weight
    "w_flow": 0.1, # Flow Loss weight (if use_flow_loss=True)

    # --- Performance & Batching ---
    "num_samples": 4096, # Number of pixels sampled for InfoNCE
    "chunk_size": 4096, # Batch size for dense evaluation (inference)
}

experiment_updates = [
    # --- Experiment 1: Elastic Aug, No Simulation, With Communicator ---
    {
        "aug_mode": "elastic", 
        "epochs": 50001, 
        "use_sparse_view_simulation": False, 
        "use_communicator": False, 
        "communicator_depth": 2, 
        "communicator_num_heads": 8,
    },
]

print(f"Total experiments to run: {len(experiment_updates)}")
for i, update in enumerate(experiment_updates):
    exp_cfg = copy.deepcopy(common_config)
    exp_cfg.update(update) 
    
    print(f"\n--- Running Experiment {i+1}/{len(experiment_updates)} ---")
    run_experiment(exp_cfg, update_dict=update)