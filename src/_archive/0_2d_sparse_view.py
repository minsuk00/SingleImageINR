# on multiple views at once?

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

# NOTE: External file dependency
from feature_communicator import FeatureCommunicator

def set_seed(seed=42):
    torch.manual_seed(seed)
    np.random.seed(seed)
    torch.backends.cudnn.deterministic = True

set_seed(42)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Running on device: {device}")

# ==========================================
# 2. CT Simulation & Augmentation (The Gauntlet)
# ==========================================

class AugmentationManager:
    def __init__(self, device="cpu", elastic_alpha=0.1, H=256, W=256):
        self.device = device
        self.elastic_alpha = elastic_alpha
        self.H = H
        self.W = W

    def get_elastic_transform(self):
        """Generates a random elastic deformation field."""
        # NOTE: Reduced grid size for smoother, more anatomical deformations
        grid_h, grid_w = 6, 6 
        # Random offsets
        coarse_flow = (torch.rand(1, 2, grid_h, grid_w, device=self.device) - 0.5) * 2 * self.elastic_alpha
        # Upsample to image size
        flow = F.interpolate(coarse_flow, size=(self.H, self.W), mode="bicubic", align_corners=False)
        flow = flow.permute(0, 2, 3, 1) # [1, H, W, 2]
        return flow

    def apply_warp(self, img_tensor, flow):
        """
        img_tensor: [B, C, H, W]
        flow: [1, H, W, 2] (Pixels relative to grid, but we need normalized for grid_sample)
        """
        B, C, H, W = img_tensor.shape
        # Create Identity Grid
        yy, xx = torch.meshgrid(torch.linspace(-1, 1, H, device=self.device), torch.linspace(-1, 1, W, device=self.device), indexing="ij")
        base_grid = torch.stack([xx, yy], dim=-1).unsqueeze(0) # [1, H, W, 2]
        
        # NOTE: The flow coming in is likely in "normalized grid units" if it came from get_elastic_transform?
        # Actually, get_elastic_transform output is directly added/subtracted.
        # Let's verify: coarse_flow is magnitude ~alpha. alpha=0.1 means 10% of image dimension.
        # grid_sample expects coordinates in [-1, 1].
        # So if flow is 0.1, that is a significant shift.
        
        # Target = Clean(Grid - Flow)  <-- "Pull" formulation
        final_grid = base_grid - flow
        
        warped = F.grid_sample(img_tensor, final_grid, align_corners=False, padding_mode="border")
        return warped

def add_poisson_noise(sinogram, I0=10000):
    """Simulates photon statistics (Poisson noise)."""
    if I0 is None:
        return sinogram

    # Beer-Lambert: I = I0 * exp(-attenuation)
    intensity = I0 * np.exp(-sinogram)

    # Poisson Noise
    noisy_intensity = np.random.poisson(intensity)

    # Prevent dead pixels
    noisy_intensity[noisy_intensity == 0] = 1

    # Recover Attenuation
    noisy_sinogram = -np.log(noisy_intensity / I0)
    return noisy_sinogram

def simulate_sparse_ct(image_tensor, num_views, noise_intensity=None):
    """
    Full Forward Model: Image -> Radon -> Noise -> FBP -> Image
    NOTE: Using skimage (CPU based). Might be slow for large batches.
    """
    device = image_tensor.device
    img_np = image_tensor.squeeze().cpu().numpy()
    
    # Handle batch dimension if present, otherwise squeeze
    if len(img_np.shape) > 2:
        img_np = img_np[0] # Take first channel/batch for simplicity if needed
        
    H, W = img_np.shape

    # 1. Angles
    theta = np.linspace(0.0, 180.0, num_views, endpoint=False)

    # 2. Radon
    sinogram = radon(img_np, theta=theta, circle=True)

    # Scale for physics realism (Max attenuation ~4.0)
    PHYSICAL_MAX_ATTENUATION = 4.0
    current_max = sinogram.max()
    scale_factor = PHYSICAL_MAX_ATTENUATION / max(current_max, 1e-6)

    # 3. Noise
    if noise_intensity is not None:
        sinogram_physical = sinogram * scale_factor
        sinogram_noisy = add_poisson_noise(sinogram_physical, I0=noise_intensity)
        sinogram = sinogram_noisy / scale_factor

    # 4. FBP (Back Projection)
    # Using 'ramp' filter (standard FBP)
    reconstruction = iradon(sinogram, theta=theta, filter_name="ramp", circle=True, output_size=max(H, W))

    # 5. Clip
    reconstruction = np.clip(reconstruction, 0, 1)

    recon_tensor = torch.from_numpy(reconstruction).float().unsqueeze(0).unsqueeze(0).to(device)
    return recon_tensor

class BatchGenerator:
    """
    Generates the 'Consensus Stack' on the fly.
    """
    def __init__(self, clean_img_tensor, config, device):
        self.clean = clean_img_tensor
        self.cfg = config
        self.device = device
        self.augmentor = AugmentationManager(device=device, elastic_alpha=config['elastic_alpha'], H=config['res'], W=config['res'])
        
    def generate_batch(self, batch_size=2):
        """
        Returns:
            anchor_img: [1, 1, H, W] (Noisy, No Warp)
            target_imgs: [B, 1, H, W] (Noisy, Warped)
            gt_flows: [B, H, W, 2] (The flow used to warp)
        """
        # 1. Create Anchor (Index 0 logic)
        # No warp, just noise + sparse view
        anchor_noisy = simulate_sparse_ct(self.clean, self.cfg['views'], self.cfg['noise_i0'])
        
        targets = []
        flows = []
        
        # 2. Create Targets
        for _ in range(batch_size):
            # A. Get Flow
            flow = self.augmentor.get_elastic_transform() # [1, H, W, 2]
            
            # B. Warp the Clean Image
            warped_clean = self.augmentor.apply_warp(self.clean, flow)
            
            # C. Corrupt the Warped Image (Noise + Streaks)
            # NOTE: CT artifacts rotate with the object if the object moves, 
            # but streaks from sparse views are fixed to the detector geometry. 
            # For simplicity here, we simulate "scan -> move -> scan", so artifacts are newly generated.
            warped_noisy = simulate_sparse_ct(warped_clean, self.cfg['views'], self.cfg['noise_i0'])
            
            targets.append(warped_noisy)
            flows.append(flow)
            
        targets = torch.cat(targets, dim=0) # [B, 1, H, W]
        flows = torch.cat(flows, dim=0)     # [B, H, W, 2]
        
        return anchor_noisy, targets, flows

# ==========================================
# 3. Models (DINO + Communicator)
# ==========================================

# (Same MatchingHead as before)
class MatchingHead(nn.Module):
    def __init__(self, input_dim=384, output_dim=24, patch_size=16):
        super().__init__()
        self.patch_size = patch_size
        self.mlp = nn.Sequential(
            nn.Linear(input_dim, 512),
            nn.GELU(),
            nn.Linear(512, output_dim * (patch_size ** 2))
        )

    def forward(self, x, H, W):
        B, N, C = x.shape
        x = self.mlp(x) 
        H_grid, W_grid = H // self.patch_size, W // self.patch_size
        x = x.transpose(1, 2).reshape(B, -1, H_grid, W_grid)
        x = F.pixel_shuffle(x, self.patch_size)
        return F.normalize(x, dim=1) 

# ==========================================
# 4. Consensus Trainer
# ==========================================
class ConsensusTrainer:
    def __init__(self, device, config, clean_image):
        self.device = device
        self.cfg = config
        self.clean_gt = clean_image # Only for validation, not training
        
        # 1. Batch Generator
        self.generator = BatchGenerator(clean_image, config, device)
        
        # 2. Models
        print("Loading DINO...")
        if config["encoder"] == "dinov3-vits":
            self.dino = AutoModel.from_pretrained('facebook/dinov3-vits16-pretrain-lvd1689m').to(device).eval()
            dim_enc = 384
        else:
            # Fallback or VitB
            self.dino = AutoModel.from_pretrained('facebook/dinov3-vitb16-pretrain-lvd1689m').to(device).eval()
            dim_enc = 768
            
        for p in self.dino.parameters(): p.requires_grad = False
        
        dim_dec = 768
        dim_total = dim_enc + dim_dec
        
        self.communicator = FeatureCommunicator(
            input_dim=dim_enc, 
            embed_dim=dim_dec, 
            grid_size=(config['res']//16, config['res']//16),
            depth=2, 
            num_heads=8
        ).to(device)
        
        self.head_ref = MatchingHead(input_dim=dim_total).to(device)
        self.head_mov = MatchingHead(input_dim=dim_total).to(device)
        
        self.optimizer = torch.optim.Adam([
            {'params': self.communicator.parameters(), 'lr': config["lr"]}, 
            {'params': self.head_ref.parameters(), 'lr': config["lr"]},
            {'params': self.head_mov.parameters(), 'lr': config["lr"]}
        ])
        
        self.ce_loss = nn.CrossEntropyLoss()
        self.mean = torch.tensor([0.485, 0.456, 0.406], device=device).view(1, 3, 1, 1)
        self.std = torch.tensor([0.229, 0.224, 0.225], device=device).view(1, 3, 1, 1)
        
        self.latest_viz_data = {}

    def get_backbone_feats(self, img):
        B, C, H, W = img.shape
        img_3ch = img.repeat(1, 3, 1, 1)
        img_norm = (img_3ch - self.mean) / self.std
        
        H_grid = H // 16; W_grid = W // 16
        num_patches = H_grid * W_grid
        
        with torch.no_grad():
            out = self.dino(pixel_values=img_norm, output_hidden_states=True)
            patch_feats = out.last_hidden_state[:, -num_patches:, :] # [B, N, 384]
        return patch_feats

    def train_step(self, step):
        self.optimizer.zero_grad()
        
        # 1. Generate Batch (The Gauntlet)
        # We only need 1 target per step for InfoNCE training, 
        # but we can generate more if we want to simulate batch alignment.
        # Let's use 1 target for efficiency in the training loop.
        anchor_img, target_imgs, gt_flows = self.generator.generate_batch(batch_size=1)
        target_img = target_imgs[0:1] # [1, 1, H, W]
        gt_flow = gt_flows[0:1]       # [1, H, W, 2] (Normalized scale)
        
        H, W = anchor_img.shape[-2:]
        
        # 2. Extract Features
        enc_anc = self.get_backbone_feats(anchor_img)
        enc_tar = self.get_backbone_feats(target_img)
        
        # 3. Communicate
        dec_anc, dec_tar = self.communicator(enc_anc, enc_tar)
        
        # 4. Heads
        feats_anc = torch.cat([enc_anc, dec_anc], dim=-1)
        feats_tar = torch.cat([enc_tar, dec_tar], dim=-1)
        
        desc_anc = self.head_ref(feats_anc, H, W) # [1, 24, H, W]
        desc_tar = self.head_mov(feats_tar, H, W) # [1, 24, H, W]
        
        # 5. InfoNCE Loss
        # We sample points in Anchor, find their GT match in Target, and pull them together.
        num_samples = 4096
        flat_inds = torch.randperm(H * W, device=self.device)[:num_samples]
        ys = flat_inds // W
        xs = flat_inds % W
        
        feats_q = desc_anc[0, :, ys, xs].T # [N, 24]
        
        # Calculate GT corresponding coordinates
        # gt_flow is (1, H, W, 2) in normalized units (because we passed it to grid_sample).
        # We need to convert it to pixel offsets.
        # Wait, AugmentationManager stores 'flow' as normalized offsets?
        # Let's check AugmentationManager.get_elastic_transform again.
        # It creates values ~ alpha (0.1). 
        # In grid_sample: output[y, x] = input[y + flow_y, x + flow_x] (roughly)
        # So the pixel at Anchor(y,x) corresponds to Target(y', x') where?
        # We warped Clean -> Target. Anchor is ~Clean.
        # Target = Warp(Clean).
        # So Target(y, x) comes from Clean(y-fy, x-fx) roughly?
        # NO. grid_sample uses Pull. Target[p] = Clean[p + flow].
        # So Clean[p + flow] moves to Target[p].
        # We want to match Anchor[p + flow] with Target[p].
        
        # Actually, let's look at it:
        # We want to match semantics.
        # The anatomy at Clean[coords] is the same anatomy.
        # In Anchor: Anatomy A is at `coords`.
        # In Target: Target[coords] = Clean[coords + flow].
        # So Anatomy A is at `coords` in Target? NO.
        # Target[coords] pulls from Clean[coords + flow].
        # That means Anatomy B (which was at coords+flow) is now at coords in Target.
        # We want to match Anatomy A.
        # Where is Anatomy A in Target?
        # It's at the position `p'` such that `p' + flow(p') = p`. Inverse problem.
        # This is hard to solve on the fly.
        
        # Alternative Strategy:
        # Sample query from TARGET at `p`.
        # This contains anatomy from Clean[`p + flow`].
        # The matching anatomy in ANCHOR is at `p + flow`.
        # YES. This is easier.
        
        # Query: Target pixel `p` (ys, xs)
        feats_q = desc_tar[0, :, ys, xs].T # [N, 24]
        
        # Key: Anchor pixel `p + flow`
        # flow is [1, H, W, 2]
        flow_sample = gt_flow[0, ys, xs] # [N, 2] (Normalized)
        
        # Convert normalized flow to pixels
        # Flow in grid_sample is [-1, 1]. 1 corresponds to Half Image width.
        flow_py = flow_sample[:, 1] * (H / 2.0)
        flow_px = flow_sample[:, 0] * (W / 2.0)
        
        # Clean coord = Target coord + flow
        # But we added noise. Anchor is geometrically aligned with Clean.
        ys_anc = torch.clamp(ys + flow_py, 0, H-1).long()
        xs_anc = torch.clamp(xs + flow_px, 0, W-1).long()
        
        feats_k_pos = desc_anc[0, :, ys_anc, xs_anc].T # [N, 24]
        
        # Contrastive Loss
        temp = 0.1
        logits = torch.matmul(feats_q, feats_k_pos.T) / temp
        labels = torch.arange(num_samples, device=self.device)
        
        loss = (self.ce_loss(logits, labels) + self.ce_loss(logits.T, labels)) / 2.0
        
        loss.backward()
        self.optimizer.step()
        
        logs = {"Loss": loss.item()}
        
        if step % 500 == 0:
            self.save_viz_data(anchor_img, target_img, desc_anc, desc_tar, step)
            
        return logs

    def save_viz_data(self, anchor, target, desc_anc, desc_tar, step):
        self.latest_viz_data = {
            "anchor": anchor.detach(),
            "target": target.detach(),
            "desc_anc": desc_anc.detach(),
            "desc_tar": desc_tar.detach(),
            "step": step
        }

    def compute_flow_from_features(self, desc_ref, desc_mov):
        """
        Computes dense flow field using feature matching.
        Ref: Anchor, Mov: Target
        We want to unwarp Target to Anchor.
        So for every pixel in Anchor, where is it in Target?
        """
        B, C, H, W = desc_ref.shape
        flat_ref = desc_ref.view(C, -1).T # [HW, C]
        flat_mov = desc_mov.view(C, -1).T # [HW, C]
        
        # Chunked computation to save memory
        chunk_size = 4096
        num_pixels = H * W
        
        flow_x_map = torch.zeros(H, W, device=self.device)
        flow_y_map = torch.zeros(H, W, device=self.device)
        
        for i in range(0, num_pixels, chunk_size):
            end = min(i + chunk_size, num_pixels)
            q_chunk = flat_ref[i:end]
            
            # Dot Product Similarity
            sim = torch.matmul(q_chunk, flat_mov.T) # [Chunk, HW]
            match_idx = sim.argmax(dim=1)
            
            # Coords in Ref (Anchor)
            indices = torch.arange(i, end, device=self.device)
            y_ref = indices // W
            x_ref = indices % W
            
            # Coords in Mov (Target)
            y_mov = match_idx // W
            x_mov = match_idx % W
            
            # Flow: We want to pull from Mov to Ref.
            # Grid_sample(img, grid). 
            # If we want to reconstruct Ref using Mov pixels:
            # Recon[y, x] = Mov[y + dy, x + dx]
            # So flow = Mov_coords - Ref_coords
            dy = y_mov - y_ref
            dx = x_mov - x_ref
            
            flow_y_map.view(-1)[i:end] = dy.float()
            flow_x_map.view(-1)[i:end] = dx.float()
            
        return torch.stack([flow_x_map, flow_y_map], dim=-1) # [H, W, 2]

    def visualize_consensus(self):
        if not self.latest_viz_data: return
        
        anchor = self.latest_viz_data["anchor"]
        target = self.latest_viz_data["target"]
        d_anc = self.latest_viz_data["desc_anc"]
        d_tar = self.latest_viz_data["desc_tar"]
        step = self.latest_viz_data["step"]
        
        H, W = anchor.shape[-2:]
        
        # 1. Infer Flow
        # [1, H, W, 2] (Pixels)
        pred_flow_px = self.compute_flow_from_features(d_anc, d_tar).unsqueeze(0)
        
        # 2. Convert Flow to Grid Sample Coords
        # grid_sample(x) gets value at x.
        # We found that Anchor(y,x) matches Target(y+dy, x+dx).
        # So we want to put Target(y+dy, x+dx) at (y,x).
        # Grid should be Identity + Flow.
        
        # Normalize flow for grid_sample
        # -1 to 1 map.
        norm_flow_x = pred_flow_px[..., 0] / (W / 2.0)
        norm_flow_y = pred_flow_px[..., 1] / (H / 2.0)
        norm_flow = torch.stack([norm_flow_x, norm_flow_y], dim=-1)
        
        # Base Grid
        yy, xx = torch.meshgrid(torch.linspace(-1, 1, H, device=self.device), torch.linspace(-1, 1, W, device=self.device), indexing="ij")
        base_grid = torch.stack([xx, yy], dim=-1).unsqueeze(0)
        
        # Sample coordinates
        sample_grid = base_grid + norm_flow
        
        # 3. Unwarp Target
        unwarped_target = F.grid_sample(target, sample_grid, align_corners=False, padding_mode="border")
        
        # 4. Consensus (Median)
        # Stack Anchor and Unwarped Target
        stack = torch.cat([anchor, unwarped_target], dim=0)
        median_img = torch.median(stack, dim=0).values
        
        # Plot
        fig, ax = plt.subplots(1, 5, figsize=(20, 5))
        ax[0].imshow(self.clean_gt.squeeze().cpu(), cmap='gray')
        ax[0].set_title("GT Clean (Ref)")
        
        ax[1].imshow(anchor.squeeze().cpu(), cmap='gray')
        ax[1].set_title("Noisy Anchor")
        
        ax[2].imshow(target.squeeze().cpu(), cmap='gray')
        ax[2].set_title("Noisy Target (Warped)")
        
        ax[3].imshow(unwarped_target.squeeze().cpu(), cmap='gray')
        ax[3].set_title("Unwarped Target")
        
        ax[4].imshow(median_img.squeeze().cpu(), cmap='gray')
        ax[4].set_title(f"Median Consensus\n(Step {step})")
        
        if self.cfg.get("wandb"):
            wandb.log({"Consensus/Viz": wandb.Image(fig)}, step=step)
            
        plt.show()

# ==========================================
# 5. Main Execution
# ==========================================

def load_image_dynamic(base_path, res):
    """Loads a PNG image for testing."""
    path = f"{base_path}/ct_chest_2_resized_{res}x{res}.png"
    try:
        img = Image.open(path).convert('L')
    except Exception as e:
        print(f"Error loading image at {path}: {e}")
        return None
    img = img.resize((res, res)) 
    img_tensor = torch.from_numpy(np.array(img)).float() / 255.0
    return img_tensor.unsqueeze(0).unsqueeze(0).to(device)

def run_experiment(exp_config):
    # Setup
    run_name = f"Consensus_{exp_config['encoder']}_views{exp_config['views']}"
    if exp_config["wandb"]:
        wandb.init(project="Semantic-Consensus", config=exp_config, name=run_name, reinit=True)
        
    base_path = "/home/minsukc/SIO/data" 
    clean_image = load_image_dynamic(base_path, exp_config['res'])
    if clean_image is None: return

    trainer = ConsensusTrainer(device, exp_config, clean_image)
    
    print(f"--- Starting Training: {run_name} ---")
    pbar = tqdm(range(exp_config["epochs"]), desc="Training")
    for step in pbar:
        logs = trainer.train_step(step)
        if step % 100 == 0:
            pbar.set_postfix(logs)
        if step % 500 == 0:
            trainer.visualize_consensus()
            
    if exp_config["wandb"]: wandb.finish()

# Config
config = {
    "wandb": True,
    "epochs": 5001,
    "lr": 5e-4,
    "res": 256,
    "encoder": "dinov3-vits", # vits or vitb
    "elastic_alpha": 0.05,
    "views": 30, # Sparse view
    "noise_i0": 5000 # Photon count (lower = noisier)
}

if __name__ == "__main__":
    run_experiment(config)