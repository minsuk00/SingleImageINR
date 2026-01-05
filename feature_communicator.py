# mast3r_arch.py
import torch
import torch.nn as nn
from functools import partial
import collections.abc
from itertools import repeat

# --- Helpers ---
def _ntuple(n):
    def parse(x):
        if isinstance(x, collections.abc.Iterable) and not isinstance(x, str):
            return x
        return tuple(repeat(x, n))
    return parse
to_2tuple = _ntuple(2)

def drop_path(x, drop_prob: float = 0., training: bool = False, scale_by_keep: bool = True):
    if drop_prob == 0. or not training:
        return x
    keep_prob = 1 - drop_prob
    shape = (x.shape[0],) + (1,) * (x.ndim - 1)
    random_tensor = x.new_empty(shape).bernoulli_(keep_prob)
    if keep_prob > 0.0 and scale_by_keep:
        random_tensor.div_(keep_prob)
    return x * random_tensor

class DropPath(nn.Module):
    def __init__(self, drop_prob: float = 0., scale_by_keep: bool = True):
        super(DropPath, self).__init__()
        self.drop_prob = drop_prob
        self.scale_by_keep = scale_by_keep
    def forward(self, x):
        return drop_path(x, self.drop_prob, self.training, self.scale_by_keep)

# --- Layers ---
class Mlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, bias=True, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        bias = to_2tuple(bias)
        drop_probs = to_2tuple(drop)
        self.fc1 = nn.Linear(in_features, hidden_features, bias=bias[0])
        self.act = act_layer()
        self.drop1 = nn.Dropout(drop_probs[0])
        self.fc2 = nn.Linear(hidden_features, out_features, bias=bias[1])
        self.drop2 = nn.Dropout(drop_probs[1])

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop1(x)
        x = self.fc2(x)
        x = self.drop2(x)
        return x

class Attention(nn.Module):
    def __init__(self, dim, num_heads=8, qkv_bias=False, attn_drop=0., proj_drop=0.):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = head_dim ** -0.5
        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x):
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).transpose(1,3)
        q, k, v = [qkv[:,:,i] for i in range(3)]
        
        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x

class CrossAttention(nn.Module):
    def __init__(self, dim, num_heads=8, qkv_bias=False, attn_drop=0., proj_drop=0.):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = head_dim ** -0.5

        self.projq = nn.Linear(dim, dim, bias=qkv_bias)
        self.projk = nn.Linear(dim, dim, bias=qkv_bias)
        self.projv = nn.Linear(dim, dim, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, query, key, value):
        B, Nq, C = query.shape
        Nk = key.shape[1]
        Nv = value.shape[1]
        
        q = self.projq(query).reshape(B,Nq,self.num_heads, C// self.num_heads).permute(0, 2, 1, 3)
        k = self.projk(key).reshape(B,Nk,self.num_heads, C// self.num_heads).permute(0, 2, 1, 3)
        v = self.projv(value).reshape(B,Nv,self.num_heads, C// self.num_heads).permute(0, 2, 1, 3)
        
        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, Nq, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x

class DecoderBlock(nn.Module):
    def __init__(self, dim, num_heads, mlp_ratio=4., qkv_bias=False, drop=0., attn_drop=0.,
                 drop_path=0., act_layer=nn.GELU, norm_layer=nn.LayerNorm, norm_mem=True):
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.attn = Attention(dim, num_heads=num_heads, qkv_bias=qkv_bias, attn_drop=attn_drop, proj_drop=drop)
        self.cross_attn = CrossAttention(dim, num_heads=num_heads, qkv_bias=qkv_bias, attn_drop=attn_drop, proj_drop=drop)
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        self.norm3 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)
        self.norm_y = norm_layer(dim) if norm_mem else nn.Identity()

    def forward(self, x, y):
        # 1. Self Attention
        x = x + self.drop_path(self.attn(self.norm1(x)))
        # 2. Cross Attention (x attends to y)
        y_ = self.norm_y(y)
        x = x + self.drop_path(self.cross_attn(self.norm2(x), y_, y_))
        # 3. MLP
        x = x + self.drop_path(self.mlp(self.norm3(x)))
        return x

# --- Main Communicator Class ---
class FeatureCommunicator(nn.Module):
    """
    Implements the asymmetric communication between two images (Ref and Mov).
    Matches ViT-Base specifications: 768 dim, 12 heads, 12 layers.
    """
    def __init__(self, input_dim=384, embed_dim=768, depth=12, num_heads=12, mlp_ratio=4):
        super().__init__()
        print(f"NEW: Initializing FeatureCommunicator (ViT-Base Sized: dim={embed_dim}, depth={depth})")
        
        # 1. Projection from DINO dimension (384) to Decoder dimension (768)
        self.enc_to_dec = nn.Linear(input_dim, embed_dim)
        
        # 2. Absolute Positional Embeddings (Standard ViT style)
        # We assume max 256 patches (16x16) for now, but making it larger to be safe (e.g. 1024 for 32x32)
        self.pos_embed = nn.Parameter(torch.zeros(1, 1024, embed_dim)) 
        nn.init.trunc_normal_(self.pos_embed, std=0.02)
        
        norm_layer = partial(nn.LayerNorm, eps=1e-6)
        
        # 3. Two separate decoder stacks (Asymmetric)
        # Stack 1: Updates Reference (using Moving as context)
        self.blocks_ref = nn.ModuleList([
            DecoderBlock(embed_dim, num_heads, mlp_ratio, qkv_bias=True, norm_layer=norm_layer)
            for _ in range(depth)
        ])
        
        # Stack 2: Updates Moving (using Reference as context)
        self.blocks_mov = nn.ModuleList([
            DecoderBlock(embed_dim, num_heads, mlp_ratio, qkv_bias=True, norm_layer=norm_layer)
            for _ in range(depth)
        ])

        self.norm_ref = norm_layer(embed_dim)
        self.norm_mov = norm_layer(embed_dim)

    def forward(self, x_ref, x_mov):
        """
        x_ref: [B, N, 384]
        x_mov: [B, N, 384]
        """
        B, N, C = x_ref.shape
        
        # Project DINO -> Decoder Dim
        f_ref = self.enc_to_dec(x_ref) # [B, N, 768]
        f_mov = self.enc_to_dec(x_mov) # [B, N, 768]
        
        # Add Positional Embeddings
        # Slice pos_embed to current sequence length N
        pos = self.pos_embed[:, :N, :]
        f_ref = f_ref + pos
        f_mov = f_mov + pos
        
        # Communication Loop
        # We update them in parallel layers, swapping context
        curr_ref = f_ref
        curr_mov = f_mov
        
        for blk_ref, blk_mov in zip(self.blocks_ref, self.blocks_mov):
            # Ref attends to Mov
            new_ref = blk_ref(curr_ref, curr_mov)
            
            # Mov attends to Ref
            new_mov = blk_mov(curr_mov, curr_ref)
            
            curr_ref = new_ref
            curr_mov = new_mov
            
        return self.norm_ref(curr_ref), self.norm_mov(curr_mov)