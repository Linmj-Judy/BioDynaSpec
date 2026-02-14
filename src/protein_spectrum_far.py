"""
Protein Spectrum FAR (Frequency Autoregressive Model)
Adapted from FAR for protein motion spectrum prediction
支持变长频谱输入的改进版本

Key adaptations:
1. Input: spectrum representations (num_residues, 3, T//2+1, 64) instead of image tokens
2. Conditions: protein structure (coords), sequence (aatype), and ESM features
3. Autoregression: over frequency dimension (T//2+1) from low to high frequency
4. Output: continuous spectrum representations instead of discrete tokens


改进点：
1. 动态位置编码生成（支持任意长度）
2. 相对位置编码 + 可学习位置编码
3. 频率归一化(将频率索引归一化到[0,1])
4. 位置插值外推（对超出训练长度的序列）
5. 可选RoPE支持

"""

from functools import partial
import numpy as np
from tqdm import tqdm
import scipy.stats as stats
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.checkpoint as ckpt
from openfold.model.primitives import Linear, LayerNorm
from openfold.utils.rigid_utils import Rigid, Rotation
from openfold.model.structure_module import InvariantPointAttention

# from timm.models.vision_transformer import Block
from typing import Optional, Tuple

from featureextractor import FeatureExtractor
from model.diffloss import SpectrumDiffLoss
from model.diffusion.denoise_net import SpectralDiTBlock
from model.modules.transformer import DiffusionTransformer
from model.modules.primitives import Attention as OpenFoldAttention
# from model.modules.olmo_transformer import OLMoBlockGroup


# ============================================================================
# FlashAttention availability
# ============================================================================
try:
    from flash_attn import flash_attn_func, flash_attn_qkvpacked_func  # noqa: F401
    try:
        from flash_attn import flash_attn_with_kvcache  # noqa: F401
        _flash_attn_with_kvcache_available = True
    except Exception:
        _flash_attn_with_kvcache_available = False
    _flash_attn_available = True
except Exception as e:
    _flash_attn_available = False
    _flash_attn_with_kvcache_available = False
    print(f"Flash Attention is not available!", e)

# ============================================================================
# RoPE (Rotary Position Embedding) Functions
# ============================================================================

def precompute_freqs_cis(dim: int, end: int, theta: float = 10000.0, device='cuda'):
    """
    Precompute the frequency tensor for complex exponentials (cis) with given dimensions.
    
    Args:
        dim (int): Dimension of the frequency tensor (should be even)
        end (int): Maximum sequence length
        theta (float): Scaling factor for frequency computation
        device: Device to create tensor on
    
    Returns:
        torch.Tensor: Precomputed frequency tensor with complex exponentials [end, dim//2]
    """
    freqs = 1.0 / (theta ** (torch.arange(0, dim, 2, device=device)[: (dim // 2)].float() / dim))
    t = torch.arange(end, device=device, dtype=torch.float32)
    freqs = torch.outer(t, freqs)  # [end, dim//2]
    freqs_cis = torch.polar(torch.ones_like(freqs), freqs)  # complex64
    return freqs_cis


def apply_rotary_emb(
    x: torch.Tensor,
    freqs_cis: torch.Tensor,
) -> torch.Tensor:
    """
    Apply rotary embeddings to input tensor.
    
    Args:
        x (torch.Tensor): Input tensor of shape [..., seq_len, dim]
        freqs_cis (torch.Tensor): Precomputed frequency tensor [seq_len, dim//2]
    
    Returns:
        torch.Tensor: Tensor with rotary embeddings applied
    """
    # Reshape x to [..., seq_len, dim//2, 2] and view as complex
    x_shape = x.shape
    x_ = x.float().reshape(*x_shape[:-1], -1, 2)  # [..., seq_len, dim//2, 2]
    x_complex = torch.view_as_complex(x_)  # [..., seq_len, dim//2]
    
    # Reshape freqs_cis for broadcasting
    # freqs_cis: [seq_len, dim//2] -> need to broadcast to [..., seq_len, dim//2]
    freqs_cis = freqs_cis.view(*([1] * (x_complex.ndim - 2)), *freqs_cis.shape)
    
    # Apply rotation
    x_out = torch.view_as_real(x_complex * freqs_cis)  # [..., seq_len, dim//2, 2]
    x_out = x_out.flatten(-2)  # [..., seq_len, dim]
    
    return x_out.type_as(x)
# ============================================================================


class Mlp(nn.Module):
    def __init__(self, in_features: int, hidden_features: Optional[int] = None, out_features: Optional[int] = None,
                 act_layer: nn.Module = nn.GELU, drop: float = 0.0):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


class Attention(nn.Module):
    def __init__(self, dim: int, num_heads: int = 8, qkv_bias: bool = True,
                 attn_drop: float = 0.0, proj_drop: float = 0.0):
        super().__init__()
        assert dim % num_heads == 0, "dim must be divisible by num_heads"
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x: torch.Tensor, *, kv_cache: dict | None = None, causal: bool = False) -> torch.Tensor:
        """
        No batch size:
        x: (N, C)
        Output: (N, C)
        kv_cache: optional dict with keys {"k_cache", "v_cache", "cache_seqlens", "cache_batch_idx"}
        causal: whether to apply causal masking
        """
        N, C = x.shape
        # B=1 for no-batch mode
        x_1b = x.unsqueeze(0)  # (1, N, C)
        qkv_linear = self.qkv(x_1b)  # (1, N, 3*C)

        if _flash_attn_available:
            orig_dtype = qkv_linear.dtype
            if orig_dtype not in (torch.bfloat16, torch.float16):
                print(f"Flash Attention QKV linear dtype must be bfloat16 or float16, but got {orig_dtype}, turning to bfloat16")
                qkv_bf16 = qkv_linear.to(torch.bfloat16)
                # (1, N, 3, H, Hd)
                qkv = qkv_bf16.reshape(1, N, 3, self.num_heads, self.head_dim)
            else:
                qkv = qkv_linear.reshape(1, N, 3, self.num_heads, self.head_dim)
            
            out = None

            if kv_cache is not None and _flash_attn_with_kvcache_available:
                # KV cache path
                q = qkv[:, :, 0]  # (1, N, H, Hd)
                k_new = qkv[:, :, 1]
                v_new = qkv[:, :, 2]
                k_cache = kv_cache.get("k_cache")
                v_cache = kv_cache.get("v_cache")
                cache_seqlens = kv_cache.get("cache_seqlens")
                cache_batch_idx = kv_cache.get("cache_batch_idx")
                block_table = kv_cache.get("block_table")

                out = flash_attn_with_kvcache(
                    q=q,
                    k_cache=k_cache,
                    v_cache=v_cache,
                    k=k_new,
                    v=v_new,
                    cache_seqlens=cache_seqlens,
                    cache_batch_idx=cache_batch_idx,
                    block_table=block_table,
                    softmax_scale=self.scale,
                    causal=causal,
                )  # (1, N, H, Hd)
            else:
                # qkv packed path
                out = flash_attn_qkvpacked_func(
                    qkv,
                    dropout_p=0.0,
                    softmax_scale=self.scale,
                    causal=causal,
                )# (1, N, H, Hd)
            if orig_dtype not in (torch.bfloat16, torch.float16):
                x_attn = out.to(orig_dtype).reshape(1, N, C)
            else:
                x_attn = out.reshape(1, N, C)
        else:
            # Fallback to standard attention
            # do not fallback
            raise Exception("Flash Attention is not available!")
            # qkv = qkv_linear.reshape(1, N, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4).contiguous()
            # q, v = qkv[0], qkv[2]  # (1, H, N, Hd)
            # k = qkv[1]
            # attn = (q @ k.transpose(-2, -1).contiguous()) * self.scale  # (1, H, N, N)
            # if causal:
            #     # Causal mask for sequence length N
            #     causal_mask = torch.triu(torch.ones(N, N, device=attn.device, dtype=torch.bool), diagonal=1)
            #     attn = attn.masked_fill(causal_mask.unsqueeze(0).unsqueeze(0), float('-inf'))
            # attn = attn.softmax(dim=-1)
            # attn = self.attn_drop(attn)
            # x_attn = (attn @ v).transpose(1, 2).reshape(1, N, C).contiguous()

        x_proj = self.proj(x_attn)
        x_proj = self.proj_drop(x_proj)
        return x_proj.squeeze(0)

# TODO: 直接找一个现成的
class Block(nn.Module):
    def __init__(self, dim: int, num_heads: int, mlp_ratio: float = 4.0, qkv_bias: bool = True,
                 norm_layer: nn.Module = nn.LayerNorm, proj_drop: float = 0.0, attn_drop: float = 0.0,
                 act_layer: nn.Module = nn.GELU):
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.attn = Attention(dim, num_heads=num_heads, qkv_bias=qkv_bias,
                              attn_drop=attn_drop, proj_drop=proj_drop)
        self.norm2 = norm_layer(dim)
        self.mlp = Mlp(in_features=dim, hidden_features=int(dim * mlp_ratio),
                       act_layer=act_layer, drop=proj_drop)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(self.norm1(x))
        x = x + self.mlp(self.norm2(x))
        if torch.cuda.is_available():   
            torch.cuda.empty_cache()
        return x


class HybridBlock(nn.Module):
    """
    Hybrid Block that combines:
    1. Cross-residue and cross-frequency attention (full attention across all tokens)
    2. Frequency-specific attention using SpectralDiTBlock (attention along frequency dimension)
    
    This allows learning both residue interactions and frequency dependencies separately.
    """
    def __init__(self, dim: int, num_heads: int, mlp_ratio: float = 4.0, qkv_bias: bool = True,
                 norm_layer: nn.Module = nn.LayerNorm, proj_drop: float = 0.0, attn_drop: float = 0.0,
                 act_layer: nn.Module = nn.GELU, 
                 use_spectral_attention: bool = True,
                 spectral_num_heads: int = None):
        super().__init__()
        # Cross-residue and cross-frequency attention (full attention)
        self.norm1 = norm_layer(dim)
        self.attn = Attention(dim, num_heads=num_heads, qkv_bias=qkv_bias,
                              attn_drop=attn_drop, proj_drop=proj_drop)
        self.norm2 = norm_layer(dim)
        self.mlp = Mlp(in_features=dim, hidden_features=int(dim * mlp_ratio),
                       act_layer=act_layer, drop=proj_drop)
        
        # Frequency-specific attention using SpectralDiTBlock
        self.use_spectral_attention = use_spectral_attention
        if use_spectral_attention:
            spectral_num_heads = spectral_num_heads or num_heads
            # Use SpectralDiTBlock directly for frequency-specific attention
            self.spectral_dit_block = SpectralDiTBlock(
                hidden_size=dim,
                num_heads=spectral_num_heads,
                mlp_ratio=mlp_ratio
            )

    def forward(self, x: torch.Tensor, num_residues: int = None, seq_len: int = None, condition: torch.Tensor = None) -> torch.Tensor:
        """
        Forward pass
        
        Args:
            x: (num_residues * seq_len, dim) if flattened, or (num_residues, seq_len, dim) if not
            num_residues: number of residues (used for reshaping if needed)
            seq_len: sequence length (used for reshaping if needed)
            condition: optional condition tensor for SpectralDiTBlock (num_residues, seq_len, dim)
        
        Returns:
            x: same shape as input
        """
        orig_shape = x.shape
        
        # 1. Cross-residue and cross-frequency attention (full attention)
        x = x + self.attn(self.norm1(x))
        x = x + self.mlp(self.norm2(x))
        
        # 2. Frequency-specific attention using SpectralDiTBlock
        if self.use_spectral_attention:
            # Reshape to (num_residues, seq_len, dim) for frequency attention
            if x.dim() == 2:
                # Flattened: (num_residues * seq_len, dim)
                if num_residues is None or seq_len is None:
                    raise ValueError("num_residues and seq_len must be provided when x is 2D")
                x_reshaped = x.reshape(num_residues, seq_len, -1)
            elif x.dim() == 3:
                # Already (num_residues, seq_len, dim)
                x_reshaped = x
                num_residues, seq_len = x.shape[:2]
            else:
                raise ValueError(f"Unexpected input shape: {x.shape}")
            
            # Prepare condition for SpectralDiTBlock
            if condition is not None:
                # Use provided condition
                c = condition
            else:
                # Use x itself as condition (self-conditioning)
                c = x_reshaped
            
            # Apply SpectralDiTBlock for frequency-specific attention
            # SpectralDiTBlock applies attention along seq_len (frequency) dimension
            x_reshaped = self.spectral_dit_block(x_reshaped, c)
            
            # Reshape back to original format
            if len(orig_shape) == 2:
                x = x_reshaped.reshape(num_residues * seq_len, -1)
            else:
                x = x_reshaped
        
        if torch.cuda.is_available():   
            torch.cuda.empty_cache()
        return x


class IPAModule(nn.Module):
    def __init__(self,
                 c_s:int,
                 c_z:int,
                 c_ipa:int,
                 no_heads_ipa:int,
                 no_qk_points:int,
                 no_v_points:int,
                 inf:float,
                 eps:float,
                 num_ipa_layers:int,
                 dropout_rate:float):
        super().__init__()
        self.c_s = c_s
        self.c_z = c_z
        self.c_ipa = c_ipa
        self.no_heads_ipa = no_heads_ipa
        self.no_qk_points = no_qk_points
        self.no_v_points = no_v_points
        self.num_ipa_layers = num_ipa_layers

        self.layer_norm_s = LayerNorm(c_s)
        self.layer_norm_z = LayerNorm(c_z)
        
        self.ipa_layers = nn.ModuleList([
            InvariantPointAttention(
                c_s=self.c_s,  # updated c_s dimension
                c_z=self.c_z,
                c_hidden=self.c_ipa,
                no_heads=self.no_heads_ipa,
                no_qk_points=self.no_qk_points,
                no_v_points=self.no_v_points,
                inf=inf,
                eps=eps,
            ) for _ in range(num_ipa_layers)
        ])

        self.ipa_dropout = nn.Dropout(dropout_rate)
        
        # Independent LayerNorm for each IPA layer
        self.layer_norms_ipa = nn.ModuleList([
            LayerNorm(c_s) for _ in range(num_ipa_layers)
        ])


    def forward(self, 
                s_s, # (B, N, c_s)
                s_z, # (B, N, c_z)
                coords, # (B, N, 3)
                rotation_matrices, # (..., 3, 3)
                aa_mask, # (B, N)
    ):
        # Normalize
        s_s = self.layer_norm_s(s_s)
        s_z = self.layer_norm_z(s_z)
        aa_mask = aa_mask.to(dtype=s_s.dtype)

        # Construct Rigid object (for IPA module)
        rigid = Rigid(
            Rotation(rot_mats=rotation_matrices),
            coords
        )

        # Through stacked IPA layers
        for ipa, layer_norm in zip(self.ipa_layers, self.layer_norms_ipa):
            s_s = s_s + ipa(s_s, s_z, rigid, aa_mask)
            s_s = self.ipa_dropout(s_s)
            s_s = layer_norm(s_s)
        return s_s


def mask_by_order(mask_len, order, bsz, seq_len):
    """Generate mask based on order and mask length"""
    masking = torch.zeros(bsz, seq_len, device=order.device)
    masking = torch.scatter(masking, dim=-1, index=order[:, :mask_len.long()], 
                           src=torch.ones(bsz, seq_len, device=order.device)).bool()
    return masking


class ProteinSpectrumFAR(nn.Module):
    """
    Protein Spectrum Frequency Autoregressive Model
    
    Performs autoregressive prediction over frequency components of protein motion
    """
    def __init__(self,
                 # Spectrum specifics
                 spectrum_token_dim=1,  # Dimension of each spectrum token
                 num_xyz_channels=3,  # x, y, z channels
                 use_complex_spectrum=True, # Whether to use complex spectrum
                 
                 # Condition specifics
                 aatype_vocab_size=21,
                 esm_dim=1280,
                 with_irfc:bool=True, # Whether to use IRFC
                 
                 # IPA param
                 c_s:int=512,
                 c_z:int=128,
                 c_ipa: int=32,
                 no_heads_ipa: int=8,
                 no_qk_points: int=4,
                 no_v_points: int=4, 
                 num_ipa_layers:int= 1,
                 inf: float = 1e5,
                 eps: float = 1e-8, 
           
                 # Encoder specifics
                 encoder_depth=16,
                 encoder_num_heads=16,
                 
                 # Decoder specifics
                 decoder_embed_dim=1024,
                 decoder_depth=16,
                 decoder_num_heads=16,
                 
                # Training specifics
                 mlp_ratio=4.,
                 norm_layer=nn.LayerNorm,
                 mask=True,
                 mask_ratio_min=0.5,
                 attn_dropout=0.1,
                 proj_dropout=0.1,
                 buffer_size=64,
                 use_checkpoint=False,
                 
                 # RoPE specifics
                 rope_theta=10000.0,  # RoPE frequency scaling
                 
                 # Diffusion loss specifics
                 diffloss_d=3,
                 diffloss_w=1024,
                 num_sampling_steps='100',
                 diffusion_batch_mul=4,
                 # Diffusion core knobs
                 learn_sigma=False,
                 predict_xstart=True,
                 sigma_small=False,
                 noise_schedule='cosine',
                 use_kl=False,
                 rescale_learned_sigmas=False,
                 diffusion_steps=1000,
                 
                 # Conditional drop for CFG
                 condition_drop_prob=0.1,
                 
                 # Frequency-specific attention
                 use_spectral_attention=True,  # Whether to use frequency-specific attention
                 spectral_num_heads=None,  # Number of heads for frequency attention (None = same as encoder_num_heads)
                 use_hybrid_block=False,  # Whether to use HybridBlock instead of regular Block
                 
                 # Spectrum length control
                 max_spectrum_len: Optional[int] = None,
                 
                 # Cross-window attention control
                 enable_cross_window: bool = False,
                 
                 # Denoise net memory control
                 denoise_seq_chunk_size=None,
                 denoise_block_type="residue",
                 denoise_use_checkpoint=False,
                 
                 # Self-forcing training mode
                 use_self_forcing=False,  # Whether to use self-forcing (teacher forcing) during training
                 self_forcing_schedule=None,  # Schedule for self-forcing: None=all steps, or list of step indices
                 ):
        super().__init__()
        
        # --------------------------------------------------------------------------
        # Spectrum specifics
        self.spectrum_token_dim = spectrum_token_dim
        self.num_xyz_channels = num_xyz_channels
        self.use_complex_spectrum = use_complex_spectrum
        
        # The token dimension after combining x,y,z channels
        if self.use_complex_spectrum:
            # Each frequency component has (6 * spectrum_token_dim) dimensional representation
            self.token_embed_dim = num_xyz_channels * 2 * spectrum_token_dim
        else:
            # Each frequency component has (3 * spectrum_token_dim) dimensional representation
            self.token_embed_dim = num_xyz_channels * spectrum_token_dim
        
        # RoPE parameters
        self.c_s = c_s
        self.decoder_embed_dim = decoder_embed_dim
        self.rope_theta = rope_theta
        self.use_checkpoint = use_checkpoint
        
        # --------------------------------------------------------------------------
        # Dimension checks for multi-head attention
        assert c_s % encoder_num_heads == 0, (
            f"c_s ({c_s}) must be divisible by encoder_num_heads ({encoder_num_heads}). "
            f"Current c_s % encoder_num_heads = {c_s % encoder_num_heads}. "
            f"Please adjust c_s to be a multiple of {encoder_num_heads}."
        )
        assert decoder_embed_dim % decoder_num_heads == 0, (
            f"decoder_embed_dim ({decoder_embed_dim}) must be divisible by decoder_num_heads ({decoder_num_heads}). "
            f"Current decoder_embed_dim % decoder_num_heads = {decoder_embed_dim % decoder_num_heads}. "
            f"Please adjust decoder_embed_dim to be a multiple of {decoder_num_heads}."
        )
        
        # Store for later use
        self.encoder_num_heads = encoder_num_heads
        self.decoder_num_heads = decoder_num_heads
        
        # Frequency-specific attention config
        self.use_spectral_attention = use_spectral_attention
        self.spectral_num_heads = spectral_num_heads
        self.use_hybrid_block = use_hybrid_block
        self.max_spectrum_len = max_spectrum_len
        
        self.enable_cross_window = enable_cross_window
        # Self-forcing training mode
        self.use_self_forcing = use_self_forcing
        self.self_forcing_schedule = self_forcing_schedule
        
        # Cross-window attention (current window attends to last window)
        self.cross_attn = None
        if self.enable_cross_window:
            self.cross_attn = OpenFoldAttention(
                c_q=c_s,
                c_k=c_s,
                c_v=c_s,
                c_hidden=c_s // encoder_num_heads,
                no_heads=encoder_num_heads,
                gating=True,
            )
        
        # --------------------------------------------------------------------------
        # Condition Encoder
        # Use FeatureExtractor to extract s and z
        self.feature_extractor = FeatureExtractor(
            num_blocks=1,  # adjustable number of blocks
            sequence_state_dim=c_s,
            pairwise_state_dim=c_z,
            sequence_head_width=32,
            pairwise_head_width=32,
            position_bins=32,
            dropout=proj_dropout,
        )
        self.bihito_proj = nn.Sequential(
            nn.Linear(128, c_s),
            nn.LayerNorm(c_s),
        )

        self.condition_drop_prob = condition_drop_prob
        
        # Fake condition for CFG's unconditional generation
        # self.fake_system_condition = nn.Parameter(torch.zeros(1, c_s))
        # self.fake_condition_residue = nn.Parameter(torch.zeros(1, c_s))
        # Note: register_buffer ensures the buffer (and hence its device) tracks the parent module.
        # So .to(device) on the module will also move this buffer. No manual device move needed.
        fake_system_condition = torch.zeros(1, c_s)
        torch.nn.init.normal_(fake_system_condition, mean=0.0, std=0.02)
        self.register_buffer("fake_system_condition", fake_system_condition)

        fake_condition_residue = torch.zeros(1, c_s)
        torch.nn.init.normal_(fake_condition_residue, mean=0.0, std=0.02)
        self.register_buffer("fake_condition_residue", fake_condition_residue)
        
        # --------------------------------------------------------------------------
        # IPA mechanism
        self.IPA_encoder = IPAModule(
            c_s=c_s,
            c_z=c_z,
            c_ipa=c_ipa,
            no_heads_ipa=no_heads_ipa,
            no_qk_points=no_qk_points,
            no_v_points=no_v_points,
            inf=inf,
            eps=eps,
            num_ipa_layers=num_ipa_layers,
            dropout_rate=proj_dropout
        )

        # --------------------------------------------------------------------------
        # Spatial transformer (residue-level attention) using DiffusionTransformer
        # system_conditioning -> a, s_s -> s, s_z -> z
        self.spatial_transformer = DiffusionTransformer(
            c_a=c_s,
            c_s=c_s,
            c_z=c_z,
            n_blocks=encoder_depth,
            n_heads=encoder_num_heads,
            blocks_per_ckpt=None,
            with_irfc=with_irfc,
        )

        # --------------------------------------------------------------------------
        # Masking for autoregressive generation
        self.mask = mask
        self.mask_ratio_generator = stats.truncnorm((mask_ratio_min - 1.0) / 0.25, 0, loc=1.0, scale=0.25)
        self.mask_token = nn.Parameter(torch.zeros(1, 1, decoder_embed_dim))
        
        # --------------------------------------------------------------------------
        # Input projection for spectrum tokens
        # Input: (B, num_residues, seq_len, token_embed_dim)
        # We need to project to c_s
        self.spectrum_proj = nn.Sequential(
            nn.Linear(self.token_embed_dim, c_s),
            nn.LayerNorm(c_s),
        )
                
        # --------------------------------------------------------------------------
        # Encoder specifics
        self.buffer_size = buffer_size
        # RoPE: No fixed position embeddings needed - computed dynamically
        
        # Choose Block type based on use_hybrid_block
        BlockType = HybridBlock if use_hybrid_block else Block
        block_kwargs = {
            'mlp_ratio': mlp_ratio,
            'qkv_bias': True,
            'norm_layer': norm_layer,
            'proj_drop': proj_dropout,
            'attn_drop': attn_dropout,
            'act_layer': nn.GELU
        }
        if use_hybrid_block:
            block_kwargs.update({
                'use_spectral_attention': use_spectral_attention,
                'spectral_num_heads': spectral_num_heads or encoder_num_heads
            })
        
        self.encoder_blocks = nn.ModuleList([
            BlockType(c_s, encoder_num_heads, **block_kwargs)
            for _ in range(encoder_depth)
        ])
        self.encoder_norm = norm_layer(c_s)
        
        # --------------------------------------------------------------------------
        # Decoder specifics
        self.decoder_embed = nn.Linear(c_s, decoder_embed_dim, bias=True)
        # RoPE: No fixed position embeddings needed - computed dynamically
        
        # Choose Block type based on use_hybrid_block
        BlockType = HybridBlock if use_hybrid_block else Block
        decoder_block_kwargs = {
            'mlp_ratio': mlp_ratio,
            'qkv_bias': True,
            'norm_layer': norm_layer,
            'proj_drop': proj_dropout,
            'attn_drop': attn_dropout,
            'act_layer': nn.GELU
        }
        if use_hybrid_block:
            decoder_block_kwargs.update({
                'use_spectral_attention': use_spectral_attention,
                'spectral_num_heads': spectral_num_heads or decoder_num_heads
            })
        
        self.decoder_blocks = nn.ModuleList([
            BlockType(decoder_embed_dim, decoder_num_heads, **decoder_block_kwargs)
            for _ in range(decoder_depth)
        ])
        
        self.decoder_norm = norm_layer(decoder_embed_dim)
        # RoPE: No fixed diffusion position embeddings needed
        
        # --------------------------------------------------------------------------
        # Diffusion Loss for continuous spectrum prediction
        
        self.spectrum_diffloss = SpectrumDiffLoss(
            target_channels=self.token_embed_dim,
            z_channels=decoder_embed_dim,
            width=diffloss_w,
            depth=diffloss_d,
            num_sampling_steps=num_sampling_steps,
            learn_sigma=learn_sigma,
            predict_xstart=predict_xstart,
            sigma_small=sigma_small,
            noise_schedule=noise_schedule,
            use_kl=use_kl,
            rescale_learned_sigmas=rescale_learned_sigmas,
            diffusion_steps=diffusion_steps,
            denoise_seq_chunk_size=denoise_seq_chunk_size,
            denoise_block_type=denoise_block_type,
            denoise_use_checkpoint=denoise_use_checkpoint,
        )        
        self.diffusion_batch_mul = diffusion_batch_mul
        
        # RoPE frequency caches to avoid repeated large allocations
        # key: (dim, total_seq_len, device_str)
        self._rope_cache_encoder = {}
        self._rope_cache_decoder = {}
        self._rope_cache_diffusion = {}
        # Cached zeros buffer used for mask prefix (num_residues, buffer_size)
        # key: (num_residues, device_str)
        self._zeros_prefix_cache = {}

        # Loss weight: 动态计算，不再需要固定的buffer
        # 在forward时根据实际seq_len计算
        
        self.initialize_weights()
    
    def _get_cached_freqs(self, cache_dict, dim: int, total_len: int, device):
        """Get cached RoPE freqs_cis tensor or compute and store it.
        Cache key is (dim, total_len, str(device)).
        """
        # Normalize device representation for cache key
        if isinstance(device, torch.device):
            device_key = str(device)
        else:
            device_key = str(torch.device(device))
        key = (int(dim), int(total_len), device_key)
        freqs = cache_dict.get(key)
        if freqs is None or freqs.device != torch.device(device):
            freqs = precompute_freqs_cis(dim, total_len, self.rope_theta, device)
            cache_dict[key] = freqs
        return freqs

    def _get_zeros_prefix(self, num_residues: int, device):
        """Get cached zeros bool tensor of shape (num_residues, buffer_size)."""
        if isinstance(device, torch.device):
            device_key = str(device)
        else:
            device_key = str(torch.device(device))
        key = (int(num_residues), device_key)
        z = self._zeros_prefix_cache.get(key)
        if z is None or z.device != torch.device(device) or z.shape[1] != self.buffer_size:
            z = torch.zeros(num_residues, self.buffer_size, device=device, dtype=torch.bool)
            self._zeros_prefix_cache[key] = z
        return z
                
    def initialize_weights(self):
        """Initialize weights"""
        # torch.nn.init.normal_(self.fake_system_condition, std=.02)
        # torch.nn.init.normal_(self.fake_condition_residue, std=.02)
        if self.mask:
            torch.nn.init.normal_(self.mask_token, std=.02)
        
        # Initialize nn.Linear and nn.LayerNorm
        self.apply(self._init_weights)
    
    def get_freq_loss_weight(self, seq_len, device):
        """
        Dynamically compute frequency-dependent loss weight
        
        Args:
            seq_len: actual sequence length
            device: device to create tensor on
            
        Returns:
            torch.Tensor: loss weight tensor [seq_len]
        """
        return torch.tensor([1.0 + np.sin(np.pi / 2. * (i + 1) / seq_len) 
                            for i in range(seq_len)], device=device)

    def generate_sliding_window_bias(self, q_len, k_len, hop_size, sigma=2.0, device='cuda', dtype=torch.float32):
        """
        Construct Gaussian bias for cross attention between sliding windows.
        q_len: length of current window
        k_len: length of last window
        hop_size: window shift (typically window_size * hop_ratio)
        """
        t_q = torch.arange(q_len, device=device).view(-1, 1)  # [Q, 1]
        t_k = torch.arange(k_len, device=device).view(1, -1)  # [1, K]
        distance = torch.abs(t_q - (t_k - hop_size))
        bias = -(distance ** 2) / (2 * sigma ** 2)
        return bias.unsqueeze(0).unsqueeze(0).to(dtype=dtype)
        
    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            torch.nn.init.xavier_uniform_(m.weight)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
            if m.weight is not None:
                nn.init.constant_(m.weight, 1.0)
                
    def prepare_spectrum_input(self, spectrum):
        """
        Prepare spectrum input
        Args:
            spectrum: (num_residues, (T//2+1), 6, spectrum_token_dim) if use_complex_spectrum else (num_residues, (T//2+1), 3, spectrum_token_dim)
        Returns:
            spectrum_tokens: (num_residues, seq_len, token_embed_dim), token_embed_dim = num_xyz_channels * spectrum_token_dim
            seq_len: actual sequence length (T//2+1)
        """
        assert spectrum.dim() == 4, f"Expected spectrum.dim() == 4, got {spectrum.dim()}, shape of spectrum: {spectrum.shape}"
        print("=="*40+f"\nshape of spectrum: {spectrum.shape}"+"=="*40)
        num_residues, seq_len, xyz, token_dim = spectrum.shape
        if self.max_spectrum_len is not None and seq_len > self.max_spectrum_len:
            spectrum = spectrum[:, : self.max_spectrum_len, :, :].contiguous()
            seq_len = self.max_spectrum_len
        
        if self.use_complex_spectrum:
            assert int(xyz) == 2 * self.num_xyz_channels, f"Expected xyz={2 * self.num_xyz_channels}, got {xyz}"
        else:
            assert int(xyz) == self.num_xyz_channels, f"Expected xyz={self.num_xyz_channels}, got {xyz}"
        
        return spectrum.reshape(num_residues, seq_len, xyz*token_dim).contiguous()
        
    def recover_spectrum_from_tokens(self, spectrum_tokens):
        """
        Recover spectrum from tokens, there is no batch_mul dim when sampling
        Args:
            spectrum_tokens: (batch_mul, num_residues, seq_len, token_embed_dim)
        Returns:
            spectrum: (batch_mul, num_residues, seq_len, 6, spectrum_token_dim) or (batch_mul, num_residues, seq_len, 3, spectrum_token_dim)
        """
        if spectrum_tokens.dim() == 4:
            batch_mul, num_residues, seq_len, token_embed_dim = spectrum_tokens.shape
            if self.use_complex_spectrum:
                expected_dim = self.num_xyz_channels * 2 * self.spectrum_token_dim
                assert expected_dim == token_embed_dim, f"Expected {expected_dim} == {token_embed_dim}, got {expected_dim} != {token_embed_dim}"
                spectrum = spectrum_tokens.reshape(batch_mul, num_residues, seq_len, self.num_xyz_channels * 2, self.spectrum_token_dim).contiguous()
            else:
                expected_dim = self.num_xyz_channels * self.spectrum_token_dim
                assert expected_dim == token_embed_dim, f"Expected {expected_dim} == {token_embed_dim}, got {expected_dim} != {token_embed_dim}"
                spectrum = spectrum_tokens.reshape(batch_mul, num_residues, seq_len, self.num_xyz_channels, self.spectrum_token_dim).contiguous()
        elif spectrum_tokens.dim() == 3:
            num_residues, seq_len, token_embed_dim = spectrum_tokens.shape
            if self.use_complex_spectrum:
                expected_dim = self.num_xyz_channels * 2 * self.spectrum_token_dim
                assert expected_dim == token_embed_dim, f"Expected {expected_dim} == {token_embed_dim}, got {expected_dim} != {token_embed_dim}"
                spectrum = spectrum_tokens.reshape(num_residues, seq_len, self.num_xyz_channels * 2, self.spectrum_token_dim).contiguous()
            else:
                expected_dim = self.num_xyz_channels * self.spectrum_token_dim
                assert expected_dim == token_embed_dim, f"Expected {expected_dim} == {token_embed_dim}, got {expected_dim} != {token_embed_dim}"
                spectrum = spectrum_tokens.reshape(num_residues, seq_len, self.num_xyz_channels, self.spectrum_token_dim).contiguous()
        else:
            raise ValueError(f"Unsupported shape of spectrum_tokens: {spectrum_tokens}")
        return spectrum
    
    # TODO：模型同时预测出实部和虚部，但这里没有加上实部和虚部是反对称的这个硬约束条件，所以应该只预测出实部，然后再根据反共轭给出虚部

    def progressive_masking(self, spectrum_tokens, num_residues, seq_len):
        """
        Progressive masking strategy for spectrum generation
        Simulates progressive resolution increase (like in FAR)
        
        Args:
            spectrum_tokens: (num_residues, seq_len, token_embed_dim)
        Returns:
            masked_tokens: progressively masked spectrum
            stage_index: current generation stage
        """
        # B = spectrum_tokens.shape[0]
        device = spectrum_tokens.device
        
        # Now assumes spectrum_tokens: (num_residues, seq_len, dim), no batch dimension

        # Random stage from 0 to seq_len
        # Stage 0: all zeros (no frequency)
        # Stage seq_len: full spectrum
        stage = torch.randint(0, seq_len + 1, (1,), device=device).item()
        
        masked_tokens = torch.zeros_like(spectrum_tokens)

        if int(stage) > 0:
            # Use interpolation to simulate progressive frequency addition
            # Take first s frequency components and interpolate
            partial_spectrum = spectrum_tokens[:, :stage, :]  # (num_residues, s, dim)
            
            # Interpolate to seq_len
            # Reshape for interpolation: (num_residues, dim, s)
            partial_spectrum = partial_spectrum.permute(0, 2, 1)  # (num_residues, dim, s)
            partial_spectrum = partial_spectrum.reshape(num_residues * self.token_embed_dim, 1, stage)
            
            # Interpolate
            interpolated = F.interpolate(partial_spectrum, size=seq_len, mode='linear', align_corners=False)
            interpolated = interpolated.reshape(num_residues, self.token_embed_dim, seq_len)
            interpolated = interpolated.permute(0, 2, 1)  # (num_residues, seq_len, dim)
            
            masked_tokens = interpolated

        return masked_tokens, stage
    
    # def sample_orders(self, bsz, seq_len, device):
    #     """Generate random generation orders for each sample"""
    #     orders = []
    #     for _ in range(bsz):
    #         order = np.array(list(range(seq_len)))
    #         np.random.shuffle(order)
    #         orders.append(order)
    #     orders = torch.tensor(np.array(orders), dtype=torch.long, device=device)
    #     return orders

    def sample_orders(self, seq_len, device):
        """Generate a random generation order (1D) for spectrum positions"""
        return torch.randperm(seq_len, device=device, dtype=torch.long)

    def random_masking(self, x, orders, stage_index, seq_len):
        """
        Generate token mask for training (no batch version)
        Args:
            x: (num_residues, seq_len, embed_dim)
            orders: (seq_len,) random order
            stage_index: int or float, current stage
            seq_len: sequence length
        Returns:
            mask: (num_residues, seq_len) boolean mask (True = masked)
        """
        num_residues, _, embed_dim = x.shape
        device = x.device

        # Calculate mask ratio based on stage
        mask_ratio = 1.0 - float(stage_index) / seq_len

        # Add randomness
        mask_ratio = mask_ratio + np.random.randn() * 0.1
        mask_ratio = np.clip(mask_ratio, 0.0, 1.0)

        # Generate mask for spectrum (no batch)
        mask = torch.zeros(num_residues, seq_len, device=device)

        num_masked = int(np.ceil(seq_len * mask_ratio))
        # Mask according to random order
        mask_indices = orders[:num_masked]
        mask[:, mask_indices] = 1.0

        return mask.bool()
        

    def forward_encoder(self, 
                        spectrum_tokens, 
                        s_s,
                        system_condition,
                        num_residues, seq_len, mask=None,
                        last_spectrum_tokens: Optional[torch.Tensor] = None,
                        hop_size: Optional[int] = None):
        """
        Forward through encoder with RoPE (no batch dimension)
        Args:
            spectrum_tokens: (num_residues, seq_len, token_embed_dim)
            s_s: (num_residues, c_s)
            system_condition: output of IPA module, (num_residues, c_s)
            num_residues: number of residues
            seq_len: sequence length
            mask: (num_residues, seq_len) optional mask
        Returns:
            x: (num_residues, seq_len_effective, embed_dim) encoded features
        """
        device = spectrum_tokens.device
        
        # Project spectrum tokens
        x = self.spectrum_proj(spectrum_tokens)  # (num_residues, seq_len, c_s)
        
        # Cross-attention with last window spectrum (optional)
        if self.enable_cross_window and last_spectrum_tokens is not None and self.cross_attn is not None:
            last_x = self.spectrum_proj(last_spectrum_tokens)  # (num_residues, seq_len, c_s)
            bias = None
            if hop_size is not None:
                bias = self.generate_sliding_window_bias(
                    q_len=x.shape[-2],
                    k_len=last_x.shape[-2],
                    hop_size=hop_size,
                    sigma=2.0,
                    device=x.device,
                    dtype=x.dtype,
                )
            biases = [bias] if bias is not None else None
            x = self.cross_attn(q_x=x, kv_x=last_x, biases=biases)
        
        # Add residue-level condition
        x = x + s_s.unsqueeze(-2)  # (num_residues, 1, c_s) + (num_residues, seq_len, c_s)
        
        # Add buffer for global condition
        buffer = system_condition.unsqueeze(-2).expand(-1, self.buffer_size, -1)  # (num_residues, buffer_size, c_s) view only
        x = torch.cat([buffer, x], dim=1)  # (num_residues, buffer_size + seq_len, c_s)
        
        # Apply RoPE: 动态计算位置编码
        total_seq_len = self.buffer_size + seq_len
        freqs_cis = self._get_cached_freqs(self._rope_cache_encoder, self.c_s, total_seq_len, device)
        x = apply_rotary_emb(x, freqs_cis)  # (num_residues, buffer_size + seq_len, c_s)
        
        # Apply mask if provided
        if mask is not None and self.mask:
            zeros_prefix = self._get_zeros_prefix(num_residues, device)
            mask_with_buffer = torch.cat([zeros_prefix, mask], dim=1)  # (num_residues, buffer_size + seq_len)
            
            # Flatten for masking: (num_residues, L, D) -> (num_residues*L, D)
            x = x.reshape(num_residues * (self.buffer_size + seq_len), -1)
            # For now, keep all tokens but the mask can be used downstream for attention
        else:
            x = x.reshape(num_residues * (self.buffer_size + seq_len), -1)
        
        # Apply transformer blocks
        for blk in self.encoder_blocks:
            if self.use_checkpoint and self.training:
                if self.use_hybrid_block:
                    # For HybridBlock, we need to reshape and provide num_residues/seq_len
                    # ckpt.checkpoint doesn't support kwargs well, so we flatten the call
                    def block_forward(x_tensor):
                        return blk(x_tensor, num_residues=num_residues, seq_len=self.buffer_size + seq_len)
                    x = ckpt.checkpoint(block_forward, x, use_reentrant=False)
                else:
                    x = ckpt.checkpoint(blk, x, use_reentrant=False)
            else:
                if self.use_hybrid_block:
                    x = blk(x, num_residues=num_residues, seq_len=self.buffer_size + seq_len)
                else:
                    x = blk(x)
        x = self.encoder_norm(x)
        # 维度爆炸，不能采用这个
        # x = self.IPA_encoder(
        #     x, 
        #     s_z.unsqueeze(-2).expand(-1, -1, -1, (self.buffer_size + seq_len), -1).reshape(-1, num_residues * (self.buffer_size + seq_len), num_residues * (self.buffer_size + seq_len) -1), 
        #     coords.unsqueeze(-2).expand(-1, -1, -1, (self.buffer_size + seq_len), -1).reshape(-1, num_residues * (self.buffer_size + seq_len), 3), 
        #     rotation_matrices.unsqueeze(-3).expand(-1, -1, (self.buffer_size + seq_len), -1,-1).reshape(-1, num_residues * (self.buffer_size + seq_len), 3, 3), 
        #     aa_mask.unsqueeze(-1).expand(-1, -1, (self.buffer_size + seq_len)).reshape(-1, num_residues * (self.buffer_size + seq_len))
        # )
        # Reshape back
        x = x.reshape(num_residues, self.buffer_size + seq_len, -1)
        
        return x # (num_residues, buffer_size + seq_len, c_s)

    def forward_decoder(self, x, num_residues, seq_len, mask=None):
        """
        Forward through decoder with RoPE (no batch dimension)
        Args:
            x: (num_residues, buffer_size + seq_len, c_s)
            num_residues: number of residues
            seq_len: sequence length
            mask: (num_residues, seq_len) optional mask
        Returns:
            z: (num_residues, seq_len, decoder_embed_dim) decoded features
        """
        device = x.device
        
        # Project to decoder dimension
        x = self.decoder_embed(x)  # (num_residues, buffer_size + seq_len, decoder_embed_dim)
        
        # Add mask tokens if masking
        if mask is not None and self.mask:
            # Extend mask with buffer
            zeros_prefix = self._get_zeros_prefix(num_residues, device)
            mask_with_buffer = torch.cat([zeros_prefix, mask], dim=1)  # (num_residues, buffer_size + seq_len)
            mask_tokens = self.mask_token.expand(num_residues, self.buffer_size + seq_len, -1)
            x = torch.where(mask_with_buffer.unsqueeze(-1), mask_tokens, x)
        
        # Apply RoPE: 动态计算位置编码
        total_seq_len = self.buffer_size + seq_len
        freqs_cis_decoder = self._get_cached_freqs(self._rope_cache_decoder, self.decoder_embed_dim, total_seq_len, device)
        x = apply_rotary_emb(x, freqs_cis_decoder)  # (num_residues, buffer_size + seq_len, decoder_embed_dim)
        
        # Flatten for transformer
        x = x.reshape(num_residues * (self.buffer_size + seq_len), -1)
        
        # Apply transformer blocks
        for blk in self.decoder_blocks:
            if self.use_checkpoint and self.training:
                if self.use_hybrid_block:
                    # For HybridBlock, we need to reshape and provide num_residues/seq_len
                    def block_forward(x_tensor):
                        return blk(x_tensor, num_residues=num_residues, seq_len=self.buffer_size + seq_len)
                    x = ckpt.checkpoint(block_forward, x, use_reentrant=False)
                else:
                    x = ckpt.checkpoint(blk, x, use_reentrant=False)
            else:
                if self.use_hybrid_block:
                    x = blk(x, num_residues=num_residues, seq_len=self.buffer_size + seq_len)
                else:
                    x = blk(x)
        x = self.decoder_norm(x)
        
        # Reshape back and remove buffer
        x = x.reshape(num_residues, self.buffer_size + seq_len, -1)
        x = x[:, self.buffer_size:, :]  # (num_residues, seq_len, decoder_embed_dim)
        
        # Apply additional RoPE for diffusion conditioning (optional, for seq_len dimension)
        freqs_cis_diffusion = self._get_cached_freqs(self._rope_cache_diffusion, self.decoder_embed_dim, seq_len, device)
        z = apply_rotary_emb(x, freqs_cis_diffusion)  # (num_residues, seq_len, decoder_embed_dim)
        
        return z

    def _forward_self_forcing(self,
                               spectrum_tokens,
                               s_s,
                               system_condition,
                               num_residues,
                               seq_len,
                               last_spectrum_tokens,
                               hop_size,
                               loss_weight,
                               return_pred_spectrum,
                               device):
        """
        Self-forcing forward pass: step-by-step generation using ground truth (teacher forcing)
        This bridges the gap between training (random stage sampling) and inference (step-by-step generation)
        
        Args:
            spectrum_tokens: (num_residues, seq_len, token_embed_dim) ground truth tokens
            s_s: (num_residues, c_s) residue features
            system_condition: (num_residues, c_s) system condition
            num_residues: number of residues
            seq_len: sequence length
            last_spectrum_tokens: optional previous window tokens
            hop_size: optional hop size for cross-window attention
            loss_weight: whether to use frequency-dependent loss weighting
            return_pred_spectrum: whether to return predicted spectrum
            device: device
            
        Returns:
            batch_pred_spectrum: (batch_mul, num_residues, seq_len, token_embed_dim) if return_pred_spectrum
            loss: scalar loss (accumulated over steps)
        """
        # Determine which steps to compute loss for
        if self.self_forcing_schedule is None:
            # Compute loss at all steps
            loss_steps = list(range(seq_len))
        else:
            # Only compute loss at specified steps
            loss_steps = [s for s in self.self_forcing_schedule if 0 <= s < seq_len]
            if not loss_steps:
                loss_steps = [seq_len - 1]  # At least compute at the last step
        
        # Initialize tokens with zeros (starting from lowest frequency)
        tokens = torch.zeros(num_residues, seq_len, self.token_embed_dim, device=device)
        
        # Accumulate losses
        total_loss = 0.0
        loss_count = 0
        batch_pred_xstarts_all = []
        
        # Step-by-step generation (similar to inference, but using ground truth)
        for step in range(seq_len):
            # Prepare input tokens: use ground truth up to current step, zeros for future (teacher forcing)
            tokens = torch.zeros_like(spectrum_tokens)
            if step > 0:
                # Use ground truth tokens up to step-1, then interpolate
                tokens[:, :step, :] = spectrum_tokens[:, :step, :]
                # Interpolate to full length for progressive resolution
                current_valid = tokens[:, :step, :]  # (num_residues, step, D)
                current_valid_reshaped = current_valid.permute(0, 2, 1)  # (num_residues, D, step)
                current_valid_flat = current_valid_reshaped.reshape(num_residues * self.token_embed_dim, 1, step)
                interpolated = F.interpolate(current_valid_flat, size=seq_len, mode='linear', align_corners=False)
                interpolated = interpolated.reshape(num_residues, self.token_embed_dim, seq_len)
                interpolated = interpolated.permute(0, 2, 1)  # (num_residues, seq_len, D)
                tokens = interpolated
            
            # Mask for current step (mask future frequencies)
            mask = torch.zeros(num_residues, seq_len, device=device, dtype=torch.bool)
            mask[:, step+1:] = True  # Mask positions after current step
            
            # Forward through encoder
            encoded = self.forward_encoder(
                tokens,
                s_s,
                system_condition,
                num_residues,
                seq_len,
                mask,
                last_spectrum_tokens=last_spectrum_tokens,
                hop_size=hop_size
            )
            
            # Forward through decoder
            decoded = self.forward_decoder(encoded, num_residues, seq_len, mask)
            
            # Compute loss only at specified steps
            if step in loss_steps:
                # Get target for current step (use full ground truth)
                target_step = spectrum_tokens
                
                # Compute loss with dynamic frequency weight
                loss_weight_tensor = self.get_freq_loss_weight(seq_len, device) if loss_weight else None
                batch_pred_xstarts, loss_step = self.forward_diffusion_loss(
                    target=target_step,
                    z=decoded,
                    stage_index=step,
                    num_residues=num_residues,
                    seq_len=seq_len,
                    loss_weight=loss_weight_tensor
                )
                
                total_loss = total_loss + loss_step
                loss_count += 1
                if return_pred_spectrum and step == loss_steps[-1]:
                    # Store prediction from the last loss step
                    batch_pred_xstarts_all = batch_pred_xstarts
        
        # Average loss over computed steps
        if loss_count > 0:
            loss = total_loss / loss_count
        else:
            loss = torch.tensor(0.0, device=device, requires_grad=True)
        
        if return_pred_spectrum:
            if batch_pred_xstarts_all is not None and len(batch_pred_xstarts_all) > 0:
                batch_pred_spectrum = self.recover_spectrum_from_tokens(batch_pred_xstarts_all)
                return batch_pred_spectrum, loss
            else:
                # Fallback: use final tokens
                final_tokens = self.recover_spectrum_from_tokens(tokens.unsqueeze(0))
                return final_tokens, loss
        
        return None, loss

    def forward_diffusion_loss(self, target, z, stage_index, num_residues, seq_len, loss_weight=None):
        """
        Compute diffusion loss (no batch dim)
        Args:
            z: (num_residues, seq_len, decoder_embed_dim) condition from AR transformer
            target: (num_residues, seq_len, token_embed_dim) target spectrum
            stage_index: int or float, generation stage
            num_residues: number of residues
            seq_len: sequence length
            loss_weight: optional per-frequency weight
        Returns:
            batch_pred_xstarts: (batch_mul, num_residues, seq_len, token_embed_dim)
            loss: scalar loss
        """
        # # Safety: clamp/normalize to avoid explosion/NaNs
        # max_val = 50.0
        # max_norm = 50.0
        # z = torch.nan_to_num(z, nan=0.0, posinf=max_val, neginf=-max_val)
        # target = torch.nan_to_num(target, nan=0.0, posinf=max_val, neginf=-max_val)
        # z = torch.clamp(z, min=-max_val, max=max_val)
        # target = torch.clamp(target, min=-max_val, max=max_val)
        # z_norm = torch.norm(z, dim=-1, keepdim=True).clamp(min=1e-6)
        # t_norm = torch.norm(target, dim=-1, keepdim=True).clamp(min=1e-6)
        # z = z * torch.clamp(max_norm / z_norm, max=1.0)
        # target = target * torch.clamp(max_norm / t_norm, max=1.0)
        
        
        # # Flatten # no more flatten
        # z_flat = z.reshape(num_residues * seq_len, -1)  # (N*L, D)
        # target_flat = target.reshape(num_residues * seq_len, -1)  # (N*L, token_dim)
        
        # Repeat for diffusion batch multiplication
        # z_flat = z_flat.reshape(self.diffusion_batch_mul, 1)
        # target_flat = target_flat.reshape(self.diffusion_batch_mul, 1)
        # Remove batch-level memory expansion; handle batch_mul inside diffloss module if needed
        # (Do not repeat/expand z_flat and target_flat here to avoid OOM.)

        # Stage index conditioning (B, 1)，不再按 M 扩展
        stage_index_float = float(stage_index)
        stage_index_tensor = torch.full(
            (num_residues, seq_len, 1),
            stage_index_float, dtype=z.dtype, device=z.device
        ) # (num_residues, seq_len, 1)
        # Loss weight
        if loss_weight is not None:
            # loss_weight: (seq_len,)
            pass
            #loss_weight_expanded = loss_weight.unsqueeze(0).expand(num_residues, seq_len) # (N, L)
            # loss_weight_flat = loss_weight_expanded.reshape(num_residues * seq_len)
            # loss_weight_flat = loss_weight_flat.repeat(self.diffusion_batch_mul)
        else:
            # loss_weight_flat = None
            loss_weight_expanded = None
        
        # Compute diffusion loss
        M = getattr(self, "diffusion_batch_mul", 1)
        batch_pred_xstarts, loss = self.spectrum_diffloss(
                             target=target,
                             z=z, 
                             index=stage_index_tensor, 
                             loss_weight=loss_weight,
                             batch_mul=M
                             )
        loss = torch.nan_to_num(loss, nan=0.0, posinf=1e4, neginf=1e4)
        
        return batch_pred_xstarts, loss

    def forward(self, 
                spectrum, 
                coords, 
                rotation_matrices,
                aatype, 
                aa_mask,
                esm_features, 
                bihito_token_repr: Optional[torch.Tensor] = None,
                loss_weight=True,
                return_pred_spectrum=True,
                last_spectrum: Optional[torch.Tensor] = None,
                window_size: Optional[int] = None,
                hop_ratio: Optional[float] = None):
        """
        Forward pass for training (no batch dimension version)
        Args:
            spectrum: (num_residues, (T//2+1), 6, spectrum_token_dim) if use_complex_spectrum else (num_residues, (T//2+1), 3, spectrum_token_dim)
            coords: (num_residues, 3)
            rotation_matrices: (num_residues, 3, 3) rotation matrices
            aatype: (num_residues,)
            aa_mask: (num_residues,) mask indicating valid residue positions
            esm_features: (num_residues, 1280)
            loss_weight: whether to use frequency-dependent loss weighting
        Returns:
            loss: scalar loss
            batch_pred_spectrum: (batch_mul, num_residues, seq_len, 6, spectrum_token_dim) or (batch_mul, num_residues, seq_len, 3, spectrum_token_dim) - reconstructed from diffusion for monitoring
        """
        # Prepare inputs
        spectrum_tokens = self.prepare_spectrum_input(spectrum)
        # spectrum_tokens: (num_residues, seq_len, token_embed_dim), seq_len=(T//2+1)
        num_residues, seq_len, _ = spectrum_tokens.shape
        
        use_cross_window = self.enable_cross_window and (window_size is not None and hop_ratio is not None)
        last_spectrum_tokens = None
        if use_cross_window:
            if last_spectrum is not None:
                last_spectrum_tokens = self.prepare_spectrum_input(last_spectrum)
                if last_spectrum_tokens.shape[:2] != (num_residues, seq_len):
                    raise ValueError(
                        f"last_spectrum shape mismatch: expected ({num_residues}, {seq_len}, *), "
                        f"got {last_spectrum_tokens.shape}"
                    )
            else:
                # For the first window, fall back to current spectrum to keep graph stable
                last_spectrum_tokens = spectrum_tokens
        
        hop_size = None
        if use_cross_window:
            hop_size = int(window_size * hop_ratio)
        
        # Encode conditions
        s_s, s_z = self.feature_extractor(aatype, esm_features, aa_mask, coords, rotation_matrices)
        # s_s: (num_residues, c_s)
        # s_z: (num_residues, num_residues, c_z)
        # coords: (num_residues, 3)
        # rotation_matrices: (num_residues, 3, 3)
        # aa_mask: (num_residues,)
        if bihito_token_repr is not None:
            s_s = s_s + self.bihito_proj(bihito_token_repr)

        system_condition = self.IPA_en、coder(s_s, s_z, coords, rotation_matrices, aa_mask.to(dtype=s_s.dtype)) # (num_residues, c_s)

        # IRFC distance (residue pairwise squared distance)
        irfc_dist_sq = torch.sum(
            (coords.unsqueeze(-2) - coords.unsqueeze(-3)) ** 2, dim=-1
        )
        
        # Random drop condition for CFG training
        if self.training:
            # drop_mask: single boolean for all residues
            drop = (torch.rand(1, device=system_condition.device) < self.condition_drop_prob).item()
            if drop:
                system_condition = self.fake_system_condition.expand(num_residues, -1)
                s_s = self.fake_condition_residue.expand(num_residues, -1)
                s_z = torch.zeros_like(s_z)

        # Residue-level spatial transformer
        system_condition = self.spatial_transformer(
            a=system_condition,
            s=s_s,
            z=s_z,
            irfc_dist_sq=irfc_dist_sq,
        )
        
        # Self-forcing mode: step-by-step generation using ground truth (teacher forcing)
        if self.training and self.use_self_forcing:
            return self._forward_self_forcing(
                spectrum_tokens=spectrum_tokens,
                s_s=s_s,
                system_condition=system_condition,
                num_residues=num_residues,
                seq_len=seq_len,
                last_spectrum_tokens=last_spectrum_tokens,
                hop_size=hop_size,
                loss_weight=loss_weight,
                return_pred_spectrum=return_pred_spectrum,
                device=s_s.device
            )
        
        # Original training mode: random stage sampling
        # Progressive masking
        masked_tokens, stage_index = self.progressive_masking(spectrum_tokens, num_residues, seq_len)
        
        # Random masking for autoregressive training
        device = s_s.device
        orders = self.sample_orders(seq_len, device)
        mask = self.random_masking(masked_tokens, orders, stage_index, seq_len) if self.mask else None
        
        # Forward through encoder
        encoded = self.forward_encoder(masked_tokens, 
                    s_s,
                    system_condition,
                    num_residues, 
                    seq_len, 
                    mask,
                    last_spectrum_tokens=last_spectrum_tokens,
                    hop_size=hop_size)
        
        # Forward through decoder
        decoded = self.forward_decoder(encoded, num_residues, seq_len, mask) # condition
        
        # Compute loss with dynamic frequency weight & Sample predicted spectrum using diffusion
        loss_weight_tensor = self.get_freq_loss_weight(seq_len, device) if loss_weight else None
        batch_pred_xstarts, loss = self.forward_diffusion_loss(target=spectrum_tokens, 
                                 z=decoded, 
                                 stage_index=stage_index, 
                                 num_residues=num_residues, 
                                 seq_len=seq_len, 
                                 loss_weight=loss_weight_tensor
                                 )
                
        if return_pred_spectrum:
            batch_pred_spectrum = self.recover_spectrum_from_tokens(batch_pred_xstarts)
            return batch_pred_spectrum, loss
        print("No need to return predicted spectrum: return_pred_spectrum={return_pred_spectrum}")
        return None, loss
    
    @torch.no_grad() # TODO: cfg的复制维度是batch，不是residues
    def sample(self, 
               coords, 
               rotation_matrices,
               aatype, 
               aa_mask,
               esm_features, 
               bihito_token_repr: Optional[torch.Tensor] = None,
               num_iter=None, 
               cfg=1.0, 
               temperature=1.0, 
               progress=False):
        """
        Sample spectrum autoregressively (single sample, no batch dimension)
        Args:
            coords: (num_residues, 3)
            rotation_matrices: (num_residues, 3, 3)
            aatype: (num_residues,)
            aa_mask: (num_residues,)
            esm_features: (num_residues, 1280)
            num_iter: number of iterations
            cfg: classifier-free guidance scale
            temperature: sampling temperature
            progress: show progress bar
        Returns:
            spectrum: generated spectrum, (num_residues, seq_len, 6, spectrum_token_dim) if use_complex_spectrum else (num_residues, seq_len, 3, spectrum_token_dim)
        """
        num_residues = coords.shape[0]
        device = coords.device
        
        if num_iter is None:
            raise ValueError("num_iter must be specified for sampling (the number of frequency components to generate)")
        
        seq_len = num_iter
        
        # Encode conditions
        s_s, s_z = self.feature_extractor(aatype, esm_features, aa_mask, coords, rotation_matrices)
        if bihito_token_repr is not None:
            s_s = s_s + self.bihito_proj(bihito_token_repr)
        system_condition = self.IPA_encoder(s_s, s_z, coords, rotation_matrices, aa_mask.to(dtype=s_s.dtype)) 
        # s_s: (num_residues, c_s)
        # system_condition: (num_residues, c_s)

        # IRFC distance (residue pairwise squared distance)
        irfc_dist_sq = torch.sum(
            (coords.unsqueeze(-2) - coords.unsqueeze(-3)) ** 2, dim=-1
        )

        # Residue-level spatial transformer
        system_condition = self.spatial_transformer(
            a=system_condition,
            s=s_s,
            z=s_z,
            irfc_dist_sq=irfc_dist_sq,
        )

        # Initialize tokens
        tokens = torch.zeros(num_residues, seq_len, self.token_embed_dim, device=device)
        
        # Generation order (low to high frequency)
        orders = torch.arange(seq_len, device=device)
        
        # Iterative generation
        indices = list(range(num_iter))
        if progress:
            indices = tqdm(indices, desc="Generating spectrum")
        
        for step in indices:
            # Progressive resolution (similar to FAR)
            # Interpolate current tokens to simulate progressive frequency addition
            if step > 0:
                current_valid = tokens[:, :step, :]  # (num_residues, step, D)
                # Interpolate
                current_valid_reshaped = current_valid.permute(0, 2, 1)  # (num_residues, D, step)
                current_valid_flat = current_valid_reshaped.reshape(num_residues * self.token_embed_dim, 1, step)
                interpolated = F.interpolate(current_valid_flat, size=seq_len, mode='linear', align_corners=False)
                interpolated = interpolated.reshape(num_residues, self.token_embed_dim, seq_len)
                interpolated = interpolated.permute(0, 2, 1)  # (num_residues, seq_len, D)
                tokens = interpolated
            
            # Prepare for CFG (double for conditional and unconditional)
            tokens_doubled = torch.cat([tokens, tokens], dim=0)  # (2*num_residues, seq_len, D)
            system_condition_doubled = torch.cat([
                system_condition, 
                self.fake_system_condition.expand(num_residues, -1)
            ], dim=0)  # (2*num_residues, c_s)
            s_s_doubled = torch.cat([
                s_s, 
                self.fake_condition_residue.expand(num_residues, -1)
            ], dim=0)  # (2*num_residues, c_s)
            
            # Mask for current step (mask未来的频率)
            mask = torch.zeros(num_residues, seq_len, device=device, dtype=torch.bool)
            mask[:, step+1:] = True  # Mask positions after current step
            mask_doubled = torch.cat([mask, mask], dim=0)  # (2*num_residues, seq_len)
            
            # Forward encoder (需要处理doubled的num_residues)
            # 将2*num_residues作为新的num_residues传入
            encoded = self.forward_encoder(tokens_doubled, 
                                           s_s_doubled,
                                           system_condition_doubled,
                                           2 * num_residues,  # doubled num_residues
                                           seq_len, 
                                           mask_doubled)
            z = self.forward_decoder(encoded, 2 * num_residues, seq_len, mask_doubled)
            
            # Sample from diffusion
            # Note: Keep z as 3D tensor (do NOT flatten) for FrequencyComponentDenoiseNet
            stage_index = torch.tensor([step], device=device, dtype=torch.float32).expand(2 * num_residues, seq_len, 1)
            
            # Adaptive temperature
            temperature_iter = temperature * (0.8 + 0.2 * step / num_iter)
            cfg_iter = cfg
            
            sampled, _ = self.spectrum_diffloss.sample(z, temperature_iter, cfg_iter, stage_index)
            # sampled: (2*num_residues, seq_len, token_embed_dim)
            
            # Apply CFG，CFG 已经在 diffloss.sample 内部完成
            sampled_cfg, _ = sampled.chunk(2, dim=0)  # (num_residues, seq_len, D) # 只取前半部分
            
            # Update tokens at current step
            tokens = sampled_cfg.clone() 
            # 更新：应该更新所有频率，不只是当前 step；因为有频率插值，需要完全替换 tokens
            # tokens[:, step, :] = sampled_cfg[:, step, :]
        
        # Reshape to (num_residues, 3, seq_len, spectrum_token_dim)
        # spectrum = tokens.reshape(num_residues, seq_len, 3, self.spectrum_token_dim)
        # spectrum = spectrum.permute(0, 2, 1, 3)  
        spectrum = self.recover_spectrum_from_tokens(tokens) # (num_residues, seq_len, 6, spectrum_token_dim) or (num_residues, seq_len, 3, spectrum_token_dim)
        return spectrum

    @torch.no_grad()
    def sample_nomask(self,
                     coords,
                     rotation_matrices, 
                     aatype,
                     aa_mask,
                     esm_features,
                     bihito_token_repr: Optional[torch.Tensor] = None,
                     num_iter=None,
                     cfg=1.0,
                     temperature=1.0,
                     progress=False,
                     freq_schedule=None):
        """
        Sample spectrum autoregressively without masking (FAR-style nomask sampling)
        
        Similar to FAR's sample_tokens_nomask method:
        - No masking mechanism
        - Direct frequency progression with interpolation
        - Faster and simpler than mask-based sampling
        
        Args:
            coords: (num_residues, 3)
            rotation_matrices: (num_residues, 3, 3)
            aatype: (num_residues,)
            aa_mask: (num_residues,)
            esm_features: (num_residues, 1280)
            num_iter: number of iterations (default: use freq_schedule length)
            cfg: classifier-free guidance scale
            temperature: sampling temperature
            progress: show progress bar
            freq_schedule: list of frequency levels to generate (e.g., [0, 2, 3, 4, 5, ...])
        
        Returns:
            spectrum:generated spectrum (num_residues, seq_len, 6, spectrum_token_dim) or (num_residues, seq_len, 3, spectrum_token_dim) 
        """
        num_residues = coords.shape[0]
        device = coords.device
        
        # Default frequency schedule (from low to high frequency)
        if freq_schedule is None:
            if num_iter is None:
                raise ValueError("Either num_iter or freq_schedule must be specified")
            # Create a simple sequential frequency schedule: [0, 1, 2, 3, ..., num_iter-1]
            # This ensures we generate all num_iter frequencies
            freq_schedule = list(range(0, num_iter))
        
        num_iter = len(freq_schedule)
        max_freq = max(freq_schedule)
        # seq_len should be the number of frequencies to generate
        # For a sequential schedule [0, 1, 2, ..., N-1], seq_len = N = num_iter
        seq_len = num_iter
        
        # Encode conditions
        s_s, s_z = self.feature_extractor(aatype, esm_features, aa_mask, coords, rotation_matrices)
        system_condition = self.IPA_encoder(s_s, s_z, coords, rotation_matrices, aa_mask.to(dtype=s_s.dtype))
        if bihito_token_repr is not None:
            s_s = s_s + self.bihito_proj(bihito_token_repr)
        # Initialize tokens (all zeros - lowest frequency)
        tokens = torch.zeros(num_residues, seq_len, self.token_embed_dim, 
                           device=device, dtype=torch.float32)
        
        # Iterative generation
        indices = list(range(num_iter))
        if progress:
            indices = tqdm(indices, desc="Generating spectrum (nomask)")
        
        for step in indices:
            current_freq_level = freq_schedule[step]
            
            # === CFG: Prepare conditional and unconditional inputs ===
            tokens_doubled = torch.cat([tokens, tokens], dim=0)  # (2*num_residues, seq_len, D)
            system_condition_doubled = torch.cat([
                system_condition,
                self.fake_system_condition.expand(num_residues, -1)
            ], dim=0)  # (2*num_residues, c_s)
            s_s_doubled = torch.cat([
                s_s,
                self.fake_condition_residue.expand(num_residues, -1)
            ], dim=0)  # (2*num_residues, c_s)
            
            # === Forward through encoder and decoder (no mask) ===
            encoded = self.forward_encoder(tokens_doubled,
                                          s_s_doubled,
                                          system_condition_doubled,
                                          2 * num_residues,  # doubled num_residues
                                          seq_len,
                                          mask=None)  # No masking
            z = self.forward_decoder(encoded, 2 * num_residues, seq_len, mask=None)
            
            # z: (2*num_residues, seq_len, decoder_embed_dim)
            # Note: Keep z as 3D tensor (do NOT flatten) for FrequencyComponentDenoiseNet
            
            # === Temperature and CFG scheduling ===
            # Temperature: low at start (deterministic structure), high at end (diverse details)
            temperature_iter = 0.8 + (1 - np.cos(math.pi / 2. * (step + 1) / num_iter)) * (temperature - 0.8)
            
            # CFG: gradually increase from 1 to cfg
            cfg_iter = 1 + (cfg - 1) * step / num_iter
            
            # === Frequency index conditioning ===
            # Normalize frequency level to [0, 1] range
            freq_index_normalized = torch.tensor([current_freq_level / max(max_freq, 1)], 
                                                device=device, dtype=torch.float32)
            # freq_index shape should match z: (2*num_residues, seq_len, 1)
            freq_index = freq_index_normalized.expand(2 * num_residues, seq_len, 1)
            
            # === Diffusion sampling ===
            # Pass 3D tensors to match training behavior
            sampled, _ = self.spectrum_diffloss.sample(z, temperature_iter, cfg_iter, freq_index)
            # sampled: (2*num_residues, seq_len, token_embed_dim)
            
            # Apply CFG
            sampled_cfg, _ = sampled.chunk(2, dim=0)  # 只取有条件的部分
            # sampled_cfg: (num_residues, seq_len, D)
            # sampled_cond = sampled[:num_residues]  # (num_residues, seq_len, D)
            # sampled_uncond = sampled[num_residues:]  # (num_residues, seq_len, D)
            # sampled_cfg = sampled_uncond + cfg_iter * (sampled_cond - sampled_uncond)
            
            # === Frequency interpolation (key operation!) ===
            # Limit frequency content to the next level
            if step < num_iter - 1 and current_freq_level > 0:
                next_freq_level = freq_schedule[step + 1]
                
                # Reshape for interpolation: (num_residues, seq_len, D) -> (num_residues*D, seq_len)
                sampled_reshaped = sampled_cfg.permute(0, 2, 1)  # (num_residues, D, seq_len)
                flat_size = num_residues * self.token_embed_dim
                sampled_flat_interp = sampled_reshaped.reshape(flat_size, seq_len)
                
                # Downsample to next frequency level
                sampled_down = F.interpolate(
                    sampled_flat_interp.unsqueeze(1),  # (flat_size, 1, seq_len)
                    size=next_freq_level,
                    mode='linear',
                    align_corners=False
                )
                
                # Upsample back to original seq_len
                sampled_up = F.interpolate(
                    sampled_down,
                    size=seq_len,
                    mode='linear',
                    align_corners=False
                )
                
                # Reshape back
                sampled_up = sampled_up.squeeze(1).reshape(num_residues, self.token_embed_dim, seq_len)
                sampled_cfg = sampled_up.permute(0, 2, 1)  # (num_residues, seq_len, D)
            
            # === Autoregressive update ===
            tokens = sampled_cfg.clone()
        
        # Reshape to spectrum format (num_residues, 3, seq_len, 64)
        # spectrum = tokens.reshape(num_residues, seq_len, 3, self.spectrum_token_dim)
        # spectrum = spectrum.permute(0, 2, 1, 3)  # (num_residues, 3, seq_len, 64)
        spectrum = self.recover_spectrum_from_tokens(tokens) # (num_residues, seq_len, 6, spectrum_token_dim) or (num_residues, seq_len, 3, spectrum_token_dim)
        return spectrum

    @torch.no_grad()
    def sample_mask(self,
                   coords,
                   rotation_matrices,
                   aatype,
                   aa_mask,
                   esm_features,
                   bihito_token_repr: Optional[torch.Tensor] = None,
                   num_iter=None,
                   cfg=1.0,
                   temperature=1.0,
                   progress=False,
                   freq_schedule=None):
        """
        Sample spectrum autoregressively with masking (MaskGIT-style sampling)
        
        Similar to FAR's sample_tokens_mask method:
        - Uses masking mechanism to control generation progress
        - Gradually unmasks tokens following a cosine schedule
        - More flexible but potentially slower than nomask sampling
        
        Args:
            coords: (num_residues, 3)
            rotation_matrices: (num_residues, 3, 3)
            aatype: (num_residues,)
            aa_mask: (num_residues,)
            esm_features: (num_residues, 1280)
            num_iter: number of iterations
            cfg: classifier-free guidance scale
            temperature: sampling temperature
            progress: show progress bar
            freq_schedule: custom frequency schedule (optional)
        
        Returns:
            spectrum: generated spectrum (num_residues, seq_len, 6, spectrum_token_dim) or (num_residues, seq_len, 3, spectrum_token_dim) 
        """
        num_residues = coords.shape[0]
        device = coords.device
        
        # Default frequency schedule
        if freq_schedule is None:
            if num_iter is None:
                raise ValueError("Either num_iter or freq_schedule must be specified")
            # Use a schedule similar to FAR's mask sampling
            # [0, 12, 12, 4, 5, 6, 7, 8, 9, 10, 11] or simpler [0, 2, 3, 4, ...]
            freq_schedule = [0] + list(range(2, min(num_iter, 12)))
            if len(freq_schedule) < num_iter:
                freq_schedule = list(range(num_iter))
        
        num_iter = len(freq_schedule)
        max_freq = max(freq_schedule)
        seq_len = max_freq if max_freq > 0 else num_iter
        
        # Encode conditions
        s_s, s_z = self.feature_extractor(aatype, esm_features, aa_mask, coords, rotation_matrices)
        system_condition = self.IPA_encoder(s_s, s_z, coords, rotation_matrices, aa_mask.to(dtype=s_s.dtype))
        if bihito_token_repr is not None:
            s_s = s_s + self.bihito_proj(bihito_token_repr)
        # Initialize
        mask = torch.ones(seq_len, device=device, dtype=torch.bool)  # All masked initially
        tokens = torch.zeros(num_residues, seq_len, self.token_embed_dim, device=device)
        cur_tokens = torch.zeros_like(tokens)
        
        # Generation orders (random shuffle)
        orders = self.sample_orders(seq_len, device)
        
        # Iterative generation
        indices = list(range(num_iter))
        if progress:
            indices = tqdm(indices, desc="Generating spectrum (mask)")
        
        for step in indices:
            current_freq_level = freq_schedule[step]
            
            # === CFG: Prepare conditional and unconditional inputs ===
            tokens_doubled = torch.cat([tokens, tokens], dim=0)  # (2*num_residues, seq_len, D)
            system_condition_doubled = torch.cat([
                system_condition,
                self.fake_system_condition.expand(num_residues, -1)
            ], dim=0)  # (2*num_residues, c_s)
            s_s_doubled = torch.cat([
                s_s,
                self.fake_condition_residue.expand(num_residues, -1)
            ], dim=0)  # (2*num_residues, c_s)
            
            # Expand mask to all residues
            mask_expanded = mask.unsqueeze(0).expand(num_residues, -1)  # (num_residues, seq_len)
            mask_doubled = torch.cat([mask_expanded, mask_expanded], dim=0)  # (2*num_residues, seq_len)
            
            # === Forward through encoder and decoder (with mask) ===
            encoded = self.forward_encoder(tokens_doubled,
                                          s_s_doubled,
                                          system_condition_doubled,
                                          2 * num_residues,  # doubled num_residues
                                          seq_len,
                                          mask_doubled)
            z = self.forward_decoder(encoded, 2 * num_residues, seq_len, mask_doubled)
            
            # z: (2*num_residues, seq_len, decoder_embed_dim)
            # Note: Keep z as 3D tensor (do NOT flatten) for FrequencyComponentDenoiseNet
            
            # === Mask ratio for next round (cosine schedule) ===
            mask_ratio = np.cos(math.pi / 2. * (step + 1) / num_iter)
            mask_len = int(np.floor(seq_len * mask_ratio))
            
            # === Temperature and CFG scheduling ===
            temperature_iter = 0.85 + (temperature - 0.85) * step / num_iter
            cfg_iter = 1 + (cfg - 1) * step / num_iter
            
            # === Frequency index conditioning ===
            freq_index_normalized = torch.tensor([current_freq_level / max(max_freq, 1)],
                                                device=device, dtype=torch.float32)
            # freq_index shape should match z: (2*num_residues, seq_len, 1)
            freq_index = freq_index_normalized.expand(2 * num_residues, seq_len, 1)
            
            # === Diffusion sampling ===
            # Pass 3D tensors to match training behavior
            sampled, _ = self.spectrum_diffloss.sample(z, temperature_iter, cfg_iter, freq_index)
            # sampled: (2*num_residues, seq_len, token_embed_dim)
            
            # Apply CFG
            sampled_cfg, _ = sampled.chunk(2, dim=0)  # (num_residues, seq_len, D)
            # sampled_cond = sampled[:num_residues]  # (num_residues, seq_len, D)
            # sampled_uncond = sampled[num_residues:]  # (num_residues, seq_len, D)
            # sampled_cfg = sampled_uncond + cfg_iter * (sampled_cond - sampled_uncond)
            
            # === Frequency interpolation (if not last step) ===
            if step < num_iter - 1 and current_freq_level > 0:
                next_freq_level = freq_schedule[step + 1]
                
                # Reshape for interpolation
                sampled_reshaped = sampled_cfg.permute(0, 2, 1)  # (num_residues, D, seq_len)
                flat_size = num_residues * self.token_embed_dim
                sampled_flat_interp = sampled_reshaped.reshape(flat_size, seq_len)
                
                # Downsample -> Upsample
                sampled_down = F.interpolate(
                    sampled_flat_interp.unsqueeze(1),
                    size=next_freq_level,
                    mode='linear',
                    align_corners=False
                )
                sampled_up = F.interpolate(
                    sampled_down,
                    size=seq_len,
                    mode='linear',
                    align_corners=False
                )
                
                sampled_up = sampled_up.squeeze(1).reshape(num_residues, self.token_embed_dim, seq_len)
                sampled_cfg = sampled_up.permute(0, 2, 1)  # (num_residues, seq_len, D)
            
            # === Update mask and tokens ===
            sampled_token = sampled_cfg
            
            # Calculate next mask (based on order)
            mask_next = torch.zeros(seq_len, device=device, dtype=torch.bool)
            if mask_len > 0:
                mask_indices = orders[:mask_len]
                mask_next[mask_indices] = True
            
            # 计算这一步要预测的位置（unmasked 的）
            mask_to_pred = torch.logical_and(mask, torch.logical_not(mask_next))
            # mask_to_pred: (seq_len,) 表示哪些频率位置在这步被预测            
            
            # Extract tokens to predict
            # 将 mask_to_pred 扩展到所有 residues
            mask_to_pred_expanded = mask_to_pred.unsqueeze(0).expand(num_residues, -1)
            # mask_to_pred_expanded: (num_residues, seq_len)

            # 更新 tokens：只更新 mask_to_pred 的位置
            cur_tokens = tokens.clone()
            cur_tokens[mask_to_pred_expanded] = sampled_token[mask_to_pred_expanded]
            
            # Update tokens and mask
            tokens = cur_tokens
            mask = mask_next
        
        # Reshape to spectrum format (num_residues, 3, seq_len, spectrum_token_dim)
        # spectrum = tokens.reshape(num_residues, seq_len, 3, self.spectrum_token_dim)
        # spectrum = spectrum.permute(0, 2, 1, 3)  # (num_residues, 3, seq_len, spectrum_token_dim)
        spectrum = self.recover_spectrum_from_tokens(tokens) # (num_residues, seq_len, 6, spectrum_token_dim) or (num_residues, seq_len, 3, spectrum_token_dim)
        return spectrum


# Model construction functions
def protein_spectrum_far_base(**kwargs):
    """
    Base model
    
    To enable frequency-specific attention using SpectralDiTBlock, pass:
        use_hybrid_block=True,
        use_spectral_attention=True
    """
    # Set default values, but allow kwargs to override
    defaults = {
        'c_s': 1024,
        'c_z': 128,
        'c_ipa': 32,
        'no_heads_ipa': 8,
        'no_qk_points': 4,
        'no_v_points': 4,
        'num_ipa_layers': 1,
        'encoder_depth': 12,
        'encoder_num_heads': 16,
        'decoder_embed_dim': 768,
        'decoder_depth': 12,
        'decoder_num_heads': 12,
        'buffer_size': 32,
        'mlp_ratio': 4,
        'norm_layer': partial(nn.LayerNorm, eps=1e-6),
        # DIFFLOSS
        'num_sampling_steps': '100',
        'diffusion_batch_mul': 4,
        'diffloss_d': 3,
        'diffloss_w': 1024,
    }
    # Update defaults with any provided kwargs
    defaults.update(kwargs)
    
    model = ProteinSpectrumFAR(**defaults)
    return model


def protein_spectrum_far_large(**kwargs):
    """Large model"""
    # Set default values, but allow kwargs to override
    defaults = {
        'c_s': 1024,
        'c_z': 128,
        'c_ipa': 32,
        'no_heads_ipa': 8,
        'no_qk_points': 4,
        'no_v_points': 4,
        'num_ipa_layers': 1,
        'encoder_depth': 16,
        'encoder_num_heads': 16,
        'decoder_embed_dim': 1024,
        'decoder_depth': 16,
        'decoder_num_heads': 16,
        'buffer_size': 64,
        'mlp_ratio': 4,
        'norm_layer': partial(nn.LayerNorm, eps=1e-6),
        'num_sampling_steps': '100',
        'diffusion_batch_mul': 4,
        'diffloss_d': 3,
        'diffloss_w': 1024,
    }
    # Update defaults with any provided kwargs
    defaults.update(kwargs)
    
    model = ProteinSpectrumFAR(**defaults)
    return model


def protein_spectrum_far_huge(**kwargs):
    """Huge model"""
    # Set default values, but allow kwargs to override
    defaults = {
        'c_s': 1024,
        'c_z': 128,
        'c_ipa': 32,
        'no_heads_ipa': 8,
        'no_qk_points': 4,
        'no_v_points': 4,
        'num_ipa_layers': 1,
        'encoder_depth': 20,
        'encoder_num_heads': 12,
        'decoder_embed_dim': 1280,
        'decoder_depth': 20,
        'decoder_num_heads': 16,
        'buffer_size': 128,
        'mlp_ratio': 4,
        'norm_layer': partial(nn.LayerNorm, eps=1e-6),
        # diffloss
        'num_sampling_steps': '100',
        'diffusion_batch_mul': 4,
        'diffloss_d': 3,
        'diffloss_w': 1024,
    }
    # Update defaults with any provided kwargs
    defaults.update(kwargs)
    
    model = ProteinSpectrumFAR(**defaults)
    return model


# ============================================================================
# Multi-sample helper functions
# ============================================================================

@torch.no_grad()
def sample_multiple(model, 
                    coords,
                    rotation_matrices,
                    aatype,
                    aa_mask,
                    esm_features,
                    bihito_token_repr: Optional[torch.Tensor] = None,
                    n_samples=1,
                    num_iter=None,
                    cfg=1.0,
                    temperature=1.0,
                    progress=False,
                    sample_method='nomask',
                    seed=None,
                    **kwargs):
    """
    Generate multiple samples from the model with different random seeds.
    
    This function calls the sampling method N_sample times to generate
    multiple predictions for the same input structure.
    
    Args:
        model: ProteinSpectrumFAR model
        coords: (num_residues, 3) coordinates
        rotation_matrices: (num_residues, 3, 3) rotation matrices
        aatype: (num_residues,) amino acid types
        aa_mask: (num_residues,) residue mask
        esm_features: (num_residues, 1280) ESM embeddings
        n_samples: number of samples to generate
        num_iter: number of iterations for sampling
        cfg: classifier-free guidance scale
        temperature: sampling temperature
        progress: show progress bar
        sample_method: 'sample', 'nomask', or 'mask'
        seed: base random seed (if None, use current random state)
        **kwargs: additional arguments for sampling method
        
    Returns:
        spectra: (n_samples, num_residues, seq_len, 3, spectrum_token_dim) generated spectra
    """
    # Get the sampling function
    if sample_method == 'sample':
        sample_fn = model.sample
    elif sample_method == 'nomask':
        sample_fn = model.sample_nomask
    elif sample_method == 'mask':
        sample_fn = model.sample_mask
    else:
        raise ValueError(f"Unknown sample_method: {sample_method}. Choose from 'sample', 'nomask', 'mask'")
    
    samples = []
    
    # Optionally wrap with progress bar
    iterator = range(n_samples)
    if progress and n_samples > 1:
        iterator = tqdm(iterator, desc=f"Generating {n_samples} samples")
    
    for i in iterator:
        # Set random seed for this sample (if provided)
        if seed is not None:
            torch.manual_seed(seed + i)
            if torch.cuda.is_available():
                torch.cuda.manual_seed(seed + i)
                torch.cuda.manual_seed_all(seed + i)
            np.random.seed(seed + i)
        
        # Generate one sample
        spectrum = sample_fn(
            coords=coords,
            rotation_matrices=rotation_matrices,
            aatype=aatype,
            aa_mask=aa_mask,
            esm_features=esm_features,
            num_iter=num_iter,
            cfg=cfg,
            temperature=temperature,
            progress=False,  # Don't show inner progress bar
            **kwargs
        )
        
        samples.append(spectrum)
    
    # Stack all samples: (n_samples, num_residues, 3, seq_len, spectrum_token_dim)
    spectra = torch.stack(samples, dim=0)
    
    return spectra

