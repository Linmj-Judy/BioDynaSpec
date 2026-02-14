import torch
import torch.nn as nn
import math
from typing import Optional
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


###########################

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
        Supports:
            x: (N, C) or (num_residues, seq_len, C)
            kv_cache: optional dict with keys {"k_cache", "v_cache", "cache_seqlens", "cache_batch_idx"}
            causal: whether to apply causal masking
        Output:
            shape matches input
        """
        orig_shape = x.shape
        squeeze_out = False

        if x.dim() == 2:
            # (N, C) → (1, N, C), return (N, C)
            x_in = x.unsqueeze(0)
            squeeze_out = True
        elif x.dim() == 3:
            # (B, N, C), usually (num_residues, seq_len, C) → use as is, return (B, N, C)
            x_in = x
        else:
            raise ValueError(f"Input x must be (N, C) or (B, N, C), got shape {x.shape}")

        B, N, C = x_in.shape
        qkv_linear = self.qkv(x_in)  # (B, N, 3*C)

        if _flash_attn_available:
            orig_dtype = qkv_linear.dtype
            if orig_dtype not in (torch.bfloat16, torch.float16):
                print(f"Flash Attention QKV linear dtype must be bfloat16 or float16, but got {orig_dtype}, turning to bfloat16")
                qkv_bf16 = qkv_linear.to(torch.bfloat16)
                # (B, N, 3, H, Hd)
                qkv = qkv_bf16.reshape(B, N, 3, self.num_heads, self.head_dim)
            else:
                qkv = qkv_linear.reshape(B, N, 3, self.num_heads, self.head_dim)

            out = None
            if kv_cache is not None and _flash_attn_with_kvcache_available:
                # KV cache path (currently only supports batch_size=1)
                if B != 1:
                    raise ValueError("KV cache attention path currently only supports batch size = 1")
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
                )   # (B, N, H, Hd)

            # back to original dtype and shape
            if orig_dtype not in (torch.bfloat16, torch.float16):
                x_attn = out.to(orig_dtype).reshape(B, N, C)
            else:
                x_attn = out.reshape(B, N, C)
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

        if squeeze_out:
            return x_proj.squeeze(0)
        else:
            return x_proj
####################

