
import torch
import torch.utils.checkpoint as ckpt
import torch.nn as nn
import math
import warnings

from typing import Optional
from typing import List
from functools import partial
from src.model.modules.attention import Mlp
from src.openfold_local.utils.chunk_utils import chunk_layer
from src.openfold_local.model.primitives import LayerNorm
from src.openfold_local.model.primitives import Attention as OpenFoldAttention
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


def modulate(x, shift, scale):
    """
    Memory‑friendlier modulation:
        y = x * (1 + scale) + shift
    尽量减少中间大张量的创建。
    """
    # shift/scale 只在本次调用中使用，可以安全地原地修改
    if scale is not None:
        scale = 1 + scale
        x = x * scale
    if shift is not None:
        x = x + shift
    return x

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


#################################################################################
#               Embedding Layers for Timesteps                                  #
#################################################################################

class TimestepEmbedder(nn.Module):
    """
    Embeds scalar timesteps into vector representations.
    """
    def __init__(self, hidden_size, frequency_embedding_size=256):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(frequency_embedding_size, hidden_size, bias=True),
            nn.SiLU(),
            nn.Linear(hidden_size, hidden_size, bias=True),
        )
        self.frequency_embedding_size = frequency_embedding_size

    @staticmethod
    def timestep_embedding(t, dim, max_period=10000):
        """
        Create sinusoidal timestep embeddings.
        :param t: a 1-D Tensor of N indices, one per batch element.
                          These may be fractional.
        :param dim: the dimension of the output.
        :param max_period: controls the minimum frequency of the embeddings.
        :return: an (N, D) Tensor of positional embeddings.
        """
        # https://github.com/openai/glide-text2im/blob/main/glide_text2im/nn.py
        half = dim // 2
        freqs = torch.exp(
            -math.log(max_period) * torch.arange(start=0, end=half, dtype=torch.float32) / half
        ).to(device=t.device)
        args = t[:, None].float() * freqs[None]
        embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
        if dim % 2:
            embedding = torch.cat([embedding, torch.zeros_like(embedding[:, :1])], dim=-1)
        return embedding

    def forward(self, t):
        t_freq = self.timestep_embedding(t, self.frequency_embedding_size)
        t_emb = self.mlp(t_freq)
        return t_emb

#################################################################################
#                                 Core DiT Model                                #
#################################################################################

class ResidueDiTBlock(nn.Module):
    """
    A DiT block with adaptive layer norm zero (adaLN-Zero) conditioning,
    modified to apply attention along the num_residues dimension, i.e., treat
    residue as the "sequence" dimension for attention.
    """
    def __init__(self, hidden_size, num_heads, mlp_ratio=4.0, **block_kwargs):
        super().__init__()
        # layer norms now over hidden_size
        self.norm1 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        # Don't change attn, but will reshape the data
        self.attn = Attention(hidden_size, num_heads=num_heads, qkv_bias=True, **block_kwargs)
        self.norm2 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        mlp_hidden_dim = int(hidden_size * mlp_ratio)
        approx_gelu = lambda: nn.GELU(approximate="tanh")
        self.mlp = Mlp(in_features=hidden_size, hidden_features=mlp_hidden_dim, act_layer=approx_gelu, drop=0)
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(hidden_size, 6 * hidden_size, bias=True)
        )

    def _forward_chunk(self, x, c):
        '''
        inputs:
            x: (num_residues, seq_len, D)
            c: (num_residues, seq_len, D)
        outputs:
            (num_residues, seq_len, D)    
        '''
        # Compute AdaLN modulations
        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = self.adaLN_modulation(c).chunk(6, dim=-1)
        # For attention over num_residues, we transpose to (seq_len, num_residues, D)
        x_res = x.transpose(0, 1)        # (seq_len, num_residues, D)
        shift_msa_res = shift_msa.transpose(0, 1)
        scale_msa_res = scale_msa.transpose(0, 1)
        gate_msa_res = gate_msa.transpose(0, 1)
        # LayerNorm and modulation
        mod_x_res = modulate(self.norm1(x_res), shift_msa_res, scale_msa_res)    # (seq_len, num_residues, D)
        # Attention along num_residues (for each seq_len separately)
        attn_out_res = self.attn(mod_x_res)                      # (seq_len, num_residues, D)
        # Gated residual
        x_res = x_res + gate_msa_res * attn_out_res              # (seq_len, num_residues, D)
        # MLP branch (apply along last dimension as normal)
        shift_mlp_res = shift_mlp.transpose(0, 1)
        scale_mlp_res = scale_mlp.transpose(0, 1)
        gate_mlp_res = gate_mlp.transpose(0, 1)
        mod_x2_res = modulate(self.norm2(x_res), shift_mlp_res, scale_mlp_res)
        mlp_out_res = self.mlp(mod_x2_res)
        x_res = x_res + gate_mlp_res * mlp_out_res
        # Transpose back to (num_residues, seq_len, D)
        x = x_res.transpose(0, 1)
        return x

    def forward(self, x, c, *, seq_chunk_size: Optional[int] = None):
        """
        inputs:
            x: (num_residues, seq_len, D)
            c: (num_residues, seq_len, D)
            seq_chunk_size: optional chunk size over seq_len to reduce memory
        """
        if seq_chunk_size is None:
            return self._forward_chunk(x, c)
        seq_len = x.shape[1]
        if seq_len <= seq_chunk_size:
            return self._forward_chunk(x, c)
        chunks = []
        for start in range(0, seq_len, seq_chunk_size):
            end = min(start + seq_chunk_size, seq_len)
            x_chunk = x[:, start:end, :]
            c_chunk = c[:, start:end, :]
            chunks.append(self._forward_chunk(x_chunk, c_chunk))
        return torch.cat(chunks, dim=1)

class SpectralDiTBlock(nn.Module):
    """
    A DiT block with adaptive layer norm zero (adaLN-Zero) conditioning,
    modified to apply attention along the second dimension of the input tensor,
    i.e., treat the second dimension as the "sequence" dimension for attention.
    Now supports chunking on the N_atom dimension (no attention on N_atom, so chunking is safe here).
    支持condition为None的情况，此时不做AdaLN调制，退化为普通Transformer block。
    """
    def __init__(self, hidden_size, num_heads, mlp_ratio=4.0, norm_layer=LayerNorm, qkv_bias = True, attn_drop=0.0, proj_drop=0.0):
        super().__init__()
        self.inf = 1e6
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        # layer norms now over hidden_size
        self.norm1 = norm_layer(hidden_size, eps=1e-6)
        
        # 检查 DeepSpeed EvoformerAttention 的限制
        self.c_hidden = hidden_size // num_heads
        if self.c_hidden > 64:
            import warnings
            warnings.warn(
                f"SpectralDiTBlock: c_hidden ({self.c_hidden}) > 64. "
                f"DeepSpeed EvoformerAttention requires c_hidden <= 64. "
                f"If use_deepspeed_evo_attention=True is used, it will be automatically disabled. "
                f"Consider increasing num_heads to at least {math.ceil(hidden_size / 64)} to use DeepSpeed EvoformerAttention.",
                UserWarning
            )
        
        # Use Protenix Attention implementation
        self.attn = OpenFoldAttention(
            c_q=hidden_size,
            c_k=hidden_size,
            c_v=hidden_size,
            c_hidden=self.c_hidden,
            no_heads=num_heads,
            gating=True,
        )
        self.norm2 = norm_layer(hidden_size, eps=1e-6)
        mlp_hidden_dim = int(hidden_size * mlp_ratio)
        approx_gelu = lambda: nn.GELU(approximate="tanh")
        self.mlp = Mlp(in_features=hidden_size, hidden_features=mlp_hidden_dim, act_layer=approx_gelu, drop=0)
        # 拆分为两套 AdaLN MLP，避免一次性产生 6*hidden 的大激活
        self.adaLN_msa = nn.Sequential(
            nn.SiLU(),
            nn.Linear(hidden_size, 3 * hidden_size, bias=True)
        )
        self.adaLN_mlp = nn.Sequential(
            nn.SiLU(),
            nn.Linear(hidden_size, 3 * hidden_size, bias=True)
        )
        
    def glorot_init(self):
        nn.init.xavier_uniform_(self.attn.linear_q.weight)
        nn.init.xavier_uniform_(self.attn.linear_k.weight)
        nn.init.xavier_uniform_(self.attn.linear_v.weight)
        nn.init.zeros_(self.attn.linear_q.bias)

    @torch.jit.ignore
    def _chunk(
        self,
        x: torch.Tensor,
        biases: List[torch.Tensor],
        chunk_size: int,
        use_memory_efficient_kernel: bool = False,
        use_deepspeed_evo_attention: bool = False,
        use_lma: bool = False,
        inplace_safe: bool = False,
        use_megatron: bool = False,
        use_sequence_parallel: bool = False,
    ) -> torch.Tensor:
        mha_inputs = {
            "q_x": x,
            "kv_x": x,
            "biases": biases,
        }

        return chunk_layer(
            partial(
                self.attn,
                use_memory_efficient_kernel=use_memory_efficient_kernel,
                use_deepspeed_evo_attention=use_deepspeed_evo_attention,
                use_lma=use_lma,
                use_megatron=use_megatron,
                use_sequence_parallel=use_sequence_parallel,
            ),
            mha_inputs,
            chunk_size=chunk_size,
            no_batch_dims=len(x.shape[:-2]),
            _out=x if inplace_safe else None,
        )

    def forward(self, x, c=None, chunk_size: int = None, attn_chunk_size: int = None, use_megatron: bool = False, use_sequence_parallel: bool = False):
        '''
        hidden_size = c_atom
        inputs:
            x: (..., N_atom, seq_len, c_atom)
            c(atom_condition): (..., N_atom, c_atom) or None
            chunk_size: if not None, chunks input on N_atom for reduced memory

        outputs:
            x: (..., N_atom, seq_len, c_atom) 
        '''
        # Handle chunking along N_atom
        n_atoms = x.shape[-3]
        if chunk_size is not None and n_atoms > chunk_size:
            # chunk along N_atom, process each chunk and concat output
            out_chunks = []
            for start in range(0, n_atoms, chunk_size):
                end = min(start + chunk_size, n_atoms)
                x_chunk = x[..., start:end, :, :]  # (..., <=chunk_size, seq_len, c_atom)
                c_chunk = c[..., start:end, :] if c is not None else None  # (..., <=chunk_size, c_atom) or None
                out_chunk = self._forward_no_chunk(x_chunk, c=c_chunk, 
                attn_chunk_size=attn_chunk_size,
                use_megatron=use_megatron,
                use_sequence_parallel=use_sequence_parallel,
                )
                out_chunks.append(out_chunk)
            x = torch.cat(out_chunks, dim=-3)
            return x
        else:
            return self._forward_no_chunk(x, c=c, attn_chunk_size=attn_chunk_size,
            use_megatron=use_megatron,
            use_sequence_parallel=use_sequence_parallel,
            )

    def _forward_no_chunk(self, 
        x, c=None,
        mask: Optional[torch.Tensor] = None,
        attn_chunk_size: int = None,
        use_memory_efficient_kernel: bool = False,
        use_deepspeed_evo_attention: bool = True,
        use_lma: bool = False,
        use_megatron: bool = False,
        use_sequence_parallel: bool = False,
    ):
        '''
        Non-chunked forward logic, 支持c为None
        '''
        # 检查 DeepSpeed EvoformerAttention 的限制：c_hidden <= 64
        # c_hidden = hidden_size // num_heads
        if use_deepspeed_evo_attention and self.c_hidden > 64:
            # DeepSpeed EvoformerAttention kernel 要求 c_hidden <= 64
            # 自动回退到标准 PyTorch attention
            warnings.warn(
                f"DeepSpeed EvoformerAttention requires c_hidden <= 64, but got {self.c_hidden} "
                f"(hidden_size={self.hidden_size}, num_heads={self.num_heads}). "
                f"Automatically disabling use_deepspeed_evo_attention and falling back to standard attention.",
                UserWarning
            )
            use_deepspeed_evo_attention = False
        
        # DeepSpeed EvoformerAttention 对序列长度也有要求：seq_len > 16
        # progressive_masking 可能导致有效序列长度仅为 buffer_size + stage，
        # 小于等于 16 时强制回退到标准注意力以避免 assertion
        seq_len_effective = x.shape[-2]
        if use_deepspeed_evo_attention and seq_len_effective <= 16:
            warnings.warn(
                f"DeepSpeed EvoformerAttention requires seq_len > 16, but got {seq_len_effective} "
                f"Automatically disabling use_deepspeed_evo_attention and falling back to standard attention.",
                UserWarning
            )
            use_deepspeed_evo_attention = False
        
        # seq_len = x.shape[-2]
        if mask is None:
            # [*, I, J]
            mask = x.new_ones(
                x.shape[:-1],
            )
        # [*, I, 1, 1, J] —— broadcast 到 attention 所需形状
        mask_bias = (self.inf * (mask - 1))[..., None, None, :].to(x.dtype)
        del mask

        if c is None:
            # 无条件时，就是标准Transformer Block
            x1 = self.norm1(x)
            biases = [] if use_deepspeed_evo_attention else [mask_bias]

            if attn_chunk_size is not None:
                x1 = self._chunk(
                    x1,
                    biases,
                    attn_chunk_size,
                    use_memory_efficient_kernel=use_memory_efficient_kernel,
                    use_deepspeed_evo_attention=use_deepspeed_evo_attention,
                    use_lma=use_lma,
                    use_megatron=use_megatron,
                    use_sequence_parallel=use_sequence_parallel,
                    inplace_safe=False,
                )
                del biases
            else:
                x1 = self.attn(
                    q_x=x1,
                    kv_x=x1,
                    biases=biases,
                    use_memory_efficient_kernel=use_memory_efficient_kernel,
                    use_deepspeed_evo_attention=use_deepspeed_evo_attention,
                    use_lma=use_lma,
                    use_megatron=use_megatron,
                    use_sequence_parallel=use_sequence_parallel,
                )
                del biases
            x = x + x1
            del x1

            x2 = self.norm2(x) # (..., N_atom, seq_len, hidden)
            del x
            out = x2 + self.mlp(x2)
            del x2
            return out
        else:
            # ---------- MSA 分支 ----------
            mods_msa = self.adaLN_msa(c)  # (..., N_atom, 3*hidden)
            mods_msa = mods_msa.unsqueeze(-2).contiguous()  # (..., N_atom, 1, 3*hidden)
            shift_msa, scale_msa, gate_msa = mods_msa.chunk(3, dim=-1) # 3个(..., N_atom, 1, hidden)
            del mods_msa
            scale_msa = 0.5 * torch.tanh(scale_msa)      # |1 + scale| 不会太大
            gate_msa  = torch.tanh(gate_msa)  # tanh ∈ [-1,1]

            x_norm1 = self.norm1(x)
            x_msa_in = modulate(x_norm1, shift_msa, scale_msa) # (..., N_atom, seq_len, hidden)
            del x_norm1, shift_msa, scale_msa
            biases = [] if use_deepspeed_evo_attention else [mask_bias]

            if attn_chunk_size is not None:
                x_msa_in = self._chunk(
                    x_msa_in,
                    biases,
                    attn_chunk_size,
                    use_memory_efficient_kernel=use_memory_efficient_kernel,
                    use_deepspeed_evo_attention=use_deepspeed_evo_attention,
                    use_lma=use_lma,
                    inplace_safe=False,
                )
                del biases
            else:
                x_msa_in = self.attn(
                    q_x=x_msa_in,
                    kv_x=x_msa_in,
                    biases=biases,
                    use_memory_efficient_kernel=use_memory_efficient_kernel,
                    use_deepspeed_evo_attention=use_deepspeed_evo_attention,
                    use_lma=use_lma,
                )
                del biases
            x = x + gate_msa * x_msa_in
            del gate_msa, x_msa_in

            # ---------- MLP 分支 ----------
            mods_mlp = self.adaLN_mlp(c)  # (..., N_atom, 3*hidden)
            mods_mlp = mods_mlp.unsqueeze(-2).contiguous()
            shift_mlp, scale_mlp, gate_mlp = mods_mlp.chunk(3, dim=-1) # 3个(..., N_atom, 1, hidden)
            del mods_mlp
            scale_mlp = 0.5 * torch.tanh(scale_mlp)
            gate_mlp  = torch.tanh(gate_mlp)

            x_norm2 = self.norm2(x)
            x_mlp_in = modulate(x_norm2, shift_mlp, scale_mlp)
            del x_norm2, shift_mlp, scale_mlp
            x = x + gate_mlp * self.mlp(x_mlp_in)
            return x


class ResBlock(nn.Module):
    """
    A residual block that can optionally change the number of channels.
    :param channels: the number of input channels.
    """

    def __init__(
        self,
        channels
    ):
        super().__init__()
        self.channels = channels

        self.in_ln = nn.LayerNorm(channels, eps=1e-6)
        self.mlp = nn.Sequential(
            nn.Linear(channels, channels, bias=True),
            nn.SiLU(),
            nn.Linear(channels, channels, bias=True),
        )

        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(channels, 3 * channels, bias=True)
        )

    def forward(self, x, y):
        shift_mlp, scale_mlp, gate_mlp = self.adaLN_modulation(y).chunk(3, dim=-1)
        h = modulate(self.in_ln(x), shift_mlp, scale_mlp)
        h = self.mlp(h)
        return x + gate_mlp * h


class FinalLayer(nn.Module):
    """
    The final layer of DiT.
    """
    def __init__(self, model_channels, out_channels):
        super().__init__()
        self.norm_final = nn.LayerNorm(model_channels, elementwise_affine=False, eps=1e-6)
        self.linear = nn.Linear(model_channels, out_channels, bias=True)
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(model_channels, 2 * model_channels, bias=True)
        )

    def forward(self, x, c):
        shift, scale = self.adaLN_modulation(c).chunk(2, dim=-1)
        x = modulate(self.norm_final(x), shift, scale)
        x = self.linear(x)
        return x


# class FrequencyComponentDenoiseNet(nn.Module):
#     """
#     The DiT model for single frequency component denoising in DiffLoss.
#     :param in_channels: channels in the input Tensor.
#     :param model_channels: base channel count for the model.
#     :param out_channels: channels in the output Tensor.
#     :param z_channels: channels in the condition.
#     :param num_res_blocks: number of residual blocks per downsample.
#     """

#     def __init__(
#         self,
#         in_channels,
#         model_channels,
#         out_channels,
#         z_channels,
#         num_dit_blocks=28,
#         num_heads=16,
#         mlp_ratio=4.0,
#         seq_chunk_size: Optional[int] = None,
#         block_type: str = "residue",
#         use_checkpoint: bool = False,
#     ):
#         super().__init__()

#         self.in_channels = in_channels
#         self.model_channels = model_channels
#         self.out_channels = out_channels
#         self.num_dit_blocks = num_dit_blocks
#         self.seq_chunk_size = seq_chunk_size
#         self.block_type = block_type
#         self.use_checkpoint = use_checkpoint

#         self.time_embed = TimestepEmbedder(model_channels)
#         self.cond_embed = nn.Linear(z_channels, model_channels)
#         self.index_cond_embed = nn.Linear(1, model_channels)

#         self.input_proj = nn.Linear(in_channels, model_channels)

#         dit_blocks = []
#         if block_type not in ("residue", "mlp"):
#             raise ValueError(f"Unsupported block_type={block_type}, expected 'residue' or 'mlp'")
#         for _ in range(num_dit_blocks):
#             if block_type == "residue":
#                 dit_blocks.append(ResidueDiTBlock(
#                     model_channels, num_heads, mlp_ratio=mlp_ratio
#                 ))
#             else:
#                 dit_blocks.append(ResBlock(model_channels))

#         self.dit_blocks = nn.ModuleList(dit_blocks)
#         self.final_layer = FinalLayer(model_channels, out_channels)

#         self.initialize_weights()

#     def initialize_weights(self):
#         # Initialize transformer layers:
#         def _basic_init(module):
#             if isinstance(module, nn.Linear):
#                 torch.nn.init.xavier_uniform_(module.weight)
#                 if module.bias is not None:
#                     nn.init.constant_(module.bias, 0)
#         self.apply(_basic_init)

#         # Initialize timestep embedding MLP
#         nn.init.normal_(self.time_embed.mlp[0].weight, std=0.02)
#         nn.init.normal_(self.time_embed.mlp[2].weight, std=0.02)

#         # Zero-out adaLN modulation layers
#         for block in self.dit_blocks:
#             nn.init.constant_(block.adaLN_modulation[-1].weight, 0)
#             nn.init.constant_(block.adaLN_modulation[-1].bias, 0)

#         # Zero-out output layers
#         nn.init.constant_(self.final_layer.adaLN_modulation[-1].weight, 0)
#         nn.init.constant_(self.final_layer.adaLN_modulation[-1].bias, 0)
#         nn.init.constant_(self.final_layer.linear.weight, 0)
#         nn.init.constant_(self.final_layer.linear.bias, 0)

#     def ckpt_wrapper(self, module):
#         def ckpt_forward(*inputs):
#             outputs = module(*inputs)
#             return outputs
#         return ckpt_forward

#     def _run_blocks(self, x, y):
#         for block in self.dit_blocks:
#             if self.use_checkpoint and self.training:
#                 if isinstance(block, ResidueDiTBlock):
#                     def _blk(x_in, y_in):
#                         return block(x_in, y_in, seq_chunk_size=self.seq_chunk_size)
#                     x = ckpt.checkpoint(_blk, x, y, use_reentrant=False)
#                 else:
#                     x = ckpt.checkpoint(block, x, y, use_reentrant=False)
#             else:
#                 if isinstance(block, ResidueDiTBlock):
#                     x = block(x, y, seq_chunk_size=self.seq_chunk_size)
#                 else:
#                     x = block(x, y)
#         return x

#     def forward(self, x_t, t, condition, index=None):
#         """
#         Apply the model to an input batch.
#         :param x_t (target): x_t(random noise), shape: (num_residues, seq_len, decoder_embed_dim)
#         :param t: a 1-D batch of timesteps. shape: (1, )
#         :param c (z): conditioning from AR transformer, shape: (num_residues, seq_len, decoder_embed_dim)
#         :return: an [N x C x ...] Tensor of outputs.
#         """
#         t_embed = self.time_embed(t) # (1, ) -> (1, model_channels)

#         # For MLP blocks, chunk along seq_len to reduce peak activation memory.
#         if self.seq_chunk_size is not None and self.block_type == "mlp":
#             seq_len = x_t.shape[1]
#             if seq_len > self.seq_chunk_size:
#                 outputs = []
#                 for start in range(0, seq_len, self.seq_chunk_size):
#                     end = min(start + self.seq_chunk_size, seq_len)
#                     x_chunk = self.input_proj(x_t[:, start:end, :])
#                     cond_chunk = self.cond_embed(condition[:, start:end, :])
#                     if index is not None:
#                         idx_chunk = self.index_cond_embed(index[:, start:end, :])
#                         y_chunk = t_embed + cond_chunk + idx_chunk
#                     else:
#                         y_chunk = t_embed + cond_chunk
#                     x_chunk = self._run_blocks(x_chunk, y_chunk)
#                     outputs.append(self.final_layer(x_chunk, y_chunk))
#                 return torch.cat(outputs, dim=1)

#         x = self.input_proj(x_t) # (num_residues, seq_len, token_embed_dim) -> (num_residues, seq_len, model_channels)
#         condition = self.cond_embed(condition) # (num_residues, seq_len, decoder_embed_dim) -> (num_residues, seq_len, model_channels)
#         if index is not None:
#             index = self.index_cond_embed(index) # (num_residues, seq_len, 1) -> (num_residues, seq_len, model_channels)
#             y = t_embed + condition + index # (num_residues, seq_len, model_channels)
#         else:
#             y = t_embed + condition

#         x = self._run_blocks(x, y)
#         return self.final_layer(x, y) # (N, T, out_channels)

#     def forward_with_cfg(self, x, t, condition, cfg_scale, index):
#         # # 这里的half是输入x的一半。通常x包含了两组输入：前一半是带条件的，后一半是不带条件的。
#         # # 这里取x的前一半（x[:len(x) // 2]），即half，然后cat两次得到一个两倍batch，
#         # # 用于分别做有条件和无条件的推理。
#         # half = x[: len(x) // 2]
#         # combined = torch.cat([half, half], dim=0)
#         # model_out = self.forward(combined, t, condition, index) # (2N, T, D)
#         # eps, rest = model_out[:, :self.in_channels], model_out[:, self.in_channels:]
#         # cond_eps, uncond_eps = torch.split(eps, len(eps) // 2, dim=0)
#         # # half_eps是两者CFG融合后的噪声预测
#         # half_eps = uncond_eps + cfg_scale * (cond_eps - uncond_eps)
#         # # 然后将half_eps复制两份，对应生成的输出与输入形状对齐
#         # eps = torch.cat([half_eps, half_eps], dim=0)
#         # return torch.cat([eps, rest], dim=1)

#         """
#         x: (2*num_residues, seq_len, decoder_embed_dim)
#         condition: (2*num_residues, seq_len, z_channels) ps: z_channels = decoder_embed_dim (now)
#         index: (2*num_residues, seq_len, 1) or None
#         We assume x, condition, index all are [2*num_residues, ...] where first num_residues rows=cond, second num_residues rows=uncond.
#         Returns [2*num_residues, seq_len, out_channels]
#         """
#         # 兼容输入未预先拼接为(2*B, ...)的情况：内部构造CFG双份批次
#         if x.shape[0] % 2 != 0:
#             # 若批大小为奇数或为单批，采用标准CFG路径：复制一份并在内部完成融合
#             half = x
#             combined_x = torch.cat([half, half], dim=0)
#             combined_condition = torch.cat([condition, condition], dim=0)
#             combined_index = torch.cat([index, index], dim=0) if index is not None else None

#             model_out = self.forward(combined_x, t, combined_condition, combined_index)
#             eps = model_out[:, :self.in_channels]
#             rest = model_out[:, self.in_channels:]

#             cond_eps, uncond_eps = torch.split(eps, len(eps) // 2, dim=0)
#             half_eps = uncond_eps + cfg_scale * (cond_eps - uncond_eps)
#             eps_final = torch.cat([half_eps, half_eps], dim=0)
#             # 若未学习方差（out_channels == in_channels），rest 为空，直接返回 eps_final
#             if self.out_channels == self.in_channels:
#                 return eps_final
#             # 确保维度匹配
#             if eps_final.shape[:2] != rest.shape[:2]:
#                 return eps_final
#             return torch.cat([eps_final, rest], dim=-1)

#         # 标准路径：输入已为(2*B, ...)，拆分为cond/uncond
#         num_residues = x.shape[0] // 2  # B
#         if num_residues == 0:
#             # 防御性分支：空批直接返回空张量，避免cat报错
#             return x.new_zeros(*x.shape[:-1], self.in_channels)

#         x_cond, x_uncond = x[:num_residues], x[num_residues:]
#         condition_cond, condition_uncond = condition[:num_residues], condition[num_residues:]
#         if index is not None:
#             index_cond, index_uncond = index[:num_residues], index[num_residues:]
#         else:
#             index_cond = index_uncond = None

#         # (B, T, D) -> (2B, T, D)
#         x_cat = torch.cat([x_cond, x_uncond], dim=0)
#         condition_cat = torch.cat([condition_cond, condition_uncond], dim=0)
#         index_cat = torch.cat([index_cond, index_uncond], dim=0) if index is not None else None

#         # 前向与通道拆分
#         model_out = self.forward(x_cat, t, condition_cat, index_cat)
#         eps = model_out[:, :self.in_channels]
#         rest = model_out[:, self.in_channels:]

#         # cond/uncond拆分
#         eps_cond, eps_uncond = eps[:num_residues], eps[num_residues:]

#         # CFG融合
#         eps_cfg = eps_uncond + cfg_scale * (eps_cond - eps_uncond)

#         # 还原为(2B, ...)
#         eps_final = torch.cat([eps_cfg, eps_cfg], dim=0)

#         # 若未学习方差（out_channels == in_channels），rest 为空，直接返回 eps_final
#         # 这是最直接的检查方式，避免依赖 rest 的形状
#         if self.out_channels == self.in_channels:
#             return eps_final

#         # 如果 out_channels > in_channels，需要拼接 rest
#         # 确保 eps_final 和 rest 的前两个维度匹配（batch 和 seq_len）
#         if eps_final.shape[:2] != rest.shape[:2]:
#             # 如果维度不匹配，只返回 eps_final（这种情况不应该发生，但防御性编程）
#             return eps_final

#         return torch.cat([eps_final, rest], dim=-1)

class FrequencyComponentDenoiseNet(nn.Module):
    """
    The DiT model for single frequency component denoising in DiffLoss.
    :param in_channels: channels in the input Tensor.
    :param model_channels: base channel count for the model.
    :param out_channels: channels in the output Tensor.
    :param z_channels: channels in the condition.
    :param num_res_blocks: number of residual blocks per downsample.
    """

    def __init__(
        self,
        in_channels,
        model_channels,
        out_channels,
        z_channels,
        num_dit_blocks=28,
        num_heads=16,
        mlp_ratio=4.0,
        seq_chunk_size: Optional[int] = None,
        block_type: str = "residue",
        use_checkpoint: bool = False,
    ):
        super().__init__()

        self.in_channels = in_channels
        self.model_channels = model_channels
        self.out_channels = out_channels
        self.num_dit_blocks = num_dit_blocks
        self.seq_chunk_size = seq_chunk_size
        self.block_type = block_type
        self.use_checkpoint = use_checkpoint

        self.time_embed = TimestepEmbedder(model_channels)
        self.cond_embed = nn.Linear(z_channels, model_channels)
        self.index_cond_embed = nn.Linear(1, model_channels)

        self.input_proj = nn.Linear(in_channels, model_channels)

        res_blocks = []
        if block_type not in ("residue", "mlp"):
            raise ValueError(f"Unsupported block_type={block_type}, expected 'residue' or 'mlp'")
        for _ in range(num_dit_blocks):
            # if block_type == "residue":
            #     res_blocks.append(ResidueDiTBlock(
            #         model_channels, num_heads, mlp_ratio=mlp_ratio
            #     ))
            # else:
            res_blocks.append(ResBlock(model_channels))

        self.res_blocks = nn.ModuleList(res_blocks)
        self.final_layer = FinalLayer(model_channels, out_channels)

        self.initialize_weights()

    def initialize_weights(self):
        # Initialize transformer layers:
        def _basic_init(module):
            if isinstance(module, nn.Linear):
                torch.nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)
        self.apply(_basic_init)

        # Initialize timestep embedding MLP
        nn.init.normal_(self.time_embed.mlp[0].weight, std=0.02)
        nn.init.normal_(self.time_embed.mlp[2].weight, std=0.02)

        # Zero-out adaLN modulation layers
        for block in self.res_blocks:
            nn.init.constant_(block.adaLN_modulation[-1].weight, 0)
            nn.init.constant_(block.adaLN_modulation[-1].bias, 0)

        # Zero-out output layers
        nn.init.constant_(self.final_layer.adaLN_modulation[-1].weight, 0)
        nn.init.constant_(self.final_layer.adaLN_modulation[-1].bias, 0)
        nn.init.constant_(self.final_layer.linear.weight, 0)
        nn.init.constant_(self.final_layer.linear.bias, 0)

    def ckpt_wrapper(self, module):
        def ckpt_forward(*inputs):
            outputs = module(*inputs)
            return outputs
        return ckpt_forward

    def _run_blocks(self, x, y):
        for block in self.res_blocks:
            if self.use_checkpoint and self.training:
                if isinstance(block, ResidueDiTBlock):
                    def _blk(x_in, y_in):
                        return block(x_in, y_in, seq_chunk_size=self.seq_chunk_size)
                    x = ckpt.checkpoint(_blk, x, y, use_reentrant=False)
                else:
                    x = ckpt.checkpoint(block, x, y, use_reentrant=False)
            else:
                if isinstance(block, ResidueDiTBlock):
                    x = block(x, y, seq_chunk_size=self.seq_chunk_size)
                else:
                    x = block(x, y)
        return x

    def forward(self, x_t, t, condition, index=None):
        """
        Apply the model to an input batch.
        :param x_t (target): x_t(random noise), shape: (num_residues, seq_len, decoder_embed_dim)
        :param t: a 1-D batch of timesteps. shape: (1, )
        :param c (z): conditioning from AR transformer, shape: (num_residues, seq_len, decoder_embed_dim)
        :return: an [N x C x ...] Tensor of outputs.
        """
        t_embed = self.time_embed(t) # (1, ) -> (1, model_channels)

        # For MLP blocks, chunk along seq_len to reduce peak activation memory.
        if self.seq_chunk_size is not None and self.block_type == "mlp":
            seq_len = x_t.shape[1]
            if seq_len > self.seq_chunk_size:
                outputs = []
                for start in range(0, seq_len, self.seq_chunk_size):
                    end = min(start + self.seq_chunk_size, seq_len)
                    x_chunk = self.input_proj(x_t[:, start:end, :])
                    cond_chunk = self.cond_embed(condition[:, start:end, :])
                    if index is not None:
                        idx_chunk = self.index_cond_embed(index[:, start:end, :])
                        y_chunk = t_embed + cond_chunk + idx_chunk
                    else:
                        y_chunk = t_embed + cond_chunk
                    x_chunk = self._run_blocks(x_chunk, y_chunk)
                    outputs.append(self.final_layer(x_chunk, y_chunk))
                return torch.cat(outputs, dim=1)

        x = self.input_proj(x_t) # (num_residues, seq_len, token_embed_dim) -> (num_residues, seq_len, model_channels)
        condition = self.cond_embed(condition) # (num_residues, seq_len, decoder_embed_dim) -> (num_residues, seq_len, model_channels)
        if index is not None:
            index = self.index_cond_embed(index) # (num_residues, seq_len, 1) -> (num_residues, seq_len, model_channels)
            y = t_embed + condition + index # (num_residues, seq_len, model_channels)
        else:
            y = t_embed + condition

        x = self._run_blocks(x, y)
        return self.final_layer(x, y) # (N, T, out_channels)

    def forward_with_cfg(self, x, t, condition, cfg_scale, index):
        # # 这里的half是输入x的一半。通常x包含了两组输入：前一半是带条件的，后一半是不带条件的。
        # # 这里取x的前一半（x[:len(x) // 2]），即half，然后cat两次得到一个两倍batch，
        # # 用于分别做有条件和无条件的推理。
        # half = x[: len(x) // 2]
        # combined = torch.cat([half, half], dim=0)
        # model_out = self.forward(combined, t, condition, index) # (2N, T, D)
        # eps, rest = model_out[:, :self.in_channels], model_out[:, self.in_channels:]
        # cond_eps, uncond_eps = torch.split(eps, len(eps) // 2, dim=0)
        # # half_eps是两者CFG融合后的噪声预测
        # half_eps = uncond_eps + cfg_scale * (cond_eps - uncond_eps)
        # # 然后将half_eps复制两份，对应生成的输出与输入形状对齐
        # eps = torch.cat([half_eps, half_eps], dim=0)
        # return torch.cat([eps, rest], dim=1)

        """
        x: (2*num_residues, seq_len, decoder_embed_dim)
        condition: (2*num_residues, seq_len, z_channels) ps: z_channels = decoder_embed_dim (now)
        index: (2*num_residues, seq_len, 1) or None
        We assume x, condition, index all are [2*num_residues, ...] where first num_residues rows=cond, second num_residues rows=uncond.
        Returns [2*num_residues, seq_len, out_channels]
        """
        # 兼容输入未预先拼接为(2*B, ...)的情况：内部构造CFG双份批次
        if x.shape[0] % 2 != 0:
            # 若批大小为奇数或为单批，采用标准CFG路径：复制一份并在内部完成融合
            half = x
            combined_x = torch.cat([half, half], dim=0)
            combined_condition = torch.cat([condition, condition], dim=0)
            combined_index = torch.cat([index, index], dim=0) if index is not None else None

            model_out = self.forward(combined_x, t, combined_condition, combined_index)
            eps = model_out[:, :self.in_channels]
            rest = model_out[:, self.in_channels:]

            cond_eps, uncond_eps = torch.split(eps, len(eps) // 2, dim=0)
            half_eps = uncond_eps + cfg_scale * (cond_eps - uncond_eps)
            eps_final = torch.cat([half_eps, half_eps], dim=0)
            # 若未学习方差（out_channels == in_channels），rest 为空，直接返回 eps_final
            if self.out_channels == self.in_channels:
                return eps_final
            # 确保维度匹配
            if eps_final.shape[:2] != rest.shape[:2]:
                return eps_final
            return torch.cat([eps_final, rest], dim=-1)

        # 标准路径：输入已为(2*B, ...)，拆分为cond/uncond
        num_residues = x.shape[0] // 2  # B
        if num_residues == 0:
            # 防御性分支：空批直接返回空张量，避免cat报错
            return x.new_zeros(*x.shape[:-1], self.in_channels)

        x_cond, x_uncond = x[:num_residues], x[num_residues:]
        condition_cond, condition_uncond = condition[:num_residues], condition[num_residues:]
        if index is not None:
            index_cond, index_uncond = index[:num_residues], index[num_residues:]
        else:
            index_cond = index_uncond = None

        # (B, T, D) -> (2B, T, D)
        x_cat = torch.cat([x_cond, x_uncond], dim=0)
        condition_cat = torch.cat([condition_cond, condition_uncond], dim=0)
        index_cat = torch.cat([index_cond, index_uncond], dim=0) if index is not None else None

        # 前向与通道拆分
        model_out = self.forward(x_cat, t, condition_cat, index_cat)
        eps = model_out[:, :self.in_channels]
        rest = model_out[:, self.in_channels:]

        # cond/uncond拆分
        eps_cond, eps_uncond = eps[:num_residues], eps[num_residues:]

        # CFG融合
        eps_cfg = eps_uncond + cfg_scale * (eps_cond - eps_uncond)

        # 还原为(2B, ...)
        eps_final = torch.cat([eps_cfg, eps_cfg], dim=0)

        # 若未学习方差（out_channels == in_channels），rest 为空，直接返回 eps_final
        # 这是最直接的检查方式，避免依赖 rest 的形状
        if self.out_channels == self.in_channels:
            return eps_final

        # 如果 out_channels > in_channels，需要拼接 rest
        # 确保 eps_final 和 rest 的前两个维度匹配（batch 和 seq_len）
        if eps_final.shape[:2] != rest.shape[:2]:
            # 如果维度不匹配，只返回 eps_final（这种情况不应该发生，但防御性编程）
            return eps_final

        return torch.cat([eps_final, rest], dim=-1)


    
# class DiT(nn.Module):
#     """
#     Diffusion model with a Transformer backbone.
#     """
#     def __init__(
#         self,
#         input_size=32,
#         patch_size=2,
#         in_channels=4,
#         hidden_size=1152,
#         depth=28,
#         num_heads=16,
#         mlp_ratio=4.0,
#         class_dropout_prob=0.1,
#         num_classes=1000,
#         learn_sigma=True,
#     ):
#         super().__init__()
#         self.learn_sigma = learn_sigma
#         self.in_channels = in_channels
#         self.out_channels = in_channels * 2 if learn_sigma else in_channels
#         self.patch_size = patch_size
#         self.num_heads = num_heads

#         self.x_embedder = PatchEmbed(input_size, patch_size, in_channels, hidden_size, bias=True)
#         self.t_embedder = TimestepEmbedder(hidden_size)
#         self.y_embedder = LabelEmbedder(num_classes, hidden_size, class_dropout_prob)
#         num_patches = self.x_embedder.num_patches
#         # Will use fixed sin-cos embedding:
#         self.pos_embed = nn.Parameter(torch.zeros(1, num_patches, hidden_size), requires_grad=False)

#         self.blocks = nn.ModuleList([
#             DiTBlock(hidden_size, num_heads, mlp_ratio=mlp_ratio) for _ in range(depth)
#         ])
#         self.final_layer = FinalLayer(hidden_size, patch_size, self.out_channels)
#         self.initialize_weights()

#     def initialize_weights(self):
#         # Initialize transformer layers:
#         def _basic_init(module):
#             if isinstance(module, nn.Linear):
#                 torch.nn.init.xavier_uniform_(module.weight)
#                 if module.bias is not None:
#                     nn.init.constant_(module.bias, 0)
#         self.apply(_basic_init)

#         # Initialize (and freeze) pos_embed by sin-cos embedding:
#         pos_embed = get_2d_sincos_pos_embed(self.pos_embed.shape[-1], int(self.x_embedder.num_patches ** 0.5))
#         self.pos_embed.data.copy_(torch.from_numpy(pos_embed).float().unsqueeze(0))

#         # Initialize patch_embed like nn.Linear (instead of nn.Conv2d):
#         w = self.x_embedder.proj.weight.data
#         nn.init.xavier_uniform_(w.view([w.shape[0], -1]))
#         nn.init.constant_(self.x_embedder.proj.bias, 0)

#         # Initialize label embedding table:
#         nn.init.normal_(self.y_embedder.embedding_table.weight, std=0.02)

#         # Initialize timestep embedding MLP:
#         nn.init.normal_(self.t_embedder.mlp[0].weight, std=0.02)
#         nn.init.normal_(self.t_embedder.mlp[2].weight, std=0.02)

#         # Zero-out adaLN modulation layers in DiT blocks:
#         for block in self.blocks:
#             nn.init.constant_(block.adaLN_modulation[-1].weight, 0)
#             nn.init.constant_(block.adaLN_modulation[-1].bias, 0)

#         # Zero-out output layers:
#         nn.init.constant_(self.final_layer.adaLN_modulation[-1].weight, 0)
#         nn.init.constant_(self.final_layer.adaLN_modulation[-1].bias, 0)
#         nn.init.constant_(self.final_layer.linear.weight, 0)
#         nn.init.constant_(self.final_layer.linear.bias, 0)

#     def unpatchify(self, x):
#         """
#         x: (N, T, patch_size**2 * C)
#         imgs: (N, H, W, C)
#         """
#         c = self.out_channels
#         p = self.x_embedder.patch_size[0]
#         h = w = int(x.shape[1] ** 0.5)
#         assert h * w == x.shape[1]

#         x = x.reshape(shape=(x.shape[0], h, w, p, p, c))
#         x = torch.einsum('nhwpqc->nchpwq', x)
#         imgs = x.reshape(shape=(x.shape[0], c, h * p, h * p))
#         return imgs
    
#     def ckpt_wrapper(self, module):
#         def ckpt_forward(*inputs):
#             outputs = module(*inputs)
#             return outputs
#         return ckpt_forward

#     def forward(self, x, t, y):
#         """
#         Forward pass of DiT.
#         x: (N, C, H, W) tensor of spatial inputs (images or latent representations of images)
#         t: (N,) tensor of diffusion timesteps
#         y: (N,) tensor of class labels
#         """
#         x = self.x_embedder(x) + self.pos_embed  # (N, T, D), where T = H * W / patch_size ** 2
#         t = self.t_embedder(t)                   # (N, D)
#         y = self.y_embedder(y, self.training)    # (N, D)
#         c = t + y                                # (N, D)
#         for block in self.blocks:
#             x = torch.utils.checkpoint.checkpoint(self.ckpt_wrapper(block), x, c)       # (N, T, D)
#         x = self.final_layer(x, c)                # (N, T, patch_size ** 2 * out_channels)
#         x = self.unpatchify(x)                   # (N, out_channels, H, W)
#         return x

#     def forward_with_cfg(self, x, t, y, cfg_scale):
#         """
#         Forward pass of DiT, but also batches the unconditional forward pass for classifier-free guidance.
#         """
#         # https://github.com/openai/glide-text2im/blob/main/notebooks/text2im.ipynb
#         half = x[: len(x) // 2]
#         combined = torch.cat([half, half], dim=0)
#         model_out = self.forward(combined, t, y)
#         # For exact reproducibility reasons, we apply classifier-free guidance on only
#         # three channels by default. The standard approach to cfg applies it to all channels.
#         # This can be done by uncommenting the following line and commenting-out the line following that.
#         # eps, rest = model_out[:, :self.in_channels], model_out[:, self.in_channels:]
#         eps, rest = model_out[:, :3], model_out[:, 3:]
#         cond_eps, uncond_eps = torch.split(eps, len(eps) // 2, dim=0)
#         half_eps = uncond_eps + cfg_scale * (cond_eps - uncond_eps)
#         eps = torch.cat([half_eps, half_eps], dim=0)
#         return torch.cat([eps, rest], dim=1)
