# Copyright 2024 ByteDance and/or its affiliates.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from functools import partial
from typing import Optional, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
import warnings
import traceback

from src.protenix.modules.primitives import (
    AdaptiveLayerNorm,
    Attention,
    BiasInitLinear,
    LinearNoBias,
    broadcast_token_to_local_atom_pair,
    rearrange_qk_to_dense_trunk,
)
from src.protenix.utils import (
    aggregate_atom_to_token,
    broadcast_token_to_atom,
    permute_final_dims,
)
from src.openfold_local.model.primitives import LayerNorm
from src.openfold_local.utils.checkpointing import checkpoint_blocks
# from src.model.modules.LoRA_module import LoRAModule

if hasattr(torch, "_dynamo"):
    disable_dynamo = torch._dynamo.disable
else:
    def disable_dynamo(fn):
        return fn

class AttentionPairBias(nn.Module):
    """
    Implements Algorithm 24 in AF3
    """

    def __init__(
        self,
        has_s: bool = True,
        n_heads: int = 16,
        c_a: int = 768,
        c_s: int = 384,
        c_z: int = 128,
        biasinit: float = -2.0,
    ) -> None:
        """
        Args:
            has_s (bool, optional):  whether s is None as stated in Algorithm 24 Line1. Defaults to True.
            n_heads (int, optional): number of attention-like head in AttentionPairBias. Defaults to 16.
            c_a (int, optional): the embedding dim of a(single feature aggregated atom info). Defaults to 768.
            c_s (int, optional):  hidden dim [for single embedding]. Defaults to 384.
            c_z (int, optional): hidden dim [for pair embedding]. Defaults to 128.
            biasinit (float, optional): biasinit for BiasInitLinear. Defaults to -2.0.
        """
        super(AttentionPairBias, self).__init__()
        assert c_a % n_heads == 0
        self.n_heads = n_heads
        self.has_s = has_s
        if has_s:
            # Line2
            self.layernorm_a = AdaptiveLayerNorm(c_a=c_a, c_s=c_s)
            # Line 13
            self.linear_a_last = BiasInitLinear(
                in_features=c_s, out_features=c_a, bias=True, biasinit=biasinit
            )
        else:
            self.layernorm_a = LayerNorm(c_a)
        # Line 6-11
        self.local_attention_method = "local_cross_attention"
        self.attention = Attention(
            c_q=c_a,
            c_k=c_a,
            c_v=c_a,
            c_hidden=c_a // n_heads,
            num_heads=n_heads,
            gating=True,
            q_linear_bias=True,
            local_attention_method=self.local_attention_method,
        )
        self.layernorm_z = LayerNorm(c_z)
        # Alg24. Line8 is scalar, but this is different for different heads
        self.linear_nobias_z = LinearNoBias(in_features=c_z, out_features=n_heads)

    def glorot_init(self):
        nn.init.xavier_uniform_(self.attention.linear_q.weight)
        nn.init.xavier_uniform_(self.attention.linear_k.weight)
        nn.init.xavier_uniform_(self.attention.linear_v.weight)
        nn.init.zeros_(self.attention.linear_q.bias)

    def local_multihead_attention(
        self,
        a: torch.Tensor,
        s: torch.Tensor,
        z: torch.Tensor,
        n_queries: int = 32,
        n_keys: int = 128,
        inplace_safe: bool = False,
        chunk_size: Optional[int] = None,
    ) -> torch.Tensor:
        """Used by Algorithm 24, with beta_ij being the local mask. Used in AtomTransformer.

        Args:
            a (torch.Tensor): atom embedding
                [..., N_atom, c_a]
            s (torch.Tensor): atom embedding
                [..., N_atom, c_s]
            z (torch.Tensor): atom-atom pair embedding, in trunked dense shape. Used for computing pair bias.
                [..., n_blocks, n_queries, n_keys, c_z]
            n_queries (int, optional): local window size of query tensor. Defaults to 32.
            n_keys (int, optional): local window size of key tensor. Defaults to 128.
            inplace_safe (bool): Whether it is safe to use inplace operations. Defaults to False.
            chunk_size (Optional[int]): Chunk size for memory-efficient operations. Defaults to None.

        Returns:
            torch.Tensor: the updated a from AttentionPairBias
                [..., N_atom, c_a]
        """

        assert n_queries == z.size(-3)
        assert n_keys == z.size(-2)
        assert len(z.shape) == len(a.shape) + 2

        # Multi-head attention bias
        bias = self.linear_nobias_z(
            self.layernorm_z(z)
        )  # [..., n_blocks, n_queries, n_keys, n_heads]
        bias = permute_final_dims(
            bias, [3, 0, 1, 2]
        )  # [..., n_heads, n_blocks, n_queries, n_keys]

        # Line 11: Multi-head attention with attention bias & gating (and optionally local attention)
        a = self.attention(
            q_x=a,
            kv_x=a,
            trunked_attn_bias=bias,
            n_queries=n_queries,
            n_keys=n_keys,
            inplace_safe=inplace_safe,
            chunk_size=chunk_size,
        )
        return a

    def standard_multihead_attention(
        self,
        a: torch.Tensor,
        s: torch.Tensor,
        z: torch.Tensor,
        inplace_safe: bool = False,
    ) -> torch.Tensor:
        """Used by Algorithm 7/20

        Args:
            a (torch.Tensor): the single feature aggregate per-atom representation
                [..., N_token, c_a]
            s (torch.Tensor): single embedding
                [..., N_token, c_s]
            z (torch.Tensor): pair embedding, used for computing pair bias.
                [..., N_token, N_token, c_z]
            inplace_safe (bool): Whether it is safe to use inplace operations. Defaults to False.

        Returns:
            torch.Tensor: the updated a from AttentionPairBias
                [..., N_token, c_a]
        """

        # Multi-head attention bias
        bias = self.linear_nobias_z(self.layernorm_z(z))
        bias = permute_final_dims(bias, [2, 0, 1])  # [..., n_heads, N_token, N_token]

        # Line 11: Multi-head attention with attention bias & gating (and optionally local attention)
        a = self.attention(q_x=a, kv_x=a, attn_bias=bias, inplace_safe=inplace_safe)

        return a

    def forward(
        self,
        a: torch.Tensor,
        s: torch.Tensor,
        z: torch.Tensor,
        n_queries: Optional[int] = None,
        n_keys: Optional[int] = None,
        inplace_safe: bool = False,
        chunk_size: Optional[int] = None,
    ) -> torch.Tensor:
        """Details are given in local_forward and standard_forward"""
        # Input projections
        if self.has_s:
            a = self.layernorm_a(a=a, s=s)
        else:
            a = self.layernorm_a(a)

        # Multihead attention with pair bias
        if n_queries and n_keys:
            a = self.local_multihead_attention(
                a,
                s,
                z,
                n_queries,
                n_keys,
                inplace_safe=inplace_safe,
                chunk_size=chunk_size,
            )
        else:
            a = self.standard_multihead_attention(a, s, z, inplace_safe=inplace_safe)

        # Output projection (from adaLN-Zero [27])
        if self.has_s:
            if inplace_safe:
                a *= torch.sigmoid(self.linear_a_last(s))
            else:
                a = torch.sigmoid(self.linear_a_last(s)) * a

        return a


class AttentionPairBiasWithIRFC(nn.Module):
    """
    Implements Algorithm 24 in AF3
    """

    def __init__(
        self,
        has_s: bool = True,
        n_heads: int = 16,
        c_a: int = 768,
        c_s: int = 384,
        c_z: int = 128,
        biasinit: float = -2.0,
    ) -> None:
        """
        Args:
            has_s (bool, optional):  whether s is None as stated in Algorithm 24 Line1. Defaults to True.
            n_heads (int, optional): number of attention-like head in AttentionPairBias. Defaults to 16.
            c_a (int, optional): the embedding dim of a(single feature aggregated atom info). Defaults to 768.
            c_s (int, optional):  hidden dim [for single embedding]. Defaults to 384.
            c_z (int, optional): hidden dim [for pair embedding]. Defaults to 128.
            biasinit (float, optional): biasinit for BiasInitLinear. Defaults to -2.0.
        """
        super(AttentionPairBiasWithIRFC, self).__init__()
        assert c_a % n_heads == 0
        self.n_heads = n_heads
        self.has_s = has_s
        if has_s:
            # Line2
            self.layernorm_a = AdaptiveLayerNorm(c_a=c_a, c_s=c_s)
            # Line 13
            self.linear_a_last = BiasInitLinear(
                in_features=c_s, out_features=c_a, bias=True, biasinit=biasinit
            )
        else:
            self.layernorm_a = LayerNorm(c_a)
        # Line 6-11
        self.local_attention_method = "local_cross_attention"
        self.attention = Attention(
            c_q=c_a,
            c_k=c_a,
            c_v=c_a,
            c_hidden=c_a // n_heads,
            num_heads=n_heads,
            gating=True,
            q_linear_bias=True,
            local_attention_method=self.local_attention_method,
        )
        self.layernorm_z = LayerNorm(c_z)
        # Alg24. Line8 is scalar, but this is different for different heads
        self.linear_nobias_z = LinearNoBias(in_features=c_z, out_features=n_heads)

        # 新增：IRFC 衰减核参数 (每个头独立)
        # 初始化为 -2.0 (exp后约为0.13)，让模型初始时具有较广的感受野
        self.irfc_log_gamma = nn.Parameter(torch.full((n_heads,), -2.0))

    def glorot_init(self):
        nn.init.xavier_uniform_(self.attention.linear_q.weight)
        nn.init.xavier_uniform_(self.attention.linear_k.weight)
        nn.init.xavier_uniform_(self.attention.linear_v.weight)
        nn.init.zeros_(self.attention.linear_q.bias)
    @disable_dynamo
    def local_multihead_attention(
        self,
        a: torch.Tensor,
        s: torch.Tensor,
        z: torch.Tensor,
        irfc_dist_sq: Optional[torch.Tensor] = None, # 新增：分块后的距离矩阵
        n_queries: int = 32,
        n_keys: int = 128,
        inplace_safe: bool = False,
        chunk_size: Optional[int] = None,
    ) -> torch.Tensor:
        """Used by Algorithm 24, with beta_ij being the local mask. Used in AtomTransformer.

        Args:
            a (torch.Tensor): atom embedding
                [..., N_atom, c_a]
            s (torch.Tensor): atom embedding
                [..., N_atom, c_s]
            z (torch.Tensor): atom-atom pair embedding, in trunked dense shape. Used for computing pair bias.
                [..., n_blocks, n_queries, n_keys, c_z]
            n_queries (int, optional): local window size of query tensor. Defaults to 32.
            n_keys (int, optional): local window size of key tensor. Defaults to 128.
            inplace_safe (bool): Whether it is safe to use inplace operations. Defaults to False.
            chunk_size (Optional[int]): Chunk size for memory-efficient operations. Defaults to None.

        Returns:
            torch.Tensor: the updated a from AttentionPairBias
                [..., N_atom, c_a]
        """

        assert n_queries == z.size(-3)
        assert n_keys == z.size(-2)
        assert len(z.shape) == len(a.shape) + 2

        # Multi-head attention bias
        bias = self.linear_nobias_z(
            self.layernorm_z(z)
        )  # [..., n_blocks, n_queries, n_keys, n_heads]
        # 2. 注入 Decay Mask (IRFC)
        if irfc_dist_sq is not None:
            # TorchDynamo 编译阶段跳过 IRFC，避免触发内部异常
            if hasattr(torch, "_dynamo") and torch._dynamo.is_compiling():
                return self.attention(
                    q_x=a,
                    kv_x=a,
                    trunked_attn_bias=permute_final_dims(
                        bias, [3, 0, 1, 2]
                    ),
                    n_queries=n_queries,
                    n_keys=n_keys,
                    inplace_safe=inplace_safe,
                    chunk_size=chunk_size,
                )
            # irfc_dist_sq 期望形状: [..., n_blocks, n_queries, n_keys]
            # 确保 dtype / device 与 bias 一致，避免运行时内部解析异常
            irfc_dist_sq = irfc_dist_sq.to(dtype=bias.dtype, device=bias.device)
            # gamma: [n_heads] -> 调整维度以便广播
            gamma = torch.exp(self.irfc_log_gamma).view(1, 1, 1, 1, self.n_heads)

            # 计算局部衰减偏置
            # [..., n_blocks, n_queries, n_keys, 1] * [1, 1, 1, 1, n_heads]
            local_irfc_bias = -gamma * irfc_dist_sq.unsqueeze(-1)

            # 累加到原有 bias
            bias = bias + local_irfc_bias

        bias = permute_final_dims(
            bias, [3, 0, 1, 2]
        )  # [..., n_heads, n_blocks, n_queries, n_keys]

        # Line 11: Multi-head attention with attention bias & gating (and optionally local attention)
        a = self.attention(
            q_x=a,
            kv_x=a,
            trunked_attn_bias=bias,
            n_queries=n_queries,
            n_keys=n_keys,
            inplace_safe=inplace_safe,
            chunk_size=chunk_size,
        )
        return a

    @disable_dynamo
    def standard_multihead_attention(
        self,
        a: torch.Tensor,
        s: torch.Tensor,
        z: torch.Tensor,
        irfc_dist_sq: Optional[torch.Tensor] = None, # 新增
        inplace_safe: bool = False,
    ) -> torch.Tensor:
        """Used by Algorithm 7/20

        Args:
            a (torch.Tensor): the single feature aggregate per-atom representation
                [..., N_token, c_a]
            s (torch.Tensor): single embedding
                [..., N_token, c_s]
            z (torch.Tensor): pair embedding, used for computing pair bias.
                [..., N_token, N_token, c_z]
            inplace_safe (bool): Whether it is safe to use inplace operations. Defaults to False.

        Returns:
            torch.Tensor: the updated a from AttentionPairBias
                [..., N_token, c_a]
        """

        # Multi-head attention bias
        bias = self.linear_nobias_z(self.layernorm_z(z))
        bias = permute_final_dims(bias, [2, 0, 1])  # [..., n_heads, N_token, N_token]

        # 2. 累加 IRFC Bias
        if irfc_dist_sq is not None:
            # TorchDynamo 编译阶段跳过 IRFC，避免触发内部异常
            if hasattr(torch, "_dynamo") and torch._dynamo.is_compiling():
                return self.attention(
                    q_x=a,
                    kv_x=a,
                    attn_bias=bias,
                    inplace_safe=inplace_safe,
                )
            # 确保 dtype / device 与 bias 一致，避免运行时内部解析异常
            irfc_dist_sq = irfc_dist_sq.to(dtype=bias.dtype, device=bias.device)
            # gamma = exp(log_gamma), 确保衰减系数为正
            # 适配 batch/seq_len 维度：对 bias 前缀维度全部广播
            head_shape = [1] * (bias.dim() - 3) + [self.n_heads, 1, 1]
            gamma = torch.exp(self.irfc_log_gamma).view(*head_shape)
            # 计算衰减核: -gamma * distance^2
            irfc_bias = -gamma * irfc_dist_sq.unsqueeze(-3)
            bias = bias + irfc_bias

        # Line 11: Multi-head attention with attention bias & gating (and optionally local attention)
        a = self.attention(q_x=a, kv_x=a, attn_bias=bias, inplace_safe=inplace_safe)

        return a

    def forward(
        self,
        a: torch.Tensor,
        s: torch.Tensor,
        z: torch.Tensor,
        irfc_dist_sq: Optional[torch.Tensor] = None, # 新增
        n_queries: Optional[int] = None,
        n_keys: Optional[int] = None,
        inplace_safe: bool = False,
        chunk_size: Optional[int] = None,
    ) -> torch.Tensor:
        """Details are given in local_forward and standard_forward"""
        # Input projections
        if self.has_s:
            a = self.layernorm_a(a=a, s=s)
        else:
            a = self.layernorm_a(a)

        # Multihead attention with pair bias
        if n_queries and n_keys:
            a = self.local_multihead_attention(
                a,
                s,
                z,
                irfc_dist_sq,
                n_queries,
                n_keys,
                inplace_safe=inplace_safe,
                chunk_size=chunk_size,
            )
        else:
            a = self.standard_multihead_attention(a, s, z, irfc_dist_sq=irfc_dist_sq, inplace_safe=inplace_safe)

        # Output projection (from adaLN-Zero [27])
        if self.has_s:
            if inplace_safe:
                a *= torch.sigmoid(self.linear_a_last(s))
            else:
                a = torch.sigmoid(self.linear_a_last(s)) * a

        return a

class DiffusionTransformerBlock(nn.Module):
    """
    Implements Algorithm 23[Line2-Line3] in AF3
    """

    def __init__(
        self,
        c_a: int,  # could be 128 or 768 in AF3
        c_s: int,  # could be c_s or c_atom
        c_z: int,  # could be c_z or c_atompair
        n_heads: int,  # could be 16 or 4 or ... in AF3
        biasinit: float = -2.0,
        with_irfc: bool = False
    ) -> None:
        """
        Args:
            c_a (int, optional): single embedding dimension.
            c_s (int, optional): single embedding dimension.
            c_z (int, optional): pair embedding dimension.
            n_heads (int, optional): number of heads for DiffusionTransformerBlock.
        """
        super(DiffusionTransformerBlock, self).__init__()
        self.n_heads = n_heads
        self.c_a = c_a
        self.c_s = c_s
        self.c_z = c_z
        if with_irfc:
            self.attention_pair_bias = AttentionPairBiasWithIRFC(
                has_s=True, n_heads=n_heads, c_a=c_a, c_s=c_s, c_z=c_z, biasinit=biasinit
            )
        else:
            self.attention_pair_bias = AttentionPairBias(
                has_s=True, n_heads=n_heads, c_a=c_a, c_s=c_s, c_z=c_z, biasinit=biasinit
            )

        self.conditioned_transition_block = ConditionedTransitionBlock(
            n=2, c_a=c_a, c_s=c_s, biasinit=biasinit
        )

    def forward(
        self,
        a: torch.Tensor,
        s: torch.Tensor,
        z: torch.Tensor,
        irfc_dist_sq: Optional[torch.Tensor] = None, # 新增
        n_queries: Optional[int] = None,
        n_keys: Optional[int] = None,
        inplace_safe: bool = False,
        chunk_size: Optional[int] = None,
    ) -> torch.Tensor:
        """
        Args:
            a (torch.Tensor): the single feature aggregate per-atom representation
                [..., N, c_a]
            s (torch.Tensor): single embedding
                [..., N, c_s]
            z (torch.Tensor): pair embedding
                [..., N, N, c_z] or [..., n_block, n_queries, n_keys, c_z]
            n_queries (int, optional): local window size of query tensor. If not None, will perform local attention. Defaults to None.
            n_keys (int, optional): local window size of key tensor. Defaults to None.
            inplace_safe (bool): Whether it is safe to use inplace operations. Defaults to False.
            chunk_size (Optional[int]): Chunk size for memory-efficient operations. Defaults to None.

        Returns:
            torch.Tensor: the output of DiffusionTransformerBlock
                [..., N, c_a]
        """
        if isinstance(self.attention_pair_bias, AttentionPairBiasWithIRFC):
            attn_out = self.attention_pair_bias(
                a=a,
                s=s,
                z=z,
                irfc_dist_sq=irfc_dist_sq,
                n_queries=n_queries,
                n_keys=n_keys,
                inplace_safe=inplace_safe,
                chunk_size=chunk_size,
            )
        else:
            attn_out = self.attention_pair_bias(
                a=a,
                s=s,
                z=z,
                n_queries=n_queries,
                n_keys=n_keys,
                inplace_safe=inplace_safe,
                chunk_size=chunk_size,
            )

        if inplace_safe:
            attn_out += a
        else:
            attn_out = attn_out + a
        ff_out = self.conditioned_transition_block(a=attn_out, s=s)
        out_a = ff_out + attn_out
        # Avoid s/z to be deleted by torch.utils.checkpoint
        return out_a, s, z


class DiffusionTransformer(nn.Module):
    """
    Implements Algorithm 23 in AF3
    """

    def __init__(
        self,
        c_a: int,  # could be 128 or 768 in AF3
        c_s: int,  # could be c_s or c_atom
        c_z: int,  # could be c_z or c_atompair
        n_blocks: int,  # could be 3 or 24 in AF3
        n_heads: int,  # could be 16 or 4 or ... in AF3
        blocks_per_ckpt: Optional[int] = None,
        with_irfc: bool = False
    ) -> None:
        """
        Args:
            c_a (int): single embedding dimension.
            c_s (int): single embedding dimension.
            c_z (int): pair embedding dimension.
            n_blocks (int): number of blocks in DiffusionTransformer.
            n_heads (int): number of heads in attention.
            blocks_per_ckpt: number of DiffusionTransformer blocks in each activation checkpoint
        """
        super(DiffusionTransformer, self).__init__()
        self.n_blocks = n_blocks
        self.n_heads = n_heads
        self.c_a = c_a
        self.c_s = c_s
        self.c_z = c_z
        self.blocks_per_ckpt = blocks_per_ckpt

        self.blocks = nn.ModuleList()
        for _ in range(n_blocks):
            block = DiffusionTransformerBlock(
                n_heads=n_heads, c_a=c_a, c_s=c_s, c_z=c_z, with_irfc=with_irfc
            )
            self.blocks.append(block)

    def _prep_blocks(
        self,
        irfc_dist_sq: Optional[torch.Tensor] = None, # 新增  
        n_queries: Optional[int] = None,
        n_keys: Optional[int] = None,
        inplace_safe: bool = False,
        chunk_size: Optional[int] = None,
        clear_cache_between_blocks: bool = False,
    ):
        blocks = [
            partial(
                b,
                irfc_dist_sq=irfc_dist_sq,
                n_queries=n_queries,
                n_keys=n_keys,
                inplace_safe=inplace_safe,
                chunk_size=chunk_size,
            )
            for b in self.blocks
        ]

        def clear_cache(b, *args, **kwargs):
            # torch.cuda.empty_cache()
            return b(*args, **kwargs)

        if clear_cache_between_blocks:
            blocks = [partial(clear_cache, b) for b in blocks]
        return blocks

    def forward(
        self,
        a: torch.Tensor,
        s: torch.Tensor,
        z: torch.Tensor,
        irfc_dist_sq: Optional[torch.Tensor] = None, # 新增
        n_queries: Optional[int] = None,
        n_keys: Optional[int] = None,
        inplace_safe: bool = False,
        chunk_size: Optional[int] = None,
    ) -> torch.Tensor:
        """
        Args:
            a (torch.Tensor): the single feature aggregate per-atom representation
                [..., N, c_a]
            s (torch.Tensor): single embedding
                [..., N, c_s]
            z (torch.Tensor): pair embedding
                [..., N, N, c_z]
            n_queries (int, optional): local window size of query tensor. If not None, will perform local attention. Defaults to None.
            n_keys (int, optional): local window size of key tensor. Defaults to None.

        Returns:
            torch.Tensor: the output of DiffusionTransformer
                [..., N, c_a]
        """
        if z.shape[-2] > 2000 and (not self.training):
            clear_cache_between_blocks = True
        else:
            clear_cache_between_blocks = False
        blocks = self._prep_blocks(
            irfc_dist_sq=irfc_dist_sq,
            n_queries=n_queries,
            n_keys=n_keys,
            inplace_safe=inplace_safe,
            chunk_size=chunk_size,
            clear_cache_between_blocks=clear_cache_between_blocks,
        )
        blocks_per_ckpt = self.blocks_per_ckpt
        if not torch.is_grad_enabled():
            blocks_per_ckpt = None
        a, s, z = checkpoint_blocks(
            blocks, args=(a, s, z), blocks_per_ckpt=blocks_per_ckpt, checkpoint_kwargs=dict(use_reentrant=False)
        )
        del s, z
        return a


class AtomTransformer(nn.Module):
    """
    Implements Algorithm 7 in AF3
    """

    def __init__(
        self,
        c_atom: int = 128,
        c_atompair: int = 16,
        n_blocks: int = 3,
        n_heads: int = 4,
        n_queries: int = 32,
        n_keys: int = 128,
        blocks_per_ckpt: Optional[int] = None,
    ) -> None:
        """Performs local transformer among atom embeddings, with bias predicted from atom pair embeddings

        Args:
            c_atom int: embedding dim for atom feature. Defaults to 128.
            c_atompair int: embedding dim for atompair feature. Defaults to 16.
            n_blocks (int, optional): number of block in AtomTransformer. Defaults to 3.
            n_heads (int, optional): nubmer of heads in attention. Defaults to 4.
            n_queries (int, optional): local window size of query tensor. If not None, will perform local attention. Defaults to 32.
            n_keys (int, optional): local window size of key tensor. Defaults to 128.
            blocks_per_ckpt: number of AtomTransformer/DiffusionTransformer blocks in each activation checkpoint
                Size of each chunk. A higher value corresponds to fewer
                checkpoints, and trades memory for speed. If None, no checkpointing
                is performed.
        """
        super(AtomTransformer, self).__init__()
        self.n_blocks = n_blocks
        self.n_heads = n_heads
        self.n_queries = n_queries
        self.n_keys = n_keys
        self.c_atom = c_atom
        self.c_atompair = c_atompair
        self.diffusion_transformer = DiffusionTransformer(
            n_blocks=n_blocks,
            n_heads=n_heads,
            c_a=c_atom,
            c_s=c_atom,
            c_z=c_atompair,
            blocks_per_ckpt=blocks_per_ckpt,
            with_irfc=False
        )

    def forward(
        self,
        q: torch.Tensor,
        c: torch.Tensor,
        p: torch.Tensor,
        inplace_safe: bool = False,
        chunk_size: Optional[int] = None,
    ) -> torch.Tensor:
        """
        Args:
            q (torch.Tensor): atom single embedding
                [..., N_atom, c_atom]
            c (torch.Tensor): atom single embedding
                [..., N_atom, c_atom]
            p (torch.Tensor): atompair embedding in dense block shape.
                [..., n_blocks, n_queries, n_keys, c_atompair]

        Returns:
            torch.Tensor: the output of AtomTransformer
                [..., N_atom, c_atom]
        """
        n_blocks, n_queries, n_keys = p.shape[-4:-1]

        assert n_queries == self.n_queries
        assert n_keys == self.n_keys
        return self.diffusion_transformer(
            a=q,
            s=c,
            z=p,
            irfc_dist_sq=None,
            n_queries=self.n_queries,
            n_keys=self.n_keys,
            inplace_safe=inplace_safe,
            chunk_size=chunk_size,
        )


class ConditionedTransitionBlock(nn.Module):
    """
    Implements Algorithm 25 in AF3
    """

    def __init__(self, c_a: int, c_s: int, n: int = 2, biasinit: float = -2.0) -> None:
        """
        Args:
            c_a (int, optional): single embedding dim (single feature aggregated atom info).
            c_s (int, optional):  single embedding dim.
            n (int, optional): channel scale factor. Defaults to 2.
        """
        super(ConditionedTransitionBlock, self).__init__()
        self.c_a = c_a
        self.c_s = c_s
        self.n = n
        self.adaln = AdaptiveLayerNorm(c_a=c_a, c_s=c_s)
        self.linear_nobias_a1 = LinearNoBias(in_features=c_a, out_features=n * c_a)
        self.linear_nobias_a2 = LinearNoBias(in_features=c_a, out_features=n * c_a)
        self.linear_nobias_b = LinearNoBias(in_features=n * c_a, out_features=c_a)
        self.linear_s = BiasInitLinear(
            in_features=c_s, out_features=c_a, bias=True, biasinit=biasinit
        )

    def forward(self, a: torch.Tensor, s: torch.Tensor) -> torch.Tensor:
        """
        Args:
            a (torch.Tensor): the single feature aggregate per-atom representation
                [..., N, c_a]
            s (torch.Tensor): single embedding
                [..., N, c_s]

        Returns:
            torch.Tensor: the updated a from ConditionedTransitionBlock
                [..., N, c_a]
        """
        a = self.adaln(a, s)
        b = F.silu((self.linear_nobias_a1(a))) * self.linear_nobias_a2(a)
        # Output projection (from adaLN-Zero [27])
        a = torch.sigmoid(self.linear_s(s)) * self.linear_nobias_b(b)
        return a


class AtomAttentionEncoder(nn.Module):
    """
    Implements Algorithm 5 in AF3
    atom attention encoder for coordinates
    """

    def __init__(
        self,
        has_coords: bool,
        c_token: int,  # 384 or 768
        c_atom: int = 128,
        c_atompair: int = 16,
        c_s: int = 384,
        c_z: int = 128,
        n_blocks: int = 3,
        n_heads: int = 4,
        n_queries: int = 32,
        n_keys: int = 128,
        blocks_per_ckpt: Optional[int] = None,
    ) -> None:
        """
        Args:
            has_coords (bool): whether the module input will contains coordinates (r_l).
            c_token (int): token embedding dim.
            c_atom (int, optional): atom embedding dim. Defaults to 128.
            c_atompair (int, optional): atompair embedding dim. Defaults to 16.
            c_s (int, optional):  single embedding dim. Defaults to 384.
            c_z (int, optional): pair embedding dim. Defaults to 128.
            n_blocks (int, optional): number of blocks in AtomTransformer. Defaults to 3.
            n_heads (int, optionall): number of heads in AtomTransformer. Defaults to 4.
            blocks_per_ckpt: number of AtomAttentionEncoder/AtomTransformer blocks in each activation checkpoint
                Size of each chunk. A higher value corresponds to fewer
                checkpoints, and trades memory for speed. If None, no checkpointing
                is performed.
        """
        super(AtomAttentionEncoder, self).__init__()
        self.has_coords = has_coords
        self.c_atom = c_atom
        self.c_atompair = c_atompair
        self.c_token = c_token
        self.c_s = c_s
        self.c_z = c_z
        self.n_queries = n_queries
        self.n_keys = n_keys
        self.local_attention_method = "local_cross_attention"

        self.input_feature = {
            "ref_pos": 3,
            "ref_charge": 1,
            "ref_mask": 1,
            "ref_element": 128,
            "ref_atom_name_chars": 4 * 64,
        }
        self.linear_no_bias_f = LinearNoBias(
            in_features=sum(self.input_feature.values()), out_features=self.c_atom
        )
        self.linear_no_bias_d = LinearNoBias(
            in_features=3, out_features=self.c_atompair
        )
        self.linear_no_bias_invd = LinearNoBias(
            in_features=1, out_features=self.c_atompair
        )
        self.linear_no_bias_v = LinearNoBias(
            in_features=1, out_features=self.c_atompair
        )

        if self.has_coords:
            # Line9
            self.layernorm_s = LayerNorm(self.c_s)
            self.linear_no_bias_s = LinearNoBias(
                in_features=self.c_s, out_features=self.c_atom
            )
            # Line10
            self.layernorm_z = LayerNorm(self.c_z)  # memory bottleneck
            self.linear_no_bias_z = LinearNoBias(
                in_features=self.c_z, out_features=self.c_atompair
            )
            # Line11
            self.linear_no_bias_r = LinearNoBias(
                in_features=3, out_features=self.c_atom
            )
        self.linear_no_bias_cl = LinearNoBias(
            in_features=self.c_atom, out_features=self.c_atompair
        )
        self.linear_no_bias_cm = LinearNoBias(
            in_features=self.c_atom, out_features=self.c_atompair
        )
        self.small_mlp = nn.Sequential(
            nn.ReLU(),
            LinearNoBias(in_features=self.c_atompair, out_features=self.c_atompair),
            nn.ReLU(),
            LinearNoBias(in_features=self.c_atompair, out_features=self.c_atompair),
            nn.ReLU(),
            LinearNoBias(in_features=self.c_atompair, out_features=self.c_atompair),
        )
        self.atom_transformer = AtomTransformer(
            n_blocks=n_blocks,
            n_heads=n_heads,
            c_atom=c_atom,
            c_atompair=c_atompair,
            n_queries=n_queries,
            n_keys=n_keys,
            blocks_per_ckpt=blocks_per_ckpt,
        )
        self.linear_no_bias_q = LinearNoBias(
            in_features=self.c_atom, out_features=self.c_token
        )

    def linear_init(
        self,
        zero_init_atom_encoder_residual_linear: bool = False,
        he_normal_init_atom_encoder_small_mlp: bool = False,
        he_normal_init_atom_encoder_output: bool = False,
    ):
        """
        Initializes the parameters of the diffusion module according to the provided initialization configuration.

        Args:
            zero_init_atom_encoder_residual_linear (bool): Whether to zero-initialize the residual linear layers.
            he_normal_init_atom_encoder_small_mlp (bool): Whether to initialize the small MLP layers with He normal initialization.
            he_normal_init_atom_encoder_output (bool): Whether to initialize the output layer with He normal initialization.
        """

        if zero_init_atom_encoder_residual_linear:
            nn.init.zeros_(self.linear_no_bias_invd.weight)
            nn.init.zeros_(self.linear_no_bias_v.weight)
            nn.init.zeros_(self.linear_no_bias_s.weight)
            nn.init.zeros_(self.linear_no_bias_z.weight)
            nn.init.zeros_(self.linear_no_bias_r.weight)
            nn.init.zeros_(self.linear_no_bias_cl.weight)
            nn.init.zeros_(self.linear_no_bias_cm.weight)
        if he_normal_init_atom_encoder_small_mlp:
            for layer in self.small_mlp:
                if not isinstance(layer, torch.nn.modules.activation.ReLU):
                    nn.init.kaiming_normal_(
                        layer.weight,
                        a=0,
                        mode="fan_in",
                        nonlinearity="relu",
                    )
        if he_normal_init_atom_encoder_output:
            nn.init.kaiming_normal_(
                self.linear_no_bias_q.weight, a=0, mode="fan_in", nonlinearity="relu"
            )

    def forward(
        self,
        input_feature_dict: dict[str, Union[torch.Tensor, int, float, dict]],
        r_l: torch.Tensor = None,
        s: torch.Tensor = None,
        z: torch.Tensor = None,
        inplace_safe: bool = False,
        chunk_size: Optional[int] = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Args:
            input_feature_dict (dict[str, Union[torch.Tensor, int, float, dict]]): input meta feature dict
            r_l (torch.Tensor, optional): noisy position.
                [..., N_sample, N_atom, 3] if has_coords else None.
            s (torch.Tensor, optional): single embedding.
                [..., N_sample, N_token, c_s] if has_coords else None.
            z (torch.Tensor, optional): pair embedding
                [..., N_sample, N_token, N_token, c_z] if has_coords else None.

        Returns:
            tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]: the output of AtomAttentionEncoder
            a:
                [..., (N_sample), N_token, c_token]
            q_l:
                [..., (N_sample), N_atom, c_atom]
            c_l:
                [..., (N_sample), N_atom, c_atom]
            p_lm:
                [..., (N_sample), N_atom, N_atom, c_atompair]

        """

        if self.has_coords:
            assert r_l is not None
            assert s is not None
            assert z is not None

        atom_to_token_idx = input_feature_dict["atom_to_token_idx"]
        # Create the atom single conditioning: Embed per-atom meta data
        # [..., N_atom, C_atom]
        batch_shape = input_feature_dict["ref_pos"].shape[:-2]
        N_atom = input_feature_dict["ref_pos"].shape[-2]
        c_l = self.linear_no_bias_f(
            torch.cat(
                [
                    input_feature_dict[name].reshape(
                        *batch_shape, N_atom, self.input_feature[name]
                    )
                    for name in self.input_feature
                ],
                dim=-1,
            )
        )

        # Line2-Line4: Embed offsets between atom reference positions

        # Prepare tensors in dense trunks for local operations
        q_trunked_list, k_trunked_list, pad_info = rearrange_qk_to_dense_trunk(
            q=[input_feature_dict["ref_pos"], input_feature_dict["ref_space_uid"]],
            k=[input_feature_dict["ref_pos"], input_feature_dict["ref_space_uid"]],
            dim_q=[-2, -1],
            dim_k=[-2, -1],
            n_queries=self.n_queries,
            n_keys=self.n_keys,
            compute_mask=True,
        )

        # Compute atom pair feature
        d_lm = (
            q_trunked_list[0][..., None, :] - k_trunked_list[0][..., None, :, :]
        )  # [..., n_blocks, n_queries, n_keys, 3]
        irfc_dist_sq_local = (d_lm**2).sum(dim=-1) # [..., n_blocks, n_queries, n_keys]
        v_lm = (
            q_trunked_list[1][..., None].int() == k_trunked_list[1][..., None, :].int()
        ).unsqueeze(
            dim=-1
        )  # [..., n_blocks, n_queries, n_keys, 1]
        p_lm = (self.linear_no_bias_d(d_lm) * v_lm) * pad_info[
            "mask_trunked"
        ].unsqueeze(
            dim=-1
        )  # [..., n_blocks, n_queries, n_keys, C_atompair]

        # Line5-Line6: Embed pairwise inverse squared distances, and the valid mask
        if inplace_safe:
            p_lm += (
                self.linear_no_bias_invd(1 / (1 + (d_lm**2).sum(dim=-1, keepdim=True)))
                * v_lm
            )
            p_lm += self.linear_no_bias_v(v_lm.to(dtype=p_lm.dtype)) * v_lm
        else:
            p_lm = (
                p_lm
                + self.linear_no_bias_invd(
                    1 / (1 + (d_lm**2).sum(dim=-1, keepdim=True))
                )
                * v_lm
            )
            p_lm = p_lm + self.linear_no_bias_v(v_lm.to(dtype=p_lm.dtype)) * v_lm

        # Line7: Initialise the atom single representation as the single conditioning
        q_l = c_l.clone()

        # If provided, add trunk embeddings and noisy positions
        n_token = None
        if r_l is not None:
            N_sample = r_l.size(-3)

            # Broadcast the single and pair embedding from the trunk
            n_token = s.size(-2)
            c_l = c_l.unsqueeze(dim=-3) + self.linear_no_bias_s(
                self.layernorm_s(
                    broadcast_token_to_atom(
                        x_token=s, atom_to_token_idx=atom_to_token_idx
                    )
                )
            )  # [..., N_sample, N_atom, c_atom]
            z_local_pairs, _ = broadcast_token_to_local_atom_pair(
                z_token=z,
                atom_to_token_idx=atom_to_token_idx,
                n_queries=self.n_queries,
                n_keys=self.n_keys,
                compute_mask=False,
            )  # [..., N_sample, n_blocks, n_queries, n_keys, c_z]
            p_lm = p_lm.unsqueeze(dim=-5) + self.linear_no_bias_z(
                self.layernorm_z(z_local_pairs)
            )  # [..., N_sample, n_blocks, n_queries, n_keys, c_atompair]

            # Add the noisy positions
            q_l = q_l.unsqueeze(dim=-3) + self.linear_no_bias_r(
                r_l
            )  # [..., N_sample, N_atom, c_atom]

        # Add the combined single conditioning to the pair representation
        c_l_q, c_l_k, _ = rearrange_qk_to_dense_trunk(
            q=c_l,
            k=c_l,
            dim_q=-2,
            dim_k=-2,
            n_queries=self.n_queries,
            n_keys=self.n_keys,
            compute_mask=False,
        )
        if inplace_safe:
            p_lm += self.linear_no_bias_cl(F.relu(c_l_q[..., None, :]))
            p_lm += self.linear_no_bias_cm(F.relu(c_l_k[..., None, :, :]))
            p_lm += self.small_mlp(p_lm)
        else:
            p_lm = (
                p_lm
                + self.linear_no_bias_cl(F.relu(c_l_q[..., None, :]))
                + self.linear_no_bias_cm(F.relu(c_l_k[..., None, :, :]))
            )  # [..., (N_sample), n_blocks, n_queries, n_keys, c_atompair]

            # Run a small MLP on the pair activations
            p_lm = p_lm + self.small_mlp(p_lm)

        # Cross attention transformer
        q_l = self.atom_transformer(
            q_l, c_l, p_lm, chunk_size=chunk_size
        )  # [..., (N_sample), N_atom, c_atom]

        # Aggregate per-atom representation to per-token representation
        a = aggregate_atom_to_token(
            x_atom=F.relu(self.linear_no_bias_q(q_l)),
            atom_to_token_idx=atom_to_token_idx,
            n_token=n_token,
            reduce="mean",
        )  # [..., (N_sample), N_token, c_token]
        if (not self.training) and (a.shape[-2] > 2000 or q_l.shape[-2] > 20000):
            torch.cuda.empty_cache()
        return a, q_l, c_l, p_lm

# class SpectrumAtomAttentionEncoder(nn.Module):
#     """
#     Modified Implements Algorithm 5 in AF3
#     atom attention encoder for spectrum
#     """

#     def __init__(
#         self,
#         has_spectrum: bool,
#         c_token: int,  # 384 or 768
#         c_atom: int = 128,
#         c_atompair: int = 16,
#         c_s: int = 384,
#         c_z: int = 128,
#         n_blocks: int = 3,
#         n_heads: int = 4,
#         n_queries: int = 32,
#         n_keys: int = 128,
#         spectrum_dim: int = 6,
#         blocks_per_ckpt: Optional[int] = None,
#         protenix= None,
#         use_lora: bool = False,
#         lora_rank: int = 8,
#         lora_alpha: float = 16.0,
#         lora_dropout: float = 0.0,
#     ) -> None:
#         """
#         Args:
#             has_spectrum (bool): whether the module input will contains spectrum (r_l).
#             c_token (int): token embedding dim.
#             c_atom (int, optional): atom embedding dim. Defaults to 128.
#             c_atompair (int, optional): atompair embedding dim. Defaults to 16.
#             c_s (int, optional):  single embedding dim. Defaults to 384.
#             c_z (int, optional): pair embedding dim. Defaults to 128.
#             n_blocks (int, optional): number of blocks in AtomTransformer. Defaults to 3.
#             n_heads (int, optionall): number of heads in AtomTransformer. Defaults to 4.
#             blocks_per_ckpt: number of AtomAttentionEncoder/AtomTransformer blocks in each activation checkpoint
#                 Size of each chunk. A higher value corresponds to fewer
#                 checkpoints, and trades memory for speed. If None, no checkpointing
#                 is performed.
#         """
#         super(SpectrumAtomAttentionEncoder, self).__init__()
#         self.has_spectrum = has_spectrum
#         self.c_atom = c_atom
#         self.c_atompair = c_atompair
#         self.c_token = c_token
#         self.c_s = c_s
#         self.c_z = c_z
#         self.n_queries = n_queries
#         self.n_keys = n_keys
#         self.spectrum_dim = spectrum_dim
#         self.local_attention_method = "local_cross_attention"

#         self.input_feature = {
#             "ref_pos": 3,
#             "ref_charge": 1,
#             "ref_mask": 1,
#             "ref_element": 128,
#             "ref_atom_name_chars": 4 * 64,
#         }
#         self.linear_no_bias_f = LinearNoBias(
#             in_features=sum(self.input_feature.values()), out_features=self.c_atom
#         )
#         self.linear_no_bias_d = LinearNoBias(
#             in_features=3, out_features=self.c_atompair
#         )
#         self.linear_no_bias_invd = LinearNoBias(
#             in_features=1, out_features=self.c_atompair
#         )
#         self.linear_no_bias_v = LinearNoBias(
#             in_features=1, out_features=self.c_atompair
#         )

#         if self.has_spectrum:
#             # Line9
#             self.layernorm_s = LayerNorm(self.c_s)
#             self.linear_no_bias_s = LinearNoBias(
#                 in_features=self.c_s, out_features=self.c_atom
#             )
#             # Line10
#             self.layernorm_z = LayerNorm(self.c_z)  # memory bottleneck
#             self.linear_no_bias_z = LinearNoBias(
#                 in_features=self.c_z, out_features=self.c_atompair
#             )
#             # Line11
#             self.linear_no_bias_r = LinearNoBias(
#                 in_features=self.spectrum_dim, out_features=self.c_atom
#             )
#         self.linear_no_bias_cl = LinearNoBias(
#             in_features=self.c_atom, out_features=self.c_atompair
#         )
#         self.linear_no_bias_cm = LinearNoBias(
#             in_features=self.c_atom, out_features=self.c_atompair
#         )
#         self.small_mlp = nn.Sequential(
#             nn.ReLU(),
#             LinearNoBias(in_features=self.c_atompair, out_features=self.c_atompair),
#             nn.ReLU(),
#             LinearNoBias(in_features=self.c_atompair, out_features=self.c_atompair),
#             nn.ReLU(),
#             LinearNoBias(in_features=self.c_atompair, out_features=self.c_atompair),
#         )
#         self.atom_transformer = AtomTransformer(
#             n_blocks=n_blocks,
#             n_heads=n_heads,
#             c_atom=c_atom,
#             c_atompair=c_atompair,
#             n_queries=n_queries,
#             n_keys=n_keys,
#             blocks_per_ckpt=blocks_per_ckpt,
#         )
#         self.linear_no_bias_q = LinearNoBias(
#             in_features=self.c_atom, out_features=self.c_token
#         )
#         self.use_lora = use_lora
#         self.lora_rank = lora_rank
#         self.lora_alpha = lora_alpha
#         self.lora_dropout = lora_dropout
#         self._register_lora_modules(protenix)

#     def _register_lora_modules(self, protenix):
#         if self.has_spectrum:
#             nn.init.zeros_(self.linear_no_bias_r.weight) # has different input shape with original atom attention encoder

#         # {self_name: module_object} - stores module object references
#         # 直接存储对象引用，如果属性不存在会在赋值时抛出 AttributeError
#         try:
#             self.layer_map = {
#                 "linear_no_bias_f": protenix.diffusion_module.atom_attention_encoder.linear_no_bias_f,
#                 "linear_no_bias_d": protenix.diffusion_module.atom_attention_encoder.linear_no_bias_d,
#                 "linear_no_bias_invd": protenix.diffusion_module.atom_attention_encoder.linear_no_bias_invd,
#                 "linear_no_bias_v": protenix.diffusion_module.atom_attention_encoder.linear_no_bias_v,
#                 "linear_no_bias_s": protenix.diffusion_module.atom_attention_encoder.linear_no_bias_s,
#                 "layernorm_s": protenix.diffusion_module.atom_attention_encoder.layernorm_s,
#                 "linear_no_bias_z": protenix.diffusion_module.atom_attention_encoder.linear_no_bias_z,
#                 "layernorm_z": protenix.diffusion_module.atom_attention_encoder.layernorm_z,
#                 "linear_no_bias_cl": protenix.diffusion_module.atom_attention_encoder.linear_no_bias_cl,
#                 "linear_no_bias_cm": protenix.diffusion_module.atom_attention_encoder.linear_no_bias_cm,
#                 "small_mlp": protenix.diffusion_module.atom_attention_encoder.small_mlp,
#                 "atom_transformer": protenix.diffusion_module.atom_attention_encoder.atom_transformer,
#                 "linear_no_bias_q": protenix.diffusion_module.atom_attention_encoder.linear_no_bias_q,
#             }
#         except AttributeError as e:
#             # 如果某个属性不存在，提供更详细的错误信息
#             available = [k for k in dir(protenix.diffusion_module) if not k.startswith('_')]
#             raise AttributeError(
#                 f"Cannot access protenix.diffusion_module attribute: {e}\n"
#                 f"Available attributes: {available}"
#             ) from e
        
#         self.lora_wrapped_modules = {}
        
#         # --- Assign mapped layers as self attributes, and optionally apply LoRA ---
#         # 注意：如果传入的 protenix 已经加载了权重，那么 base_module 中天生就有权重
#         # LoRA 包装时，base_layer 直接引用原始的 Linear 层，权重会自动保留
#         # 因此不需要再次加载权重，除非需要从 checkpoint 重新加载
#         failed_modules = []
#         for self_name, base_module in self.layer_map.items():
#             try:
#                 # Check if base_module is None
#                 if base_module is None:
#                     raise ValueError(f"base_module for '{self_name}' is None")
                
#                 getattr(self, self_name).load_state_dict(base_module.state_dict(), strict=True)

#                 if self.use_lora:
#                     # Only wrap if not already wrapped
#                     if not isinstance(getattr(self, self_name), LoRAModule):
#                         module_obj=LoRAModule(
#                             base_module=getattr(self, self_name),
#                             target_modules=None, # Apply to all Linear layers
#                             rank=self.lora_rank,
#                             alpha=self.lora_alpha,
#                             dropout=self.lora_dropout,
#                         )
#                     else:
#                         module_obj = getattr(self, self_name)
#                 else:
#                     module_obj = getattr(self, self_name)

#                 setattr(self, self_name, module_obj)
#                 if isinstance(getattr(self, self_name), LoRAModule):
#                     self.lora_wrapped_modules[self_name] = module_obj

#             except Exception as e:
#                 failed_modules.append(self_name)
#                 warnings.warn(f"Failed to process module '{self_name}': {traceback.format_exc()}")

#     def linear_init(
#         self,
#         zero_init_atom_encoder_residual_linear: bool = False,
#         he_normal_init_atom_encoder_small_mlp: bool = False,
#         he_normal_init_atom_encoder_output: bool = False,
#     ):
#         """
#         Initializes the parameters of the diffusion module according to the provided initialization configuration.

#         Args:
#             zero_init_atom_encoder_residual_linear (bool): Whether to zero-initialize the residual linear layers.
#             he_normal_init_atom_encoder_small_mlp (bool): Whether to initialize the small MLP layers with He normal initialization.
#             he_normal_init_atom_encoder_output (bool): Whether to initialize the output layer with He normal initialization.
#         """

#         if zero_init_atom_encoder_residual_linear:
#             nn.init.zeros_(self.linear_no_bias_invd.weight)
#             nn.init.zeros_(self.linear_no_bias_v.weight)
#             nn.init.zeros_(self.linear_no_bias_s.weight)
#             nn.init.zeros_(self.linear_no_bias_z.weight)
#             nn.init.zeros_(self.linear_no_bias_r.weight)
#             nn.init.zeros_(self.linear_no_bias_cl.weight)
#             nn.init.zeros_(self.linear_no_bias_cm.weight)
#         if he_normal_init_atom_encoder_small_mlp:
#             for layer in self.small_mlp:
#                 if not isinstance(layer, torch.nn.modules.activation.ReLU):
#                     nn.init.kaiming_normal_(
#                         layer.weight,
#                         a=0,
#                         mode="fan_in",
#                         nonlinearity="relu",
#                     )
#         if he_normal_init_atom_encoder_output:
#             nn.init.kaiming_normal_(
#                 self.linear_no_bias_q.weight, a=0, mode="fan_in", nonlinearity="relu"
#             )

#     def forward(
#         self,
#         input_feature_dict: dict[str, Union[torch.Tensor, int, float, dict]],
#         r_l: torch.Tensor = None,
#         s: torch.Tensor = None,
#         z: torch.Tensor = None,
#         inplace_safe: bool = False,
#         chunk_size: Optional[int] = None,
#     ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
#         """
#         Args:
#             input_feature_dict (dict[str, Union[torch.Tensor, int, float, dict]]): input meta feature dict
#             r_l (torch.Tensor, optional): noisy spectrum.
#                 [..., seq_len, N_atom, spectrum_dim] if has_spectrum else None.
#             s (torch.Tensor, optional): single embedding.
#                 [..., seq_len, N_token, c_s] if has_spectrum else None.
#             z (torch.Tensor, optional): pair embedding.
#                 [..., seq_len, N_token, N_token, c_z] if has_spectrum else None.

#         Returns:
#             tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]: the output of SpectrumAtomAttentionEncoder
#             a:
#                 [..., (seq_len), N_token, c_token]
#             q_l:
#                 [..., (seq_len), N_atom, c_atom]
#             c_l:
#                 [..., (seq_len), N_atom, c_atom]
#             p_lm:
#                 [..., (seq_len), N_atom, N_atom, c_atompair]

#         """

#         if self.has_spectrum:
#             assert r_l is not None
#             assert s is not None
#             assert z is not None

#         atom_to_token_idx = input_feature_dict["atom_to_token_idx"]
#         # Create the atom single conditioning: Embed per-atom meta data
#         # [..., N_atom, C_atom]
#         batch_shape = input_feature_dict["ref_pos"].shape[:-2]
#         N_atom = input_feature_dict["ref_pos"].shape[-2]
#         c_l = self.linear_no_bias_f(
#             torch.cat(
#                 [
#                     input_feature_dict[name].reshape(
#                         *batch_shape, N_atom, self.input_feature[name]
#                     )
#                     for name in self.input_feature
#                 ],
#                 dim=-1,
#             )
#         )

#         # Line2-Line4: Embed offsets between atom reference positions

#         # Prepare tensors in dense trunks for local operations
#         q_trunked_list, k_trunked_list, pad_info = rearrange_qk_to_dense_trunk(
#             q=[input_feature_dict["ref_pos"], input_feature_dict["ref_space_uid"]],
#             k=[input_feature_dict["ref_pos"], input_feature_dict["ref_space_uid"]],
#             dim_q=[-2, -1],
#             dim_k=[-2, -1],
#             n_queries=self.n_queries,
#             n_keys=self.n_keys,
#             compute_mask=True,
#         )

#         # Compute atom pair feature
#         d_lm = (
#             q_trunked_list[0][..., None, :] - k_trunked_list[0][..., None, :, :]
#         )  # [..., n_blocks, n_queries, n_keys, 3]
#         v_lm = (
#             q_trunked_list[1][..., None].int() == k_trunked_list[1][..., None, :].int()
#         ).unsqueeze(
#             dim=-1
#         )  # [..., n_blocks, n_queries, n_keys, 1]
#         p_lm = (self.linear_no_bias_d(d_lm) * v_lm) * pad_info[
#             "mask_trunked"
#         ].unsqueeze(
#             dim=-1
#         )  # [..., n_blocks, n_queries, n_keys, C_atompair]

#         # Line5-Line6: Embed pairwise inverse squared distances, and the valid mask
#         if inplace_safe:
#             p_lm += (
#                 self.linear_no_bias_invd(1 / (1 + (d_lm**2).sum(dim=-1, keepdim=True)))
#                 * v_lm
#             )
#             p_lm += self.linear_no_bias_v(v_lm.to(dtype=p_lm.dtype)) * v_lm
#         else:
#             p_lm = (
#                 p_lm
#                 + self.linear_no_bias_invd(
#                     1 / (1 + (d_lm**2).sum(dim=-1, keepdim=True))
#                 )
#                 * v_lm
#             )
#             p_lm = p_lm + self.linear_no_bias_v(v_lm.to(dtype=p_lm.dtype)) * v_lm

#         # Line7: Initialise the atom single representation as the single conditioning
#         q_l = c_l.clone()

#         # If provided, add trunk embeddings and noisy positions
#         n_token = None
#         if r_l is not None:
#             seq_len = r_l.size(-3)

#             # Broadcast the single and pair embedding from the trunk
#             n_token = s.size(-2)
#             c_l = c_l.unsqueeze(dim=-3) + self.linear_no_bias_s(
#                 self.layernorm_s(
#                     broadcast_token_to_atom(
#                         x_token=s, atom_to_token_idx=atom_to_token_idx
#                     )
#                 )
#             )  # [..., seq_len, N_atom, c_atom]
#             z_local_pairs, _ = broadcast_token_to_local_atom_pair(
#                 z_token=z,
#                 atom_to_token_idx=atom_to_token_idx,
#                 n_queries=self.n_queries,
#                 n_keys=self.n_keys,
#                 compute_mask=False,
#             )  # [..., seq_len, n_blocks, n_queries, n_keys, c_z]
#             p_lm = p_lm.unsqueeze(dim=-5) + self.linear_no_bias_z(
#                 self.layernorm_z(z_local_pairs)
#             )  # [..., seq_len, n_blocks, n_queries, n_keys, c_atompair]

#             # Add the noisy positions
#             q_l = q_l.unsqueeze(dim=-3) + self.linear_no_bias_r(
#                 r_l
#             )  # [..., seq_len, N_atom, c_atom]

#         # Add the combined single conditioning to the pair representation
#         c_l_q, c_l_k, _ = rearrange_qk_to_dense_trunk(
#             q=c_l,
#             k=c_l,
#             dim_q=-2,
#             dim_k=-2,
#             n_queries=self.n_queries,
#             n_keys=self.n_keys,
#             compute_mask=False,
#         )
#         if inplace_safe:
#             p_lm += self.linear_no_bias_cl(F.relu(c_l_q[..., None, :]))
#             p_lm += self.linear_no_bias_cm(F.relu(c_l_k[..., None, :, :]))
#             p_lm += self.small_mlp(p_lm)
#         else:
#             p_lm = (
#                 p_lm
#                 + self.linear_no_bias_cl(F.relu(c_l_q[..., None, :]))
#                 + self.linear_no_bias_cm(F.relu(c_l_k[..., None, :, :]))
#             )  # [..., (seq_len), n_blocks, n_queries, n_keys, c_atompair]

#             # Run a small MLP on the pair activations
#             p_lm = p_lm + self.small_mlp(p_lm)

#         # Cross attention transformer
#         q_l = self.atom_transformer(
#             q_l, c_l, p_lm, chunk_size=chunk_size
#         )  # [..., (seq_len), N_atom, c_atom]

#         # Aggregate per-atom representation to per-token representation
#         a = aggregate_atom_to_token(
#             x_atom=F.relu(self.linear_no_bias_q(q_l)),
#             atom_to_token_idx=atom_to_token_idx,
#             n_token=n_token,
#             reduce="mean",
#         )  # [..., (seq_len), N_token, c_token]
#         if (not self.training) and (a.shape[-2] > 2000 or q_l.shape[-2] > 20000):
#             torch.cuda.empty_cache()
#         return a, q_l, c_l, p_lm

#     def forward_simple(
#         self,
#         input_feature_dict: dict[str, Union[torch.Tensor, int, float, dict]],
#         r_l: torch.Tensor = None,
#         s: torch.Tensor = None,
#         z: torch.Tensor = None,
#         c_l: torch.Tensor = None, # atom single representation
#         p_lm: torch.Tensor = None, # atom pair representation
#         inplace_safe: bool = False,
#         chunk_size: Optional[int] = None,
#     ):
#         if self.has_spectrum:
#             assert r_l is not None
#             assert s is not None
#             assert z is not None
#         atom_to_token_idx = input_feature_dict["atom_to_token_idx"]

#         if p_lm is None: # use_bihito=True
#             # Create the atom single conditioning: Embed per-atom meta data
#             # [..., N_atom, C_atom]
#             batch_shape = input_feature_dict["ref_pos"].shape[:-2]
#             N_atom = input_feature_dict["ref_pos"].shape[-2]
#             c_l = self.linear_no_bias_f(
#                 torch.cat(
#                     [
#                         input_feature_dict[name].reshape(
#                             *batch_shape, N_atom, self.input_feature[name]
#                         )
#                         for name in self.input_feature
#                     ],
#                     dim=-1,
#                 )
#             )

#             # Line2-Line4: Embed offsets between atom reference positions

#             # Prepare tensors in dense trunks for local operations
#             q_trunked_list, k_trunked_list, pad_info = rearrange_qk_to_dense_trunk(
#                 q=[input_feature_dict["ref_pos"], input_feature_dict["ref_space_uid"]],
#                 k=[input_feature_dict["ref_pos"], input_feature_dict["ref_space_uid"]],
#                 dim_q=[-2, -1],
#                 dim_k=[-2, -1],
#                 n_queries=self.n_queries,
#                 n_keys=self.n_keys,
#                 compute_mask=True,
#             )

#             # Compute atom pair feature
#             d_lm = (
#                 q_trunked_list[0][..., None, :] - k_trunked_list[0][..., None, :, :]
#             )  # [..., n_blocks, n_queries, n_keys, 3]
#             v_lm = (
#                 q_trunked_list[1][..., None].int() == k_trunked_list[1][..., None, :].int()
#             ).unsqueeze(
#                 dim=-1
#             )  # [..., n_blocks, n_queries, n_keys, 1]
#             p_lm = (self.linear_no_bias_d(d_lm) * v_lm) * pad_info[
#                 "mask_trunked"
#             ].unsqueeze(
#                 dim=-1
#             )  # [..., n_blocks, n_queries, n_keys, C_atompair]

#             # Line5-Line6: Embed pairwise inverse squared distances, and the valid mask
#             if inplace_safe:
#                 p_lm += (
#                     self.linear_no_bias_invd(1 / (1 + (d_lm**2).sum(dim=-1, keepdim=True)))
#                     * v_lm
#                 )
#                 p_lm += self.linear_no_bias_v(v_lm.to(dtype=p_lm.dtype)) * v_lm
#             else:
#                 p_lm = (
#                     p_lm
#                     + self.linear_no_bias_invd(
#                         1 / (1 + (d_lm**2).sum(dim=-1, keepdim=True))
#                     )
#                     * v_lm
#                 )
#                 p_lm = p_lm + self.linear_no_bias_v(v_lm.to(dtype=p_lm.dtype)) * v_lm

#         q_l = c_l.clone()
#         if r_l is not None:
#             seq_len = r_l.size(-3)

#             # Broadcast the single and pair embedding from the trunk
#             n_token = s.size(-2)
#             c_l = c_l.unsqueeze(dim=-3) + self.linear_no_bias_s(
#                 self.layernorm_s(
#                     broadcast_token_to_atom(
#                         x_token=s, atom_to_token_idx=atom_to_token_idx
#                     )
#                 )
#             )  # [..., seq_len, N_atom, c_atom]
#             z_local_pairs, _ = broadcast_token_to_local_atom_pair(
#                 z_token=z,
#                 atom_to_token_idx=atom_to_token_idx,
#                 n_queries=self.n_queries,
#                 n_keys=self.n_keys,
#                 compute_mask=False,
#             )  # [..., seq_len, n_blocks, n_queries, n_keys, c_z]
#             p_lm = p_lm.unsqueeze(dim=-5) + self.linear_no_bias_z(
#                 self.layernorm_z(z_local_pairs)
#             )  # [..., seq_len, n_blocks, n_queries, n_keys, c_atompair]

#             # Add the noisy positions
#             q_l = q_l.unsqueeze(dim=-3) + self.linear_no_bias_r(
#                 r_l
#             )  # [..., seq_len, N_atom, c_atom]
#         # Add the combined single conditioning to the pair representation
#         c_l_q, c_l_k, _ = rearrange_qk_to_dense_trunk(
#             q=c_l,
#             k=c_l,
#             dim_q=-2,
#             dim_k=-2,
#             n_queries=self.n_queries,
#             n_keys=self.n_keys,
#             compute_mask=False,
#         )
#         if inplace_safe:
#             p_lm += self.linear_no_bias_cl(F.relu(c_l_q[..., None, :]))
#             p_lm += self.linear_no_bias_cm(F.relu(c_l_k[..., None, :, :]))
#             p_lm += self.small_mlp(p_lm)
#         else:
#             p_lm = (
#                 p_lm
#                 + self.linear_no_bias_cl(F.relu(c_l_q[..., None, :]))
#                 + self.linear_no_bias_cm(F.relu(c_l_k[..., None, :, :]))
#             )  # [..., (seq_len), n_blocks, n_queries, n_keys, c_atompair]

#             # Run a small MLP on the pair activations
#             p_lm = p_lm + self.small_mlp(p_lm)

#         # Cross attention transformer
#         q_l = self.atom_transformer(
#             q_l, c_l, p_lm, chunk_size=chunk_size
#         )  # [..., (seq_len), N_atom, c_atom]

#         # Aggregate per-atom representation to per-token representation
#         a = aggregate_atom_to_token(
#             x_atom=F.relu(self.linear_no_bias_q(q_l)),
#             atom_to_token_idx=atom_to_token_idx,
#             n_token=n_token,
#             reduce="mean",
#         )  # [..., (seq_len), N_token, c_token]
#         if (not self.training) and (a.shape[-2] > 2000 or q_l.shape[-2] > 20000):
#             torch.cuda.empty_cache()
#         return a, q_l, c_l, p_lm

#     def merge_lora_weights(self):
#         if self.use_lora and hasattr(self, 'lora_wrapped_modules'):
#             for m in self.lora_wrapped_modules.values():
#                 if hasattr(m, 'merge_weights'):
#                     m.merge_weights()
#         print("[SpectrumAtomAttentionEncoder] LoRA weights merged into base layers.")

class AtomAttentionDecoder(nn.Module):
    """
    Implements Algorithm 6 in AF3
    """

    def __init__(
        self,
        n_blocks: int = 3,
        n_heads: int = 4,
        c_token: int = 384,
        c_atom: int = 128,
        c_atompair: int = 16,
        n_queries: int = 32,
        n_keys: int = 128,
        blocks_per_ckpt: Optional[int] = None,
    ) -> None:
        """
        Args:
            n_blocks (int, optional): number of blocks for AtomTransformer. Defaults to 3.
            n_heads (int, optional): number of heads for AtomTransformer. Defaults to 4.
            c_token (int, optional): feature channel of token (single a). Defaults to 384.
            c_atom (int, optional): embedding dim for atom embedding. Defaults to 128.
            c_atompair (int, optional): embedding dim for atom pair embedding.
            blocks_per_ckpt: number of AtomAttentionDecoder/AtomTransformer blocks in each activation checkpoint
                Size of each chunk. A higher value corresponds to fewer
                checkpoints, and trades memory for speed. If None, no checkpointing
                is performed.
        """
        super(AtomAttentionDecoder, self).__init__()
        self.n_blocks = n_blocks
        self.n_heads = n_heads
        self.c_token = c_token
        self.c_atom = c_atom
        self.c_atompair = c_atompair
        self.n_queries = n_queries
        self.n_keys = n_keys
        self.linear_no_bias_a = LinearNoBias(in_features=c_token, out_features=c_atom)
        self.layernorm_q = LayerNorm(c_atom)
        self.linear_no_bias_out = LinearNoBias(in_features=c_atom, out_features=3)
        self.atom_transformer = AtomTransformer(
            n_blocks=n_blocks,
            n_heads=n_heads,
            c_atom=c_atom,
            c_atompair=c_atompair,
            n_queries=n_queries,
            n_keys=n_keys,
            blocks_per_ckpt=blocks_per_ckpt,
        )

    def forward(
        self,
        input_feature_dict: dict[str, Union[torch.Tensor, int, float, dict]],
        a: torch.Tensor,
        q_skip: torch.Tensor,
        c_skip: torch.Tensor,
        p_skip: torch.Tensor,
        inplace_safe: bool = False,
        chunk_size: Optional[int] = None,
    ) -> torch.Tensor:
        """
        Args:
            input_feature_dict (dict[str, Union[torch.Tensor, int, float, dict]]): input meta feature dict
            a (torch.Tensor): the single feature aggregate per-atom representation
                [..., N_token, c_token]
            q_skip (torch.Tensor): atom single embedding
                [..., N_atom, c_atom]
            c_skip (torch.Tensor): atom single embedding
                [..., N_atom, c_atom]
            p_skip (torch.Tensor): atompair single embedding
                [..., n_blocks, n_queries, n_keys, c_atompair]

        Returns:
            torch.Tensor: the updated nosiy coordinates
                [..., N_atom, 3]
        """
        # Broadcast per-token activiations to per-atom activations and add the skip connection
        q = (
            self.linear_no_bias_a(
                broadcast_token_to_atom(
                    x_token=a, atom_to_token_idx=input_feature_dict["atom_to_token_idx"]
                )  # [..., N_atom, c_token]
            )  # [..., N_atom, c_atom]
            + q_skip
        )

        # Cross attention transformer
        q = self.atom_transformer(
            q, c_skip, p_skip, inplace_safe=inplace_safe, chunk_size=chunk_size
        )

        # Map to positions update
        r = self.linear_no_bias_out(self.layernorm_q(q))

        return r

class SpectrumAtomAttentionDecoder(nn.Module):
    """
    Modified Implements Algorithm 6 in AF3
    atom attention decoder for spectrums
    """

    def __init__(
        self,
        n_blocks: int = 3,
        n_heads: int = 4,
        c_token: int = 384,
        c_atom: int = 128,
        c_atompair: int = 16,
        n_queries: int = 32,
        n_keys: int = 128,
        spectrum_dim: int = 6,
        blocks_per_ckpt: Optional[int] = None,
        protenix = None,
        use_lora: bool = False,
        lora_rank: int = 8,
        lora_alpha: float = 16.0,
        lora_dropout: float = 0.0,
    ) -> None:
        """
        Args:
            n_blocks (int, optional): number of blocks for AtomTransformer. Defaults to 3.
            n_heads (int, optional): number of heads for AtomTransformer. Defaults to 4.
            c_token (int, optional): feature channel of token (single a). Defaults to 384.
            c_atom (int, optional): embedding dim for atom embedding. Defaults to 128.
            c_atompair (int, optional): embedding dim for atom pair embedding.
            spectrum_dim (int): 3+3=6(default) for real and imaginary parts of spectrums
            blocks_per_ckpt: number of AtomAttentionDecoder/AtomTransformer blocks in each activation checkpoint
                Size of each chunk. A higher value corresponds to fewer
                checkpoints, and trades memory for speed. If None, no checkpointing
                is performed.
        """
        super(SpectrumAtomAttentionDecoder, self).__init__()
        self.n_blocks = n_blocks
        self.n_heads = n_heads
        self.c_token = c_token
        self.c_atom = c_atom
        self.c_atompair = c_atompair
        self.n_queries = n_queries
        self.n_keys = n_keys
        self.spectrum_dim = spectrum_dim
        self.linear_no_bias_a = LinearNoBias(in_features=c_token, out_features=c_atom)
        self.layernorm_q = LayerNorm(c_atom)
        self.linear_no_bias_out = LinearNoBias(in_features=c_atom, out_features=spectrum_dim)
        self.atom_transformer = AtomTransformer(
            n_blocks=n_blocks,
            n_heads=n_heads,
            c_atom=c_atom,
            c_atompair=c_atompair,
            n_queries=n_queries,
            n_keys=n_keys,
            blocks_per_ckpt=blocks_per_ckpt,
        )
        self.use_lora = use_lora
        self.lora_rank = lora_rank
        self.lora_alpha = lora_alpha
        self.lora_dropout = lora_dropout
        self._register_lora_modules(protenix)

    def _register_lora_modules(self, protenix):
        nn.init.zeros_(self.linear_no_bias_out.weight) # has different input shape with original atom attention decoder

        # {self_name: module_object} - stores module object references
        # 直接存储对象引用，如果属性不存在会在赋值时抛出 AttributeError
        try:
            self.layer_map = {
                "linear_no_bias_a": protenix.diffusion_module.atom_attention_decoder.linear_no_bias_a,
                "layernorm_q": protenix.diffusion_module.atom_attention_decoder.layernorm_q,
                "atom_transformer": protenix.diffusion_module.atom_attention_decoder.atom_transformer,
            }
        except AttributeError as e:
            # 如果某个属性不存在，提供更详细的错误信息
            available = [k for k in dir(protenix.diffusion_module) if not k.startswith('_')]
            raise AttributeError(
                f"Cannot access protenix.diffusion_module attribute: {e}\n"
                f"Available attributes: {available}"
            ) from e
        
        self.lora_wrapped_modules = {}
        
        # --- Assign mapped layers as self attributes, and optionally apply LoRA ---
        # 注意：如果传入的 protenix 已经加载了权重，那么 base_module 中天生就有权重
        # LoRA 包装时，base_layer 直接引用原始的 Linear 层，权重会自动保留
        # 因此不需要再次加载权重，除非需要从 checkpoint 重新加载
        failed_modules = []
        for self_name, base_module in self.layer_map.items():
            try:
                # Check if base_module is None
                if base_module is None:
                    raise ValueError(f"base_module for '{self_name}' is None")
                
                getattr(self, self_name).load_state_dict(base_module.state_dict(), strict=True)

                if self.use_lora:
                    # Only wrap if not already wrapped
                    if not isinstance(getattr(self, self_name), LoRAModule):
                        module_obj=LoRAModule(
                            base_module=getattr(self, self_name),
                            target_modules=None, # Apply to all Linear layers
                            rank=self.lora_rank,
                            alpha=self.lora_alpha,
                            dropout=self.lora_dropout,
                        )
                    else:
                        module_obj = getattr(self, self_name)
                else:
                    module_obj = getattr(self, self_name)

                setattr(self, self_name, module_obj)
                if isinstance(getattr(self, self_name), LoRAModule):
                    self.lora_wrapped_modules[self_name] = module_obj

            except Exception as e:
                failed_modules.append(self_name)
                warnings.warn(f"Failed to process module '{self_name}': {traceback.format_exc()}")

    def forward(
        self,
        input_feature_dict: dict[str, Union[torch.Tensor, int, float, dict]],
        a: torch.Tensor,
        q_skip: torch.Tensor,
        c_skip: torch.Tensor,
        p_skip: torch.Tensor,
        inplace_safe: bool = False,
        chunk_size: Optional[int] = None,
    ) -> torch.Tensor:
        """
        Args:
            input_feature_dict (dict[str, Union[torch.Tensor, int, float, dict]]): input meta feature dict
            a (torch.Tensor): the single feature aggregate per-atom representation
                [..., N_token, c_token]
            q_skip (torch.Tensor): atom single embedding
                [..., N_atom, c_atom]
            c_skip (torch.Tensor): atom single embedding
                [..., N_atom, c_atom]
            p_skip (torch.Tensor): atompair single embedding
                [..., n_blocks, n_queries, n_keys, c_atompair]

        Returns:
            torch.Tensor: the updated nosiy coordinates
                [..., N_atom, 3]
        """
        # Broadcast per-token activiations to per-atom activations and add the skip connection
        q = (
            self.linear_no_bias_a(
                broadcast_token_to_atom(
                    x_token=a, atom_to_token_idx=input_feature_dict["atom_to_token_idx"]
                )  # [..., N_atom, c_token]
            )  # [..., N_atom, c_atom]
            + q_skip
        )

        # Cross attention transformer
        q = self.atom_transformer(
            q, c_skip, p_skip, inplace_safe=inplace_safe, chunk_size=chunk_size
        )

        # Map to positions update
        r = self.linear_no_bias_out(self.layernorm_q(q))

        return r
