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

from typing import Any, Optional, Union

import torch
import torch.nn as nn
import torch.nn.functional as F

from src.protenix.modules.primitives import LinearNoBias
from src.protenix.modules.transformer import AtomAttentionEncoder


class InputFeatureEmbedder(nn.Module):
    """
    Implements Algorithm 2 in AF3
    """

    def __init__(
        self,
        c_atom: int = 128,
        c_atompair: int = 16,
        c_token: int = 384,
        esm_configs: dict = {},
    ) -> None:
        """
        Args:
            c_atom (int, optional): atom embedding dim. Defaults to 128.
            c_atompair (int, optional): atom pair embedding dim. Defaults to 16.
            c_token (int, optional): token embedding dim. Defaults to 384.
        """
        super(InputFeatureEmbedder, self).__init__()
        self.c_atom = c_atom
        self.c_atompair = c_atompair
        self.c_token = c_token
        self.atom_attention_encoder = AtomAttentionEncoder(
            c_atom=c_atom,
            c_atompair=c_atompair,
            c_token=c_token,
            has_coords=False,
        )
        # Line2
        self.esm_configs = {
            "enable": esm_configs.get("enable", True),
            "model_name": esm_configs.get("model_name", "esm2-3b"),
            "embedding_dim": esm_configs.get("embedding_dim", 2560),
        }
        if self.esm_configs["enable"]:
            self.linear_esm = LinearNoBias(
                self.esm_configs["embedding_dim"],
                self.c_token + 32 + 32 + 1,
            )
            nn.init.zeros_(self.linear_esm.weight)
        self.input_feature = {"restype": 32, "profile": 32, "deletion_mean": 1}

    def forward(
        self,
        input_feature_dict: dict[str, Any],
        inplace_safe: bool = False,
        chunk_size: Optional[int] = None,
    ) -> torch.Tensor:
        """
        Args:
            input_feature_dict (Dict[str, Any]): dict of input features
            inplace_safe (bool): Whether it is safe to use inplace operations. Defaults to False.
            chunk_size (Optional[int]): Chunk size for memory-efficient operations. Defaults to None.

        Returns:
            torch.Tensor: token embedding
                [..., N_token, 384 (c_token) + 32 + 32 + 1 :=449]
        """
        # Embed per-atom features.
        a, _, _, _ = self.atom_attention_encoder(
            input_feature_dict=input_feature_dict,
            inplace_safe=inplace_safe,
            chunk_size=chunk_size,
        )  # [..., N_token, c_token]
        # Concatenate the per-token features.
        batch_shape = input_feature_dict["restype"].shape[:-1]
        s_inputs = torch.cat(
            [a]
            + [
                input_feature_dict[name].reshape(*batch_shape, d)
                for name, d in self.input_feature.items()
            ],
            dim=-1,
        )
        if self.esm_configs["enable"]:
            esm_embeddings = self.linear_esm(input_feature_dict["esm_token_embedding"])
            s_inputs = s_inputs + esm_embeddings
        return s_inputs


class RelativePositionEncoding(nn.Module):
    """
    Implements Algorithm 3 in AF3
    """

    def __init__(self, r_max: int = 32, s_max: int = 2, c_z: int = 128) -> None:
        """
        Args:
            r_max (int, optional): Relative position indices clip value. Defaults to 32.
            s_max (int, optional): Relative chain indices clip value. Defaults to 2.
            c_z (int, optional): hidden dim [for pair embedding]. Defaults to 128.
        """
        super(RelativePositionEncoding, self).__init__()
        self.r_max = r_max
        self.s_max = s_max
        self.c_z = c_z
        self.linear_no_bias = LinearNoBias(
            in_features=(4 * self.r_max + 2 * self.s_max + 7), out_features=self.c_z
        )
        self.input_feature = {
            "asym_id": 1,
            "residue_index": 1,
            "entity_id": 1,
            "sym_id": 1,
            "token_index": 1,
        }

    def forward(self, input_feature_dict: dict[str, Any]) -> torch.Tensor:
        """
        Args:
            input_feature_dict (Dict[str, Any]): input meta feature dict.
            asym_id / residue_index / entity_id / sym_id / token_index
                [..., N_tokens]
        Returns:
            torch.Tensor: relative position encoding
                [..., N_token, N_token, c_z]
        """
        b_same_chain = (
            input_feature_dict["asym_id"][..., :, None]
            == input_feature_dict["asym_id"][..., None, :]
        ).long()  # [..., N_token, N_token]
        b_same_residue = (
            input_feature_dict["residue_index"][..., :, None]
            == input_feature_dict["residue_index"][..., None, :]
        ).long()  # [..., N_token, N_token]
        b_same_entity = (
            input_feature_dict["entity_id"][..., :, None]
            == input_feature_dict["entity_id"][..., None, :]
        ).long()  # [..., N_token, N_token]
        d_residue = torch.clip(
            input=input_feature_dict["residue_index"][..., :, None]
            - input_feature_dict["residue_index"][..., None, :]
            + self.r_max,
            min=0,
            max=2 * self.r_max,
        ) * b_same_chain + (1 - b_same_chain) * (
            2 * self.r_max + 1
        )  # [..., N_token, N_token]
        a_rel_pos = F.one_hot(d_residue, 2 * (self.r_max + 1))
        d_token = torch.clip(
            input=input_feature_dict["token_index"][..., :, None]
            - input_feature_dict["token_index"][..., None, :]
            + self.r_max,
            min=0,
            max=2 * self.r_max,
        ) * b_same_chain * b_same_residue + (1 - b_same_chain * b_same_residue) * (
            2 * self.r_max + 1
        )  # [..., N_token, N_token]
        a_rel_token = F.one_hot(d_token, 2 * (self.r_max + 1))
        d_chain = torch.clip(
            input=input_feature_dict["sym_id"][..., :, None]
            - input_feature_dict["sym_id"][..., None, :]
            + self.s_max,
            min=0,
            max=2 * self.s_max,
        ) * b_same_entity + (1 - b_same_entity) * (
            2 * self.s_max + 1
        )  # [..., N_token, N_token]
        a_rel_chain = F.one_hot(d_chain, 2 * (self.s_max + 1))
        # 一视同仁，都不chunk，不然推理阶段容易报错
        p = self.linear_no_bias(
            torch.cat(
                [a_rel_pos, a_rel_token, b_same_entity[..., None], a_rel_chain],
                dim=-1,
            ).float()
        )  # [..., N_token, N_token, 2 * (self.r_max + 1)+ 2 * (self.r_max + 1)+ 1 + 2 * (self.s_max + 1)] -> [..., N_token, N_token, c_z]
        return p

        # if self.training:
        #     p = self.linear_no_bias(
        #         torch.cat(
        #             [a_rel_pos, a_rel_token, b_same_entity[..., None], a_rel_chain],
        #             dim=-1,
        #         ).float()
        #     )  # [..., N_token, N_token, 2 * (self.r_max + 1)+ 2 * (self.r_max + 1)+ 1 + 2 * (self.s_max + 1)] -> [..., N_token, N_token, c_z]
        #     return p
        # else:
        #     del d_chain, d_token, d_residue, b_same_chain, b_same_residue
        #     origin_shape = a_rel_pos.shape[:-1]
        #     Ntoken = a_rel_pos.shape[-2]
        #     a_rel_pos = a_rel_pos.reshape(-1, a_rel_pos.shape[-1])
        #     chunk_num = 1 if Ntoken < 3200 else 8
        #     a_rel_pos_chunks = torch.chunk(
        #         a_rel_pos.reshape(-1, a_rel_pos.shape[-1]), chunk_num, dim=-2
        #     )
        #     a_rel_token_chunks = torch.chunk(
        #         a_rel_token.reshape(-1, a_rel_token.shape[-1]), chunk_num, dim=-2
        #     )
        #     b_same_entity_chunks = torch.chunk(
        #         b_same_entity.reshape(-1, 1), chunk_num, dim=-2
        #     )
        #     a_rel_chain_chunks = torch.chunk(
        #         a_rel_chain.reshape(-1, a_rel_chain.shape[-1]), chunk_num, dim=-2
        #     )
        #     start = 0
        #     p = None
        #     for i in range(len(a_rel_pos_chunks)):
        #         data = torch.cat(
        #             [
        #                 a_rel_pos_chunks[i],
        #                 a_rel_token_chunks[i],
        #                 b_same_entity_chunks[i],
        #                 a_rel_chain_chunks[i],
        #             ],
        #             dim=-1,
        #         ).float()
        #         result = self.linear_no_bias(data)
        #         del data
        #         if p is None:
        #             p = torch.empty(
        #                 (a_rel_pos.shape[-2], self.c_z),
        #                 device=a_rel_pos.device,
        #                 dtype=result.dtype,
        #             )
        #         p[start : start + result.shape[0]] = result
        #         start += result.shape[0]
        #         del result
        #     del a_rel_pos, a_rel_token, b_same_entity, a_rel_chain
        #     p = p.reshape(*origin_shape, self.c_z)
        #     assert p.shape[-1] == self.c_z, \
        #         f"RelativePositionEncoding output last dim mismatch: {p.shape}"
        #     return p


class FourierEmbedding(nn.Module):
    """
    Implements Algorithm 22 in AF3
    """

    def __init__(self, c: int, seed: int = 42) -> None:
        """
        Args:
            c (int): embedding dim.
        """
        super(FourierEmbedding, self).__init__()
        self.c = c
        self.seed = seed
        generator = torch.Generator()
        generator.manual_seed(seed)
        w_value = torch.randn(size=(c,), generator=generator)
        self.w = nn.Parameter(w_value, requires_grad=False)
        b_value = torch.randn(size=(c,), generator=generator)
        self.b = nn.Parameter(b_value, requires_grad=False)

    def forward(self, t_hat_noise_level: torch.Tensor) -> torch.Tensor:
        """
        Args:
            t_hat_noise_level (torch.Tensor): the noise level
                [..., N_sample]

        Returns:
            torch.Tensor: the output fourier embedding
                [..., N_sample, c]
        """
        return torch.cos(
            input=2 * torch.pi * (t_hat_noise_level.unsqueeze(dim=-1) * self.w + self.b)
        )


class ConstraintEmbedder(nn.Module):
    """
    Implements Constraint Embedder
    """

    def __init__(
        self,
        pocket_embedder: dict[str:int],
        contact_embedder: dict[str:int],
        c_s,
        c_z,
        initialize_method: str = "kaiming",
    ) -> None:
        """

        Args:

        """
        super(ConstraintEmbedder, self).__init__()

        # contact embedder
        self.contact_embedder = LinearNoBias(
            in_features=contact_embedder.get("c_z_input", 2), out_features=c_z
        )

        # pocket embedder. Not used by now, coming soon
        self.is_pocket_s_embedder = LinearNoBias(
            in_features=pocket_embedder.get("c_s_input", 1),
            out_features=c_s,
        )
        self.is_pocket_z_embedder = LinearNoBias(
            in_features=pocket_embedder.get("c_z_input", 1),
            out_features=c_z,
        )

        if initialize_method == "zero":
            nn.init.zeros_(self.is_pocket_s_embedder.weight)
            nn.init.zeros_(self.is_pocket_z_embedder.weight)
            nn.init.zeros_(self.contact_embedder.weight)

    def forward(
        self,
        constraint_feature_dict: dict[str, Union[torch.Tensor, int, float, dict]],
    ) -> torch.Tensor:
        """

        Args:
            constraint_feature_dict (dict[str, Union[torch.Tensor, int, float, dict]]): dict of input features

        Returns:
            torch.Tensor: token embedding
                [..., N_token, c_s]
            torch.Tensor: token-pair embedding
                [..., N_token, N_token, c_s]
        """
        s_constraint, z_constraint = None, None

        z_constraint = self.contact_embedder(constraint_feature_dict["contact"])
        return s_constraint, z_constraint

class SimpleNexusAblation(nn.Module):
    """
    A minimal ablation module for NexusEncoder.
    It aggregates s_inputs, s_trunk, and z_trunk into a global_repr
    using simple pooling and fixed projections, with no trainable parameters.
    The other "enhanced" outputs are simply pass-throughs of the original inputs
    to maintain API compatibility.
    """
    def __init__(
        self,
        c_s=128,
        c_z=128,
        c_s_inputs=512,
        device=None,
        dtype=None,
        **kwargs  # Absorbs unused arguments like num_heads from NexusEncoder's init
    ):
        super(SimpleNexusAblation, self).__init__()
        self.c_s = c_s
        self.c_z = c_z  # Not directly used for output shaping if projecting all to c_s
        self.c_s_inputs = c_s_inputs # Not directly used for output shaping if projecting all to c_s
        self.device = device

        if isinstance(dtype, str):
            dtype_map = {
                "float16": torch.float16,
                "float32": torch.float32,
                "bfloat16": torch.bfloat16,
                "fp16": torch.float16,
                "fp32": torch.float32,
                "bf16": torch.bfloat16
            }
            self.dtype = dtype_map.get(dtype.lower(), torch.float32) # Default to float32 if string is unknown
        elif isinstance(dtype, torch.dtype):
            self.dtype = dtype
        elif dtype is None: # Default if not provided at all
             self.dtype = torch.float32
        else:
            raise ValueError(f"Invalid dtype: {dtype}")
        
        # print(f"------------------SimpleNexusAblation dtype: {self.dtype}------------------")
        # No trainable parameters are defined in this simple module.

    def _fixed_project(self, x, target_dim):
        """
        Projects a tensor x to a target_dim along its last dimension.
        Uses simple mean pooling (no trainable parameters, deterministic).
        Input x shape: [B, Features_in]
        Output shape: [B, target_dim]
        """
        if x.shape[-1] == target_dim:
            return x
        B, Features_in = x.shape
        # 计算每个目标维度对应的原始区间
        splits = torch.linspace(0, Features_in, steps=target_dim+1, dtype=torch.long)
        out = []
        for i in range(target_dim):
            start, end = splits[i].item(), splits[i+1].item()
            if end > start:
                out.append(x[:, start:end].mean(dim=1, keepdim=True))
            else:
                # 极端情况下，区间为空，补零
                out.append(torch.zeros(B, 1, device=x.device, dtype=x.dtype))
        return torch.cat(out, dim=1)

    def forward(self, s_inputs, s_trunk, z_trunk):
        """
        Aggregates the three input representations into a global representation.
        
        Args:
            s_inputs: Input features [N_tokens, c_s_inputs]
            s_trunk: Single-residue features [N_tokens, c_s]
            z_trunk: Residue-pair features [N_tokens, N_tokens, c_z]
            
        Returns:
            tuple: (s_inputs, s_trunk, z_trunk, global_repr)
                   The first three are pass-throughs of the original inputs (after type/device casting and batching).
                   global_repr has shape [c_s].
        """
        # Add batch dimension (B=1) and move to target device/dtype, similar to NexusEncoder
        _s_inputs = s_inputs.unsqueeze(0).to(device=self.device, dtype=self.dtype)
        _s_trunk = s_trunk.unsqueeze(0).to(device=self.device, dtype=self.dtype)
        _z_trunk = z_trunk.unsqueeze(0).to(device=self.device, dtype=self.dtype)

        # 1. Pool s_inputs over token dimension
        # _s_inputs shape: [B, N_tokens, c_s_inputs] -> s_inputs_pooled shape: [B, c_s_inputs]
        s_inputs_pooled = torch.mean(_s_inputs, dim=1)

        # 2. Pool s_trunk over token dimension
        # _s_trunk shape: [B, N_tokens, c_s] -> s_trunk_pooled shape: [B, c_s]
        s_trunk_pooled = torch.mean(_s_trunk, dim=1)

        # 3. Pool z_trunk over both token dimensions
        # _z_trunk shape: [B, N_tokens, N_tokens, c_z] -> z_trunk_pooled shape: [B, c_z]
        z_trunk_pooled = torch.mean(_z_trunk, dim=(1, 2))

        # 4. Project pooled s_inputs to self.c_s dimension
        # Output shape: [B, self.c_s]
        s_inputs_projected = self._fixed_project(s_inputs_pooled, self.c_s)

        # 5. s_trunk_pooled is already in self.c_s dimension
        # Output shape: [B, self.c_s]
        s_trunk_projected = s_trunk_pooled 

        # 6. Project pooled z_trunk to self.c_s dimension
        # Output shape: [B, self.c_s]
        z_trunk_projected = self._fixed_project(z_trunk_pooled, self.c_s)
        
        # 7. Combine the three projected representations by averaging
        # Output shape: [B, self.c_s]
        global_repr_batched = (s_inputs_projected + s_trunk_projected + z_trunk_projected) / 3.0
        
        # Prepare outputs, squeezing the batch dimension to match NexusEncoder's typical output for single instance
        # The original NexusEncoder returns .squeeze(0) for all its outputs.
        enhanced_s_inputs_out = _s_inputs.squeeze(0)
        enhanced_s_trunk_out = _s_trunk.squeeze(0)
        enhanced_z_trunk_out = _z_trunk.squeeze(0)
        global_repr_out = global_repr_batched.squeeze(0) # Shape: [c_s]
        
        return enhanced_s_inputs_out, enhanced_s_trunk_out, enhanced_z_trunk_out, global_repr_out