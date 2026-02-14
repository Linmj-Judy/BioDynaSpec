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

import logging
import math
from typing import Any, Optional, Union
import numpy as np
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from biotite.structure import AtomArray
from scipy.optimize import linear_sum_assignment
import torch.distributed as dist

from src.protenix.metrics.rmsd import weighted_rigid_align
from src.protenix.modules.frames import (
    expressCoordinatesInFrame,
    gather_frame_atom_by_indices,
)
from src.protenix.utils import expand_at_dim
from src.openfold_local.utils.checkpointing import get_checkpoint_fn
from src.utils.torch_utils import cdist


def loss_reduction(loss: torch.Tensor, method: str = "mean") -> torch.Tensor:
    """reduction wrapper

    Args:
        loss (torch.Tensor): loss
            [...]
        method (str, optional): reduction method. Defaults to "mean".

    Returns:
        torch.Tensor: reduced loss
            [] or [...]
    """

    if method is None:
        return loss
    assert method in ["mean", "sum", "add", "max", "min"]
    if method == "add":
        method = "sum"
    return getattr(torch, method)(loss)


class SmoothLDDTLoss(nn.Module):
    """
    Implements Algorithm 27 [SmoothLDDTLoss] in AF3
    """

    def __init__(
        self,
        eps: float = 1e-10,
        reduction: str = "mean",
    ) -> None:
        """SmoothLDDTLoss

        Args:
            eps (float, optional): avoid nan. Defaults to 1e-10.
            reduction (str, optional): reduction method for the batch dims. Defaults to mean.
        """
        super(SmoothLDDTLoss, self).__init__()
        self.eps = eps
        self.reduction = reduction

    def _chunk_forward(self, pred_distance, true_distance, c_lm=None):
        dist_diff = torch.abs(pred_distance - true_distance)
        # For save cuda memory we use inplace op
        dist_diff_epsilon = 0
        for threshold in [0.5, 1, 2, 4]:
            dist_diff_epsilon += 0.25 * torch.sigmoid(threshold - dist_diff)

        # Compute mean
        if c_lm is not None:
            lddt = torch.sum(c_lm * dist_diff_epsilon, dim=(-1, -2)) / (
                torch.sum(c_lm, dim=(-1, -2)) + self.eps
            )  # [..., N_sample]
        else:
            # It's for sparse forward mode
            lddt = torch.mean(dist_diff_epsilon, dim=-1)
        return lddt

    def forward(
        self,
        pred_distance: torch.Tensor,
        true_distance: torch.Tensor,
        distance_mask: torch.Tensor,
        lddt_mask: torch.Tensor,
        diffusion_chunk_size: Optional[int] = None,
    ) -> torch.Tensor:
        """SmoothLDDTLoss

        Args:
            pred_distance (torch.Tensor): the diffusion denoised atom-atom distance
                [..., N_sample, N_atom, N_atom]
            true_distance (torch.Tensor): the ground truth coordinates
                [..., N_atom, N_atom]
            distance_mask (torch.Tensor): whether true coordinates exist.
                [N_atom, N_atom]
            lddt_mask (torch.Tensor, optional): whether true distance is within radius (30A for nuc and 15A for others)
                [N_atom, N_atom]
            diffusion_chunk_size (Optional[int]): Chunk size over the N_sample dimension. Defaults to None.

        Returns:
            torch.Tensor: the smooth lddt loss
                [...] if reduction is None else []
        """
        c_lm = lddt_mask.bool().unsqueeze(dim=-3).detach()  # [..., 1, N_atom, N_atom]
        # Compute distance error
        # [...,  N_sample , N_atom, N_atom]
        if diffusion_chunk_size is None:
            lddt = self._chunk_forward(
                pred_distance=pred_distance, true_distance=true_distance, c_lm=c_lm
            )
        else:
            # Default use checkpoint for saving memory
            checkpoint_fn = get_checkpoint_fn()
            lddt = []
            N_sample = pred_distance.shape[-3]
            no_chunks = N_sample // diffusion_chunk_size + (
                N_sample % diffusion_chunk_size != 0
            )
            for i in range(no_chunks):
                lddt_i = checkpoint_fn(
                    self._chunk_forward,
                    pred_distance[
                        ...,
                        i * diffusion_chunk_size : (i + 1) * diffusion_chunk_size,
                        :,
                        :,
                    ],
                    true_distance,
                    c_lm,
                    use_reentrant=False,
                )
                lddt.append(lddt_i)
            lddt = torch.cat(lddt, dim=-1)

        lddt = lddt.mean(dim=-1)  # [...]
        return 1 - loss_reduction(lddt, method=self.reduction)

    def sparse_forward_subsample(
        self,
        pred_coordinate: torch.Tensor,
        true_coordinate: torch.Tensor,
        lddt_mask: torch.Tensor,
        diffusion_chunk_size: Optional[int] = None,
        sample_ratio: Optional[float] = None,
        max_samples: Optional[int] = None,
        generator: Optional[torch.Generator] = None,
    ) -> torch.Tensor:
        """
        在 N_sample 维上随机子采样，只对采样到的样本计算 sparse lddt。

        Args:
            pred_coordinate (torch.Tensor): [..., N_sample, N_atom, 3]
            true_coordinate (torch.Tensor): [..., N_atom, 3]
            lddt_mask (torch.Tensor): [N_atom, N_atom]
            diffusion_chunk_size (Optional[int]): N_sample 维分块大小
            sample_ratio (Optional[float]): 采样比例 (0,1]，与 max_samples 取最小
            max_samples (Optional[int]): 最多保留的样本数
            generator (Optional[torch.Generator]): 随机生成器，便于可复现

        Returns:
            torch.Tensor: lddt loss
        """
        n_sample = pred_coordinate.shape[-3]
        # 计算采样数
        target = n_sample
        if sample_ratio is not None:
            target = min(target, max(1, int(math.ceil(n_sample * sample_ratio))))
        if max_samples is not None:
            target = min(target, max_samples)
        if target <= 0:
            target = 1

        # 随机采样 N_sample 维
        device = pred_coordinate.device
        idx = torch.randperm(n_sample, device=device, generator=generator)[:target]
        pred_coordinate_sub = pred_coordinate.index_select(-3, idx)

        return self.sparse_forward(
            pred_coordinate=pred_coordinate_sub,
            true_coordinate=true_coordinate,
            lddt_mask=lddt_mask,
            diffusion_chunk_size=diffusion_chunk_size,
        )

    def sparse_forward(
        self,
        pred_coordinate: torch.Tensor,
        true_coordinate: torch.Tensor,
        lddt_mask: torch.Tensor,
        diffusion_chunk_size: Optional[int] = None,
    ) -> torch.Tensor:
        """SmoothLDDTLoss sparse implementation

        Args:
            pred_coordinate (torch.Tensor): the diffusion denoised atom coordinates
                [..., N_sample, N_atom, 3]
            true_coordinate (torch.Tensor): the ground truth atom coordinates
                [..., N_atom, 3]
            lddt_mask (torch.Tensor, optional): whether true distance is within radius (30A for nuc and 15A for others)
                [N_atom, N_atom]
            diffusion_chunk_size (Optional[int]): Chunk size over the N_sample dimension. Defaults to None.

        Returns:
            torch.Tensor: the smooth lddt loss
                [...] if reduction is None else []
        """
        lddt_indices = torch.nonzero(lddt_mask, as_tuple=True)
        true_coords_l = true_coordinate.index_select(-2, lddt_indices[0])
        true_coords_m = true_coordinate.index_select(-2, lddt_indices[1])
        true_distance_sparse_lm = torch.norm(true_coords_l - true_coords_m, p=2, dim=-1)
        if diffusion_chunk_size is None:
            pred_coords_l = pred_coordinate.index_select(-2, lddt_indices[0])
            pred_coords_m = pred_coordinate.index_select(-2, lddt_indices[1])
            # \delta x_{lm} and \delta x_{lm}^{GT} in the Algorithm 27
            pred_distance_sparse_lm = torch.norm(
                pred_coords_l - pred_coords_m, p=2, dim=-1
            )
            lddt = self._chunk_forward(
                pred_distance_sparse_lm, true_distance_sparse_lm, c_lm=None
            )
        else:
            checkpoint_fn = get_checkpoint_fn()
            lddt = []
            N_sample = pred_coordinate.shape[-3]
            no_chunks = N_sample // diffusion_chunk_size + (
                N_sample % diffusion_chunk_size != 0
            )
            for i in range(no_chunks):
                pred_coords_i_l = pred_coordinate[
                    i * diffusion_chunk_size : (i + 1) * diffusion_chunk_size, :, :
                ].index_select(-2, lddt_indices[0])
                pred_coords_i_m = pred_coordinate[
                    i * diffusion_chunk_size : (i + 1) * diffusion_chunk_size, :, :
                ].index_select(-2, lddt_indices[1])

                # \delta x_{lm} and \delta x_{lm}^{GT} in the Algorithm 27
                pred_distance_sparse_i_lm = torch.norm(
                    pred_coords_i_l - pred_coords_i_m, p=2, dim=-1
                )
                lddt_i = checkpoint_fn(
                    self._chunk_forward,
                    pred_distance_sparse_i_lm,
                    true_distance_sparse_lm,
                    checkpoint_kwargs=dict(use_reentrant=False),
                )
                lddt.append(lddt_i)
            lddt = torch.cat(lddt, dim=-1)

        lddt = lddt.mean(dim=-1)  # [...]
        return 1 - loss_reduction(lddt, method=self.reduction)

    def dense_forward(
        self,
        pred_coordinate: torch.Tensor,
        true_coordinate: torch.Tensor,
        lddt_mask: torch.Tensor,
        diffusion_chunk_size: Optional[int] = None,
    ) -> torch.Tensor:
        """SmoothLDDTLoss sparse implementation

        Args:
            pred_coordinate (torch.Tensor): the diffusion denoised atom coordinates
                [..., N_sample, N_atom, 3]
            true_coordinate (torch.Tensor): the ground truth atom coordinates
                [..., N_atom, 3]
            lddt_mask (torch.Tensor, optional): whether true distance is within radius (30A for nuc and 15A for others)
                [N_atom, N_atom]
            diffusion_chunk_size (Optional[int]): Chunk size over the N_sample dimension. Defaults to None.

        Returns:
            torch.Tensor: the smooth lddt loss
                [...] if reduction is None else []
        """
        c_lm = lddt_mask.bool().unsqueeze(dim=-3).detach()  # [..., 1, N_atom, N_atom]
        # Compute distance error
        # [...,  N_sample , N_atom, N_atom]
        true_distance = torch.cdist(true_coordinate, true_coordinate)
        if diffusion_chunk_size is None:
            pred_distance = torch.cdist(pred_coordinate, pred_coordinate)
            lddt = self._chunk_forward(
                pred_distance=pred_distance, true_distance=true_distance, c_lm=c_lm
            )
        else:
            checkpoint_fn = get_checkpoint_fn()
            lddt = []
            N_sample = pred_coordinate.shape[-3]
            no_chunks = N_sample // diffusion_chunk_size + (
                N_sample % diffusion_chunk_size != 0
            )
            for i in range(no_chunks):
                pred_distance_i = torch.cdist(
                    pred_coordinate[
                        i * diffusion_chunk_size : (i + 1) * diffusion_chunk_size,
                        :,
                        :,
                    ],
                    pred_coordinate[
                        i * diffusion_chunk_size : (i + 1) * diffusion_chunk_size,
                        :,
                        :,
                    ],
                )
                lddt_i = checkpoint_fn(
                    self._chunk_forward,
                    pred_distance_i,
                    true_distance,
                    c_lm,
                    use_reentrant=False,
                )
                lddt.append(lddt_i)
            lddt = torch.cat(lddt, dim=-1)

        lddt = lddt.mean(dim=-1)  # [...]
        return 1 - loss_reduction(lddt, method=self.reduction)


class BondLoss(nn.Module):
    """
    Implements Formula 5 [BondLoss] in AF3
    """

    def __init__(self, eps: float = 1e-6, reduction: str = "mean") -> None:
        """BondLoss

        Args:
            eps (float, optional): avoid nan. Defaults to 1e-6.
            reduction (str, optional): reduction method for the batch dims. Defaults to mean.
        """
        super(BondLoss, self).__init__()
        self.eps = eps
        self.reduction = reduction

    def _chunk_forward(self, pred_distance, true_distance, bond_mask):
        # Distance squared error
        # [...,  N_sample , N_atom, N_atom]
        dist_squared_err = (pred_distance - true_distance.unsqueeze(dim=-3)) ** 2
        bond_loss = torch.sum(dist_squared_err * bond_mask, dim=(-1, -2)) / torch.sum(
            bond_mask + self.eps, dim=(-1, -2)
        )  # [..., N_sample]
        return bond_loss

    def forward(
        self,
        pred_distance: torch.Tensor,
        true_distance: torch.Tensor,
        distance_mask: torch.Tensor,
        bond_mask: torch.Tensor,
        per_sample_scale: torch.Tensor = None,
        diffusion_chunk_size: Optional[int] = None,
    ) -> torch.Tensor:
        """BondLoss

        Args:
            pred_distance (torch.Tensor): the diffusion denoised atom-atom distance
                [..., N_sample, N_atom, N_atom]
            true_distance (torch.Tensor): the ground truth coordinates
                [..., N_atom, N_atom]
            distance_mask (torch.Tensor): whether true coordinates exist.
                [N_atom, N_atom] or [..., N_atom, N_atom]
            bond_mask (torch.Tensor): bonds considered in this loss
                [N_atom, N_atom] or [..., N_atom, N_atom]
            per_sample_scale (torch.Tensor, optional): whether to scale the loss by the per-sample noise-level.
                [..., N_sample]
            diffusion_chunk_size (Optional[int]): Chunk size over the N_sample dimension. Defaults to None.

        Returns:
            torch.Tensor: the bond loss
                [...] if reduction is None else []
        """

        bond_mask = (bond_mask * distance_mask).unsqueeze(
            dim=-3
        )  # [1, N_atom, N_atom] or [..., 1, N_atom, N_atom]
        # Bond Loss
        if diffusion_chunk_size is None:
            bond_loss = self._chunk_forward(
                pred_distance=pred_distance,
                true_distance=true_distance,
                bond_mask=bond_mask,
            )
        else:
            checkpoint_fn = get_checkpoint_fn()
            bond_loss = []
            N_sample = pred_distance.shape[-3]
            no_chunks = N_sample // diffusion_chunk_size + (
                N_sample % diffusion_chunk_size != 0
            )
            for i in range(no_chunks):
                bond_loss_i = checkpoint_fn(
                    self._chunk_forward,
                    pred_distance[
                        ...,
                        i * diffusion_chunk_size : (i + 1) * diffusion_chunk_size,
                        :,
                        :,
                    ],
                    true_distance,
                    bond_mask,
                    use_reentrant=False,
                )
                bond_loss.append(bond_loss_i)
            bond_loss = torch.cat(bond_loss, dim=-1)
        if per_sample_scale is not None:
            bond_loss = bond_loss * per_sample_scale

        bond_loss = bond_loss.mean(dim=-1)  # [...]
        return loss_reduction(bond_loss, method=self.reduction)

    def sparse_forward(
        self,
        pred_coordinate: torch.Tensor,
        true_coordinate: torch.Tensor,
        distance_mask: torch.Tensor,
        bond_mask: torch.Tensor,
        per_sample_scale: torch.Tensor = None,
    ) -> torch.Tensor:
        """BondLoss sparse implementation

        Args:
            pred_coordinate (torch.Tensor): the diffusion denoised atom coordinates
                [..., N_sample, N_atom, 3]
            true_coordinate (torch.Tensor): the ground truth atom coordinates
                [..., N_atom, 3]
            distance_mask (torch.Tensor): whether true coordinates exist.
                [N_atom, N_atom] or [..., N_atom, N_atom]
            bond_mask (torch.Tensor): bonds considered in this loss
                [N_atom, N_atom] or [..., N_atom, N_atom]
            per_sample_scale (torch.Tensor, optional): whether to scale the loss by the per-sample noise-level.
                [..., N_sample]
        Returns:
            torch.Tensor: the bond loss
                [...] if reduction is None else []
        """

        bond_mask = bond_mask * distance_mask
        bond_indices = torch.nonzero(bond_mask, as_tuple=True)
        pred_coords_i = pred_coordinate.index_select(-2, bond_indices[0])
        pred_coords_j = pred_coordinate.index_select(-2, bond_indices[1])
        true_coords_i = true_coordinate.index_select(-2, bond_indices[0])
        true_coords_j = true_coordinate.index_select(-2, bond_indices[1])

        pred_distance_sparse = torch.norm(pred_coords_i - pred_coords_j, p=2, dim=-1)
        true_distance_sparse = torch.norm(true_coords_i - true_coords_j, p=2, dim=-1)
        dist_squared_err_sparse = (pred_distance_sparse - true_distance_sparse) ** 2
        # Protecting special data that has size: tensor([], size=(x, 0), grad_fn=<PowBackward0>)
        if dist_squared_err_sparse.numel() == 0:
            return torch.tensor(
                0.0, device=dist_squared_err_sparse.device, requires_grad=True
            )
        bond_loss = torch.mean(dist_squared_err_sparse, dim=-1)  # [..., N_sample]
        if per_sample_scale is not None:
            bond_loss = bond_loss * per_sample_scale

        bond_loss = bond_loss.mean(dim=-1)  # [...]
        return bond_loss


def compute_lddt_mask(
    true_distance: torch.Tensor,
    distance_mask: torch.Tensor,
    is_nucleotide: torch.Tensor,
    is_nucleotide_threshold: float = 30.0,
    is_not_nucleotide_threshold: float = 15.0,
) -> torch.Tensor:
    """calculate the atom pair mask with the bespoke radius

    Args:
        true_distance (torch.Tensor): the ground truth coordinates
            [..., N_atom, N_atom]
        distance_mask (torch.Tensor): whether true coordinates exist.
            [..., N_atom, N_atom] or [N_atom, N_atom]
        is_nucleotide (torch.Tensor): Indicator for nucleotide atoms.
            [..., N_atom] or [N_atom]
        is_nucleotide_threshold (float): Threshold distance for nucleotide atoms. Defaults to 30.0.
        is_not_nucleotide_threshold (float): Threshold distance for non-nucleotide atoms. Defaults to 15.0.

    Returns:
        c_lm (torch.Tenson): the atom pair mask c_lm, not symmetric
            [..., N_atom, N_atom]
    """
    # Restrict to bespoke inclusion radius
    is_nucleotide_mask = is_nucleotide.bool()
    c_lm = (true_distance < is_nucleotide_threshold) * is_nucleotide_mask[..., None] + (
        true_distance < is_not_nucleotide_threshold
    ) * (
        ~is_nucleotide_mask[..., None]
    )  # [..., N_atom, N_atom]

    # Zero-out diagonals of c_lm and cast to float
    c_lm = c_lm * (
        1 - torch.eye(n=c_lm.size(-1), device=c_lm.device, dtype=true_distance.dtype)
    )
    # Zero-out atom pairs without true coordinates
    # Note: the sparsity of c_lm is ~10% in 5000 atom-pairs,
    # and becomes more sparse as the number of atoms increases,
    # change to sparse implementation can reduce cuda memory
    c_lm = c_lm * distance_mask  # [..., N_atom, N_atom]
    return c_lm


def softmax_cross_entropy(logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
    """Softmax cross entropy

    Args:
        logits (torch.Tensor): classification logits
            [..., num_class]
        labels (torch.Tensor): classification labels (value = probability)
            [..., num_class]

    Returns:
        torch.Tensor: softmax cross entropy
            [...]
    """
    loss = -1 * torch.sum(
        labels * F.log_softmax(logits, dim=-1),
        dim=-1,
    )
    return loss


class DistogramLoss(nn.Module):
    """
    Implements DistogramLoss in AF3
    """

    def __init__(
        self,
        min_bin: float = 2.3125,
        max_bin: float = 21.6875,
        no_bins: int = 64,
        eps: float = 1e-6,
        reduction: str = "mean",
    ) -> None:
        """Distogram loss
        This head and loss are identical to AlphaFold 2, where the pairwise token distances use the representative atom for each token:
            Cβ for protein residues (Cα for glycine),
            C4 for purines and C2 for pyrimidines.
            All ligands already have a single atom per token.

        Args:
            min_bin (float, optional): min boundary of bins. Defaults to 2.3125.
            max_bin (float, optional): max boundary of bins. Defaults to 21.6875.
            no_bins (int, optional): number of bins. Defaults to 64.
            eps (float, optional): small number added to denominator. Defaults to 1e-6.
            reduce (bool, optional): reduce dim. Defaults to True.
        """
        super(DistogramLoss, self).__init__()
        self.min_bin = min_bin
        self.max_bin = max_bin
        self.no_bins = no_bins
        self.eps = eps
        self.reduction = reduction

    def calculate_label(
        self,
        true_coordinate: torch.Tensor,
        coordinate_mask: torch.Tensor,
        rep_atom_mask: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """calculate the label as bins

        Args:
            true_coordinate (torch.Tensor): true coordinates.
                [..., N_atom, 3]
            coordinate_mask (torch.Tensor): whether true coordinates exist.
                [N_atom] or [..., N_atom]
            rep_atom_mask (torch.Tensor): representative atom mask
                [N_atom]

        Returns:
            true_bins (torch.Tensor): distance error assigned into bins (one-hot).
                [..., N_token, N_token, no_bins]
            pair_coordinate_mask (torch.Tensor): whether the coordinates of representative atom pairs exist.
                [N_token, N_token] or [..., N_token, N_token]
        """

        boundaries = torch.linspace(
            start=self.min_bin,
            end=self.max_bin,
            steps=self.no_bins - 1,
            device=true_coordinate.device,
        )

        # Compute label: the true bins
        # True distance
        rep_atom_mask = rep_atom_mask.bool()
        true_coordinate = true_coordinate[..., rep_atom_mask, :]  # [..., N_token, 3]
        gt_dist = cdist(true_coordinate, true_coordinate)  # [..., N_token, N_token]
        # Assign distance to bins
        true_bins = torch.sum(
            gt_dist.unsqueeze(dim=-1) > boundaries, dim=-1
        )  # range in [0, no_bins-1], shape = [..., N_token, N_token]

        # Mask
        token_mask = coordinate_mask[..., rep_atom_mask]
        pair_mask = token_mask[..., None] * token_mask[..., None, :]

        return F.one_hot(true_bins, self.no_bins), pair_mask

    def forward(
        self,
        logits: torch.Tensor,
        true_coordinate: torch.Tensor,
        coordinate_mask: torch.Tensor,
        rep_atom_mask: torch.Tensor,
    ) -> torch.Tensor:
        """Distogram loss

        Args:
            logits (torch.Tensor): logits.
                [..., N_token, N_token, no_bins]
            true_coordinate (torch.Tensor): true coordinates.
                [..., N_atom, 3]
            coordinate_mask (torch.Tensor): whether true coordinates exist.
                [N_atom] or [..., N_atom]
            rep_atom_mask (torch.Tensor): representative atom mask.
                [N_atom]

        Returns:
            torch.Tensor: the return loss.
                [...] if self.reduction is not None else []
        """

        with torch.no_grad():
            true_bins, pair_mask = self.calculate_label(
                true_coordinate=true_coordinate,
                coordinate_mask=coordinate_mask,
                rep_atom_mask=rep_atom_mask,
            )

        errors = softmax_cross_entropy(
            logits=logits,
            labels=true_bins,
        )  # [..., N_token, N_token]

        denom = self.eps + torch.sum(pair_mask, dim=(-1, -2))
        loss = torch.sum(errors * pair_mask, dim=(-1, -2))
        loss = loss / denom

        return loss_reduction(loss, method=self.reduction)


class PDELoss(nn.Module):
    """
    Implements Predicted distance loss in AF3
    """

    def __init__(
        self,
        min_bin: float = 0,
        max_bin: float = 32,
        no_bins: int = 64,
        eps: float = 1e-6,
        reduction: str = "mean",
    ) -> None:
        """PDELoss
        This loss are between representative token atoms i and j in the mini-rollout prediction

        Args:
            min_bin (float, optional): min boundary of bins. Defaults to 0.
            max_bin (float, optional): max boundary of bins. Defaults to 32.
            no_bins (int, optional): number of bins. Defaults to 64.
            eps (float, optional): small number added to denominator. Defaults to 1e-6.
            reduction (str, optional): reduction method for the batch dims. Defaults to mean.
        """
        super(PDELoss, self).__init__()
        self.min_bin = min_bin
        self.max_bin = max_bin
        self.no_bins = no_bins
        self.eps = eps
        self.reduction = reduction

    def calculate_label(
        self,
        pred_coordinate: torch.Tensor,
        true_coordinate: torch.Tensor,
        coordinate_mask: torch.Tensor,
        rep_atom_mask: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """calculate the label as bins

        Args:
            pred_coordinate (torch.Tensor): predicted coordinates.
                [..., N_sample, N_atom, 3]
            true_coordinate (torch.Tensor): true coordinates.
                [..., N_atom, 3]
            coordinate_mask (torch.Tensor): whether true coordinates exist.
                [N_atom] or [..., N_atom]
            rep_atom_mask (torch.Tensor):
                [N_atom]

        Returns:
            true_bins (torch.Tensor): distance error assigned into bins (one-hot).
                [..., N_sample, N_token, N_token, no_bins]
            pair_coordinate_mask (torch.Tensor): whether the coordinates of representative atom pairs exist.
                [N_token, N_token] or [..., N_token, N_token]
        """

        boundaries = torch.linspace(
            start=self.min_bin,
            end=self.max_bin,
            steps=self.no_bins + 1,
            device=pred_coordinate.device,
        )

        # Compute label: the true bins
        # True distance
        rep_atom_mask = rep_atom_mask.bool()
        true_coordinate = true_coordinate[..., rep_atom_mask, :]  # [..., N_token, 3]
        gt_dist = cdist(true_coordinate, true_coordinate)  # [..., N_token, N_token]
        # Predicted distance
        pred_coordinate = pred_coordinate[..., rep_atom_mask, :]
        pred_dist = cdist(
            pred_coordinate, pred_coordinate
        )  # [..., N_sample, N_token, N_token]
        # Distance error
        dist_error = torch.abs(pred_dist - gt_dist.unsqueeze(dim=-3))

        # Assign distance error to bins
        true_bins = torch.sum(
            dist_error.unsqueeze(dim=-1) > boundaries, dim=-1
        )  # range in [1, no_bins + 1], shape = [..., N_sample, N_token, N_token]
        true_bins = torch.clamp(
            true_bins, min=1, max=self.no_bins
        )  # just in case bin=0 occurs

        # Mask
        token_mask = coordinate_mask[..., rep_atom_mask]
        pair_mask = token_mask[..., None] * token_mask[..., None, :]

        return F.one_hot(true_bins - 1, self.no_bins).detach(), pair_mask.detach()

    def forward(
        self,
        logits: torch.Tensor,
        pred_coordinate: torch.Tensor,
        true_coordinate: torch.Tensor,
        coordinate_mask: torch.Tensor,
        rep_atom_mask: torch.Tensor,
    ) -> torch.Tensor:
        """PDELoss

        Args:
            logits (torch.Tensor): logits
                [..., N_sample, N_token, N_token, no_bins]
            pred_coordinate: (torch.Tensor): predict coordinates
                [..., N_sample, N_atom, 3]
            true_coordinate (torch.Tensor): true coordinates
                [..., N_atom, 3]
            coordinate_mask (torch.Tensor): whether true coordinates exist
                [N_atom] or [..., N_atom]
            rep_atom_mask (torch.Tensor): representative atom mask for this loss
                [N_atom]

        Returns:
            torch.Tensor: the return loss
                [...] if reduction is None else []
        """

        with torch.no_grad():
            true_bins, pair_mask = self.calculate_label(
                pred_coordinate=pred_coordinate,
                true_coordinate=true_coordinate,
                coordinate_mask=coordinate_mask,
                rep_atom_mask=rep_atom_mask,
            )

        errors = softmax_cross_entropy(
            logits=logits,
            labels=true_bins,
        )  # [..., N_sample, N_token, N_token]

        denom = self.eps + torch.sum(pair_mask, dim=(-1, -2))  # [...]
        loss = errors * pair_mask.unsqueeze(dim=-3)  # [..., N_sample, N_token, N_token]
        loss = torch.sum(loss, dim=(-1, -2))  # [..., N_sample]
        loss = loss / denom.unsqueeze(dim=-1)  # [..., N_sample]
        loss = loss.mean(dim=-1)  # [...]

        return loss_reduction(loss, method=self.reduction)


# Algorithm 30 Compute alignment error
def compute_alignment_error_squared(
    pred_coordinate: torch.Tensor,
    true_coordinate: torch.Tensor,
    pred_frames: torch.Tensor,
    true_frames: torch.Tensor,
) -> torch.Tensor:
    """Implements Algorithm 30 Compute alignment error, but do not take the square root

    Args:
        pred_coordinate (torch.Tensor): the predict coords [frame center]
            [..., N_sample, N_token, 3]
        true_coordinate (torch.Tensor): the ground truth coords [frame center]
            [..., N_token, 3]
        pred_frames (torch.Tensor): the predict frame
            [..., N_sample, N_frame, 3, 3]
        true_frames (torch.Tensor): the ground truth frame
            [..., N_frame, 3, 3]

    Returns:
        torch.Tensor: the computed alignment error
            [..., N_sample, N_frame, N_token]
    """
    x_transformed_pred = expressCoordinatesInFrame(
        coordinate=pred_coordinate, frames=pred_frames
    )  # [..., N_sample, N_frame, N_token, 3]
    x_transformed_true = expressCoordinatesInFrame(
        coordinate=true_coordinate, frames=true_frames
    )  # [..., N_frame, N_token, 3]
    squared_pae = torch.sum(
        (x_transformed_pred - x_transformed_true.unsqueeze(dim=-4)) ** 2, dim=-1
    )  # [..., N_sample, N_frame, N_token]
    return squared_pae


class PAELoss(nn.Module):
    """
    Implements Predicted Aligned distance loss in AF3
    """

    def __init__(
        self,
        min_bin: float = 0,
        max_bin: float = 32,
        no_bins: int = 64,
        eps: float = 1e-6,
        reduction: str = "mean",
    ) -> None:
        """PAELoss
        This loss are between representative token atoms i and j in the mini-rollout prediction

        Args:
            min_bin (float, optional): min boundary of bins. Defaults to 0.
            max_bin (float, optional): max boundary of bins. Defaults to 32.
            no_bins (int, optional): number of bins. Defaults to 64.
            eps (float, optional): small number added to denominator. Defaults to 1e-6.
            reduce (bool, optional): reduce dim. Defaults to True.
        """
        super(PAELoss, self).__init__()
        self.min_bin = min_bin
        self.max_bin = max_bin
        self.no_bins = no_bins
        self.eps = eps
        self.reduction = reduction

    def calculate_label(
        self,
        pred_coordinate: torch.Tensor,
        true_coordinate: torch.Tensor,
        coordinate_mask: torch.Tensor,
        rep_atom_mask: torch.Tensor,
        frame_atom_index: torch.Tensor,
        has_frame: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """calculate true PAE (squared) and true bins

        Args:
            pred_coordinate: (torch.Tensor): predict coordinates.
                [..., N_sample, N_atom, 3]
            true_coordinate (torch.Tensor): true coordinates.
                [..., N_atom, 3]
            coordinate_mask (torch.Tensor): whether true coordinates exist
                [N_atom]
            rep_atom_mask (torch.Tensor): masks of the representative atom for each token.
                [N_atom]
            frame_atom_index (torch.Tensor): indices of frame atoms (three atoms per token(=per frame)).
                [N_token, 3[three atom]]
            has_frame (torch.Tensor): indicates whether token_i has a valid frame.
                [N_token]
        Returns:
            squared_pae (torch.Tensor): pairwise alignment error squared
                [..., N_sample, N_frame, N_token] where N_token = rep_atom_mask.sum()
            true_bins (torch.Tensor): the true bins
                [..., N_sample, N_frame, N_token, no_bins]
            frame_token_pair_mask (torch.Tensor): whether frame_i token_j both have true coordinates.
                [N_frame, N_token]
        """

        coordinate_mask = coordinate_mask.bool()
        rep_atom_mask = rep_atom_mask.bool()
        has_frame = has_frame.bool()

        # NOTE: to support frame_atom_index with batch_dims, need to expand its dims before constructing frames.
        assert len(frame_atom_index.shape) == 2

        # Take valid frames: N_token -> N_frame
        frame_atom_index = frame_atom_index[has_frame, :]  # [N_frame, 3[three atom]]

        # Get predicted frames and true frames
        pred_frames = gather_frame_atom_by_indices(
            coordinate=pred_coordinate, frame_atom_index=frame_atom_index, dim=-2
        )  # [..., N_sample, N_frame, 3[three atom], 3[coordinates]]
        true_frames = gather_frame_atom_by_indices(
            coordinate=true_coordinate, frame_atom_index=frame_atom_index, dim=-2
        )  # [..., N_frame, 3[three atom], 3[coordinates]]

        # Get pair_mask for computing the loss
        true_frame_coord_mask = gather_frame_atom_by_indices(
            coordinate=coordinate_mask, frame_atom_index=frame_atom_index, dim=-1
        )  # [N_frame, 3[three atom]]
        true_frame_coord_mask = (
            true_frame_coord_mask.sum(dim=-1) >= 3
        )  # [N_frame] whether all atoms in the frame has coordinates
        token_mask = coordinate_mask[rep_atom_mask]  # [N_token]
        frame_token_pair_mask = (
            true_frame_coord_mask[..., None] * token_mask[..., None, :]
        )  # [N_frame, N_token]

        squared_pae = (
            compute_alignment_error_squared(
                pred_coordinate=pred_coordinate[..., rep_atom_mask, :],
                true_coordinate=true_coordinate[..., rep_atom_mask, :],
                pred_frames=pred_frames,
                true_frames=true_frames,
            )
            * frame_token_pair_mask
        )  # [..., N_sample, N_frame, N_token]

        # Compute true bins
        boundaries = torch.linspace(
            start=self.min_bin,
            end=self.max_bin,
            steps=self.no_bins + 1,
            device=pred_coordinate.device,
        )
        boundaries = boundaries**2

        true_bins = torch.sum(
            squared_pae.unsqueeze(dim=-1) > boundaries, dim=-1
        )  # range [1, no_bins + 1]
        true_bins = torch.where(
            frame_token_pair_mask,
            true_bins,
            torch.ones_like(true_bins) * self.no_bins,
        )
        true_bins = torch.clamp(
            true_bins, min=1, max=self.no_bins
        )  # just in case bin=0 occurs

        return (
            squared_pae.detach(),
            F.one_hot(true_bins - 1, self.no_bins).detach(),
            frame_token_pair_mask.detach(),
        )

    def forward(
        self,
        logits: torch.Tensor,
        pred_coordinate: torch.Tensor,
        true_coordinate: torch.Tensor,
        coordinate_mask: torch.Tensor,
        frame_atom_index: torch.Tensor,
        rep_atom_mask: torch.Tensor,
        has_frame: torch.Tensor,
    ) -> torch.Tensor:
        """PAELoss

        Args:
            logits (torch.Tensor): logits
                [..., N_sample, N_token, N_token, no_bins]
            pred_coordinate: (torch.Tensor): predict coordinates
                [..., N_sample, N_atom, 3]
            true_coordinate (torch.Tensor): true coordinates
                [..., N_atom, 3]
            coordinate_mask (torch.Tensor): whether true coordinates exist
                [N_atom]
            rep_atom_mask (torch.Tensor): masks of the representative atom for each token.
                [N_atom]
            frame_atom_index (torch.Tensor): indices of frame atoms (three atoms per token(=per frame)).
                [N_token, 3[three atom]]
            has_frame (torch.Tensor): indicates whether token_i has a valid frame.
                [N_token]
        Returns:
            torch.Tensor: the return loss
                [] if reduce
                [..., n] else
        """

        has_frame = has_frame.bool()
        rep_atom_mask = rep_atom_mask.bool()
        assert len(has_frame.shape) == 1
        assert len(frame_atom_index.shape) == 2

        with torch.no_grad():
            # true_bins: [..., N_sample, N_frame, N_token, no_bins]
            # pair_mask: [N_frame, N_token]
            _, true_bins, pair_mask = self.calculate_label(
                pred_coordinate=pred_coordinate,
                true_coordinate=true_coordinate,
                frame_atom_index=frame_atom_index,
                rep_atom_mask=rep_atom_mask,
                coordinate_mask=coordinate_mask,
                has_frame=has_frame,
            )

        loss = softmax_cross_entropy(
            logits=logits[
                ..., has_frame, :, :
            ],  # [..., N_sample, N_frame, N_token, no_bins]
            labels=true_bins,
        )  # [..., N_sample, N_frame, N_token]

        denom = self.eps + torch.sum(pair_mask, dim=(-1, -2))  # []
        loss = loss * pair_mask.unsqueeze(dim=-3)  # [..., N_sample, N_token, N_token]
        loss = torch.sum(loss, dim=(-1, -2))  # [..., N_sample]
        loss = loss / denom.unsqueeze(dim=-1)  # [..., N_sample]
        loss = loss.mean(dim=-1)  # [...]

        return loss_reduction(loss, self.reduction)


class ExperimentallyResolvedLoss(nn.Module):
    def __init__(
        self,
        eps: float = 1e-6,
        reduction: str = "mean",
    ) -> None:
        """
        Args:
            eps (float, optional): avoid nan. Defaults to 1e-6.
        """
        super(ExperimentallyResolvedLoss, self).__init__()
        self.eps = eps
        self.reduction = reduction

    def forward(
        self,
        logits: torch.Tensor,
        coordinate_mask: torch.Tensor,
        atom_mask: torch.Tensor = None,
    ) -> torch.Tensor:
        """
        Args:
            logits (torch.Tensor): logits
                [..., N_sample, N_atom, no_bins:=2]
            coordinate_mask (torch.Tensor): whether true coordinates exist
                [..., N_atom] | [N_atom]
            atom_mask (torch.Tensor, optional): whether to conside the atom in the loss
                [..., N_atom]
        Returns:
            torch.Tensor: the experimentally resolved loss
        """
        is_resolved = F.one_hot(
            coordinate_mask.long(), 2
        )  # [..., N_atom, 2] or [N_atom, 2]
        errors = softmax_cross_entropy(
            logits=logits, labels=is_resolved.unsqueeze(dim=-3)
        )  # [..., N_sample, N_atom]
        if atom_mask is None:
            loss = errors.mean(dim=-1)  # [..., N_sample]
        else:
            loss = torch.sum(
                errors * atom_mask[..., None, :], dim=-1
            )  # [..., N_sample]
            loss = loss / (
                self.eps + torch.sum(atom_mask[..., None, :], dim=-1)
            )  # [..., N_sample]

        loss = loss.mean(dim=-1)  # [...]
        return loss_reduction(loss, method=self.reduction)


class MSELoss(nn.Module):
    """
    Implements Formula 2-4 [MSELoss] in AF3
    """

    def __init__(
        self,
        weight_mse: float = 1 / 3,
        weight_dna: float = 5.0,
        weight_rna=5.0,
        weight_ligand=10.0,
        eps=1e-6,
        reduction: str = "mean",
    ) -> None:
        super(MSELoss, self).__init__()
        self.weight_mse = weight_mse
        self.weight_dna = weight_dna
        self.weight_rna = weight_rna
        self.weight_ligand = weight_ligand
        self.eps = eps
        self.reduction = reduction

    def weighted_rigid_align(
        self,
        pred_coordinate: torch.Tensor,
        true_coordinate: torch.Tensor,
        coordinate_mask: torch.Tensor,
        is_dna: torch.Tensor,
        is_rna: torch.Tensor,
        is_ligand: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """compute weighted rigid alignment results

        Args:
            pred_coordinate (torch.Tensor): the denoised coordinates from diffusion module
                [..., N_sample, N_atom, 3]
            true_coordinate (torch.Tensor): the ground truth coordinates
                [..., N_atom, 3]
            coordinate_mask (torch.Tensor): whether true coordinates exist
                [N_atom] or [..., N_atom]
            is_dna / is_rna / is_ligand (torch.Tensor): mol type mask
                [N_atom] or [..., N_atom]

        Returns:
            true_coordinate_aligned (torch.Tensor): aligned coordinates for each sample
                [..., N_sample, N_atom, 3]
            weight (torch.Tensor): weights for each atom
                [N_atom] or [..., N_sample, N_atom]
        """
        N_sample = pred_coordinate.size(-3)
        weight = (
            1
            + self.weight_dna * is_dna
            + self.weight_rna * is_rna
            + self.weight_ligand * is_ligand
        )  # [N_atom] or [..., N_atom]

        # Apply coordinate_mask
        weight = weight * coordinate_mask  # [N_atom] or [..., N_atom]
        true_coordinate = true_coordinate * coordinate_mask.unsqueeze(dim=-1)
        pred_coordinate = pred_coordinate * coordinate_mask[..., None, :, None]

        # Reshape to add "N_sample" dimension when missing
        # pred_coordinate: [..., N_sample, N_atom, 3]
        # true_coordinate: [..., N_atom, 3] or [..., N_sample, N_atom, 3]
        if true_coordinate.dim() == pred_coordinate.dim() - 1:
            true_coordinate = expand_at_dim(
                true_coordinate, dim=-3, n=N_sample
            )  # [..., N_sample, N_atom, 3]
        # weight: [..., N_atom] or [..., N_sample, N_atom]
        if weight.dim() == pred_coordinate.dim() - 2:
            weight = expand_at_dim(
                weight, dim=-2, n=N_sample
            )  # [..., N_sample, N_atom]

        # 对齐 atom_weight 的前缀维度到 pred_coordinate（支持 batch_shape）
        target_prefix = pred_coordinate.shape[:-2]
        n_atom = pred_coordinate.shape[-2]
        if weight.shape[-1] != n_atom:
            raise ValueError(
                f"atom_weight last dim {weight.shape[-1]} does not match N_atom {n_atom}"
            )

        def _is_broadcastable(shape, target):
            return all(s == 1 or s == t for s, t in zip(shape, target))

        weight_prefix = weight.shape[:-1]
        if weight_prefix != target_prefix:
            if len(weight_prefix) == len(target_prefix):
                if not _is_broadcastable(weight_prefix, target_prefix):
                    # 若目标维为 1，允许在该维度上取第一个样本以对齐
                    slices = []
                    for w_dim, t_dim in zip(weight_prefix, target_prefix):
                        if w_dim == t_dim:
                            slices.append(slice(None))
                        elif t_dim == 1:
                            slices.append(slice(0, 1))
                        else:
                            slices = None
                            break
                    if slices is not None:
                        weight = weight[tuple(slices)]
                        weight_prefix = weight.shape[:-1]

            if len(weight_prefix) > len(target_prefix):
                # 允许去掉多余的前导 1 维
                while len(weight_prefix) > len(target_prefix) and weight.shape[0] == 1:
                    weight = weight.squeeze(0)
                    weight_prefix = weight.shape[:-1]
                if len(weight_prefix) > len(target_prefix):
                    raise ValueError(
                        f"atom_weight prefix {weight_prefix} cannot align to target prefix {target_prefix}"
                    )

            if len(weight_prefix) < len(target_prefix):
                # 同时尝试前缀对齐 / 后缀对齐（允许 1 广播）
                prefix_target = target_prefix[: len(weight_prefix)]
                suffix_target = target_prefix[-len(weight_prefix) :]
                prefix_ok = _is_broadcastable(weight_prefix, prefix_target)
                suffix_ok = _is_broadcastable(weight_prefix, suffix_target)

                if prefix_ok and not suffix_ok:
                    # 在 atom 维前补 1
                    for _ in range(len(target_prefix) - len(weight_prefix)):
                        weight = weight.unsqueeze(-2)
                elif suffix_ok and not prefix_ok:
                    # 在最前补 1
                    for _ in range(len(target_prefix) - len(weight_prefix)):
                        weight = weight.unsqueeze(0)
                elif prefix_ok and suffix_ok:
                    # 都可行时，优先匹配更“具体”的（非1维更多）
                    prefix_score = sum(
                        1 for s, t in zip(weight_prefix, prefix_target) if s == t
                    )
                    suffix_score = sum(
                        1 for s, t in zip(weight_prefix, suffix_target) if s == t
                    )
                    if suffix_score > prefix_score:
                        for _ in range(len(target_prefix) - len(weight_prefix)):
                            weight = weight.unsqueeze(0)
                    else:
                        for _ in range(len(target_prefix) - len(weight_prefix)):
                            weight = weight.unsqueeze(-2)
                else:
                    raise ValueError(
                        f"atom_weight prefix {weight_prefix} cannot broadcast to target prefix {target_prefix}"
                    )

            # 广播到目标前缀
            weight = weight.expand(*target_prefix, n_atom)

        # Align GT coords to predicted coords
        d = pred_coordinate.dtype
        # Some ops in weighted_rigid_align do not support BFloat16 training
        with torch.amp.autocast("cuda",enabled=False):
            true_coordinate_aligned = weighted_rigid_align(
                x=true_coordinate.to(torch.float32),  # [..., N_sample, N_atom, 3]
                x_target=pred_coordinate.to(
                    torch.float32
                ),  # [..., N_sample, N_atom, 3]
                atom_weight=weight.to(
                    torch.float32
                ),  # [N_atom] or [..., N_sample, N_atom]
                stop_gradient=True,
            )  # [..., N_sample, N_atom, 3]
            true_coordinate_aligned = true_coordinate_aligned.to(d)

        return (true_coordinate_aligned.detach(), weight.detach())

    def forward(
        self,
        pred_coordinate: torch.Tensor,
        true_coordinate: torch.Tensor,
        coordinate_mask: torch.Tensor,
        is_dna: torch.Tensor,
        is_rna: torch.Tensor,
        is_ligand: torch.Tensor,
        per_sample_scale: torch.Tensor = None,
    ) -> torch.Tensor:
        """MSELoss

        Args:
            pred_coordinate (torch.Tensor): the denoised coordinates from diffusion module.
                [..., N_sample, N_atom, 3]
            true_coordinate (torch.Tensor): the ground truth coordinates.
                [..., N_atom, 3]
            coordinate_mask (torch.Tensor): whether true coordinates exist.
                [N_atom] or [..., N_atom]
            is_dna / is_rna / is_ligand (torch.Tensor): mol type mask.
                [N_atom] or [..., N_atom]
            per_sample_scale (torch.Tensor, optional): whether to scale the loss by the per-sample noise-level.
                [..., N_sample]

        Returns:
            torch.Tensor: the weighted mse loss.
                [...] is self.reduction is None else []
        """
        # True_coordinate_aligned: [..., N_sample, N_atom, 3]
        # Weight: [N_atom] or [..., N_sample, N_atom]
        with torch.no_grad():
            true_coordinate_aligned, weight = self.weighted_rigid_align(
                pred_coordinate=pred_coordinate,
                true_coordinate=true_coordinate,
                coordinate_mask=coordinate_mask,
                is_dna=is_dna,
                is_rna=is_rna,
                is_ligand=is_ligand,
            )

        # Calculate MSE loss
        per_atom_se = ((pred_coordinate - true_coordinate_aligned) ** 2).sum(
            dim=-1
        )  # [..., N_sample, N_atom]
        per_sample_weighted_mse = (weight * per_atom_se).sum(dim=-1) / (
            coordinate_mask.sum(dim=-1, keepdim=True) + self.eps
        )  # [..., N_sample]

        if per_sample_scale is not None:
            per_sample_weighted_mse = per_sample_weighted_mse * per_sample_scale

        weighted_align_mse_loss = self.weight_mse * (per_sample_weighted_mse).mean(
            dim=-1
        )  # [...]

        loss = loss_reduction(weighted_align_mse_loss, method=self.reduction)

        return loss


class PLDDTLoss(nn.Module):
    """
    Implements PLDDT Loss in AF3, different from the paper description.
    Main changes:
    1. use difference of distance instead of predicted distance when calculating plddt
    2. normalize each plddt score within 0-1
    """

    def __init__(
        self,
        min_bin: float = 0,
        max_bin: float = 1,
        no_bins: int = 50,
        is_nucleotide_threshold: float = 30.0,
        is_not_nucleotide_threshold: float = 15.0,
        eps: float = 1e-6,
        normalize: bool = True,
        reduction: str = "mean",
    ) -> None:
        """PLDDT loss
        This loss are between atoms l and m (has some filters) in the mini-rollout prediction

        Args:
            min_bin (float, optional): min boundary of bins. Defaults to 0.
            max_bin (float, optional): max boundary of bins. Defaults to 1.
            no_bins (int, optional): number of bins. Defaults to 50.
            is_nucleotide_threshold (float, optional): threshold for nucleotide atoms. Defaults 30.0.
            is_not_nucleotide_threshold (float, optional): threshold for non-nucleotide atoms. Defaults 15.0
            eps (float, optional): small number added to denominator. Defaults to 1e-6.
            reduction (str, optional): reduction method for the batch dims. Defaults to mean.
        """
        super(PLDDTLoss, self).__init__()
        self.normalize = normalize
        self.min_bin = min_bin
        self.max_bin = max_bin
        self.no_bins = no_bins
        self.eps = eps
        self.reduction = reduction
        self.is_nucleotide_threshold = is_nucleotide_threshold
        self.is_not_nucleotide_threshold = is_not_nucleotide_threshold

    def calculate_label(
        self,
        pred_coordinate: torch.Tensor,
        true_coordinate: torch.Tensor,
        is_nucleotide: torch.Tensor,
        is_polymer: torch.Tensor,
        rep_atom_mask: torch.Tensor,
    ) -> torch.Tensor:
        """calculate the lddt as described in Sec 4.3.1.

        Args:
            pred_coordinate (torch.Tensor):
                [..., N_sample, N_atom, 3]
            true_coordinate (torch.Tensor):
                [..., N_atom]
            is_nucleotide (torch.Tensor):
                [N_atom] or [..., N_atom]
            is_polymer (torch.Tensor):
                [N_atom]
            rep_atom_mask (torch.Tensor):
                [N_atom]

        Returns:
            torch.Tensor: per-atom lddt
                [..., N_sample, N_atom]
        """

        N_atom = true_coordinate.size(-2)
        atom_m_mask = (rep_atom_mask * is_polymer).bool()  # [N_atom]
        # Distance: d_lm
        pred_d_lm = torch.cdist(
            pred_coordinate, pred_coordinate[..., atom_m_mask, :]
        )  # [..., N_sample, N_atom, N_atom(m)]
        true_d_lm = torch.cdist(
            true_coordinate, true_coordinate[..., atom_m_mask, :]
        )  # [..., N_atom, N_atom(m)]
        delta_d_lm = torch.abs(
            pred_d_lm - true_d_lm.unsqueeze(dim=-3)
        )  # [..., N_sample, N_atom, N_atom(m)]

        # Pair-wise lddt
        thresholds = [0.5, 1, 2, 4]
        lddt_lm = (
            torch.stack([delta_d_lm < t for t in thresholds], dim=-1)
            .to(dtype=delta_d_lm.dtype)
            .mean(dim=-1)
        )  # [..., N_sample, N_atom, N_atom(m)]

        # Select atoms that are within certain threshold to l in ground truth
        # Restrict to bespoke inclusion radius
        is_nucleotide = is_nucleotide[
            ..., atom_m_mask
        ].bool()  # [N_atom(m)] or [..., N_atom(m)]
        locality_mask = (
            true_d_lm < self.is_nucleotide_threshold
        ) * is_nucleotide.unsqueeze(dim=-2) + (
            true_d_lm < self.is_not_nucleotide_threshold
        ) * (
            ~is_nucleotide.unsqueeze(dim=-2)
        )  # [..., N_atom, N_atom(m)]

        # Remove self-distance computation
        diagonal_mask = ((1 - torch.eye(n=N_atom)).bool().to(true_d_lm.device))[
            ..., atom_m_mask
        ]  # [N_atom, N_atom(m)]

        pair_mask = (locality_mask * diagonal_mask).unsqueeze(
            dim=-3
        )  # [..., 1, N_atom, N_atom(m)]

        per_atom_lddt = torch.sum(
            lddt_lm * pair_mask, dim=-1, keepdim=True
        )  # [...,  N_sample, N_atom, 1]
        if self.normalize:
            per_atom_lddt = per_atom_lddt / (
                torch.sum(pair_mask.to(dtype=per_atom_lddt.dtype), dim=-1, keepdim=True)
                + self.eps
            )
        # Distribute into bins
        boundaries = torch.linspace(
            start=self.min_bin,
            end=self.max_bin,
            steps=self.no_bins + 1,
            device=true_coordinate.device,
        )  # [N_bins]

        true_bins = torch.sum(
            per_atom_lddt > boundaries, dim=-1
        )  # [...,  N_sample, N_atom], range in [1, no_bins]
        true_bins = torch.clamp(
            true_bins, min=1, max=self.no_bins
        )  # just in case bin=0/no_bins+1 occurs
        true_bins = F.one_hot(
            true_bins - 1, self.no_bins
        )  # [...,  N_sample, N_atom, N_bins]

        return true_bins

    def forward(
        self,
        logits: torch.Tensor,
        pred_coordinate: torch.Tensor,
        true_coordinate: torch.Tensor,
        coordinate_mask: torch.Tensor,
        rep_atom_mask: torch.Tensor,
        is_nucleotide: torch.Tensor,
        is_polymer: torch.Tensor,
    ) -> torch.Tensor:
        """PLDDT loss

        Args:
            logits (torch.Tensor): logits
                [..., N_sample, N_atom, no_bins:=50]
            pred_coordinate (torch.Tensor): predicted coordinates
                [..., N_sample, N_atom, 3]
            true_coordinate (torch.Tensor): true coordinates
                [..., N_atom, 3]
            coordinate_mask (torch.Tensor): whether true coordinates exist
                [N_atom]
            is_nucleotide (torch.Tensor): "is_rna" or "is_dna"
                [N_atom]
            is_polymer (torch.Tensor): not "is_ligand"
                [N_atom]
            rep_atom_mask (torch.Tensor): representative atom of each token
                [N_atom]

        Returns:
            torch.Tensor: the return loss
                [...] if self.reduction is None else []
        """
        assert (
            is_nucleotide.shape
            == is_polymer.shape
            == rep_atom_mask.shape
            == coordinate_mask.shape
            == coordinate_mask.view(-1).shape
        )

        coordinate_mask = coordinate_mask.bool()
        rep_atom_mask = rep_atom_mask.bool()
        is_nucleotide = is_nucleotide.bool()
        is_polymer = is_polymer.bool()

        with torch.no_grad():
            true_bins = self.calculate_label(
                pred_coordinate=pred_coordinate[..., coordinate_mask, :],
                true_coordinate=true_coordinate[..., coordinate_mask, :],
                is_nucleotide=is_nucleotide[coordinate_mask],
                is_polymer=is_polymer[coordinate_mask],
                rep_atom_mask=rep_atom_mask[coordinate_mask],
            ).detach()  # [..., N_sample, N_atom_with_coords, N_bins]

        plddt_loss = softmax_cross_entropy(
            logits=logits[..., coordinate_mask, :],
            labels=true_bins,
        )  # [..., N_sample, N_atom_with_coords]

        # Average over atoms
        plddt_loss = plddt_loss.mean(dim=-1)  # [..., N_sample]

        # Average over samples
        plddt_loss = plddt_loss.mean(dim=-1)  # [...]

        return loss_reduction(plddt_loss, method=self.reduction)

class ClashLoss(nn.Module):
    """
    A soft clash loss based on overlap of van der Waals spheres.
    Reference: https://pmc.ncbi.nlm.nih.gov/articles/PMC3058769/
    """

    def __init__(
        self,
        tolerance: float = 0.4,
        square: bool = True,
        eps: float = 1e-6,
        reduction: str = "mean",
    ) -> None:
        """
        Args:
            tolerance (float, optional): allowed overlap (Å). Defaults to 0.4.
            square (bool, optional): square the overlap penalty. Defaults to True.
            eps (float, optional): avoid nan. Defaults to 1e-6.
            reduction (str, optional): reduction method for the batch dims. Defaults to mean.
        """
        super(ClashLoss, self).__init__()
        self.tolerance = tolerance
        self.square = square
        self.eps = eps
        self.reduction = reduction

    def _chunk_forward(
        self,
        pred_coordinate: torch.Tensor,
        radii_sum: torch.Tensor,
        pair_mask: torch.Tensor,
    ) -> torch.Tensor:
        """
        Args:
            pred_coordinate: [..., N_sample_chunk, N_atom, 3]
            radii_sum: [..., N_atom, N_atom]
            pair_mask: [..., 1, N_atom, N_atom]
        Returns:
            torch.Tensor: [..., N_sample_chunk]
        """
        # Pairwise distances
        pred_distance = torch.cdist(pred_coordinate, pred_coordinate)

        # Effective minimum distance: r_i + r_j - tolerance
        effective_min_dist = torch.clamp(radii_sum - self.tolerance, min=0.0)

        # Overlap = max(0, effective_min_dist - d_ij)
        overlap = F.relu(effective_min_dist.unsqueeze(dim=-3) - pred_distance)
        if self.square:
            overlap = overlap**2

        loss = torch.sum(overlap * pair_mask, dim=(-1, -2))
        denom = torch.sum(pair_mask, dim=(-1, -2)) + self.eps
        return loss / denom

    def forward(
        self,
        pred_coordinate: torch.Tensor,
        coordinate_mask: torch.Tensor,
        atom_vdw_radius: torch.Tensor,
        pair_mask: Optional[torch.Tensor] = None,
        per_sample_scale: Optional[torch.Tensor] = None,
        diffusion_chunk_size: Optional[int] = None,
    ) -> torch.Tensor:
        """
        Args:
            pred_coordinate (torch.Tensor): predicted coordinates
                [..., N_sample, N_atom, 3]
            coordinate_mask (torch.Tensor): whether coordinates exist
                [N_atom] or [..., N_atom]
            atom_vdw_radius (torch.Tensor): vdw radius per atom
                [N_atom] or [..., N_atom]
            pair_mask (Optional[torch.Tensor]): extra pair mask to exclude pairs
                [N_atom, N_atom] or [..., N_atom, N_atom]
            per_sample_scale (Optional[torch.Tensor]): scale by per-sample noise
                [..., N_sample]
            diffusion_chunk_size (Optional[int]): chunk size over N_sample
        Returns:
            torch.Tensor: the clash loss
                [...] if reduction is None else []
        """
        coordinate_mask = coordinate_mask.bool()

        # Base pair mask from coordinate existence
        base_pair_mask = (
            coordinate_mask[..., None] * coordinate_mask[..., None, :]
        ).to(dtype=pred_coordinate.dtype)

        # Remove diagonal (self pairs)
        diag = torch.eye(
            n=base_pair_mask.size(-1),
            device=base_pair_mask.device,
            dtype=base_pair_mask.dtype,
        )
        base_pair_mask = base_pair_mask * (1 - diag)

        if pair_mask is not None:
            base_pair_mask = base_pair_mask * pair_mask.to(base_pair_mask.dtype)

        # [ ..., 1, N_atom, N_atom ] for broadcasting over N_sample
        base_pair_mask = base_pair_mask.unsqueeze(dim=-3)

        # Radii sum [ ..., N_atom, N_atom ]
        radii_sum = atom_vdw_radius[..., None] + atom_vdw_radius[..., None, :]

        # Compute loss
        if diffusion_chunk_size is None:
            clash_loss = self._chunk_forward(
                pred_coordinate=pred_coordinate,
                radii_sum=radii_sum,
                pair_mask=base_pair_mask,
            )
        else:
            checkpoint_fn = get_checkpoint_fn()
            clash_loss = []
            N_sample = pred_coordinate.shape[-3]
            no_chunks = N_sample // diffusion_chunk_size + (
                N_sample % diffusion_chunk_size != 0
            )
            for i in range(no_chunks):
                clash_loss_i = checkpoint_fn(
                    self._chunk_forward,
                    pred_coordinate[
                        ...,
                        i * diffusion_chunk_size : (i + 1) * diffusion_chunk_size,
                        :,
                        :,
                    ],
                    radii_sum,
                    base_pair_mask,
                    use_reentrant=False,
                )
                clash_loss.append(clash_loss_i)
            clash_loss = torch.cat(clash_loss, dim=-1)

        if per_sample_scale is not None:
            clash_loss = clash_loss * per_sample_scale

        clash_loss = clash_loss.mean(dim=-1)  # [...]
        return loss_reduction(clash_loss, method=self.reduction)

class ProtenixLoss(nn.Module):
    """Aggregation of the various losses"""

    def __init__(self, configs) -> None:
        super(ProtenixLoss, self).__init__()
        self.configs = configs

        self.alpha_confidence = self.configs.loss.weight.alpha_confidence
        self.alpha_pae = self.configs.loss.weight.alpha_pae
        self.alpha_except_pae = self.configs.loss.weight.alpha_except_pae
        self.alpha_diffusion = self.configs.loss.weight.alpha_diffusion
        self.alpha_distogram = self.configs.loss.weight.alpha_distogram
        self.alpha_bond = self.configs.loss.weight.alpha_bond
        self.weight_smooth_lddt = self.configs.loss.weight.smooth_lddt

        self.lddt_radius = {
            "is_nucleotide_threshold": 30.0,
            "is_not_nucleotide_threshold": 15.0,
        }

        self.loss_weight = {
            # confidence
            "plddt_loss": self.alpha_confidence * self.alpha_except_pae,
            "pde_loss": self.alpha_confidence * self.alpha_except_pae,
            "resolved_loss": self.alpha_confidence * self.alpha_except_pae,
            "pae_loss": self.alpha_confidence * self.alpha_pae,
            # diffusion
            "mse_loss": self.alpha_diffusion,
            "bond_loss": self.alpha_diffusion * self.alpha_bond,
            "smooth_lddt_loss": self.alpha_diffusion
            * self.weight_smooth_lddt,  # Different from AF3 appendix eq(6), where smooth_lddt has no weight
            # distogram
            "distogram_loss": self.alpha_distogram,
        }

        # Loss
        self.plddt_loss = PLDDTLoss(**configs.loss.plddt, **self.lddt_radius)
        self.pde_loss = PDELoss(**configs.loss.pde)
        self.resolved_loss = ExperimentallyResolvedLoss(**configs.loss.resolved)
        self.pae_loss = PAELoss(**configs.loss.pae)
        self.mse_loss = MSELoss(**configs.loss.diffusion.mse)
        self.bond_loss = BondLoss(**configs.loss.diffusion.bond)
        self.smooth_lddt_loss = SmoothLDDTLoss(**configs.loss.diffusion.smooth_lddt)
        self.distogram_loss = DistogramLoss(**configs.loss.distogram)

    def calculate_label(
        self,
        feat_dict: dict[str, Any],
        label_dict: dict[str, Any],
    ) -> dict[str, Any]:
        """calculate true distance, and atom pair mask

        Args:
            feat_dict (dict): Feature dictionary containing additional features.
            label_dict (dict): Label dictionary containing ground truth data.

        Returns:
            label_dict (dict): with the following updates:
                distance (torch.Tensor): true atom-atom distance.
                    [..., N_atom, N_atom]
                distance_mask (torch.Tensor): atom-atom mask indicating whether true distance exists.
                    [..., N_atom, N_atom]
        """
        # Distance mask
        distance_mask = (
            label_dict["coordinate_mask"][..., None]
            * label_dict["coordinate_mask"][..., None, :]
        )
        # Distances for all atom pairs
        # Note: we convert to bf16 for saving cuda memory, if performance drops, do not convert it
        distance = (
            cdist(label_dict["coordinate"], label_dict["coordinate"]) * distance_mask
        ).to(
            label_dict["coordinate"].dtype
        )  # [..., N_atom, N_atom]

        lddt_mask = compute_lddt_mask(
            true_distance=distance,
            distance_mask=distance_mask,
            is_nucleotide=feat_dict["is_rna"].bool() + feat_dict["is_dna"].bool(),
            **self.lddt_radius,
        )

        label_dict["lddt_mask"] = lddt_mask
        label_dict["distance_mask"] = distance_mask
        if not self.configs.loss_metrics_sparse_enable:
            label_dict["distance"] = distance
        del distance, distance_mask, lddt_mask
        return label_dict

    def calculate_prediction(
        self,
        pred_dict: dict[str, torch.Tensor],
    ) -> dict[str, torch.Tensor]:
        """get more predictions used for calculating difference losses

        Args:
            pred_dict (dict[str, torch.Tensor]): raw prediction dict given by the model

        Returns:
            dict[str, torch.Tensor]: updated predictions
        """
        if not self.configs.loss_metrics_sparse_enable:
            pred_dict["distance"] = torch.cdist(
                pred_dict["coordinate"], pred_dict["coordinate"]
            ).to(
                pred_dict["coordinate"].dtype
            )  # [..., N_atom, N_atom]
        return pred_dict

    def aggregate_losses(
        self, loss_fns: dict, has_valid_resolution: Optional[torch.Tensor] = None
    ) -> tuple[torch.Tensor, dict]:
        """
        Aggregates multiple loss functions and their respective metrics.

        Args:
            loss_fns (dict): Dictionary of loss functions to be aggregated.
            has_valid_resolution (Optional[torch.Tensor]): Tensor indicating valid resolutions. Defaults to None.

        Returns:
            tuple[torch.Tensor, dict]:
                - cum_loss (torch.Tensor): Cumulative loss.
                - all_metrics (dict): Dictionary containing all metrics.
        """
        cum_loss = 0.0
        all_metrics = {}
        for loss_name, loss_fn in loss_fns.items():
            weight = self.loss_weight[loss_name]
            loss_outputs = loss_fn()
            if isinstance(loss_outputs, tuple):
                loss, metrics = loss_outputs
            else:
                assert isinstance(loss_outputs, torch.Tensor)
                loss, metrics = loss_outputs, {}

            all_metrics.update(
                {f"{loss_name}/{key}": val for key, val in metrics.items()}
            )
            if torch.isnan(loss) or torch.isinf(loss):
                logging.warning(f"{loss_name} loss is NaN. Skipping...")
            if (
                (has_valid_resolution is not None)
                and (has_valid_resolution.sum() == 0)
                and (
                    loss_name in ["plddt_loss", "pde_loss", "resolved_loss", "pae_loss"]
                )
            ):
                loss = 0.0 * loss
            else:
                all_metrics[loss_name] = loss.detach().clone()
                all_metrics[f"weighted_{loss_name}"] = weight * loss.detach().clone()

            cum_loss = cum_loss + weight * loss
        all_metrics["loss"] = cum_loss.detach().clone()

        return cum_loss, all_metrics

    def calculate_losses(
        self,
        feat_dict: dict[str, Any],
        pred_dict: dict[str, torch.Tensor],
        label_dict: dict[str, Any],
        mode: str = "train",
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        """
        Calculate the cumulative loss and aggregated metrics for the given predictions and labels.

        Args:
            feat_dict (dict[str, Any]): Feature dictionary containing additional features.
            pred_dict (dict[str, torch.Tensor]): Prediction dictionary containing model outputs.
            label_dict (dict[str, Any]): Label dictionary containing ground truth data.
            mode (str): Mode of operation ('train', 'eval', 'inference'). Defaults to 'train'.

        Returns:
            tuple[torch.Tensor, dict[str, torch.Tensor]]:
                - cum_loss (torch.Tensor): Cumulative loss.
                - metrics (dict[str, torch.Tensor]): Dictionary containing aggregated metrics.
        """
        assert mode in ["train", "eval", "inference"]
        if mode == "train":
            # Confidence Loss: use mini-rollout coordinates
            confidence_coordinate = "coordinate_mini"
            if not self.configs.train_confidence_only:
                # Scale diffusion loss with noise-level
                if pred_dict.get("noise_level") is None:
                    diffusion_per_sample_scale = None
                else:
                    diffusion_per_sample_scale = (
                        pred_dict["noise_level"] ** 2 + self.configs.sigma_data**2
                    ) / (self.configs.sigma_data * pred_dict["noise_level"]) ** 2

        else:
            # Confidence Loss: use diffusion coordinates
            confidence_coordinate = "coordinate"
            # No scale is required
            diffusion_per_sample_scale = None

        if self.configs.train_confidence_only and mode == "train":
            # Skip Diffusion Loss and distogram loss
            loss_fns = {}
        else:
            # Diffusion Loss: SmoothLDDTLoss / BondLoss / MSELoss
            loss_fns = {}
            if self.configs.loss.diffusion_lddt_loss_dense:
                loss_fns.update(
                    {
                        "smooth_lddt_loss": lambda: self.smooth_lddt_loss.dense_forward(
                            pred_coordinate=pred_dict["coordinate"],
                            true_coordinate=label_dict["coordinate"],
                            lddt_mask=label_dict["lddt_mask"],
                            diffusion_chunk_size=self.configs.loss.diffusion_lddt_chunk_size,
                        )  # it's faster is not OOM
                    }
                )
            elif self.configs.loss.diffusion_sparse_subsample_loss.enable:
                loss_fns.update(
                    {
                        "smooth_lddt_loss": lambda: self.smooth_lddt_loss.sparse_forward_subsample(
                            pred_coordinate=pred_dict["coordinate"],
                            true_coordinate=label_dict["coordinate"],
                            lddt_mask=label_dict["lddt_mask"],
                            diffusion_chunk_size=self.configs.loss.diffusion_lddt_chunk_size,
                            sample_ratio=self.configs.loss.diffusion_sparse_subsample_loss.sample_ratio,
                            max_samples=self.configs.loss.diffusion_sparse_subsample_loss.max_samples,
                        )
                    }
                )
            elif self.configs.loss.diffusion_sparse_loss_enable:
                loss_fns.update(
                    {
                        "smooth_lddt_loss": lambda: self.smooth_lddt_loss.sparse_forward(
                            pred_coordinate=pred_dict["coordinate"],
                            true_coordinate=label_dict["coordinate"],
                            lddt_mask=label_dict["lddt_mask"],
                            diffusion_chunk_size=self.configs.loss.diffusion_lddt_chunk_size,
                        )
                    }
                )
            else:
                loss_fns.update(
                    {
                        "smooth_lddt_loss": lambda: self.smooth_lddt_loss(
                            pred_distance=pred_dict["distance"],
                            true_distance=label_dict["distance"],
                            distance_mask=label_dict["distance_mask"],
                            lddt_mask=label_dict["lddt_mask"],
                            diffusion_chunk_size=self.configs.loss.diffusion_lddt_chunk_size,
                        )
                    }
                )
            loss_fns.update(
                {
                    "bond_loss": lambda: (
                        self.bond_loss.sparse_forward(
                            pred_coordinate=pred_dict["coordinate"],
                            true_coordinate=label_dict["coordinate"],
                            distance_mask=label_dict["distance_mask"],
                            bond_mask=feat_dict["bond_mask"],
                            per_sample_scale=diffusion_per_sample_scale,
                        )
                        if self.configs.loss.diffusion_sparse_loss_enable
                        else self.bond_loss(
                            pred_distance=pred_dict["distance"],
                            true_distance=label_dict["distance"],
                            distance_mask=label_dict["distance_mask"],
                            bond_mask=feat_dict["bond_mask"],
                            per_sample_scale=diffusion_per_sample_scale,
                            diffusion_chunk_size=self.configs.loss.diffusion_bond_chunk_size,
                        )
                    ),
                    "mse_loss": lambda: self.mse_loss(
                        pred_coordinate=pred_dict["coordinate"],
                        true_coordinate=label_dict["coordinate"],
                        coordinate_mask=label_dict["coordinate_mask"],
                        is_rna=feat_dict["is_rna"],
                        is_dna=feat_dict["is_dna"],
                        is_ligand=feat_dict["is_ligand"],
                        per_sample_scale=diffusion_per_sample_scale,
                    ),
                }
            )
            # Distogram Loss
            if "distogram" in pred_dict:
                loss_fns.update(
                    {
                        "distogram_loss": lambda: self.distogram_loss(
                            logits=pred_dict["distogram"],
                            true_coordinate=label_dict["coordinate"],
                            coordinate_mask=label_dict["coordinate_mask"],
                            rep_atom_mask=feat_dict["distogram_rep_atom_mask"],
                        )
                    }
                )

        # Confidence Loss:
        # Only when resoluton is in [min_resolution, max_resolution] the confidence loss is considered
        # NOTE: here we assume batch_size == 1
        resolution = feat_dict["resolution"].item()
        has_valid_resolution = (resolution >= self.configs.loss.resolution.min) & (
            resolution <= self.configs.loss.resolution.max
        )

        if has_valid_resolution:
            has_valid_resolution = torch.tensor(
                [1.0],
                dtype=label_dict["coordinate"].dtype,
                device=label_dict["coordinate"].device,
            )
        else:
            has_valid_resolution = torch.tensor(
                [0.0],
                dtype=label_dict["coordinate"].dtype,
                device=label_dict["coordinate"].device,
            )

        if all(x in pred_dict for x in ["plddt", "pde", "pae", "resolved"]):
            loss_fns.update(
                {
                    "plddt_loss": lambda: self.plddt_loss(
                        logits=pred_dict["plddt"],
                        pred_coordinate=pred_dict[confidence_coordinate].detach(),
                        true_coordinate=label_dict["coordinate"],
                        coordinate_mask=label_dict["coordinate_mask"],
                        rep_atom_mask=feat_dict["plddt_m_rep_atom_mask"],
                        is_nucleotide=feat_dict["is_rna"] + feat_dict["is_dna"],
                        is_polymer=1 - feat_dict["is_ligand"],
                    ),
                    "pde_loss": lambda: self.pde_loss(
                        logits=pred_dict["pde"],
                        pred_coordinate=pred_dict[confidence_coordinate].detach(),
                        true_coordinate=label_dict["coordinate"],
                        coordinate_mask=label_dict["coordinate_mask"],
                        rep_atom_mask=feat_dict["distogram_rep_atom_mask"],
                    ),
                    "resolved_loss": lambda: self.resolved_loss(
                        logits=pred_dict["resolved"],
                        coordinate_mask=label_dict["coordinate_mask"],
                    ),
                    "pae_loss": lambda: self.pae_loss(
                        logits=pred_dict["pae"],
                        pred_coordinate=pred_dict[confidence_coordinate].detach(),
                        true_coordinate=label_dict["coordinate"],
                        coordinate_mask=label_dict["coordinate_mask"],
                        frame_atom_index=feat_dict["frame_atom_index"],
                        rep_atom_mask=feat_dict["pae_rep_atom_mask"],
                        has_frame=feat_dict["has_frame"],
                    ),
                }
            )

        cum_loss, metrics = self.aggregate_losses(loss_fns, has_valid_resolution)
        return cum_loss, metrics

    def forward(
        self,
        feat_dict: dict[str, Any],
        pred_dict: dict[str, torch.Tensor],
        label_dict: dict[str, Any],
        mode: str = "train",
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        """
        Forward pass for calculating the cumulative loss and aggregated metrics.

        Args:
            feat_dict (dict[str, Any]): Feature dictionary containing additional features.
            pred_dict (dict[str, torch.Tensor]): Prediction dictionary containing model outputs.
            label_dict (dict[str, Any]): Label dictionary containing ground truth data.
            mode (str): Mode of operation ('train', 'eval', 'inference'). Defaults to 'train'.

        Returns:
            tuple[torch.Tensor, dict[str, torch.Tensor]]:
                - cum_loss (torch.Tensor): Cumulative loss.
                - losses (dict[str, torch.Tensor]): Dictionary containing aggregated metrics.
        """
        diffusion_chunk_size = self.configs.loss.diffusion_chunk_size_outer
        assert mode in ["train", "eval", "inference"]
        # Pre-computations
        with torch.no_grad():
            label_dict = self.calculate_label(feat_dict, label_dict)

        for k in ["coordinate", "coordinate_mini"]:
            if k in pred_dict:
                pred_dict[k] = pred_dict[k].to(torch.float32)
        pred_dict = self.calculate_prediction(pred_dict)

        if diffusion_chunk_size <= 0:
            # Calculate losses
            cum_loss, losses = self.calculate_losses(
                feat_dict=feat_dict,
                pred_dict=pred_dict,
                label_dict=label_dict,
                mode=mode,
            )
        else:
            if "coordinate" in pred_dict:
                N_sample = pred_dict["coordinate"].shape[-3]
            elif self.configs.train_confidence_only:
                N_sample = pred_dict["coordinate_mini"].shape[-3]
            else:
                raise KeyError("Missing key: coordinate (in pred_dict).")
            no_chunks = N_sample // diffusion_chunk_size + (
                N_sample % diffusion_chunk_size != 0
            )
            cum_loss = 0.0
            losses = {}
            for i in range(no_chunks):
                cur_sample_num = min(
                    diffusion_chunk_size, N_sample - i * diffusion_chunk_size
                )
                pred_dict_i = {}
                for key, value in pred_dict.items():
                    if key in ["coordinate"] and mode == "train":
                        pred_dict_i[key] = value[
                            i * diffusion_chunk_size : (i + 1) * diffusion_chunk_size,
                            :,
                            :,
                        ]
                    elif (
                        key in ["coordinate", "plddt", "pae", "pde", "resolved"]
                        and mode != "train"
                    ):
                        pred_dict_i[key] = value[
                            i * diffusion_chunk_size : (i + 1) * diffusion_chunk_size,
                            :,
                            :,
                        ]
                    elif key == "noise_level":
                        pred_dict_i[key] = value[
                            i * diffusion_chunk_size : (i + 1) * diffusion_chunk_size
                        ]
                    else:
                        pred_dict_i[key] = value
                pred_dict_i = self.calculate_prediction(pred_dict_i)
                cum_loss_i, losses_i = self.calculate_losses(
                    feat_dict=feat_dict,
                    pred_dict=pred_dict_i,
                    label_dict=label_dict,
                    mode=mode,
                )
                cum_loss += cum_loss_i * cur_sample_num
                # Aggregate metrics
                for key, value in losses_i.items():
                    if key in losses:
                        losses[key] += value * cur_sample_num
                    else:
                        losses[key] = value * cur_sample_num
            cum_loss /= N_sample
            for key in losses.keys():
                losses[key] /= N_sample

        return cum_loss, losses

class GlobalReprLoss(nn.Module):
    def __init__(self):
        super(GlobalReprLoss, self).__init__()
        
    def forward(self, repr1, repr2):
        if torch.isnan(repr1).any() or torch.isinf(repr1).any():
            print(f"GlobalReprLoss: repr1 contains NaN/Inf before cleaning. Shape: {repr1.shape}, dtype: {repr1.dtype}")
        if torch.isnan(repr2).any() or torch.isinf(repr2).any():
            print(f"GlobalReprLoss: repr2 contains NaN/Inf before cleaning. Shape: {repr2.shape}, dtype: {repr2.dtype}")

        clean_val_limit = 1e4 # Value for replacing Inf and for clamping range

        # Step 1: Convert NaNs to 0.0, and Inf to clean_val_limit.
        repr1 = torch.nan_to_num(repr1, nan=0.0, posinf=clean_val_limit, neginf=-clean_val_limit)
        repr2 = torch.nan_to_num(repr2, nan=0.0, posinf=clean_val_limit, neginf=-clean_val_limit)
        
        # Step 2: Clamp other extreme finite values to the same limit.
        # This ensures that values originally finite but very large (e.g., 1e20) are also capped.
        repr1 = torch.clamp(repr1, min=-clean_val_limit, max=clean_val_limit)
        repr2 = torch.clamp(repr2, min=-clean_val_limit, max=clean_val_limit)
        
        if torch.isnan(repr1).any() or torch.isinf(repr1).any():
             print(f"GlobalReprLoss: ERROR - repr1 STILL contains NaN/Inf AFTER cleaning/clamping. Shape: {repr1.shape}")
        if torch.isnan(repr2).any() or torch.isinf(repr2).any():
             print(f"GlobalReprLoss: ERROR - repr2 STILL contains NaN/Inf AFTER cleaning/clamping. Shape: {repr2.shape}")

        # Calculate MSE loss with cleaned and clamped inputs
        mse_loss = F.mse_loss(repr1, repr2)
        if torch.isnan(mse_loss) or torch.isinf(mse_loss):
            mse_val_item = mse_loss.item() # Store before overwriting
            print(f"GlobalReprLoss: mse_loss is NaN/Inf ({mse_val_item}). This is unexpected after input cleaning & clamping.")
            # Fallback to a high, but finite MSE value based on clamp limits
            # Max possible MSE: (limit - (-limit))^2 = (2*limit)^2, but F.mse_loss is mean, so (limit-0)^2 is a safer upper bound if one is 0
            mse_loss = torch.tensor(clean_val_limit**2, device=repr1.device, dtype=repr1.dtype) 
            print(f"GlobalReprLoss: mse_loss reset to {mse_loss.item()}")

        # Prepare inputs for cosine similarity
        _repr1 = repr1.clone()
        _repr2 = repr2.clone()

        if _repr1.dim() == 1:
            _repr1 = _repr1.unsqueeze(0)
        if _repr2.dim() == 1:
            _repr2 = _repr2.unsqueeze(0)
        
        norm_repr1 = torch.linalg.norm(_repr1, dim=-1)
        norm_repr2 = torch.linalg.norm(_repr2, dim=-1)

        if (norm_repr1 == 0).any():
            print(f"GlobalReprLoss: repr1 has zero norm for some samples after cleaning/clamping. Norms: {norm_repr1}")
        if (norm_repr2 == 0).any():
            print(f"GlobalReprLoss: repr2 has zero norm for some samples after cleaning/clamping. Norms: {norm_repr2}")
        
        cosine_sim = F.cosine_similarity(_repr1, _repr2, dim=-1, eps=1e-8) 
        
        if torch.isnan(cosine_sim).any():
            # This should be rare if inputs _repr1, _repr2 are finite and eps is used.
            # Can happen if a batch element in _repr1 AND _repr2 are both exactly zero vector causing 0/0 even with eps in some cases.
            # PyTorch F.cosine_similarity with eps should yield 0 if both are zero vectors.
            print(f"GlobalReprLoss: cosine_similarity output is NaN. Norms: repr1_norm={norm_repr1}, repr2_norm={norm_repr2}. Values (first sample if batched): _repr1[0]={_repr1[0] if _repr1.numel() > 0 else 'empty'}, _repr2[0]={_repr2[0] if _repr2.numel() > 0 else 'empty'}")
            # Replace NaN similarities with 0 (implies no similarity or undefined)
            cosine_sim = torch.nan_to_num(cosine_sim, nan=0.0)
            print(f"GlobalReprLoss: cosine_similarity NaNs replaced with 0.")

        cosine_loss = 1 - cosine_sim
        cosine_loss = cosine_loss.mean() # Mean over batch and other dimensions if cosine_sim was multi-dim

        if torch.isnan(cosine_loss) or torch.isinf(cosine_loss):
            # Cosine loss is 1-sim; sim in [-1,1] -> loss in [0,2]. Should not be Inf.
            # NaN can happen if cosine_sim.mean() processes NaNs not caught above (e.g. if cosine_sim was all NaNs).
            cl_val_item = cosine_loss.item() # Store before overwriting
            print(f"GlobalReprLoss: cosine_loss is NaN/Inf ({cl_val_item}). Cosine similarity mean was: {cosine_sim.mean().item() if cosine_sim.numel() > 0 else 'N/A'}")
            cosine_loss = torch.tensor(1.0, device=repr1.device, dtype=repr1.dtype) # Default to 1.0 (e.g. orthogonal)
            print(f"GlobalReprLoss: cosine_loss reset to {cosine_loss.item()}")

        final_loss = mse_loss + cosine_loss
        if torch.isnan(final_loss) or torch.isinf(final_loss):
            fl_val_item = final_loss.item() # Store before overwriting
            print(f"GlobalReprLoss: final_loss is NaN/Inf ({fl_val_item}). MSE: {mse_loss.item()}, CosineLoss: {cosine_loss.item()}")
            # Fallback to sum of capped/defaulted components
            # Max mse_loss capped at clean_val_limit**2, max cosine_loss (1- (-1))=2, default is 1
            final_loss = torch.tensor(clean_val_limit**2 + 1.0, device=repr1.device, dtype=repr1.dtype)
            print(f"GlobalReprLoss: final_loss was NaN/Inf, reset to {final_loss.item()}")
        return final_loss

'''
1. align loss进行归一化
2. 只对齐多肽部分
3. 主链骨架原子和seq模态对齐的权重更大，侧链原子和smiles模态对齐的权重更小
'''
class MultiPSP_align_loss(nn.Module):
    def __init__(self, configs):
        super().__init__()
        self.configs = configs

        self.alpha_confidence = self.configs.loss.weight.alpha_confidence
        self.alpha_pae = self.configs.loss.weight.alpha_pae
        self.alpha_except_pae = self.configs.loss.weight.alpha_except_pae
        self.alpha_diffusion = self.configs.loss.weight.alpha_diffusion
        self.alpha_distogram = self.configs.loss.weight.alpha_distogram
        self.alpha_bond = self.configs.loss.weight.alpha_bond
        self.weight_smooth_lddt = self.configs.loss.weight.smooth_lddt
        self.alpha_rmsd = self.configs.loss.weight.alpha_rmsd

        self.lddt_radius = {
            "is_nucleotide_threshold": 30.0,
            "is_not_nucleotide_threshold": 15.0,
        }

        self.loss_weight = {
            # confidence
            "plddt_loss": self.alpha_confidence * self.alpha_except_pae,
            # "pde_loss": self.alpha_confidence * self.alpha_except_pae,
            "resolved_loss": self.alpha_confidence * self.alpha_except_pae,
            # "pae_loss": self.alpha_confidence * self.alpha_pae,
            # diffusion
            "mse_loss": self.alpha_diffusion,
            # "bond_loss": self.alpha_diffusion * self.alpha_bond,
            "smooth_lddt_loss": self.alpha_diffusion * self.weight_smooth_lddt,  # Different from AF3 appendix eq(6), where smooth_lddt has no weight
            # distogram
            "distogram_loss": self.alpha_distogram,
            "rmsd_loss": self.alpha_rmsd,
        }

        self.mse_loss = MSELoss(**configs.loss.diffusion.mse)
        self.global_repr_loss = GlobalReprLoss()
        # 新增 —— 直接沿用 Protenix 的实现
        self.smooth_lddt = SmoothLDDTLoss(**configs.loss.diffusion.smooth_lddt)
        # self.pde_loss     = PDELoss(**configs.loss.pde)
        self.plddt_loss   = PLDDTLoss(**configs.loss.plddt,
                                      **{"is_nucleotide_threshold":self.lddt_radius["is_nucleotide_threshold"],
                                         "is_not_nucleotide_threshold":self.lddt_radius["is_not_nucleotide_threshold"]})

    def kabsch_algorithm(self, P, Q):
        """
        Kabsch算法计算最优旋转矩阵和平移向量
        
        参数:
            P: 参考坐标集 (tensor, n x 3)
            Q: 待对齐坐标集 (tensor, n x 3)
        
        返回:
            R: 旋转矩阵 (tensor, 3 x 3)
            t: 平移向量 (tensor, 3)
        """
        # 确保输入是数组，并且具有相同的大小
        # print("P.shape, Q.shape", P.shape, Q.shape) # Already exists
        if torch.isnan(P).any() or torch.isinf(P).any():
            print(f"Kabsch: Input P contains NaN/Inf. Shape: {P.shape}")
        if torch.isnan(Q).any() or torch.isinf(Q).any():
            print(f"Kabsch: Input Q contains NaN/Inf. Shape: {Q.shape}")

        assert P.shape == Q.shape, f"Kabsch P shape {P.shape} and Q shape {Q.shape} must be equal."
        if P.shape[0] < 1: # Need at least 1 point, though SVD is more stable with >=3
             print(f"Kabsch: Not enough points. P shape: {P.shape}. Returning identity and zero translation.")
             return torch.eye(3, device=P.device, dtype=P.dtype), torch.zeros(3, device=P.device, dtype=P.dtype)

        # 计算质心
        centroid_P = torch.mean(P, dim=0)
        centroid_Q = torch.mean(Q, dim=0)
        if torch.isnan(centroid_P).any() or torch.isinf(centroid_P).any():
            print(f"Kabsch: centroid_P contains NaN/Inf.")
        if torch.isnan(centroid_Q).any() or torch.isinf(centroid_Q).any():
            print(f"Kabsch: centroid_Q contains NaN/Inf.")

        # 将坐标中心化
        P_centered = P - centroid_P
        Q_centered = Q - centroid_Q
        
        # 计算协方差矩阵 H = Q^T @ P (修正：原来是 P^T @ Q)
        H = Q_centered.transpose(-2, -1) @ P_centered
        if torch.isnan(H).any() or torch.isinf(H).any():
            print(f"Kabsch: Covariance matrix H contains NaN/Inf. H: {H}")
        
        H_detached = H.detach() 
        
        # 奇异值分解
        try:
            U, S, V = torch.linalg.svd(H_detached)
        except Exception as e:
            print(f"Kabsch: SVD failed. H_detached: {H_detached}. Error: {e}")
            # Return identity if SVD fails
            return torch.eye(3, device=P.device, dtype=P.dtype), torch.zeros(3, device=P.device, dtype=P.dtype)

        if torch.isnan(U).any() or torch.isinf(U).any() or \
           torch.isnan(S).any() or torch.isinf(S).any() or \
           torch.isnan(V).any() or torch.isinf(V).any():
            print(f"Kabsch: SVD output U, S, or V contains NaN/Inf. H_detached was: {H_detached}")
            print(f"U: {U}, S: {S}, V: {V}")
            return torch.eye(3, device=P.device, dtype=P.dtype), torch.zeros(3, device=P.device, dtype=P.dtype)

        # 计算旋转矩阵 R = V^T @ U^T (修正：原来是 U @ V^T)
        R = V.transpose(-2, -1) @ U.transpose(-2, -1)
        
        # 确保我们有一个右手系统
        det_R = torch.linalg.det(R)
        if torch.isnan(det_R):
            print(f"Kabsch: Determinant of R is NaN. R: {R}")
        if det_R < 0:
            # 修正反射：翻转V的最后一列
            V_corrected = V.clone()
            V_corrected[-1, :] *= -1
            R = V_corrected.transpose(-2, -1) @ U.transpose(-2, -1)
            new_det_R = torch.linalg.det(R)
            if torch.isnan(new_det_R):
                 print(f"Kabsch: Determinant of corrected R is NaN. Corrected R: {R}")
        
        # 计算平移向量 t = centroid_P - centroid_Q @ R^T (修正：原来是 centroid_P - R @ centroid_Q)
        t_translation_term = centroid_Q @ R.transpose(-2, -1)
        if torch.isnan(t_translation_term).any() or torch.isinf(t_translation_term).any():
            print(f"Kabsch: centroid_Q @ R^T for translation calculation contains NaN/Inf.")
        t = (centroid_P - t_translation_term).detach()

        if torch.isnan(R).any() or torch.isinf(R).any():
            print(f"Kabsch: Final R contains NaN/Inf. R: {R}")
        if torch.isnan(t).any() or torch.isinf(t).any():
            print(f"Kabsch: Final t contains NaN/Inf. t: {t}")
            
        return R.detach(), t # Ensure R is also detached

    def apply_transformation(self, coords, R, t):
        """
        应用旋转矩阵和平移向量到坐标集
        
        参数:
            coords: 坐标集 (tensor, n x 3)
            R: 旋转矩阵 (tensor, 3 x 3)
            t: 平移向量 (tensor, 3)
        
        返回:
            transformed_coords: 变换后的坐标集 (tensor, n x 3)
        """
        # 修正：应该是 coords @ R^T + t，而不是 (R @ coords^T)^T + t
        return coords @ R.transpose(-2, -1) + t

    def calculate_rmsd(self, P, Q, epsilon=1e-10):
        """
        计算两组坐标之间的RMSD，添加数值稳定性
        
        参数:
            P: 第一组坐标 (tensor, n x 3)
            Q: 第二组坐标 (tensor, n x 3)
            epsilon: 数值稳定性常数
        
        返回:
            rmsd: 均方根偏差 (tensor, scalar)
        """
        diff = P - Q
        return torch.sqrt(torch.mean(torch.sum(diff * diff, dim=1)) + epsilon)

    def get_atom_from_atom_array(self, atom_array: AtomArray):
        """
        从atom_array中获取原子
        """
        atom_instance_list = []
        num_atoms = atom_array.array_length()
        for i in range(num_atoms):
            atom = atom_array.get_atom(i) # 格式：Atom(np.array([193.553, 174.792, 227.732], dtype=float32), chain_id="B", res_id=674, ins_code="", res_name="ARG", hetero=False, atom_name="N", element="N", b_factor=87.52, occupancy=1.0, charge=0, auth_asym_id="K", label_alt_id="", auth_seq_id="674", label_entity_id="2", label_seq_id="674", label_asym_id="B", is_resolved=True, mol_type="protein", centre_atom_mask=0, chain_mol_type="protein", is_ligand=0, is_dna=0, is_rna=0, is_protein=1, distogram_rep_atom_mask=0, plddt_m_rep_atom_mask=0, cano_seq_resname="ARG", tokatom_idx=0, modified_res_mask=0, assembly_1=True, asym_id_int=1, entity_id_int=1, sym_id_int=0, mol_id=1, entity_mol_id=1, mol_atom_index=5246, ref_space_uid=697, ref_pos=[ 3.14147449  0.78444302 -0.24782169], ref_charge=0, ref_mask=1, res_perm="0_0", resolution=3.37)
            atom_instance_list.append(atom)
        return atom_instance_list

    def _get_atom_environment_vectorized(self, atoms, coords, radius=5.0):
        """向量化版本的原子环境计算，纯Torch实现"""
        # 确保 coords 为 Tensor
        if not isinstance(coords, torch.Tensor):
            coords = torch.as_tensor(coords)

        n_atoms = coords.shape[0]
        # 计算所有原子对之间的距离矩阵
        diffs = coords.unsqueeze(1) - coords.unsqueeze(0)  # shape: [n_atoms, n_atoms, 3]
        dist_matrix = torch.norm(diffs, p=2, dim=-1)        # shape: [n_atoms, n_atoms]

        # 构建环境列表
        all_envs = [[] for _ in range(n_atoms)]
        idx_range = torch.arange(n_atoms, device=coords.device)
        for i in range(n_atoms):
            # 排除自身并应用距离阈值
            mask = (dist_matrix[i] <= radius) & (idx_range != i)
            neigh_idxs = torch.nonzero(mask, as_tuple=False).view(-1)
            for j in neigh_idxs.tolist():
                all_envs[i].append((atoms[j].element, dist_matrix[i, j].item()))

        return all_envs
    
    def get_match_indices(self, pdb_id, ref_coords, pred_coords, ref_atoms, pred_atoms):
        cache_key = id(pdb_id)
        if hasattr(self, '_match_cache') and cache_key in self._match_cache:
            matched_indices = self._match_cache[cache_key]
            if matched_indices:
                return matched_indices

        device = ref_coords.device

        # detach 掉 coordinate 的梯度（仅用于匹配，不影响后续对齐梯度）
        ref_coords = ref_coords.detach()
        pred_coords = pred_coords.detach()

        # 1. 按元素类型分组
        ref_by_elem, pred_by_elem = {}, {}
        for i, a in enumerate(ref_atoms):
            ref_by_elem.setdefault(a.element, []).append(i)
        for j, a in enumerate(pred_atoms):
            pred_by_elem.setdefault(a.element, []).append(j)

        matched_indices = []
        common_elems = set(ref_by_elem.keys()) & set(pred_by_elem.keys())

        # 2. 对每种元素内部做贪心最小距离匹配
        for elem in common_elems:
            ref_idx_list = ref_by_elem[elem]
            pred_idx_list = pred_by_elem[elem]
            if not ref_idx_list or not pred_idx_list:
                continue

            # 提取坐标子集
            R = ref_coords[ref_idx_list]    # [n,3]
            P = pred_coords[pred_idx_list]  # [m,3]

            # 计算欧氏距离矩阵
            # dist_mat[i,j] = ||R[i]-P[j]||
            dist_mat = torch.cdist(R, P, p=2)  # [n, m]

            used_cols = set()
            # 对每个参考原子贪心找最近的预测原子
            for i, ridx in enumerate(ref_idx_list):
                drow = dist_mat[i]                   # [m]
                sorted_cols = torch.argsort(drow)   # 距离从小到大
                for c in sorted_cols.tolist():
                    if c not in used_cols:
                        used_cols.add(c)
                        matched_indices.append((ridx, pred_idx_list[c]))
                        break

        # 3. 如果没有匹配到任何原子对，使用全局匈牙利算法
        if not matched_indices and len(ref_coords) > 0 and len(pred_coords) > 0:
            from scipy.optimize import linear_sum_assignment
            dist_mat = torch.cdist(ref_coords, pred_coords, p=2)
            row_ind, col_ind = linear_sum_assignment(dist_mat.cpu().numpy())
            matched_indices = [(i, j) for i, j in zip(row_ind, col_ind)]
        if not hasattr(self, '_match_cache'):
            self._match_cache = {}
        if not matched_indices:
            self._match_cache[cache_key] = matched_indices

        return matched_indices

    def match_atoms(self, pdb_id, ref_coords, pred_coords, ref_atoms, pred_atoms):
        """
        匹配两个结构中的原子，即使原子数量不同
        
        参数:
            ref_coords: 参考结构坐标 (tensor, n x 3)
            pred_coords: 预测结构坐标 (tensor, m x 3)
            ref_atoms: 参考结构原子列表
            pred_atoms: 预测结构原子列表
        
        返回:
            ref_matched: 匹配后的参考结构坐标 (tensor)
            pred_matched: 匹配后的预测结构坐标 (tensor)
        """
        # 如果原子数相同，直接返回
        if len(ref_coords) == len(pred_coords):
            return ref_coords, pred_coords, ref_atoms, pred_atoms
        print(f"匹配原子中: 参考结构有 {len(ref_coords)} 个原子，预测结构有 {len(pred_coords)} 个原子")
        
        # ============ 首选：基于图的原子匹配（参考 synthetic_bench 的分链图匹配策略） ============
        graph_matched_ref_indices = []
        graph_matched_pred_indices = []
        try:
            import os
            import tempfile
            import subprocess
            import networkx as nx

            # 优先尝试 synthetic_bench 的 parse_pdb_to_graph
            parse_pdb_to_graph = None
            try:
                from synthetic_bench.scripts.utils.align import parse_pdb_to_graph as _parse
                parse_pdb_to_graph = _parse
            except Exception:
                try:
                    from multipsp.synthetic_bench.scripts.utils.align import parse_pdb_to_graph as _parse  # type: ignore
                    parse_pdb_to_graph = _parse
                except Exception:
                    try:
                        from ..synthetic_bench.scripts.utils.align import parse_pdb_to_graph as _parse  # type: ignore
                        parse_pdb_to_graph = _parse
                    except Exception:
                        parse_pdb_to_graph = None

            if parse_pdb_to_graph is not None and len(ref_atoms) > 0 and len(pred_atoms) > 0:
                # 将 Atom 列表转换为轻量 atom_info（用于写入 PDB）
                def _to_atom_info_list(atoms):
                    infos = []
                    for a in atoms:
                        chain_id = getattr(a, 'chain_id', 'A') or 'A'
                        res_id = getattr(a, 'res_id', 1)
                        try:
                            res_id = int(res_id)
                        except Exception:
                            res_id = 1
                        res_name = getattr(a, 'res_name', 'UNK') or 'UNK'
                        atom_name = getattr(a, 'atom_name', getattr(a, 'name', 'C')) or 'C'
                        element = getattr(a, 'element', (atom_name[0] if atom_name else 'C')) or (atom_name[0] if atom_name else 'C')
                        infos.append({'chain_id': str(chain_id), 'res_id': int(res_id), 'res_name': str(res_name), 'atom_name': str(atom_name), 'element': str(element)})
                    return infos

                ref_infos = _to_atom_info_list(ref_atoms)
                pred_infos = _to_atom_info_list(pred_atoms)

                # 分析各自的链并按原子数量降序排序
                def _analyze_chains(atom_info_list):
                    chain_stats = {}
                    for i, info in enumerate(atom_info_list):
                        cid = info['chain_id']
                        if cid not in chain_stats:
                            chain_stats[cid] = {'indices': [], 'atom_count': 0}
                        chain_stats[cid]['indices'].append(i)
                        chain_stats[cid]['atom_count'] += 1
                    return sorted(chain_stats.items(), key=lambda x: x[1]['atom_count'], reverse=True)

                pred_chains = _analyze_chains(pred_infos)
                ref_chains = _analyze_chains(ref_infos)

                if len(pred_chains) != len(ref_chains):
                    print(f"[GraphMatch] 链数量不匹配: pred={len(pred_chains)}, ref={len(ref_chains)}，将按较小数量截断")
                    min_chains = min(len(pred_chains), len(ref_chains))
                    pred_chains = pred_chains[:min_chains]
                    ref_chains = ref_chains[:min_chains]

                with tempfile.TemporaryDirectory() as tmp_dir:
                    def _format_pdb_atom_line(serial, atom_dict):
                        try:
                            line = "ATOM  "
                            line += f"{serial:5d}"
                            line += " "
                            line += f"{atom_dict['atom_name']:<4s}"
                            line += f"{atom_dict['res_name']:>3s}"
                            line += " "
                            line += f"{atom_dict['chain_id']:1s}"
                            line += f"{atom_dict['res_id']:4d}"
                            line += "    "
                            line += f"{atom_dict['x']:8.3f}"
                            line += f"{atom_dict['y']:8.3f}"
                            line += f"{atom_dict['z']:8.3f}"
                            line += f"{atom_dict.get('occupancy', 1.00):6.2f}"
                            line += f"{atom_dict.get('bfactor', 0.0):6.2f}"
                            line += "          "
                            line += f"{atom_dict['element']:>2s}"
                            return line
                        except Exception:
                            return ""

                    def _write_chain_pdb(atom_indices, coords_arr, atom_info_local, chain_id, out_path, serial_map):
                        with open(out_path, "w") as fh:
                            fh.write("HEADER    CHAIN_EXTRACT\n")
                            for serial, g_idx in enumerate(atom_indices, start=1):
                                x, y, z = coords_arr[g_idx]
                                info = atom_info_local[serial - 1]
                                atom_dict = {
                                    'atom_name': info['atom_name'],
                                    'res_name': info['res_name'],
                                    'chain_id': chain_id,
                                    'res_id': info['res_id'],
                                    'x': float(x),
                                    'y': float(y),
                                    'z': float(z),
                                    'occupancy': 1.00,
                                    'bfactor': 0.0,
                                    'element': (info['atom_name'][0] if info['atom_name'] else 'C')
                                }
                                fh.write(_format_pdb_atom_line(serial, atom_dict) + "\n")
                                serial_map[serial] = g_idx
                            fh.write("END\n")

                    ref_np = ref_coords.detach().cpu().numpy()
                    pred_np = pred_coords.detach().cpu().numpy()

                    for (pred_chain_id, pred_chain_info), (ref_chain_id, ref_chain_info) in zip(pred_chains, ref_chains):
                        pred_chain_indices = pred_chain_info['indices']
                        ref_chain_indices = ref_chain_info['indices']

                        pred_chain_infos = [pred_infos[i] for i in pred_chain_indices]
                        ref_chain_infos  = [ref_infos[i] for i in ref_chain_indices]

                        pred_pdb_path = os.path.join(tmp_dir, f"pred_{pred_chain_id}.pdb")
                        ref_pdb_path  = os.path.join(tmp_dir, f"ref_{ref_chain_id}.pdb")

                        serial2global_pred = {}
                        serial2global_ref  = {}

                        _write_chain_pdb(pred_chain_indices, pred_np, pred_chain_infos, pred_chain_id, pred_pdb_path, serial2global_pred)
                        _write_chain_pdb(ref_chain_indices,  ref_np,  ref_chain_infos,  ref_chain_id,  ref_pdb_path,  serial2global_ref)

                        # 使用 openbabel 尝试补全 CONECT（可选）
                        for _p in (pred_pdb_path, ref_pdb_path):
                            try:
                                subprocess.run(["obabel", _p, "-O", _p], capture_output=True, text=True, timeout=60)
                            except FileNotFoundError:
                                pass
                            except Exception as _ge:
                                print(f"[GraphMatch] obabel 处理失败: {_ge}")

                        graph_pred, _ = parse_pdb_to_graph(pred_pdb_path, pred_chain_id)
                        graph_ref, _  = parse_pdb_to_graph(ref_pdb_path,  ref_chain_id)

                        if len(graph_pred) > 0 and len(graph_ref) > 0:
                            def _node_match(n1, n2):
                                return n1.get("type") == n2.get("type")

                            # 令较大的图作为 graph_a
                            if len(graph_pred) >= len(graph_ref):
                                graph_a, graph_b = graph_pred, graph_ref
                                map_pred_to_ref = True
                            else:
                                graph_a, graph_b = graph_ref, graph_pred
                                map_pred_to_ref = False

                            matcher = nx.algorithms.isomorphism.GraphMatcher(graph_a, graph_b, node_match=_node_match)
                            mapping = None
                            try:
                                mapping = next(matcher.subgraph_isomorphisms_iter())
                            except StopIteration:
                                mapping = None
                            except Exception as _me:
                                print(f"[GraphMatch] GraphMatcher 失败: {_me}")
                                mapping = None

                            if mapping:
                                for node_a, node_b in mapping.items():
                                    pred_serial, ref_serial = (node_a, node_b) if map_pred_to_ref else (node_b, node_a)
                                    if pred_serial in serial2global_pred and ref_serial in serial2global_ref:
                                        graph_matched_pred_indices.append(serial2global_pred[pred_serial])
                                        graph_matched_ref_indices.append(serial2global_ref[ref_serial])
                            else:
                                # 链级图匹配失败，回退到链内 (res_id, atom_name) 键匹配，尽量收集匹配点
                                def _key(info):
                                    return (info['res_id'], info['atom_name'])
                                pred_map = { _key(info): idx for idx, info in enumerate(pred_chain_infos) }
                                ref_map  = { _key(info): idx for idx, info in enumerate(ref_chain_infos) }
                                common_keys = set(pred_map.keys()) & set(ref_map.keys())
                                for k in sorted(common_keys):
                                    p_local = pred_map[k]
                                    r_local = ref_map[k]
                                    graph_matched_pred_indices.append(pred_chain_indices[p_local])
                                    graph_matched_ref_indices.append(ref_chain_indices[r_local])

            else:
                print("[GraphMatch] 未找到 parse_pdb_to_graph，跳过图匹配")
        except Exception as e:
            print(f"[GraphMatch] 图匹配流程异常: {e}")

        # 若图匹配得到有效匹配对，直接返回
        if len(graph_matched_ref_indices) > 0 and len(graph_matched_pred_indices) > 0:
            ref_ids, pred_ids = graph_matched_ref_indices, graph_matched_pred_indices
            ref_matched_coords  = ref_coords[ref_ids]
            pred_matched_coords = pred_coords[pred_ids]
            ref_matched_atoms   = [ref_atoms[i] for i in ref_ids]
            pred_matched_atoms  = [pred_atoms[j] for j in pred_ids]
            return ref_ids, pred_ids, ref_matched_coords, pred_matched_coords, ref_matched_atoms, pred_matched_atoms
        
        # ============ 回退：沿用现有匹配逻辑 ============
        matched_indices = self.get_match_indices(pdb_id, ref_coords, pred_coords, ref_atoms, pred_atoms)
        
        # 3. 构造输出 Tensor / Atom 列表
        if not matched_indices:
            device = ref_coords.device
            return ([], [], torch.empty((0,3), device=device),
                    torch.empty((0,3), device=device),
                    [], [])
            
        ref_ids, pred_ids = zip(*matched_indices)
        ref_ids, pred_ids = list(ref_ids), list(pred_ids)

        ref_matched_coords  = ref_coords[ref_ids]
        pred_matched_coords = pred_coords[pred_ids]
        ref_matched_atoms   = [ref_atoms[i] for i in ref_ids]
        pred_matched_atoms  = [pred_atoms[j] for j in pred_ids]

        return ref_ids, pred_ids, ref_matched_coords, pred_matched_coords, ref_matched_atoms, pred_matched_atoms


    # 计算两个化学环境的相似性
    def _calculate_environment_similarity(self, env1, env2):
        """计算两个化学环境的相似性得分"""
        if not env1 or not env2:
            return 100.0  # 惩罚没有环境的情况
            
        # 基于元素类型和距离的匹配评分
        score = 0.0
        
        # 统计环境中的元素
        elements1 = {}
        for element, dist in env1:
            if element not in elements1:
                elements1[element] = []
            elements1[element].append(dist)
            
        elements2 = {}
        for element, dist in env2:
            if element not in elements2:
                elements2[element] = []
            elements2[element].append(dist)
        
        # 比较共有元素的分布
        for element in set(elements1.keys()) & set(elements2.keys()):
            dists1 = sorted(elements1[element])
            dists2 = sorted(elements2[element])
            
            # 计算距离分布的差异
            min_len = min(len(dists1), len(dists2))
            for i in range(min_len):
                score += abs(dists1[i] - dists2[i])
                
            # 惩罚数量不匹配
            score += abs(len(dists1) - len(dists2)) * 2.0
        
        # 惩罚不共有的元素
        missing_elements = len(set(elements1.keys()) ^ set(elements2.keys()))
        score += missing_elements * 5.0
        
        return score

    def chunked_cdist(self, a, b, chunk_size=1024):
        """
        分块计算欧几里得距离矩阵，避免内存溢出
        
        参数:
            a: 第一组坐标 (tensor, n x 3)
            b: 第二组坐标 (tensor, m x 3)
            chunk_size: 每个块的最大尺寸
            
        返回:
            dist_matrix: 距离矩阵 (tensor, n x m)
        """
        n_a = a.size(0)
        n_b = b.size(0)
        device = a.device
        
        # 对于小规模数据，直接使用torch.cdist
        if n_a * n_b <= chunk_size * chunk_size:
            return torch.cdist(a, b)
        
        # 对于大规模数据，分块计算
        dist_matrix = torch.zeros((n_a, n_b), device=device)
        for i in range(0, n_a, chunk_size):
            end_i = min(i + chunk_size, n_a)
            a_chunk = a[i:end_i]
            
            for j in range(0, n_b, chunk_size):
                end_j = min(j + chunk_size, n_b)
                b_chunk = b[j:end_j]
                dist_chunk = torch.cdist(a_chunk, b_chunk)
                dist_matrix[i:end_i, j:end_j] = dist_chunk
        
        return dist_matrix

    def superimpose_and_calculate_rmsd(self, pdb_id, ref_coords, pred_coords, ref_atoms, pred_atoms, ref_mask=None, pred_mask=None):
        """优化版的配准和RMSD计算"""
        try:
            # Initial checks for NaNs/Infs in raw inputs
            if torch.isnan(ref_coords).any(): print(f"RMSD PDB {pdb_id}: Initial ref_coords has NaN")
            if torch.isinf(ref_coords).any(): print(f"RMSD PDB {pdb_id}: Initial ref_coords has Inf")
            if torch.isnan(pred_coords).any(): print(f"RMSD PDB {pdb_id}: Initial pred_coords has NaN")
            if torch.isinf(pred_coords).any(): print(f"RMSD PDB {pdb_id}: Initial pred_coords has Inf")

            if not isinstance(ref_coords, torch.Tensor) or not isinstance(pred_coords, torch.Tensor):
                print(f"RMSD PDB {pdb_id}: Error - coordinates are not torch.Tensor type.")
                raise TypeError("坐标必须是torch.Tensor类型")
            
            if len(ref_coords.shape) > 2 or len(pred_coords.shape) > 2:
                print(f"RMSD PDB {pdb_id}: Error - superimpose_and_calculate_rmsd expects non-batched input. Got shapes {ref_coords.shape}, {pred_coords.shape}")
                raise ValueError("superimpose_and_calculate_rmsd 只接受无批次维度的输入")
            
            device = ref_coords.device
            epsilon = 1e-10 # For sqrt and division stability, not for general float comparisons
            
            # Make copies for modification
            _ref_coords = ref_coords.clone()
            _pred_coords = pred_coords.clone()
            _ref_atoms = list(ref_atoms) # shallow copy
            _pred_atoms = list(pred_atoms)

            # 初始化ref_ids和pred_ids，防止未定义
            ref_ids, pred_ids = [], []

            # 应用掩码和过滤无效值
            if ref_mask is not None:
                if len(ref_mask) != len(_ref_coords):
                    print(f"警告 (PDB {pdb_id}): 参考掩码长度 {len(ref_mask)} 与坐标数量 {len(_ref_coords)} 不匹配. Truncating/Padding.")
                    if len(ref_mask) > len(_ref_coords):
                        ref_mask = ref_mask[:len(_ref_coords)]
                    else:
                        padding = torch.zeros(len(_ref_coords) - len(ref_mask), dtype=torch.bool, device=ref_mask.device)
                        ref_mask = torch.cat([ref_mask, padding])

                valid_ref = ref_mask.bool() & (~torch.isnan(_ref_coords).any(dim=1)) & (~torch.isinf(_ref_coords).any(dim=1))
                _ref_coords = _ref_coords[valid_ref]
                _ref_atoms = [atom for i, atom in enumerate(_ref_atoms) if valid_ref[i].item()]
            else:
                valid_ref = (~torch.isnan(_ref_coords).any(dim=1)) & (~torch.isinf(_ref_coords).any(dim=1))
                _ref_coords = _ref_coords[valid_ref]
                _ref_atoms = [atom for i, atom in enumerate(_ref_atoms) if valid_ref[i].item()]
            
            if pred_mask is not None:
                if len(pred_mask) != len(_pred_coords):
                    print(f"警告 (PDB {pdb_id}): 预测掩码长度 {len(pred_mask)} 与坐标数量 {len(_pred_coords)} 不匹配. Truncating/Padding.")
                    if len(pred_mask) > len(_pred_coords):
                        pred_mask = pred_mask[:len(_pred_coords)]
                    else:
                        padding = torch.zeros(len(_pred_coords) - len(pred_mask), dtype=torch.bool, device=pred_mask.device)
                        pred_mask = torch.cat([pred_mask, padding])

                valid_pred = pred_mask.bool() & (~torch.isnan(_pred_coords).any(dim=1)) & (~torch.isinf(_pred_coords).any(dim=1))
                _pred_coords = _pred_coords[valid_pred]
                _pred_atoms = [atom for i, atom in enumerate(_pred_atoms) if valid_pred[i].item()]
            else:
                valid_pred = (~torch.isnan(_pred_coords).any(dim=1)) & (~torch.isinf(_pred_coords).any(dim=1))
                _pred_coords = _pred_coords[valid_pred]
                _pred_atoms = [atom for i, atom in enumerate(_pred_atoms) if valid_pred[i].item()]

            print(f"RMSD PDB {pdb_id}: Valid ref atoms after filtering: {len(_ref_coords)}, Valid pred atoms: {len(_pred_coords)}")
            if torch.isnan(_ref_coords).any(): print(f"RMSD PDB {pdb_id}: Filtered ref_coords has NaN")
            if torch.isinf(_ref_coords).any(): print(f"RMSD PDB {pdb_id}: Filtered ref_coords has Inf")
            if torch.isnan(_pred_coords).any(): print(f"RMSD PDB {pdb_id}: Filtered pred_coords has NaN")
            if torch.isinf(_pred_coords).any(): print(f"RMSD PDB {pdb_id}: Filtered pred_coords has Inf")

            # 处理空输入或有效坐标太少
            if len(_ref_coords) < 3 or len(_pred_coords) < 3:
                print(f"警告 (PDB {pdb_id}): 有效原子数量不足 (参考: {len(_ref_coords)}, 预测: {len(_pred_coords)}). Returning default high RMSD.")
                return torch.tensor(10.0, device=device, dtype=ref_coords.dtype), torch.eye(3, device=device, dtype=ref_coords.dtype), torch.zeros(3, device=device, dtype=ref_coords.dtype), [], [], ref_ids, pred_ids
            
            # 原子匹配 (如果需要)
            R_kabsch, t_kabsch = torch.eye(3, device=device, dtype=ref_coords.dtype), torch.zeros(3, device=device, dtype=ref_coords.dtype)
            if len(_ref_coords) != len(_pred_coords):
                # print(f"警告 (PDB {pdb_id}): 参考结构有 {len(_ref_coords)} 个有效原子，预测结构有 {len(_pred_coords)} 个有效原子. Matching atoms...")
                ref_ids, pred_ids, ref_coords_matched, pred_coords_matched, _, _ = self.match_atoms(pdb_id, _ref_coords, _pred_coords, _ref_atoms, _pred_atoms)
                
                print(f"RMSD PDB {pdb_id}: Matched atoms for Kabsch: {len(ref_coords_matched)}")
                if len(ref_coords_matched) < 3:
                    print(f"错误 (PDB {pdb_id}): 匹配的原子对不足3对 ({len(ref_coords_matched)}). Returning default high RMSD.")
                    return torch.tensor(10.0, device=device, dtype=ref_coords.dtype), R_kabsch, t_kabsch, [], [], ref_ids, pred_ids
                
                # 使用匹配的坐标子集计算变换
                R_kabsch, t_kabsch = self.kabsch_algorithm(ref_coords_matched, pred_coords_matched)
            else:
                # 原子数量相同，直接计算
                R_kabsch, t_kabsch = self.kabsch_algorithm(_ref_coords, _pred_coords)

            if torch.isnan(R_kabsch).any() or torch.isinf(R_kabsch).any():
                print(f"RMSD PDB {pdb_id}: Kabsch R contains NaN/Inf. R: {R_kabsch}. Using identity.")
                R_kabsch = torch.eye(3, device=device, dtype=ref_coords.dtype)
            if torch.isnan(t_kabsch).any() or torch.isinf(t_kabsch).any():
                print(f"RMSD PDB {pdb_id}: Kabsch t contains NaN/Inf. t: {t_kabsch}. Using zeros.")
                t_kabsch = torch.zeros(3, device=device, dtype=ref_coords.dtype)
            
            # 向量化应用变换 to original _pred_coords (before matching for Kabsch)
            aligned_pred_coords = self.apply_transformation(_pred_coords, R_kabsch, t_kabsch)
            if torch.isnan(aligned_pred_coords).any() or torch.isinf(aligned_pred_coords).any():
                print(f"RMSD PDB {pdb_id}: aligned_pred_coords contains NaN/Inf.")
            
            # RMSD calculation part
            # If numbers of atoms were different, we need to match again for RMSD calculation
            # between the original _ref_coords and the transformed _pred_coords.
            # Or, more robustly, use the initially matched sets if they were used for Kabsch.
            
            final_ref_coords_for_rmsd = _ref_coords
            final_aligned_pred_coords_for_rmsd = aligned_pred_coords

            # If Kabsch was on matched subset, RMSD should be on that same subset's original reference and transformed prediction
            if 'ref_coords_matched' in locals() and len(ref_coords_matched) > 0 : # Check if matching was done
                 # Align the *subset* of pred_coords that was matched
                 # This is tricky. Kabsch used (ref_coords_matched, pred_coords_matched).
                 # So aligned_pred_coords are based on transforming the *entire* _pred_coords.
                 # For RMSD, we need to compare the *corresponding* atoms.
                 # The safest is to match _ref_coords and aligned_pred_coords if their sizes differ.
                 if len(_ref_coords) != len(aligned_pred_coords):
                     # print(f"RMSD PDB {pdb_id}: Rematching for RMSD calculation as _ref_coords ({len(_ref_coords)}) and aligned_pred_coords ({len(aligned_pred_coords)}) differ.")
                     ref_ids, pred_ids, final_ref_coords_for_rmsd, final_aligned_pred_coords_for_rmsd, _, _ = self.match_atoms(pdb_id, _ref_coords, aligned_pred_coords, _ref_atoms, _pred_atoms) # Match _ref_coords with the *transformed* full _pred_coords
                     print(f"RMSD PDB {pdb_id}: Matched atoms for final RMSD: {len(final_ref_coords_for_rmsd)}")
                     if len(final_ref_coords_for_rmsd) == 0:
                         print(f"警告 (PDB {pdb_id}): No atoms matched for final RMSD. Returning high RMSD.")
                         return torch.tensor(10.0, device=device, dtype=ref_coords.dtype), R_kabsch, t_kabsch, ref_ids, pred_ids, ref_ids, pred_ids
                 # If they were same length originally, but Kabsch was on a subset (should not happen with current logic), this branch is complex.
                 # Current logic: if lengths differ, match for Kabsch. Then transform full pred. Then if lengths still differ for RMSD, rematch.

            num_atoms_for_rmsd = len(final_ref_coords_for_rmsd)
            if num_atoms_for_rmsd == 0: # Should be caught by earlier checks if match_atoms returns empty
                print(f"警告 (PDB {pdb_id}): num_atoms_for_rmsd is 0. Returning high RMSD.")
                return torch.tensor(10.0, device=device, dtype=ref_coords.dtype), R_kabsch, t_kabsch, ref_ids, pred_ids, ref_ids, pred_ids

            diff_for_rmsd = final_ref_coords_for_rmsd - final_aligned_pred_coords_for_rmsd
            if torch.isnan(diff_for_rmsd).any() or torch.isinf(diff_for_rmsd).any():
                print(f"RMSD PDB {pdb_id}: Difference for RMSD calc contains NaN/Inf.")
            
            # Using L1-like loss as in the existing code for `rmsd` variable
            rmsd_val = torch.mean(torch.abs(diff_for_rmsd) + epsilon) # Mean of abs differences per coordinate
            if torch.isnan(rmsd_val) or torch.isinf(rmsd_val):
                print(f"RMSD PDB {pdb_id}: Raw rmsd_val is NaN/Inf: {rmsd_val}. Num atoms: {num_atoms_for_rmsd}")
                # Fallback for safety
                sum_sq_diff = torch.sum(diff_for_rmsd**2, dim=-1)
                mean_sq_diff = torch.mean(sum_sq_diff)
                if torch.isnan(mean_sq_diff) or torch.isinf(mean_sq_diff) or mean_sq_diff < 0:
                     print(f"RMSD PDB {pdb_id}: mean_sq_diff for sqrt is invalid: {mean_sq_diff}. Using default high RMSD.")
                     rmsd_val = torch.tensor(10.0, device=device, dtype=ref_coords.dtype)
                else:
                     rmsd_val = torch.sqrt(mean_sq_diff + epsilon) # Classic RMSD as fallback if L1 version fails
                     print(f"RMSD PDB {pdb_id}: Fell back to sqrt-based RMSD: {rmsd_val.item()}")

            # Normalization
            log_term_input = torch.tensor(num_atoms_for_rmsd, dtype=torch.float, device=device) + 1e-10 # add eps before log
            if log_term_input <= 0:
                print(f"RMSD PDB {pdb_id}: Invalid input for log10 in normalization: {log_term_input}. Using num_atoms_for_rmsd = 1 for log.")
                log_term_input = torch.tensor(1.0, dtype=torch.float, device=device) # Avoid log(<=0)
            
            log_val = torch.log10(log_term_input)
            if torch.isnan(log_val) or torch.isinf(log_val):
                print(f"RMSD PDB {pdb_id}: log_val in normalization is NaN/Inf. Input to log: {log_term_input}. Using log_val = 0.")
                log_val = torch.tensor(0.0, device=device, dtype=ref_coords.dtype)

            denominator = 1.0 + 0.1 * log_val
            if denominator == 0: # Avoid division by zero
                print(f"RMSD PDB {pdb_id}: Denominator in normalization is zero. num_atoms: {num_atoms_for_rmsd}. Using denominator = 1.")
                normalized_rmsd = rmsd_val
            elif torch.isnan(denominator) or torch.isinf(denominator):
                print(f"RMSD PDB {pdb_id}: Denominator in normalization is NaN/Inf: {denominator}. Using denominator = 1.")
                normalized_rmsd = rmsd_val
            else:
                normalized_rmsd = rmsd_val / denominator
            
            if torch.isnan(normalized_rmsd) or torch.isinf(normalized_rmsd):
                print(f"警告 (PDB {pdb_id}): Normalized_rmsd 对齐误差值为 {normalized_rmsd.item()}，raw rmsd_val: {rmsd_val.item()}, denom: {denominator.item()}, num_atoms: {num_atoms_for_rmsd}. 替换为默认值 10.0")
                normalized_rmsd = torch.tensor(10.0, device=device, dtype=ref_coords.dtype)
                
            normalized_rmsd = torch.clamp(normalized_rmsd, min=0.0, max=10.0) # Ensure non-negative and capped
            
            print(f"RMSD PDB {pdb_id}: Final normalized RMSD: {normalized_rmsd.item()}, R_kabsch_det: {torch.linalg.det(R_kabsch).item() if R_kabsch is not None else 'None'}, t_kabsch: {t_kabsch if t_kabsch is not None else 'None'}")
            return normalized_rmsd, R_kabsch, t_kabsch, \
                   final_ref_coords_for_rmsd.detach(), \
                   final_aligned_pred_coords_for_rmsd.detach(), \
                   ref_ids, pred_ids
            
        except Exception as e:
            print(f"RMSD计算错误 (PDB {pdb_id}): {e}")
            import traceback
            traceback.print_exc()
            # Ensure dtype matches expected output
            dummy_device = ref_coords.device if isinstance(ref_coords, torch.Tensor) else torch.device('cpu')
            dummy_dtype = ref_coords.dtype if isinstance(ref_coords, torch.Tensor) else torch.float32
            # 返回空ref_ids和pred_ids，防止未定义
            return torch.tensor(10.0, device=dummy_device, dtype=dummy_dtype), torch.eye(3, device=dummy_device, dtype=dummy_dtype), torch.zeros(3, device=dummy_device, dtype=dummy_dtype), torch.empty(0, device=dummy_device, dtype=dummy_dtype), torch.empty(0, device=dummy_device, dtype=dummy_dtype), [], []

    # def align_loss(self, pdb_id, pred_coordinate, true_coordinate, pred_atoms, true_atoms, pred_mask=None, true_mask=None, seq_is_nucleotide=None, smiles_is_nucleotide=None, seq_is_polymer=None, smiles_is_polymer=None):
    #     """
    #     计算对齐损失，正确处理批次和坐标掩码
    #     参数:
    #         pdb_id: 蛋白质ID
    #         pred_coordinate: 预测结构坐标 (tensor, [batch_size, n_atoms, 3] 或 [n_atoms, 3])
    #         true_coordinate: 参考结构坐标 (tensor, [batch_size, n_atoms, 3] 或 [n_atoms, 3])
    #         pred_atoms: 预测结构原子列表
    #         true_atoms: 参考结构原子列表
    #         pred_mask: 预测坐标掩码 (可选)
    #         true_mask: 参考坐标掩码 (可选)
            
    #     返回:
    #         rmsd: 对齐损失 (tensor, scalar)
    #     """
    #     # 检查维度并处理
    #     pred_has_batch = len(pred_coordinate.shape) == 3
    #     true_has_batch = len(true_coordinate.shape) == 3
    #     pred_mask_has_batch = pred_mask is not None and len(pred_mask.shape) == 2
    #     true_mask_has_batch = true_mask is not None and len(true_mask.shape) == 2
        
    #     # 调整掩码维度与坐标维度匹配
    #     if pred_has_batch and pred_mask is not None and not pred_mask_has_batch:
    #         pred_mask = pred_mask.unsqueeze(0).expand(pred_coordinate.shape[0], -1)
        
    #     if true_has_batch and true_mask is not None and not true_mask_has_batch:
    #         true_mask = true_mask.unsqueeze(0).expand(true_coordinate.shape[0], -1)
        
    #     # 如果预测坐标有批次维度但真实坐标没有，则复制真实坐标到批次维度
    #     if pred_has_batch and not true_has_batch:
    #         batch_size = pred_coordinate.shape[0]
    #         true_coordinate = true_coordinate.unsqueeze(0).expand(batch_size, -1, -1)
    #         if true_mask is not None:
    #             true_mask = true_mask.unsqueeze(0).expand(batch_size, -1)
        
    #     # 如果真实坐标有批次维度但预测坐标没有，则复制预测坐标到批次维度（不太可能发生）
    #     elif true_has_batch and not pred_has_batch:
    #         batch_size = true_coordinate.shape[0]
    #         pred_coordinate = pred_coordinate.unsqueeze(0).expand(batch_size, -1, -1)
    #         if pred_mask is not None:
    #             pred_mask = pred_mask.unsqueeze(0).expand(batch_size, -1)
        
    #     # 如果都没有批次维度，直接计算
    #     if not pred_has_batch and not true_has_batch:
    #         rmsd, R, t, ref_aligned, pred_aligned = self.superimpose_and_calculate_rmsd(
    #             pdb_id, true_coordinate, pred_coordinate, true_atoms, pred_atoms, 
    #             ref_mask=true_mask, pred_mask=pred_mask
    #         )
    #         # 在单样本情况下也添加NaN检查
    #         if torch.isnan(rmsd).any() or torch.isinf(rmsd).any():
    #             current_rank = dist.get_rank() if dist.is_initialized() else "N/A_no_dist"
    #             print(f"CRITICAL align_loss (single sample): rmsd is NaN/Inf for PDB {pdb_id} on rank {current_rank}. Value: {rmsd.item()}")
    #             # rmsd = torch.tensor(10.0, device=rmsd.device, dtype=rmsd.dtype) # 可选：替换为默认值
    #         return rmsd
        
    #     # 批量情况 - 处理每个样本
    #     device = pred_coordinate.device
    #     batch_size = pred_coordinate.shape[0]
    #     valid_count = 0
    #     total_rmsd = torch.tensor(0.0, device=device, dtype=pred_coordinate.dtype) # 确保dtype匹配
        
    #     for i in range(batch_size):
    #         curr_true_mask = None if true_mask is None else true_mask[i]
    #         curr_pred_mask = None if pred_mask is None else pred_mask[i]
            
    #         sample_rmsd, _, _, ref_aligned, pred_aligned = self.superimpose_and_calculate_rmsd(
    #             pdb_id, true_coordinate[i], pred_coordinate[i], true_atoms, pred_atoms,
    #             ref_mask=curr_true_mask, pred_mask=curr_pred_mask
    #         )
            
    #         # ---- 新增的NaN/Inf检查 ----
    #         if torch.isnan(sample_rmsd).any() or torch.isinf(sample_rmsd).any():
    #             current_rank = dist.get_rank() if dist.is_initialized() else "N/A_no_dist"
    #             print(f"CRITICAL align_loss (batch loop): sample_rmsd is NaN/Inf for PDB {pdb_id}, sample_idx {i} on rank {current_rank}. Value: {sample_rmsd.item()}")
    #             # 为了尝试继续运行并收集更多信息，可以将 NaN 替换为一个高值，但要注意这可能掩盖根本原因
    #             # sample_rmsd = torch.tensor(10.0, device=sample_rmsd.device, dtype=sample_rmsd.dtype)
    #         # ---- 结束新增检查 ----
            
    #         # 检查是否为有效值 (已有的逻辑)
    #         if not torch.isnan(sample_rmsd) and not torch.isinf(sample_rmsd):
    #             total_rmsd += sample_rmsd
    #             valid_count += 1
    #         else: # 如果sample_rmsd是NaN/Inf，也在这里打印信息
    #             current_rank = dist.get_rank() if dist.is_initialized() else "N/A_no_dist"
    #             print(f"INFO align_loss (batch loop): sample_rmsd was NaN/Inf for PDB {pdb_id}, sample_idx {i} on rank {current_rank}, NOT added to total_rmsd. Value: {sample_rmsd.item()}")

    #     # 计算平均RMSD，避免除零
    #     if valid_count > 0:
    #         avg_rmsd = total_rmsd / valid_count
    #     else:
    #         current_rank = dist.get_rank() if dist.is_initialized() else "N/A_no_dist"
    #         print(f"警告 align_loss: 批次中没有有效的RMSD值 for PDB {pdb_id} on rank {current_rank}. total_rmsd: {total_rmsd.item()}, valid_count: {valid_count}. Returning default 10.0")
    #         avg_rmsd = torch.tensor(10.0, device=device, dtype=pred_coordinate.dtype) # 确保dtype匹配
        
    #     # ---- 新增的NaN/Inf检查 ----
    #     if torch.isnan(avg_rmsd).any() or torch.isinf(avg_rmsd).any():
    #         current_rank = dist.get_rank() if dist.is_initialized() else "N/A_no_dist"
    #         print(f"CRITICAL align_loss: avg_rmsd is NaN/Inf for PDB {pdb_id} on rank {current_rank} BEFORE RETURN. Value: {avg_rmsd.item()}. valid_count: {valid_count}, total_rmsd: {total_rmsd.item()}")
            # avg_rmsd = torch.tensor(10.0, device=avg_rmsd.device, dtype=avg_rmsd.dtype) # 可选：替换为默认值
        # ---- 结束新增检查 ----
        
        # -------------------------
        # # 0. 取坐标及掩码
        # seq_pred_coordinate   = seq_pred_dict.get("coordinate")   # [N_atom, 3]
        # seq_true_coordinate   = seq_label_dict.get("coordinate")  # [N_atom, 3]
        # seq_coordinate_mask   = seq_label_dict.get("coordinate_mask")  # [N_atom]

        # # 若缺关键输入直接抛错，便于调用方补充
        # if seq_pred_coordinate is None or seq_true_coordinate is None:
        #     raise ValueError("[align_geometric_loss] coordinate 信息缺失，无法计算对齐损失！")

        # # -------------------------
        # # 1. 原子数组（用于匹配）
        # seq_atoms = self.get_atom_from_atom_array(seq_atom_array) if seq_atom_array is not None else []

        # # 2. 刚体超叠 & 匹配
        # rmsd, R, t, ref_aligned, pred_aligned = self.superimpose_and_calculate_rmsd(
        #     pdb_id,
        #     pred_coordinate, true_coordinate
        #     true_atoms, pred_atoms,
        #     ref_mask=seq_coordinate_mask,
        #     pred_mask=seq_coordinate_mask,
        # )

    def align_geometric_loss(self, pdb_id, pred_coordinate, true_coordinate, pred_atoms, true_atoms, 
                             pred_mask=None, true_mask=None, true_is_nucleotide=None, pred_is_nucleotide=None, true_is_polymer=None, pred_is_polymer=None):
                             #seq_pred_dict, smiles_pred_dict, seq_label_dict, smiles_label_dict, seq_atom_array, smiles_atom_array):
        """
        计算对齐几何损失（严格按照各损失原生接口调用，无任何简化）
        TODO: 标注的占位张量 / 掩码 / logits 需由上游调用方在正式使用时提供真实数据。
        参数:
            pdb_id: str
            seq_pred_dict / smiles_pred_dict: Dict[str, Tensor]
            seq_label_dict / smiles_label_dict: Dict[str, Tensor]
            seq_atom_array / smiles_atom_array: AtomArray
        返回:
            align_geometric_loss: Tensor (scalar)
        """
        # 与 align_loss 对齐的批次与掩码/类别标记处理逻辑
        pred_has_batch = len(pred_coordinate.shape) == 3
        true_has_batch = len(true_coordinate.shape) == 3
        pred_mask_has_batch = pred_mask is not None and len(pred_mask.shape) == 2
        true_mask_has_batch = true_mask is not None and len(true_mask.shape) == 2
        true_is_nucleotide_has_batch = true_is_nucleotide is not None and len(true_is_nucleotide.shape) == 2
        pred_is_nucleotide_has_batch = pred_is_nucleotide is not None and len(pred_is_nucleotide.shape) == 2
        true_is_polymer_has_batch = true_is_polymer is not None and len(true_is_polymer.shape) == 2
        pred_is_polymer_has_batch = pred_is_polymer is not None and len(pred_is_polymer.shape) == 2

        # 掩码/标记与坐标批次维度对齐（广播）
        if pred_has_batch and pred_mask is not None and not pred_mask_has_batch:
            pred_mask = pred_mask.unsqueeze(0).expand(pred_coordinate.shape[0], -1)
        if true_has_batch and true_mask is not None and not true_mask_has_batch:
            true_mask = true_mask.unsqueeze(0).expand(true_coordinate.shape[0], -1)
        if pred_has_batch and pred_is_nucleotide is not None and not pred_is_nucleotide_has_batch:
            pred_is_nucleotide = pred_is_nucleotide.unsqueeze(0).expand(pred_coordinate.shape[0], -1)
        if true_has_batch and true_is_nucleotide is not None and not true_is_nucleotide_has_batch:
            true_is_nucleotide = true_is_nucleotide.unsqueeze(0).expand(true_coordinate.shape[0], -1)
        if pred_has_batch and pred_is_polymer is not None and not pred_is_polymer_has_batch:
            pred_is_polymer = pred_is_polymer.unsqueeze(0).expand(pred_coordinate.shape[0], -1)
        if true_has_batch and true_is_polymer is not None and not true_is_polymer_has_batch:
            true_is_polymer = true_is_polymer.unsqueeze(0).expand(true_coordinate.shape[0], -1)

        # 坐标批次维度对齐
        if pred_has_batch and not true_has_batch:
            batch_size = pred_coordinate.shape[0]
            true_coordinate = true_coordinate.unsqueeze(0).expand(batch_size, -1, -1)
            if true_mask is not None:
                true_mask = true_mask.unsqueeze(0).expand(batch_size, -1)
            if true_is_nucleotide is not None and len(true_is_nucleotide.shape) == 1:
                true_is_nucleotide = true_is_nucleotide.unsqueeze(0).expand(batch_size, -1)
            if true_is_polymer is not None and len(true_is_polymer.shape) == 1:
                true_is_polymer = true_is_polymer.unsqueeze(0).expand(batch_size, -1)
        elif true_has_batch and not pred_has_batch:
            batch_size = true_coordinate.shape[0]
            pred_coordinate = pred_coordinate.unsqueeze(0).expand(batch_size, -1, -1)
            if pred_mask is not None:
                pred_mask = pred_mask.unsqueeze(0).expand(batch_size, -1)
            if pred_is_nucleotide is not None and len(pred_is_nucleotide.shape) == 1:
                pred_is_nucleotide = pred_is_nucleotide.unsqueeze(0).expand(batch_size, -1)
            if pred_is_polymer is not None and len(pred_is_polymer.shape) == 1:
                pred_is_polymer = pred_is_polymer.unsqueeze(0).expand(batch_size, -1)

        # ----------- 重要说明 -----------
        # 经过上述批次维度对齐（广播）后，mask 变量（如 true_mask, pred_mask）如果原本是一维（[N_atom]），
        # 会变成二维（[batch, N_atom]）。此时如果后续还想用一维的 ref_ids/pred_ids（如 [idx1, idx2, ...]）
        # 直接对 mask 做索引（如 true_mask[ref_ids]），会导致 shape 不匹配或语义错误。
        # 正确做法是：如果 mask 已经是二维（[batch, N_atom]），
        # 应该用 mask[:, ref_ids] 这样带 batch 维度的索引，或者对每个 batch 单独索引。
        # 如果后续代码还需要用一维索引，请务必检查 mask 的 shape，并做相应处理。
        # --------------------------------

        # 工具函数：根据mask的shape和ref_ids/pred_ids做对齐索引
        def _index_mask(mask, ids):
            if mask is None:
                return None
            if len(mask.shape) == 1:
                # [N_atom]
                return mask[ids]
            elif len(mask.shape) == 2:
                # [batch, N_atom]，此时通常在batch模式下，单样本时取[0]
                # 这里假设mask已经被切分到当前batch（如curr_true_mask），所以shape为[N_atom]
                if mask.shape[0] == len(ids):
                    # 罕见情况，mask本身就是[N, N]，不常见
                    return mask[range(mask.shape[0]), ids]
                else:
                    return mask[ids]
            elif len(mask.shape) == 3:
                # [B, N, ...]，不常见
                # 这里不做特殊处理，直接取batch=0
                return mask[0, ids]
            else:
                raise ValueError(f"Unsupported mask shape: {mask.shape}")

        # 单样本路径
        if not pred_has_batch and not true_has_batch:
            normalized_rmsd, R_kabsch, t_kabsch, ref_aligned, pred_aligned, ref_ids, pred_ids = self.superimpose_and_calculate_rmsd(
                pdb_id,
                ref_coords=true_coordinate,
                pred_coords=pred_coordinate,
                ref_atoms=true_atoms,
                pred_atoms=pred_atoms,
                ref_mask=true_mask,
                pred_mask=pred_mask,
            )

            # 形状与有效性防护，避免 cdist 遇到 1D/空张量
            if isinstance(ref_aligned, torch.Tensor) and ref_aligned.dim() == 1:
                ref_aligned = ref_aligned.unsqueeze(0)
            if isinstance(pred_aligned, torch.Tensor) and pred_aligned.dim() == 1:
                pred_aligned = pred_aligned.unsqueeze(0)
            if (not isinstance(ref_aligned, torch.Tensor)) or (not isinstance(pred_aligned, torch.Tensor)) or \
               (ref_aligned.dim() < 2) or (pred_aligned.dim() < 2) or \
               (ref_aligned.numel() == 0) or (pred_aligned.numel() == 0):
                align_geometric_loss = self.loss_weight["rmsd_loss"] * normalized_rmsd
                return normalized_rmsd, align_geometric_loss

            # 3.1 MSE
            mse_align = F.mse_loss(ref_aligned, pred_aligned)

            # 3.2 Smooth-LDDT
            dist_true = torch.cdist(ref_aligned, ref_aligned)   # [N_atom, N_atom]
            dist_pred = torch.cdist(pred_aligned, pred_aligned) # [N_atom, N_atom]
            distance_mask = torch.ones_like(dist_true)
            lddt_mask = torch.ones_like(dist_true)
            smooth_lddt = self.smooth_lddt(
                pred_distance=dist_pred.unsqueeze(0),
                true_distance=dist_true,
                distance_mask=distance_mask,
                lddt_mask=lddt_mask,
                diffusion_chunk_size=None,
            )

            # # 3.3 PDELoss
            n_atom = ref_aligned.shape[0]
            # pde_bins = self.pde_loss.no_bins
            # logits_pde = torch.zeros((1, n_atom, n_atom, pde_bins), device=ref_aligned.device, dtype=ref_aligned.dtype)
            if true_mask is None:
                coordinate_mask = torch.ones(n_atom, device=ref_aligned.device, dtype=ref_aligned.dtype)
            else:
                if len(ref_ids) == 0 or min(ref_ids) < 0 or max(ref_ids) >= true_mask.shape[-1]:
                    coordinate_mask = torch.ones(n_atom, device=true_mask.device, dtype=true_mask.dtype)
                else:
                    coordinate_mask = _index_mask(true_mask, ref_ids)
            if pred_mask is None:
                rep_atom_mask = torch.ones(n_atom, device=ref_aligned.device, dtype=ref_aligned.dtype)
            else:
                if len(pred_ids) == 0 or min(pred_ids) < 0 or max(pred_ids) >= pred_mask.shape[-1]:
                    rep_atom_mask = torch.ones(n_atom, device=pred_mask.device, dtype=pred_mask.dtype)
                else:
                    rep_atom_mask = _index_mask(pred_mask, pred_ids)
            # pde_loss_val = self.pde_loss(
            #     logits=logits_pde,
            #     pred_coordinate=pred_aligned.unsqueeze(0),
            #     true_coordinate=ref_aligned,
            #     coordinate_mask=coordinate_mask,
            #     rep_atom_mask=rep_atom_mask,
            # )

            # 3.4 PLDDT Loss
            plddt_bins = self.plddt_loss.no_bins
            logits_plddt = torch.zeros((1, n_atom, plddt_bins), device=ref_aligned.device, dtype=ref_aligned.dtype)
            if true_is_nucleotide is None or pred_is_nucleotide is None or true_is_polymer is None or pred_is_polymer is None:
                raise ValueError("true_is_nucleotide, pred_is_nucleotide, true_is_polymer, pred_is_polymer must be provided")
            if len(ref_ids) == 0 or len(pred_ids) == 0:
                is_nucleotide = torch.zeros(n_atom, device=ref_aligned.device, dtype=torch.bool)
                is_polymer = torch.zeros(n_atom, device=ref_aligned.device, dtype=torch.bool)
            else:
                valid_ref_nuc = (min(ref_ids) >= 0 and max(ref_ids) < true_is_nucleotide.shape[-1])
                valid_pred_nuc = (min(pred_ids) >= 0 and max(pred_ids) < pred_is_nucleotide.shape[-1])
                if valid_ref_nuc and valid_pred_nuc:
                    nuc_ref = _index_mask(true_is_nucleotide, ref_ids).bool()
                    nuc_pred = _index_mask(pred_is_nucleotide, pred_ids).bool()
                    is_nucleotide = nuc_ref | nuc_pred
                else:
                    is_nucleotide = torch.zeros(n_atom, device=ref_aligned.device, dtype=torch.bool)
                valid_ref_poly = (min(ref_ids) >= 0 and max(ref_ids) < true_is_polymer.shape[-1])
                valid_pred_poly = (min(pred_ids) >= 0 and max(pred_ids) < pred_is_polymer.shape[-1])
                if valid_ref_poly and valid_pred_poly:
                    poly_ref = _index_mask(true_is_polymer, ref_ids).bool()
                    poly_pred = _index_mask(pred_is_polymer, pred_ids).bool()
                    is_polymer = poly_ref | poly_pred
                else:
                    is_polymer = torch.zeros(n_atom, device=ref_aligned.device, dtype=torch.bool)
            plddt_loss_val = self.plddt_loss(
                logits=logits_plddt,
                pred_coordinate=pred_aligned.unsqueeze(0),
                true_coordinate=ref_aligned,
                coordinate_mask=coordinate_mask,
                rep_atom_mask=rep_atom_mask,
                is_nucleotide=is_nucleotide,
                is_polymer=is_polymer,
            )

            align_geometric_loss = (
                self.loss_weight["rmsd_loss"]   * normalized_rmsd +
                self.loss_weight["mse_loss"]    * mse_align +
                self.loss_weight["smooth_lddt_loss"]   * smooth_lddt +
                # self.loss_weight["pde_loss"]    * pde_loss_val +
                self.loss_weight["plddt_loss"]  * plddt_loss_val
            )

            return normalized_rmsd, align_geometric_loss

        # 批量路径
        device = pred_coordinate.device
        dtype = pred_coordinate.dtype
        batch_size = pred_coordinate.shape[0]
        total_norm_rmsd = torch.tensor(0.0, device=device, dtype=dtype)
        total_align_geo = torch.tensor(0.0, device=device, dtype=dtype)
        valid_count = 0

        for i in range(batch_size):
            curr_true_mask = None if true_mask is None else true_mask[i]
            curr_pred_mask = None if pred_mask is None else pred_mask[i]
            curr_true_is_nuc = None if true_is_nucleotide is None else true_is_nucleotide[i]
            curr_pred_is_nuc = None if pred_is_nucleotide is None else pred_is_nucleotide[i]
            curr_true_is_poly = None if true_is_polymer is None else true_is_polymer[i]
            curr_pred_is_poly = None if pred_is_polymer is None else pred_is_polymer[i]

            sample_norm_rmsd, R_kabsch, t_kabsch, ref_aligned, pred_aligned, ref_ids, pred_ids = self.superimpose_and_calculate_rmsd(
                pdb_id,
                ref_coords=true_coordinate[i],
                pred_coords=pred_coordinate[i],
                ref_atoms=true_atoms,
                pred_atoms=pred_atoms,
                ref_mask=curr_true_mask,
                pred_mask=curr_pred_mask,
            )

            # NaN/Inf 检查（对齐 RMSD）
            if torch.isnan(sample_norm_rmsd).any() or torch.isinf(sample_norm_rmsd).any():
                current_rank = dist.get_rank() if dist.is_initialized() else "N/A_no_dist"
                print(f"CRITICAL align_geometric_loss (batch loop): normalized_rmsd is NaN/Inf for PDB {pdb_id}, sample_idx {i} on rank {current_rank}. Value: {sample_norm_rmsd.item()}")
                continue

            # 形状与有效性防护，避免 cdist 遇到 1D/空张量
            if isinstance(ref_aligned, torch.Tensor) and ref_aligned.dim() == 1:
                ref_aligned = ref_aligned.unsqueeze(0)
            if isinstance(pred_aligned, torch.Tensor) and pred_aligned.dim() == 1:
                pred_aligned = pred_aligned.unsqueeze(0)
            if (not isinstance(ref_aligned, torch.Tensor)) or (not isinstance(pred_aligned, torch.Tensor)) or \
               (ref_aligned.dim() < 2) or (pred_aligned.dim() < 2) or \
               (ref_aligned.numel() == 0) or (pred_aligned.numel() == 0):
                align_geo = self.loss_weight["rmsd_loss"] * sample_norm_rmsd
                if (not torch.isnan(align_geo).any()) and (not torch.isinf(align_geo).any()):
                    total_norm_rmsd += sample_norm_rmsd
                    total_align_geo += align_geo
                    valid_count += 1
                continue

            # 3.1 MSE
            mse_align = F.mse_loss(ref_aligned, pred_aligned)

            # 3.2 Smooth-LDDT
            dist_true = torch.cdist(ref_aligned, ref_aligned)
            dist_pred = torch.cdist(pred_aligned, pred_aligned)
            distance_mask = torch.ones_like(dist_true)
            lddt_mask = torch.ones_like(dist_true)
            smooth_lddt = self.smooth_lddt(
                pred_distance=dist_pred.unsqueeze(0),
                true_distance=dist_true,
                distance_mask=distance_mask,
                lddt_mask=lddt_mask,
                diffusion_chunk_size=None,
            )

            # # 3.3 PDELoss
            n_atom = ref_aligned.shape[0]
            # pde_bins = self.pde_loss.no_bins
            # logits_pde = torch.zeros((1, n_atom, n_atom, pde_bins), device=ref_aligned.device, dtype=ref_aligned.dtype)
            if curr_true_mask is None:
                coordinate_mask = torch.ones(n_atom, device=ref_aligned.device, dtype=ref_aligned.dtype)
            else:
                if len(ref_ids) == 0 or min(ref_ids) < 0 or max(ref_ids) >= curr_true_mask.shape[-1]:
                    coordinate_mask = torch.ones(n_atom, device=curr_true_mask.device, dtype=curr_true_mask.dtype)
                else:
                    coordinate_mask = _index_mask(curr_true_mask, ref_ids)
            if curr_pred_mask is None:
                rep_atom_mask = torch.ones(n_atom, device=ref_aligned.device, dtype=ref_aligned.dtype)
            else:
                if len(pred_ids) == 0 or min(pred_ids) < 0 or max(pred_ids) >= curr_pred_mask.shape[-1]:
                    rep_atom_mask = torch.ones(n_atom, device=curr_pred_mask.device, dtype=curr_pred_mask.dtype)
                else:
                    rep_atom_mask = _index_mask(curr_pred_mask, pred_ids)
            # pde_loss_val = self.pde_loss(
            #     logits=logits_pde,
            #     pred_coordinate=pred_aligned.unsqueeze(0),
            #     true_coordinate=ref_aligned,
            #     coordinate_mask=coordinate_mask,
            #     rep_atom_mask=rep_atom_mask,
            # )

            # 3.4 PLDDT Loss
            plddt_bins = self.plddt_loss.no_bins
            logits_plddt = torch.zeros((1, n_atom, plddt_bins), device=ref_aligned.device, dtype=ref_aligned.dtype)
            if curr_true_is_nuc is None or curr_pred_is_nuc is None or curr_true_is_poly is None or curr_pred_is_poly is None:
                raise ValueError("true_is_nucleotide, pred_is_nucleotide, true_is_polymer, pred_is_polymer must be provided for batch mode as well")
            if len(ref_ids) == 0 or len(pred_ids) == 0:
                is_nucleotide = torch.zeros(n_atom, device=ref_aligned.device, dtype=torch.bool)
                is_polymer = torch.zeros(n_atom, device=ref_aligned.device, dtype=torch.bool)
            else:
                valid_ref_nuc = (min(ref_ids) >= 0 and max(ref_ids) < curr_true_is_nuc.shape[-1])
                valid_pred_nuc = (min(pred_ids) >= 0 and max(pred_ids) < curr_pred_is_nuc.shape[-1])
                if valid_ref_nuc and valid_pred_nuc:
                    nuc_ref = _index_mask(curr_true_is_nuc, ref_ids).bool()
                    nuc_pred = _index_mask(curr_pred_is_nuc, pred_ids).bool()
                    is_nucleotide = nuc_ref | nuc_pred
                else:
                    is_nucleotide = torch.zeros(n_atom, device=ref_aligned.device, dtype=torch.bool)
                valid_ref_poly = (min(ref_ids) >= 0 and max(ref_ids) < curr_true_is_poly.shape[-1])
                valid_pred_poly = (min(pred_ids) >= 0 and max(pred_ids) < curr_pred_is_poly.shape[-1])
                if valid_ref_poly and valid_pred_poly:
                    poly_ref = _index_mask(curr_true_is_poly, ref_ids).bool()
                    poly_pred = _index_mask(curr_pred_is_poly, pred_ids).bool()
                    is_polymer = poly_ref | poly_pred
                else:
                    is_polymer = torch.zeros(n_atom, device=ref_aligned.device, dtype=torch.bool)
            plddt_loss_val = self.plddt_loss(
                logits=logits_plddt,
                pred_coordinate=pred_aligned.unsqueeze(0),
                true_coordinate=ref_aligned,
                coordinate_mask=coordinate_mask,
                rep_atom_mask=rep_atom_mask,
                is_nucleotide=is_nucleotide,
                is_polymer=is_polymer,
            )

            align_geo = (
                self.loss_weight["rmsd_loss"] * sample_norm_rmsd +
                self.loss_weight["mse_loss"] * mse_align +
                self.loss_weight["smooth_lddt_loss"] * smooth_lddt +
                # self.loss_weight["pde_loss"] * pde_loss_val +
                self.loss_weight["plddt_loss"] * plddt_loss_val
            )

            if (not torch.isnan(align_geo).any()) and (not torch.isinf(align_geo).any()):
                total_norm_rmsd += sample_norm_rmsd
                total_align_geo += align_geo
                valid_count += 1
            else:
                current_rank = dist.get_rank() if dist.is_initialized() else "N/A_no_dist"
                print(f"INFO align_geometric_loss (batch loop): align_geo was NaN/Inf for PDB {pdb_id}, sample_idx {i} on rank {current_rank}, skipped.")

        if valid_count > 0:
            avg_norm_rmsd = total_norm_rmsd / valid_count
            avg_align_geo = total_align_geo / valid_count
        else:
            current_rank = dist.get_rank() if dist.is_initialized() else "N/A_no_dist"
            print(f"警告 align_geometric_loss: 批次中没有有效的样本 for PDB {pdb_id} on rank {current_rank}. Returning defaults.")
            avg_norm_rmsd = torch.tensor(10.0, device=device, dtype=dtype)
            avg_align_geo = self.loss_weight["rmsd_loss"] * avg_norm_rmsd

        if torch.isnan(avg_norm_rmsd).any() or torch.isinf(avg_norm_rmsd).any():
            current_rank = dist.get_rank() if dist.is_initialized() else "N/A_no_dist"
            print(f"CRITICAL align_geometric_loss: avg_norm_rmsd is NaN/Inf for PDB {pdb_id} on rank {current_rank} BEFORE RETURN. Value: {avg_norm_rmsd.item()}")
        if torch.isnan(avg_align_geo).any() or torch.isinf(avg_align_geo).any():
            current_rank = dist.get_rank() if dist.is_initialized() else "N/A_no_dist"
            print(f"CRITICAL align_geometric_loss: avg_align_geo is NaN/Inf for PDB {pdb_id} on rank {current_rank} BEFORE RETURN.")

        return avg_norm_rmsd, avg_align_geo
    
    def prevent_nan_inf(self, tensor: torch.Tensor, eps: float = 1e-8, max_value: float = 1e8):
        # 检查NaN，若有则整体加一个eps
        if torch.isnan(tensor).any():
            tensor = tensor + eps
        # 检查Inf，若有则用一个极大但可微的值替换（如用自身乘以一个很大的数，保持梯度流动）
        if torch.isinf(tensor).any():
            tensor = tensor.sign() * max_value + eps
        return tensor

    def forward(self, pdb_id, seq_pred_dict, smiles_pred_dict, seq_label_dict, smiles_label_dict, seq_input_feature_dict, smiles_input_feature_dict, seq_atom_array, smiles_atom_array, mode="train"):
        """优化的前向传播函数"""
        
        print(f"\n--- MultiPSP_align_loss forward for PDB ID: {pdb_id}, Mode: {mode} ---")

        # 模式设置
        assert mode in ["train", "eval", "inference"]
        
        seq_diffusion_per_sample_scale = None
        smiles_diffusion_per_sample_scale = None

        if mode == "train":
            confidence_coordinate = "coordinate" # "coordinate_mini"
            if not self.configs.train_confidence_only:
                if "noise_level" not in seq_pred_dict or seq_pred_dict["noise_level"] is None:
                    print(f"WARNING (PDB {pdb_id}): seq_pred_dict missing 'noise_level' in train mode.")
                else:
                    seq_nl = seq_pred_dict["noise_level"]
                    print(f"PDB {pdb_id}: seq_noise_level: shape={seq_nl.shape}, values={seq_nl}, any_zero={(seq_nl == 0).any()}")
                    if (seq_nl == 0).any():
                        print(f"ERROR (PDB {pdb_id}): seq_noise_level contains zero, may lead to NaN/Inf scale!")
                    # Calculate scale, add epsilon to denominator for safety against noise_level being exactly 0
                    denominator_seq = (self.configs.sigma_data * seq_nl)**2 + 1e-12 # Added epsilon
                    seq_diffusion_per_sample_scale = (seq_nl**2 + self.configs.sigma_data**2) / denominator_seq
                    if torch.isnan(seq_diffusion_per_sample_scale).any() or torch.isinf(seq_diffusion_per_sample_scale).any():
                        print(f"ERROR (PDB {pdb_id}): seq_diffusion_per_sample_scale is NaN/Inf. NL: {seq_nl}, Denom: {denominator_seq}")


                if "noise_level" not in smiles_pred_dict or smiles_pred_dict["noise_level"] is None:
                    print(f"WARNING (PDB {pdb_id}): smiles_pred_dict missing 'noise_level' in train mode.")
                else:
                    smiles_nl = smiles_pred_dict["noise_level"]
                    print(f"PDB {pdb_id}: smiles_noise_level: shape={smiles_nl.shape}, values={smiles_nl}, any_zero={(smiles_nl == 0).any()}")
                    if (smiles_nl == 0).any():
                        print(f"ERROR (PDB {pdb_id}): smiles_noise_level contains zero, may lead to NaN/Inf scale!")
                    denominator_smiles = (self.configs.sigma_data * smiles_nl)**2 + 1e-12 # Added epsilon
                    smiles_diffusion_per_sample_scale = (smiles_nl**2 + self.configs.sigma_data**2) / denominator_smiles
                    if torch.isnan(smiles_diffusion_per_sample_scale).any() or torch.isinf(smiles_diffusion_per_sample_scale).any():
                        print(f"ERROR (PDB {pdb_id}): smiles_diffusion_per_sample_scale is NaN/Inf. NL: {smiles_nl}, Denom: {denominator_smiles}")
            else: # train_confidence_only = True
                seq_diffusion_per_sample_scale = None
                smiles_diffusion_per_sample_scale = None
        else: # eval or inference mode
            confidence_coordinate = "coordinate"
            seq_diffusion_per_sample_scale = None
            smiles_diffusion_per_sample_scale = None

        # 获取坐标和坐标掩码
        # Print details of input coordinates and masks
        def print_coord_stats(name, coord, mask):
            if coord is None:
                print(f"PDB {pdb_id}: {name}_coordinate is None")
                return
            print(f"PDB {pdb_id}: {name}_coordinate - shape={coord.shape}, dtype={coord.dtype}, has_nan={(torch.isnan(coord).any())}, has_inf={(torch.isinf(coord).any())}")
            if mask is None:
                 print(f"PDB {pdb_id}: {name}_coordinate_mask is None")
            else:
                 print(f"PDB {pdb_id}: {name}_coordinate_mask - shape={mask.shape}, dtype={mask.dtype}, sum={mask.sum().item() if mask.numel() > 0 else 'empty'}")

        seq_pred_coordinate = seq_pred_dict.get(confidence_coordinate)
        smiles_pred_coordinate = smiles_pred_dict.get(confidence_coordinate)
        seq_true_coordinate = seq_label_dict.get("coordinate")
        smiles_true_coordinate = smiles_label_dict.get("coordinate")
        seq_coordinate_mask = seq_label_dict.get("coordinate_mask")
        smiles_coordinate_mask = smiles_label_dict.get("coordinate_mask")

        print_coord_stats("seq_pred", seq_pred_coordinate, None) # Mask for pred is not directly used here, but through true_mask in align_loss
        print_coord_stats("smiles_pred", smiles_pred_coordinate, None)
        print_coord_stats("seq_true", seq_true_coordinate, seq_coordinate_mask)
        print_coord_stats("smiles_true", smiles_true_coordinate, smiles_coordinate_mask)

        default_device = torch.device('cpu')
        if seq_pred_coordinate is not None: default_device = seq_pred_coordinate.device
        elif smiles_pred_coordinate is not None: default_device = smiles_pred_coordinate.device
        elif seq_true_coordinate is not None: default_device = seq_true_coordinate.device
        
        # Handle cases where coordinates might be missing
        if seq_pred_coordinate is None or smiles_pred_coordinate is None or \
           seq_true_coordinate is None or smiles_true_coordinate is None:
            print(f"ERROR (PDB {pdb_id}): One or more coordinate sets are None. Skipping align_loss calculation.")
            seq_smiles_align_loss = torch.tensor(50.0, device=default_device, dtype=torch.float32) # Assign high loss
        else:
            # 处理原子数组
            seq_atoms = self.get_atom_from_atom_array(seq_atom_array) if seq_atom_array is not None else []
            smiles_atoms = self.get_atom_from_atom_array(smiles_atom_array) if smiles_atom_array is not None else []
            seq_is_nucleotide = seq_input_feature_dict.get("is_rna") + seq_input_feature_dict.get("is_dna")
            smiles_is_nucleotide = smiles_input_feature_dict.get("is_rna") + smiles_input_feature_dict.get("is_dna")
            seq_is_polymer    = torch.ones_like(seq_coordinate_mask) - seq_input_feature_dict.get("is_ligand")   # TODO: 1 - is_ligand  seq_input_feature_dict.get("is_ligand")
            smiles_is_polymer    = torch.ones_like(smiles_coordinate_mask) - smiles_input_feature_dict.get("is_ligand")   # TODO: 1 - is_ligand  smiles_input_feature_dict.get("is_ligand")
            
            # 计算序列和SMILES之间的对齐损失
            try:
                seq_smiles_norm_rmsd, raw_seq_smiles_align_loss = self.align_geometric_loss(
                    pdb_id, seq_pred_coordinate, smiles_pred_coordinate, 
                    seq_atoms, smiles_atoms,
                    pred_mask=seq_coordinate_mask, # Using true masks for predicted as well, common practice
                    true_mask=smiles_coordinate_mask,
                    true_is_nucleotide=seq_is_nucleotide,
                    pred_is_nucleotide=smiles_is_nucleotide,
                    true_is_polymer=seq_is_polymer,
                    pred_is_polymer=smiles_is_polymer
                )
                print(f"PDB ID {pdb_id}: raw_seq_smiles_align_loss (before scaling) = {raw_seq_smiles_align_loss.item()}")

                smiles_seq_norm_rmsd, raw_smiles_seq_align_loss = self.align_geometric_loss(
                    pdb_id, smiles_pred_coordinate, seq_pred_coordinate, 
                    smiles_atoms, seq_atoms,
                    pred_mask=smiles_coordinate_mask, # Using true masks for predicted as well, common practice
                    true_mask=seq_coordinate_mask,
                    true_is_nucleotide=smiles_is_nucleotide,
                    pred_is_nucleotide=seq_is_nucleotide,
                    true_is_polymer=smiles_is_polymer,
                    pred_is_polymer=seq_is_polymer
                )
                print(f"PDB ID {pdb_id}: raw_smiles_seq_align_loss (before scaling) = {raw_smiles_seq_align_loss.item()}")

                seq_smiles_align_loss = self.prevent_nan_inf(raw_seq_smiles_align_loss, eps=1e-8, max_value=1e8)
                smiles_seq_align_loss = self.prevent_nan_inf(raw_smiles_seq_align_loss, eps=1e-8, max_value=1e8)
                # -------------------------
                # 应用缩放因子
                if seq_diffusion_per_sample_scale is not None and smiles_diffusion_per_sample_scale is not None:
                    # 检查缩放因子是否为有效tensor
                    if not isinstance(seq_diffusion_per_sample_scale, torch.Tensor):
                        print(f"ERROR (PDB {pdb_id}): seq_diffusion_per_sample_scale 不是Tensor，使用1.0")
                        seq_scale_factor_val = 1.0
                    elif torch.isnan(seq_diffusion_per_sample_scale).any() or torch.isinf(seq_diffusion_per_sample_scale).any():
                        print(f"ERROR (PDB {pdb_id}): seq_diffusion_per_sample_scale 存在NaN/Inf，使用1.0")
                        seq_scale_factor_val = 1.0
                    elif seq_diffusion_per_sample_scale.numel() == 0:
                        print(f"ERROR (PDB {pdb_id}): seq_diffusion_per_sample_scale 为空，使用1.0")
                        seq_scale_factor_val = 1.0
                    else:
                        seq_scale_factor_val = seq_diffusion_per_sample_scale.mean().item()

                    if not isinstance(smiles_diffusion_per_sample_scale, torch.Tensor):
                        print(f"ERROR (PDB {pdb_id}): smiles_diffusion_per_sample_scale 不是Tensor，使用1.0")
                        smiles_scale_factor_val = 1.0
                    elif torch.isnan(smiles_diffusion_per_sample_scale).any() or torch.isinf(smiles_diffusion_per_sample_scale).any():
                        print(f"ERROR (PDB {pdb_id}): smiles_diffusion_per_sample_scale 存在NaN/Inf，使用1.0")
                        smiles_scale_factor_val = 1.0
                    elif smiles_diffusion_per_sample_scale.numel() == 0:
                        print(f"ERROR (PDB {pdb_id}): smiles_diffusion_per_sample_scale 为空，使用1.0")
                        smiles_scale_factor_val = 1.0
                    else:
                        smiles_scale_factor_val = smiles_diffusion_per_sample_scale.mean().item()

                    # 由于mean().item()已经是float，这里用math.isnan和math.isinf
                    import math
                    if math.isnan(seq_scale_factor_val) or math.isinf(seq_scale_factor_val) or \
                       math.isnan(smiles_scale_factor_val) or math.isinf(smiles_scale_factor_val):
                        print(f"ERROR (PDB {pdb_id}): seq_scale_factor_val 或 smiles_scale_factor_val 是NaN/Inf，不进行缩放。")
                    else:
                        seq_smiles_align_loss = seq_smiles_align_loss * seq_scale_factor_val
                        smiles_seq_align_loss = smiles_seq_align_loss * smiles_scale_factor_val
                        seq_smiles_align_loss = (seq_smiles_align_loss + smiles_seq_align_loss) / 2.0
                
                # 再次检查NaN和Inf，并限制最大值
                if torch.isnan(seq_smiles_align_loss) or torch.isinf(seq_smiles_align_loss):
                    print(f"警告 (PDB {pdb_id}): Scaled seq_smiles_align_loss为NaN或Inf ({seq_smiles_align_loss.item()})")
                    seq_smiles_align_loss = self.prevent_nan_inf(seq_smiles_align_loss, eps=1e-8, max_value=1e8)
                
                # clamp
                # seq_smiles_align_loss = torch.clamp(seq_smiles_align_loss, min=0.0, max=50.0) # Clamp after scaling
                    
            except Exception as e:
                print(f"计算序列-SMILES对齐损失时出错 (PDB {pdb_id}): {e}")
                import traceback
                traceback.print_exc()
                seq_smiles_align_loss = torch.tensor(100.0, device=seq_pred_coordinate.device if seq_pred_coordinate is not None else default_device, dtype=torch.float32)
        
        # Global Representation Loss
        seq_global_repr = seq_pred_dict.get("global_repr")
        smiles_global_repr = smiles_pred_dict.get("global_repr")

        if seq_global_repr is None or smiles_global_repr is None:
            print(f"ERROR (PDB {pdb_id}): seq_global_repr or smiles_global_repr is None. Skipping global_repr_loss.")
            global_repr_loss = torch.tensor(0.0, device=default_device, dtype=torch.float32) # Or a high value if preferred
        else:
            print(f"PDB {pdb_id}: seq_global_repr - shape={seq_global_repr.shape}, dtype={seq_global_repr.dtype}, has_nan={torch.isnan(seq_global_repr).any()}, has_inf={torch.isinf(seq_global_repr).any()}")
            print(f"PDB {pdb_id}: smiles_global_repr - shape={smiles_global_repr.shape}, dtype={smiles_global_repr.dtype}, has_nan={torch.isnan(smiles_global_repr).any()}, has_inf={torch.isinf(smiles_global_repr).any()}")
            global_repr_loss = self.global_repr_loss(seq_global_repr, smiles_global_repr)

        # Final aggregation and checks
        loss_dict = {
            "seq_smiles_align_loss": seq_smiles_align_loss.detach().clone() if isinstance(seq_smiles_align_loss, torch.Tensor) else torch.tensor(seq_smiles_align_loss), # Ensure tensor for dict
            "global_repr_loss": global_repr_loss.detach().clone() if isinstance(global_repr_loss, torch.Tensor) else torch.tensor(global_repr_loss)
        }
        print(f"PDB ID {pdb_id} (MultiPSP_align_loss): seq_smiles_align_loss_val = {loss_dict['seq_smiles_align_loss'].item()}, global_repr_loss_val = {loss_dict['global_repr_loss'].item()}")
        if torch.isnan(loss_dict['seq_smiles_align_loss']).any(): print(f"PDB ID {pdb_id}: 最终 seq_smiles_align_loss is NaN")
        if torch.isnan(loss_dict['global_repr_loss']).any(): print(f"PDB ID {pdb_id}: 最终 global_repr_loss is NaN")
        
        if os.getenv("ALIGN_MODE") == "struct":
            final_align_loss = seq_smiles_align_loss
        elif os.getenv("ALIGN_MODE") == "repr":
            final_align_loss = global_repr_loss
        elif os.getenv("ALIGN_MODE") == "both":
            final_align_loss = seq_smiles_align_loss + global_repr_loss
        elif os.getenv("ALIGN_MODE") == "no":
            final_align_loss = 0.0
        else:
            raise ValueError(f"env ALIGN_MODE must be either 'no', 'struct', 'repr', or 'both', got {os.getenv('ALIGN_MODE')}")

        if torch.isnan(final_align_loss).any() or torch.isinf(final_align_loss).any():
            print(f"CRITICAL NaN/Inf DETECTED on rank {dist.get_rank() if dist.is_initialized() else 'N/A'} for PDB ID {pdb_id} BEFORE RETURN!")
            print(f"Components: seq_smiles_align_loss={seq_smiles_align_loss.item()}, global_repr_loss={global_repr_loss.item()}")
            # Potentially raise an error here to stop execution immediately on the faulty rank
            # raise RuntimeError(f"NaN/Inf in final_align_loss for PDB {pdb_id} on rank {dist.get_rank()}")
        
        print(f"--- End MultiPSP_align_loss for PDB ID: {pdb_id}, Final Align Loss: {final_align_loss.item() if isinstance(final_align_loss, torch.Tensor) else final_align_loss} ---\n")
        return final_align_loss, loss_dict

