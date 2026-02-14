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

from typing import Optional

import torch


def rmsd(
    pred_pose: torch.Tensor,
    true_pose: torch.Tensor,
    mask: torch.Tensor = None,
    eps: float = 0.0,
    reduce: bool = True,
):
    """
    compute rmsd between two poses, with the same shape
    Arguments:
        pred_pose/true_pose: [...,N,3], two poses with the same shape
        mask: [..., N], mask to indicate which atoms/pseudo_betas/etc to compute
        eps: add a tolerance to avoid floating number issue
        reduce: decide the return shape of rmsd;
    Return:
        rmsd: if reduce = true, return the mean of rmsd over batches;
            else return a tensor containing each rmsd separately
    """

    # mask [..., N]
    assert pred_pose.shape == true_pose.shape  # [..., N, 3]

    if mask is None:
        mask = torch.ones(true_pose.shape[:-1], device=true_pose.device)

    # [...]
    err2 = (torch.square(pred_pose - true_pose).sum(dim=-1) * mask).sum(
        dim=-1
    ) / mask.sum(dim=-1)
    rmsd = err2.add(eps).sqrt()
    if reduce:
        rmsd = rmsd.mean()
    return rmsd


def align_pred_to_true(
    pred_pose: torch.Tensor,
    true_pose: torch.Tensor,
    atom_mask: Optional[torch.Tensor] = None,
    weight: Optional[torch.Tensor] = None,
    allowing_reflection: bool = False,
):
    """Find optimal transformation, rotation (and reflection) of two poses.
    Arguments:
        pred_pose: [...,N,3] the pose to perform transformation on
        true_pose: [...,N,3] the target pose to align pred_pose to
        atom_mask: [..., N] a mask for atoms
        weight: [..., N] a weight vector to be applied.
        allow_reflection: whether to allow reflection when finding optimal alignment
    return:
        aligned_pose: [...,N,3] the transformed pose
        rot: optimal rotation
        translate: optimal translation
    """
    if atom_mask is not None:
        pred_pose = pred_pose * atom_mask.unsqueeze(-1)
        true_pose = true_pose * atom_mask.unsqueeze(-1)
    else:
        atom_mask = torch.ones(*pred_pose.shape[:-1]).to(pred_pose.device)

    if weight is None:
        weight = atom_mask
    else:
        weight = weight * atom_mask

    weighted_n_atoms = torch.sum(weight, dim=-1, keepdim=True).unsqueeze(-1)
    pred_pose_centroid = (
        torch.sum(pred_pose * weight.unsqueeze(-1), dim=-2, keepdim=True)
        / weighted_n_atoms
    )
    pred_pose_centered = pred_pose - pred_pose_centroid
    true_pose_centroid = (
        torch.sum(true_pose * weight.unsqueeze(-1), dim=-2, keepdim=True)
        / weighted_n_atoms
    )
    true_pose_centered = true_pose - true_pose_centroid
    H_mat = torch.matmul(
        (pred_pose_centered * weight.unsqueeze(-1)).transpose(-2, -1),
        true_pose_centered * atom_mask.unsqueeze(-1),
    )
    # torch.svd 在 CUDA 上不支持 bfloat16/float16，需要临时升精度到 float32
    orig_dtype = H_mat.dtype
    if H_mat.is_floating_point() and H_mat.dtype in (torch.float16, torch.bfloat16):
        H_for_svd = H_mat.float()
    else:
        H_for_svd = H_mat

    u, s, v = torch.svd(H_for_svd)
    u = u.transpose(-1, -2)

    if not allowing_reflection:

        # torch.linalg.det 也不支持 bf16/fp16，需要临时升精度
        det_input = torch.matmul(v, u)
        orig_det_dtype = det_input.dtype
        if det_input.is_floating_point() and det_input.dtype in (torch.float16, torch.bfloat16):
            det_input = det_input.float()
        det = torch.linalg.det(det_input)

        diagonal = torch.stack(
            [torch.ones_like(det), torch.ones_like(det), det], dim=-1
        )
        rot = torch.matmul(
            torch.diag_embed(diagonal).to(u.device),
            u,
        )
        rot = torch.matmul(v, rot)
    else:
        rot = torch.matmul(v, u)
    translate = true_pose_centroid - torch.matmul(
        pred_pose_centroid, rot.transpose(-1, -2)
    )

    pred_pose_translated = (
        torch.matmul(pred_pose_centered, rot.transpose(-1, -2)) + true_pose_centroid
    )

    # 将输出转换回与输入相同的 dtype，以兼容混合精度训练（例如 bf16）
    pred_pose_translated = pred_pose_translated.to(pred_pose.dtype)
    rot = rot.to(pred_pose.dtype)
    translate = translate.to(pred_pose.dtype)

    return pred_pose_translated, rot, translate


def partially_aligned_rmsd(
    pred_pose: torch.Tensor,
    true_pose: torch.Tensor,
    align_mask: torch.Tensor,
    atom_mask: torch.Tensor,
    weight: Optional[torch.Tensor] = None,
    eps: float = 0.0,
    reduce: bool = True,
    allowing_reflection: bool = False,
):
    """RMSD when aligning parts of the complex coordinate, does NOT take permutation symmetricity into consideration
    Arguments:
        pred_pose: native predicted pose, [..., N,3]
        true_pose: ground truth pose, [..., N, 3]
        align_mask: a mask representing which coordinates to align [..., N]
        atom_mask: a mask representing which coordinates to compute loss [..., N]
        weight: a weight tensor assining weights in alignment for each atom [..., N]
        eps: add a tolerance to avoid floating number issue in sqrt
        reduce: decide the return shape of rmsd;
        allowing_reflection: whether to allow reflection when finding optimal alignment
    return:
        aligned_part_rmsd: the rmsd of part being align_masked
        unaligned_part_rmsd: the rmsd of unaligned part
        transformed_pred_pose:
        rot: optimal rotation
        trans: optimal translation
    """
    _, rot, translate = align_pred_to_true(
        pred_pose,
        true_pose,
        atom_mask=atom_mask * align_mask,
        weight=weight,
        allowing_reflection=allowing_reflection,
    )
    transformed_pose = torch.matmul(pred_pose, rot.transpose(-1, -2)) + translate
    err_atom = torch.square(transformed_pose - true_pose).sum(dim=-1) * atom_mask
    aligned_mask, unaligned_mask = atom_mask * align_mask.float(), atom_mask * (
        1 - align_mask.float()
    )
    aligned_part_err_square = (err_atom * aligned_mask).sum(dim=-1) / aligned_mask.sum(
        dim=-1
    )
    unaligned_part_err_square = (err_atom * unaligned_mask).sum(
        dim=-1
    ) / unaligned_mask.sum(dim=-1)
    aligned_part_rmsd = aligned_part_err_square.add(eps).sqrt()
    unaligned_part_rmsd = unaligned_part_err_square.add(eps).sqrt()
    if reduce:
        aligned_part_rmsd = aligned_part_rmsd.mean()
        unaligned_part_rmsd = unaligned_part_rmsd.mean()
    return aligned_part_rmsd, unaligned_part_rmsd, transformed_pose, rot, translate


def self_aligned_rmsd(
    pred_pose: torch.Tensor,
    true_pose: torch.Tensor,
    atom_mask: torch.Tensor,
    eps: float = 0.0,
    reduce: bool = True,
    allowing_reflection: bool = False,
):
    """RMSD when aligning one molecule with ground truth and compute rmsd.
    Arguments:
        pred_pose: native predicted pose, [..., N,3]
        true_pose: ground truth pose, [..., N, 3]
        atom_mask: a mask representing which coordinates to compute loss [..., N]
        eps: add a tolerance to avoid floating number issue in sqrt
        reduce: decide the return shape of rmsd;
        allowing_reflection: whether to allow reflection when finding optimal alignment
    return:
        aligned_rmsd: the rmsd of part being align_masked
        transformed_pred_pose: the aligned pose
        rot: optimal rotation matrix
        trans: optimal translation
    """
    aligned_rmsd, _, transformed_pred_pose, rot, trans = partially_aligned_rmsd(
        pred_pose=pred_pose,
        true_pose=true_pose,
        align_mask=atom_mask,
        atom_mask=atom_mask,
        eps=eps,
        reduce=reduce,
        allowing_reflection=allowing_reflection,
    )
    return aligned_rmsd, transformed_pred_pose, rot, trans


def weighted_rigid_align(
    x: torch.Tensor,
    x_target: torch.Tensor,
    atom_weight: torch.Tensor,
    stop_gradient: bool = True,
) -> tuple[torch.Tensor]:
    """Implements Algorithm 28 in AF3. Wrap `align_pred_to_true`.

    Args:
        x (torch.Tensor): input coordinates, it will be moved to match x_target.
            [..., N_atom, 3]
        x_target (torch.Tensor): target coordinates for the input to match.
            [..., N_atom, 3]
        atom_weight (torch.Tensor): weights for each atom.
            [..., N_atom] or [N_atom]
        stop_gradient (bool): whether to detach the output. If true, will run it with no_grad() ctx.

    Returns:
        x_aligned (torch.Tensor): rotated, translated x which should be closer to x_target.
            [..., N_atom, 3]
    """

    x_prefix = x.shape[:-2]
    n_atom = x.shape[-2]

    def _is_broadcastable(shape, target):
        return all(s == 1 or s == t for s, t in zip(shape, target))

    if atom_weight.dim() == 1:
        assert (
            atom_weight.shape[-1] == n_atom
        ), f"atom_weight shape {atom_weight.shape} does not match x shape {x.shape}"
        # Expand to match x batch prefix
        atom_weight = atom_weight.view(*([1] * len(x_prefix)), n_atom).expand(
            *x_prefix, n_atom
        )
    elif atom_weight.dim() == len(x.shape) - 1:
        if atom_weight.shape[-1] != n_atom:
            raise AssertionError(
                f"atom_weight shape {atom_weight.shape} does not match x shape {x.shape}"
            )

        weight_prefix = atom_weight.shape[:-1]
        # Drop redundant leading singleton dims
        while len(weight_prefix) > len(x_prefix) and atom_weight.shape[0] == 1:
            atom_weight = atom_weight.squeeze(0)
            weight_prefix = atom_weight.shape[:-1]

        # Add missing leading singleton dims
        if len(weight_prefix) < len(x_prefix):
            for _ in range(len(x_prefix) - len(weight_prefix)):
                atom_weight = atom_weight.unsqueeze(0)
            weight_prefix = atom_weight.shape[:-1]

        if weight_prefix != x_prefix:
            slices = []
            for w_dim, t_dim in zip(weight_prefix, x_prefix):
                if w_dim == t_dim:
                    slices.append(slice(None))
                elif t_dim == 1:
                    slices.append(slice(0, 1))
                else:
                    slices = None
                    break
            if slices is not None:
                atom_weight = atom_weight[tuple(slices)]
                weight_prefix = atom_weight.shape[:-1]

        assert _is_broadcastable(
            weight_prefix, x_prefix
        ), f"atom_weight shape {atom_weight.shape} does not match x shape {x.shape}"
        if weight_prefix != x_prefix:
            atom_weight = atom_weight.expand(*x_prefix, n_atom)
    else:
        raise AssertionError(
            f"atom_weight shape {atom_weight.shape} does not match x shape {x.shape}"
        )

    if stop_gradient:
        with torch.no_grad():
            x_aligned, rot, trans = align_pred_to_true(
                pred_pose=x,
                true_pose=x_target,
                atom_mask=None,
                weight=atom_weight,
                allowing_reflection=False,
            )
            return x_aligned.detach()
    else:
        x_aligned, rot, trans = align_pred_to_true(
            pred_pose=x,
            true_pose=x_target,
            atom_mask=None,
            weight=atom_weight,
            allowing_reflection=False,
        )
        return x_aligned
