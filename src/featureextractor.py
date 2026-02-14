import torch
import torch.nn as nn
from esm.esmfold.v1.tri_self_attn_block import TriangularSelfAttentionBlock
from torch.nn import LayerNorm, Linear


class RelativePosition(nn.Module):
    def __init__(self, bins, pairwise_state_dim):
        super().__init__()
        self.bins = bins
        self.embedding = nn.Embedding(2 * bins + 2, pairwise_state_dim)

    def forward(self, residue_index, mask=None):
        # residue_index: (N, N)
        # mask: (N,) or None
        # No batch dimension
        diff = residue_index - residue_index.t()
        diff = diff.clamp(-self.bins, self.bins) + self.bins + 1  # (N, N)
        if mask is not None:
            # mask: (N,)
            mask = mask.bool()
            mask_2d = mask.unsqueeze(0) & mask.unsqueeze(1)  # (N, N), ensure bool type
            diff = diff.masked_fill(~mask_2d.bool(), 0)
        return self.embedding(diff)


class FeatureExtractor(nn.Module):
    """
    特征提取器。

    - s_s 代表每个残基（Residue）的序列特征（Sequence Feature），用于描述单个残基自身的信息，包含来自ESM蛋白嵌入和氨基酸类型嵌入的融合特征，维度为 (B, N, sequence_state_dim)。常用于单体注意力和序列层处理。
    - s_z 代表残基-残基对的成对特征（Pairwise Feature），用于描述蛋白质中残基与残基之间的关系，初始包含坐标距离、是否同类型、以及相对位置编码等，维度为 (B, N, N, pairwise_state_dim)。常用于三角自注意力层和关系建模。
    """

    def __init__(
        self,
        num_blocks=2,
        sequence_state_dim=1024,
        pairwise_state_dim=128,
        sequence_head_width=32,
        pairwise_head_width=32,
        position_bins=32,
        dropout=0.1,
    ):
        super().__init__()
        self.sequence_state_dim = sequence_state_dim
        self.pairwise_state_dim = pairwise_state_dim
        self.esm_feats_protein = 1280

        # Embedding layer for aatype
        self.aatype_embedding = nn.Embedding(21, self.sequence_state_dim)

        # Project ESM embeddings to sequence state dim
        self.esm_s_mlp = nn.Sequential(
            LayerNorm(self.esm_feats_protein),
            Linear(self.esm_feats_protein, self.sequence_state_dim),
            nn.ReLU(),
            Linear(self.sequence_state_dim, self.sequence_state_dim),
        )

        # Position embedding for pairwise features
        self.pairwise_positional_embedding = RelativePosition(position_bins, pairwise_state_dim)
        self.pair_norm = LayerNorm(pairwise_state_dim)

        # Attention blocks
        self.blocks = nn.ModuleList([
            TriangularSelfAttentionBlock(
                sequence_state_dim=sequence_state_dim,
                pairwise_state_dim=pairwise_state_dim,
                sequence_head_width=sequence_head_width,
                pairwise_head_width=pairwise_head_width,
                dropout=dropout,
            ) for _ in range(num_blocks)
        ])

        self.use_checkpoint = False

    def gradient_checkpointing_enable(self):
        self.use_checkpoint = True

    def gradient_checkpointing_disable(self):
        self.use_checkpoint = False

    def _forward_impl(self, aatype, esm_s_protein, mask, coords, rotation_matrices):
        max_N, _ = esm_s_protein.shape

        # Step 1: Process sequence features (s_s)
        # s_s: 序列级特征, 每个残基一个向量，含ESM特征和氨基酸类型嵌入
        aatype_emb = self.aatype_embedding(aatype)  # (max{N}, sequence_state_dim)
        esm_s_projected = self.esm_s_mlp(esm_s_protein)  # (max{N}, sequence_state_dim)
        s_s = esm_s_projected + aatype_emb  # Combine ESM and aatype embeddings
        s_s = s_s * mask.unsqueeze(-1)  # Apply mask to ignore padding regions

        # Step 2: Compute transformed coordinates using rotation matrices (No batch size)
        def apply_rotation(coords, rotation_matrices):
            """
            Apply rotation matrices to coordinates.
            Args:
                coords: [N, 3] Residue coordinates
                rotation_matrices: [N, 3, 3] Rotation matrices
            Returns:
                Transformed coordinates: [N, 3]
            """
            return torch.einsum('nij,nj->ni', rotation_matrices, coords)

        transformed_coords = apply_rotation(coords, rotation_matrices)  # Transform coordinates

        # Step 3: Initialize pairwise features (s_z) using transformed coordinates and aatype
        # s_z: 残基-残基对特征，包括距离和类型相同标记
        pairwise_distances = torch.cdist(transformed_coords, transformed_coords, p=2)  # (max{N}, max{N})
        pairwise_aatype = (aatype.unsqueeze(1) == aatype.unsqueeze(0)).float()  # (max{N}, max{N})
        s_z = pairwise_distances.unsqueeze(-1).expand(max_N, max_N, self.pairwise_state_dim)  # (max{N}, max{N}, pairwise_state_dim)
        s_z = s_z + pairwise_aatype.unsqueeze(-1)  # Incorporate aatype similarity

        # Add positional embeddings
        residue_indices = torch.arange(max_N, device=coords.device).expand(max_N, max_N)  # (max{N}, max{N})
        pos_embedding = self.pairwise_positional_embedding(residue_indices, mask=mask)  # (max{N}, max{N}, pairwise_state_dim)
        s_z = s_z + pos_embedding

        # Normalize pairwise features
        s_z = self.pair_norm(s_z)

        # Convert mask to float for attention blocks
        mask = mask.float()

        # Step 4: Process through attention blocks
        # TriangularSelfAttentionBlock need batch dimention
        s_s_with_batch = s_s.unsqueeze(0)
        s_z_with_batch = s_z.unsqueeze(0)
        mask_with_batch = mask.unsqueeze(0)
        for block in self.blocks:
            # Pass mask with shape (max{N}) to the block
            s_s_with_batch, s_z_with_batch = block(s_s_with_batch, s_z_with_batch, mask=mask_with_batch)
        s_s = s_s_with_batch.squeeze(0)
        s_z = s_z_with_batch.squeeze(0)
        return s_s, s_z  # s_s: (N, sequence_state_dim)，每个残基的表示；s_z: (N, N, pairwise_state_dim)，每对残基的关系特征

    def forward(self, aatype, esm_s_protein, mask, coords, rotation_matrices):
        if self.use_checkpoint and self.training:
            return torch.utils.checkpoint.checkpoint(
                self._forward_impl,
                aatype, esm_s_protein, mask, coords, rotation_matrices,
                use_reentrant=False
            )
        else:
            return self._forward_impl(aatype, esm_s_protein, mask, coords, rotation_matrices)


if __name__ == "__main__":
    # 下面的测试要改，已经没有batch size了
    # Test data
    B, max_N = 2, 10  # batch size and maximum sequence length
    C_s, C_z = 1024, 128  # sequence and pairwise feature dimensions

    aatype = torch.randint(1, 21, (B, max_N)).cuda()  # (B, max{N})
    esm_s_protein = torch.randn(B, max_N, 1280).cuda()  # (B, max{N}, C_s)
    mask = torch.randint(0, 2, (B, max_N)).cuda()  # (B, max{N})
    coords = torch.randn(B, max_N, 3).cuda()  # (B, max{N}, 3)
    rotation_matrices = torch.eye(3).unsqueeze(0).unsqueeze(0).expand(B, max_N, 3, 3).cuda()  # (B, max{N}, 3, 3)

    # Initialize model
    model = FeatureExtractor(num_blocks=2, sequence_state_dim=C_s, pairwise_state_dim=C_z).cuda()
    s_s, s_z = model(aatype, esm_s_protein, mask, coords, rotation_matrices)

    print("s_s shape:", s_s.shape)  # (B, max{N}, sequence_state_dim)
    print("s_z shape:", s_z.shape)  # (B, max{N}, max{N}, pairwise_state_dim)
