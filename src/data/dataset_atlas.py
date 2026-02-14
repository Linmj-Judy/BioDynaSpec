import torch
from torch.utils.data import Dataset, DataLoader, IterableDataset
import numpy as np
import random
from torchdrug import data
import os

class ProteinMotionDataset(Dataset):
    def __init__(self, data_path, bihito_repr_dir=None):
        """
        Args:
            data_path (str): Path to the .pth file containing the data
        """
        self.data = torch.load(data_path, weights_only=False)
        self.bihito_repr_dir = bihito_repr_dir
        # Remove all proteins with length over 1000
        self.data = [data for data in self.data if data['positions']['x0'].__len__() <= 1000]
        self.protein_ids = list(range(len(self.data)))

    def __len__(self):
        return len(self.protein_ids)

    def _build_geometric_edges(self, coords):
        """Build geometric graph edge list
        
        Args:
            coords: [N, 3] protein coordinates
                
        Returns:
            edge_list: [num_edges, 3] (src_idx, dst_idx, edge_type)
            edge_feature: [num_edges, 7] [relative position(3) + distance(1) + direction vector(3)]
        """
        num_nodes = coords.shape[0]
        device = coords.device
        edge_list = []
        
        # 1. Sequential adjacent edges (max_distance=2)
        for d in range(1, 3):
            # Forward edges
            src = torch.arange(num_nodes-d, device=device)
            dst = torch.arange(d, num_nodes, device=device)
            edge_type = torch.full((len(src),), d-1, device=device)
            seq_edges = torch.stack([src, dst, edge_type], dim=1)
            edge_list.append(seq_edges)
            
            # Reverse edges
            edge_type = torch.full((len(src),), d-1+3, device=device)  # +3 indicates reverse
            seq_edges_rev = torch.stack([dst, src, edge_type], dim=1)
            edge_list.append(seq_edges_rev)
        
        # 2. Spatial adjacent edges (radius=7.0, max_distance=5)
        dist = torch.cdist(coords, coords)
        spatial_mask = (dist < 7.0) & (dist > 0)
        spatial_edges = torch.nonzero(spatial_mask)
        seq_dist = torch.abs(spatial_edges[:,1] - spatial_edges[:,0])
        valid_spatial = seq_dist > 5
        spatial_edges = spatial_edges[valid_spatial]
        
        # Forward edges
        edge_type = torch.full((len(spatial_edges),), 2, device=device)
        spatial_edges = torch.cat([spatial_edges, edge_type.unsqueeze(1)], dim=1)
        edge_list.append(spatial_edges)
        
        # Reverse edges
        edge_type = torch.full((len(spatial_edges),), 5, device=device)  # 2+3=5
        spatial_edges_rev = torch.cat([spatial_edges[:, [1,0]], edge_type.unsqueeze(1)], dim=1)
        edge_list.append(spatial_edges_rev)
        
        # 3. KNN edges (k=10, max_distance=5)
        dist_wo_seq = dist.clone()
        seq_mask = torch.abs(torch.arange(num_nodes).unsqueeze(0) - 
                        torch.arange(num_nodes).unsqueeze(1)) <= 5
        dist_wo_seq[seq_mask] = float('inf')
        _, knn_indices = torch.topk(dist_wo_seq, k=10, dim=-1, largest=False)
        rows = torch.arange(num_nodes, device=device).unsqueeze(-1).expand(-1, 10)
        src = rows.reshape(-1)
        dst = knn_indices.reshape(-1)
        
        # Forward edges
        edge_type = torch.full((len(src),), 3, device=device)
        knn_edges = torch.stack([src, dst, edge_type], dim=1)
        edge_list.append(knn_edges)
        
        # Reverse edges
        edge_type = torch.full((len(src),), 6, device=device)  # 3+3=6
        knn_edges_rev = torch.stack([dst, src, edge_type], dim=1)
        edge_list.append(knn_edges_rev)
        
        # 4. Self-loop edges
        self_loop = torch.arange(num_nodes, device=device)
        edge_type = torch.full((num_nodes,), 6, device=device)
        self_edges = torch.stack([self_loop, self_loop, edge_type], dim=1)
        edge_list.append(self_edges)
        
        # Merge all edges
        edge_list = torch.cat(edge_list, dim=0)
        
        # Calculate edge features
        src_pos = coords[edge_list[:,0]]
        dst_pos = coords[edge_list[:,1]]
        rel_pos = dst_pos - src_pos
        dist = torch.norm(rel_pos, dim=-1, keepdim=True)
        direction = rel_pos / (dist + 1e-10)
        
        # Merge edge features
        edge_feature = torch.cat([
            rel_pos,    # relative position
            dist,       # distance
            direction   # direction vector
        ], dim=-1)
        
        return edge_list, edge_feature

    def __getitem__(self, idx):
        """
        Returns:
            graph (data.Graph): Protein graph structure
            coords (torch.Tensor): [N, 3] First frame residue coordinates
            spectrum (torch.Tensor): [N, 96] Motion spectrum
            rotation_matrices (torch.Tensor): [N, 3, 3] Rotation matrices
            aatype (torch.Tensor): [N] Amino acid types
            esm_s_protein (torch.Tensor): [N, 1280] ESM embeddings
        """
        # idx: integer index, get protein id string
        protein_id = self.protein_ids[idx]
        protein_data = self.data[protein_id]
        id_str = protein_data['id']

        # Build coordinates
        coords = torch.tensor(np.stack([
            protein_data['positions']['x0'],
            protein_data['positions']['y0'],
            protein_data['positions']['z0']
        ], axis=-1), dtype=torch.float32)

        # Build spectrum features [N, 96]
        spectrum_x_real = [item[:, 0] for item in protein_data['positions']['x']]  # [N, 16]
        spectrum_x_imag = [item[:, 1] for item in protein_data['positions']['x']]
        spectrum_y_real = [item[:, 0] for item in protein_data['positions']['y']]
        spectrum_y_imag = [item[:, 1] for item in protein_data['positions']['y']]
        spectrum_z_real = [item[:, 0] for item in protein_data['positions']['z']]
        spectrum_z_imag = [item[:, 1] for item in protein_data['positions']['z']]

        spectrum = np.stack([
            spectrum_x_real, spectrum_x_imag,
            spectrum_y_real, spectrum_y_imag,
            spectrum_z_real, spectrum_z_imag,
        ], axis=-1)  # [N, 16, 6]
        # Convert spectrum features to torch.Tensor
        spectrum = torch.tensor(spectrum, dtype=torch.float32) # [N, 16, 6]
        #spectrum=spectrum.view(coords.shape[0], -1) # [N, 96]
        spectrum = spectrum.unsqueeze(-1) # [N, 16, 6，1]

        # Build rotation matrices
        rotation_matrices = protein_data['rotation_matrices0'].clone().detach().requires_grad_(True)

        # Build amino acid types
        aatype = protein_data['amino_acid_types'].clone().detach()
        aa_mask = torch.ones(aatype.shape, dtype=torch.bool)

        # Build ESM features
        esm_s_protein = protein_data['esm']#[1:-1].clone().detach().requires_grad_(True)

        # # Build geometric graph
        # edge_list, edge_feature = self._build_geometric_edges(coords)
        
        # # Build graph
        # graph = data.Graph(
        #     edge_list=edge_list,
        #     edge_feature=edge_feature,
        #     num_node=coords.shape[0],
        #     num_relation=7 # sequential edges(2) + spatial edges(2) + KNN edges(2) + self-loop edges(1)
        # )
        # # ===== BiHiTo representation =====
        # bihito_repr_path = os.path.join(self.bihito_repr_dir, f"{id_str.lower()}.pt")
        # if os.path.exists(bihito_repr_path):
        #     bihito_repr_dict = torch.load(bihito_repr_path, weights_only=False)
        #     for key, value in bihito_repr_dict.items():
        #         if isinstance(value, torch.Tensor):
        #             bihito_repr_dict[key] = value.detach().requires_grad_(False)
        #         if isinstance(value, dict):
        #             for sub_key, sub_value in value.items():
        #                 if isinstance(sub_value, torch.Tensor):
        #                     value[sub_key] = sub_value.detach().requires_grad_(False)
        # else:
        #     raise ValueError(f"bihito representation not found for {id_str.lower()}, please run BiHiTo_private-mix1/scripts/test_pdb_dir.sh first to generate it")
        # first_frame_bihito_repr_dict = bihito_repr_dict[0]
        if "bihito_token_repr" not in protein_data:
            bihito_token_repr = None
        else:
            bihito_token_repr = protein_data['bihito_token_repr']
        return coords.detach(), spectrum.detach(), rotation_matrices.detach(), aatype.detach(), aa_mask.detach(), torch.tensor(esm_s_protein), bihito_token_repr
    @staticmethod
    def load_full_trajectory_data(data_path):
        """
        从features_fouriered_list.pth文件中加载完整的轨迹数据
        用于inference时需要原始轨迹进行评估
        
        Args:
            data_path (str): Path to the .pth file (e.g., atlas_10_features_fouriered_list_test.pth)
        
        Returns:
            dict: 以protein_id为key的字典，每个值包含完整的轨迹数据
                  格式: {protein_id: {'positions': {'x0', 'y0', 'z0', 'x', 'y', 'z'}, 
                                      'num_frames': int, 'amino_acid_types': tensor, ...}}
        """
        data = torch.load(data_path, weights_only=False)
        
        # 构建以protein_id为key的字典
        trajectory_dict = {}
        for protein_data in data:
            protein_id = protein_data['id']
            trajectory_dict[protein_id] = protein_data
        
        return trajectory_dict
    
    @staticmethod
    def reconstruct_trajectory_from_spectrum(protein_data, scaling_factors, num_frames=100):
        """
        从频谱数据重建完整轨迹
        
        Args:
            protein_data (dict): 包含频谱数据的字典，格式同features_fouriered_list.pth中的单个protein数据
            scaling_factors (torch.Tensor): [K] 频谱缩放因子
            num_frames (int): 要重建的帧数
        
        Returns:
            torch.Tensor: [num_frames, num_residues, 3] 重建的轨迹坐标
        """
        # 获取frame 0的坐标
        frame0_positions = torch.tensor(np.stack([
            protein_data['positions']['x0'],
            protein_data['positions']['y0'],
            protein_data['positions']['z0']
        ], axis=-1), dtype=torch.float32)  # [num_residues, 3]
        
        num_residues = frame0_positions.shape[0]
        
        # 从频谱数据重建位移序列
        scaling_factors_np = scaling_factors.cpu().numpy()
        
        # 初始化轨迹数组
        trajectory = np.zeros((num_residues, num_frames, 3))
        
        # 处理x, y, z三个维度
        for dim_idx, dim_key in enumerate(['x', 'y', 'z']):
            spectrum_data = protein_data['positions'][dim_key]
            
            for res_idx in range(num_residues):
                # spectrum_data[res_idx] 是 [K, 2]，其中K是频率分量数，2是实部和虚部
                spectrum_res = spectrum_data[res_idx]  # [K, 2]
                
                # 转换为 [2, K] 格式（实部，虚部）
                data_normalized = np.array([spectrum_res[:, 0], spectrum_res[:, 1]])  # [2, K]
                
                # 反归一化和IFFT
                displacement = ProteinMotionDataset._inverse_normalize_and_ifft(
                    data_normalized, num_frames, scaling_factors_np
                )
                
                trajectory[res_idx, :, dim_idx] = displacement
        
        # 设置frame 0的坐标
        frame0_positions_np = frame0_positions.cpu().numpy()
        trajectory[:, 0, :] = frame0_positions_np
        
        # 将位移累加到初始坐标
        for i in range(1, num_frames):
            trajectory[:, i, :] = frame0_positions_np + trajectory[:, i, :]
        
        # 转换为torch tensor: [num_frames, num_residues, 3]
        trajectory_tensor = torch.from_numpy(trajectory).permute(1, 0, 2).contiguous()
        
        return trajectory_tensor
    
    @staticmethod
    def _inverse_normalize_and_ifft(data_normalized, num_frames, scale):
        """
        反归一化并执行逆傅里叶变换
        
        Args:
            data_normalized: [2, K] 归一化数据（实部和虚部），K是频率分量数
            num_frames: 目标时域长度（帧数T）
            scale: [K] 每个频率分量的缩放因子
        
        Returns:
            重建的位移序列: [num_frames] array
        """
        # 分离实部和虚部
        real_normalized = data_normalized[0, ...]
        imag_normalized = data_normalized[1, ...]
        # 合并实部和虚部
        data_fft = real_normalized + 1j * imag_normalized
        
        # 获取实际的频率分量长度
        actual_freq_len = len(data_fft)
        
        # 只使用对应长度的scale
        scale_used = scale[:actual_freq_len] if len(scale) >= actual_freq_len else scale
        
        # 反归一化
        magnitude = np.abs(data_fft)
        data_fft_denorm = np.where(
            magnitude > 0,
            (magnitude ** 2) * scale_used * (data_fft / magnitude),
            0
        )
        
        # 执行逆FFT，指定输出长度为num_frames
        return np.fft.irfft(data_fft_denorm, n=num_frames)

class ProteinMotionWindowDataset(IterableDataset):
    """
    Iterable dataset that yields per-window samples from STFT windowed spectra.
    Each window uses its own first frame as the anchor.
    """
    def __init__(self, data_path: str, bihito_repr_dir=None):
        super().__init__()
        self.data = torch.load(data_path, weights_only=False)
        self.data = [data for data in self.data if data["positions"].get("x0_windows") is not None]
        self.bihito_repr_dir = bihito_repr_dir

        self._num_windows = None
        self._num_samples = None
        self._compute_num_samples()

    def _build_geometric_edges(self, coords):
        return ProteinMotionDataset._build_geometric_edges(self, coords)

    def _compute_num_samples(self):
        # 预计算窗口样本数量，避免在__len__里反复遍历
        total = 0
        for protein_data in self.data:
            positions = protein_data["positions"]
            x_windows = positions.get("x_windows", [])
            y_windows = positions.get("y_windows", [])
            z_windows = positions.get("z_windows", [])
            if not x_windows or not y_windows or not z_windows:
                continue
            total += len(x_windows)
        self._num_samples = total

    def __len__(self):
        # 返回可迭代数据集的样本数量（窗口数）
        return int(self._num_samples) if self._num_samples is not None else 0

    def __iter__(self):
        for protein_data in self.data:
            positions = protein_data["positions"]
            x_windows = positions.get("x_windows", [])
            y_windows = positions.get("y_windows", [])
            z_windows = positions.get("z_windows", [])
            x0_windows = positions.get("x0_windows", [])
            y0_windows = positions.get("y0_windows", [])
            z0_windows = positions.get("z0_windows", [])
            window_size = positions.get("window_size", None)
            hop_ratio = positions.get("hop_ratio", None)
            id_str = protein_data['id']
            if not x_windows or not y_windows or not z_windows:
                continue

            # # ===== BiHiTo representation =====
            # bihito_repr_path = os.path.join(self.bihito_repr_dir, f"{id_str.lower()}.pt")
            # if os.path.exists(bihito_repr_path):
            #     bihito_repr_dict = torch.load(bihito_repr_path, weights_only=False)
            #     for key, value in bihito_repr_dict.items():
            #         if isinstance(value, torch.Tensor):
            #             bihito_repr_dict[key] = value.detach().requires_grad_(False)
            #         if isinstance(value, dict):
            #             for sub_key, sub_value in value.items():
            #                 if isinstance(sub_value, torch.Tensor):
            #                     value[sub_key] = sub_value.detach().requires_grad_(False)
            # else:
            #     raise ValueError(f"bihito representation not found for {id_str.lower()}, please run BiHiTo_private-mix1/scripts/test_pdb_dir.sh first to generate it")
            
            num_windows = len(x_windows)
            last_spectrum = None
            for w in range(num_windows):
                first_frame_idx = int(w * window_size * hop_ratio)
                spectrum_x_real = [item[:, 0] for item in x_windows[w]]
                spectrum_x_imag = [item[:, 1] for item in x_windows[w]]
                spectrum_y_real = [item[:, 0] for item in y_windows[w]]
                spectrum_y_imag = [item[:, 1] for item in y_windows[w]]
                spectrum_z_real = [item[:, 0] for item in z_windows[w]]
                spectrum_z_imag = [item[:, 1] for item in z_windows[w]]

                spectrum = np.stack(
                    [
                        spectrum_x_real, spectrum_x_imag,
                        spectrum_y_real, spectrum_y_imag,
                        spectrum_z_real, spectrum_z_imag,
                    ],
                    axis=-1,
                )  # [N, F, 6]
                spectrum = torch.tensor(spectrum, dtype=torch.float32).unsqueeze(-1)

                coords = torch.tensor(
                    np.stack([x0_windows[w], y0_windows[w], z0_windows[w]], axis=-1),
                    dtype=torch.float32,
                )
                rotation_matrices = protein_data["rotation_matrices0"].clone().detach().requires_grad_(True)
                aatype = protein_data["amino_acid_types"].clone().detach()
                aa_mask = torch.ones(aatype.shape, dtype=torch.bool)
                esm_s_protein = protein_data["esm"]
                if "bihito_token_repr" not in protein_data:
                    bihito_token_repr = None
                else:
                    bihito_token_repr = protein_data['bihito_token_repr']

                # first_frame_bihito_repr_dict = bihito_repr_dict[first_frame_idx]

                # edge_list, edge_feature = self._build_geometric_edges(coords)
                # graph = data.Graph(
                #     edge_list=edge_list,
                #     edge_feature=edge_feature,
                #     num_node=coords.shape[0],
                #     num_relation=7,
                # )
                yield (
                    # graph.detach(),
                    coords.detach(),
                    spectrum.detach(),
                    rotation_matrices.detach(),
                    aatype.detach(),
                    aa_mask.detach(),
                    torch.tensor(esm_s_protein),
                    bihito_token_repr,
                    last_spectrum.detach() if last_spectrum is not None else None,
                    window_size,
                    hop_ratio,
                )
                last_spectrum = spectrum


class ProteinMotionFFTWindowDataset(IterableDataset):
    """
    Iterable dataset that yields per-window samples from full FFT windows.
    Each window is treated as an independent FFT spectrum (no COLA).
    """
    def __init__(self, data_path: str, bihito_repr_dir=None):
        super().__init__()
        self.data = torch.load(data_path, weights_only=False)
        self.data = [data for data in self.data if data["positions"].get("x_fft_windows") is not None]
        self.bihito_repr_dir = bihito_repr_dir
        self._num_samples = None
        self._compute_num_samples()

    def _compute_num_samples(self):
        total = 0
        for protein_data in self.data:
            positions = protein_data["positions"]
            x_windows = positions.get("x_fft_windows", [])
            if not x_windows:
                continue
            total += len(x_windows)
        self._num_samples = total

    def __len__(self):
        return int(self._num_samples) if self._num_samples is not None else 0

    def __iter__(self):
        for protein_data in self.data:
            positions = protein_data["positions"]
            x_windows = positions.get("x_fft_windows", [])
            y_windows = positions.get("y_fft_windows", [])
            z_windows = positions.get("z_fft_windows", [])
            x0_windows = positions.get("x0_fft_windows", [])
            y0_windows = positions.get("y0_fft_windows", [])
            z0_windows = positions.get("z0_fft_windows", [])
            window_size = positions.get("fft_window_size", None)
            id_str = protein_data["id"]
            if not x_windows or not y_windows or not z_windows:
                continue

            # bihito_repr_path = os.path.join(self.bihito_repr_dir, f"{id_str.lower()}.pt")
            # if os.path.exists(bihito_repr_path):
            #     bihito_repr_dict = torch.load(bihito_repr_path, weights_only=False)
            #     for key, value in bihito_repr_dict.items():
            #         if isinstance(value, torch.Tensor):
            #             bihito_repr_dict[key] = value.detach().requires_grad_(False)
            #         if isinstance(value, dict):
            #             for sub_key, sub_value in value.items():
            #                 if isinstance(sub_value, torch.Tensor):
            #                     value[sub_key] = sub_value.detach().requires_grad_(False)
            # else:
            #     raise ValueError(
            #         f"bihito representation not found for {id_str.lower()}, "
            #         "please run BiHiTo_private-mix1/scripts/test_pdb_dir.sh first to generate it"
            #     )

            num_windows = len(x_windows)
            for w in range(num_windows):
                spectrum_x_real = [item[:, 0] for item in x_windows[w]]
                spectrum_x_imag = [item[:, 1] for item in x_windows[w]]
                spectrum_y_real = [item[:, 0] for item in y_windows[w]]
                spectrum_y_imag = [item[:, 1] for item in y_windows[w]]
                spectrum_z_real = [item[:, 0] for item in z_windows[w]]
                spectrum_z_imag = [item[:, 1] for item in z_windows[w]]

                spectrum = np.stack(
                    [
                        spectrum_x_real, spectrum_x_imag,
                        spectrum_y_real, spectrum_y_imag,
                        spectrum_z_real, spectrum_z_imag,
                    ],
                    axis=-1,
                )  # [N, F, 6]
                spectrum = torch.tensor(spectrum, dtype=torch.float32).unsqueeze(-1)

                coords = torch.tensor(
                    np.stack([x0_windows[w], y0_windows[w], z0_windows[w]], axis=-1),
                    dtype=torch.float32,
                )
                rotation_matrices = protein_data["rotation_matrices0"].clone().detach().requires_grad_(True)
                aatype = protein_data["amino_acid_types"].clone().detach()
                aa_mask = torch.ones(aatype.shape, dtype=torch.bool)
                esm_s_protein = protein_data["esm"]
                first_frame_idx = int(w * window_size) if window_size is not None else 0
                if "bihito_token_repr" not in protein_data:
                    bihito_token_repr = None
                else:
                    bihito_token_repr = protein_data['bihito_token_repr']

                yield (
                    coords.detach(),
                    spectrum.detach(),
                    rotation_matrices.detach(),
                    aatype.detach(),
                    aa_mask.detach(),
                    torch.tensor(esm_s_protein),
                    bihito_token_repr,
                )



def collate_fn(batch):
    """
    Args:
        batch: List of (graph, coords, spectrum, rotation_matrices, aatype, esm_s_protein)
    Returns:
        packed_graph (data.PackedGraph): Packed protein graphs
        coords (torch.Tensor): [total_N, 3] Concatenated coordinates
        spectrum (torch.Tensor): [total_N, 96] Concatenated spectrum
        rotation_matrices (torch.Tensor): [total_N, 3, 3] Concatenated rotation matrices
        aatype (torch.Tensor): [total_N] Concatenated amino acid types
        esm_s_protein (torch.Tensor): [total_N, 1280] Concatenated ESM embeddings
    """
    graphs, coords, spectrum, rotation_matrices, aatype, esm_s_protein = zip(*batch)
    
    # Pack graph structure
    packed_graph = data.PackedGraph.pack(graphs)
    
    # Concatenate other features
    coords = torch.cat(coords, dim=0)
    spectrum = torch.cat(spectrum, dim=0)
    rotation_matrices = torch.cat(rotation_matrices, dim=0)
    aatype = torch.cat(aatype, dim=0)
    esm_s_protein = torch.cat(esm_s_protein, dim=0)
    
    return packed_graph, coords, spectrum, rotation_matrices, aatype, esm_s_protein


# 专用于“单蛋白、无 batch”模式：直接返回样本本身（不做 concat / pack）
def collate_fn_single(batch):
    return batch[0]

# Example usage
if __name__ == "__main__":
    dataset = ProteinMotionDataset(
        data_path="/path/to/features_fouriered_list.pth"
    )
    dataloader = DataLoader(dataset, batch_size=4, shuffle=True, collate_fn=collate_fn)
    for batch in dataloader:
        print(batch)
        break
