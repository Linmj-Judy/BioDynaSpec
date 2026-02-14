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

import os

import torch
import torch.distributed as dist
import torch.nn.functional as F


def distributed_available() -> bool:
    return torch.distributed.is_available() and torch.distributed.is_initialized()


class DistWrapper:
    def __init__(self) -> None:
        self.rank = int(os.environ.get("RANK", 0))
        self.local_rank = int(os.environ.get("LOCAL_RANK", 0))
        self.local_world_size = int(os.environ.get("LOCAL_WORLD_SIZE", 1))
        self.world_size = int(os.environ.get("WORLD_SIZE", 1))
        self.num_nodes = int(self.world_size // self.local_world_size)
        self.node_rank = int(self.rank // self.local_world_size)

    def all_gather_object(self, obj, group=None):
        """Function to gather objects from several distributed processes.
        It is now only used by sync metrics in logger due to security reason.
        """
        if self.world_size > 1 and distributed_available():
            with torch.no_grad():
                obj_list = [None for _ in range(self.world_size)]
                torch.distributed.all_gather_object(obj_list, obj, group=group)
                return obj_list
        else:
            return [obj]


DIST_WRAPPER = DistWrapper()


def traverse_and_aggregate(dict_list, aggregation_func=None):
    """Traverse list of dicts and merge into a single dict with leaf values joined to list."""
    merged_dict = {}
    all_keys = set().union(*dict_list)
    for key in all_keys:
        agg_value = [m[key] for m in dict_list if key in m]

        if isinstance(agg_value[0], dict):
            merged_dict[key] = traverse_and_aggregate(
                agg_value, aggregation_func=aggregation_func
            )
        else:
            if aggregation_func is not None:
                agg_value = aggregation_func(agg_value)
            merged_dict[key] = agg_value

    return merged_dict


def gather_and_merge(metrics, aggregation_func=None):
    """Gather metrics from ddp workers and aggregate leaf metrics."""
    gathered_metrics = DIST_WRAPPER.all_gather_object(metrics)  # list of metrics
    merged_metrics = traverse_and_aggregate(gathered_metrics, aggregation_func)
    return merged_metrics

def _gather_along_dim(input, dim, group, original_size=None):
    """
    辅助函数：沿着指定维度 dim 进行 all_gather 并拼接
    
    Args:
        input: 输入 tensor
        dim: 要 gather 的维度
        group: 通信组
        original_size: 原始大小（如果之前进行了 padding，需要 truncate 回原始大小）
    """
    world_size = dist.get_world_size(group)
    
    # 1. 先收集所有 rank 的 chunk 大小（处理不同 rank chunk 大小不同的情况）
    local_chunk_size = input.shape[dim]
    chunk_sizes_tensor = torch.tensor([local_chunk_size], device=input.device, dtype=torch.long)
    chunk_sizes_list = [torch.zeros_like(chunk_sizes_tensor) for _ in range(world_size)]
    dist.all_gather(chunk_sizes_list, chunk_sizes_tensor, group=group)
    chunk_sizes = [size.item() for size in chunk_sizes_list]
    max_chunk_size = max(chunk_sizes)
    
    # 2. 如果不同 rank 的 chunk 大小不同，需要先 pad 到相同大小
    if local_chunk_size < max_chunk_size:
        pad_size = max_chunk_size - local_chunk_size
        pad_dims = [0] * (input.ndim * 2)
        pad_dims[dim * 2 + 1] = pad_size  # 在 dim 维度的末尾 pad
        input_padded = F.pad(input, pad_dims, mode='constant', value=0)
    else:
        input_padded = input
    
    # 3. 准备接收 buffer（现在所有 rank 的大小都相同）
    tensor_list = [torch.empty_like(input_padded) for _ in range(world_size)]
    
    # 4. 执行通信
    dist.all_gather(tensor_list, input_padded, group=group)
    
    # 5. 拼接
    result = torch.cat(tensor_list, dim=dim)
    
    # 6. 根据实际 chunk 大小截取（去除 padding）
    # 计算实际的总长度
    actual_total_size = sum(chunk_sizes)
    if result.shape[dim] != actual_total_size:
        slice_obj = [slice(None)] * result.ndim
        slice_obj[dim] = slice(0, actual_total_size)
        result = result[tuple(slice_obj)]
    
    # 7. 如果提供了 original_size，进一步 truncate（处理 scatter 时的 padding）
    if original_size is not None and result.shape[dim] != original_size:
        slice_obj = [slice(None)] * result.ndim
        slice_obj[dim] = slice(0, original_size)
        result = result[tuple(slice_obj)]
    
    return result

def _reduce_scatter_along_dim(input, dim, group):
    """
    辅助函数：沿着指定维度 dim 进行切分 (Scatter)
    
    如果维度不能被 world_size 整除，会先 pad 到可整除的大小，然后切分。
    返回 (chunk, original_size)，其中 original_size 用于后续 gather 时恢复原始大小。
    """
    world_size = dist.get_world_size(group)
    rank = dist.get_rank(group)
    
    original_size = input.shape[dim]
    padded_size = ((original_size + world_size - 1) // world_size) * world_size  # 向上取整到 world_size 的倍数
    
    # 如果需要进行 padding
    if padded_size != original_size:
        pad_size = padded_size - original_size
        # 构造 padding 参数：在 dim 维度的末尾 pad
        pad_dims = [0] * (input.ndim * 2)
        pad_dims[dim * 2 + 1] = pad_size  # 在 dim 维度的末尾 pad
        input = F.pad(input, pad_dims, mode='constant', value=0)
    
    # torch.chunk 会把 tensor 切分成 tensor tuple
    chunks = input.chunk(world_size, dim=dim)
    
    # 拿出属于当前 rank 的那一份
    # contiguous() 很重要，否则后续 view/reshape 操作可能会报错
    chunk = chunks[rank].contiguous()
    
    # 返回 chunk 和原始大小信息
    return chunk, original_size

class ScatterToSequenceParallelRegion(torch.autograd.Function):
    """
    前向传播：把完整 Tensor 切分给各个 GPU (Scatter)
    反向传播：把各个 GPU 的梯度收集起来拼成完整梯度 (Gather)
    
    如果维度不能被 world_size 整除，会自动进行 padding 和 truncation。
    """
    @staticmethod
    def forward(ctx, input, group=None, dim=-2):
        ctx.group = group
        ctx.dim = dim
        chunk, original_size = _reduce_scatter_along_dim(input, dim, group)
        # 保存原始大小，用于 backward 时 truncate
        ctx.original_size = original_size
        return chunk

    @staticmethod
    def backward(ctx, grad_output):
        # 反向时，把大家的梯度 Gather 起来
        # 使用保存的 original_size 来 truncate
        return _gather_along_dim(grad_output, ctx.dim, ctx.group, ctx.original_size), None, None

class GatherFromSequenceParallelRegion(torch.autograd.Function):
    """
    前向传播：把各个 GPU 的 Tensor 收集起来拼成完整 Tensor (Gather)
    反向传播：把完整梯度切分给各个 GPU (Scatter)
    
    如果之前进行了 padding，会自动 truncate 回原始大小。
    注意：需要传入 original_size 参数（从 scatter 操作中获取）。
    """
    @staticmethod
    def forward(ctx, input, group=None, dim=-2, original_size=None):
        ctx.group = group
        ctx.dim = dim
        # 如果提供了 original_size，保存它用于 backward
        if original_size is None:
            # 如果没有提供，尝试从输入形状推断（假设所有 rank 的 chunk 大小相同）
            world_size = dist.get_world_size(group)
            original_size = input.shape[dim] * world_size
        ctx.original_size = original_size
        return _gather_along_dim(input, dim, group, original_size)

    @staticmethod
    def backward(ctx, grad_output):
        # 反向时，把梯度 Scatter 回去
        chunk, _ = _reduce_scatter_along_dim(grad_output, ctx.dim, ctx.group)
        # 必须返回4个梯度，对应forward的4个输入：input, group, dim, original_size
        # group, dim, original_size 都不需要梯度，所以返回 None
        return chunk, None, None, None