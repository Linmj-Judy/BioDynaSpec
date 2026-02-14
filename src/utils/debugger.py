
import os
from typing import Any, Iterable

import torch


def ipdb_in_parallel() -> None:
    """多卡分布式环境下的 ipdb 调试辅助。只在 LOCAL_RANK=0 进入 ipdb，其它 rank 在 barrier 等待。"""
    try:
        import ipdb

        local_rank = int(os.environ.get("LOCAL_RANK", 0))
        world_size = int(os.environ.get("WORLD_SIZE", 1))
        if local_rank == 0:
            print("======== Entering ipdb.set_trace() on LOCAL_RANK=0 ========")
            ipdb.set_trace()
        else:
            print(f"Rank {local_rank}: Waiting at ipdb barrier...")
            # 等待主卡调完之后其他才继续
            import torch.distributed as dist

            if dist.is_available() and dist.is_initialized():
                dist.barrier()
            else:
                # 不是分布式环境，直接pass
                pass
    except Exception as e:
        print(f"[ipdb debug hook] Exception: {e}")


def _iter_tensors(x: Any) -> Iterable[torch.Tensor]:
    """递归遍历嵌套结构中的所有 Tensor。"""
    if isinstance(x, torch.Tensor):
        yield x
    elif isinstance(x, (list, tuple)):
        for v in x:
            yield from _iter_tensors(v)
    elif isinstance(x, dict):
        for v in x.values():
            yield from _iter_tensors(v)


def register_nan_hooks_for_module(
    module: torch.nn.Module,
    module_name: str = "model",
    enable_backward: bool = True,
    only_leaf: bool = True,
    enable_gradient_stats: bool = False,
    grad_explosion_threshold: float = 1e3,
    grad_vanishing_threshold: float = 1e-7,
) -> None:
    """
    在整个 Module 上注册 NaN 监控钩子，用于排查"从哪一层开始出现 NaN"。

    - 前向：检查输入 / 输出是否包含 NaN/Inf
    - 反向（可选）：检查 grad_input / grad_output 是否包含 NaN/Inf
    - 梯度统计（可选）：检查梯度范围、均值，检测梯度爆炸/消失

    检测到 NaN 时，会打印：
        [NaN DETECT][forward] rank=R path=xxx has_nan_input=True has_nan_output=False

    Args:
        module: 需要监控的顶层模块（例如 ProteinSpectrumFAR 或 LightningModule 本身）
        module_name: 日志中展示的顶层模块名前缀
        enable_backward: 是否同时监控反向传播中的 NaN
        only_leaf: 仅在 leaf module 上挂 hook，减少日志量
        enable_gradient_stats: 是否启用详细的梯度统计（范围、均值、爆炸/消失检测）
        grad_explosion_threshold: 梯度爆炸阈值（默认1e3）
        grad_vanishing_threshold: 梯度消失阈值（默认1e-7）
    """
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    world_size = int(os.environ.get("WORLD_SIZE", 1))

    state = {"forward_triggered": False, "backward_triggered": False}

    def _compute_tensor_stats(tensor: torch.Tensor) -> dict:
        """计算张量的统计信息"""
        if tensor.numel() == 0:
            return {"min": 0.0, "max": 0.0, "mean": 0.0, "has_nan": False, "has_inf": False}
        
        abs_tensor = tensor.abs()
        stats = {
            "min": abs_tensor.min().item(),
            "max": abs_tensor.max().item(),
            "mean": abs_tensor.mean().item(),
            "has_nan": torch.isnan(tensor).any().item(),
            "has_inf": torch.isinf(tensor).any().item(),
        }
        return stats

    def make_forward_hook(path: str):
        def _hook(mod, inputs, outputs):
            if state["forward_triggered"]:
                return
            try:
                # 检查NaN/Inf
                has_nan_in = any(
                    torch.isnan(t).any().item() for t in _iter_tensors(inputs) if isinstance(t, torch.Tensor) and t.numel() > 0
                )
                has_inf_in = any(
                    torch.isinf(t).any().item() for t in _iter_tensors(inputs) if isinstance(t, torch.Tensor) and t.numel() > 0
                )
                has_nan_out = any(
                    torch.isnan(t).any().item() for t in _iter_tensors(outputs) if isinstance(t, torch.Tensor) and t.numel() > 0
                )
                has_inf_out = any(
                    torch.isinf(t).any().item() for t in _iter_tensors(outputs) if isinstance(t, torch.Tensor) and t.numel() > 0
                )
                
                if has_nan_in or has_inf_in or has_nan_out or has_inf_out:
                    state["forward_triggered"] = True
                    details = []
                    if has_nan_in:
                        details.append("NaN_in")
                    if has_inf_in:
                        details.append("Inf_in")
                    if has_nan_out:
                        details.append("NaN_out")
                    if has_inf_out:
                        details.append("Inf_out")
                    
                    # 如果启用详细统计，添加张量信息
                    info_parts = [
                        f"rank={local_rank}/{world_size}",
                        f"module={module_name}.{path}",
                        f"异常={'、'.join(details)}"
                    ]
                    
                    if enable_gradient_stats:
                        # 添加输入输出张量的形状和类型信息
                        input_tensors = [t for t in _iter_tensors(inputs) if isinstance(t, torch.Tensor)]
                        output_tensors = [t for t in _iter_tensors(outputs) if isinstance(t, torch.Tensor)]
                        
                        if input_tensors:
                            input_info = [f"输入[{i}]: shape={t.shape}, dtype={t.dtype}" 
                                        for i, t in enumerate(input_tensors)]
                            info_parts.append(" | ".join(input_info))
                        if output_tensors:
                            output_info = [f"输出[{i}]: shape={t.shape}, dtype={t.dtype}" 
                                         for i, t in enumerate(output_tensors)]
                            info_parts.append(" | ".join(output_info))
                    
                    print(f"[NaN DETECT][forward] {' | '.join(info_parts)}")
            except Exception as e:
                print(f"[NaN forward hook error] module={path} err={e}")

        return _hook

    def make_backward_hook(path: str):
        def _hook(mod, grad_input, grad_output):
            if state["backward_triggered"]:
                return
            try:
                # 检查NaN/Inf
                has_nan_in = any(
                    torch.isnan(t).any().item() for t in _iter_tensors(grad_input) 
                    if isinstance(t, torch.Tensor) and t.numel() > 0
                )
                has_inf_in = any(
                    torch.isinf(t).any().item() for t in _iter_tensors(grad_input) 
                    if isinstance(t, torch.Tensor) and t.numel() > 0
                )
                has_nan_out = any(
                    torch.isnan(t).any().item() for t in _iter_tensors(grad_output) 
                    if isinstance(t, torch.Tensor) and t.numel() > 0
                )
                has_inf_out = any(
                    torch.isinf(t).any().item() for t in _iter_tensors(grad_output) 
                    if isinstance(t, torch.Tensor) and t.numel() > 0
                )
                
                if has_nan_in or has_inf_in or has_nan_out or has_inf_out or enable_gradient_stats:
                    # 如果启用梯度统计，即使没有NaN也要检查梯度爆炸/消失
                    info_parts = [
                        f"rank={local_rank}/{world_size}",
                        f"module={module_name}.{path}"
                    ]
                    
                    if has_nan_in or has_inf_in or has_nan_out or has_inf_out:
                        state["backward_triggered"] = True
                        details = []
                        if has_nan_in:
                            details.append("NaN_grad_in")
                        if has_inf_in:
                            details.append("Inf_grad_in")
                        if has_nan_out:
                            details.append("NaN_grad_out")
                        if has_inf_out:
                            details.append("Inf_grad_out")
                        info_parts.append(f"异常={'、'.join(details)}")
                    
                    # 梯度统计信息
                    if enable_gradient_stats:
                        grad_info = []
                        
                        # 输入梯度统计
                        for i, grad in enumerate(_iter_tensors(grad_input)):
                            if isinstance(grad, torch.Tensor) and grad.numel() > 0:
                                stats = _compute_tensor_stats(grad)
                                grad_str = f"grad_in[{i}]: shape={grad.shape}, range={stats['min']:.2e}-{stats['max']:.2e}, mean={stats['mean']:.2e}"
                                
                                if stats['max'] > grad_explosion_threshold:
                                    grad_str += ", 梯度爆炸!"
                                    state["backward_triggered"] = True
                                if stats['max'] < grad_vanishing_threshold and stats['max'] > 0:
                                    grad_str += ", 梯度消失!"
                                    state["backward_triggered"] = True
                                
                                grad_info.append(grad_str)
                        
                        # 输出梯度统计
                        for i, grad in enumerate(_iter_tensors(grad_output)):
                            if isinstance(grad, torch.Tensor) and grad.numel() > 0:
                                stats = _compute_tensor_stats(grad)
                                grad_str = f"grad_out[{i}]: shape={grad.shape}, range={stats['min']:.2e}-{stats['max']:.2e}, mean={stats['mean']:.2e}"
                                
                                if stats['max'] > grad_explosion_threshold:
                                    grad_str += ", 梯度爆炸!"
                                    state["backward_triggered"] = True
                                if stats['max'] < grad_vanishing_threshold and stats['max'] > 0:
                                    grad_str += ", 梯度消失!"
                                    state["backward_triggered"] = True
                                
                                grad_info.append(grad_str)
                        
                        if grad_info:
                            info_parts.append(" | ".join(grad_info))
                    
                    # 只有在检测到异常时才打印
                    if state["backward_triggered"]:
                        print(f"[NaN DETECT][backward] {' | '.join(info_parts)}")
            except Exception as e:
                print(f"[NaN backward hook error] module={path} err={e}")

        return _hook

    for name, submodule in module.named_modules():
        # 顶层 module 自己也会出现在 named_modules() 中
        if only_leaf and any(submodule.children()):
            continue
        path = name or submodule.__class__.__name__
        submodule.register_forward_hook(make_forward_hook(path))
        if enable_backward:
            # PyTorch >=1.9 推荐使用 register_full_backward_hook
            submodule.register_full_backward_hook(make_backward_hook(path))

    print(
        f"[NaN hooks] Registered NaN forward/backward hooks on '{module_name}' "
        f"(rank={local_rank}, world_size={world_size}, "
        f"gradient_stats={'enabled' if enable_gradient_stats else 'disabled'})"
    )