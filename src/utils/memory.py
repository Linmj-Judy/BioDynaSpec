import os
# os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'max_split_size_mb:128'

import torch
import numpy as np
from collections import defaultdict

class MemoryProfiler:
    """显存监控器，记录每个层在前向和反向传播时的显存占用"""
    
    def __init__(self, enabled=True):
        self.enabled = enabled and torch.cuda.is_available()
        self.forward_records = defaultdict(list)  # {layer_name: [{...}]}
        self.backward_records = defaultdict(list)
        self.hooks = []
        self.current_step = 0
        self._forward_mem_before = {}
        self._backward_mem_before = {}
        
    def _get_memory_mb(self):
        """获取当前显存占用（MB）"""
        if not torch.cuda.is_available():
            return 0.0
        torch.cuda.synchronize()
        return torch.cuda.memory_allocated() / 1024**2

    # ----------------------- forward hooks -----------------------
    def _forward_pre_hook(self, name):
        def hook(module, inputs):
            if not self.enabled:
                return
            self._forward_mem_before[name] = self._get_memory_mb()
        return hook

    def _forward_hook(self, name):
        def hook(module, inputs, output):
            if not self.enabled:
                return
            mem_before = self._forward_mem_before.pop(name, None)
            mem_after = self._get_memory_mb()
            if mem_before is None:
                mem_before = mem_after
            self.forward_records[name].append({
                "step": self.current_step,
                "mem_before": mem_before,
                "mem_after": mem_after,
                "mem_delta": mem_after - mem_before,
            })
        return hook

    # ----------------------- backward hooks -----------------------
    def _backward_pre_hook(self, name):
        def hook(module, grad_output):
            if not self.enabled:
                return
            self._backward_mem_before[name] = self._get_memory_mb()
        return hook
    
    def _backward_hook(self, name):
        """反向传播 hook"""
        def hook(module, grad_input, grad_output):
            if not self.enabled:
                return
            mem_before = self._backward_mem_before.pop(name, None)
            mem_after = self._get_memory_mb()
            if mem_before is None:
                mem_before = mem_after
            self.backward_records[name].append({
                'step': self.current_step,
                'mem_before': mem_before,
                'mem_after': mem_after,
                'mem_delta': mem_after - mem_before
            })
        return hook
    
    def register_hooks(self, model):
        """为模型的所有子模块注册 hooks"""
        if not self.enabled:
            return
        
        for name, module in model.named_modules():
            if len(list(module.children())) == 0:  # 只注册叶子模块
                f_pre = module.register_forward_pre_hook(self._forward_pre_hook(name))
                f_post = module.register_forward_hook(self._forward_hook(name))
                b_pre = module.register_full_backward_pre_hook(self._backward_pre_hook(name))
                b_post = module.register_full_backward_hook(self._backward_hook(name))
                self.hooks.append((f_pre, f_post, b_pre, b_post))
    
    def remove_hooks(self):
        """移除所有 hooks"""
        for handles in self.hooks:
            for handle in handles:
                handle.remove()
        self.hooks.clear()
    
    def update_step(self, step):
        """更新当前训练步数"""
        self.current_step = step

    def print_summary(self, file_path=None):
        """打印显存使用摘要"""
        if not self.enabled:
            return

        import numpy as np

        print("\n" + "="*80)
        print("GPU Memory Profiling Summary")
        print("="*80)

        # 前向传播统计
        print("\n[Forward Pass Memory Usage]")
        print("-" * 80)
        print(f"{'Layer Name':<50} {'次数':<6} {'均值 (MB)':<12} {'最大 (MB)':<12} {'最小 (MB)':<12}")
        print("-" * 80)
        forward_stats = []
        for name, records in self.forward_records.items():
            if not records:
                continue
            deltas = [r["mem_delta"] for r in records]
            forward_stats.append((
                name,
                len(deltas),
                float(np.mean(deltas)),
                float(np.max(deltas)),
                float(np.min(deltas))
            ))
        # 按均值排序
        forward_stats.sort(key=lambda x: x[2], reverse=True)
        for name, cnt, avg, maxv, minv in forward_stats:
            print(f"{name:<50} {cnt:<6} {avg:>10.2f}   {maxv:>10.2f}   {minv:>10.2f}")

        # 反向传播统计
        print("\n[Backward Pass Memory Usage]")
        print("-" * 80)
        print(f"{'Layer Name':<50} {'次数':<6} {'均值 (MB)':<12} {'最大 (MB)':<12} {'最小 (MB)':<12}")
        print("-" * 80)
        backward_stats = []
        for name, records in self.backward_records.items():
            if not records:
                continue
            deltas = [r["mem_delta"] for r in records]
            backward_stats.append((
                name,
                len(deltas),
                float(np.mean(deltas)),
                float(np.max(deltas)),
                float(np.min(deltas))
            ))
        # 按均值排序，仅前20
        backward_stats.sort(key=lambda x: x[2], reverse=True)
        for name, cnt, avg, maxv, minv in backward_stats:
            print(f"{name:<50} {cnt:<6} {avg:>10.2f}   {maxv:>10.2f}   {minv:>10.2f}")

        # 总体统计
        print("\n[Overall Statistics]")
        print("-" * 80)
        if self.forward_records:
            total_forward_delta = sum(
                sum(r['mem_delta'] for r in records)
                for records in self.forward_records.values()
            )
            total_forward_steps = sum(
                len(records) for records in self.forward_records.values()
            )
            print(f"Total Forward Memory Delta: {total_forward_delta:.2f} MB over {total_forward_steps} layer records")
        if self.backward_records:
            total_backward_delta = sum(
                sum(r['mem_delta'] for r in records)
                for records in self.backward_records.values()
            )
            total_backward_steps = sum(
                len(records) for records in self.backward_records.values()
            )
            print(f"Total Backward Memory Delta: {total_backward_delta:.2f} MB over {total_backward_steps} layer records")
        print("="*80 + "\n")

        # 如果指定了文件路径，也保存到文件
        if file_path:
            with open(file_path, 'w') as f:
                f.write("GPU Memory Profiling Summary\n")
                f.write("="*80 + "\n\n")

                f.write("[Forward Pass Memory Usage]\n")
                f.write("-" * 80 + "\n")
                f.write(f"{'Layer Name':<50} {'次数':<6} {'均值 (MB)':<12} {'最大 (MB)':<12} {'最小 (MB)':<12}\n")
                f.write("-" * 80 + "\n")
                for name, cnt, avg, maxv, minv in forward_stats:
                    f.write(f"{name:<50} {cnt:<6} {avg:>10.2f}   {maxv:>10.2f}   {minv:>10.2f}\n")
                
                f.write("\n[Backward Pass Memory Usage]\n")
                f.write("-" * 80 + "\n")
                f.write(f"{'Layer Name':<50} {'次数':<6} {'均值 (MB)':<12} {'最大 (MB)':<12} {'最小 (MB)':<12}\n")
                f.write("-" * 80 + "\n")
                for name, cnt, avg, maxv, minv in backward_stats:
                    f.write(f"{name:<50} {cnt:<6} {avg:>10.2f}   {maxv:>10.2f}   {minv:>10.2f}\n")

                f.write("\n[Overall Statistics]\n")
                f.write("-" * 80 + "\n")
                if self.forward_records:
                    f.write(f"Total Forward Memory Delta: {total_forward_delta:.2f} MB over {total_forward_steps} layer records\n")
                if self.backward_records:
                    f.write(f"Total Backward Memory Delta: {total_backward_delta:.2f} MB over {total_backward_steps} layer records\n")
                f.write("="*80 + "\n")
