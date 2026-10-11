"""速度与资源占用评测。

无人机场景里 FPS 和显存是硬指标，不是可选项 —— 审稿人一定会问
"这能上机吗"。所以速度评测必须和精度评测同等对待，并且用同一套协议：

    预热 -> 同步 -> 多次测量 -> 取中位数（不是均值，避免首个迭代的抖动）

三个容易做错的地方
------------------
1. **不预热**：首次前向包含 CUDA kernel 编译、显存分配、cudnn autotune，
   不预热会把速度低估 2~5 倍。
2. **不同步**：CUDA 是异步的，不 ``torch.cuda.synchronize()`` 会测出
   接近 0 的"假延迟"。
3. **只测一次**：GPU 频率抖动会让单次测量偏差 10% 以上，必须多次取中位数。
"""

from __future__ import annotations

import statistics
import time
from dataclasses import asdict, dataclass, field
from typing import Any, Callable, Dict, List, Optional

import numpy as np

# 依赖探测集中放在 core/deps.py，避免每处各写一份 try/except
from detbench.core.deps import torch_available  # noqa: F401  (对外转出)


@dataclass
class SpeedResult:
    """一次速度评测的结果。"""

    fps: float = 0.0
    latency_ms: float = 0.0
    latency_ms_std: float = 0.0
    peak_vram_mb: float = 0.0
    params_m: float = 0.0
    flops_g: float = 0.0
    batch_size: int = 1
    input_size: str = ""
    device: str = ""
    warmup: int = 0
    iters: int = 0
    notes: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


def _sync(device: str) -> None:
    if device.startswith("cuda"):
        import torch

        torch.cuda.synchronize()


def measure_speed(
    forward_fn: Callable[[], Any],
    input_size: str = "",
    batch_size: int = 1,
    device: str = "cuda",
    warmup: int = 20,
    iters: int = 100,
    model: Any = None,
) -> SpeedResult:
    """测量 ``forward_fn`` 的延迟与吞吐。

    Parameters
    ----------
    forward_fn : 无参可调用对象，执行一次完整前向（含预处理和后处理，
                 这样测出来的才是端到端部署延迟，而不是纯网络时间）
    warmup : 预热次数
    iters : 测量次数
    model : 可选，用于统计参数量与峰值显存
    """
    if device.startswith("cuda"):
        import torch

        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()

    # ---- 预热 ----
    for _ in range(max(0, warmup)):
        forward_fn()
    _sync(device)

    # ---- 测量 ----
    latencies: List[float] = []
    for _ in range(max(1, iters)):
        t0 = time.perf_counter()
        forward_fn()
        _sync(device)
        latencies.append((time.perf_counter() - t0) * 1000.0)

    median_ms = statistics.median(latencies)
    std_ms = statistics.pstdev(latencies) if len(latencies) > 1 else 0.0

    peak_mb = 0.0
    if device.startswith("cuda"):
        import torch

        peak_mb = torch.cuda.max_memory_allocated() / (1024 ** 2)

    params_m = 0.0
    if model is not None:
        try:
            params_m = sum(p.numel() for p in model.parameters()) / 1e6
        except Exception:
            params_m = 0.0

    return SpeedResult(
        fps=(1000.0 / median_ms) * batch_size if median_ms > 0 else 0.0,
        latency_ms=median_ms,
        latency_ms_std=std_ms,
        peak_vram_mb=peak_mb,
        params_m=params_m,
        batch_size=batch_size,
        input_size=input_size,
        device=device,
        warmup=warmup,
        iters=iters,
    )


def count_flops(model: Any, input_shape: tuple, device: str = "cuda") -> float:
    """统计 FLOPs（单位 G）。需要 ``thop`` 或 ``fvcore``；缺依赖时返回 0。

    注意：FLOPs 与真实延迟的相关性有限（受显存带宽、算子实现影响），
    论文里**报告 FPS 比报告 FLOPs 更有说服力**。FLOPs 只作为参考列。
    """
    try:
        import torch
    except Exception:
        return 0.0

    dummy = torch.randn(*input_shape, device=device)
    try:
        from thop import profile  # type: ignore

        macs, _ = profile(model, inputs=(dummy,), verbose=False)
        return float(macs) * 2.0 / 1e9  # MACs -> FLOPs
    except Exception:
        pass
    try:
        from fvcore.nn import FlopCountAnalysis  # type: ignore

        return float(FlopCountAnalysis(model, dummy).total()) / 1e9
    except Exception:
        return 0.0


def estimate_train_time(
    iters_per_epoch: int,
    epochs: int,
    sec_per_iter: float,
    num_gpus: int = 1,
    ddp_efficiency: float = 0.9,
) -> Dict[str, float]:
    """根据实测 ``sec_per_iter`` 估算训练总时长。

    这个函数存在的意义：在动手训练前先算清总账。
    一个典型的坑是"以为两卡能省一半时间" —— 实际 DDP 效率常只有
    0.85~0.95，而且 4060 Ti 这类窄带宽卡在多卡下的收益更差。
    """
    total_iters = iters_per_epoch * epochs
    effective = max(num_gpus * ddp_efficiency, 1e-9)
    total_sec = total_iters * sec_per_iter / effective
    return {
        "total_iters": float(total_iters),
        "total_hours": total_sec / 3600.0,
        "total_gpu_hours": total_sec * num_gpus / 3600.0,
        "hours_per_epoch": total_sec / max(epochs, 1) / 3600.0,
    }
