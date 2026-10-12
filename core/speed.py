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

    #: 测量是否可信。计时器精度有限（Windows 上 time.perf_counter 约 100ns
    #: 量级），如果单次前向快到接近这个量级，算出的 FPS 会是个天文数字
    #: （实测见过 166666.7 FPS），放进论文表格只会闹笑话。
    reliable: bool = True
    warning: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


def cuda_is_usable(device: str) -> bool:
    """判断 ``device`` 是否真的能用 CUDA。

    ⚠️ 光看 ``device.startswith("cuda")`` 不够。**CPU-only 版的 torch
    里 ``torch.cuda`` 的一些函数根本不存在**（不是返回 False，而是
    直接抛 ``AttributeError: module 'torch._C' has no attribute
    '_cuda_resetPeakMemoryStats'``）。

    实测：``torch 2.14.1+cpu`` 上调用 ``torch.cuda.reset_peak_memory_stats()``
    就是这个报错。而 CPU-only 安装非常常见 —— CI、无独显的笔记本、
    只用 CPU 推理的部署环境都是。所以必须显式判断，不能只看设备字符串。
    """
    if not device.startswith("cuda"):
        return False
    try:
        import torch

        return bool(torch.cuda.is_available()) and hasattr(
            torch.cuda, "reset_peak_memory_stats"
        )
    except Exception:
        return False


def _sync(device: str) -> None:
    """CUDA 是异步的，不 synchronize 会测出接近 0 的"假延迟"。

    测速本身失败不该让整个评测崩掉，所以这里吞掉异常。
    """
    if not cuda_is_usable(device):
        return
    try:
        import torch

        torch.cuda.synchronize()
    except Exception:
        pass


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

    Notes
    -----
    显存统计是**尽力而为**：CPU-only 的 torch 构建、或 CUDA 不可用时，
    ``peak_vram_mb`` 记为 0，但延迟与 FPS 照常测量。
    """
    use_cuda = cuda_is_usable(device)
    if use_cuda:
        try:
            import torch

            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats()
        except Exception:
            use_cuda = False

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
    if use_cuda:
        try:
            import torch

            peak_mb = torch.cuda.max_memory_allocated() / (1024 ** 2)
        except Exception:
            peak_mb = 0.0  # 显存统计尽力而为，拿不到就记 0

    params_m = 0.0
    if model is not None:
        try:
            params_m = sum(p.numel() for p in model.parameters()) / 1e6
        except Exception:
            params_m = 0.0

    # 合理性检查：单次前向快到接近计时器分辨率时，FPS 没有意义。
    # 阈值取 50 微秒 —— 真实检测模型在 CPU 上也不可能快到这一步，
    # 出现这种数字只可能是"模型几乎没干活"或"计时坏了"。
    reliable = True
    warning = ""
    if median_ms < 0.05:
        reliable = False
        warning = (
            f"单次前向仅 {median_ms * 1000:.1f} 微秒，已接近计时器分辨率，"
            f"由此算出的 {1000.0 / max(median_ms, 1e-9):.0f} FPS 不可信。"
            "常见原因：模型几乎不做计算（如占位检测器），或推理被缓存/短路。"
            "请勿把这个 FPS 写进论文。"
        )

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
        reliable=reliable,
        warning=warning,
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
