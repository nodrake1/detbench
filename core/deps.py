"""可选依赖探测。

框架的核心（数据、指标、分层评测、结果汇总）**只依赖 numpy 和 Pillow**。
torch / mmdet / ultralytics / matplotlib 都是按需加载，缺失时应该给出
明确提示而不是抛一个裸 ImportError。

这个模块把这些探测集中到一处，避免每个适配器各写一份 try/except。
"""

from __future__ import annotations

import functools
from typing import Callable, Optional, Tuple

# 依赖 -> 安装提示
INSTALL_HINT = {
    "torch": "pip install torch",
    "torchvision": "pip install torchvision",
    "mmdet": "pip install mmdet  (DETR 系模型都在 mmdetection 里)",
    "mmengine": "pip install mmengine",
    "mmcv": "pip install mmcv",
    "ultralytics": "pip install ultralytics",
    "matplotlib": "pip install matplotlib",
    "thop": "pip install thop",
    "yaml": "pip install pyyaml  (不装也行，框架内置了 YAML 子集解析器)",
}


@functools.lru_cache(maxsize=None)
def module_version(name: str) -> Optional[str]:
    """返回模块版本号；未安装返回 None。结果被缓存。"""
    try:
        import importlib

        return getattr(importlib.import_module(name), "__version__", "?")
    except Exception:
        return None


@functools.lru_cache(maxsize=None)
def probe(name: str) -> Tuple[bool, Optional[str]]:
    """探测模块是否可用，返回 ``(可用, 错误信息)``。"""
    try:
        import importlib

        importlib.import_module(name)
        return True, None
    except Exception as exc:
        return False, f"{type(exc).__name__}: {exc}"


def require(name: str) -> None:
    """断言某个模块可用，否则抛出带安装提示的 ImportError。"""
    ok, err = probe(name)
    if not ok:
        hint = INSTALL_HINT.get(name, f"pip install {name}")
        raise ImportError(
            f"本功能需要 '{name}'，但当前环境不可用。\n"
            f"    底层错误：{err}\n"
            f"    安装命令：{hint}"
        )


def torch_available() -> bool:
    return probe("torch")[0]


def mmdet_available() -> bool:
    """mmdet + mmengine 是否都可用（DETR 系适配器的前提）。"""
    return probe("mmdet")[0] and probe("mmengine")[0]


def describe_environment() -> dict:
    """把关键依赖的可用性汇总成一个 dict（写进 result.json 便于追溯）。"""
    out = {}
    for name in ("numpy", "torch", "torchvision", "mmdet", "mmengine", "mmcv",
                 "ultralytics", "matplotlib"):
        ok, _ = probe(name)
        if ok:
            out[name] = module_version(name) or "?"
    return out
