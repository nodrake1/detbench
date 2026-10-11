"""模型适配器注册表。

本框架不绑定任何检测库。任何检测器只要实现 :class:`DetectorAdapter`
的两个方法，就能接入统一评测。

这样做的理由：你未来一定会同时用 YOLO（ultralytics）、
DETR 系（mmdetection / detectron2）和自研模型。如果评测代码为每个库
写一份，三份一定会逐渐不一致，而"对比不公平"是审稿人最容易打死的点。
"""

from __future__ import annotations

import abc
from typing import Any, Callable, Dict, List, Optional, Sequence

import numpy as np

from detbench.core.io import DetPrediction, DetSample


class DetectorAdapter(abc.ABC):
    """检测器适配器接口。

    子类必须实现 :meth:`predict` 与 :meth:`describe`。
    """

    #: 注册名，用于配置文件里的 ``model.type``
    name: str = "base"

    def __init__(self, device: str = "cuda", **kwargs: Any) -> None:
        self.device = device
        self.kwargs = kwargs

    @abc.abstractmethod
    def predict(self, sample: DetSample, image_root: str) -> DetPrediction:
        """对单张图做推理，返回 **原图坐标系下** 的 xyxy 预测。

        实现要点：如果模型内部做了 letterbox / resize，**必须把框映射回
        原图尺寸**。忘记反变换是检测评测中最常见的 bug，会让 mAP 掉一半
        且很难发现（指标只是"偏低"，不报错）。
        """

    @abc.abstractmethod
    def describe(self) -> Dict[str, Any]:
        """返回模型元信息：method / backbone / input_size / epochs / seed。"""

    def predict_batch(
        self, samples: Sequence[DetSample], image_root: str
    ) -> Dict[int, DetPrediction]:
        """批量推理。默认逐个调用 :meth:`predict`；子类可覆写以启用真批处理。"""
        return {s.image_id: self.predict(s, image_root) for s in samples}

    @property
    def torch_model(self) -> Optional[Any]:
        """返回底层 ``nn.Module``（若可用），用于统计参数量与显存。"""
        return None

    def warmup(self, sample: DetSample, image_root: str, iters: int = 3) -> None:
        """速度评测前的预热钩子。"""
        for _ in range(iters):
            self.predict(sample, image_root)


# --------------------------------------------------------------------------- #
# 注册表
# --------------------------------------------------------------------------- #

_REGISTRY: Dict[str, Callable[..., DetectorAdapter]] = {}


def register(
    name: str,
) -> Callable[[Callable[..., DetectorAdapter]], Callable[..., DetectorAdapter]]:
    """装饰器：把一个适配器工厂注册到全局表。

    Usage::

        @register("yolov8")
        def build_yolov8(weights, device="cuda", **kw):
            return YOLOv8Adapter(weights, device=device, **kw)
    """

    def deco(fn: Callable[..., DetectorAdapter]) -> Callable[..., DetectorAdapter]:
        if name in _REGISTRY:
            raise KeyError(f"模型名 '{name}' 已注册")
        _REGISTRY[name] = fn
        return fn

    return deco


def build(spec: Dict[str, Any], device: str = "cuda") -> DetectorAdapter:
    """按配置字典构建适配器。

    ``spec`` 形如::

        {"type": "yolov8", "weights": "runs/exp1/weights/best.pt", "conf": 0.001}
    """
    spec = dict(spec)
    mtype = spec.pop("type", None)
    if mtype is None:
        raise ValueError("模型配置缺少 'type' 字段")
    if mtype not in _REGISTRY:
        raise KeyError(f"未注册的模型类型 '{mtype}'；已注册：{sorted(_REGISTRY)}")
    return _REGISTRY[mtype](device=device, **spec)


def available() -> List[str]:
    return sorted(_REGISTRY)
