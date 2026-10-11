"""各检测框架的适配器实现。

每个适配器只需实现两件事：``predict`` 和 ``describe``。
所有预处理差异（letterbox、归一化、坐标反变换）在适配器内部消化，
对外暴露的永远是**原图坐标系下的 xyxy**。

新增一个检测库时，照抄 ``yolov8.py`` 的结构即可。
"""

from .dummy import DummyAdapter  # noqa: F401

__all__ = ["DummyAdapter"]

# 可选依赖：缺库时不影响框架其他部分使用
try:  # pragma: no cover
    from .yolov8 import YOLOv8Adapter  # noqa: F401

    __all__.append("YOLOv8Adapter")
except Exception:  # pragma: no cover
    pass

try:  # pragma: no cover
    from .torchvision_det import TorchvisionAdapter  # noqa: F401

    __all__.append("TorchvisionAdapter")
except Exception:  # pragma: no cover
    pass

try:  # pragma: no cover
    from .mmdet_detr import MMDetAdapter  # noqa: F401

    __all__.append("MMDetAdapter")
except Exception:  # pragma: no cover
    # mmdet / mmengine 未安装时走到这里。这是正常情况，不影响其他适配器。
    pass
