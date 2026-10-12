"""torchvision 检测模型适配器（Faster R-CNN / RetinaNet / FCOS ...）。

用途有两个：

1. **当经典基线**：Faster R-CNN (ResNet-50 FPN) 是几乎所有检测论文的必做对比。
2. **当闭环自检**：即使你最终用 YOLO 或 DETR，也可以用这个适配器在
   占位数据集上验证整条评测链路是通的（见 ``models/dummy.py`` 的说明）。

注意坐标反变换：torchvision 检测模型不做 letterbox，它内部用
``GeneralizedRCNNTransform`` 把图缩放到 ``min_size``/``max_size``，
并自动把预测框还原到**原图尺寸**。所以这里同样不需要手动还原。
但**输入必须自己构造为 0~1 的 float RGB tensor**。
"""

from __future__ import annotations

from typing import Any, Dict, Optional

import numpy as np

from detbench.core.io import DetPrediction, DetSample
from detbench.core.registry import DetectorAdapter, register


class TorchvisionAdapter(DetectorAdapter):
    name = "torchvision"

    def __init__(
        self,
        weights: Optional[str] = None,
        arch: str = "fasterrcnn_resnet50_fpn",
        num_classes: int = 2,  # 1 类目标 + 背景
        score_thresh: float = 0.001,  # 同 YOLO：算 mAP 必须很低
        min_size: int = 800,
        max_size: int = 1333,
        device: str = "cuda",
        **kwargs: Any,
    ) -> None:
        super().__init__(device=device, **kwargs)
        import torch
        from torchvision import models as tvm
        from torchvision.models import detection as tvd

        self.torch = torch
        self.arch = arch
        self.min_size = min_size
        self.max_size = max_size
        self.score_thresh = score_thresh

        builder = getattr(tvd, arch, None) or getattr(tvm, arch, None)
        if builder is None:
            raise ValueError(f"torchvision 里找不到模型 {arch}")

        model = builder(weights=None)

        # 把分类头替换成你的类别数
        if hasattr(model, "roi_heads"):  # Faster R-CNN / Mask R-CNN 系
            in_features = model.roi_heads.box_predictor.cls_score.in_features
            model.roi_heads.box_predictor = tvd.faster_rcnn.FastRCNNPredictor(
                in_features, num_classes
            )
        elif arch.startswith("retinanet"):
            from torchvision.models.detection.retinanet import RetinaNetClassificationHead

            num_anchors = model.head.classification_head.num_anchors
            model.head.classification_head = RetinaNetClassificationHead(
                model.backbone.out_channels, num_anchors, num_classes, norm_layer=None
            )
        elif arch.startswith("fcos"):
            from torchvision.models.detection.fcos import FCOSClassificationHead

            num_anchors = model.head.classification_head.num_anchors
            model.head.classification_head = FCOSClassificationHead(
                model.backbone.out_channels, num_anchors, num_classes
            )

        if weights:
            state = torch.load(weights, map_location="cpu")
            state = state.get("model", state.get("state_dict", state))
            model.load_state_dict(state, strict=False)

        # 推理时把后处理的分数阈值调到很低，让评测阶段自己截断
        if hasattr(model, "roi_heads"):
            model.roi_heads.score_thresh = score_thresh
        if hasattr(model, "score_thresh"):
            model.score_thresh = score_thresh

        model.eval()
        self.model = model.to(device)
        self._last_input = f"min{min_size}/max{max_size}"

    def predict(self, sample: DetSample, image_root: str) -> DetPrediction:
        import torch
        from PIL import Image
        from pathlib import Path

        img = Image.open(Path(image_root) / sample.file_name).convert("RGB")
        arr = np.asarray(img, dtype=np.float32) / 255.0

        # 用 as_tensor 而不是 from_numpy。
        #
        # 原因：某些环境下 torch 的 C 扩展与 numpy 的 C API 版本不匹配，
        # 会让 from_numpy 直接抛 `TypeError: expected np.ndarray (got numpy.ndarray)`
        # —— 报错信息看起来像天方夜谭，但会让所有走 numpy 的推理代码全线崩溃。
        # 实测（torch 2.7.1+cu128 + numpy 1.26.4）：
        #     torch.from_numpy(a)  -> FAIL
        #     torch.as_tensor(a)   -> OK
        #     torch.tensor(a)      -> OK
        # as_tensor 本身也是官方推荐的现代写法（能接受 list/ndarray/标量）。
        tensor = torch.as_tensor(arr).permute(2, 0, 1).to(self.device)

        with torch.no_grad():
            out = self.model([tensor])[0]

        boxes = out["boxes"].detach().cpu().numpy().astype(np.float32)
        scores = out["scores"].detach().cpu().numpy().astype(np.float32)
        labels = out["labels"].detach().cpu().numpy().astype(np.int64)

        keep = scores >= self.score_thresh
        boxes, scores, labels = boxes[keep], scores[keep], labels[keep]
        # torchvision 的 label 0 是背景；本框架约定 0-based 前景类别，故减 1
        labels = np.clip(labels - 1, 0, None)

        h, w = arr.shape[:2]
        self._last_input = f"{w}x{h}->min{self.min_size}"
        return DetPrediction(sample.image_id, boxes, scores, labels)

    def describe(self) -> Dict[str, Any]:
        return {
            "method": self.arch,
            "backbone": "resnet50-fpn",
            "input_size": self._last_input,
            "epochs": 0,
            "seed": 0,
        }

    @property
    def torch_model(self) -> Optional[Any]:
        return self.model


@register("torchvision")
def _build_tv(device: str = "cuda", **kwargs: Any) -> TorchvisionAdapter:
    return TorchvisionAdapter(device=device, **kwargs)


register("faster_rcnn")(_build_tv)
register("retinanet")(_build_tv)
