"""Ultralytics YOLO 适配器（v5 / v8 / v11 通用）。

关键实现点
----------
1. **坐标反变换**：ultralytics 的 ``results[0].boxes.xyxy`` 已经在原图坐标系下
   （它内部处理了 letterbox 反变换），所以这里不需要手动还原。
   但如果你换成自己写的推理管线（ONNX / TensorRT），**必须自己反变换**。
2. **低置信度阈值**：算 mAP 时必须传 ``conf=0.001``。默认的 ``conf=0.25``
   会砍掉大量低分预测，使 precision 虚高、mAP 失真。这是最常见的评测错误。
3. **max_det 设大一些**（如 300）：让评测阶段自己做 top-k 截断，
   保持各方法可比。
"""

from __future__ import annotations

from typing import Any, Dict, Optional

import numpy as np

from detbench.core.io import DetPrediction, DetSample
from detbench.core.registry import DetectorAdapter, register


class YOLOv8Adapter(DetectorAdapter):
    name = "yolov8"

    def __init__(
        self,
        weights: str,
        imgsz: int = 640,
        conf: float = 0.001,
        iou: float = 0.7,
        max_det: int = 300,
        device: str = "cuda",
        half: bool = False,
        **kwargs: Any,
    ) -> None:
        super().__init__(device=device, **kwargs)
        from ultralytics import YOLO  # 延迟导入，缺库时不影响其他适配器

        self.model = YOLO(weights)
        self.weights = weights
        self.imgsz = imgsz
        self.conf = conf
        self.iou = iou
        self.max_det = max_det
        self.half = half

    def predict(self, sample: DetSample, image_root: str) -> DetPrediction:
        from pathlib import Path

        img_path = str(Path(image_root) / sample.file_name)
        res = self.model.predict(
            img_path,
            imgsz=self.imgsz,
            conf=self.conf,
            iou=self.iou,
            max_det=self.max_det,
            device=self.device,
            half=self.half,
            verbose=False,
        )[0]

        if res.boxes is None or len(res.boxes) == 0:
            return DetPrediction(
                sample.image_id,
                np.zeros((0, 4), np.float32),
                np.zeros((0,), np.float32),
                np.zeros((0,), np.int64),
            )

        boxes = res.boxes.xyxy.detach().cpu().numpy().astype(np.float32)
        scores = res.boxes.conf.detach().cpu().numpy().astype(np.float32)
        labels = res.boxes.cls.detach().cpu().numpy().astype(np.int64)
        return DetPrediction(sample.image_id, boxes, scores, labels)

    def describe(self) -> Dict[str, Any]:
        return {
            "method": f"YOLO({self.weights.replace(chr(92), '/').split('/')[-1]})",
            "backbone": "-",
            "input_size": str(self.imgsz),
            "epochs": 0,
            "seed": 0,
        }

    @property
    def torch_model(self) -> Optional[Any]:
        try:
            return self.model.model
        except Exception:
            return None

    def warmup(self, sample: DetSample, image_root: str, iters: int = 3) -> None:
        # ultralytics 首次调用会做内部 warmup，这里再跑几次确保显存分配稳定
        for _ in range(max(1, iters)):
            self.predict(sample, image_root)


@register("yolov8")
def _build_yolov8(device: str = "cuda", **kwargs: Any) -> YOLOv8Adapter:
    return YOLOv8Adapter(device=device, **kwargs)


# 别名，方便配置里直接写 yolov11 / yolov5
register("yolov11")(_build_yolov8)
register("yolov5")(_build_yolov8)
