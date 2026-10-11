"""占位检测器：不做任何真实推理，用于端到端打通管线。

作用：在没有真实模型、没有真实数据的时候，验证
"数据加载 -> 评测 -> 分层统计 -> 结果落盘 -> 生成论文表" 整条链路是通的。

它有三种模式，故意设计成可预测的行为，便于验证指标实现是否正确：

- ``mode="perfect"``  ：直接输出 GT。mAP 应接近 1.0（上限检查）。
- ``mode="empty"``    ：什么都不输出。mAP 应为 0.0（下限检查）。
- ``mode="jitter"``   ：把 GT 做小幅平移 + 加入一部分误检。
  mAP 应落在 0 和 1 之间（中间档检查）。
- ``mode="random"``   ：输出随机框。

⚠️ 关于 ``random`` 模式的一个重要事实
-------------------------------------
在 960x640 图像上，随机框命中一个 15px 小目标（IoU>0.5）的概率极低，
所以 ``random`` 的 mAP **合理地就是 0.0000**，不是 bug。

这本身是一个值得记住的结论：**小目标检测任务里，"随机猜"是彻底无效的**，
因此任何非零 mAP 都来自真实的检测能力。这也意味着验证"中间档"行为
必须用 ``jitter`` 模式，而不是 ``random``。
"""

from __future__ import annotations

from typing import Any, Dict, Optional

import numpy as np

from detbench.core.io import DetPrediction, DetSample
from detbench.core.registry import DetectorAdapter, register


class DummyAdapter(DetectorAdapter):
    name = "dummy"

    def __init__(
        self,
        mode: str = "perfect",
        seed: int = 0,
        num_random: int = 10,
        jitter: float = 0.15,
        fp_ratio: float = 0.3,
        device: str = "cpu",
        **kwargs: Any,
    ) -> None:
        super().__init__(device=device, **kwargs)
        if mode not in ("perfect", "empty", "jitter", "random"):
            raise ValueError("mode 必须是 perfect / empty / jitter / random 之一")
        self.mode = mode
        self.rng = np.random.RandomState(seed)
        self.num_random = num_random
        self.jitter = jitter
        self.fp_ratio = fp_ratio

    def predict(self, sample: DetSample, image_root: str) -> DetPrediction:
        if self.mode == "empty":
            return DetPrediction(
                sample.image_id,
                np.zeros((0, 4), np.float32),
                np.zeros((0,), np.float32),
                np.zeros((0,), np.int64),
            )

        if self.mode == "perfect":
            # 输出 GT 本身，并给不同分数保证排序稳定
            n = sample.num_objects
            scores = (
                np.linspace(0.99, 0.90, n, dtype=np.float32) if n else np.zeros((0,), np.float32)
            )
            return DetPrediction(sample.image_id, sample.boxes.copy(), scores, sample.labels.copy())

        if self.mode == "jitter":
            # 把 GT 按框尺寸的比例做平移，使 IoU 落在 0.5~0.9 之间；
            # 再掺入一部分完全错误的预测，模拟真实的 FP。
            n = sample.num_objects
            if n == 0:
                return DetPrediction(
                    sample.image_id,
                    np.zeros((0, 4), np.float32),
                    np.zeros((0,), np.float32),
                    np.zeros((0,), np.int64),
                )
            wh = np.clip(sample.boxes[:, 2:4] - sample.boxes[:, 0:2], 1e-6, None)
            dx = (self.rng.uniform(-self.jitter, self.jitter, n) * wh[:, 0]).reshape(-1, 1)
            dy = (self.rng.uniform(-self.jitter, self.jitter, n) * wh[:, 1]).reshape(-1, 1)
            # 平移量同时作用到左上与右下角（形状 (n,4)），保持框的尺寸不变
            shift = np.concatenate([dx, dy, dx, dy], axis=1)
            boxes = sample.boxes + shift
            scores = self.rng.uniform(0.5, 0.99, n).astype(np.float32)
            labels = sample.labels.copy()

            n_fp = int(round(n * self.fp_ratio))
            if n_fp > 0:
                w, h = float(sample.width), float(sample.height)
                cx = self.rng.uniform(0.05, 0.95, n_fp) * w
                cy = self.rng.uniform(0.05, 0.95, n_fp) * h
                bw = self.rng.uniform(0.02, 0.15, n_fp) * w
                bh = self.rng.uniform(0.02, 0.15, n_fp) * h
                fp_boxes = np.stack(
                    [cx - bw / 2, cy - bh / 2, cx + bw / 2, cy + bh / 2], axis=1
                ).astype(np.float32)
                boxes = np.concatenate([boxes, fp_boxes], axis=0)
                scores = np.concatenate([scores, self.rng.uniform(0.05, 0.4, n_fp).astype(np.float32)])
                labels = np.concatenate([labels, np.zeros((n_fp,), dtype=np.int64)])

            order = np.argsort(-scores)
            return DetPrediction(sample.image_id, boxes[order], scores[order], labels[order])

        # random：随机位置 + 随机尺寸，模拟一个几乎没有检测能力的模型
        w, h = float(sample.width), float(sample.height)
        k = self.num_random
        cx = self.rng.uniform(0.1, 0.9, k) * w
        cy = self.rng.uniform(0.1, 0.9, k) * h
        bw = self.rng.uniform(0.02, 0.2, k) * w
        bh = self.rng.uniform(0.02, 0.2, k) * h
        boxes = np.stack(
            [cx - bw / 2, cy - bh / 2, cx + bw / 2, cy + bh / 2], axis=1
        ).astype(np.float32)
        scores = self.rng.uniform(0.05, 0.95, k).astype(np.float32)
        labels = np.zeros((k,), dtype=np.int64)
        return DetPrediction(sample.image_id, boxes, scores, labels)

    def describe(self) -> Dict[str, Any]:
        return {
            "method": f"Dummy({self.mode})",
            "backbone": "-",
            "input_size": "-",
            "epochs": 0,
            "seed": 0,
        }

    @property
    def torch_model(self) -> Optional[Any]:
        return None


@register("dummy")
def _build_dummy(device: str = "cpu", **kwargs: Any) -> DummyAdapter:
    return DummyAdapter(device=device, **kwargs)
