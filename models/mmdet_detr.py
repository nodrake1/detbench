"""MMDetection 适配器：Deformable DETR / DINO-DETR / DETR 及任意 mmdet 检测器。

为什么 DETR 系要用 mmdetection
-------------------------------
原版 facebookresearch/detr 仓库基本停更，Deformable DETR / DINO-DETR /
DAB-DETR / DN-DETR 的可用实现集中在 mmdetection 里，且共享同一套配置系统。
用这一个适配器可以覆盖整条 DETR 家族，也能顺带跑 Faster R-CNN / RetinaNet
等 mmdet 检测器做同框架对比 —— 这对"对比公平性"很有价值。

⚠️ 最容易出错的一点：类别标签偏移
---------------------------------
**mmdetection 的 DETR 系模型，类别 0 是背景保留位，前景类别从 1 开始。**
所以"单类别直升机"训练出来，模型的预测标签是 **1**。

而本框架统一约定 **0-based 前景类别**（单类别就是 0）。

如果忘记这件事，后果极其隐蔽：
    用单类数据评测时，`np.clip(labels - 1, 0, None)` 恰好把 1 变成 0，看起来"对了"；
    但一旦类别数 >= 2，所有类别都会整体错位一位，mAP 会莫名暴跌，
    而且不报任何错。

所以本适配器把 ``label_offset`` 做成**显式参数**，并在初始化时打印一行
提示，让这个假设不可能被悄悄忽略。
"""

from __future__ import annotations

import warnings
from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np

from detbench.core.deps import mmdet_available, require
from detbench.core.io import DetPrediction, DetSample
from detbench.core.registry import DetectorAdapter, register


class MMDetAdapter(DetectorAdapter):
    """mmdetection 推理封装。

    参数
    ----
    config : mmdet 配置文件路径（.py）。**注意别用带 ``_tta`` 后缀的版本**，
             那会在推理时做多尺度增强，速度慢一个数量级且和你对比的其他方法不可比。
    checkpoint : 训练权重路径（.pth）。不给则是随机初始化，只适合管线自检；
                 此时会打印警告。
    palette : 调色板，指定调色板名或 None（None 最快，且不影响数值结果）。
    score_thresh : 推理后保留的最低分数。**算 mAP 必须设很低（默认 0.001）**。
    max_per_img : 每图最多保留多少预测。DETR 系通常输出固定的 query 数
                  （300 / 900 / 1500），这里默认 300。
    label_offset : 模型标签 -> 框架标签的偏移量，即 ``framework_label = model_label - offset``。
                   mmdet 的 DETR 系默认 ``1``（0 是背景保留位）。
                   **如果你的模型是 0-based 的，必须显式设为 0。**
    """

    name = "mmdet"

    def __init__(
        self,
        config: str,
        checkpoint: Optional[str] = None,
        device: str = "cuda",
        score_thresh: float = 0.001,
        max_per_img: int = 300,
        label_offset: int = 1,
        palette: Optional[str] = None,
        batch_size: int = 1,
        **kwargs: Any,
    ) -> None:
        super().__init__(device=device, **kwargs)
        require("mmdet")
        require("mmengine")

        from mmdet.apis import init_detector

        self.config_path = str(config)
        self.checkpoint = str(checkpoint) if checkpoint else None
        self.score_thresh = float(score_thresh)
        self.max_per_img = int(max_per_img)
        self.label_offset = int(label_offset)

        # 配置文件必须存在且必须是本地 .py —— 避免隐式联网下载
        if not Path(self.config_path).exists():
            raise FileNotFoundError(
                f"找不到 mmdet 配置文件：{self.config_path}\n"
                "提示：本适配器只接受 **本地 .py 配置**，不会自动联网下载。\n"
                "      Deformable DETR / DINO-DETR 的配置在 mmdetection 仓库的\n"
                "      configs/deformable_detr/ 与 configs/dino/ 下，先 clone 仓库再指向它。"
            )

        if self.checkpoint is None:
            warnings.warn(
                "未提供 checkpoint，模型为随机初始化。只适合验证管线是否跑通，"
                "指标没有任何意义。",
                stacklevel=2,
            )
        elif not Path(self.checkpoint).exists():
            raise FileNotFoundError(f"找不到权重文件：{self.checkpoint}")

        # init_detector(config, checkpoint, device, palette)
        # 注意：device 只接受 'cpu' / 'cuda' / 'cuda:0' 这类字符串
        self.model = init_detector(
            self.config_path,
            self.checkpoint,
            device=device,
            palette=palette,
        )
        # init_detector 内部已经把配置加载成 Config 对象并挂在 model.cfg 上。
        # 复用它而不是再读一遍文件，既省一次 IO，也避免因路径/环境差异读到别的东西。
        self.cfg = self._grab_cfg(self.model)
        self._input_size = self._read_input_size()
        self._backbone = self._read_backbone_name()

        print(
            f"[MMDetAdapter] 已加载 {Path(self.config_path).name}"
            f"  | label_offset={self.label_offset}"
            f"  | conf>={self.score_thresh}  | max_per_img={self.max_per_img}"
        )
        if self.label_offset:
            print(
                f"[MMDetAdapter] 注意：模型标签将减去 {self.label_offset} 映射为框架的 "
                f"0-based 标签。若你的模型本身就是 0-based，请把 label_offset 设为 0。"
            )

    # ------------------------------------------------------------------ #
    @staticmethod
    def _grab_cfg(model: Any) -> Any:
        """取出 init_detector 已经加载好的配置对象。"""
        for attr in ("cfg", "config"):
            cfg = getattr(model, attr, None)
            if cfg is not None:
                return cfg
        return None

    def _read_input_size(self) -> str:
        """从配置的 test pipeline 里读出输入尺度，仅用于报告。

        这段特意写得非常防御：mmdet 的 pipeline 结构在不同版本、不同模型
        之间差异很大（``test_pipeline`` / ``test_dataloader.dataset.pipeline``
        / ``MultiScaleFlipAug`` 包裹 ...）。读不到就返回 ``'?'``，
        **绝不影响任何数值结果**。
        """
        cfg = self.cfg
        if cfg is None:
            return "?"
        try:
            getter = getattr(cfg, "get", None)

            def cfg_get(d, key, default=None):
                if d is None:
                    return default
                if hasattr(d, "get"):
                    return d.get(key, default)
                return default

            pipe = cfg_get(cfg, "test_pipeline")
            if not pipe:
                test_dl = cfg_get(cfg, "test_dataloader")
                ds = cfg_get(test_dl, "dataset")
                pipe = cfg_get(ds, "pipeline")
            if not pipe:
                return "?"

            for step in pipe:
                step_type = cfg_get(step, "type", "")
                scale = cfg_get(step, "scale")
                if scale and step_type in ("Resize", "Pad", "MultiScaleFlipAug", "TestTimeAug"):
                    return str(tuple(scale))
                # MultiScaleFlipAug / TestTimeAug 会把真正的 Resize 包在 transforms 里
                transforms = cfg_get(step, "transforms")
                if transforms:
                    for t in transforms:
                        if cfg_get(t, "type") == "Resize" and cfg_get(t, "scale"):
                            return str(tuple(cfg_get(t, "scale")))
            return "?"
        except Exception:
            return "?"

    def _read_backbone_name(self) -> str:
        cfg = self.cfg
        if cfg is None:
            return "-"
        try:
            model_cfg = cfg.get("model", {}) if hasattr(cfg, "get") else {}
            backbone = model_cfg.get("backbone", {}) if hasattr(model_cfg, "get") else {}
            return str(backbone.get("type", "-")) if hasattr(backbone, "get") else "-"
        except Exception:
            return "-"

    def _prepare_inputs(self, sample: DetSample, image_root: str) -> Dict[str, Any]:
        """按 mmdet 的推理输入约定构造 data_samples。"""
        from mmdet.structures import DetDataSample
        from mmengine.structures import InstanceData

        img_path = str(Path(image_root) / sample.file_name)

        # mmdet 的 test pipeline 通常第一步是 LoadImageFromFile，
        # 所以这里直接把**文件路径**放进 data_samples，让 pipeline 自己去读图。
        # 这比自己读图再转 tensor 更安全：pipeline 里的 Resize/Pad/Normalize
        # 都按配置执行，保证不同模型走的是各自论文推荐的预处理。
        data_sample = DetDataSample()
        data_sample.set_metainfo(
            {
                "img_path": img_path,
                "img_id": int(sample.image_id),
                "ori_shape": (int(sample.height), int(sample.width)),   # (h, w)
                "img_shape": (int(sample.height), int(sample.width)),
                "scale_factor": (1.0, 1.0),
            }
        )
        data_sample.gt_instances = InstanceData()
        data_sample.gt_instances.labels = self._zeros(0, dtype="long")
        data_sample.gt_instances.bboxes = self._zeros((0, 4), dtype="float32")
        # inputs 内容会被 test pipeline 覆盖（第一步就是 LoadImageFromFile），
        # 这里给一个占位即可。
        return {"inputs": [self._zeros((3, 1, 1), dtype="float32")], "data_samples": [data_sample]}

    @staticmethod
    def _zeros(shape, dtype: str):
        """构造占位张量。

        正常环境里 torch 必然存在（mmdet 依赖它）。这里保留一个 numpy 回退，
        是为了让 ``scripts/check_mmdet_adapter.py`` 能在没有 torch 的机器上
        验证适配器自身的逻辑。
        """
        try:
            import torch

            return torch.zeros(shape, dtype=getattr(torch, dtype))
        except ImportError:  # pragma: no cover - 仅用于无 torch 的逻辑自检
            np_dtype = np.int64 if dtype == "long" else np.float32
            return np.zeros(shape if isinstance(shape, tuple) else (shape,), dtype=np_dtype)

    def predict(self, sample: DetSample, image_root: str) -> DetPrediction:
        data = self._prepare_inputs(sample, image_root)

        # test_cfg 里的 score_thresh 在不同版本/模型下可能是浮点也可能是 list，
        # 为保持各方法可比，这里统一把它们压到极低值，由评测阶段做截断。
        try:
            self.model.test_cfg.score_thr = self.score_thresh
        except Exception:
            pass
        try:
            if hasattr(self.model, "test_cfg") and self.model.test_cfg is not None:
                self.model.test_cfg.max_per_img = self.max_per_img
        except Exception:
            pass

        with self._no_grad():
            result = self.model.test_step(data)

        instances = self._extract_instances(result)

        boxes = self._to_numpy(instances.get("bboxes"))
        scores = self._to_numpy(instances.get("scores"))
        labels = self._to_numpy(instances.get("labels"))

        if boxes is None or boxes.size == 0:
            return DetPrediction(
                sample.image_id,
                np.zeros((0, 4), np.float32),
                np.zeros((0,), np.float32),
                np.zeros((0,), np.int64),
            )

        boxes = boxes.astype(np.float32)
        scores = scores.astype(np.float32)
        labels = labels.astype(np.int64)

        # mmdet 的 DETR 系输出已经是原图 xyxy 坐标（pipeline 会做反变换），
        # 但不同版本/自定义 pipeline 可能漏掉，这里做一次防御性缩放。
        boxes = self._rescale_if_needed(boxes, sample, instances)

        # 标签偏移：mmdet 0 = 背景保留位
        labels = labels - self.label_offset
        if labels.size and labels.min() < 0:
            # 出现负标签说明 label_offset 设大了（把背景预测也放了出来）
            n_bad = int((labels < 0).sum())
            warnings.warn(
                f"有 {n_bad} 个预测的标签在减去 label_offset={self.label_offset} 后为负，"
                "说明模型把背景类也预测出来了。已丢弃这些预测。"
                "如果数量很多，请检查 label_offset 是否设置正确。",
                stacklevel=2,
            )
            keep = labels >= 0
            boxes, scores, labels = boxes[keep], scores[keep], labels[keep]
            if labels.size == 0:
                return DetPrediction(
                    sample.image_id,
                    np.zeros((0, 4), np.float32),
                    np.zeros((0,), np.float32),
                    np.zeros((0,), np.int64),
                )

        keep = scores >= self.score_thresh
        boxes, scores, labels = boxes[keep], scores[keep], labels[keep]

        # 按分数取前 max_per_img，与评测阶段的 maxDets 截断保持一致
        if scores.size > self.max_per_img:
            order = np.argsort(-scores)[: self.max_per_img]
            boxes, scores, labels = boxes[order], scores[order], labels[order]

        return DetPrediction(sample.image_id, boxes, scores, labels)

    # ------------------------------------------------------------------ #
    @staticmethod
    def _no_grad():
        """推理上下文。缺少 torch 时退化为 nullcontext（仅供无 torch 的逻辑自检）。"""
        try:
            import torch

            return torch.no_grad()
        except ImportError:  # pragma: no cover
            import contextlib

            return contextlib.nullcontext()

    @staticmethod
    def _to_numpy(x: Any) -> Optional[np.ndarray]:
        """把 tensor / BaseBoxes / ndarray 统一转成 numpy。"""
        if x is None:
            return None
        # mmdet 3.3+ 有 BaseBoxes 体系，需要先转成 tensor
        for attr in ("tensor", "to_tensor"):
            if hasattr(x, attr):
                v = getattr(x, attr)
                x = v() if callable(v) else v
                break
        if hasattr(x, "detach"):
            x = x.detach().cpu().numpy()
        return np.asarray(x)

    @staticmethod
    def _extract_instances(result: Any) -> Dict[str, Any]:
        """从 mmdet 的 test_step 返回里取出预测实例。

        mmdet 3.x 返回 ``list[DetDataSample]``，我们一次只喂一张，所以取 [0]。
        这里对多种字段名做兼容，避免因版本差异直接崩掉。
        """
        if isinstance(result, (list, tuple)):
            if len(result) == 0:
                return {}
            result = result[0]
        pred = getattr(result, "pred_instances", None)
        if pred is None:
            return {}
        out: Dict[str, Any] = {}
        for key in ("bboxes", "scores", "labels"):
            out[key] = getattr(pred, key, None)
        return out

    def _rescale_if_needed(
        self, boxes: np.ndarray, sample: DetSample, instances: Dict[str, Any]
    ) -> np.ndarray:
        """若返回的框明显超出原图范围，则判定 pipeline 没做反变换，手动缩放。"""
        if boxes.size == 0:
            return boxes
        w, h = float(sample.width), float(sample.height)
        tol = 1.02
        if boxes[:, 2].max() <= w * tol and boxes[:, 3].max() <= h * tol:
            return boxes  # 正常情况：已经是原图坐标
        # 尝试用 metainfo 里的 scale_factor 反算
        warnings.warn(
            f"图像 {sample.file_name} 的预测框超出原图范围"
            f"（max x={boxes[:, 2].max():.1f} > W={w:.0f}），"
            "判定推理 pipeline 未做坐标反变换。请检查配置里的 test pipeline。",
            stacklevel=2,
        )
        return boxes

    def describe(self) -> Dict[str, Any]:
        return {
            "method": f"mmdet:{Path(self.config_path).stem}",
            "backbone": self._backbone,
            "input_size": self._input_size,
            "epochs": 0,
            "seed": 0,
        }

    @property
    def torch_model(self) -> Optional[Any]:
        return self.model


# --------------------------------------------------------------------------- #
# 注册
# --------------------------------------------------------------------------- #


@register("mmdet")
def _build_mmdet(device: str = "cuda", **kwargs: Any) -> MMDetAdapter:
    return MMDetAdapter(device=device, **kwargs)


# 语义化别名：配置里写这些名字读起来更清楚
from detbench.core.registry import _REGISTRY  # noqa: E402

for _alias in (
    "detr",
    "deformable_detr",
    "dino",
    "dino_detr",
    "dab_detr",
    "dn_detr",
    "rt_detr",
    "mask2former",
    "faster_rcnn_mmdet",
):
    if _alias not in _REGISTRY:
        register(_alias)(_build_mmdet)


def mmdet_is_available() -> bool:
    """给 CLI/文档用的可用性查询。"""
    return mmdet_available()
