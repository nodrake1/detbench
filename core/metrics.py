"""分层 AP 计算（本框架的核心）。

标准 mAP 只给你一个数。但研究结论藏在条件分布里：

    "目标等效边长 < 16px 时 mAP 从 0.83 掉到 0.21"
    "在重模糊帧上 YOLO 领先 DETR 0.04，但在清晰帧上落后 0.02"

要得到这类结论，必须能"按条件切片后再算 AP"。本模块提供：

1. :func:`compute_ap` —— 不依赖 pycocotools 的 AP 实现，
   支持按目标尺寸分桶与按最大检测数截断。
2. :class:`GroupSpec` / :func:`apply_group` —— 按属性把数据集切片，
   用于分组评测（距离档 / 模糊档 / 光照档 ...）。

关于"分组评测时假正例怎么算"
----------------------------
本实现采用**全局评测协议**：把不符合条件的 GT 从该组剔除，
但保留模型在**整张图**上的全部预测。这样落在天空背景上的误检
仍被计为假正例，指标才反映真实部署表现。

若改用"只在组内区域评测"的协议，会人为放大分数。
两种协议结果差别很大，论文里必须写清用的是哪一种。
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

from detbench.core.io import DetPrediction, DetSample

# --------------------------------------------------------------------------- #
# 分桶定义
# --------------------------------------------------------------------------- #

#: 目标尺寸分桶（按 bbox 面积，单位 px^2）。阈值对齐 COCO 定义。
#: 等效边长 = sqrt(area)，便于直观理解：
#:     small  < 32^2    -> 边长 < 32 px
#:     medium 32^2~96^2 -> 边长 32~96 px
#:     large  > 96^2    -> 边长 > 96 px
AREA_BINS: Tuple[Tuple[str, float, float], ...] = (
    ("small", 0.0, 32.0 ** 2),
    ("medium", 32.0 ** 2, 96.0 ** 2),
    ("large", 96.0 ** 2, float("inf")),
)

#: 面向对空/航拍小目标的细分桶（比 COCO 更细）。
#: 远距离直升机常落在 nano / tiny 桶里，细分后才能看出方法差异。
FINE_AREA_BINS: Tuple[Tuple[str, float, float], ...] = (
    ("nano(<8px)", 0.0, 8.0 ** 2),
    ("tiny(8-16px)", 8.0 ** 2, 16.0 ** 2),
    ("small(16-32px)", 16.0 ** 2, 32.0 ** 2),
    ("medium(32-96px)", 32.0 ** 2, 96.0 ** 2),
    ("large(>96px)", 96.0 ** 2, float("inf")),
)

NO_BIN = "__nobin__"


def bin_of_area(area: float, bins: Sequence[Tuple[str, float, float]]) -> Optional[str]:
    """返回面积所属桶名；不在任何桶内返回 None。"""
    for name, lo, hi in bins:
        if lo <= area < hi:
            return name
    return None


def boxes_area(boxes: np.ndarray) -> np.ndarray:
    """xyxy 框的面积（负宽高裁剪为 0）。"""
    if boxes.size == 0:
        return np.zeros((0,), dtype=np.float32)
    wh = np.clip(boxes[:, 2:4] - boxes[:, 0:2], 0.0, None)
    return (wh[:, 0] * wh[:, 1]).astype(np.float32)


# --------------------------------------------------------------------------- #
# 基础几何
# --------------------------------------------------------------------------- #


def box_iou_matrix(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """两组框的 IoU 矩阵，形状 ``(len(a), len(b))``；输入均为 xyxy。"""
    if a.size == 0 or b.size == 0:
        return np.zeros((a.shape[0], b.shape[0]), dtype=np.float32)

    area_a = boxes_area(a)
    area_b = boxes_area(b)

    lt = np.maximum(a[:, None, :2], b[None, :, :2])
    rb = np.minimum(a[:, None, 2:], b[None, :, 2:])
    wh = np.clip(rb - lt, 0.0, None)
    inter = wh[..., 0] * wh[..., 1]

    union = area_a[:, None] + area_b[None, :] - inter
    return (inter / np.maximum(union, 1e-9)).astype(np.float32)


def _voc_ap(recall: np.ndarray, precision: np.ndarray) -> float:
    """VOC2010+ 全点插值 AP（与 COCO 的 101 点采样精度等价）。"""
    if recall.size == 0:
        return 0.0
    mrec = np.concatenate(([0.0], recall, [1.0]))
    mpre = np.concatenate(([0.0], precision, [0.0]))
    for i in range(mpre.size - 1, 0, -1):
        mpre[i - 1] = max(mpre[i - 1], mpre[i])
    idx = np.where(mrec[1:] != mrec[:-1])[0]
    return float(np.sum((mrec[idx + 1] - mrec[idx]) * mpre[idx + 1]))


# --------------------------------------------------------------------------- #
# AP 主函数
# --------------------------------------------------------------------------- #


def compute_ap(
    samples: Sequence[DetSample],
    predictions: Dict[int, DetPrediction],
    iou_threshold: float = 0.5,
    area_bins: Optional[Sequence[Tuple[str, float, float]]] = None,
    max_dets: int = 100,
    num_classes: Optional[int] = None,
    return_curves: bool = False,
) -> Dict[str, Any]:
    """计算 AP 及其按目标尺寸分桶的分解。

    Parameters
    ----------
    samples : 该组的 GT 样本（标注应已按分组条件裁剪）
    predictions : ``image_id -> DetPrediction``（该组图像的完整预测）
    iou_threshold : IoU 判定阈值（0.5 或 0.75）
    area_bins : 尺寸分桶；None 表示不做尺寸分解
    max_dets : 每图最多参与评测的预测数（按分数取前 k），对齐 COCO 的 maxDets
    num_classes : 类别数；默认从 GT 推断
    return_curves : 是否额外返回 PR 曲线数据（画图用）

    Returns
    -------
    dict: ``{"AP":..., "num_gt":..., "num_pred":..., "AP/<桶>":..., "nGT/<桶>":...}``

    Notes
    -----
    尺寸桶的 AP 定义为 ``桶内被正确匹配的 GT 数 / 桶内 GT 总数``（per-bin recall），
    FP 不归属到任何桶 —— 这与 COCO 对 AP_small/medium/large 的处理一致。
    因此各桶 AP 不加权求和不等于总体 AP，这是预期行为。
    """
    bins = list(area_bins) if area_bins else []
    flat_labels = [int(l) for s in samples for l in s.labels]
    if num_classes is None:
        num_classes = (max(flat_labels) + 1) if flat_labels else 1

    img_index = {s.image_id: i for i, s in enumerate(samples)}

    # ---- 收集 GT（按类别） ----
    gt_boxes: List[List[np.ndarray]] = [[] for _ in range(num_classes)]
    gt_img: List[List[int]] = [[] for _ in range(num_classes)]
    gt_bin: List[List[str]] = [[] for _ in range(num_classes)]

    for s in samples:
        idx = img_index[s.image_id]
        if s.num_objects == 0:
            continue
        areas = boxes_area(s.boxes)
        for b, l, a in zip(s.boxes, s.labels, areas):
            c = int(l)
            if c >= num_classes:
                continue
            gt_boxes[c].append(b)
            gt_img[c].append(idx)
            gt_bin[c].append(bin_of_area(float(a), bins) or NO_BIN)

    n_gt_total = sum(len(gt_img[c]) for c in range(num_classes))
    n_gt_per_bin = {name: 0 for name, _, _ in bins}
    for c in range(num_classes):
        for bn in gt_bin[c]:
            if bn in n_gt_per_bin:
                n_gt_per_bin[bn] += 1

    # 便于按 (图, 类别) 快速取 GT 索引
    gt_lookup: Dict[Tuple[int, int], List[int]] = {}
    for c in range(num_classes):
        for gi, img_i in enumerate(gt_img[c]):
            gt_lookup.setdefault((img_i, c), []).append(gi)

    # ---- 收集预测并按分数降序 ----
    records: List[Tuple[float, int, int, np.ndarray]] = []  # (score, class, img_idx, box)
    for s in samples:
        pred = predictions.get(s.image_id)
        if pred is None or pred.scores.size == 0:
            continue
        pred = pred.topk(max_dets)
        idx = img_index[s.image_id]
        for b, sc, l in zip(pred.boxes, pred.scores, pred.labels):
            c = int(l)
            if c >= num_classes:
                continue
            records.append((float(sc), c, idx, b))
    records.sort(key=lambda r: -r[0])

    # ---- 贪心匹配 ----
    matched = [np.zeros(len(gt_img[c]), dtype=bool) for c in range(num_classes)]
    tp = np.zeros(len(records), dtype=np.float64)
    fp = np.zeros(len(records), dtype=np.float64)
    hit_bin: List[Optional[str]] = [None] * len(records)

    for k, (_score, c, img_i, box) in enumerate(records):
        cand = gt_lookup.get((img_i, c), [])
        if not cand:
            fp[k] = 1.0
            continue
        cand_boxes = np.stack([gt_boxes[c][gi] for gi in cand])
        ious = box_iou_matrix(box.reshape(1, 4), cand_boxes)[0]
        # 取 IoU 最大且尚未被匹配的 GT（标准贪心）
        order = np.argsort(-ious)
        chosen, chosen_iou = -1, 0.0
        for oi in order:
            gi = cand[oi]
            if not matched[c][gi]:
                chosen, chosen_iou = gi, float(ious[oi])
                break
        if chosen >= 0 and chosen_iou >= iou_threshold:
            matched[c][chosen] = True
            tp[k] = 1.0
            hit_bin[k] = gt_bin[c][chosen]
        else:
            fp[k] = 1.0

    # ---- 汇总 ----
    result: Dict[str, Any] = {
        "AP": _ap_from_curve(tp, fp, n_gt_total),
        "num_gt": int(n_gt_total),
        "num_pred": int(len(records)),
        "num_tp": int(tp.sum()),
    }
    if abs(iou_threshold - 0.5) < 1e-9:
        result["AP50"] = result["AP"]
    elif abs(iou_threshold - 0.75) < 1e-9:
        result["AP75"] = result["AP"]

    for name, _, _ in bins:
        n_gt_b = n_gt_per_bin[name]
        result[f"nGT/{name}"] = int(n_gt_b)
        if n_gt_b > 0:
            hit = sum(1 for k in range(len(records)) if hit_bin[k] == name)
            result[f"AP/{name}"] = hit / n_gt_b

    if return_curves and n_gt_total > 0:
        tp_cum = np.cumsum(tp)
        fp_cum = np.cumsum(fp)
        result["_curve"] = {
            "recall": (tp_cum / n_gt_total).tolist(),
            "precision": (tp_cum / np.maximum(tp_cum + fp_cum, 1e-9)).tolist(),
        }
    return result


def _ap_from_curve(tp: np.ndarray, fp: np.ndarray, num_gt: int) -> float:
    if num_gt == 0 or tp.size == 0:
        return 0.0
    tp_cum = np.cumsum(tp)
    fp_cum = np.cumsum(fp)
    recall = tp_cum / num_gt
    precision = tp_cum / np.maximum(tp_cum + fp_cum, 1e-9)
    return _voc_ap(recall, precision)


def compute_map_range(
    samples: Sequence[DetSample],
    predictions: Dict[int, DetPrediction],
    iou_thresholds: Sequence[float] = tuple(np.arange(0.5, 1.0, 0.05)),
    area_bins: Optional[Sequence[Tuple[str, float, float]]] = None,
    max_dets: int = 100,
    num_classes: Optional[int] = None,
) -> Dict[str, Any]:
    """在多个 IoU 阈值上求平均，得到 mAP@[.5:.95] 及其尺寸分解。

    当 ``area_bins`` 给定时，同时给出每个桶在 IoU 范围内的平均 AP ——
    这就是论文里的 ``AP_small`` / ``AP_tiny`` 等指标。
    """
    per_iou: List[Dict[str, Any]] = []
    for t in iou_thresholds:
        per_iou.append(
            compute_ap(
                samples,
                predictions,
                iou_threshold=float(t),
                area_bins=area_bins,
                max_dets=max_dets,
                num_classes=num_classes,
            )
        )

    out: Dict[str, Any] = {
        "mAP": float(np.mean([r["AP"] for r in per_iou])),
        "num_gt": per_iou[0]["num_gt"],
        "num_pred": per_iou[0]["num_pred"],
        "num_tp": per_iou[0]["num_tp"],
    }
    if abs(iou_thresholds[0] - 0.5) < 1e-9:
        out["mAP50"] = per_iou[0]["AP"]
    if area_bins:
        for name, _, _ in area_bins:
            key = f"AP/{name}"
            vals = [r[key] for r in per_iou if key in r]
            if vals:
                out[f"mAP/{name}"] = float(np.mean(vals))
    return out


# --------------------------------------------------------------------------- #
# 分组切片
# --------------------------------------------------------------------------- #


@dataclass
class GroupSpec:
    """分组评测配置。

    Examples
    --------
    按"距离档"分组（属性表里有 ``distance_bin`` 列）::

        GroupSpec(key="distance_bin")

    按目标尺寸细分桶（无需属性表）::

        GroupSpec(key="__size__", bins=FINE_AREA_BINS)

    ``key="__size__"`` 为保留字，表示按 GT 框面积分桶。
    属性表里某列的值也可以是分号分隔的目标级列表，此时同一张图会被
    按目标拆到多个组里。
    """

    key: str
    bins: Optional[Sequence[Tuple[str, float, float]]] = None
    min_gt: int = 5  # 组内 GT 少于该数量则跳过（指标不可信）

    def groups(self, samples: Sequence[DetSample]) -> Dict[str, List[DetSample]]:
        out: Dict[str, List[DetSample]] = {}

        if self.key == "__size__":
            bins = self.bins or FINE_AREA_BINS
            for s in samples:
                if s.num_objects == 0:
                    continue
                areas = boxes_area(s.boxes)
                names = [bin_of_area(float(a), bins) for a in areas]
                for name in sorted({n for n in names if n}):
                    keep = np.array([n == name for n in names], dtype=bool)
                    out.setdefault(name, []).append(s.filtered(keep))
            return out

        for s in samples:
            raw = s.attrs.get(self.key)
            if raw is None or raw == "":
                out.setdefault("unknown", []).append(s)
                continue
            if isinstance(raw, (list, tuple, np.ndarray)) and len(raw) == s.num_objects:
                vals = [str(v) for v in raw]
                for v in sorted(set(vals)):
                    keep = np.array([x == v for x in vals], dtype=bool)
                    out.setdefault(v, []).append(s.filtered(keep))
            else:
                out.setdefault(str(raw), []).append(s)
        return out


def apply_group(
    samples: Sequence[DetSample],
    predictions: Dict[int, DetPrediction],
    spec: GroupSpec,
) -> List[Tuple[str, List[DetSample], Dict[int, DetPrediction]]]:
    """把数据集切成分组，返回 ``[(组名, 组内样本, 组内预测), ...]``。

    关键点：**预测不做裁剪** —— 模型在整张图上的预测全部保留参与该组评测，
    这样背景误检不会被洗掉，指标反映真实表现。详见模块文档。
    """
    groups = spec.groups(samples)
    out: List[Tuple[str, List[DetSample], Dict[int, DetPrediction]]] = []
    for name, group_samples in groups.items():
        if sum(s.num_objects for s in group_samples) < spec.min_gt:
            continue
        ids = {s.image_id for s in group_samples}
        group_preds = {i: p for i, p in predictions.items() if i in ids}
        out.append((name, group_samples, group_preds))
    out.sort(key=lambda t: t[0])  # 稳定顺序，论文表格才可复现
    return out
