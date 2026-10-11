"""可视化工具：把评测结果变成论文里的图。

论文里真正有说服力的不是 mAP 数字，而是这些图：

- **尺度-mAP 曲线**：横轴目标等效边长，纵轴 mAP。直观展示小目标崩盘。
- **条件 mAP 柱状图**：距离档 / 模糊档 / 光照档分组对比。
- **PR 曲线**：多方法对比。
- **失败案例网格**：TP 绿框、FP 红框、FN 黄框。
  主动展示失败案例比藏起来强得多，审稿人很吃这套。

依赖 matplotlib；缺库时各函数会抛出 :class:`VizUnavailable` 并给出明确安装提示，
而不是让人对着一个裸 ImportError 发愁。
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

from detbench.core.metrics import box_iou_matrix, boxes_area
from detbench.core.io import DetPrediction, DetSample


class VizUnavailable(RuntimeError):
    """缺少绘图依赖。"""


def _plt():
    try:
        import matplotlib

        matplotlib.use("Agg")  # 服务器/无显示器环境必须
        import matplotlib.pyplot as plt
    except ImportError as exc:
        raise VizUnavailable(
            "绘图功能需要 matplotlib。请在你的训练环境里安装：\n"
            "    pip install matplotlib\n"
            "（评测、分层统计、结果汇总都不依赖它，只有 core/viz.py 出图需要。）"
        ) from exc

    plt.rcParams.update(
        {
            "font.size": 9,
            "axes.grid": True,
            "grid.alpha": 0.3,
            "figure.dpi": 160,
            "savefig.bbox": "tight",
        }
    )
    return plt


def _rect(box, color, lw=1.0):
    from matplotlib.patches import Rectangle

    x1, y1, x2, y2 = box
    return Rectangle((x1, y1), x2 - x1, y2 - y1, fill=False, edgecolor=color, linewidth=lw)


# --------------------------------------------------------------------------- #
# 分层柱状图
# --------------------------------------------------------------------------- #


def plot_stratified_bars(
    stratified_by_method: Dict[str, Dict[str, Dict[str, float]]],
    group_key: str,
    out_path: str | Path,
    metric: str = "mAP",
    title: Optional[str] = None,
    group_order: Optional[Sequence[str]] = None,
) -> Path:
    """画条件 mAP 对比柱状图。

    Parameters
    ----------
    stratified_by_method : ``{方法名: {组名: {指标: 值}}}``，
        即把多个 ``ExperimentResult.stratified[group_key]`` 收集到一起。
    """
    plt = _plt()
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    groups: List[str] = list(group_order or [])
    if not groups:
        seen: List[str] = []
        for m in stratified_by_method.values():
            for g in m:
                if g not in seen:
                    seen.append(g)
        groups = sorted(seen)
    if not groups:
        raise ValueError(f"分组 '{group_key}' 没有任何数据")

    methods = list(stratified_by_method)
    x = np.arange(len(groups))
    width = 0.8 / max(len(methods), 1)

    fig, ax = plt.subplots(figsize=(max(4.5, 1.2 * len(groups)), 3.2))
    for i, method in enumerate(methods):
        vals = [stratified_by_method[method].get(g, {}).get(metric, 0.0) for g in groups]
        ax.bar(x + i * width - 0.4 + width / 2, vals, width, label=method)
    ax.set_xticks(x)
    ax.set_xticklabels(groups, rotation=20, ha="right")
    ax.set_ylabel(metric)
    ax.set_title(title or f"{metric} by {group_key}")
    ax.legend(fontsize=7, frameon=False)
    fig.savefig(out_path)
    plt.close(fig)
    return out_path


# --------------------------------------------------------------------------- #
# 尺度-mAP 曲线
# --------------------------------------------------------------------------- #


def parse_bin_center(name: str) -> Optional[float]:
    """从桶名里解析出等效边长中心，作为曲线横轴坐标。

    支持 ``"tiny(8-16px)"`` / ``"nano(<8px)"`` / ``"large(>96px)"`` 三种写法。
    """
    m = re.search(r"(\d+(?:\.\d+)?)\s*-\s*(\d+(?:\.\d+)?)\s*px", name)
    if m:
        return (float(m.group(1)) + float(m.group(2))) / 2.0
    m = re.search(r"<\s*(\d+(?:\.\d+)?)", name)
    if m:
        return float(m.group(1)) / 2.0
    m = re.search(r">\s*(\d+(?:\.\d+)?)", name)
    if m:
        return float(m.group(1)) * 1.5
    return None


def plot_scale_curve(
    curves_by_method: Dict[str, Dict[str, float]],
    out_path: str | Path,
    title: str = "Detection performance vs. object scale",
    log_x: bool = True,
) -> Path:
    """画"目标等效边长 -> mAP"曲线 —— 小目标论文最有说服力的一张图。

    ``curves_by_method`` 的值形如
    ``{"nano(<8px)": 0.12, "tiny(8-16px)": 0.41, ...}``，
    即 :func:`detbench.core.metrics.compute_map_range` 输出的 ``mAP/<桶名>`` 字段。
    """
    plt = _plt()
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(4.5, 3.2))
    for method, buckets in curves_by_method.items():
        pts: List[Tuple[float, float]] = []
        for name, val in buckets.items():
            c = parse_bin_center(name)
            if c is not None:
                pts.append((c, float(val)))
        pts.sort()
        if pts:
            xs, ys = zip(*pts)
            ax.plot(xs, ys, marker="o", ms=3, label=method)
    if log_x:
        ax.set_xscale("log")
    ax.set_xlabel("Equivalent object side length (px)" + (" — log scale" if log_x else ""))
    ax.set_ylabel("mAP")
    ax.set_title(title)
    ax.legend(fontsize=7, frameon=False)
    fig.savefig(out_path)
    plt.close(fig)
    return out_path


def plot_pr_curves(
    curves: Dict[str, Tuple[Sequence[float], Sequence[float]]],
    out_path: str | Path,
    title: str = "Precision-Recall (IoU=0.5)",
) -> Path:
    """画多方法 PR 曲线。输入 ``{方法名: (recall, precision)}``。"""
    plt = _plt()
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(4.2, 3.4))
    for method, (rec, prec) in curves.items():
        ax.plot(rec, prec, lw=1.2, label=method)
    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1.02)
    ax.set_title(title)
    ax.legend(fontsize=7, frameon=False)
    fig.savefig(out_path)
    plt.close(fig)
    return out_path


# --------------------------------------------------------------------------- #
# 失败案例网格
# --------------------------------------------------------------------------- #


def plot_failure_cases(
    samples: Sequence[DetSample],
    predictions: Dict[int, DetPrediction],
    image_root: str | Path,
    out_path: str | Path,
    iou_threshold: float = 0.5,
    score_threshold: float = 0.3,
    max_images: int = 12,
    cols: int = 4,
    rank_by: str = "errors",
) -> Path:
    """把模型的错误画出来：TP 绿 / FP 红 / FN 黄。

    ``rank_by`` 取 ``"errors"``（按总错误数排序，展示最差的图）或
    ``"fn"``（按漏检数排序）。
    """
    plt = _plt()
    from PIL import Image

    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    records = []
    for s in samples:
        pred = predictions.get(s.image_id)
        p_boxes = pred.boxes if pred is not None else np.zeros((0, 4), np.float32)
        p_scores = pred.scores if pred is not None else np.zeros((0,), np.float32)
        keep = p_scores >= score_threshold
        p_boxes = p_boxes[keep]

        tp_mask = np.zeros(len(p_boxes), dtype=bool)
        fn_mask = np.ones(s.num_objects, dtype=bool)
        if len(p_boxes) and s.num_objects:
            ious = box_iou_matrix(p_boxes, s.boxes)
            for pi in range(len(p_boxes)):
                for gi in np.argsort(-ious[pi]):
                    if fn_mask[gi] and ious[pi, gi] >= iou_threshold:
                        fn_mask[gi] = False
                        tp_mask[pi] = True
                        break
        n_fp = int((~tp_mask).sum())
        n_fn = int(fn_mask.sum())
        records.append((s, p_boxes, tp_mask, fn_mask, n_fp + n_fn, n_fn))

    key_idx = 4 if rank_by == "errors" else 5
    records.sort(key=lambda r: -r[key_idx])
    records = records[:max_images]
    if not records:
        return out_path

    rows = int(np.ceil(len(records) / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(3.0 * cols, 3.0 * rows))
    axes = np.atleast_1d(axes).ravel()

    for ax_i, (s, p_boxes, tp_mask, fn_mask, _n_err, _n_fn) in enumerate(records):
        ax = axes[ax_i]
        try:
            ax.imshow(Image.open(Path(image_root) / s.file_name).convert("RGB"))
        except Exception:
            ax.set_facecolor("0.9")
        for pi, b in enumerate(p_boxes):
            ax.add_patch(_rect(b, "lime" if tp_mask[pi] else "red", 1.2))
        for gi, b in enumerate(s.boxes):
            if fn_mask[gi]:
                ax.add_patch(_rect(b, "yellow", 1.2))
        ax.set_title(
            f"{s.file_name}  FP={int((~tp_mask).sum())}  FN={int(fn_mask.sum())}", fontsize=6
        )
        ax.axis("off")
    for j in range(len(records), len(axes)):
        axes[j].axis("off")
    fig.suptitle("TP=green  FP=red  FN=yellow", fontsize=8)
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)
    return out_path


# --------------------------------------------------------------------------- #
# 数据集统计图（Dataset 章节用）
# --------------------------------------------------------------------------- #


def plot_dataset_stats(dataset, out_path: str | Path) -> Path:
    """画数据集统计组合图：尺寸分布 / 宽高比 / 位置热力图 / 每图目标数。

    这四张子图基本就是论文 Dataset 章节的标准配置。
    """
    plt = _plt()
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    areas, ars, cxs, cys, nobj = [], [], [], [], []
    for s in dataset.samples:
        nobj.append(s.num_objects)
        if s.num_objects == 0:
            continue
        areas.append(boxes_area(s.boxes))
        wh = np.clip(s.boxes[:, 2:4] - s.boxes[:, 0:2], 1e-6, None)
        ars.append(wh[:, 0] / wh[:, 1])
        cxs.append(((s.boxes[:, 0] + s.boxes[:, 2]) / 2) / max(s.width, 1))
        cys.append(((s.boxes[:, 1] + s.boxes[:, 3]) / 2) / max(s.height, 1))

    fig, axes = plt.subplots(1, 4, figsize=(13, 2.9))

    if areas:
        axes[0].hist(np.sqrt(np.concatenate(areas)), bins=40, color="steelblue")
    axes[0].set_xlabel("Equivalent side length (px)")
    axes[0].set_xscale("log")
    axes[0].set_ylabel("#objects")
    axes[0].set_title("(a) Object size distribution")

    if ars:
        axes[1].hist(np.concatenate(ars), bins=40, color="darkorange")
    axes[1].set_xlabel("Aspect ratio (w/h)")
    axes[1].set_title("(b) Aspect ratio")

    if cxs:
        axes[2].hist2d(np.concatenate(cxs), np.concatenate(cys), bins=24, cmap="magma")
        axes[2].invert_yaxis()
    axes[2].set_xlabel("x / W")
    axes[2].set_ylabel("y / H")
    axes[2].set_title("(c) Center position heatmap")

    if nobj:
        axes[3].hist(nobj, bins=min(30, max(5, int(max(nobj)))), color="seagreen")
    axes[3].set_xlabel("#objects per image")
    axes[3].set_title("(d) Objects per image")

    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)
    return out_path
