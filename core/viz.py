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

    # 中文字体：必须显式指定。matplotlib 默认字体不含 CJK 字形，
    # 中文标签会渲染成一排方框（"豆腐块"）而且**不报任何错**，
    # 往往到把图贴进论文时才发现。
    cjk = _pick_cjk_font()
    if cjk:
        plt.rcParams["font.sans-serif"] = [cjk, "DejaVu Sans"]
        # 中文环境下负号也会变成方框，必须关掉 unicode_minus
        plt.rcParams["axes.unicode_minus"] = False

    return plt


def _pick_cjk_font() -> Optional[str]:
    """找一个能显示中文的字体名；找不到返回 None。"""
    try:
        from matplotlib import font_manager
    except Exception:
        return None

    # 按优先级覆盖 Windows / macOS / Linux 常见中文字体
    preferred = (
        "Microsoft YaHei",   # Windows 默认中文界面字体
        "SimHei",            # Windows 黑体
        "SimSun",            # Windows 宋体
        "PingFang SC",       # macOS
        "Hiragino Sans GB",  # macOS
        "Heiti SC",          # macOS
        "Noto Sans CJK SC",  # Linux
        "Source Han Sans SC",
        "WenQuanYi Micro Hei",
        "Arial Unicode MS",
    )
    try:
        available = {f.name for f in font_manager.fontManager.ttflist}
    except Exception:
        return None
    for name in preferred:
        if name in available:
            return name
    return None


def _rect(box, color, lw=1.0):
    from matplotlib.patches import Rectangle

    x1, y1, x2, y2 = box
    return Rectangle((x1, y1), x2 - x1, y2 - y1, fill=False, edgecolor=color, linewidth=lw)


# --------------------------------------------------------------------------- #
# 分层柱状图
# --------------------------------------------------------------------------- #


def _extract_metric(value: Any, metric: str) -> float:
    """从一个分组的取值里取出指标数值。

    兼容两种写法：
      * ``{"mAP": 0.81, "mAP50": 0.93}``  —— 完整指标字典（ExperimentResult 的格式）
      * ``0.81``                          —— 直接给数值

    做这个兼容是因为只给数值是更自然的写法，而只给数值时原实现会抛
    ``'float' object has no attribute 'get'``，完全看不出问题在哪。
    """
    if isinstance(value, (int, float, np.floating)):
        return float(value)
    if isinstance(value, dict):
        if metric in value:
            v = value[metric]
            return float(v) if isinstance(v, (int, float, np.floating)) else 0.0
        # 常见别名兜底
        for alias in ("mAP", "AP", "map", "AP50", "mAP50"):
            if alias in value and isinstance(value[alias], (int, float, np.floating)):
                return float(value[alias])
        return 0.0
    raise TypeError(
        f"分组取值类型不支持: {type(value).__name__}。\n"
        "期望 {组名: {'mAP': 数值}} 或 {组名: 数值}。"
    )


def plot_stratified_bars(
    stratified_by_method: Dict[str, Dict[str, Any]],
    group_key: str,
    out_path: str | Path,
    metric: str = "mAP",
    title: Optional[str] = None,
    group_order: Optional[Sequence[str]] = None,
) -> Path:
    """画条件 mAP 对比柱状图。

    Parameters
    ----------
    stratified_by_method : ``{方法名: {组名: 取值}}``。取值可以是

        * 完整指标字典（``ExperimentResult.stratified[group_key]`` 的格式）::

              {"YOLO11": {"near": {"mAP": 0.93, "mAP50": 0.97}, ...}}

        * 直接给数值（更省事）::

              {"YOLO11": {"near": 0.93, "mid": 0.78, "far": 0.31}}

    group_key : 仅用于默认标题与报错信息
    metric : 取值是字典时，取哪个键
    group_order : 指定柱子顺序；不指定则按组名字母序
    """
    # 先做输入校验，避免后续抛出难以理解的 AttributeError
    if not isinstance(stratified_by_method, dict) or not stratified_by_method:
        raise ValueError(
            f"分组 '{group_key}' 没有任何数据。\n"
            "stratified_by_method 应为 {方法名: {组名: 取值}}，且不能为空。"
        )
    for method, groups_of_method in stratified_by_method.items():
        if not isinstance(groups_of_method, dict):
            raise TypeError(
                f"方法 '{method}' 的取值应是 {{组名: 取值}} 字典，"
                f"实际是 {type(groups_of_method).__name__}。\n"
                "正确写法：{'方法名': {'组名': 0.81}} 或 "
                "{'方法名': {'组名': {'mAP': 0.81}}}"
            )

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
        vals = [
            _extract_metric(stratified_by_method[method].get(g, 0.0), metric)
            for g in groups
        ]
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
    all_x: set[float] = set()
    for method, buckets in curves_by_method.items():
        pts: List[Tuple[float, float]] = []
        for name, val in buckets.items():
            c = parse_bin_center(name)
            if c is not None:
                pts.append((c, float(val)))
        pts.sort()
        if pts:
            xs, ys = zip(*pts)
            ax.plot(xs, ys, marker="o", ms=4, lw=1.4, label=method)
            all_x.update(xs)

    if log_x and all_x:
        ax.set_xscale("log")
        # 关键：显式指定刻度位置，否则 matplotlib 会在对数轴上堆一堆
        # 密集的次刻度标签（"2×10¹ 3×10¹ 4×10¹..."），互相重叠完全不可读。
        ticks = sorted(all_x)
        ax.set_xticks(ticks)
        # 用普通数字格式，避免科学计数法标签过长
        ax.xaxis.set_major_formatter(
            plt.FuncFormatter(lambda v, _pos: f"{v:g}")
        )
        ax.xaxis.set_minor_formatter(plt.NullFormatter())
        ax.tick_params(axis="x", which="minor", length=0)

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
        side = np.sqrt(np.concatenate(areas))
        # 用对数分箱：目标尺寸跨度常有 2~3 个数量级，线性分箱会把
        # 小目标全挤在第一个箱里，看不出分布形状。
        bins = np.logspace(np.log10(max(side.min(), 0.5)), np.log10(side.max() + 1e-6), 25)
        axes[0].hist(side, bins=bins, color="steelblue")
        axes[0].set_xscale("log")
        # 对数轴必须显式控制刻度，否则主/次刻度标签会糊成一团
        axes[0].xaxis.set_major_formatter(plt.FuncFormatter(lambda v, _p: f"{v:g}"))
        axes[0].xaxis.set_minor_formatter(plt.NullFormatter())
        axes[0].tick_params(axis="x", which="minor", length=0)
    axes[0].set_xlabel("Equivalent side length (px)")
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
