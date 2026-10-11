"""评测主循环：标准 mAP + 分层 mAP + 速度，一次跑完并落盘。

用法（在配置好模型适配器之后）::

    from detbench.core.eval import evaluate

    result = evaluate(
        adapter=adapter,
        dataset=test_ds,
        image_root="data/HeliAir/images/test",
        exp_id="exp001_dino_r50",
        group_specs=[GroupSpec("distance_bin"), GroupSpec("blur"), GroupSpec("__size__")],
        speed=True,
        out_dir="runs/exp001",
    )
"""

from __future__ import annotations

import time
from typing import Any, Dict, List, Optional, Sequence

import numpy as np

from detbench.core.io import DetDataset, DetPrediction, DetSample
from detbench.core.metrics import (
    AREA_BINS,
    FINE_AREA_BINS,
    GroupSpec,
    apply_group,
    compute_map_range,
)
from detbench.core.registry import DetectorAdapter
from detbench.core.results import ExperimentResult
from detbench.core.speed import measure_speed, torch_available

#: COCO 标准的评测 IoU 序列（0.50 : 0.05 : 0.95）
COCO_IOU_THRESHOLDS = tuple(float(t) for t in np.arange(0.5, 1.0, 0.05))


# --------------------------------------------------------------------------- #
# 推理
# --------------------------------------------------------------------------- #


def run_inference(
    adapter: DetectorAdapter,
    dataset: DetDataset,
    image_root: str,
    verbose: bool = True,
    log_every: int = 200,
) -> Dict[int, DetPrediction]:
    """在数据集上跑完推理，返回 ``image_id -> DetPrediction``。"""
    preds: Dict[int, DetPrediction] = {}
    t0 = time.time()
    for i, s in enumerate(dataset.samples, 1):
        try:
            preds[s.image_id] = adapter.predict(s, image_root)
        except Exception as exc:  # 单图失败不应中断整个评测
            print(f"[warn] 图像 {s.file_name} 推理失败：{exc}")
            preds[s.image_id] = DetPrediction(
                s.image_id,
                np.zeros((0, 4), dtype=np.float32),
                np.zeros((0,), dtype=np.float32),
                np.zeros((0,), dtype=np.int64),
            )
        if verbose and log_every and i % log_every == 0:
            rate = i / max(time.time() - t0, 1e-9)
            print(f"  推理进度 {i}/{dataset.num_images}  ({rate:.1f} img/s)")
    return preds


# --------------------------------------------------------------------------- #
# 分层评测
# --------------------------------------------------------------------------- #


def evaluate_stratified(
    dataset: DetDataset,
    predictions: Dict[int, DetPrediction],
    specs: Sequence[GroupSpec],
    fine_bins: bool = True,
) -> Dict[str, Dict[str, Dict[str, float]]]:
    """按多个属性维度做分组评测。

    Returns
    -------
    ``{group_key: {group_name: {"mAP": ..., "mAP/<桶名>": ...}}}``
    """
    out: Dict[str, Dict[str, Dict[str, float]]] = {}
    bins = FINE_AREA_BINS if fine_bins else AREA_BINS

    for spec in specs:
        spec_bins = spec.bins if spec.bins is not None else bins
        groups = apply_group(dataset.samples, predictions, spec)
        if not groups:
            continue
        per_group: Dict[str, Dict[str, float]] = {}
        for gname, g_samples, g_preds in groups:
            metrics = compute_map_range(
                g_samples,
                g_preds,
                iou_thresholds=COCO_IOU_THRESHOLDS,
                # __size__ 分组时组内已是单一尺寸桶，再做尺寸分解没有意义
                area_bins=None if spec.key == "__size__" else spec_bins,
                num_classes=dataset.num_classes,
            )
            metrics["num_images"] = float(len(g_samples))
            per_group[gname] = {k: float(v) for k, v in metrics.items() if v is not None}
        out[spec.key] = per_group
    return out


def build_size_decomposition(
    dataset: DetDataset,
    predictions: Dict[int, DetPrediction],
    fine_bins: bool = True,
) -> Dict[str, float]:
    """整体指标 + 目标尺寸桶分解（论文里的 AP_nano/tiny/small 列）。"""
    bins = FINE_AREA_BINS if fine_bins else AREA_BINS
    m = compute_map_range(
        dataset.samples,
        predictions,
        iou_thresholds=COCO_IOU_THRESHOLDS,
        area_bins=bins,
        num_classes=dataset.num_classes,
    )
    return {k: float(v) for k, v in m.items() if v is not None}


# --------------------------------------------------------------------------- #
# 顶层入口
# --------------------------------------------------------------------------- #


def evaluate(
    adapter: DetectorAdapter,
    dataset: DetDataset,
    image_root: str,
    exp_id: str,
    out_dir: Optional[str] = None,
    group_specs: Optional[Sequence[GroupSpec]] = None,
    speed: bool = True,
    speed_warmup: int = 20,
    speed_iters: int = 100,
    predict_conf: float = 0.001,
    max_dets: int = 100,
    fine_bins: bool = True,
    config: Optional[Dict[str, Any]] = None,
    notes: str = "",
    train_seconds: float = 0.0,
) -> ExperimentResult:
    """完整评测流程：推理 -> 标准 mAP -> 分层 mAP -> 速度 -> 落盘。

    Parameters
    ----------
    predict_conf : 推理时保留的最低分数。**算 mAP 必须设得很低（默认 0.001）**，
                   否则会人为砍掉低分预测、虚高 precision。
                   部署时的实际阈值是另一回事，不要混用。
    max_dets : 每图参与评测的最大预测数，对齐 COCO 的 maxDets=100
    """
    print(f"[{exp_id}] 数据集 {dataset.name}: {dataset.num_images} 图 / {dataset.num_objects} 目标")

    # ---- 1. 推理 ----
    t_inf = time.time()
    preds = run_inference(adapter, dataset, image_root)
    infer_seconds = time.time() - t_inf

    # 评测时统一做低阈值过滤，保证各方法使用同一标准
    preds_eval = {i: p.filter_by_score(predict_conf) for i, p in preds.items()}

    # ---- 2. 标准 mAP ----
    metrics = build_size_decomposition(dataset, preds_eval, fine_bins=fine_bins)
    print(
        f"[{exp_id}] mAP@[.5:.95]={metrics.get('mAP', 0):.4f}  "
        f"mAP@0.5={metrics.get('mAP50') or 0:.4f}"
    )

    # ---- 3. 分层评测 ----
    stratified: Dict[str, Dict[str, Dict[str, float]]] = {}
    if group_specs:
        stratified = evaluate_stratified(dataset, preds_eval, group_specs, fine_bins=fine_bins)
        for k, groups in stratified.items():
            preview = ", ".join(
                f"{g}={v.get('mAP', float('nan')):.3f}" for g, v in list(groups.items())[:6]
            )
            print(f"[{exp_id}] 分层[{k}]: {preview}")

    # ---- 4. 速度 ----
    # 测速依赖 torch（需要 cuda synchronize / 显存统计）。没装 torch 时
    # 只跳过测速，精度评测照常完成 —— 这样在只有 numpy 的环境里也能打通全链路。
    speed_res: Dict[str, Any] = {}
    want_speed = speed and bool(dataset.samples)
    if want_speed and not torch_available():
        print(f"[{exp_id}] 未检测到 torch，跳过速度评测（精度指标不受影响）")
        want_speed = False
    if want_speed:
        probe = dataset.samples[0]
        adapter.warmup(probe, image_root, iters=3)
        sr = measure_speed(
            forward_fn=lambda: adapter.predict(probe, image_root),
            input_size=str(adapter.describe().get("input_size", "")),
            batch_size=1,
            device=adapter.device,
            warmup=speed_warmup,
            iters=speed_iters,
            model=adapter.torch_model,
        )
        speed_res = sr.to_dict()
        print(
            f"[{exp_id}] {speed_res['fps']:.1f} FPS  "
            f"{speed_res['latency_ms']:.1f} ms  "
            f"峰值显存 {speed_res['peak_vram_mb']:.0f} MB"
        )

    # ---- 5. 汇总落盘 ----
    desc = adapter.describe()
    result = ExperimentResult(
        exp_id=exp_id,
        method=str(desc.get("method", adapter.name)),
        dataset=dataset.name,
        split=str(desc.get("split", "test")),
        backbone=str(desc.get("backbone", "")),
        input_size=str(desc.get("input_size", "")),
        epochs=int(desc.get("epochs", 0)),
        seed=int(desc.get("seed", 0)),
        metrics=metrics,
        fps=float(speed_res.get("fps", 0.0)),
        latency_ms=float(speed_res.get("latency_ms", 0.0)),
        peak_vram_mb=float(speed_res.get("peak_vram_mb", 0.0)),
        params_m=float(speed_res.get("params_m", 0.0)),
        flops_g=float(speed_res.get("flops_g", 0.0)),
        stratified=stratified,
        config=dict(config or {}),
        dataset_summary=dataset.summary(),
        train_seconds=train_seconds,
        notes=notes or f"infer={infer_seconds:.1f}s, max_dets={max_dets}, conf>={predict_conf}",
    )
    if out_dir:
        path = result.save(out_dir)
        print(f"[{exp_id}] 结果已保存：{path}")
        # 顺手存一份原始预测，方便事后换指标重算而不用重跑推理
        _dump_predictions(preds_eval, out_dir)
    return result


def _dump_predictions(preds: Dict[int, DetPrediction], out_dir: str) -> None:
    """把预测存成 npz。

    这一步很关键：**换了评测协议或指标定义时，可以直接重算，不用重跑推理。**
    推理往往是整个流程里最慢的一环（尤其 DETR 系）。
    """
    from pathlib import Path

    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    ids = sorted(preds)
    np.savez_compressed(
        out / "predictions.npz",
        image_ids=np.asarray(ids, dtype=np.int64),
        boxes=np.concatenate([preds[i].boxes for i in ids]) if ids else np.zeros((0, 4), dtype=np.float32),
        scores=np.concatenate([preds[i].scores for i in ids]) if ids else np.zeros((0,), dtype=np.float32),
        labels=np.concatenate([preds[i].labels for i in ids]) if ids else np.zeros((0,), dtype=np.int64),
        offsets=np.cumsum([0] + [preds[i].scores.size for i in ids]).astype(np.int64),
    )
