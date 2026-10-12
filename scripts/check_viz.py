"""检查 core/viz.py 的全部绘图函数能否正常出图。

需要 matplotlib。用占位数据集 + 占位检测器产生真实形状的数据，
然后调用全部 5 个绘图函数，检查输出文件是否有效（存在、非空、是合法图片）。

之所以要单独测：绘图函数是最容易被忽略的一类代码 —— 它们不参与训练和评测，
一旦出错往往到写论文要出图时才发现，那时候时间最紧。

用法::

    pip install matplotlib
    python scripts/check_viz.py
"""

from __future__ import annotations

import shutil
import struct
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT.parent) not in sys.path:
    sys.path.insert(0, str(ROOT.parent))

import numpy as np  # noqa: E402

from detbench.core.eval import build_size_decomposition, evaluate_stratified  # noqa: E402
from detbench.core.io import load_coco  # noqa: E402
from detbench.core.metrics import FINE_AREA_BINS, GroupSpec, box_iou_matrix  # noqa: E402
from detbench.models.dummy import DummyAdapter  # noqa: E402

DATA = ROOT / "data" / "demo"
OUT = ROOT / ".viz_check"

RESULTS: list[tuple[str, bool, str]] = []


def check(name: str, cond: bool, detail: str = "") -> None:
    RESULTS.append((name, cond, detail))
    print(f"[{'PASS' if cond else 'FAIL'}] {name}" + (f"  -- {detail}" if detail else ""))


def image_info(p: Path) -> str:
    """读 PNG 头，返回 '宽x高, N KB'。顺便验证它是合法 PNG。"""
    raw = p.read_bytes()
    if len(raw) < 24 or raw[:8] != b"\x89PNG\r\n\x1a\n":
        return "不是合法 PNG"
    w, h = struct.unpack(">II", raw[16:24])
    return f"{w}x{h}, {len(raw) / 1024:.1f} KB"


def main() -> int:
    print("=" * 74)
    print("可视化函数检查 (core/viz.py)")
    print("=" * 74)

    try:
        import matplotlib

        print(f"matplotlib {matplotlib.__version__}\n")
    except ImportError:
        print("❌ 需要 matplotlib：pip install matplotlib")
        return 1

    ann = DATA / "annotations" / "instances_test.json"
    if not ann.exists():
        print("找不到占位数据集，请先运行：python tools/main.py make-demo")
        return 1
    if OUT.exists():
        shutil.rmtree(OUT, ignore_errors=True)
    OUT.mkdir(parents=True, exist_ok=True)

    from detbench.core import viz

    ds = load_coco(
        ann,
        str(DATA / "images"),
        name="demo-test",
        attr_file=str(DATA / "metadata" / "frames.csv"),
    )

    # ---- 1. 数据集统计图（Dataset 章节用，不需要模型）----
    p = viz.plot_dataset_stats(ds, OUT / "01_dataset_stats.png")
    check("plot_dataset_stats", p.exists() and p.stat().st_size > 5000, image_info(p))

    # ---- 准备两个"方法"的预测，用于多方法对比图 ----
    preds_a = {s.image_id: DummyAdapter(mode="perfect", seed=0).predict(s, "") for s in ds.samples}
    preds_b = {s.image_id: DummyAdapter(mode="jitter", seed=1).predict(s, "") for s in ds.samples}

    # ---- 2. 分层柱状图：条件 mAP ----
    specs = [GroupSpec("distance_bin", min_gt=2), GroupSpec("__size__", min_gt=2)]
    strat_a = evaluate_stratified(ds, preds_a, specs)
    strat_b = evaluate_stratified(ds, preds_b, specs)
    by_method = {
        "Detector A": strat_a["distance_bin"],
        "Detector B": strat_b["distance_bin"],
    }
    p = viz.plot_stratified_bars(
        by_method, "distance_bin", OUT / "02_stratified_bars.png",
        title="mAP by distance bin",
    )
    check("plot_stratified_bars", p.exists() and p.stat().st_size > 4000, image_info(p))

    # ---- 3. 尺度-mAP 曲线（小目标论文最有力的一张图）----
    def size_buckets(m):
        return {k.split("/", 1)[1]: v for k, v in m.items() if k.startswith("mAP/")}

    curves = {
        "Detector A": size_buckets(build_size_decomposition(ds, preds_a)),
        "Detector B": size_buckets(build_size_decomposition(ds, preds_b)),
    }
    print(f"    尺度桶: {sorted(curves['Detector A'])}")
    p = viz.plot_scale_curve(curves, OUT / "03_scale_curve.png")
    check("plot_scale_curve", p.exists() and p.stat().st_size > 4000, image_info(p))

    # ---- 4. PR 曲线 ----
    from detbench.core.metrics import compute_ap

    pr = {}
    for name, pr_preds in (("Detector A", preds_a), ("Detector B", preds_b)):
        res = compute_ap(ds.samples, pr_preds, iou_threshold=0.5, num_classes=1, return_curves=True)
        pr[name] = (res["_curve"]["recall"], res["_curve"]["precision"])
    p = viz.plot_pr_curves(pr, OUT / "04_pr_curves.png")
    check("plot_pr_curves", p.exists() and p.stat().st_size > 3000, image_info(p))

    # ---- 5. 失败案例网格（需要真实图像文件）----
    p = viz.plot_failure_cases(
        ds.samples, preds_b, DATA / "images", OUT / "05_failure_cases.png",
        score_threshold=0.0, max_images=8, cols=4,
    )
    check("plot_failure_cases", p.exists() and p.stat().st_size > 10000, image_info(p))

    # ---- 6. 边界情况：空数据不应崩溃 ----
    try:
        viz.plot_stratified_bars({}, "nope", OUT / "06_empty.png")
        check("空分组应抛明确错误", False, "没有抛异常")
    except ValueError as exc:
        check("空分组抛 ValueError 且信息清晰", "没有任何数据" in str(exc), str(exc)[:50])
    except Exception as exc:
        check("空分组应抛 ValueError", False, f"抛的是 {type(exc).__name__}")

    # ---- 7. 边界情况：没有目标的数据集 ----
    from detbench.core.io import DetDataset

    empty_ds = DetDataset(name="empty", samples=[], class_names=["x"])
    try:
        viz.plot_dataset_stats(empty_ds, OUT / "07_empty_ds.png")
        check("空数据集统计图不崩溃", True, "生成成功")
    except Exception as exc:
        check("空数据集统计图不崩溃", False, f"{type(exc).__name__}: {exc}")

    # ---- 8. 中文字体检查（论文图里常有中文标签）----
    from matplotlib import font_manager

    cjk = [
        f.name for f in font_manager.fontManager.ttflist
        if any(k in f.name for k in ("SimHei", "Microsoft YaHei", "SimSun", "Noto Sans CJK", "Source Han"))
    ]
    if cjk:
        check("系统有中文字体可用", True, f"例如 {sorted(set(cjk))[:3]}")
    else:
        check("系统有中文字体可用", False, "没有中文字体，中文标签会显示成方框")

    # ---- 汇总 ----
    print("\n生成的图片：")
    for f in sorted(OUT.glob("*.png")):
        print(f"  {f.name:28s} {image_info(f)}")

    print("=" * 74)
    passed = sum(1 for _, c, _ in RESULTS if c)
    total = len(RESULTS)
    if passed == total:
        print(f"全部 {total} 项通过。")
        print(f"图片在 {OUT}，可以直接打开看效果。")
        return 0
    print(f"{total - passed}/{total} 项失败：")
    for n, c, d in RESULTS:
        if not c:
            print(f"  - {n}  {d}")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
