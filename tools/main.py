#!/usr/bin/env python
"""DetBench 命令行入口。

四个子命令::

    # 1. 生成占位数据集（无需真实数据即可打通管线）
    python tools/main.py make-demo --out data/demo

    # 2. 自检：用占位检测器验证评测实现是否正确（强烈建议每次改完代码都跑）
    python tools/main.py selftest

    # 3. 评测一个模型
    python tools/main.py eval --config configs/example.yaml

    #   4. 汇总所有实验，生成论文表格
    python tools/main.py report --runs runs --out report.md

    # 5. 生成论文配图（需要 matplotlib）
    python tools/main.py figures --runs runs --out figs --config configs/example.yaml
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

# 让脚本能直接运行：把 **包所在的父目录** 加入 sys.path，
# 这样 `detbench` 是一个正规包，内部一律使用绝对导入。
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT.parent) not in sys.path:
    sys.path.insert(0, str(ROOT.parent))

from detbench.core.eval import COCO_IOU_THRESHOLDS, evaluate  # noqa: E402
from detbench.core.io import DetDataset, load_coco  # noqa: E402
from detbench.core.metrics import FINE_AREA_BINS, GroupSpec  # noqa: E402
from detbench.core.results import (  # noqa: E402
    collect_results,
    to_latex_table,
    to_markdown_table,
    to_stratified_markdown,
    write_report,
)


# --------------------------------------------------------------------------- #
# 配置载入
# --------------------------------------------------------------------------- #


def _selftest_dir() -> Path:
    """自检的临时输出目录（放在项目内，避开不可写的系统临时目录）。"""
    return ROOT / ".selftest_out"


def _load_config(path: str | Path, sets=None, data_root=None, check_paths=True) -> dict:
    """载入 YAML / JSON 配置，并完成环境变量展开、``--set`` 覆盖、路径解析。

    YAML 走 detbench 内置的零依赖解析器（装了 PyYAML 则优先用），
    所以在离线或受限环境里也不会有"装不上依赖"的问题。
    路径全部解析为绝对路径：数据路径相对 data_root，权重路径相对配置文件目录，
    因此从任何工作目录调用都能得到同样的结果。
    """
    from detbench.core.config import load_and_resolve

    return load_and_resolve(path, overrides=sets, data_root=data_root, check_paths=check_paths)


def _build_dataset(cfg: dict) -> DetDataset:
    d = cfg.get("dataset", {})
    fmt = d.get("format", "coco")
    attr_file = d.get("attr_file")
    if fmt == "coco":
        return load_coco(
            d["ann_file"],
            d.get("images_root", ""),
            name=d.get("name", "dataset"),
            attr_file=attr_file,
        )
    if fmt == "yolo":
        from detbench.core.io import load_yolo

        return load_yolo(
            d["images_dir"],
            d["labels_dir"],
            class_names=d["class_names"],
            name=d.get("name", "dataset"),
            attr_file=attr_file,
        )
    if fmt == "voc":
        from detbench.core.io import load_voc

        return load_voc(
            d["ann_dir"],
            d.get("images_root", ""),
            class_names=d["class_names"],
            name=d.get("name", "dataset"),
            attr_file=attr_file,
        )
    raise SystemExit(f"不支持的 dataset.format: {fmt}")


def _build_groups(cfg: dict) -> list[GroupSpec]:
    specs: list[GroupSpec] = []
    for g in cfg.get("eval", {}).get("groups", []) or []:
        key = g["key"] if isinstance(g, dict) else str(g)
        bins = FINE_AREA_BINS if (isinstance(g, dict) and g.get("bins") == "fine") else None
        min_gt = g.get("min_gt", 5) if isinstance(g, dict) else 5
        specs.append(GroupSpec(key=key, bins=bins, min_gt=min_gt))
    return specs


# --------------------------------------------------------------------------- #
# 子命令
# --------------------------------------------------------------------------- #


def cmd_make_demo(args: argparse.Namespace) -> int:
    from detbench.scripts.make_demo_dataset import generate

    generate(args.out, args.clips, args.frames, args.width, args.height, args.seed)
    return 0


def cmd_selftest(args: argparse.Namespace) -> int:
    """用占位检测器验证评测实现。

    这是本框架最重要的一个自检：指标实现的 bug 往往表现为"mAP 偏低"，
    不报错、不崩溃，靠肉眼极难发现。用已知答案的合成场景才能测出来。
    """
    import numpy as np

    from detbench.core.io import DetPrediction, DetSample
    from detbench.models.dummy import DummyAdapter

    print("=" * 68)
    print("DetBench 自检")
    print("=" * 68)
    failures: list[str] = []

    def check(name: str, cond: bool, detail: str = "") -> None:
        status = "PASS" if cond else "FAIL"
        print(f"[{status}] {name}" + (f"  -- {detail}" if detail else ""))
        if not cond:
            failures.append(name)

    # ---- 1. IoU 几何 ----
    from detbench.core.metrics import box_iou_matrix, compute_map_range

    a = np.array([[0, 0, 10, 10]], dtype=np.float32)
    b = np.array([[0, 0, 10, 10], [5, 0, 15, 10], [20, 20, 30, 30]], dtype=np.float32)
    iou = box_iou_matrix(a, b)[0]
    check("IoU: 完全重合 = 1.0", abs(iou[0] - 1.0) < 1e-6, f"got {iou[0]:.6f}")
    check("IoU: 半重叠 = 1/3", abs(iou[1] - 1.0 / 3.0) < 1e-5, f"got {iou[1]:.6f}")
    check("IoU: 不相交 = 0.0", abs(iou[2]) < 1e-9, f"got {iou[2]:.6f}")

    # ---- 2. 构造已知答案的场景 ----
    samples = [
        DetSample(
            image_id=1,
            file_name="a.jpg",
            width=1000,
            height=1000,
            boxes=np.array([[10, 10, 30, 30], [100, 100, 200, 200]], dtype=np.float32),  # 20px, 100px
            labels=np.array([0, 0]),
            attrs={"blur": ["light", "heavy"], "distance_bin": "near"},
        ),
        DetSample(
            image_id=2,
            file_name="b.jpg",
            width=1000,
            height=1000,
            boxes=np.array([[50, 50, 56, 56]], dtype=np.float32),  # 6px -> nano 桶
            labels=np.array([0]),
            attrs={"blur": ["clear"], "distance_bin": "far"},
        ),
    ]

    perfect = DummyAdapter(mode="perfect")
    preds_perfect = {s.image_id: perfect.predict(s, "") for s in samples}
    m_perfect = compute_map_range(
        samples, preds_perfect, iou_thresholds=COCO_IOU_THRESHOLDS, area_bins=FINE_AREA_BINS, num_classes=1
    )
    check(
        "perfect 模式 mAP ≈ 1.0",
        m_perfect["mAP"] > 0.99,
        f"got {m_perfect['mAP']:.4f}",
    )

    empty = DummyAdapter(mode="empty")
    preds_empty = {s.image_id: empty.predict(s, "") for s in samples}
    m_empty = compute_map_range(
        samples, preds_empty, iou_thresholds=COCO_IOU_THRESHOLDS, area_bins=FINE_AREA_BINS, num_classes=1
    )
    check("empty 模式 mAP = 0.0", abs(m_empty["mAP"]) < 1e-9, f"got {m_empty['mAP']:.4f}")

    # ---- 3. 尺寸分桶是否落在正确的桶里 ----
    check(
        "尺寸分桶: nano 桶含 6px 目标",
        abs(m_perfect.get("mAP/nano(<8px)", -1) - 1.0) < 1e-6,
        f"got {m_perfect.get('mAP/nano(<8px)')}",
    )
    check(
        "尺寸分桶: small 桶含 20px 目标",
        abs(m_perfect.get("mAP/small(16-32px)", -1) - 1.0) < 1e-6,
        f"got {m_perfect.get('mAP/small(16-32px)')}",
    )
    check(
        "尺寸分桶: large 桶含 100px 目标",
        abs(m_perfect.get("mAP/large(>96px)", -1) - 1.0) < 1e-6,
        f"got {m_perfect.get('mAP/large(>96px)')}",
    )

    # ---- 4. 多预测时的贪心匹配（一个 GT 只能被命中一次）----
    dup = {
        1: DetPrediction(
            1,
            np.array([[10, 10, 30, 30], [11, 11, 31, 31]], dtype=np.float32),
            np.array([0.9, 0.8], dtype=np.float32),
            np.array([0, 0], dtype=np.int64),
        ),
        2: DetPrediction(2, np.zeros((0, 4), np.float32), np.zeros((0,), np.float32), np.zeros((0,), np.int64)),
    }
    m_dup = compute_map_range(
        samples, dup, iou_thresholds=(0.5,), area_bins=None, num_classes=1
    )
    # 图 1 的两个预测都与同一个 GT 高度重合 -> 只有 1 个算 TP，
    # 第 2 个预测因为找不到"尚未被匹配的 GT"而降级为 FP（任何 IoU 阈值下都一样）。
    # 这正是贪心匹配的核心行为，也是最容易实现错的地方。
    check(
        "重复预测只算一次 TP（第二个降级为 FP）",
        m_dup["num_tp"] == 1 and m_dup["num_pred"] == 2,
        f"num_tp={m_dup['num_tp']}, num_pred={m_dup['num_pred']} (期望 1 / 2)",
    )

    # ---- 5. 分层评测：目标级属性拆分 ----
    specs = [GroupSpec(key="blur", min_gt=1), GroupSpec(key="distance_bin", min_gt=1)]
    from detbench.core.eval import evaluate_stratified

    ds_selftest = DetDataset(name="selftest", samples=samples, class_names=["helicopter"])
    strat = evaluate_stratified(ds_selftest, preds_perfect, specs)
    check("分层: blur 组存在", "blur" in strat and len(strat["blur"]) >= 2, f"groups={list(strat.get('blur', {}))}")
    check(
        "分层: 目标级属性把同图拆到不同组",
        set(strat.get("blur", {})) >= {"light", "heavy", "clear"},
        f"groups={sorted(strat.get('blur', {}))}",
    )

    # ---- 6. 端到端落盘与汇总 ----
    import shutil

    # 注意：这里刻意不用 tempfile —— 在沙箱/受限环境下系统临时目录可能不可写。
    # 统一落到工作区内的目录，跑完删掉。
    tmp_root = Path(_selftest_dir())
    if tmp_root.exists():
        shutil.rmtree(tmp_root, ignore_errors=True)
    tmp_root.mkdir(parents=True, exist_ok=True)
    try:
        ds = DetDataset(name="selftest", samples=samples, class_names=["helicopter"])
        r = evaluate(
            adapter=DummyAdapter(mode="perfect"),
            dataset=ds,
            image_root="",
            exp_id="selftest_perfect",
            out_dir=str(tmp_root),
            group_specs=specs,
            speed=False,
            config={"note": "selftest"},
        )
        check("结果落盘 result.json", (tmp_root / "result.json").exists())
        check("预测落盘 predictions.npz", (tmp_root / "predictions.npz").exists())
        loaded = collect_results(tmp_root)
        check("结果可回读", len(loaded) == 1 and loaded[0].exp_id == "selftest_perfect")
        md = to_markdown_table(loaded)
        check("生成 Markdown 表", "mAP" in md and len(md.splitlines()) >= 3)
        tex = to_latex_table(loaded)
        check("生成 LaTeX 表", "\\begin{table}" in tex)
        smd = to_stratified_markdown(loaded)
        check("生成分层表", "blur" in smd)
    finally:
        shutil.rmtree(tmp_root, ignore_errors=True)

    print("=" * 68)
    if failures:
        print(f"自检未通过：{len(failures)} 项失败 -> {failures}")
        return 1
    print("全部自检通过。评测实现可信，可以开始跑真实实验。")
    return 0


def _print_cfg(cfg: dict) -> None:
    """打印解析后的关键路径，便于换设备后确认路径是否指向正确位置。"""
    print(f"  配置文件  : {cfg.get('_config_path')}")
    print(f"  data_root : {cfg.get('data_root', '(未设置)')}")
    ds = cfg.get("dataset", {})
    for k in ("format", "name", "ann_file", "images_root", "images_dir", "labels_dir", "attr_file"):
        if ds.get(k):
            print(f"  dataset.{k:12s}: {ds[k]}")
    m = cfg.get("model", {})
    for k in ("type", "config", "checkpoint", "weights"):
        if m.get(k):
            print(f"  model.{k:14s}: {m[k]}")

    # 顺手检查路径是否存在 —— 换设备后最常见的问题就是路径没对上
    print("\n  路径存在性检查：")
    for label, p in (
        ("ann_file", ds.get("ann_file")),
        ("images_root", ds.get("images_root") or ds.get("images_dir")),
        ("attr_file", ds.get("attr_file")),
        ("checkpoint", m.get("checkpoint") or m.get("weights")),
        ("model.config", m.get("config")),
    ):
        if not p:
            continue
        exists = Path(p).exists()
        print(f"    [{'OK ' if exists else '缺失'}] {label:14s} {p}")


def cmd_doctor(args: argparse.Namespace) -> int:
    """依赖体检：一眼看出哪些功能可用、哪些装不上就用不了。

    先跑这个再跑 eval，能省掉很多"跑到一半才报 ImportError"的时间。
    """
    import importlib

    def probe(mod: str) -> Optional[str]:
        try:
            m = importlib.import_module(mod)
            return getattr(m, "__version__", "?")
        except Exception:
            return None

    rows = [
        ("numpy", "核心（必装）", True),
        ("PIL", "核心（读写图像）", True),
        ("detbench.core.config", "核心（配置解析，内置）", True),
        ("torch", "测速 / 所有真实模型", False),
        ("torchvision", "Faster R-CNN / RetinaNet 基线", False),
        ("mmengine", "mmdet 的前置依赖", False),
        ("mmdet", "★ DETR 全家族 / Deformable DETR / DINO-DETR", False),
        ("mmcv", "mmdet 的前置依赖", False),
        ("ultralytics", "YOLOv5/v8/v11", False),
        ("matplotlib", "论文配图 (core/viz.py)", False),
        ("yaml", "YAML 配置（未装则用内置解析器，不影响）", False),
        ("thop", "FLOPs 统计（可选）", False),
    ]

    print("=" * 72)
    print("DetBench 依赖体检")
    print("=" * 72)
    print(f"{'模块':<24} {'版本':<12} {'状态':<8} 用途")
    print("-" * 72)
    missing_required: List[str] = []
    for mod, purpose, required in rows:
        ver = probe(mod)
        status = "OK" if ver else ("缺失!" if required else "未装")
        print(f"{mod:<24} {str(ver or '-'):<12} {status:<8} {purpose}")
        if required and not ver:
            missing_required.append(mod)

    print("-" * 72)
    if missing_required:
        print(f"❌ 缺少必需依赖：{missing_required}")
        return 1
    print("✅ 核心依赖完整。评测 / 分层统计 / 结果汇总均可用。")
    try:
        import torch  # noqa: F401

        print("✅ torch 可用，速度评测（FPS / 延迟 / 峰值显存）已启用。")
    except Exception:
        print("ℹ️  未装 torch：速度评测会自动跳过，精度指标不受影响。")
    print()
    print("建议安装（按需）：")
    print("    pip install torch torchvision      # 真实模型与测速")
    print("    pip install mmengine mmcv mmdet    # ★ DETR 全家族（Deformable DETR / DINO-DETR）")
    print("    pip install ultralytics            # YOLO 系")
    print("    pip install matplotlib             # 论文配图")
    print()
    print("提示：DETR 系适配器还需要把 mmdetection 仓库 clone 到本地，")
    print("      因为 config 参数只接受本地 .py 文件（不会联网下载配置）。")
    return 0


def cmd_eval(args: argparse.Namespace) -> int:
    dry_run = bool(getattr(args, "dry_run", False))
    # dry-run 的目的就是"诊断路径"，所以不能因为路径不存在就直接报错退出 ——
    # 否则用户永远看不到解析结果。改成把缺失项作为诊断信息打印出来。
    cfg = _load_config(
        args.config,
        sets=getattr(args, "set", None),
        data_root=getattr(args, "data_root", None),
        check_paths=not dry_run and not getattr(args, "no_check_paths", False),
    )

    if dry_run:
        print("[dry-run] 配置解析结果（不执行任何推理）：")
        _print_cfg(cfg)
        out_dir = getattr(args, "out", None) or cfg.get("output", {}).get("dir", "runs/exp")
        print(f"\n  输出目录  : {out_dir}")
        # 顺便试着加载数据，确认标注文件本身是可解析的
        try:
            from detbench.core.registry import build  # noqa: F401

            dataset = _build_dataset(cfg)
            print(f"  数据集    : {dataset.num_images} 图 / {dataset.num_objects} 目标 / "
                  f"{dataset.num_classes} 类 {dataset.class_names}")
        except Exception as exc:
            print(f"  ⚠️ 数据集加载失败（路径存在但内容有问题？）：{type(exc).__name__}: {exc}")
            return 1
        print("\ndry-run 通过：配置与数据都能正确解析，可以正式评测。")
        return 0

    from detbench.core.registry import build

    dataset = _build_dataset(cfg)
    groups = _build_groups(cfg)

    image_root = cfg["dataset"].get("images_root") or cfg["dataset"].get("images_dir") or ""
    device = getattr(args, "device", None) or cfg.get("runtime", {}).get("device", "cuda")

    import detbench.models  # noqa: F401  触发适配器注册

    adapter = build(cfg["model"], device=device)

    exp_id = getattr(args, "exp_id", None) or cfg.get("exp_id") or cfg["model"].get("type", "exp")
    out_dir = getattr(args, "out", None) or cfg.get("output", {}).get("dir", f"runs/{exp_id}")

    evaluate(
        adapter=adapter,
        dataset=dataset,
        image_root=image_root,
        exp_id=exp_id,
        out_dir=out_dir,
        group_specs=groups,
        speed=cfg.get("eval", {}).get("speed", True),
        predict_conf=cfg.get("eval", {}).get("predict_conf", 0.001),
        max_dets=cfg.get("eval", {}).get("max_dets", 100),
        fine_bins=cfg.get("eval", {}).get("fine_bins", True),
        config=cfg,
        notes=cfg.get("notes", ""),
    )
    return 0


def cmd_report(args: argparse.Namespace) -> int:
    results = collect_results(args.runs)
    if not results:
        print(f"在 {args.runs} 下没找到任何 result.json，先跑 eval。")
        return 1

    print("=" * 100)
    print(to_markdown_table(results))
    print("=" * 100)

    strat = to_stratified_markdown(results)
    if strat:
        print()
        print(strat)

    out = Path(args.out)
    write_report(results, out)
    print(f"\n报告已写入：{out.resolve()}")

    if args.latex:
        tex = to_latex_table(results)
        tex_path = out.with_suffix(".tex")
        tex_path.write_text(tex, encoding="utf-8")
        print(f"LaTeX 表已写入：{tex_path.resolve()}")
    return 0


def cmd_figures(args: argparse.Namespace) -> int:
    """把 runs/ 里的实验记录画成论文配图。

    这是 core/viz.py 的入口 —— 没有它，那些绘图函数就是死代码。

    能画什么，取决于手里有什么数据：

    ============================  ==========================================
    图                            依赖
    ============================  ==========================================
    尺度-mAP 曲线                 result.json 里的 mAP/<尺寸桶> 指标
    条件 mAP 柱状图（每个属性）    result.json 里的 stratified
    PR 曲线                       runs/*/predictions.npz
    失败案例网格                  predictions.npz + 原始图像
    数据集统计图                  --config 指定的数据集
    ============================  ==========================================
    """
    try:
        from detbench.core import viz
    except Exception:
        pass
    try:
        import matplotlib  # noqa: F401
    except ImportError:
        print(
            "❌ 出图需要 matplotlib：\n"
            "    pip install matplotlib\n"
            "（评测、分层统计、结果汇总都不依赖它。）"
        )
        return 1

    from detbench.core import viz
    from detbench.core.eval import load_predictions

    results = collect_results(args.runs)
    if not results:
        print(f"在 {args.runs} 下没找到任何 result.json，先跑 eval。")
        return 1

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"找到 {len(results)} 个实验，输出目录 {out_dir.resolve()}\n")

    made: list[Path] = []

    def try_plot(label: str, fn, *a, **kw) -> None:
        """单个图失败不应中断其余图 —— 出图是最后一步，能出一张是一张。"""
        try:
            p = fn(*a, **kw)
            made.append(p)
            print(f"  ✓ {label:34s} {p.name}")
        except viz.VizUnavailable:
            raise
        except Exception as exc:
            print(f"  ✗ {label:34s} {type(exc).__name__}: {exc}")

    # ---- 1. 尺度-mAP 曲线（小目标论文最有力的一张图）----
    curves: dict[str, dict[str, float]] = {}
    for r in results:
        buckets = {k.split("/", 1)[1]: v for k, v in r.metrics.items() if k.startswith("mAP/")}
        if buckets:
            curves[r.method if r.method not in curves else r.exp_id] = buckets
    if curves:
        try_plot("尺度-mAP 曲线", viz.plot_scale_curve, curves, out_dir / "scale_curve.png")
    else:
        print("  - 尺度-mAP 曲线                    跳过（result.json 里没有 mAP/<尺寸桶> 指标）")

    # ---- 2. 每个属性维度的条件柱状图 ----
    group_keys: list[str] = []
    for r in results:
        for k in r.stratified:
            if k not in group_keys:
                group_keys.append(k)
    for key in sorted(group_keys):
        by_method = {}
        for r in results:
            if key in r.stratified:
                by_method[r.method if r.method not in by_method else r.exp_id] = r.stratified[key]
        if by_method:
            try_plot(
                f"条件 mAP 柱状图 [{key}]",
                viz.plot_stratified_bars,
                by_method, key, out_dir / f"stratified_{key}.png",
            )

    # ---- 3. PR 曲线（需要 predictions.npz）----
    pr: dict[str, tuple] = {}
    for r in results:
        npz = Path(args.runs) / r.exp_id / "predictions.npz"
        if not npz.exists():
            # 也试试 result.json 所在目录
            cands = list(Path(args.runs).glob(f"**/{r.exp_id}/predictions.npz"))
            npz = cands[0] if cands else npz
        if not npz.exists():
            continue
        try:
            preds = load_predictions(npz)
            ds = _dataset_for_figures(r, args)
            if ds is None:
                continue
            from detbench.core.metrics import compute_ap

            res = compute_ap(
                ds.samples, preds, iou_threshold=0.5,
                num_classes=ds.num_classes, return_curves=True,
            )
            if "_curve" in res:
                pr[r.method] = (res["_curve"]["recall"], res["_curve"]["precision"])
        except Exception as exc:
            print(f"  - PR 曲线 [{r.exp_id}] 跳过：{exc}")
    if pr:
        try_plot("PR 曲线 (IoU=0.5)", viz.plot_pr_curves, pr, out_dir / "pr_curves.png")
    else:
        print("  - PR 曲线                          跳过（找不到 predictions.npz 或数据集）")

    # ---- 4. 失败案例网格（需要 predictions + 图像）----
    for r in results[: args.max_failure_sets]:
        npz_cands = list(Path(args.runs).glob(f"**/{r.exp_id}/predictions.npz"))
        if not npz_cands:
            continue
        ds = _dataset_for_figures(r, args)
        if ds is None:
            continue
        try:
            preds = load_predictions(npz_cands[0])
            image_root = _image_root_for_figures(r, args)
            if image_root is None:
                continue
            try_plot(
                f"失败案例 [{r.method}]",
                viz.plot_failure_cases,
                ds.samples, preds, image_root,
                out_dir / f"failure_cases_{r.exp_id}.png",
                max_images=args.max_images,
            )
        except Exception as exc:
            print(f"  - 失败案例 [{r.exp_id}] 跳过：{exc}")

    # ---- 5. 数据集统计图（需要 --config）----
    if args.config:
        try:
            cfg = _load_config(args.config, data_root=args.data_root, check_paths=False)
            ds = _build_dataset(cfg)
            try_plot("数据集统计图", viz.plot_dataset_stats, ds, out_dir / "dataset_stats.png")
        except Exception as exc:
            print(f"  - 数据集统计图                      跳过：{exc}")
    else:
        print("  - 数据集统计图                      跳过（加 --config 可生成）")

    print()
    if made:
        print(f"共生成 {len(made)} 张图，在 {out_dir.resolve()}")
        print("这些图可直接放进论文；矢量图请在 viz.py 里把 savefig 后缀改成 .pdf")
    else:
        print("没有生成任何图。请检查上面的跳过原因。")
        return 1
    return 0


def _dataset_for_figures(result, args):
    """为某个实验重建数据集对象（画 PR 曲线与失败案例需要）。"""
    cfg = result.config or {}
    ds_cfg = (cfg.get("dataset") or {}) if isinstance(cfg, dict) else {}
    ann = ds_cfg.get("ann_file")
    if not ann or not Path(str(ann)).exists():
        return None
    from detbench.core.io import load_coco

    attr = ds_cfg.get("attr_file")
    return load_coco(
        ann,
        ds_cfg.get("images_root", ""),
        name=ds_cfg.get("name", "dataset"),
        attr_file=attr if attr and Path(str(attr)).exists() else None,
    )


def _image_root_for_figures(result, args):
    """取图像根目录（画失败案例需要读原图）。"""
    cfg = result.config or {}
    ds_cfg = (cfg.get("dataset") or {}) if isinstance(cfg, dict) else {}
    root = ds_cfg.get("images_root") or ds_cfg.get("images_dir")
    if root and Path(str(root)).exists():
        return str(root)
    if args.images_root and Path(args.images_root).exists():
        return args.images_root
    return None


# --------------------------------------------------------------------------- #
# 参数解析
# --------------------------------------------------------------------------- #


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(
        prog="detbench",
        description="面向小目标检测的可复现评测框架",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    sub = ap.add_subparsers(dest="cmd", required=True)

    p = sub.add_parser("make-demo", help="生成占位数据集")
    p.add_argument("--out", default="data/demo")
    # 默认值必须与 scripts/make_demo_dataset.generate 的签名保持一致，
    # 否则子命令的默认值会静默覆盖函数默认值（这是个很容易踩的坑）。
    p.add_argument("--clips", type=int, default=12)
    p.add_argument("--frames", type=int, default=16)
    p.add_argument("--width", type=int, default=960)
    p.add_argument("--height", type=int, default=640)
    p.add_argument("--seed", type=int, default=42)
    p.set_defaults(func=cmd_make_demo)

    p = sub.add_parser("selftest", help="用占位检测器验证评测实现是否正确")
    p.set_defaults(func=cmd_selftest)

    p = sub.add_parser("doctor", help="依赖体检：哪些功能可用")
    p.set_defaults(func=cmd_doctor)

    p = sub.add_parser("eval", help="按配置评测一个模型")
    p.add_argument("--config", required=True, help="YAML 或 JSON 配置路径")
    p.add_argument(
        "--data-root",
        default=None,
        help="数据根目录。配置里的 ann_file/images_root/attr_file 若是相对路径，"
             "会拼到它下面。换设备时用它，不用改配置文件。",
    )
    p.add_argument(
        "--set",
        action="append",
        default=None,
        metavar="KEY=VALUE",
        help="覆盖配置项，可重复。例：--set runtime.device=cuda:1 --set eval.predict_conf=0.01",
    )
    p.add_argument("--device", default=None, help="覆盖 runtime.device")
    p.add_argument("--exp-id", default=None, help="覆盖 exp_id")
    p.add_argument("--out", default=None, help="覆盖输出目录")
    p.add_argument(
        "--dry-run",
        action="store_true",
        help="只解析配置与数据路径并打印，不执行推理。换设备后先跑这个。",
    )
    p.add_argument(
        "--no-check-paths",
        action="store_true",
        help="跳过启动时的路径存在性检查（默认会检查，缺失即报错）。",
    )
    p.set_defaults(func=cmd_eval)

    p = sub.add_parser("report", help="汇总所有实验生成论文表格")
    p.add_argument("--runs", default="runs", help="包含 result.json 的根目录")
    p.add_argument("--out", default="report.md")
    p.add_argument("--latex", action="store_true", help="额外输出 LaTeX 表")
    p.set_defaults(func=cmd_report)

    p = sub.add_parser("figures", help="把 runs/ 里的实验画成论文配图（需要 matplotlib）")
    p.add_argument("--runs", default="runs", help="包含 result.json 的根目录")
    p.add_argument("--out", default="figs", help="图片输出目录")
    p.add_argument("--config", default=None,
                   help="评测配置，用于生成数据集统计图（并把图存到 result.json 里）")
    p.add_argument("--data-root", default=None, help="数据根目录")
    p.add_argument("--images-root", default=None, help="图像根目录（失败案例图用，兜底）")
    p.add_argument("--max-images", type=int, default=12, help="失败案例图最多放几张")
    p.add_argument("--max-failure-sets", type=int, default=3,
                   help="最多为几个实验画失败案例图（每个实验一张大图）")
    p.set_defaults(func=cmd_figures)

    return ap


def main() -> int:
    ap = build_parser()
    args = ap.parse_args()
    try:
        return int(args.func(args) or 0)
    except FileNotFoundError as exc:
        # 路径类错误是换设备后最常见的问题，输出要干净、可操作，
        # 不要甩一屏 traceback 让人去猜。
        print(f"\n❌ 路径错误\n{exc}\n", file=sys.stderr)
        return 2
    except KeyboardInterrupt:
        print("\n已中断。", file=sys.stderr)
        return 130


if __name__ == "__main__":
    raise SystemExit(main())
