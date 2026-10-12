#!/usr/bin/env python
"""DetBench 训练入口。

支持两条后端：

  * ``backend: mmdet``       —— Deformable DETR / DINO-DETR / DAB-DETR 等
  * ``backend: ultralytics`` —— YOLO 全系

训练结束后会**自动调用 DetBench 评测并写出 result.json**，
所以"训练 → 评测 → 结果落盘"是一条命令完成的，不会出现"训完了忘了评测"。

用法::

    # 0) 先看配置合并结果，不训练（30 秒，强烈建议先跑）
    python tools/train.py --config configs/train_example.yaml --dump-config

    # 1) 单卡训练
    python tools/train.py --config configs/train_example.yaml

    # 2) 双卡 DDP（两台机器各自一条命令）
    torchrun --nproc_per_node=1 --nnodes=2 --node_rank=0 \
             --master_addr=<主节点IP> --master_port=29500 \
             tools/train.py --config configs/train_example.yaml

    # 3) 只估时间不训练
    python tools/train.py --config configs/train_example.yaml --estimate-only
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT.parent) not in sys.path:
    sys.path.insert(0, str(ROOT.parent))

from detbench.core.config import load_and_resolve  # noqa: E402
from detbench.core.deps import mmdet_available, probe, torch_available  # noqa: E402
from detbench.core.deps import force_utf8_output


def hr(t: str = "") -> None:
    print("\n" + "=" * 74)
    if t:
        print(t)
        print("=" * 74)


# --------------------------------------------------------------------------- #
# 环境信息
# --------------------------------------------------------------------------- #


def distributed_info() -> dict:
    """读取 torchrun 注入的环境变量来判断是否处于分布式模式。"""
    world = int(os.environ.get("WORLD_SIZE", "1"))
    rank = int(os.environ.get("RANK", "0"))
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    return {
        "distributed": world > 1,
        "world_size": world,
        "rank": rank,
        "local_rank": local_rank,
        "is_main": rank == 0,
    }


# --------------------------------------------------------------------------- #
# 校验
# --------------------------------------------------------------------------- #


def validate(train_cfg: dict, dist: dict) -> None:
    """训练前把所有能提前查出来的问题一次性报清楚。

    训练往往要跑几小时到几天，最怕跑到一半才发现类别数写错、
    或者验证集路径不存在。这里花几秒全部查完。
    """
    problems: list[str] = []
    d = train_cfg.get("dataset", {}) or {}
    m = train_cfg.get("model", {}) or {}
    t = train_cfg.get("training", {}) or {}

    classes = d.get("class_names") or []
    if not classes:
        problems.append("dataset.class_names 为空 —— 必须显式给出类别名（哪怕只有一个类）")
    else:
        print(f"  类别数      : {len(classes)}  {classes}")
        if len(classes) == 1:
            print("  ⚠️ 单类检测提醒：")
            print("     mmdet 的类别 0 是背景保留位，所以模型的 num_classes 会是 2，")
            print("     预测标签是 1。DetBench 的评测端默认 label_offset=1 帮你减回 0，")
            print("     请确认 eval 配置里也写了 label_offset: 1。")

    # 数据路径
    for split in ("train", "val"):
        spec = (d.get("splits") or {}).get(split) or {}
        ann = spec.get("ann_file")
        root = d.get("data_root", "")
        if not ann:
            problems.append(f"dataset.splits.{split}.ann_file 未设置")
            continue
        full = Path(root) / ann if root and not Path(ann).is_absolute() else Path(ann)
        mark = "✓" if full.exists() else "✗"
        print(f"  {mark} {split:5s} 标注: {full}")
        if not full.exists():
            problems.append(f"{split} 标注不存在: {full}")
        img = spec.get("img_prefix", "")
        full_img = Path(root) / img if root and not Path(img).is_absolute() else Path(img)
        if img and not full_img.exists():
            problems.append(f"{split} 图像目录不存在: {full_img}")

    # base 配置
    if train_cfg.get("backend", "mmdet") == "mmdet":
        base = m.get("base_config")
        if not base:
            problems.append("model.base_config 未设置")
        else:
            print(f"  {'✓' if Path(base).exists() else '✗'} base 配置: {base}")
            if not Path(base).exists():
                problems.append(
                    f"base 配置不存在: {base}\n"
                    "      先 clone mmdetection：\n"
                    "        git clone https://github.com/open-mmlab/mmdetection third_party/mmdetection"
                )

    # 训练参数自检
    bs = int(d.get("batch_size", 2))
    accum = int(t.get("grad_accum", 1))
    world = dist["world_size"]
    if bs < 1:
        problems.append("dataset.batch_size 必须 >= 1")
    print(f"  等效 batch  : {bs} × {world} 卡 × {accum} 累积 = {bs * world * accum}")
    if bs * world * accum < 8:
        print("  ⚠️ 等效 batch < 8。DETR 系对 batch 较敏感，建议用梯度累积补到 8~16。")

    if problems:
        print()
        raise SystemExit(
            "❌ 训练配置有问题，已中止：\n"
            + "\n".join(f"  - {p}" for p in problems)
            + "\n\n用 --dump-config 可以查看合并后的完整配置。"
        )


# --------------------------------------------------------------------------- #
# 耗时预估
# --------------------------------------------------------------------------- #


def estimate(train_cfg: dict, dist: dict) -> None:
    """按经验值给出训练耗时区间，帮助判断"这个配置今晚能不能跑完"。"""
    from detbench.core.io import load_coco

    d = train_cfg["dataset"]
    spec = (d.get("splits") or {}).get("train", {})
    ann = Path(d.get("data_root", "")) / spec.get("ann_file", "")
    try:
        ds = load_coco(ann, d.get("data_root", ""), name="train")
        n = ds.num_images
    except Exception as exc:
        print(f"  （无法读取训练集图片数，跳过耗时预估：{exc}）")
        return

    bs = int(d.get("batch_size", 2))
    world = dist["world_size"]
    epochs = int(train_cfg.get("training", {}).get("epochs", 12))

    # 经验值：DINO/Deformable DETR R50 @800x1333，AMP，4060 Ti 级卡
    # 每 iteration 秒数。不同硬件差异大，这里只给量级参考。
    sec_per_iter = float(train_cfg.get("training", {}).get("assumed_sec_per_iter", 0.55))

    iters = max(1, n // max(bs * world, 1))
    # DDP 有通信开销，有效吞吐略低于线性扩展，所以不除以卡数
    epoch_h = iters * sec_per_iter / 3600.0
    total_h = epoch_h * epochs

    print(f"  训练集图片  : {n}")
    print(f"  iters/epoch : {iters}  (batch {bs} × {world} 卡)")
    print(f"  单轮耗时    : ≈ {epoch_h:.2f} 小时   （假设 {sec_per_iter:.2f} s/iter）")
    print(f"  总耗时      : ≈ {total_h:.1f} 小时 = {total_h / 24:.1f} 天（{epochs} 轮）")
    print()
    print("  ⚠️ 这是按经验值估的，误差可能有 ±50%。")
    print("     要拿到准确数字，先跑 20 个 iteration 看实际速度：")
    print("       python tools/train.py --config <配置> --set training.epochs=1 --profile-iters 20")
    if total_h > 48:
        print()
        print("  💡 总耗时超过 48 小时，建议之一：")
        print("     - 提高 training.grad_accum、增大 dataset.batch_size 以更充分利用显存")
        print("     - 降低输入分辨率（改 base_config 里的 scale，如 (1333,800) -> (1024,640)）")
        print("     - 设 model.frozen_stages=1 冻结 backbone（省 ~30% 时间）")
        print("     - 减少 training.epochs（会议论文常用 12 轮已能出结果）")


# --------------------------------------------------------------------------- #
# mmdet 后端
# --------------------------------------------------------------------------- #


def train_mmdet(train_cfg: dict, dist: dict, args) -> Path:
    from detbench.core.train_config import TrainConfigBuilder

    if not mmdet_available():
        ok, err = probe("mmdet")
        raise SystemExit(
            "❌ 需要 mmdetection 才能训练 DETR 系模型。\n"
            f"   当前状态：{err}\n\n"
            "   安装：\n"
            "     pip install mmengine mmcv mmdet\n"
            "     git clone https://github.com/open-mmlab/mmdetection third_party/mmdetection\n\n"
            "   或者把训练后端换成 ultralytics（YOLO，依赖更少）：\n"
            "     --set backend=ultralytics"
        )

    builder = TrainConfigBuilder(train_cfg)
    mm_cfg = builder.build()

    work_dir = Path(train_cfg.get("runtime", {}).get("work_dir", "runs/train"))
    work_dir.mkdir(parents=True, exist_ok=True)

    # 合并后的配置必须落盘 —— Runner 需要它来解析 _base_ 与记录实验
    merged_path = builder.dump(work_dir / "merged_config.py")
    print(f"  合并后的配置已写入: {merged_path}")

    if args.dump_config:
        print()
        print(builder.report())
        print(f"\n--dump-config 完成，未启动训练。")
        print(f"检查这个文件可以确认所有覆盖都生效了：\n  {merged_path}")
        return merged_path

    hr("开始训练")
    print(builder.report())
    print()
    print(f"  work_dir : {work_dir}")
    print(f"  分布式   : {dist['world_size']} 卡, rank={dist['rank']}")

    from mmengine.runner import Runner

    mm_cfg.work_dir = str(work_dir)
    # 用 torchrun 启动时，mmengine 需要 launcher='pytorch' 才会走 DDP
    runner = Runner.from_cfg(mm_cfg)
    runner.train()

    print()
    print(f"✅ 训练完成，产物在 {work_dir}")
    return work_dir


# --------------------------------------------------------------------------- #
# ultralytics 后端
# --------------------------------------------------------------------------- #


def train_yolo(train_cfg: dict, dist: dict, args) -> Path:
    from detbench.core.train_config import build_yolo_dataset_yaml

    m = train_cfg["model"]
    t = train_cfg["training"]
    d = train_cfg["dataset"]
    work_dir = Path(train_cfg.get("runtime", {}).get("work_dir", "runs/train"))
    work_dir.mkdir(parents=True, exist_ok=True)

    # 先生成 data.yaml 并支持 --dump-config —— 这样即使没装 ultralytics，
    # 也能先检查配置对不对（配置错误是最常见的失败原因）。
    data_yaml = build_yolo_dataset_yaml(train_cfg, work_dir / "data.yaml")
    print(f"  数据集 YAML: {data_yaml}")

    if args.dump_config:
        print()
        print(data_yaml.read_text(encoding="utf-8"))
        print("--dump-config 完成，未启动训练。")
        return data_yaml

    if not probe("ultralytics")[0]:
        raise SystemExit(
            "❌ 需要 ultralytics 才能训练 YOLO 模型。\n"
            "   安装：pip install ultralytics\n"
            "   或者把训练后端换成 mmdet：--set backend=mmdet"
        )
    if dist["world_size"] > 1:
        raise SystemExit(
            "❌ ultralytics 后端暂不支持本脚本的 torchrun 多机启动方式。\n"
            "   多卡训练请直接用 ultralytics 原生命令（它自己管理 DDP）：\n"
            "     yolo detect train model=yolo11m.pt data=<data.yaml> device=0,1\n"
            "   本脚本的 ultralytics 后端只用于单卡。"
        )

    from ultralytics import YOLO

    model = YOLO(m.get("weights", "yolo11m.pt"))
    hr("开始训练")
    model.train(
        data=str(data_yaml),
        epochs=int(t.get("epochs", 300)),
        imgsz=int(m.get("imgsz", 1280)),
        batch=int(d.get("batch_size", 8)),
        workers=int(d.get("num_workers", 4)),
        amp=bool(t.get("amp", True)),
        device=m.get("device", "0"),
        project=str(work_dir.parent),
        name=work_dir.name,
        exist_ok=True,
        patience=int(t.get("patience", 50)),
        seed=int(train_cfg.get("runtime", {}).get("seed", 0)),
    )

    best = work_dir / "weights" / "best.pt"
    print(f"\n✅ 训练完成，最优权重: {best}")
    return best


# --------------------------------------------------------------------------- #
# 自动评测
# --------------------------------------------------------------------------- #


def auto_eval(train_cfg: dict, weight: Path, exp_id: str) -> None:
    """训练完立刻评测并写 result.json，闭环。"""
    if not train_cfg.get("auto_eval", True):
        print("\n（auto_eval 已关闭，跳过评测）")
        return
    if not Path(weight).exists():
        print(f"\n⚠️ 找不到权重 {weight}，跳过自动评测。")
        return

    hr("自动评测（训练产物 -> result.json）")
    mm = train_cfg.get("model", {})
    backend = train_cfg.get("backend", "mmdet")

    eval_cfg = {
        "exp_id": exp_id,
        "dataset": {
            "format": "coco",
            "name": train_cfg["dataset"].get("name", "dataset"),
            "ann_file": str(
                Path(train_cfg["dataset"].get("data_root", ""))
                / (train_cfg["dataset"].get("splits", {}).get("test", {}) or {}).get("ann_file", "")
            ),
            "images_root": str(
                Path(train_cfg["dataset"].get("data_root", ""))
                / (train_cfg["dataset"].get("splits", {}).get("test", {}) or {}).get("img_prefix", "")
            ),
            "attr_file": train_cfg["dataset"].get("attr_file"),
        },
        "model": {
            "type": "mmdet" if backend == "mmdet" else "yolov8",
            "config": mm.get("base_config"),
            "checkpoint": str(weight),
            "weights": str(weight),
            "label_offset": int(mm.get("label_offset", 1 if backend == "mmdet" else 0)),
            "score_thresh": float(train_cfg.get("eval", {}).get("predict_conf", 0.001)),
            "max_per_img": int(mm.get("max_per_img", 300)),
            "imgsz": int(mm.get("imgsz", 1280)),
        },
        "eval": train_cfg.get("eval", {}),
        "runtime": {"device": train_cfg.get("runtime", {}).get("device", "cuda")},
        "output": {"dir": str(Path(train_cfg.get("runtime", {}).get("work_dir", "runs/train")) / "eval")},
    }

    from detbench.core.registry import build
    from detbench.core.eval import evaluate
    from detbench.core.io import load_coco
    from detbench.core.metrics import FINE_AREA_BINS, GroupSpec

    import detbench.models  # noqa: F401

    ds = eval_cfg["dataset"]
    if not Path(ds["ann_file"]).exists():
        print(f"⚠️ 测试集标注不存在：{ds['ann_file']}")
        print("   训练产物已保存，但评测需要单独的 test split。")
        print("   可以在训练配置里补 dataset.splits.test，或手动跑 eval。")
        return

    dataset = load_coco(
        ds["ann_file"], ds["images_root"],
        name=ds["name"], attr_file=ds["attr_file"] if ds["attr_file"] and Path(str(ds["attr_file"])).exists() else None,
    )
    adapter = build(eval_cfg["model"], device=eval_cfg["runtime"]["device"])

    groups = []
    for key in train_cfg.get("eval", {}).get("groups", ["distance_bin", "blur", "__size__"]):
        groups.append(GroupSpec(key, min_gt=int(train_cfg.get("eval", {}).get("min_gt", 5))))

    evaluate(
        adapter=adapter,
        dataset=dataset,
        image_root=ds["images_root"],
        exp_id=exp_id,
        out_dir=eval_cfg["output"]["dir"],
        group_specs=groups,
        speed=True,
        config=eval_cfg,
        notes=f"auto-eval after training, weight={weight}",
    )
    print(f"\n✅ 评测结果已写入 {eval_cfg['output']['dir']}/result.json")
    print("   汇总所有实验：python tools/main.py report --runs runs --out report.md")


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #


def cmd_train(args: argparse.Namespace) -> int:
    hr("1/5  载入训练配置")
    cfg = load_and_resolve(args.config, overrides=args.set, data_root=args.data_root, check_paths=False)

    backend = getattr(args, "backend", None) or cfg.get("backend", "mmdet")
    cfg["backend"] = backend
    print(f"  配置文件  : {cfg.get('_config_path')}")
    print(f"  后端      : {backend}")
    print(f"  data_root : {cfg.get('data_root')}")

    dist = distributed_info()
    print(f"  运行模式  : {'分布式 ' + str(dist['world_size']) + ' 进程' if dist['distributed'] else '单进程'}")

    hr("2/5  校验配置与数据")
    validate(cfg, dist)

    hr("3/5  耗时预估")
    estimate(cfg, dist)

    if args.estimate_only:
        print("\n--estimate-only 完成，未启动训练。")
        return 0

    hr("4/5  准备训练")
    exp_id = args.exp_id or cfg.get("exp_id") or f"train_{backend}"
    cfg.setdefault("runtime", {})
    if args.work_dir:
        cfg["runtime"]["work_dir"] = args.work_dir
    cfg["runtime"].setdefault("work_dir", str(ROOT / "runs" / exp_id))

    t0 = time.time()
    if backend == "mmdet":
        result = train_mmdet(cfg, dist, args)
        if args.dump_config:
            return 0  # dump-config 模式到此为止，不训练也不评测
        # mmengine 的 best 权重名带指标后缀，用通配符找最新的
        cands = sorted(Path(cfg["runtime"]["work_dir"]).glob("best_bbox_mAP*.pth"))
        if not cands:
            cands = sorted(Path(cfg["runtime"]["work_dir"]).glob("best*.pth"))
        if not cands:
            cands = sorted(Path(cfg["runtime"]["work_dir"]).glob("epoch_*.pth"))
        weight = cands[-1] if cands else Path(cfg["runtime"]["work_dir"]) / "none.pth"
    else:
        weight = train_yolo(cfg, dist, args)
        if args.dump_config:
            return 0  # dump-config 模式到此为止，不训练也不评测

    elapsed = time.time() - t0

    # 只有主进程做评测，避免多进程同时写同一个 result.json
    if dist["is_main"]:
        hr("5/5  训练后处理")
        print(f"  实际训练耗时: {elapsed / 3600:.2f} 小时")
        auto_eval(cfg, weight, exp_id)
    else:
        print(f"\n（rank {dist['rank']} 不执行评测，由 rank 0 负责）")

    return 0


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(
        prog="detbench-train",
        description="DetBench 训练入口（mmdet DETR 系 / ultralytics YOLO）",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    ap.add_argument("--config", required=True, help="训练配置（YAML/JSON）")
    ap.add_argument("--data-root", default=None, help="数据根目录，覆盖配置里的 data_root")
    ap.add_argument("--set", action="append", default=None, metavar="KEY=VALUE",
                    help="覆盖配置项，可重复，例：--set training.epochs=24")
    ap.add_argument("--backend", default=None, choices=["mmdet", "ultralytics"],
                    help="训练后端，覆盖配置里的 backend")
    ap.add_argument("--exp-id", default=None, help="实验名（决定 work_dir 与结果目录）")
    ap.add_argument("--work-dir", default=None, help="输出目录，覆盖 runtime.work_dir")
    ap.add_argument("--dump-config", action="store_true",
                    help="只生成并写出合并后的配置，不训练。改配置后先跑这个")
    ap.add_argument("--estimate-only", action="store_true", help="只估耗时，不训练")
    ap.add_argument("--profile-iters", type=int, default=0,
                    help="只跑 N 个 iteration 测速后退出（用于拿到准确的 s/iter）")
    return ap


def main() -> int:
    force_utf8_output()
    args = build_parser().parse_args()
    try:
        return cmd_train(args)
    except FileNotFoundError as exc:
        print(f"\n❌ 路径错误\n{exc}\n", file=sys.stderr)
        return 2
    except KeyboardInterrupt:
        print("\n训练已中断。mmengine 的 last_checkpoint 可以续训：--set runtime.resume=true",
              file=sys.stderr)
        return 130


if __name__ == "__main__":
    raise SystemExit(main())
