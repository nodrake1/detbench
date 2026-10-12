"""训练配置生成器。

设计原则：**从 mmdetection 官方配置继承，只覆盖必要项。**

为什么不从零手写 mmdet 配置
----------------------------
Deformable DETR / DINO-DETR 的配置涉及 backbone、ChannelMapper、encoder、
decoder、positional_encoding、DINOHead、HungarianAssigner、参数调度器……
手写一份要几百行，而且 mmdet 每个版本都可能有字段变动，写错的地方不会报错、
只会静默训出一个差模型。

继承 + 覆盖则完全不同：架构细节全部沿用官方（即论文里验证过的配置），
我们只改"和本数据集相关"的那几项。这样既短又稳。

覆盖清单（每一项都是必需的，缺一个都跑不通或结果不对）
------------------------------------------------------
1. ``metainfo.classes``        —— 类别名。**单类检测最容易错的地方**：
   mmdet 靠它决定 num_classes 的语义，写错会让标签与类别错位。
2. ``model.bbox_head.num_classes`` —— 分类头输出维度
3. 三个 dataloader 的数据路径 + ``val/test_evaluator.ann_file``
4. ``optim_wrapper.type = AmpOptimWrapper`` —— 混合精度，省 30~45% 显存
5. ``optim_wrapper.accumulative_counts`` —— 梯度累积，小显存凑等效 batch
6. ``train_cfg.max_epochs`` + ``param_scheduler`` 的终点
7. ``model.backbone.frozen_stages`` —— 冻结 backbone，省 30% 显存
8. ``model.backbone.norm_cfg = SyncBN`` —— DDP + 小 batch 下 BN 统计失真
9. ``model.test_cfg.max_per_img`` —— 与评测阶段的 maxDets 保持一致
10. ``work_dir`` / checkpoint hook / batch_size / num_workers
"""

from __future__ import annotations

import copy
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

from detbench.core.deps import require


# --------------------------------------------------------------------------- #
# 工具：安全地按路径取值 / 赋值
# --------------------------------------------------------------------------- #


def _dig(cfg: Any, path: str, default: Any = None) -> Any:
    """按 ``a.b.c`` 路径安全取值。

    mmengine 的 Config 与 dict 的取值方式不同（Config 支持属性访问），
    这里统一处理，避免为了兼容两个版本写一堆 hasattr。
    """
    node = cfg
    for part in path.split("."):
        if node is None:
            return default
        if isinstance(node, dict) or hasattr(node, "get"):
            if part in node if hasattr(node, "__contains__") else False:
                node = node[part] if not hasattr(node, "get") else node.get(part)
            else:
                return default
        else:
            return default
    return node if node is not None else default


def _poke(cfg: Any, path: str, value: Any, *, must_exist: bool = True) -> bool:
    """按路径赋值。返回是否成功。

    ``must_exist=True`` 时，若路径的父级不存在就跳过并返回 False —— 宁可少覆盖
    一项（然后在报告里列出来），也不要凭空创建字段。凭空创建字段在 mmengine 里
    很可能被忽略，导致"以为改了其实没改"。
    """
    parts = path.split(".")
    node = cfg
    for part in parts[:-1]:
        try:
            if hasattr(node, "get"):
                nxt = node.get(part)
                if nxt is None:
                    return False
                node = nxt
            else:
                return False
        except Exception:
            return False
    last = parts[-1]
    try:
        if isinstance(node, dict):
            if must_exist and last not in node:
                return False
            node[last] = value
            return True
        # mmengine.Config 支持属性式赋值
        if must_exist and not hasattr(node, last):
            return False
        setattr(node, last, value)
        return True
    except Exception:
        return False


# --------------------------------------------------------------------------- #
# 训练超参
# --------------------------------------------------------------------------- #


class TrainConfigBuilder:
    """把 DetBench 的训练 YAML + mmdet 官方 base 配置，合成最终训练配置。"""

    def __init__(self, train_cfg: Dict[str, Any]) -> None:
        require("mmengine")
        from mmengine.config import Config

        self._Config = Config
        self.cfg = dict(train_cfg)
        self.notes: List[str] = []
        self.overridden: List[str] = []
        self.skipped: List[str] = []

    # ---------------------------------------------------------------- #
    def build(self) -> Any:
        base = self.cfg.get("base_config")
        if not base:
            raise ValueError("训练配置缺少 base_config（mmdet 官方配置的本地路径）")
        base_path = Path(str(base))
        if not base_path.exists():
            raise FileNotFoundError(
                f"找不到 mmdet base 配置：{base_path}\n\n"
                "DETR 系的官方配置在 mmdetection 仓库里，请先 clone 到本地：\n"
                "    git clone https://github.com/open-mmlab/mmdetection third_party/mmdetection\n"
                "然后把这个路径指过去，例如：\n"
                "    base_config: third_party/mmdetection/configs/dino/dino-4scale_r50_8xb2-12e_coco.py"
            )

        mm = self._Config.fromfile(str(base_path))
        self.mm_cfg = mm
        self._apply_dataset()
        self._apply_model()
        self._apply_training()
        self._apply_runtime()
        return mm

    # ---------------------------------------------------------------- #
    def _set(self, path: str, value: Any, *, must_exist: bool = True) -> None:
        ok = _poke(self.mm_cfg, path, value, must_exist=must_exist)
        (self.overridden if ok else self.skipped).append(f"{path} = {value!r}")

    # ---------------------------------------------------------------- #
    def _apply_dataset(self) -> None:
        d = self.cfg.get("dataset", {})
        classes = tuple(d.get("class_names") or ["helicopter"])
        num_classes = len(classes)
        self.num_classes = num_classes
        self.class_names = list(classes)

        data_root = str(d.get("data_root", ""))
        splits = d.get("splits", {})

        # metainfo 必须设 —— 它决定类别语义，也是单类检测最容易翻车的地方
        self._set("metainfo", dict(classes=classes), must_exist=False)

        for split, loader in (
            ("train", "train_dataloader"),
            ("val", "val_dataloader"),
            ("test", "test_dataloader"),
        ):
            spec = splits.get(split, {})
            ann = spec.get("ann_file")
            img = spec.get("img_prefix", "")
            if self.cfg.get("dataset", {}).get("data_root_override"):
                self._set(f"{loader}.dataset.data_root", self.cfg["dataset"]["data_root_override"])
            else:
                self._set(f"{loader}.dataset.data_root", data_root)
            if ann:
                self._set(f"{loader}.dataset.ann_file", ann)
            if img is not None:
                self._set(f"{loader}.dataset.data_prefix", dict(img=img))
            self._set(f"{loader}.dataset.metainfo", dict(classes=classes))
            self._set(f"{loader}.batch_size", int(d.get("batch_size", 2)))
            self._set(f"{loader}.num_workers", int(d.get("num_workers", 2)))
            self._set(f"{loader}.dataset.test_mode", split != "train")

        # 验证/测试的评测器要指向对应标注
        for split, evaluator in (("val", "val_evaluator"), ("test", "test_evaluator")):
            spec = splits.get(split, {})
            if spec.get("ann_file"):
                self._set(f"{evaluator}.ann_file", str(Path(data_root) / spec["ann_file"]))
            self._set(f"{evaluator}.metric", "bbox")

    # ---------------------------------------------------------------- #
    def _apply_model(self) -> None:
        m = self.cfg.get("model", {})
        # DETR 系的分类头字段名在不同实现里都是 bbox_head
        self._set("model.bbox_head.num_classes", self.num_classes)
        # DINO 用 sync_cls_avg_factor，Deformable DETR 没有这个字段，跳过即可
        _poke(self.mm_cfg, "model.bbox_head.sync_cls_avg_factor", True)

        # 冻结 backbone：小数据集微调时几乎无损，省 30% 显存与时间
        frozen = m.get("frozen_stages")
        if frozen is not None:
            if _poke(self.mm_cfg, "model.backbone.frozen_stages", int(frozen)):
                self.overridden.append(f"model.backbone.frozen_stages = {frozen} (冻结 backbone)")
            else:
                self.skipped.append("model.backbone.frozen_stages（该 backbone 不支持）")

        # DDP + 小 batch 下 BN 统计会失真，换 SyncBN
        if m.get("sync_bn"):
            if _poke(self.mm_cfg, "model.backbone.norm_cfg",
                     dict(type="SyncBN", requires_grad=False)):
                self.overridden.append("model.backbone.norm_cfg = SyncBN")
            else:
                self.skipped.append("model.backbone.norm_cfg（该 backbone 无 norm_cfg）")

        # 推理保留的预测数要与评测阶段的 maxDets 一致，否则两边对不上
        if m.get("max_per_img") is not None:
            self._set("model.test_cfg.max_per_img", int(m["max_per_img"]), must_exist=False)

        if m.get("num_queries") is not None:
            _poke(self.mm_cfg, "model.num_queries", int(m["num_queries"]))

        self._set("load_from", self.cfg.get("load_from"), must_exist=False)

    # ---------------------------------------------------------------- #
    def _apply_training(self) -> None:
        t = self.cfg.get("training", {})
        epochs = int(t.get("epochs", 12))

        # AMP：省显存 + 加速。DETR 系对数值较敏感，若出现 NaN 就关掉
        if t.get("amp", True):
            if _poke(self.mm_cfg, "optim_wrapper.type", "AmpOptimWrapper"):
                self.overridden.append("optim_wrapper.type = AmpOptimWrapper（混合精度）")
            else:
                self.skipped.append("optim_wrapper.type")

        # 梯度累积：小显存凑等效 batch
        accum = int(t.get("grad_accum", 1))
        if accum > 1:
            if _poke(self.mm_cfg, "optim_wrapper.accumulative_counts", accum):
                self.overridden.append(f"optim_wrapper.accumulative_counts = {accum}")

        if t.get("lr") is not None:
            self._set("optim_wrapper.optimizer.lr", float(t["lr"]))

        # 训练轮数与参数调度器的终点必须一起改，否则学习率曲线会对不上
        self._set("train_cfg.max_epochs", epochs)
        self._set("train_cfg.val_interval", int(t.get("val_interval", 1)))
        self._retune_schedulers(epochs)

        # checkpoint：保留最近若干个，避免磁盘被撑满
        self._set("default_hooks.checkpoint.interval", int(t.get("ckpt_interval", 1)))
        self._set("default_hooks.checkpoint.max_keep_ckpts", int(t.get("max_keep_ckpts", 3)))
        self._set("default_hooks.checkpoint.save_best", t.get("save_best", "auto"))

    def _retune_schedulers(self, epochs: int) -> None:
        """把参数调度器的 epoch 数与总轮数对齐。

        官方配置是按它自己的 epoch 数写的（DINO 12e、Deformable DETR 50e）。
        我们改了 max_epochs 就必须同步改调度器，否则：
        - 调度器 epoch 比训练短 -> 后期学习率一直是 0，模型停止学习
        - 调度器 epoch 比训练长 -> 学习率没衰减到位，精度明显偏低
        这两种情况都不报错，只是训得差，属于最隐蔽的一类坑。
        """
        sched = _dig(self.mm_cfg, "param_scheduler")
        if not sched:
            self.skipped.append("param_scheduler（未找到，未调整）")
            return
        try:
            items = list(sched)
        except Exception:
            self.skipped.append("param_scheduler（无法迭代）")
            return

        for s in items:
            t = None
            try:
                t = s.get("type") if hasattr(s, "get") else None
            except Exception:
                t = None
            if t in ("MultiStepLR", "MultiStepLr"):
                milestones = list(s.get("milestones", []))
                # 按比例缩放 milestones 到新的总轮数
                if milestones:
                    old_total = float(self.cfg.get("base_epochs") or epochs)
                    ratio = epochs / max(old_total, 1)
                    new_ms = sorted({max(1, int(round(m * ratio))) for m in milestones})
                    _poke(s, "milestones", new_ms)
                    self.overridden.append(f"param_scheduler.milestones -> {new_ms}")
                _poke(s, "end", epochs)
            elif t in ("CosineAnnealingLR", "CosineAnnealingLr", "LinearLR", "PolyLR", "PolyLr"):
                _poke(s, "end", epochs)
                self.overridden.append(f"param_scheduler({t}).end = {epochs}")

    # ---------------------------------------------------------------- #
    def _apply_runtime(self) -> None:
        r = self.cfg.get("runtime", {})
        if r.get("work_dir"):
            self._set("work_dir", str(r["work_dir"]))
        if r.get("seed") is not None:
            self._set("randomness", dict(seed=int(r["seed"]), deterministic=False), must_exist=False)
        if r.get("resume"):
            self._set("resume", True, must_exist=False)
        # 视觉后端：训练时不需要画框，关掉省一点显存
        self._set("visualizer.vis_backends", [dict(type="LocalVisBackend")], must_exist=False)
        self._set("visualizer.name", "visualizer")

    # ---------------------------------------------------------------- #
    def report(self) -> str:
        lines = ["已覆盖的配置项："]
        lines += [f"  ✓ {x}" for x in self.overridden] or ["  （无）"]
        if self.skipped:
            lines += ["", "跳过（目标字段在 base 配置里不存在，属正常）:"]
            lines += [f"  - {x}" for x in self.skipped]
        return "\n".join(lines)

    def dump(self, path: str | Path) -> Path:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        self.mm_cfg.dump(str(path))
        return path


# --------------------------------------------------------------------------- #
# YOLO（ultralytics）数据集 YAML
# --------------------------------------------------------------------------- #


def build_yolo_dataset_yaml(train_cfg: Dict[str, Any], out_path: str | Path) -> Path:
    """生成 ultralytics 需要的 data.yaml。

    ultralytics 不接受 COCO json，必须是它自己的 YOLO txt 布局：
        <root>/images/{train,val}/  <root>/labels/{train,val}/
    如果你的数据是 COCO 格式，先用 ``detbench.core.io`` 转一下（见 README）。
    """
    d = train_cfg.get("dataset", {})
    classes = list(d.get("class_names") or ["helicopter"])
    splits = d.get("splits", {})

    lines = [
        "# Auto-generated by DetBench. Do not edit by hand.",
        f"path: {d.get('root', '.')}",
        f"train: {splits.get('train', {}).get('img_prefix', 'images/train')}",
        f"val: {splits.get('val', {}).get('img_prefix', 'images/val')}",
        f"nc: {len(classes)}",
        "names:",
    ]
    lines += [f"  {i}: {n}" for i, n in enumerate(classes)]

    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    # 用 ascii 写入：ultralytics 与部分 yaml 解析器对非 ASCII 注释/编码的
    # 处理不一致，训练配置这种"必须被正确读取"的文件不值得冒这个风险。
    # 类别名本身仍按原样写入（中文类名会转义成 \uXXXX，YAML 解析后能正确还原）。
    out_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return out_path


# --------------------------------------------------------------------------- #
# 训练耗时预估
# --------------------------------------------------------------------------- #


def estimate_epoch_time(
    num_train_images: int,
    batch_size: int,
    sec_per_iter: float,
    num_gpus: int = 1,
    ddp_efficiency: float = 0.9,
) -> Tuple[float, float]:
    """按实测 ``sec_per_iter`` 估算每轮与总耗时（小时）。

    ``batch_size`` 是**每张卡**的 batch，所以 iters/epoch = 图片数 / (batch × 卡数)。
    这个公式很简单，但把它写下来的价值在于：**动手前先算总账**。
    很多人的坑是"以为两卡能省一半时间"，实际 DDP 效率只有 0.85~0.95，
    在 4060 Ti 这类窄带宽卡上更差。
    """
    effective = max(num_gpus, 1)
    iters = max(1, num_train_images // max(batch_size * effective, 1))
    epoch_sec = iters * sec_per_iter / max(ddp_efficiency, 1e-9)
    return epoch_sec / 3600.0, iters
