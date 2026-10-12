"""生成一个合成"占位数据集"，用于在没有真实直升机数据时打通整条评测链路。

产物结构（与未来正式数据集完全一致）::

    <out>/
      images/{train,val,test}/*.jpg
      annotations/instances_{train,val,test}.json   # COCO 格式
      metadata/clips.csv                            # 采集元数据（属性表）
      metadata/frames.csv                           # 帧级属性（距离/模糊/光照）

为什么要专门造这个
------------------
等真实数据到手时，你只需要替换数据目录，其余代码一行不改。
这就是把"数据等待期"变成"管线开发期"的办法。

合成图的视觉内容是随机噪声 + 亮斑，**没有任何科研意义**，只是为了让
数据加载、评测、分层统计、结果汇总这条链路能被端到端验证。
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np

# 让脚本能独立运行：把包所在的父目录加入 sys.path
_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT.parent) not in sys.path:
    sys.path.insert(0, str(_ROOT.parent))

from detbench.core.deps import force_utf8_output  # noqa: E402


def _make_image(rng: np.random.RandomState, w: int, h: int, boxes: List[List[float]]) -> np.ndarray:
    """画一张"天空 + 若干小目标"的合成图。"""
    # 天空背景：从亮到暗的竖直渐变 + 噪声
    grad = np.linspace(200, 120, h, dtype=np.float32)[:, None]
    img = np.repeat(grad, w, axis=1)
    img = img + rng.normal(0, 6.0, (h, w)).astype(np.float32)
    img = np.stack([img * 0.95, img * 0.98, img], axis=-1)  # 略偏蓝

    # 目标：深色矩形斑点（模拟远处的直升机剪影）
    for x1, y1, x2, y2 in boxes:
        xi1, yi1, xi2, yi2 = (int(round(v)) for v in (x1, y1, x2, y2))
        xi1, yi1 = max(0, xi1), max(0, yi1)
        xi2, yi2 = min(w, xi2), min(h, yi2)
        if xi2 <= xi1 or yi2 <= yi1:
            continue
        patch = rng.normal(60, 12, (yi2 - yi1, xi2 - xi1, 3)).astype(np.float32)
        img[yi1:yi2, xi1:xi2] = patch

    return np.clip(img, 0, 255).astype(np.uint8)


def generate(
    out_dir: str | Path,
    num_clips: int = 12,
    frames_per_clip: int = 16,
    img_w: int = 960,
    img_h: int = 640,
    seed: int = 42,
) -> Path:
    """生成占位数据集。

    默认 12 段 clip x 16 帧 = 192 张图。规模刻意选得"刚好够分层评测用"：
    取 60/20/20 划分后测试集有 2~3 段 clip、几十个目标，
    足以让 ``min_gt`` 检查通过、每个分组都出得来数。
    真实数据集上这个数字要放大两个数量级。
    """
    from PIL import Image

    out = Path(out_dir)
    rng = np.random.RandomState(seed)

    # 属性取值池：距离三档、模糊三档、光照三类 —— 与真实数据集同构
    distance_bins = ["near", "mid", "far"]
    blur_levels = ["clear", "light", "heavy"]
    lightings = ["frontlit", "backlit", "overcast"]

    # 按 clip 划分：绝不按帧随机划分（防 scene leakage）。
    # 采用"轮流分配"而不是纯随机，是为了保证每个划分里距离档/模糊档/光照档
    # 都有覆盖 —— 否则随机性可能让测试集只落到一个档位，分层评测直接失效。
    # 真实数据集也应这样做分层划分，并把这个策略写进论文。
    clip_attr_all: Dict[str, Dict[str, str]] = {}
    for i, cid in enumerate(sorted([f"clip{i:03d}" for i in range(num_clips)])):
        clip_attr_all[cid] = {
            "clip_id": cid,
            "distance_bin": distance_bins[i % 3],
            "blur": blur_levels[(i // 2) % 3],
            "lighting": lightings[(i // 3) % 3],
        }

    # 按 distance_bin 分桶后轮流分给 train/val/test，保证三档都出现在测试集
    buckets: Dict[str, List[str]] = {b: [] for b in distance_bins}
    for cid in sorted(clip_attr_all):
        buckets[clip_attr_all[cid]["distance_bin"]].append(cid)

    splits: Dict[str, List[str]] = {"train": [], "val": [], "test": []}
    target_ratio = {"train": 0.6, "val": 0.2, "test": 0.2}
    for b in distance_bins:
        ids = buckets[b]
        n = len(ids)
        if n == 0:
            continue
        n_train = max(1, int(round(n * target_ratio["train"])))
        n_val = max(1, int(round(n * target_ratio["val"]))) if n >= 3 else 1
        if n_train + n_val >= n:
            n_train = max(1, n - 2) if n >= 3 else 1
            n_val = max(1, n - n_train - 1) if n >= 3 else 0
        splits["train"] += ids[:n_train]
        splits["val"] += ids[n_train : n_train + n_val]
        splits["test"] += ids[n_train + n_val :]
    for k in splits:
        splits[k] = sorted(set(splits[k]))

    # 每段 clip 固定它的属性（真实采集里一段飞行通常固定一个配置）
    clip_attr = clip_attr_all

    clip_rows = ["clip_id,distance_bin,blur,lighting,shutter,focal_mm,fps"]
    frame_rows = ["image_name,clip_id,distance_bin,blur,lighting,obj_blur"]

    for split, ids in splits.items():
        img_dir = out / "images" / split
        img_dir.mkdir(parents=True, exist_ok=True)
        images, annotations = [], []
        ann_id = 0

        for cid in ids:
            attr = clip_attr[cid]
            shutter = {"clear": "1/2000", "light": "1/1000", "heavy": "1/250"}[attr["blur"]]
            clip_rows.append(
                f"{cid},{attr['distance_bin']},{attr['blur']},{attr['lighting']},{shutter},50,30"
            )

            for f in range(frames_per_clip):
                image_id = len(images) + 1
                fname = f"{cid}_f{f:04d}.jpg"

                # 目标尺寸由距离档决定，档内再取一个连续斜坡，
                # 让尺寸分布连续而不是三个离散值 —— 这样 nano/tiny/small/large
                # 各个尺寸桶都能出现，尺寸-精度分解才有意义。
                if attr["distance_bin"] == "near":
                    side = 40.0 * (1.6 ** rng.uniform(-1.0, 1.0))   # ~18..100 px
                elif attr["distance_bin"] == "mid":
                    side = 16.0 * (1.8 ** rng.uniform(-1.0, 1.0))   # ~5..52 px
                else:
                    side = 7.0 * (1.9 ** rng.uniform(-1.0, 1.0))    # ~2..25 px

                boxes = []
                n_obj = int(rng.choice([1, 1, 1, 2], p=[0.5, 0.25, 0.15, 0.1]))
                for _ in range(n_obj):
                    cx = rng.uniform(side, img_w - side)
                    cy = rng.uniform(side, img_h - side)
                    bw = side * rng.uniform(0.8, 1.6)
                    bh = side * rng.uniform(0.4, 1.0)
                    boxes.append(
                        [
                            float(np.clip(cx - bw / 2, 0, img_w)),
                            float(np.clip(cy - bh / 2, 0, img_h)),
                            float(np.clip(cx + bw / 2, 0, img_w)),
                            float(np.clip(cy + bh / 2, 0, img_h)),
                        ]
                    )

                arr = _make_image(rng, img_w, img_h, boxes)
                Image.fromarray(arr).save(img_dir / fname, quality=92)

                images.append(
                    {"id": image_id, "file_name": f"{split}/{fname}", "width": img_w, "height": img_h}
                )
                for b in boxes:
                    ann_id += 1
                    annotations.append(
                        {
                            "id": ann_id,
                            "image_id": image_id,
                            "category_id": 1,
                            "bbox": [b[0], b[1], b[2] - b[0], b[3] - b[1]],
                            "area": (b[2] - b[0]) * (b[3] - b[1]),
                            "iscrowd": 0,
                        }
                    )

                frame_rows.append(
                    f"{split}/{fname},{cid},{attr['distance_bin']},{attr['blur']},{attr['lighting']},"
                    + ";".join([attr["blur"]] * len(boxes))
                )

        coco = {
            "info": {"description": "DetBench synthetic placeholder dataset"},
            "licenses": [],
            "images": images,
            "annotations": annotations,
            "categories": [{"id": 1, "name": "helicopter", "supercategory": "aircraft"}],
        }
        ann_dir = out / "annotations"
        ann_dir.mkdir(parents=True, exist_ok=True)
        (ann_dir / f"instances_{split}.json").write_text(
            json.dumps(coco, ensure_ascii=False, indent=1), encoding="utf-8"
        )

    meta_dir = out / "metadata"
    meta_dir.mkdir(parents=True, exist_ok=True)
    (meta_dir / "clips.csv").write_text("\n".join(clip_rows) + "\n", encoding="utf-8")
    (meta_dir / "frames.csv").write_text("\n".join(frame_rows) + "\n", encoding="utf-8")

    print(f"占位数据集已生成：{out.resolve()}")
    for split, ids in splits.items():
        print(f"  {split:5s}: {len(ids)} clip")
    return out


def main() -> None:
    force_utf8_output()
    ap = argparse.ArgumentParser(description="生成用于打通管线的占位数据集")
    ap.add_argument("--out", default="data/demo", help="输出目录")
    ap.add_argument("--clips", type=int, default=8)
    ap.add_argument("--frames", type=int, default=12)
    ap.add_argument("--width", type=int, default=960)
    ap.add_argument("--height", type=int, default=640)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()
    generate(args.out, args.clips, args.frames, args.width, args.height, args.seed)


if __name__ == "__main__":
    main()
