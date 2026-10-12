#!/usr/bin/env python
"""DetBench 数据集工具：格式转换 / 分层划分 / 质检 / 去冗余。

五个子命令::

    # 1. 格式转换（COCO <-> YOLO <-> VOC）
    python tools/dataset.py convert --from coco --to yolo \
        --ann data/HeliAir/annotations/instances_train.json \
        --images data/HeliAir/images/train --out data/HeliAir_yolo

    # 2. 按飞行架次分层划分（**绝不能按帧随机划分**）
    python tools/dataset.py split \
        --ann data/HeliAir/annotations/instances_all.json \
        --attr data/HeliAir/metadata/frames.csv \
        --group-key clip_id --out data/HeliAir/annotations

    # 3. 质检：框的合法性、类别分布、图像完整性、重复图
    python tools/dataset.py check \
        --ann data/HeliAir/annotations/instances_all.json \
        --images data/HeliAir/images

    # 4. 标注一致性：两个标注员互评（论文里报告"标注一致性 mAP"就靠它）
    python tools/dataset.py agreement \
        --ann-a ann_by_alice.json --ann-b ann_by_bob.json

    # 5. 抽帧去冗余：检测近似重复帧（视频逐帧导出会让 mAP 虚高）
    python tools/dataset.py dedupe \
        --images data/HeliAir/raw_frames --threshold 8 --move-to data/HeliAir/quarantine
"""

from __future__ import annotations

import argparse
import csv
import json
import shutil
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT.parent) not in sys.path:
    sys.path.insert(0, str(ROOT.parent))

import numpy as np  # noqa: E402

IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}


def hr(t: str = "") -> None:
    print("\n" + "=" * 74)
    if t:
        print(t)
        print("=" * 74)


# --------------------------------------------------------------------------- #
# 内部：读写 COCO
# --------------------------------------------------------------------------- #


def _read_coco(path: str | Path) -> dict:
    with Path(path).open("r", encoding="utf-8") as f:
        return json.load(f)


def _write_coco(path: str | Path, images: list, annotations: list, categories: list) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    coco = {
        "info": {"description": "DetBench", "version": "1.0"},
        "licenses": [],
        "images": images,
        "annotations": annotations,
        "categories": categories,
    }
    with path.open("w", encoding="utf-8") as f:
        json.dump(coco, f, ensure_ascii=False, indent=1)
    return path


def _xywh_to_xyxy(b):
    return [b[0], b[1], b[0] + b[2], b[1] + b[3]]


def _xyxy_to_xywh(b):
    return [b[0], b[1], b[2] - b[0], b[3] - b[1]]


def _load_attr(path: Optional[str | Path]) -> Dict[str, Dict[str, str]]:
    """读属性表 CSV，返回 ``{文件名: {列: 值}}``。"""
    if not path:
        return {}
    path = Path(path)
    if not path.exists():
        print(f"  ⚠️ 属性表不存在，忽略：{path}")
        return {}
    out: Dict[str, Dict[str, str]] = {}
    with path.open("r", encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        if not reader.fieldnames:
            return {}
        key = reader.fieldnames[0]
        for row in reader:
            name = Path((row.get(key) or "").strip()).name
            if name:
                out[name] = {k: (v or "").strip() for k, v in row.items() if k != key}
    return out


# --------------------------------------------------------------------------- #
# 1. 格式转换
# --------------------------------------------------------------------------- #


def cmd_convert(args: argparse.Namespace) -> int:
    hr(f"格式转换 {args.src_format} -> {args.dst_format}")
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    if args.src_format == args.dst_format:
        print("源格式与目标格式相同，无需转换。")
        return 0

    # 统一先转成内存中的 COCO 结构
    if args.src_format == "coco":
        coco = _read_coco(args.ann)
    elif args.src_format == "yolo":
        coco = _yolo_to_coco(args.images, args.labels, args.class_names, args.images)
    elif args.src_format == "voc":
        coco = _voc_to_coco(args.ann, args.images, args.class_names)
    else:
        raise SystemExit(f"不支持的源格式：{args.src_format}")

    n_img, n_ann = len(coco["images"]), len(coco["annotations"])
    print(f"  读入 {n_img} 图 / {n_ann} 标注 / {len(coco['categories'])} 类")

    if args.dst_format == "coco":
        p = _write_coco(out / "instances.json", coco["images"], coco["annotations"], coco["categories"])
        print(f"  写出 COCO: {p}")
    elif args.dst_format == "yolo":
        _coco_to_yolo(coco, out)
    elif args.dst_format == "voc":
        _coco_to_voc(coco, out)
    else:
        raise SystemExit(f"不支持的目标格式：{args.dst_format}")

    print(f"\n✅ 转换完成，输出目录 {out.resolve()}")
    return 0


def _yolo_to_coco(images_dir, labels_dir, class_names, data_root) -> dict:
    """YOLO txt -> 内存 COCO 结构。

    路径匹配做两级兜底，因为 COCO 的 ``file_name`` 常带 split 前缀
    （``train/xxx.jpg``），而 YOLO 标签可能是平铺的也可能是同构嵌套的：

    1. 先按 ``labels_dir / <相对图像根目录的路径>.txt`` 找（同构布局）
    2. 找不到再按**文件名**在 ``labels_dir`` 下递归索引里查（平铺布局）

    没有这个兜底，布局不一致时会**静默产出 0 个标注** ——
    转换"成功"但结果全空，是最难发现的一类错误。
    """
    from PIL import Image

    images_dir, labels_dir = Path(images_dir), Path(labels_dir)
    if not images_dir.exists():
        raise FileNotFoundError(f"图像目录不存在: {images_dir}")
    if not labels_dir.exists():
        raise FileNotFoundError(f"标签目录不存在: {labels_dir}")

    # 按文件名建索引，作为路径匹配失败时的兜底
    by_stem: Dict[str, Path] = {}
    for t in labels_dir.rglob("*.txt"):
        by_stem.setdefault(t.stem, t)

    imgs, anns = [], []
    aid = 0
    n_no_label = 0
    n_from_fallback = 0
    for i, p in enumerate(sorted(x for x in images_dir.rglob("*") if x.suffix.lower() in IMG_EXTS)):
        rel = p.relative_to(images_dir).as_posix()
        with Image.open(p) as im:
            w, h = im.width, im.height
        imgs.append({"id": i + 1, "file_name": rel, "width": w, "height": h})

        label = labels_dir / Path(rel).with_suffix(".txt")
        if not label.exists():
            alt = by_stem.get(p.stem)
            if alt is not None:
                label = alt
                n_from_fallback += 1
            else:
                n_no_label += 1
                continue

        for line in label.read_text(encoding="utf-8").splitlines():
            parts = line.split()
            if len(parts) < 5:
                continue
            c = int(float(parts[0]))
            cx, cy, bw, bh = (float(v) for v in parts[1:5])
            x1 = max(0.0, (cx - bw / 2) * w)
            y1 = max(0.0, (cy - bh / 2) * h)
            x2 = min(float(w), (cx + bw / 2) * w)
            y2 = min(float(h), (cy + bh / 2) * h)
            aid += 1
            anns.append({
                "id": aid, "image_id": i + 1, "category_id": c + 1,
                "bbox": [x1, y1, x2 - x1, y2 - y1],
                "area": (x2 - x1) * (y2 - y1), "iscrowd": 0,
            })

    if n_from_fallback:
        print(f"  提示：{n_from_fallback} 张图是通过文件名兜底匹配到标签的（布局与图像目录不同构）")
    if n_no_label and n_no_label > len(imgs) * 0.5:
        print(
            f"  ⚠️ {n_no_label}/{len(imgs)} 张图找不到对应标签文件，结果可能是空的。\n"
            f"     请确认 --labels 与 --images 的目录结构是否对应。\n"
            f"     当前：\n"
            f"       --images {images_dir}\n"
            f"       --labels {labels_dir}\n"
            f"     标签目录下的顶层条目：{[x.name for x in list(labels_dir.iterdir())[:6]]}"
        )
    if not anns:
        print("  ❌ 没有解析出任何标注。请检查上面的路径提示。")

    cats = [{"id": j + 1, "name": n, "supercategory": "object"} for j, n in enumerate(class_names)]
    return {"images": imgs, "annotations": anns, "categories": cats}


def _voc_to_coco(ann_dir, images_root, class_names) -> dict:
    import xml.etree.ElementTree as ET
    from PIL import Image

    ann_dir = Path(ann_dir)
    imgs, anns = [], []
    aid = 0
    name_to_id = {n: i + 1 for i, n in enumerate(class_names)}
    for i, xml in enumerate(sorted(ann_dir.glob("*.xml"))):
        root = ET.parse(xml).getroot()
        fname = root.findtext("filename") or xml.with_suffix(".jpg").name
        size = root.find("size")
        w = int(float(size.findtext("width", "0"))) if size is not None else 0
        h = int(float(size.findtext("height", "0"))) if size is not None else 0
        if w == 0 or h == 0:
            with Image.open(Path(images_root) / fname) as im:
                w, h = im.width, im.height
        imgs.append({"id": i + 1, "file_name": fname, "width": w, "height": h})
        for obj in root.findall("object"):
            cname = obj.findtext("name", "")
            if cname not in name_to_id:
                continue
            bb = obj.find("bndbox")
            if bb is None:
                continue
            x1, y1 = float(bb.findtext("xmin", "0")), float(bb.findtext("ymin", "0"))
            x2, y2 = float(bb.findtext("xmax", "0")), float(bb.findtext("ymax", "0"))
            aid += 1
            anns.append({
                "id": aid, "image_id": i + 1, "category_id": name_to_id[cname],
                "bbox": [x1, y1, x2 - x1, y2 - y1],
                "area": max(0.0, x2 - x1) * max(0.0, y2 - y1), "iscrowd": 0,
            })
    cats = [{"id": i + 1, "name": n, "supercategory": "object"} for i, n in enumerate(class_names)]
    return {"images": imgs, "annotations": anns, "categories": cats}


def _coco_to_yolo(coco: dict, out: Path) -> None:
    (out / "labels").mkdir(parents=True, exist_ok=True)
    by_img: Dict[int, list] = defaultdict(list)
    for a in coco["annotations"]:
        if a.get("iscrowd", 0):
            continue
        by_img[a["image_id"]].append(a)
    id_to_idx = {c["id"]: i for i, c in enumerate(sorted(coco["categories"], key=lambda x: x["id"]))}

    n_lines = 0
    for img in coco["images"]:
        w, h = float(img["width"]), float(img["height"])
        lines = []
        for a in by_img.get(img["id"], []):
            x, y, bw, bh = a["bbox"]
            cx, cy = (x + bw / 2) / w, (y + bh / 2) / h
            lines.append(f"{id_to_idx[a['category_id']]} {cx:.6f} {cy:.6f} {bw / w:.6f} {bh / h:.6f}")
        p = out / "labels" / (Path(img["file_name"]).with_suffix(".txt"))
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text("\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")
        n_lines += len(lines)

    (out / "classes.txt").write_text(
        "\n".join(c["name"] for c in sorted(coco["categories"], key=lambda x: x["id"])) + "\n",
        encoding="utf-8",
    )
    with (out / "image_list.txt").open("w", encoding="utf-8") as f:
        for img in coco["images"]:
            f.write(img["file_name"] + "\n")
    print(f"  写出 YOLO: {out / 'labels'} （{len(coco['images'])} 个 txt / {n_lines} 行）")
    print(f"  类别清单:  {out / 'classes.txt'}")
    print(f"  图像清单:  {out / 'image_list.txt'}  （用 rsync 按它拷图）")


def _coco_to_voc(coco: dict, out: Path) -> None:
    from xml.etree.ElementTree import Element, SubElement, ElementTree

    (out / "Annotations").mkdir(parents=True, exist_ok=True)
    (out / "ImageSets" / "Main").mkdir(parents=True, exist_ok=True)
    by_img: Dict[int, list] = defaultdict(list)
    for a in coco["annotations"]:
        by_img[a["image_id"]].append(a)
    id_to_name = {c["id"]: c["name"] for c in coco["categories"]}

    stems = []
    for img in coco["images"]:
        stem = Path(img["file_name"]).stem
        stems.append(stem)
        root = Element("annotation")
        SubElement(root, "folder").text = "JPEGImages"
        SubElement(root, "filename").text = Path(img["file_name"]).name
        size = SubElement(root, "size")
        SubElement(size, "width").text = str(img["width"])
        SubElement(size, "height").text = str(img["height"])
        SubElement(size, "depth").text = "3"
        for a in by_img.get(img["id"], []):
            obj = SubElement(root, "object")
            SubElement(obj, "name").text = id_to_name.get(a["category_id"], "unknown")
            SubElement(obj, "difficult").text = "0"
            bb = SubElement(obj, "bndbox")
            x, y, bw, bh = a["bbox"]
            SubElement(bb, "xmin").text = str(int(round(x)))
            SubElement(bb, "ymin").text = str(int(round(y)))
            SubElement(bb, "xmax").text = str(int(round(x + bw)))
            SubElement(bb, "ymax").text = str(int(round(y + bh)))
        ElementTree(root).write(out / "Annotations" / f"{stem}.xml", encoding="utf-8")

    (out / "ImageSets" / "Main" / "trainval.txt").write_text("\n".join(stems) + "\n", encoding="utf-8")
    print(f"  写出 VOC: {out / 'Annotations'} （{len(stems)} 个 xml）")


# --------------------------------------------------------------------------- #
# 2. 分层划分
# --------------------------------------------------------------------------- #


def cmd_split(args: argparse.Namespace) -> int:
    hr("按组分层划分数据集")
    coco = _read_coco(args.ann)
    attr = _load_attr(args.attr)

    id_to_img = {im["id"]: im for im in coco["images"]}
    # 每个图像的"组"（通常是飞行架次）
    img_group: Dict[int, str] = {}
    for im in coco["images"]:
        a = attr.get(Path(im["file_name"]).name, {})
        img_group[im["id"]] = str(a.get(args.group_key, "__nogroup__"))

    groups: Dict[str, List[int]] = defaultdict(list)
    for im in coco["images"]:
        groups[img_group[im["id"]]].append(im["id"])

    if "__nogroup__" in groups:
        print(f"  ⚠️ 有 {len(groups['__nogroup__'])} 张图没有组标签（属性表里缺 {args.group_key} 列）。")
        print("     没有组标签就无法保证划分无泄漏，建议先补齐属性表。")

    keys = sorted(groups)
    rng = np.random.RandomState(args.seed)

    # 分层：按组的主导属性轮流分配，保证每个划分里各档都有覆盖
    strat_key = args.stratify_by
    if strat_key:
        buckets: Dict[str, List[str]] = defaultdict(list)
        for k in keys:
            vals = [attr.get(Path(id_to_img[i]["file_name"]).name, {}).get(strat_key, "unknown")
                    for i in groups[k]]
            buckets[Counter(vals).most_common(1)[0][0]].append(k)
    else:
        buckets = {"__all__": keys}

    assign: Dict[str, str] = {}
    ratios = (args.ratio_train, args.ratio_val, args.ratio_test)
    for bname in sorted(buckets):
        gs = buckets[bname][:]
        rng.shuffle(gs)
        n = len(gs)
        if n == 1:
            assign[gs[0]] = "train"
            continue
        if n == 2:
            assign[gs[0]], assign[gs[1]] = "train", "test"
            continue
        n_train = max(1, int(round(n * ratios[0])))
        n_val = max(1, int(round(n * ratios[1])))
        if n_train + n_val >= n:
            n_train = max(1, n - 2)
            n_val = 1
        for g in gs[:n_train]:
            assign[g] = "train"
        for g in gs[n_train:n_train + n_val]:
            assign[g] = "val"
        for g in gs[n_train + n_val:]:
            assign[g] = "test"

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    ann_by_img: Dict[int, list] = defaultdict(list)
    for a in coco["annotations"]:
        ann_by_img[a["image_id"]].append(a)

    print(f"  组数 {len(keys)}，按 '{args.group_key}' 划分" + (f"，用 '{strat_key}' 分层" if strat_key else ""))
    split_files = []
    for split in ("train", "val", "test"):
        ids = [im["id"] for im in coco["images"] if assign.get(img_group[im["id"]]) == split]
        imgs = [im for im in coco["images"] if im["id"] in set(ids)]
        anns = [a for i in ids for a in ann_by_img.get(i, [])]
        n_groups = len({img_group[i] for i in ids})
        if args.keep_groups and ids:
            # 把组标签写进 images 的 extra 字段，便于事后核对
            for im in imgs:
                im["group"] = img_group[im["id"]]
        p = _write_coco(out / f"instances_{split}.json", imgs, anns, coco["categories"])
        split_files.append((split, p, len(imgs), len(anns), n_groups))
        print(f"  {split:5s}: {len(imgs):5d} 图 / {len(anns):5d} 标注 / {n_groups} 组  -> {p.name}")

    # 顺带写一份划分清单，便于核对与复现
    manifest = out / "split_manifest.csv"
    with manifest.open("w", encoding="utf-8-sig", newline="") as f:
        wr = csv.writer(f)
        wr.writerow(["image_name", "group", "split"])
        for im in coco["images"]:
            wr.writerow([im["file_name"], img_group[im["id"]], assign.get(img_group[im["id"]], "?")])
    print(f"\n  划分清单: {manifest}")

    if len(split_files) == 3 and split_files[2][2] == 0:
        print("\n  ⚠️ 测试集为空。组数太少时无法切出三个划分，建议增加采集架次。")
    print("\n✅ 划分完成。注意：**同一组的帧只会出现在一个划分里**，这是防泄漏的关键。")
    return 0


# --------------------------------------------------------------------------- #
# 3. 质检
# --------------------------------------------------------------------------- #


def cmd_check(args: argparse.Namespace) -> int:
    hr("数据集质检")
    coco = _read_coco(args.ann)
    images, anns, cats = coco["images"], coco["annotations"], coco["categories"]
    id_to_img = {im["id"]: im for im in images}
    id_to_cat = {c["id"]: c["name"] for c in cats}

    problems: List[str] = []
    warnings: List[str] = []

    print(f"  图像 {len(images)} / 标注 {len(anns)} / 类别 {len(cats)}  {[c['name'] for c in cats]}")

    # ---- 3.1 图像文件是否存在 ----
    if args.images:
        root = Path(args.images)
        missing = [im["file_name"] for im in images if not (root / im["file_name"]).exists()]
        if missing:
            problems.append(f"{len(missing)} 张图的文件不存在（例：{missing[:3]}）")
        else:
            print("  ✓ 所有图像文件都存在")
    else:
        warnings.append("未提供 --images，跳过了图像存在性检查")

    # ---- 3.2 bbox 合法性 ----
    bad_zero, bad_neg, bad_oob, bad_tiny, bad_ar = [], [], [], [], []
    areas, ars, per_img = [], [], Counter()
    for a in anns:
        im = id_to_img.get(a["image_id"])
        if im is None:
            problems.append(f"标注 {a['id']} 指向不存在的图像 {a['image_id']}")
            continue
        x, y, w, h = a["bbox"]
        per_img[a["image_id"]] += 1
        area = w * h
        areas.append(area)
        if w <= 0 or h <= 0:
            bad_neg.append(a["id"])
            continue
        if w * h <= 0:
            bad_zero.append(a["id"])
        if x < -1 or y < -1 or x + w > im["width"] + 1 or y + h > im["height"] + 1:
            bad_oob.append(a["id"])
        side = float(np.sqrt(area))
        if side < args.min_side:
            bad_tiny.append(a["id"])
        ar = max(w, h) / max(min(w, h), 1e-6)
        ars.append(ar)
        if ar > args.max_aspect:
            bad_ar.append(a["id"])

    if bad_neg:
        problems.append(f"{len(bad_neg)} 个框宽或高 <= 0（例 id {bad_neg[:5]}）")
    if bad_oob:
        problems.append(f"{len(bad_oob)} 个框超出图像范围（例 id {bad_oob[:5]}）")
    if bad_tiny:
        warnings.append(
            f"{len(bad_tiny)} 个框等效边长 < {args.min_side}px（例 id {bad_tiny[:5]}）。"
            "这么小的目标标注一致性很差，建议标为 ignore 而不是硬标"
        )
    if bad_ar:
        warnings.append(f"{len(bad_ar)} 个框长宽比 > {args.max_aspect}（例 id {bad_ar[:5]}），检查是否标错了")

    # ---- 3.3 分布 ----
    if areas:
        side = np.sqrt(np.array(areas))
        print("\n  目标等效边长分布（px）:")
        for p in (5, 25, 50, 75, 95):
            print(f"    p{p:<2d} = {np.percentile(side, p):7.2f}")
        print(f"    最小 {side.min():.2f} / 最大 {side.max():.2f}  （跨度 {side.max()/max(side.min(),1e-6):.0f} 倍）")

    n_per_img = np.array([per_img.get(im["id"], 0) for im in images])
    print(f"\n  每图目标数: 均值 {n_per_img.mean():.2f} / 最大 {n_per_img.max()} / 空图 {int((n_per_img==0).sum())}")
    if (n_per_img == 0).sum() > len(images) * 0.3:
        warnings.append(
            f"空图占比 {100*(n_per_img==0).mean():.0f}% 偏高。"
            "适当保留空图有助于降低误检，但过多会拖慢训练"
        )

    cat_counter = Counter(id_to_cat.get(a["category_id"], "?") for a in anns)
    print("\n  类别分布:")
    for name, c in cat_counter.most_common():
        print(f"    {name:20s} {c:6d}  ({100*c/max(len(anns),1):5.1f}%)")
    if len(cat_counter) > 1:
        mx, mn = max(cat_counter.values()), min(cat_counter.values())
        if mx / max(mn, 1) > 10:
            warnings.append(f"类别严重不均衡（最多/最少 = {mx/max(mn,1):.0f}:1），考虑重采样或调 loss 权重")

    # ---- 3.4 重复标注（同一图内几乎重合的框）----
    by_img_boxes: Dict[int, list] = defaultdict(list)
    for a in anns:
        by_img_boxes[a["image_id"]].append(a)
    dup_pairs = 0
    for img_id, boxes in by_img_boxes.items():
        if len(boxes) < 2:
            continue
        arr = np.array([_xywh_to_xyxy(b["bbox"]) for b in boxes], dtype=np.float32)
        for i in range(len(arr)):
            for j in range(i + 1, len(arr)):
                if _iou(arr[i], arr[j]) > 0.85 and boxes[i]["category_id"] == boxes[j]["category_id"]:
                    dup_pairs += 1
    if dup_pairs:
        warnings.append(f"发现 {dup_pairs} 对高度重合的同类框（IoU>0.85），可能是重复标注")

    # ---- 汇总 ----
    print()
    if problems:
        print("❌ 必须修的问题:")
        for p in problems:
            print(f"   - {p}")
    if warnings:
        print("⚠️ 建议关注:")
        for w in warnings:
            print(f"   - {w}")
    if not problems and not warnings:
        print("✅ 未发现问题。")
    return 1 if problems else 0


def _iou(a: np.ndarray, b: np.ndarray) -> float:
    x1, y1 = max(a[0], b[0]), max(a[1], b[1])
    x2, y2 = min(a[2], b[2]), min(a[3], b[3])
    inter = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    ua = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / max(ua, 1e-9)


# --------------------------------------------------------------------------- #
# 4. 标注一致性（双人互评）
# --------------------------------------------------------------------------- #


def cmd_agreement(args: argparse.Namespace) -> int:
    """把标注员 A 当 GT、B 当预测算 mAP，得到"标注一致性"这个硬指标。

    这是论文里回应"标注质量如何保证"最有力的证据：不留主观描述，
    直接给出"两位标注员互评 mAP = 0.91"这样的数字。
    """
    hr("标注一致性（双人互评）")
    from detbench.core.io import DetDataset, DetPrediction, DetSample
    from detbench.core.metrics import compute_map_range, boxes_area

    coco_a, coco_b = _read_coco(args.ann_a), _read_coco(args.ann_b)
    cats = {c["id"]: c["name"] for c in coco_a.get("categories", [])}
    print(f"  A: {len(coco_a['images'])} 图 / {len(coco_a['annotations'])} 标注")
    print(f"  B: {len(coco_b['images'])} 图 / {len(coco_b['annotations'])} 标注")

    # 按文件名对齐（两个文件的 image_id 可能不同）
    name_a = {Path(im["file_name"]).name: im for im in coco_a["images"]}
    name_b = {Path(im["file_name"]).name: im for im in coco_b["images"]}
    common = sorted(set(name_a) & set(name_b))
    if not common:
        raise SystemExit("❌ 两份标注没有同名图像，无法对比。请确认用的是同一批图。")
    only_a = len(set(name_a) - set(name_b))
    only_b = len(set(name_b) - set(name_a))
    print(f"  共同图像 {len(common)}（A 独有 {only_a}，B 独有 {only_b}）")

    # image_id -> 文件名 的映射。**必须建这个映射**：标注里的 image_id 是数字，
    # 而两份文件对同一张图可能给出不同的 id，只有文件名是可靠的对齐键。
    a_id2name = {im["id"]: Path(im["file_name"]).name for im in coco_a["images"]}
    b_id2name = {im["id"]: Path(im["file_name"]).name for im in coco_b["images"]}

    boxes_a: Dict[str, list] = defaultdict(list)
    for a in coco_a["annotations"]:
        fn = a_id2name.get(a["image_id"])
        if fn:
            boxes_a[fn].append(a)
    boxes_b: Dict[str, list] = defaultdict(list)
    for a in coco_b["annotations"]:
        fn = b_id2name.get(a["image_id"])
        if fn:
            boxes_b[fn].append(a)

    class_names = [cats[k] for k in sorted(cats)] or ["object"]
    cat_to_idx = {k: i for i, k in enumerate(sorted(cats))}

    samples, preds = [], {}
    for i, fn in enumerate(common):
        im = name_a[fn]
        aa = boxes_a.get(fn, [])
        bb = boxes_b.get(fn, [])
        s = DetSample(
            image_id=i + 1, file_name=fn, width=im["width"], height=im["height"],
            boxes=np.array([_xywh_to_xyxy(x["bbox"]) for x in aa], dtype=np.float32).reshape(-1, 4),
            labels=np.array([cat_to_idx.get(x["category_id"], 0) for x in aa], dtype=np.int64),
        )
        preds[i + 1] = DetPrediction(
            i + 1,
            np.array([_xywh_to_xyxy(x["bbox"]) for x in bb], dtype=np.float32).reshape(-1, 4),
            np.ones(len(bb), dtype=np.float32),
            np.array([cat_to_idx.get(x["category_id"], 0) for x in bb], dtype=np.int64),
        )
        samples.append(s)

    m = compute_map_range(samples, preds, num_classes=len(class_names))
    print(f"\n  以 A 为真值、B 为预测：")
    print(f"    一致性 mAP@[.5:.95] = {m['mAP']:.4f}")
    print(f"    一致性 mAP@0.5      = {m.get('mAP50', 0):.4f}")

    # 每图逐目标 IoU 的中位数 —— 更直观的"标注重合度"
    ious = []
    for s in samples:
        pb = preds[s.image_id].boxes
        if s.num_objects == 0 or len(pb) == 0:
            continue
        for gb in s.boxes:
            best = max((_iou(gb, pbx) for pbx in pb), default=0.0)
            ious.append(best)
    if ious:
        arr = np.array(ious)
        print(f"\n  逐目标最佳 IoU（A 的每个框对 B 的最佳匹配）:")
        for p in (5, 25, 50, 75, 95):
            print(f"    p{p:<2d} = {np.percentile(arr, p):.3f}")
        print(f"    IoU > 0.5 的比例 = {100*(arr>0.5).mean():.1f}%")
        print(f"    IoU > 0.9 的比例 = {100*(arr>0.9).mean():.1f}%")

    n_a = sum(s.num_objects for s in samples)
    n_b = sum(p.scores.size for p in preds.values())
    print(f"\n  目标数: A={n_a}  B={n_b}  差异 {100*abs(n_a-n_b)/max(n_a,1):.1f}%")
    if abs(n_a - n_b) / max(n_a, 1) > 0.1:
        print("  ⚠️ 两位标注员的目标数差异超过 10%，说明存在系统性漏标/多标，建议复核。")
    else:
        print("  ✓ 目标数差异在 10% 以内。")

    print("\n  论文里可以这样写：")
    print(f"    「随机抽取 {len(common)} 帧由两位标注员独立标注，")
    print(f"      以一方为真值评测另一方，一致性 mAP@0.5 为 {m.get('mAP50', 0):.3f}，")
    print(f"      逐目标最佳 IoU 中位数为 {np.median(arr) if ious else 0:.3f}。」")
    return 0


# --------------------------------------------------------------------------- #
# 5. 抽帧去冗余
# --------------------------------------------------------------------------- #


def _ahash(path: Path, size: int = 8) -> Optional[int]:
    """均值哈希：缩到 size×size 灰度，与均值比较得到位图。

    纯 PIL + numpy 实现，不依赖 imagehash。足够检测视频相邻帧这类近重复。
    """
    try:
        from PIL import Image
    except ImportError:
        raise SystemExit("需要 Pillow：pip install pillow")
    try:
        with Image.open(path) as im:
            g = im.convert("L").resize((size, size), Image.BILINEAR)
            a = np.asarray(g, dtype=np.float32)
    except Exception:
        return None
    bits = (a > a.mean()).flatten()
    v = 0
    for b in bits:
        v = (v << 1) | int(b)
    return v


def _dhash(path: Path, size: int = 8) -> Optional[int]:
    """差值哈希：缩到 (size+1)×size，比较水平相邻像素。

    对亮度整体变化更鲁棒，配合 aHash 一起用效果更好。
    """
    from PIL import Image

    try:
        with Image.open(path) as im:
            g = im.convert("L").resize((size + 1, size), Image.BILINEAR)
            a = np.asarray(g, dtype=np.float32)
    except Exception:
        return None
    bits = (a[:, 1:] > a[:, :-1]).flatten()
    v = 0
    for b in bits:
        v = (v << 1) | int(b)
    return v


def cmd_dedupe(args: argparse.Namespace) -> int:
    """检测近似重复帧。

    **这是最容易被忽略、但影响最大的一步。** 把视频逐帧导出会让相邻帧几乎相同，
    造成两个后果：
      1. 训练集内部冗余 —— 模型反复看同一画面，泛化变差
      2. train/test 之间泄漏 —— 测试集 mAP 虚高，论文结论站不住
    审稿人对这一点很敏感。所以抽帧必须去冗余。

    已知局限（用 aHash + dHash）
    ---------------------------
    感知哈希只看**灰度结构**，这正是它能忽略曝光/白平衡变化的原因，
    但也带来两个盲区：

    * **纯色或近似平坦的图无法区分**。实测若干不同颜色的纯色图，
      aHash 与 dHash 全都得到 0。这是设计使然，不是 bug。
      本工具会在发现大量图哈希相同时给出警告。
    * **只做全局比较**。目标在画面中移动但背景不变的帧（航拍最常见的情形）
      全局哈希距离仍然很小，会被判为重复 —— 对"抽帧去冗余"这个目的而言
      这是期望行为，但如果你要保留"目标位置变化"的样本，需要调小阈值
      或改用目标区域裁剪后再哈希。
    """
    hr("抽帧去冗余（近似重复帧检测）")
    src = Path(args.images)
    if not src.exists():
        raise SystemExit(f"❌ 目录不存在：{src}")

    files = sorted(p for p in src.rglob("*") if p.suffix.lower() in IMG_EXTS)
    print(f"  扫描 {len(files)} 个文件，哈希阈值 {args.threshold}（汉明距离）")

    # 计算哈希
    hashes: List[Tuple[Path, int, int]] = []
    failed = 0
    for i, p in enumerate(files, 1):
        ha, hd = _ahash(p), _dhash(p)
        if ha is None or hd is None:
            failed += 1
            continue
        hashes.append((p, ha, hd))
        if args.verbose and i % 500 == 0:
            print(f"    哈希进度 {i}/{len(files)}")
    if failed:
        print(f"  ⚠️ {failed} 个文件无法读取，已跳过")
    print(f"  成功计算 {len(hashes)} 个哈希")
    if not hashes:
        return 1

    # 哈希退化检查：大量图哈希相同说明这些图没有可区分的灰度结构
    # （纯色、极简纹理、或者严重过曝/欠曝），此时去重结果不可信。
    hash_counts = Counter(h for _, h, _ in hashes)
    top_hash, top_n = hash_counts.most_common(1)[0]
    if top_n > max(10, len(hashes) * 0.1):
        print(
            f"\n  ⚠️ 有 {top_n}/{len(hashes)} 张图的哈希完全相同"
            f"（{100*top_n/len(hashes):.1f}%）。\n"
            "     这些图缺少可区分的灰度结构（纯色/极简纹理/严重过曝过暗），\n"
            "     感知哈希无法区分它们，本次去重结果不可信。\n"
            "     建议：确认图像内容是否正常，或降低 --threshold 后人工抽查。"
        )

    # 贪心去重：按文件名排序（视频帧天然有序），保留第一个，后续与已保留的比对
    kept: List[Tuple[Path, int, int]] = []
    dropped: List[Tuple[Path, Path, int]] = []  # (被丢的, 与谁重复, 距离)
    for p, ha, hd in hashes:
        dup_of, dup_d = None, 999
        # 只与"最近保留的 N 个"比较：视频冗余是局部的，全量比较是 O(n²)
        for kp, kha, khd in reversed(kept[-args.window:]):
            d = bin(ha ^ kha).count("1") + bin(hd ^ khd).count("1")
            if d < dup_d:
                dup_of, dup_d = kp, d
            if dup_d == 0:
                break
        if dup_of is not None and dup_d <= args.threshold:
            dropped.append((p, dup_of, dup_d))
        else:
            kept.append((p, ha, hd))

    print(f"\n  保留 {len(kept)} 帧 / 判定重复 {len(dropped)} 帧"
          f"（冗余率 {100*len(dropped)/max(len(hashes),1):.1f}%）")

    if args.move_to:
        dst = Path(args.move_to)
        dst.mkdir(parents=True, exist_ok=True)
        for p, _, _ in dropped:
            target = dst / p.name
            shutil.move(str(p), str(target))
        print(f"  重复帧已移动到: {dst}")

    if args.write_keep_list:
        kp = Path(args.write_keep_list)
        kp.parent.mkdir(parents=True, exist_ok=True)
        kp.write_text("\n".join(str(p) for p, _, _ in kept) + "\n", encoding="utf-8")
        print(f"  保留清单已写入: {kp}")

    if args.report and dropped:
        rp = Path(args.report)
        with rp.open("w", encoding="utf-8-sig", newline="") as f:
            wr = csv.writer(f)
            wr.writerow(["dropped", "duplicate_of", "hamming_distance"])
            for p, q, d in dropped[:5000]:
                wr.writerow([p.name, q.name, d])
        print(f"  重复明细已写入: {rp}（最多 5000 行）")

    # 给一点可操作的建议
    print()
    if len(dropped) / max(len(hashes), 1) > 0.5:
        print("  💡 冗余率超过 50%，说明确实是逐帧导出的。这是常见做法但不可取：")
        print("     即使去重后，也要确保**同一段视频的帧不跨 train/test 划分**。")
        print("     用 tools/dataset.py split --group-key clip_id 按架次划分。")
    print("\n✅ 去冗余完成。")
    return 0


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(
        prog="detbench-dataset",
        description="DetBench 数据集工具：转换 / 划分 / 质检 / 一致性 / 去冗余",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    sub = ap.add_subparsers(dest="cmd", required=True)

    p = sub.add_parser("convert", help="格式转换（coco/yolo/voc 之间）")
    p.add_argument("--from", dest="src_format", required=True, choices=["coco", "yolo", "voc"])
    p.add_argument("--to", dest="dst_format", required=True, choices=["coco", "yolo", "voc"])
    p.add_argument("--ann", default=None, help="源标注路径（coco json 或 voc 目录）")
    p.add_argument("--images", default=None, help="图像目录（yolo/voc 源需要）")
    p.add_argument("--labels", default=None, help="YOLO 标签目录")
    p.add_argument("--class-names", nargs="*", default=["helicopter"], help="类别名列表")
    p.add_argument("--out", required=True)
    p.set_defaults(func=cmd_convert)

    p = sub.add_parser("split", help="按组（飞行架次）分层划分")
    p.add_argument("--ann", required=True)
    p.add_argument("--attr", default=None, help="属性表 CSV（提供 group-key 与分层列）")
    p.add_argument("--group-key", default="clip_id", help="分组列名（默认 clip_id）")
    p.add_argument("--stratify-by", default=None, help="按哪一列分层，保证各划分覆盖各档")
    p.add_argument("--ratio-train", type=float, default=0.7)
    p.add_argument("--ratio-val", type=float, default=0.1)
    p.add_argument("--ratio-test", type=float, default=0.2)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--keep-groups", action="store_true", help="把组标签写进输出的 COCO")
    p.add_argument("--out", required=True)
    p.set_defaults(func=cmd_split)

    p = sub.add_parser("check", help="质检：框合法性 / 分布 / 重复标注")
    p.add_argument("--ann", required=True)
    p.add_argument("--images", default=None, help="图像根目录（检查文件是否存在）")
    p.add_argument("--min-side", type=float, default=4.0, help="小于该等效边长的框给警告")
    p.add_argument("--max-aspect", type=float, default=8.0, help="长宽比超过该值给警告")
    p.set_defaults(func=cmd_check)

    p = sub.add_parser("agreement", help="标注一致性：两人互评 mAP")
    p.add_argument("--ann-a", required=True)
    p.add_argument("--ann-b", required=True)
    p.set_defaults(func=cmd_agreement)

    p = sub.add_parser("dedupe", help="抽帧去冗余（近似重复帧检测）")
    p.add_argument("--images", required=True)
    p.add_argument("--threshold", type=int, default=8,
                   help="汉明距离阈值，越小越严格。8 是常用起点")
    p.add_argument("--window", type=int, default=200,
                   help="只与最近保留的 N 帧比较（视频冗余是局部的，避免 O(n^2)）")
    p.add_argument("--move-to", default=None, help="把重复帧移动到这个目录（默认只报告不移动）")
    p.add_argument("--write-keep-list", default=None, help="把保留帧清单写入文件")
    p.add_argument("--report", default=None, help="把重复明细写入 CSV")
    p.add_argument("--verbose", action="store_true")
    p.set_defaults(func=cmd_dedupe)

    return ap


def main() -> int:
    args = build_parser().parse_args()
    try:
        return int(args.func(args) or 0)
    except FileNotFoundError as exc:
        print(f"\n❌ 路径错误\n{exc}\n", file=sys.stderr)
        return 2
    except KeyboardInterrupt:
        print("\n已中断。", file=sys.stderr)
        return 130


if __name__ == "__main__":
    raise SystemExit(main())
