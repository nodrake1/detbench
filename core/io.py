"""统一数据接口。

把 COCO / YOLO / VOC 三种标注格式归一化成同一个内存结构，
并挂载"图像属性元数据"（距离档 / 模糊档 / 光照档 ...），供分层评测使用。

核心概念
--------
DetSample : 一张图 + 其全部标注 + 其属性元数据
AttrTable: 图像属性表，把属性名映射到"每张图的取值"，用于分组评测

为什么要有属性表
----------------
普通评测只告诉你 mAP=0.83。但研究结论藏在条件分布里：
    "在目标像素高度 < 16px 时 mAP 掉到 0.21"
    "在重模糊帧上 DETR 反超 YOLO 0.04"
这些结论只能靠按属性分组评测得到，而分组需要每张图的属性标签。
所以属性表是一等公民，不是可选项。
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np

# --------------------------------------------------------------------------- #
# 数据结构
# --------------------------------------------------------------------------- #


@dataclass
class DetSample:
    """单张图像及其标注与属性。

    Attributes
    ----------
    image_id : int
        全局唯一图像 id。
    file_name : str
        图像文件名（相对 ``images_root``）。
    width, height : int
        图像尺寸（像素）。
    boxes : np.ndarray, shape (N, 4)
        标注框，**xyxy** 绝对像素坐标。
    labels : np.ndarray, shape (N,)
        类别 id（0-based，与 ``class_names`` 对应）。
    attrs : dict
        图像级属性，例如 ``{"distance_bin": "far", "blur": "heavy"}``。
        也支持目标级属性：值为长度 N 的 list，例如 ``{"blur": ["light","heavy"]}``。
    """

    image_id: int
    file_name: str
    width: int
    height: int
    boxes: np.ndarray
    labels: np.ndarray
    attrs: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.boxes = np.asarray(self.boxes, dtype=np.float32).reshape(-1, 4)
        self.labels = np.asarray(self.labels, dtype=np.int64).reshape(-1)

    @property
    def num_objects(self) -> int:
        return int(self.boxes.shape[0])

    def box_areas(self) -> np.ndarray:
        wh = np.clip(self.boxes[:, 2:4] - self.boxes[:, 0:2], 0.0, None)
        return wh[:, 0] * wh[:, 1]

    def filtered(self, keep: np.ndarray) -> "DetSample":
        """返回只保留 ``keep`` 掩码内标注的副本（属性表同步裁剪）。"""
        keep = np.asarray(keep, dtype=bool)
        new_attrs: Dict[str, Any] = {}
        for k, v in self.attrs.items():
            if isinstance(v, (list, tuple, np.ndarray)) and len(v) == self.num_objects:
                new_attrs[k] = list(np.asarray(v, dtype=object)[keep])
            else:
                new_attrs[k] = v
        return DetSample(
            image_id=self.image_id,
            file_name=self.file_name,
            width=self.width,
            height=self.height,
            boxes=self.boxes[keep],
            labels=self.labels[keep],
            attrs=new_attrs,
        )


@dataclass
class DetPrediction:
    """单个模型对一张图的预测结果。

    boxes 为 **xyxy** 绝对像素坐标；scores 与 labels 一一对应。
    """

    image_id: int
    boxes: np.ndarray
    scores: np.ndarray
    labels: np.ndarray

    def __post_init__(self) -> None:
        self.boxes = np.asarray(self.boxes, dtype=np.float32).reshape(-1, 4)
        self.scores = np.asarray(self.scores, dtype=np.float32).reshape(-1)
        self.labels = np.asarray(self.labels, dtype=np.int64).reshape(-1)

    def topk(self, k: int) -> "DetPrediction":
        """按分数保留前 k 个预测。"""
        if k <= 0 or self.scores.size <= k:
            return self
        idx = np.argsort(-self.scores)[:k]
        return DetPrediction(self.image_id, self.boxes[idx], self.scores[idx], self.labels[idx])

    def filter_by_score(self, thr: float) -> "DetPrediction":
        keep = self.scores >= thr
        return DetPrediction(self.image_id, self.boxes[keep], self.scores[keep], self.labels[keep])


@dataclass
class DetDataset:
    """一个数据集划分（train / val / test）。"""

    name: str
    samples: List[DetSample]
    class_names: List[str]

    def __len__(self) -> int:
        return len(self.samples)

    def __iter__(self):
        return iter(self.samples)

    @property
    def num_images(self) -> int:
        return len(self.samples)

    @property
    def num_objects(self) -> int:
        return int(sum(s.num_objects for s in self.samples))

    @property
    def num_classes(self) -> int:
        return len(self.class_names)

    def by_id(self) -> Dict[int, DetSample]:
        return {s.image_id: s for s in self.samples}

    def summary(self) -> Dict[str, Any]:
        """数据集概览，写进 result.json 便于事后核对。"""
        areas = np.concatenate([s.box_areas() for s in self.samples]) if self.samples else np.zeros(0)
        labels = np.concatenate([s.labels for s in self.samples]) if self.samples else np.zeros(0, dtype=int)
        per_class = {self.class_names[c]: int((labels == c).sum()) for c in range(self.num_classes)}
        return {
            "name": self.name,
            "num_images": self.num_images,
            "num_objects": self.num_objects,
            "num_classes": self.num_classes,
            "class_names": list(self.class_names),
            "objects_per_class": per_class,
            "objects_per_image": round(self.num_objects / max(self.num_images, 1), 3),
            "box_area_px": {
                "min": float(areas.min()) if areas.size else 0.0,
                "median": float(np.median(areas)) if areas.size else 0.0,
                "max": float(areas.max()) if areas.size else 0.0,
            },
            # 目标等效边长分布是"小目标难度"最直观的刻画
            "box_side_px_percentile": {
                f"p{p}": float(np.percentile(np.sqrt(areas), p)) if areas.size else 0.0
                for p in (5, 25, 50, 75, 95)
            },
        }


# --------------------------------------------------------------------------- #
# 属性表
# --------------------------------------------------------------------------- #


class AttrTable:
    """图像/目标属性表。

    从 CSV 或 JSON 载入，形如::

        image_name,distance_bin,blur,lighting
        DJI_0001.jpg,far,heavy,backlit
        DJI_0002.jpg,mid,light,normal

    第一列默认为图像文件名（或 path），其余列是属性。
    列名以 ``obj_`` 开头的列会被当作"目标级属性"（值为分号分隔的列表）。
    """

    def __init__(self, table: Dict[str, Dict[str, Any]] | None = None) -> None:
        self._table: Dict[str, Dict[str, Any]] = table or {}

    def __len__(self) -> int:
        return len(self._table)

    def get(self, file_name: str) -> Dict[str, Any]:
        """取某张图的属性；查不到返回空 dict（分层评测时会归入 ``unknown``）。"""
        return self._table.get(os.path.basename(file_name), {})

    @classmethod
    def from_csv(cls, path: str | Path, key_col: Optional[str] = None) -> "AttrTable":
        import csv

        path = Path(path)
        table: Dict[str, Dict[str, Any]] = {}
        with path.open("r", encoding="utf-8-sig", newline="") as f:
            reader = csv.DictReader(f)
            if reader.fieldnames is None:
                return cls()
            key = key_col or reader.fieldnames[0]
            for row in reader:
                raw_key = (row.get(key) or "").strip()
                if not raw_key:
                    continue
                rec: Dict[str, Any] = {}
                for col, val in row.items():
                    if col == key:
                        continue
                    val = (val or "").strip()
                    if col.startswith("obj_"):
                        rec[col[4:]] = [v.strip() for v in val.split(";") if v.strip()]
                    else:
                        rec[col] = val
                table[os.path.basename(raw_key)] = rec
        return cls(table)

    @classmethod
    def from_json(cls, path: str | Path) -> "AttrTable":
        with Path(path).open("r", encoding="utf-8") as f:
            raw = json.load(f)
        return cls({os.path.basename(k): v for k, v in raw.items()})

    @classmethod
    def load(cls, path: str | Path) -> "AttrTable":
        path = Path(path)
        if path.suffix.lower() == ".json":
            return cls.from_json(path)
        return cls.from_csv(path)

    def attach(self, dataset: DetDataset) -> DetDataset:
        """把属性挂到数据集里的每个样本上（就地修改并返回）。"""
        for s in dataset.samples:
            s.attrs.update(self.get(s.file_name))
        return dataset


# --------------------------------------------------------------------------- #
# 各格式加载器
# --------------------------------------------------------------------------- #


def _xywh_to_xyxy(boxes: np.ndarray) -> np.ndarray:
    out = np.empty_like(boxes, dtype=np.float32)
    out[:, 0] = boxes[:, 0]
    out[:, 1] = boxes[:, 1]
    out[:, 2] = boxes[:, 0] + boxes[:, 2]
    out[:, 3] = boxes[:, 1] + boxes[:, 3]
    return out


def load_coco(
    ann_file: str | Path,
    images_root: str | Path,
    name: str = "coco",
    attr_file: Optional[str | Path] = None,
    category_map: Optional[Dict[int, int]] = None,
) -> DetDataset:
    """载入 COCO 格式标注。

    Parameters
    ----------
    ann_file : COCO ``instances_*.json`` 路径
    images_root : 图像根目录
    attr_file : 可选的属性表（CSV/JSON）
    category_map : 可选的类别 id 重映射（原 category_id -> 0-based 索引）。
                   默认按 COCO 文件里的类别出现顺序重新编号为 0..K-1。
    """
    ann_file = Path(ann_file)
    with ann_file.open("r", encoding="utf-8") as f:
        data = json.load(f)

    cat_id_to_idx: Dict[int, int] = {}
    class_names: List[str] = []
    if category_map is not None:
        cat_id_to_idx = dict(category_map)
        by_idx = {v: k for k, v in cat_id_to_idx.items()}
        cats = {c["id"]: c["name"] for c in data.get("categories", [])}
        class_names = [cats.get(by_idx[i], f"class_{i}") for i in range(len(cat_id_to_idx))]
    else:
        for i, c in enumerate(sorted(data.get("categories", []), key=lambda x: x["id"])):
            cat_id_to_idx[c["id"]] = i
            class_names.append(c["name"])
        if not class_names:  # 无 categories 字段时的兜底
            ids = sorted({a["category_id"] for a in data.get("annotations", [])})
            cat_id_to_idx = {cid: i for i, cid in enumerate(ids)}
            class_names = [f"class_{i}" for i in range(len(ids))]

    per_image: Dict[int, List[dict]] = {}
    for a in data.get("annotations", []):
        if a.get("iscrowd", 0):
            continue  # 本框架不处理 crowd 区域；如需，请显式扩展
        per_image.setdefault(a["image_id"], []).append(a)

    samples: List[DetSample] = []
    for img in data.get("images", []):
        anns = per_image.get(img["id"], [])
        if anns:
            boxes = _xywh_to_xyxy(
                np.asarray([[a["bbox"][0], a["bbox"][1], a["bbox"][2], a["bbox"][3]] for a in anns], dtype=np.float32)
            )
            labels = np.asarray([cat_id_to_idx[a["category_id"]] for a in anns], dtype=np.int64)
        else:
            boxes = np.zeros((0, 4), dtype=np.float32)
            labels = np.zeros((0,), dtype=np.int64)
        samples.append(
            DetSample(
                image_id=int(img["id"]),
                file_name=img["file_name"],
                width=int(img["width"]),
                height=int(img["height"]),
                boxes=boxes,
                labels=labels,
            )
        )

    ds = DetDataset(name=name, samples=samples, class_names=class_names)
    if attr_file is not None:
        AttrTable.load(attr_file).attach(ds)
    return ds


def load_yolo(
    images_dir: str | Path,
    labels_dir: str | Path,
    class_names: Sequence[str],
    name: str = "yolo",
    attr_file: Optional[str | Path] = None,
    image_sizes: Optional[Dict[str, Tuple[int, int]]] = None,
) -> DetDataset:
    """载入 YOLO 格式标注（每图一个 .txt，``cls cx cy w h`` 归一化坐标）。

    ``image_sizes`` 为空时会用 Pillow 读取每张图的真实尺寸（较慢但正确）。
    """
    images_dir, labels_dir = Path(images_dir), Path(labels_dir)
    exts = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}
    img_files = sorted(p for p in images_dir.rglob("*") if p.suffix.lower() in exts)

    samples: List[DetSample] = []
    for i, img_path in enumerate(img_files):
        rel = img_path.relative_to(images_dir).as_posix()
        label_path = labels_dir / (Path(rel).with_suffix(".txt"))
        if image_sizes and rel in image_sizes:
            w, h = image_sizes[rel]
        else:
            w, h = _read_image_size(img_path)

        boxes: List[List[float]] = []
        labels: List[int] = []
        if label_path.exists():
            for line in label_path.read_text(encoding="utf-8").splitlines():
                parts = line.split()
                if len(parts) < 5:
                    continue
                c = int(float(parts[0]))
                cx, cy, bw, bh = (float(v) for v in parts[1:5])
                # 归一化 cxcywh -> 绝对 xyxy，并裁剪到图像范围
                x1 = (cx - bw / 2.0) * w
                y1 = (cy - bh / 2.0) * h
                x2 = (cx + bw / 2.0) * w
                y2 = (cy + bh / 2.0) * h
                boxes.append([max(0.0, x1), max(0.0, y1), min(float(w), x2), min(float(h), y2)])
                labels.append(c)

        samples.append(
            DetSample(
                image_id=i,
                file_name=rel,
                width=w,
                height=h,
                boxes=np.asarray(boxes, dtype=np.float32).reshape(-1, 4),
                labels=np.asarray(labels, dtype=np.int64),
            )
        )

    ds = DetDataset(name=name, samples=samples, class_names=list(class_names))
    if attr_file is not None:
        AttrTable.load(attr_file).attach(ds)
    return ds


def load_voc(
    ann_dir: str | Path,
    images_root: str | Path,
    class_names: Sequence[str],
    name: str = "voc",
    attr_file: Optional[str | Path] = None,
) -> DetDataset:
    """载入 Pascal VOC 格式标注（每图一个 XML）。"""
    import xml.etree.ElementTree as ET

    ann_dir, images_root = Path(ann_dir), Path(images_root)
    name_to_idx = {n: i for i, n in enumerate(class_names)}
    samples: List[DetSample] = []

    for i, xml_path in enumerate(sorted(ann_dir.glob("*.xml"))):
        root = ET.parse(xml_path).getroot()
        size = root.find("size")
        w = int(float(size.findtext("width", "0"))) if size is not None else 0
        h = int(float(size.findtext("height", "0"))) if size is not None else 0
        fname = root.findtext("filename") or xml_path.with_suffix(".jpg").name

        boxes, labels = [], []
        for obj in root.findall("object"):
            cname = obj.findtext("name", "")
            if cname not in name_to_idx:
                continue
            bb = obj.find("bndbox")
            if bb is None:
                continue
            boxes.append(
                [
                    float(bb.findtext("xmin", "0")),
                    float(bb.findtext("ymin", "0")),
                    float(bb.findtext("xmax", "0")),
                    float(bb.findtext("ymax", "0")),
                ]
            )
            labels.append(name_to_idx[cname])

        if w == 0 or h == 0:
            w, h = _read_image_size(images_root / fname)
        samples.append(
            DetSample(
                image_id=i,
                file_name=fname,
                width=w,
                height=h,
                boxes=np.asarray(boxes, dtype=np.float32).reshape(-1, 4),
                labels=np.asarray(labels, dtype=np.int64),
            )
        )

    ds = DetDataset(name=name, samples=samples, class_names=list(class_names))
    if attr_file is not None:
        AttrTable.load(attr_file).attach(ds)
    return ds


def _read_image_size(path: Path) -> Tuple[int, int]:
    from PIL import Image

    with Image.open(path) as im:
        return int(im.width), int(im.height)


# --------------------------------------------------------------------------- #
# 划分工具
# --------------------------------------------------------------------------- #


def split_by_group(
    dataset: DetDataset,
    group_key: str,
    ratios: Tuple[float, float, float] = (0.7, 0.1, 0.2),
    seed: int = 42,
) -> Tuple[DetDataset, DetDataset, DetDataset]:
    """按"组"（例如飞行架次 clip_id）划分 train/val/test。

    ⚠️ 绝不能按帧随机划分。同一段视频的相邻帧高度相似，
    按帧随机划分会造成 scene leakage，使得测试集 mAP 虚高、泛化性虚假。
    这是数据集论文最常被审稿人挑出的问题之一。

    ``group_key`` 是样本属性表中的某个键（如 ``clip_id``）。
    """
    rng = np.random.RandomState(seed)
    groups: Dict[str, List[DetSample]] = {}
    for s in dataset.samples:
        g = str(s.attrs.get(group_key, "__nogroup__"))
        groups.setdefault(g, []).append(s)

    keys = sorted(groups.keys())
    rng.shuffle(keys)

    n = len(keys)
    n_train = int(round(n * ratios[0]))
    n_val = int(round(n * ratios[1]))
    # 至少给每个划分留一组（组数足够时）
    if n >= 3:
        n_train = max(1, min(n_train, n - 2))
        n_val = max(1, min(n_val, n - n_train - 1))
    else:
        n_train, n_val = max(1, n - 1), 0

    train_keys = set(keys[:n_train])
    val_keys = set(keys[n_train : n_train + n_val])
    test_keys = set(keys[n_train + n_val :])

    def build(subset_keys: set, suffix: str) -> DetDataset:
        smp = [s for k in sorted(subset_keys) for s in groups[k]]
        return DetDataset(name=f"{dataset.name}_{suffix}", samples=smp, class_names=list(dataset.class_names))

    return build(train_keys, "train"), build(val_keys, "val"), build(test_keys, "test")
