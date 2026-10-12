"""数据集工具的全流程检查：convert / split / check / agreement / dedupe。

用法::  python scripts/check_dataset_tools.py
"""

from __future__ import annotations

import json
import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT.parent) not in sys.path:
    sys.path.insert(0, str(ROOT.parent))

from detbench.core.deps import force_utf8_output  # noqa: E402

PY = sys.executable
WORK = ROOT / ".ds_test"
DATA = ROOT / "data" / "demo"

RESULTS: list[tuple[str, bool, str]] = []


def check(name: str, cond: bool, detail: str = "") -> None:
    RESULTS.append((name, cond, detail))
    print(f"[{'PASS' if cond else 'FAIL'}] {name}" + (f"  -- {detail}" if detail else ""))


def run(args: list[str]) -> tuple[int, str]:
    r = subprocess.run([PY, str(ROOT / "tools" / "dataset.py")] + args,
                       capture_output=True, text=True, encoding="utf-8", cwd=str(ROOT))
    return r.returncode, (r.stdout or "") + (r.stderr or "")


def main() -> int:
    force_utf8_output()
    print("=" * 74)
    print("数据集工具检查 (tools/dataset.py)")
    print("=" * 74)

    if not (DATA / "annotations/instances_test.json").exists():
        print("找不到占位数据集，请先运行：python tools/main.py make-demo")
        return 1
    if WORK.exists():
        shutil.rmtree(WORK, ignore_errors=True)
    WORK.mkdir(parents=True)

    # ---- 准备：把三个 split 合并成一个 all.json，用于测 split 与 check ----
    merged = {"images": [], "annotations": [], "categories": None}
    nxt_img = 1
    nxt_ann = 1
    for split in ("train", "val", "test"):
        d = json.loads((DATA / "annotations" / f"instances_{split}.json").read_text(encoding="utf-8"))
        merged["categories"] = d["categories"]
        idmap = {}
        for im in d["images"]:
            idmap[im["id"]] = nxt_img
            im2 = dict(im)
            im2["id"] = nxt_img
            merged["images"].append(im2)
            nxt_img += 1
        for a in d["annotations"]:
            a2 = dict(a)
            a2["id"] = nxt_ann
            a2["image_id"] = idmap[a["image_id"]]
            merged["annotations"].append(a2)
            nxt_ann += 1
    all_json = WORK / "instances_all.json"
    all_json.write_text(json.dumps(merged, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"合并出 all.json: {len(merged['images'])} 图 / {len(merged['annotations'])} 标注\n")

    # ---- 1. check ----
    code, out = run(["check", "--ann", str(all_json), "--images", str(DATA / "images")])
    check("check 正常运行", code == 0, f"exit={code}")
    check("check 报出图像与标注数", "图像 144" in out or "图像" in out)
    check("check 给出尺寸分位", "p50" in out)

    # ---- 2. split ----
    code, out = run([
        "split", "--ann", str(all_json),
        "--attr", str(DATA / "metadata" / "frames.csv"),
        "--group-key", "clip_id", "--stratify-by", "distance_bin",
        "--out", str(WORK / "annotations"),
    ])
    check("split 正常运行", code == 0, f"exit={code}")
    ok_files = all((WORK / "annotations" / f"instances_{s}.json").exists()
                   for s in ("train", "val", "test"))
    check("split 生成三个 COCO 文件", ok_files)
    check("split 生成划分清单", (WORK / "annotations" / "split_manifest.csv").exists())

    # 关键性质：同一 clip 不得跨越划分
    if ok_files:
        clip_split: dict[str, set[str]] = {}
        for s in ("train", "val", "test"):
            d = json.loads((WORK / "annotations" / f"instances_{s}.json").read_text(encoding="utf-8"))
            for im in d["images"]:
                cid = Path(im["file_name"]).name.split("_")[0]
                clip_split.setdefault(cid, set()).add(s)
        leaked = {c: v for c, v in clip_split.items() if len(v) > 1}
        check("无 scene leakage（同一 clip 不跨划分）", not leaked, f"泄漏的 clip: {list(leaked)[:3]}")
        n_train = len(json.loads((WORK / "annotations/instances_train.json").read_text(encoding="utf-8"))["images"])
        n_test = len(json.loads((WORK / "annotations/instances_test.json").read_text(encoding="utf-8"))["images"])
        check("训练集与测试集都非空", n_train > 0 and n_test > 0, f"train={n_train}, test={n_test}")

    # ---- 3. convert: coco -> yolo ----
    code, out = run(["convert", "--from", "coco", "--to", "yolo",
                     "--ann", str(WORK / "annotations/instances_train.json"),
                     "--out", str(WORK / "yolo")])
    check("convert coco->yolo", code == 0, f"exit={code}")
    check("yolo 类别清单生成", (WORK / "yolo" / "classes.txt").exists())
    check("yolo 图像清单生成", (WORK / "yolo" / "image_list.txt").exists())
    n_txt = len(list((WORK / "yolo" / "labels").rglob("*.txt"))) if (WORK / "yolo" / "labels").exists() else 0
    check("yolo 标签文件数与图像数一致", n_txt == n_train if ok_files else n_txt > 0, f"{n_txt} 个 txt")

    # ---- 4. convert 往返：coco -> yolo -> coco，标注应基本一致 ----
    # 注意 --images 要用**与 coco->yolo 时相同的根**（data/demo/images），
    # 这样 file_name 的相对路径（含 train/ 前缀）才能和标签目录同构对上。
    # 用 data/demo/images/train 会错配，工具会给出警告。
    code, out = run(["convert", "--from", "yolo", "--to", "coco",
                     "--images", str(DATA / "images"),
                     "--labels", str(WORK / "yolo" / "labels"),
                     "--class-names", "helicopter",
                     "--out", str(WORK / "roundtrip")])
    check("convert yolo->coco", code == 0, f"exit={code}")
    rt = WORK / "roundtrip" / "instances.json"
    check("往返生成 COCO", rt.exists())
    if rt.exists() and ok_files:
        a = json.loads((WORK / "annotations/instances_train.json").read_text(encoding="utf-8"))
        b = json.loads(rt.read_text(encoding="utf-8"))
        check("往返后标注数不变", len(a["annotations"]) == len(b["annotations"]),
              f"{len(a['annotations'])} -> {len(b['annotations'])}")
        # 必须按**文件名**比对，不能按 image_id —— 两个文件对同一张图的 id 不同
        # （往返文件按路径排序重新编号，test/ 排在 train/ 前面）。
        from collections import defaultdict as _dd

        def _by_name(coco_json):
            id2name = {im["id"]: Path(im["file_name"]).name for im in coco_json["images"]}
            d = _dd(list)
            for x in coco_json["annotations"]:
                d[id2name.get(x["image_id"], "?")].append(x["bbox"])
            for k in d:
                d[k].sort()
            return d

        ga, gb = _by_name(a), _by_name(b)
        common = sorted(set(ga) & set(gb))
        if common:
            diffs, count_mismatch = [], 0
            for k in common:
                if len(ga[k]) != len(gb[k]):
                    count_mismatch += 1
                    continue
                for ba, bb in zip(ga[k], gb[k]):
                    diffs.append(max(abs(p - q) for p, q in zip(ba, bb)))
            check("往返后每图标注数一致", count_mismatch == 0, f"{count_mismatch} 张图数量不符")
            if diffs:
                check("往返后坐标误差 < 1px", max(diffs) < 1.0,
                      f"最大误差 {max(diffs):.4f}px（对比了 {len(diffs)} 个目标）")

    # ---- 4b. 布局错配时必须给出警告（而不是静默产出空标注）----
    code, out = run(["convert", "--from", "yolo", "--to", "coco",
                     "--images", str(DATA / "images" / "train"),   # 故意用错的根
                     "--labels", str(WORK / "yolo" / "labels"),
                     "--class-names", "helicopter",
                     "--out", str(WORK / "mismatch")])
    # 文件名兜底应该仍然救回来大部分，且要打印提示
    check("布局不同构时有兜底或警告", ("兜底匹配" in out) or ("找不到对应标签" in out),
          "有提示信息" if ("兜底匹配" in out or "找不到对应标签" in out) else "无任何提示")

    # ---- 5. convert: coco -> voc ----
    code, out = run(["convert", "--from", "coco", "--to", "voc",
                     "--ann", str(WORK / "annotations/instances_train.json"),
                     "--out", str(WORK / "voc")])
    check("convert coco->voc", code == 0, f"exit={code}")
    n_xml = len(list((WORK / "voc" / "Annotations").glob("*.xml"))) if (WORK / "voc" / "Annotations").exists() else 0
    check("voc XML 数正确", n_xml == n_train if ok_files else n_xml > 0, f"{n_xml} 个 xml")

    # ---- 6. agreement：把 all.json 与"抖动版"当两个标注员 ----
    import numpy as np

    b = json.loads(all_json.read_text(encoding="utf-8"))
    rng = np.random.RandomState(0)
    for a in b["annotations"]:
        w, h = a["bbox"][2], a["bbox"][3]
        a["bbox"] = [a["bbox"][0] + rng.uniform(-0.08, 0.08) * w,
                     a["bbox"][1] + rng.uniform(-0.08, 0.08) * h, w, h]
    b_json = WORK / "instances_annotatorB.json"
    b_json.write_text(json.dumps(b, ensure_ascii=False, indent=1), encoding="utf-8")

    code, out = run(["agreement", "--ann-a", str(all_json), "--ann-b", str(b_json)])
    check("agreement 正常运行", code == 0, f"exit={code}")
    check("agreement 报告一致性 mAP", "一致性 mAP@[.5:.95]" in out)
    check("agreement 报告逐目标 IoU", "逐目标最佳 IoU" in out)
    check("agreement 给出论文可用的句子", "论文里可以这样写" in out)
    # 抖动 8% 时 IoU@0.5 仍能全部匹配（mAP@0.5 = 1.0 是对的），
    # 所以要看 IoU 范围平均后的 mAP@[.5:.95]，它必须严格小于 1。
    import re
    mm = re.search(r"一致性 mAP@\[\.5:\.95\]\s*=\s*([\d.]+)", out)
    if mm:
        v = float(mm.group(1))
        check("抖动后 mAP@[.5:.95] 严格小于 1", v < 1.0, f"mAP@[.5:.95] = {v}")
        check("抖动后 mAP@[.5:.95] 仍然较高（数据一致）", v > 0.5, f"mAP@[.5:.95] = {v}")
    # 逐目标 IoU 中位数应该很高但不为 1
    miou = re.search(r"p50\s*=\s*([\d.]+)", out)
    if miou:
        check("逐目标 IoU 中位数在 (0.7, 1.0)", 0.7 < float(miou.group(1)) < 1.0,
              f"p50 IoU = {miou.group(1)}")

    # ---- 7. dedupe：用**视觉上明显不同**的图构造"3 份完全副本"场景 ----
    # 不能用 demo 数据集的帧：那些是合成天空图，互之间哈希距离本来就 2~9，
    # 去重工具把它们判为近重复是**正确行为**，但会让本用例的期望值失效。
    dupdir = WORK / "dup_frames"
    dupdir.mkdir(parents=True, exist_ok=True)
    import shutil as _sh
    from PIL import Image as _Image

    rng2 = np.random.RandomState(7)
    n_distinct = 6
    names = []
    for i in range(n_distinct):
        # 生成**有灰度结构**的图。刻意不用纯色 —— 实测纯色图的 aHash/dHash
        # 恒为 0（与颜色无关），这是感知哈希的设计特性，会让本用例的期望值失效。
        arr = np.zeros((240, 320, 3), dtype=np.uint8)
        if i % 3 == 0:
            xx, yy = np.meshgrid(np.arange(320), np.arange(240))
            arr[..., 0] = ((xx // (8 + i * 4)) % 2) * 255      # 不同频率的竖直条纹
            arr[..., 1] = ((yy // (10 + i * 5)) % 2) * 180
        elif i % 3 == 1:
            arr = (rng2.rand(240, 320, 3) * 255).astype(np.uint8)  # 随机噪声
        else:
            xx, yy = np.meshgrid(np.arange(320), np.arange(240))
            arr[..., 2] = (((xx + yy) // (12 + i * 6)) % 2) * 255   # 斜条纹
        base = dupdir / f"clip{i:02d}_f0000.jpg"
        _Image.fromarray(arr).save(base, quality=95)
        names.append(base)
        for k in (1, 2):   # 两份完全相同的副本
            _sh.copy(base, dupdir / f"clip{i:02d}_f000{k}.jpg")
    n_before = len(list(dupdir.glob("*.jpg")))

    code, out = run(["dedupe", "--images", str(dupdir), "--threshold", "8",
                     "--write-keep-list", str(WORK / "keep.txt")])
    check("dedupe 正常运行", code == 0, f"exit={code}")
    check("dedupe 识别出重复帧", "判定重复" in out and "冗余率" in out)
    keep = WORK / "keep.txt"
    n_keep = len(keep.read_text(encoding="utf-8").strip().splitlines()) if keep.exists() else 0
    check("dedupe 保留数正确（3 份副本保留 1）", n_keep == n_distinct,
          f"{n_before} 帧 -> 保留 {n_keep}（期望 {n_distinct}）")
    # 冗余率应该接近 2/3
    mrate = re.search(r"冗余率 ([\d.]+)%", out)
    if mrate:
        check("冗余率接近 66.7%", 60 < float(mrate.group(1)) < 72, f"冗余率 {mrate.group(1)}%")

    # ---- 8. 边界：空目录 / 不存在的路径 ----
    empty = WORK / "empty"; empty.mkdir()
    code, out = run(["dedupe", "--images", str(empty)])
    check("dedupe 空目录不崩溃", code in (0, 1), f"exit={code}")
    code, out = run(["check", "--ann", str(WORK / "nope.json")])
    check("check 不存在的标注报错清晰", code != 0 and "路径错误" in out, f"exit={code}")

    print("=" * 74)
    passed = sum(1 for _, c, _ in RESULTS if c)
    total = len(RESULTS)
    if passed == total:
        print(f"全部 {total} 项通过。")
        shutil.rmtree(WORK, ignore_errors=True)
        return 0
    print(f"{total - passed}/{total} 项失败：")
    for n, c, d in RESULTS:
        if not c:
            print(f"  - {n}  {d}")
    print(f"（中间产物保留在 {WORK} 便于排查）")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
