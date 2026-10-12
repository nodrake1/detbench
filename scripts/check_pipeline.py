"""开发期自检脚本：验证指标单调性、分层评测与结果汇总。

与 ``tools/main.py selftest`` 的区别：
- ``selftest`` 用构造好的小型场景验证指标实现的**正确性**（有已知答案）。
- 本脚本在**占位数据集**上验证**行为合理性**（单调性、分组是否出得来数）。

用法::

    python scripts/check_pipeline.py
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT.parent) not in sys.path:
    sys.path.insert(0, str(ROOT.parent))

from detbench.core.eval import build_size_decomposition, evaluate_stratified  # noqa: E402
from detbench.core.io import load_coco  # noqa: E402
from detbench.core.metrics import GroupSpec  # noqa: E402
from detbench.models.dummy import DummyAdapter  # noqa: E402
from detbench.core.deps import force_utf8_output

DATA = ROOT / "data" / "demo"


def _ensure_demo_dataset(ann: Path) -> bool:
    """占位数据集不存在时自动生成。

    没有这一步，本脚本在全新 checkout 上必然失败 —— ``data/`` 被 .gitignore
    排除，CI 上拿不到数据集。让检查脚本自己准备前置数据，
    比在每个调用点都记得先跑 make-demo 更可靠。
    """
    if ann.exists():
        return True
    print("占位数据集不存在，正在自动生成……")
    try:
        from detbench.scripts.make_demo_dataset import generate

        generate(DATA, num_clips=12, frames_per_clip=16)
        print()
        return ann.exists()
    except Exception as exc:
        print(f"自动生成失败：{type(exc).__name__}: {exc}")
        print("请手动运行：python tools/main.py make-demo")
        return False


def main() -> int:
    force_utf8_output()
    ann = DATA / "annotations" / "instances_test.json"
    if not _ensure_demo_dataset(ann):
        return 1

    ds = load_coco(
        ann,
        str(DATA / "images"),
        name="demo-test",
        attr_file=str(DATA / "metadata" / "frames.csv"),
    )
    s = ds.summary()
    print(f"测试集：{s['num_images']} 图 / {s['num_objects']} 目标 / "
          f"平均 {s['objects_per_image']} 目标每图")
    print(f"目标等效边长分位：{s['box_side_px_percentile']}")
    print()

    # ---- 1. 指标单调性：perfect > jitter > random ----
    from detbench.core.metrics import FINE_AREA_BINS

    def _side_buckets(n: int) -> str:
        return "  ".join(f"{name:>16s}" for name, _, _ in FINE_AREA_BINS)

    print(f"{'模式':10s} {'mAP@[.5:.95]':>12s} {'mAP@0.5':>9s}   {_side_buckets(0)}")
    scores = {}
    for mode in ("perfect", "jitter", "random", "empty"):
        preds = {sm.image_id: DummyAdapter(mode=mode, seed=0).predict(sm, "") for sm in ds.samples}
        m = build_size_decomposition(ds, preds)
        scores[mode] = m
        buckets = "  ".join(
            f"{m.get(f'mAP/{n}', float('nan')):>16.3f}" for n, _, _ in FINE_AREA_BINS
        )
        print(f"{mode:10s} {m['mAP']:>12.4f} {m['mAP50']:>9.4f}   {buckets}")
    print()

    ok = True
    if not (scores["perfect"]["mAP"] > scores["jitter"]["mAP"] > scores["random"]["mAP"]):
        print(
            "[FAIL] 单调性：应满足 perfect > jitter > random，实际 "
            f"{scores['perfect']['mAP']:.4f} / {scores['jitter']['mAP']:.4f} / "
            f"{scores['random']['mAP']:.4f}"
        )
        ok = False
    else:
        print("[PASS] 单调性：perfect > jitter > random")
    if abs(scores["empty"]["mAP"]) > 1e-9:
        print("[FAIL] empty 模式的 mAP 应为 0")
        ok = False
    else:
        print("[PASS] empty 模式 mAP = 0")

    # jitter 模式必须落在中间档：说明指标对"预测质量下降"是敏感的，
    # 而不是只会输出 0 或 1 这种退化解。
    j = scores["jitter"]["mAP"]
    if 0.05 < j < 0.98:
        print(f"[PASS] jitter 模式落在中间档（mAP={j:.4f}），指标对质量变化敏感")
    else:
        print(f"[FAIL] jitter 模式 mAP={j:.4f}，未落在中间档，指标可能退化了")
        ok = False
    print()

    # ---- 2. 分层评测：每个维度都要出得来数 ----
    preds = {sm.image_id: DummyAdapter(mode="perfect", seed=0).predict(sm, "") for sm in ds.samples}
    specs = [
        GroupSpec("distance_bin", min_gt=2),
        GroupSpec("blur", min_gt=2),
        GroupSpec("lighting", min_gt=2),
        GroupSpec("__size__", min_gt=2),
    ]
    strat = evaluate_stratified(ds, preds, specs)
    for key, groups in strat.items():
        detail = ", ".join(f"{g}={v['mAP']:.3f}(n={int(v['num_gt'])})" for g, v in sorted(groups.items()))
        print(f"  {key:14s}: {detail}")
        if not groups:
            print(f"[FAIL] 分组 {key} 没有任何输出")
            ok = False
    print()

    if ok:
        print("管线自检通过：指标行为合理，分层评测可用。")
        return 0
    print("管线自检未通过。")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
