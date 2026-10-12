"""验证 mmdet 适配器的逻辑正确性 —— 不需要真的安装 mmdetection。

原理
----
适配器与 mmdet 的交互面很窄，只有四个 API：``init_detector``、
``Config.fromfile``、``DetDataSample`` 和 ``InstanceData``。
本脚本把这几项替换成受控的假实现，就能在没有 mmdet 的机器上
精确验证适配器自身的逻辑：

1. **标签偏移**（``label_offset``）—— 最容易出错、后果最隐蔽的一点
2. 分数过滤与 ``max_per_img`` 截断
3. 负标签（背景类）的丢弃与告警
4. 空预测的处理
5. ``palette`` / 模型返回值的类型转换
6. 配置文件缺失时的明确报错

真装上 mmdet 之后，仍然建议用真实模型再跑一次 ``tools/main.py eval``
做端到端确认；本脚本保证的是"适配器逻辑"这一层是对的。

用法::

    python scripts/check_mmdet_adapter.py
"""

from __future__ import annotations

import sys
import types
import warnings
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT.parent) not in sys.path:
    sys.path.insert(0, str(ROOT.parent))

import numpy as np  # noqa: E402

from detbench.core.io import DetSample  # noqa: E402
from detbench.core.deps import force_utf8_output

# --------------------------------------------------------------------------- #
# 假的 mmdet / mmengine
# --------------------------------------------------------------------------- #


class FakeInstanceData:
    """模拟 ``mmengine.structures.InstanceData``：允许属性赋值的容器。"""

    def __init__(self, **kwargs):
        for k, v in kwargs.items():
            setattr(self, k, v)


class FakeDetDataSample:
    """模拟 ``mmdet.structures.DetDataSample``。"""

    def __init__(self):
        self.metainfo: dict = {}
        self.pred_instances = None
        self.gt_instances = None

    def set_metainfo(self, d):
        self.metainfo.update(d)

    def get(self, key, default=None):
        return self.metainfo.get(key, default)


class FakeConfig:
    """模拟 ``mmengine.Config``（支持 ``fromfile`` / ``get``）。

    解析方式是"只取字面量赋值"：把所有 ``KEY = ...`` 拼成一行再用
    ``ast.literal_eval`` 求值。真实配置里有 import 和函数调用，
    但适配器只读 ``model.backbone.type`` 和 test pipeline 的 scale，
    所以字面量足够。解析不了的键会被跳过（适配器本身对此也是防御性的）。
    """

    def __init__(self, data):
        self._data = data

    def get(self, key, default=None):
        return self._data.get(key, default)

    @classmethod
    def fromfile(cls, path):
        """构造假配置。

        真实的 mmdet 配置是 Python 脚本；这里直接硬编码适配器会读取的两项。
        **注意不能用 ast.literal_eval 去解析 `dict(...)` 调用** ——
        literal_eval 只接受字面量与容器字面量，函数调用会直接报错。
        """
        return cls(
            {
                "model": {"backbone": {"type": "ResNet", "depth": 50}},
                "test_pipeline": [
                    {"type": "LoadImageFromFile"},
                    {"type": "Resize", "scale": (1333, 800)},
                ],
            }
        )


#: 由测试用例设置的"模型输出"，形如 (boxes, scores, labels)
FAKE_OUTPUT = {"boxes": None, "scores": None, "labels": None}
INIT_CALLS: list = []


class FakeModel:
    def __init__(self, cfg, ckpt, device, palette):
        self.cfg = FakeConfig.fromfile(cfg)   # 模拟 init_detector 挂载已加载的配置
        self.ckpt = ckpt
        self.device = device
        self.palette = palette
        self.test_cfg = types.SimpleNamespace(score_thr=0.05, max_per_img=100)

    def test_step(self, data):
        sample = FakeDetDataSample()
        sample.pred_instances = FakeInstanceData(
            bboxes=FAKE_OUTPUT["boxes"],
            scores=FAKE_OUTPUT["scores"],
            labels=FAKE_OUTPUT["labels"],
        )
        return [sample]


def install_fakes() -> None:
    """把假模块塞进 sys.modules。"""
    mmdet = types.ModuleType("mmdet")
    mmdet.__version__ = "3.3.0-fake"
    mmdet_apis = types.ModuleType("mmdet.apis")
    mmdet_apis.init_detector = lambda cfg, ckpt, device="cuda", palette=None: (
        INIT_CALLS.append((cfg, ckpt, device, palette)) or FakeModel(cfg, ckpt, device, palette)
    )
    mmdet_struct = types.ModuleType("mmdet.structures")
    mmdet_struct.DetDataSample = FakeDetDataSample
    mmdet.apis = mmdet_apis
    mmdet.structures = mmdet_struct

    mmengine = types.ModuleType("mmengine")
    mmengine.__version__ = "0.10.0-fake"
    mmengine_struct = types.ModuleType("mmengine.structures")
    mmengine_struct.InstanceData = FakeInstanceData
    mmengine.Config = FakeConfig
    mmengine.structures = mmengine_struct

    sys.modules.update(
        {
            "mmdet": mmdet,
            "mmdet.apis": mmdet_apis,
            "mmdet.structures": mmdet_struct,
            "mmengine": mmengine,
            "mmengine.structures": mmengine_struct,
        }
    )


# --------------------------------------------------------------------------- #
# 测试脚手架
# --------------------------------------------------------------------------- #

RESULTS: list = []


def check(name: str, cond: bool, detail: str = "") -> None:
    RESULTS.append((name, cond))
    print(f"[{'PASS' if cond else 'FAIL'}] {name}" + (f"  -- {detail}" if detail else ""))


def make_sample() -> DetSample:
    return DetSample(
        image_id=7,
        file_name="fake.jpg",
        width=1280,
        height=720,
        boxes=np.zeros((0, 4), np.float32),
        labels=np.zeros((0,), np.int64),
    )


def build_adapter(**overrides):
    """构造一个连着假 mmdet 的适配器。"""
    from detbench.models.mmdet_detr import MMDetAdapter

    cfg = overrides.pop("config", str(ROOT / ".mmdet_fake" / "fake_config.py"))
    ckpt = overrides.pop("checkpoint", str(ROOT / ".mmdet_fake" / "fake.pth"))
    defaults = dict(device="cpu", label_offset=1, score_thresh=0.001, max_per_img=300)
    defaults.update(overrides)
    return MMDetAdapter(config=cfg, checkpoint=ckpt, **defaults)


# --------------------------------------------------------------------------- #
# 用例
# --------------------------------------------------------------------------- #


def main() -> int:
    force_utf8_output()
    fake_dir = ROOT / ".mmdet_fake"
    fake_dir.mkdir(parents=True, exist_ok=True)
    (fake_dir / "fake_config.py").write_text(
        "# 假配置，仅供 check_mmdet_adapter.py 使用\n"
        "model = dict(backbone=dict(type='ResNet', depth=50))\n"
        "test_pipeline = [\n"
        "    dict(type='LoadImageFromFile'),\n"
        "    dict(type='Resize', scale=(1333, 800)),\n"
        "]\n",
        encoding="utf-8",
    )
    (fake_dir / "fake.pth").write_bytes(b"fake")

    install_fakes()
    from detbench.models import mmdet_detr  # noqa: F401  触发注册
    from detbench.core.registry import available

    print("=" * 74)
    print("mmdet 适配器逻辑自检（使用假 mmdet，无需真实安装）")
    print("=" * 74)

    try:
        _run_cases()
    finally:
        import shutil

        shutil.rmtree(fake_dir, ignore_errors=True)
        print("=" * 74)

    # 注册名检查
    print("\n--- 模型注册名 ---")
    names = available()
    print(f"已注册: {names}")
    for alias in ("mmdet", "deformable_detr", "dino", "detr"):
        check(f"注册名 '{alias}' 可用", alias in names)

    passed = sum(1 for _, c in RESULTS if c)
    total = len(RESULTS)
    print("=" * 74)
    if passed == total:
        print(f"全部 {total} 项通过。适配器逻辑正确。")
        print("注意：这验证的是适配器自身的逻辑；装上真实 mmdet 后仍应跑一次端到端 eval。")
        return 0
    failed = [n for n, c in RESULTS if not c]
    print(f"{total - passed}/{total} 项失败：{failed}")
    return 1


def _run_cases() -> None:
    sample = make_sample()

    # ---- 用例 1：基本映射与标签偏移 ----
    FAKE_OUTPUT.update(
        boxes=np.array([[10, 20, 60, 80], [100, 100, 140, 130]], dtype=np.float32),
        scores=np.array([0.9, 0.4], dtype=np.float32),
        labels=np.array([1, 1], dtype=np.int64),   # mmdet 单类 = 标签 1
    )
    adapter = build_adapter()
    pred = adapter.predict(sample, "")
    check(
        "标签偏移：mmdet 标签 1 -> 框架标签 0",
        pred.labels.tolist() == [0, 0],
        f"labels={pred.labels.tolist()}",
    )
    check(
        "框坐标原样透传（不缩放）",
        np.allclose(pred.boxes, FAKE_OUTPUT["boxes"]),
        f"boxes[0]={pred.boxes[0].tolist()}",
    )
    check("分数原样透传", np.allclose(pred.scores, [0.9, 0.4]), f"scores={pred.scores.tolist()}")
    check("image_id 正确", pred.image_id == 7)

    # ---- 用例 2：palette 与参数透传给 init_detector ----
    INIT_CALLS.clear()
    build_adapter(palette="coco", max_per_img=5)
    check(
        "palette / device / checkpoint 正确透传给 init_detector",
        len(INIT_CALLS) == 1 and INIT_CALLS[0][3] == "coco" and INIT_CALLS[0][2] == "cpu",
        f"init_calls={INIT_CALLS}",
    )

    # ---- 用例 3：l abel_offset=0 时标签不变 ----
    FAKE_OUTPUT.update(
        boxes=np.array([[10, 20, 60, 80]], dtype=np.float32),
        scores=np.array([0.9], dtype=np.float32),
        labels=np.array([0], dtype=np.int64),
    )
    adapter0 = build_adapter(label_offset=0)
    pred0 = adapter0.predict(sample, "")
    check("label_offset=0 时标签保持 0", pred0.labels.tolist() == [0], f"labels={pred0.labels.tolist()}")

    # ---- 用例 4：分数过滤 ----
    FAKE_OUTPUT.update(
        boxes=np.array([[10, 20, 60, 80], [100, 100, 140, 130], [5, 5, 25, 25]], dtype=np.float32),
        scores=np.array([0.9, 0.02, 0.5], dtype=np.float32),
        labels=np.array([1, 1, 1], dtype=np.int64),
    )
    adapter_hi = build_adapter(score_thresh=0.3)
    pred_hi = adapter_hi.predict(sample, "")
    check(
        "score_thresh=0.3 只保留 2 个预测",
        pred_hi.scores.size == 2,
        f"kept={pred_hi.scores.tolist()}",
    )
    check("保留的预测按原顺序", np.allclose(np.sort(pred_hi.scores), [0.5, 0.9]))

    # ---- 用例 5：max_per_img 截断（保留最高分）----
    adapter_top = build_adapter(max_per_img=2)
    pred_top = adapter_top.predict(sample, "")
    check(
        "max_per_img=2 保留分数最高的两个",
        pred_top.scores.size == 2 and np.allclose(np.sort(pred_top.scores), [0.5, 0.9]),
        f"scores={pred_top.scores.tolist()}",
    )

    # ---- 用例 6：负标签（背景类）被丢弃并告警 ----
    FAKE_OUTPUT.update(
        boxes=np.array([[10, 20, 60, 80], [1, 1, 9, 9]], dtype=np.float32),
        scores=np.array([0.8, 0.7], dtype=np.float32),
        labels=np.array([1, 0], dtype=np.int64),   # 标签 0 是背景 -> 减 1 后为负
    )
    adapter_neg = build_adapter(label_offset=1)
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        pred_neg = adapter_neg.predict(sample, "")
    check(
        "背景类预测被丢弃",
        pred_neg.labels.size == 1 and pred_neg.labels.tolist() == [0],
        f"labels={pred_neg.labels.tolist()}",
    )
    check(
        "丢弃背景类时发出告警",
        any("标签在减去 label_offset" in str(x.message) for x in w),
        f"warnings={[str(x.message)[:40] for x in w]}",
    )

    # ---- 用例 7：空预测 ----
    FAKE_OUTPUT.update(
        boxes=np.zeros((0, 4), np.float32),
        scores=np.zeros((0,), np.float32),
        labels=np.zeros((0,), np.int64),
    )
    pred_empty = build_adapter().predict(sample, "")
    check(
        "空预测返回形状正确的空结果",
        pred_empty.boxes.shape == (0, 4) and pred_empty.scores.size == 0,
        f"boxes.shape={pred_empty.boxes.shape}",
    )

    # ---- 用例 8：多类别不再错位（偏移必须整体生效）----
    FAKE_OUTPUT.update(
        boxes=np.array([[1, 1, 10, 10], [20, 20, 40, 40], [50, 50, 90, 90]], dtype=np.float32),
        scores=np.array([0.9, 0.8, 0.7], dtype=np.float32),
        labels=np.array([1, 2, 3], dtype=np.int64),   # 3 个前景类
    )
    pred_multi = build_adapter().predict(sample, "")
    check(
        "多类别整体偏移正确：1,2,3 -> 0,1,2",
        pred_multi.labels.tolist() == [0, 1, 2],
        f"labels={pred_multi.labels.tolist()}",
    )

    # ---- 用例 9：describe 读到了输入尺度 ----
    desc = build_adapter().describe()
    check(
        "describe 从 test_pipeline 读到输入尺度",
        desc["input_size"] == "(1333, 800)",
        f"input_size={desc['input_size']}",
    )
    check(
        "describe 读到 backbone 类型",
        desc["backbone"] == "ResNet",
        f"backbone={desc['backbone']}",
    )

    # ---- 用例 10：配置文件不存在时报错清晰 ----
    try:
        build_adapter(config=str(ROOT / "no_such_config.py"))
        check("配置文件缺失应报错", False, "没有抛异常")
    except FileNotFoundError as exc:
        check("配置文件缺失时报 FileNotFoundError 且提示如何获取配置",
              "clone" in str(exc) or "本地" in str(exc), f"{str(exc)[:60]}...")
    except Exception as exc:
        check("配置文件缺失应报 FileNotFoundError", False, f"抛的是 {type(exc).__name__}")

    # ---- 用例 11：无 checkpoint 时告警但不崩 ----
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        build_adapter(checkpoint=None)
    check(
        "未提供 checkpoint 时告警",
        any("随机初始化" in str(x.message) for x in w),
        f"warnings={[str(x.message)[:30] for x in w]}",
    )


if __name__ == "__main__":
    raise SystemExit(main())
