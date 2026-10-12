# DetBench

面向**小目标 / 对空目标检测**的可复现评测框架。

它解决的问题不是"再实现一个检测器"，而是三件让你的实验结果站得住脚的事：

1. **统一预处理** —— 所有模型走同一套 letterbox 与归一化，绝不让各模型自己处理。
2. **统一评测** —— 所有模型走同一个评测脚本、同一套指标定义。
3. **全部落盘** —— 每次实验自动记录 git commit / 配置 / 随机种子 / 环境版本，
   三个月后你还能知道某个数字是怎么来的。

外加一个常规框架不提供、但对写论文最关键的模块：**分层评测**。

---

## 为什么需要分层评测

标准 mAP 只给你一个数：

```
mAP@[.5:.95] = 0.741
```

但研究结论藏在条件分布里：

```
目标等效边长 < 8px 时 mAP = 0.069，8~16px 时 0.312，>32px 时 0.823
重模糊帧上 YOLO 领先 DETR 0.04，清晰帧上反而落后 0.02
远距离档检出率 41%，近距离档 96%
```

**"在什么条件下谁更好"才是论文的贡献**，而普通评测脚本算不出来。
本框架把"按属性切片后再算 AP"做成了一等公民。

---

## 安装

```bash
# 核心依赖（很少）
pip install numpy pillow

# 画论文配图
pip install matplotlib

# 真实模型（按需）
pip install torch torchvision
pip install ultralytics
```

配置解析**不需要 PyYAML**：`core/config.py` 内置了一个覆盖常用子集的零依赖
YAML 解析器（装了 PyYAML 会自动优先使用）。

先跑体检确认环境：

```bash
python tools/main.py doctor
```

---

## 5 分钟跑通全链路

不需要任何真实数据、任何模型、任何 GPU。占位数据集是自动生成的。

```bash
cd detbench

# 1. 生成占位数据集（12 段 clip x 16 帧，含距离/模糊/光照属性）
python tools/main.py make-demo --out data/demo

# 2. 自检：验证指标实现是否正确（用有已知答案的场景）
python tools/main.py selftest

# 3. 评测（占位检测器）
python tools/main.py eval --config configs/example.yaml

# 4. 汇总成论文表格
python tools/main.py report --runs runs --out report.md --latex
```

第 2 步的 `selftest` 是整个框架的安全网。它会验证：

- IoU 几何计算正确
- `perfect` 模式 mAP ≈ 1.0（指标上限）
- `empty` 模式 mAP = 0.0（指标下限）
- 尺寸分桶落在正确的桶里
- 重复预测只算一次 TP（贪心匹配的核心行为）
- 目标级属性能把同一张图拆到不同的分组
- 结果能落盘、能回读、能生成 Markdown / LaTeX / 分层表

**每次改完评测代码都跑一遍。** 指标实现的 bug 通常只表现为"mAP 偏低"，
不报错、不崩溃，光靠肉眼极难发现。

---

## 目录结构

```
detbench/
├── core/
│   ├── io.py            统一数据接口：COCO / YOLO / VOC + 属性表
│   ├── metrics.py       ★ 分层 AP 计算（不依赖 pycocotools）
│   ├── eval.py          评测主循环：mAP + 分层 mAP + 速度
│   ├── train_config.py  ★ 训练配置生成（mmdet 继承+覆盖 / YOLO data.yaml）
│   ├── speed.py         FPS / 延迟 / 峰值显存测量
│   ├── results.py       实验记录 + 一键生成论文表
│   ├── registry.py      模型适配器注册表
│   ├── config.py        配置解析（内置零依赖 YAML）
│   ├── deps.py          可选依赖探测
│   └── viz.py           论文配图：尺度曲线 / 条件柱状图 / PR / 失败案例
├── models/              各检测库的适配器
│   ├── dummy.py         占位检测器（自检与管线验证用）
│   ├── mmdet_detr.py    ★ mmdet DETR 全家族 + 任意 mmdet 检测器
│   ├── yolov8.py        ultralytics YOLO（v5/v8/v11）
│   └── torchvision_det.py   Faster R-CNN / RetinaNet / FCOS
├── tools/
│   ├── main.py          CLI：doctor / selftest / make-demo / eval / report
│   └── train.py         ★ CLI：训练（mmdet / ultralytics）+ 训练后自动评测
├── scripts/
│   ├── bootstrap.py           新设备一键引导
│   ├── make_demo_dataset.py   生成占位数据集
│   ├── check_pipeline.py      行为合理性检查（单调性等）
│   └── check_mmdet_adapter.py DETR 适配器逻辑自检（不需要装 mmdet）
├── configs/
│   ├── example.yaml           评测配置示例
│   └── train_example.yaml     ★ 训练配置示例
└── requirements.txt
```

---

## 训练

`tools/train.py` 支持两条后端，**训练结束后自动评测并写出 `result.json`**，
所以"训练 → 评测 → 结果落盘"是一条命令完成的，不会出现"训完了忘了评测"。

| 后端 | 覆盖模型 | 依赖 |
|---|---|---|
| `mmdet` | Deformable DETR / DINO-DETR / DAB-DETR / 任意 mmdet 检测器 | `mmengine mmcv mmdet` + clone mmdetection |
| `ultralytics` | YOLO 全系 | `ultralytics` |

### 1. 准备

```bash
# mmdet 后端
pip install torch torchvision mmengine mmcv mmdet
git clone https://github.com/open-mmlab/mmdetection third_party/mmdetection

# 或 YOLO 后端（依赖少、速度快，适合先跑通）
pip install torch torchvision ultralytics
```

> ⚠️ **如果你的环境里已经有 torch**，装新包时请带上约束文件锁住 numpy 大版本：
>
> ```bash
> pip install -c constraints.txt ultralytics
> ```
>
> 原因见 `constraints.txt` 的注释：`opencv-python` 4.8 / `matplotlib` 3.7 /
> `pandas` 2.0 都只支持 numpy 1.x。pip 装新包时若顺手把 numpy 升到 2.x，
> 这三个包会立刻坏掉，而且**报错五花八门、很难定位到 numpy** ——
> 例如 `torch.from_numpy` 抛 `expected np.ndarray (got numpy.ndarray)`，
> 或 `numpy.core.multiarray` 缺 `number` 属性。
> 更隐蔽的一种坏法是 1.x 与 2.x 的文件混在一起（覆盖安装没清干净），
> 表现为随机崩溃。用 `-c constraints.txt` 可以从源头避免。

### 2. 改配置

复制 `configs/train_example.yaml`，主要改这几项：

```yaml
backend: mmdet
data_root: ..                    # 换设备时用 --data-root 覆盖，不用改这里
dataset:
  class_names: [helicopter]      # ⚠️ 必须显式给，哪怕只有一个类
  batch_size: 2                  # 每张卡
  splits:                        # ann_file / img_prefix 都相对 data_root
    train: {ann_file: annotations/instances_train.json, img_prefix: images/train}
    val:   {ann_file: annotations/instances_val.json,   img_prefix: images/val}
    test:  {ann_file: annotations/instances_test.json,  img_prefix: images/test}
model:
  base_config: third_party/mmdetection/configs/dino/dino-4scale_r50_8xb2-12e_coco.py
  frozen_stages: 1               # 冻结 backbone，省 ~30% 显存与时间
  max_per_img: 300               # 必须与评测的 maxDets 一致
  label_offset: 1                # mmdet 类别 0 是背景保留位
training:
  epochs: 12
  grad_accum: 4                  # 等效 batch = 2 × 1卡 × 4 = 8
  amp: true
```

### 3. 训练

```bash
# 0) 先看合并后的配置对不对（30 秒，强烈建议先跑）
python tools/train.py --config configs/train_example.yaml --dump-config

# 1) 估耗时
python tools/train.py --config configs/train_example.yaml --estimate-only

# 2) 单卡训练
python tools/train.py --config configs/train_example.yaml

# 3) 双卡 DDP（两台机器各自跑一条，master_addr 填 rank0 的 IP）
torchrun --nproc_per_node=1 --nnodes=2 --node_rank=0 \
         --master_addr=<主节点IP> --master_port=29500 \
         tools/train.py --config configs/train_example.yaml
```

### 3b. 先跑冒烟配置验证链路（强烈建议）

```bash
python tools/main.py make-demo --out data/demo
python tools/train.py --config configs/train_smoke.yaml
```

`train_smoke.yaml` 用 **YOLO11n 从零初始化**（不下载预训练权重）、320px、2 轮，
几十秒跑完，用来确认"训练 → 自动评测 → `result.json`"整条链路是通的。
**它不是用来出结果的。**

实测数据（RTX 4060 Ti，50 轮约 2 分钟）：

```
最终评测：mAP@[.5:.95] = 0.2080    mAP@0.5 = 0.5112    57.5 FPS / 128 MB

分层结果：
  距离档：near 0.459  >  mid 0.153  >  far 0.025
  模糊档：light 0.459 >  heavy 0.084
  尺寸桶：medium 0.429 > small 0.354 > tiny 0.053 > nano 0.000
```

**三个维度的排序完全符合物理直觉**（越近/越清晰/越大越准）——
这既是"训练有效"的证据，也证明分层评测本身是自洽的。

### 4. 训练配置的三个隐蔽坑

`tools/train.py` 帮你在开训前就把这些查出来：

**① `class_names` 必须显式给。** mmdet 靠 `metainfo.classes` 决定类别语义。
单类检测时 mmdet 的 `num_classes` 是 2（0 是背景），模型预测标签是 `1`，
而 DetBench 评测端统一 0-based，靠 `label_offset: 1` 对齐。
**单类场景下 `labels - 1` 恰好把 1 变成 0，看起来完全正确**；
但类别数 ≥ 2 时所有类别会整体错位一位，mAP 莫名暴跌且不报错。

**② 改了 `max_epochs` 必须同步改参数调度器。** 官方配置的学习率调度是按它
自己的轮数写的（DINO 12e、Deformable DETR 50e）。如果只改 `epochs` 不改调度器：
调度器比训练短 → 后期学习率一直是 0，模型停止学习；比训练长 → 学习率没衰减到位。
**两种情况都不报错，只是训得差。** 本框架会自动按比例重设 `milestones` 和 `end`。

**③ 等效 batch 太小会掉点。** DETR 系对 batch size 敏感。脚本会打印
`batch × 卡数 × 累积`，低于 8 时给警告。4060 Ti 16GB 上 DETR@800×1333
建议 `batch_size: 2` + `grad_accum: 4`。

### 5. 训练后

训练结束会自动：

1. 找到最优权重（`best_bbox_mAP*.pth` / `best.pt`）
2. 在 test split 上跑完整评测（含分层 mAP + 测速）
3. 写出 `result.json` 到 `<work_dir>/eval/`

然后照常汇总：

```bash
python tools/main.py report --runs runs --out report.md --latex
```

---

## 把真实数据集接进来

### 1. 数据格式

**COCO（推荐）**：

```python
from detbench.core.io import load_coco

ds = load_coco(
    "data/HeliAir/annotations/instances_test.json",
    "data/HeliAir/images",
    attr_file="data/HeliAir/metadata/frames.csv",   # 可选，分层评测需要
)
```

**YOLO**：

```python
from detbench.core.io import load_yolo

ds = load_yolo("data/images/test", "data/labels/test", class_names=["helicopter"])
```

**VOC**：

```python
from detbench.core.io import load_voc

ds = load_voc("data/Annotations", "data/JPEGImages", class_names=["helicopter"])
```

### 2. 属性表（分层评测的前提）

CSV，第一列是图像文件名，其余列是属性。列名以 `obj_` 开头表示**目标级属性**
（分号分隔），同一张图会被按目标拆到多个分组：

```csv
image_name,clip_id,distance_bin,blur,lighting,obj_blur
clip000_f0000.jpg,clip000,far,heavy,backlit,heavy;light
clip000_f0001.jpg,clip000,far,heavy,backlit,heavy
```

对应的采集元数据模板见 `scripts/make_demo_dataset.py` 生成的
`metadata/clips.csv` 与 `metadata/frames.csv`。

### 3. 划分数据（千万别按帧随机划分）

```python
from detbench.core.io import split_by_group

train, val, test = split_by_group(ds, group_key="clip_id", ratios=(0.7, 0.1, 0.2))
```

同一段视频的相邻帧高度相似。**按帧随机划分会造成 scene leakage**，
使测试集 mAP 虚高、泛化性虚假 —— 这是数据集类论文最常被审稿人挑出的问题之一。
必须按 `clip_id`（飞行架次）分组划分。

---

## 数据集工具

`tools/dataset.py` 覆盖做数据集时的五件事。这些不是"锦上添花"——
它们各自对应我在数据集指南里提过的一个具体坑。

### 1. 格式转换（COCO ↔ YOLO ↔ VOC）

```bash
python tools/dataset.py convert --from coco --to yolo \
    --ann data/HeliAir/annotations/instances_train.json \
    --out data/HeliAir_yolo

# 反向：YOLO -> COCO
python tools/dataset.py convert --from yolo --to coco \
    --images data/HeliAir/images \
    --labels data/HeliAir_yolo/labels \
    --class-names helicopter \
    --out data/HeliAir_coco
```

`coco → yolo` 会额外生成 `classes.txt` 和 `image_list.txt`，后者可以直接喂给
`rsync --files-from` 来拷图，不用手写脚本。

**已处理的坑**：COCO 的 `file_name` 常带 split 前缀（`train/xxx.jpg`），
而 YOLO 标签可能是平铺的也可能是同构嵌套的。反方向转换做了两级路径兜底
（同构路径 → 文件名索引），并在布局错配到会产出空结果时**明确警告**。
没有这个兜底，布局不一致时会静默产出 0 个标注 —— "转换成功"但结果全空，
是最难发现的一类错误。

### 2. 按架次分层划分

```bash
python tools/dataset.py split \
    --ann data/HeliAir/annotations/instances_all.json \
    --attr data/HeliAir/metadata/frames.csv \
    --group-key clip_id --stratify-by distance_bin \
    --out data/HeliAir/annotations
```

- 按 `clip_id`（飞行架次）分组，**同一架次绝不跨越划分** —— 防 scene leakage
- `--stratify-by` 按属性轮流分配，保证每个划分里距离档/光照档都有覆盖。
  纯随机划分可能让测试集只落到一个档位，分层评测直接失效
- 输出三个 COCO 文件 + 一份 `split_manifest.csv` 划分清单（便于核对与复现）

### 3. 质检

```bash
python tools/dataset.py check \
    --ann data/HeliAir/annotations/instances_all.json \
    --images data/HeliAir/images
```

检查项：图像文件是否存在、框宽高是否 ≤0、框是否越界、过小目标、
极端长宽比、每图目标数、空图占比、类别不均衡、**同类高度重合的重复标注**。
分「必须修」和「建议关注」两级，退出码可用于 CI。

### 4. 标注一致性（两人互评）⭐

```bash
python tools/dataset.py agreement \
    --ann-a ann_by_alice.json --ann-b ann_by_bob.json
```

把标注员 A 当真值、B 当预测算 mAP。**这是回应"标注质量如何保证"最有力的证据** ——
不留主观描述，直接给出可写进论文的数字。工具会直接打印成句：

```
  以 A 为真值、B 为预测：
    一致性 mAP@[.5:.95] = 0.703
    一致性 mAP@0.5      = 1.000

  逐目标最佳 IoU（A 的每个框对 B 的最佳匹配）:
    p50 = 0.850
    IoU > 0.5 的比例 = 91.3%

  论文里可以这样写：
    「随机抽取 192 帧由两位标注员独立标注，以一方为真值评测另一方，
      一致性 mAP@0.5 为 1.000，逐目标最佳 IoU 中位数为 0.850。」
```

两个文件用**文件名**对齐（两份标注的 `image_id` 往往不同），并会报告
目标数差异 —— 超过 10% 说明存在系统性漏标/多标。

### 5. 抽帧去冗余 ⭐

```bash
python tools/dataset.py dedupe \
    --images data/HeliAir/raw_frames \
    --threshold 8 \
    --move-to data/HeliAir/quarantine \
    --report data/HeliAir/duplicates.csv
```

把视频逐帧导出是最常见也最有害的做法：相邻帧几乎相同，导致训练集内部冗余、
以及 train/test 之间泄漏。这个命令用 aHash + dHash 检测近似重复帧：

```
  扫描 5400 个文件，哈希阈值 8（汉明距离）
  保留 412 帧 / 判定重复 4988 帧（冗余率 92.4%）
```

**已知局限**（写在 `--help` 里）：aHash/dHash 只看**灰度结构**，
这正是它能忽略曝光变化的原因，但也带来两个盲区：

- **纯色或近似平坦的图无法区分**。实测若干不同颜色的纯色图，aHash 与 dHash
  全都得到 0。这是设计使然。工具在发现大量图哈希相同时会明确警告，不会给你
  一个看起来正常其实无意义的结果。
- **只做全局比较**。目标移动但背景不变的帧全局哈希距离仍很小，会被判为重复 ——
  对"抽帧去冗余"这个目的是期望行为，但若要保留目标位置变化的样本，
  需调小阈值或先裁剪目标区域再哈希。

---

## 接入一个新模型

任何检测器只要实现两个方法就能接入统一评测。以自研模型为例：

```python
# models/my_detector.py
from detbench.core.io import DetPrediction, DetSample
from detbench.core.registry import DetectorAdapter, register


class MyDetectorAdapter(DetectorAdapter):
    name = "my_detector"

    def __init__(self, weights: str, imgsz: int = 1280, conf: float = 0.001,
                 device: str = "cuda", **kw):
        super().__init__(device=device, **kw)
        self.model = load_my_model(weights).to(device).eval()
        self.imgsz, self.conf = imgsz, conf

    def predict(self, sample: DetSample, image_root: str) -> DetPrediction:
        img = load_image(image_root, sample.file_name)
        out = self.model(img)
        boxes = out["boxes"]                      # ⚠️ 必须是 xyxy、原图坐标系
        scores = out["scores"]
        keep = scores >= self.conf
        return DetPrediction(sample.image_id, boxes[keep], scores[keep], out["labels"][keep])

    def describe(self) -> dict:
        return {"method": "MyDetector", "backbone": "R50",
                "input_size": str(self.imgsz), "epochs": 50, "seed": 0}

    @property
    def torch_model(self):
        return self.model


@register("my_detector")
def _build(device="cuda", **kw):
    return MyDetectorAdapter(device=device, **kw)
```

然后在配置里写 `model: {type: my_detector, weights: ...}` 即可。

### 内置适配器

| `model.type` | 覆盖的模型 |
|---|---|
| `dummy` | 占位检测器（`perfect` / `jitter` / `random` / `empty` 四种模式） |
| `yolov8` / `yolov11` / `yolov5` | ultralytics YOLO 全系 |
| `torchvision` / `faster_rcnn` / `retinanet` | Faster R-CNN / RetinaNet / FCOS |
| `mmdet` / `detr` / `deformable_detr` / `dino` / `dino_detr` / `dab_detr` / `dn_detr` / `rt_detr` / `mask2former` | **mmdetection 里的整条 DETR 家族**，以及任意其他 mmdet 检测器 |

### DETR 系（走 mmdetection）

原版 `facebookresearch/detr` 仓库基本停更，Deformable DETR / DINO-DETR /
DAB-DETR 的可用实现都在 mmdetection 里。用一个适配器覆盖整条家族：

```yaml
model:
  type: deformable_detr
  config: third_party/mmdetection/configs/deformable_detr/deformable-detr_r50_16xb2-50e_coco.py
  checkpoint: runs/deformable_detr/epoch_50.pth
  label_offset: 1        # ← 见下面警告
  score_thresh: 0.001
  max_per_img: 300
```

两个使用注意：

- **`config` 必须是本地 `.py` 路径**，适配器不会联网下载配置文件。
  Deformable DETR / DINO-DETR 的配置在 mmdetection 仓库的
  `configs/deformable_detr/` 与 `configs/dino/` 下，先 clone 仓库再指向它。
- **别用带 `_tta` 后缀的配置**。那会在推理时做多尺度增强，速度慢一个数量级，
  且和其他方法不可比 —— 直接违反"对比公平"原则。

#### ⚠️ 最容易出错的一点：类别标签偏移

**mmdetection 的 DETR 系模型，类别 0 是背景保留位，前景类别从 1 开始。**
所以"单类别直升机"训练出来，模型预测的标签是 **1**。
而本框架统一约定 **0-based 前景类别**（单类别就是 0）。

忘记这件事的后果极其隐蔽：

- 单类别数据上，`labels - 1` 恰好把 1 变成 0，**看起来是对的**；
- 一旦类别数 ≥ 2，所有类别整体错位一位，mAP 莫名暴跌，**而且不报任何错**。

所以 `label_offset` 是**显式参数**（默认 `1`），适配器初始化时会打印一行提示，
并且在减去 offset 后出现负标签时会告警并丢弃这些预测。
**如果你的模型本身是 0-based，必须显式设 `label_offset: 0`。**

适配器本身的逻辑（标签偏移、分数过滤、`max_per_img` 截断、空预测、
背景类丢弃、配置缺失报错等 21 项）有一套**不依赖真实安装**的验证脚本：

```bash
python scripts/check_mmdet_adapter.py
```

它用假的 mmdet 模块验证适配器逻辑。**装上真实 mmdet 后，仍应跑一次端到端
`eval` 做最终确认。**

---

## 换到另一台设备继续项目

整个项目源码只有 **约 200 KB / 26 个文件**，而 `data/` 和 `runs/` 都可以重新生成，
所以迁移本身很轻。真正容易出问题的只有一件事：**数据路径在新设备上对不上**。
框架已经把这件事处理掉了。

### 迁移清单：哪些要带走，哪些不用

| 内容 | 大小 | 是否要带走 | 说明 |
|---|---|---|---|
| `core/` `models/` `tools/` `scripts/` `configs/` | ~180 KB | ✅ **必须** | 源码，唯一的真正资产 |
| `README.md` `requirements.txt` `.gitignore` | ~20 KB | ✅ 必须 | |
| `runs/**/result.json` | 很小 | ⚠️ **建议** | 已完成实验的指标记录，丢了得重跑 |
| `runs/**/predictions.npz` | 中等 | ⚠️ 可选 | 换指标重算用；必要时可重跑推理 |
| `runs/**/*.pth` 权重 | 大 | ⚠️ 可选 | 能重训就不带；要带就单独拷 |
| `data/demo/` 占位数据集 | 35 MB | ❌ **不要** | `make-demo` 一条命令就能重建 |
| `__pycache__/` `.selftest_out/` | 小 | ❌ 不要 | 已在 `.gitignore` |

**核心结论：真正必须带走的只有约 200 KB 源码。**

### 在你现在的设备上（准备迁移）

```bash
cd detbench
python scripts/bootstrap.py --with-git      # 体检 + 冒烟测试 + 建 git 仓库并首次提交
```

然后按情况选一种打包方式：

```bash
# 方式 A：有远程仓库（最推荐，可后续持续同步）
git remote add origin <你的仓库地址>
git push -u origin main

# 方式 B：没有远程仓库，只打包源码（约 200 KB，可直接邮件发送）
git archive --format=zip -o detbench.zip HEAD

# 方式 C：连提交历史一起带走
git bundle create detbench.bundle --all

# 方式 D：直接拷目录，但必须排除 data/ 和 runs/
```

### 在新设备上（恢复项目）

```bash
git clone <仓库地址> detbench      # 或解压 detbench.zip
cd detbench

pip install numpy pillow           # 最少只要这两个
# 真要跑模型时再装：
pip install torch torchvision matplotlib
pip install mmengine mmcv mmdet    # DETR 全家族

python scripts/bootstrap.py        # 一键引导
```

`bootstrap.py` 做四件事，任何一步失败都会明确告诉你原因：

```
步骤 1/4  依赖体检   —— 缺什么、装什么命令
步骤 2/4  冒烟测试   —— 生成占位数据集，跑通 数据→评测→分层→落盘→论文表
步骤 3/4  路径自检   —— 打印解析后的绝对路径并逐项检查存在性
步骤 4/4  git 仓库   —— 状态确认
```

**冒烟测试是关键**：它证明新设备的完整链路真的能跑，
而不是"依赖装上了但跑不通"。

### 数据在新设备上的位置不一样，怎么办

**不要改配置文件。** 用 `--data-root` 覆盖：

```bash
# 旧设备：数据在 D:/datasets/HeliAir
# 新设备：数据在 /mnt/nas/HeliAir
python tools/main.py eval --config configs/exp01.yaml --data-root /mnt/nas/HeliAir
```

路径解析规则（刻意只保留一条，避免多条规则互相争夺）：

```
数据路径（ann_file / images_root / attr_file ...）
    相对路径 -> 拼到「有效数据根」下
    有效数据根 = --data-root  >  配置里的 data_root  >  项目根目录

权重路径（checkpoint / weights / model.config）
    相对路径 -> 相对「配置文件所在目录」
    因为权重通常和配置一起管理，而不是和数据集放一起
```

**数据路径不会相对 `configs/` 解析** —— 这是刻意的。如果同时支持两种基准，
它们会争夺同一批键，导致 `--data-root` 被静默忽略，那是最糟糕的失败模式。

其他不用改配置文件的做法：

```bash
# 环境变量（配置里写 data_root: ${DETBENCH_DATA:-.}）
export DETBENCH_DATA=/mnt/nas/HeliAir

# 临时覆盖单个键，可重复
python tools/main.py eval --config configs/exp01.yaml \
    --set runtime.device=cuda:1 --set eval.predict_conf=0.01

# 覆盖设备与实验名
python tools/main.py eval --config configs/exp01.yaml --device cuda:1 --exp-id exp01_rerun
```

### 到新设备后先跑 dry-run

这是最省时间的一个习惯：

```bash
python tools/main.py eval --config configs/exp01.yaml --dry-run
```

它打印**全部解析后的绝对路径**并逐项检查存在性，还会试加载一次数据集：

```
  配置文件  : /home/me/detbench/configs/exp01.yaml
  data_root : /mnt/nas/HeliAir
  dataset.ann_file    : /mnt/nas/HeliAir/annotations/instances_test.json
  dataset.images_root : /mnt/nas/HeliAir/images

  路径存在性检查：
    [OK ] ann_file       /mnt/nas/HeliAir/annotations/instances_test.json
    [缺失] attr_file      /mnt/nas/HeliAir/metadata/frames.csv

  数据集    : 48 图 / 52 目标 / 1 类 ['helicopter']
```

`dry-run` 不会因路径缺失就直接退出（否则你看不到解析结果）；
而正式评测会**在启动时立即检查并列出所有缺失路径**，附带排查建议。

### 迁移后的常见问题

| 症状 | 原因 | 解决 |
|---|---|---|
| `FileNotFoundError: 以下路径不存在` | 数据没拷过来或位置不同 | `--data-root` 指向新位置 |
| `ModuleNotFoundError: No module named 'detbench'` | 从错误目录调用 | 在项目根目录下运行 |
| 中文输出乱码 | Windows 终端编码 | PowerShell: `$env:PYTHONUTF8="1"` |
| 速度评测被跳过 | 没装 torch | `pip install torch`；精度指标不受影响 |
| 换到 Linux 后路径报错 | 配置里写了 Windows 绝对路径 | 改相对路径 + `--data-root` |
| 权重找不到 | 权重没拷或路径写死 | `--set model.checkpoint=<新路径>` |

**建议从一开始就这么做**：配置里所有路径都写**相对路径**，数据位置由
`data_root` 或 `--data-root` 决定。这样在两台设备之间切换时，配置文件一个字都不用改。

---

---

## 上传到 GitHub

`.gitignore` 已经配好了，`data/`、`runs/`、权重、缓存都不会被上传，
所以仓库只有 **约 210 KB / 27 个文件**，推送很快。

### 1. 先设置 git 身份（只需一次）

```bash
git config --global user.name  "你的名字"
git config --global user.email "你的邮箱@example.com"
```

> 邮箱建议用 GitHub 账号里的邮箱（或 GitHub 提供的
> `@users.noreply.github.com` 隐私邮箱），否则提交不会关联到你的账号。

### 2. 修正已有的提交署名

本项目的初始提交是由引导脚本用占位身份 `bootstrap <bootstrap@local>` 创建的。
上传前建议改掉，否则 GitHub 上显示的提交者不是你自己：

```bash
# 方式 A：一条命令改掉全部历史（推荐）
git filter-branch -f --env-filter '
export GIT_AUTHOR_NAME="你的名字"
export GIT_AUTHOR_EMAIL="你的邮箱@example.com"
export GIT_COMMITTER_NAME="你的名字"
export GIT_COMMITTER_EMAIL="你的邮箱@example.com"
' -- --all

# 方式 B：不装了，直接压成一次全新提交（最干净）
rm -rf .git
git init -b main
git add -A
git commit -m "DetBench: 小目标检测可复现评测框架"
```

方式 B 会丢掉之前两次提交的历史，但对一个新项目来说完全无所谓，
而且署名一定正确。

### 3. 在 GitHub 上建仓库

打开 <https://github.com/new>：

- **Repository name**：`detbench`（或你喜欢的名字）
- **Description**：`面向小目标检测的可复现评测框架（分层 mAP / 统一评测 / 论文表格自动生成）`
- **Public / Private**：自选。如果打算在论文里提供代码链接，选 Public
- ⚠️ **不要勾选** "Add a README file"、"Add .gitignore"、"Choose a license"
  —— 本地已经有了，勾了会在推送时冲突

### 4. 关联远程仓库并推送

```bash
git remote add origin https://github.com/<你的用户名>/detbench.git
git branch -M main
git push -u origin main
```

**认证说明**：GitHub 从 2021 年起不再接受账号密码。

- **Windows**：`credential.helper = manager`（本机已配置）会在首次推送时
  自动弹出浏览器窗口让你登录 GitHub，登录一次之后就免密了。**这是最省事的方式。**
- **Linux / 手动**：需要 Personal Access Token。
  到 <https://github.com/settings/tokens> 生成一个带 `repo` 权限的
  Fine-grained 或 Classic token，推送时把它当密码填入。
- **SSH（可选，更省心）**：
  ```bash
  ssh-keygen -t ed25519 -C "你的邮箱@example.com"
  # 把 ~/.ssh/id_ed25519.pub 的内容贴到 https://github.com/settings/keys
  git remote set-url origin git@github.com:<你的用户名>/detbench.git
  ```

### 5. 以后更新

```bash
git add -A
git commit -m "说明这次改了什么"
git push
```

### 网络不通怎么办

如果 `git push` 卡住或报连接错误，通常是网络问题（GitHub 在国内访问不稳定）。
三个办法：

```bash
# 1. 给 git 单独挂代理（端口按你自己的代理软件改）
git config --global http.proxy  http://127.0.0.1:7890
git config --global https.proxy http://127.0.0.1:7890
# 取消：
git config --global --unset http.proxy
git config --global --unset https.proxy

# 2. 用 SSH 走 443 端口（能绕过部分限制）
#    ~/.ssh/config 里加：
#      Host github.com
#        Hostname ssh.github.com
#        Port 443
#        User git

# 3. 改用 Gitee（国内）或校内 GitLab，命令完全一样，只换仓库地址
```

### 上传前的自查清单

```bash
git status                     # 应该是干净的
git ls-files | wc -l           # 应约 27 个文件
git ls-files | grep -E "\.pth|\.pt$|\.onnx"   # 应该没有输出（权重没被跟踪）
du -sh .git                    # 应该只有几百 KB
```

如果 `du` 显示 `.git` 有几十 MB，说明之前误提交过大文件。
用 `git rm --cached <文件>` 移除后提交即可（文件本身不会被删）。

---

## 三个必须记住的评测陷阱

### ⚠️ 1. 算 mAP 时置信度阈值必须设到极低

```python
conf = 0.001     # ✅
conf = 0.25      # ❌ 默认值，会砍掉大量低分预测，precision 虚高
```

评测阶段让所有预测都参与，由 `max_dets` 做 top-k 截断，保持各方法可比。
部署用的实际阈值是另一回事，**不要混用**。

### ⚠️ 2. 忘记坐标反变换

如果模型内部做了 letterbox / resize，**必须把框映射回原图尺寸**。
这是最常见的 bug，而且不会报错 —— 指标只是"偏低"。

`models/yolov8.py` 和 `models/torchvision_det.py` 里都标注了各自的处理方式。
自己写 ONNX / TensorRT 推理管线时必须手动反变换。

### ⚠️ 3. 对比不公平

所有方法必须使用**同一输入分辨率、同一套增强、同一 epoch 预算、同一个评测脚本**。
本框架通过 `DetectorAdapter` 强制统一了推理接口，但训练侧的一致性仍然靠你自律。

---

## 分层评测的两种协议

本框架采用**全局评测协议**：把不符合条件的 GT 从该组剔除，
但保留模型在**整张图**上的全部预测。

```python
from detbench.core.metrics import GroupSpec, FINE_AREA_BINS

specs = [
    GroupSpec("distance_bin", min_gt=30),      # 按距离档
    GroupSpec("blur",         min_gt=30),      # 按模糊档
    GroupSpec("lighting",     min_gt=30),      # 按光照档
    GroupSpec("__size__", bins=FINE_AREA_BINS, min_gt=30),   # 按目标尺寸
]
```

`__size__` 是保留字，按 GT 框面积分桶，不需要属性表。默认使用细分桶：

| 桶名 | 等效边长 |
|---|---|
| `nano(<8px)` | < 8 px |
| `tiny(8-16px)` | 8 ~ 16 px |
| `small(16-32px)` | 16 ~ 32 px |
| `medium(32-96px)` | 32 ~ 96 px |
| `large(>96px)` | > 96 px |

**为什么这样设计**：落在天空背景上的误检仍被计为假正例，指标才反映真实部署表现。
若改用"只在组内区域评测"的协议，会人为放大分数。
**两种协议结果差别很大，论文里必须写清用的是哪一种。**

`min_gt` 用于跳过样本过少的分组 —— 组内 GT 少于该数量时 AP 不可信。
真实数据集上建议设 30~50；占位数据集很小，示例配置里设的是 2。

---

## 输出：一键生成论文表格

每次 `eval` 会在输出目录写入：

- `result.json` —— 全部指标 + 分层结果 + 配置 + 环境 + git commit
- `predictions.npz` —— 原始预测

存 `predictions.npz` 的意义：**换了评测协议或指标定义时可以直接重算，不用重跑推理。**
推理往往是整个流程里最慢的一环。

汇总所有实验：

```bash
python tools/main.py report --runs runs --out report.md --latex
```

输出主对比表：

| Method | Backbone | Input | mAP@[.5:.95] | mAP@0.5 | AP_small | Params(M) | FPS | Latency(ms) | VRAM(MB) |
|---|---|---|---|---|---|---|---|---|---|
| ... | | | | | | | | | |

外加按每个属性维度拆开的分层表，以及可直接粘贴的 LaTeX 版本。

---

## 论文配图

一条命令把所有实验结果画成论文配图：

```bash
python tools/main.py figures --runs runs --out figs --config configs/example.yaml
```

输出（能画哪些取决于手里有什么数据）：

```
找到 2 个实验，输出目录 .../figs

  ✓ 尺度-mAP 曲线                          scale_curve.png
  ✓ 条件 mAP 柱状图 [__size__]              stratified___size__.png
  ✓ 条件 mAP 柱状图 [blur]                  stratified_blur.png
  ✓ 条件 mAP 柱状图 [distance_bin]          stratified_distance_bin.png
  ✓ 条件 mAP 柱状图 [lighting]              stratified_lighting.png
  ✓ PR 曲线 (IoU=0.5)                      pr_curves.png
  ✓ 失败案例 [Dummy(jitter)]               failure_cases_demo_dummy_jitter.png
  ✓ 失败案例 [Dummy(perfect)]              failure_cases_demo_dummy_perfect.png
  ✓ 数据集统计图                             dataset_stats.png
```

| 图 | 依赖 | 论文里的位置 |
|---|---|---|
| 尺度-mAP 曲线 | `result.json` 里的 `mAP/<尺寸桶>` | 实验章核心图，一眼看出小目标崩盘 |
| 条件 mAP 柱状图 | `result.json` 里的 `stratified` | 回答"在什么条件下谁更好" |
| PR 曲线 | `runs/*/predictions.npz` | 多方法对比 |
| 失败案例网格 | `predictions.npz` + 原始图像 | 定性分析；主动展示失败案例比藏着强 |
| 数据集统计图 | `--config` 指定的数据集 | Dataset 章节标准四联图 |

单张图失败不会中断其余图（出图是最后一步，能出一张是一张），跳过原因会打印出来。

### 当成库调用

```python
from detbench.core.viz import (
    plot_scale_curve,        # 目标尺寸 -> mAP 曲线（小目标论文最有力的一张图）
    plot_stratified_bars,    # 距离/模糊/光照 条件柱状图
    plot_pr_curves,          # 多方法 PR 曲线
    plot_failure_cases,      # TP 绿 / FP 红 / FN 黄 失败案例网格
    plot_dataset_stats,      # 尺寸分布/宽高比/位置热力图/每图目标数
)

# 尺度曲线：横轴等效边长，纵轴 mAP
plot_scale_curve(
    {"YOLO11": {"nano(<8px)": 0.11, "tiny(8-16px)": 0.35, "small(16-32px)": 0.62},
     "DINO-DETR": {"nano(<8px)": 0.07, "tiny(8-16px)": 0.41, "small(16-32px)": 0.66}},
    "figs/scale_curve.pdf",
)

# 条件柱状图支持两种写法，任选
plot_stratified_bars({"本方法": {"near": 0.95, "far": 0.43}}, "距离档", "figs/bars.pdf")
plot_stratified_bars({"本方法": {"near": {"mAP": 0.95}}}, "距离档", "figs/bars.pdf")
```

### 三个已处理好的坑

1. **中文字体自动选择**。matplotlib 默认字体不含 CJK 字形，中文标签会渲染成
   一排方框（"豆腐块"）**且不报任何错**，往往到把图贴进论文时才发现。
   `_pick_cjk_font()` 会按 Windows/macOS/Linux 常见字体优先级自动挑选，
   并关掉 `axes.unicode_minus`（否则负号也是方框）。
2. **对数轴刻度不再重叠**。尺度曲线和尺寸分布图都用对数横轴，
   不显式控制刻度就会堆出 `4×10⁰ 6×10⁰ 10¹ 2×10¹ 3×10¹…` 这种糊成一团的标签。
   现在用普通数字格式并关闭次刻度标签。
3. **参数校验前置**。传错数据层级时会抛出说明清楚的 `TypeError`
   （而不是 `'float' object has no attribute 'get'`）。

需要 matplotlib。缺失时会抛出 `VizUnavailable` 并给出安装提示，
评测、分层统计、结果汇总都不依赖它。

想输出矢量图，把 `viz.py` 里的 `savefig` 后缀从 `.png` 改成 `.pdf` 即可，
论文排版时放大不失真。

---

## 复现性保证

`result.json` 里自动记录：

- `git_commit`（含 `-dirty` 标记，能看出实验时代码有没有未提交改动）
- 完整 `config`（所有影响结果的参数）
- `env`：Python / PyTorch / CUDA / cuDNN 版本、GPU 型号与显存
- `dataset_summary`：图像数、目标数、每类目标数、**目标等效边长分位分布**
- `seed`、`timestamp`、`train_seconds`

`dataset_summary` 里的尺寸分位分布尤其重要 —— 换数据集后如果 mAP 变了，
你可以立刻判断是因为方法变了还是因为目标尺度分布变了。

---

## 已知边界

- **不支持 crowd 区域**（COCO `iscrowd=1`）：`load_coco` 会直接跳过并计入不了 GT。
  需要时请显式扩展。
- **AP 实现与 pycocotools 在数值上可能有极小差异**：本实现用 VOC2010+ 全点插值，
  COCO 用 101 点采样，理论上等价但浮点上不保证逐位相同。
  **同一个项目内始终用本框架即可保持自洽**；若要和论文里的 pycocotools 数字对比，
  请注意这一点。
- **`speed.py` 的测速包含预处理与后处理**，测的是端到端部署延迟，而不是纯网络前向。
  这对无人机场景是有意为之（部署关心的是端到端），但不要和只报网络时间的论文直接比。
- **速度评测需要 torch**；未安装时会自动跳过，精度指标不受影响。

---

## 快速命令参考

```bash
python tools/main.py doctor                          # 依赖体检
python tools/main.py make-demo --out data/demo       # 生成占位数据集
python tools/main.py selftest                        # 指标实现自检（改代码后必跑）
python tools/main.py eval --config configs/exp1.yaml # 评测一个模型
python tools/main.py report --runs runs --out report.md --latex   # 汇总出论文表
python tools/main.py figures --runs runs --out figs --config configs/example.yaml  # 生成论文配图

python scripts/check_viz.py                          # 绘图函数自检（需要 matplotlib）
python scripts/check_dataset_tools.py                # 数据集工具全流程检查（33 项）

python tools/dataset.py convert --from coco --to yolo --ann <标注> --out <输出>
python tools/dataset.py split  --ann <标注> --attr <属性表> --group-key clip_id --out <输出>
python tools/dataset.py check  --ann <标注> --images <图像根>
python tools/dataset.py agreement --ann-a <标注A> --ann-b <标注B>
python tools/dataset.py dedupe --images <帧目录> --threshold 8

python scripts/check_pipeline.py                     # 行为合理性检查（单调性等）
python scripts/check_mmdet_adapter.py                # DETR 适配器逻辑自检（不需要装 mmdet）

python tools/train.py --config configs/train_example.yaml --dump-config    # 检查训练配置
python tools/train.py --config configs/train_example.yaml --estimate-only  # 估训练耗时
python tools/train.py --config configs/train_example.yaml                  # 训练 + 自动评测
```
