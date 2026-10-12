# AGENTS.md — 交接文档

> 这份文件是给**下一个接手本项目的 AI agent**（或未来的你自己）看的。
> `README.md` 讲"怎么用"，这份讲**"现在处于什么状态、哪些验证过、哪些是陷阱、别重做什么"**。
>
> **先读这份，再读 README。** 最后更新：commit `936cce6`。

---

## 1. 三十秒定位

**DetBench** —— 面向**小目标 / 对空目标检测**的可复现评测框架。

它存在的理由不是"再实现一个检测器"，而是解决三个写论文时的具体痛点：

1. **分层评测**：标准 mAP 只给一个数，但论文结论藏在条件分布里
   （"< 8px 时 mAP 掉到 0.07"、"重模糊帧上 YOLO 反超 DETR"）。
   本框架把"按属性切片后再算 AP"做成一等公民。
2. **统一评测协议**：所有模型走同一套预处理与指标定义，消除"对比不公平"。
3. **全链路可追溯**：每次实验自动记录 git commit / 配置 / 环境 / 数据集统计。

**真实课题背景**：无人机对直升机做单类别检测，目标是发**会议论文**
（导师要求仅为会议）。算力是 2×4060 Ti 分处两台机器，时间预算充足。

**当前最大瓶颈不是代码，是数据** —— 手上还没有真实直升机数据集。

---

## 2. 当前状态

| 项目 | 值 |
|---|---|
| 项目路径 | `C:\Users\nichenh\Desktop\detbench` |
| Git commit | `936cce6`（工作区干净，已推送） |
| 远程仓库 | https://github.com/nodrake1/detbench （Public） |
| 文件数 | 38 个（源码约 380 KB） |
| CI | **9/9 全绿**（ubuntu + windows × py3.10/3.12） |
| 代码规模 | 约 4,500 行 Python |

```
core/     11 个  数据接口 / 分层 mAP / 评测 / 训练配置 / 测速 / 结果 / 配图 / 依赖探测 / 配置解析
models/    5 个  mmdet DETR 全家族 / YOLO / torchvision / 占位检测器
tools/     4 个  main.py（eval·report·figures·doctor·selftest）/ train.py / dataset.py
scripts/   7 个  bootstrap + 5 套自检 + 数据生成
configs/   3 个  example.yaml（评测）/ train_example.yaml / train_smoke.yaml
.github/workflows/ci.yml
```

---

## 3. 环境：用哪个解释器

这台机器上有**两个可用的 Python**，用途不同：

### A. `conda-transformer`（有 GPU，跑真实模型用这个）

```
解释器: D:\Anaconda3\envs\transformer\python.exe
Python 3.10.20

torch          2.7.1+cu128
torchvision    0.22.1+cu128
ultralytics    8.4.175
numpy          1.26.4      ← 必须是 1.x，见第 5 节
matplotlib     3.7.5
pandas         2.0.3
cv2            4.8.1
PIL            12.3.0
yaml           6.0.3
thop           2.2.2
CUDA           True | NVIDIA GeForce RTX 4060 Ti | 8.0 GB

缺失: mmdet / mmcv / mmengine   ← 所以 mmdet 后端尚未验证
```

### B. `sandbox-bundled`（无 GPU，跑无 torch 依赖的验证）

```
解释器: C:\Users\nichenh\.dsh\dsh-runtimes\dsh-primary-runtime\dependencies\python\python.exe
Python 3.12.14

torch          2.14.1+cpu   ← CPU-only！cuda 相关 API 不存在
numpy          2.3.5        ← 注意这里是 2.x，与新环境不同
matplotlib     3.11.2
pandas         3.0.1
yaml           6.0.3
CUDA           False
```

**两套环境都要跑回归** —— 它们覆盖了 CPU-only torch（第 5 节的坑）与 GPU torch 两条路径。

---

## 4. 已验证 vs 未验证（**这一节最重要**）

### ✅ 已用真实 torch 端到端验证

| 功能 | 证据 |
|---|---|
| 评测链路（真实 GPU） | Faster R-CNN 随机权重跑通，报出 **13.3 FPS / 75.3 ms / 623 MB 峰值显存** |
| **训练闭环** | YOLO11n 从零训练 50 轮 → 自动评测 → `result.json`。**学习曲线真实上涨** 0.146→0.221 |
| **分层评测的正确性** | 三个维度排序全符合物理直觉：near 0.459 > mid 0.153 > far 0.025；light 0.459 > heavy 0.084；medium 0.429 > small 0.354 > tiny 0.053 > nano 0.000 |
| 报告 / 配图 | 4 个实验（2 个 dummy + YOLO + FRCNN）汇总成表，11 张图全部生成 |
| 数据集工具 | 33 项检查全过，含 COCO↔YOLO 往返转换（坐标误差 < 0.001px） |
| CI | 9/9 作业全绿 |

### ❌ 未验证

| 功能 | 原因 | 风险 |
|---|---|---|
| **mmdet 后端**（Deformable DETR / DINO-DETR） | `mmcv` 在 Windows 上需编译，未安装 | 中。代码路径写了但从未真跑 |
| 多机 DDP（`torchrun --nnodes=2`） | 需要两台机器同时在 | 中 |
| **真实数据** | 还没有真实直升机数据集 | **高，这是当前最大未知** |
| 可视化在真实论文图上的效果 | 只用合成数据出过图 | 低 |

**接手时不要假设 mmdet 能用。** 第一次用请先跑：
```bash
python tools/train.py --config configs/train_example.yaml --dump-config
```

---

## 5. 环境陷阱（会浪费你几小时的那种）

### ⚠️ 1. `transformer` 环境的 numpy 曾被装坏（已修，但会复发）

**原始症状**（非常有误导性）：
```
torch.from_numpy  -> TypeError: expected np.ndarray (got numpy.ndarray)
numpy.core.multiarray -> AttributeError: no attribute 'number'
```

**根因**：numpy 被覆盖安装，1.26.4 与 2.2.6 的**文件混在一起**
（`numpy/core` 与 `numpy/_core` 并存、两个 `dist-info` 同时存在、
两个不同大小的 `_multiarray_umath.pyd`）。

**已处理**：备份损坏目录 + 干净重装 numpy 1.26.4。
备份在 `D:\Anaconda3\envs\transformer\Lib\site-packages\`：
```
numpy_BROKEN_20261012_134438
numpy-1.26.4.dist-info.bak_20261012_134438
numpy-2.2.6.dist-info.bak_20261012_134438
```

**防复发**：装任何新包都用约束文件锁住 numpy 大版本：
```bash
pip install -c constraints.txt <包名>
```
原因：该环境的 `opencv-python 4.8` / `matplotlib 3.7` / `pandas 2.0`
**都只支持 numpy 1.x**。pip 顺手升到 2.x 会让这三个包一起坏。

### ⚠️ 2. CPU-only 的 torch 里 `torch.cuda.*` 根本不存在

不是返回 False，而是**抛 `AttributeError`**。实测：
```
torch 2.14.1+cpu -> torch.cuda.reset_peak_memory_stats() 抛
    AttributeError: module 'torch._C' has no attribute '_cuda_resetPeakMemoryStats'
```
已修：`core/speed.py:cuda_is_usable()` 显式判断，显存统计改为尽力而为。
**不要退回成只判断 `device.startswith("cuda")`。**

### ⚠️ 3. Windows 控制台编码（cp1252 / cp936 都会崩）

- 英文 Windows = `cp1252` → 打印中文直接 `UnicodeEncodeError`
- 简体中文 Windows = `cp936`(GBK) → 中文可以，但打印 `⚠️` 这类字符照样崩

已修：所有入口脚本调用 `core/deps.py:force_utf8_output()`（含 `errors="replace"`）。
**新增入口脚本时记得也调它**，否则 CI 的 Windows 作业会红。

### ⚠️ 4. 网络/代理

- **git push 依赖代理**：`~/.ssh/config` 里配了
  `ProxyCommand D:/Git-Installed/mingw64/bin/connect.exe -H 127.0.0.1:7890 %h %p`
- **代理不开就报** `FATAL: Unable to connect to relay host, errno=10061`。
  看到这个先开代理，别怀疑别的。
- HTTPS 直连 github.com 会被重置；**走 SSH + 443 端口**是可靠的
- pip 装包也要代理：`--proxy http://127.0.0.1:7890`

### ⚠️ 5. 沙箱（如果 agent 在沙箱里跑）

- 受限模式下**写用户主目录会被拒绝**（`~/.ssh/config` 写不了）。
  变通：把 SSH 配置写进仓库本地 git config
  （`git config core.sshCommand "ssh -p 443 ..."`）
- **git 会派生 `sh.exe`**，受限模式禁止创建命名管道
  （`couldn't create signal pipe, Win32 error 5`）。需要放宽权限或让用户手动执行
- **PowerShell 的 `Add-Content -Encoding utf8` / `Set-Content` 会写 BOM 或破坏中文**。
  改文件请用 `write`/`edit` 工具，或 Python 的 `read_text/write_text(encoding="utf-8")`。
  这个坑在本项目里已经踩坏过 7 个文件的中文注释
- **`String.Replace(a,b,count)` 三参数重载在 PowerShell 里不存在**，
  且抛的是**终止性错误**，会让整个脚本中止（不会静默跳过）

---

## 6. 别重复做的事

以下都是**已经排查过的，不要重新"发现"一遍**：

1. **不要重写评测指标** —— `core/metrics.py` 的实现用 `selftest` 的 18 项验证过，
   包括"重复预测只算一次 TP"这类容易错的贪心匹配细节。
   改完**必须**跑 `python tools/main.py selftest`。
2. **不要把 `from_numpy` 改回去** —— 在 numpy ABI 不匹配的环境会崩，`as_tensor` 更稳。
3. **不要退回"只判断设备字符串"的 CUDA 检查** —— 见第 5 节陷阱 2。
4. **不要按帧随机划分数据集** —— 同一段视频的相邻帧高度相似，会造成
   scene leakage 使 mAP 虚高。必须用 `tools/dataset.py split --group-key clip_id`。
5. **不要在算 mAP 时用高置信度阈值** —— 必须 `conf=0.001`，否则 precision 虚高。
   代码默认已经是这个值，别"优化"它。
6. **不要删 `setup_github.ps1`** —— 虽然已经没用，但用户没明确说删。
7. **不要把 `data/` `runs/` `figs/` 加进 git** —— 已在 `.gitignore`，权重用 Releases 发。

---

## 7. 看起来像 bug 但**不是** bug 的地方

避免浪费时间排查这些：

| 现象 | 真相 |
|---|---|
| `Dummy(perfect)` 的 `mAP@[.5:.95]` 是 1.00，但某些尺寸桶是 0.97~0.98 | float32 下 GT 与预测框在 IoU=0.95 阈值上边界抖动，正常 |
| `Dummy(random)` 的 mAP 是 **0.0000** | 在 960×640 图上随机框命中 15px 小目标的概率极低，**合理** |
| `Dummy(perfect)` 测速报 **163934 FPS** | dummy 在 CPU 上几乎不耗时，落到了计时器分辨率以下。代码已标记 `reliable=False` 并警告；**别把这个数字写进论文** |
| demo 数据集的相邻帧被判为"重复" | 合成天空图本来就几乎一样，去重工具**判断正确**。真实视频帧不会这样 |
| 纯色图的 aHash/dHash 全部相同 | 感知哈希只看**灰度结构**，这是设计特性，已写进 `dedupe --help` 的"已知局限" |
| PR 曲线是水平直线（precision 恒为 1.0） | dummy 的 FP 分数低、TP 分数高，按分数排序时 TP 全在前面。真实模型会有形状 |

---

## 8. 已验证的常用命令

```bash
# ---- 环境自检（换设备后先跑这个）----
python scripts/bootstrap.py          # 体检 + 冒烟测试 + 路径自检
python tools/main.py doctor          # 只做依赖体检

# ---- 自检（改任何评测代码后必跑）----
python tools/main.py selftest              # 18 项指标实现自检
python scripts/check_pipeline.py           # 行为合理性（单调性）
python scripts/check_mmdet_adapter.py      # 21 项 DETR 适配器逻辑（不需要装 mmdet）
python scripts/check_viz.py                # 8 项绘图函数
python scripts/check_dataset_tools.py      # 33 项数据集工具

# ---- 端到端 ----
python tools/main.py make-demo --out data/demo
python tools/main.py eval --config configs/example.yaml
python tools/main.py report --runs runs --out report.md --latex
python tools/main.py figures --runs runs --out figs --config configs/example.yaml

# ---- 训练（先冒烟，再正式）----
python tools/train.py --config configs/train_smoke.yaml                    # 几十秒
python tools/train.py --config configs/train_example.yaml --dump-config    # 检查配置
python tools/train.py --config configs/train_example.yaml                  # 正式训练

# ---- 数据集工具 ----
python tools/dataset.py check     --ann <标注> --images <图像根>
python tools/dataset.py convert   --from coco --to yolo --ann <标注> --out <输出>
python tools/dataset.py split     --ann <标注> --attr <属性表> --group-key clip_id --out <输出>
python tools/dataset.py agreement --ann-a <标注A> --ann-b <标注B>
python tools/dataset.py dedupe    --images <帧目录> --threshold 8
```

**全量回归（两套环境各 13 项）** —— 改动后建议都跑：
```powershell
# 用 conda（GPU）
& "D:\Anaconda3\envs\transformer\python.exe" tools\main.py selftest
# 用沙箱自带（CPU-only torch）
& "C:\Users\nichenh\.dsh\dsh-runtimes\dsh-primary-runtime\dependencies\python\python.exe" tools\main.py selftest
```

---

## 9. 下一步（按建议优先级）

### 🥇 优先级 1：把**真实数据集**接进来

这是当前最大未知。链路已完全验证，但只用过合成占位数据。

```bash
# 1) 先质检
python tools/dataset.py check --ann <真实标注> --images <图像根>

# 2) 按架次分层划分（防泄漏）
python tools/dataset.py split --ann <标注> --attr <属性表> \
    --group-key clip_id --stratify-by distance_bin --out annotations/

# 3) 先 dry-run 确认路径（换设备后必做）
python tools/main.py eval --config configs/exp01.yaml --dry-run

# 4) 训练
python tools/train.py --config configs/exp01.yaml
```

**需要用户提供的信息**：真实数据在哪、有多少帧、标注格式、有没有采集元数据
（距离/模糊/光照档 —— 分层评测依赖它）。

### 🥈 优先级 2：打通 mmdet 后端（DETR 系）

```bash
pip install -c constraints.txt mmengine mmcv mmdet
git clone https://github.com/open-mmlab/mmdetection third_party/mmdetection
python tools/train.py --config configs/train_example.yaml --dump-config
```
Windows 上 `mmcv` 可能需要预编译 wheel（OpenMMLab 有 wheel 源）或编译工具链。
**先试装，失败就退回 YOLO 后端**（已完全可用）。

### 🥉 优先级 3：论文相关的收尾

- `CITATION.cff`
- 仓库 Topics（`object-detection` `small-object-detection` `benchmark` `detr` …）
- 删掉已无用的 `setup_github.ps1`（**先问用户**）
- 双机实验调度脚本（用户之前说"先放一边"，随时可捡起）
- `viz.py` 输出矢量图（把 `savefig` 后缀改 `.pdf`）

---

## 10. 与用户协作的注意事项

- **用户很在意"真的验证过"**，不喜欢只跑通 hello world 就宣称完成。
  本项目正是在"用真实 torch 端到端跑一遍"时才发现 8 个 bug。
  **能真跑就跑真跑，不能跑要明说"未验证"。**
- **报告结论时要区分**"代码逻辑正确"与"端到端验证过"。本文件第 4 节就是范例。
- 用户偏好**中文回复**，且喜欢先给结论再给依据。
- 用户会直接说"进行 A/B/C"来选择方案，所以提问时给编号选项。
- **改用户的 conda 环境前先说明**（本项目改过：备份并重装 numpy、装了 ultralytics）。
- 用户对代理/网络的坑不熟，涉及 `git push` 失败时应直接指向"代理没开"。

---

## 11. 一句话交接

**代码已经完整且经过真实 GPU 端到端验证，评测与训练链路都可用；
下一步的价值不在写代码，而在把真实数据接进来。
接数据前先读第 5 节的环境陷阱和第 6 节的"别重复做的事"，能省你几个小时。**
