"""实验记录与结果汇总。

这个模块解决一个非常具体、也非常致命的问题：

    三个月后你看到一个 mAP=0.741 的数字，想不起来它是哪次实验、用什么配置、
    哪个 commit、哪个数据划分跑出来的。

对策很简单但必须自动化：**每次实验往磁盘写一个 result.json，里面包含
配置、环境、指标、耗时**。然后一键扫描所有 result.json 生成论文表格。

之后论文的主对比表直接从脚本生成，不用手抄数字。手抄必错，
而且审稿人要求补实验时你会感谢自己。
"""

from __future__ import annotations

import json
import os
import platform
import socket
import subprocess
import sys
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence

# --------------------------------------------------------------------------- #
# 环境快照
# --------------------------------------------------------------------------- #


def git_commit(cwd: Optional[str | Path] = None) -> str:
    """当前 git commit hash（含 ``-dirty`` 标记），非 git 仓库返回 ``unknown``。"""
    try:
        h = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            cwd=str(cwd) if cwd else None,
            capture_output=True,
            text=True,
            timeout=5,
        )
        if h.returncode != 0:
            return "unknown"
        commit = h.stdout.strip()
        st = subprocess.run(
            ["git", "status", "--porcelain"],
            cwd=str(cwd) if cwd else None,
            capture_output=True,
            text=True,
            timeout=5,
        )
        return f"{commit}-dirty" if st.stdout.strip() else commit
    except Exception:
        return "unknown"


def env_snapshot() -> Dict[str, Any]:
    """记录可复现所需的环境信息。"""
    snap: Dict[str, Any] = {
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "hostname": socket.gethostname(),
    }
    try:
        import torch

        snap["torch"] = torch.__version__
        snap["cuda"] = torch.version.cuda
        snap["cudnn"] = torch.backends.cudnn.version()
        snap["gpu_count"] = torch.cuda.device_count()
        snap["gpus"] = [torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())]
        if torch.cuda.is_available():
            props = torch.cuda.get_device_properties(0)
            snap["gpu_mem_gb"] = round(props.total_memory / 1024 ** 3, 1)
    except Exception as exc:  # torch 未安装时也能记录
        snap["torch_error"] = repr(exc)
    try:
        import numpy as np

        snap["numpy"] = np.__version__
    except Exception:
        pass
    # 记录装了的可选依赖，便于事后判断"这次实验用的是哪个版本的 DETR 实现"
    try:
        from detbench.core.deps import describe_environment

        snap["optional_deps"] = describe_environment()
    except Exception:
        pass
    return snap


# --------------------------------------------------------------------------- #
# 结果记录
# --------------------------------------------------------------------------- #


@dataclass
class ExperimentResult:
    """一次评测实验的完整记录。"""

    exp_id: str
    method: str
    dataset: str
    split: str = "test"
    backbone: str = ""
    input_size: str = ""
    epochs: int = 0
    seed: int = 0

    # 精度指标：扁平的 key-value，便于任意模型塞入自定义指标
    metrics: Dict[str, float] = field(default_factory=dict)

    # 速度指标
    fps: float = 0.0
    latency_ms: float = 0.0
    peak_vram_mb: float = 0.0
    params_m: float = 0.0
    flops_g: float = 0.0

    # 分层评测结果：{"distance_bin": {"near": {...}, "mid": {...}}, ...}
    stratified: Dict[str, Dict[str, Dict[str, float]]] = field(default_factory=dict)

    # 元信息
    config: Dict[str, Any] = field(default_factory=dict)
    dataset_summary: Dict[str, Any] = field(default_factory=dict)
    env: Dict[str, Any] = field(default_factory=dict)
    git_commit: str = ""
    timestamp: str = ""
    train_seconds: float = 0.0
    notes: str = ""

    #: 本次实验的产物目录（result.json / predictions.npz 所在处）。
    #: **必须记录**：exp_id 与目录名不一定相同（例如 exp_id=frcnn_random
    #: 但 out_dir=runs/frcnn_test）。下游（如 figures）靠它定位
    #: predictions.npz；若按 exp_id 拼路径会找不到并静默跳过出图。
    out_dir: str = ""

    def __post_init__(self) -> None:
        if not self.timestamp:
            self.timestamp = time.strftime("%Y-%m-%d %H:%M:%S")
        if not self.git_commit:
            self.git_commit = git_commit()
        if not self.env:
            self.env = env_snapshot()

    # -- 便捷访问器，把常用指标统一到固定 key，方便跨方法汇总 --
    @property
    def map50(self) -> Optional[float]:
        for k in ("mAP50", "AP50", "map50", "mAP@0.5"):
            if k in self.metrics:
                return float(self.metrics[k])
        return None

    @property
    def map5095(self) -> Optional[float]:
        for k in ("mAP", "mAP@[.5:.95]", "map5095"):
            if k in self.metrics:
                return float(self.metrics[k])
        return None

    @property
    def ap_small(self) -> Optional[float]:
        for k, v in self.metrics.items():
            if k.startswith("mAP/") and ("small" in k or "tiny" in k or "nano" in k):
                return float(v)
        return None

    def save(self, out_dir: str | Path) -> Path:
        out = Path(out_dir)
        out.mkdir(parents=True, exist_ok=True)
        path = out / "result.json"
        # 记录产物目录，供下游（figures / 事后排查）定位同目录下的 predictions.npz
        self.out_dir = str(out.resolve())
        with path.open("w", encoding="utf-8") as f:
            json.dump(asdict(self), f, ensure_ascii=False, indent=2, default=str)
        return path

    @classmethod
    def load(cls, path: str | Path) -> "ExperimentResult":
        with Path(path).open("r", encoding="utf-8") as f:
            raw = json.load(f)
        known = {f for f in cls.__dataclass_fields__}  # type: ignore[attr-defined]
        return cls(**{k: v for k, v in raw.items() if k in known})


def collect_results(root: str | Path, pattern: str = "**/result.json") -> List[ExperimentResult]:
    """递归收集某个目录下所有 result.json。"""
    root = Path(root)
    out: List[ExperimentResult] = []
    for p in sorted(root.glob(pattern)):
        try:
            out.append(ExperimentResult.load(p))
        except Exception as exc:
            print(f"[warn] 跳过无法解析的结果文件 {p}: {exc}", file=sys.stderr)
    return out


# --------------------------------------------------------------------------- #
# 论文表格生成
# --------------------------------------------------------------------------- #


def _fmt(v: Optional[float], nd: int = 4) -> str:
    if v is None:
        return "-"
    return f"{v:.{nd}f}"


def to_markdown_table(
    results: Sequence[ExperimentResult],
    columns: Optional[Sequence[str]] = None,
    sort_by: str = "map5095",
    ndigits: int = 4,
) -> str:
    """把结果列表渲染成 Markdown 表，可直接贴进论文草稿。

    默认列刻意做成"论文主对比表"的样子。
    """
    columns = list(columns or [
        "method", "backbone", "input_size", "map5095", "map50", "ap_small",
        "params_m", "fps", "latency_ms", "peak_vram_mb",
    ])
    header_map = {
        "method": "Method",
        "backbone": "Backbone",
        "input_size": "Input",
        "epochs": "Epochs",
        "map5095": "mAP@[.5:.95]",
        "map50": "mAP@0.5",
        "ap_small": "AP_small",
        "params_m": "Params(M)",
        "fps": "FPS",
        "latency_ms": "Latency(ms)",
        "peak_vram_mb": "VRAM(MB)",
        "flops_g": "FLOPs(G)",
        "exp_id": "ExpID",
    }

    def value_of(r: ExperimentResult, col: str) -> str:
        if col in ("method", "backbone", "input_size", "exp_id"):
            return str(getattr(r, col, ""))
        if col == "epochs":
            return str(r.epochs)
        if col in ("map50", "map5095", "ap_small"):
            return _fmt(getattr(r, col), ndigits)
        if col in ("fps", "latency_ms"):
            return _fmt(getattr(r, col, 0.0), 1)
        if col in ("params_m", "flops_g"):
            return _fmt(getattr(r, col, 0.0), 2)
        if col == "peak_vram_mb":
            return _fmt(getattr(r, col, 0.0), 0)
        if col in r.metrics:
            return _fmt(float(r.metrics[col]), ndigits)
        return "-"

    key_fn = {
        "map5095": lambda r: (r.map5095 is None, -(r.map5095 or 0.0)),
        "map50": lambda r: (r.map50 is None, -(r.map50 or 0.0)),
        "fps": lambda r: -r.fps,
        "method": lambda r: r.method,
    }.get(sort_by, lambda r: r.method)
    rows = sorted(results, key=key_fn)

    lines = ["| " + " | ".join(header_map.get(c, c) for c in columns) + " |"]
    lines.append("|" + "|".join(["---"] * len(columns)) + "|")
    for r in rows:
        lines.append("| " + " | ".join(value_of(r, c) for c in columns) + " |")
    return "\n".join(lines)


def to_latex_table(
    results: Sequence[ExperimentResult],
    columns: Optional[Sequence[str]] = None,
    caption: str = "Comparison of detection methods.",
    label: str = "tab:main",
    sort_by: str = "map5095",
) -> str:
    """生成 LaTeX 表格（booktabs 风格）。"""
    md = to_markdown_table(results, columns=columns, sort_by=sort_by)
    rows = [ln.strip("|").split(" | ") for ln in md.strip().splitlines()]
    header, body = rows[0], rows[2:]

    colspec = "l" + "c" * (len(header) - 1)
    out = [
        "\\begin{table}[t]",
        "  \\centering",
        f"  \\caption{{{caption}}}",
        f"  \\label{{{label}}}",
        f"  \\begin{{tabular}}{{{colspec}}}",
        "    \\toprule",
        "    " + " & ".join(header) + " \\\\",
        "    \\midrule",
    ]
    for row in body:
        out.append("    " + " & ".join(row) + " \\\\")
    out += ["    \\bottomrule", "  \\end{tabular}", "\\end{table}"]
    return "\n".join(out)


def to_stratified_markdown(
    results: Sequence[ExperimentResult],
    group_keys: Optional[Sequence[str]] = None,
    ndigits: int = 4,
) -> str:
    """把分层评测结果渲染成表格：每行一个方法，每列一个分组。

    这是论文里最有价值的那张表 —— 它回答"在什么条件下谁更好"。
    """
    if not results:
        return ""
    keys = list(group_keys or [])
    if not keys:
        keys = sorted({k for r in results for k in r.stratified})
    blocks: List[str] = []

    for key in keys:
        # 收集所有出现过的组名
        groups: List[str] = []
        for r in results:
            for g in r.stratified.get(key, {}):
                if g not in groups:
                    groups.append(g)
        groups.sort()
        if not groups:
            continue

        header = ["Method"] + [f"{key}={g}" for g in groups]
        lines = ["| " + " | ".join(header) + " |", "|" + "|".join(["---"] * len(header)) + "|"]
        for r in results:
            row = [r.method]
            for g in groups:
                m = r.stratified.get(key, {}).get(g, {})
                val = None
                for cand in ("mAP", "AP", "AP50"):
                    if cand in m:
                        val = m[cand]
                        break
                row.append(_fmt(val, ndigits))
            lines.append("| " + " | ".join(row) + " |")
        blocks.append(f"**分层评测 — 按 `{key}` 分组**\n\n" + "\n".join(lines))

    return "\n\n".join(blocks)


def write_report(
    results: Sequence[ExperimentResult],
    out_path: str | Path = "report.md",
    title: str = "Detection Benchmark Report",
) -> Path:
    """生成一份完整的 Markdown 报告：主表 + 分层表 + 环境信息。"""
    out_path = Path(out_path)
    parts = [f"# {title}", "", f"生成时间：{time.strftime('%Y-%m-%d %H:%M:%S')}", ""]

    parts += ["## 1. 主对比表", "", to_markdown_table(results), ""]

    strat = to_stratified_markdown(results)
    if strat:
        parts += ["## 2. 分层评测（条件 mAP）", "", strat, ""]

    parts += ["## 3. 实验清单", ""]
    parts.append("| ExpID | Method | Dataset | Git | Time | Notes |")
    parts.append("|---|---|---|---|---|---|")
    for r in results:
        parts.append(f"| {r.exp_id} | {r.method} | {r.dataset}/{r.split} | {r.git_commit} | {r.timestamp} | {r.notes} |")

    if results:
        env = results[0].env
        parts += ["", "## 4. 环境", "", "```json", json.dumps(env, ensure_ascii=False, indent=2), "```", ""]

    out_path.write_text("\n".join(parts), encoding="utf-8")
    return out_path
