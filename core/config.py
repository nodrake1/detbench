"""配置文件载入：JSON 原生支持，YAML 走内置的零依赖解析器。

为什么自己写 YAML 解析
----------------------
评测框架的配置只用到 YAML 的一个很小的子集：嵌套映射、列表、标量、行内列表、
``#`` 注释。为了这点需求让每个用户都去装 PyYAML（尤其在离线/受限环境里
根本装不上）不划算，所以这里内置一个覆盖该子集的解析器。

如果环境里装了 PyYAML，仍然优先用它（更完整）；否则自动回退。

**不支持的 YAML 特性**（如果用了会报错，而不是静默给错结果）：
锚点/引用（``&`` / ``*``）、多行块标量（``|`` / ``>``）、复杂流式映射、
显式类型标签。配置文件里不需要这些。
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple


class ConfigError(ValueError):
    """配置文件格式错误。"""


# --------------------------------------------------------------------------- #
# 标量解析
# --------------------------------------------------------------------------- #


def _parse_scalar(text: str) -> Any:
    """把 YAML 标量文本转成 Python 值。"""
    s = text.strip()
    if s == "" or s in ("~", "null", "Null", "NULL"):
        return None
    low = s.lower()
    if low in ("true", "yes", "on"):
        return True
    if low in ("false", "no", "off"):
        return False

    # 去掉包裹的引号
    if len(s) >= 2 and s[0] == s[-1] and s[0] in ("'", '"'):
        return s[1:-1]

    # 数字（含科学计数法、小数、负号）
    try:
        if any(ch in s for ch in ".eE") and not s.startswith(("0x", "0X")):
            return float(s)
        return int(s)
    except ValueError:
        pass
    return s


def _split_inline_list(text: str) -> List[str]:
    """切分 ``[a, b, c]``，正确处理引号内的逗号。"""
    inner = text.strip()[1:-1]
    parts: List[str] = []
    buf: List[str] = []
    quote: Optional[str] = None
    for ch in inner:
        if quote:
            buf.append(ch)
            if ch == quote:
                quote = None
        elif ch in ("'", '"'):
            quote = ch
            buf.append(ch)
        elif ch == ",":
            parts.append("".join(buf))
            buf = []
        else:
            buf.append(ch)
    if buf:
        parts.append("".join(buf))
    return [p for p in (p.strip() for p in parts) if p != ""]


def _parse_value(text: str) -> Any:
    s = text.strip()
    if s.startswith("[") and s.endswith("]"):
        return [_parse_scalar(p) for p in _split_inline_list(s)]
    if s.startswith("{") and s.endswith("}"):
        out: Dict[str, Any] = {}
        for pair in _split_inline_list(s):
            if ":" not in pair:
                raise ConfigError(f"行内映射项缺少冒号: {pair!r}")
            k, v = pair.split(":", 1)
            out[_parse_scalar(k)] = _parse_value(v)
        return out
    return _parse_scalar(s)


# --------------------------------------------------------------------------- #
# 迷你 YAML 解析器（缩进驱动）
# --------------------------------------------------------------------------- #


def _strip_comment(line: str) -> str:
    """去掉行尾注释，但保留引号内的 ``#``。"""
    out: List[str] = []
    quote: Optional[str] = None
    for i, ch in enumerate(line):
        if quote:
            out.append(ch)
            if ch == quote:
                quote = None
        elif ch in ("'", '"'):
            quote = ch
            out.append(ch)
        elif ch == "#" and (i == 0 or line[i - 1] in " \t"):
            break
        else:
            out.append(ch)
    return "".join(out).rstrip()


def parse_yaml(text: str) -> Dict[str, Any]:
    """解析 YAML 子集，返回 dict。

    实现方式：先把 ``(缩进, 内容)`` 展开成序列，再递归下降。
    """
    raw_lines: List[Tuple[int, str]] = []
    for lineno, line in enumerate(text.splitlines(), 1):
        if "\t" in line[: len(line) - len(line.lstrip())]:
            raise ConfigError(f"第 {lineno} 行用了 Tab 缩进；YAML 只允许空格")
        stripped = _strip_comment(line)
        if not stripped.strip():
            continue
        indent = len(stripped) - len(stripped.lstrip(" "))
        raw_lines.append((indent, stripped.strip()))

    if not raw_lines:
        return {}

    pos = 0

    def parse_block(indent: int) -> Any:
        nonlocal pos
        # 判断这一层是列表还是映射
        if pos < len(raw_lines) and raw_lines[pos][1].startswith("- "):
            items: List[Any] = []
            while pos < len(raw_lines):
                ind, content = raw_lines[pos]
                if ind < indent or not content.startswith("- "):
                    break
                if ind > indent:
                    raise ConfigError(f"列表项缩进异常: {content!r}")
                body = content[2:].strip()
                pos += 1
                if ":" in body and not body.startswith(("[", "{")):
                    # 列表项是一个映射，且首个键写在 "- " 同一行
                    k, v = body.split(":", 1)
                    item: Dict[str, Any] = {}
                    item[_parse_scalar(k)] = _parse_value(v) if v.strip() else parse_block(indent + 2)
                    # 继续吸收同一缩进层级的后续键
                    while pos < len(raw_lines) and raw_lines[pos][0] > indent and not raw_lines[pos][1].startswith("- "):
                        sub_ind, sub = raw_lines[pos]
                        if ":" not in sub:
                            raise ConfigError(f"映射项缺少冒号: {sub!r}")
                        sk, sv = sub.split(":", 1)
                        pos += 1
                        item[_parse_scalar(sk)] = _parse_value(sv) if sv.strip() else parse_block(sub_ind + 2)
                    items.append(item)
                else:
                    items.append(_parse_value(body) if body else parse_block(indent + 2))
            return items

        mapping: Dict[str, Any] = {}
        while pos < len(raw_lines):
            ind, content = raw_lines[pos]
            if ind < indent:
                break
            if ind > indent:
                raise ConfigError(f"映射缩进异常: {content!r}")
            if content.startswith("- "):
                break
            if ":" not in content:
                raise ConfigError(f"映射项缺少冒号: {content!r}")
            key, val = content.split(":", 1)
            pos += 1
            k = _parse_scalar(key)
            if val.strip():
                mapping[k] = _parse_value(val)
            else:
                # 值为空 -> 看下一层；没有下一层就是 None
                if pos < len(raw_lines) and raw_lines[pos][0] > indent:
                    mapping[k] = parse_block(raw_lines[pos][0])
                elif pos < len(raw_lines) and raw_lines[pos][0] == indent and raw_lines[pos][1].startswith("- "):
                    mapping[k] = parse_block(indent)
                else:
                    mapping[k] = None
        return mapping

    result = parse_block(raw_lines[0][0])
    if not isinstance(result, dict):
        raise ConfigError("配置文件顶层必须是映射（key: value）")
    return result


# --------------------------------------------------------------------------- #
# 对外入口
# --------------------------------------------------------------------------- #


def load_config(path: str | Path) -> Dict[str, Any]:
    """载入 YAML 或 JSON 配置。

    优先用 PyYAML（若已安装），否则回退到内置解析器。
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"配置文件不存在: {path}")
    text = path.read_text(encoding="utf-8")

    if path.suffix.lower() == ".json":
        data = json.loads(text)
        if not isinstance(data, dict):
            raise ConfigError("配置文件顶层必须是映射")
        return data

    try:  # 装了 PyYAML 就用它
        import yaml  # type: ignore

        data = yaml.safe_load(text)
        if not isinstance(data, dict):
            raise ConfigError("配置文件顶层必须是映射")
        return data
    except ImportError:
        return parse_yaml(text)


# --------------------------------------------------------------------------- #
# 环境变量展开
# --------------------------------------------------------------------------- #

_ENV_PATTERN = None


def expand_env(value: Any) -> Any:
    """递归展开字符串里的 ``${VAR}`` 与 ``${VAR:-默认值}``。

    这让配置可以写成机器无关的形式，例如::

        data_root: ${DETBENCH_DATA:-./data/HeliAir}

    换设备时只需要设一个环境变量，不用改配置文件里的每一行路径。
    """
    global _ENV_PATTERN
    if _ENV_PATTERN is None:
        import re

        _ENV_PATTERN = re.compile(r"\$\{([A-Za-z_][A-Za-z0-9_]*)(?::-([^}]*))?\}")

    if isinstance(value, str):

        def repl(m):
            name, default = m.group(1), m.group(2)
            return os.environ.get(name, default if default is not None else m.group(0))

        return _ENV_PATTERN.sub(repl, value)
    if isinstance(value, dict):
        return {k: expand_env(v) for k, v in value.items()}
    if isinstance(value, list):
        return [expand_env(v) for v in value]
    return value


# --------------------------------------------------------------------------- #
# 路径解析（跨设备可移植的关键）
# --------------------------------------------------------------------------- #

#: 会被解析为"相对于 data_root"的键名
_DATA_PATH_KEYS = ("ann_file", "images_root", "images_dir", "labels_dir", "ann_dir", "attr_file")


def apply_overrides(cfg: Dict[str, Any], overrides: Optional[Sequence[str]]) -> Dict[str, Any]:
    """应用形如 ``dataset.images_root=/path/a`` 的点号路径覆盖。

    用途：换设备或换数据集时，不必改配置文件，命令行直接盖掉即可::

        --set dataset.images_root=/data/HeliAir/images
        --set eval.predict_conf=0.01
        --set runtime.device=cuda:1
    """
    for item in overrides or []:
        if "=" not in item:
            raise ConfigError(f"--set 参数必须是 key=value 形式，得到: {item!r}")
        dotted, raw = item.split("=", 1)
        # 值按 YAML 标量解析，这样 true / 0.5 / [a,b] 都能正确转型
        val = _parse_value(raw.strip()) if raw.strip() else None
        node: Any = cfg
        parts = dotted.strip().split(".")
        for p in parts[:-1]:
            if p not in node or not isinstance(node[p], dict):
                node[p] = {}
            node = node[p]
        node[parts[-1]] = val
    return cfg


def resolve_paths(cfg: Dict[str, Any], data_root: Optional[str] = None) -> Dict[str, Any]:
    """把配置里的数据相关路径解析为绝对路径。

    单一规则（刻意做得简单，避免多条规则互相争夺同一批键）
    ------------------------------------------------------
    ``ann_file`` / ``images_root`` / ``attr_file`` 等数据路径若为相对路径，
    一律拼到 **有效数据根目录** 下：

        有效数据根 = 命令行 --data-root
                   > 配置顶层的 data_root（相对路径时相对配置文件目录解析）
                   > 项目根目录

    **数据路径不会相对配置文件所在目录解析。** 这一点是刻意的：
    如果既支持"相对 configs/"又支持"相对 data_root"，两者会打架，
    ``--data-root`` 会被静默忽略 —— 那正是最糟糕的失败模式。

    ``checkpoint`` / ``weights`` 这类模型文件走另一条规则（相对配置文件目录），
    因为权重通常和配置一起管理，而不是和数据集放一起。
    """
    cfg = dict(cfg)

    # 1) 确定有效数据根（绝对路径）
    root: Optional[str] = None
    if data_root:
        root = str(Path(str(data_root)).expanduser().resolve())
    elif cfg.get("data_root"):
        p = Path(str(cfg["data_root"])).expanduser()
        root = str(p if p.is_absolute() else (Path(cfg["_config_dir"]) / p).resolve())
    else:
        root = _project_root(cfg)

    cfg["data_root"] = root

    # 2) 数据路径挂到有效数据根下
    ds = dict(cfg.get("dataset", {}))
    for key in _DATA_PATH_KEYS:
        if ds.get(key):
            p = Path(str(ds[key])).expanduser()
            ds[key] = str(p if p.is_absolute() else (Path(root) / p))
    cfg["dataset"] = ds

    # 3) 权重路径挂到配置文件目录下
    #
    #    ⚠️ 但**只在这个文件真的存在时才锚定**。
    #    ultralytics 的权重可以是"模型名"而不是路径，例如::
    #        yolo11n.yaml   -> 用内置结构从零初始化
    #        yolo11n.pt     -> 用内置预训练权重（必要时自动下载）
    #    如果无条件锚定成 <配置文件目录>/yolo11n.yaml，
    #    ultralytics 就会去找一个不存在的本地文件并报错，
    #    而这个报错看起来像"配置文件路径写错了"，非常误导。
    m = cfg.get("model")
    if isinstance(m, dict):
        m = dict(m)
        for key in ("checkpoint", "weights", "config"):
            if not m.get(key):
                continue
            p = Path(str(m[key])).expanduser()
            if p.is_absolute():
                continue
            # 相对路径：先看相对配置文件目录，再看相对当前工作目录；
            # 两处都不存在就保持原样，交给后端自己解释
            # （mmdet 的 config 缺失会在适配器里报出带 clone 提示的清晰错误）。
            for base in (Path(cfg["_config_dir"]), Path.cwd()):
                cand = base / p
                if cand.exists():
                    m[key] = str(cand.resolve())
                    break
        cfg["model"] = m

    return cfg


def _project_root(cfg: Dict[str, Any]) -> str:
    """推断项目根目录：配置文件在 configs/ 下时取父目录，否则取配置目录本身。"""
    d = cfg.get("_config_dir")
    if not d:
        return str(Path.cwd())
    p = Path(d)
    return str(p.parent if p.name.lower() in ("configs", "config", "conf") else p)


def _anchor_output_to_project(cfg: Dict[str, Any]) -> None:
    """把 ``output.dir`` 的相对路径锚定到项目根目录。

    这样 ``output.dir: runs/exp1`` 无论配置放在哪一层，结果都写到
    ``<项目根>/runs/exp1``，而不是 ``<项目根>/configs/runs/exp1``。
    """
    out = cfg.get("output")
    if not isinstance(out, dict) or not out.get("dir"):
        return
    p = Path(str(out["dir"])).expanduser()
    if p.is_absolute():
        return
    out = dict(out)
    out["dir"] = str((Path(_project_root(cfg)) / p).resolve())
    cfg["output"] = out


def load_and_resolve(
    path: str | Path,
    overrides: Optional[Sequence[str]] = None,
    data_root: Optional[str] = None,
    check_paths: bool = True,
) -> Dict[str, Any]:
    """``load_config`` + 环境变量展开 + ``--set`` 覆盖 + 路径解析，一步到位。

    ``check_paths=True`` 时会在解析完成后检查关键路径是否存在，缺失就
    **立刻抛错并列出所有缺失项 + 排查建议**。换设备后最常见的问题就是
    路径没对上，早报错比跑到一半才炸好得多。
    """
    path = Path(path).resolve()
    cfg = load_config(path)
    cfg["_config_path"] = str(path)
    cfg["_config_dir"] = str(path.parent)
    cfg = expand_env(cfg)
    cfg = apply_overrides(cfg, overrides)
    cfg = resolve_paths(cfg, data_root=data_root)
    _anchor_output_to_project(cfg)

    if check_paths:
        missing = find_missing_paths(cfg)
        if missing:
            lines = "\n".join(f"    [{label}] {p}" for label, p in missing)
            raise FileNotFoundError(
                "以下路径不存在，无法继续：\n"
                f"{lines}\n\n"
                f"当前 data_root = {cfg.get('data_root')}\n\n"
                "排查建议：\n"
                "  1. 加 --dry-run 查看解析后的全部路径：\n"
                "       python tools/main.py eval --config <配置> --dry-run\n"
                "  2. 换设备后数据位置变了，用 --data-root 指向新的数据根目录\n"
                "  3. 首次运行先生成占位数据集：python tools/main.py make-demo"
            )
    return cfg


def find_missing_paths(cfg: Dict[str, Any]) -> List[Tuple[str, str]]:
    """返回 ``[(说明, 路径), ...]``，只包含**填写了但不存在**的路径。"""
    ds = cfg.get("dataset", {}) or {}
    m = cfg.get("model", {}) or {}
    candidates = [
        ("dataset.ann_file", ds.get("ann_file")),
        ("dataset.images_root", ds.get("images_root") or ds.get("images_dir")),
        ("dataset.labels_dir", ds.get("labels_dir")),
        ("dataset.ann_dir", ds.get("ann_dir")),
        ("dataset.attr_file", ds.get("attr_file")),
        ("model.config", m.get("config")),
        ("model.checkpoint", m.get("checkpoint") or m.get("weights")),
    ]
    out: List[Tuple[str, str]] = []
    for label, p in candidates:
        if p and not Path(str(p)).exists():
            out.append((label, str(p)))
    return out
