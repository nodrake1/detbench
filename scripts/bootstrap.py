#!/usr/bin/env python
"""在新设备上一键完成项目引导。

做四件事：

1. **依赖体检** —— 告诉你缺什么、怎么装，而不是让你跑到一半才遇到 ImportError
2. **冒烟测试** —— 生成占位数据集并跑通完整评测链路，证明环境真的能用
3. **路径自检** —— 打印解析后的路径，方便和数据实际位置对照
4. **git 初始化**（可选）—— 建仓库并做首次提交，方便迁到下一台设备

为什么需要这个脚本
------------------
换设备时最容易踩的坑不是"装依赖"，而是**装了依赖但没验证链路能跑通**，
或者**数据路径在另一台机器上对不上**。这个脚本把这两件事一次性做完。

用法::

    python scripts/bootstrap.py                 # 体检 + 冒烟测试
    python scripts/bootstrap.py --with-git      # 额外初始化 git 仓库
    python scripts/bootstrap.py --data-root /mnt/nas/HeliAir   # 顺便自检数据路径
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT.parent) not in sys.path:
    sys.path.insert(0, str(ROOT.parent))


def hr(title: str = "") -> None:
    print("\n" + "=" * 74)
    if title:
        print(title)
        print("=" * 74)


def step_doctor() -> bool:
    hr("步骤 1/4  依赖体检")
    sys.path.insert(0, str(ROOT))
    from tools.main import cmd_doctor  # noqa: E402

    class _Args:
        pass

    # cmd_doctor 的返回值：0 = 核心依赖完整
    return cmd_doctor(_Args()) == 0


def step_smoke() -> bool:
    hr("步骤 2/4  冒烟测试：生成占位数据集并跑通评测链路")
    py = sys.executable

    demo_dir = ROOT / "data" / "demo"
    if (demo_dir / "annotations" / "instances_test.json").exists():
        print(f"占位数据集已存在，跳过生成：{demo_dir}")
    else:
        r = subprocess.run(
            [py, str(ROOT / "tools" / "main.py"), "make-demo", "--out", str(demo_dir)],
            cwd=str(ROOT),
        )
        if r.returncode != 0:
            print("❌ 生成占位数据集失败")
            return False

    print("\n--- 指标实现自检 ---")
    r = subprocess.run([py, str(ROOT / "tools" / "main.py"), "selftest"], cwd=str(ROOT))
    if r.returncode != 0:
        print("❌ 自检未通过。评测实现有问题，先别急着跑真实实验。")
        return False

    print("\n--- 端到端评测（占位检测器）---")
    r = subprocess.run(
        [
            py,
            str(ROOT / "tools" / "main.py"),
            "eval",
            "--config",
            str(ROOT / "configs" / "example.yaml"),
        ],
        cwd=str(ROOT),
    )
    if r.returncode != 0:
        print("❌ 端到端评测失败")
        return False

    print("\n--- 汇总出论文表格 ---")
    r = subprocess.run(
        [
            py,
            str(ROOT / "tools" / "main.py"),
            "report",
            "--runs",
            str(ROOT / "runs"),
            "--out",
            str(ROOT / "report.md"),
            "--latex",
        ],
        cwd=str(ROOT),
    )
    if r.returncode != 0:
        print("❌ 结果汇总失败")
        return False

    print("\n✅ 全链路跑通：数据 -> 评测 -> 分层统计 -> 落盘 -> 论文表格")
    return True


def step_paths(data_root: str | None, config: str | None) -> bool:
    hr("步骤 3/4  路径自检")
    cfg_path = Path(config) if config else (ROOT / "configs" / "example.yaml")
    if not cfg_path.exists():
        print(f"⚠️ 找不到配置 {cfg_path}，跳过路径自检")
        return True

    py = sys.executable
    cmd = [py, str(ROOT / "tools" / "main.py"), "eval", "--config", str(cfg_path), "--dry-run"]
    if data_root:
        cmd += ["--data-root", data_root]
    r = subprocess.run(cmd, cwd=str(ROOT))
    if r.returncode != 0:
        print("\n⚠️ 路径自检未通过。")
        print("   这通常意味着数据还没拷到这台设备上，或者位置和配置里写的不一样。")
        print("   数据到位后用 --data-root 指向它即可，不需要改配置文件：")
        print(f"     python tools/main.py eval --config {cfg_path.name} --data-root <新位置>")
        return False
    return True


def step_git(with_git: bool) -> bool:
    hr("步骤 4/4  git 仓库")
    if not with_git:
        print("跳过（加 --with-git 可初始化仓库）")
        print("\n建议：用 git 管理源码，这是迁移到新设备最省事的方式。")
        print("      data/ 与 runs/ 已在 .gitignore 里排除，所以仓库很小（<200 KB）。")
        return True

    if (ROOT / ".git").exists():
        print("已经是 git 仓库，跳过初始化。")
        r = subprocess.run(["git", "status", "--short"], cwd=str(ROOT))
        return r.returncode == 0

    try:
        subprocess.run(["git", "--version"], check=True, capture_output=True)
    except Exception:
        print("⚠️ 未找到 git，跳过。请先安装 git。")
        return False

    commands = [
        ["git", "init", "-b", "main"],
        ["git", "add", "-A"],
        ["git", "-c", "user.name=bootstrap", "-c", "user.email=bootstrap@local",
         "commit", "-m", "DetBench: 初始提交（小目标检测可复现评测框架）"],
    ]
    for cmd in commands:
        r = subprocess.run(cmd, cwd=str(ROOT))
        if r.returncode != 0:
            print(f"❌ 执行失败：{' '.join(cmd)}")
            return False

    print("\n✅ git 仓库已建立。")
    print("\n迁到新设备的最省事方式：")
    print("  1. 在有远程仓库的情况下： git remote add origin <url> && git push -u origin main")
    print("  2. 没有远程仓库时，直接打包源码即可（很小）：")
    print("       git archive --format=zip -o detbench.zip HEAD")
    print("     或者用 bundle 把历史一起带走：")
    print("       git bundle create detbench.bundle --all")
    return True


def main() -> int:
    ap = argparse.ArgumentParser(
        description="DetBench 新设备引导脚本",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    ap.add_argument("--with-git", action="store_true", help="额外初始化 git 仓库并首次提交")
    ap.add_argument("--data-root", default=None, help="顺便用这个数据根做路径自检")
    ap.add_argument("--config", default=None, help="自检用的配置文件（默认 configs/example.yaml）")
    ap.add_argument("--skip-smoke", action="store_true", help="跳过冒烟测试（只做体检）")
    args = ap.parse_args()

    print("DetBench 新设备引导")
    print(f"项目目录：{ROOT}")

    ok_doctor = step_doctor()
    ok_smoke = True if args.skip_smoke else step_smoke()
    ok_paths = step_paths(args.data_root, args.config)
    ok_git = step_git(args.with_git)

    hr("总结")
    print(f"  依赖体检   : {'✅ 通过' if ok_doctor else '❌ 缺少必需依赖'}")
    print(f"  冒烟测试   : {'跳过' if args.skip_smoke else ('✅ 通过' if ok_smoke else '❌ 失败')}")
    print(f"  路径自检   : {'✅ 通过' if ok_paths else '⚠️ 数据尚未就位'}")
    print(f"  git 仓库   : {'✅ 就绪' if ok_git else '⚠️ 未就绪'}")

    if not ok_doctor:
        print("\n先解决依赖问题：pip install numpy pillow")
        return 1
    if not (ok_smoke):
        print("\n冒烟测试失败。先把链路打通再考虑真实实验。")
        return 1

    print("\n下一步：")
    if not ok_paths:
        print("  1. 把数据拷到这台设备，用 --data-root 指向它（不用改配置文件）")
    print("  2. 准备真实数据的配置：cp configs/example.yaml configs/exp01.yaml，改 dataset 段")
    print("  3. 先 dry-run 确认路径：python tools/main.py eval --config configs/exp01.yaml --dry-run")
    print("  4. 正式评测：python tools/main.py eval --config configs/exp01.yaml")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
