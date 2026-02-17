"""MlxVadSRT GUI 启动入口"""

import os
import sys
import subprocess

if sys.platform == "darwin" and "PATH" in os.environ:
    try:
        # 启动一个登录 Shell (zsh -l) 获取真实的 PATH
        user_shell = os.environ.get("SHELL", "/bin/zsh")
        user_path = subprocess.check_output(
            [user_shell, "-l", "-c", "echo $PATH"],
            text=True,
            stderr=subprocess.DEVNULL
        ).strip()

        # 将获取到的 PATH 应用到当前进程
        os.environ["PATH"] = user_path
        print(f"已同步系统 PATH: {len(user_path)} chars")
    except Exception as e:
        print(f"警告: 同步 Shell PATH 失败: {e}")

# 确保项目根目录在 sys.path 中
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from gui.app import run

if __name__ == "__main__":
    run()
