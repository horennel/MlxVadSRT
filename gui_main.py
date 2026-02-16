"""MlxVadSRT GUI 启动入口"""

import os
import sys

# 确保项目根目录在 sys.path 中
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from gui.app import run

if __name__ == "__main__":
    run()
