# -*- coding: utf-8 -*-
"""play_ai.py —— 启动当前战术 AI 的入口。

真正的实现放在 ``isaac_ai/play_ai.py``；这里只做一层转发，
这样在仓库根目录直接运行 ``python play_ai.py`` 就行。

常用：
    python play_ai.py --attach              # 接管已开着的游戏
    python play_ai.py --launch 1            # 启动一个新实例
    python play_ai.py --duration 120        # 跑 2 分钟自动停
"""
import sys

from isaac_ai.play_ai import main

if __name__ == "__main__":
    sys.exit(main())
