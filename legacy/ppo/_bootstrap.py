# -*- coding: utf-8 -*-
"""让 legacy/ppo 下的脚本可以直接运行。

这些脚本仍在同一个目录里互相 import（isaac_env、combat_env 等），
但 multi_instance 已经搬进 ``isaac_ai`` 包，所以要先把仓库根目录
放进 sys.path。

用法（在仓库根目录或 legacy/ppo 里都可以）：
    python legacy/ppo/train_multi.py ...
    python train_multi.py ...
"""
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))

for _path in (ROOT, HERE):
    if _path not in sys.path:
        sys.path.insert(0, _path)
