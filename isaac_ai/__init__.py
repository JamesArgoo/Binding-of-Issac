# -*- coding: utf-8 -*-
"""isaac_ai —— 当前可实机游玩的战术 AI。

三个模块，职责分开：

* ``tactical_agent``：只做决策，不碰窗口、不碰网络，纯标准库。
* ``multi_instance``：找窗口、连端口、发按键、管理多个游戏实例。
* ``play_ai``：把两者串起来的主循环（Runner）。

旧 PPO 训练/观测代码不在这里，见仓库根目录的 ``legacy/ppo/``。
"""
