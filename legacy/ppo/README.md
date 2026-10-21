# 旧 PPO 实验（已归档）

这里是项目早期用强化学习打以撒的代码，**当前 AI 不使用，也不再维护**。
保留它只是为了对照实验结果，历史记录见 `../../docs/EXPERIMENTS.md`。

运行这些脚本需要额外安装 `requirements.txt` 里的 torch / stable-baselines3 等依赖，
并先进入本目录：

```bash
cd legacy/ppo
python train_multi.py --attach --steps 20000
python eval_combat.py --model ../../models/best_combat_r5.zip --episodes 10
```

目录里各文件的作用：

| 文件 | 用途 |
|---|---|
| `combat_env.py` | 单房间战斗 Gym 环境：reset/goto、观测、奖励 |
| `isaac_env.py` | 状态 → 神经网络观测的构建 |
| `isaac_bridge.py` | 状态解析（敌人、敌方子弹、Pickup） |
| `isaac_data.py` | 离线演示用的合成房间/状态 |
| `room_catalog.py` | 探测 `goto d.<variant>` 有敌人的房间，生成 `room_catalog.json` |
| `train_multi.py` | 多实例并行 PPO 训练 |
| `eval_combat.py` | 量化评估模型（清房率、命中率、受伤） |
| `diag_combat.py` | 逐项诊断模型行为 |
| `test_combat.py` | 战斗环境自检 |
| `play_combat.py` | 用训练好的模型实机游玩 |
| `bc_pretrain.py` | 行为克隆预训练 |
| `oracle_eval.py` | 规则教师 / 随机策略的上下限对照 |
| `inspect_policy.py` | 检查模型权重和输出分布 |
| `selftest_learn.py` | 学习流程自检 |
| `dump_state.py` | 打印原始状态字段 |
| `play_rule.py` / `ai_basic_4.py` | 旧规则 AI（8 方向移动 + 4 方向射击） |
| `viz_viewer.py` | 状态可视化查看器 |
| `test_speed.py` + `turbo_*.c` | 游戏加速注入实验 |
| `_bootstrap.py` | 让本目录脚本能找到仓库里的 `isaac_ai` 包 |
