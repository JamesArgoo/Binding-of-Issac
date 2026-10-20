# -*- coding: utf-8 -*-
"""eval_combat.py —— 量化评估训练好的单房间战斗模型

输出指标（每个回合一局）：
  清房率      房间清空的比例（最重要）
  存活率      没死的比例
  平均步数    回合长度（越短说明清得越快，而时间扣分）
  平均伤害    对敌人造成的总伤害
  平均命中率  泪弹命中率
  平均奖励    回合总奖励

用法：
    python eval_combat.py --model models/combat_v2.zip --episodes 10
    python eval_combat.py --model models/combat_v2.zip --episodes 10 --stochastic
"""
import argparse
import sys
import time

import numpy as np

try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

import multi_instance as mi
from combat_env import CombatEnv
from train_multi import load_variants


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="latest",
                    help="模型路径；默认 latest = models/ 下最新的 checkpoint")
    ap.add_argument("--episodes", type=int, default=10)
    ap.add_argument("--catalog", default="room_catalog.json")
    ap.add_argument("--stochastic", action="store_true", help="用随机策略而不是取最大概率动作")
    args = ap.parse_args()

    from train_multi import resolve_model_path
    try:
        args.model = resolve_model_path(args.model)
    except FileNotFoundError as exc:
        print(exc)
        return 1
    print(f"使用模型 {args.model}")

    from stable_baselines3 import PPO
    from isaac_env import TERRAIN_FULL, TERRAIN_LOCAL, obs_dim, terrain_mode_from_dim
    model = PPO.load(args.model, device="cpu")
    dim = int(model.observation_space.shape[0])
    try:
        terrain_mode = terrain_mode_from_dim(dim)
    except ValueError as exc:
        print(f"✗ {exc}")
        print(f"  这个模型（{dim} 维）是用旧版观测训的，和当前代码不兼容。"
              f"当前是 {obs_dim(TERRAIN_LOCAL)} 维（local）/ "
              f"{obs_dim(TERRAIN_FULL)} 维（full），请用新代码重新训练。")
        return 1
    print(f"模型 {args.model} 加载完成（地形模式 {terrain_mode}，观测 {dim} 维）")

    variants = load_variants(args.catalog)
    mgr = mi.InstanceManager(n=1)
    mgr.attach()
    if not mgr.wait_connections(60):
        print("没有实例在对局里")
        return 1
    mgr.resolve_mapping()

    inst = mgr.instances[0]
    ch = mgr.bridge.channels[inst.port]
    env = CombatEnv(mi.ChannelAdapter(ch), inst.controller, variants=variants,
                    frame_skip=2, max_steps=900, terrain_mode=terrain_mode)

    rows = []
    t0 = time.time()
    obs, info = env.reset()
    for ep in range(args.episodes):
        ep_rew = 0.0
        ep_dealt = 0.0
        ep_fired = 0.0
        steps = 0
        cleared = died = False
        while True:
            action, _ = model.predict(obs, deterministic=not args.stochastic)
            obs, r, term, trunc, info = env.step(action)
            ep_rew += r
            ep_dealt += info.get("dealt", 0.0)
            ep_fired += info.get("fired", 0.0)
            steps += 1
            if info.get("clear"):
                cleared = True
            if info.get("dead"):
                died = True
            if term or trunc:
                break

        rows.append(dict(rew=ep_rew, dealt=ep_dealt, fired=ep_fired,
                         steps=steps, cleared=cleared, died=died,
                         variant=info.get("variant")))
        print(f"  回合{ep + 1}: 步数={steps:3d} 伤害={ep_dealt:5.0f} "
              f"发弹={ep_fired:3.0f} 清房={cleared} 死亡={died} 奖励={ep_rew:+.2f}")
        obs, info = env.reset()

    n = len(rows)
    print(f"\n==== {args.episodes} 回合汇总（{time.time() - t0:.0f}s）====")
    print(f"  清房率:   {sum(r['cleared'] for r in rows)}/{n} = "
          f"{sum(r['cleared'] for r in rows) / n:.0%}")
    print(f"  存活率:   {sum(not r['died'] for r in rows)}/{n} = "
          f"{sum(not r['died'] for r in rows) / n:.0%}")
    print(f"  平均步数: {np.mean([r['steps'] for r in rows]):.0f}")
    print(f"  平均伤害: {np.mean([r['dealt'] for r in rows]):.0f}")
    print(f"  平均发弹: {np.mean([r['fired'] for r in rows]):.0f}")
    tot_dealt = sum(r['dealt'] for r in rows)
    tot_fired = sum(r['fired'] for r in rows)
    if tot_fired > 0:
        print(f"  等效命中率(伤害/发弹/单发伤害): 见各回合")
    print(f"  平均奖励: {np.mean([r['rew'] for r in rows]):+.3f}")

    env.close()
    mgr.stop_all()
    return 0


if __name__ == "__main__":
    sys.exit(main())
