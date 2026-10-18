# -*- coding: utf-8 -*-
"""play_combat.py —— 让训练好的模型实际玩游戏给你看

用法：
    python play_combat.py --model models/multi_355000_steps.zip
    python play_combat.py --model models/xxx.zip --stochastic   # 用采样动作（更像训练时）

会用确定性策略（取最大概率动作）驱动所有已开着的实例，你直接看窗口就行。
"""
import argparse
import sys
import time

try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

import multi_instance as mi
from stable_baselines3 import PPO
from train_multi import build_envs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="models/multi_355000_steps.zip")
    ap.add_argument("--stochastic", action="store_true")
    ap.add_argument("--catalog", default="room_catalog.json")
    args = ap.parse_args()

    model = PPO.load(args.model, device="cpu")
    print(f"模型已加载: {args.model}")

    mgr = mi.InstanceManager(n=1)
    mgr.attach()
    if not mgr.wait_connections(120):
        print("没有实例在对局里 —— 请先在每个游戏窗口里进入一局")
        mgr.stop_all()
        return 1
    mgr.resolve_mapping()

    envs = build_envs(mgr, combat=True, catalog=args.catalog)
    print(f"\n开始让模型玩（{len(envs)} 个实例，Ctrl+C 停止）")
    print("画面里以撒的动作就是这个模型的真实水平。\n")

    obs_list = []
    for k, e in enumerate(envs):
        o, info = e.reset()
        obs_list.append(o)
        print(f"  实例{k}: 进入房间 variant={info.get('variant')}")

    ep, rew, steps = [0] * len(envs), [0.0] * len(envs), [0] * len(envs)
    try:
        while True:
            for k, e in enumerate(envs):
                a, _ = model.predict(obs_list[k],
                                     deterministic=not args.stochastic)
                obs, r, term, trunc, info = e.step(a)
                rew[k] += r
                steps[k] += 1
                if term or trunc:
                    print(f"  实例{k} 回合{ep[k] + 1}: {steps[k]} 步  "
                          f"清房={info.get('clear')} 死亡={info.get('dead')}  "
                          f"奖励={rew[k]:+.2f}")
                    obs, info = e.reset()
                    ep[k] += 1
                    rew[k], steps[k] = 0.0, 0
                obs_list[k] = obs
    except KeyboardInterrupt:
        print("\n停止")
    finally:
        for e in envs:
            e.close()
        mgr.stop_all()
    return 0


if __name__ == "__main__":
    sys.exit(main())
