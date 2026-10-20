# -*- coding: utf-8 -*-
"""play_combat.py —— 让训练好的模型实际玩游戏给你看

用法：
    python play_combat.py --model models/best_combat_r5          # 看最佳模型
    python play_combat.py --model models/best_combat_r5 --max-enemies 1
    python play_combat.py --stochastic                            # 用采样动作

会用确定性策略（取最大概率动作）驱动所有已开着的实例，你直接看窗口就行。

注意必须用 `--max-enemies` 限制房间难度：模型是按某个课程（比如 ≤1 或 ≤2 个敌人）
训的，直接扔进 16 个敌人的房间会秒死，看起来像"模型没用"，其实是分布不对。
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
    ap.add_argument("--model", default="models/best_combat_r5",
                    help="模型路径（默认用第 5 轮那个最佳模型）")
    ap.add_argument("--stochastic", action="store_true",
                    help="用采样动作而不是取最大概率动作")
    ap.add_argument("--catalog", default="room_catalog.json")
    ap.add_argument("--max-enemies", type=int, default=2,
                    help="只玩敌人数 <= N 的房间（默认 2，和训练课程对齐）")
    ap.add_argument("--min-enemies", type=int, default=None)
    ap.add_argument("--enemy-types", default=None,
                    help="只玩全部敌人都在这个类型列表里的房间，逗号分隔")
    ap.add_argument("--reset-mode", default="auto", choices=["restart", "keep", "auto"],
                    help="回合之间怎么切房（默认 auto：血量健康就快速 goto）")
    ap.add_argument("--min-hp", type=float, default=3.0)
    ap.add_argument("--hurt-scale", type=float, default=10.0)
    ap.add_argument("--instances", type=int, default=0,
                    help="只驱动前 N 个实例（0 = 全部）")
    args = ap.parse_args()

    from train_multi import resolve_model_path
    try:
        args.model = resolve_model_path(args.model)
    except FileNotFoundError as exc:
        print(exc)
        return 1
    print(f"使用模型 {args.model}")

    model = PPO.load(args.model, device="cpu")
    from isaac_env import (TERRAIN_FULL, TERRAIN_LOCAL, obs_dim,
                           terrain_mode_from_dim)
    dim = int(model.observation_space.shape[0])
    try:
        terrain_mode = terrain_mode_from_dim(dim)
    except ValueError as exc:
        print(f"✗ {exc}")
        print(f"  这个模型（{dim} 维）是用旧版观测训的，和当前代码不兼容。"
              f"当前是 {obs_dim(TERRAIN_LOCAL)} 维（local）/ "
              f"{obs_dim(TERRAIN_FULL)} 维（full），请用新代码重新训练。")
        return 1
    print(f"模型已加载: {args.model}（地形模式 {terrain_mode}，观测 {dim} 维）")

    mgr = mi.InstanceManager(n=1)
    mgr.attach()
    if not mgr.wait_connections(120):
        print("没有实例在对局里 —— 请先在每个游戏窗口里进入一局")
        mgr.stop_all()
        return 1
    mgr.resolve_mapping()

    if args.instances and args.instances > 0:
        for inst in mgr.instances[args.instances:]:
            ch = mgr.bridge.channels.get(inst.port)
            if ch is not None:
                ch.connected = False
        mgr.instances = mgr.instances[:args.instances]
        mgr.n = len(mgr.instances)
        print(f"只驱动前 {mgr.n} 个实例")

    envs = build_envs(mgr, combat=True, catalog=args.catalog,
                      terrain_mode=terrain_mode,
                      min_enemies=args.min_enemies,
                      max_enemies=args.max_enemies,
                      enemy_types=args.enemy_types,
                      reset_mode=args.reset_mode,
                      min_hp=args.min_hp,
                      hurt_scale=args.hurt_scale)
    print(f"\n开始让模型玩（{len(envs)} 个实例，Ctrl+C 停止）")
    print(f"  房间课程：敌人数 ≤ {args.max_enemies}"
          + (f"，类型 {args.enemy_types}" if args.enemy_types else ""))
    print("画面里以撒的动作就是这个模型的真实水平。\n")

    obs_list = []
    for k, e in enumerate(envs):
        o, info = e.reset()
        obs_list.append(o)
        print(f"  实例{k}: 进入房间 variant={info.get('variant')}")

    ep, rew, steps = [0] * len(envs), [0.0] * len(envs), [0] * len(envs)
    t0 = time.time()
    try:
        while True:
            for k, e in enumerate(envs):
                a, _ = model.predict(obs_list[k],
                                     deterministic=not args.stochastic)
                obs, r, term, trunc, info = e.step(a)
                rew[k] += r
                steps[k] += 1
                if term or trunc:
                    how = ("清房 ✅" if info.get("clear")
                           else "受伤终止" if info.get("hurt_fail")
                           else "死亡" if info.get("dead") else "超时")
                    print(f"  实例{k} 回合{ep[k] + 1}: {steps[k]:4d} 步  {how}  "
                          f"受伤 {info.get('hurt', 0):.2f}  输出 {info.get('ep_dealt', 0):.0f}  "
                          f"奖励 {rew[k]:+.2f}   [已跑 {(time.time() - t0) / 60:.1f} 分钟]")
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
