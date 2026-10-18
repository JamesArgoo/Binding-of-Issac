# -*- coding: utf-8 -*-
"""diag_combat.py —— 量化诊断训练出来的战斗AI到底差在哪

针对用户观察到的现象逐条量化：
  "乱走动"        -> 统计移动/射击动作分布（如果某个动作占 90%，说明策略退化成常数）
  "会撞到怪物"    -> 统计玩家和敌人接触的步数比例 + 受伤事件
  "打人完全瞄不准" -> 统计发弹数 / 伤害 / 命中率

用法：
    python diag_combat.py --model models/combat_v2.zip --episodes 6
"""
import argparse
import sys
import time
from collections import Counter

import numpy as np

try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

import multi_instance as mi
from combat_env import CombatEnv, DIRS8, SHOOT_DIRS
from train_multi import load_variants

MOVE_NAMES = ["右", "右下", "下", "左下", "左", "左上", "上", "右上", "停"]
SHOOT_NAMES = ["不射", "上", "右", "下", "左"]


def contact_count(s):
    """玩家和敌人重叠的数量（距离 < 双方碰撞半径之和）。"""
    if not s or "player" not in s:
        return 0
    p = s["player"]
    ps = float(p.get("size", 10.0) or 10.0)
    n = 0
    for e in s.get("entities", []):
        if not (e.get("enemy") or e.get("boss")):
            continue
        r = ps + float(e.get("size", 10.0) or 10.0)
        if (e["x"] - p["x"]) ** 2 + (e["y"] - p["y"]) ** 2 <= r * r:
            n += 1
    return n


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="models/combat_v2.zip")
    ap.add_argument("--episodes", type=int, default=6)
    ap.add_argument("--catalog", default="room_catalog.json")
    args = ap.parse_args()

    from stable_baselines3 import PPO
    model = PPO.load(args.model, device="cpu")
    print(f"模型 {args.model}")

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
                    frame_skip=2, max_steps=900)

    move_cnt = Counter()
    shoot_cnt = Counter()
    total_steps = 0
    contact_steps = 0
    hurt_events = 0
    hurt_total = 0.0
    dealt_total = 0.0
    fired_total = 0.0
    cleared = 0
    died = 0
    ep_rewards = []

    obs, info = env.reset()
    prev_hp = None
    for ep in range(args.episodes):
        ep_rew = 0.0
        while True:
            a, _ = model.predict(obs, deterministic=True)
            obs, r, term, trunc, info = env.step(a)
            ep_rew += r
            total_steps += 1
            move_cnt[int(a[0])] += 1
            shoot_cnt[int(a[1])] += 1

            s = env.last_state
            if s and "player" in s:
                if contact_count(s) > 0:
                    contact_steps += 1
                hp = s["player"]["hp"] + s["player"]["soul_hp"]
                if prev_hp is not None and hp < prev_hp:
                    hurt_events += 1
                    hurt_total += prev_hp - hp
                prev_hp = hp

            dealt_total += info.get("dealt", 0.0)
            fired_total += info.get("fired", 0.0)
            if term or trunc:
                if info.get("clear"):
                    cleared += 1
                if info.get("dead"):
                    died += 1
                break
        ep_rewards.append(ep_rew)
        print(f"  回合{ep + 1}: 奖励={ep_rew:+.2f} 清房={info.get('clear')} 死亡={info.get('dead')}")
        obs, info = env.reset()
        prev_hp = None

    n = args.episodes
    print(f"\n==== 诊断结果（{n} 回合 / {total_steps} 步）====")
    print(f"清房 {cleared}/{n}   死亡 {died}/{n}   平均奖励 {np.mean(ep_rewards):+.2f}")

    print(f"\n[会撞到怪物]")
    print(f"  和敌人重叠的步数: {contact_steps}/{total_steps} = "
          f"{contact_steps / max(total_steps, 1):.1%}")
    print(f"  受伤事件 {hurt_events} 次，共掉 {hurt_total:.0f} 点血")
    print(f"  平均每回合掉血 {hurt_total / n:.1f} 点（满血 6 点）")

    print(f"\n[打人瞄不准]")
    print(f"  发弹总数 {fired_total:.0f}，造成伤害 {dealt_total:.0f}")
    per_hit = 3.5
    if fired_total > 0:
        print(f"  等效命中率: {min(1.0, dealt_total / per_hit / fired_total):.1%}"
              f"（伤害/{per_hit} 点 = 命中发数）")

    print(f"\n[乱走动] 移动动作分布:")
    for i in range(9):
        c = move_cnt.get(i, 0)
        if c:
            print(f"    {MOVE_NAMES[i]:<3} {c / total_steps:6.1%}  {'#' * int(c / total_steps * 50)}")
    print(f"\n[乱走动] 射击动作分布:")
    for i in range(5):
        c = shoot_cnt.get(i, 0)
        if c:
            print(f"    {SHOOT_NAMES[i]:<3} {c / total_steps:6.1%}  {'#' * int(c / total_steps * 50)}")

    env.close()
    mgr.stop_all()
    return 0


if __name__ == "__main__":
    sys.exit(main())
