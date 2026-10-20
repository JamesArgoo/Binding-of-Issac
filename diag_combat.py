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
    ap.add_argument("--model", default="latest",
                    help="模型路径；默认 latest = models/ 下最新的 checkpoint")
    ap.add_argument("--episodes", type=int, default=6)
    ap.add_argument("--catalog", default="room_catalog.json")
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
    print(f"模型 {args.model}（地形模式 {terrain_mode}，观测 {dim} 维）")

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
    ep_lens = []
    # 关键诊断：拿不到玩家状态的步数（菜单/死亡结算/换房转场）
    # 这些步 reward=0 且不终止，会稀释梯度，却看不出来
    nostate_steps = 0
    nostate_rew = 0.0
    comp = Counter()          # 奖励成分累计（原始值，未缩放）
    obs_min = np.full(model.observation_space.shape[0], np.inf)
    obs_max = np.full(model.observation_space.shape[0], -np.inf)

    obs, info = env.reset()
    prev_hp = None
    ep_steps = 0
    for ep in range(args.episodes):
        ep_rew = 0.0
        ep_steps = 0
        while True:
            a, _ = model.predict(obs, deterministic=True)
            obs, r, term, trunc, info = env.step(a)
            ep_rew += r
            total_steps += 1
            ep_steps += 1
            move_cnt[int(a[0])] += 1
            shoot_cnt[int(a[1])] += 1
            np.minimum(obs_min, obs, out=obs_min)
            np.maximum(obs_max, obs, out=obs_max)

            comp["dealt"] += info.get("dealt", 0.0)
            comp["kills"] += info.get("kills", 0)
            comp["fired"] += info.get("fired", 0)
            comp["hurt"] += info.get("hurt", 0.0)
            comp["wall"] += info.get("wall", 0)
            if info.get("clear"):
                comp["clear"] += 1

            s = env.last_state
            if s and "player" in s:
                if contact_count(s) > 0:
                    contact_steps += 1
                hp = s["player"]["hp"] + s["player"]["soul_hp"]
                if prev_hp is not None and hp < prev_hp:
                    hurt_events += 1
                    hurt_total += prev_hp - hp
                prev_hp = hp
            else:
                nostate_steps += 1
                nostate_rew += r

            dealt_total += info.get("dealt", 0.0)
            fired_total += info.get("fired", 0.0)
            if term or trunc:
                if info.get("clear"):
                    cleared += 1
                if info.get("dead"):
                    died += 1
                break
        ep_rewards.append(ep_rew)
        ep_lens.append(ep_steps)
        print(f"  回合{ep + 1}: {ep_steps} 步  奖励={ep_rew:+.2f} "
              f"清房={info.get('clear')} 死亡={info.get('dead')}")
        obs, info = env.reset()
        prev_hp = None

    n = args.episodes
    print(f"\n==== 诊断结果（{n} 回合 / {total_steps} 步）====")
    print(f"清房 {cleared}/{n}   死亡 {died}/{n}   "
          f"平均奖励 {np.mean(ep_rewards):+.2f}   平均长度 {np.mean(ep_lens):.0f} 步")

    # ---- 奖励成分（这是判断"信号有没有到网络"的关键）----
    print(f"\n[奖励成分] 原始值累计 / 折算到每步（整体再 x0.1）")
    print(f"  命中伤害  dealt={comp['dealt']:.0f}  -> +{comp['dealt'] / 5.0:.2f}")
    print(f"  击杀      kills={comp['kills']}    -> +{comp['kills'] * 0.25:.2f}")
    print(f"  受伤      hurt={comp['hurt']:.2f}    -> -{comp['hurt']:.2f}")
    print(f"  顶墙      wall={comp['wall']} 步   -> -{comp['wall'] * 0.6 * 2 / 60:.2f}")
    print(f"  清房      clear={comp['clear']}       -> +{comp['clear'] * 40.0:.2f}")
    time_cost = total_steps * 2.0 * (2 / 60)
    print(f"  时间代价  {total_steps} 步 @2.0/秒  -> -{time_cost:.2f}"
          f"   <-- 早期命中率=0 时按最贵算，通常压倒其它项")

    # ---- 拿不到状态的步数 ----
    print(f"\n[无效步] 拿不到玩家状态（菜单/死亡结算/转场）")
    print(f"  {nostate_steps}/{total_steps} = {nostate_steps / max(total_steps, 1):.1%}"
          f"   这些步 reward=0 且不终止，纯粹稀释梯度")
    print(f"  它们贡献的奖励合计 {nostate_rew:+.3f}")

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

    print(f"\n[乱走动] 移动动作分布（均匀随机应各约 11%）:")
    for i in range(9):
        c = move_cnt.get(i, 0)
        if c:
            print(f"    {MOVE_NAMES[i]:<3} {c / total_steps:6.1%}  {'#' * int(c / total_steps * 50)}")
    print(f"\n[乱走动] 射击动作分布（均匀随机应各 20%）:")
    for i in range(5):
        c = shoot_cnt.get(i, 0)
        if c:
            print(f"    {SHOOT_NAMES[i]:<3} {c / total_steps:6.1%}  {'#' * int(c / total_steps * 50)}")

    # ---- 策略熵：判断"还是随机"最直接的数 ----
    move_p = np.array([move_cnt.get(i, 0) for i in range(9)], dtype=float)
    shoot_p = np.array([shoot_cnt.get(i, 0) for i in range(5)], dtype=float)
    def _ent(p):
        p = p[p > 0] / p.sum()
        return float(-(p * np.log(p)).sum())
    h_move, h_shoot = _ent(move_p), _ent(shoot_p)
    print(f"\n[随机程度] 策略熵（越大越随机）")
    print(f"  移动 H={h_move:.3f} / 上限 {np.log(9):.3f} = {h_move / np.log(9):.0%}")
    print(f"  射击 H={h_shoot:.3f} / 上限 {np.log(5):.3f} = {h_shoot / np.log(5):.0%}")
    print(f"  合计 {h_move + h_shoot:.3f} / {np.log(9) + np.log(5):.3f} = "
          f"{(h_move + h_shoot) / (np.log(9) + np.log(5)):.0%}"
          f"   {'<- 基本等于随机策略' if (h_move + h_shoot) / (np.log(9) + np.log(5)) > 0.85 else ''}")

    # ---- 观测是否真的在变 ----
    const = int((obs_max - obs_min < 1e-6).sum())
    print(f"\n[观测] {len(obs_min)} 维里全程恒定的有 {const} 维"
          f"（{const / max(1, len(obs_min)):.0%}）")

    env.close()
    mgr.stop_all()
    return 0


if __name__ == "__main__":
    sys.exit(main())
