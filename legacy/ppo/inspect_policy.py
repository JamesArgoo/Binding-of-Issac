# -*- coding: utf-8 -*-
"""inspect_policy.py —— 不开游戏，直接看模型"想干什么"

回答一个问题：**这个 checkpoint 的策略到底是均匀随机，还是已经偏向某些动作了？**

做法：把 frames.json（或合成房间）里的真实状态喂进模型，读它的动作概率分布和
价值估计。不需要启动游戏、不需要 --luadebug、不需要游戏在对局中。

用法：
    python inspect_policy.py                       # 用最新的模型 + frames.json
    python inspect_policy.py --model models/x.zip --frames frames.json
    python inspect_policy.py --demo                # 不用 frames.json，用合成房间

看什么：
  * 熵 / 上限  ——  接近 100% 说明还基本是随机策略
  * 移动/射击分布  ——  如果某个动作占 90%，那是退化成常数（另一个极端）
  * V(s)  ——  价值网络的输出，长期贴在 0 附近说明 critic 也没学到东西
"""
import argparse
import json
import math
import os
import sys

import numpy as np

try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

MOVE_NAMES = ["右", "右下", "下", "左下", "左", "左上", "上", "右上", "停"]
SHOOT_NAMES = ["不射", "上", "右", "下", "左"]


def entropy(p):
    p = np.asarray(p, dtype=float)
    p = p[p > 0]
    return float(-(p * np.log(p)).sum())


def bar(p, width=40):
    return "#" * int(round(p * width))


def main():
    ap = argparse.ArgumentParser(description="离线检查策略分布")
    ap.add_argument("--model", default="latest",
                    help="模型路径；默认 latest = models/ 下最新 checkpoint")
    ap.add_argument("--frames", default="frames.json",
                    help="dump_state.py --json 存的状态帧（没有就用 --demo）")
    ap.add_argument("--demo", action="store_true", help="不用 frames.json，合成房间")
    ap.add_argument("--samples", type=int, default=2000,
                    help="拿不到概率时用采样估计分布的样本数")
    args = ap.parse_args()

    from train_multi import peek_obs_dim, resolve_model_path
    try:
        args.model = resolve_model_path(args.model)
    except FileNotFoundError as exc:
        print(exc)
        return 1

    import isaac_data as D
    from isaac_env import terrain_mode_from_dim

    dim = peek_obs_dim(args.model)
    terrain_mode = terrain_mode_from_dim(dim)
    print(f"模型   {args.model}")
    print(f"观测   {dim} 维（地形模式 {terrain_mode}）")

    from stable_baselines3 import PPO
    model = PPO.load(args.model, device="cpu")
    nvec = list(model.action_space.nvec)
    print(f"动作   {nvec}   （共 {int(np.prod(nvec))} 种组合）")

    # ---------- 收集若干帧观测 ----------
    states = []
    if args.demo or not os.path.exists(args.frames):
        if not args.demo:
            print(f"（{args.frames} 不存在，改用合成房间）")
        states = [D.demo_state(i / 15.0) for i in range(30)]
    else:
        with open(args.frames, encoding="utf-8") as f:
            data = json.load(f)
        states = [s for s in (data if isinstance(data, list) else [data])
                  if "player" in s]
        print(f"状态   {args.frames} 里 {len(states)} 帧")

    grids, cur = [], None
    for s in states:                     # 地形缓存：和真实环境一致
        if s.get("grid"):
            cur = D.normalize_grid(dict(s["grid"]))
        grids.append(cur)

    probs_move, probs_shoot, values, ents = [], [], [], []
    for s, g in zip(states, grids):
        obs = D.compute_obs(s, g, terrain_mode)
        if obs.shape[0] != dim:
            print(f"✗ 观测维度 {obs.shape[0]} != 模型 {dim}")
            return 1
        import torch
        t = torch.as_tensor(obs[None], dtype=torch.float32)
        with torch.no_grad():
            dist = model.policy.get_distribution(t)
            inner = dist.distribution
            # SB3 的 MultiDiscrete 里 .distribution 是「每个动作维一个 Categorical 的列表」
            if isinstance(inner, (list, tuple)):
                parts = [d.probs.cpu().numpy()[0] for d in inner]
                pm, ps = parts[0], parts[1]
            elif getattr(inner, "probs", None) is not None:
                p = inner.probs.cpu().numpy()[0]
                parts = np.split(p, np.cumsum(nvec)[:-1])
                pm, ps = parts[0], parts[1]
            else:                        # 兜底：采样估计
                acts = np.array([dist.get_actions().cpu().numpy()[0]
                                 for _ in range(args.samples)])
                pm = np.bincount(acts[:, 0], minlength=nvec[0]) / len(acts)
                ps = np.bincount(acts[:, 1], minlength=nvec[1]) / len(acts)
            v = model.policy.predict_values(t).cpu().numpy().ravel()[0]
        probs_move.append(pm)
        probs_shoot.append(ps)
        values.append(float(v))
        ents.append(entropy(pm) + entropy(ps))

    pm = np.mean(probs_move, axis=0)
    ps = np.mean(probs_shoot, axis=0)
    h_move, h_shoot = entropy(pm), entropy(ps)
    h_max = math.log(nvec[0]) + math.log(nvec[1])
    h_now = h_move + h_shoot

    print()
    print("=" * 62)
    print("① 随机程度（关键：接近 100% = 还是随机策略）")
    print("=" * 62)
    print(f"  移动熵 {h_move:.3f} / 上限 {math.log(nvec[0]):.3f}"
          f"  = {h_move / math.log(nvec[0]):.0%}")
    print(f"  射击熵 {h_shoot:.3f} / 上限 {math.log(nvec[1]):.3f}"
          f"  = {h_shoot / math.log(nvec[1]):.0%}")
    print(f"  合计   {h_now:.3f} / {h_max:.3f} = {h_now / h_max:.0%}"
          + ("   <-- 基本等于均匀随机" if h_now / h_max > 0.85 else
             "   <-- 已经开始有偏好" if h_now / h_max > 0.6 else
             "   <-- 策略已经明显收敛"))
    print(f"  （逐帧熵范围 {min(ents):.3f} ~ {max(ents):.3f}）")

    print()
    print("=" * 62)
    print("② 平均动作分布")
    print("=" * 62)
    print("  移动：")
    for i in range(nvec[0]):
        print(f"    {MOVE_NAMES[i] if i < len(MOVE_NAMES) else i:<3} "
              f"{pm[i]:6.1%}  {bar(pm[i])}")
    print("  射击：")
    for i in range(nvec[1]):
        print(f"    {SHOOT_NAMES[i] if i < len(SHOOT_NAMES) else i:<3} "
              f"{ps[i]:6.1%}  {bar(ps[i])}")

    # 极端情况提示
    for name, arr, names in (("移动", pm, MOVE_NAMES), ("射击", ps, SHOOT_NAMES)):
        top = int(np.argmax(arr))
        if arr[top] > 0.5:
            print(f"  ⚠ {name}有 {arr[top]:.0%} 集中在"
                  f"「{names[top] if top < len(names) else top}」"
                  f" —— 如果游戏画面里也是这个表现，说明策略退化成常数了")

    print()
    print("=" * 62)
    print("③ 价值网络")
    print("=" * 62)
    print(f"  V(s) 平均 {np.mean(values):+.4f}   "
          f"范围 {min(values):+.4f} ~ {max(values):+.4f}")
    print(f"  对输入变化的敏感度（标准差）{np.std(values):.4f}"
          + ("   <-- 几乎不随状态变化，critic 没学到东西" if np.std(values) < 1e-4 else ""))

    # 同一状态重复喂几次，确认是确定性的（不随机的模型应该每次一样）
    print()
    print("=" * 62)
    print("④ 确定性检查（同一状态喂 3 次）")
    print("=" * 62)
    obs = D.compute_obs(states[len(states) // 2], grids[len(states) // 2],
                        terrain_mode)
    import torch
    for k in range(3):
        a, _ = model.predict(obs, deterministic=True)
        print(f"  第{k + 1}次 deterministic 动作: 移动={MOVE_NAMES[int(a[0])]} "
              f"射击={SHOOT_NAMES[int(a[1])]}")
    picks = {}
    for k in range(20):
        a, _ = model.predict(obs, deterministic=False)
        picks[int(a[0])] = picks.get(int(a[0]), 0) + 1
    print(f"  随机采样 20 次选中过 {len(picks)} 种移动动作（越多越随机）")

    # ---------- ⑥ 归因：敌人换方向，动作跟不跟 ----------
    # 这是判断"到底学没学会瞄准"最直接的实验：把同一个敌人分别放到玩家的
    # 上/右/下/左，如果射击动作四个方向都一样，说明它压根没看敌人。
    print()
    print("=" * 62)
    print("⑤ 归因实验：敌人放在四个方向，看动作跟不跟")
    print("=" * 62)
    base = None
    for s, g in zip(states, grids):
        if s.get("entities") and g:
            base = (s, g)
            break
    if base is None:
        print("  没有带敌人+地形的帧，跳过")
        return 0
    s0, g0 = base
    p0 = s0["player"]
    aim = {"上": (0, -1), "右": (1, 0), "下": (0, 1), "左": (-1, 0)}
    want_shoot = {"上": 1, "右": 2, "下": 3, "左": 4}

    def shoot_with(enemy_xy, dist_from_player=150.0):
        st = dict(s0)
        ents = [e for e in s0["entities"] if not (e.get("enemy") or e.get("boss"))]
        dx, dy = enemy_xy
        ents.append({"id": 9001, "type": 10, "variant": 0, "subtype": 0,
                     "x": p0["x"] + dx * dist_from_player,
                     "y": p0["y"] + dy * dist_from_player,
                     "vx": 0.0, "vy": 0.0, "hp": 20.0, "boss": False,
                     "enemy": True, "dead": False, "spawner_type": 0,
                     "size": 12.0, "height": 0.0, "coll": 1, "cdmg": 0.0,
                     "gcoll": 1, "grid_idx": p0.get("grid_idx", 0)})
        st["entities"] = ents
        o = D.compute_obs(st, g0, terrain_mode)
        a, _ = model.predict(o, deterministic=True)
        return int(a[0]), int(a[1])

    print("  (A) 瞄准测试：单个敌人放在 150 单位外")
    hit = 0
    for name, (dx, dy) in aim.items():
        mv, sh = shoot_with((dx, dy))
        ok = (sh == want_shoot[name])
        hit += ok
        dist = math.hypot(dx, dy) * 150.0
        print(f"    敌人在{name:<2} (距离 {dist:.0f})  ->  移动={MOVE_NAMES[mv]:<3} "
              f"射击={SHOOT_NAMES[sh]:<3}  {'✓ 对' if ok else '✗ 错'}"
              f"（期望射击{name}）")
    print(f"  四方向命中 {hit}/4"
          + ("   <-- 完全没学会瞄准，射击动作和敌人位置无关"
             if hit <= 1 else ""))

    print()
    print("  (B) 距离测试：敌人在右边，看会不会主动靠近到 ~150 单位")
    for d in (60.0, 150.0, 400.0):
        mv, sh = shoot_with((1, 0), d)
        print(f"    敌人距离 {d:>5.0f}  ->  移动={MOVE_NAMES[mv]:<3} "
              f"({'靠近' if mv in (0, 1, 7) else '远离/横向'})  "
              f"射击={SHOOT_NAMES[sh]}")
    print("  期望：60 太近应该远离，400 太远应该靠近，150 附近应该稳住/绕圈")
    return 0


if __name__ == "__main__":
    sys.exit(main())
