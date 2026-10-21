# -*- coding: utf-8 -*-
"""oracle_eval.py —— 测"命中率天花板"和"随机下限"

为什么需要它：
    训练了几十万步，命中率一直卡在 27~46% 来回摆。这时候必须分清两件事：
      (a) 是**策略没学会对齐射击**（奖励/信用分配的问题，可以调奖励去修）
      (b) 还是**这套"4 方向射击 + 泪弹飞行时间"的几何本来就只有这么高**
          （那就是指标/环境问题，再怎么调惩罚都是白费力气）

    做法：让两个**已知水平**的策略在同一套真实环境里打同样的房间：
      oracle —— bc_pretrain.rule_action 的规则老师（每步预演 9 个移动、朝敌人瞄）
               它"永远朝敌人所在轴开枪、永远往同一行/列挪"，是**策略层面几乎满分**的
      random —— 纯随机动作，代表"瞎打乱打"的垃圾下限

    同时统计**决策级对齐率**：开枪的那些步里，垂直偏移 <= --align-tol 的比例。
    把这三个数放在一起看就很清楚了：
      * oracle 对齐率高、命中率也高 → 天花板够高，是 PPO 没学会，去修奖励
      * oracle 对齐率很高、命中率却一样低 → 是泪弹飞行期敌人位移/判定几何的限制，
        继续加大对齐惩罚只会让模型不敢开枪（清房率掉），该换思路

用法：
    # 先让 test_multi 退出、确保游戏实例空着（同一端口只有一个客户端）
    python oracle_eval.py --episodes 20                      # 跑教师
    python oracle_eval.py --episodes 20 --policy random      # 跑随机下限
    python oracle_eval.py --episodes 20 --max-enemies 1      # 和训练课程对齐

注意：必须在"4 个实例空着"的时候跑，别和 train_multi --attach 同时用同一批端口。
"""
import argparse
import math
import sys

import numpy as np

try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

import multi_instance as mi
from bc_pretrain import rule_action
from combat_env import CombatEnv
from isaac_env import TERRAIN_FULL, TERRAIN_LOCAL, build_obs, obs_dim
from train_multi import load_variants

N_MOVE, N_SHOOT = 9, 5
SHOOT_DIRS = [(0, 0), (0, -1), (1, 0), (0, 1), (-1, 0)]   # 0不射 1上 2右 3下 4左
SHOOT_NAMES = ["不射", "上", "右", "下", "左"]

# 距离分桶：小于这个值归到第 k 桶（单位：世界像素，一格 40）
DIST_BINS = [80.0, 120.0, 180.0, 300.0, 1e9]
DIST_LABELS = ["<80", "80-120", "120-180", "180-300", ">300"]


BLOCKING = (2, 3, 4)
LOS_MAX_LONG = 40          # 整间房 28×16，扫 40 格足够


def los_long(px, py, grid, max_cells=LOS_MAX_LONG):
    """从 (px,py) 看 4 个正方向能通多少格（上/右/下/左）。

    不能直接用 bc_pretrain._los：那个只扫 15 格（=600 单位），
    远距离的敌人会被误判成"视线被挡"，把天花板分析全污染掉。
    """
    out = [max_cells] * 4
    if not grid:
        return out
    w, h = grid["w"], grid["h"]
    tw, th, ox, oy = grid["tw"], grid["th"], grid["ox"], grid["oy"]
    coll = grid.get("coll") or []
    gx, gy = int((px - ox) / tw), int((py - oy) / th)
    if not (0 <= gx < w and 0 <= gy < h):
        return out
    for k, (dx, dy) in enumerate(((0, -1), (1, 0), (0, 1), (-1, 0))):
        span = max_cells
        for step in range(1, max_cells + 1):
            cx, cy = gx + dx * step, gy + dy * step
            if not (0 <= cx < w and 0 <= cy < h):
                span = step - 1
                break
            i = cy * w + cx
            if (int(coll[i]) if i < len(coll) else 4) in BLOCKING:
                span = step - 1
                break
        out[k] = span
    return out


def shot_feasibility(px, py, ex, ey, evx, evy, shot_speed, grid,
                     through=0.5, radi=(18.0, 26.0)):
    """如果**现在**朝敌人所在轴开枪，泪弹飞到时能不能打中？

    这是"命中率天花板"的真正度量。关键在**提前量**：
        泪弹速度 shot_speed（单位/帧），距离 d → 飞行 T = d / shot_speed 帧；
        这 T 帧里敌人已经顺着 (evx, evy) 挪了 T 步。
    泪弹沿着开枪那一刻的**固定轴**飞，所以敌人只要在垂直方向漂出判定半径就打空。
    这正是"4 方向射击打不准"的物理根源。

    返回 dict：dist / T / off_pred / align18 / align26 / los_ok / hit18 / hit26
    """
    dx, dy = ex - px, ey - py
    dist = math.hypot(dx, dy)
    if dist < 1e-6 or shot_speed <= 0:
        return None
    horizontal = abs(dx) >= abs(dy)
    axis = (2 if dx > 0 else 4) if horizontal else (3 if dy > 0 else 1)

    # 飞行时间内敌人的预测位置（泪速是像素/帧，所以 T 就是帧数）
    T = dist / max(shot_speed, 1e-6)
    ex2, ey2 = ex + evx * T, ey + evy * T
    off_pred = abs(ey2 - py) if horizontal else abs(ex2 - px)

    los_ok = True
    if grid:
        span = los_long(px, py, grid)[axis - 1]
        tw = max(grid["tw"], 1e-6)
        los_ok = span + through >= dist / tw

    out = dict(dist=dist, T=T, off_pred=off_pred, los_ok=los_ok)
    for r in radi:
        aligned = off_pred <= r
        out[f"align{int(r)}"] = aligned
        out[f"hit{int(r)}"] = aligned and los_ok
    return out


def alignment_stats(state, shoot, align_tol):
    """这一步的开枪离"同轴"有多远。

    返回 (on_axis, off, wrong_axis)：
      on_axis   —— 垂直偏移 <= align_tol 且 选的是主轴方向（真能打到）
      off       —— 垂直偏移绝对值（像素）
      wrong_axis—— 开火了，但选的不是敌人所在的主轴（必然打空）
    """
    p = state["player"]
    ents = state.get("entities", [])
    enemies = [e for e in ents if e.get("enemy") or e.get("boss")]
    if not enemies or shoot <= 0:
        return None
    e = min(enemies, key=lambda e: (e["x"] - p["x"]) ** 2 + (e["y"] - p["y"]) ** 2)
    dx, dy = e["x"] - p["x"], e["y"] - p["y"]
    horizontal = abs(dx) >= abs(dy)
    off = abs(dy) if horizontal else abs(dx)
    want = (2 if dx > 0 else 4) if horizontal else (3 if dy > 0 else 1)
    wrong = int(shoot) != want
    return (off <= align_tol and not wrong), off, wrong


def run(args):
    mgr = mi.InstanceManager(n=1)
    mgr.attach()
    print("等实例连上 ...")
    if not mgr.wait_connections(args.timeout):
        print("超时：确认游戏窗口在对局里，且 test_multi / train_multi 没在占用同一批端口")
        return 1
    mgr.resolve_mapping()

    inst = mgr.instances[0]
    adapter = mi.ChannelAdapter(mgr.bridge.channels[inst.port])
    terrain = TERRAIN_FULL if args.full_terrain else TERRAIN_LOCAL
    variants = load_variants(args.catalog, args.min_enemies, args.max_enemies,
                             args.enemy_types, None)
    print(f"房间 variant: {len(variants)} 个   策略: {args.policy}   "
          f"回合数: {args.episodes}   align_tol={args.align_tol}")

    env = CombatEnv(adapter, inst.controller, variants=variants,
                    frame_skip=args.frame_skip, chunk=1, terrain_mode=terrain,
                    reset_mode=args.reset_mode, min_hp=args.min_hp,
                    seed=args.seed)
    rng = np.random.default_rng(args.seed)

    model = None
    models = {}          # --models 多模型轮流评测用
    if args.policy == "model":
        from stable_baselines3 import PPO
        from train_multi import resolve_model_path
        want = obs_dim(terrain)
        names = ([s.strip() for s in args.models.split(",") if s.strip()]
                 if args.models else [args.model])
        for nm in names:
            path = resolve_model_path(nm)
            m = PPO.load(path, device="cpu")
            got = int(np.prod(m.observation_space.shape))
            if got != want:
                print(f"⚠ {path} 观测 {got} 维 ≠ 环境 {want} 维（地形模式 {terrain}）")
                return 1
            models[nm] = m
            print(f"载入模型 {nm}（{got} 维）")
        model = models[names[0]]

    # 多模型时按回合轮转：同一批房间从小到大依次分给不同模型，
    # 这样"房间难度"这个最大噪声源被平均掉。
    # 单独跑两个进程各测一个模型是不行的：oracle_eval 每个进程会 attach 到
    # 不同的游戏实例，各实例的对局状态（道具/层数/属性）早就不同了 ——
    # 实测同一个模型在两次运行里能测出 41% 和 26% 的命中率。
    mnames = list(models.keys()) if models else []

    rows = []
    gate = {0: [0, 0], 1: [0, 0], 2: [0, 0]}    # 对齐分组: 未开枪 / 同轴 / 不同轴 -> [发数, ]
    on_axis_steps = off_axis_steps = no_shoot_steps = 0
    shots_total = 0
    # 距离桶 -> [步数, 预测可命中18, 预测可命中26, 视线通]
    bins = {k: [0, 0, 0, 0, 0] for k in range(len(DIST_BINS))}

    for ep in range(args.episodes):
        obs, _ = env.reset()
        steps = 0
        info = {}
        while True:
            s = env._state()
            if not s or "player" not in s:
                break
            if args.policy == "oracle":
                move, shoot = rule_action(s, env.grid or {}, tear_range=220.0)
            elif args.policy == "model":
                o = build_obs(s, env.grid, terrain)
                cur = mnames[ep % len(mnames)] if mnames else None
                use = models[cur] if cur else model
                a, _ = use.predict(o, deterministic=not args.stochastic)
                move, shoot = int(a[0]), int(a[1])
            else:
                move, shoot = int(rng.integers(N_MOVE)), int(rng.integers(N_SHOOT))

            # ---- 完美门控：只在"真的能打中"的那一步开枪 ----
            # 这不是一个能直接部署的策略（它作弊，提前知道敌人会不会走开），
            # 而是用来回答："如果开火时机被完美控制，命中率最高能到多少？"
            # 如果这个数也只有 40%，那 35% 就没什么提升空间了，别再调奖励；
            # 如果它能到 70%，说明是学习问题，值得继续投入。
            if args.gate:
                moved = False
                try:
                    p_, e_ = s["player"], s.get("entities", [])
                    ens_ = [e for e in e_ if e.get("enemy") or e.get("boss")]
                    if ens_:
                        e_n = min(ens_, key=lambda q: (q["x"] - p_["x"]) ** 2
                                  + (q["y"] - p_["y"]) ** 2)
                        fe_ = shot_feasibility(p_["x"], p_["y"], e_n["x"], e_n["y"],
                                               float(e_n.get("vx", 0) or 0),
                                               float(e_n.get("vy", 0) or 0),
                                               float(p_.get("shot_speed", 10) or 10),
                                               env.grid, through=0.5)
                        if fe_ and fe_[f"align{int(args.align_tol)}"]:
                            # 对齐了就朝敌人所在轴开火
                            dx_ = e_n["x"] - p_["x"]
                            dy_ = e_n["y"] - p_["y"]
                            if abs(dx_) >= abs(dy_):
                                shoot = 2 if dx_ > 0 else 4
                            else:
                                shoot = 3 if dy_ > 0 else 1
                            if args.gate_los and not fe_["los_ok"]:
                                shoot = 0
                            moved = shoot != 0
                        else:
                            shoot = 0
                    else:
                        shoot = 0
                except Exception:
                    pass
                if not moved:
                    shoot = 0

            # ---- 天花板分析：每步都算"现在开枪能不能中"，与动作无关 ----
            try:
                p, ents = s["player"], s.get("entities", [])
                ens = [e for e in ents if e.get("enemy") or e.get("boss")]
                if ens:
                    e = min(ens, key=lambda q: (q["x"] - p["x"]) ** 2
                            + (q["y"] - p["y"]) ** 2)
                    fe = shot_feasibility(p["x"], p["y"], e["x"], e["y"],
                                          float(e.get("vx", 0) or 0),
                                          float(e.get("vy", 0) or 0),
                                          float(p.get("shot_speed", 10) or 10),
                                          env.grid, through=0.5)
                    if fe:
                        for k, hi in enumerate(DIST_BINS):
                            if fe["dist"] < hi:
                                b = bins[k]
                                b[0] += 1
                                b[1] += 1 if fe["align18"] else 0
                                b[2] += 1 if fe["align26"] else 0
                                b[3] += 1 if fe["hit18"] else 0
                                b[4] += 1 if fe["los_ok"] else 0
                                break
            except Exception:
                pass

            if shoot > 0:
                shots_total += 1
                a = alignment_stats(s, shoot, args.align_tol)
                if a is not None:
                    ok, off, wrong = a
                    if wrong:
                        off_axis_steps += 1
                        gate[2][0] += 1
                    elif ok:
                        on_axis_steps += 1
                        gate[1][0] += 1
                    else:
                        off_axis_steps += 1
                        gate[2][0] += 1
            obs, r, term, trunc, info = env.step(np.array([int(move), int(shoot)]))
            steps += 1
            if term or trunc or steps >= args.max_steps:
                break

        dmg = float(info.get("ep_dealt", 0.0) or 0.0)
        fired = float(info.get("ep_fired", 0.0) or 0.0)
        hurt = float(info.get("hurt", 0.0) or 0.0)
        dm = max(float(s["player"].get("damage", 3.5) or 3.5), 0.5) if s and "player" in s else 3.5
        hits = dmg / dm
        rate = min(1.0, hits / fired) if fired > 0 else 0.0
        rows.append(dict(ep=ep + 1, steps=steps, dealt=dmg, fired=fired,
                         hits=hits, rate=rate, hurt=hurt,
                         who=(mnames[ep % len(mnames)] if mnames else args.policy),
                         clear=bool(info.get("clear")), dead=bool(info.get("dead"))))
        print(f"  回合 {ep+1:2d}"
              + (f" [{mnames[ep % len(mnames)].split(chr(92))[-1][:28]}]" if mnames else "")
              + f": {steps:4d} 步  伤害 {dmg:6.1f}  发弹 {fired:4.0f}"
              f"  命中 {rate*100:5.1f}%  受伤 {hurt:4.1f}"
              f"  {'清房' if info.get('clear') else '——'}"
              f"{'  死亡' if info.get('dead') else ''}")

    env.close()
    try:
        mgr.bridge.stop()
    except Exception:
        pass

    if not rows:
        print("没有跑出任何回合")
        return 1

    def avg(k):
        return sum(r[k] for r in rows) / len(rows)

    # 命中率按"总伤害/总发弹"汇总，比"每回合命中率平均"稳得多
    sum_dealt = sum(r["dealt"] for r in rows)
    sum_fired = sum(r["fired"] for r in rows)
    sum_hits = sum(r["hits"] for r in rows)
    agg_rate = sum_hits / sum_fired if sum_fired > 0 else 0.0
    on_axis_frac = on_axis_steps / shots_total if shots_total else 0.0

    print("\n" + "=" * 64)
    if len({r["who"] for r in rows}) > 1:
        # 多模型：按模型分组给同口径汇总（房间轮转，难度被平均掉）
        print(f"{'模型':<34}{'回合':>5}{'清房':>7}{'受伤':>7}{'发弹':>7}{'伤害':>7}{'命中率':>8}")
        for nm in mnames:
            sub = [r for r in rows if r["who"] == nm]
            if not sub:
                continue
            f_ = sum(r["fired"] for r in sub)
            h_ = sum(r["hits"] for r in sub)
            print(f"{nm.split(chr(92))[-1][:32]:<34}{len(sub):>5}"
                  f"{sum(r['clear'] for r in sub)/len(sub)*100:>6.0f}%"
                  f"{sum(r['hurt'] for r in sub)/len(sub):>7.2f}"
                  f"{sum(r['fired'] for r in sub)/len(sub):>7.1f}"
                  f"{sum(r['dealt'] for r in sub)/len(sub):>7.1f}"
                  f"{(h_/f_*100 if f_ else 0):>7.1f}%")
        print("=" * 64)
    print(f"策略 {args.policy}   回合 {len(rows)}")
    print(f"  平均步数     {avg('steps'):7.1f}")
    print(f"  清房率       {sum(r['clear'] for r in rows)/len(rows)*100:6.1f}%")
    print(f"  死亡回合     {sum(r['dead'] for r in rows)}")
    print(f"  平均受伤     {avg('hurt'):7.2f} 点")
    print(f"  平均发弹     {avg('fired'):7.1f} 发/回合")
    print(f"  平均伤害     {avg('dealt'):7.1f}")
    print(f"  **命中率**   {agg_rate*100:6.1f}%   (汇总: 伤害{sum_dealt:.0f}/发弹{sum_fired:.0f})")
    print(f"  **决策对齐率** {on_axis_frac*100:6.1f}%   （开枪步中真正同轴的比例）")
    print("=" * 64)

    # ---- 距离 → 命中率天花板（含提前量预测） ----
    tot = sum(b[0] for b in bins.values())
    if tot:
        print("\n距离 → 现在开枪的命中窗口（含提前量；判定半径 R=18/26 像素）")
        print(f"{'距离区间':>10} {'占比':>7} {'对齐R18':>8} {'对齐R26':>8}"
              f" {'视线通':>8} {'可命中':>8}")
        for k, lab in enumerate(DIST_LABELS):
            b = bins[k]
            if b[0] == 0:
                continue
            print(f"{lab:>10} {b[0]/tot*100:6.1f}% {b[1]/b[0]*100:7.1f}%"
                  f" {b[2]/b[0]*100:7.1f}% {b[4]/b[0]*100:7.1f}%"
                  f" {b[3]/b[0]*100:7.1f}%")
        near = sum(bins[k][3] for k in range(2))
        near_n = sum(bins[k][0] for k in range(2))
        far = sum(bins[k][3] for k in range(2, len(DIST_BINS)))
        far_n = sum(bins[k][0] for k in range(2, len(DIST_BINS)))
        if near_n and far_n:
            print(f"\n  近距(<120) 可命中 {near/near_n*100:.1f}%（{near_n} 步）"
                  f"    远距(>120) 可命中 {far/far_n*100:.1f}%（{far_n} 步）")
            if near / near_n > far / far_n * 1.5:
                print("  → 近距命中窗口明显更宽：该把交战距离 D_OPT 从 150 往下调")
            else:
                print("  → 距离不是主要瓶颈，别再动 D_OPT")
    print("=" * 64)
    if args.policy == "oracle":
        if on_axis_frac > 0.85 and agg_rate < 0.55:
            print("结论：教师对齐率很高但命中率仍低 → 是泪弹飞行期敌人位移/判定的几何上限，")
            print("      继续加大对齐惩罚只会让它不敢开枪，应该换思路（预判提前量/缩短交战距离）")
        elif on_axis_frac <= 0.85:
            print("结论：连教师都没做到同轴（可能被石头挡/被弹幕逼走）→ 先看地形和交战距离")
        else:
            print("结论：天花板明显高于训练中的 30% → 是 PPO 没学会，该继续修奖励/热启动")
    return 0


def main():
    ap = argparse.ArgumentParser(description="测命中率天花板（规则教师）和随机下限")
    ap.add_argument("--policy", default="oracle", choices=["oracle", "random", "model"])
    ap.add_argument("--model", default=None, help="--policy model 时的模型路径（可用 latest）")
    ap.add_argument("--models", default=None,
                    help="逗号分隔的多个模型，**同一个进程里按回合轮转评测**。"
                         "比较模型务必用这个：分两次进程跑会 attach 到不同的游戏实例，"
                         "各实例的道具/层数早就不同，结论会翻转")
    ap.add_argument("--stochastic", action="store_true", help="模型采样动作而不是取 argmax")
    ap.add_argument("--gate", action="store_true",
                    help="完美门控：只在「提前量预测能命中」的那一步开枪（测命中率天花板）")
    ap.add_argument("--gate-los", action="store_true",
                    help="门控时额外要求视线畅通（否则照样打石头）")
    ap.add_argument("--episodes", type=int, default=20)
    ap.add_argument("--max-steps", type=int, default=600)
    ap.add_argument("--frame-skip", type=int, default=1)
    ap.add_argument("--catalog", default="room_catalog.json")
    ap.add_argument("--min-enemies", type=int, default=None)
    ap.add_argument("--max-enemies", type=int, default=None)
    ap.add_argument("--enemy-types", default=None)
    ap.add_argument("--full-terrain", action="store_true")
    ap.add_argument("--reset-mode", default="auto", choices=["restart", "keep", "auto"])
    ap.add_argument("--min-hp", type=float, default=3.0)
    ap.add_argument("--align-tol", type=float, default=18.0)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--timeout", type=float, default=60.0)
    args = ap.parse_args()
    if args.enemy_types:
        args.enemy_types = [int(t) for t in args.enemy_types.split(",") if t.strip()]
    return run(args)


if __name__ == "__main__":
    sys.exit(main())
