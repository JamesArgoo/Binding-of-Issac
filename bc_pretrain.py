# -*- coding: utf-8 -*-
"""bc_pretrain.py —— 用规则给模型"热启动"瞄准能力

为什么需要它（实测结论，不是猜）：
    inspect_policy.py 的归因实验显示，从零 PPO 训出来的模型（45k / 300k / 355k 步
    都一样）对"敌人放在上/右/下/左"输出**几乎完全相同**的射击概率 —— 它根本没学会
    瞄准。而用同一套观测做监督学习（BC），15 个 epoch 就 4/4、每个方向 100% 正确。

    原因是探索/信用分配：PPO 自己的移动策略会把敌人方位压到很窄的分布里，
    射中又是稀疏奖励（每 10~20 步才中一次），梯度根本分不出"哪个方向更好"。
    瞄准这类一学就会、但靠随机探索碰不到的子技能，用规则监督是最划算的。

做法：
    1. 从真实帧（dump_state.py --json 存的）或合成房间取状态
    2. 对每个状态做**敌人位置增强**：把敌人摆到玩家周围很多方位/距离上
    3. 用解析规则打标签：射击 = 朝最近的敌人；移动 = 往"刚好打到"的距离靠/退
    4. 只对策略头做交叉熵训练（默认只训射击头，移动留给 PPO 自己学）
    5. 存成 SB3 模型，train_multi.py --load 接着 PPO 微调

用法：
    python bc_pretrain.py                                   # 合成房间，验证流程
    python bc_pretrain.py --frames frames.json               # 用真实地形/玩家属性
    python bc_pretrain.py --frames frames.json --out models/bc_seed.zip
    python train_multi.py --attach --combat --full-terrain --max-enemies 2 \
        --load models/bc_seed --steps 200000

注意：BC 只给策略一个"会瞄准"的起点，最终水平仍要靠 PPO 在真实奖励上微调。
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

import isaac_data as D
from isaac_env import (DIRS8, TERRAIN_FULL, TERRAIN_LOCAL, build_obs, obs_dim,
                       terrain_mode_from_dim)

SHOOT_DIRS = [(0, 0), (0, -1), (1, 0), (0, 1), (-1, 0)]   # 同 combat_env
MOVE_NAMES = ["右", "右下", "下", "左下", "左", "左上", "上", "右上", "停"]
SHOOT_NAMES = ["不射", "上", "右", "下", "左"]

D_OPT = 150.0     # 理想交战距离（和 combat_env 一致）
D_NEAR = 90.0     # 更近就往后退
STEP = 14.0       # 每步位移估计（实测真游戏约 12 单位/步）
PROJECTILE_TYPES = (2, 7, 8, 9)
LOS_MAX = 15.0
BLOCKING = (2, 3, 4)      # 网格碰撞类：物体/实心/墙 —— 挡子弹


def _walkable(x, y, grid):
    w, h = grid["w"], grid["h"]
    tw, th, ox, oy = grid["tw"], grid["th"], grid["ox"], grid["oy"]
    gx, gy = int((x - ox) / tw), int((y - oy) / th)
    if not (0 <= gx < w and 0 <= gy < h):
        return False
    cells = grid.get("cells")
    i = gy * w + gx
    return bool(cells[i]) if (cells and i < len(cells)) else False


def _los(px, py, grid):
    """从某个世界坐标点看 4 个方向（上/右/下/左）的视线长度（格）。"""
    out = [LOS_MAX] * 4
    if not grid:
        return out
    w, h = grid["w"], grid["h"]
    tw, th, ox, oy = grid["tw"], grid["th"], grid["ox"], grid["oy"]
    coll = grid.get("coll") or []
    gx, gy = int((px - ox) / tw), int((py - oy) / th)
    if not (0 <= gx < w and 0 <= gy < h):
        return out
    for k, (dx, dy) in enumerate(((0, -1), (1, 0), (0, 1), (-1, 0))):
        span = LOS_MAX
        for step in range(1, int(LOS_MAX) + 1):
            cx, cy = gx + dx * step, gy + dy * step
            if not (0 <= cx < w and 0 <= cy < h):
                span = step - 1
                break
            i = cy * w + cx
            c = int(coll[i]) if i < len(coll) else 4
            if c in BLOCKING:
                span = step - 1
                break
        out[k] = span
    return out


def _threat(bullet, x, y, margin=26.0, horizon=20.0):
    """这颗敌方子弹会不会在接下来 ~20 帧内打到 (x,y)。"""
    vx, vy = float(bullet.get("vx", 0) or 0), float(bullet.get("vy", 0) or 0)
    sp2 = vx * vx + vy * vy
    if sp2 < 0.25:
        return False
    rx, ry = x - bullet["x"], y - bullet["y"]
    t = (rx * vx + ry * vy) / sp2          # 最近接近时刻
    if t < 0 or t > horizon:
        return False
    cx, cy = bullet["x"] + vx * t, bullet["y"] + vy * t
    return math.hypot(x - cx, y - cy) < margin


def _shoot_axis(px, py, ex, ey):
    """朝敌人该按哪个射击键（1上 2右 3下 4左）。"""
    dx, dy = ex - px, ey - py
    if abs(dx) >= abs(dy):
        return 2 if dx > 0 else 4
    return 3 if dy > 0 else 1


def rule_action(state, grid, tear_range=220.0):
    """规则老师：射击朝最近敌人（视线被挡就不射）；移动按 1 步预演打分。

    移动打分同时考虑你提的四件事：
      1. **对齐**：让"非射击轴"上的偏移尽量小 —— 敌人斜着时先挪到同一行/列，
         否则泪弹只会沿着主轴飞过去打空
      2. **距离**：留在射程内、靠近 D_OPT
      3. **视线**：新位置的射击方向不能被石头/墙挡住
      4. **躲弹**：新位置会不会被敌方弹幕打到（垂直躲开再回来）
    """
    p = state["player"]
    px, py = p["x"], p["y"]
    ents = state.get("entities", [])
    enemies = [e for e in ents if e.get("enemy") or e.get("boss")]
    if not enemies:
        return 8, 0
    e = min(enemies, key=lambda e: (e["x"] - px) ** 2 + (e["y"] - py) ** 2)
    ex, ey = e["x"], e["y"]
    bullets = [b for b in ents
               if int(b.get("type", 0)) in PROJECTILE_TYPES
               and int(b.get("spawner_type", 0) or 0) != 1]

    # ---- 射击：始终朝最近敌人瞄准 ----
    # 不在这里判"视线被挡就不射"：那样标签里会塞满「不射」（真实房间石头很多），
    # BC 只会学成一个几乎不开枪的哑巴。瞄准是技能，开不开枪是决策 ——
    # 前者用 BC 教，后者交给 PPO 用命中率奖励去学（现在观测里有视线那 4 维了）。
    shoot = _shoot_axis(px, py, ex, ey)

    # ---- 移动：9 个动作各预演一步，取最高分 ----
    bdx, bdy = ex - px, ey - py
    horizontal = abs(bdx) >= abs(bdy)
    off_old = (py - ey) if horizontal else (px - ex)
    threat_now = any(_threat(b, px, py) for b in bullets)

    cands = list(enumerate(DIRS8)) + [(8, (0, 0))]
    best_i, best_s = 8, -1e18
    for i, (mx, my) in cands:
        nx, ny = px + mx * STEP, py + my * STEP
        if not _walkable(nx, ny, grid):
            continue
        nd = math.hypot(ex - nx, ey - ny)
        s = 0.0
        # 距离
        if nd > tear_range:
            s -= 6.0 + (nd - tear_range) / 40.0
        s -= abs(nd - D_OPT) / 90.0
        # 对齐（让非射击轴偏移变小）
        off_new = (ny - ey) if horizontal else (nx - ex)
        s -= abs(off_new) / 45.0
        if abs(off_new) <= 14.0:
            s += 1.5
        # 视线：新位置的射击轴别被挡（移动时就该往能打到的位置挪）
        los_new = _los(nx, ny, grid)
        if los_new[shoot - 1] + 0.5 < nd / max(grid["tw"], 1e-6):
            s -= 2.0
        # 躲弹：只有当前确实被威胁时才躲
        if threat_now:
            s += 1.0 if not any(_threat(b, nx, ny) for b in bullets) else -3.0
        if s > best_s:
            best_s, best_i = s, i
    return best_i, shoot


def _vec_to_move(ux, uy):
    """连续方向 -> DIRS8 索引（0=右 1=右下 2=下 3=左下 4=左 5=左上 6=上 7=右上）。"""
    ang = math.atan2(uy, ux)                  # y 向下为正
    idx = int(round(ang / (math.pi / 4))) % 8
    return idx


def _make_state(px, py, ex, ey, grid, evx=0.0, evy=0.0, shot_speed=10.0):
    """拼一个符合模组 schema 的最小状态，直接喂 build_obs。

    evx/evy 是**敌人的速度**：BC 的门控标签要用它算提前量（"这发飞到时还在轴上吗"）。
    shot_speed 是玩家的泪速（模组里实测约 10 单位/帧）。
    """
    w = grid["w"]
    tw, th, ox, oy = grid["tw"], grid["th"], grid["ox"], grid["oy"]
    idx = int((py - oy) / th) * w + int((px - ox) / tw)
    return {
        "frame": 0,
        "player": {
            "x": px, "y": py, "vx": 0.0, "vy": 0.0,
            "hp": 6.0, "max_hp": 6.0, "soul_hp": 0.0,
            "coins": 0, "bombs": 0, "keys": 0,
            "active_item": 0, "active_charge": 0, "pocket_item": 0, "card": 0,
            "damage": 3.5, "move_speed": 1.0, "tear_range": 260.0,
            "shot_speed": shot_speed, "fire_delay": 8.0, "size": 10.0,
            "grid_idx": idx, "fire_dir": -1, "head_dir": -1, "iframes": 0,
        },
        "room": {"type": 1, "index": -3, "enemies_alive": 1,
                 "bosses_alive": 0, "clear": False},
        "entities": [
            {"id": 1, "type": 1, "variant": 0, "subtype": 0, "x": px, "y": py,
             "vx": 0.0, "vy": 0.0, "hp": 6.0, "boss": False, "enemy": False,
             "dead": False, "spawner_type": 0, "size": 10.0, "height": 0.0,
             "coll": 4, "cdmg": 0.0, "gcoll": 1, "grid_idx": idx},
            {"id": 100, "type": 10, "variant": 0, "subtype": 0, "x": ex, "y": ey,
             "vx": evx, "vy": evy, "hp": 20.0, "boss": False, "enemy": True,
             "dead": False, "spawner_type": 0, "size": 12.0, "height": 0.0,
             "coll": 1, "cdmg": 0.0, "gcoll": 1, "grid_idx": 0},
        ],
        "doors": [],
    }


def pick_grid(frames):
    """从帧里挑一份真实地形；没有就用合成房间。"""
    for s in reversed(frames or []):
        if s.get("grid"):
            return D.normalize_grid(dict(s["grid"])), s
    return D.normalize_grid(dict(D.demo_grid())), None


def gate_label(px, py, ex, ey, evx, evy, shot_speed, sh, grid, tol=18.0,
               need_los=False):
    """「该不该开枪」的规则标签：只在**泪弹飞到时**还打在轴上时才开火。

    这是 BC 里最关键的一块。原来的规则老师只会"永远朝敌人瞄"，开不开枪交给 PPO
    去学；实测 PPO 学不会（命中率卡在 30~45%，而完美门控能到 47~65%）。
    既然 BC 一学就会（回归自检 4/4），就把"何时该开火"也做成监督标签。

    物理依据：泪弹速度 ~10 单位/帧，距离 d 要飞 T=d/10 帧；
    这 T 帧里敌人已经顺着 (evx,evy) 挪开了。所以判的是
        |垂直偏移(敌人预测位置)| <= tol
    而不是"现在"的偏移 —— 这是 `oracle_eval.py` 量出来的、唯一和真实命中对得上的判据。
    """
    if sh == 0:
        return 0
    dx, dy = ex - px, ey - py
    dist = math.hypot(dx, dy)
    T = dist / max(shot_speed, 1e-6)
    ex_p, ey_p = ex + evx * T, ey + evy * T
    off = abs(ey_p - py) if sh in (2, 4) else abs(ex_p - px)
    if off > tol:
        return 0
    if need_los and grid:
        span = _los(px, py, grid)[sh - 1]
        tw = max(grid["tw"], 1e-6)
        if span + 0.5 < dist / tw:
            return 0
    return sh


def collect(frames, grid, terrain_mode, per_frame=120, seed=0, gate=False,
            gate_tol=18.0, gate_los=False, enemy_speed=0.0):
    """造 (obs, target_move, target_shoot) 监督数据。

    以真实帧里的玩家位置为锚点，把敌人摆到周围的随机方位/距离上 ——
    这样少量真实地形也能产出足够多、覆盖全方位的样本。

    gate=True 时用 `gate_label` 把"不该开枪"的样本标成 0（教它等时机）；
    enemy_speed>0 时给敌人随机速度，教它算**提前量**（观测里本来就有 vx/vy）。
    """
    rng = np.random.default_rng(seed)
    w, h = grid["w"], grid["h"]
    tw, th, ox, oy = grid["tw"], grid["th"], grid["ox"], grid["oy"]

    anchors = []
    for s in (frames or []):
        p = s.get("player")
        if p:
            anchors.append((p["x"], p["y"]))
    if not anchors:                            # 合成锚点
        for _ in range(20):
            anchors.append((ox + rng.uniform(3, w - 3) * tw,
                            oy + rng.uniform(2, h - 2) * th))

    X, Ym, Ys = [], [], []
    n_gate_off = 0
    for ax, ay in anchors:
        for _ in range(per_frame):
            ang = rng.uniform(0, 2 * math.pi)
            r = rng.uniform(60.0, 420.0)
            ex, ey = ax + r * math.cos(ang), ay + r * math.sin(ang)
            evx = evy = 0.0
            if enemy_speed > 0:
                evx = float(rng.uniform(-enemy_speed, enemy_speed))
                evy = float(rng.uniform(-enemy_speed, enemy_speed))
            st = _make_state(ax, ay, ex, ey, grid, evx, evy)
            m, sh = rule_action(st, grid)
            if gate:
                sh2 = gate_label(ax, ay, ex, ey, evx, evy, 10.0, sh, grid,
                                 tol=gate_tol, need_los=gate_los)
                if sh2 == 0 and sh != 0:
                    n_gate_off += 1
                sh = sh2
            X.append(build_obs(st, grid, terrain_mode))
            Ym.append(m)
            Ys.append(sh)
    if gate:
        total = len(Ys)
        quiet = sum(1 for v in Ys if v == 0)
        print(f"门控标签：{quiet}/{total} = {quiet/max(total,1)*100:.1f}% 标为「不射」"
              f"（其中 {n_gate_off} 条是「该射但现在不该射」）")
    return (np.asarray(X, dtype=np.float32),
            np.asarray(Ym, dtype=np.int64),
            np.asarray(Ys, dtype=np.int64))


def main():
    ap = argparse.ArgumentParser(description="用规则 BC 热启动策略头")
    ap.add_argument("--frames", default=None,
                    help="dump_state.py --json 存的状态帧；不给就用合成房间")
    ap.add_argument("--out", default="models/bc_seed",
                    help="输出模型路径（不用写 .zip）")
    ap.add_argument("--steps-per-anchor", type=int, default=120,
                    help="每个玩家位置生成多少个敌人方位样本")
    ap.add_argument("--epochs", type=int, default=15)
    ap.add_argument("--head", default="both", choices=["shoot", "both"],
                    help="both=瞄准和移动都教（默认）；shoot=只教瞄准，移动留给 PPO")
    ap.add_argument("--terrain", default=None, choices=[TERRAIN_LOCAL, TERRAIN_FULL],
                    help="地形模式；默认按 --frames 的模型推断，没有就用 full")
    ap.add_argument("--load", default=None,
                    help="在这个模型上继续 BC（不给就新建随机权重的网络）")
    ap.add_argument("--gate", action="store_true",
                    help="教「何时该开枪」：只在提前量预测能命中时才标成开火。"
                         "实测这是 PPO 学不会、BC 一学就会的部分")
    ap.add_argument("--gate-tol", type=float, default=18.0,
                    help="门控容差（世界单位，默认 18 ≈ 半格）：预测垂直偏移超过它就不射")
    ap.add_argument("--gate-los", action="store_true",
                    help="门控时额外要求视线畅通（避免对着石头开枪）")
    ap.add_argument("--enemy-speed", type=float, default=0.0,
                    help="给敌人加多大随机速度（单位/帧）。设 1.5~2.5 能教它算提前量")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    frames = None
    if args.frames and os.path.exists(args.frames):
        with open(args.frames, encoding="utf-8") as f:
            data = json.load(f)
        frames = [s for s in (data if isinstance(data, list) else [data])
                  if "player" in s]
        print(f"读入 {args.frames}：{len(frames)} 帧")
    elif args.frames:
        print(f"⚠ {args.frames} 不存在，改用合成房间")

    grid, _ = pick_grid(frames)
    terrain_mode = args.terrain or TERRAIN_FULL
    dim = obs_dim(terrain_mode)
    print(f"地形 {terrain_mode} → 观测 {dim} 维；房间 {grid['w']}×{grid['h']}")

    X, Ym, Ys = collect(frames, grid, terrain_mode, args.steps_per_anchor,
                        seed=args.seed, gate=args.gate, gate_tol=args.gate_tol,
                        gate_los=args.gate_los, enemy_speed=args.enemy_speed)
    print(f"监督样本 {len(X)} 条（射击标签 5 类，移动标签 9 类）")

    # ---- 给出网络 ----
    from stable_baselines3 import PPO
    from train_multi import resolve_model_path
    if args.load:
        path = resolve_model_path(args.load)
        model = PPO.load(path, device="cpu")
        got = int(model.observation_space.shape[0])
        if got != dim:
            print(f"✗ {path} 是 {got} 维，和当前地形模式({dim} 维)不匹配。"
                  f"用 --terrain 指定它的地形模式。")
            return 1
        print(f"在 {path} 基础上继续 BC")
    else:
        import gymnasium as gym
        from gymnasium import spaces
        dummy = gym.Env()
        dummy.action_space = spaces.MultiDiscrete([9, 5])
        dummy.observation_space = spaces.Box(-10.0, 10.0, shape=(dim,),
                                             dtype=np.float32)
        dummy.reset = lambda **kw: (np.zeros(dim, dtype=np.float32), {})
        dummy.step = lambda a: (np.zeros(dim, dtype=np.float32), 0.0, False, False, {})
        model = PPO("MlpPolicy", dummy, n_steps=1024, batch_size=256,
                    policy_kwargs=dict(net_arch=dict(pi=[256, 256], vf=[256, 256])),
                    device="cpu", verbose=0)

    # ---- 只训策略头 ----
    import torch
    import torch.nn.functional as F
    pol = model.policy
    opt = torch.optim.Adam(pol.parameters(), lr=1e-3)
    Xt = torch.as_tensor(X)
    Yst = torch.as_tensor(Ys)
    Ymt = torch.as_tensor(Ym)
    nvec = [9, 5]
    N = len(X)

    for ep in range(args.epochs):
        idx = torch.randperm(N)
        tot_s = tot_m = 0.0
        for k in range(0, N, 256):
            b = idx[k:k + 256]
            feat = pol.extract_features(Xt[b])
            logits = pol.action_net(pol.mlp_extractor.forward_actor(feat))
            l_move, l_shoot = torch.split(logits, nvec, dim=1)
            loss = F.cross_entropy(l_shoot, Yst[b])
            if args.head == "both":
                loss = loss + F.cross_entropy(l_move, Ymt[b])
            opt.zero_grad()
            loss.backward()
            opt.step()
            tot_s += F.cross_entropy(l_shoot.detach(), Yst[b]).item() * len(b)
            if args.head == "both":
                tot_m += F.cross_entropy(l_move.detach(), Ymt[b]).item() * len(b)
        line = f"  epoch {ep + 1:2d}  射击交叉熵 {tot_s / N:.4f}"
        if args.head == "both":
            line += f"  移动交叉熵 {tot_m / N:.4f}"
        print(line)

    # ---- 自检 ----
    print("\n自检 1（把敌人放在玩家四个方向，看射击跟不跟）：")
    off = {"上": (0, -1), "右": (1, 0), "下": (0, 1), "左": (-1, 0)}
    want = {"上": 1, "右": 2, "下": 3, "左": 4}
    ax, ay = grid["ox"] + grid["w"] * grid["tw"] / 2, grid["oy"] + grid["h"] * grid["th"] / 2
    hit = 0
    for name, (dx, dy) in off.items():
        o = build_obs(_make_state(ax, ay, ax + dx * D_OPT, ay + dy * D_OPT, grid),
                      grid, terrain_mode)
        with torch.no_grad():
            ds = pol.get_distribution(
                torch.as_tensor(o[None], dtype=torch.float32)).distribution
            p = ds[1].probs.cpu().numpy()[0]
        am = SHOOT_NAMES[int(np.argmax(p))]
        ok = int(np.argmax(p)) == want[name]
        hit += ok
        print(f"  敌人在{name:<2} -> 射击={am:<3} {'✓' if ok else '✗'}   "
              + " ".join(f"{SHOOT_NAMES[i]}{p[i]:.0%}" for i in range(5)))
    print(f"  四方向命中 {hit}/4" + ("  ✅" if hit == 4 else "  ⚠"))

    print("\n自检 2（敌人斜着放，看移动会不会往同一行/列上靠）：")
    for label, (dx, dy) in {"敌人在右下 300,80": (300, 80),
                            "敌人在左上 -300,-80": (-300, -80),
                            "敌人在右上 250,-70": (250, -70),
                            "敌人在左下 -250,70": (-250, 70)}.items():
        ex, ey = ax + dx, ay + dy
        st = _make_state(ax, ay, ex, ey, grid)
        o = build_obs(st, grid, terrain_mode)
        a, _ = model.predict(o, deterministic=True)
        mi = int(a[0])
        mx, my = DIRS8[mi] if mi < 8 else (0, 0)
        # 主轴 = 偏移大的那个轴；对齐 = 消掉另一轴上的偏移
        horizontal = abs(dx) >= abs(dy)
        off0 = (0 - dy) if horizontal else (0 - dx)
        off1 = (my * STEP - dy) if horizontal else (mx * STEP - dx)
        better = abs(off1) < abs(off0)
        print(f"  {label:<20} -> 移动={MOVE_NAMES[mi]:<3} "
              f"非主轴偏移 {off0:+.0f} -> {off1:+.0f} "
              f"{'✓ 往对齐方向挪' if better else '✗ 没往对齐方向'}"
              f"   射击={SHOOT_NAMES[int(a[1])]}")

    print("\n自检 3（视线观测有没有正常工作 —— 挡子弹的是 2物体/3实心/4墙）：")
    w, h = grid["w"], grid["h"]
    tw, th, ox, oy = grid["tw"], grid["th"], grid["ox"], grid["oy"]
    px, py = ox + 3.5 * tw, oy + (h - 3) * th     # 房间左侧、靠下（左边就是墙）
    los = _los(px, py, grid)
    st = _make_state(px, py, px - 300, py, grid)
    o = build_obs(st, grid, terrain_mode)
    l0, l1 = D.block_slices(terrain_mode)["los"]
    print(f"  玩家格=({int((px - ox) / tw)},{int((py - oy) / th)})  "
          f"视线 上{los[0]:.0f} 右{los[1]:.0f} 下{los[2]:.0f} 左{los[3]:.0f} 格")
    print(f"  观测里的视线块 = {np.round(o[l0:l1], 2)}（左=0 说明左边紧贴墙）")
    print("  BC 不教「该不该开枪」，这部分交给 PPO：打空会拉低命中率 -> 时间代价变贵")

    # ---- 自检 4：提前量门控（只有 --gate 才有意义）----
    if args.gate:
        print("\n自检 4（提前量门控：敌人横着漂，该等到漂回轴上再开枪）：")
        for label, vy in (("静止", 0.0), ("下漂 +1/帧", 1.0),
                          ("下漂 +3/帧", 3.0), ("下漂 +6/帧", 6.0)):
            st = _make_state(ax, ay, ax + 150, ay, grid, 0.0, vy)
            o = build_obs(st, grid, terrain_mode)
            gate_v = gate_label(ax, ay, ax + 150, ay, 0.0, vy, 10.0, 2, grid,
                                tol=args.gate_tol)
            with torch.no_grad():
                ds = pol.get_distribution(
                    torch.as_tensor(o[None], dtype=torch.float32)).distribution
                p = ds[1].probs.cpu().numpy()[0]
            a_shot = SHOOT_NAMES[int(np.argmax(p))]
            print(f"  敌人正右150 {label:<10} 规则标签={SHOOT_NAMES[gate_v]:<3} "
                  f"模型={a_shot:<3}  "
                  + " ".join(f"{SHOOT_NAMES[i]}{p[i]:.0%}" for i in range(5)))

    out = args.out
    model.save(out)
    print(f"\n已保存 {out}.zip")
    print(f"接着微调：\n  python train_multi.py --attach --combat "
          f"{'--full-terrain ' if terrain_mode == TERRAIN_FULL else ''}"
          f"--max-enemies 2 --load {out} --steps 200000")
    return 0


if __name__ == "__main__":
    sys.exit(main())
