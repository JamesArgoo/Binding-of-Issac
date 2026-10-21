# -*- coding: utf-8 -*-
"""selftest_learn.py —— 不开游戏，验证"这套观测+动作能不能学会"

目的：把"学不会"拆成两种可能
   A. 观测/动作设计有问题，网络根本表达不出"朝敌人射"这件事
   B. 观测没问题，是真实环境的奖励/吞吐/信号有问题

做法：用完全相同的 `build_obs`、相同的动作空间，造一个**假环境**：
玩家和敌人都在一张 15×9 网格上，动一下就按 DIRS8 走 12 单位，
奖励 = 距离项 + 命中项。然后跑 PPO，最后用和真实模型一样的归因实验检查它。

如果这里 PPO 能学会（四方向命中 4/4），说明 A 排除，问题在真实环境那边。

用法：
    python selftest_learn.py                 # 默认 full 地形，20k 步
    python selftest_learn.py --mode local --steps 20000
"""
import argparse
import math
import sys

import numpy as np

try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

import gymnasium as gym
from gymnasium import spaces

import isaac_data as D
from isaac_env import DIRS8, TERRAIN_FULL, TERRAIN_LOCAL, build_obs, obs_dim

SHOOT_DIRS = [(0, 0), (0, -1), (1, 0), (0, 1), (-1, 0)]   # 同 combat_env
MOVE_NAMES = ["右", "右下", "下", "左下", "左", "左上", "上", "右上", "停"]
SHOOT_NAMES = ["不射", "上", "右", "下", "左"]
CELL = 12.0          # 每步位移（实测真游戏约 12 单位/步）
D_OPT = 150.0


def _make_state(px, py, ex, ey, grid):
    """拼一个符合模组 schema 的最小状态，直接喂 build_obs。"""
    w, h = grid["w"], grid["h"]
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
            "shot_speed": 10.0, "fire_delay": 8.0, "size": 10.0,
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
             "vx": 0.0, "vy": 0.0, "hp": 20.0, "boss": False, "enemy": True,
             "dead": False, "spawner_type": 0, "size": 12.0, "height": 0.0,
             "coll": 1, "cdmg": 0.0, "gcoll": 1, "grid_idx": 0},
        ],
        "doors": [],
    }


def _bearing(shoot_a, dx, dy):
    """射击动作是否指向 (dx,dy) 的方向（四方向 + 30° 容差）。"""
    ax, ay = SHOOT_DIRS[shoot_a]
    if ax == 0 and ay == 0:
        return False
    n = math.hypot(dx, dy) or 1.0
    cos = (ax * dx + ay * dy) / n
    return cos > math.cos(math.radians(45))


class FakeCombat(gym.Env):
    """完全用真实 build_obs 的最小战斗环境，只为验证可学性。

    ablate: 把某一块观测清零，用来定位"是哪一块把有用信号淹了"
      None=完整观测 / "terrain"=清零地形块 / "tear"=清零弹幕块
    """

    def __init__(self, terrain_mode=TERRAIN_FULL, max_steps=120, ablate=None,
                 hit_bonus=0.5, base=0.02):
        super().__init__()
        self.mode = terrain_mode
        self.hit_bonus = hit_bonus
        self.base = base
        # 命令行叫 terrain，block_slices 里的键叫 grid
        self.ablate = {"terrain": "grid"}.get(ablate, ablate)
        self.slices = D.block_slices(terrain_mode)
        self.grid = D.normalize_grid(dict(D.demo_grid()))
        self.action_space = spaces.MultiDiscrete([9, 5])
        self.observation_space = spaces.Box(-10.0, 10.0,
                                            shape=(obs_dim(terrain_mode),),
                                            dtype=np.float32)
        self.max_steps = max_steps
        self.rng = np.random.default_rng(0)
        self.t = 0
        self.px = self.py = self.ex = self.ey = 0.0
        self.reset()

    def _clamp(self, x, y):
        g = self.grid
        tw, th, ox, oy = g["tw"], g["th"], g["ox"], g["oy"]
        w, h = g["w"], g["h"]
        x = min(max(x, ox + 2 * tw), ox + (w - 2) * tw)
        y = min(max(y, oy + 2 * th), oy + (h - 2) * th)
        return x, y

    def _obs(self):
        o = build_obs(_make_state(self.px, self.py, self.ex, self.ey, self.grid),
                      self.grid, self.mode)
        if self.ablate and self.ablate in self.slices:
            lo, hi = self.slices[self.ablate]
            o[lo:hi] = 0.0
        return o

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        g = self.grid
        ox, oy, tw, th = g["ox"], g["oy"], g["tw"], g["th"]
        self.px = ox + self.rng.uniform(3, g["w"] - 3) * tw
        self.py = oy + self.rng.uniform(2, g["h"] - 2) * th
        ang = self.rng.uniform(0, 2 * math.pi)
        r = self.rng.uniform(120, 380)
        self.ex, self.ey = self._clamp(self.px + r * math.cos(ang),
                                       self.py + r * math.sin(ang))
        self.t = 0
        return self._obs(), {}

    def step(self, action):
        a = int(np.asarray(action).ravel()[0])
        s = int(np.asarray(action).ravel()[1])
        dx, dy = DIRS8[a] if a < 8 else (0, 0)
        self.px, self.py = self._clamp(self.px + CELL * dx, self.py + CELL * dy)
        # 敌人慢慢朝玩家飘（制造必须射击的压力）
        vx, vy = self.px - self.ex, self.py - self.ey
        n = math.hypot(vx, vy) or 1.0
        self.ex += vx / n * 3.0
        self.ey += vy / n * 3.0

        vx, vy = self.px - self.ex, self.py - self.ey
        d = math.hypot(vx, vy)
        # 奖励：站在合适距离 + 朝敌人射击
        r = -abs(d - D_OPT) / 400.0
        hit = _bearing(s, -vx, -vy)          # 从玩家指向敌人
        if hit:
            r += self.hit_bonus
        r += self.base                       # 每步基础存活分（抵消纯时间惩罚的学习惯性）

        self.t += 1
        terminated = False
        truncated = self.t >= self.max_steps
        return self._obs(), float(r), terminated, truncated, {"hit": hit, "d": d}


def attribution_test(model, env):
    """和 inspect_policy.py 同样的四方向归因，检查"射击跟不跟敌人"。"""
    g = env.grid
    ox, oy, tw, th = g["ox"], g["oy"], g["tw"], g["th"]
    px, py = ox + 7.5 * tw, oy + 4.5 * th
    want = {"上": 1, "右": 2, "下": 3, "左": 4}
    off = {"上": (0, -1), "右": (1, 0), "下": (0, 1), "左": (-1, 0)}
    hits = 0
    for name, (dx, dy) in off.items():
        st = _make_state(px, py, px + dx * 150, py + dy * 150, g)
        o = build_obs(st, g, env.mode)
        if getattr(env, "ablate", None):
            lo, hi = env.slices[env.ablate]
            o[lo:hi] = 0.0
        a, _ = model.predict(o, deterministic=True)
        ok = int(a[1]) == want[name]
        hits += ok
        print(f"    敌人在{name:<2} -> 射击={SHOOT_NAMES[int(a[1])]:<3} "
              f"{'✓' if ok else '✗'}   移动={MOVE_NAMES[int(a[0])]}")
    return hits


def run_one(mode, steps, ablate, seed=0, hit_bonus=0.5):
    """训练一次并做归因测试；返回 (四方向命中, 训练后平均回报, 策略熵占比)。"""
    tag = f"地形={mode}" + (f" 清零[{ablate}]" if ablate else "")
    if hit_bonus != 0.5:
        tag += f"  瞄准奖励=+{hit_bonus}"
    print(f"\n{'=' * 64}\n{tag}   观测 {obs_dim(mode)} 维\n{'=' * 64}")
    env = FakeCombat(terrain_mode=mode, ablate=ablate, hit_bonus=hit_bonus)
    from stable_baselines3 import PPO
    from stable_baselines3.common.monitor import Monitor
    menv = Monitor(env)
    model = PPO("MlpPolicy", menv, n_steps=1024, batch_size=256, n_epochs=10,
                gamma=0.99, ent_coef=0.01, learning_rate=3e-4, seed=seed,
                policy_kwargs=dict(net_arch=dict(pi=[128, 128], vf=[128, 128])),
                device="cpu", verbose=0)
    model.learn(total_timesteps=steps, progress_bar=False)

    # 训练后表现
    env.rng = np.random.default_rng(1)
    tot = 0.0
    for _ in range(20):
        o, _ = env.reset()
        done = False
        while not done:
            a, _ = model.predict(o, deterministic=False)
            o, r, term, trunc, _ = env.step(a)
            tot += r
            done = term or trunc

    # 策略熵占比
    import math

    import torch
    from isaac_env import build_obs
    g = env.grid
    w, h = g["w"], g["h"]
    tw, th, ox, oy = g["tw"], g["th"], g["ox"], g["oy"]
    st = _make_state(ox + 7.5 * tw, oy + 4.5 * th,
                     ox + 7.5 * tw + 150, oy + 4.5 * th, g)
    ob = build_obs(st, g, env.mode)
    with torch.no_grad():
        ds = model.policy.get_distribution(
            torch.as_tensor(ob[None], dtype=torch.float32)).distribution
        ent = float(sum(d.entropy().sum().item() for d in ds))
    h_max = math.log(9) + math.log(5)

    hits = attribution_test(model, env)
    print(f"  -> 四方向命中 {hits}/4   平均回报 {tot / 20:+.2f}   "
          f"熵 {ent:.3f}/{h_max:.3f} = {ent / h_max:.0%}")
    return hits, tot / 20, ent / h_max


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", default="full",
                    choices=[TERRAIN_LOCAL, TERRAIN_FULL])
    ap.add_argument("--steps", type=int, default=20000)
    ap.add_argument("--ablate", default=None,
                    choices=["terrain", "tear", "enemy", "player"],
                    help="清零某一块观测，用来定位是哪块把信号淹了")
    ap.add_argument("--hit-bonus", type=float, default=0.5,
                    help="假环境里「朝敌人射」的每步奖励。真实环境一次命中约 +0.014")
    ap.add_argument("--compare", action="store_true",
                    help="一次跑完四组对照：full / full-清零地形 / local / local-清零地形")
    ap.add_argument("--sweep", action="store_true",
                    help="扫描瞄准奖励强度，看多小的信号才会学不会")
    args = ap.parse_args()

    mode = TERRAIN_FULL if args.mode == "full" else TERRAIN_LOCAL

    if args.sweep:
        # 真实环境一次命中 +0.014（3.5 伤害 / 5 × 0.1），这里从强到弱扫描
        amounts = [0.5, 0.1, 0.03, 0.014, 0.005]
        print(f"扫描：瞄准奖励从 +0.5 一路降到 +0.005（真实环境约 +0.014）")
        print(f"每组 {args.steps} 步，地形 {mode}\n")
        rows = []
        for amt in amounts:
            hits, rew, entr = run_one(mode, args.steps, None, hit_bonus=amt)
            rows.append((amt, hits, rew, entr))
        print(f"\n{'=' * 64}\n汇总：瞄准奖励强度 vs 能不能学会\n{'=' * 64}")
        print(f"{'瞄准奖励':>10} {'四方向命中':>10} {'平均回报':>10} {'熵占比':>8}")
        for amt, hits, rew, entr in rows:
            print(f"{'+' + str(amt):>10} {hits:>8}/4 {rew:>+10.2f} {entr:>8.0%}")
        print("\n如果小奖励学不会、大奖励学得会 —— 说明真实环境的奖励尺度就是瓶颈，"
              "要放大命中奖励或做 reward normalization")
        return 0

    if args.compare:
        res = {}
        res["full"] = run_one(TERRAIN_FULL, args.steps, None)[0]
        res["full_无地形"] = run_one(TERRAIN_FULL, args.steps, "terrain")[0]
        res["local"] = run_one(TERRAIN_LOCAL, args.steps, None)[0]
        res["local_无地形"] = run_one(TERRAIN_LOCAL, args.steps, "terrain")[0]
        print(f"\n{'=' * 64}\n汇总（每格满分 4）\n{'=' * 64}")
        for k, v in res.items():
            print(f"  {k:<14} {v}/4  {'#' * v * 5}")
        best = max(res, key=res.get)
        print(f"\n最好的配置：{best} ({res[best]}/4)")
        if res["full_无地形"] >= 3 and res["full"] <= 1:
            print("=> 地形块（896 维）把敌人特征淹了。要么单独编码地形，要么给各块做尺度归一。")
        elif res["local"] >= 3 and res["full"] <= 1:
            print("=> 全房间地形不可学，退回 local 的 7×7。")
        elif max(res.values()) <= 1:
            print("=> 连最简单的配置都学不会，问题在观测/动作的表达本身，不在某一块。")
        return 0

    hits, rew, entr = run_one(mode, args.steps, args.ablate,
                              hit_bonus=args.hit_bonus)
    print(f"\n四方向命中 {hits}/4  ->  "
          + ("这套观测+动作是可以学会的 ✅" if hits >= 3
             else "这个奖励强度下学不会 ❌"))
    return 0


if __name__ == "__main__":
    sys.exit(main())
