# -*- coding: utf-8 -*-
"""combat_env.py —— 单房间战斗环境（照搬 AIsaac 的任务设定）

模仿的是 `AIIsaac/AIsaac/bridge/python/isaac_bridge/` 里的设计：

1. 任务范围（abplus_tasks.py）—— 原文："RL for combat only, room-level mixture"
   每回合只打一个房间：reset 时用控制台命令 `restart` + `goto d.<variant>` 传送进去。
   终止条件：房间清空 或 玩家死亡。

2. 奖励（abplus_reward.py 的 CombatHitRate）：
       hit  = 敌人掉血 / 5.0                    （每 5 点伤害 +1）
       kill = 击杀数 * 0.25
       time = -(2.0 + (0.5-2.0) * 命中率) * 秒数 （命中率越高，时间越便宜）
   精妙点：不直接惩罚打空，而是把"时间代价"和命中率挂钩，
   打得准自然就少被扣时间。整体再乘 scale=0.1。

3. 动作（env.py 的 Action）：
   移动 8 方向 + 停，射击 4 方向 + 不射（泪弹只有四个方向）。

为什么不用"让 AI 自己走整层楼"的旧思路：实测 AI 会在两个房间之间无限来回弹，
录了几分钟才走 2 个房间，永远遇不到敌人。AIsaac 直接传送进战斗房间规避了这个问题。
"""
import math
import time
from collections import deque

import gymnasium as gym
import numpy as np
from gymnasium import spaces

from isaac_bridge import get_enemies, get_player
from isaac_env import (TERRAIN_FULL, TERRAIN_LOCAL, build_obs, cell_walkable,
                       normalize_grid, obs_dim)

# ---- 动作：移动 0-7 = DIRS8 八方向，8 = 停；射击 0 = 不射，1=上 2=右 3=下 4=左 ----
DIRS8 = [(1, 0), (1, 1), (0, 1), (-1, 1), (-1, 0), (-1, -1), (0, -1), (1, -1)]
SHOOT_DIRS = [(0, 0), (0, -1), (1, 0), (0, 1), (-1, 0)]   # 0=不射 1=上 2=右 3=下 4=左
N_MOVE, N_SHOOT = 9, 5

# ---- 奖励常数（照 AIsaac CombatHitRateHurt / CombatHp）----
HIT_REW = 1.0 / 5.0     # 每造成 1 点伤害的奖励（= 每 5 点伤害 +1）
KILL = 0.25             # 每杀一个 +0.25
COST_HIT = 0.5          # 命中率=1 时每秒的时间代价
COST_MISS = 2.0         # 命中率=0 时每秒的时间代价
CLEAR = 40.0            # 清房（AIsaac ROOM_POINTS['normal']=40）
DEATH = 5.0             # 死亡固定惩罚
ALIGN = 0.2             # 距离势权重（AIsaac HRW align=0.2）：鼓励站在刚好打得到的距离
ALIGN_OFF = 1.0         # **对齐势**权重：奖励"把自己的行/列挪到和敌人一致"
                        # 以撒的泪弹只有 4 个方向，敌人斜着时无论怎么按都打不中，
                        # 必须先把"非射击轴"上的偏移消掉。原来的奖励里只有距离势
                        # (每步 ±0.02 缩放分)，对齐完全没被激励 —— 这是"打不准"的
                        # 头号原因。1.0 时：挪 1 格(40 单位)对齐 ≈ +0.1 缩放分，
                        # 而掉 1 点血 = -1.0，量级终于能比了。
GAMMA = 0.99            # 势函数折扣，必须和 PPO 的 gamma 一致
D_OPT = 150.0           # 理想射击距离（像素）：太近会被撞，太远打不着
CELL = 40.0             # 一格 40 世界单位（AIsaac 的 CELL）
WALL_PUSH = 0.6         # 顶着墙/障碍物推的每秒惩罚（治"怼着石头走"）
SCALE = 0.1             # 整体缩放
# 每秒多少个**决策步**。默认 60/frame_skip（假设游戏跑满 60fps）：
# 注意这和实测不符 —— 本机游戏帧率约 29.6fps，frame_skip=2 时真实只有 ~14.8 步/秒。
# 这个值只影响"死亡时罚掉剩余时间"那项。用默认 30 时，死亡惩罚(2.0/FPS)和
# 时间代价(2.0*frame_skip/60)恰好相等，于是"早死"和"晚死"总分一模一样(都是 -6.5)，
# 时间那一项的设计意图（早死不该省时间成本）被抵消掉了。按实测帧率传进来才是对的。
FPS = 30.0
RATE_WINDOW = 45        # 命中率统计窗口（步）
HP_FULL = 6.0           # 3 颗红心 = 6 个半心


def _curve(x):
    return (x + 1.0 - (1.0 - x) ** 4) / 2.0


# 血量权重：让"丢最后一个半心"的代价约为丢第一个的十几倍
HP_WEIGHT = 1.0 / (_curve(1.0) - _curve((HP_FULL - 1) / HP_FULL))


def health_value(h):
    """血量价值：凹形曲线，越接近死亡每一滴血越贵（AIsaac 的做法）。

    直接用固定"每点血扣 X"会鼓励 AI 拿血硬换伤害；
    凹形曲线让它在残血时格外小心，这才像人在玩。
    """
    h = max(0.0, float(h))
    if h >= HP_FULL:
        return HP_WEIGHT + (h - HP_FULL)
    return HP_WEIGHT * _curve(h / HP_FULL)


def time_price(rate, cost_miss=COST_MISS, cost_hit=COST_HIT):
    """命中率 -> 每秒时间代价。命中率越高越便宜（线性插值）。"""
    return cost_miss + (cost_hit - cost_miss) * min(1.0, max(0.0, float(rate)))


class CombatEnv(gym.Env):
    """一个实例 = 一个单房间战斗环境。

    bridge: 要有 get_state() 和 send(text)（multi_instance.ChannelAdapter）
    controller: 要有 set_move(mx,my) / set_shoot(ax,ay) / release_all()
    variants: 可用的房间 variant 列表（用 room_catalog.py 探出来）
    """

    metadata = {"render_modes": []}

    def __init__(self, bridge, controller, variants=(12, 13, 14),
                 frame_skip=2, max_steps=None, seed=None, chunk=1,
                 terrain_mode=TERRAIN_LOCAL, fps=None,
                 hit_rew=HIT_REW, cost_miss=COST_MISS, reset_mode="restart",                 hurt_scale=1.0, fail_on_hurt=False, min_hp=3.0, align_off=ALIGN_OFF):
        super().__init__()
        self.bridge = bridge
        self.kc = controller
        self.variants = list(variants) or [12]
        self.frame_skip = frame_skip
        # 奖励尺度。命中奖励和"时间代价"的**比值**决定瞄准学不学得会：
        # 实测（selftest_learn.py --sweep）每步瞄准信号低于 ~0.1 时 PPO 完全学不动，
        # 熵会一直卡在 92%。真实环境一次命中约 +0.07、每 10~20 步才中一次
        # -> 每步约 0.005，正好落在这个学不动的区间里。所以这两个值要一起调。
        self.hit_rew = float(hit_rew)
        self.cost_miss = float(cost_miss)
        # reset 策略：
        #   "restart"（默认，照 AIsaac）每回合整局重开，保证满血、楼层一致。
        #   "keep"   玩家还活着就只 goto 下一间房，不重开 —— 重开要好几次秒，
        #            而实测回合本身只有 ~180 步（6 秒），重开能吃掉 30-40% 的训练时间。
        #            代价：血量会跨回合累积（死了模组照样会自动重开，等于回满血）。
        self.reset_mode = reset_mode
        # 受伤惩罚倍率。默认 1.0（= AIsaac 的凹形血量曲线，掉光 6 点血约 -1.2 缩放分）。
        # 但命中奖励 --hit-rew 调到 1.0 之后，一个回合打出的伤害值 ~28 分，
        # 而掉光血只扣 11.95 分 —— 打输出比挨打贵 2.3 倍，于是"贴脸换血"成了最优解。
        # 想让 AI 学会躲，就得把这个倍数拉到能压过输出收益（实测 10 左右起效）。
        self.hurt_scale = float(hurt_scale)
        # "不允许受伤"的严格版：掉任何一点血就结束本回合（不给 DEATH 惩罚，
        # 因为这是课程阶段的手段，不是真死亡）。
        self.fail_on_hurt = bool(fail_on_hurt)
        # auto 模式的补血阈值：血量低于这个值就整局重开
        self.min_hp = float(min_hp)
        # 对齐势权重：奖励"把非射击轴上的偏移消掉"（泪弹只有 4 方向，不对齐打不中）
        self.align_off = float(align_off)
        self.last_reset_s = 0.0     # 上一次 reset 花了多少秒（诊断）
        self.last_ep_steps = 0      # 上一回合走了多少步（诊断）
        self._reset_fallbacks = 0   # goto 连试 4 次都没进战斗房的次数（诊断）
        # 本次对局已经进过的房间 variant：避免重进已被清空的房间白拿清房奖励
        self._used_variants = set()
        # 决策频率（步/秒）。默认假设游戏跑满 60fps；实测只有 ~30fps 时应该传 15，
        # 否则"早死"和"晚死"的奖励会完全相等（见文件顶部 FPS 的注释）。
        self.fps = float(fps) if fps else 60.0 / max(1, frame_skip)
        # 地形编码：local = 玩家周围 7×7（445 维）；full = 整间房 28×16×2 通道（1292 维）
        self.terrain_mode = terrain_mode
        self.obs_dim = obs_dim(terrain_mode)
        # action chunking（借鉴 ACT）：一次决策输出连续 chunk 组动作，
        # 缩短有效决策步数 -> PPO 信用分配更容易、动作更平滑。chunk=1 就是每次一步。
        self.chunk = max(1, int(chunk))
        self.max_steps = int(max_steps) if max_steps else max(50, 900 // self.chunk)
        self.rng = np.random.default_rng(seed)

        self.action_space = spaces.MultiDiscrete([N_MOVE, N_SHOOT] * self.chunk)
        self.observation_space = spaces.Box(-10.0, 10.0, shape=(self.obs_dim,),
                                            dtype=np.float32)

        self.grid = None
        self.last_state = None
        self.steps = 0

        self._prev_enemy_hp = {}
        self._prev_alive = set()
        self._prev_clear = False
        self._prev_room = None
        self._prev_hp = HP_FULL
        self._prev_phi = 0.0
        self._prev_phi_off = 0.0

        self._started_clear = False
        self._room_idx0 = None     # 开局房间索引：变了就说明不在战斗房里了
        self._seen_tears = set()
        self._rate_window = deque(maxlen=RATE_WINDOW)
        self._ep_dealt = 0.0
        self._ep_fired = 0.0
        self._ep_clear = 0
        self._ep_hurt = 0.0
        self._ep_wall = 0

    # ---------------- 工具 ----------------
    def _state(self):
        s = self.bridge.get_state()
        if s and "grid" in s:
            self.grid = normalize_grid(s["grid"])
        return s

    def _alive(self, s):
        if not s or "player" not in s:
            return False
        p = s["player"]
        return (p["hp"] + p["soul_hp"]) > 0

    def _wait(self, pred, timeout=15.0, poll=0.02):
        t0 = time.time()
        while time.time() - t0 < timeout:
            s = self._state()
            if pred(s):
                return s
            time.sleep(poll)
        return None

    def _wait_frames(self, n, timeout=1.0):
        """等游戏**逻辑帧**推进 n 帧（而不是按墙钟睡固定时间）。

        这是把 turbo 加速转成训练吞吐的关键：固定 time.sleep(2/60) 把决策频率
        锁死在 ~30 次/秒，跟游戏跑多快无关；turbo 让游戏快 5 倍，决策也该快 5 倍。

        等待用**条件变量**（模组推来新状态时唤醒），不能忙轮询 ——
        实测 0.5ms 忙轮询会把 GIL 抢光，吞吐从 52 掉到 16。
        """
        s0 = self._state()
        f0 = int(s0.get("frame", 0)) if (s0 and "frame" in s0) else 0
        seq = self.bridge.seq() if hasattr(self.bridge, "seq") else 0
        t0 = time.time()
        s = s0
        while time.time() - t0 < timeout:
            s = self._state()
            if s and int(s.get("frame", 0)) - f0 >= n:
                return s
            if hasattr(self.bridge, "wait_new"):
                seq = self.bridge.wait_new(seq, timeout=0.1)
            else:
                time.sleep(0.002)
        return s

    def _cmd(self, text):
        return self.bridge.send(text)

    def _tear_damage(self, s):
        return max(float(s["player"].get("damage", 3.5)), 0.5)

    def _scan_enemies(self, s):
        """返回 {id: hp} 以及当前存活 id 集合。"""
        hp = {}
        for e in get_enemies(s):
            hp[e["id"]] = float(e["hp"])
        return hp, set(hp.keys())

    def _count_fired_tears(self, s):
        """数"新出现的玩家泪弹" = 射了几发（用实体 id 去重）。"""
        fired = 0
        cur = set()
        for e in s.get("entities", []):
            if e.get("type") == 2 and e.get("spawner_type") == 1:
                eid = e.get("id")
                cur.add(eid)
                if eid not in self._seen_tears:
                    fired += 1
        # 把已经消失的泪弹从记忆里去掉，避免集合无限增长
        self._seen_tears = cur
        return fired

    def hit_rate(self):
        fired = sum(f for f, _ in self._rate_window)
        hits = sum(h for _, h in self._rate_window)
        if fired <= 0:
            return 0.0
        return min(1.0, hits / fired)

    # ---------------- Gym 接口 ----------------
    def _combat_ready(self, st):
        """指定房间已经加载好、玩家活着、房间还没清空、**而且真的还有敌人** —— goto 成功判据。

        必须要求"有敌人"：goto 有时会短暂返回一个 clear=False 但实际已经清空的
        房间状态，接受它会导致回合刚开局就被判成清房（实测出现 2~9 步、
        输出 0 的假回合），还会让 `_started_clear` 记错。
        """
        if not st or "player" not in st or "room" not in st:
            return False
        p = st["player"]
        if (p.get("hp", 0) or 0) + (p.get("soul_hp", 0) or 0) <= 0:
            return False
        room = st["room"]
        if room.get("clear"):
            return False
        if "grid" not in st:
            return False
        # 至少要有一个敌人实体（或房间自己报告还有活敌）
        if int(room.get("enemies_alive", 0) or 0) > 0:
            return True
        for e in st.get("entities", []):
            if e.get("enemy") or e.get("boss"):
                return True
        return False

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        t_reset = time.time()
        self.kc.release_all()

        # 1) 需要整局重开吗？
        #    reset_mode="restart"（默认，照 AIsaac）：每回合都重开，保证满血、楼层一致。
        #    reset_mode="keep"   ：只要玩家还活着就不重开，直接 goto 下一间房。
        #    reset_mode="auto"   ：血量 >= min_hp 就 goto（快，0.4s），
        #      血量太低才 restart 补满（4.4s）。实测 restart 要重载整层楼，
        #      是 reset 里最贵的一步；auto 让"健康回合"不付这笔钱，
        #      同时保证回合总在接近满血下进行 —— keep 那种血量一路累积
        #      会让回合越来越短、清房率假性崩塌。
        s = self._state()
        need_restart = not self._alive(s)
        if self.reset_mode == "restart":
            need_restart = True
        elif self.reset_mode == "auto":
            p = (s or {}).get("player") or {}
            hp_now = float(p.get("hp", 0) or 0) + float(p.get("soul_hp", 0) or 0)
            if hp_now < self.min_hp:
                need_restart = True
        if need_restart:
            self._cmd("restart")
            self._wait(lambda st: self._alive(st), timeout=15.0)

        # 2) 传送进一个有敌人的房间。
        #    原来这里用固定 time.sleep(0.8) 盲等，改成轮询到"真的进了战斗房间"为止：
        #    顺利时 0.1~0.3 秒就返回，失败才退到超时。
        #
        #    **必须避开本次对局已经进过的房间**：goto d.<variant> 重进同一间房时，
        #    那间房如果已经被清空过，进去就是 clear=True —— 等于白送一个 +40 清房奖励。
        #    实测演示里 13% 的回合是这种"6 步、零输出、零受伤"的白捡。
        #    进满一轮房间就整局重开一次（35 个房间只要重开 1 次，成本可忽略）。
        variant = None
        s = None
        ready = False
        pool = [v for v in self.variants if v not in self._used_variants]
        if not pool:
            if self.reset_mode != "restart":
                self._cmd("restart")
                self._wait(lambda st: self._alive(st), timeout=15.0)
            self._used_variants.clear()
            pool = list(self.variants)
        for attempt in range(4):
            if not pool:
                break
            variant = int(self.rng.choice(pool))
            self._used_variants.add(variant)
            try:
                pool.remove(variant)
            except ValueError:
                pass
            self._cmd(f"goto d.{variant}")
            time.sleep(0.2)                       # 让房间转场先开始
            s = self._wait(self._combat_ready, timeout=4.0, poll=0.02)
            if s is not None:
                ready = True
                break
        if not ready:
            # 兜底：4 次都没进到战斗房。拿一个"活着"的状态让回合能跑，
            # 但**必须禁用 room_left 判据**（等下 _room_idx0 设成 None）——
            # 否则 _room_idx0 记的是起始房的 index，下一刻房间真的转场就会
            # 被判成"死亡"，白给一个 -DEATH。实测这会造成 3 步就结束的假回合。
            s = self._wait(lambda st: self._alive(st), timeout=6.0, poll=0.02)
            self._reset_fallbacks += 1

        # 3) 兜底
        if s is None:
            s = {"player": {"x": 0, "y": 0, "vx": 0, "vy": 0, "hp": 0, "soul_hp": 0,
                            "grid_idx": 0, "damage": 3.5},
                 "room": {"index": -1, "clear": False, "enemies_alive": 0},
                 "entities": []}

        # 4) 初始化统计
        ep_steps = self.steps            # 上一回合走了多少步（诊断用）
        self.last_state = s
        self.steps = 0
        self._prev_enemy_hp, self._prev_alive = self._scan_enemies(s)
        self._prev_clear = bool(s.get("room", {}).get("clear"))
        # 开局的房间索引：goto 出来的战斗房恒为 -3。之后只要这个值变了，
        # 就说明玩家已经不在开局那个房间里了（死了被重开 / 被传走）。
        # 没成功进战斗房时（ready=False）设成 None —— 这时"房间发生变化"
        # 只是 goto 迟到的转场，不能当成死亡。
        self._room_idx0 = s.get("room", {}).get("index") if ready else None
        # 如果 goto 落到了一个本来就已清空的房间，不能立刻判 terminated，
        # 否则会陷入 reset→立即结束→reset 的死循环
        self._started_clear = self._prev_clear
        self._prev_room = s.get("room", {}).get("index")
        self._seen_tears = set()
        self._rate_window.clear()
        self._ep_dealt = 0.0
        self._ep_fired = 0.0
        self._ep_clear = 0
        self._ep_hurt = 0.0
        self._ep_wall = 0

        # 血量与两个势函数的初值（势函数必须从当前状态起算，否则第一步会凭空多出一笔奖励）
        pp = s.get("player", {}) if s else {}
        self._prev_hp = float(pp.get("hp", 0) or 0) + float(pp.get("soul_hp", 0) or 0)
        enemies0 = get_enemies(s) if (s and "player" in s) else []
        if enemies0:
            near0 = min(enemies0,
                        key=lambda e: (e["x"] - pp["x"]) ** 2 + (e["y"] - pp["y"]) ** 2)
            d0 = math.hypot(near0["x"] - pp["x"], near0["y"] - pp["y"])
            self._prev_phi = -ALIGN * abs(d0 - D_OPT) / CELL
            dx0 = near0["x"] - pp["x"]
            dy0 = near0["y"] - pp["y"]
            off0 = abs(dy0) if abs(dx0) >= abs(dy0) else abs(dx0)
            self._prev_phi_off = -self.align_off * off0 / CELL
        else:
            self._prev_phi = 0.0
            self._prev_phi_off = 0.0


        if "player" not in s:
            self.last_reset_s = time.time() - t_reset
            self.last_ep_steps = ep_steps
            return np.zeros(self.obs_dim, dtype=np.float32), {}
        # 记下本回合还没进对局时的重置耗时，供训练时打印（判断时间花在哪）
        self.last_reset_s = time.time() - t_reset
        self.last_ep_steps = ep_steps
        return (build_obs(s, self.grid, self.terrain_mode),
                {"variant": variant, "room": self._prev_room})

    def step(self, action):
        """一次决策输出 chunk 组动作（借鉴 ACT 的 action chunking）。

        每组动作执行 frame_skip 帧，中途一旦结束就提前退出。
        """
        a = [int(x) for x in np.asarray(action).ravel()]
        total_r = 0.0
        terminated = False
        info = {}
        s = None

        for k in range(self.chunk):
            move_a = a[2 * k] if 2 * k < len(a) else 8
            shoot_a = a[2 * k + 1] if 2 * k + 1 < len(a) else 0
            mx, my = DIRS8[move_a] if move_a < 8 else (0, 0)
            ax, ay = SHOOT_DIRS[shoot_a] if shoot_a < len(SHOOT_DIRS) else (0, 0)
            self.kc.set_move(mx, my)
            self.kc.set_shoot(ax, ay)

            # 按游戏逻辑帧推进（不是墙钟）——turbo 下这一步会快约 5 倍
            s = self._wait_frames(self.frame_skip)
            self.last_state = s
            if not s or "player" not in s:
                break

            r, term, info = self._reward(s, mx, my)
            total_r += r
            if term:
                terminated = True
                break

        self.steps += 1
        obs = (build_obs(s, self.grid, self.terrain_mode) if (s and "player" in s)
               else np.zeros(self.obs_dim, dtype=np.float32))

        truncated = self.steps >= self.max_steps
        if terminated or truncated:
            self.kc.release_all()
        return obs, float(total_r), bool(terminated), bool(truncated), info

    def _reward(self, s, mx=0, my=0):
        p = s["player"]
        cur_hp, cur_alive = self._scan_enemies(s)
        hp_now = float(p["hp"]) + float(p["soul_hp"])

        # ---- 伤害（照 AIsaac：lineage_damage 的差值）----
        dealt = 0.0
        for eid, hp in self._prev_enemy_hp.items():
            if eid in cur_hp and hp > cur_hp[eid]:
                dealt += hp - cur_hp[eid]
        # ---- 击杀 ----
        kills = len(self._prev_alive - cur_alive)

        # ---- 命中率窗口：命中数用"伤害/单发伤害"估算，发弹数用新出现的泪弹数 ----
        fired = self._count_fired_tears(s)
        hits = dealt / self._tear_damage(s)
        self._rate_window.append((fired, hits))
        self._ep_fired += fired
        self._ep_dealt += dealt

        rate = self.hit_rate()

        r = 0.0
        r += max(0.0, dealt) * self.hit_rew
        r += kills * KILL
        r -= time_price(rate, self.cost_miss, COST_HIT) * (self.frame_skip / 60.0)

        # ---- 顶墙/顶障碍物惩罚 ----
        #     想走的方向是墙、石头、便便就直接扣分：逼 AI 别对着障碍物干瞪眼。
        #     只看网格碰撞，不依赖物理位移，所以没有延迟。
        if (mx or my) and self.grid is not None and p.get("grid_idx") is not None:
            w = self.grid["w"]
            gx, gy = p["grid_idx"] % w, p["grid_idx"] // w
            if not cell_walkable(gx + mx, gy + my, self.grid):
                r -= WALL_PUSH * (self.frame_skip / 60.0)
                self._ep_wall += 1

        # ---- 受伤：凹形血量曲线，越接近死亡掉血越贵 ----
        #     之前照抄的 hit-rate 配置里压根没有这一项（那是"玩家无敌"的实验配置），
        #     所以 AI 完全没有躲怪的动机，会直接往怪身上撞。
        #     hurt_scale 用来压过"贴脸换血"的收益，fail_on_hurt 用来实现
        #     "本阶段不允许受伤"（掉血即终局）。
        took_damage = hp_now < self._prev_hp
        if took_damage:
            hurt = health_value(self._prev_hp) - health_value(hp_now)
            r -= hurt * self.hurt_scale
            self._ep_hurt += hurt

        # ---- 对齐势（potential-based，不改变结果排序）----
        #     Φ = -ALIGN * |d - D_OPT| / CELL：鼓励停在"刚好能打到敌人"的距离，
        #     太近会被撞、太远打不着。势函数形式保证不会扭曲最终收益排序。
        #
        #     另外再加一个**对齐势** Φ_off：泪弹只有 4 个方向，敌人斜着时
        #     必须先消掉"非射击轴"上的偏移才能打中。只给距离势的话，
        #     策略会学会"保持距离边走边射"，但永远学不会"先对齐再开枪"，
        #     表现出来就是"打不准、一直擦边"。
        enemies_all = get_enemies(s)
        if enemies_all:
            near = min(enemies_all,
                       key=lambda e: (e["x"] - p["x"]) ** 2 + (e["y"] - p["y"]) ** 2)
            d = math.hypot(near["x"] - p["x"], near["y"] - p["y"])
            phi = -ALIGN * abs(d - D_OPT) / CELL
            # 主轴 = 偏移大的那个轴（泪弹就往那个方向射），另一轴就是"对齐误差"
            dx, dy = near["x"] - p["x"], near["y"] - p["y"]
            off = abs(dy) if abs(dx) >= abs(dy) else abs(dx)
            phi_off = -self.align_off * off / CELL
        else:
            phi, phi_off = 0.0, 0.0
        r += GAMMA * phi - self._prev_phi
        r += GAMMA * phi_off - self._prev_phi_off
        self._prev_phi = phi
        self._prev_phi_off = phi_off

        # ---- 清房 ----
        # 注意：必须确认玩家还站在开局那个战斗房间里。
        # 死亡后模组会 AUTO_RESTART 把玩家送回**起始房间**（那个房间本来就 clear=True），
        # 如果只看 clear 就会把"死了被重开"误判成"清房成功"，白送 +40 —— 等于奖励自杀。
        room = s.get("room", {})
        cleared = bool(room.get("clear"))
        room_idx = room.get("index")
        room_left = (self._room_idx0 is not None and room_idx is not None
                     and room_idx != self._room_idx0)
        if cleared and not room_left and not self._prev_clear and not self._started_clear:
            r += CLEAR
            self._ep_clear += 1

        # ---- 死亡：固定惩罚 + 罚掉剩余时间（早死不该省下时间成本）----
        # room_left 同样按死亡结算：被打死/被传走都算这一局失败。
        #
        # 剩余时间必须用**游戏帧**折算，不能用实测 fps：
        #   * 每步时间代价 = time_price * frame_skip/60 （按"60 帧 = 1 秒"记账）
        #   * 剩余秒数     = 剩余步数   * frame_skip/60
        # 两者用同一个基准，"早死省下的时间代价"正好被"死亡惩罚"抵掉，
        # 净惩罚只剩固定的 DEATH —— 这才是原本的设计意图。
        #
        # 之前用 (max_steps-steps)/fps，一旦游戏帧率不是 60 就会失衡：
        # 用户机器实测 112fps 时，时间代价速率是死亡惩罚速率的 1.87 倍，
        # 等于「死得早反而省钱」，反向鼓励送死。改成按帧记账后与帧率无关。
        dead = (hp_now <= 0) or room_left
        if dead:
            remaining_s = max(0.0, (self.max_steps - self.steps)
                              * self.frame_skip / 60.0)
            r -= DEATH + self.cost_miss * min(remaining_s, 30.0)

        # ---- 本阶段"不允许受伤"：掉血即终局 ----
        # 注意这里**不**追加 DEATH 惩罚：这是课程手段，不是真死亡。
        # 真正的惩罚是"丢掉后续清房奖励 + 已经扣掉的受伤分"。
        hurt_fail = bool(self.fail_on_hurt and took_damage and not dead)
        if hurt_fail:
            terminated_extra = True
        else:
            terminated_extra = False

        # 起点就已清空时不把"clear"当终止条件，避免 reset 死循环
        terminated = bool(dead or hurt_fail
                          or (cleared and not room_left and not self._started_clear))

        info = {
            "dealt": dealt, "kills": kills, "fired": fired,
            "hit_rate": rate, "clear": cleared, "dead": dead,
            "left_room": room_left, "hurt_fail": hurt_fail,
            "hurt": self._ep_hurt, "wall": self._ep_wall,
            "ep_dealt": self._ep_dealt, "ep_fired": self._ep_fired,
        }

        self._prev_enemy_hp, self._prev_alive = cur_hp, cur_alive
        self._prev_clear = cleared
        self._prev_hp = hp_now
        return r * SCALE, terminated, info

    def close(self):
        try:
            self.kc.release_all()
        except Exception:
            pass
