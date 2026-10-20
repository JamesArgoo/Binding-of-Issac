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
ALIGN = 0.2             # 对齐势权重（AIsaac HRW align=0.2）
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
                 hit_rew=HIT_REW, cost_miss=COST_MISS):
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
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.kc.release_all()

        # 1) 重开一局：每回合都重开，保证满血、楼层一致
        #    注意死亡结算画面要好几秒才结束，restart 在期间不生效，所以超时给长一点
        if not self._alive(self._state()):
            for _ in range(2):
                self._cmd("restart")
                if self._wait(lambda st: self._alive(st), timeout=12.0):
                    break
        self._cmd("restart")
        self._wait(lambda st: self._alive(st), timeout=15.0)
        # 关键：重开的房间转场会持续一会儿，紧接着发的 goto 会被吞掉 ——
        # 实测 reset 后玩家还留在起始房间。这里先等转场结束。
        time.sleep(0.9)

        # 2) 传送进一个有敌人的房间
        #    goto 造出来的房间 room.index 恒为 -3（不在楼层地图上），
        #    所以用"房间是否已清空"判断有没有真的进到战斗房间，不行就重发。
        variant = int(self.rng.choice(self.variants))
        s = None
        for _ in range(5):
            self._cmd(f"goto d.{variant}")
            time.sleep(0.8)
            s = self._state()
            if s and "room" in s and not s["room"]["clear"]:
                break
        # 等 grid 到位（地形每 15 帧刷新一次）
        s2 = self._wait(lambda st: self._alive(st) and "grid" in st, timeout=6.0)
        if s2 is not None:
            s = s2

        # 3) 兜底
        if s is None:
            s = {"player": {"x": 0, "y": 0, "vx": 0, "vy": 0, "hp": 0, "soul_hp": 0,
                            "grid_idx": 0, "damage": 3.5},
                 "room": {"index": -1, "clear": False, "enemies_alive": 0},
                 "entities": []}

        # 4) 初始化统计
        self.last_state = s
        self.steps = 0
        self._prev_enemy_hp, self._prev_alive = self._scan_enemies(s)
        self._prev_clear = bool(s.get("room", {}).get("clear"))
        # 开局的房间索引：goto 出来的战斗房恒为 -3。之后只要这个值变了，
        # 就说明玩家已经不在开局那个房间里了（死了被重开 / 被传走）。
        self._room_idx0 = s.get("room", {}).get("index")
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

        # 血量与对齐势的初值（势函数必须从当前状态起算，否则第一步会凭空多出一笔奖励）
        pp = s.get("player", {}) if s else {}
        self._prev_hp = float(pp.get("hp", 0) or 0) + float(pp.get("soul_hp", 0) or 0)
        enemies0 = get_enemies(s) if (s and "player" in s) else []
        if enemies0:
            d0 = min(math.hypot(e["x"] - pp["x"], e["y"] - pp["y"]) for e in enemies0)
            self._prev_phi = -ALIGN * abs(d0 - D_OPT) / CELL
        else:
            self._prev_phi = 0.0

        if "player" not in s:
            return np.zeros(self.obs_dim, dtype=np.float32), {}
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
        if hp_now < self._prev_hp:
            hurt = health_value(self._prev_hp) - health_value(hp_now)
            r -= hurt
            self._ep_hurt += hurt

        # ---- 对齐势（potential-based，不改变结果排序）----
        #     Φ = -ALIGN * |d - D_OPT| / CELL：鼓励停在"刚好能打到敌人"的距离，
        #     太近会被撞、太远打不着。势函数形式保证不会扭曲最终收益排序。
        enemies_all = get_enemies(s)
        if enemies_all:
            d = min(math.hypot(e["x"] - p["x"], e["y"] - p["y"]) for e in enemies_all)
            phi = -ALIGN * abs(d - D_OPT) / CELL
        else:
            phi = 0.0
        r += GAMMA * phi - self._prev_phi
        self._prev_phi = phi

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

        # 起点就已清空时不把"clear"当终止条件，避免 reset 死循环
        terminated = bool(dead or (cleared and not room_left and not self._started_clear))

        info = {
            "dealt": dealt, "kills": kills, "fired": fired,
            "hit_rate": rate, "clear": cleared, "dead": dead,
            "left_room": room_left,
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
