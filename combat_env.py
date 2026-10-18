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
import time
from collections import deque

import gymnasium as gym
import numpy as np
from gymnasium import spaces

from isaac_bridge import get_enemies, get_player
from isaac_env import OBS_DIM, build_obs, normalize_grid

# ---- 动作：移动 0-7 = DIRS8 八方向，8 = 停；射击 0 = 不射，1=上 2=右 3=下 4=左 ----
DIRS8 = [(1, 0), (1, 1), (0, 1), (-1, 1), (-1, 0), (-1, -1), (0, -1), (1, -1)]
SHOOT_DIRS = [(0, 0), (0, -1), (1, 0), (0, 1), (-1, 0)]   # 0=不射 1=上 2=右 3=下 4=左
N_MOVE, N_SHOOT = 9, 5

# ---- 奖励常数（照 AIsaac CombatHitRate 的比例）----
HIT_HP = 5.0            # 每造成 5 点伤害 +1
KILL = 0.25             # 每杀一个 +0.25
COST_HIT = 0.5          # 命中率=1 时每秒的时间代价
COST_MISS = 2.0         # 命中率=0 时每秒的时间代价
DEATH = 5.0             # 死亡
CLEAR = 40.0            # 清房（AIsaac ROOM_POINTS['normal']=40）
SCALE = 0.1             # 整体缩放
FPS = 30.0              # 决策频率（frame_skip=2 对应 30Hz）
RATE_WINDOW = 45        # 命中率统计窗口（步）


def time_price(rate):
    """命中率 -> 每秒时间代价。命中率越高越便宜（线性插值）。"""
    return COST_MISS + (COST_HIT - COST_MISS) * min(1.0, max(0.0, float(rate)))


class CombatEnv(gym.Env):
    """一个实例 = 一个单房间战斗环境。

    bridge: 要有 get_state() 和 send(text)（multi_instance.ChannelAdapter）
    controller: 要有 set_move(mx,my) / set_shoot(ax,ay) / release_all()
    variants: 可用的房间 variant 列表（用 room_catalog.py 探出来）
    """

    metadata = {"render_modes": []}

    def __init__(self, bridge, controller, variants=(12, 13, 14),
                 frame_skip=2, max_steps=900, seed=None):
        super().__init__()
        self.bridge = bridge
        self.kc = controller
        self.variants = list(variants) or [12]
        self.frame_skip = frame_skip
        self.max_steps = max_steps
        self.rng = np.random.default_rng(seed)

        self.action_space = spaces.MultiDiscrete([N_MOVE, N_SHOOT])
        self.observation_space = spaces.Box(-10.0, 10.0, shape=(OBS_DIM,), dtype=np.float32)

        self.grid = None
        self.last_state = None
        self.steps = 0

        self._prev_enemy_hp = {}
        self._prev_alive = set()
        self._prev_clear = False
        self._prev_room = None
        self._seen_tears = set()
        self._rate_window = deque(maxlen=RATE_WINDOW)
        self._ep_dealt = 0.0
        self._ep_fired = 0.0
        self._ep_clear = 0

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

        # 2) 传送进一个有敌人的房间
        #    goto 造出来的房间 room.index 恒为 -3（不在楼层地图上），
        #    所以不能拿"房间号变了"当成功标志，改成固定 settle + 等 grid 到位。
        variant = int(self.rng.choice(self.variants))
        self._cmd(f"goto d.{variant}")
        time.sleep(0.6)
        s = self._wait(lambda st: self._alive(st) and "grid" in st, timeout=8.0)
        if s is None:
            s = self._state()

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
        self._prev_room = s.get("room", {}).get("index")
        self._seen_tears = set()
        self._rate_window.clear()
        self._ep_dealt = 0.0
        self._ep_fired = 0.0
        self._ep_clear = 0

        if "player" not in s:
            return np.zeros(OBS_DIM, dtype=np.float32), {}
        return build_obs(s, self.grid), {"variant": variant, "room": self._prev_room}

    def step(self, action):
        move_a, shoot_a = int(action[0]), int(action[1])
        mx, my = DIRS8[move_a] if move_a < 8 else (0, 0)
        ax, ay = SHOOT_DIRS[shoot_a] if shoot_a < len(SHOOT_DIRS) else (0, 0)
        self.kc.set_move(mx, my)
        self.kc.set_shoot(ax, ay)

        time.sleep(self.frame_skip / 60.0)
        self.steps += 1

        s = self._state()
        self.last_state = s

        if not s or "player" not in s:
            return np.zeros(OBS_DIM, dtype=np.float32), 0.0, False, False, {}

        reward, terminated, info = self._reward(s)
        obs = build_obs(s, self.grid)

        truncated = self.steps >= self.max_steps
        if terminated or truncated:
            self.kc.release_all()
        return obs, float(reward), bool(terminated), bool(truncated), info

    def _reward(self, s):
        p = s["player"]
        cur_hp, cur_alive = self._scan_enemies(s)

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
        r += max(0.0, dealt) / HIT_HP
        r += kills * KILL
        r -= time_price(rate) * (self.frame_skip / 60.0)

        # ---- 清房 ----
        room = s.get("room", {})
        cleared = bool(room.get("clear"))
        if cleared and not self._prev_clear:
            r += CLEAR
            self._ep_clear += 1

        # ---- 死亡 ----
        dead = (p["hp"] + p["soul_hp"]) <= 0
        if dead:
            r -= DEATH

        terminated = bool(dead or cleared)

        info = {
            "dealt": dealt, "kills": kills, "fired": fired,
            "hit_rate": rate, "clear": cleared, "dead": dead,
            "ep_dealt": self._ep_dealt, "ep_fired": self._ep_fired,
        }

        self._prev_enemy_hp, self._prev_alive = cur_hp, cur_alive
        self._prev_clear = cleared
        return r * SCALE, terminated, info

    def close(self):
        try:
            self.kc.release_all()
        except Exception:
            pass
