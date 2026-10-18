# -*- coding: utf-8 -*-
"""
isaac_env.py —— 把"以撒 + Lua 桥接 + pyautogui"包装成 Gymnasium 强化学习环境

设计：
  观测：玩家(6) + 最近16个敌人(6*16) + 最近48个敌方泪(4*48) + 玩家周围7x7网格(49) + 房间(2)
  动作：MultiDiscrete([9, 9]) —— 移动(8方向+不动) × 射击(8方向+不射)
  奖励：
    * 受伤：-5/点血
    * 打中敌人：+0.2/点血
    * 击杀：+10
    * 清房：+50
    * 死亡：-100（episode 结束）
    * 每帧：-0.005（逼它别挂机）
  重置：玩家死亡后，模组会自动 ExecuteCommand("restart") 重开，环境等待新局开始。

注意：这是实时环境，step() 会 sleep 若干帧，训练速度受游戏帧率限制。
"""
import math
import time

import gymnasium as gym
import numpy as np
import pyautogui
from gymnasium import spaces

# 关键！pyautogui 默认每次调用后 sleep 0.1s，会让训练慢到 2 FPS，必须清零
pyautogui.PAUSE = 0.0

from isaac_bridge import (IsaacBridge, get_enemies, get_enemy_projectiles, get_player)

# ---- 尺寸参数 ----
MAX_ENEMIES = 16
MAX_TEARS = 48
GRID_WIN = 7  # 7x7 局部网格
# 观测维度（借鉴 AIsaac：实体带碰撞体积/类别，子弹带高度）
#   玩家6 + 敌人16*8 + 子弹48*5 + 网格49 + 房间2 + 门12
OBS_DIM = 6 + MAX_ENEMIES * 8 + MAX_TEARS * 5 + GRID_WIN * GRID_WIN + 2 + 4 * 3  # 437

# 8 方向：0=右 1=右下 2=下 3=左下 4=左 5=左上 6=上 7=右上
DIRS8 = [(1, 0), (1, 1), (0, 1), (-1, 1), (-1, 0), (-1, -1), (0, -1), (1, -1)]

# 归一化尺度（位置改成按房间尺寸归一化，见 _room_scale）
POS_SCALE = 500.0        # 无网格数据时的兜底房间尺寸
VEL_SCALE = 10.0
HP_SCALE = 12.0
ENEMY_HP_SCALE = 50.0

# 进度势（借鉴 AIsaac combat-v5 的对齐势）：Φ = -POTENTIAL_SCALE * d / POTENTIAL_DIST
# 用 γΦ(s') − Φ(s) 鼓励接近敌人，替代"每步扣分"（每步扣分会逼 AI 无意义绕圈）
POTENTIAL_SCALE = 0.3
POTENTIAL_DIST = 300.0
GAMMA = 0.99


def _clip(v, lo=-10.0, hi=10.0):
    return max(lo, min(hi, v))


class KeyController:
    def __init__(self):
        self._move = set()
        self._shoot = set()

    def _apply(self, current, want):
        for k in current - want:
            pyautogui.keyUp(k)
        for k in want - current:
            pyautogui.keyDown(k)
        return want

    def set_move(self, mx, my):
        want = set()
        if mx > 0:
            want.add("d")
        elif mx < 0:
            want.add("a")
        if my > 0:
            want.add("s")
        elif my < 0:
            want.add("w")
        self._move = self._apply(self._move, want)

    def set_shoot(self, ax, ay):
        want = set()
        if ax > 0:
            want.add("right")
        elif ax < 0:
            want.add("left")
        if ay > 0:
            want.add("down")
        elif ay < 0:
            want.add("up")
        self._shoot = self._apply(self._shoot, want)

    def release_all(self):
        for k in list(self._move):
            pyautogui.keyUp(k)
        for k in list(self._shoot):
            pyautogui.keyUp(k)
        self._move.clear()
        self._shoot.clear()


def cell_walkable(gx, gy, grid):
    if grid is None:
        return False
    if gx < 0 or gy < 0 or gx >= grid["w"] or gy >= grid["h"]:
        return False
    return grid["cells"][gy * grid["w"] + gx] == 1


# 玩家能不能走：0=空地；5=WALL_EXCEPT_PLAYER（门就是这一类，玩家能穿过去）
WALKABLE_COLL = (0, 5)


def normalize_grid(grid):
    """按"玩家能不能走"重算 grid["cells"]。

    模组里 cells 只把碰撞类别 0 当可走，可门是 coll=5（WALL_EXCEPT_PLAYER）。
    结果 A* 永远到不了门 —— 规则AI被锁死在房间里出不去，这曾是最致命的 bug。
    这里直接用 coll 重算：0 和 5 都算可走，坑(1)/物体(2)/实心(3)/真墙(4)都不算。
    """
    if not grid:
        return grid
    coll = grid.get("coll")
    if not coll:
        return grid
    grid["cells"] = [1 if c in WALKABLE_COLL else 0 for c in coll]
    return grid


def _room_scale(grid):
    """房间在世界坐标里的宽、高、左上角（借鉴 AIsaac：按房间尺寸归一化，而不是固定 /500）。"""
    if grid is not None and "ox" in grid and "tw" in grid and "th" in grid:
        rw = max(grid["w"] * grid["tw"], 1.0)
        rh = max(grid["h"] * grid["th"], 1.0)
        return rw, rh, grid["ox"], grid["oy"]
    return POS_SCALE, POS_SCALE, 0.0, 0.0


def build_obs(state, grid):
    """把一帧状态压成固定长度观测向量。

    相比旧版（借鉴 AIsaac observation.rs 的实体特征设计）：
      * 位置按房间真实尺寸归一化（0~1），不再用固定 /500
      * 敌人额外带碰撞体积 size 和碰撞类别 coll（判断接触威胁）
      * 子弹额外带高度 height（抛射/越障弹需要）
    """
    obs = []
    p = state["player"]
    rw, rh, ox, oy = _room_scale(grid)

    # 玩家
    obs.append(_clip((p["x"] - ox) / rw))
    obs.append(_clip((p["y"] - oy) / rh))
    obs.append(_clip(p["vx"] / VEL_SCALE))
    obs.append(_clip(p["vy"] / VEL_SCALE))
    obs.append(_clip((p["hp"] + p["soul_hp"]) / HP_SCALE))
    obs.append(1.0 if p["hp"] > 0 else 0.0)

    # 敌人（按距离排序，取最近的 MAX_ENEMIES 个；每个 8 维）
    enemies = get_enemies(state)
    enemies.sort(key=lambda e: (e["x"] - p["x"]) ** 2 + (e["y"] - p["y"]) ** 2)
    for i in range(MAX_ENEMIES):
        if i < len(enemies):
            e = enemies[i]
            obs.append(_clip((e["x"] - p["x"]) / rw))
            obs.append(_clip((e["y"] - p["y"]) / rh))
            obs.append(_clip(e["vx"] / VEL_SCALE))
            obs.append(_clip(e["vy"] / VEL_SCALE))
            obs.append(_clip(e["hp"] / ENEMY_HP_SCALE))
            obs.append(1.0 if e["boss"] else 0.0)
            obs.append(_clip(e.get("size", 10.0) / 40.0))   # 碰撞体积（一格 40 单位）
            obs.append(_clip(e.get("coll", 0) / 5.0))        # 碰撞类别
        else:
            obs.extend([0.0] * 8)

    # 子弹（泪/激光/刀/投射物，每个 5 维，含高度）
    tears = get_enemy_projectiles(state)
    tears.sort(key=lambda t: (t["x"] - p["x"]) ** 2 + (t["y"] - p["y"]) ** 2)
    for i in range(MAX_TEARS):
        if i < len(tears):
            t = tears[i]
            obs.append(_clip((t["x"] - p["x"]) / rw))
            obs.append(_clip((t["y"] - p["y"]) / rh))
            obs.append(_clip(t["vx"] / VEL_SCALE))
            obs.append(_clip(t["vy"] / VEL_SCALE))
            obs.append(_clip(t.get("height", 0.0) / 40.0))   # 泪弹高度
        else:
            obs.extend([0.0] * 5)

    # 玩家周围 7x7 可行走网格
    if grid is not None and p.get("grid_idx") is not None:
        w, h = grid["w"], grid["h"]
        gx, gy = p["grid_idx"] % w, p["grid_idx"] // w
        r = GRID_WIN // 2
        for dy in range(-r, r + 1):
            for dx in range(-r, r + 1):
                obs.append(1.0 if cell_walkable(gx + dx, gy + dy, grid) else 0.0)
    else:
        obs.extend([0.0] * (GRID_WIN * GRID_WIN))

    # 房间
    obs.append(_clip(state["room"]["enemies_alive"] / 20.0))
    obs.append(1.0 if state["room"]["clear"] else 0.0)

    # 门（4 个正方向：相对玩家的网格方向 + 是否打开）
    doors = {d["slot"]: d for d in state.get("doors", [])}
    if grid is not None and p.get("grid_idx") is not None:
        w, h = grid["w"], grid["h"]
        gx, gy = p["grid_idx"] % w, p["grid_idx"] // w
        for slot in range(4):
            d = doors.get(slot)
            if d is not None:
                dgx, dgy = d["grid"] % w, d["grid"] // w
                obs.append(_clip((dgx - gx) / w))
                obs.append(_clip((dgy - gy) / h))
                obs.append(1.0 if d["open"] else 0.0)
            else:
                obs.extend([0.0, 0.0, 0.0])
    else:
        obs.extend([0.0] * 12)

    return np.array(obs, dtype=np.float32)


class IsaacEnv(gym.Env):
    metadata = {"render_modes": []}

    def __init__(self, frame_skip=2, port=None, controller=None, bridge=None):
        super().__init__()
        self.frame_skip = frame_skip
        self.action_space = spaces.MultiDiscrete([9, 9])
        self.observation_space = spaces.Box(-10.0, 10.0, shape=(OBS_DIM,), dtype=np.float32)

        # 多开时由 InstanceManager 注入 bridge + controller（各实例端口/窗口都不同）；
        # 单开时自己起桥接、用 pyautogui 打当前焦点窗口。
        if bridge is not None:
            self.bridge = bridge
            self._owns_bridge = False
        else:
            self.bridge = IsaacBridge(port=port) if port is not None else IsaacBridge()
            self.bridge.start()
            self._owns_bridge = True
        self.kc = controller if controller is not None else KeyController()
        self.grid = None

        self.prev_hp = 0.0
        self.prev_enemy_hp = {}
        self.prev_clear = False
        self.prev_room_index = None
        self.last_state = None
        self.prev_phi = 0.0  # 进度势的上一帧值（借鉴 AIsaac 的势函数）

    # ---------- 工具 ----------
    def _alive(self, s):
        if not s or "player" not in s:
            return False
        p = s["player"]
        return (p["hp"] + p["soul_hp"]) > 0

    def _wait_alive(self, timeout=60.0):
        t0 = time.time()
        while time.time() - t0 < timeout:
            s = self.bridge.get_state()
            if s and "grid" in s:
                self.grid = normalize_grid(s["grid"])
            if self._alive(s):
                return s
            time.sleep(0.05)
        return None

    # ---------- Gym 接口 ----------
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.kc.release_all()
        s = self._wait_alive()
        if s is None:
            # 兜底：返回全零观测，避免训练直接崩
            return np.zeros(OBS_DIM, dtype=np.float32), {}

        p = s["player"]
        self.last_state = s
        self.prev_hp = p["hp"] + p["soul_hp"]
        self.prev_enemy_hp = {e["id"]: e["hp"] for e in get_enemies(s)}
        self.prev_clear = s["room"]["clear"]
        self.prev_phi = 0.0
        return build_obs(s, self.grid), {}

    def step(self, action):
        move_a, shoot_a = int(action[0]), int(action[1])
        mx, my = DIRS8[move_a] if move_a < 8 else (0, 0)
        ax, ay = DIRS8[shoot_a] if shoot_a < 8 else (0, 0)
        self.kc.set_move(mx, my)
        self.kc.set_shoot(ax, ay)

        # 按住动作跑 frame_skip 帧
        time.sleep(self.frame_skip / 60.0)

        s = self.bridge.get_state()
        self.last_state = s
        if s and "grid" in s:
            self.grid = normalize_grid(s["grid"])

        terminated = False
        reward = 0.0
        if s and "player" in s:
            reward, terminated = self._reward(s, shoot_a)
        else:
            reward = 0.0

        obs = build_obs(s, self.grid) if (s and "player" in s) else np.zeros(OBS_DIM, dtype=np.float32)

        if terminated:
            self.kc.release_all()

        return obs, float(reward), bool(terminated), False, {}

    def _reward(self, s, shoot_a=8):
        p = s["player"]
        total_hp = p["hp"] + p["soul_hp"]
        room_idx = s["room"]["index"]
        room_changed = (self.prev_room_index is not None and room_idx != self.prev_room_index)

        reward = 0.0

        # 受伤（红心+魂心减少）—— 换房间时也继续算（掉了血就是掉了）
        dmg = self.prev_hp - total_hp
        if dmg > 0:
            reward -= dmg * 5.0

        # 打中 / 击杀 —— 换房间时敌人列表全变，跳过（避免误判成击杀）
        cur_enemy_hp = {e["id"]: e["hp"] for e in get_enemies(s)}
        if not room_changed:
            dealt = 0.0
            for eid, hp in self.prev_enemy_hp.items():
                if eid in cur_enemy_hp:
                    d = hp - cur_enemy_hp[eid]
                    if d > 0:
                        dealt += d
            reward += dealt * 0.2

            cur_alive = {eid for eid, hp in cur_enemy_hp.items() if hp > 0}
            for eid, hp in self.prev_enemy_hp.items():
                if hp > 0 and eid not in cur_alive:
                    reward += 10.0

        # 清房
        if not self.prev_clear and s["room"]["clear"]:
            reward += 50.0

        # 进入新房间（鼓励推进，而不是在清空房间里瞎逛）
        if room_changed:
            reward += 5.0

        # 进度势（借鉴 AIsaac combat-v5）：Φ = -POTENTIAL_SCALE * d / POTENTIAL_DIST
        # 用 γΦ(s') − Φ(s) 鼓励接近敌人；不再有"每步扣分"（那会逼 AI 无意义绕圈）
        enemies = get_enemies(s)
        if enemies:
            d = min(math.hypot(e["x"] - p["x"], e["y"] - p["y"]) for e in enemies)
            phi = -POTENTIAL_SCALE * d / POTENTIAL_DIST
        else:
            phi = 0.0
        reward += GAMMA * phi - self.prev_phi
        self.prev_phi = phi

        # 死亡
        terminated = total_hp <= 0
        if terminated:
            reward -= 100.0

        # 更新上一帧
        self.prev_hp = total_hp
        self.prev_enemy_hp = cur_enemy_hp
        self.prev_clear = s["room"]["clear"]
        self.prev_room_index = room_idx
        return reward, terminated

    def close(self):
        self.kc.release_all()
        if self._owns_bridge:
            self.bridge.stop()
