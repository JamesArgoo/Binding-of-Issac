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
# 玩家特征：x,y,vx,vy,hp,alive + fire_dir(x,y) + head_dir(x,y) + iframes + damage
#           + shot_speed + move_speed = 14
PLAYER_DIM = 14

# ---- 地形编码模式 ----
# "local"：玩家周围 7×7 局部窗口（旧版，49 维）—— 平移不变、房间尺寸无关，
#          但实测只覆盖房间 20%，84% 的墙/石头/坑落在盲区里。
# "full" ：整间房铺到 28×16 固定画布，2 通道（可走 / 危险），896 维 —— 全房间可见。
#          房间贴画布左上角，画布多出来的部分填 0（和墙一样是"走不了"）。
TERRAIN_LOCAL = "local"
TERRAIN_FULL = "full"
GRID_WIN = 7                       # local 模式的窗口边长
CANVAS_W, CANVAS_H = 28, 16        # full 模式的画布：标准房间最大外接框
                                   #   1×1=15×9 / 1×2=15×16 / 2×1=28×9 / 2×2=28×16
TERRAIN_CHANNELS = 2               # full 模式：ch0 可走, ch1 危险

# ---- 视线（line of fire）----
# 从玩家所在格沿 4 个射击方向（上/右/下/左，和 SHOOT_DIRS 1..4 对齐）
# 到第一个"挡子弹"的格子有多远（格数，归一化）。
# 挡子弹 = 网格碰撞类 2物体/3实心/4墙；坑(1)和开着的门(5)不挡。
# 没有这 4 个数，AI 只知道"那里走不进去"，不知道"子弹会撞上去"。
LOS_DIM = 4
LOS_MAX_CELLS = 15.0               # 超过这个距离就当"一路畅通"
LOS_DIRS = ((0, -1), (1, 0), (0, 1), (-1, 0))   # 上 右 下 左

TERRAIN_DIMS = {
    TERRAIN_LOCAL: GRID_WIN * GRID_WIN,                    # 49
    TERRAIN_FULL: CANVAS_W * CANVAS_H * TERRAIN_CHANNELS,  # 896
}


def terrain_dim(terrain_mode=TERRAIN_LOCAL):
    """地形块占多少维。"""
    return TERRAIN_DIMS.get(terrain_mode, TERRAIN_DIMS[TERRAIN_LOCAL])


def obs_dim(terrain_mode=TERRAIN_LOCAL):
    """观测总维度：玩家14 + 敌人16*8 + 子弹48*5 + 地形 + 视线4 + 房间2 + 门12。"""
    return (PLAYER_DIM + MAX_ENEMIES * 8 + MAX_TEARS * 5
            + terrain_dim(terrain_mode) + LOS_DIM + 2 + 4 * 3)


OBS_DIM = obs_dim(TERRAIN_LOCAL)        # 449
OBS_DIM_FULL = obs_dim(TERRAIN_FULL)    # 1296


def terrain_mode_from_dim(dim):
    """反查：观测维度 -> 地形模式。

    加载已有模型时必须用它对齐 —— 模型是按 1292 维（full）训的话，
    环境还用 445 维（local）就会在 predict 时形状不匹配。
    """
    for _m in (TERRAIN_LOCAL, TERRAIN_FULL):
        if obs_dim(_m) == int(dim):
            return _m
    raise ValueError(
        f"观测维度 {dim} 不对应任何已知地形模式"
        f"（local={obs_dim(TERRAIN_LOCAL)}, full={obs_dim(TERRAIN_FULL)}）")

# Direction 枚举: 0=左 1=上 2=右 3=下
_DIR_XY = {0: (-1.0, 0.0), 1: (0.0, -1.0), 2: (1.0, 0.0), 3: (0.0, 1.0)}


def _dir_xy(d):
    """Direction -> 单位向量；无朝向返回 (0,0)。"""
    try:
        return _DIR_XY.get(int(d), (0.0, 0.0))
    except (TypeError, ValueError):
        return (0.0, 0.0)

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


HAZARD_PAD = 28.3   # 危险判定半径的修正量，照 AIsaac abplus_obs._terrain_with_hazards

# 子弹类实体：泪(2) 激光(7) 刀(8) 投射物(9)
_PROJECTILE_TYPES = (2, 7, 8, 9)


def _hazard_cells(state, grid):
    """带接触伤害的实体（火堆、火焰等）会伤到的格子集合。

    AIsaac 的做法：cdmg > 0 的实体，凡是格心到它中心距离 <= size + 28.3 的格子都算危险。
    没有这个，AI 完全看不见火堆，只能靠掉血后才知道疼。

    **实测踩过的坑（2026-10 用真数据查出来）**：原判据只跳过玩家本体 `type == 1`，
    于是玩家自己射出的泪弹（`type=2 spawner_type=1 cdmg=3.5`）也被标成危险格 ——
    那一帧 7×7 视野里唯一的 0.5 就是玩家刚打出去的那发子弹，等于在教 AI
    "别站在自己子弹的位置"。现在同时跳过：玩家本体、玩家发射的一切、所有弹幕。

    弹幕不进危险格的理由：弹幕块（5×48）已经有精确位置和速度用于预判，而地形
    每 15 帧才刷新一次，把会动的东西画进地形只会得到过期 250ms 的噪声，
    还会让"地形"这个本该静态的通道时刻抖动。
    """
    if not grid:
        return frozenset()
    coll = grid.get("coll")
    if not coll:
        return frozenset()
    tw = float(grid.get("tw", 40.0)) or 40.0
    th = float(grid.get("th", 40.0)) or 40.0
    ox = float(grid.get("ox", 0.0))
    oy = float(grid.get("oy", 0.0))
    w, h = grid["w"], grid["h"]
    out = set()
    for e in state.get("entities", []):
        try:
            cdmg = float(e.get("cdmg", 0) or 0)
        except (TypeError, ValueError):
            continue
        if cdmg <= 0:
            continue
        if int(e.get("type", 0)) == 1:                          # 玩家本体
            continue
        if int(e.get("spawner_type", 0) or 0) == 1:             # 玩家发射的东西
            continue
        if int(e.get("type", 0)) in _PROJECTILE_TYPES:          # 会动的弹幕
            continue
        r = float(e.get("size", 10.0) or 0) + HAZARD_PAD
        r2 = r * r
        ex, ey = e["x"], e["y"]
        gx0 = max(0, int((ex - r - ox) / tw))
        gx1 = min(w - 1, int((ex + r - ox) / tw) + 1)
        gy0 = max(0, int((ey - r - oy) / th))
        gy1 = min(h - 1, int((ey + r - oy) / th) + 1)
        for gy in range(gy0, gy1 + 1):
            cy = oy + (gy + 0.5) * th
            for gx in range(gx0, gx1 + 1):
                cx = ox + (gx + 0.5) * tw
                if (cx - ex) ** 2 + (cy - ey) ** 2 <= r2:
                    out.add(gy * w + gx)
    return out


def _room_scale(grid):
    """房间在世界坐标里的宽、高、左上角（借鉴 AIsaac：按房间尺寸归一化，而不是固定 /500）。"""
    if grid is not None and "ox" in grid and "tw" in grid and "th" in grid:
        rw = max(grid["w"] * grid["tw"], 1.0)
        rh = max(grid["h"] * grid["th"], 1.0)
        return rw, rh, grid["ox"], grid["oy"]
    return POS_SCALE, POS_SCALE, 0.0, 0.0


# ---------------- 地形编码：两种模式 ----------------

def _terrain_local(state, grid):
    """玩家周围 7×7：0=走不了 / 0.5=能走但危险 / 1=能走且安全。"""
    if grid is None or state.get("player", {}).get("grid_idx") is None:
        return [0.0] * (GRID_WIN * GRID_WIN)
    p = state["player"]
    w, h = grid["w"], grid["h"]
    gx, gy = p["grid_idx"] % w, p["grid_idx"] // w
    r = GRID_WIN // 2
    haz = _hazard_cells(state, grid)
    out = []
    for dy in range(-r, r + 1):
        for dx in range(-r, r + 1):
            cx, cy = gx + dx, gy + dy
            if not cell_walkable(cx, cy, grid):
                out.append(0.0)
            elif cy * w + cx in haz:
                out.append(0.5)          # 能走，但站上去会掉血
            else:
                out.append(1.0)
    return out


def terrain_full(state, grid):
    """整间房铺到 CANVAS_H×CANVAS_W 画布，返回 (2, H, W) 的 float32。

        ch0 可走   : 1=玩家能站
        ch1 危险   : 1=能站但会掉血（火堆/火焰等接触伤害实体覆盖到）

    房间贴画布左上角；画布比房间大的部分保持 0（等价于"走不了"）。
    比 7×7 窗口的代价：失去平移不变性（网络得自己学"墙在哪"），
    但换来的是整间房可见 —— 实测 7×7 有 84% 的墙/障碍在盲区。
    """
    out = np.zeros((TERRAIN_CHANNELS, CANVAS_H, CANVAS_W), dtype=np.float32)
    if not grid:
        return out
    gw, gh = int(grid["w"]), int(grid["h"])
    cells = grid.get("cells")
    if not cells:
        return out
    need = gw * gh
    arr = np.asarray(cells, dtype=np.float32)
    if arr.size < need:
        return out
    walk = arr[:need].reshape(gh, gw)

    h = min(gh, CANVAS_H)
    w = min(gw, CANVAS_W)
    out[0, :h, :w] = walk[:h, :w]

    haz = _hazard_cells(state, grid)
    if haz:
        hm = np.zeros(need, dtype=np.float32)
        hm[[i for i in haz if 0 <= i < need]] = 1.0
        hm = hm.reshape(gh, gw)
        # 危险只在"能走上去"的格子上有意义，别和墙的信息混在一起
        out[1, :h, :w] = hm[:h, :w] * walk[:h, :w]
    return out


def _terrain_block(state, grid, terrain_mode):
    """按模式返回摊平后的地形特征（list）。"""
    if terrain_mode == TERRAIN_FULL:
        return terrain_full(state, grid).ravel().tolist()
    return _terrain_local(state, grid)


def line_of_fire(state, grid):
    """玩家沿 4 个射击方向（上/右/下/左）到第一个挡子弹格子的距离。

    返回长度 4 的 list，值 0~1（格数 / LOS_MAX_CELLS）。
    索引顺序与射击动作 1=上 2=右 3=下 4=左 一一对应，方便网络直接对照。

    为什么需要：泪弹撞到石头/墙就没了，但地形块只告诉 AI"那一格走不进去"，
    没有任何一维告诉它"往这个方向射会打在石头上"。实测对局里玩家泪弹
    大量打在墙上，这些发数全被算成"打空"，命中率被拉低、时间代价被拉高。
    """
    blocked = (2, 3, 4)          # 物体 / 实心 / 墙
    # 没有地形时说"一路畅通"是危险的假信息（会鼓励它乱开枪）。
    # 反向对齐地形块：grid 缺失时地形块全是 0（哪里都走不了），
    # 所以视线也给 0（哪里都打不通），两者语义一致。
    if not grid:
        return [0.0] * LOS_DIM
    p = state.get("player", {})
    gidx = p.get("grid_idx")
    if gidx is None:
        return [0.0] * LOS_DIM
    out = [LOS_MAX_CELLS] * LOS_DIM
    w, h = grid["w"], grid["h"]
    coll = grid.get("coll") or []
    gx, gy = int(gidx) % w, int(gidx) // w
    span = int(LOS_MAX_CELLS)
    for k, (dx, dy) in enumerate(LOS_DIRS):
        dist = LOS_MAX_CELLS
        for step in range(1, span + 1):
            cx, cy = gx + dx * step, gy + dy * step
            if not (0 <= cx < w and 0 <= cy < h):
                dist = step - 1               # 出界 = 走到房间边缘
                break
            i = cy * w + cx
            c = int(coll[i]) if i < len(coll) else 4
            if c in blocked:
                dist = step - 1               # 停在障碍前的最后一格
                break
        out[k] = dist
    return [min(v, LOS_MAX_CELLS) / LOS_MAX_CELLS for v in out]


def build_obs(state, grid, terrain_mode=TERRAIN_LOCAL):
    """把一帧状态压成固定长度观测向量。

    相比旧版（借鉴 AIsaac observation.rs 的实体特征设计）：
      * 位置按房间真实尺寸归一化（0~1），不再用固定 /500
      * 敌人额外带碰撞体积 size 和碰撞类别 coll（判断接触威胁）
      * 子弹额外带高度 height（抛射/越障弹需要）

    terrain_mode:
      * TERRAIN_LOCAL（默认，445 维）：地形只看玩家周围 7×7
      * TERRAIN_FULL （1292 维）：地形铺满整间房（28×16 画布 × 可走/危险 2 通道）
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

    # 朝向：没有这个 AI 不知道自己正朝哪射（泪弹方向 = 射击键 + 玩家速度），"瞄不准"的直接原因
    fx, fy = _dir_xy(p.get("fire_dir"))
    obs.append(fx)
    obs.append(fy)
    hx, hy = _dir_xy(p.get("head_dir"))
    obs.append(hx)
    obs.append(hy)
    obs.append(_clip(float(p.get("iframes", 0) or 0) / 60.0))          # 无敌帧
    obs.append(_clip(float(p.get("damage", 3.5) or 3.5) / 10.0))        # 单发伤害
    obs.append(_clip(float(p.get("shot_speed", 10.0) or 10.0) / 10.0))  # 泪速（算提前量）
    obs.append(_clip(float(p.get("move_speed", 1.0) or 1.0) / 2.0))     # 移速（甩枪幅度）

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

    # 地形：local=玩家周围 7×7（0/0.5/1），full=整间房 28×16×2 通道
    obs.extend(_terrain_block(state, grid, terrain_mode))
    # 视线：4 个射击方向到障碍物的距离（解决"子弹打在墙上"）
    obs.extend(line_of_fire(state, grid))

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
