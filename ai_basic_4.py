# -*- coding: utf-8 -*-
"""
ai_basic_4.py —— 规则版 AI：8 方向移动 + 4 方向射击（甩泪）

与 ai_basic.py 的区别：
  * 移动：8 方向（与 ai_basic 相同，W+A 即左上 45°）
  * 射击：**4 方向**（上/下/左/右），但泪弹会继承人物移动速度，
    斜向命中靠"边移动边打"的矢量合成（甩泪弹）；
    选射击方向时综合考虑了 泪速 + 人物移速，见 choose_aim4()

行为：
  * 房间有敌人：追最近的敌人（A* 绕障碍）+ 打 + 躲敌方泪弹
  * 房间清空：A* 走到最近的开门（自动进入下一间）

用法：游戏进对局保持前台 -> python ai_basic_4.py [--debug] -> 5 秒内切回游戏。
按 'p' 暂停/继续（需 pip install keyboard），Ctrl+C 退出。
"""
import argparse
import math
import time

import cv2
import numpy as np
import pyautogui

from isaac_bridge import (IsaacBridge, get_enemies, get_enemy_projectiles,
                          get_player, nearest_enemy)

pyautogui.FAILSAFE = True
pyautogui.PAUSE = 0.0

# ---- 参数（Isaac 世界坐标；Velocity 是"每帧位移"，tca 单位=帧）----
IDEAL_DIST = 160.0
DEFAULT_SIZE = 10.0      # 实体缺省碰撞半径（读不到 size 时兜底）
DODGE_FRAMES = 25.0      # 子弹多少帧内会命中就躲（≈0.4 秒）
DODGE_MARGIN = 25.0      # 在"双方碰撞半径之和"外再留的安全距离
DODGE_STRENGTH = 6.0
CHASE_WEIGHT = 1.0
DODGE_WEIGHT = 3.0
ENEMY_MARGIN = 20.0      # 敌人排斥：双方半径之和 + 此余量内就推开
ENEMY_REPEL_WEIGHT = 1.5
LEAD_MAX_FRAMES = 60.0   # 瞄准预判最大帧数（≈1 秒）
LOOKAHEAD_FRAMES = 8     # 前瞻模拟帧数（躲弹/避障用）

# 8 方向（射击用）：0=右 1=右下 2=下 3=左下 4=左 5=左上 6=上 7=右上
DIRS8 = [(1, 0), (1, 1), (0, 1), (-1, 1), (-1, 0), (-1, -1), (0, -1), (1, -1)]
# 4 方向（移动用）：右/下/左/上
DIRS4 = [(1, 0), (0, 1), (-1, 0), (0, -1)]

try:
    import keyboard
    HAS_KEYBOARD = True
except ImportError:
    HAS_KEYBOARD = False


def norm(x, y):
    m = math.hypot(x, y)
    if m < 1e-6:
        return 0.0, 0.0
    return x / m, y / m


def dir8(ax, ay):
    if abs(ax) < 1e-6 and abs(ay) < 1e-6:
        return 0, 0
    angle = math.atan2(ay, ax)
    return DIRS8[int(round(angle / (math.pi / 4.0))) % 8]


def dir4(ax, ay):
    """把方向向量量化到 4 方向（取主轴）。射击方向用。"""
    if abs(ax) < 1e-6 and abs(ay) < 1e-6:
        return (0, 0)
    if abs(ax) >= abs(ay):
        return (1 if ax > 0 else -1, 0)
    return (0, 1 if ay > 0 else -1)


def choose_aim4(desired, player_vel, tear_speed):
    """在 4 个射击方向里，选"加上人物速度后"最接近 desired 的那个（甩泪）。

    泪弹实际速度 ≈ 射击方向 × 泪速 + 人物移动速度，
    所以斜向命中靠"边移动边打"的矢量合成（甩泪弹）。
    """
    best_dir, best_dot = (0, 0), -2.0
    for d in DIRS4:
        tvx = d[0] * tear_speed + player_vel[0]
        tvy = d[1] * tear_speed + player_vel[1]
        m = math.hypot(tvx, tvy)
        if m < 1e-6:
            continue
        dot = (tvx / m) * desired[0] + (tvy / m) * desired[1]
        if dot > best_dot:
            best_dot = dot
            best_dir = d
    return best_dir


def _ang_diff(a, b):
    d = (a - b) % (2 * math.pi)
    if d > math.pi:
        d -= 2 * math.pi
    return abs(d)


def cell_walkable(gx, gy, grid):
    if grid is None or gx < 0 or gy < 0 or gx >= grid["w"] or gy >= grid["h"]:
        return False
    return grid["cells"][gy * grid["w"] + gx] == 1


def wall_steer(mx, my, gidx, grid):
    gx, gy = gidx % grid["w"], gidx // grid["w"]
    angle = math.atan2(my, mx)
    ordered = sorted(DIRS8, key=lambda d: _ang_diff(math.atan2(d[1], d[0]), angle))
    for dx, dy in ordered:
        if cell_walkable(gx + dx, gy + dy, grid):
            return dx, dy
    return 0, 0


def astar(grid, start, goal):
    w, h = grid["w"], grid["h"]
    cells = grid["cells"]
    n = w * h
    if not (0 <= start < n and 0 <= goal < n):
        return None
    if cells[start] == 0 or cells[goal] == 0:
        return None

    def heuristic(a, b):
        return abs(a % w - b % w) + abs(a // w - b // w)

    import heapq
    open_heap = [(0, start)]
    came = {}
    g = {start: 0}
    while open_heap:
        _, cur = heapq.heappop(open_heap)
        if cur == goal:
            path = [goal]
            while path[-1] != start:
                path.append(came[path[-1]])
            path.reverse()
            return path
        cx, cy = cur % w, cur // w
        for dx, dy in ((1, 0), (-1, 0), (0, 1), (0, -1)):
            nx, ny = cx + dx, cy + dy
            if 0 <= nx < w and 0 <= ny < h:
                nb = ny * w + nx
                if cells[nb] == 1:
                    ng = g[cur] + 1
                    if nb not in g or ng < g[nb]:
                        g[nb] = ng
                        came[nb] = cur
                        heapq.heappush(open_heap, (ng + heuristic(nb, goal), nb))
    return None


def nearest_open_door_grid(state, grid):
    """返回离玩家最近的开门格子索引（没有则 None）。"""
    p = get_player(state)
    if grid is None or p.get("grid_idx") is None:
        return None
    open_doors = [d for d in state.get("doors", []) if d.get("open")]
    if not open_doors:
        return None
    w = grid["w"]

    def gdist(d):
        return abs(d["grid"] % w - p["grid_idx"] % w) + abs(d["grid"] // w - p["grid_idx"] // w)

    return min(open_doors, key=gdist)["grid"]


def _neighbor_room_grid(room_grid, slot):
    """门 slot: 0=上 1=右 2=下 3=左；楼层房间地图是 13x13。"""
    return room_grid + {0: -13, 1: 1, 2: 13, 3: -1}.get(slot, 0)


def _go_to_door(state, grid, last_door=None, recent=None):
    """房间清空时，走到最近的开门。返回 8 方向 (mx, my)。"""
    p = get_player(state)
    if grid is None or p.get("grid_idx") is None:
        return (0, 0)
    open_doors = [d for d in state.get("doors", []) if d.get("open")]
    if not open_doors:
        return (0, 0)
    w = grid["w"]

    def gdist(d):
        return abs(d["grid"] % w - p["grid_idx"] % w) + abs(d["grid"] // w - p["grid_idx"] // w)

    # 第一优先：通往没探过的房间的门。
    # 只靠"避开刚来的那扇门"是不够的 —— 实测会在两个已清空房间之间无限来回弹，
    # 永远遇不到敌人（录了4分钟只走了2个房间）。
    room_grid = state["room"]["index"]
    visited = {r["grid"]: r["visited"] for r in state.get("map", [])}

    def _ng(d):
        return _neighbor_room_grid(room_grid, d.get("slot", -1))

    # 按"邻居房间被访问过几次"打分选门 —— 这就是记忆性：
    #   去过越多次的房间越不去（0 次 = 没去过，最优先）
    #   最近刚去过的房间强烈回避（治 A-B-A-B 打转）
    #   刚走过来的那扇门也回避
    # 这样即使 B 是只有一扇门的死胡同，去过几次后 visit 变大，就不再反复进去了。
    recent_set = set(recent or [])

    def _door_score(d):
        ng = _ng(d)
        v = visited.get(ng, 0)
        pen = 0
        if ng in recent_set:
            pen += 100
        if d["grid"] == last_door:
            pen += 50
        return (v + pen, gdist(d))

    candidates = list(open_doors)
    best = min(candidates, key=_door_score)
    path = astar(grid, p["grid_idx"], best["grid"])
    if path and len(path) >= 2:
        cur, nxt = path[0], path[1]
        return dir8((nxt % w) - (cur % w), (nxt // w) - (cur // w))

    # 已经站在门格上：A* 的路径长度只有 1，如果直接返回 (0,0) 就会永远卡在门上。
    # 这里改为继续朝"门外"推 —— 门在房间边界上，方向 = 从房间中心指向门的那一侧。
    dgx, dgy = best["grid"] % w, best["grid"] // w
    pgx, pgy = p["grid_idx"] % w, p["grid_idx"] // w
    if (dgx, dgy) == (pgx, pgy):
        h = grid["h"]
        out = dir8(dgx - w // 2, dgy - h // 2)
        if out != (0, 0):
            return out
        # 兜底：从玩家真实世界位置往门格中心推
        ddx = (dgx + 0.5) * grid.get("tw", 40) + grid.get("ox", 0) - p["x"]
        ddy = (dgy + 0.5) * grid.get("th", 40) + grid.get("oy", 0) - p["y"]
        if abs(ddx) > 1e-3 or abs(ddy) > 1e-3:
            return dir8(ddx, ddy)
    return (0, 0)


def choose_target(state, p):
    """按威胁度选目标：Boss > 正在靠近的 > 距离近的（不只看最近）。"""
    enemies = get_enemies(state)
    if not enemies:
        return None, enemies
    best, best_score = None, -1e18
    for e in enemies:
        dx, dy = p["x"] - e["x"], p["y"] - e["y"]
        dist = math.hypot(dx, dy) or 1.0
        score = 0.0
        if e.get("boss"):
            score += 100.0
        approach = (e["vx"] * dx + e["vy"] * dy) / dist  # 朝玩家方向的速度分量
        score += approach * 8.0
        score += 200.0 / (dist + 60.0)
        if score > best_score:
            best, best_score = e, score
    return best, enemies


def world_to_grid(wx, wy, grid):
    """世界坐标 -> 网格坐标（需要 grid 里有 ox/oy/tw/th）。"""
    if grid is None or "ox" not in grid:
        return None
    gx = int((wx - grid["ox"]) // grid["tw"])
    gy = int((wy - grid["oy"]) // grid["th"])
    return gx, gy


def cell_walkable_world(wx, wy, grid):
    """世界坐标所在格子是否可走（无网格数据时默认 True）。"""
    if grid is None or "ox" not in grid:
        return True
    g = world_to_grid(wx, wy, grid)
    gx, gy = g
    if gx < 0 or gy < 0 or gx >= grid["w"] or gy >= grid["h"]:
        return False
    return grid["cells"][gy * grid["w"] + gx] == 1


def plan_move(state, grid, p, target, enemies, projectiles, debug=None):
    """候选方向采样 + 前瞻模拟：选未来最安全的方向（替代力场求和，避免局部最优）。"""
    p_size = p.get("size", DEFAULT_SIZE)
    move_speed = max(p.get("move_speed", 2.5), 1e-3)  # 单位/帧
    K = LOOKAHEAD_FRAMES
    candidates = list(DIRS8) + [(0, 0)]
    best_dir, best_score = (0, 0), -1e18
    dist_now = math.hypot(target["x"] - p["x"], target["y"] - p["y"]) if target is not None else 0.0
    results = []  # 每个候选方向的得分（调试可视化用）

    for d in candidates:
        score = 0.0
        sx, sy = p["x"], p["y"]
        for t in range(1, K + 1):
            sx += d[0] * move_speed
            sy += d[1] * move_speed
            weight = 1.0 - (t - 1) / K  # 越早撞越严重
            # 墙 / 障碍
            if not cell_walkable_world(sx, sy, grid):
                score -= 1000.0 * weight
                break
            # 子弹（所有敌方子弹，预判位置）
            for tpro in projectiles:
                bx = tpro["x"] + tpro["vx"] * t
                by = tpro["y"] + tpro["vy"] * t
                dd = math.hypot(bx - sx, by - sy)
                danger = p_size + tpro.get("size", DEFAULT_SIZE)
                if dd < danger:
                    score -= (danger - dd + 1.0) * 20.0 * weight
            # 敌人（所有敌人，预判位置）
            for e in enemies:
                ex = e["x"] + e["vx"] * t
                ey = e["y"] + e["vy"] * t
                dd = math.hypot(ex - sx, ey - sy)
                danger = p_size + e.get("size", DEFAULT_SIZE) + 5.0
                if dd < danger:
                    score -= (danger - dd + 1.0) * 15.0 * weight
        # 前进：目标在理想距离外时，奖励"靠近目标"的方向
        if target is not None and dist_now > IDEAL_DIST:
            ex_x = p["x"] + d[0] * move_speed * K
            ex_y = p["y"] + d[1] * move_speed * K
            dt = math.hypot(target["x"] - ex_x, target["y"] - ex_y)
            score += 2.0 * max(0.0, 1.0 - dt / max(dist_now, 1.0))
        # 永不停在原地：原来这里给 (0,0) 加 0.05 分，导致没威胁时它站着不动。
        # 改成轻微惩罚，让任何安全方向都优先于静止。
        if d == (0, 0):
            score -= 0.30
        results.append((d, score))
        if score > best_score:
            best_dir, best_score = d, score
    if debug is not None:
        debug["candidates"] = results
    return best_dir


def _fallback_move(state, grid, p, enemies=None, projectiles=None):
    """兜底方向：绝不返回 (0,0)。

    原来 _go_to_door 在"房间没清空 / 门没开 / 已站在门上"时会返回 (0,0)，
    表现就是站在门边发呆。这里改成朝房间中心走（中心通常最开阔），
    走不了就挑任意可走方向。
    """
    if grid is None or p.get("grid_idx") is None:
        return (0, 0)
    w, h = grid["w"], grid["h"]
    gx, gy = p["grid_idx"] % w, p["grid_idx"] // w
    threats = list(enemies or []) + list(projectiles or [])
    best, best_key = None, None
    for d in DIRS8:
        if not cell_walkable(gx + d[0], gy + d[1], grid):
            continue
        # 这一步走完之后的预想世界坐标（一格 40 单位）
        nx = p["x"] + d[0] * 40.0
        ny = p["y"] + d[1] * 40.0
        to_center = abs(gx + d[0] - w // 2) + abs(gy + d[1] - h // 2)
        if threats:
            dmin = min(math.hypot(t["x"] - nx, t["y"] - ny) for t in threats)
            key = (-dmin, to_center)      # 离威胁越远越好，其次靠中心
        else:
            key = (0.0, to_center)
        if best_key is None or key < best_key:
            best, best_key = d, key
    return best if best is not None else (0, 0)


def decide(state, grid, last_door=None, debug=None, recent=None):
    """返回 (移动x, 移动y, 射击x, 射击y)，均为 8 方向 {-1,0,1}。"""
    p = get_player(state)
    target, enemies = choose_target(state, p)
    room_clear = state["room"]["clear"]

    # --- 房间清空 / 无敌人：去最近的开门 ---
    if room_clear or target is None:
        # 兜底走路时也要能躲子弹，所以这里先取一次敌方子弹
        projectiles = get_enemy_projectiles(state)
        move = _go_to_door(state, grid, last_door, recent)
        if move == (0, 0):
            # 门还没开 / 已经站在门上 —— 不允许原地发呆，朝房间中心走
            move = _fallback_move(state, grid, p, enemies, projectiles)
        if debug is not None:
            debug["target"] = None
            debug["candidates"] = []
            debug["move_dir"] = move
            debug["aim_dir"] = (0, 0)
        return move[0], move[1], 0, 0

    # --- 战斗 ---
    # 1) 移动：8 方向候选采样 + 前瞻模拟（综合所有子弹/敌人/墙，避免局部最优）
    projectiles = get_enemy_projectiles(state)
    move = plan_move(state, grid, p, target, enemies, projectiles, debug=debug)
    if move == (0, 0):
        move = _fallback_move(state, grid, p, enemies, projectiles)
    if grid is not None and p.get("grid_idx") is not None and move != (0, 0):
        move = wall_steer(move[0], move[1], p["grid_idx"], grid)

    # 2) 期望的泪弹方向：朝目标预判点（带过冲保护）
    dx, dy = target["x"] - p["x"], target["y"] - p["y"]
    dist = math.hypot(dx, dy)
    dir_x, dir_y = norm(dx, dy)              # 直接朝敌人的方向
    shot_speed = max(state["player"].get("shot_speed", 10.0), 1e-3)
    lead_frames = min(dist / shot_speed, LEAD_MAX_FRAMES)
    desired_x = dx + target["vx"] * lead_frames
    desired_y = dy + target["vy"] * lead_frames
    desired = norm(desired_x, desired_y)
    if desired[0] * dir_x + desired[1] * dir_y <= 0:
        desired = (dir_x, dir_y)             # 过冲指反时退回直接瞄准

    # 3) 4 方向射击 + 甩泪：泪弹会继承人物移动速度，
    #    在 4 个射击方向里挑一个，使 (射击方向×泪速 + 人物速度) 最接近期望方向
    move_speed = max(p.get("move_speed", 2.5), 1e-3)
    player_vel = (move[0] * move_speed, move[1] * move_speed)
    aim4 = choose_aim4(desired, player_vel, shot_speed)

    if debug is not None:
        debug["target"] = target
        debug["move_dir"] = move
        debug["aim_dir"] = aim4
        debug["aim_point"] = (p["x"] + desired_x, p["y"] + desired_y)
        debug["desired"] = desired

    return move[0], move[1], aim4[0], aim4[1]


def dirs_to_action(mx, my):
    """8 方向 -> RL 动作索引（0-7 方向，8 = 不动）。"""
    if mx == 0 and my == 0:
        return 8
    return DIRS8.index((mx, my))


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


CELL_PX = 26  # 调试窗口里每格像素


def render_debug(state, grid, dbg):
    """把 AI 的世界模型 + 决策画成一张图（用于调试观察）。"""
    if grid is None or "ox" not in grid:
        canvas = np.zeros((220, 460, 3), np.uint8)
        cv2.putText(canvas, "no grid data (restart game?)", (10, 110),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
        return canvas

    w, h = grid["w"], grid["h"]
    cells = grid["cells"]
    W, H = w * CELL_PX, h * CELL_PX
    canvas = np.full((H, W, 3), 25, np.uint8)

    # 地形：可走=深灰，不可走=亮灰
    for gy in range(h):
        for gx in range(w):
            color = (65, 65, 65) if cells[gy * w + gx] == 1 else (150, 150, 150)
            cv2.rectangle(canvas, (gx * CELL_PX, gy * CELL_PX),
                          ((gx + 1) * CELL_PX - 1, (gy + 1) * CELL_PX - 1), color, -1)

    ox, oy, tw, th = grid["ox"], grid["oy"], grid["tw"], grid["th"]

    def to_px(x, y):
        return int((x - ox) / tw * CELL_PX), int((y - oy) / th * CELL_PX)

    p = get_player(state)
    px, py = to_px(p["x"], p["y"])

    # 候选方向打分热力：绿=好（安全/该走），红=差
    cands = dbg.get("candidates") or []
    if cands:
        scores = [sc for _, sc in cands]
        lo, hi = min(scores), max(scores)
        rng = hi - lo if hi > lo else 1.0
        for d, sc in cands:
            t = (sc - lo) / rng
            col = (0, int(255 * t), int(255 * (1 - t)))  # BGR：t=1→绿，t=0→红
            ex = px + int(d[0] * CELL_PX * 1.6)
            ey = py + int(d[1] * CELL_PX * 1.6)
            cv2.line(canvas, (px, py), (ex, ey), col, 2)

    # 敌人（红点）
    for e in get_enemies(state):
        ex, ey = to_px(e["x"], e["y"])
        cv2.circle(canvas, (ex, ey), max(4, int(e.get("size", 10) / 40 * CELL_PX)), (0, 0, 255), -1)
    # 敌方子弹（黄点）
    for t in get_enemy_projectiles(state):
        bx, by = to_px(t["x"], t["y"])
        cv2.circle(canvas, (bx, by), 3, (0, 200, 255), -1)

    # 选中的目标（红圈）
    tgt = dbg.get("target")
    if tgt:
        tx, ty = to_px(tgt["x"], tgt["y"])
        cv2.circle(canvas, (tx, ty), 12, (0, 0, 255), 2)

    # 玩家（蓝点）
    cv2.circle(canvas, (px, py), max(5, int(p.get("size", 10) / 40 * CELL_PX)), (255, 150, 0), -1)

    # 移动方向箭头（绿）
    mv = dbg.get("move_dir", (0, 0))
    if mv != (0, 0):
        cv2.arrowedLine(canvas, (px, py),
                        (int(px + mv[0] * CELL_PX * 2), int(py + mv[1] * CELL_PX * 2)),
                        (0, 255, 0), 3, tipLength=0.3)
    # 瞄准方向箭头（黄）
    ad = dbg.get("aim_dir", (0, 0))
    if ad != (0, 0):
        cv2.arrowedLine(canvas, (px, py),
                        (int(px + ad[0] * CELL_PX * 2), int(py + ad[1] * CELL_PX * 2)),
                        (0, 255, 255), 2, tipLength=0.3)
    # 预判瞄准点（黄圈）
    ap = dbg.get("aim_point")
    if ap:
        axp, ayp = to_px(ap[0], ap[1])
        cv2.circle(canvas, (axp, ayp), 5, (0, 255, 255), 1)

    # 期望泪弹方向（青线）：对比 4 向瞄准，能看出甩泪补偿
    ds = dbg.get("desired")
    if ds:
        cv2.line(canvas, (px, py),
                 (int(px + ds[0] * CELL_PX * 2.5), int(py + ds[1] * CELL_PX * 2.5)),
                 (255, 255, 0), 1)

    # 图例
    cv2.putText(canvas, "green=move yellow=aim cyan=desired", (5, H - 6),
                cv2.FONT_HERSHEY_SIMPLEX, 0.4, (210, 210, 210), 1)
    return canvas


def main():
    ap = argparse.ArgumentParser(description="规则版以撒 AI")
    ap.add_argument("--debug", action="store_true", help="打开可视化调试窗口（看 AI 为什么这么决策）")
    args = ap.parse_args()

    bridge = IsaacBridge()
    bridge.start()
    kc = KeyController()
    grid = None
    last_door = None

    print("等待游戏数据...（先让游戏进一局，保持游戏窗口在前台）")
    while bridge.get_state() is None:
        time.sleep(0.1)
    print("拿到数据！5 秒倒计时，快把焦点切回游戏窗口...")
    for i in range(5, 0, -1):
        print(f"  {i}...")
        time.sleep(1)
    if args.debug:
        print("调试窗口已打开（绿=移动 黄=瞄准 红=目标；候选方向绿好红差）")

    paused = False
    prev_room = None
    prev_door = None
    try:
        while True:
            if HAS_KEYBOARD and keyboard.is_pressed("p"):
                paused = not paused
                kc.release_all()
                print("已暂停" if paused else "继续")
                time.sleep(0.3)

            s = bridge.get_state()
            if s is None or "player" not in s or paused:
                time.sleep(0.01)
                continue

            if "grid" in s:
                grid = s["grid"]

            # 检测换房间：刚从 prev_door 那扇门过来
            room_idx = s["room"]["index"]
            if prev_room is not None and room_idx != prev_room:
                last_door = prev_door
            prev_room = room_idx
            prev_door = nearest_open_door_grid(s, grid)

            dbg = {} if args.debug else None
            mx, my, ax, ay = decide(s, grid, last_door, debug=dbg)
            kc.set_move(mx, my)
            kc.set_shoot(ax, ay)

            if args.debug:
                canvas = render_debug(s, grid, dbg)
                cv2.imshow("AI Debug (q 退出)", canvas)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break

            time.sleep(0.016)  # ~60Hz
    except KeyboardInterrupt:
        pass
    finally:
        kc.release_all()
        bridge.stop()
        if args.debug:
            cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
