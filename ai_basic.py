# -*- coding: utf-8 -*-
"""
ai_basic.py —— 规则版 AI v3：清房进下一间 + A* 追击 + 墙感知躲弹幕 + 8 方向射击

行为：
  * 房间有敌人：追最近的敌人（A* 绕障碍）+ 打 + 躲敌方泪弹
  * 房间清空：A* 走到最近的开门（自动进入下一间）
  * 记住"上一扇走过的门"，优先走别的门，减少来回横跳

用法：游戏进对局保持前台 -> python ai_basic.py -> 5 秒内切回游戏。
按 'p' 暂停/继续（需 pip install keyboard），Ctrl+C 退出。
"""
import math
import time

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

# 8 方向：0=右 1=右下 2=下 3=左下 4=左 5=左上 6=上 7=右上
DIRS8 = [(1, 0), (1, 1), (0, 1), (-1, 1), (-1, 0), (-1, -1), (0, -1), (1, -1)]

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


def _go_to_door(state, grid, last_door=None):
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

    # 优先不走上一次走过的门（减少来回横跳）
    candidates = [d for d in open_doors if d["grid"] != last_door] or open_doors
    best = min(candidates, key=gdist)
    path = astar(grid, p["grid_idx"], best["grid"])
    if path and len(path) >= 2:
        cur, nxt = path[0], path[1]
        return dir8((nxt % w) - (cur % w), (nxt // w) - (cur // w))
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


def plan_move(state, grid, p, target, enemies, projectiles):
    """候选方向采样 + 前瞻模拟：选未来最安全的方向（替代力场求和，避免局部最优）。"""
    p_size = p.get("size", DEFAULT_SIZE)
    move_speed = max(p.get("move_speed", 2.5), 1e-3)  # 单位/帧
    K = LOOKAHEAD_FRAMES
    candidates = list(DIRS8) + [(0, 0)]
    best_dir, best_score = (0, 0), -1e18
    dist_now = math.hypot(target["x"] - p["x"], target["y"] - p["y"]) if target is not None else 0.0

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
        # 无威胁时轻微偏好原地，避免无意义漂移
        if d == (0, 0):
            score += 0.05
        if score > best_score:
            best_dir, best_score = d, score
    return best_dir


def decide(state, grid, last_door=None):
    """返回 (移动x, 移动y, 射击x, 射击y)，均为 8 方向 {-1,0,1}。"""
    p = get_player(state)
    target, enemies = choose_target(state, p)
    room_clear = state["room"]["clear"]

    # --- 房间清空 / 无敌人：去最近的开门 ---
    if room_clear or target is None:
        move = _go_to_door(state, grid, last_door)
        return move[0], move[1], 0, 0

    # --- 战斗 ---
    # 瞄准：预判敌人未来位置，但防止"过冲指反"
    dx, dy = target["x"] - p["x"], target["y"] - p["y"]
    dist = math.hypot(dx, dy)
    dir_x, dir_y = norm(dx, dy)              # 直接朝敌人的方向
    shot_speed = max(state["player"].get("shot_speed", 10.0), 1e-3)
    lead_frames = min(dist / shot_speed, LEAD_MAX_FRAMES)
    aim_x = dx + target["vx"] * lead_frames
    aim_y = dy + target["vy"] * lead_frames
    aim = norm(aim_x, aim_y)
    # 预判方向若偏离"直接朝敌人"超过 90°（过冲会导致指反），退回直接瞄准
    if aim[0] * dir_x + aim[1] * dir_y <= 0:
        aim = (dir_x, dir_y)

    # 移动：候选方向采样 + 前瞻模拟（综合所有子弹/敌人/墙，避免局部最优）
    projectiles = get_enemy_projectiles(state)
    move = plan_move(state, grid, p, target, enemies, projectiles)
    if grid is not None and p.get("grid_idx") is not None and move != (0, 0):
        move = wall_steer(move[0], move[1], p["grid_idx"], grid)

    aim8 = dir8(aim[0], aim[1])
    return move[0], move[1], aim8[0], aim8[1]


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


def main():
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

            mx, my, ax, ay = decide(s, grid, last_door)
            kc.set_move(mx, my)
            kc.set_shoot(ax, ay)
            time.sleep(0.016)  # ~60Hz
    except KeyboardInterrupt:
        pass
    finally:
        kc.release_all()
        bridge.stop()
        print("\n已退出，按键已全部松开")


if __name__ == "__main__":
    main()
