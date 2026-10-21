# -*- coding: utf-8 -*-
"""tactical_agent.py —— 可实机运行的战术 AI（不依赖 PPO）

为什么要这个文件
----------------
项目原来的 PPO 模型在真实对局里命中率 ~20%、清房率 ~65%，而且动作会在
"敢开枪 <> 不敢开枪"两个模式之间漂移。继续调奖励权重的收益已经很低。
这里换一条路：直接把"瞄得准、会躲、会走位"写成可解释的搜索策略。

它和 RL 模型使用同一份内存状态（模组 uplink），但决策由三块组成：

1. **弹道求解**（aim）：对 4 个射击方向，解析计算"泪弹飞出去后与敌人的
   最近距离"，把玩家当前速度对泪弹的继承也算进去；只有预测能命中、且
   中间没有石头/墙时才开火。
2. **威胁搜索**（move）：把 8 个移动方向 + 停止逐一向前模拟 20 帧，
   同时躲避敌方弹幕、敌人身体、火堆/尖刺和墙角；安全优先，其次才是
   保持交战距离、对齐射击轴、取得视线。
3. **导航**（navigate）：清房后用 A* 走到掉落物/门；用楼层地图 BFS 找
   最近的未探索房间，避免在两个清空的房间之间来回弹。

这个模块只负责"给定一帧状态，返回按键"，不碰 socket / 窗口。这样可以在
离线测试里构造假局面验证瞄准和躲弹逻辑。
"""
from __future__ import annotations

import heapq
import math
from collections import deque
from dataclasses import dataclass, field
from typing import Iterable, Optional

# 和 combat_env / isaac_env 保持一致
DIRS8 = [(1, 0), (1, 1), (0, 1), (-1, 1), (-1, 0), (-1, -1), (0, -1), (1, -1)]
# 射击动作：0=不射，1上 2右 3下 4左
SHOOT_DIRS = {1: (0, -1), 2: (1, 0), 3: (0, 1), 4: (-1, 0)}
PROJECTILE_TYPES = (2, 7, 8, 9)
WALKABLE_COLL = (0, 5)          # 空地和 WALL_EXCEPT_PLAYER（门）
BLOCKING_COLL = (2, 3, 4)       # 石头/实心/墙：挡泪弹
HARMFUL_GRID_ENT = (4, 8)       # 火堆 / 尖刺（GridEntityType）
TRAPDOOR_GRID_ENT = (10, 11)    # 陷阱门 / 楼梯

# Isaac 的 DoorSlot 顺序是 左0 / 上1 / 右2 / 下3（不是上右下左！）
# 大房间同一侧的第二扇门是 左4 / 上5 / 右6 / 下7。
# 楼层房间网格宽 13：左 -1，右 +1，上 -13，下 +13。
ROOM_DELTA = {0: -1, 1: -13, 2: 1, 3: 13,
              4: -1, 5: -13, 6: 1, 7: 13}


def _norm(x: float, y: float) -> tuple[float, float]:
    m = math.hypot(x, y)
    if m < 1e-9:
        return 0.0, 0.0
    return x / m, y / m


def _dir8_to_vec(mx: int, my: int) -> tuple[float, float]:
    """把 8 方向按键向量归一化（斜向的长度也是 1）。"""
    return _norm(float(mx), float(my))


def _vec_to_dir8(x: float, y: float) -> tuple[int, int]:
    """连续方向 -> 8 方向按键。"""
    if abs(x) < 1e-9 and abs(y) < 1e-9:
        return 0, 0
    ang = math.atan2(y, x)
    idx = int(round(ang / (math.pi / 4.0))) % 8
    return DIRS8[idx]


def _dir8_index(mx: int, my: int) -> int:
    if mx == 0 and my == 0:
        return 8
    return DIRS8.index((mx, my))


# ---------------------------------------------------------------------------
# 房间地形
# ---------------------------------------------------------------------------
class RoomGrid:
    """对模组发来的网格做一层包装，提供世界坐标/格子坐标的常用查询。"""

    def __init__(self, raw: dict):
        self.raw = dict(raw)
        self.w = int(raw["w"])
        self.h = int(raw["h"])
        self.tw = float(raw.get("tw", 40.0) or 40.0)
        self.th = float(raw.get("th", 40.0) or 40.0)
        self.ox = float(raw.get("ox", 0.0) or 0.0)
        self.oy = float(raw.get("oy", 0.0) or 0.0)
        coll = raw.get("coll") or []
        if len(coll) < self.w * self.h:
            coll = list(coll) + [0] * (self.w * self.h - len(coll))
        self.coll = [int(c or 0) for c in coll[: self.w * self.h]]
        # 旧版模组的 bug：把“可走区域”的宽高除以了含墙的 w/h，原点也少外推；
        # 窄房里可走区域还不从第 1 行/列开始。用碰撞格包围盒就地还原。
        ver = int(raw.get("gv", 1) or 1)
        if ver < 3 and self.w > 2 and self.h > 2:
            xs = [i % self.w for i, c in enumerate(self.coll) if c]
            ys = [i // self.w for i, c in enumerate(self.coll) if c]
            if xs and ys:
                minx, maxx = min(xs), max(xs)
                miny, maxy = min(ys), max(ys)
                cols, rows = maxx - minx - 1, maxy - miny - 1
                if cols > 0 and rows > 0:
                    if ver <= 1:
                        # 旧版 tw/th 是“可走跨度 / 含墙格数”，先还原跨度再求格宽
                        self.tw *= self.w / cols
                        self.th *= self.h / rows
                        self.ox -= (minx + 1) * self.tw
                        self.oy -= (miny + 1) * self.th
                    else:
                        # gv=2 格宽已正确，只是原点按正常房型固定外推了一格
                        self.ox -= minx * self.tw
                        self.oy -= miny * self.th
            elif ver <= 1:
                self.tw *= self.w / (self.w - 2)
                self.th *= self.h / (self.h - 2)
                self.ox -= self.tw
                self.oy -= self.th
        # cells 以 coll 重算（门是 5，玩家能穿过）
        self.cells = [1 if c in WALKABLE_COLL else 0 for c in self.coll]
        ent = raw.get("ent") or []
        if len(ent) < self.w * self.h:
            ent = list(ent) + [0] * (self.w * self.h - len(ent))
        self.ent = [int(e or 0) for e in ent[: self.w * self.h]]
        self._clearance = None

    # ---- 坐标 ----
    def cell_of(self, x: float, y: float) -> tuple[int, int]:
        return int((x - self.ox) // self.tw), int((y - self.oy) // self.th)

    def idx_of(self, x: float, y: float) -> int:
        gx, gy = self.cell_of(x, y)
        if not (0 <= gx < self.w and 0 <= gy < self.h):
            return -1
        return gy * self.w + gx

    def center(self, idx: int) -> tuple[float, float]:
        gx, gy = idx % self.w, idx // self.w
        return self.ox + (gx + 0.5) * self.tw, self.oy + (gy + 0.5) * self.th

    def idx_center(self, gx: int, gy: int) -> tuple[float, float]:
        return self.ox + (gx + 0.5) * self.tw, self.oy + (gy + 0.5) * self.th

    def walkable_idx(self, idx: int) -> bool:
        return 0 <= idx < len(self.cells) and self.cells[idx] == 1

    def walkable_cell(self, gx: int, gy: int) -> bool:
        if not (0 <= gx < self.w and 0 <= gy < self.h):
            return False
        return self.cells[gy * self.w + gx] == 1

    def blocked_cell(self, gx: int, gy: int) -> bool:
        """挡泪弹的格子：石头/实心/墙；出界也算。"""
        if not (0 <= gx < self.w and 0 <= gy < self.h):
            return True
        return self.coll[gy * self.w + gx] in BLOCKING_COLL

    def hazardous_idx(self, idx: int) -> bool:
        if not (0 <= idx < len(self.ent)):
            return False
        return self.ent[idx] in HARMFUL_GRID_ENT

    def trapdoor_cells(self) -> list[int]:
        return [i for i, e in enumerate(self.ent) if e in TRAPDOOR_GRID_ENT]

    def point_clear(self, x: float, y: float, radius: float) -> bool:
        """半径 radius 的圆是否完全落在可走区域内。

        用 9 个点采样（中心 + 8 个方向）近似；比逐像素碰撞便宜得多。
        """
        if not self.walkable_cell(*self.cell_of(x, y)):
            return False
        for dx, dy in ((1, 0), (-1, 0), (0, 1), (0, -1),
                       (0.707, 0.707), (-0.707, 0.707), (0.707, -0.707), (-0.707, -0.707)):
            if not self.walkable_cell(*self.cell_of(x + dx * radius, y + dy * radius)):
                return False
        return True

    def point_hazard(self, x: float, y: float) -> bool:
        idx = self.idx_of(x, y)
        return self.hazardous_idx(idx) if idx >= 0 else True

    def clearance(self) -> list[float]:
        """每个格子到最近不可走格子的近似距离（世界单位），用于避免贴墙。"""
        if self._clearance is not None:
            return self._clearance
        w, h = self.w, self.h
        INF = 10 ** 6
        dist = [0.0 if not self.cells[i] else float(INF) for i in range(w * h)]
        dq = deque(i for i in range(w * h) if not self.cells[i])
        # 8 邻域 BFS（斜向按 1.414 计）
        while dq:
            cur = dq.popleft()
            cx, cy = cur % w, cur // w
            dc = dist[cur]
            for dx, dy, cost in ((1, 0, 1), (-1, 0, 1), (0, 1, 1), (0, -1, 1),
                                 (1, 1, 1.414), (-1, 1, 1.414),
                                 (1, -1, 1.414), (-1, -1, 1.414)):
                nx, ny = cx + dx, cy + dy
                if not (0 <= nx < w and 0 <= ny < h):
                    continue
                ni = ny * w + nx
                nd = dc + cost * min(self.tw, self.th)
                if nd + 1e-9 < dist[ni]:
                    dist[ni] = nd
                    dq.append(ni)
        # 格子中心到不可走格中心的距离；再减去半格当作"到边界"的近似
        half = 0.5 * min(self.tw, self.th)
        self._clearance = [max(0.0, d - half) if self.cells[i] else 0.0
                           for i, d in enumerate(dist)]
        return self._clearance

    def clearance_at(self, x: float, y: float) -> float:
        idx = self.idx_of(x, y)
        if idx < 0:
            return 0.0
        return self.clearance()[idx]

    # ---- 视线 ----
    def segment_clear(self, x0: float, y0: float, x1: float, y1: float,
                      radius: float = 0.0) -> bool:
        """从 (x0,y0) 到 (x1,y1) 是否不被石头/墙挡住。

        radius > 0 时同时检查射线两侧偏移，避免"擦着石头角飞过去"的假阳性。
        采样步长取半格，既不会漏掉单格障碍，也不至于太慢。
        """
        dx, dy = x1 - x0, y1 - y0
        dist = math.hypot(dx, dy)
        if dist < 1e-6:
            return not self.blocked_cell(*self.cell_of(x0, y0))
        steps = max(2, int(dist / (min(self.tw, self.th) * 0.45)) + 1)
        px, py = -dy / dist * radius, dx / dist * radius
        offsets = [(0.0, 0.0)]
        if radius > 0:
            offsets += [(px, py), (-px, -py)]
        for k in range(1, steps + 1):
            t = k / steps
            for ox, oy in offsets:
                sx = x0 + dx * t + ox
                sy = y0 + dy * t + oy
                if self.blocked_cell(*self.cell_of(sx, sy)):
                    return False
        return True

    def grid_segment_clear(self, x0: float, y0: float, x1: float, y1: float,
                           radius: float = 7.0) -> bool:
        """走路用的粗线段检查：既要无遮挡，也要离墙留出半径。"""
        dx, dy = x1 - x0, y1 - y0
        dist = math.hypot(dx, dy)
        if dist < 1e-6:
            return self.point_clear(x0, y0, radius)
        steps = max(2, int(dist / (min(self.tw, self.th) * 0.5)) + 1)
        for k in range(1, steps + 1):
            t = k / steps
            if not self.point_clear(x0 + dx * t, y0 + dy * t, radius):
                return False
        return True


# ---------------------------------------------------------------------------
# 状态整理
# ---------------------------------------------------------------------------
@dataclass
class Entity:
    id: int
    type: int
    x: float
    y: float
    vx: float
    vy: float
    hp: float = 0.0
    size: float = 10.0
    boss: bool = False
    enemy: bool = False
    spawner_type: int = 0
    cdmg: float = 0.0
    raw: dict = field(default_factory=dict, repr=False)


def _entity(e: dict) -> Entity:
    return Entity(
        id=int(e.get("id", 0)),
        type=int(e.get("type", 0) or 0),
        x=float(e.get("x", 0.0) or 0.0),
        y=float(e.get("y", 0.0) or 0.0),
        vx=float(e.get("vx", 0.0) or 0.0),
        vy=float(e.get("vy", 0.0) or 0.0),
        hp=float(e.get("hp", 0.0) or 0.0),
        size=float(e.get("size", 10.0) or 10.0),
        boss=bool(e.get("boss")),
        enemy=bool(e.get("enemy")),
        spawner_type=int(e.get("spawner_type", 0) or 0),
        cdmg=float(e.get("cdmg", 0.0) or 0.0),
        raw=e,
    )


def split_entities(state: dict) -> tuple[list[Entity], list[Entity], list[Entity], list[Entity]]:
    """返回 (敌人, 敌方弹幕, 掉落物, 其他带伤害实体)。"""
    enemies, projectiles, pickups, hazards = [], [], [], []
    for e in state.get("entities", []):
        ent = _entity(e)
        if ent.enemy or ent.boss:
            if ent.hp > 0:
                enemies.append(ent)
            continue
        if ent.type in PROJECTILE_TYPES and ent.spawner_type != 1:
            projectiles.append(ent)
            continue
        if ent.type == 5:                  # Pickup
            pickups.append(ent)
            continue
        if ent.cdmg > 0:
            hazards.append(ent)
    return enemies, projectiles, pickups, hazards


@dataclass
class Action:
    move: tuple[int, int] = (0, 0)
    shoot: tuple[int, int] = (0, 0)
    mode: str = "idle"
    target_id: Optional[int] = None
    aim_dir: int = 0
    debug: dict = field(default_factory=dict)


@dataclass
class Config:
    d_opt: float = 108.0
    d_min: float = 58.0
    d_far: float = 200.0
    horizon: int = 20
    speed_default: float = 3.6
    player_radius: float = 10.0
    tear_radius: float = 5.0
    tear_speed_default: float = 10.0
    aim_slack: float = 0.95
    dodge_margin: float = 7.0
    body_margin: float = 5.0
    wall_radius: float = 8.0
    # 评分权重
    w_danger: float = 3.0
    w_body: float = 2.0
    w_distance: float = 1.0
    w_align: float = 1.25
    w_los: float = 0.85
    w_open: float = 0.28
    stop_bonus: float = 0.32
    w_aim: float = 1.15          # 移动方向带来的射击机会（含速度补偿）
    # 切换目标的滞回
    target_hysteresis: float = 0.35


class TacticalAgent:
    """战术决策器。每个游戏实例一个。"""

    def __init__(self, cfg: Optional[Config] = None, log=None):
        self.cfg = cfg or Config()
        self.log = log or (lambda *a, **k: None)
        self.grid: Optional[RoomGrid] = None
        self.room_index: Optional[int] = None
        self.map_rooms: list[dict] = []
        self.doors: list[dict] = []
        self.last_target_id: Optional[int] = None
        self.last_move = (0, 0)
        self.last_shoot = (0, 0)
        self.prev_pos: Optional[tuple[float, float]] = None
        self.stuck_frames = 0
        self.recover_frames = 0
        self.recover_dir = (0, 0)
        self.no_los_frames = 0
        self.no_aim_frames = 0
        self.path: list[int] = []
        self.path_goal: Optional[int] = None
        self.path_kind: str = ""
        self.pickup_blacklist: set[int] = set()
        self.pickup_wait = 0
        self.door_push = 0
        self.door_pushing: Optional[int] = None
        # 楼层连通图：room -> {door_slot: neighbor}。只记录真正走过的门，
        # 不能靠 13x13 相邻推断 —— 格子相邻不代表两间房之间有门。
        self.floor_edges: dict[int, dict[int, int]] = {}
        self.pushed_slot: Optional[int] = None
        self.pushed_from: Optional[int] = None
        self._map_sig: Optional[frozenset[int]] = None
        self.visited_rooms: set[int] = set()
        self.seen_door_slots: dict[int, set[int]] = {}
        self.last_room_clear = False
        self.speed_est = self.cfg.speed_default
        self._vel_samples: deque[float] = deque(maxlen=30)
        self.path_age = 0

    # ------------------------------------------------------------------
    # 状态更新
    # ------------------------------------------------------------------
    def update_state(self, state: dict):
        p = state.get("player") or {}
        room = state.get("room") or {}
        idx = room.get("index")
        if idx is not None and self.room_index is not None and idx != self.room_index:
            old = self.room_index
            # 房号 -3 是每局开局房间；从这里进来说明重开了，楼层记忆作废
            if idx == -3 and old != -3:
                self.reset_floor()
            # 刚推过的门真的把我们送到了邻房：记下这条连通边
            if self.pushed_from == old and self.pushed_slot is not None:
                slot = self.pushed_slot
                if old + ROOM_DELTA.get(slot, 0) == idx:
                    self.floor_edges.setdefault(old, {})[slot] = idx
                    # 反向门：左0↔右2、上1↔下3；第二扇门 4↔6、5↔7
                    self.floor_edges.setdefault(idx, {})[slot ^ 2] = old
            self.pushed_slot = None
            self.pushed_from = None
            # 换房间：所有本房间的路径/记忆都失效
            self.grid = None
            self.path = []
            self.path_goal = None
            self.pickup_blacklist.clear()
            self.pickup_wait = 0
            self.door_push = 0
            self.door_pushing = None
            self.stuck_frames = 0
            self.recover_frames = 0
            self.no_los_frames = 0
            self.no_aim_frames = 0
        self.room_index = idx
        if idx is not None:
            self.visited_rooms.add(idx)
        if "grid" in state and state["grid"]:
            self.grid = RoomGrid(state["grid"])
        if "doors" in state:
            self.doors = list(state.get("doors") or [])
            if idx is not None:
                # 记录“当前仍开着的门”；门锁上后不能还当它是可探索出口
                self.seen_door_slots[idx] = {
                    int(d.get("slot", -1)) for d in self.doors if d.get("open")}
        if "map" in state and state["map"]:
            new_map = list(state.get("map") or [])
            sig = frozenset(int(r.get("grid", 0)) for r in new_map)
            if self._map_sig is not None and sig != self._map_sig:
                # 换层（或秘密房新出现）：旧连通图不再可信
                self.floor_edges.clear()
                self.visited_rooms = {idx} if idx is not None else set()
                self.seen_door_slots.clear()
            self._map_sig = sig
            self.map_rooms = new_map
        # 估计实际移动速度（游戏里 Speed 属性与 velocity 的关系不固定，直接实测）
        sp = math.hypot(float(p.get("vx", 0) or 0), float(p.get("vy", 0) or 0))
        if sp > 0.8:
            self._vel_samples.append(sp)
            vals = sorted(self._vel_samples)
            self.speed_est = vals[int(len(vals) * 0.9)] if vals else self.cfg.speed_default
            self.speed_est = max(2.4, min(5.2, self.speed_est))

    def reset_floor(self):
        """换层/重开时清空楼层记忆（连通图不能跨层复用）。"""
        self.floor_edges.clear()
        self.map_rooms = []
        self._map_sig = None
        self.visited_rooms.clear()
        self.seen_door_slots.clear()
        self.pushed_slot = None
        self.pushed_from = None
        self.door_push = 0
        self.door_pushing = None

    def reset_tracking(self):
        self.last_target_id = None
        self.last_move = (0, 0)
        self.last_shoot = (0, 0)
        self.prev_pos = None
        self.stuck_frames = 0
        self.recover_frames = 0
        self.no_los_frames = 0
        self.no_aim_frames = 0
        self.path = []
        self.path_goal = None
        self.path_age = 0
        self.pickup_blacklist.clear()
        self.pickup_wait = 0
        self.door_push = 0
        self.door_pushing = None

    # ------------------------------------------------------------------
    # 弹道
    # ------------------------------------------------------------------
    def _tear_speed(self, p: dict) -> float:
        try:
            ss = float(p.get("shot_speed", 1.0) or 1.0)
        except (TypeError, ValueError):
            ss = 1.0
        # 模组里的 shot_speed 是属性倍率，默认 1.0 对应约 10 单位/帧。
        return max(4.0, 10.0 * ss)

    def aim_solution(self, p: dict, e: Entity, vel: tuple[float, float],
                     grid: Optional[RoomGrid] = None) -> Optional[dict]:
        """4 个射击方向里，找一个预测能命中的解。

        返回 {dir, miss, t, point}；没有可行解返回 None。
        物理：泪弹速度 = 射击方向单位向量 × 泪速 + 玩家当前速度。
        """
        grid = grid or self.grid
        px, py = float(p["x"]), float(p["y"])
        S = self._tear_speed(p)
        life = max(8.0, float(p.get("tear_range", 260.0) or 260.0) / S)
        vpx, vpy = vel
        best = None
        ex0, ey0 = e.x - px, e.y - py
        for sd, (sx, sy) in SHOOT_DIRS.items():
            vx = sx * S + vpx
            vy = sy * S + vpy
            if math.hypot(vx, vy) < 1e-6:
                continue
            # 玩家实际弹道方向必须朝敌人（不然就是背身开枪）
            if ex0 * vx + ey0 * vy <= 0:
                continue
            rvx = e.vx - vx
            rvy = e.vy - vy
            vv = rvx * rvx + rvy * rvy
            if vv < 1e-9:
                t = 0.0
            else:
                t = -(ex0 * rvx + ey0 * rvy) / vv
            t = min(max(t, 0.0), life)
            mx = ex0 + rvx * t
            my = ey0 + rvy * t
            miss = math.hypot(mx, my)
            # 泪弹飞行距离不能超过射程
            travel = math.hypot(vx, vy) * t
            rng = float(p.get("tear_range", 260.0) or 260.0)
            if travel > rng * 1.02:
                continue
            ex_t, ey_t = e.x + e.vx * t, e.y + e.vy * t
            if grid is not None and not grid.segment_clear(px, py, ex_t, ey_t,
                                                           radius=self.cfg.tear_radius * 0.6):
                continue
            hit_radius = max(8.0, min(22.0, e.size * 0.95) + self.cfg.tear_radius)
            cand = {"dir": sd, "miss": miss, "t": t,
                    "point": (ex_t, ey_t), "hit_radius": hit_radius,
                    "vx": vx, "vy": vy}
            if best is None or cand["miss"] < best["miss"]:
                best = cand
        if best is None:
            return None
        if best["miss"] > best["hit_radius"] * self.cfg.aim_slack:
            return None
        return best

    # ------------------------------------------------------------------
    # 目标选择
    # ------------------------------------------------------------------
    def choose_target(self, p: dict, enemies: list[Entity]) -> Optional[Entity]:
        if not enemies:
            return None
        px, py = float(p["x"]), float(p["y"])
        best, best_score = None, -1e18
        for e in enemies:
            dx, dy = e.x - px, e.y - py
            d = math.hypot(dx, dy) or 1.0
            approach = (e.vx * dx + e.vy * dy) / d
            los = (self.grid.segment_clear(px, py, e.x, e.y, radius=2.0)
                   if self.grid else True)
            s = 0.0
            if e.boss:
                s += 4.0
            s += 2.2 / (0.45 + d / 110.0)
            s += 0.75 * max(0.0, approach)
            s += 0.55 * max(0.0, approach) / 1.0
            if los:
                s += 1.1
            else:
                s -= 0.6
            s -= 0.012 * max(0.0, e.hp)
            if d < 48.0:
                s -= 1.4
            if e.id == self.last_target_id:
                s += self.cfg.target_hysteresis
            if s > best_score:
                best, best_score = e, s
        return best

    # ------------------------------------------------------------------
    # 移动评分
    # ------------------------------------------------------------------
    def _body_radius(self, e: Entity) -> float:
        return max(7.0, e.size) + self.cfg.player_radius + self.cfg.body_margin

    def _bullet_radius(self, b: Entity) -> float:
        return max(4.0, b.size) + self.cfg.player_radius + self.cfg.dodge_margin

    def _threat_now(self, p: dict, projectiles: list[Entity], enemies: list[Entity],
                    hazards: list[Entity]) -> float:
        """当前是否已经在威胁里（用于进入紧急躲避模式）。"""
        px, py = float(p["x"]), float(p["y"])
        level = 0.0
        for b in projectiles:
            rr = self._bullet_radius(b)
            dist = math.hypot(b.x - px, b.y - py)
            if dist <= rr:
                level = max(level, 1.0)
                continue
            sp2 = b.vx * b.vx + b.vy * b.vy
            if sp2 < 1e-4:
                continue
            t = ((px - b.x) * b.vx + (py - b.y) * b.vy) / sp2
            if 0 < t <= 16.0:
                cx, cy = b.x + b.vx * t, b.y + b.vy * t
                if math.hypot(cx - px, cy - py) < rr:
                    level = max(level, 1.0 - t / 22.0)
        for e in enemies:
            if e.cdmg <= 0 and not e.boss:
                continue
            rr = self._body_radius(e)
            dist = math.hypot(e.x - px, e.y - py)
            if dist < rr:
                level = max(level, 0.8)
        for hz in hazards:
            if math.hypot(hz.x - px, hz.y - py) < hz.size + self.cfg.player_radius + 6:
                level = max(level, 0.6)
        return level

    def _danger_at(self, x: float, y: float, t: float,
                   projectiles: list[Entity], enemies: list[Entity],
                   hazards: list[Entity], grid: RoomGrid, urgent: float) -> float:
        """位置 (x,y) 在 t 帧后的危险度。数值越大越危险。"""
        danger = 0.0
        urgency = 1.0 + max(0.0, 1.0 - t / max(1.0, self.cfg.horizon)) * 1.5
        for b in projectiles:
            bx, by = b.x + b.vx * t, b.y + b.vy * t
            rr = self._bullet_radius(b)
            d = math.hypot(bx - x, by - y)
            if d < rr:
                danger += (rr - d + 2.0) * 5.0 * urgency
                if d < rr * 0.55:
                    danger += 45.0 * urgency
        for e in enemies:
            if e.cdmg <= 0 and not e.boss:
                continue
            ex, ey = e.x + e.vx * t, e.y + e.vy * t
            rr = self._body_radius(e)
            d = math.hypot(ex - x, ey - y)
            if d < rr:
                danger += (rr - d + 2.0) * 3.5 * urgency
        for hz in hazards:
            rr = hz.size + self.cfg.player_radius + 3
            d = math.hypot(hz.x - x, hz.y - y)
            if d < rr:
                danger += (rr - d + 2.0) * 4.0 * urgency
        # 静态地面危险物（火堆/尖刺）
        if grid.point_hazard(x, y):
            danger += 140.0 * urgency
        return danger * urgent

    def _align_offset(self, x: float, y: float, tx: float, ty: float) -> tuple[float, float]:
        """返回 (射击轴, 对齐误差)。射击轴 = 主轴（delta 大的那个）。"""
        dx, dy = tx - x, ty - y
        if abs(dx) >= abs(dy):
            return 1, abs(dy)
        return 0, abs(dx)

    def plan_move(self, p: dict, target: Optional[Entity], enemies: list[Entity],
                  projectiles: list[Entity], hazards: list[Entity],
                  aim: Optional[dict], clear: bool) -> tuple[tuple[int, int], dict]:
        grid = self.grid
        px, py = float(p["x"]), float(p["y"])
        if grid is None:
            return (0, 0), {"reason": "no-grid"}
        speed = self.speed_est
        H = max(6, self.cfg.horizon)
        rg = self._threat_now(p, projectiles, enemies, hazards)
        urgent = 1.0 + min(2.0, rg) * 1.2
        obs_times = [1, 2, 3, 4, 6, 8, 10, 12, 15, 18, 22, 26]
        obs_times = [t for t in obs_times if t <= H] or [H]
        # 预计算每个时间点的目标位置
        best_dir, best_score, best_detail = (0, 0), -1e18, {}
        clearance = grid.clearance()
        vpx = float(p.get("vx", 0) or 0.0)
        vpy = float(p.get("vy", 0) or 0.0)
        for mx, my in DIRS8 + [(0, 0)]:
            ux, uy = _dir8_to_vec(mx, my)
            score = 0.0
            danger_sum = 0.0
            sampled = 0
            blocked_at = None
            for t in obs_times:
                sx = px + ux * speed * t
                sy = py + uy * speed * t
                if grid.idx_of(sx, sy) < 0 or not grid.point_clear(sx, sy, self.cfg.wall_radius):
                    blocked_at = t
                    break
                danger_sum += self._danger_at(sx, sy, t, projectiles, enemies,
                                              hazards, grid, urgent)
                sampled += 1
            if sampled:
                score -= danger_sum / sampled
            # 撞墙不是伤害：按住这个方向只会在墙前停下。只有"一动就撞"才说明
            # 这个方向没用。旧版对 18 帧内任何时刻撞墙都扣 -900，结果靠墙时
            # 几乎所有方向都被淘汰，AI 只剩站桩可选（实测靠下墙时完全不动）。
            if blocked_at is not None and blocked_at <= 2:
                score -= 1.6
            # 评估点：能走多远就评多远，绝不用墙里的位置去算对齐/开阔度
            mid_t = obs_times[len(obs_times) // 2]
            eval_t = mid_t if blocked_at is None else min(mid_t, max(1, blocked_at - 1))
            mxp, myp = px + ux * speed * eval_t, py + uy * speed * eval_t
            # ---- 射击潜力：这个方向下（泪弹继承移动速度）能不能打中目标 ----
            cand_aim = None
            if target is not None:
                if (mx, my) == (0, 0):
                    cand_aim = aim          # 停止时保持当前速度，act 已算过
                else:
                    cand_aim = self.aim_solution(
                        p, target, (ux * speed, uy * speed), grid)
                if cand_aim is not None:
                    quality = 1.0 - min(
                        1.0, cand_aim["miss"] / max(1e-6, cand_aim["hit_radius"]))
                    # 正在挨打时别为了开枪往危险里走，射击潜力只留一点权重
                    score += self.cfg.w_aim * (0.55 + 0.45 * quality) * (
                        1.0 if rg < 0.4 else 0.25)
                elif rg < 0.4:
                    score -= 0.10
            # ---- 目标导向 ----
            if target is not None:
                tx, ty = target.x + target.vx * eval_t, target.y + target.vy * eval_t
                d = math.hypot(tx - mxp, ty - myp)
                # 距离：靠近 d_opt，太远/太近都罚
                score -= self.cfg.w_distance * abs(d - self.cfg.d_opt) / 55.0
                if d > self.cfg.d_far:
                    score -= 1.4 * (d - self.cfg.d_far) / 50.0
                if d < self.cfg.d_min:
                    score -= 1.8 * (self.cfg.d_min - d) / 30.0
                # 对齐：减少非射击轴偏移
                axis, off = self._align_offset(mxp, myp, tx, ty)
                score -= self.cfg.w_align * off / 42.0
                if off <= 16.0:
                    score += 0.22
                # 视线
                if grid.segment_clear(mxp, myp, tx, ty, radius=2.0):
                    score += self.cfg.w_los
                else:
                    score -= self.cfg.w_los * 0.55
                # 能开枪时不要横向自毁准星；站桩只在有射击解时给奖励
                if cand_aim is not None:
                    ax, ay = SHOOT_DIRS[cand_aim["dir"]]
                    if (mx, my) == (0, 0):
                        score += self.cfg.stop_bonus
                    else:
                        dot = ax * ux + ay * uy
                        if abs(dot) < 0.5:
                            score -= 0.55
                        else:
                            score += 0.08
                elif (mx, my) == (0, 0):
                    score -= 0.28
            # 开阔度：别贴着墙角/石头（用能走到的评估点，不是墙里的点）
            idx = grid.idx_of(mxp, myp)
            if idx >= 0:
                cl = clearance[idx]
                score += self.cfg.w_open * min(cl, 90.0) / 90.0
                if cl < self.cfg.wall_radius + 4:
                    score -= 0.55
            # 连续性（轻微，避免 30Hz 抖动）
            if (mx, my) != (0, 0) and self.last_move == (mx, my):
                score += 0.025
            # 原地不动：远距离目标别站着，评价里已经给过"没射击解"的惩罚
            if (mx, my) == (0, 0):
                if target is not None and math.hypot(target.x - px, target.y - py) > self.cfg.d_far:
                    score -= 0.25
            if score > best_score:
                best_score, best_dir = score, (mx, my)
                best_detail = {"score": score, "danger": danger_sum,
                               "blocked_at": blocked_at}
        # 紧急恢复：如果所有方向都撞墙，停住等下一帧
        return best_dir, best_detail

    # ------------------------------------------------------------------
    # A* 导航
    # ------------------------------------------------------------------
    def astar(self, start: int, goal: int, hazard_penalty: float = 10.0) -> Optional[list[int]]:
        grid = self.grid
        if grid is None:
            return None
        n = grid.w * grid.h
        if not (0 <= start < n) or not (0 <= goal < n):
            return None
        if not grid.walkable_idx(goal):
            # 目标格不可走（比如掉落物在石头上），找一个可走邻居
            gx, gy = goal % grid.w, goal // grid.w
            cand = []
            for dx, dy in DIRS8:
                nx, ny = gx + dx, gy + dy
                if grid.walkable_cell(nx, ny):
                    cand.append(ny * grid.w + nx)
            if not cand:
                return None
            goal = min(cand, key=lambda i: abs(i % grid.w - gx) + abs(i // grid.w - gy))
        if start == goal:
            return [start]
        if not grid.walkable_idx(start):
            gx, gy = start % grid.w, start // grid.w
            cand = [i for i in (gy * grid.w + gx + d[1] * grid.w + d[0]
                                for d in DIRS8)
                    if 0 <= i < n and grid.walkable_idx(i)]
            if not cand:
                return None
            start = min(cand, key=lambda i: (i % grid.w - gx) ** 2 + (i // grid.w - gy) ** 2)
        def h(i):
            return abs(i % grid.w - goal % grid.w) + abs(i // grid.w - goal // grid.w)
        open_heap = [(h(start), 0.0, start)]
        came = {}
        gscore = {start: 0.0}
        seen = set()
        while open_heap:
            _, curg, cur = heapq.heappop(open_heap)
            if cur in seen:
                continue
            seen.add(cur)
            if cur == goal:
                path = [goal]
                while path[-1] != start:
                    path.append(came[path[-1]])
                path.reverse()
                return path
            cx, cy = cur % grid.w, cur // grid.w
            for dx, dy, cost in ((1, 0, 1.0), (-1, 0, 1.0), (0, 1, 1.0), (0, -1, 1.0),
                                 (1, 1, 1.414), (-1, 1, 1.414),
                                 (1, -1, 1.414), (-1, -1, 1.414)):
                nx, ny = cx + dx, cy + dy
                if not grid.walkable_cell(nx, ny):
                    continue
                # 斜向不能穿过两个墙角
                if dx != 0 and dy != 0:
                    if not (grid.walkable_cell(cx + dx, cy) and grid.walkable_cell(cx, cy + dy)):
                        continue
                ni = ny * grid.w + nx
                ng = curg + cost
                if grid.hazardous_idx(ni):
                    ng += hazard_penalty
                if ng + 1e-9 < gscore.get(ni, 1e18):
                    gscore[ni] = ng
                    came[ni] = cur
                    heapq.heappush(open_heap, (ng + h(ni), ng, ni))
        return None

    def _follow_path(self, p: dict, goal_idx: int, radius: float = 7.0) -> Optional[tuple[int, int]]:
        grid = self.grid
        if grid is None:
            return None
        start = grid.idx_of(float(p["x"]), float(p["y"]))
        if start < 0:
            return None
        self.path_age += 1
        if self.path_goal != goal_idx or not self.path or self.path_age > 18:
            self.path = self.astar(start, goal_idx) or []
            self.path_goal = goal_idx
            self.path_age = 0
        if not self.path:
            return None
        px, py = float(p["x"]), float(p["y"])
        # 路径平滑：找最远的、可直线走到的路径点
        best_i = 0
        for i in range(1, min(len(self.path), 10)):
            cx, cy = grid.center(self.path[i])
            if grid.grid_segment_clear(px, py, cx, cy, radius=radius):
                best_i = i
            else:
                break
        if best_i == len(self.path) - 1:
            cx, cy = grid.center(self.path[-1])
            if math.hypot(cx - px, cy - py) < 5.0:
                return (0, 0)
        cx, cy = grid.center(self.path[best_i])
        return _vec_to_dir8(cx - px, cy - py)

    # ------------------------------------------------------------------
    # 门/探索/拾取
    # ------------------------------------------------------------------
    def _map_step_toward(self, goal_pred) -> Optional[int]:
        """在“已经走过的门”构成的连通图上 BFS。

        不能用 13x13 格子相邻来推断连通性 —— 实测房间 84 的正上方格子是
        72，但 84 的上门其实通到 71，而 85 的上门是锁着的。按格子 BFS
        会得出 84→85→72 这种根本不存在的路，AI 就在两间房之间反复弹。
        """
        rooms = {int(r.get("grid", 0)): r for r in self.map_rooms}
        cur = self.room_index
        if cur is None or cur not in rooms:
            return None
        q = deque([cur])
        first = {cur: None}
        while q:
            r = q.popleft()
            if r != cur and goal_pred(rooms.get(r, {})):
                return first.get(r)
            for slot, nr in self.floor_edges.get(r, {}).items():
                if nr in rooms and nr not in first:
                    first[nr] = nr if r == cur else first[r]
                    q.append(nr)
        return None

    def _nearest_unvisited_step(self) -> Optional[int]:
        """朝最近的未访问房间走的第一跳；已知连通图里到不了就返回 None。"""
        return self._map_step_toward(
            lambda room: (int(room.get("visited", 0) or 0) == 0
                          and int(room.get("grid", 0)) not in self.visited_rooms))

    def _visited_map(self) -> dict[int, int]:
        return {int(r.get("grid", 0)): int(r.get("visited", 0) or 0)
                for r in self.map_rooms}

    def _door_neighbor(self, door: dict) -> Optional[int]:
        if self.room_index is None:
            return None
        delta = ROOM_DELTA.get(int(door.get("slot", -1)))
        return None if delta is None else self.room_index + delta

    def _is_unvisited(self, nr: Optional[int], vm: dict[int, int]) -> bool:
        # 地图的 visited 每 30 帧才刷新；刚进屋的这半秒要用本地记忆兜底，
        # 否则会立刻掉头再进一次刚出来的房间。
        return nr is not None and vm.get(nr, 0) == 0 and nr not in self.visited_rooms

    def _floor_fully_explored(self) -> bool:
        vm = self._visited_map()
        if len(vm) < 6:
            return False
        if any(v <= 0 and g not in self.visited_rooms for g, v in vm.items()):
            return False
        opens = [d for d in self.doors if d.get("open")]
        if any(self._is_unvisited(self._door_neighbor(d), vm) for d in opens):
            return False
        return not self._frontier_rooms()

    def _frontier_rooms(self) -> set[int]:
        """还有“见过的门但没走过”的房间。"""
        out = set()
        for r, slots in self.seen_door_slots.items():
            if slots - set(self.floor_edges.get(r, {})):
                out.add(r)
        return out

    def choose_door(self) -> Optional[dict]:
        if not self.doors:
            return None
        open_doors = [d for d in self.doors if d.get("open")]
        if not open_doors:
            return None
        vm = self._visited_map()

        # 1) 这扇门后面就是没去过的房间：直接探索
        unexplored = [d for d in open_doors
                      if self._is_unvisited(self._door_neighbor(d), vm)]
        if unexplored:
            return min(unexplored, key=lambda d: int(d.get("slot", 9)))

        # 2) 靠已经走过的连通图，去更远的未探索房间
        step = self._nearest_unvisited_step()
        # 3) 全图走完就去 Boss 房（type 5）
        if step is None:
            step = self._map_step_toward(
                lambda room: int(room.get("type", 0) or 0) == 5)
        if step is not None:
            toward = [d for d in open_doors
                      if self._door_neighbor(d) == step]
            if toward:
                return toward[0]

        # 4) 有些去过的房间还有没走过的门：沿已知连通图回去补图
        frontier = self._frontier_rooms()
        frontier.discard(self.room_index)
        if frontier:
            step = self._map_step_toward(
                lambda room: int(room.get("grid", 0)) in frontier)
            if step is not None:
                toward = [d for d in open_doors
                          if self._door_neighbor(d) == step]
                if toward:
                    return toward[0]

        # 5) 还有没走过的门：去补连通图，避免"反复去同一间已访问房"
        known = self.floor_edges.get(self.room_index, {})
        unknown = [d for d in open_doors
                   if int(d.get("slot", -1)) not in known]
        if unknown:
            return min(unknown, key=lambda d: (
                vm.get(self._door_neighbor(d), 0), int(d.get("slot", 9))))

        # 6) 都走过了：挑去得最少的门
        return min(open_doors, key=lambda d: (
            vm.get(self._door_neighbor(d), 0), int(d.get("slot", 9))))

    def _pickup_target(self, p: dict, pickups: list[Entity]) -> Optional[Entity]:
        if not pickups or self.grid is None:
            return None
        px, py = float(p["x"]), float(p["y"])
        cand = []
        for e in pickups:
            if e.id in self.pickup_blacklist:
                continue
            # 太远的房间边缘掉落物也一样去拿；这里只排除明显在不可达位置的
            cand.append(e)
        if not cand:
            return None
        return min(cand, key=lambda e: math.hypot(e.x - px, e.y - py))

    # ------------------------------------------------------------------
    # 主决策
    # ------------------------------------------------------------------
    def act(self, state: dict) -> Action:
        p = state.get("player") or {}
        room = state.get("room") or {}
        self.update_state(state)
        if not p or (float(p.get("hp", 0) or 0) + float(p.get("soul_hp", 0) or 0)) <= 0:
            return Action(mode="dead")
        px, py = float(p["x"]), float(p["y"])
        enemies, projectiles, pickups, hazards = split_entities(state)
        grid = self.grid
        if grid is None:
            return Action(mode="no-grid")

        # 卡住检测：只有"想走却几乎没位移"才算卡住。
        # 战斗中站桩射击是正常行为，旧版把它当卡死，每秒强制往房间中心跑一次。
        if self.prev_pos is not None and not room.get("clear") and self.last_move != (0, 0):
            moved = math.hypot(px - self.prev_pos[0], py - self.prev_pos[1])
            if moved < 0.6:
                self.stuck_frames += 1
            else:
                self.stuck_frames = max(0, self.stuck_frames - 2)
        else:
            self.stuck_frames = max(0, self.stuck_frames - 2)
        self.prev_pos = (px, py)
        if self.stuck_frames > 30:
            # 用一个短促的脱困方向打破卡死（下一帧重新评估）
            self.stuck_frames = 0
            self.path = []
            self.path_goal = None
            self.recover_frames = 12
            # 朝房间中心走通常最开阔
            cx = grid.ox + grid.w * grid.tw / 2
            cy = grid.oy + grid.h * grid.th / 2
            self.recover_dir = _vec_to_dir8(cx - px, cy - py)
        if self.recover_frames > 0:
            self.recover_frames -= 1
            return Action(move=self.recover_dir, mode="recover")

        # ---- 战斗 ----
        if enemies and not room.get("clear"):
            target = self.choose_target(p, enemies)
            self.last_target_id = target.id if target else None

            # 先用当前速度求一次射击解，给移动评分提供"现在能不能打"的信号
            vx0, vy0 = float(p.get("vx", 0) or 0), float(p.get("vy", 0) or 0)
            aim0 = self.aim_solution(p, target, (vx0, vy0), grid) if target else None
            move, detail = self.plan_move(p, target, enemies, projectiles, hazards, aim0, False)

            # 用计划移动后的稳态速度重新求射击解（泪弹继承人物速度）。
            # 按满速算：按键后人物几帧内就到全速，而泪弹是在 fire_delay 之后才出膛的。
            ux, uy = _dir8_to_vec(*move)
            planned_vel = ((ux * self.speed_est, uy * self.speed_est)
                           if move != (0, 0) else (vx0, vy0))
            aim = self.aim_solution(p, target, planned_vel, grid) if target else None

            # "长时间打不到"用真正的射击解计数，而不是到敌人中心的视线：
            # 移动目标经常中心被挡但弹道仍可命中（反之亦然）。
            if target is None or aim is not None:
                self.no_aim_frames = 0
            else:
                self.no_aim_frames += 1

            # 如果已经在挨打，先躲，不开枪（少一发射击不影响生存）
            threat = self._threat_now(p, projectiles, enemies, hazards)
            shoot = SHOOT_DIRS[aim["dir"]] if (aim is not None and threat < 0.65) else (0, 0)
            if aim is not None and threat >= 0.65:
                aim = None
                detail["dodge_overrides_aim"] = True
            # 没有射击解且长时间打不到 -> 用 A* 走到有视线的位置（交战点）
            if aim is None and self.no_aim_frames > 24 and target is not None:
                spots = self._engage_spots(target)
                if spots:
                    goal = min(spots, key=lambda i: (i % grid.w - grid.idx_of(px, py) % grid.w) ** 2
                               + (i // grid.w - grid.idx_of(px, py) // grid.w) ** 2)
                    nav = self._follow_path(p, goal, radius=7.0)
                    if nav is not None and nav != (0, 0):
                        move = nav
                        detail["reposition"] = True
                        self.no_aim_frames = 0
            self.last_move = move
            self.last_shoot = shoot
            return Action(move=move, shoot=shoot, mode="combat",
                          target_id=target.id if target else None,
                          aim_dir=aim["dir"] if aim else 0,
                          debug={**detail, "target": (target.x, target.y) if target else None,
                                 "aim_miss": aim["miss"] if aim else None,
                                 "threat": threat,
                                 "speed": self.speed_est})

        # ---- 清房后的拾取/开门 ----
        if room.get("clear"):
            self.recover_frames = 0
            # 1) 先拿房间里的掉落物
            pk = self._pickup_target(p, pickups)
            if pk is not None and self.pickup_wait < 240:
                goal = grid.idx_of(pk.x, pk.y)
                if goal >= 0:
                    # 已经站在掉落物附近 -> 等游戏拾取
                    if math.hypot(pk.x - px, pk.y - py) < 22:
                        self.pickup_wait += 1
                        if self.pickup_wait > 6:
                            self.pickup_blacklist.add(pk.id)
                            self.pickup_wait = 0
                        self.last_move = (0, 0)
                        return Action(mode="pickup", target_id=pk.id)
                    nav = self._follow_path(p, goal, radius=6.0)
                    if nav is not None and nav != (0, 0):
                        self.pickup_wait += 1
                        self.last_move = nav
                        return Action(move=nav, mode="pickup", target_id=pk.id,
                                      debug={"pickup": (pk.x, pk.y)})
                    self.pickup_wait += 10
                    if self.pickup_wait > 30:
                        self.pickup_blacklist.add(pk.id)
                        self.pickup_wait = 0
            # 2) 陷阱门/楼梯：全图探索完后优先下去
            trapdoors = grid.trapdoor_cells()
            if trapdoors and self._floor_fully_explored():
                pidx = grid.idx_of(px, py)
                goal = min(trapdoors, key=lambda i: abs(i % grid.w - pidx % grid.w)
                           + abs(i // grid.w - pidx // grid.w))
                nav = self._follow_path(p, goal, radius=6.0)
                if nav is not None:
                    self.last_move = nav
                    return Action(move=nav, mode="trapdoor")
            # 3) 去下一扇门
            door = self.choose_door()
            if door is not None:
                dg = int(door.get("grid", -1))
                # 站在门格里了：朝门外推
                dx0 = px - (grid.ox + (grid.w // 2) * grid.tw)
                dy0 = py - (grid.oy + (grid.h // 2) * grid.th)
                slot = int(door.get("slot", -2))
                if self.door_pushing != dg:
                    self.door_pushing = dg
                    self.door_push = 0
                near = (grid.idx_of(px, py) == dg)
                if near or (dg >= 0 and math.hypot(
                        *[a - b for a, b in zip(grid.center(dg), (px, py))]) < 16):
                    # slot: 0=左 1=上 2=右 3=下
                    out = {0: (-1, 0), 1: (0, -1), 2: (1, 0), 3: (0, 1),
                           4: (-1, 0), 5: (0, -1), 6: (1, 0), 7: (0, 1)}.get(slot)
                    if out is None:
                        out = _vec_to_dir8(dx0, dy0)
                    self.door_push += 1
                    self.pushed_slot = slot
                    self.pushed_from = self.room_index
                    self.last_move = out
                    return Action(move=out, mode="door", debug={"door": dg, "push": True})
                if dg >= 0:
                    nav = self._follow_path(p, dg, radius=6.0)
                    if nav is not None:
                        self.last_move = nav
                        return Action(move=nav, mode="door", debug={"door": dg})
            self.last_move = (0, 0)
            return Action(mode="idle")

        # 有敌人但房间标记清空（过渡帧）
        self.last_move = (0, 0)
        return Action(mode="idle")

    def _engage_spots(self, target: Entity) -> list[int]:
        """目标周围能打到它的候选格子：同一行/列、理想距离附近。"""
        grid = self.grid
        if grid is None:
            return []
        spots = []
        cs = max(grid.tw, grid.th)
        for axis in (0, 1):
            for sign in (-1, 1):
                for off in (self.cfg.d_opt - 25, self.cfg.d_opt, self.cfg.d_opt + 25):
                    if axis == 0:       # 水平射击：站在同一 y
                        x, y = target.x + sign * off, target.y
                    else:
                        x, y = target.x, target.y + sign * off
                    if not (grid.ox <= x < grid.ox + grid.w * grid.tw and
                            grid.oy <= y < grid.oy + grid.h * grid.th):
                        continue
                    if not grid.point_clear(x, y, self.cfg.wall_radius):
                        continue
                    if not grid.segment_clear(x, y, target.x, target.y,
                                              radius=self.cfg.tear_radius * 0.6):
                        continue
                    idx = grid.idx_of(x, y)
                    if idx >= 0 and not grid.hazardous_idx(idx):
                        spots.append(idx)
        return spots
