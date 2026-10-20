# -*- coding: utf-8 -*-
"""isaac_data.py —— 模组上报数据 → 「人话」/ 445 维观测对照表 / 共用数据源

`dump_state.py`（文本探针）和 `viz_viewer.py`（可视化窗口）都依赖本模块，
保证两边看到的枚举名字、观测切分完全一致，不会各写一套而对不上。

**单一事实来源**：观测向量的形状与顺序一律来自 `isaac_env.build_obs` / `OBS_DIM`。
本模块只负责"给每一维起个名字"，绝不重复实现观测逻辑 —— 否则改了一边忘了另一边，
看到的可视化就是骗人的。
"""
from __future__ import annotations

import json
import math
import os
import socket
import sys
import threading
import time

import numpy as np

# 观测的真正定义在这里；本模块只做标注
try:
    from isaac_env import (CANVAS_H, CANVAS_W, GRID_WIN, MAX_ENEMIES, MAX_TEARS,
                           OBS_DIM, OBS_DIM_FULL, PLAYER_DIM, TERRAIN_CHANNELS,
                           TERRAIN_FULL, TERRAIN_LOCAL, _hazard_cells, build_obs,
                           normalize_grid, obs_dim, terrain_dim)
except ImportError as _exc:                       # pragma: no cover - 环境问题
    _prefix = os.environ.get("CONDA_PREFIX")
    _hint = (f"  或者显式指定：\"{_prefix}\\python.exe\" dump_state.py\n"
             if _prefix else
             "  先激活环境： conda activate Issac_env\n")
    raise ImportError(
        f"导入 isaac_env 失败：{_exc}\n"
        f"\n"
        f"  当前解释器     : {sys.executable}\n"
        f"  已激活的 conda : {_prefix or '(没有激活任何环境)'}\n"
        f"\n"
        f"  最常见原因：Windows 下直接敲 `dump_state.py` 会交给文件关联的\n"
        f"  py 启动器（C:\\Windows\\py.exe），而那用的是 D:\\Python313\\python.exe，\n"
        f"  不是你以为的 Issac_env。pip 装到的和脚本跑的不是同一个 Python。\n"
        f"\n"
        f"  正确用法：   python dump_state.py      （带 python 前缀）\n"
        f"{_hint}"
        f"  确认环境：   python -c \"import sys, gymnasium; print(sys.executable)\"\n"
    ) from _exc

# ---------------------------------------------------------------- 枚举 → 中文

# EntityType（只列房间里实际会遇到的，其余显示成 type=NN）
ENTITY_NAMES = {
    0: "空", 1: "玩家", 2: "泪弹", 3: "跟班", 4: "炸弹", 5: "掉落物", 6: "老虎机",
    7: "激光", 8: "刀", 9: "敌方弹",
    10: "跛行者", 11: "喷吐者", 12: "霍尔夫", 13: "苍蝇", 14: "噗噗怪", 15: "凝块",
    16: "毛利根", 17: "店主", 18: "攻击蝇", 19: "拉里", 20: "蒙斯特罗", 21: "蛆",
    22: "蜂巢", 23: "冲锋者", 24: "哥布林", 25: "爆蝇", 26: "大嘴", 27: "宿主",
    28: "丘布", 29: "跳跳", 30: "煮沸", 31: "喷溅", 32: "大脑", 33: "壁炉",
    34: "跳跃者", 35: "蒙斯特罗之嘴", 36: "格蒂", 38: "婴儿", 39: "维丝",
    40: "肠子", 41: "骑士", 42: "石头头", 43: "蒙斯特罗2", 44: "波奇", 45: "妈",
    46: "懒惰", 47: "色欲", 48: "愤怒", 49: "暴食", 50: "贪婪", 51: "嫉妒",
    52: "傲慢", 53: "多普", 54: "火焰跳跳", 55: "水蛭", 56: "肉块", 57: "脑怪",
    58: "寄生咬", 59: "弗雷德", 60: "眼球", 61: "吸盘", 62: "钉刺", 63: "饥荒",
    64: "瘟疫", 65: "战争", 66: "死亡", 67: "苍蝇公爵", 68: "窥视", 69: "洛基",
    71: "瘘管怪", 1000: "特效",
}

# GridEntityType（地形格上的静态物件）
GRID_ENTITY_NAMES = {
    0: "空", 1: "装饰", 2: "岩石", 3: "方块", 4: "彩色岩", 5: "可炸岩", 6: "罐/蘑菇/头骨",
    7: "坑", 8: "尖刺", 9: "开关尖刺", 10: "蛛网", 11: "钥匙砖", 12: "TNT",
    13: "壁炉(未用)", 14: "便便", 15: "墙", 16: "门", 17: "活板门", 18: "楼梯",
    19: "重力", 20: "压力板", 21: "雕像", 22: "超级彩岩", 23: "传送器", 24: "柱子",
    25: "尖刺岩", 26: "彩头骨", 27: "金岩",
}

# GridCollisionClass：这一格"走不走得进去"
GRID_COLL_NAMES = {
    0: "可走", 1: "坑", 2: "物体", 3: "实心", 4: "墙", 5: "墙(玩家可过)",
}

# EntityCollisionClass：这个实体"会不会撞到我"
ENTITY_COLL_NAMES = {
    0: "不碰任何东西", 1: "只碰玩家", 2: "碰玩家/泪弹/跟班", 3: "碰敌人", 4: "碰一切",
}

ROOM_TYPE_NAMES = {
    1: "普通", 2: "商店", 3: "错误房", 4: "宝箱房", 5: "Boss房", 6: "小Boss房",
    7: "隐藏房", 8: "超级隐藏房", 9: "赌博房", 10: "刺房", 11: "挑战房",
    12: "书房", 13: "骰子房", 14: "恶魔房", 15: "天使房", 16: "秘密出口",
    17: "普通Boss", 18: "蓝子宫", 19: "虚空", 20: "MegaSatan", 21: "牢房",
    22: "暗室", 23: "死寂", 24: "宝箱房(已开)", 25: "贪婪模式", 26: "贪婪Boss",
    27: "狗窝", 28: "天文台", 29: "超宝箱房", 30: "净化房",
}

PICKUP_VARIANT_NAMES = {
    10: "红心", 20: "金币", 30: "钥匙", 40: "炸弹", 90: "道具底座",
    100: "道具底座", 300: "箱子", 350: "饰品", 360: "卡牌", 370: "药丸",
}

# Direction 枚举
DIR_NAMES = {-1: "无", 0: "左", 1: "上", 2: "右", 3: "下"}


def entity_name(t, variant=None):
    """实体类型的可读名字；未知类型显示成 type=NN（而不是瞎猜）。"""
    t = int(t)
    if t == 5 and variant is not None:
        return PICKUP_VARIANT_NAMES.get(int(variant), f"掉落#{int(variant)}")
    return ENTITY_NAMES.get(t, f"type={t}")


def grid_ent_name(v):
    v = int(v)
    return GRID_ENTITY_NAMES.get(v, f"地形#{v}")


def grid_coll_name(c):
    c = int(c)
    return GRID_COLL_NAMES.get(c, f"碰撞#{c}")


def entity_coll_name(c):
    c = int(c)
    return ENTITY_COLL_NAMES.get(c, f"碰撞#{c}")


def room_type_name(v):
    v = int(v)
    return ROOM_TYPE_NAMES.get(v, f"房间类型#{v}")


def dir_name(d):
    try:
        return DIR_NAMES.get(int(d), str(d))
    except (TypeError, ValueError):
        return str(d)


# ---------------------------------------------------------------- 观测标签

# (键, 中文名, 维度, 配色)
def obs_groups(terrain_mode=TERRAIN_LOCAL):
    """观测分块：(键, 中文名, 维度, 配色)。地形块的维度随模式变化。"""
    return [
        ("player", "玩家", PLAYER_DIM, "#4C78A8"),
        ("enemy", "敌人", MAX_ENEMIES * 8, "#E45756"),
        ("tear", "敌弹", MAX_TEARS * 5, "#F58518"),
        ("grid", "地形", terrain_dim(terrain_mode), "#54A24B"),
        ("room", "房间", 2, "#B279A2"),
        ("door", "门", 12, "#72B7B2"),
    ]


OBS_GROUPS = obs_groups(TERRAIN_LOCAL)      # 默认模式的兼容别名

_ENEMY_FEATS = ("相对x", "相对y", "vx", "vy", "血量", "是boss", "体积", "碰撞类")
_TEAR_FEATS = ("相对x", "相对y", "vx", "vy", "高度")

_LABEL_CACHE = {}


def obs_labels(terrain_mode=TERRAIN_LOCAL):
    """返回长度 == obs_dim(terrain_mode) 的标签表：每一维在网络眼里叫什么。

    顺序与 `isaac_env.build_obs` 严格一致；对不上就直接抛异常，
    免得可视化跟真实观测错位还看不出来。
    """
    if terrain_mode in _LABEL_CACHE:
        return _LABEL_CACHE[terrain_mode]

    lab = [
        ("player", "玩家x"), ("player", "玩家y"), ("player", "玩家vx"), ("player", "玩家vy"),
        ("player", "血量"), ("player", "存活"),
        ("player", "射击向x"), ("player", "射击向y"), ("player", "头向x"), ("player", "头向y"),
        ("player", "无敌帧"), ("player", "伤害"), ("player", "泪速"), ("player", "移速"),
    ]
    for i in range(MAX_ENEMIES):
        lab += [("enemy", f"敌{i}·{f}") for f in _ENEMY_FEATS]
    for i in range(MAX_TEARS):
        lab += [("tear", f"弹{i}·{f}") for f in _TEAR_FEATS]

    if terrain_mode == TERRAIN_FULL:
        # 展平顺序 = 通道优先：先整张"可走"图，再整张"危险"图
        for ch, chn in ((0, "可走"), (1, "危险")):
            for y in range(CANVAS_H):
                for x in range(CANVAS_W):
                    lab.append(("grid", f"{chn}{y},{x}"))
    else:
        for dy in range(-(GRID_WIN // 2), GRID_WIN // 2 + 1):
            for dx in range(-(GRID_WIN // 2), GRID_WIN // 2 + 1):
                lab.append(("grid", f"格{dx:+d},{dy:+d}"))

    lab += [("room", "敌人数量"), ("room", "已清空")]
    for s in range(4):
        lab += [("door", f"门{s}·x"), ("door", f"门{s}·y"), ("door", f"门{s}·开")]

    want = obs_dim(terrain_mode)
    if len(lab) != want:
        raise RuntimeError(
            f"观测标签数 {len(lab)} != obs_dim({terrain_mode})={want}："
            f"isaac_env.build_obs 改了但本模块没同步")
    _LABEL_CACHE[terrain_mode] = lab
    return lab


def block_slices(terrain_mode=TERRAIN_LOCAL):
    """各块在观测向量里的切片（顺序同 build_obs）。"""
    t0 = PLAYER_DIM + MAX_ENEMIES * 8 + MAX_TEARS * 5
    t1 = t0 + terrain_dim(terrain_mode)
    return {
        "player": (0, PLAYER_DIM),
        "enemy": (PLAYER_DIM, PLAYER_DIM + MAX_ENEMIES * 8),
        "tear": (PLAYER_DIM + MAX_ENEMIES * 8, t0),
        "grid": (t0, t1),
        "room": (t1, t1 + 2),
        "door": (t1 + 2, obs_dim(terrain_mode)),
    }


BLOCK_SLICES = block_slices(TERRAIN_LOCAL)   # 默认模式的兼容别名


def obs_blocks(obs, terrain_mode=TERRAIN_LOCAL):
    """把观测向量按语义切块，返回 dict：键 -> ndarray（已 reshape 成直观形状）。

    * enemy: (16, 8) —— 一行一个敌人槽位，一列一个特征
    * tear : (5, 48) —— 转置成一列一个弹槽位，便于横向展示
    * grid : local=(7, 7) 玩家为中心的局部地形
             full =(2, 16, 28) 整间房画布：ch0 可走 / ch1 危险
    """
    obs = np.asarray(obs, dtype=np.float32).ravel()
    sl = block_slices(terrain_mode)
    g0, g1 = sl["grid"]
    d0, d1 = sl["door"]
    grid = (obs[g0:g1].reshape(TERRAIN_CHANNELS, CANVAS_H, CANVAS_W)
            if terrain_mode == TERRAIN_FULL
            else obs[g0:g1].reshape(GRID_WIN, GRID_WIN))
    return {
        "player": obs[slice(*sl["player"])],
        "enemy": obs[sl["enemy"][0]:sl["enemy"][1]].reshape(MAX_ENEMIES, 8),
        "tear": obs[sl["tear"][0]:sl["tear"][1]].reshape(MAX_TEARS, 5).T,
        "grid": grid,
        "room": obs[slice(*sl["room"])],
        "door": obs[d0:d1].reshape(4, 3),
    }


def compute_obs(state, grid, terrain_mode=TERRAIN_LOCAL):
    """按 isaac_env 的口径算观测；不在对局里时返回全零。"""
    if not state or "player" not in state:
        return np.zeros(obs_dim(terrain_mode), dtype=np.float32)
    return build_obs(state, grid, terrain_mode)


# ---------------------------------------------------------------- 地形分类


CAT_KEYS = ("floor", "wall", "rock", "poop", "pit", "spike", "door", "other")
CAT_NAMES = ("地板", "墙", "石头", "便便", "坑", "尖刺", "门", "其它")
CAT_COLORS = ("#f2efe9", "#2f2f38", "#8d8d99", "#8b5a2b",
              "#0b0b10", "#c8b6ff", "#3fa9f5", "#666666")
_CAT = {k: i for i, k in enumerate(CAT_KEYS)}
CAT_GLYPHS = (".", "#", "o", "p", " ", "^", "+", "?")

# 算作"石头/障碍"的地形实体
_ROCK_ENTS = (2, 3, 4, 5, 6, 11, 22, 25, 26, 27)
_SPIKE_ENTS = (8, 9)


def cell_categories(state, grid):
    """把房间网格归类成 8 类，返回 (h, w) 的 int8 数组。

    `coll` 决定走不走得进去，`ent` 决定这块地"长什么样"。
    只有 `coll` 的话，石头和便便、坑和墙全都长一个样 —— 那正是当前观测的缺陷。
    """
    if not grid:
        return None
    w, h = int(grid["w"]), int(grid["h"])
    coll = grid.get("coll") or [0] * (w * h)
    ent = grid.get("ent") or [0] * (w * h)
    out = np.zeros((h, w), dtype=np.int8)
    for i in range(w * h):
        c = int(coll[i]) if i < len(coll) else 0
        e = int(ent[i]) if i < len(ent) else 0
        if c == 5 or e == 16:
            k = "door"
        elif c == 1 or e == 7:
            k = "pit"
        elif e in _SPIKE_ENTS:
            k = "spike"
        elif c == 4:
            k = "wall"
        elif e == 14:
            k = "poop"
        elif c in (2, 3) or e in _ROCK_ENTS:
            k = "rock"
        elif c == 0:
            k = "floor"
        else:
            k = "other"
        out[i // w, i % w] = _CAT[k]
    return out


def ascii_room(state, grid, max_w=40):
    """把房间画成字符画，终端里直接看。墙# 石头o 便便p 尖刺^ 门+ 地板.

    实体叠加在上面：@ 玩家 / E 敌人 / * 弹幕 / F 火堆 / $ 掉落。
    """
    if not grid:
        return ["(没有地形数据：模组还没发出 grid，或不在对局中)"]
    cats = cell_categories(state, grid)
    h, w = cats.shape
    if w > max_w:
        return [f"(房间 {w}x{h} 太宽，字符画跳过)"]

    g = [[CAT_GLYPHS[c] for c in row] for row in cats]

    def put(idx, ch):
        try:
            gy, gx = int(idx) // w, int(idx) % w
        except (TypeError, ValueError):
            return
        if 0 <= gy < h and 0 <= gx < w:
            g[gy][gx] = ch

    for e in state.get("entities", []):
        t = int(e.get("type", 0))
        cdmg = float(e.get("cdmg", 0) or 0)
        if t == 1:
            put(e.get("grid_idx"), "@")
        elif t == 5:
            put(e.get("grid_idx"), "$")
        elif t == 33 or cdmg > 0:
            put(e.get("grid_idx"), "F")
        elif e.get("enemy") or e.get("boss"):
            put(e.get("grid_idx"), "E")
        elif t in (2, 9, 7, 8):
            put(e.get("grid_idx"), "*")
    head = "   " + "".join(str(i % 10) for i in range(w))
    lines = [head]
    for gy in range(h):
        lines.append(f"{gy:2d} " + "".join(g[gy]))
    return lines


# ---------------------------------------------------------------- 数据源


class StateFeed:
    """在指定端口上等游戏连上来，随时取最新一帧。

    只做"收"，不参与训练 —— 训练用的是 multi_instance.MultiPortBridge。
    单独的轻量实现是为了让探针/可视化能在没装训练依赖时也跑得动。
    """

    def __init__(self, port=48532, host="127.0.0.1"):
        self.port = port
        self.host = host
        self._latest = None
        self._lock = threading.Lock()
        self._running = False
        self._thread = None
        self._server = None
        self._conn = None
        self.connected = False
        self._rate_ref = None      # (墙钟, 游戏帧) —— 算实际帧率
        self.fps = 0.0
        self.error = None

    def start(self):
        self._server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self._server.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        try:
            self._server.bind((self.host, self.port))
        except OSError as exc:
            self.error = f"端口 {self.port} 被占用（{exc}）。换个 --port 或者先关掉占用它的程序。"
            raise
        self._server.listen(4)
        self._server.settimeout(1.0)
        self._running = True
        self._thread = threading.Thread(target=self._loop, daemon=True)
        self._thread.start()
        return self

    def _loop(self):
        while self._running:
            try:
                conn, _ = self._server.accept()
            except socket.timeout:
                continue
            except OSError:
                break
            with self._lock:
                self._conn = conn
                self.connected = True
            buf = b""
            conn.settimeout(1.0)
            while self._running:
                try:
                    data = conn.recv(65536)
                except socket.timeout:
                    continue
                except OSError:
                    break
                if not data:
                    break
                buf += data
                while b"\n" in buf:
                    line, buf = buf.split(b"\n", 1)
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        msg = json.loads(line.decode("utf-8"))
                    except (json.JSONDecodeError, UnicodeDecodeError):
                        continue
                    self._update(msg)
            with self._lock:
                self._conn = None
                self.connected = False
            try:
                conn.close()
            except OSError:
                pass

    def _update(self, msg):
        with self._lock:
            self._latest = msg
        fr = msg.get("frame")
        if isinstance(fr, int):
            now = time.time()
            if self._rate_ref is None:
                self._rate_ref = (now, fr)
            else:
                t0, f0 = self._rate_ref
                if now - t0 >= 1.5:
                    if fr > f0:
                        self.fps = (fr - f0) / (now - t0)
                    self._rate_ref = (now, fr)

    def latest(self):
        with self._lock:
            return self._latest

    def send(self, text):
        """下发一行命令（restart / goto d.12 / exec ...）。"""
        with self._lock:
            conn = self._conn
        if conn is None:
            return False
        try:
            conn.sendall((text.rstrip("\n") + "\n").encode("utf-8"))
            return True
        except OSError:
            return False

    def wait_first(self, timeout=60.0, poll=0.05):
        t0 = time.time()
        while time.time() - t0 < timeout:
            s = self.latest()
            if s and "player" in s:
                return s
            time.sleep(poll)
        return None

    def close(self):
        self._running = False
        for s in (self._server,):
            try:
                if s:
                    s.close()
            except OSError:
                pass


# ---------------------------------------------------------------- 演示数据

DEMO_W, DEMO_H = 15, 9
CELL = 40.0
DEMO_OX, DEMO_OY = -280.0, -140.0     # 房间左上角的世界坐标
_DEMO_GRID_CACHE = {}


def demo_grid(w=DEMO_W, h=DEMO_H):
    """合成一张典型房间网格（没开游戏时用来看可视化长什么样）。"""
    key = (w, h)
    if key in _DEMO_GRID_CACHE:
        return _DEMO_GRID_CACHE[key]
    coll = [0] * (w * h)
    ent = [0] * (w * h)

    def put(gx, gy, c, e):
        if 0 <= gx < w and 0 <= gy < h:
            coll[gy * w + gx] = c
            ent[gy * w + gx] = e

    for gx in range(w):                      # 上下墙
        put(gx, 0, 4, 15)
        put(gx, h - 1, 4, 15)
    for gy in range(h):                      # 左右墙
        put(0, gy, 4, 15)
        put(w - 1, gy, 4, 15)
    for gx, gy in ((w // 2, 0), (w // 2, h - 1), (0, h // 2), (w - 1, h // 2)):
        put(gx, gy, 5, 16)                   # 门：coll=5 玩家可过

    for gx, gy in ((3, 2), (4, 2), (3, 3), (4, 3), (11, 2)):   # 石头
        put(gx, gy, 3, 2)
    put(10, 6, 3, 14)                        # 便便
    put(12, 6, 1, 7)                         # 坑
    put(5, 6, 2, 8)                          # 尖刺

    grid = {
        "w": w, "h": h,
        "cells": [1 if c in (0, 5) else 0 for c in coll],
        "coll": coll, "ent": ent,
        "ox": DEMO_OX, "oy": DEMO_OY,
        "tw": CELL, "th": CELL,
    }
    _DEMO_GRID_CACHE[key] = grid
    return grid


def demo_state(t=0.0):
    """随时间走的假状态，用来在不启动游戏的情况下预览可视化。"""
    grid = demo_grid()
    w, h = grid["w"], grid["h"]
    tw, th = grid["tw"], grid["th"]
    ox, oy = grid["ox"], grid["oy"]

    def wx(gx):
        return ox + (gx + 0.5) * tw

    def wy(gy):
        return oy + (gy + 0.5) * th

    px = wx(7) + 26 * math.sin(t * 1.3)
    py = wy(5) + 18 * math.cos(t * 0.9)

    enemies_xy = [(wx(11), wy(2)), (wx(4), wy(3)), (wx(12), wy(5))]
    enemies = []
    for i, (ex, ey) in enumerate(enemies_xy):
        ox_ = 6 * math.sin(t * (0.7 + 0.2 * i))
        oy_ = 6 * math.cos(t * (0.9 + 0.15 * i))
        enemies.append({
            "id": 100 + i, "type": (14, 13, 15)[i], "variant": 0, "subtype": 0,
            "x": ex + ox_, "y": ey + oy_, "vx": ox_ * -1.2, "vy": oy_ * -1.2,
            "hp": (20.0, 8.0, 14.0)[i], "boss": False, "enemy": True, "dead": False,
            "spawner_type": 0, "size": (12.0, 8.0, 11.0)[i], "height": 0.0,
            "coll": 1, "cdmg": 0.0, "gcoll": 1,
            "grid_idx": int(int((enemies_xy[i][1] - oy) / th) * w +
                            int((enemies_xy[i][0] - ox) / tw)),
        })

    tears = []
    for k in range(7):
        phase = (t * 0.5 + k / 7.0) % 1.0
        sx, sy = enemies_xy[k % 3]
        tears.append({
            "id": 200 + k, "type": 2, "variant": 0, "subtype": 0,
            "x": sx + (px - sx) * phase, "y": sy + (py - sy) * phase,
            "vx": (px - sx) * 0.4, "vy": (py - sy) * 0.4,
            "hp": 0.0, "boss": False, "enemy": False, "dead": False,
            "spawner_type": 0, "size": 8.0, "height": 0.0,
            "coll": 0, "cdmg": 0.0, "gcoll": 0,
            "grid_idx": int(int(py - oy) / th) * w + int((px - ox) / tw),
        })

    ents = [{
        "id": 1, "type": 1, "variant": 0, "subtype": 0,
        "x": px, "y": py, "vx": 0.0, "vy": 0.0,
        "hp": 6.0, "boss": False, "enemy": False, "dead": False,
        "spawner_type": 0, "size": 10.0, "height": 0.0,
        "coll": 4, "cdmg": 0.0, "gcoll": 1,
        "grid_idx": int(int(py - oy) / th) * w + int((px - ox) / tw),
    }]
    ents += enemies
    ents += tears
    ents.append({                              # 火堆：cdmg>0 → 观测里的危险格
        "id": 300, "type": 33, "variant": 0, "subtype": 0,
        "x": wx(9), "y": wy(5), "vx": 0.0, "vy": 0.0,
        "hp": 100.0, "boss": False, "enemy": True, "dead": False,
        "spawner_type": 0, "size": 16.0, "height": 0.0,
        "coll": 1, "cdmg": 1.0, "gcoll": 1,
        "grid_idx": 5 * w + 9,
    })
    ents.append({                              # 掉落的红心
        "id": 400, "type": 5, "variant": 10, "subtype": 0,
        "x": wx(2), "y": wy(7), "vx": 0.0, "vy": 0.0,
        "hp": 0.0, "boss": False, "enemy": False, "dead": False,
        "spawner_type": 0, "size": 10.0, "height": 0.0,
        "coll": 1, "cdmg": 0.0, "gcoll": 1,
        "grid_idx": 7 * w + 2,
    })

    nearest = min(enemies_xy, key=lambda p: (p[0] - px) ** 2 + (p[1] - py) ** 2)
    dx, dy = nearest[0] - px, nearest[1] - py
    fire_dir = 2 if abs(dx) >= abs(dy) and dx > 0 else \
               0 if abs(dx) >= abs(dy) else \
               3 if dy > 0 else 1

    return {
        "frame": int(t * 30) + 1000,
        "player": {
            "x": px, "y": py, "vx": 26 * 1.3 * math.cos(t * 1.3),
            "vy": -18 * 0.9 * math.sin(t * 0.9),
            "hp": 5.0, "max_hp": 6.0, "soul_hp": 2.0,
            "coins": 4, "bombs": 1, "keys": 0,
            "active_item": 520, "active_charge": 2, "pocket_item": 0, "card": 0,
            "damage": 3.5, "move_speed": 1.0, "tear_range": 260.0,
            "shot_speed": 10.0, "fire_delay": 10.0, "size": 10.0,
            "grid_idx": int(int(py - oy) / th) * w + int((px - ox) / tw),
            "fire_dir": fire_dir, "head_dir": fire_dir, "iframes": 0,
        },
        "room": {
            "type": 1, "index": -3, "enemies_alive": len(enemies),
            "bosses_alive": 0, "clear": False,
        },
        "entities": ents,
        "doors": [
            {"slot": 0, "grid": 0 * w + w // 2, "open": False},
            {"slot": 1, "grid": (h // 2) * w + (w - 1), "open": True},
            {"slot": 2, "grid": (h - 1) * w + w // 2, "open": False},
            {"slot": 3, "grid": (h // 2) * w + 0, "open": False},
        ],
        "grid": grid,
        "map": [
            {"grid": 84, "type": 1, "visited": 3},
            {"grid": 83, "type": 1, "visited": 1},
            {"grid": 85, "type": 5, "visited": 0},
        ],
        "_demo": True,
    }
