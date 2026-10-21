# -*- coding: utf-8 -*-
"""test_tactical_agent.py —— 不用开游戏，验证战术 AI 的瞄准/门控/躲弹/绕障。

运行：
    python tests/test_tactical_agent.py
"""
import json
import math
import os
import sys

try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

# 允许直接 `python tests/test_tactical_agent.py`
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from isaac_ai.tactical_agent import (Config, TacticalAgent, RoomGrid, DIRS8,
                                     SHOOT_DIRS, ROOM_DELTA, _dir8_to_vec,
                                     split_entities)

DEMO_W, DEMO_H = 15, 9
CELL = 40.0
DEMO_OX, DEMO_OY = -280.0, -140.0     # 房间左上角的世界坐标
_DEMO_GRID_CACHE = {}


def demo_grid(w=DEMO_W, h=DEMO_H):
    """合成一张典型房间网格，离线测试用（不依赖任何第三方库）。"""
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
        "gv": 2,
        "ox": DEMO_OX, "oy": DEMO_OY,
        "tw": CELL, "th": CELL,
    }
    _DEMO_GRID_CACHE[key] = grid
    return grid


def mk_state(grid, px, py, enemies=(), bullets=(), pickups=(), clear=False,
             room_index=50, player_v=(0.0, 0.0), hp=6.0, doors=None, bombs=3):
    ent = [{
        "id": 1, "type": 1, "variant": 0, "x": px, "y": py,
        "vx": player_v[0], "vy": player_v[1], "hp": hp, "enemy": False,
        "boss": False, "spawner_type": 0, "size": 10.0, "cdmg": 0.0,
    }]
    for i, e in enumerate(enemies):
        rec = {
            "id": 100 + i, "type": int(e.get("type", 10)), "variant": 0,
            "x": e["x"], "y": e["y"], "vx": e.get("vx", 0.0), "vy": e.get("vy", 0.0),
            "hp": e.get("hp", 10.0), "enemy": True, "boss": bool(e.get("boss")),
            "spawner_type": 0, "size": e.get("size", 13.0), "cdmg": e.get("cdmg", 1.0),
        }
        for k in ("anim", "height", "fall_speed", "fall_accel", "sprite_off_y",
                  "flying", "visible", "invincible"):
            if k in e:
                rec[k] = e[k]
        ent.append(rec)
    for i, b in enumerate(bullets):
        rec = {
            "id": 300 + i, "type": 2, "variant": 0,
            "x": b["x"], "y": b["y"], "vx": b.get("vx", 0.0), "vy": b.get("vy", 0.0),
            "hp": 0.0, "enemy": False, "boss": False,
            "spawner_type": 0, "size": b.get("size", 5.0), "cdmg": b.get("cdmg", 0.0),
        }
        for k in ("height", "fall_speed", "fall_accel"):
            if k in b:
                rec[k] = b[k]
        ent.append(rec)
    for i, pk in enumerate(pickups):
        ent.append({
            "id": 500 + i, "type": 5, "variant": pk.get("variant", 10),
            "x": pk["x"], "y": pk["y"], "vx": 0.0, "vy": 0.0,
            "hp": 0.0, "enemy": False, "boss": False,
            "spawner_type": 0, "size": pk.get("size", 10.0), "cdmg": 0.0,
        })
    return {
        "frame": 1,
        "player": {
            "x": px, "y": py, "vx": player_v[0], "vy": player_v[1],
            "hp": hp, "max_hp": 6, "soul_hp": 0,
            "damage": 3.5, "move_speed": 1.0, "tear_range": 260.0,
            "shot_speed": 1.0, "fire_delay": 8.0, "size": 10.0,
            "grid_idx": 0, "fire_dir": -1, "head_dir": 3, "iframes": 0,
            "bombs": bombs,
        },
        "room": {"type": 1, "index": room_index, "enemies_alive": len(enemies),
                 "bosses_alive": 0, "clear": clear},
        "entities": ent,
        "grid": grid,
        "doors": doors if doors is not None else [
            {"slot": 1, "grid": 7, "open": True},      # 上
            {"slot": 2, "grid": 74, "open": True},     # 右（DoorSlot 2）
            {"slot": 3, "grid": 112, "open": True},    # 下
            {"slot": 0, "grid": 60, "open": True}],    # 左
        "map": [{"grid": 50, "type": 1, "visited": 1},
                {"grid": 51, "type": 1, "visited": 0},
                {"grid": 63, "type": 1, "visited": 0}],
    }


def center(g, gx, gy):
    return g["ox"] + (gx + 0.5) * g["tw"], g["oy"] + (gy + 0.5) * g["th"]


def mech_grid(plate=(7, 4), bombable=(6, 4), tnt=()):
    """复刻实机房间 73：压力板被一圈石头围死，其中一块是炸弹石。

    bombable 里的格子用 ent=5（炸弹石），tnt 里的用 ent=12；其余围一圈
    ent=2 的石头。压力板本身可走（coll=0, ent=20）。
    """
    w, h = 15, 9
    coll = [0] * (w * h)
    ent = [0] * (w * h)

    def put(gx, gy, c, e=0):
        coll[gy * w + gx] = c
        ent[gy * w + gx] = e

    for gx in range(w):
        put(gx, 0, 4, 15)
        put(gx, h - 1, 4, 15)
    for gy in range(h):
        put(0, gy, 4, 15)
        put(w - 1, gy, 4, 15)
    for gx, gy in ((w // 2, 0), (w // 2, h - 1), (0, h // 2), (w - 1, h // 2)):
        put(gx, gy, 5, 16)                  # 门

    pgx, pgy = plate
    put(pgx, pgy, 0, 20)                    # 压力板
    for dx, dy in DIRS8:
        gx, gy = pgx + dx, pgy + dy
        if (gx, gy) in tnt:
            put(gx, gy, 3, 12)
        elif (gx, gy) in bombable:
            put(gx, gy, 3, 5)
        else:
            put(gx, gy, 3, 2)
    return {
        "w": w, "h": h, "cells": [0] * (w * h),
        "coll": coll, "ent": ent,
        "gv": 3, "ox": 20.0, "oy": 100.0, "tw": 40.0, "th": 40.0,
    }


def run_case(name, fn):
    try:
        fn()
        print(f"  ✅ {name}")
        return True
    except AssertionError as exc:
        print(f"  ❌ {name}: {exc}")
        return False


def main():
    grid = demo_grid(15, 9)
    px, py = center(grid, 7, 4)
    print(f"测试房间 {grid['w']}x{grid['h']} player=({px},{py})")
    ok = []

    def t_stationary():
        st = mk_state(grid, px, py, enemies=[{"x": px + 100, "y": py}])
        ag = TacticalAgent()
        a = ag.act(st)
        assert a.shoot == (1, 0), f"应朝右射，实际 {a.shoot}"
        aim = ag.aim_solution(st["player"], split_entities(st)[0][0],
                              (0, 0), ag.grid)
        assert aim and aim["dir"] == 2, f"瞄准解错误 {aim}"
        print(f"     动作 move={a.move} shoot={a.shoot} miss={aim['miss']:.1f}")

    def t_diagonal_gate():
        st = mk_state(grid, px, py, enemies=[{"x": px + 70, "y": py + 70}])
        ag = TacticalAgent()
        a = ag.act(st)
        assert a.shoot == (0, 0), f"斜向未对齐不应开枪，实际 {a.shoot}"
        m = _dir8_to_vec(*a.move)
        assert abs(m[1]) > 0.5, f"应该上下移动对齐 y 轴，实际 {a.move}"
        print(f"     动作 move={a.move} shoot={a.shoot}")

    def t_lead_move():
        # 敌人正右 150，向下漂 3/帧：正右直射会偏，移动下甩枪才有解
        st = mk_state(grid, px, py, enemies=[{"x": px + 150, "y": py, "vy": 3.0}])
        ag = TacticalAgent()
        a = ag.act(st)
        print(f"     动作 move={a.move} shoot={a.shoot} 目标vy=3")
        # 允许按下/右方向补偿；至少不能在不补偿的情况下开枪
        assert a.shoot == (0, 0) or a.move != (0, 0), "移动目标不应站桩直射"

    def t_wall():
        # 在玩家正右 2 格处放石头（demo grid 的石头在 (11,2) 附近，这里改造一份）
        g2 = json.loads(json.dumps(grid))
        w = g2["w"]
        for gy in range(3, 6):
            g2["coll"][gy * w + 9] = 3
            g2["cells"][gy * w + 9] = 0
        st = mk_state(g2, px, py, enemies=[{"x": px + 160, "y": py}])
        ag = TacticalAgent()
        a = ag.act(st)
        assert a.shoot == (0, 0), f"石头后面不应开枪，实际 {a.shoot}"
        print(f"     动作 move={a.move} shoot={a.shoot}")

    def t_dodge():
        bullets = [{"x": px - 80, "y": py, "vx": 8.0, "vy": 0.0}]
        st = mk_state(grid, px, py, enemies=[{"x": px + 220, "y": py}],
                      bullets=bullets)
        ag = TacticalAgent()
        a = ag.act(st)
        m = _dir8_to_vec(*a.move)
        assert abs(m[1]) > 0.7, f"右方来弹应该上下躲，实际 {a.move}"
        assert a.shoot == (0, 0), f"紧急躲弹不应开枪，实际 {a.shoot}"
        print(f"     动作 move={a.move} shoot={a.shoot}")

    def t_lob_landing():
        # 高抛弹：影子就在玩家脚下，height 从 -180 升到 -160（10/帧），
        # 约 14 帧后落到普通命中高度。二维速度是 0，只能靠落点预警发现。
        st1 = mk_state(grid, px, py, enemies=[{"x": px + 200, "y": py}],
                       bullets=[{"x": px, "y": py, "height": -180.0}])
        st1["frame"] = 10
        ag = TacticalAgent()
        ag.act(st1)
        st2 = mk_state(grid, px, py, enemies=[{"x": px + 200, "y": py}],
                       bullets=[{"x": px, "y": py, "height": -160.0}])
        st2["frame"] = 12
        a = ag.act(st2)
        assert a.move != (0, 0), "高抛弹落点就在脚下，必须移动"
        enemies, projectiles, _, hazards = split_entities(st2)
        danger_here = ag._danger_at(px, py, 8, projectiles, enemies, hazards, ag.grid, 1.0)
        danger_away = ag._danger_at(px + 120, py, 8, projectiles, enemies, hazards, ag.grid, 1.0)
        assert danger_here > danger_away + 100, \
            f"落点危险场没生效 {danger_here:.0f} vs {danger_away:.0f}"
        print(f"     动作 move={a.move} 落点危险 {danger_here:.0f} vs {danger_away:.0f}")

    def t_jump_up_warning():
        # 实机 Monstro：JumpUp 约 0.4 秒后砸向玩家当前位置，动画名是唯一信号
        st = mk_state(grid, px, py, enemies=[{
            "x": px + 150, "y": py, "hp": 500, "boss": True,
            "size": 42, "anim": "JumpUp"}])
        ag = TacticalAgent()
        a = ag.act(st)
        m = _dir8_to_vec(*a.move)
        assert a.mode == "jump_dodge", f"跳跃时应接管移动，实际 {a.mode}"
        assert a.debug.get("threat", 0) >= 0.65, f"起跳威胁没触发 {a.debug.get('threat')}"
        assert a.shoot == (0, 0), f"跳跃砸地时不该站桩开枪 {a.shoot}"
        assert a.move != (0, 0), f"起跳就该横移 {a.move}"
        assert m[0] < 0.1, f"不该往起跳的 BOSS 身上跑 {a.move}"
        print(f"     动作 move={a.move} threat={a.debug.get('threat'):.2f} "
              f"落点=({a.debug.get('jump_land')})")

    def t_jump_down_landing():
        # JumpDown：影子=Position，按速度前推；预警圈在落点上，AI 应往外走
        st = mk_state(grid, px, py, enemies=[{
            "x": px + 100, "y": py, "vx": -5.0, "hp": 500, "boss": True,
            "size": 42, "anim": "JumpDown"}])
        ag = TacticalAgent()
        a = ag.act(st)
        m = _dir8_to_vec(*a.move)
        assert a.shoot == (0, 0), f"下落砸地时不应开枪 {a.shoot}"
        assert a.move != (0, 0), "下落阶段必须离开落点"
        assert m[0] < 0.3, f"不该迎着砸下来的 BOSS 走 {a.move}"
        print(f"     动作 move={a.move} threat={a.debug.get('threat'):.2f}")

    def t_static_hazard():
        # 尖刺在 (5,6)，玩家在右边，目标格子设到尖刺左侧：A* 不该穿尖刺
        sx, sy = center(grid, 5, 6)
        st = mk_state(grid, center(grid, 8, 6)[0], center(grid, 8, 6)[1], clear=True)
        ag = TacticalAgent()
        ag.update_state(st)
        # 直接检查 A* 路径不经过尖刺格
        hz = room_hazard_idx(grid, 5, 6)
        path = ag.astar(ag.grid.idx_of(sx + 120, sy), ag.grid.idx_of(sx - 20, sy))
        assert path is not None, "A* 无路径"
        assert hz not in path, "A* 穿过了尖刺"
        print(f"     路径长度 {len(path)}，未踩尖刺")

    def t_pickup_then_door():
        st = mk_state(grid, px, py, clear=True,
                      pickups=[{"x": px + 60, "y": py}])
        ag = TacticalAgent()
        a = ag.act(st)
        assert a.mode == "pickup", f"清房应先去拾取，实际 {a.mode}"
        # 走到掉落物旁边会等游戏拾取
        st2 = mk_state(grid, px + 58, py, clear=True,
                       pickups=[{"x": px + 60, "y": py}])
        a2 = ag.act(st2)
        assert a2.mode == "pickup", f"站在掉落物上应等待，实际 {a2.mode}"
        print(f"     拾取模式 {a.mode}/{a2.mode}")

    def t_door_explore():
        st = mk_state(grid, px, py, clear=True)
        ag = TacticalAgent()
        a = ag.act(st)
        assert a.mode == "door" and a.move != (0, 0), f"清房后应去开门，实际 {a.mode} {a.move}"
        print(f"     开门动作 {a.move} door")

    def t_door_push_east():
        # 站在右侧门格里：DoorSlot 2 = 右，必须往右推，不能往下（旧版把 slot
        # 顺序当成上右下左，结果在右门一直往墙里推，永远换不了房）
        dg = 74
        gx, gy = dg % grid["w"], dg // grid["w"]
        wx, wy = center(grid, gx, gy)
        st = mk_state(grid, wx, wy, clear=True,
                      doors=[{"slot": 2, "grid": dg, "open": True}])
        ag = TacticalAgent()
        a = ag.act(st)
        assert a.mode == "door", f"应在开门，实际 {a.mode}"
        assert _dir8_to_vec(*a.move)[0] > 0.7, f"右门应向右推，实际 {a.move}"
        print(f"     DoorSlot2 推门动作 {a.move}")

    def t_legacy_grid_normalize():
        # 旧模组把可走区域宽高除以含墙的 w/h：15x9 房间会给出 tw=34.67,ox=60。
        # 用实机采样验证：玩家(536.4,329.6) 游戏内 grid_idx=87，即 (12,5)。
        old = {
            "w": 15, "h": 9,
            "cells": [0] * 135, "coll": [0] * 135, "ent": [0] * 135,
            "ox": 60.0, "oy": 140.0,
            "tw": 520.0 / 15, "th": 280.0 / 9,
        }
        rg = RoomGrid(old)
        assert abs(rg.tw - 40.0) < 1e-6, f"格宽应还原为40，实际 {rg.tw}"
        assert abs(rg.ox - 20.0) < 1e-6, f"原点应还原为20，实际 {rg.ox}"
        assert rg.cell_of(536.4, 329.6) == (12, 5), f"实际 {rg.cell_of(536.4, 329.6)}"
        # 新版模组（gv=2）不能被二次修正
        new = dict(old, gv=2, ox=20.0, oy=100.0, tw=40.0, th=40.0)
        rg2 = RoomGrid(new)
        assert (rg2.tw, rg2.ox) == (40.0, 20.0)
        assert rg2.cell_of(536.4, 329.6) == (12, 5)
        print(f"     旧网格还原 {rg.tw}x{rg.th} ox={rg.ox}；新网格不变")

    def t_narrow_room_grid():
        # 实机 1x1 窄房：15x9 网格，可走区域只占第 3~5 行（上下各空两行）。
        # 玩家 (390.4,329.5) 游戏内 grid_idx=84，即 (9,5)。
        w, h = 15, 9
        coll = [0] * (w * h)

        def putrow(y, vals):
            for x, c in enumerate(vals):
                coll[y * w + x] = c

        putrow(2, [4] * 15)
        putrow(3, [4, 0, 0, 0, 0, 0, 3, 0, 3, 0, 0, 0, 0, 0, 4])
        putrow(4, [5, 0, 0, 0, 0, 0, 0, 3, 0, 0, 0, 0, 0, 0, 5])
        putrow(5, [4, 0, 3, 0, 0, 0, 0, 0, 0, 0, 0, 0, 3, 0, 4])
        putrow(6, [4] * 15)
        base = {"w": w, "h": h, "cells": [0] * (w * h),
                "coll": coll, "ent": [0] * (w * h)}
        # 新版模组（gv=3）：直接给出正确值
        rg = RoomGrid(dict(base, gv=3, ox=20.0, oy=100.0, tw=40.0, th=40.0))
        assert rg.cell_of(390.4, 329.5) == (9, 5), f"gv3 映射 {rg.cell_of(390.4, 329.5)}"
        # gv=2：格宽对，但原点按“可走区域从第1行开始”外推，窄房少了 2 行
        rg2 = RoomGrid(dict(base, gv=2, ox=20.0, oy=180.0, tw=40.0, th=40.0))
        assert abs(rg2.oy - 100.0) < 1e-6, f"gv2 原点还原 {rg2.oy}"
        assert rg2.cell_of(390.4, 329.5) == (9, 5), f"gv2 映射 {rg2.cell_of(390.4, 329.5)}"
        # 更老的 gv=1：tw/th 也是错的
        rg1 = RoomGrid(dict(base, ox=60.0, oy=220.0,
                           tw=520.0 / 15, th=120.0 / 9))
        assert abs(rg1.tw - 40.0) < 1e-6 and abs(rg1.th - 40.0) < 1e-6
        assert rg1.cell_of(390.4, 329.5) == (9, 5), f"gv1 映射 {rg1.cell_of(390.4, 329.5)}"
        print(f"     窄房映射 gv3/gv2/gv1 均为 (9,5)，gv2 oy={rg2.oy}")

    def t_door_no_pingpong():
        # 实机踩到过：84 的右边是 85，85 的正上方 72 还没探索，但 85 的上门
        # 是锁着的。按格子相邻做 BFS 会得出 84→85→72，AI 就在 84/85 之间弹。
        # 正确做法是只看本房间真实存在的门：84 的上门通往未探索的 71。
        ag = TacticalAgent()
        ag.room_index = 84
        ag.doors = [{"slot": 0, "grid": 60, "open": True},    # 左 -> 83
                    {"slot": 1, "grid": 7, "open": True},     # 上 -> 71
                    {"slot": 2, "grid": 74, "open": True}]    # 右 -> 85
        ag.map_rooms = [
            {"grid": 84, "type": 1, "visited": 1},
            {"grid": 83, "type": 10, "visited": 1},
            {"grid": 85, "type": 4, "visited": 58},
            {"grid": 71, "type": 1, "visited": 0},
            {"grid": 72, "type": 7, "visited": 0},
        ]
        ag.floor_edges = {}
        d = ag.choose_door()
        assert d and d["slot"] == 1, f"应走上门去 71，实际 {d}"

    def t_door_route_boss_edges():
        # 邻居都已访问，Boss 在 71 的正上方 70：只能靠走过的连通图绕过去
        ag = TacticalAgent()
        ag.room_index = 84
        ag.doors = [{"slot": 0, "grid": 60, "open": True},
                    {"slot": 1, "grid": 7, "open": True},
                    {"slot": 2, "grid": 74, "open": True}]
        ag.map_rooms = [
            {"grid": 84, "type": 1, "visited": 3},
            {"grid": 83, "type": 1, "visited": 2},
            {"grid": 85, "type": 4, "visited": 2},
            {"grid": 71, "type": 1, "visited": 1},
            {"grid": 70, "type": 5, "visited": 0},
        ]
        ag.floor_edges = {84: {1: 71}, 71: {3: 84, 1: 70}, 70: {3: 71}}
        d = ag.choose_door()
        assert d and d["slot"] == 1, f"应沿已知连通图上走，实际 {d}"

    def t_door_frontier_route():
        # 84 的门都走过了，但 83 还有一扇没走过的门；Boss 又不在已知图上。
        # 应该沿已知连通图回 83 补图，而不是在 84/85 之间来回。
        ag = TacticalAgent()
        ag.room_index = 84
        ag.visited_rooms = {84, 83, 85, 71}
        ag.doors = [{"slot": 0, "grid": 60, "open": True},    # 左 -> 83
                    {"slot": 2, "grid": 74, "open": True}]    # 右 -> 85
        ag.map_rooms = [
            {"grid": 84, "type": 1, "visited": 4},
            {"grid": 83, "type": 1, "visited": 2},
            {"grid": 85, "type": 4, "visited": 2},
            {"grid": 71, "type": 1, "visited": 1},
            {"grid": 30, "type": 5, "visited": 0},
        ]
        ag.floor_edges = {84: {0: 83, 2: 85}, 83: {2: 84}, 85: {0: 84}}
        ag.seen_door_slots = {84: {0, 2}, 83: {0, 2}, 85: {0}}
        d = ag.choose_door()
        assert d and d["slot"] == 0, f"应回 83 补连通图，实际 {d}"

    def t_second_door_slot():
        # 大房间同一侧有两扇门：slot 4/5/6/7 的方向和反向边
        for slot in (4, 5, 6, 7):
            assert ROOM_DELTA[slot] == ROOM_DELTA[slot - 4], f"slot{slot} 方向错误"
        dg = 74
        gx, gy = dg % grid["w"], dg // grid["w"]
        wx, wy = center(grid, gx, gy)
        st = mk_state(grid, wx, wy, clear=True,
                      doors=[{"slot": 6, "grid": dg, "open": True}])
        ag = TacticalAgent()
        a = ag.act(st)
        assert a.mode == "door" and _dir8_to_vec(*a.move)[0] > 0.7, \
            f"slot6 应往右推，实际 {a.move}"
        ag.pushed_slot, ag.pushed_from = 6, 84
        ag.room_index = 84
        ag.update_state({"room": {"index": 85}, "player": {}})
        assert ag.floor_edges[84][6] == 85 and ag.floor_edges[85][4] == 84, \
            f"连通边反向错误 {ag.floor_edges}"
        print("     slot6 往右推，反向边 6<->4")

    def t_keep_distance():
        # 怪物贴到 60px：应该主动往后退，而不是继续贴脸
        st = mk_state(grid, px, py, enemies=[{"x": px + 60, "y": py}])
        ag = TacticalAgent()
        a = ag.act(st)
        m = _dir8_to_vec(*a.move)
        assert m[0] < -0.5, f"贴脸时应后撤拉开距离，实际 {a.move}"
        # 远距离（180px）不需要后撤
        st2 = mk_state(grid, px, py, enemies=[{"x": px + 180, "y": py}])
        a2 = TacticalAgent().act(st2)
        m2 = _dir8_to_vec(*a2.move)
        assert m2[0] > -0.5, f"中距离不应一直后退，实际 {a2.move}"
        print(f"     近身 move={a.move}，180px move={a2.move}")

    def t_crowd_escape():
        # 三只怪从不同方向围上来：不能继续站桩对枪，也不能朝最近的那只走
        st = mk_state(grid, px, py, enemies=[
            {"x": px - 60, "y": py, "size": 13},
            {"x": px + 60, "y": py, "size": 13},
            {"x": px, "y": py - 70, "size": 13}])
        ag = TacticalAgent()
        a = ag.act(st)
        assert a.mode == "crowd_escape", f"被三面包围应脱离，实际 {a.mode}"
        assert a.move != (0, 0), "脱离时不能站着"
        ux, uy = _dir8_to_vec(*a.move)
        for e in st["entities"][1:]:
            dx, dy = e["x"] - px, e["y"] - py
            d = math.hypot(dx, dy)
            dot = (ux * dx + uy * dy) / d
            assert dot < 0.6, f"脱离方向正对敌人({e['x']},{e['y']})，实际 {a.move}"
        assert ag._escape_cost is not None, "撤离应带敌人代价图"
        print(f"     三面包围 -> {a.mode} {a.move} goal={a.debug.get('goal')}")

    def t_two_chasers_escape():
        # 左右各一只正在冲过来：还没贴脸也要提前拉开，不能等进 90px
        st = mk_state(grid, px, py, enemies=[
            {"x": px - 150, "y": py, "vx": 2.5},
            {"x": px + 150, "y": py, "vx": -2.5}])
        ag = TacticalAgent()
        a = ag.act(st)
        assert a.mode == "crowd_escape", f"两只冲锋怪应提前脱离，实际 {a.mode}"
        assert a.debug.get("chasing") == 2, f"追击计数错误 {a.debug}"
        print(f"     两怪冲锋 -> {a.mode} {a.move} goal={a.debug.get('goal')}")

    def t_escape_path_avoids_enemy():
        # 目标格在怪的正后方：带代价的寻路不能直接从怪身上穿过去
        gopen = json.loads(json.dumps(grid))
        for i in range(gopen["w"] * gopen["h"]):
            gopen["cells"][i] = 1
            gopen["coll"][i] = 0
            gopen["ent"][i] = 0
        ex, ey = center(gopen, 7, 4)
        st = mk_state(gopen, *center(gopen, 3, 4), clear=True)
        ag = TacticalAgent()
        ag.update_state(st)
        pidx = ag.grid.idx_of(*center(gopen, 3, 4))
        goal = ag.grid.idx_of(*center(gopen, 11, 4))
        n = gopen["w"] * gopen["h"]
        cost = [0.0] * n
        for i in range(n):
            cx, cy = ag.grid.center(i)
            d = math.hypot(cx - ex, cy - ey)
            if d < 150:
                cost[i] = ((150 - d) / 150) ** 2 * 12 + (40.0 if d < 60 else 0.0)
        path = ag.astar(pidx, goal, extra_cost=cost)
        assert path, "带代价寻路无解"
        dmin = min(math.hypot(ag.grid.center(i)[0] - ex,
                              ag.grid.center(i)[1] - ey) for i in path)
        assert dmin > 45.0, f"撤离路径穿怪，最近只有 {dmin:.0f}px"
        print(f"     绕怪路径 {len(path)} 格，最近 {dmin:.0f}px")

    def t_corner_steer():
        # 紧贴尖刺右下角：朝正上方会被拐角挡住，局部转向必须选一个能走的方向
        st = mk_state(grid, -31.5, 145.5, clear=True)
        ag = TacticalAgent()
        ag.update_state(st)
        want = (0, -1)
        assert not ag.grid.grid_segment_clear(
            -31.5, 145.5, -31.5, 145.5 - 12, radius=10), "测试前提：正上方应被挡住"
        mv = ag._steer(st["player"], want, -20, 120, look=12)
        assert mv != want, f"顶住拐角还朝正上走：{mv}"
        ux, uy = _dir8_to_vec(*mv)
        assert ag.grid.grid_segment_clear(
            -31.5, 145.5, -31.5 + ux * 12, 145.5 + uy * 12, radius=10), \
            f"修正后的方向仍然撞墙：{mv}"
        print(f"     墙角修正 {want} -> {mv}")

    def t_plate_reachable():
        g = json.loads(json.dumps(grid))
        pgx, pgy = 7, 4
        g["ent"][pgy * g["w"] + pgx] = 20
        x, y = center(g, 4, 4)
        ag = TacticalAgent()
        a = ag.act(mk_state(g, x, y, clear=False))
        assert a.mode == "plate", f"有可达压力板应去踩，实际 {a.mode}"
        assert a.move != (0, 0), f"应朝压力板移动，实际 {a.move}"
        # 站上压力板后原地等清房
        bx, by = center(g, pgx, pgy)
        a2 = ag.act(mk_state(g, bx, by, clear=False))
        assert a2.mode == "plate" and a2.move == (0, 0), \
            f"站在板上应原地等，实际 {a2.mode} {a2.move}"
        print(f"     走板 move={a.move}，站板 {a2.mode}/{a2.move}")

    def t_bomb_open_plate():
        g = mech_grid()
        x, y = center(g, 3, 4)
        ag = TacticalAgent()
        a = ag.act(mk_state(g, x, y, clear=False, bombs=3))
        assert a.mode == "bomb", f"压力板不可达应去炸石头，实际 {a.mode}"
        assert ag.bomb_target == 4 * g["w"] + 6, \
            f"应选 (6,4) 炸弹石，实际 {ag.bomb_target}"
        # 走到炸点后放下炸弹并往安全处撤
        bx, by = center(g, 5, 4)
        a2 = ag.act(mk_state(g, bx, by, clear=False, bombs=2))
        assert a2.bomb, f"到炸点应放炸弹，实际 {a2.mode} bomb={a2.bomb}"
        assert a2.move != (0, 0), "放完炸弹应撤离"
        # 等爆炸期间不能重复放弹
        a3 = ag.act(mk_state(g, bx, by, clear=False, bombs=2))
        assert not a3.bomb and a3.mode == "bomb", \
            f"等爆炸期间不应再放弹，实际 {a3.mode} bomb={a3.bomb}"
        print(f"     规划 {a.mode} -> 放弹 {a2.bomb} 撤离 {a2.move} -> 等待 {a3.mode}")

    def t_bomb_no_stock():
        g = mech_grid()
        x, y = center(g, 3, 4)
        a = TacticalAgent().act(mk_state(g, x, y, clear=False, bombs=0))
        assert a.mode == "plate-blocked", f"没炸弹应停在 plate-blocked，实际 {a.mode}"
        assert not a.bomb, "没炸弹不能放"
        print(f"     无库存 {a.mode}")

    def t_bomb_prefers_rock_over_tnt():
        g = mech_grid(bombable={(6, 4), (8, 4)}, tnt={(8, 4)})
        x, y = center(g, 3, 4)
        ag = TacticalAgent()
        a = ag.act(mk_state(g, x, y, clear=False, bombs=3))
        assert a.mode == "bomb", f"应规划炸路，实际 {a.mode}"
        assert ag.bomb_target == 4 * g["w"] + 6, \
            f"应优先普通炸弹石而非 TNT，实际 {ag.bomb_target}"
        print(f"     TNT 在右，选择炸左侧石头 {ag.bomb_target}")

    def t_bomb_escape_avoid_tnt():
        # 炸点左侧有 TNT：撤离点不能贴着它
        g = mech_grid(tnt={(5, 3), (4, 4), (5, 5)})
        x, y = center(g, 5, 4)
        ag = TacticalAgent()
        ag.update_state(mk_state(g, x, y, clear=False, bombs=2))
        esc = ag._bomb_escape_cell(mk_state(g, x, y)["player"], 4 * g["w"] + 6)
        assert esc is not None, "应能找到撤离点"
        ex, ey = ag.grid.center(esc)
        bx, by = ag.grid.center(4 * g["w"] + 6)
        assert math.hypot(ex - bx, ey - by) >= 3.0 * 40 - 1e-6, \
            f"撤离点太近 {esc}"
        assert not ag.grid.near_tnt(esc, 2.2), f"撤离点贴着 TNT {esc}"
        print(f"     撤离格 {esc} 距爆点 {math.hypot(ex - bx, ey - by):.0f}px")

    def t_bomb_full_sequence():
        # 完整走一遍：规划→放弹→撤离→等爆炸→石头消失→踩板→清房
        g = mech_grid()
        ag = TacticalAgent()
        gx, gy = center(g, 3, 4)

        def mk(px, py, gg=g, bombs=2):
            return mk_state(gg, px, py, clear=False, bombs=bombs)

        a = ag.act(mk(gx, gy, bombs=3))
        assert a.mode == "bomb" and ag.bomb_spot is not None
        bx, by = center(g, 5, 4)
        a2 = ag.act(mk(bx, by, bombs=3))
        assert a2.bomb and ag.bomb_escape is not None
        ex, ey = ag.grid.center(ag.bomb_escape)
        for _ in range(120):
            ag.act(mk(ex, ey))
        assert ag.bomb_target is None and not ag.bomb_dropped, \
            f"爆炸后没清账 target={ag.bomb_target}"

        g2 = json.loads(json.dumps(g))
        g2["coll"][4 * g2["w"] + 6] = 0
        g2["ent"][4 * g2["w"] + 6] = 0
        a3 = ag.act(mk(ex, ey, gg=g2))
        assert a3.mode == "plate", f"路炸开后应去踩板，实际 {a3.mode}"
        pxp, pyp = center(g2, 7, 4)
        a4 = ag.act(mk(pxp, pyp, gg=g2))
        assert a4.mode == "plate" and a4.move == (0, 0), \
            f"站上板应等清房，实际 {a4.mode} {a4.move}"
        print(f"     规划→放弹→撤离→清账→踩板 全流程通过")

    def t_explore_via_seen_door():
        # 实机地图：84 的下方是 97，97 的下门通往未探索的 110。
        # 即使 97→110 还没走过，只要在 97 见过下门开着，就应该规划过去。
        ag = TacticalAgent()
        ag.room_index = 84
        ag.doors = [{"slot": 0, "grid": 60, "open": True},
                    {"slot": 2, "grid": 74, "open": True},
                    {"slot": 3, "grid": 127, "open": True}]
        ag.map_rooms = [
            {"grid": 84, "type": 1, "visited": 17},
            {"grid": 83, "type": 1, "visited": 18},
            {"grid": 85, "type": 1, "visited": 2},
            {"grid": 97, "type": 1, "visited": 1},
            {"grid": 110, "type": 1, "visited": 0},
            {"grid": 112, "type": 5, "visited": 0},
        ]
        ag.visited_rooms = {84, 83, 85, 97}
        ag.seen_door_slots = {84: {0, 2, 3}, 97: {1, 3}, 85: {0, 3}}
        ag.floor_edges = {84: {0: 83, 2: 85, 3: 97},
                          83: {2: 84}, 85: {0: 84}, 97: {1: 84}}
        step = ag._nearest_unvisited_step()
        assert step == 97, f"应经 97 去 110，实际第一跳 {step}"
        d = ag.choose_door()
        assert d and d["slot"] == 3, f"应走 84 下门，实际 {d}"

        # 没见过的门不能凭空推断（85 的上门锁着，72 不能从 85 进）
        edges = ag._known_edges()
        assert 1 not in edges.get(85, {}), f"85 上锁的门被误当成通路: {edges.get(85)}"
        print(f"     第一跳 97 -> 110，84 选下门；未见的 85 上门未连通")

    def t_no_pingpong_prev_room():
        # 两个邻居都去过、都没有其它未探索出口：不要立刻回到刚出来的房间
        ag = TacticalAgent()
        ag.room_index = 84
        ag.prev_room = 83
        ag.doors = [{"slot": 0, "grid": 60, "open": True},
                    {"slot": 2, "grid": 74, "open": True}]
        ag.map_rooms = [
            {"grid": 84, "type": 1, "visited": 8},
            {"grid": 83, "type": 1, "visited": 1},   # 刚出来、显得“去得少”
            {"grid": 85, "type": 1, "visited": 9},
        ]
        ag.visited_rooms = {84, 83, 85}
        ag.seen_door_slots = {84: {0, 2}, 83: {2}, 85: {0}}
        ag.floor_edges = {84: {0: 83, 2: 85}, 83: {2: 84}, 85: {0: 84}}
        d = ag.choose_door()
        assert d and d["slot"] == 2, f"应避开刚出来的 83，实际 {d}"
        print(f"     prev_room=83 时改走 85")

    ok.append(run_case("静止敌人：开火且方向正确", t_stationary))
    ok.append(run_case("斜向敌人：未对齐不开火", t_diagonal_gate))
    ok.append(run_case("移动敌人：不能站桩直射", t_lead_move))
    ok.append(run_case("石头挡视线：不开火", t_wall))
    ok.append(run_case("弹幕来袭：横移躲避且停火", t_dodge))
    ok.append(run_case("高抛弹幕：按落点躲避", t_lob_landing))
    ok.append(run_case("BOSS 起跳：提前离开落点", t_jump_up_warning))
    ok.append(run_case("BOSS 下落：不迎着落点走", t_jump_down_landing))
    ok.append(run_case("A* 绕开尖刺", t_static_hazard))
    ok.append(run_case("清房先拾取再开门", t_pickup_then_door))
    ok.append(run_case("清房后按地图开门", t_door_explore))
    ok.append(run_case("右门(DoorSlot2)往右推", t_door_push_east))
    ok.append(run_case("旧网格元数据自动还原", t_legacy_grid_normalize))
    ok.append(run_case("窄房网格原点还原", t_narrow_room_grid))
    ok.append(run_case("只走真实门，不在相邻房之间弹", t_door_no_pingpong))
    ok.append(run_case("沿已知连通图去 Boss 房", t_door_route_boss_edges))
    ok.append(run_case("回访有未走门的房间补图", t_door_frontier_route))
    ok.append(run_case("大房间第二扇门 slot4-7", t_second_door_slot))
    ok.append(run_case("近身主动拉开距离", t_keep_distance))
    ok.append(run_case("三面包围：进入脱离模式", t_crowd_escape))
    ok.append(run_case("两怪冲锋：提前脱离", t_two_chasers_escape))
    ok.append(run_case("撤离寻路不穿怪", t_escape_path_avoids_enemy))
    ok.append(run_case("墙角局部转向", t_corner_steer))
    ok.append(run_case("可达压力板：走过去踩", t_plate_reachable))
    ok.append(run_case("压力板被围：炸弹石开路", t_bomb_open_plate))
    ok.append(run_case("没炸弹时报告 plate-blocked", t_bomb_no_stock))
    ok.append(run_case("炸弹石优先于 TNT", t_bomb_prefers_rock_over_tnt))
    ok.append(run_case("撤离点避开 TNT", t_bomb_escape_avoid_tnt))
    ok.append(run_case("机关房炸弹开路全流程", t_bomb_full_sequence))
    ok.append(run_case("用见过的开门的门探索未访问房", t_explore_via_seen_door))
    ok.append(run_case("清房后不立刻折返上一间", t_no_pingpong_prev_room))

    # 真实帧回放：每帧只做决策，检查不抛异常 + 统计开火倾向
    print("\n真实帧回放 (frames.json) ...")
    try:
        frames = json.load(open("frames.json", encoding="utf-8"))
        ag = TacticalAgent()
        shots = 0
        for s in frames[-60:]:
            if "grid" not in s:
                continue
            a = ag.act(s)
            if a.shoot != (0, 0):
                shots += 1
        print(f"  60 帧中开火 {shots} 帧，无异常")
        ok.append(True)
    except Exception as exc:
        print(f"  ❌ 回放异常: {exc}")
        ok.append(False)

    print(f"\n结果：{sum(ok)}/{len(ok)} 通过")
    return 0 if all(ok) else 1


def room_hazard_idx(grid, gx, gy):
    return gy * grid["w"] + gx


if __name__ == "__main__":
    sys.exit(main())
