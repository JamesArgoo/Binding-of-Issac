# -*- coding: utf-8 -*-
"""test_tactical_agent.py —— 不用开游戏，验证战术 AI 的瞄准/门控/躲弹/绕障。

运行：
    python test_tactical_agent.py
"""
import json
import math
import sys

try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

import isaac_data as D
from tactical_agent import (Config, TacticalAgent, RoomGrid, DIRS8,
                            SHOOT_DIRS, ROOM_DELTA, _dir8_to_vec)


def mk_state(grid, px, py, enemies=(), bullets=(), pickups=(), clear=False,
             room_index=50, player_v=(0.0, 0.0), hp=6.0, doors=None):
    ent = [{
        "id": 1, "type": 1, "variant": 0, "x": px, "y": py,
        "vx": player_v[0], "vy": player_v[1], "hp": hp, "enemy": False,
        "boss": False, "spawner_type": 0, "size": 10.0, "cdmg": 0.0,
    }]
    for i, e in enumerate(enemies):
        ent.append({
            "id": 100 + i, "type": int(e.get("type", 10)), "variant": 0,
            "x": e["x"], "y": e["y"], "vx": e.get("vx", 0.0), "vy": e.get("vy", 0.0),
            "hp": e.get("hp", 10.0), "enemy": True, "boss": False,
            "spawner_type": 0, "size": e.get("size", 13.0), "cdmg": e.get("cdmg", 1.0),
        })
    for i, b in enumerate(bullets):
        ent.append({
            "id": 300 + i, "type": 2, "variant": 0,
            "x": b["x"], "y": b["y"], "vx": b.get("vx", 0.0), "vy": b.get("vy", 0.0),
            "hp": 0.0, "enemy": False, "boss": False,
            "spawner_type": 0, "size": b.get("size", 5.0), "cdmg": b.get("cdmg", 0.0),
        })
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


def run_case(name, fn):
    try:
        fn()
        print(f"  ✅ {name}")
        return True
    except AssertionError as exc:
        print(f"  ❌ {name}: {exc}")
        return False


def main():
    grid = D.demo_grid(15, 9)
    px, py = center(grid, 7, 4)
    print(f"测试房间 {grid['w']}x{grid['h']} player=({px},{py})")
    ok = []

    def t_stationary():
        st = mk_state(grid, px, py, enemies=[{"x": px + 100, "y": py}])
        ag = TacticalAgent()
        a = ag.act(st)
        assert a.shoot == (1, 0), f"应朝右射，实际 {a.shoot}"
        aim = ag.aim_solution(st["player"], __import__("tactical_agent").split_entities(st)[0][0], (0, 0), ag.grid)
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

    ok.append(run_case("静止敌人：开火且方向正确", t_stationary))
    ok.append(run_case("斜向敌人：未对齐不开火", t_diagonal_gate))
    ok.append(run_case("移动敌人：不能站桩直射", t_lead_move))
    ok.append(run_case("石头挡视线：不开火", t_wall))
    ok.append(run_case("弹幕来袭：横移躲避且停火", t_dodge))
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
