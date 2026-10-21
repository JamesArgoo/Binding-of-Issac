# -*- coding: utf-8 -*-
"""play_rule.py —— 让规则AI（ai_basic_4）实际打给你看

用 PostMessage 注入按键，**不需要游戏窗口有焦点**，所以你可以随便切窗口看。

用法：
    python play_rule.py                # 接管已开着的游戏
    python play_rule.py --launch 1     # 自己启动 1 个实例
"""
import argparse
import sys
from collections import deque
import time

try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

import ai_basic_4 as teacher
import multi_instance as mi
from isaac_env import normalize_grid


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--launch", type=int, default=0)
    ap.add_argument("--run-timeout", type=float, default=600.0)
    args = ap.parse_args()

    if args.launch:
        mgr = mi.InstanceManager(n=args.launch)
        mgr.launch_all(wait_runs=True, run_timeout=args.run_timeout)
    else:
        mgr = mi.InstanceManager(n=1)
        mgr.attach()
        if not mgr.wait_connections(args.run_timeout):
            print("没有实例在对局里 —— 请先在游戏窗口里进入一局")
            mgr.stop_all()
            return 1
        mgr.resolve_mapping()

    inst = mgr.instances[0]
    ch = mgr.bridge.channels[inst.port]
    kc = inst.controller

    print(f"\n规则AI 开始 (实例{inst.index} port={inst.port} hwnd={inst.hwnd})")
    print("它用的策略：8方向移动 + 4方向射击 + 甩枪补偿 + 前瞻躲弹")
    print("按 Ctrl+C 停止\n")

    grid = None
    map_cache = None   # 模组每 30 帧才发一次 map，必须自己缓存住
    recent_rooms = deque(maxlen=3)   # 记录**来源房间**（防 A-B-A-B 打转）
    last_door = None
    prev_room = None
    prev_door = None
    n = 0
    stuck = 0
    last_pos = None
    t0 = time.time()

    try:
        while True:
            s = ch.state()
            if not s or "player" not in s:
                time.sleep(0.02)
                continue
            if "grid" in s:
                grid = normalize_grid(s["grid"])
            # 地图（房间 visited 计数）每 30 帧才来一次；不缓存的话
            # _go_to_door 里"优先去未探索房间"永远拿不到数据 -> 两个房间无限横跳
            if "map" in s:
                map_cache = s["map"]
            elif map_cache is not None:
                s["map"] = map_cache

            # grid 还没到就算不了门，跳过本帧（首帧常见）
            if grid is None:
                time.sleep(0.005)
                continue

            room_idx = s["room"]["index"]

            # 【修复1】先算"本帧（新房间）的最近门"
            cur_door = teacher.nearest_open_door_grid(s, grid)

            # 【修复2】原来这里缺 prev_room = room_idx，导致 prev_room 永远是 None，
            #         整个换房间分支从来没执行过 —— last_door 恒为 None、
            #         recent_rooms 恒为空，所有门选择逻辑全是死代码。
            if prev_room is not None and room_idx != prev_room:
                # 刚跨门那一帧人紧贴入口门，cur_door 就是入口门 ✅
                last_door = cur_door
                # 【关键】append 的是**来源房间**，不是刚进入的房间。
                # decide 里 recent 的语义是"要避开的房间"（通往它们的门 +100 分惩罚）。
                # 原来 append(room_idx) 放的是"我现在所在的房间"，没有任何门通向它，
                # 这条逻辑等于空转。
                recent_rooms.append(prev_room)
                # 换房后地图缓存必然陈旧（map 每 30 帧才发一次），先清掉，
                # 靠 last_door + recent 的惩罚顶过这段空窗期
                map_cache = None
                p0 = s["player"]
                print(f"[换房] {prev_room} -> {room_idx}  "
                      f"player=({p0['x']:.0f},{p0['y']:.0f})  cur_door={cur_door}  "
                      f"来源队列={list(recent_rooms)}")
            prev_room = room_idx
            prev_door = cur_door

            mx, my, ax, ay = teacher.decide(s, grid, last_door, recent=recent_rooms)
            kc.set_move(mx, my)
            kc.set_shoot(ax, ay)

            # 卡住检测：位置几乎不变但仍在按键
            p = s["player"]
            if last_pos is not None:
                moved = abs(p["x"] - last_pos[0]) + abs(p["y"] - last_pos[1])
                if moved < 0.5 and (mx or my):
                    stuck += 1
                else:
                    stuck = 0
            last_pos = (p["x"], p["y"])

            n += 1
            if n % 60 == 0:
                el = time.time() - t0
                print(f"[{el:5.0f}s] 移动=({mx:+d},{my:+d}) 瞄准=({ax:+d},{ay:+d}) "
                      f"HP={p['hp']} 房间={room_idx} "
                      f"{'⚠️ 卡住 %d 帧' % stuck if stuck > 15 else ''}")
            time.sleep(1 / 60.0)
    except KeyboardInterrupt:
        print("\n停止")
    finally:
        kc.release_all()
        mgr.stop_all()
    return 0


if __name__ == "__main__":
    sys.exit(main())
