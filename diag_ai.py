# -*- coding: utf-8 -*-
"""诊断：看规则AI当前面对的真实状态，以及它为什么做出某个决策。

用法：python diag_ai.py
"""
import sys

try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

import ai_basic_4 as teacher
import multi_instance as mi
from isaac_env import normalize_grid


def sample_with_grid(mgr, index, timeout=6.0):
    """连续采样，直到拿到带 grid 的一帧（模组每 15 帧才发一次）。

    返回 (最后一帧状态, 采到的帧数, 带 grid 的帧数)。
    """
    import time
    t0 = time.time()
    seen = 0
    with_grid = 0
    last_with_grid = None
    last = None
    while time.time() - t0 < timeout:
        s = mgr.state(index)
        if s and "player" in s:
            seen += 1
            if "grid" in s:
                with_grid += 1
                normalize_grid(s["grid"])   # 按玩家口径重算可走性（门是 coll=5）
                last_with_grid = s
            last = s
        if with_grid >= 2:
            break
        time.sleep(0.05)
    return (last_with_grid or last), seen, with_grid


def main():
    mgr = mi.InstanceManager(n=1)
    mgr.attach(verbose=False)
    if not mgr.wait_connections(90, verbose=False):
        print("没有实例在对局里")
        return 1

    for inst in mgr.instances:
        s, seen, with_grid = sample_with_grid(mgr, inst.index)
        if not s or "player" not in s:
            print(f"实例{inst.index}: 无有效状态")
            continue

        p = s["player"]
        g = s.get("grid")
        enemies = [e for e in s["entities"] if e.get("enemy") or e.get("boss")]
        doors = s.get("doors", [])
        n_open = sum(1 for d in doors if d.get("open"))

        print(f"\n===== 实例{inst.index}  port={inst.port}  hwnd={inst.hwnd} =====")
        print(f"采样 {seen} 帧，带 grid 的 {with_grid} 帧 "
              f"(模组每 15 帧发一次，比例 {with_grid / max(seen, 1):.2f})")
        print(f"玩家 ({p['x']:.0f},{p['y']:.0f})  grid_idx={p['grid_idx']}  "
              f"HP={p['hp']}/{p['max_hp']}  魂心={p['soul_hp']}")
        print(f"房间 type={s['room']['type']}  clear={s['room']['clear']}  "
              f"敌人={len(enemies)}  门={len(doors)}(开{n_open})")

        if g is None:
            print("!! 一直没收到 grid —— 模组的 build_grid 可能在报错")
            continue
        print(f"grid: {g['w']}x{g['h']}  ox={g['ox']:.0f} oy={g['oy']:.0f} "
              f"tw={g['tw']:.1f} th={g['th']:.1f}")
        walkable = sum(1 for c in g["cells"] if c == 1)
        print(f"可行走格 {walkable}/{g['w'] * g['h']}")

        if enemies:
            ne = min(enemies, key=lambda e: (e["x"] - p["x"]) ** 2 + (e["y"] - p["y"]) ** 2)
            print(f"最近敌人 ({ne['x']:.0f},{ne['y']:.0f}) hp={ne['hp']} "
                  f"type={ne['type']} enemy={ne.get('enemy')} boss={ne.get('boss')}")
        else:
            print("没有敌人 → 老师应该走去开门")

        d = teacher.decide(s, g, None)
        print(f"老师决策: move=({d[0]},{d[1]})  aim=({d[2]},{d[3]})")

        if n_open:
            w = g["w"]
            od = [x for x in doors if x.get("open")]
            best = min(od, key=lambda x: abs(x["grid"] % w - p["grid_idx"] % w)
                       + abs(x["grid"] // w - p["grid_idx"] // w))
            path = teacher.astar(g, p["grid_idx"], best["grid"])
            print(f"到最近开门(grid {best['grid']})的路径: "
                  f"{'无路!' if not path else str(len(path)) + ' 格'}")
            print(f"  玩家格可走? {teacher.cell_walkable(p['grid_idx'] % w, p['grid_idx'] // w, g)}"
                  f"  门格可走? {teacher.cell_walkable(best['grid'] % w, best['grid'] // w, g)}")
            # 关键：直接看门格的碰撞类别和网格实体类型
            gi = best["grid"]
            coll = g.get("coll", [])
            ent = g.get("ent", [])
            print(f"  门格 coll={coll[gi] if gi < len(coll) else '?'} "
                  f"ent={ent[gi] if gi < len(ent) else '?'}  "
                  f"(GridCollisionClass: 0=NONE可走 1=坑 2=物体 3=实心 4=墙 5=墙但玩家可过)")
            pi = p["grid_idx"]
            print(f"  玩家格 coll={coll[pi] if pi < len(coll) else '?'} "
                  f"ent={ent[pi] if pi < len(ent) else '?'}")
            # 统计整个房间各碰撞类别有多少格
            from collections import Counter
            print(f"  房间碰撞类别分布: {dict(Counter(coll))}")
        else:
            print("房间没有开着的门 → 老师会原地不动")

    mgr.stop_all()
    return 0


if __name__ == "__main__":
    sys.exit(main())
