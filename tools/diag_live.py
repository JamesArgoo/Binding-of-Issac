# -*- coding: utf-8 -*-
"""诊断：接管正在运行的以撒，检查状态/网格映射/按键是否真的生效。

用法：
    python tools/diag_live.py            # 只读取状态
    python tools/diag_live.py --press    # 依次按住 下/右/上/左 各 1.2 秒，观察速度与位移
    python tools/diag_live.py --press --focus   # 把游戏切到前台再按键（排查焦点问题）
"""
import argparse
import json
import math
import os
import sys
import time

try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from isaac_ai import multi_instance as mi  # noqa: E402
from isaac_ai.tactical_agent import RoomGrid  # noqa: E402


def pause(sec):
    time.sleep(sec)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--press", action="store_true")
    ap.add_argument("--focus", action="store_true")
    ap.add_argument("--seconds", type=float, default=1.2)
    args = ap.parse_args()

    wins = mi.isaac_windows()
    print("游戏窗口：", [(w[0], w[1], w[3]) for w in wins])
    if not wins:
        print("✗ 没有 Isaac 窗口")
        return 1
    hwnd = wins[0][0]

    bridge = mi.MultiPortBridge(base_port=48532, max_instances=1)
    ch = bridge.open_port(48532)
    t0 = time.time()
    while not ch.connected and time.time() - t0 < 20:
        pause(0.2)
    if not ch.connected:
        print("✗ 模组没有连上 48532")
        return 2
    ch.wait_new(0, timeout=1.0)
    st = ch.state()
    p = st.get("player") or {}
    room = st.get("room") or {}
    grid = st.get("grid")
    print(f"房 {room.get('index')} 类型{room.get('type')} 清房{room.get('clear')} "
          f"HP {p.get('hp')} 位置({p.get('x')},{p.get('y')}) 速度({p.get('vx')},{p.get('vy')}) "
          f"grid_idx={p.get('grid_idx')} fire_dir={p.get('fire_dir')}")
    print(f"门 {[(d.get('slot'), d.get('grid'), d.get('open')) for d in st.get('doors') or []]}")
    print(f"地图 {[(r.get('grid'), r.get('type'), r.get('visited')) for r in st.get('map') or []]}")
    if grid:
        rg = RoomGrid(grid)
        print(f"网格 {rg.w}x{rg.h} gv={grid.get('gv')} ox={rg.ox:.2f} oy={rg.oy:.2f} "
              f"tw={rg.tw:.2f} th={rg.th:.2f}")
        print(f"玩家映射格 {rg.cell_of(float(p['x']), float(p['y']))} / 游戏给 "
              f"({int(p['grid_idx']) % rg.w}, {int(p['grid_idx']) // rg.w})")
        if st.get("doors"):
            for d in st["doors"]:
                g = int(d.get("grid", -1))
                if 0 <= g < rg.w * rg.h:
                    print(f"  门 slot={d.get('slot')} grid={g} cell=({g % rg.w},{g // rg.w}) "
                          f"coll={rg.coll[g]} center={tuple(round(v, 1) for v in rg.center(g))}")
    else:
        print("（这一帧没有 grid）")

    if not args.press:
        return 0

    if args.focus:
        user32 = mi.user32
        fg = user32.GetForegroundWindow()
        user32.SetForegroundWindow(hwnd)
        print(f"--focus：前台 {fg} -> {hwnd}")
        pause(0.5)

    ctrl = mi.Win32KeyController(hwnd)
    seq = ch.seq()
    for name, mv in (("下", (0, 1)), ("右", (1, 0)), ("上", (0, -1)), ("左", (-1, 0))):
        p0 = (ch.state() or {}).get("player") or {}
        ctrl.set_move(*mv)
        t1 = time.time()
        samples = []
        while time.time() - t1 < args.seconds:
            s = ch.seq()
            if s != seq:
                seq = s
                q = (ch.state() or {}).get("player") or {}
                samples.append((round(float(q.get("vx", 0)), 2),
                                round(float(q.get("vy", 0)), 2)))
            pause(0.03)
        ctrl.set_move(0, 0)
        pause(0.15)
        p1 = (ch.state() or {}).get("player") or {}
        moved = math.hypot(float(p1.get("x", 0)) - float(p0.get("x", 0)),
                           float(p1.get("y", 0)) - float(p0.get("y", 0)))
        nz = sum(1 for vx, vy in samples if abs(vx) + abs(vy) > 0.1)
        print(f"按{name}：位移 {moved:.1f}  速度非零 {nz}/{len(samples)}  "
              f"末速度({p1.get('vx')},{p1.get('vy')})")
    ctrl.release_all()
    return 0


if __name__ == "__main__":
    sys.exit(main())
