# -*- coding: utf-8 -*-
"""诊断：实机观察清房后的选门/楼层连通图，排查房间之间来回走。

用法：
    python tools/diag_explore.py --seconds 120
"""
import argparse
import os
import sys
import time

try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from isaac_ai import multi_instance as mi  # noqa: E402
from isaac_ai.play_ai import Runner, restore_window  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seconds", type=float, default=120.0)
    ap.add_argument("--hz", type=float, default=30.0)
    args = ap.parse_args()

    wins = mi.isaac_windows()
    if not wins:
        print("✗ 没有 Isaac 窗口")
        return 1
    mgr = mi.InstanceManager(n=1, base_port=48532)
    mgr.attach()
    inst = mgr.instances[0]
    restore_window(inst.hwnd, verbose=False)
    ch = mgr.bridge.channels.get(inst.port)
    if ch is None:
        print("✗ 没有端口通道")
        return 2
    r = Runner(inst, ch, 0)
    last_room = object()
    last_choice = None
    try:
        t0 = time.time()
        while time.time() - t0 < args.seconds:
            r.step(time.time(), args.hz)
            s = r.last_state or {}
            room = s.get("room") or {}
            ridx = room.get("index")
            a = r.last_action
            ag = r.agent
            choice = None
            if room.get("clear") and a:
                d = ag.choose_door()
                choice = (d.get("slot"), d.get("grid")) if d else None
            if ridx != last_room or choice != last_choice:
                last_room = ridx
                last_choice = choice
                print(f"[{time.time()-t0:5.1f}s] room={ridx} clear={room.get('clear')} "
                      f"mode={a.mode if a else None} door={a.debug.get('door') if a else None} "
                      f"choice={choice} hp={s.get('player', {}).get('hp')}")
                if a and a.mode == "door":
                    print(f"    edges={ {k: dict(v) for k, v in ag.floor_edges.items()} }")
                    print(f"    seen={ {k: sorted(v) for k, v in ag.seen_door_slots.items()} }")
                    print(f"    unvisited={[(x.get('grid'), x.get('type'), x.get('visited')) for x in ag.map_rooms if not x.get('visited')]}")
                    print(f"    doors={[(d.get('slot'), d.get('grid'), d.get('open')) for d in ag.doors]}")
            time.sleep(0.01)
    except KeyboardInterrupt:
        pass
    finally:
        r._release()
        inst.controller.release_all()
        try:
            mgr.bridge.close()
        except Exception:
            pass
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
