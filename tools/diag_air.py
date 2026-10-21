# -*- coding: utf-8 -*-
"""诊断：实机采样高度/下落速度字段，确认跳跃 BOSS 与高抛弹幕的数据形态。

用法：
    python tools/diag_air.py                 # 只看当前房间
    python tools/diag_air.py --goto-boss     # 传送到本层 Boss 房再采样
    python tools/diag_air.py --seconds 20
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


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--goto-boss", action="store_true")
    ap.add_argument("--seconds", type=float, default=20.0)
    args = ap.parse_args()

    wins = mi.isaac_windows()
    if not wins:
        print("✗ 没有 Isaac 窗口")
        return 1
    bridge = mi.MultiPortBridge(base_port=48532, max_instances=1)
    ch = bridge.open_port(48532)
    t0 = time.time()
    while not ch.connected and time.time() - t0 < 20:
        time.sleep(0.2)
    if not ch.connected:
        print("✗ 模组没有连上 48532")
        return 2
    ch.wait_new(0, timeout=2.0)
    st = ch.state() or {}
    p = st.get("player") or {}
    room = st.get("room") or {}
    print(f"起始 room={room.get('index')} type={room.get('type')} clear={room.get('clear')} "
          f"hp={p.get('hp')} frame={st.get('frame')}")
    bosses = [r for r in st.get("map") or [] if int(r.get("type", 0) or 0) == 5]
    print("地图 Boss 房:", [(r.get("grid"), r.get("visited")) for r in bosses])
    if args.goto_boss:
        if not bosses:
            print("✗ 本层没有 Boss 房")
        else:
            bg = bosses[0].get("grid")
            print(f"下发 goto b.{bg}")
            ch.send(f"goto b.{bg}")
            time.sleep(1.0)

    seen = {}
    seq = ch.seq()
    end = time.time() + args.seconds
    last_print = 0
    while time.time() < end:
        s = ch.seq()
        if s == seq:
            time.sleep(0.005)
            continue
        seq = s
        st = ch.state() or {}
        p = st.get("player") or {}
        room = st.get("room") or {}
        ents = st.get("entities") or []
        frame = st.get("frame")
        for e in ents:
            h = float(e.get("height", 0) or 0)
            if h <= 1.0:
                continue
            key = (int(e.get("id", 0)), int(e.get("type", 0)),
                   int(e.get("variant", 0) or 0), int(e.get("subtype", 0) or 0))
            rec = seen.setdefault(key, {"n": 0, "hmin": 1e9, "hmax": -1e9,
                                        "vmin": 1e9, "vmax": -1e9,
                                        "fs": set(), "fa": set(),
                                        "boss": bool(e.get("boss")),
                                        "enemy": bool(e.get("enemy")),
                                        "dmg": e.get("cdmg"),
                                        "spawner": e.get("spawner_type")})
            rec["n"] += 1
            rec["hmin"] = min(rec["hmin"], h)
            rec["hmax"] = max(rec["hmax"], h)
            sp = (float(e.get("vx", 0) or 0) ** 2 + float(e.get("vy", 0) or 0) ** 2) ** 0.5
            rec["vmin"] = min(rec["vmin"], sp)
            rec["vmax"] = max(rec["vmax"], sp)
            if "falling_speed" in e:
                rec["fs"].add(round(float(e.get("falling_speed", 0) or 0), 2))
            if "falling_accel" in e:
                rec["fa"].add(round(float(e.get("falling_accel", 0) or 0), 2))
        if time.time() - last_print > 2.0:
            last_print = time.time()
            names = ", ".join(f"id{k[0]}/t{k[1]}" for k in list(seen)[:6])
            print(f"[{time.time() + args.seconds - end:4.0f}s] room={room.get('index')} "
                  f"hp={p.get('hp')} ents={len(ents)} 高空实体种类={len(seen)} {names}")
    print("\n高度采样汇总（height>1）：")
    for key, rec in sorted(seen.items()):
        print(f"  id={key[0]} type={key[1]} var={key[2]} sub={key[3]} "
              f"boss={rec['boss']} enemy={rec['enemy']} spawner={rec['spawner']} cdmg={rec['dmg']} "
              f"帧数={rec['n']} h=[{rec['hmin']},{rec['hmax']}] |v|=[{rec['vmin']:.2f},{rec['vmax']:.2f}] "
              f"fall_speed={sorted(rec['fs'])[:8]} fall_accel={sorted(rec['fa'])[:8]}")
    try:
        bridge.close()
    except Exception:
        pass
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
