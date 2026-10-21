# -*- coding: utf-8 -*-
"""诊断：实机生成指定实体，采样高度/空中偏移/动画，确认跳跃 BOSS 的数据形态。

用法：
    python tools/diag_spawn.py --type 20 --variant 0 --seconds 15
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
    ap.add_argument("--type", type=int, default=20)
    ap.add_argument("--variant", type=int, default=0)
    ap.add_argument("--subtype", type=int, default=0)
    ap.add_argument("--seconds", type=float, default=15.0)
    args = ap.parse_args()

    bridge = mi.MultiPortBridge(base_port=48532, max_instances=1)
    ch = bridge.open_port(48532)
    t0 = time.time()
    while not ch.connected and time.time() - t0 < 40:
        time.sleep(0.25)
    if not ch.connected:
        print("✗ 模组没有连上 48532")
        return 2
    print("模组已连接")
    # 等进入对局
    room = {}
    end_wait = time.time() + 40
    while time.time() < end_wait:
        ch.wait_new(0, timeout=0.5)
        st = ch.state() or {}
        room = st.get("room") or {}
        if room.get("index") is not None:
            break
    if room.get("index") is None:
        print("✗ 40 秒内没有进入对局")
        return 3
    p = (ch.state() or {}).get("player") or {}
    print(f"对局 room={room.get('index')} hp={p.get('hp')} frame={(ch.state() or {}).get('frame')}")
    print(f"spawn {args.type} {args.variant} {args.subtype}")
    ch.send(f"spawn {args.type} {args.variant} {args.subtype}")
    time.sleep(1.5)

    st = ch.state() or {}
    room = st.get("room") or {}
    enemies = [e for e in st.get("entities") or [] if int(e.get("type", 0)) == args.type]
    print("生成后实体:", [(e.get("id"), e.get("x"), e.get("y"), e.get("hp")) for e in enemies])
    if not enemies:
        print("✗ 没生成出来（Boss 可能不允许直接 spawn，或控制台被禁用）")
        bridge.close()
        return 4

    seq = ch.seq()
    end = time.time() + args.seconds
    last = None
    rows = 0
    while time.time() < end:
        s = ch.seq()
        if s == seq:
            time.sleep(0.004)
            continue
        seq = s
        st = ch.state() or {}
        room = st.get("room") or {}
        p = st.get("player") or {}
        ents = st.get("entities") or []
        boss = next((e for e in ents if int(e.get("type", 0)) == args.type), None)
        if boss is not None:
            row = (round(float(boss.get("x", 0) or 0), 1),
                   round(float(boss.get("y", 0) or 0), 1),
                   round(float(boss.get("vx", 0) or 0), 2),
                   round(float(boss.get("vy", 0) or 0), 2),
                   float(boss.get("height", 0) or 0),
                   float(boss.get("sprite_off_y", 0) or 0),
                   boss.get("anim"), boss.get("visible"), boss.get("invincible"),
                   boss.get("hp"))
            if row != last:
                last = row
                rows += 1
                print(f"[{time.time() - (end - args.seconds):5.1f}s] room={room.get('index')} hp={p.get('hp')} "
                      f"pos=({row[0]},{row[1]}) v=({row[2]},{row[3]}) "
                      f"h={row[4]} off_y={row[5]} anim={row[6]} vis={row[7]} "
                      f"inv={row[8]} bhp={row[9]}")
        else:
            if last is not None or room.get("index") is None:
                print(f"[{time.time() - (end - args.seconds):5.1f}s] BOSS 不在实体列表 "
                      f"(room={room.get('index')} hp={p.get('hp')}) 可能是跳出场/死亡/重开")
            last = None
        for e in ents:
            h = float(e.get("height", 0) or 0)
            if h < -25 and h != -23:
                print(f"    高抛弹幕 id={e.get('id')} t={e.get('type')} h={h} "
                      f"fs={e.get('fall_speed')} fa={e.get('fall_accel')} "
                      f"pos=({e.get('x')},{e.get('y')}) v=({e.get('vx')},{e.get('vy')}) "
                      f"spawner={e.get('spawner_type')}")
    print(f"共 {rows} 条 Boss 状态变化")
    try:
        bridge.close()
    except Exception:
        pass
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
