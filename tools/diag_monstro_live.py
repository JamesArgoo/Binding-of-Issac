# -*- coding: utf-8 -*-
"""诊断：实机生成 Monstro，观察 AI 是否在 JumpUp 阶段就横移躲开。

用法：
    python tools/diag_monstro_live.py --seconds 30
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
from isaac_ai.play_ai import Runner, restore_window, hp_of  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seconds", type=float, default=30.0)
    ap.add_argument("--hz", type=float, default=30.0)
    ap.add_argument("--type", type=int, default=20)
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
    try:
        # 先等游戏真正进入对局（刚重启时可能还在菜单），再重开一局
        ch.send("restart")
        t0 = time.time()
        s = {}
        while time.time() - t0 < 40.0:
            r.step(time.time(), args.hz)
            s = r.last_state or {}
            if (s.get("room") or {}).get("index") is not None and \
                    (s.get("player") or {}).get("hp") is not None:
                break
            if time.time() - t0 > 2.0 and int(time.time() - t0) % 4 == 0:
                ch.send("restart")
            time.sleep(0.02)
        if (s.get("room") or {}).get("index") is None:
            print("✗ 40 秒内没有进入对局")
            return 3
        print(f"进入对局 room={s['room'].get('index')} hp={hp_of(s)}")
        time.sleep(0.5)
        boss_seen = False
        for attempt in range(4):
            s = ch.state() or {}
            if any(int(e.get("type", 0)) == args.type for e in s.get("entities") or []):
                boss_seen = True
                break
            print(f"spawn Monstro（第 {attempt + 1} 次）")
            ch.send(f"spawn {args.type} 0 0")
            time.sleep(0.8)
        if not boss_seen:
            print("✗ 连续 4 次 spawn 都没生成出 BOSS")
            return 4

        last_hp = None
        last_anim = None
        last_mode = None
        damage_events = []
        moved_in_jump = 0
        boss_gone_at = None
        end = time.time() + args.seconds
        while time.time() < end:
            r.step(time.time(), args.hz)
            s = r.last_state or {}
            a = r.last_action
            if not s or a is None:
                time.sleep(0.01)
                continue
            p = s.get("player") or {}
            hp = hp_of(s)
            boss = next((e for e in s.get("entities") or []
                         if int(e.get("type", 0)) == args.type), None)
            if boss is None:
                if boss_gone_at is None:
                    boss_gone_at = time.time()
                elif time.time() - boss_gone_at > 2.0:
                    print(f"[{time.time() - (end - args.seconds):5.2f}s] BOSS 消失（死亡/换房/重开）")
                    break
            else:
                boss_gone_at = None
            anim = (boss or {}).get("anim")
            jumping = bool(anim and "jump" in str(anim).lower())
            t = time.time() - (end - args.seconds)
            if last_hp is not None and hp < last_hp:
                damage_events.append((t, last_anim, round(float((boss or {}).get("x", 0)), 1),
                                      round(float((boss or {}).get("y", 0)), 1), a.mode))
                print(f"  !! {t:5.2f}s 掉血 {last_hp:.0f}->{hp:.0f} "
                      f"boss_anim={last_anim} boss=({(boss or {}).get('x')},{(boss or {}).get('y')}) "
                      f"player=({p.get('x')},{p.get('y')}) mode={a.mode} move={a.move}")
            last_hp = hp
            if jumping and a.move != (0, 0):
                moved_in_jump += 1
            if anim != last_anim or a.mode != last_mode:
                print(f"[{t:5.2f}s] hp={hp:.0f} anim={anim} mode={a.mode} move={a.move} "
                      f"shoot={a.shoot} threat={a.debug.get('threat')} "
                      f"jump_land={a.debug.get('jump_land')} "
                      f"pl=({p.get('x')},{p.get('y')}) "
                      f"B=({(boss or {}).get('x')},{(boss or {}).get('y')})")
                last_anim, last_mode = anim, a.mode
            time.sleep(0.005)
        print(f"\n掉血次数 {len(damage_events)}，跳跃阶段有移动的帧数 {moved_in_jump}")
        for d in damage_events:
            print(f"  掉血于 {d[0]:.2f}s Boss 动画={d[1]} 位置=({d[2]},{d[3]}) mode={d[4]}")
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
