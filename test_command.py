# -*- coding: utf-8 -*-
"""测试双向命令通道：restart / goto 到底有没有生效。

用法：python test_command.py
"""
import sys
import time

try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

import multi_instance as mi


def snap(ch):
    s = ch.state()
    if not s or "player" not in s:
        return dict(room="?", hp=None, enemies=None, clear=None)
    enemies = [e for e in s.get("entities", []) if e.get("enemy") or e.get("boss")]
    return dict(room=s["room"]["index"], hp=s["player"]["hp"],
                enemies=len(enemies), clear=s["room"]["clear"],
                type=s["room"]["type"])


def main():
    mgr = mi.InstanceManager(n=1)
    mgr.attach()
    if not mgr.wait_connections(60):
        print("没有实例在对局里")
        return 1

    for inst in mgr.instances:
        ch = mgr.bridge.channels[inst.port]
        print(f"\n===== 实例{inst.index} port={inst.port} =====")
        print(f"  初始: {snap(ch)}")

        for v in (4, 10, 20, 40, 4):
            ok = ch.send(f"goto d.{v}")
            time.sleep(0.6)
            print(f"  goto d.{v}: send={ok}  ->  {snap(ch)}")

        print("  测试 restart ...")
        ch.send("restart")
        time.sleep(1.5)
        print(f"  restart 后: {snap(ch)}")

    mgr.stop_all()
    return 0


if __name__ == "__main__":
    sys.exit(main())
