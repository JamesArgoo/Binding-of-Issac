# -*- coding: utf-8 -*-
"""测试用控制台命令给玩家加无敌/回血，解决"探测和训练时站着挨打会死"的问题。

Isaac 的 debug 命令（控制台开启时可用）常见几个：
    debug 1  回满血 / 无限血
    debug 3  无敌
    debug 8  无限道具
具体哪个生效，这里逐个试，看 HP 会不会掉。
"""
import sys
import time

try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

import multi_instance as mi


def hp(ch):
    s = ch.state()
    if not s or "player" not in s:
        return None
    return s["player"]["hp"] + s["player"]["soul_hp"]


def main():
    mgr = mi.InstanceManager(n=1)
    mgr.attach()
    if not mgr.wait_connections(45):
        print("没有实例在对局里（可能都死了）")
        mgr.stop_all()
        return 1

    for inst in mgr.instances:
        ch = mgr.bridge.channels[inst.port]
        h0 = hp(ch)
        print(f"\n===== 实例{inst.index} port={inst.port} 初始 HP={h0} =====")

        for cmd in ("debug 3", "debug 1", "debug 8"):
            ch.send(f"exec {cmd}")
            time.sleep(0.4)
            print(f"  发送 exec {cmd} 后 HP={hp(ch)}")

        print("  观察 8 秒，看 HP 还掉不掉 ...")
        t0 = time.time()
        last = hp(ch)
        while time.time() - t0 < 8:
            time.sleep(0.5)
            cur = hp(ch)
            if cur != last:
                print(f"    +{time.time() - t0:4.1f}s HP {last} -> {cur}")
                last = cur
        final = hp(ch)
        print(f"  8 秒后 HP={final}")
        if h0 is not None and final is not None and final >= h0:
            print("  => 无敌/回血生效 ✅ （可以用它保护探测和训练）")
        else:
            print("  => 没保住血，这些 debug 命令不管用")

    mgr.stop_all()
    return 0


if __name__ == "__main__":
    sys.exit(main())
