# -*- coding: utf-8 -*-
"""探针：在端口段上听，看游戏模组会不会连上来。用来判断 socket 通路是否正常。

用法：
    python probe.py              # 听 48532，30 秒
    python probe.py --port 48533 --seconds 60
"""
import argparse
import sys
import time

try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

from multi_instance import MultiPortBridge


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, default=48532)
    ap.add_argument("--seconds", type=float, default=30.0)
    args = ap.parse_args()

    b = MultiPortBridge(base_port=args.port, max_instances=1)
    b.open_port(args.port)
    print(f"监听 127.0.0.1:{args.port} ...")
    t0 = time.time()
    last = None
    while time.time() - t0 < args.seconds:
        s = b.channels[args.port].state()
        if s is not None and s is not last:
            last = s
            elapsed = time.time() - t0
            keys = list(s.keys())
            if "hello" in s:
                print(f"[{elapsed:5.1f}s] hello 握手成功! port={s.get('port')}")
            elif "menu" in s:
                print(f"[{elapsed:5.1f}s] menu 帧={s.get('frame')} 自动开局尝试={s.get('start_attempts')}")
            elif "player" in s:
                p = s["player"]
                print(f"[{elapsed:5.1f}s] 对局状态 玩家=({p['x']:.0f},{p['y']:.0f}) "
                      f"HP={p['hp']} 房间type={s.get('room', {}).get('type')}")
            else:
                print(f"[{elapsed:5.1f}s] 其他消息 keys={keys}")
        time.sleep(0.2)

    if b.channels[args.port].state() is None:
        print(f"\n{args.seconds:.0f} 秒内没有任何连接。")
        print("=> 模组的 socket 通路有问题（require(\"socket\") 大概率失败了）")
        rc = 1
    else:
        print("\n=> socket 通路正常 ✅")
        rc = 0
    b.close()
    return rc


if __name__ == "__main__":
    sys.exit(main())
