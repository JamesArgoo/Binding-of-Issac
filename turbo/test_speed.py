"""test_speed.py —— 测 turbo 注入后的游戏逻辑帧率，判断加速是否生效

用法（在 turbo/ 目录里）：
    python test_speed.py --speed 20 --skip-present 0     # 只变速时钟
    python test_speed.py --speed 20 --skip-present 1     # 时钟 + 跳过呈现
    python test_speed.py --speed 1  --skip-present 1     # 只跳过呈现（基线对照）

它会：注入启动一个新实例 → 等窗口出现 → 自动按 Enter 开局 → 连上模组测帧率。
"""
import argparse
import ctypes
import os
import subprocess
import sys
import time
from ctypes import wintypes

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))
import multi_instance as mi  # noqa: E402

EXE = r"D:\SteamLibrary\steamapps\common\The Binding of Isaac Rebirth\isaac-ng.exe"
INJECTOR = os.path.join(HERE, "turbo_inject.exe")
DLL = os.path.join(HERE, "turbo_hook.dll")

user32 = ctypes.WinDLL("user32")
user32.PostMessageW.argtypes = [wintypes.HWND, wintypes.UINT,
                                wintypes.WPARAM, wintypes.LPARAM]


def press(hwnd, vk=0x0D):
    user32.PostMessageW(hwnd, 0x100, vk, 1)
    time.sleep(0.05)
    user32.PostMessageW(hwnd, 0x101, vk, 1 | (1 << 30) | (1 << 31))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--speed", type=float, default=20.0)
    ap.add_argument("--skip-present", type=int, default=1)
    ap.add_argument("--render-every", type=int, default=0)
    ap.add_argument("--port", type=int, default=48536)
    ap.add_argument("--seconds", type=float, default=16.0)
    args = ap.parse_args()

    env = os.environ.copy()
    env["SteamAppId"] = "250900"
    env["SteamGameId"] = "250900"
    env["SteamClientLaunch"] = "1"
    env["ISAAC_TURBO_SPEED"] = str(args.speed)
    env["ISAAC_TURBO_SKIP_PRESENT"] = str(args.skip_present)
    env["ISAAC_TURBO_RENDER_EVERY"] = str(args.render_every)
    log = os.path.join(HERE, "turbo.log")
    env["ISAAC_TURBO_LOG"] = log

    before = {w[1] for w in mi.isaac_windows()}
    print(f"注入参数: speed={args.speed} skip_present={args.skip_present} "
          f"render_every={args.render_every}")

    subprocess.Popen([INJECTOR, DLL, EXE, "--luadebug"], env=env,
                     stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

    hwnd = None
    t0 = time.time()
    while time.time() - t0 < 90:
        for h, pid, cls, title in mi.isaac_windows():
            if pid not in before:
                hwnd = h
                break
        if hwnd:
            break
        time.sleep(0.5)
    if not hwnd:
        print("没等到新窗口")
        return 1
    print(f"新实例窗口 hwnd={hwnd}，等 12 秒后按 Enter 开局")
    time.sleep(12)
    for _ in range(4):
        press(hwnd, 0x0D)
        time.sleep(1.2)
    press(hwnd, 0x20)
    time.sleep(1.0)

    bridge = mi.MultiPortBridge(base_port=args.port, max_instances=1)
    bridge.open_port(args.port)
    print(f"监听 {args.port} 测帧率 ...")
    ch = bridge.channels[args.port]
    t0 = time.time()
    ref = None
    rates = []
    while time.time() - t0 < args.seconds:
        s = ch.state()
        if s and "frame" in s and "player" in s:
            now = time.time()
            if ref is None:
                ref = (now, s["frame"])
            elif now - ref[0] >= 2.0:
                fps = (s["frame"] - ref[1]) / (now - ref[0])
                rates.append(fps)
                print(f"  frame {ref[1]} -> {s['frame']} 于 {now - ref[0]:.1f}s "
                      f"=> 逻辑帧率 {fps:.1f} fps")
                ref = (now, s["frame"])
        time.sleep(0.05)

    if rates:
        avg = sum(rates) / len(rates)
        print(f"\n平均逻辑帧率: {avg:.1f} fps")
        print(f"相对 30fps 基线: {avg / 30.0:.2f} 倍")
        if avg > 45:
            print("=> ✅ 加速生效")
        else:
            print("=> ❌ 没有加速（仍是 30fps 基线）")
    else:
        print("没测到帧率（实例没进对局？）")
    bridge.close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
