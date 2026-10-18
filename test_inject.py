# -*- coding: utf-8 -*-
"""对已经在运行的游戏做 PostMessage 注入测试（不需要脚本启动游戏）。

用法：
    python test_inject.py             # 测试焦点内 + 非焦点
    python test_inject.py --focused-only
"""
import argparse
import sys
import time

try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

import multi_instance as mi


def find_non_isaac_window():
    """找一个非 Isaac 的可见窗口，用来把焦点抢走。"""
    for hwnd, _pid, _cls, title in mi.list_windows():
        if "isaac" not in title.lower():
            return hwnd, title
    return None, None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--focused-only", action="store_true")
    ap.add_argument("--port", type=int, default=48532)
    args = ap.parse_args()

    wins = mi.isaac_windows()
    if not wins:
        print("没找到 Isaac 窗口，先把游戏开起来并进一局")
        return 1
    hwnd, pid, cls, title = wins[0]
    print(f"目标窗口: hwnd={hwnd} pid={pid} class={cls!r}")
    print(f"窗口类名 = {cls!r}  <-- 能看出输入后端（SDL/GLFW/自定义）")

    b = mi.MultiPortBridge(base_port=args.port, max_instances=1)
    b.open_port(args.port)
    print(f"\n等模组连上 {args.port} ...")
    if not b.wait_connected(args.port, 30):
        print("模组 30 秒内没连上（确认游戏已经进到对局里）")
        b.close()
        return 2
    print("模组已连接 ✅")

    ctrl = mi.Win32KeyController(hwnd)

    def pos():
        s = b.channels[args.port].state()
        if s and "player" in s:
            return (s["player"]["x"], s["player"]["y"])
        return None

    def run_case(label):
        p0 = pos()
        if p0 is None:
            print(f"  {label}: 拿不到玩家状态")
            return None
        ctrl.set_move(1, 0)          # 按住右
        time.sleep(0.8)
        p1 = pos()
        ctrl.set_move(0, 0)          # 松手
        time.sleep(0.4)
        p2 = pos()
        if p1 is None or p2 is None:
            print(f"  {label}: 中途丢了状态")
            return None
        dx = p1[0] - p0[0]
        # 松手后应该明显减速，用来区分"真的在走"和"自己飘"
        drift = abs(p2[0] - p1[0])
        ok = abs(dx) > 3.0 and drift < abs(dx) * 0.6
        print(f"  {label}: 起点={p0[0]:.1f} 按住后={p1[0]:.1f} 松手后={p2[0]:.1f}  "
              f"dx={dx:+.1f} 松手漂移={drift:.1f}  -> {'✅ 生效' if ok else '❌ 无效'}")
        return ok

    print("\n==== 测试 1：窗口在前台（焦点内）====")
    fg = mi.user32.GetForegroundWindow()
    if fg != hwnd:
        print(f"  当前前台不是游戏窗口(hwnd={fg})，先切到后台测")
    else:
        print("  游戏当前在前台")
    r1 = run_case("焦点内")

    r2 = None
    if not args.focused_only:
        print("\n==== 测试 2：窗口失焦（多开的真实场景）====")
        other, otitle = find_non_isaac_window()
        if other is None:
            print("  找不到别的窗口来抢焦点，跳过")
        else:
            mi.user32.SetForegroundWindow(int(other))
            time.sleep(0.6)
            fg2 = mi.user32.GetForegroundWindow()
            print(f"  已把焦点切给 {otitle!r} (hwnd={other})，当前前台={fg2}")
            if fg2 == hwnd:
                print("  ⚠️ 焦点没切走，这次测试不算数")
            r2 = run_case("非焦点")

    print("\n==== 结论 ====")
    if r1 is True and (r2 is True or r2 is None):
        print("✅ PostMessage 能驱动 Isaac")
        if r2 is True:
            print("   而且**失焦也能驱动** —— 多开方案成立！")
        rc = 0
    elif r1 is True and r2 is False:
        print("⚠️ 焦点内能驱动，失焦不能。")
        print("   说明游戏在失焦时会忽略输入，多开得另想办法。")
        rc = 3
    else:
        print("❌ PostMessage 驱动不了 Isaac。")
        print("   游戏大概率用 GetAsyncKeyState / 原始输入轮询键盘，不吃投递消息。")
        print("   备选方案：模组侧直接改玩家速度/发弹（绕开键盘）。")
        rc = 4

    ctrl.release_all()
    b.close()
    return rc


if __name__ == "__main__":
    sys.exit(main())
