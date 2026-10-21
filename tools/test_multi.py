# -*- coding: utf-8 -*-
"""多开可行性验证。

验证两件决定性的事：
  1. 能不能同时跑 N 个游戏实例，且各自的端口/窗口映射正确
  2. PostMessage 能不能驱动**没有焦点**的窗口（多开的命门）

用法：
    python tools/test_multi.py --launch 2      # 启动 2 个实例，跑完整验证
    python tools/test_multi.py --attach        # 用已经开着的游戏验证（单实例）
    python tools/test_multi.py --launch 2 --keep   # 测完不关游戏

注意：新启动的实例停在主菜单，需要**手动在每个窗口里开一局**，脚本会等。
"""
import argparse
import os
import sys
import time

try:
    sys.stdout.reconfigure(encoding="utf-8")
    sys.stderr.reconfigure(encoding="utf-8")
except Exception:
    pass

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from isaac_ai import multi_instance as mi  # noqa: E402


def describe(msg):
    if not msg:
        return "<还没数据>"
    if "hello" in msg:
        return f"hello 已握手 (port={msg.get('port')})"
    if "error" in msg:
        return f"模组报错: {msg['error']}"
    if "menu" in msg:
        return f"主菜单（自动开局已尝试 {msg.get('start_attempts', 0)} 次）"
    p = msg.get("player")
    if not p:
        return "有数据但没有 player"
    enemies = [e for e in msg.get("entities", []) if e.get("enemy") or e.get("boss")]
    return (f"对局中 玩家=({p['x']:.0f},{p['y']:.0f}) "
            f"HP={p['hp']}/{p['max_hp']} 敌人={len(enemies)}")


def wait_for_runs(mgr, timeout=180.0):
    """等所有实例都进到对局里，每 5 秒打印一次进度。"""
    t0 = time.time()
    last_note = 0.0
    while time.time() - t0 < timeout:
        ready = 0
        for inst in mgr.instances:
            s = mgr.state(inst.index)
            if s and "player" in s:
                ready += 1
        if ready == len(mgr.instances):
            return True
        if time.time() - last_note > 5.0:
            last_note = time.time()
            parts = []
            for i in mgr.instances:
                st = mgr.state(i.index) or {}
                parts.append(f"#{i.index}:{'对局中' if st.get('player') else '菜单'}")
            print(f"  [{time.time() - t0:5.0f}s] {' | '.join(parts)}")
        time.sleep(0.5)
    return False


def test_injection(inst, mgr, hold=0.8):
    """在该实例上按 D（右）一下，看玩家有没有往右走。返回 (是否生效, dx, dy)。"""
    s0 = mgr.state(inst.index)
    if not s0 or "player" not in s0:
        return None, 0.0, 0.0
    p0 = s0["player"]

    inst.controller.set_move(1, 0)      # 按住右
    time.sleep(hold)
    s1 = mgr.state(inst.index)
    inst.controller.set_move(0, 0)      # 松手
    time.sleep(0.3)
    s2 = mgr.state(inst.index)

    if not s1 or "player" not in s1:
        return None, 0.0, 0.0
    dx = s1["player"]["x"] - p0["x"]
    dy = s1["player"]["y"] - p0["y"]
    # 看第二帧确认松手后停下（避免把惯性误判成按键生效）
    dx2 = (s2["player"]["x"] - s1["player"]["x"]) if (s2 and "player" in s2) else 0.0
    return (abs(dx) > 3.0 and abs(dx2) < abs(dx) * 0.5), dx, dy


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--launch", type=int, default=0, help="启动几个实例")
    ap.add_argument("--attach", action="store_true", help="用已开着的游戏")
    ap.add_argument("--keep", action="store_true", help="测完不关游戏")
    ap.add_argument("--exe", default=mi.DEFAULT_EXE)
    ap.add_argument("--run-timeout", type=float, default=180.0)
    args = ap.parse_args()

    mgr = None
    if args.launch:
        print(f"==== 启动 {args.launch} 个实例 ====")
        print(f"exe: {args.exe}")
        mgr = mi.InstanceManager(n=args.launch, exe=args.exe)
        t0 = time.time()
        print("窗口都出来后会等你在每个窗口里开局（任意顺序）...\n")
        try:
            mgr.launch_all(wait_runs=True, run_timeout=args.run_timeout)
        except Exception as e:
            print(f"\n启动/对号失败: {e}")
            mgr.stop_all()
            return 1
        print(f"\n全部就绪，耗时 {time.time() - t0:.1f}s")
        print("\n==== 实例映射 ====")
        for inst in mgr.instances:
            print(f"  {inst!r}")
    elif args.attach:
        mgr = mi.InstanceManager(n=1)
        try:
            mgr.attach()
        except Exception as e:
            print(f"接管失败: {e}")
            return 1
        print("\n等各实例的模组连上来（确认每个窗口都在对局里）...")
        if not mgr.wait_connections(args.run_timeout):
            print("超时：有窗口没进对局")
            mgr.stop_all()
            return 2
        print("\n==== 端口 ↔ 窗口 对号 ====")
        mgr.resolve_mapping()
    else:
        print("需要 --launch N 或 --attach")
        return 1

    try:
        print("\n==== PostMessage 注入测试 ====")
        fg = mi.user32.GetForegroundWindow()
        print(f"当前前台窗口 hwnd={fg}\n")
        results = []
        for inst in mgr.instances:
            focused = (inst.hwnd == fg)
            ok, dx, dy = test_injection(inst, mgr)
            tag = "焦点内" if focused else "**非焦点**"
            mark = {True: "✅ 生效", False: "❌ 无反应", None: "⚠️ 拿不到状态"}[ok]
            print(f"  实例{inst.index} port={inst.port} ({tag}): dx={dx:+.1f} dy={dy:+.1f}  {mark}")
            results.append((inst, ok, focused))

        print("\n==== 结论 ====")
        non_focus = [(i, ok) for i, ok, f in results if not f]
        if all(ok for _i, ok, _f in results):
            print("✅ 所有实例都能被 PostMessage 驱动，多开在输入这一环通了。")
            if non_focus:
                print(f"   其中 {len(non_focus)} 个是**非焦点**窗口，说明焦点问题已解决。")
            rc = 0
        elif any(ok for _i, ok, _f in results):
            print("⚠️ 部分实例生效，部分不生效——需要逐个排查（可能是暂停/菜单/撞墙）。")
            rc = 3
        else:
            print("❌ PostMessage 完全驱动不了游戏。")
            print("   可能：游戏用 GetAsyncKeyState/原始输入轮询键盘，不吃 PostMessage。")
            print("   下一步只能考虑 SendInput（需抢焦点、不能真并行）或改模组侧直接控速。")
            rc = 4
        return rc
    finally:
        if mgr and not args.keep:
            print("\n关闭测试实例 ...")
            mgr.stop_all()
        elif mgr:
            print("\n--keep：实例保持运行。")


if __name__ == "__main__":
    sys.exit(main())
