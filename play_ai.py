# -*- coding: utf-8 -*-
"""play_ai.py —— 让战术 AI 真正玩《以撒的结合：忏悔》

和训练脚本的区别：这里不加载 PPO 模型，决策全部来自 tactical_agent.py
（弹道求解 + 视线门控 + 威胁预测 + A* 导航），按键通过 PostMessage 投递，
不要求游戏窗口有焦点。

用法：
    python play_ai.py                    # 接管已开着的游戏；没有就启动一个
    python play_ai.py --launch 1         # 启动一个新实例并开始玩
    python play_ai.py --attach           # 只接管，不启动
    python play_ai.py --launch 2 --hz 30 # 两个实例并排跑
    python play_ai.py --duration 120     # 跑 2 分钟后自动停
    python play_ai.py --focus            # 调试：允许抢一次焦点

注意：
  * 游戏窗口被最小化时，Isaac 会暂停 onUpdate，模组不发数据、PostMessage
    也进不去。脚本启动后会先用 SW_SHOWNOACTIVATE 把窗口恢复出来（不抢焦点）。
  * 死亡后模组会自己 restart；如果卡在菜单/结算，脚本每 2 秒补一次
    Enter / restart 命令。
  * 多开时先用探针把"窗口 ↔ 端口"对上号，避免两个 AI 抢同一个窗口。
"""
import argparse
import ctypes
import json
import math
import os
import sys
import time
import traceback
from ctypes import wintypes

try:
    sys.stdout.reconfigure(encoding="utf-8")
    sys.stderr.reconfigure(encoding="utf-8")
except Exception:
    pass

import multi_instance as mi
from tactical_agent import Config, TacticalAgent, split_entities

user32 = ctypes.WinDLL("user32", use_last_error=True)

SW_SHOWNOACTIVATE = 4
SWP_NOSIZE = 0x0001
SWP_NOMOVE = 0x0002
SWP_NOACTIVATE = 0x0010
SWP_SHOWWINDOW = 0x0040

user32.IsIconic.restype = wintypes.BOOL
user32.IsIconic.argtypes = [wintypes.HWND]
user32.ShowWindow.restype = wintypes.BOOL
user32.ShowWindow.argtypes = [wintypes.HWND, ctypes.c_int]
user32.SetWindowPos.restype = wintypes.BOOL
user32.SetWindowPos.argtypes = [
    wintypes.HWND, wintypes.HWND, ctypes.c_int, ctypes.c_int,
    ctypes.c_int, ctypes.c_int, wintypes.UINT]
user32.SetForegroundWindow.restype = wintypes.BOOL
user32.SetForegroundWindow.argtypes = [wintypes.HWND]


def restore_window(hwnd, verbose=True):
    """把最小化/隐藏的游戏窗口恢复成可见但不抢焦点。"""
    if not hwnd:
        return False
    was_iconic = bool(user32.IsIconic(hwnd))
    user32.ShowWindow(hwnd, SW_SHOWNOACTIVATE)
    user32.SetWindowPos(hwnd, 0, 0, 0, 0, 0,
                        SWP_NOMOVE | SWP_NOSIZE | SWP_NOACTIVATE | SWP_SHOWWINDOW)
    still = bool(user32.IsIconic(hwnd))
    if verbose:
        if was_iconic:
            print(f"  窗口 {hwnd} 之前是最小化，已尝试恢复"
                  f"{'（仍是 minimized，可能需要在任务栏点开）' if still else '（未抢焦点）'}")
        elif still:
            print(f"  窗口 {hwnd} 仍处于最小化状态")
    return not still


def hp_of(state):
    p = (state or {}).get("player") or {}
    return float(p.get("hp", 0) or 0) + float(p.get("soul_hp", 0) or 0)


def in_run(state):
    """有玩家、且至少还有一滴血。菜单/结算/hello 都不算。"""
    return bool(state) and "player" in state and hp_of(state) > 0


class Runner:
    """一个实例的状态、战术 AI 和统计。"""

    def __init__(self, inst, channel, index, auto_start=True):
        self.inst = inst
        self.ch = channel
        self.index = index
        self.agent = TacticalAgent(Config())
        self.ctrl = inst.controller
        self.auto_start = auto_start

        self.last_decision = 0.0
        self.last_seq = -1
        self.last_change = time.time()
        self.last_state = None
        self.last_action = None
        self.failed = False

        self.prev_hp = None
        self.prev_room = None
        self.prev_clear = False
        self.prev_enemy_hp = {}
        self.prev_enemy_count = 0
        self.prev_shoot = (0, 0)
        self.last_alive = False
        self.room_had_enemies = False

        self.started = time.time()
        self.damage_taken = 0.0
        self.damage_dealt = 0.0
        self.kills = 0
        self.rooms_cleared = 0
        self.deaths = 0
        self.shot_frames = 0
        self.shot_bursts = 0
        self.hit_events = 0
        self.rooms_seen = set()

        self.last_menu_poke = 0.0
        self.last_restart = 0.0
        self.last_report = 0.0

    # ---------- 指标 ----------
    def observe(self, s):
        """根据两帧状态更新受伤/输出/击杀/清房统计。"""
        if not s or "player" not in s:
            return
        hp = hp_of(s)
        room = s.get("room") or {}
        ridx = room.get("index")

        if self.prev_room is not None and ridx != self.prev_room:
            # 换房：敌人列表整体失效，不能把"消失"算成击杀
            self.agent.reset_tracking()
            self.prev_enemy_hp = {}
            self.prev_enemy_count = 0
            self.prev_clear = bool(room.get("clear"))
            self.room_had_enemies = False

        if self.prev_hp is not None and hp < self.prev_hp - 1e-6:
            self.damage_taken += self.prev_hp - hp

        enemies, _proj, _pk, _hz = split_entities(s)
        cur = {e.id: max(0.0, e.hp) for e in enemies if e.hp > 0}
        if cur:
            self.room_had_enemies = True
        if self.prev_enemy_hp:
            dealt = 0.0
            for eid, old in self.prev_enemy_hp.items():
                if eid in cur and cur[eid] < old - 1e-6:
                    dealt += old - cur[eid]
                    self.hit_events += 1
            self.damage_dealt += dealt
            for eid, old in self.prev_enemy_hp.items():
                if old > 0 and eid not in cur and self.prev_room == ridx:
                    self.kills += 1

        if (not self.prev_clear and room.get("clear")
                and (self.prev_enemy_count > 0 or self.room_had_enemies)
                and ridx == self.prev_room):
            self.rooms_cleared += 1
            self.room_had_enemies = False
            self.agent.reset_tracking()

        if self.prev_hp is not None and self.prev_hp > 0 and hp <= 0:
            self.deaths += 1
            self.agent.reset_tracking()
            self.agent.reset_floor()

        if self.prev_hp is not None and self.prev_hp <= 0 and hp > 0:
            self.agent.reset_tracking()

        self.prev_hp = hp
        self.prev_room = ridx
        self.prev_clear = bool(room.get("clear"))
        self.prev_enemy_hp = cur
        self.prev_enemy_count = len(cur)
        if ridx is not None:
            self.rooms_seen.add(ridx)

    # ---------- 菜单/死亡 ----------
    def service_meta(self, s, now, auto_start=True):
        """没有对局时按 Enter / 发 restart，把游戏推进对局。"""
        if not auto_start:
            return
        need = (not s) or s.get("menu") or ("player" in s and hp_of(s) <= 0) \
            or ("player" not in s)
        if not need:
            return
        if now - self.last_menu_poke > 2.0:
            self.last_menu_poke = now
            try:
                mi.press_enter(self.inst.hwnd)
            except Exception:
                pass
        if now - self.last_restart > 5.0:
            self.last_restart = now
            try:
                self.ch.send("restart")
            except Exception:
                pass

    # ---------- 决策 ----------
    def step(self, now, hz):
        s = self.ch.state()
        seq = self.ch.seq()
        if s is not None and seq != self.last_seq:
            self.last_seq = seq
            self.last_change = now

        self.service_meta(s, now, auto_start=self.auto_start)

        if s is None or "player" not in s or "error" in s:
            if s is not None and "error" in s and not getattr(self, "_err_shown", False):
                print(f"  实例{self.index}: 模组报错 {s['error']}")
                self._err_shown = True
            self._release()
            return

        # 先统计再判断死亡：死亡那一帧也要计入 deaths / 受伤
        self.observe(s)

        if hp_of(s) <= 0:
            self.last_alive = False
            self._release()
            return

        self.last_alive = True

        if now - self.last_decision < 1.0 / max(1.0, hz):
            return
        self.last_decision = now

        try:
            a = self.agent.act(s)
        except Exception:
            self.failed = True
            self._release()
            print(f"  实例{self.index}: 战术 AI 异常（已停用该实例）")
            traceback.print_exc()
            return

        self.ctrl.set_move(*a.move)
        self.ctrl.set_shoot(*a.shoot)
        # 换房那一帧的 keydown 可能被游戏丢掉，周期性补发（否则会"假按着"）
        try:
            self.ctrl.refresh()
        except AttributeError:
            pass
        self.last_state = s
        self.last_action = a

        if a.shoot != (0, 0):
            self.shot_frames += 1
            if self.prev_shoot == (0, 0):
                self.shot_bursts += 1
        self.prev_shoot = a.shoot

    def _release(self):
        if self.ctrl is not None:
            try:
                self.ctrl.release_all()
            except Exception:
                pass
        self.prev_shoot = (0, 0)

    # ---------- 控制台 ----------
    def status_line(self):
        s = self.last_state
        if not s or "player" not in s:
            return (f"实例{self.index} port={self.inst.port} 等待对局 "
                    f"（死亡 {self.deaths}，清房 {self.rooms_cleared}）")
        room = s.get("room") or {}
        enemies, proj, _pk, _hz = split_entities(s)
        a = self.last_action
        mode = a.mode if a is not None else ""
        extra = ""
        if a is not None and a.debug:
            miss = a.debug.get("aim_miss")
            if miss is not None:
                extra += f" miss={miss:.0f}"
            thr = a.debug.get("threat")
            if thr:
                extra += f" 威胁{thr:.1f}"
            if a.debug.get("reposition"):
                extra += " 换位"
        return (f"实例{self.index} port={self.inst.port} "
                f"房{room.get('index')} HP={hp_of(s):.0f} "
                f"敌{len(enemies)} 弹{len(proj)} {mode}{extra} "
                f"受伤{self.damage_taken:.0f} 输出{self.damage_dealt:.0f} "
                f"击杀{self.kills} 清房{self.rooms_cleared} 死{self.deaths}")

    def summary(self):
        mins = max(1e-6, (time.time() - self.started) / 60.0)
        return (f"实例{self.index}: 清房 {self.rooms_cleared}，击杀≈{self.kills}，"
                f"受伤 {self.damage_taken:.0f}，输出≈{self.damage_dealt:.0f}，"
                f"死亡 {self.deaths}，开火 {self.shot_bursts} 次/"
                f"{self.shot_frames} 决策帧，经过 {len(self.rooms_seen)} 个房间"
                f"（{mins:.1f} 分钟）")


def wait_runs(runners, timeout=180.0, auto_start=True):
    """等所有实例真正进入对局（有 player 且 HP>0），而不是只连上 socket。"""
    t0 = time.time()
    last_note = 0.0
    last_poke = 0.0
    while time.time() - t0 < timeout:
        now = time.time()
        ready = [r for r in runners if in_run(r.ch.state())]
        if len(ready) == len(runners):
            return True
        if auto_start and now - last_poke > 2.0:
            last_poke = now
            for r in runners:
                s = r.ch.state()
                if in_run(s):
                    continue
                try:
                    mi.press_enter(r.inst.hwnd)
                except Exception:
                    pass
                # 多开时端口还没对号，往通道发 restart 可能重启别人的对局；
                # 这时只按 Enter，开局交给模组自己的 AUTO_START。
                if len(runners) == 1:
                    try:
                        r.ch.send("restart")
                    except Exception:
                        pass
        if now - last_note > 5.0:
            last_note = now
            parts = []
            for r in runners:
                s = r.ch.state()
                if in_run(s):
                    parts.append(f"#{r.index}:对局中")
                elif s and s.get("menu"):
                    parts.append(f"#{r.index}:菜单(尝试{s.get('start_attempts', 0)})")
                elif s and "hello" in s:
                    parts.append(f"#{r.index}:已连接")
                elif s:
                    parts.append(f"#{r.index}:数据")
                else:
                    parts.append(f"#{r.index}:未连接")
            print(f"  [{now - t0:5.0f}s] {' | '.join(parts)}"
                  f"{'（自动按 Enter）' if auto_start else ''}")
        time.sleep(0.25)
    return False


def main():
    ap = argparse.ArgumentParser(description="战术 AI 实机游玩")
    ap.add_argument("--launch", type=int, default=0,
                    help="启动 N 个新实例（0 = 优先接管已开着的）")
    ap.add_argument("--attach", action="store_true",
                    help="只接管已开着的实例，不启动新游戏")
    ap.add_argument("--attach-all", action="store_true",
                    help="接管所有 Isaac 窗口（默认只接管一个）")
    ap.add_argument("--exe", default=mi.DEFAULT_EXE)
    ap.add_argument("--base-port", type=int, default=48532)
    ap.add_argument("--hz", type=float, default=30.0,
                    help="决策频率（默认 30，游戏 60 帧）")
    ap.add_argument("--duration", type=float, default=0.0,
                    help="运行多少秒后自动停（0 = 一直跑）")
    ap.add_argument("--close", action="store_true",
                    help="退出时关掉由本脚本启动的游戏")
    ap.add_argument("--focus", action="store_true",
                    help="调试用：启动时抢一次焦点（默认绝不抢焦点）")
    ap.add_argument("--no-auto-start", action="store_true",
                    help="不自动按 Enter / restart，由你手动进对局")
    ap.add_argument("--log", default=None,
                    help="把状态摘要写到 JSONL 文件")
    ap.add_argument("--quiet", action="store_true", help="少打印")
    args = ap.parse_args()

    mgr = None
    launched = False
    runners = []
    log_fp = None
    if args.log:
        os.makedirs(os.path.dirname(os.path.abspath(args.log)) or ".", exist_ok=True)
        log_fp = open(args.log, "w", encoding="utf-8")

    def log(obj):
        if log_fp:
            log_fp.write(json.dumps(obj, ensure_ascii=False) + "\n")
            log_fp.flush()

    try:
        wins = mi.isaac_windows()
        if args.launch > 0 and wins:
            # 端口段是固定的（48532 起），旧实例会先抢到 48532，导致脚本把旧游戏
            # 的状态当成新实例，而按键却发到新窗口 —— 表现为"AI 完全不动"。
            print(f"✗ 启动新实例前发现已有 {len(wins)} 个 Isaac 窗口。")
            print("  旧实例会抢占 48532 端口，造成状态和窗口错配。请先关掉旧游戏；")
            print("  或改用 --attach / --attach-all 接管已开着的实例。")
            return 1
        if args.launch <= 0 and wins:
            fg = user32.GetForegroundWindow()
            if not args.attach_all:
                wins = sorted(wins, key=lambda w: 0 if w[0] == fg else 1)[:1]
            mgr = mi.InstanceManager(n=len(wins), base_port=args.base_port,
                                     exe=args.exe)
            mgr.attach()
            print(f"接管 {len(mgr.instances)} 个已开着的实例")
        elif args.launch <= 0 and args.attach:
            print("✗ 没找到 Isaac 窗口。请先打开游戏，或用 --launch 1 启动。")
            return 1
        else:
            n = args.launch if args.launch > 0 else 1
            print(f"==== 启动 {n} 个实例 ====")
            print(f"exe: {args.exe}")
            mgr = mi.InstanceManager(n=n, base_port=args.base_port, exe=args.exe)
            mgr.launch_all(wait_runs=False, verbose=not args.quiet)
            launched = True

        # 关键：最小化窗口会让游戏暂停，先恢复（不抢焦点）
        for inst in mgr.instances:
            restore_window(inst.hwnd, verbose=not args.quiet)
        if args.focus:
            fg = user32.GetForegroundWindow()
            try:
                user32.SetForegroundWindow(mgr.instances[0].hwnd)
                print("  --focus：已尝试把游戏切到前台")
                if fg and fg != mgr.instances[0].hwnd:
                    user32.SetForegroundWindow(fg)
            except Exception:
                pass

        # 先等模组连上来（菜单也会连），再等真正进入对局
        print("等模组连接 ...")
        if not mgr.wait_connections(60.0, verbose=not args.quiet,
                                    auto_start=not args.no_auto_start):
            print("✗ 60 秒内没有模组连上。")
            print("  若游戏是直接双击 exe 开的，可能没带 --luadebug，模组 socket 用不了；")
            print("  请用 `python play_ai.py --launch 1` 启动。")
            return 2

        # 先用"初始端口"建临时 runner，把每个窗口都推进对局。
        # 对号之前 inst.port 可能和窗口不一致，但这一步只关心"每个端口都有人"。
        temp = []
        for k, inst in enumerate(mgr.instances):
            ch = mgr.bridge.channels.get(inst.port)
            if ch is None:
                print(f"✗ 实例{k} 没有端口通道（port={inst.port}）")
                return 3
            temp.append(Runner(inst, ch, k, auto_start=not args.no_auto_start))

        print("等进入对局（会自动按 Enter；也可手动在窗口里开局）...")
        if not wait_runs(temp, timeout=180.0,
                         auto_start=not args.no_auto_start):
            print("✗ 3 分钟内没有全部进入对局。")
            print("  窗口是否被最小化/游戏是否暂停在菜单？可试 --focus 后手动开一局。")
            return 4

        # 多开：用短促方向键探针把窗口↔端口对上号
        if len(mgr.instances) > 1:
            print("多开对号（会向每个窗口发短促的方向键）...")
            if not mgr.resolve_mapping(verbose=not args.quiet):
                print("✗ 窗口↔端口对号失败。为避免两个 AI 抢同一个窗口，已停止。")
                return 3
            runners = []
            for k, inst in enumerate(mgr.instances):
                ch = mgr.bridge.channels.get(inst.port)
                if ch is None:
                    print(f"✗ 实例{k} 没有端口通道（port={inst.port}）")
                    return 3
                runners.append(Runner(inst, ch, k,
                                      auto_start=not args.no_auto_start))
        else:
            runners = temp

        print("\n==== 开始游玩（Ctrl+C 停止）====")
        for r in runners:
            print(f"  {r.inst!r}")
        print()

        t0 = time.time()
        last_report = 0.0
        last_alive_check = 0.0
        stale_warned = set()

        while True:
            now = time.time()
            if args.duration and now - t0 >= args.duration:
                print(f"\n到达 --duration {args.duration:.0f}s，停止。")
                break

            alive_runners = 0
            for r in runners:
                if r.failed:
                    continue
                alive_runners += 1
                try:
                    r.step(now, args.hz)
                except Exception:
                    r.failed = True
                    r._release()
                    print(f"  实例{r.index}: 主循环异常（已停用该实例）")
                    traceback.print_exc()

                # 状态长时间不更新：游戏可能被最小化/暂停了
                if now - r.last_change > 1.5 and r.index not in stale_warned:
                    stale_warned.add(r.index)
                    if not args.quiet:
                        print(f"  实例{r.index}: 1.5s 没有新状态 —— 可能被最小化/暂停，"
                              f"尝试恢复窗口 ...")
                        restore_window(r.inst.hwnd)
                elif now - r.last_change < 1.0:
                    stale_warned.discard(r.index)

            if not alive_runners:
                print("所有实例都已停用，退出。")
                break

            # 游戏窗口没了就退出（attach 场景；launch 场景进程退出也会关窗口）
            if now - last_alive_check > 2.0:
                last_alive_check = now
                if not mi.isaac_windows():
                    print("Isaac 窗口已不存在，退出。")
                    break

            if not args.quiet and now - last_report > 3.0:
                last_report = now
                for r in runners:
                    print("  " + r.status_line())
                    s = r.last_state or {}
                    p = s.get("player") or {}
                    a = r.last_action
                    log({"t": round(now - t0, 2), "inst": r.index,
                         "hp": hp_of(s) if s else None,
                         "room": (s.get("room") or {}).get("index"),
                         "x": p.get("x"), "y": p.get("y"),
                         "gx": p.get("grid_idx"),
                         "mode": a.mode if a else None,
                         "move": list(a.move) if a else None,
                         "shoot": list(a.shoot) if a else None,
                         "enemies": (s.get("room") or {}).get("enemies_alive"),
                         "damage_taken": round(r.damage_taken, 1),
                         "damage_dealt": round(r.damage_dealt, 1),
                         "kills": r.kills, "cleared": r.rooms_cleared,
                         "deaths": r.deaths,
                         "shots": r.shot_bursts})

            time.sleep(0.003)

    except KeyboardInterrupt:
        print("\n收到 Ctrl+C，停止。")
    except Exception:
        print("\n运行异常：")
        traceback.print_exc()
    finally:
        for r in runners:
            r._release()
        if mgr is not None:
            for inst in mgr.instances:
                try:
                    inst.controller.release_all()
                except Exception:
                    pass
            if launched and args.close:
                print("关闭本脚本启动的游戏 ...")
                mgr.stop_all()
            elif launched:
                print("游戏保持运行（--close 才会关闭）。")
            else:
                try:
                    mgr.bridge.close()
                except Exception:
                    pass
        if log_fp:
            log_fp.close()
        if runners:
            print("\n==== 本次统计 ====")
            for r in runners:
                print("  " + r.summary())

    return 0


if __name__ == "__main__":
    sys.exit(main())
