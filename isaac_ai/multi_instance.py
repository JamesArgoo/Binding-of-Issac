# -*- coding: utf-8 -*-
"""multi_instance.py —— Isaac 多开的基础设施

当前 AI 的所有“游戏外”操作都在这里：找窗口、连端口、注入按键、管理实例。
解决三个问题：
  1. 端口分配：每个实例连自己的端口（模组在 48532..48547 里自己找）
  2. 窗口映射：Python 顺序启动实例 + 顺序等连接，天然得到 实例号 ↔ PID ↔ 窗口句柄
  3. 输入注入：PostMessage 直接投递到目标窗口，不依赖窗口焦点

为什么输入必须走 PostMessage：
  多开时只有一个窗口有焦点，pyautogui 只会打进焦点窗口。PostMessage 把
  WM_KEYDOWN/WM_KEYUP 直接塞进目标窗口的消息队列，绕过焦点限制。

用法：
    mgr = InstanceManager(n=2)
    mgr.launch_all()          # 顺序启动并建立映射
    inst = mgr.instances[0]
    inst.controller.set_move(1, 0)   # 往右
    ...
    mgr.stop_all()
"""
import ctypes
import json
import os
import socket
import subprocess
import threading
import time
from ctypes import wintypes

# ---------------- Win32 ----------------
user32 = ctypes.WinDLL("user32", use_last_error=True)

WM_KEYDOWN = 0x0100
WM_KEYUP = 0x0101
MAPVK_VK_TO_VSC = 0

user32.PostMessageW.restype = wintypes.BOOL
user32.PostMessageW.argtypes = [
    wintypes.HWND, wintypes.UINT, wintypes.WPARAM, wintypes.LPARAM]
user32.MapVirtualKeyW.restype = wintypes.UINT
user32.MapVirtualKeyW.argtypes = [wintypes.UINT, wintypes.UINT]
user32.IsWindow.restype = wintypes.BOOL
user32.IsWindow.argtypes = [wintypes.HWND]
user32.IsWindowVisible.argtypes = [wintypes.HWND]
user32.GetWindowTextLengthW.argtypes = [wintypes.HWND]
user32.GetWindowTextW.argtypes = [wintypes.HWND, wintypes.LPWSTR, ctypes.c_int]
user32.GetClassNameW.argtypes = [wintypes.HWND, wintypes.LPWSTR, ctypes.c_int]
user32.GetWindowThreadProcessId.argtypes = [
    wintypes.HWND, ctypes.POINTER(wintypes.DWORD)]

EnumWindowsProc = ctypes.WINFUNCTYPE(wintypes.BOOL, wintypes.HWND, wintypes.LPARAM)
user32.EnumWindows.argtypes = [EnumWindowsProc, wintypes.LPARAM]

# 按键名 -> (虚拟键码, 是否扩展键)
VK_MAP = {
    "w": (0x57, False), "a": (0x41, False), "s": (0x53, False), "d": (0x44, False),
    "up": (0x26, True), "down": (0x28, True), "left": (0x25, True), "right": (0x27, True),
    "e": (0x45, False),          # 默认炸弹键
}


def press_enter(hwnd, vk=0x0D):
    """给指定窗口投递一次 Enter。

    用途：模组的 onUpdate 在主菜单不触发，游戏不会自己进对局，
    所以要从外部按键把它推进去（实测有效，可全自动开局）。
    """
    if not hwnd:
        return False
    scan = user32.MapVirtualKeyW(vk, MAPVK_VK_TO_VSC)
    user32.PostMessageW(hwnd, WM_KEYDOWN, vk, 1 | (scan << 16))
    time.sleep(0.04)
    user32.PostMessageW(hwnd, WM_KEYUP, vk,
                        1 | (scan << 16) | (1 << 30) | (1 << 31))
    return True


def list_windows():
    """返回 [(hwnd, pid, class_name, title), ...]，只保留可见窗口。"""
    out = []

    @EnumWindowsProc
    def _cb(hwnd, _lparam):
        if not user32.IsWindowVisible(hwnd):
            return True
        n = user32.GetWindowTextLengthW(hwnd)
        if n <= 0:
            return True
        tbuf = ctypes.create_unicode_buffer(n + 1)
        user32.GetWindowTextW(hwnd, tbuf, n + 1)
        cbuf = ctypes.create_unicode_buffer(256)
        user32.GetClassNameW(hwnd, cbuf, 256)
        pid = wintypes.DWORD()
        user32.GetWindowThreadProcessId(hwnd, ctypes.byref(pid))
        out.append((hwnd, pid.value, cbuf.value, tbuf.value))
        return True

    user32.EnumWindows(_cb, 0)
    return out


def isaac_windows():
    """所有 Isaac 游戏窗口。

    必须同时按窗口类名 GLFW30 过滤 —— 只按标题里的 "isaac" 匹配会误伤：
    实测文件资源管理器的标题「AIsaac 和 1 个其他选项卡」也含 "isaac"，
    被当成了第 3 个游戏实例，导致"已连上 2/3"永远等不到。
    """
    return [w for w in list_windows()
            if w[2] == "GLFW30" and "isaac" in w[3].lower()]


def windows_for_pid(pid):
    """某个进程的所有可见窗口句柄。"""
    return [w[0] for w in list_windows() if w[1] == pid]


def wait_for_window(pid, timeout=90.0, poll=0.5):
    """等某个进程的窗口出现（游戏启动到出窗口要十几秒）。"""
    t0 = time.time()
    while time.time() - t0 < timeout:
        hwnds = windows_for_pid(pid)
        if hwnds:
            return hwnds[0]
        time.sleep(poll)
    return None


def wait_for_new_isaac_window(before_hwnds, timeout=120.0, poll=0.5):
    """等一个"之前不存在"的 Isaac 窗口出现。

    不能按 Popen 的 PID 找窗口：绕过 Steam 直接启动 exe 时，Steam 会重新拉起
    真正的游戏进程，PID 完全不是 Popen 返回的那个。用窗口差集才靠谱。
    返回 (hwnd, pid)；超时返回 (None, None)。
    """
    t0 = time.time()
    while time.time() - t0 < timeout:
        for hwnd, pid, _cls, _title in isaac_windows():
            if hwnd not in before_hwnds:
                return hwnd, pid
        time.sleep(poll)
    return None, None


# ---------------- 输入注入 ----------------
class Win32KeyController:
    """和 isaac_env.KeyController 同接口，但往指定窗口 PostMessage。

    多开时每个实例一个实例化对象，互不干扰，且不要求窗口有焦点。
    """

    def __init__(self, hwnd):
        self.hwnd = hwnd
        self._move = set()
        self._shoot = set()
        self._down = set()      # 记录已经按下的键，避免重复发 keydown
        self._last_refresh = 0.0

    def _post(self, key, down):
        vk, extended = VK_MAP[key]
        scan = user32.MapVirtualKeyW(vk, MAPVK_VK_TO_VSC)
        if down:
            lparam = 1 | (scan << 16) | ((1 if extended else 0) << 24)
        else:
            lparam = 1 | (scan << 16) | ((1 if extended else 0) << 24) | (1 << 30) | (1 << 31)
        return bool(user32.PostMessageW(
            self.hwnd, WM_KEYDOWN if down else WM_KEYUP, vk, lparam))

    def _apply(self, current, want):
        for k in current - want:
            self._post(k, False)
        for k in want - current:
            self._post(k, True)
        return want

    def set_move(self, mx, my):
        want = set()
        if mx > 0:
            want.add("d")
        elif mx < 0:
            want.add("a")
        if my > 0:
            want.add("s")
        elif my < 0:
            want.add("w")
        self._move = self._apply(self._move, want)

    def set_shoot(self, ax, ay):
        want = set()
        if ax > 0:
            want.add("right")
        elif ax < 0:
            want.add("left")
        if ay > 0:
            want.add("down")
        elif ay < 0:
            want.add("up")
        self._shoot = self._apply(self._shoot, want)

    def tap(self, key, hold=0.05):
        """短按一次（炸弹/主动道具这类一次性动作不能按住不放）。"""
        self._post(key, True)
        time.sleep(hold)
        self._post(key, False)
        return True

    def tap_bomb(self):
        return self.tap("e")

    def release_all(self):
        for k in list(self._move) + list(self._shoot):
            self._post(k, False)
        self._move.clear()
        self._shoot.clear()

    def refresh(self, every=0.35):
        """周期性补发 keydown。

        换房/过场那一帧投递的 keydown 可能被游戏丢掉；而 _apply 只在
        状态切换时发一次，之后就会一直"以为按着"，实际游戏里是松开的。
        实测表现为 AI 站在空地上一动不动。重复 keydown 等价于键盘自动
        重复，没有副作用。
        """
        now = time.time()
        if now - self._last_refresh < every:
            return
        self._last_refresh = now
        for k in self._move | self._shoot:
            self._post(k, True)


# ---------------- 多端口桥接 ----------------
class PortChannel:
    """一个端口上的一条游戏连接（双向：收状态 + 下发命令）。"""

    def __init__(self, port):
        self.port = port
        self.latest = None
        self.connected = False
        self._lock = threading.Lock()
        self._conn = None
        # 条件变量：模组推来新状态时唤醒等待的 env 线程。
        # 用它替代忙轮询 —— 忙轮询会抢光 GIL（实测把吞吐从 52 打到 16）。
        self._cond = threading.Condition()
        self._seq = 0

    def set(self, msg):
        with self._lock:
            self.latest = msg
            self.connected = True
        with self._cond:
            self._seq += 1
            self._cond.notify_all()

    def seq(self):
        with self._cond:
            return self._seq

    def wait_new(self, after, timeout=0.1):
        """阻塞到"有比 after 更新的状态到达"，最多等 timeout。返回当前 seq。"""
        with self._cond:
            if self._seq <= after:
                self._cond.wait(timeout)
            return self._seq

    def state(self):
        with self._lock:
            return self.latest

    def attach(self, conn):
        with self._lock:
            self._conn = conn

    def detach(self):
        with self._lock:
            self._conn = None

    def send(self, text):
        """往游戏端发一行命令：restart / goto d.12 / exec <控制台命令>。

        模组每帧轮询一次 socket，所以发出去后下一帧就会执行。
        """
        with self._lock:
            conn = self._conn
        if conn is None:
            return False
        try:
            conn.sendall((text.rstrip("\n") + "\n").encode("utf-8"))
            return True
        except OSError:
            return False


class MultiPortBridge:
    """在 BASE_PORT..+N-1 上监听。

    关键规则：**每个端口只接受一条连接**。第二个连上来的立刻被踢掉，
    这样后启动的游戏实例会被挤到下一个空闲端口，端口分配自动确定。
    """

    def __init__(self, base_port=48532, max_instances=16, host="127.0.0.1"):
        self.base_port = base_port
        self.max_instances = max_instances
        self.host = host
        self.channels = {}          # port -> PortChannel
        self._servers = {}          # port -> socket
        self._running = True
        self._lock = threading.Lock()

    def open_port(self, port):
        """开始监听某个端口（按需逐个打开）。"""
        with self._lock:
            if port in self._servers:
                return self.channels[port]
            ch = PortChannel(port)
            srv = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            srv.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            srv.bind((self.host, port))
            srv.listen(4)
            srv.settimeout(1.0)
            self._servers[port] = srv
            self.channels[port] = ch
        threading.Thread(target=self._accept_loop, args=(srv, ch), daemon=True).start()
        return ch

    def _accept_loop(self, srv, ch):
        port = ch.port
        while self._running:
            try:
                conn, _ = srv.accept()
            except socket.timeout:
                continue
            except OSError:
                break
            # 该端口已经有主了：直接踢掉新来的，让后启动的实例去试下一个端口
            if ch.connected:
                try:
                    conn.close()
                except OSError:
                    pass
                continue
            threading.Thread(target=self._read_loop, args=(conn, ch),
                             daemon=True).start()

    def _read_loop(self, conn, ch):
        buf = b""
        conn.settimeout(1.0)
        ch.attach(conn)
        try:
            while self._running:
                try:
                    data = conn.recv(65536)
                except socket.timeout:
                    continue
                except OSError:
                    break
                if not data:
                    break
                buf += data
                while b"\n" in buf:
                    line, buf = buf.split(b"\n", 1)
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        msg = json.loads(line.decode("utf-8"))
                    except (json.JSONDecodeError, UnicodeDecodeError):
                        continue
                    ch.set(msg)
        finally:
            ch.detach()
            try:
                conn.close()
            except OSError:
                pass
            with ch._lock:
                ch.connected = False

    def wait_connected(self, port, timeout=90.0):
        """等某个端口上的游戏连上来并发出一帧数据。"""
        ch = self.channels.get(port)
        if ch is None:
            return False
        t0 = time.time()
        while time.time() - t0 < timeout:
            if ch.state() is not None:
                return True
            time.sleep(0.1)
        return False

    def close(self):
        self._running = False
        for srv in list(self._servers.values()):
            try:
                srv.close()
            except OSError:
                pass
        self._servers.clear()


# ---------------- 实例管理器 ----------------
DEFAULT_EXE = (r"D:\SteamLibrary\steamapps\common"
               r"\The Binding of Isaac Rebirth\isaac-ng.exe")


class Instance:
    """一个被托管的游戏实例。"""

    def __init__(self, index, port, proc=None):
        self.index = index
        self.port = port
        self.proc = proc
        self.pid = proc.pid if proc else None
        self.hwnd = None
        self.controller = None

    def __repr__(self):
        return (f"Instance#{self.index} pid={self.pid} port={self.port} "
                f"hwnd={self.hwnd}")


class InstanceManager:
    """管理 N 个游戏实例：启动、端口分配、窗口认领、端口↔窗口对号。

    启动流程（并行友好，不用按顺序一个个开局）：
      1. 先把 N 个端口全部开好，再逐个启动 exe
      2. 每启动一个，用"新出现的 Isaac 窗口"认领实例号
         （Popen 返回的 PID 不可信：Steam 会重新拉起真正的游戏进程）
      3. 你在任意窗口里开局都行，等 N 个端口都连上
      4. 用探针把 端口 和 窗口 对上号 —— 往某个窗口发方向键，看哪个端口的玩家动了
    """

    PROBE_DIRS = [(1, 0), (0, 1), (-1, 0), (0, -1), (1, 1), (-1, -1), (1, -1), (-1, 1)]

    def __init__(self, n=1, base_port=48532, exe=DEFAULT_EXE,
                 max_instances=16, launch_timeout=120.0,
                 extra_args=("--luadebug",), injector=None, dll=None,
                 env_extra=None, viewer_count=0):
        if n < 1 or n > max_instances:
            raise ValueError(f"n 必须在 1..{max_instances} 之间")
        self.n = n
        self.exe = exe
        self.launch_timeout = launch_timeout
        # 绕过 Steam 直接启动 exe 时必须自己带 --luadebug，否则模组里的 socket 用不了
        self.extra_args = list(extra_args)
        # 给了 injector+dll 就用"挂起启动 + 注入"来起进程（turbo 加速层）
        self.injector = injector
        self.dll = dll
        self.env_extra = dict(env_extra or {})
        # 最后 viewer_count 个实例不开 turbo：正常速度 + 正常渲染，供人实时观看
        self.viewer_count = max(0, int(viewer_count))
        self.bridge = MultiPortBridge(base_port, max_instances)
        self.instances = []
        self._started = False
        self.mapping_resolved = False

    # ---------- 启动 ----------
    def launch_all(self, verbose=True, wait_runs=False, run_timeout=900.0):
        """启动 N 个实例并认领窗口；wait_runs=True 时再等对局、做端口对号。"""
        for k in range(self.n):
            self.bridge.open_port(self.bridge.base_port + k)

        for k in range(self.n):
            before = {w[0] for w in isaac_windows()}
            use_turbo = (self.injector and self.dll
                         and k < self.n - self.viewer_count)
            if verbose:
                kind = "turbo(不渲染)" if use_turbo else "正常渲染(观看用)"
                print(f"[{k + 1}/{self.n}] 启动实例 {kind} ...")
            # Steam 伪装：不设这几个变量，游戏会调 RestartAppIfNecessary 自己退出
            env = os.environ.copy()
            env.setdefault("SteamAppId", "250900")
            env.setdefault("SteamGameId", "250900")
            env.setdefault("SteamClientLaunch", "1")
            if self.injector and self.dll:
                env.update(self.env_extra)
                if not use_turbo:
                    env["ISAAC_TURBO_SPEED"] = "1"
                    env["ISAAC_TURBO_SKIP_PRESENT"] = "0"
                proc = subprocess.Popen(
                    [self.injector, self.dll, self.exe] + self.extra_args,
                    cwd=os.path.dirname(self.exe), env=env,
                    stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            else:
                proc = subprocess.Popen([self.exe] + self.extra_args,
                                        cwd=os.path.dirname(self.exe), env=env)
            inst = Instance(k, self.bridge.base_port + k, proc)
            self.instances.append(inst)

            hwnd, pid = wait_for_new_isaac_window(before, self.launch_timeout)
            if hwnd is None:
                raise RuntimeError(
                    f"实例 {k} 启动后 {self.launch_timeout}s 内没出现新窗口。")
            inst.hwnd, inst.pid = hwnd, pid
            inst.controller = Win32KeyController(hwnd)
            if verbose:
                print(f"        窗口已认领 {inst!r}（端口待对号）")

        self._started = True
        if wait_runs:
            if not self.wait_connections(run_timeout, verbose=verbose):
                raise RuntimeError(
                    f"{run_timeout}s 内没有全部连上 —— 确认每个窗口都开局了")
            self.resolve_mapping(verbose=verbose)
        return self.instances

    # ---------- 接管已在运行的游戏 ----------
    def attach(self, verbose=True):
        """不启动新游戏，直接接管当前所有开着的 Isaac 窗口。"""
        wins = isaac_windows()
        if not wins:
            raise RuntimeError("没有找到 Isaac 窗口")
        wins = wins[:self.bridge.max_instances]
        for k, (hwnd, pid, _cls, _title) in enumerate(wins):
            self.bridge.open_port(self.bridge.base_port + k)
            inst = Instance(k, self.bridge.base_port + k, None)
            inst.hwnd, inst.pid = hwnd, pid
            inst.controller = Win32KeyController(hwnd)
            self.instances.append(inst)
        self.n = len(self.instances)
        self._started = True
        if verbose:
            print(f"接管了 {self.n} 个实例：")
            for inst in self.instances:
                print(f"  {inst}")
        return self.instances

    # ---------- 等对局 ----------
    def wait_connections(self, timeout=900.0, verbose=True, auto_start=True):
        """等所有端口都连上（等价于每个窗口都进了对局）。

        auto_start=True 时，对还没连上的窗口自动发 Enter 帮它开局
        （模组的 onUpdate 在主菜单不触发，所以只能靠外部按键把游戏推进对局）。
        """
        ports = [self.bridge.base_port + k for k in range(self.n)]
        t0 = time.time()
        last = 0.0
        last_press = 0.0
        while time.time() - t0 < timeout:
            done = [p for p in ports if self.bridge.channels[p].state() is not None]
            if len(done) == len(ports):
                return True
            # 自动开局：每 2 秒给"还没连上"的实例窗口按一次 Enter
            if auto_start and time.time() - last_press > 2.0:
                last_press = time.time()
                for inst in self.instances:
                    ch = self.bridge.channels.get(inst.port)
                    if ch is not None and ch.state() is not None:
                        continue
                    if inst.hwnd:
                        press_enter(inst.hwnd)
            if verbose and time.time() - last > 5.0:
                last = time.time()
                print(f"  [{time.time() - t0:5.0f}s] 已连上 {len(done)}/{len(ports)}"
                      f"{'（自动按 Enter 推进中）' if auto_start else ''}")
            time.sleep(0.3)
        return False

    # ---------- 端口 ↔ 窗口 对号 ----------
    def _player_xy(self, port):
        ch = self.bridge.channels.get(port)
        s = ch.state() if ch else None
        if s and "player" in s:
            return (s["player"]["x"], s["player"]["y"])
        return None

    def resolve_mapping(self, probes_per_instance=3, probe_s=0.7, settle_s=0.4,
                        verbose=True):
        """往实例的窗口发方向键，看哪个端口的玩家动了，把端口认领回来。

        每实例最多试几个方向（撞墙就换方向重试），命中即锁定。
        """
        ports = [self.bridge.base_port + k for k in range(self.n)]
        taken = {}
        for k, inst in enumerate(self.instances):
            if inst.hwnd is None:
                continue
            for attempt in range(probes_per_instance):
                dx, dy = self.PROBE_DIRS[(k + attempt) % len(self.PROBE_DIRS)]
                before = {p: self._player_xy(p) for p in ports}
                inst.controller.set_move(dx, dy)
                time.sleep(probe_s)
                inst.controller.set_move(0, 0)
                time.sleep(settle_s)
                after = {p: self._player_xy(p) for p in ports}

                hits = []
                for p in ports:
                    if p in taken.values():
                        continue
                    b, a = before[p], after[p]
                    if b is None or a is None:
                        continue
                    mvx, mvy = a[0] - b[0], a[1] - b[1]
                    if abs(mvx) > 3.0 and (dx == 0 or (mvx > 0) == (dx > 0)):
                        hits.append(p)
                    elif abs(mvy) > 3.0 and (dy == 0 or (mvy > 0) == (dy > 0)):
                        hits.append(p)
                if len(hits) == 1:
                    inst.port = hits[0]
                    taken[k] = hits[0]
                    if verbose:
                        print(f"  实例{k}: 窗口 {inst.hwnd} ↔ 端口 {hits[0]} ✅")
                    break
            else:
                if verbose:
                    print(f"  实例{k}: 对号失败（撞墙/有人在操作？）"
                          f"暂用 {inst.port}")

        self.mapping_resolved = len(taken) == len(self.instances)
        return self.mapping_resolved

    def state(self, index):
        return self.bridge.channels[self.instances[index].port].state()

    def stop_all(self):
        """停掉所有由本管理器启动的游戏进程。"""
        for inst in self.instances:
            if inst.controller:
                try:
                    inst.controller.release_all()
                except Exception:
                    pass
            if inst.proc and inst.proc.poll() is None:
                try:
                    inst.proc.terminate()
                except Exception:
                    pass
        for inst in self.instances:
            if inst.proc:
                try:
                    inst.proc.wait(timeout=10)
                except Exception:
                    try:
                        inst.proc.kill()
                    except Exception:
                        pass
        self.bridge.close()
        self.instances.clear()
        self._started = False


class ChannelAdapter:
    """把 MultiPortBridge 的单个端口包装成 IsaacBridge 的接口（get_state/stop），
    这样 IsaacEnv 既能单开自己起桥接，也能多开直接吃管理器分配好的端口。"""

    def __init__(self, channel):
        self.channel = channel

    def get_state(self):
        return self.channel.state()

    def send(self, text):
        """下发一行命令给游戏（restart / goto d.12 / exec ...）。"""
        return self.channel.send(text)

    def seq(self):
        return self.channel.seq()

    def wait_new(self, after, timeout=0.1):
        return self.channel.wait_new(after, timeout)

    def stop(self):
        # 生命周期由 InstanceManager 统一管理，这里不动
        pass


def make_envs(mgr, frame_skip=2):
    """用已启动的实例构造 gym 环境列表（一个实例一个环境）。"""
    # 旧 PPO 训练路径；当前战术 AI 不依赖 gym/numpy。
    # 旧模块之间仍用顶层导入，这里把 legacy/ppo 临时放进搜索路径。
    import sys
    legacy = os.path.join(os.path.dirname(os.path.dirname(
        os.path.abspath(__file__))), "legacy", "ppo")
    if legacy not in sys.path:
        sys.path.insert(0, legacy)
    from isaac_env import IsaacEnv
    envs = []
    for inst in mgr.instances:
        ch = mgr.bridge.channels[inst.port]
        envs.append(IsaacEnv(frame_skip=frame_skip,
                             bridge=ChannelAdapter(ch),
                             controller=inst.controller))
    return envs


if __name__ == "__main__":
    # 简易自检：只打印当前能看到什么
    print("==== 可见的 Isaac 窗口 ====")
    for hwnd, pid, cls, title in isaac_windows():
        print(f"  hwnd={hwnd} pid={pid} class={cls!r} title={title!r}")
    print(f"\n默认 exe: {DEFAULT_EXE}")
