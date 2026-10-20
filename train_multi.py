# -*- coding: utf-8 -*-
"""多开并行训练：N 个游戏实例同时采样，PPO 一起学。

为什么能加速：每个实例的游戏都按自己 60FPS 实时推进，N 个实例并行就是
N 倍的经验采集速度。单实例不加速（Windows 忏悔做不到虚拟时钟），
所以吞吐 ≈ N 倍，N 受 CPU/GPU 限制（24 核大概 8~12 个比较稳）。

用法：
    # 接管已经开着的游戏（每个窗口都要在对局里）
    python train_multi.py --attach --steps 20000

    # 自己启动 3 个实例（会等你在每个窗口开局）
    python train_multi.py --launch 3 --steps 100000

    # 从 BC 模型热启动（推荐：纯 RL 从零开始太慢）
    python train_multi.py --launch 3 --load models/isaac_ppo_bc --steps 200000
"""
import argparse
import os
import sys
import threading
import time

import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback, CheckpointCallback
from stable_baselines3.common.vec_env import VecEnv, VecMonitor

try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

import multi_instance as mi
from isaac_env import (OBS_DIM, TERRAIN_FULL, TERRAIN_LOCAL, IsaacEnv, obs_dim,
                       terrain_mode_from_dim)


def resolve_model_path(path, models_dir="models"):
    """把 --model 的输入解析成真实存在的模型文件。

    支持：
      * "latest" / None  -> 自动挑 models/ 下最新的 checkpoint
      * 少写 .zip 或多写 .zip 都能认（SB3 自己补 .zip 会拼成 foo.zip.zip，
        报错信息完全看不出问题在哪，这里提前处理掉）

    找不到时抛 FileNotFoundError，附上现有模型清单（按时间倒序）。
    """
    import glob
    import os

    def _pick_latest():
        cands = [p for p in glob.glob(os.path.join(models_dir, "*.zip"))]
        if not cands:
            raise FileNotFoundError(f"{models_dir}/ 里没有任何 .zip 模型")
        cands.sort(key=os.path.getmtime, reverse=True)
        return cands[0]

    if path is None or str(path).lower() in ("latest", "newest", "@latest"):
        return _pick_latest()

    p = str(path)
    for cand in (p, p + ".zip", p[:-4] if p.lower().endswith(".zip") else None):
        if cand and os.path.isfile(cand):
            return cand

    lines = [f"✗ 找不到模型文件：{p}"]
    cands = sorted(glob.glob(os.path.join(models_dir, "*.zip")),
                   key=os.path.getmtime, reverse=True)[:8]
    if cands:
        import time as _t
        lines.append(f"  {models_dir}/ 下最近的模型：")
        for c in cands:
            stamp = _t.strftime("%m-%d %H:%M", _t.localtime(os.path.getmtime(c)))
            lines.append(f"    {stamp}  {os.path.basename(c)}"
                         f"  ({os.path.getsize(c) // 1024} KB)")
        lines.append(f"  提示：直接用 --model latest 就会选最新的那个。")
    else:
        lines.append(f"  {models_dir}/ 下没有任何 .zip；训练中断时"
                     f"CheckpointCallback 会每 5000 步存一个，看看 models/ 目录。")
    raise FileNotFoundError("\n".join(lines))


def peek_obs_dim(path):
    """不开模型就读出它训练时的观测维度。

    SB3 存的 .zip 里 `data` 是 JSON，`observation_space` 序列化成
    {":type:": Box, "_shape": [...], ":serialized:": base64}，
    优先读 _shape，读不到再解 base64 的 pickle。
    用它对齐地形模式，避免"模型 1292 维、环境 445 维"这种加载后才炸的错。
    """
    import base64
    import json
    import pickle
    import zipfile

    with zipfile.ZipFile(path) as z:
        raw = z.read("data")
    try:
        data = json.loads(raw)
    except (ValueError, UnicodeDecodeError):
        import cloudpickle
        data = cloudpickle.loads(raw)

    sp = data["observation_space"]
    if isinstance(sp, dict):
        shape = sp.get("_shape")
        if shape is None and sp.get(":serialized:"):
            shape = pickle.loads(base64.b64decode(sp[":serialized:"])).shape
        if shape is None:
            raise ValueError("模型里的 observation_space 读不出形状")
        return int(shape[0])
    return int(sp.shape[0])


class ParallelIsaacVecEnv(VecEnv):
    """把 N 个实时 IsaacEnv 包成 SB3 能吃的向量环境。

    每个 step 用线程并行推进所有实例 —— 各自 sleep frame_skip/60，
    并行后总耗时 ≈ 单实例耗时，而不是 N 倍。
    """

    def __init__(self, envs):
        # 必须先挂上 self.envs 再调 super().__init__：
        # SB3 的 VecEnv.__init__ 里会 get_attr("render_mode")，它走的是
        # self.envs[i]；envs 还没赋值就会抛 AttributeError，被 SB3 捕获后
        # 打出那条 "render_mode attribute is not defined" 的 UserWarning。
        self.envs = envs
        super().__init__(len(envs), envs[0].observation_space, envs[0].action_space)
        self.render_mode = None
        self._actions = None
        self._lock = threading.Lock()

    def reset(self):
        obs = [e.reset()[0] for e in self.envs]
        return np.stack(obs)

    def step_async(self, actions):
        self._actions = np.asarray(actions)

    def step_wait(self):
        n = len(self.envs)
        results = [None] * n

        def work(i):
            try:
                results[i] = self.envs[i].step(self._actions[i])
            except Exception as e:      # 单个实例炸了不能拖垮整个训练
                print(f"[vecenv] 实例{i} step 异常: {e}")
                # 用该实例自己的观测维度补零：full 模式是 1296 维，
                # 写死 OBS_DIM（local 的 449）会让后面 np.stack 直接形状不匹配
                n_obs = int(self.envs[i].observation_space.shape[0])
                results[i] = (np.zeros(n_obs, dtype=np.float32), 0.0, True, False, {})

        threads = [threading.Thread(target=work, args=(i,)) for i in range(n)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        obs, rews, dones, infos = [], [], [], []
        for i, (o, r, term, trunc, info) in enumerate(results):
            terminal_obs = o
            if term or trunc:
                o, info = self.envs[i].reset()
                info = dict(info or {})
                info["terminal_observation"] = terminal_obs
                info["TimeLimit.truncated"] = bool(trunc and not term)
                dones.append(True)
            else:
                dones.append(False)
            obs.append(o)
            rews.append(float(r))
            infos.append(info)

        return (np.stack(obs),
                np.array(rews, dtype=np.float32),
                np.array(dones, dtype=bool),
                infos)

    def close(self):
        for e in self.envs:
            try:
                e.close()
            except Exception:
                pass

    # ---- SB3 VecEnv 要求的抽象方法 ----
    def _resolve(self, indices):
        return range(len(self.envs)) if indices is None else indices

    def get_attr(self, attr_name, indices=None):
        return [getattr(self.envs[i], attr_name) for i in self._resolve(indices)]

    def set_attr(self, attr_name, value, indices=None):
        for i in self._resolve(indices):
            setattr(self.envs[i], attr_name, value)

    def env_method(self, method_name, *method_args, indices=None, **method_kwargs):
        return [getattr(self.envs[i], method_name)(*method_args, **method_kwargs)
                for i in self._resolve(indices)]

    def env_is_wrapped(self, wrapper_class, indices=None):
        return [isinstance(self.envs[i], wrapper_class)
                for i in self._resolve(indices)]


def load_variants(path, min_enemies=None, max_enemies=None):
    """从 room_catalog.py 探出的目录里取有敌人的房间 variant 列表。

    min/max_enemies 用来做**课程学习**：先只练 1~3 个敌人的房间降低方差，
    练稳了再放开到全难度（AIsaac 的 TaskSampler 也是按权重分难度的思路）。
    """
    import json
    with open(path, encoding="utf-8") as f:
        data = json.load(f)
    combat = data.get("combat") or []
    picked = []
    for r in combat:
        n = int(r.get("n_enemies", 0))
        if n <= 0:
            continue
        if min_enemies is not None and n < min_enemies:
            continue
        if max_enemies is not None and n > max_enemies:
            continue
        picked.append(int(r["variant"]))
    variants = sorted(set(picked))
    if not variants:
        raise SystemExit(
            f"{path} 里没有符合条件的房间（min={min_enemies} max={max_enemies}），"
            f"先跑 room_catalog.py 或放宽条件")
    return variants


class ProgressCallback(BaseCallback):
    """每 N 秒打一行训练进度。

    SB3 默认只在每个 PPO 迭代**结束**时打一次日志表。一次迭代 =
    n_steps × n_envs 步，在真实游戏环境里（每秒几十步）要几十秒才出一行，
    看起来就像训练卡死了 —— 实测 4 实例时一次迭代约 34 秒。

    这里同时打印「本段速率」（最近一个窗口）和「平均速率」（全程），
    以及 reset 耗时 / 回合长度 —— 因为 ETA 会跳，根源就是回合结束时
    要传送进新房间，那段时间不产生任何步数。
    """

    def __init__(self, every=10.0):
        super().__init__()
        self.every = float(every)
        self.t0 = None
        self.last_t = None
        self.last_steps = 0
        self.episodes = 0
        self.ep_rews = []
        self.ep_hurts = []
        self.ep_dealts = []
        self.clear_cnt = 0
        self.hurtfail_cnt = 0

    def _on_step(self):
        now = time.time()
        if self.t0 is None:
            self.t0 = self.last_t = now
            self.last_steps = self.num_timesteps
            print("  （每个 PPO 迭代结束会打一张日志表；下面是滚动进度）")
            return True
        for info in (self.locals.get("infos") or []):
            ep = info.get("episode")
            if ep:
                self.episodes += 1
                self.ep_rews.append(float(ep.get("r", 0.0)))
                self.ep_rews = self.ep_rews[-30:]
                # 受伤 / 输出 / 清房率 是"学会躲没有"最直接的指标
                self.ep_hurts.append(float(info.get("hurt", 0.0) or 0.0))
                self.ep_hurts = self.ep_hurts[-30:]
                self.ep_dealts.append(float(info.get("ep_dealt", 0.0) or 0.0))
                self.ep_dealts = self.ep_dealts[-30:]
                if info.get("clear"):
                    self.clear_cnt += 1
                if info.get("hurt_fail"):
                    self.hurtfail_cnt += 1

        if now - self.last_t < self.every:
            return True

        dt_win = max(now - self.last_t, 1e-9)
        dt_all = max(now - self.t0, 1e-9)
        steps = self.num_timesteps
        rate_win = (steps - self.last_steps) / dt_win
        rate_avg = steps / dt_all
        self.last_t, self.last_steps = now, steps

        total = getattr(self.model, "_total_timesteps", None) or 0
        eta = ""
        if total and rate_avg > 0:
            eta = f"  剩余约 {(total - steps) / rate_avg / 60:.1f} 分钟"
        recent = (f"  最近回合回报 {sum(self.ep_rews) / len(self.ep_rews):+.2f}"
                  if self.ep_rews else "")

        # reset 耗时 / 回合长度（判断时间到底花在哪）
        extra = ""
        try:
            rs = [x for x in self.training_env.get_attr("last_reset_s") or []
                  if isinstance(x, (int, float))]
            ep_len = [x for x in self.training_env.get_attr("last_ep_steps") or []
                      if isinstance(x, (int, float))]
            if rs and sum(rs) > 0:
                extra = (f"  reset {sum(rs) / len(rs):.1f}s"
                         f" / 回合 {sum(ep_len) / max(1, len(ep_len)):.0f} 步")
        except Exception:
            pass

        print(f"  [{dt_all:6.0f}s] {steps:>7} 步  "
              f"本段 {rate_win:5.1f} / 平均 {rate_avg:5.1f} 步/秒  "
              f"完成 {self.episodes} 回合{recent}{extra}{eta}")
        if self.ep_hurts:
            avg_hurt = sum(self.ep_hurts) / len(self.ep_hurts)
            avg_dealt = sum(self.ep_dealts) / len(self.ep_dealts)
            clr = 100.0 * self.clear_cnt / max(1, self.episodes)
            hf = (f"  因受伤终止 {self.hurtfail_cnt}"
                  if self.hurtfail_cnt else "")
            print(f"           最近 30 回合：平均受伤 {avg_hurt:.2f} 点  "
                  f"平均输出 {avg_dealt:.0f} 伤害  清房率 {clr:.0f}%{hf}")
        return True


def measure_game_fps(adapter, seconds=1.5):
    """实测游戏逻辑帧率，用来推算真实的决策频率。

    项目原来的 fps 默认值 60/frame_skip 假设游戏跑满 60fps；但实测不少机器
    只有 ~30fps，这时真实决策频率是 15 而不是 30 —— 死亡惩罚和时间代价会
    恰好互相抵消，奖励就废了。所以启动时测一下，别再让人手填。
    """
    s = adapter.get_state()
    if not s or "frame" not in s:
        return None
    t0, f0 = time.time(), s["frame"]
    last = s
    while time.time() - t0 < seconds:
        cur = adapter.get_state()
        if cur and "frame" in cur:
            last = cur
        time.sleep(0.02)
    dt = time.time() - t0
    df = last["frame"] - f0
    if dt <= 0 or df <= 0:
        return None
    return df / dt


def build_envs(mgr, combat=False, catalog="room_catalog.json", frame_skip=2,
               min_enemies=None, max_enemies=None, chunk=1,
               terrain_mode="local", hit_rew=None, cost_miss=None, fps=None,
               reset_mode="restart", hurt_scale=None, fail_on_hurt=False):
    """用管理器里的实例构造环境列表。

    combat=True 用单房间战斗环境（照 AIsaac 的设定，推荐）；
    否则用旧的整层楼环境。

    terrain_mode: "local"（地形只看玩家周围 7×7，445 维）
                  "full" （地形铺满整间房 28×16×2 通道，1292 维）
    hit_rew / cost_miss: 奖励尺度。命中奖励和每秒时间代价的**比值**决定瞄准学不学得会
                  （selftest_learn.py --sweep 实测：每步瞄准信号 <0.1 时 PPO 学不动）
    """
    kw = {}
    if hit_rew is not None:
        kw["hit_rew"] = hit_rew
    if cost_miss is not None:
        kw["cost_miss"] = cost_miss
    if fps is not None:
        kw["fps"] = fps
    if hurt_scale is not None:
        kw["hurt_scale"] = hurt_scale
    kw["fail_on_hurt"] = bool(fail_on_hurt)
    # fail_on_hurt + keep 不相容：keep 不重开，血量会一路往下掉，
    # 于是"掉血即结束"的回合越跑越短、最后卡在濒死状态。这种组合强制整局重开。
    if fail_on_hurt and reset_mode == "keep":
        print("⚠ --fail-on-hurt 需要每回合满血，已把 reset-mode 从 keep 改回 restart")
        reset_mode = "restart"
    kw["reset_mode"] = reset_mode

    if combat:
        from combat_env import CombatEnv
        variants = load_variants(catalog, min_enemies, max_enemies)
        rng = ""
        if min_enemies is not None or max_enemies is not None:
            rng = f"（敌人数 {min_enemies or 0}~{max_enemies or '∞'} 的课程）"
        print(f"单房间战斗环境：{len(variants)} 个可用房间 variant{rng}，chunk={chunk}")
    else:
        from isaac_env import IsaacEnv

    envs = []
    for inst in mgr.instances:
        ch = mgr.bridge.channels[inst.port]
        adapter = mi.ChannelAdapter(ch)
        if combat:
            envs.append(CombatEnv(adapter, inst.controller,
                                  variants=variants, frame_skip=frame_skip,
                                  chunk=chunk, terrain_mode=terrain_mode, **kw))
        else:
            envs.append(IsaacEnv(frame_skip=frame_skip, bridge=adapter,
                                 controller=inst.controller))
    return envs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--attach", action="store_true", help="接管已在运行的游戏")
    ap.add_argument("--launch", type=int, default=0, help="启动 N 个新实例")
    ap.add_argument("--steps", type=int, default=50000)
    ap.add_argument("--load", default=None, help="从已有模型热启动（推荐用 BC 模型）")
    ap.add_argument("--resume", action="store_true", help="接着 --load 的模型继续训练")
    ap.add_argument("--out", default="models/isaac_ppo_multi")
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--ent-coef", type=float, default=0.01)
    ap.add_argument("--run-timeout", type=float, default=900.0)
    ap.add_argument("--combat", action="store_true",
                    help="用单房间战斗环境（照 AIsaac 的设定，推荐）")
    ap.add_argument("--catalog", default="room_catalog.json",
                    help="room_catalog.py 探出的房间目录，--combat 时使用")
    ap.add_argument("--min-enemies", type=int, default=None,
                    help="课程学习：只练敌人数量 >= N 的房间")
    ap.add_argument("--max-enemies", type=int, default=None,
                    help="课程学习：只练敌人数量 <= N 的房间（先易后难）")
    ap.add_argument("--chunk", type=int, default=1,
                    help="action chunking：一次决策输出连续 N 组动作（借鉴 ACT，建议 2~4）")
    ap.add_argument("--reset-mode", default="restart",
                    choices=["restart", "keep"],
                    help="restart=每回合整局重开（照 AIsaac，保证满血）；"
                         "keep=玩家活着就只 goto 下一间房，不重开。"
                         "实测回合只有 ~180 步（6 秒）而重开要 2-4 秒，"
                         "keep 能省掉 30-40% 的训练时间")
    ap.add_argument("--hurt-scale", type=float, default=None,
                    help="受伤惩罚倍率（默认 1.0）。--hit-rew 调到 1.0 后，一个回合"
                         "打出的伤害约 28 分、掉光血只扣 12 分，于是贴脸换血最优；"
                         "想让它学会躲，把这里拉到 10 左右")
    ap.add_argument("--fail-on-hurt", action="store_true",
                    help="「不允许受伤」：掉任何一点血就结束本回合（不加死亡惩罚）。"
                         "适合先练躲避的课程阶段")
    ap.add_argument("--save-every", type=int, default=25000,
                    help="每训练多少总步数存一个 checkpoint（默认 25000）。"
                         "多开 + 高帧率下吞吐能到几百步/秒，按步数存会一秒一个、"
                         "很快把硬盘塞满，用总步数而不是 vec-step 次数来算")
    ap.add_argument("--full-terrain", action="store_true",
                    help="地形铺满整间房（28×16 画布 × 可走/危险 2 通道，观测 445→1292），"
                         "而不是只看玩家周围 7×7")
    ap.add_argument("--hit-rew", type=float, default=None,
                    help="每点伤害的奖励，默认 0.2（= 每 5 点伤害 +1）。"
                         "调大它相对时间代价的比值，瞄准才学得会")
    ap.add_argument("--cost-miss", type=float, default=None,
                    help="命中率为 0 时每秒的时间代价，默认 2.0。调小它等于抬高命中奖励的相对权重")
    ap.add_argument("--fps", type=float, default=None,
                    help="每秒决策步数。不给就启动时实测游戏帧率再除以 frame_skip")
    ap.add_argument("--frame-skip", type=int, default=1,
                    help="每步等几个游戏帧（默认 1）。模组已改成每帧都发数据，"
                         "所以 1 能让决策频率等于游戏帧率")
    ap.add_argument("--n-steps", type=int, default=1024,
                    help="PPO 每次更新收集多少步（越大 advantage 估计越稳）")
    ap.add_argument("--batch", type=int, default=256, help="PPO minibatch 大小")
    ap.add_argument("--turbo", action="store_true",
                    help="用 turbo 加速层启动实例（虚拟时钟+跳渲染，实测约 5 倍）")
    ap.add_argument("--turbo-speed", type=float, default=20.0,
                    help="turbo 时钟倍速（20 和 60 效果一样，CPU 才是瓶颈）")
    ap.add_argument("--viewer", type=int, default=0,
                    help="最后 N 个实例不开 turbo：正常速度+正常渲染，供人观看")
    ap.add_argument("--device", default="cpu",
                    help="MlpPolicy 用 cpu 反而更快，默认 cpu")
    ap.add_argument("--tb", default=None, help="tensorboard 日志目录（需要装 tensorboard）")
    args = ap.parse_args()

    # ---------- 准备实例 ----------
    if args.attach:
        mgr = mi.InstanceManager(n=1)
        mgr.attach()
        print("\n等各实例连上 ...")
        if not mgr.wait_connections(args.run_timeout):
            print("超时：确认每个窗口都在对局里")
            return 1
        print("\n==== 端口 ↔ 窗口 对号 ====")
        mgr.resolve_mapping()
    elif args.launch:
        kw = {}
        if args.turbo:
            here = os.path.dirname(os.path.abspath(__file__))
            kw = dict(
                injector=os.path.join(here, "turbo", "turbo_inject.exe"),
                dll=os.path.join(here, "turbo", "turbo_hook.dll"),
                env_extra=dict(ISAAC_TURBO_SPEED=str(args.turbo_speed),
                               ISAAC_TURBO_SKIP_PRESENT="1",
                               ISAAC_TURBO_RENDER_EVERY="0"),
                viewer_count=args.viewer,
            )
            print(f"turbo 开启：{args.turbo_speed}x + 跳过呈现；"
                  f"其中 {args.viewer} 个实例正常渲染供观看")
        mgr = mi.InstanceManager(n=args.launch, **kw)
        print(f"启动 {args.launch} 个实例（每个窗口出来后在任意窗口开局）...")
        try:
            mgr.launch_all(wait_runs=True, run_timeout=args.run_timeout)
        except Exception as e:
            print(f"启动失败: {e}")
            mgr.stop_all()
            return 1
    else:
        print("需要 --attach 或 --launch N")
        return 1

    terrain_mode = TERRAIN_FULL if args.full_terrain else TERRAIN_LOCAL

    # ---------- 实测游戏帧率 → 推出真实决策频率 ----------
    decision_fps = args.fps
    if decision_fps is None:
        try:
            ch0 = mgr.bridge.channels[mgr.instances[0].port]
            g_fps = measure_game_fps(mi.ChannelAdapter(ch0))
        except Exception:
            g_fps = None
        if g_fps:
            decision_fps = g_fps / max(1, args.frame_skip)
            print(f"实测游戏帧率 {g_fps:.1f} fps ÷ frame_skip {args.frame_skip}"
                  f" → 决策 {decision_fps:.1f} 步/秒")
        else:
            decision_fps = 60.0 / max(1, args.frame_skip)
            print(f"⚠ 测不出游戏帧率，按 60fps 估算 → 决策 {decision_fps:.1f} 步/秒")

    # 热启动：以模型自己的观测维度为准，否则 predict 时形状对不上
    if args.load:
        try:
            args.load = resolve_model_path(args.load)     # 允许省略 .zip / 用 latest
        except FileNotFoundError as exc:
            print(exc)
            return 1
        try:
            loaded_dim = peek_obs_dim(args.load)
            mode_from_model = terrain_mode_from_dim(loaded_dim)
            if mode_from_model != terrain_mode:
                print(f"⚠ {args.load} 是 {loaded_dim} 维（地形 {mode_from_model}），"
                      f"按模型来（忽略 --full-terrain）")
            terrain_mode = mode_from_model
        except Exception as exc:
            print(f"⚠ 读不出 {args.load} 的观测维度（{exc}），按命令行参数 {terrain_mode} 继续")

    envs = build_envs(mgr, combat=args.combat, catalog=args.catalog,
                      min_enemies=args.min_enemies,
                      max_enemies=args.max_enemies,
                      chunk=args.chunk,
                      frame_skip=args.frame_skip,
                      terrain_mode=terrain_mode,
                      hit_rew=args.hit_rew, cost_miss=args.cost_miss,
                      fps=decision_fps, reset_mode=args.reset_mode,
                      hurt_scale=args.hurt_scale, fail_on_hurt=args.fail_on_hurt)
    venv = VecMonitor(ParallelIsaacVecEnv(envs))
    odim = obs_dim(terrain_mode)
    print(f"\n并行环境就绪：{venv.num_envs} 个实例")
    print(f"地形模式 {terrain_mode} → 观测 {odim} 维 "
          + ("（整间房 28×16×2 通道）" if terrain_mode == TERRAIN_FULL
             else "（玩家周围 7×7）"))
    e0 = envs[0]
    if hasattr(e0, "hit_rew"):
        print(f"奖励尺度：命中 {e0.hit_rew}/点伤害   时间代价 {e0.cost_miss}/秒   "
              f"受伤 x{e0.hurt_scale}   决策 {e0.fps:.1f} 步/秒")
        if e0.fail_on_hurt:
            print("  课程：不允许受伤 —— 掉任何一点血就结束本回合")
        print(f"  回合重置：{e0.reset_mode}"
              + ("" if e0.reset_mode == "restart" else "（不整局重开，血量跨回合累积）"))
    # full 地形把输入从 445 拉到 1292，128 宽的两层太窄，加宽到 256
    net_arch = [256, 256] if terrain_mode == TERRAIN_FULL else [128, 128]

    # ---------- 模型 ----------
    if args.load:
        print(f"加载模型 {args.load}")
        model = PPO.load(args.load, env=venv, device=args.device)
        model.learning_rate = args.lr
        model.ent_coef = args.ent_coef
        if not args.resume:
            model.num_timesteps = 0
            model._episode_num = 0
    else:
        model = PPO(
            "MlpPolicy", venv,
            n_steps=args.n_steps, batch_size=args.batch, n_epochs=10,
            gamma=0.99, gae_lambda=0.95, clip_range=0.2,
            ent_coef=args.ent_coef, learning_rate=args.lr,
            policy_kwargs=dict(net_arch=dict(pi=list(net_arch),
                                             vf=list(net_arch))),
            tensorboard_log=args.tb,
            device=args.device,
            verbose=1,
        )

    # checkpoint 名字带启动时间戳：CheckpointCallback 的计数每次训练都从 0 开始，
    # 只用 "multi" 前缀会让新训练**静默覆盖**上一次的同名 checkpoint
    # （实测：10/20 那次把 10/18 的 multi_5000..45000 全盖掉了）。
    # save_freq 的单位是「vec-step 调用次数」，所以要用总步数 // num_envs。
    run_tag = time.strftime("r%m%d-%H%M")
    save_freq = max(args.save_every // max(1, venv.num_envs), 100)
    ckpt = CheckpointCallback(save_freq=save_freq,
                              save_path="models",
                              name_prefix=f"multi_{run_tag}")
    print(f"checkpoint：每 {args.save_every} 总步（{save_freq} vec-step）存一次，"
          f"前缀 models/multi_{run_tag}_")

    print(f"\n==== 开始训练 {args.steps} 步（{venv.num_envs} 实例并行）====")
    t0 = time.time()
    model.learn(total_timesteps=args.steps,
                callback=[ckpt, ProgressCallback(every=10.0)],
                reset_num_timesteps=not args.resume)
    dt = time.time() - t0
    model.save(args.out)
    print(f"\n训练完成：{args.steps} 步 / {dt:.1f}s "
          f"= {args.steps / max(dt, 1e-9):.1f} 步/秒")
    print(f"模型已保存: {args.out}.zip")

    venv.close()
    mgr.stop_all()
    return 0


if __name__ == "__main__":
    sys.exit(main())
