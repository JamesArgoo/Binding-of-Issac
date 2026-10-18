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
import sys
import threading
import time

import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import CheckpointCallback
from stable_baselines3.common.vec_env import VecEnv, VecMonitor

try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

import multi_instance as mi
from isaac_env import OBS_DIM, IsaacEnv


class ParallelIsaacVecEnv(VecEnv):
    """把 N 个实时 IsaacEnv 包成 SB3 能吃的向量环境。

    每个 step 用线程并行推进所有实例 —— 各自 sleep frame_skip/60，
    并行后总耗时 ≈ 单实例耗时，而不是 N 倍。
    """

    def __init__(self, envs):
        super().__init__(len(envs), envs[0].observation_space, envs[0].action_space)
        self.render_mode = None
        self.envs = envs
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
                results[i] = (np.zeros(OBS_DIM, dtype=np.float32), 0.0, True, False, {})

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


def build_envs(mgr):
    """用管理器里的实例构造环境列表。"""
    envs = []
    for inst in mgr.instances:
        ch = mgr.bridge.channels[inst.port]
        envs.append(IsaacEnv(frame_skip=2,
                             bridge=mi.ChannelAdapter(ch),
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
        mgr = mi.InstanceManager(n=args.launch)
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

    envs = build_envs(mgr)
    venv = VecMonitor(ParallelIsaacVecEnv(envs))
    print(f"\n并行环境就绪：{venv.num_envs} 个实例")

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
            n_steps=256, batch_size=64, n_epochs=10,
            gamma=0.99, gae_lambda=0.95, clip_range=0.2,
            ent_coef=args.ent_coef, learning_rate=args.lr,
            policy_kwargs=dict(net_arch=dict(pi=[128, 128], vf=[128, 128])),
            tensorboard_log=args.tb,
            device=args.device,
            verbose=1,
        )

    ckpt = CheckpointCallback(save_freq=max(5000 // max(1, venv.num_envs), 500),
                              save_path="models", name_prefix="multi")

    print(f"\n==== 开始训练 {args.steps} 步（{venv.num_envs} 实例并行）====")
    t0 = time.time()
    model.learn(total_timesteps=args.steps, callback=ckpt,
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
