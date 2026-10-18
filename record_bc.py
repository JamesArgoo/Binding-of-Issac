# -*- coding: utf-8 -*-
"""record_bc.py —— 用规则AI当老师录行为克隆（BC）数据

老师 = ai_basic_4（8方向移动 + 4方向射击 + 甩枪）。让它真的去玩，
边玩边把 (观测, 动作) 记下来。之后 bc_train.py 监督学习出一个初始策略，
比随机初始化强一大截，再拿去热启动 PPO —— 这就是"BC 热启动 + RL 微调"的路线。

为什么不让人类当老师：人类示范前后不一致（犹豫、回头、手抖），
而且观测分布窄；规则AI的决策是确定性的、可复现的，克隆起来干净得多。

多开：同时从多个实例录，数据来得更快。

用法：
    python record_bc.py --samples 20000
    python record_bc.py --samples 20000 --out bc_data_teacher.npz --hz 30
"""
import argparse
import sys
import threading
import time

import numpy as np

try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

import ai_basic_4 as teacher
import multi_instance as mi
from isaac_env import DIRS8, build_obs, normalize_grid


def action_index(dx, dy):
    """方向 -> 动作索引 0..7； (0,0) 表示不动/不射 -> 8"""
    try:
        return DIRS8.index((int(dx), int(dy)))
    except ValueError:
        return 8


class Recorder(threading.Thread):
    """一个实例一个录音线程：规则AI决策 + 驱动游戏 + 记录样本。"""

    def __init__(self, inst, channel, hz=30.0, verbose=True):
        super().__init__(daemon=True)
        self.inst = inst
        self.ch = channel
        self.kc = inst.controller
        self.dt = 1.0 / hz
        self.verbose = verbose
        self.stop_flag = threading.Event()
        self.obs = []
        self.acts = []
        self.grid = None
        self.last_door = None
        self.prev_room = None
        self.prev_door = None
        self.errors = 0

    def run(self):
        while not self.stop_flag.is_set():
            s = self.ch.state()
            if not s or "player" not in s:
                time.sleep(0.02)
                continue
            try:
                if "grid" in s:
                    self.grid = normalize_grid(s["grid"])

                # 换房间时告诉老师"我是从哪扇门过来的"，避免原地打转
                room_idx = s["room"]["index"]
                if self.prev_room is not None and room_idx != self.prev_room:
                    self.last_door = self.prev_door
                self.prev_room = room_idx
                self.prev_door = teacher.nearest_open_door_grid(s, self.grid)

                # 先记观测（决策时刻的状态），再执行动作
                obs = build_obs(s, self.grid)
                mx, my, ax, ay = teacher.decide(s, self.grid, self.last_door)

                self.kc.set_move(mx, my)
                self.kc.set_shoot(ax, ay)

                self.obs.append(obs)
                self.acts.append((action_index(mx, my), action_index(ax, ay)))
            except Exception as e:
                self.errors += 1
                if self.verbose and self.errors <= 3:
                    print(f"[实例{self.inst.index}] 录制异常: {e}")
            time.sleep(self.dt)

    def shutdown(self):
        self.stop_flag.set()
        self.kc.release_all()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--samples", type=int, default=20000, help="目标样本数（所有实例合计）")
    ap.add_argument("--out", default="bc_data_teacher.npz")
    ap.add_argument("--hz", type=float, default=30.0, help="录制频率（对应 frame_skip=2）")
    ap.add_argument("--run-timeout", type=float, default=300.0)
    args = ap.parse_args()

    mgr = mi.InstanceManager(n=1)
    mgr.attach()
    print("\n等各实例连上（每个窗口都要在对局里）...")
    if not mgr.wait_connections(args.run_timeout):
        print("超时：有窗口没进对局")
        return 1
    print("\n==== 端口 ↔ 窗口 对号 ====")
    mgr.resolve_mapping()

    recs = []
    for inst in mgr.instances:
        ch = mgr.bridge.channels[inst.port]
        r = Recorder(inst, ch, hz=args.hz)
        r.start()
        recs.append(r)

    n_inst = len(recs)
    eta = args.samples / max(n_inst * args.hz, 1e-9)
    print(f"\n{n_inst} 个实例同时录制，目标 {args.samples} 条（约 {eta:.0f} 秒）")

    t0 = time.time()
    last_report = 0.0
    try:
        while True:
            total = sum(len(r.acts) for r in recs)
            now = time.time()
            if total >= args.samples:
                break
            if now - last_report >= 10.0:
                last_report = now
                el = now - t0
                print(f"  [{el:5.0f}s] 已录 {total}/{args.samples} "
                      f"({total / max(el, 1e-9):.1f} 条/秒)", flush=True)
            time.sleep(0.5)
    except KeyboardInterrupt:
        print("\n手动结束")

    for r in recs:
        r.shutdown()
    for r in recs:
        r.join(timeout=2.0)

    parts_o = [np.stack(r.obs) for r in recs if r.obs]
    parts_a = [np.array(r.acts, dtype=np.int64) for r in recs if r.acts]
    if not parts_o:
        print("一条样本都没录到（老师一直没拿到有效状态？）")
        mgr.stop_all()
        return 2

    obs = np.concatenate(parts_o).astype(np.float32)
    act = np.concatenate(parts_a)
    el = time.time() - t0
    print(f"\n合计 {len(obs)} 条 / {el:.0f}s = {len(obs) / max(el, 1e-9):.1f} 条/秒")
    print(f"观测 {obs.shape}  动作 {act.shape}")

    # 动作分布：如果某一个动作占比极高，说明老师行为退化，BC 会学成常数
    m_d = np.bincount(act[:, 0], minlength=9) / len(act)
    s_d = np.bincount(act[:, 1], minlength=9) / len(act)
    print("移动动作分布(0-7=八方向, 8=不动):", np.round(m_d, 3))
    print("射击动作分布(0-7=八方向, 8=不射):", np.round(s_d, 3))

    np.savez_compressed(args.out, obs=obs, act=act)
    print(f"已保存 {args.out}")
    print(f"下一步：python bc_train.py --data {args.out} --epochs 40")

    mgr.stop_all()
    return 0


if __name__ == "__main__":
    sys.exit(main())
