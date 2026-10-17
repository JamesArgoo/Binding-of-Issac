# -*- coding: utf-8 -*-
"""
play_safe.py —— BC 模型 + 卡住兜底（规则 AI 脱困）

行为：
  * 正常情况下用训练好的 BC/RL 模型按策略玩；
  * 如果检测到"卡住"（连续若干帧位置几乎没变），临时切换成规则 AI 的
    A* 寻路脱困，脱困后自动切回模型。

用法：
  python play_safe.py --model models/isaac_ppo_bc.zip
游戏需在前台。Ctrl+C 退出。
"""
import argparse
import time
from collections import deque

import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv

from ai_basic import decide, dirs_to_action, nearest_open_door_grid
from isaac_env import IsaacEnv


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="models/isaac_ppo_bc.zip")
    ap.add_argument("--frame-skip", type=int, default=2)
    ap.add_argument("--stuck-window", type=int, default=20, help="连续多少帧几乎没动算卡住")
    ap.add_argument("--stuck-dist", type=float, default=6.0, help="位移小于这个值算没动")
    ap.add_argument("--deterministic", action="store_true", default=True)
    args = ap.parse_args()

    env = DummyVecEnv([lambda: IsaacEnv(frame_skip=args.frame_skip)])
    inner = env.envs[0]
    model = PPO.load(args.model, env=env)

    print(f"加载 {args.model}（BC + 卡住兜底）。5 秒倒计时...")
    for i in range(5, 0, -1):
        print(f"  {i}...")
        time.sleep(1)

    obs = env.reset()
    stuck_pos = deque(maxlen=args.stuck_window)
    last_door = None
    prev_room = None
    prev_door = None
    rescued = 0

    try:
        while True:
            s = inner.last_state
            grid = inner.grid

            if s and "player" in s:
                room_idx = s["room"]["index"]
                if prev_room is not None and room_idx != prev_room:
                    last_door = prev_door
                prev_room = room_idx
                prev_door = nearest_open_door_grid(s, grid)
                p = s["player"]
                stuck_pos.append((p["x"], p["y"]))

            # 卡住检测
            stuck = False
            if len(stuck_pos) == stuck_pos.maxlen:
                xs = [x for x, y in stuck_pos]
                ys = [y for x, y in stuck_pos]
                if max(xs) - min(xs) < args.stuck_dist and max(ys) - min(ys) < args.stuck_dist:
                    stuck = True

            action, _ = model.predict(obs, deterministic=args.deterministic)

            if stuck and s and "player" in s:
                mx, my, ax, ay = decide(s, grid, last_door)
                action = np.array([[dirs_to_action(mx, my), dirs_to_action(ax, ay)]],
                                  dtype=np.int64)
                stuck_pos.clear()
                rescued += 1
                if rescued % 5 == 1:
                    print(f"[play] 卡住，规则 AI 脱困（累计 {rescued} 次）")

            obs, rewards, dones, infos = env.step(action)
            if dones[0]:
                obs = env.reset()
                stuck_pos.clear()
    except KeyboardInterrupt:
        pass
    finally:
        env.close()
        print("\n已退出")


if __name__ == "__main__":
    main()
