# -*- coding: utf-8 -*-
"""
play.py —— 纯播放：加载模型用策略玩，不训练

用来诊断"BC 模型本身像不像你"（排除 PPO 微调带来的干扰）。

用法：
  python play.py --model models/isaac_ppo_bc.zip
  python play.py --model models/isaac_ppo_bc.zip --deterministic   # 用确定性策略（更稳）
游戏需在前台。Ctrl+C 退出。
"""
import argparse
import time

from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv

from isaac_env import IsaacEnv


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="models/isaac_ppo_bc.zip")
    ap.add_argument("--frame-skip", type=int, default=2)
    ap.add_argument("--deterministic", action="store_true",
                    help="用确定性动作（推荐，模仿类模型更稳）")
    args = ap.parse_args()

    env = DummyVecEnv([lambda: IsaacEnv(frame_skip=args.frame_skip)])
    model = PPO.load(args.model, env=env)
    print(f"加载 {args.model}，纯播放不训练。5 秒倒计时...")
    for i in range(5, 0, -1):
        print(f"  {i}...")
        time.sleep(1)

    obs = env.reset()
    try:
        while True:
            action, _ = model.predict(obs, deterministic=args.deterministic)
            obs, rewards, dones, infos = env.step(action)
            if dones[0]:
                obs = env.reset()
    except KeyboardInterrupt:
        pass
    finally:
        env.close()
        print("\n已退出")


if __name__ == "__main__":
    main()
