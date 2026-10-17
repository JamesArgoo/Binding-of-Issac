# -*- coding: utf-8 -*-
"""
bc_train.py —— 用录制的 (状态,动作) 对，监督预训练 PPO 的策略（行为克隆）

思路：不单独建网络，而是直接拿 PPO 自己的 actor 网络做监督学习——
      这样预训练出来的权重和 PPO 完全兼容，之后 train.py --load 就能接着 PPO 微调。

用法：
  python bc_train.py --data bc_data.npz --epochs 30
输出：
  models/isaac_ppo_bc.zip
"""
import argparse

import numpy as np
import torch
import torch.nn.functional as F
import gymnasium as gym
from gymnasium import spaces
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv

from isaac_env import OBS_DIM


class DummyEnv(gym.Env):
    """只为构造 PPO 的观测/动作空间，不接真实游戏。"""
    observation_space = spaces.Box(-10.0, 10.0, (OBS_DIM,), dtype=np.float32)
    action_space = spaces.MultiDiscrete([9, 9])

    def reset(self, seed=None, options=None):
        return np.zeros(OBS_DIM, dtype=np.float32), {}

    def step(self, action):
        return np.zeros(OBS_DIM, dtype=np.float32), 0.0, False, False, {}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", nargs="+", default=["bc_data.npz"],
                    help="一个或多个 .npz 数据文件（可把规则 AI 和人类示范合并）")
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--batch", type=int, default=256)
    ap.add_argument("--out", default="models/isaac_ppo_bc")
    args = ap.parse_args()

    obs_parts, act_parts = [], []
    for f in args.data:
        d = np.load(f)
        obs_parts.append(d["obs"].astype(np.float32))
        act_parts.append(d["act"].astype(np.int64))
        print(f"  载入 {f}: {len(d['obs'])} 条")
    obs = np.concatenate(obs_parts)
    act = np.concatenate(act_parts)
    n = len(obs)
    print(f"合计 {n} 条样本，观测维度 {obs.shape[1]}")

    # 用和 train.py 一致的网络结构构建 PPO
    env = DummyVecEnv([lambda: DummyEnv()])
    model = PPO(
        "MlpPolicy",
        env,
        verbose=0,
        device="cpu",
        policy_kwargs=dict(net_arch=dict(pi=[128, 128], vf=[128, 128])),
    )
    policy = model.policy
    device = model.device

    obs_t = torch.as_tensor(obs, dtype=torch.float32, device=device)
    act_t = torch.as_tensor(act, dtype=torch.long, device=device)
    opt = torch.optim.Adam(policy.parameters(), lr=args.lr)

    bs = args.batch
    n_disp = 9
    for epoch in range(args.epochs):
        perm = torch.randperm(n, device=device)
        total = 0.0
        for i in range(0, n, bs):
            idx = perm[i:i + bs]
            ob = obs_t[idx]
            ac = act_t[idx]
            features = policy.extract_features(ob)
            latent_pi, _ = policy.mlp_extractor(features)
            logits = policy.action_net(latent_pi)  # (B, 18) = 9 移动 + 9 射击
            loss = (F.cross_entropy(logits[:, :n_disp], ac[:, 0])
                    + F.cross_entropy(logits[:, n_disp:], ac[:, 1]))
            opt.zero_grad()
            loss.backward()
            opt.step()
            total += loss.item() * len(idx)
        print(f"epoch {epoch + 1}/{args.epochs}  loss={total / n:.4f}")

    model.save(args.out)
    print(f"已保存 {args.out}.zip")
    print(f"下一步：python train.py --load {args.out}.zip  （接着 PPO 微调）")


if __name__ == "__main__":
    main()
