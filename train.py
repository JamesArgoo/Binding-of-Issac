# -*- coding: utf-8 -*-
"""
train.py —— 用 PPO 训练以撒 AI

用法：
  python train.py                      # 从头训练（新模型，随机初始化）
  python train.py --resume             # 自动加载 models/ 里最新的存档，接着练
  python train.py --load models/isaac_ppo_final.zip   # 从指定模型继续
  python train.py --steps 2000000      # 指定本次训练步数

训练前：游戏进对局、把以撒停在一个有敌人的房间里、保持游戏窗口在前台。
训练中：AI 自己按键盘玩，你看着；死亡后模组自动重开一局，继续训练。
按 Ctrl+C 停止并保存。

注意：实时训练很慢（每步约 30ms），几十万步可能要好几个小时。
"""
import argparse
import glob
import os

from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import CheckpointCallback
from stable_baselines3.common.vec_env import DummyVecEnv

from isaac_env import IsaacEnv

MODELS_DIR = "models"


def find_latest_model():
    files = glob.glob(os.path.join(MODELS_DIR, "isaac_ppo_*.zip"))
    if not files:
        return None
    return max(files, key=os.path.getmtime)


def build_model(env, device):
    return PPO(
        "MlpPolicy",
        env,
        verbose=1,
        learning_rate=3e-4,
        n_steps=1024,
        batch_size=64,
        gamma=0.99,
        gae_lambda=0.95,
        ent_coef=0.01,
        policy_kwargs=dict(net_arch=dict(pi=[128, 128], vf=[128, 128])),
        device=device,
    )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--steps", type=int, default=500000, help="本次训练步数")
    ap.add_argument("--resume", action="store_true", help="自动加载最新存档继续训练")
    ap.add_argument("--load", default=None, help="从指定模型继续训练（如 models/isaac_ppo_bc.zip）")
    ap.add_argument("--frame-skip", type=int, default=2, help="每个动作保持几帧")
    ap.add_argument("--device", default="auto")
    ap.add_argument("--save-every", type=int, default=5000, help="每多少步存一次档")
    ap.add_argument("--lr", type=float, default=None, help="学习率；BC 微调建议 1e-4")
    ap.add_argument("--ent-coef", type=float, default=0.01, help="熵系数；BC 微调建议 0.001")
    args = ap.parse_args()

    env = DummyVecEnv([lambda: IsaacEnv(frame_skip=args.frame_skip)])
    os.makedirs(MODELS_DIR, exist_ok=True)

    load_path = None
    if args.load:
        if not os.path.exists(args.load):
            print(f"[错误] --load 指定的文件不存在：{args.load}")
            print("  检查路径：应该用斜杠 models/isaac_ppo_bc.zip，不是 models.issac_...")
            env.close()
            return
        load_path = args.load
    elif args.resume:
        load_path = find_latest_model()
        if load_path is None:
            print("[训练] --resume 但 models/ 里没有存档，改为从头训练")

    if load_path:
        model = PPO.load(load_path, env=env, device=args.device)
        if args.lr is not None:
            model.learning_rate = args.lr
        model.ent_coef = args.ent_coef
        print(f"[训练] 从 {load_path} 继续训练（lr={model.learning_rate}, ent_coef={model.ent_coef}）")
    else:
        print("[训练] 从头训练（新模型，随机初始化）")
        model = build_model(env, args.device)

    ckpt = CheckpointCallback(
        save_freq=args.save_every, save_path=MODELS_DIR, name_prefix="isaac_ppo"
    )

    print("开始训练。保持游戏在前台；按 Ctrl+C 保存并停止。")
    try:
        model.learn(total_timesteps=args.steps, callback=ckpt)
    except KeyboardInterrupt:
        print("\n中断，保存模型...")
    finally:
        model.save(os.path.join(MODELS_DIR, "isaac_ppo_final"))
        env.close()
        print(f"已保存 {MODELS_DIR}/isaac_ppo_final.zip")


if __name__ == "__main__":
    main()
