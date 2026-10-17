# -*- coding: utf-8 -*-
"""
record_human.py —— 录制你（人类）的操作，供 AI 行为克隆模仿

用法：
  1) 游戏进对局，保持游戏窗口在前台
  2) python record_human.py
  3) 正常打，AI 不会抢你的键；脚本只旁观记录
  4) 打够了按 Ctrl+C 保存

原理：用 keyboard 库读你当前按住的键（默认 WASD 移动 + 方向键射击），
      每帧和游戏状态配对，存成 (obs, action)，格式和规则 AI 录的一样，
      所以可以直接喂给 bc_train.py。
"""
import argparse
import time

import numpy as np
import keyboard

from isaac_bridge import IsaacBridge
from isaac_env import build_obs
from ai_basic import DIRS8


def to_idx(x, y):
    if x == 0 and y == 0:
        return 8
    return DIRS8.index((x, y))


def read_human_action():
    """读当前按住的键，转成 (移动索引, 射击索引)。默认 WASD 移动 + 方向键射击。"""
    mx = (1 if keyboard.is_pressed("d") else 0) - (1 if keyboard.is_pressed("a") else 0)
    my = (1 if keyboard.is_pressed("s") else 0) - (1 if keyboard.is_pressed("w") else 0)
    ax = (1 if keyboard.is_pressed("right") else 0) - (1 if keyboard.is_pressed("left") else 0)
    ay = (1 if keyboard.is_pressed("down") else 0) - (1 if keyboard.is_pressed("up") else 0)
    return to_idx(mx, my), to_idx(ax, ay)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--steps", type=int, default=8000, help="最多录多少步")
    ap.add_argument("--out", default="bc_data_human.npz")
    args = ap.parse_args()

    bridge = IsaacBridge()
    bridge.start()

    obs_list, act_list = [], []
    grid = None

    print(f"开始录制你的操作（目标 {args.steps} 步）。正常打，Ctrl+C 停止。")
    print("默认按键：WASD 移动 + 方向键射击（若你改过键位，改 read_human_action 里的键名）")
    try:
        while len(obs_list) < args.steps:
            s = bridge.get_state()
            if s is None or "player" not in s:
                time.sleep(0.05)
                continue
            if "grid" in s:
                grid = s["grid"]

            mi, si = read_human_action()
            obs_list.append(build_obs(s, grid))
            act_list.append([mi, si])
            time.sleep(0.033)
    except KeyboardInterrupt:
        print("\n停止录制，保存...")
    finally:
        bridge.stop()

    np.savez(args.out, obs=np.array(obs_list, dtype=np.float32),
             act=np.array(act_list, dtype=np.int64))
    print(f"已保存 {args.out}（{len(obs_list)} 条样本）")
    print("下一步：python bc_train.py --data", args.out)


if __name__ == "__main__":
    main()
