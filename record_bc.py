# -*- coding: utf-8 -*-
"""
record_bc.py —— 跑规则 AI，录制 (状态, 动作) 对，供行为克隆预训练

用法：
  python record_bc.py --steps 5000
输出：
  bc_data.npz（obs: [N, 357]  float32,  act: [N, 2] int64）

录制时规则 AI 会真的操作游戏，所以你坐在旁边看它跑就行。
"""
import argparse
import time

import numpy as np
import pyautogui

from isaac_bridge import IsaacBridge
from isaac_env import build_obs, KeyController
from ai_basic import decide, dirs_to_action, nearest_open_door_grid


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--steps", type=int, default=5000, help="录制多少步")
    ap.add_argument("--out", default="bc_data.npz")
    args = ap.parse_args()

    pyautogui.PAUSE = 0.0
    bridge = IsaacBridge()
    bridge.start()
    kc = KeyController()

    obs_list, act_list = [], []
    grid = None
    last_door = None
    prev_room = None
    prev_door = None

    print(f"开始录制规则 AI 行为（目标 {args.steps} 步）。保持游戏前台，Ctrl+C 停止并保存。")
    try:
        while len(obs_list) < args.steps:
            s = bridge.get_state()
            if s is None or "player" not in s:
                time.sleep(0.05)
                continue
            if "grid" in s:
                grid = s["grid"]

            room_idx = s["room"]["index"]
            if prev_room is not None and room_idx != prev_room:
                last_door = prev_door
            prev_room = room_idx
            prev_door = nearest_open_door_grid(s, grid)

            mx, my, ax, ay = decide(s, grid, last_door)
            obs_list.append(build_obs(s, grid))
            act_list.append([dirs_to_action(mx, my), dirs_to_action(ax, ay)])

            kc.set_move(mx, my)
            kc.set_shoot(ax, ay)
            time.sleep(0.033)
    except KeyboardInterrupt:
        print("\n中断，保存...")
    finally:
        kc.release_all()
        bridge.stop()

    np.savez(args.out, obs=np.array(obs_list, dtype=np.float32),
             act=np.array(act_list, dtype=np.int64))
    print(f"已保存 {args.out}（{len(obs_list)} 条样本）")


if __name__ == "__main__":
    main()
