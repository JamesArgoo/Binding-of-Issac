# -*- coding: utf-8 -*-
"""单房间战斗环境自检：跑一段随机动作，看奖励信号是否正常。

检查项：
  - reset 能不能 restart + goto 进战斗房间
  - 随机乱按能不能打出伤害/发弹（说明动作通道和统计都通）
  - 清房、死亡、命中率这些 reward 成分有没有产生

用法：python test_combat.py --steps 200
"""
import argparse
import sys
import time

import numpy as np

try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

import multi_instance as mi
from combat_env import CombatEnv
from train_multi import load_variants


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--steps", type=int, default=200)
    ap.add_argument("--catalog", default="room_catalog.json")
    args = ap.parse_args()

    variants = load_variants(args.catalog)
    print(f"房间目录：{len(variants)} 个战斗房间 variant")

    mgr = mi.InstanceManager(n=1)
    mgr.attach()
    if not mgr.wait_connections(60):
        print("没有实例在对局里")
        return 1
    mgr.resolve_mapping()

    inst = mgr.instances[0]
    ch = mgr.bridge.channels[inst.port]
    env = CombatEnv(mi.ChannelAdapter(ch), inst.controller, variants=variants, frame_skip=2)

    t0 = time.time()
    obs, info = env.reset()
    print(f"reset 耗时 {time.time() - t0:.1f}s，进到的房间 variant={info.get('variant')}，"
          f"obs 形状 {obs.shape}")

    def dump(tag):
        s = env.last_state or {}
        p = s.get("player") or {}
        enemies = [e for e in s.get("entities", []) if e.get("enemy") or e.get("boss")]
        tears = [e for e in s.get("entities", [])
                 if e.get("type") == 2 and e.get("spawner_type") == 1]
        print(f"  [{tag}] HP={p.get('hp')} 房间={s.get('room', {}).get('index')} "
              f"clear={s.get('room', {}).get('clear')} 敌={len(enemies)} "
              f"我方泪={len(tears)}")

    dump("reset后")

    tot_dealt = tot_fired = 0.0
    tot_reward = 0.0
    clears = deaths = 0
    rates = []
    ep = 1
    for i in range(args.steps):
        a = env.action_space.sample()
        obs, r, term, trunc, info = env.step(a)
        tot_reward += r
        tot_dealt += info.get("dealt", 0.0)
        tot_fired += info.get("fired", 0.0)
        rates.append(info.get("hit_rate", 0.0))
        if info.get("clear"):
            clears += 1
        if info.get("dead"):
            deaths += 1
        if i % 40 == 0:
            dump(f"第{i}步 action={tuple(int(x) for x in a)}")
        if term or trunc:
            print(f"  回合{ep} 结束：步数={i + 1} 累计伤害={tot_dealt:.1f} "
                  f"发弹={tot_fired:.0f} 清房={info.get('clear')} 死亡={info.get('dead')} "
                  f"本回合奖励={tot_reward:+.2f}")
            ep += 1
            tot_dealt = tot_fired = 0.0
            tot_reward = 0.0
            obs, info = env.reset()

    print(f"\n{args.steps} 步耗时 {time.time() - t0:.1f}s")
    print(f"  发弹总数: {sum(1 for _ in rates)} 步里命中率均值 {np.mean(rates):.3f}")
    print(f"  清房次数: {clears}  死亡次数: {deaths}")

    env.close()
    mgr.stop_all()
    return 0


if __name__ == "__main__":
    sys.exit(main())
