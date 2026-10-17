# -*- coding: utf-8 -*-
"""
diag.py —— 诊断：实时打印实体统计 + 子弹明细，确认敌方子弹有没有被正确识别

用法：
  python diag.py
游戏进对局、正常打（让敌人朝你射）。每秒打印一行汇总 + 子弹明细：
  [诊断] 敌人=.. 子弹类(type 2/7/8/9)=.. | type计数={...}
    子弹: type=9 var=0 spawner_type=.. pos=(..)
"""
import time
from collections import Counter

from isaac_bridge import (IsaacBridge, get_enemies, get_enemy_projectiles,
                          get_enemy_tears, get_projectiles, get_tears)


def main():
    bridge = IsaacBridge()
    bridge.start()
    print("诊断模式：每秒打印一次。进对局、正常打。Ctrl+C 退出。")
    try:
        while True:
            s = bridge.get_state()
            if s and "player" in s:
                ents = s.get("entities", [])
                tc = Counter(e.get("type") for e in ents)
                allb = get_projectiles(s)
                enemyb = get_enemy_projectiles(s)
                print(f"[诊断] 敌人={len(get_enemies(s))} "
                      f"全部子弹(2/7/8/9)={len(allb)} 敌方子弹={len(enemyb)} "
                      f"| type计数={dict(tc)}")
                for b in enemyb[:10]:
                    print(f"    敌方子弹: type={b['type']} var={b['variant']} "
                          f"spawner_type={b['spawner_type']} "
                          f"pos=({b['x']:.0f},{b['y']:.0f}) vx={b['vx']:.1f} vy={b['vy']:.1f}")
            else:
                print("[诊断] 还没数据 / 不在对局")
            time.sleep(1)
    except KeyboardInterrupt:
        pass
    finally:
        bridge.stop()


if __name__ == "__main__":
    main()
