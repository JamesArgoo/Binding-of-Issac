# -*- coding: utf-8 -*-
"""
test_dixing.py —— 验证地形（网格可走性）读取是否正确

检查内容：
  1. 打印当前房间的可行走地图（. 可走 / # 不可走）
  2. 按"网格实体类型"统计每类格子数量，以及其中被判为可走的数量
  3. 针对性验证：空地(0)、石头(2)、罐子(6)、便便(14)、坑(7) 的走法是否符合预期
  4. 找出"应不可走却被判为可走"的类型（真正的 bug）

用法：
  游戏进一个有石头/罐子/便便的房间，然后运行：
    python test_dixing.py
（需要先重启游戏加载最新模组，否则拿不到 coll/ent）
"""
import time
from collections import Counter

from isaac_bridge import IsaacBridge

# GridEntityType -> 名字（官方枚举 https://moddingofisaac.com/docs/rep/enums/GridEntityType.html）
GRID_NAMES = {
    0: "空", 1: "装饰",
    2: "石头", 3: "方块石", 4: "染色石", 5: "炸弹石", 6: "罐子/蘑菇/头骨",
    7: "坑", 8: "尖刺", 9: "开关尖刺", 10: "蛛网",
    11: "锁块", 12: "TNT", 13: "壁炉",
    14: "便便", 15: "墙", 16: "门", 17: "活板门", 18: "楼梯",
    19: "重力", 20: "压力板", 21: "雕像", 22: "超级染色石",
    23: "传送器", 24: "柱子", 25: "尖刺石", 26: "染色头骨", 27: "金石头",
}

# 预期"不可走"的实体类型
BLOCKING_TYPES = {2, 3, 4, 5, 6, 7, 8, 9, 11, 12, 14, 15, 22, 24, 25, 26, 27}
# 预期"可走"的实体类型（空/装饰/蛛网/门类/楼梯/活板门/压力板/传送器）
WALKABLE_TYPES = {0, 1, 10, 16, 17, 18, 20, 23}


def name_of(t):
    return GRID_NAMES.get(t, f"未知{t}")


def main():
    bridge = IsaacBridge()
    bridge.start()
    try:
        print("等待网格数据...（游戏进一个有石头/便便/罐子的房间更方便验证）")
        grid = None
        while grid is None:
            s = bridge.get_state()
            if s and "grid" in s:
                grid = s["grid"]
            time.sleep(0.1)

        w, h = grid["w"], grid["h"]
        cells = grid["cells"]
        coll = grid.get("coll")
        ent = grid.get("ent")

        print(f"\n房间网格: {w} x {h} = {w * h} 格")
        if coll is None or ent is None:
            print("[警告] 模组没有导出 coll/ent 字段 —— 请重启游戏加载最新模组后再测")
            return

        # 1) 可行走地图
        print("\n===== 可行走地图（. 可走  # 不可走）=====")
        for gy in range(h):
            row = "".join("." if cells[gy * w + gx] == 1 else "#" for gx in range(w))
            print(row)

        # 2) 按实体类型统计
        print("\n===== 实体类型统计（格子数 / 其中判可走数）=====")
        counts = Counter(ent)
        illegal = []
        for t, cnt in sorted(counts.items()):
            walk_n = sum(1 for i in range(w * h) if ent[i] == t and cells[i] == 1)
            mark = ""
            if t in BLOCKING_TYPES and walk_n > 0:
                mark = f"  ❌ 应不可走，却有 {walk_n} 格可走"
                illegal.append((t, walk_n))
            elif t in WALKABLE_TYPES and walk_n < cnt:
                mark = f"  ⚠ 应可走，却有 {cnt - walk_n} 格不可走"
            print(f"  {name_of(t):14s}(type={t:2d}): {cnt:3d} 格, 可走 {walk_n:3d}{mark}")

        # 3) 针对性验证
        print("\n===== 针对性验证 =====")
        checks = [(0, "空", True), (2, "石头", False), (6, "罐子", False),
                  (14, "便便", False), (7, "坑", False)]
        for t, label, should_walk in checks:
            cnt = counts.get(t, 0)
            if cnt == 0:
                print(f"  {label}(type={t}): 本房间没有")
                continue
            walk_n = sum(1 for i in range(w * h) if ent[i] == t and cells[i] == 1)
            ok = (walk_n == cnt) if should_walk else (walk_n == 0)
            print(f"  {label}(type={t}): {cnt} 格, 可走 {walk_n} → {'✓' if ok else '❌ 不符合预期'}")

        # 4) 结论
        print("\n===== 结论 =====")
        if illegal:
            for t, n in illegal:
                print(f"  ✗ {name_of(t)}(type={t}) 有 {n} 格被误判为可走")
        else:
            print("  ✓ 石头/罐子/便便/坑等均被正确判为不可走")
    except KeyboardInterrupt:
        pass
    finally:
        bridge.stop()


if __name__ == "__main__":
    main()
