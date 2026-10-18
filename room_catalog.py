# -*- coding: utf-8 -*-
"""room_catalog.py —— 探出哪些 `goto d.<variant>` 会加载出有敌人的房间

模仿 AIsaac 的 `abplus_room_catalog.py`：逐个 goto，记录房间类型和敌人组成，
把"确实有敌人"的房间收进 JSON。单房间战斗训练就从这份目录里随机抽房间。

多开并行：把 variant 区间切给各个实例同时探，速度直接翻倍。

用法：
    python room_catalog.py --max 300 --out room_catalog.json
    python room_catalog.py --min 0 --max 1200 --settle 0.35
"""
import argparse
import json
import sys
import threading
import time

try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

import multi_instance as mi


def alive(s):
    return bool(s and "player" in s
                and (s["player"]["hp"] + s["player"]["soul_hp"]) > 0)


def ensure_alive(ch, timeout=20.0):
    """探测时玩家是不动的，会被敌人打死。死了就 restart 再来。

    注意：死亡结算画面要几秒才结束，restart 短时间内不生效，所以超时给得比较长。
    """
    if alive(ch.state()):
        return True
    for _ in range(2):
        ch.send("restart")
        t0 = time.time()
        while time.time() - t0 < timeout / 2:
            if alive(ch.state()):
                return True
            time.sleep(0.05)
    return False


def probe_one(ch, variant, settle=0.5):
    """goto 一个 variant，返回房间信息。

    注意：goto 造出来的房间 `room.index` 恒为 -3（不在楼层地图上），
    所以不能用"房间号变没变"判断成功与否 —— 只能看这次 settle 后的敌人组成。
    """
    if not ensure_alive(ch):
        return None

    if not ch.send(f"goto d.{variant}"):
        return None
    time.sleep(settle)

    s = ch.state()
    if not s or "player" not in s or "room" not in s:
        return None
    if not alive(s):
        return None

    enemies = [e for e in s.get("entities", []) if e.get("enemy") or e.get("boss")]
    return dict(
        variant=variant,
        room_index=s["room"]["index"],
        room_type=s["room"]["type"],
        clear=bool(s["room"]["clear"]),
        n_enemies=len(enemies),
        enemies=[dict(type=e["type"], variant=e["variant"], boss=bool(e.get("boss")),
                      hp=e["hp"]) for e in enemies][:16],
    )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--min", type=int, default=0)
    ap.add_argument("--max", type=int, default=300)
    ap.add_argument("--settle", type=float, default=0.35, help="每次 goto 后等多久(秒)")
    ap.add_argument("--out", default="room_catalog.json")
    ap.add_argument("--run-timeout", type=float, default=120.0)
    args = ap.parse_args()

    mgr = mi.InstanceManager(n=1)
    mgr.attach()
    print("\n等各实例连上（每个窗口都要在对局里）...")
    if not mgr.wait_connections(args.run_timeout):
        print("超时：有窗口没进对局")
        return 1

    n = len(mgr.instances)
    ids = list(range(args.min, args.max + 1))
    # 切成 n 段，每个实例一段
    slices = [ids[i::n] for i in range(n)]
    results = [[] for _ in range(n)]
    done = [0] * n

    def worker(k, inst, my_ids):
        ch = mgr.bridge.channels[inst.port]
        for v in my_ids:
            try:
                r = probe_one(ch, v, args.settle)
            except Exception as e:
                print(f"[实例{k}] variant {v} 异常: {e}")
                r = None
            if r is not None:
                results[k].append(r)
            done[k] += 1

    print(f"\n{n} 个实例并行探测 variant {args.min}..{args.max}"
          f"（每个约 {len(slices[0])} 个，预计 {len(slices[0]) * args.settle:.0f}s）")

    threads = [threading.Thread(target=worker, args=(k, inst, slices[k]))
               for k, inst in enumerate(mgr.instances)]
    t0 = time.time()
    for t in threads:
        t.start()

    last = 0.0
    while any(t.is_alive() for t in threads):
        now = time.time()
        if now - last >= 10.0:
            last = now
            total = sum(done)
            found = sum(len(r) for r in results)
            print(f"  [{now - t0:5.0f}s] 已探 {total}/{len(ids)}，"
                  f"有效房间 {found}", flush=True)
        time.sleep(0.5)

    for t in threads:
        t.join()

    all_rooms = [r for rs in results for r in rs]
    # 去重（同一个房间可能被不同 variant 命中）
    seen = set()
    uniq = []
    for r in all_rooms:
        key = (r["room_index"], r["room_type"], r["n_enemies"],
               tuple(sorted((e["type"], e["variant"]) for e in r["enemies"])))
        if key in seen:
            continue
        seen.add(key)
        uniq.append(r)

    combat = [r for r in uniq if r["n_enemies"] > 0 and not r["clear"]]
    print(f"\n探测完成，耗时 {time.time() - t0:.0f}s")
    print(f"  加载成功的 room: {len(all_rooms)}，去重后 {len(uniq)}，"
          f"其中**有敌人且未清空** {len(combat)} 个")

    if combat:
        variants = sorted({r["variant"] for r in combat})
        print(f"  可用 variant: {variants[:40]}{' ...' if len(variants) > 40 else ''}")
        from collections import Counter
        types = Counter(r["room_type"] for r in combat)
        print(f"  房间类型分布: {dict(types)}")
        cnt = Counter(r["n_enemies"] for r in combat)
        print(f"  敌人数分布: {dict(sorted(cnt.items())[:10])}")

    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(dict(combat=combat, all_rooms=uniq,
                       range=[args.min, args.max]), f, ensure_ascii=False, indent=1)
    print(f"已保存 {args.out}")

    mgr.stop_all()
    return 0


if __name__ == "__main__":
    sys.exit(main())
