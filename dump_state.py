# -*- coding: utf-8 -*-
"""dump_state.py —— 数据探针：把模组发来的一帧"翻译成人话"并逐项摊开

回答的问题：**游戏内存里到底有什么、AI 实际拿到了什么**。

用法：
    python dump_state.py                 # 连 48532，取一帧详细打印
    python dump_state.py --frames 120    # 连续取 120 帧后打印（带取值统计）
    python dump_state.py --json out.json # 顺便把那几帧原始 JSON 存下来
    python dump_state.py --demo          # 不开游戏，用合成房间看效果

游戏侧准备：Steam 启动项加 --luadebug，且模组 isaac_ai_bridge 已启用。

和 `viz_viewer.py` 的区别：这个只输出文字，适合对着看数据有没有、
取值范围对不对；要看"AI 眼里的世界长什么样"用 viz_viewer.py。
"""
from __future__ import annotations

import argparse
import json
import sys
import time
import unicodedata

import numpy as np

for _stream in (sys.stdout, sys.stderr):
    try:
        _stream.reconfigure(encoding="utf-8")
    except Exception:
        pass

try:
    import isaac_data as D
    from isaac_env import OBS_DIM
except ImportError as exc:
    print(f"\n✗ {exc}", file=sys.stderr)
    sys.exit(1)

# ---------------------------------------------------------------- 排版小工具


def dwidth(s):
    """显示宽度：中日韩全角字符算 2 列，否则表头会歪。"""
    return sum(2 if unicodedata.east_asian_width(ch) in ("W", "F") else 1
               for ch in str(s))


def pad(s, w, align="l"):
    s = str(s)
    gap = " " * max(0, w - dwidth(s))
    return s + gap if align == "l" else gap + s


def table(headers, rows, aligns=None, indent="  "):
    if not rows:
        print(indent + "(无)")
        return
    aligns = aligns or ["l"] * len(headers)
    widths = [dwidth(h) for h in headers]
    for r in rows:
        for i, cell in enumerate(r):
            widths[i] = max(widths[i], dwidth(cell))
    line = indent + "  ".join(pad(h, widths[i]) for i, h in enumerate(headers))
    print(line)
    print(indent + "─" * dwidth(line))
    for r in rows:
        print(indent + "  ".join(pad(c, widths[i], aligns[i]) for i, c in enumerate(r)))


def sec(title):
    print()
    print("=" * 78)
    print(title)
    print("=" * 78)


def fnum(v, nd=1):
    try:
        return f"{float(v):.{nd}f}"
    except (TypeError, ValueError):
        return str(v)


# ---------------------------------------------------------------- 各段打印


def show_channel(state, feed=None, has_grid=False, has_map=False):
    sec("① 通路状态")
    rows = [
        ("frame", state.get("frame", "?"), "游戏逻辑帧计数"),
        ("实测帧率", f"{feed.fps:.1f} fps" if (feed and feed.fps) else "(demo/未知)",
         "模组每 2 帧发一次"),
        ("房间类型", D.room_type_name(state.get("room", {}).get("type", -1)),
         "room.type"),
        ("房间索引", state.get("room", {}).get("index"),
         "-3 = 由 goto 命令临时造出来的房间"),
        ("有 grid", "有" if has_grid else "无", "地形每 15 帧才发一次"),
        ("有 map", "有" if has_map else "无", "整层地图每 30 帧才发一次"),
        ("实体数", len(state.get("entities", [])), "含玩家自己"),
    ]
    table(["字段", "值", "说明"], rows)


def show_player(state):
    sec("② 玩家（player）—— 每帧都发")
    p = state.get("player", {})
    rows = [
        ("x, y", f"({fnum(p.get('x'))}, {fnum(p.get('y'))})", "世界坐标，屏幕外也照样有值"),
        ("vx, vy", f"({fnum(p.get('vx'))}, {fnum(p.get('vy'))})", "速度"),
        ("hp / max_hp", f"{p.get('hp')} / {p.get('max_hp')}", "红心，单位是半颗心"),
        ("soul_hp", p.get("soul_hp"), "魂心（掉血顺序在红心之后）"),
        ("coins/bombs/keys", f"{p.get('coins')}/{p.get('bombs')}/{p.get('keys')}", "资源"),
        ("active_item", p.get("active_item"), "主动道具 ID"),
        ("active_charge", p.get("active_charge"), "主动道具充能"),
        ("pocket_item / card", f"{p.get('pocket_item')} / {p.get('card')}", "胶囊 / 卡牌"),
        ("damage", fnum(p.get("damage"), 2), "单发伤害（命中率估算用）"),
        ("move_speed", fnum(p.get("move_speed"), 2), "移速"),
        ("tear_range", fnum(p.get("tear_range")), "射程 —— 关键！现在观测里没用"),
        ("shot_speed", fnum(p.get("shot_speed"), 2), "泪速"),
        ("fire_delay", fnum(p.get("fire_delay")), "射击间隔 —— 现在观测里没用"),
        ("size", fnum(p.get("size")), "玩家碰撞半径"),
        ("grid_idx", p.get("grid_idx"), "所在格"),
        ("fire_dir", f"{p.get('fire_dir')} ({D.dir_name(p.get('fire_dir'))})", "射击朝向"),
        ("head_dir", f"{p.get('head_dir')} ({D.dir_name(p.get('head_dir'))})", "头部朝向"),
        ("iframes", p.get("iframes"), "剩余无敌帧"),
    ]
    table(["字段", "值", "说明"], rows)


def show_entities(state):
    sec("③ 实体（entities）—— 每帧都发，房间里所有东西")
    ents = state.get("entities", [])
    p = state.get("player", {})

    # 分组统计
    counts = {}
    for e in ents:
        key = (int(e.get("type", 0)), int(e.get("spawner_type", 0) or 0))
        counts[key] = counts.get(key, 0) + 1
    rows = []
    for (t, sp), n in sorted(counts.items(), key=lambda kv: -kv[1]):
        owner = "玩家发射" if sp == 1 else ("无主" if sp == 0 else f"来自type={sp}")
        rows.append((n, t, D.entity_name(t, None), owner))
    table(["数量", "type", "名称", "spawner"], rows,
          aligns=["r", "r", "l", "l"])

    # 明细表
    print()
    print("  明细（按离玩家的距离排序，最多 25 行）：")
    def dist2(e):
        return (e.get("x", 0) - p.get("x", 0)) ** 2 + (e.get("y", 0) - p.get("y", 0)) ** 2
    rows = []
    for e in sorted(ents, key=dist2)[:25]:
        t = int(e.get("type", 0))
        flags = "".join((
            "E" if e.get("enemy") else "-",
            "B" if e.get("boss") else "-",
            "D" if e.get("dead") else "-",
        ))
        rows.append((
            e.get("id"), t, D.entity_name(t, e.get("variant")),
            e.get("variant"), e.get("subtype"),
            fnum(e.get("x"), 0), fnum(e.get("y"), 0),
            fnum(e.get("vx"), 1), fnum(e.get("vy"), 1),
            fnum(e.get("hp"), 1), fnum(e.get("size"), 0),
            e.get("coll"), fnum(e.get("cdmg"), 0),
            e.get("spawner_type"), flags,
        ))
    table(["id", "type", "名称", "var", "sub", "x", "y", "vx", "vy",
           "hp", "size", "coll", "cdmg", "spwn", "标志"], rows,
          aligns=["r", "r", "l", "r", "r", "r", "r", "r", "r", "r", "r", "r", "r", "r", "l"])
    print("  标志 = 敌人 / Boss / 已死")
    print("  cdmg>0 的实体 → 站上去会掉血（火堆、火焰），观测里会变成 0.5 的危险格")


def show_grid(state, grid, cached=False):
    sec("④ 地形（grid）—— 每 15 帧刷新一次")
    if not grid:
        print("  这一帧没有地形数据（模组每 15 帧才发一次，用 --frames 30 多取几帧）")
        return
    if cached:
        print("  ⚠ 用最近一次收到的地形（本帧没有带 grid）。真实环境也是这样缓存的 ——"
              "所以 terrain 观测最多滞后 15 帧 ≈ 250ms。")

    print("  字符画：墙# 石头o 便便p 尖刺^ 门+ 地板. | @玩家 E敌人 *弹幕 F火堆 $掉落 坑=空格")
    for line in D.ascii_room(state, grid):
        print("  " + line)

    w, h = int(grid["w"]), int(grid["h"])
    print()
    print(f"  尺寸 {w}x{h} = {w * h} 格，格宽 {grid.get('tw')} x 格高 {grid.get('th')}，"
          f"左上角 ({grid.get('ox')}, {grid.get('oy')})")

    print()
    print("  coll 分布（这一格走不走得进去）：")
    coll = grid.get("coll") or []
    hist = {}
    for c in coll:
        hist[int(c)] = hist.get(int(c), 0) + 1
    table(["coll", "含义", "格数", "占比"],
          [(c, D.grid_coll_name(c), n, f"{100.0 * n / max(1, len(coll)):.0f}%")
           for c, n in sorted(hist.items())],
          aligns=["r", "l", "r", "r"])

    print()
    print("  ent 分布（这块地长什么样 —— 观测里**完全没用**，是当前最大的浪费）：")
    ent = grid.get("ent") or []
    hist = {}
    for e in ent:
        hist[int(e)] = hist.get(int(e), 0) + 1
    table(["ent", "含义", "格数"],
          [(e, D.grid_ent_name(e), n) for e, n in sorted(hist.items()) if e != 0] or [("-", "全空", "")],
          aligns=["r", "l", "r"])

    haz = D._hazard_cells(state, grid)
    print()
    print(f"  危险格（cdmg>0 的实体覆盖到、且玩家能走上去）：{len(haz)} 格")


def show_doors_map(state, map_rooms=None):
    sec("⑤ 门（doors）与整层地图（map）")
    rows = [(d.get("slot"), d.get("grid"), "开" if d.get("open") else "关")
            for d in state.get("doors", [])]
    table(["slot", "格索引", "状态"], rows, aligns=["r", "r", "l"])

    rooms = map_rooms if map_rooms is not None else state.get("map", [])
    if not rooms:
        print("\n  还没收到 map（每 30 帧才发一次）")
        return
    print(f"\n  整层房间数：{len(rooms)}")
    kinds = {}
    for r in rooms:
        kinds[r.get("type")] = kinds.get(r.get("type"), 0) + 1
    table(["房间类型", "数量", "已探明"],
          [(D.room_type_name(t), n, "") for t, n in sorted(kinds.items())],
          aligns=["l", "r", "l"])
    print("  注意：map 只给了 房间格索引 + 类型 + 访问次数，"
          "**没有**房间之间的连接关系，也没有楼梯位置。")


def show_obs(state, grid, cached=False):
    sec("⑥ AI 的输入：445 维观测（这就是网络的眼睛）")
    obs = D.compute_obs(state, grid)
    labels = D.obs_labels()
    blocks = D.obs_blocks(obs)
    if cached:
        print("  （地形块来自缓存的上一份 grid）")

    rows = []
    for key, name, dim, _color in D.OBS_GROUPS:
        blk = blocks[key]
        nz = int(np.count_nonzero(np.abs(blk) > 1e-6))
        rows.append((f"{name}（{key}）", dim, nz,
                     fnum(float(np.min(blk)), 2), fnum(float(np.max(blk)), 2),
                     fnum(float(np.mean(np.abs(blk))), 3)))
    table(["块", "维度", "非零维数", "最小值", "最大值", "平均|值|"], rows,
          aligns=["l", "r", "r", "r", "r", "r"])
    print(f"  合计 {OBS_DIM} 维（isaac_env.build_obs 的定义）")

    p = state.get("player", {})
    print()
    print("  玩家块逐维：")
    table(["#", "含义", "值"],
          [(i, labels[i][1], fnum(v, 3)) for i, v in enumerate(blocks["player"])],
          aligns=["r", "l", "r"])

    print()
    print("  敌人块（16 个槽位 × 8 个特征，按离玩家最近的排前面）：")
    enemies = [e for e in state.get("entities", []) if e.get("enemy") or e.get("boss")]
    if enemies:
        enemies.sort(key=lambda e: (e["x"] - p.get("x", 0)) ** 2 + (e["y"] - p.get("y", 0)) ** 2)
    rows = []
    for i in range(D.MAX_ENEMIES):
        vals = blocks["enemy"][i]
        if not np.any(np.abs(vals) > 1e-6):
            continue
        who = D.entity_name(enemies[i].get("type")) if i < len(enemies) else "?"
        rows.append([i] + [who] + [fnum(v, 2) for v in vals])
    table(["槽", "实体"] + list(D._ENEMY_FEATS), rows,
          aligns=["r", "l"] + ["r"] * 8)
    if len(enemies) > D.MAX_ENEMIES:
        print(f"  ⚠ 房间里有 {len(enemies)} 个敌人，但观测只有 {D.MAX_ENEMIES} 个槽位 —— "
              f"多出来的 {len(enemies) - D.MAX_ENEMIES} 个 AI 完全看不见")
    else:
        print(f"  敌人 {len(enemies)} 个，槽位 {D.MAX_ENEMIES} 个"
              + ("" if enemies else "（本帧没有敌人）"))

    print()
    tears = [e for e in state.get("entities", [])
             if int(e.get("type", 0)) in (2, 7, 8, 9) and int(e.get("spawner_type", 0) or 0) != 1]
    print(f"  敌弹块：{len(tears)} 个敌弹 / {D.MAX_TEARS} 个槽位")
    if len(tears) > D.MAX_TEARS:
        print(f"  ⚠ 超出的 {len(tears) - D.MAX_TEARS} 个弹幕 AI 看不见")

    print()
    print("  地形块（玩家为中心的 7×7，0=走不了 0.5=能走但会掉血 1=安全）：")
    for row in blocks["grid"]:
        print("    " + " ".join(f"{v:>4.1f}" for v in row))

    print()
    print("  房间块 / 门块：")
    print(f"    敌人数量={fnum(blocks['room'][0], 3)}  已清空={fnum(blocks['room'][1], 3)}")
    table(["门", "相对x", "相对y", "开"],
          [(i, fnum(v[0], 2), fnum(v[1], 2), fnum(v[2], 0))
           for i, v in enumerate(blocks["door"])],
          aligns=["r", "r", "r", "r"])


def show_stats(frames, grids):
    sec(f"⑦ 多帧统计（{len(frames)} 帧）—— 各维真实取值范围")
    if len(frames) < 2:
        print("  至少要 --frames 2 才有统计")
        return

    obs_list = []
    for st, g in zip(frames, grids):
        obs_list.append(D.compute_obs(st, g))
    mat = np.stack(obs_list)
    labels = D.obs_labels()

    print("  按块统计（如果某块 min==max==0，说明这段时间它一直在喂常数 —— 相当于死输入）：")
    rows = []
    for key, name, dim, _c in D.OBS_GROUPS:
        lo, hi = D.BLOCK_SLICES[key]
        sl = mat[:, lo:hi]
        rows.append((name, dim, fnum(sl.min(), 2), fnum(sl.max(), 2),
                     fnum(float(np.mean(np.abs(sl))), 3),
                     fnum(float(np.mean(np.std(sl, axis=0) < 1e-6)), 2)))
    table(["块", "维度", "最小", "最大", "平均|值|", "恒定维占比"], rows,
          aligns=["l", "r", "r", "r", "r", "r"])

    # 找出"一直是常数"的维
    stds = mat.std(axis=0)
    const_idx = np.where(stds < 1e-6)[0]
    print()
    print(f"  全程恒定的维：{len(const_idx)} / {OBS_DIM}")
    if len(const_idx):
        names = {}
        for i in const_idx:
            names.setdefault(labels[i][0], []).append(labels[i][1])
        for key, name, dim, _c in D.OBS_GROUPS:
            if key in names:
                show = names[key][:6]
                more = "" if len(names[key]) <= 6 else f" …共{len(names[key])}维"
                print(f"    {name}：{', '.join(show)}{more}")


# ---------------------------------------------------------------- 主流程


def capture(args):
    """取帧：返回 (frames, feed, elapsed)。demo 模式即时生成。"""
    if args.demo:
        t0 = time.time()
        frames = []
        # 合成一串连续状态，中间混入几帧不带 grid 的（模拟 15 帧才发一次地形）
        for i in range(max(args.frames, 1)):
            t = i / 30.0
            st = D.demo_state(t)
            if i % 15 != 0 and i > 0:
                st = {k: v for k, v in st.items() if k != "grid"}
            if i % 30 != 0 and i > 0:
                st = {k: v for k, v in st.items() if k != "map"}
            frames.append(st)
        return frames, None, time.time() - t0

    feed = D.StateFeed(port=args.port)
    print(f"[探针] 监听 127.0.0.1:{args.port}，等游戏连上来"
          f"（最多 {args.timeout:.0f} 秒）...")
    feed.start()
    first = feed.wait_first(timeout=args.timeout)
    if first is None:
        feed.close()
        print("\n✗ 没等到数据。检查清单：")
        print("  1) Steam 启动项有没有加 --luadebug（socket 通信必需）")
        print("  2) mods/isaac_ai_bridge 是否已启用，主菜单里 Mods 能看到")
        print("  3) 端口是否被别的程序占着 → 换 --port")
        print("  4) 是否已经有一局在进行（不然模组只会发 menu）")
        return None, None, 0.0
    if "hello" in first:
        print(f"[探针] 收到 hello，端口 {first.get('port')} 握手成功 ✅")
        first = feed.wait_first(timeout=args.timeout)
        if first is None:
            feed.close()
            print("✗ 握手成功但没收到对局数据 —— 大概率还停在主菜单，进一局试试")
            return None, None, 0.0
    if "menu" in first:
        print(f"[探针] 模组还停在主菜单（自动开局尝试 {first.get('start_attempts')} 次），"
              f"请手动进一局")

    t0 = time.time()
    frames = [first]
    last = first
    while len(frames) < max(args.frames, 1) and time.time() - t0 < args.timeout:
        cur = feed.latest()
        if cur is not None and cur is not last and "player" in cur:
            frames.append(cur)
            last = cur
        time.sleep(0.01)
    return frames, feed, time.time() - t0


def _grid_series(frames):
    """地形序列：每帧用"截至该帧最近收到的 grid"，和真实环境的缓存行为一致。"""
    out, cur = [], None
    for st in frames:
        g = st.get("grid")
        if g:
            cur = D.normalize_grid(g)
        out.append(cur)
    return out


def main():
    ap = argparse.ArgumentParser(description="以撒模组数据探针")
    ap.add_argument("--port", type=int, default=48532)
    ap.add_argument("--frames", type=int, default=1, help="收集多少帧（>1 才有取值统计）")
    ap.add_argument("--timeout", type=float, default=60.0)
    ap.add_argument("--json", type=str, default=None, help="把这些帧的原始 JSON 存到文件")
    ap.add_argument("--demo", action="store_true", help="不连游戏，用合成数据看效果")
    ap.add_argument("--quiet", action="store_true", help="只打印观测部分")
    args = ap.parse_args()

    frames, feed, elapsed = capture(args)
    if not frames:
        return 1

    grids = _grid_series(frames)
    state, grid = frames[-1], grids[-1]
    cached = "grid" not in frames[-1]
    map_rooms = None
    for st in reversed(frames):
        if st.get("map"):
            map_rooms = st["map"]
            break
    if args.json:
        with open(args.json, "w", encoding="utf-8") as f:
            json.dump(frames, f, ensure_ascii=False, indent=1)
        print(f"[探针] 已保存 {len(frames)} 帧到 {args.json}")

    if not args.quiet:
        print(f"[探针] 收到 {len(frames)} 帧，耗时 {elapsed:.1f}s")
        show_channel(state, feed, has_grid=grid is not None, has_map=map_rooms is not None)
        show_player(state)
        show_entities(state)
        show_grid(state, grid, cached=cached)
        show_doors_map(state, map_rooms)
    show_obs(state, grid, cached=cached)
    if len(frames) > 1:
        show_stats(frames, grids)

    if feed is not None:
        feed.close()
    print()
    print("提示：想看「AI 眼里的世界」，跑 python viz_viewer.py")
    return 0


if __name__ == "__main__":
    sys.exit(main())
