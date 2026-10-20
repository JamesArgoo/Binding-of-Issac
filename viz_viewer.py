# -*- coding: utf-8 -*-
"""viz_viewer.py —— 可视化：把「AI 看到的」和「人看到的」并排放在一起

左边是人类视角的完整房间，右边是 AI 真正喂给神经网络的东西。
看这个窗口就明白：哪些信息 AI 有、哪些没有、哪些被压成了几个数字。

用法：
    python viz_viewer.py                 # 实时看（需要游戏在跑 + 模组已连上）
    python viz_viewer.py --demo          # 不开游戏，用合成房间看效果
    python viz_viewer.py --snapshot a.png        # 存一张图就退出
    python viz_viewer.py --demo --snapshot a.png # 不开游戏存图（用于写文档）

面板说明
--------
  ┌ 世界俯视图 ─────┬ AI 地形视野 ┬ 敌人槽位 ┐
  │ 完整房间+实体    │ 7×7 局部    │ 16×8 数字 │
  ├ 敌弹槽位 ───────┼ 状态文本 ───┴ 分块强度 ─┤
  └─────────────────┴─────────────────────────┘

关键差异（也是当前观测设计的两个"盲点"）：
  * **地形只看得见玩家周围 7×7**，房间另一头的墙石头 AI 完全不知道；
  * **敌人/子弹是全房间的**，但只有最近的 16 个敌人 / 48 个弹幕进得了观测。
"""
from __future__ import annotations

import argparse
import json
import sys
import time

import numpy as np

for _stream in (sys.stdout, sys.stderr):
    try:
        _stream.reconfigure(encoding="utf-8")
    except Exception:
        pass

import matplotlib
# --snapshot 只要出图，用无窗口后端，避免在没显示器的环境下开窗口
if "--snapshot" in sys.argv:
    matplotlib.use("Agg")

import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from matplotlib.markers import MarkerStyle
from matplotlib.patches import Rectangle

try:
    import isaac_data as D
    from isaac_env import (CANVAS_H, CANVAS_W, TERRAIN_FULL, TERRAIN_LOCAL,
                           obs_dim)
except ImportError as exc:
    print(f"\n✗ {exc}", file=sys.stderr)
    sys.exit(1)

# 中文显示：Windows 上默认字体没有汉字，不设会全变方块
for _f in ("Microsoft YaHei", "SimHei", "Noto Sans CJK SC", "DejaVu Sans"):
    try:
        matplotlib.rcParams["font.sans-serif"] = [_f]
        break
    except Exception:
        continue
matplotlib.rcParams["axes.unicode_minus"] = False

CAT_CMAP = ListedColormap(list(D.CAT_COLORS))
VISION_CMAP = ListedColormap(["#5c1a1a", "#d98c26", "#8fc98f"])   # 0 / 0.5 / 1
# full 模式：0=走不了/界外 1=可走 2=可走但危险
CANVAS_CMAP = ListedColormap(["#3a3a44", "#e8e4dc", "#d9622b"])

# head_dir/fire_dir -> 三角标记旋转角（默认 '^' 指向上）
_DIR_ROT = {-1: 0.0, 1: 0.0, 2: -90.0, 3: 180.0, 0: 90.0}


def _triangle(rot):
    return MarkerStyle((3, 0.0, rot))


class Viewer:
    def __init__(self, args):
        self.args = args
        self.terrain_mode = TERRAIN_FULL if getattr(args, "full_terrain", False) \
            else TERRAIN_LOCAL
        self.odim = obs_dim(self.terrain_mode)
        self.fig = plt.figure(figsize=(16.5, 9.6))
        gs = self.fig.add_gridspec(2, 3, width_ratios=[2.05, 0.95, 1.35],
                                   height_ratios=[1.0, 0.62],
                                   left=0.045, right=0.985, top=0.905, bottom=0.115,
                                   wspace=0.20, hspace=0.30)
        self.ax_world = self.fig.add_subplot(gs[0, 0])
        self.ax_vision = self.fig.add_subplot(gs[0, 1])
        self.ax_enemy = self.fig.add_subplot(gs[0, 2])
        self.ax_tear = self.fig.add_subplot(gs[1, 0])
        self.ax_info = self.fig.add_subplot(gs[1, 1])
        self.ax_groups = self.fig.add_subplot(gs[1, 2])
        self.ax_info.axis("off")
        try:
            self.fig.canvas.manager.set_window_title("以撒 AI 之眼")
        except Exception:
            pass
        self._build_legend()

    def _build_legend(self):
        """图例做成整张图共享的一条，放在底部预留区（不会被裁掉）。"""
        handles = [Rectangle((0, 0), 1, 1, fc=c, ec="#888") for c in D.CAT_COLORS]
        labels = list(D.CAT_NAMES) + ["危险格", "玩家", "敌人", "敌弹", "我弹", "掉落"]
        handles += [
            Rectangle((0, 0), 1, 1, fc="#ff3b30", alpha=0.35, ec="none"),
            plt.Line2D([], [], marker=_triangle(0), ls="", ms=9, mfc="#1565c0",
                       mec="#0b1d33"),
            plt.Line2D([], [], marker="o", ls="", ms=8, mfc="#b3001b", mec="#3b0006"),
            plt.Line2D([], [], marker="o", ls="", ms=6, mfc="#ffd400", mec="#6b5900"),
            plt.Line2D([], [], marker="o", ls="", ms=6, mfc="#17a2b8", mec="none"),
            plt.Line2D([], [], marker="s", ls="", ms=6, mfc="#2e9e44", mec="none"),
        ]
        self.fig.legend(handles, labels, loc="lower center",
                        bbox_to_anchor=(0.5, 0.004), ncol=7, fontsize=8,
                        frameon=False, handlelength=1.4, columnspacing=1.2,
                        handletextpad=0.5)

    # ------------------------------------------------------------ 世界俯视图
    def draw_world(self, state, grid, cats):
        ax = self.ax_world
        ax.clear()
        if cats is None:
            ax.text(0.5, 0.5, "还没有地形数据\n（模组每 15 帧发一次 grid）",
                    ha="center", va="center", fontsize=12, transform=ax.transAxes)
            ax.set_xticks([])
            ax.set_yticks([])
            ax.set_title("① 世界俯视图（人类视角）", fontsize=11)
            return

        h, w = cats.shape
        ax.imshow(cats, cmap=CAT_CMAP, vmin=-0.5, vmax=len(D.CAT_KEYS) - 0.5,
                  extent=(0, w, h, 0), interpolation="nearest", zorder=1)

        # 危险格（接触伤害覆盖 + 玩家能走）
        haz = D._hazard_cells(state, grid)
        if haz:
            mask = np.zeros((h, w), dtype=bool)
            walk = np.array(grid["cells"], dtype=np.int8).reshape(h, w) > 0
            for idx in haz:
                gy, gx = idx // w, idx % w
                if 0 <= gy < h and 0 <= gx < w:
                    mask[gy, gx] = True
            mask &= walk
            if mask.any():
                ax.imshow(np.ma.masked_where(~mask, np.ones_like(mask, float)),
                          cmap=ListedColormap(["#ff3b30"]), alpha=0.30,
                          extent=(0, w, h, 0), interpolation="nearest", zorder=2)

        tw = float(grid.get("tw") or 40.0)
        th = float(grid.get("th") or 40.0)
        ox, oy = float(grid.get("ox") or 0.0), float(grid.get("oy") or 0.0)

        def to_grid(x, y):
            return (x - ox) / tw, (y - oy) / th

        p = state.get("player", {})
        px, py = to_grid(p.get("x", 0), p.get("y", 0))

        # 7×7 地形视野框（只在 local 模式下画；full 模式整间房都看得见，画框会误导）
        if self.terrain_mode != TERRAIN_FULL:
            gx, gy = int(px), int(py)
            r = D.GRID_WIN // 2
            ax.add_patch(Rectangle((gx - r, gy - r), D.GRID_WIN, D.GRID_WIN,
                                   fill=False, lw=1.8, ls="--", ec="#000000", zorder=6))
            ax.text(gx - r + 0.1, gy - r - 0.25, "AI 的地形视野 7×7",
                    fontsize=7.5, color="#000000", zorder=7)

        for e in state.get("entities", []):
            t = int(e.get("type", 0))
            ex, ey = to_grid(e.get("x", 0), e.get("y", 0))
            size = float(e.get("size", 10) or 10)
            s = 22 + size * 5
            if t == 1:
                continue
            if e.get("enemy") or e.get("boss"):
                friendly = bool(e.get("boss"))
                ax.scatter([ex], [ey], s=s, marker="o",
                           c="#b3001b" if not friendly else "#7b00b3",
                           edgecolors="#3b0006", linewidths=0.8, zorder=4)
                cdmg = float(e.get("cdmg", 0) or 0)
                if cdmg > 0:                      # 火堆之类：接触就掉血
                    ax.scatter([ex], [ey], s=s * 2.4, marker="o",
                               facecolors="none", edgecolors="#ff7b00",
                               linewidths=1.2, zorder=4)
            elif t in (2, 9, 7, 8):
                own = int(e.get("spawner_type", 0) or 0)
                if own == 1:
                    ax.scatter([ex], [ey], s=18, marker="o", c="#17a2b8", zorder=3)
                else:
                    ax.scatter([ex], [ey], s=22, marker="o", c="#ffd400",
                               edgecolors="#6b5900", linewidths=0.5, zorder=3)
            elif t == 5:
                ax.scatter([ex], [ey], s=40, marker="s", c="#2e9e44", zorder=3)

        # 玩家：三角朝 head_dir
        rot = _DIR_ROT.get(int(p.get("head_dir", -1) or -1), 0.0)
        ax.scatter([px], [py], s=150, marker=_triangle(rot), c="#1565c0",
                   edgecolors="#0b1d33", linewidths=1.2, zorder=5)
        ax.text(px + 0.35, py - 0.35, "玩家", fontsize=7.5, color="#1565c0", zorder=7)

        ax.set_xlim(0, w)
        ax.set_ylim(h, 0)
        ax.set_aspect("equal")
        ax.set_xticks(range(0, w + 1, 5))
        ax.set_yticks(range(0, h + 1))
        ax.tick_params(labelsize=6.5)
        ax.set_title(f"① 世界俯视图（人类视角）{w}×{h}", fontsize=11)

    # ------------------------------------------------------------ AI 地形视野
    def draw_vision(self, blocks, grid=None):
        ax = self.ax_vision
        ax.clear()
        win = blocks["grid"]
        if self.terrain_mode == TERRAIN_FULL:
            walk_ch, haz_ch = win[0], win[1]
            cat = np.where(haz_ch > 0.5, 2, np.where(walk_ch > 0.5, 1, 0))
            ax.imshow(cat, cmap=CANVAS_CMAP, vmin=-0.5, vmax=2.5,
                      interpolation="nearest")
            if grid:      # 标出房间真实边界（画布其余部分是界外填充）
                ax.add_patch(Rectangle((-0.5, -0.5), grid["w"], grid["h"],
                                       fill=False, lw=1.2, ls=":",
                                       ec="#00c8ff", zorder=5))
            ax.set_xticks([])
            ax.set_yticks([])
            ax.set_title(f"② AI 的地形视野\n整间房 {CANVAS_W}×{CANVAS_H}"
                         f"×2通道=896维", fontsize=10.0)
            return
        ax.imshow(win, cmap=VISION_CMAP, vmin=-0.25, vmax=1.25,
                  interpolation="nearest")
        r = D.GRID_WIN // 2
        for i in range(D.GRID_WIN):
            for j in range(D.GRID_WIN):
                v = win[i, j]
                ax.text(j, i, "0" if v < 0.25 else ("½" if v < 0.75 else "1"),
                        ha="center", va="center", fontsize=8.5,
                        color="#ffffff" if v < 0.25 else "#20301f")
        ax.plot([r], [r], marker="+", ms=10, mew=1.6, color="#0b1d33")
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_title("② AI 的地形视野\n只有这 7×7 格（49 维）", fontsize=10.0)

    # ------------------------------------------------------------ 敌人 / 敌弹槽
    def draw_enemy_block(self, blocks, n_enemies):
        ax = self.ax_enemy
        ax.clear()
        blk = blocks["enemy"]
        ax.imshow(blk, cmap="RdBu_r", vmin=-1, vmax=1, aspect="auto",
                  interpolation="nearest")
        ax.set_xticks(range(8))
        ax.set_xticklabels(list(D._ENEMY_FEATS), fontsize=6.5, rotation=40, ha="right")
        ax.set_yticks(range(D.MAX_ENEMIES))
        ax.set_yticklabels([f"槽{i}" for i in range(D.MAX_ENEMIES)], fontsize=6.5)
        for i in range(D.MAX_ENEMIES):          # 空槽位画灰
            if not np.any(np.abs(blk[i]) > 1e-6):
                ax.axhspan(i - 0.5, i + 0.5, color="#dddddd", alpha=0.55, zorder=0)
        ax.set_title(f"③ 敌人槽位 16×8\n本帧 {n_enemies} 个敌人", fontsize=10.5)

    def draw_tear_block(self, blocks, n_tears):
        ax = self.ax_tear
        ax.clear()
        blk = blocks["tear"]
        ax.imshow(blk, cmap="RdBu_r", vmin=-1, vmax=1, aspect="auto",
                  interpolation="nearest")
        ax.set_yticks(range(5))
        ax.set_yticklabels(list(D._TEAR_FEATS), fontsize=7)
        ax.set_xticks(range(0, D.MAX_TEARS, 4))
        ax.set_xticklabels(range(0, D.MAX_TEARS, 4), fontsize=6.5)
        for i in range(D.MAX_TEARS):
            if not np.any(np.abs(blk[:, i]) > 1e-6):
                ax.axvspan(i - 0.5, i + 0.5, color="#dddddd", alpha=0.55, zorder=0)
        ax.set_title(f"④ 敌弹槽位 5×48（按距离排序）\n本帧 {n_tears} 个敌弹", fontsize=10.5)

    # ------------------------------------------------------------ 状态文本
    def draw_info(self, state, grid, obs, blocks, n_enemies, n_tears, mode):
        ax = self.ax_info
        ax.clear()
        ax.axis("off")
        p = state.get("player", {})
        room = state.get("room", {})
        nz = int(np.count_nonzero(np.abs(obs) > 1e-6))
        w = grid["w"] if grid else "?"
        h = grid["h"] if grid else "?"
        gidx = p.get("grid_idx")
        gxy = (f"{int(gidx) % w},{int(gidx) // w}"
               if (grid and gidx is not None) else "?")

        doors = state.get("doors", [])
        door_txt = " ".join(f"{d.get('slot')}:{'开' if d.get('open') else '关'}"
                            for d in doors) or "—"
        if self.terrain_mode == TERRAIN_FULL:
            terr_txt = f"地形 {w}×{h} 全房间(28×16画布)"
        else:
            terr_txt = f"地形 {w}×{h}，只看 7×7"

        lines = [
            f"数据源 {mode}   帧 {state.get('frame')}",
            f"地形 {self.terrain_mode} · 观测 {self.odim} 维",
            "── 玩家 ──",
            f"格位置 ({gxy})   血 {p.get('hp')}+{p.get('soul_hp')}魂",
            f"伤害 {p.get('damage')}  射程 {p.get('tear_range')}  "
            f"泪速 {p.get('shot_speed')}",
            f"朝向 射{D.dir_name(p.get('fire_dir'))}/头{D.dir_name(p.get('head_dir'))}"
            f"   无敌帧 {p.get('iframes')}",
            "── 房间 ──",
            f"{D.room_type_name(room.get('type', -1))} index {room.get('index')} "
            f"{'已清空' if room.get('clear') else '战斗中'}",
            f"活敌 {room.get('enemies_alive')}  boss {room.get('bosses_alive')}",
            f"门 {door_txt}",
            "── AI 拿到什么 ──",
            f"观测 {nz}/{self.odim} 维非零",
            f"敌人 {n_enemies}/{D.MAX_ENEMIES} 槽"
            + ("  ⚠溢出" if n_enemies > D.MAX_ENEMIES else "")
            + f"   敌弹 {n_tears}/{D.MAX_TEARS} 槽"
            + ("  ⚠溢出" if n_tears > D.MAX_TEARS else ""),
            terr_txt + ("" if grid else "  ⚠本帧无grid"),
        ]
        ax.text(0.0, 0.99, "\n".join(lines), va="top", ha="left",
                fontsize=7.6, transform=ax.transAxes, linespacing=1.40)
        ax.set_title("⑤ 状态与覆盖率", fontsize=10.5)

    # ------------------------------------------------------------ 分块强度
    def draw_groups(self, blocks):
        ax = self.ax_groups
        ax.clear()
        groups = D.obs_groups(self.terrain_mode)
        names, vals, ns, colors = [], [], [], []
        for key, name, dim, color in groups:
            blk = blocks[key]
            names.append(name)
            vals.append(float(np.mean(np.abs(blk))))
            ns.append(int(np.count_nonzero(np.abs(blk) > 1e-6)))
            colors.append(color)
        y = np.arange(len(names))[::-1]
        ax.barh(y, vals, color=colors, height=0.62)
        for yi, v, n, (_k, _nm, dim, _c) in zip(y, vals, ns, groups):
            ax.text(v + max(vals) * 0.02, yi, f"{v:.3f}  ({n}/{dim}非零)",
                    va="center", fontsize=7.5)
        ax.set_yticks(y)
        ax.set_yticklabels([f"{n}  {d}维" for n, (_k, _nm, d, _c)
                            in zip(names, groups)], fontsize=8)
        ax.set_xlim(0, max(vals + [0.01]) * 1.55)
        ax.set_xlabel("该块内 |值| 的平均（越大 = 这块信息越强）", fontsize=8)
        ax.set_title(f"⑥ {self.odim} 维的分块强度", fontsize=10.5)
        ax.tick_params(labelsize=7.5)
        ax.grid(axis="x", ls=":", alpha=0.4)

    # ------------------------------------------------------------ 一帧
    def render(self, state, grid, mode):
        cats = D.cell_categories(state, grid) if grid else None
        obs = D.compute_obs(state, grid, self.terrain_mode)
        blocks = D.obs_blocks(obs, self.terrain_mode)
        p = state.get("player", {})
        enemies = [e for e in state.get("entities", [])
                   if e.get("enemy") or e.get("boss")]
        tears = [e for e in state.get("entities", [])
                 if int(e.get("type", 0)) in (2, 7, 8, 9)
                 and int(e.get("spawner_type", 0) or 0) != 1]

        self.draw_world(state, grid, cats)
        self.draw_vision(blocks, grid)
        self.draw_enemy_block(blocks, len(enemies))
        self.draw_tear_block(blocks, len(tears))
        self.draw_info(state, grid, obs, blocks, len(enemies), len(tears), mode)
        self.draw_groups(blocks)

        room = state.get("room", {})
        self.fig.suptitle(
            f"以撒 AI 之眼 —— 人类的房间 vs 网络收到的 {self.odim} 个数字"
            f"    |  地形 {self.terrain_mode}  帧 {state.get('frame')}  "
            f"房间 {D.room_type_name(room.get('type', -1))}  "
            f"{'已清空' if room.get('clear') else '战斗中'}",
            fontsize=13.5)


def main():
    ap = argparse.ArgumentParser(description="以撒 AI 观测可视化")
    ap.add_argument("--port", type=int, default=48532)
    ap.add_argument("--demo", action="store_true", help="用合成房间（不需要开游戏）")
    ap.add_argument("--replay", type=str, default=None,
                    help="回放 dump_state.py --json 存的帧文件（不需要开游戏）")
    ap.add_argument("--snapshot", type=str, default=None, help="存一张 PNG 后退出")
    ap.add_argument("--interval", type=float, default=0.12, help="刷新间隔（秒）")
    ap.add_argument("--timeout", type=float, default=60.0, help="等游戏连上来的超时")
    ap.add_argument("--full-terrain", action="store_true",
                    help="按全房间地形观测（28×16×2 通道，1292 维）显示，"
                         "而不是默认的 7×7 局部（445 维）")
    args = ap.parse_args()

    viewer = Viewer(args)
    feed = None

    # ---- 数据源：replay / demo / 实时 ----
    replay_frames, replay_grids, replay_i = None, None, 0
    if args.replay:
        with open(args.replay, encoding="utf-8") as f:
            data = json.load(f)
        data = data if isinstance(data, list) else [data]
        replay_frames = [st for st in data if "player" in st] or data
        # 地形缓存：模组 15 帧才发一次 grid，真实环境也是这样缓存的
        cur, replay_grids = None, []
        for st in replay_frames:
            if st.get("grid"):
                cur = D.normalize_grid(dict(st["grid"]))
            replay_grids.append(cur)
        mode = f"回放 {args.replay}（{len(replay_frames)} 帧）"
        print(f"[viz] 回放 {args.replay}，共 {len(replay_frames)} 帧")

        def get_state(_t=0.0):
            nonlocal replay_i
            if args.snapshot:
                replay_i = len(replay_frames) - 1     # 存图用最后一帧
            else:
                replay_i = (replay_i + 1) % len(replay_frames)
            return replay_frames[replay_i]

        def grid_for(state):
            return replay_grids[replay_i]
    elif args.demo:
        def get_state(_t=0.0):
            return D.demo_state(time.time())
        mode = "合成数据（--demo）"

        grid_cache = {"g": None}

        def grid_for(state):
            raw = state.get("grid")
            if raw:
                grid_cache["g"] = D.normalize_grid(dict(raw))
            return grid_cache["g"]
    else:
        feed = D.StateFeed(port=args.port).start()
        print(f"[viz] 监听 127.0.0.1:{args.port} ...")
        first = feed.wait_first(timeout=args.timeout)
        if first is None:
            feed.close()
            print("✗ 没等到数据。确认 --luadebug 已开、模组已启用、并且已经进了一局。")
            return 1

        def get_state(_t=0.0):
            return feed.latest()
        mode = f"LIVE 127.0.0.1:{args.port}"

        grid_cache = {"g": None}

        def grid_for(state):
            raw = state.get("grid")
            if raw:
                grid_cache["g"] = D.normalize_grid(dict(raw))
            return grid_cache["g"]

    if args.snapshot:
        state = get_state()
        g = grid_for(state)
        viewer.render(state, g, mode)
        viewer.fig.savefig(args.snapshot, dpi=110)
        print(f"[viz] 已保存 {args.snapshot}")
        if feed is not None:
            feed.close()
        return 0

    plt.ion()
    plt.show(block=False)
    print("[viz] 窗口已打开（关掉窗口即退出）")
    last_frame = None
    try:
        while plt.fignum_exists(viewer.fig.number):
            state = get_state()
            if state is None or "player" not in state:
                plt.pause(args.interval)
                continue
            g = grid_for(state)
            if state.get("frame") != last_frame or args.demo or args.replay:
                last_frame = state.get("frame")
                viewer.render(state, g, mode)
                viewer.fig.canvas.draw_idle()
            viewer.fig.canvas.flush_events()
            plt.pause(args.interval)
    except KeyboardInterrupt:
        pass
    finally:
        if feed is not None:
            feed.close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
