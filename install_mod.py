# -*- coding: utf-8 -*-
"""install_mod.py —— 把 isaac_ai_bridge 装进游戏的 mods 目录

改了 main.lua 之后必须重新安装（游戏不会读仓库里的这份）。

用法：
    python install_mod.py              # 自动找游戏目录并复制
    python install_mod.py --check      # 只检查，比较两边的 main.lua 是否一致
    python install_mod.py --game "D:\\...\\The Binding of Isaac Rebirth"
"""
import argparse
import hashlib
import os
import shutil
import sys

try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

HERE = os.path.dirname(os.path.abspath(__file__))
SRC = os.path.join(HERE, "isaac_ai_bridge")

CANDIDATES = [
    r"D:\SteamLibrary\steamapps\common\The Binding of Isaac Rebirth",
    r"C:\Program Files (x86)\Steam\steamapps\common\The Binding of Isaac Rebirth",
    r"D:\Steam\steamapps\common\The Binding of Isaac Rebirth",
    r"E:\SteamLibrary\steamapps\common\The Binding of Isaac Rebirth",
]


def find_game(explicit=None):
    if explicit:
        return explicit if os.path.isfile(os.path.join(explicit, "isaac-ng.exe")) else None
    # 先读 steam 的 libraryfolders.vdf，把所有库路径都试一遍
    roots = []
    for vdf in (r"C:\Program Files (x86)\Steam\steamapps\libraryfolders.vdf",
                r"D:\SteamLibrary\steamapps\libraryfolders.vdf"):
        if os.path.isfile(vdf):
            try:
                with open(vdf, encoding="utf-8", errors="ignore") as f:
                    for line in f:
                        if '"path"' in line:
                            parts = line.split('"')
                            if len(parts) >= 4:
                                roots.append(parts[3].replace("\\\\", "\\"))
            except OSError:
                pass
    for r in roots:
        p = os.path.join(r, "steamapps", "common",
                         "The Binding of Isaac Rebirth")
        if os.path.isfile(os.path.join(p, "isaac-ng.exe")):
            return p
    for p in CANDIDATES:
        if os.path.isfile(os.path.join(p, "isaac-ng.exe")):
            return p
    return None


def md5(path):
    h = hashlib.md5()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


def main():
    ap = argparse.ArgumentParser(description="安装/更新以撒 AI 桥接模组")
    ap.add_argument("--game", default=None, help="游戏根目录（含 isaac-ng.exe）")
    ap.add_argument("--check", action="store_true", help="只比较，不复制")
    args = ap.parse_args()

    game = find_game(args.game)
    if game is None:
        print("✗ 找不到游戏目录。用 --game 手动指定（指向含 isaac-ng.exe 的目录）")
        return 1
    dst = os.path.join(game, "mods", "isaac_ai_bridge")
    print(f"游戏目录 {game}")
    print(f"目标     {dst}")

    src_main = os.path.join(SRC, "main.lua")
    dst_main = os.path.join(dst, "main.lua")
    same = (os.path.isfile(dst_main)
            and md5(src_main) == md5(dst_main))
    print(f"main.lua 仓库版 {os.path.getsize(src_main)} 字节"
          + (f"，已安装版 {os.path.getsize(dst_main)} 字节" if os.path.isfile(dst_main) else "，未安装"))

    if same:
        print("✅ 两边一致，不需要更新")
        return 0
    if args.check:
        print("⚠ 两边不一致 —— 跑 `python install_mod.py` 更新，然后在游戏里重新加载模组")
        return 0

    os.makedirs(dst, exist_ok=True)
    for name in ("main.lua", "metadata.xml"):
        shutil.copy2(os.path.join(SRC, name), os.path.join(dst, name))
    print("✅ 已复制 main.lua / metadata.xml")
    print("\n接下来：在游戏里 Mods 菜单里把 Isaac AI Bridge 关掉再打开"
          "（或直接重启游戏），新代码才生效。")
    print("验证：python dump_state.py —— ① 通路状态里应该出现「模组耗时」和更高的实测帧率")
    return 0


if __name__ == "__main__":
    sys.exit(main())
