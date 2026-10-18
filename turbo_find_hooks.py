"""turbo_find_hooks.py —— 在 Windows 版 isaac-ng.exe 里找 isaac_turbo 的 hook 目标

AIsaac 的 isaac_turbo（turbo/src/turbo.cpp, j460_targets.json）按绝对地址挂 hook，
绑定 J460 build。但它同时存了每个目标的 prologue 字节签名（含 ?? 通配）。

J460 和 J835 是相近版本，签名很可能还在 —— 这里就用签名去扫我们的 exe，
如果命中，就能白捡地址、直接复用他们的 turbo 逻辑。

用法：
    python turbo_find_hooks.py                       # 默认扫 Steam 安装目录
    python turbo_find_hooks.py <isaac-ng.exe 路径>
"""
import json
import struct
import sys

DEFAULT_EXE = r"D:\SteamLibrary\steamapps\common\The Binding of Isaac Rebirth\isaac-ng.exe"

# 来自 AIsaac turbo/j460_targets.json
J460_BUILD = dict(timestamp=0x69E6E3A7, size_of_image=0x0092C000)
SIGS = {
    "glfw_get_time": "55 8B EC 83 EC 08 83 3D ?? ?? ?? ?? 00 75 16",
    "render_frame": ("53 8B DC 83 EC 08 83 E4 F8 83 C4 04 55 8B 6B 04 "
                     "89 6C 24 04 8B EC 6A FF"),
    "manager_update": "55 8B EC 83 E4 F0 81 EC 78 04 00 00 A1 ?? ?? ?? ?? 33 C4",
    "game_update": "55 8B EC 6A FF 68 ?? ?? ?? ?? 64 A1 ?? ?? ?? ?? 50 83 EC 4C 56 57",
    "push_shader": ("53 8B DC 83 EC 08 83 E4 F8 83 C4 04 55 8B 6B 04 "
                    "89 6C 24 04 8B EC 83 EC 18"),
    "font_draw": "55 8B EC 83 E4 F8 83 EC 40 8B C1 F3 0F 11 54 24 34 56 0F 28 CB",
}


def parse_sig(text):
    """'55 8B ?? 00' -> (bytes值, bytes掩码)，掩码 1 表示该字节要精确匹配。"""
    vals, mask = [], []
    for tok in text.split():
        if tok == "??":
            vals.append(0)
            mask.append(0)
        else:
            vals.append(int(tok, 16))
            mask.append(1)
    return bytes(vals), bytes(mask)


def parse_pe(data):
    if data[:2] != b"MZ":
        raise SystemExit("不是 PE 文件")
    e_lfanew = struct.unpack_from("<I", data, 0x3C)[0]
    if data[e_lfanew:e_lfanew + 4] != b"PE\0\0":
        raise SystemExit("找不到 PE 头")
    coff = e_lfanew + 4
    machine, nsec, timestamp = struct.unpack_from("<HHI", data, coff)
    opt_size = struct.unpack_from("<H", data, coff + 16)[0]
    opt = coff + 20
    magic = struct.unpack_from("<H", data, opt)[0]
    sec_align = struct.unpack_from("<I", data, opt + 32)[0]
    file_align = struct.unpack_from("<I", data, opt + 36)[0]
    size_image = struct.unpack_from("<I", data, opt + 56)[0]
    sec_off = opt + opt_size
    sections = []
    for i in range(nsec):
        o = sec_off + i * 40
        name = data[o:o + 8].rstrip(b"\0").decode("latin1")
        vsize, vaddr, rawsize, rawptr = struct.unpack_from("<IIII", data, o + 8)
        sections.append(dict(name=name, vaddr=vaddr, vsize=vsize,
                             rawptr=rawptr, rawsize=rawsize))
    return dict(machine=machine, nsec=nsec, timestamp=timestamp, magic=magic,
                sec_align=sec_align, file_align=file_align,
                size_image=size_image, sections=sections)


def off_to_rva(pe, off):
    for s in pe["sections"]:
        if s["rawptr"] <= off < s["rawptr"] + s["rawsize"]:
            return s["vaddr"] + (off - s["rawptr"]), s["name"]
    return None, None


def scan(data, vals, mask):
    """用最长的连续精确字节做锚点，再用掩码逐字节校验。命中的文件偏移列表。"""
    # 找最长精确串
    best_start, best_len = 0, 0
    i = 0
    while i < len(mask):
        if mask[i]:
            j = i
            while j < len(mask) and mask[j]:
                j += 1
            if j - i > best_len:
                best_len, best_start = j - i, i
            i = j
        else:
            i += 1
    if best_len < 4:
        return []
    anchor = vals[best_start:best_start + best_len]
    hits = []
    pos = data.find(anchor)
    while pos != -1:
        start = pos - best_start
        if start >= 0 and start + len(vals) <= len(data):
            ok = True
            for k in range(len(vals)):
                if mask[k] and data[start + k] != vals[k]:
                    ok = False
                    break
            if ok:
                hits.append(start)
        pos = data.find(anchor, pos + 1)
    return hits


def main():
    exe = sys.argv[1] if len(sys.argv) > 1 else DEFAULT_EXE
    with open(exe, "rb") as f:
        data = f.read()
    pe = parse_pe(data)

    print(f"文件: {exe}")
    print(f"大小: {len(data)} 字节 ({len(data) / 1048576:.2f} MB)")
    print(f"PE 时间戳: 0x{pe['timestamp']:08X}   "
          f"(AIsaac J460 是 0x{J460_BUILD['timestamp']:08X}"
          f"{' ✅ 相同' if pe['timestamp'] == J460_BUILD['timestamp'] else ' ❌ 不同'})")
    print(f"SizeOfImage: 0x{pe['size_image']:08X} "
          f"(J460 是 0x{J460_BUILD['size_of_image']:08X}"
          f"{' ✅' if pe['size_image'] == J460_BUILD['size_of_image'] else ' ❌'})")
    print(f"节数: {pe['nsec']}  节表:")
    for s in pe["sections"]:
        print(f"  {s['name']:<8} vaddr=0x{s['vaddr']:08X} vsize=0x{s['vsize']:X} "
              f"rawptr=0x{s['rawptr']:X} rawsize=0x{s['rawsize']:X}")
    print()
    print("==== 按 AIsaac 的 prologue 签名扫描 ====")
    found = {}
    for name, sig in SIGS.items():
        vals, mask = parse_sig(sig)
        hits = scan(data, vals, mask)
        exact = len(vals) - mask.count(0)
        if hits:
            print(f"  {name:<16} ✅ 命中 {len(hits)} 处（签名 {exact}/{len(vals)} 字节精确）")
            for h in hits[:3]:
                rva, sec = off_to_rva(pe, h)
                print(f"      file_off=0x{h:X}  RVA=0x{rva:08X}  节={sec}")
            found[name] = [off_to_rva(pe, h)[0] for h in hits]
        else:
            print(f"  {name:<16} ❌ 没找到（签名 {exact}/{len(vals)} 字节精确）")

    print()
    n = len(found)
    print(f"命中 {n}/{len(SIGS)} 个目标")
    if n == len(SIGS):
        print("=> 全部命中，可以直接复用 AIsaac 的 turbo 逻辑，只换地址！")
    elif n >= 3:
        print("=> 部分命中，核心的时钟/渲染 hook 若都在，就足以做虚拟时钟加速")
    else:
        print("=> 命中太少，需要自己逆向定位")

    if found:
        with open("turbo_targets.json", "w", encoding="utf-8") as f:
            json.dump(dict(build="J835", exe=exe, size=len(data),
                           timestamp=f"0x{pe['timestamp']:08X}",
                           size_of_image=f"0x{pe['size_image']:08X}",
                           found=found), f, ensure_ascii=False, indent=1)
        print("已保存 turbo_targets.json")
    return 0


if __name__ == "__main__":
    sys.exit(main())
