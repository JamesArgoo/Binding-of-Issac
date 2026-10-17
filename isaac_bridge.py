import json
import os
import socket
import threading
import time
HOST="127.0.0.1"
PORT=48532

#实体
EntityType={
    "PLAYER":1,
    "TEAR":2,
    "FAMILIAR":3,#跟班
    "BOMB":4,
    "PICKUP":5,#掉落物
    "SLOT":6,
    "LASER":7,
    "KNIFE":8,
    "PROJECTILE":9,
    "EFFECT":1000,
}

#房间类型
RoomType={
    "DEFAULT":1,
    "SHOP":2,#商店
    "TREASURE":4,#宝箱房
    "BOSS":5,#头目房
    "SECRET":7,#隐藏房
    "SUPERSECRET":8,#超级隐藏房
    "ARCADE":9,#赌博房
    "CURSE":10,#刺房
    "DEVIL":14,#恶魔房
    "ANGEL":15,#天使房
    "LIBRARY":12,#书房
    "PLANETARIUM":28,
}


class IsaacBridge:
    #在后台线程监听游戏发来的 JSON 状态，供主循环随时读取最新一帧。
    def __init__(self,host=HOST,port=PORT):
        self.host,self.port=host,port
        self._latest=None
        self._lock=threading.Lock()
        self._running=False
        self._thread=None
        self._server=None
        self._last_time=0.0

    def start(self):
        self._running = True
        self._server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self._server.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self._server.bind((self.host, self.port))
        self._server.listen(5)
        self._server.settimeout(1.0)
        self._thread = threading.Thread(target=self._loop, daemon=True)
        self._thread.start()
        print(f"[bridge] 监听 {self.host}:{self.port}，等待游戏连接...")

    def _loop(self):
        while self._running:
            try:
                conn, _ = self._server.accept()
            except socket.timeout:
                continue
            except OSError:
                break
            print("[bridge] 游戏已连接")
            buf = b""
            conn.settimeout(1.0)
            while self._running:
                try:
                    data = conn.recv(65536)
                except socket.timeout:
                    continue
                except OSError:
                    break
                if not data:
                    break
                buf += data
                while b"\n" in buf:
                    line, buf = buf.split(b"\n", 1)
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        state = json.loads(line.decode("utf-8"))
                        with self._lock:
                            self._latest = state
                            self._last_time = time.time()
                    except (json.JSONDecodeError, UnicodeDecodeError):
                        pass
            conn.close()
            print("[bridge] 游戏断开，等待重连...")

    def get_state(self):
        """返回最新一帧状态 dict；还没数据时返回 None。"""
        with self._lock:
            return self._latest

    def stop(self):
        self._running = False
        if self._server:
            try:
                self._server.close()
            except OSError:
                pass


# ---------- 供 AI 使用的小工具 ----------

def get_player(state):
    return state.get("player")


def get_enemies(state):
    """返回所有敌人实体（含 Boss，按 IsEnemy/IsBoss 标记）。"""
    return [e for e in state.get("entities", []) if e.get("enemy") or e.get("boss")]


def get_tears(state):
    """返回所有泪（子弹）实体。"""
    return [e for e in state.get("entities", []) if e.get("type") == EntityType["TEAR"]]


def get_enemy_tears(state):
    return [e for e in state.get("entities", [])
            if e.get("type") == EntityType["TEAR"]
            and e.get("spawner_type", 0) != EntityType["PLAYER"]]


def get_player_tears(state):
    return [e for e in state.get("entities", [])
            if e.get("type") == EntityType["TEAR"]
            and e.get("spawner_type", 0) == EntityType["PLAYER"]]


# "子弹类"实体：泪(2) 激光(7) 刀(8) 投射物(9)
PROJECTILE_TYPES = (2, 7, 8, 9)


def get_projectiles(state):
    """所有子弹类实体（泪/激光/刀/投射物），供躲弹幕和观测使用。

    注意：很多敌人打的是 ENTITY_PROJECTILE(9)，不是 ENTITY_TEAR(2)，
    所以躲弹幕必须同时覆盖 2/7/8/9，否则会漏掉敌人子弹。
    """
    return [e for e in state.get("entities", []) if e.get("type") in PROJECTILE_TYPES]


def get_enemy_projectiles(state):
    """敌方子弹（泪/激光/刀/投射物，且不是玩家自己射的）。"""
    return [e for e in state.get("entities", [])
            if e.get("type") in PROJECTILE_TYPES
            and e.get("spawner_type", 0) != EntityType["PLAYER"]]


def get_pickups(state):
    """掉落物 / 道具底座 / 饰品（都是 type=5，靠 variant 区分）。"""
    return [e for e in state.get("entities", []) if e.get("type") == EntityType["PICKUP"]]


# PickupVariant
_PICKUP_NAMES = {10: "红心", 20: "金币", 30: "钥匙", 40: "炸弹",
                 100: "道具底座", 350: "饰品"}


def pickup_label(e):
    return _PICKUP_NAMES.get(e.get("variant"), f"variant={e.get('variant')}")


def nearest_entity(state, entities):
    """返回离以撒最近的实体（dict 或 None）。"""
    p = get_player(state)
    if not p or not entities:
        return None
    return min(entities, key=lambda e: (e["x"] - p["x"]) ** 2 + (e["y"] - p["y"]) ** 2)


def nearest_enemy(state):
    """最近的敌人（返回实体 dict 或 None）。"""
    return nearest_entity(state, get_enemies(state))


def get_secret_rooms(state):
    """从楼层地图里找隐藏房 / 超级隐藏房，返回 [{grid, type, visited}]。"""
    out = []
    for r in state.get("map", []):
        if r["type"] in (RoomType["SECRET"], RoomType["SUPERSECRET"]):
            out.append(r)
    return out


def demo_loop():  # 主循环：清屏固定刷新的仪表盘
    bridge = IsaacBridge()
    bridge.start()
    try:
        while True:
            s = bridge.get_state()
            if s:
                os.system("cls")  # 清屏，避免滚动刷屏看不清
                if "hello" in s:
                    print(f"[bridge] 收到 hello (frame={s.get('frame')}) —— socket 数据通路 OK，进一局")
                elif "error" in s:
                    print(f"[bridge] 模组报错:\n  {s['error']}")
                elif "menu" in s:
                    print(f"[bridge] 模组在菜单/未进对局 (frame={s.get('frame')})，进一局试试")
                else:
                    p = s["player"]
                    enemies = get_enemies(s)
                    enemy_tears = get_enemy_tears(s)
                    player_tears = get_player_tears(s)
                    pickups = get_pickups(s)

                    print("==== Isaac 实时状态 (Ctrl+C 退出) ====")
                    print(f"玩家: ({p['x']:.0f}, {p['y']:.0f})  HP={p['hp']}/{p['max_hp']} 魂心={p['soul_hp']}  "
                          f"金币={p['coins']} 钥匙={p['keys']} 炸弹={p['bombs']}")
                    print(f"房间: 类型={s['room']['type']} 已清={s['room']['clear']} | "
                          f"敌人={len(enemies)} 敌方泪={len(enemy_tears)} 我方泪={len(player_tears)} "
                          f"掉落/道具={len(pickups)}")

                    ne = nearest_entity(s, enemies)
                    if ne:
                        print(f"最近敌人: ({ne['x']:.0f},{ne['y']:.0f}) type={ne['type']} "
                              f"hp={ne['hp']} boss={ne['boss']}")
                    nt = nearest_entity(s, enemy_tears)
                    if nt:
                        print(f"最近敌方泪: ({nt['x']:.0f},{nt['y']:.0f}) "
                              f"速度=({nt['vx']:.1f},{nt['vy']:.1f})")
                    if pickups:
                        labels = ", ".join(f"{pickup_label(e)}@({e['x']:.0f},{e['y']:.0f})" for e in pickups[:5])
                        print(f"掉落/道具: {labels}")
            time.sleep(0.2)  # 5Hz 刷新，眼睛看得清
    except KeyboardInterrupt:
        print("\n退出")
    finally:
        bridge.stop()


if __name__ == "__main__":
    demo_loop()
