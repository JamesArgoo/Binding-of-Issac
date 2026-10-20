# Binding-of-Issac — 用强化学习打以撒

不截屏识图，直接把游戏内存里的状态经 Lua 模组 + socket 送到 Python，
用 PPO 学**单房间战斗**，输出按键再注入回游戏。

```
Isaac(模组 main.lua) --状态/JSON/socket--> Python(观测/奖励) --PPO--> 动作
        ^                                                          |
        +---------------- PostMessage 按键注入 <--------------------+
```

- **模组**：每 2 帧发一帧状态（玩家/敌人/子弹/网格/门/地图），并接收 Python 下发的
  控制台命令（`restart` / `goto d.<id>`）。需要 Steam 启动项 `--luadebug`。
- **多开**：N 个实例，端口段 48532~48547 自动分配；用探针把"端口↔窗口"对上号；
  按键用 `PostMessage` 直接投给目标窗口 —— **不需要窗口焦点**（这是多开的关键）。
- **训练**：每实例一个 `CombatEnv`，线程并行 step，包成 SB3 的 `VecEnv` 交给 PPO。

## 核心设计（为什么这么做）

**1. 只练单房间战斗，不练走整层楼。**
reset 时用控制台命令 `restart` + `goto d.<variant>` 把玩家传送进有敌人的房间，
清房或死亡即结束。照搬 AIsaac 的任务设定 —— 让 AI 自己走整层楼难一个量级
（实测会在两个已清空房间之间无限来回弹，永远遇不到敌人）。624 个可用房间由
`room_catalog.py` 探出来。

**2. 观测对房间尺寸不变。**
位置按房间尺寸归一化；地形只看玩家周围 7×7 局部窗口；敌人/子弹用固定槽位按距离排序。
所以 15×9 和 26×14 的房间进网络后分布一致。共 **445 维**（含玩家朝向 `fire_dir`/`head_dir`、
无敌帧、伤害/泪速/移速，以及网格三值编码：0 走不了 / 0.5 能走但危险(火堆) / 1 安全）。

**3. 奖励**

```
hit    +1 每造成 5 点伤害
kill   +0.25 每次击杀
time   -(2.0 + (0.5-2.0)*命中率) 每秒               # 打得准，时间更便宜
hurt   -凹形血量曲线                                 # 丢最后一颗心比第一颗贵 4 倍
align  gamma*Phi(s')-Phi(s), Phi = -0.2*|d-150|/40  # 站在刚好能打到的距离
wall   -0.6/秒 顶着墙/石头/便便推
clear  +40
death  -5 - 2.0*min(剩余秒数, 30)                    # 早死不该省时间成本
整体 x0.1
```

要点：**不直接惩罚打空**，而是让时间代价随命中率下降；血量用凹形曲线让 AI 残血时更谨慎。

**4. 动作**：移动 8 方向 + 停，射击 4 方向 + 不射（泪滴只有四个方向）。

## 文件

| 文件 | 作用 |
|---|---|
| `isaac_ai_bridge/main.lua` | 游戏模组：状态上行 + 命令下行 |
| `isaac_data.py` | 枚举翻译成中文 + 445 维观测对照表 + `StateFeed` 数据源（探针/可视化共用） |
| `dump_state.py` | **数据探针**：把一帧状态逐字段摊开，看内存里到底有什么 |
| `viz_viewer.py` | **可视化**：人类看到的房间 vs 网络收到的 445 个数字，并排对照 |
| `isaac_bridge.py` | 状态解析小工具（敌人/敌方子弹/Pickup） |
| `isaac_env.py` | 观测构建 `build_obs`、网格可走性归一化、危险格 |
| `combat_env.py` | 单房间战斗环境（奖励 + reset + chunk） |
| `multi_instance.py` | 多开：PostMessage 控制器、端口桥接、实例管理、端口对号 |
| `train_multi.py` | 多开并行 PPO 训练 |
| `room_catalog.py` / `.json` | 探测/记录有敌人的房间（624 个） |
| `eval_combat.py` / `diag_combat.py` / `test_combat.py` | 评估 / 诊断 / 自检 |
| `test_multi.py` / `probe.py` | 多开启动验证 / socket 探针 |
| `ai_basic_4.py` | 规则AI（8方向移动 + 4方向甩枪），备选与对照 |

## 先看清 AI 能看到什么（改观测/调奖励前必做）

```bash
# 不开游戏，直接看效果
python dump_state.py --demo --frames 30
python viz_viewer.py --demo --snapshot demo_viz.png     # 存一张图慢慢看

# 开着游戏（steam 启动项带 --luadebug，模组已启用，且已经进了一局）
python dump_state.py --frames 120 --json frames.json
python viz_viewer.py
```

`dump_state.py` 输出七段：通路状态 / 玩家字段全表 / 实体明细 / 地形（含字符画、
`coll` `ent` 直方图、危险格）/ 门与地图 / **445 维观测逐维** / 多帧取值统计
（会点出"哪些维全程恒定 = 死输入"）。

`viz_viewer.py` 一张图里六个面板：① 世界俯视图（人类视角，含 7×7 地形视野框）
② AI 的地形视野 7×7 ③ 敌人槽位 16×8 ④ 敌弹槽位 5×48 ⑤ 状态与覆盖率 ⑥ 445 维分块强度。

看这两个工具能立刻发现的两个观测缺陷：

- **地形只喂了玩家周围 7×7**，房间另一头的石头/坑 AI 完全不知道；
- **`grid.ent`（每格是石头/便便/坑/尖刺）收到了却没用**，观测里只剩 `coll` 二值，
  所以"石头"和"便便"、"墙"和"实心障碍"在 AI 眼里长得一模一样。

## 用法

```bash
# 1) 启动 N 个实例（之后在每个游戏窗口手动按 Enter 开一局）
python test_multi.py --launch 4 --keep

# 2) 训练（--max-enemies 4 = 课程学习，先只练 <=4 敌人的房间）
python train_multi.py --attach --combat --chunk 1 --max-enemies 4 --steps 200000

# 3) 评估与诊断
python eval_combat.py --model models/combat_v4.zip --episodes 10
python diag_combat.py --model models/combat_v4.zip --episodes 6
```

## 已知约束（踩过的坑，别踩第二遍）

- **必须带 `python` 前缀**：Windows 把 `.py` 关联到 `C:\Windows\py.exe`，它默认用
  `D:\Python313\python.exe` —— 不是激活的 Issac_env。直接敲 `dump_state.py` 会报
  `No module named 'gymnasium'`（pip 装到的和脚本跑的不是同一个 Python）。
- **必须设工作目录**：启动 exe 要 `cwd=游戏目录`，否则找不到 `resources/`，着色器全失败直接崩。
- **Steam 默认只允许一个实例**：要设 `SteamAppId` / `SteamGameId` / `SteamClientLaunch`
  伪装成"由 Steam 启动"，绕过 `SteamAPI_RestartAppIfNecessary`。
- **`PauseOnFocusLost=0`**（`options.ini`）：否则失焦的实例会暂停，多开等于没开。
- **死亡必须立刻 `restart`**：一旦进入死亡结算画面，`Isaac.ExecuteCommand("restart")` 就失效，
  玩家会永远卡在 game over。
- **`goto` 造出来的房间 `room.index` 恒为 -3**，不能用"房间号变了"判断传送是否成功，
  要看"房间是否已清空"。
- **不要在游戏写存档时强杀进程**：会把存档重置成 3956 字节的空档
  （`YYYYMMDD.rep_persistentgamedataN.dat` 是历史备份，可改名恢复）。

## 吞吐现状

单实例被锁在 60fps 实时，吞吐只能靠多开线性堆：

| 配置 | 步/秒 |
|---|---|
| 单实例 | ~25-30 |
| 2 实例 | ~49 |
| 4 实例 | ~52（含 reset 开销，已接近瓶颈） |

`--chunk > 1`（ACT 式动作分块）实测**负收益**：动作空间从 2 维扩到 8 维，
熵卡在上限（15.2）不下降，且慢 4 倍。默认 `chunk=1`。
