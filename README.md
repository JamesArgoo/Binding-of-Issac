# 以撒战术 AI

用 Lua 模组直接读取《以撒的结合：忏悔》的内存状态，Python 决策后把按键注入游戏。
不截屏、不训练、不加载模型；**当前 AI 只用 Python 标准库**。

```
游戏（isaac_ai_bridge/main.lua）
        │  JSON 状态，TCP 48532+
        ▼
Python 战术 AI（isaac_ai/）
        │  PostMessage 按键
        ▼
游戏窗口（不需要焦点）
```

## 现在能做什么

- **战斗**：预测泪弹飞行轨迹，只有“打得中且视线没被挡”才开枪；保持约 4 格交战距离。
- **走位**：模拟未来 20 帧躲弹幕、躲敌人身体；被多只怪围攻时进入脱离模式，绕开包围圈。
- **贴墙处理**：路径带开阔度代价，墙角和障碍物旁用局部转向绕行，不在原地左右横跳。
- **特殊攻击**：识别 Monstro 的 `JumpUp`/`JumpDown` 提前离开落点；估算高抛弹幕的落点。
- **探索**：清房后先拾取，再按楼层连通图去未访问的房间，不在两个已清房间之间折返。
- **机关房**：会找压力板、用炸弹炸开石头、撤离时避开 TNT。
- **整局运行**：死亡自动重开；支持同时接管多个游戏实例。

## 快速开始

### 1. 准备

- Windows + Steam 版《以撒的结合：忏悔》，Python 3.9 以上。
- Steam 里右键游戏 → 属性 → 启动选项，填 `--luadebug`（socket 通信必需）。
- 关闭失焦暂停：在 `文档\My Games\Binding of Isaac Repentance\options.ini`
  里设 `PauseOnFocusLost=0`。否则窗口一失焦，游戏就暂停。

### 2. 安装模组

```bash
python install_mod.py
```

脚本会自动找游戏目录；找不到时用 `--game "D:\...\The Binding of Isaac Rebirth"` 指定。
安装后进入游戏，在 Mods 菜单启用 **Isaac AI Bridge**。

### 3. 让 AI 玩

```bash
# 已经开好游戏并进入一局：直接接管
python play_ai.py --attach

# 或者让脚本启动游戏（启动前请关掉已有的 Isaac 窗口）
python play_ai.py --launch 1
```

运行后按 `Ctrl+C` 停止。窗口可以放在后台，但**不要最小化**——最小化会让游戏暂停。

### 4. 离线自检（不用开游戏）

```bash
python tests/test_tactical_agent.py
```

## 工作原理

1. **状态上行**：`main.lua` 每 2 帧发送一帧状态，包括玩家、敌人、子弹、网格、门、
   地图和掉落物。Python 同时可以下发 `restart` / `goto` / `spawn` 命令。
2. **决策**：`tactical_agent.py` 做三件事——解析弹道决定往哪开枪、逐方向模拟 20 帧
   决定怎么走、用 A* 和楼层图决定去哪。
3. **按键下行**：Win32 `PostMessage` 把方向键和射击键直接投给游戏窗口。多开时每个
   实例有自己的端口和窗口句柄，互不干扰。

## 文件地图

| 路径 | 用途 |
|---|---|
| `play_ai.py` | 仓库根目录的启动入口，转发到 `isaac_ai/play_ai.py` |
| `install_mod.py` | 把 `isaac_ai_bridge/` 安装到游戏 mods 目录 |
| `isaac_ai/tactical_agent.py` | **决策核心**：瞄准、走位、躲弹、A* 导航、探索、机关房 |
| `isaac_ai/multi_instance.py` | 多开基础设施：窗口查找、端口桥接、按键注入、实例管理 |
| `isaac_ai/play_ai.py` | 主循环：收状态 → 决策 → 发按键 → 死亡重开 |
| `isaac_ai/__init__.py` | 包说明，无逻辑 |
| `isaac_ai_bridge/main.lua` | 游戏模组：状态上行 + 控制命令下行 |
| `isaac_ai_bridge/metadata.xml` | 模组元数据 |
| `tests/test_tactical_agent.py` | 离线回归测试，覆盖瞄准、躲弹、导航、机关房等 32 项 |
| `tools/diag_live.py` | 诊断：状态、网格映射、按键是否生效 |
| `tools/diag_explore.py` | 诊断：清房后的选门和楼层连通图 |
| `tools/diag_air.py` | 诊断：跳跃 BOSS / 高抛弹幕的高度数据 |
| `tools/diag_monstro_live.py` | 诊断：生成 Monstro，观察跳跃躲避 |
| `tools/diag_spawn.py` | 诊断：生成指定实体并采样动画/高度 |
| `tools/probe.py` | 探针：看模组有没有连上端口 |
| `tools/test_multi.py` | 多开验证：端口↔窗口映射、无焦点按键 |
| `room_catalog.json` | 旧 PPO 用的房间目录，当前 AI 不需要 |
| `legacy/ppo/` | 归档的旧 PPO 训练/评估代码，不维护 |
| `docs/EXPERIMENTS.md` | 旧 PPO 实验记录 |

## 常用命令

| 目的 | 命令 |
|---|---|
| 接管当前游戏 | `python play_ai.py --attach` |
| 启动并自动开局 | `python play_ai.py --launch 1` |
| 跑 2 分钟自动停 | `python play_ai.py --attach --duration 120` |
| 同时跑 2 个实例 | `python play_ai.py --launch 2` |
| 接管所有已开窗口 | `python play_ai.py --attach-all` |
| 记录状态日志 | `python play_ai.py --attach --log logs/run.jsonl` |
| 检查模组版本是否一致 | `python install_mod.py --check` |
| 离线测试 | `python tests/test_tactical_agent.py` |
| 多开环境验证 | `python tools/test_multi.py --launch 2 --keep` |

新启动的实例如果停在主菜单，脚本会自动按 Enter；也可以在窗口里手动按 Enter。

## 验证情况

- 离线回归：`32/32 通过`。
- 已实机确认：Monstro 跳跃躲避不掉血；清房后能连续进入未访问房间；多怪近身会触发脱离。
- 已离线覆盖、实机尚未专门复现：墙角/障碍物绕行、机关房炸路与压力板、高抛弹幕。

## 已知限制

- **不要最小化游戏窗口**：Isaac 最小化后会暂停 `onUpdate`，模组不再发状态。
- **一个端口只能有一个客户端**：跑 `--launch` 前先关掉已有窗口，否则状态和按键会串台。
- **死亡要立刻重开**：进入死亡结算后 `restart` 会失效，主循环会自动补发。
- **不要在建档写入时强杀进程**：可能把存档写成空档。正常停止请用 `Ctrl+C`。
- **必须用 `--luadebug` 启动**：直接双击 exe 时模组无法建立 socket。
- 当前策略是确定性的规则/搜索 AI，不会从对局中学习；调整行为改 `Config` 里的权重。

## 旧 PPO 实验

`legacy/ppo/` 是早期强化学习路线的存档，需要 `requirements.txt` 里的
torch/SB3 等依赖，当前 AI 不使用，也不再维护。相关背景见 `docs/EXPERIMENTS.md`。
