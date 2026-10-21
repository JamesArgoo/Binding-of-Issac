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

**2. 观测：位置归一化 + 固定槽位，地形两种模式。**

位置按房间尺寸归一化，敌人/子弹用固定槽位按距离排序，所以 15×9 和 15×16 的房间
进网络后分布一致。基础部分共 **396 维**（玩家朝向 `fire_dir`/`head_dir`、无敌帧、
伤害/泪速/移速、敌人 16×8、敌弹 48×5、房间 2、门 4×3），地形的编码有两种：

| 模式 | 地形 | 总维度 | 说明 |
|---|---|---|---|
| `local`（默认） | 玩家周围 7×7 三值窗口 | **445** | 平移不变；但实测只覆盖房间 **20.4%**，84% 的墙/石头/坑在盲区 |
| `full`（`--full-terrain`） | 整间房铺进 28×16 画布 × 2 通道（可走/危险） | **1292** | 全房间可见；代价是失去平移不变性 |

`full` 模式把房间贴画布左上角，画布 28×16 是标准房间最大外接框
（1×1=15×9 / 1×2=15×16 / 2×1=28×9 / 2×2=28×16），多出来的部分填 0（等价"走不了"）。
用 `python dump_state.py --replay frames.json --full-terrain` 可以直接对比两种编码。

**2b. 危险格只收静态地面危险物。**
`cdmg > 0` 的实体标成"能走但会掉血"。必须排除：玩家本体、玩家发射的一切、所有会动的弹幕
—— 弹幕块已经有精确位置+速度，而地形每 15 帧才刷新，把会动的东西画进地形只会得到
过期 250ms 的噪声。（实测踩过：只排除玩家本体时，玩家自己的泪弹会被标成危险格，
等于教 AI 躲自己的子弹。）

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
| `viz_viewer.py` | **可视化**：人类看到的房间 vs 网络收到的观测，并排对照 |
| `inspect_policy.py` | **离线查策略**：不开游戏，看模型的动作分布/熵/价值，并做"敌人换方向动作跟不跟"的归因实验 |
| `selftest_learn.py` | **可学性自检**：用同一套观测造可控假环境，验证 PPO 到底能不能学会"朝敌人射" |
| `bc_pretrain.py` | **BC 热启动**：用规则给策略头灌"瞄准"能力（PPO 从零探索不出来，见下） |
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
python viz_viewer.py --replay frames.json

# 对比两种地形编码（同一个 frames.json，不用再开游戏）
python dump_state.py --replay frames.json                 # 445 维，7×7 局部
python dump_state.py --replay frames.json --full-terrain   # 1292 维，整间房
python viz_viewer.py  --replay frames.json --full-terrain
```

`dump_state.py` 输出七段：通路状态 / 玩家字段全表 / 实体明细 / 地形（含字符画、
`coll` `ent` 直方图、危险格）/ 门与地图 / **观测逐维** / 多帧取值统计
（会点出"哪些维全程恒定 = 死输入"）。

`viz_viewer.py` 一张图里六个面板：① 世界俯视图（人类视角）② AI 的地形视野
③ 敌人槽位 16×8 ④ 敌弹槽位 5×48 ⑤ 状态与覆盖率 ⑥ 分块强度。

看这两个工具能立刻发现的两个观测缺陷：

- **地形只喂了玩家周围 7×7**（默认 `local` 模式），房间另一头的石头/坑 AI 完全不知道
  —— 实测 84% 的非空地格在窗口外。`--full-terrain` 可消除；
- **`grid.ent`（每格是石头/便便/坑/尖刺）收到了却没用**，观测里只剩 `coll` 二值，
  所以"石头"和"便便"、"墙"和"实心障碍"在 AI 眼里长得一模一样。（待做）

## 用法

```bash
# 1) 启动 N 个实例（之后在每个游戏窗口手动按 Enter 开一局）
python test_multi.py --launch 4 --keep

# 2) 训练（--max-enemies 4 = 课程学习，先只练 <=4 敌人的房间）
python train_multi.py --attach --combat --chunk 1 --max-enemies 4 --steps 200000

# 2b) 全房间地形（观测 445→1292，网络自动加宽到 256×256）
python train_multi.py --attach --combat --full-terrain --steps 200000

# 3) 评估与诊断
python eval_combat.py --model models/combat_v4.zip --episodes 10
python diag_combat.py --model models/combat_v4.zip --episodes 6
```

注意：`--full-terrain` 训练的模型不能和 `local` 的模型互换（观测维度不同），
`--load` 旧模型时会自动按模型自己的观测维度对齐地形模式；`--model latest` 直接选最新 checkpoint。

## 训练不动时先查这两件事（都不用开游戏）

```bash
# 1) 模型现在到底想干什么：熵接近 100% = 还是随机；归因实验 1/4 = 完全不看敌人
python inspect_policy.py --model latest

# 2) 这套观测+动作有没有可能学会（可控假环境，几十秒）
python selftest_learn.py --steps 20000     # 强奖励下能学会 -> 观测没问题
python selftest_learn.py --sweep           # 扫奖励强度，找"学得动"的阈值
```

实测结论（2026-10）：真实环境一次命中约 +0.07，每 10~20 步才中一次，
**每步瞄准信号约 0.005**；而 `--sweep` 显示每步信号低于 ~0.1 时 PPO 完全学不动
（熵卡在 92%，动作与敌人位置无关）。所以**命中奖励和时间代价的比值**是当前头号瓶颈：

```bash
python train_multi.py --attach --combat --full-terrain --max-enemies 2 \
    --hit-rew 1.0 --cost-miss 0.5 --fps 15 --steps 200000
```

## 关键：瞄准必须用 BC 热启动，PPO 从零探索不出来

归因实验（`inspect_policy.py`）在多个模型上一致地显示：敌人放在玩家上/右/下/左，
**射击概率几乎完全一样**（差 <1%）。PPO 训 45k / 300k / 355k 步都一样，
可控假环境里即使给 +0.5/步的密集瞄准奖励、跑 30k 步，也只学会"一直射同一个方向"。

而用**同一套观测、同一个网络**做监督学习，15 个 epoch 就四方向 4/4、每方向 100% 正确。

原因：PPO 自己的移动策略会把敌人方位压到很窄的分布里，而射中又是稀疏奖励，
梯度分不出"哪个方向更好"。瞄准是"一学就会、但随机探索碰不到"的子技能。

```bash
# 1) 生成 BC 种子（几秒钟，不用开游戏；有真实帧更好）
python bc_pretrain.py --frames frames.json --out models/bc_seed

# 2) 从这个种子接着 PPO 微调（移动策略仍由 PPO 学）
python train_multi.py --attach --combat --full-terrain --max-enemies 2 \
    --load models/bc_seed --hit-rew 1.0 --cost-miss 0.5 --fps 15 --steps 200000

# 3) 随时检查：归因应该是 4/4，且熵要往下走
python inspect_policy.py
```

**第二次验证（更狠的用法）：连"何时该开枪"也交给 BC。**

命中率一直卡在 30~45%，而完美门控能到 47~65%。用 `--gate` 把"该不该开火"
也做成监督标签（判据是**提前量**：泪弹飞到时敌人还在不在轴上），
再用 `--load` + `--head shoot` **只重写射击头**、保留 PPO 学到的走位：

```bash
python bc_pretrain.py --frames frames.json --terrain full \
    --load models/<现有模型> --head shoot \
    --gate --gate-tol 20 --enemy-speed 2.0 --epochs 20 --out models/bc_gate
```

自检 4 会打印门控行为（敌人正右 150、下漂 +3/帧 → 应该输出「不射」）。
接上去 PPO 微调后命中率从 30~45% 抬到 **40~54%**，受伤反而更低。

## 先量天花板再调参：`oracle_eval.py`

调奖励权重之前，先搞清楚"最好能有多准"。这个脚本把**规则教师**、**纯随机**、
**训练好的模型**放进**同一套真实环境**打同样一批房间，输出：
命中率、**决策对齐率**、清房率、受伤，以及一张按距离分桶的"此刻开枪能不能中"表。

```bash
python oracle_eval.py --policy oracle       --max-enemies 1     # 规则教师（下限参照）
python oracle_eval.py --policy random       --max-enemies 1     # 瞎打
python oracle_eval.py --policy model --model latest             # 当前模型
python oracle_eval.py --policy oracle --gate --gate-los         # 完美门控＝命中率天花板
```

**比较模型时必须用 `--models` 在同一进程里轮转，否则结论会翻转。**
`oracle_eval` 每个进程 attach 到的**游戏实例不同**，而多个实例跑几小时后
道具/层数/属性早就分叉了，"命中数 = 伤害 ÷ 单发伤害"这个估算会直接失真
（实测出现过 111.7% 的命中率）。另外房间是随机挑的，10 个回合方差极大：

```bash
# ✅ 正确：一个进程里轮流打同一批房间
python oracle_eval.py --policy model \
    --models models/best_aim,models/best_combat_r5 \
    --episodes 30 --seed 7 --max-enemies 1

# ❌ 错误：分两次跑再比
python oracle_eval.py --policy model --model models/best_aim --episodes 20
python oracle_eval.py --policy model --model models/best_combat_r5 --episodes 20
```

实测得到的三个关键数字（换房间/换敌人会变，但量级稳定）：

- 规则教师只有 **9.5~18.9%** 命中率（它不看视线，55 发 0 伤害是常事）
  → 所以"PPO 学不会瞄准"这个判断是错的，PPO 早就比教师准得多
- 完美门控（只在预测能命中时开枪）**46.8~65%**，而且用 **1/3 的子弹打出同样伤害**
- 距离 80~120 的命中窗口 57%，120~180 只有 14%，>300 只有 6.7%
  → 这就是 `--d-opt 100` 的依据（默认 150 落在窗口外）

## 现在有两个版本，按需要选

| 模型 | 命中率 | 清房率 | 平均受伤 | 发弹/回合 | 特点 |
|---|---|---|---|---|---|
| `models/best_aim.zip` | **33%** | 78% | **0.71** | **10.6** | **推荐**：打得准、挨打少、最省子弹 |
| `models/best_combat_r5.zip` | 24% | **100%** | 1.20 | 16.6 | 老基线：清房最稳，但靠"多喷"堆伤害 |

> 上表是**单进程轮转、同一批 27 个房间**（`--models A,B,C --seed 7`）的结果，
> 每个模型各 9 回合。不同房间集下命中率的绝对值会大幅波动（实测 19%~83%），
> 但 `best_aim` 在多次对比里都稳定优于旧基线，差距在 1.35~4 倍之间。

```bash
# 看"打得准"的版本玩
python play_combat.py --model models/best_aim --max-enemies 1
# 看"稳"的版本玩
python play_combat.py --model models/best_combat_r5 --max-enemies 1
```

命中率提升了 2.4 倍是实测确认的；清房率和受伤在种子之间摇摆，
主要被"敌人打不到、空喷 55 发 0 伤害"这类病理回合拖累 —— 这是下一步最该修的东西。

## 打准相关的参数速查

| 参数 | 默认 | 作用 | 实测结论 |
|---|---|---|---|
| `--d-opt` | 150 | 理想交战距离 | **改成 100 是最大的一次提升**：清房率+5、受伤−40%、回报从 −127 升到 −25、回合长度 3.5 倍变短 |
| `--align-off` | 1.0 | 对齐势权重（奖励把非主轴偏移消掉） | 2.5 略好于 1.0，但不是决定性的 |
| `--align-shoot-pen` | 3.0 | 没对准就开枪的惩罚（决策级） | 有效，但**必须按发结算**（见 `--no-align-on-spawn`） |
| `--align-shoot-bonus` | 1.0 | 对准了开枪的小奖励 | 同上 |
| `--align-tol` | 18 | 判定对齐的容差（约半格） | 和敌人半径+泪弹半径对得上，一般不用动 |
| `--dist-w` | 0.2 | 距离势权重 | 0.5 只把清房率抬了一点，命中率不动；别指望它治龟缩 |
| `--lead-align` | 关 | 用提前量预测判对齐 | **默认关**：罚更严 → 不敢开枪 → 回合拖长、清房率掉 |
| `--no-align-on-spawn` | （默认按发判） | 退回"每帧按着开火键都判" | 会让惩罚密度变成实际发弹数的 ~10 倍，学成"松手不射" |
| `--hit-rew` / `--miss-pen` | 2.0 / 2.0 | 命中加分 / 空打扣分（结果级） | 盈亏平衡命中率 ≈ `miss_pen/(hit_rew·伤害+miss_pen)` |
| `--hurt-scale` | 1.0 | 受伤惩罚倍率 | 25 才压得住"贴脸换血"；配合 `--reset-mode auto` 用 |


## 阶段化训练：先学不挨打，再学打输出

`--hit-rew 1.0` 把命中奖励抬上去之后出现了一个副作用：**一个回合打出的伤害值约 28 分，
而掉光 6 点血只扣 11.95 分 —— 打输出比挨打"贵"2.3 倍，于是贴脸换血成了最优解**
（实测回合回报 +5.6，同时每回合掉光血）。

按阶段调：

```bash
# 阶段 A：重罚受伤（10 倍），先把它从"撞怪换血"掰回来
python train_multi.py --attach --combat --full-terrain --max-enemies 2 \
    --load models/bc_seed2 --hit-rew 1.0 --cost-miss 0.5 --reset-mode keep \
    --hurt-scale 10 --steps 200000

# 阶段 B：等"平均受伤"降下来之后，加严格版"不允许受伤"
python train_multi.py --attach --combat --full-terrain --max-enemies 2 \
    --load latest --hit-rew 1.0 --cost-miss 0.5 \
    --hurt-scale 10 --fail-on-hurt --steps 200000
```

| 参数 | 作用 |
|---|---|
| `--hurt-scale N` | 受伤惩罚倍率（默认 1.0）。`10` 时掉 1 点血 ≈ 3 次泪弹命中，能压过换血收益 |
| `--fail-on-hurt` | 掉**任何**一点血就结束本回合（不加死亡惩罚）。会自动把 `--reset-mode` 改回 `restart` |

进度行第二行直接给这三个数，用来判断"学会躲没有"：

```
  最近 30 回合：平均受伤 0.42 点  平均输出 26 伤害  清房率 87%
```

`平均受伤` 从 6 降到 1 以下，就说明躲避学会了。

## 怎么练得快（实测瓶颈与放大倍数）

`time/fps` 的物理上限是 **游戏逻辑帧率 ÷ frame_skip**。项目原来的数字（单实例
25-30 步/秒）是在游戏跑满 60fps 时测的；如果你的游戏只有 ~30fps，单实例就只有
~15 步/秒。三个独立的乘数：

| 手段 | 效果 | 怎么做 |
|---|---|---|
| **模组瘦身**（默认已开） | JSON 12.5KB → 6.2KB（-50%），实体 31 → 12（特效占 61%） | `python install_mod.py` |
| **每帧都发 + frame_skip=1** | 决策频率 = 游戏帧率，单实例 15 → 30 步/秒 | 模组 `SEND_EVERY=1`（已改）+ `--frame-skip 1`（默认） |
| **多开** | 线性，4 实例 ≈ 4x（CPU 线程数就是上限，24 线程可开 6~8） | `python test_multi.py --launch 4 --keep` 再 `--attach` |
| **turbo**（DLL 加速时钟+跳渲染） | README 记录约 5x，需 `--launch` 而非 `--attach` | `--launch N --turbo --turbo-speed 20` |

`dump_state.py` 的 ① 通路状态里现在会显示 **模组耗时 ms/帧**：超过 ~8ms（30fps 时
帧预算 33ms 的四分之一）说明是模组在拖慢游戏，先优化模组；如果很小而帧率仍是 30，
那就是 VSync / 电源模式 / 游戏跑在核显上的问题，去改游戏侧设置。

```bash
# 完整加速组合（4 实例 + 瘦身后的模组 + frame_skip=1）
python install_mod.py
python test_multi.py --launch 4 --keep
python train_multi.py --attach --combat --full-terrain --max-enemies 2 \
    --load models/bc_seed2 --hit-rew 1.0 --cost-miss 0.5 --steps 400000
```

启动时会**实测游戏帧率**并自动算出决策频率（不再需要手填 `--fps`），打印类似
`实测游戏帧率 59.8 fps ÷ frame_skip 1 → 决策 59.8 步/秒`。

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
