# Binding-of-Issac
An AI to play Issac，不采取图像识别的方法，而是去提取游戏运行内存中的数据。文件夹isaac_ai_bridge是一个mod，放置到你的游戏目录（比如D:\SteamLibrary\steamapps\common\The Binding of Isaac Rebirth\mods）里面，steam中Isaac游戏设置-属性-高级选项里面填上“--luadebug”，用于启动这个提取游戏数据的mod.打开游戏金额isaac_bridge.py如果能够看到输出说明能够连接到游戏。



ai_basic.py是一个基于规则的AI，目前还算可以。

bc_data_human.npz是人类玩家游玩的数据记录，产生于record_human.py

bc_train用来把人类玩家的游玩数据训练给AI，这个效果不佳

isaac_bridge.py  diag.py仅用于测试，测试是否能够获取数据。

两个有帮助的以撒的结合接口网址：https://cuerzor.github.io/IsaacDocs/rep/Options.html，https://moddingofisaac.com/docs/rep/enums/GridEntityType.html

---

# 多开并行训练（branch: `Aisaac_learn`）

## 为什么要多开

实时 RL 训练慢的根因是：游戏逻辑锁在 60 FPS 实时，一步决策就吃掉 1/30 秒。
参考项目 AIsaac 靠"改 C 代码做虚拟时钟 + 跳渲染"拿到单实例 20~28 倍加速，
再叠加 16/64 并行，得到 450×/1300×。**但那是 Linux AB+ 上的二进制改造，Windows 忏悔上做不到。**

所以这里只做能做的：**并行多开**，吞吐 ≈ N 倍。单实例不加速，靠数量堆。

> 换算一下期望：AIsaac 展示模型约等于 3 游戏年训练量。
> 1300× 下约 20 小时；我们 8× 下要 137 天。**所以多开救不了"从零 RL"**，
> 真正的杠杆是先用规则AI做 BC 热启动，再用 RL 微调。

## 实测结论

| 环节 | 状态 | 说明 |
|---|---|---|
| 同时跑多个实例 | ✅ 可行 | 实测 2 进程 + 2 窗口并存 10 分钟稳定 |
| 非焦点窗口注入按键 | ✅ 可行 | `PostMessage` 打 `GLFW30` 窗口，失焦照样驱动 |
| 端口分配 | ✅ 可行 | 模组在 `48532..48547` 里自己找空闲端口 |
| 端口 ↔ 窗口对号 | ✅ 可行 | 探针法：往窗口发方向键，看哪个端口的玩家动了 |
| **存档隔离** | ❌ 做不到 | 所有实例共用同一存档目录，有串写风险 |
| 菜单自动开局 | ❌ 做不到 | `MC_POST_UPDATE` 在主菜单不触发 |

## 踩过的坑（每一条都实际卡过一次）

1. **启动 exe 必须设工作目录 `cwd=游戏目录`**
   否则游戏按当前目录找 `resources/`，全部着色器加载失败 → `Caught exception` 直接崩。
   （PowerShell 的 `Start-Process` 尤其容易漏 `-WorkingDirectory`。）

2. **`Popen` 返回的 PID 不是游戏 PID**
   绕过 Steam 直接启动 exe 时，Steam 会把游戏重新拉起，真正的游戏进程是另一个 PID
   （父进程是 `steam.exe`）。所以要按"**新出现的 Isaac 窗口**"认领实例，不能按 PID 找窗口。

3. **Steam 默认只让开一个实例**
   直接第二次启动会立刻退出，因为 `SteamAPI_RestartAppIfNecessary` 会请 Steam 重新拉起自己，
   而 Steam 发现已经在跑就什么都不做。解法：给子进程设环境变量伪装成"由 Steam 启动"：

   ```
   SteamAppId=250900
   SteamGameId=250900
   SteamClientLaunch=1
   ```

4. **`USERPROFILE` 覆盖隔离存档：没用，还会崩**
   Isaac 用 Windows Known Folder API 取存档路径，不吃环境变量。强行改会让游戏 11 秒后崩溃，
   而且新目录里根本不会生成存档。这条路走不通。

5. **`MC_POST_UPDATE` 在主菜单不触发**
   模组只在进入对局后才跑，主菜单里连 socket 都不会连。
   所以**每个实例必须手动开一局**（开完就靠死亡自动重开，不用再管）。
   这也是为什么"等模组自己连上来"在主菜单会一直超时。

6. **`PauseOnFocusLost` 必须改成 0**
   在 `Documents\My Games\Binding of Isaac Repentance\options.ini` 里。
   是 1 的话，失焦的实例会暂停 → 多开等于没开。改前记得备份。

7. **mods 路径是相对工作目录的** `<cwd>/mods/`
   （见游戏自己写出的 `savedatapath.txt`）。所以理论上可以给训练实例配一个**精简 mods 目录**，
   只放桥接模组，去掉百科/掉率显示之类的模组，跑得更轻。资源目录 `resources/` 也在 cwd 下。

## 规则AI的两个致命 bug（已修复）

这两个 bug 让规则AI"看着能跑，实际寸步难行"，是它此前效果差的真正根因。

### bug 1：门被判为不可走 → 永远出不了房间

模组把可走性写成 `cells[i] = (coll == 0)`，但门是 `GridCollisionClass = 5`
（`WALL_EXCEPT_PLAYER`，玩家能穿过去）。于是 A\* 永远到不了门格，
`_go_to_door` 恒返回 `(0,0)` —— **AI 被锁死在房间里**。

修复：`isaac_env.normalize_grid()` 用 `coll` 重算可走性，0 和 5 都算可走
（Python 侧立即生效，不用重启游戏；模组 `main.lua` 也同步改了）。

### bug 2：走到门格上就停住 → 永远推不出门

A\* 从玩家格到门格，当玩家已经站在门格上时 `path` 长度 = 1，
`_go_to_door` 因为 `len(path) < 2` 直接返回 `(0,0)` —— **AI 走到门口就杵在那不动**。

修复：站在门上后改为继续朝"门外"推，方向 = 从房间中心指向门所在的那一侧。

### 诊断教训：采样别混叠

采样频率必须远高于 grid 发送频率（模组约 0.5s 才发一次 grid）。
用 0.4s 采样看状态会**混叠**，让某个实例持续错过 grid，
把完全正常的决策误报成 `(0,0)` —— 这个假象一度把排查带偏。
`diag_ai.py --watch` 改成 0.03s 采样后真相才显现。

## 照搬 AIsaac（重要转折）

`D:\DAERSHANG\AI_learning\screen_project\AIIsaac\AIsaac` 是前人（AB+ Linux）的完整实现。
读了它的 `bridge/python/isaac_bridge/` 之后发现我之前走错了方向：

| | AIsaac | 本项目旧做法 |
|---|---|---|
| 任务 | **只练单房间战斗**：`restart` + `goto d.<variant>` 传送进房间 | 想打整层楼 |
| episode | 清房 或 死亡 即结束 | 一直玩下去 |
| 奖励 | `abplus_reward.CombatHitRate` | 自己凑的几项 |
| 动作 | 移动 8 方向 + **射击 4 方向** + 炸弹/道具 | 移动 9 + 射击 9 |
| 同步 | 游戏阻塞等 Python 回动作（锁步） | 实时流式 |

**关键教训**："让 AI 自己走整层楼"比前人的设定难一个量级 —— 实测 AI 会在两个已清空
房间之间无限来回弹，录 4 分钟只走了 2 个房间，永远遇不到敌人，于是**一份战斗数据都录不到**。
AIsaac 直接用控制台命令传送进战斗房间，彻底规避了这个问题。

### 照搬下来的奖励（见 `combat_env.py`）

```
hit  = 敌人掉血 / 5.0                  # 每 5 点伤害 +1
kill = 击杀数 * 0.25
time = -(2.0 + (0.5 - 2.0) * 命中率) * 秒数
```

精妙点：**不直接惩罚打空**，而是让"时间代价"随命中率线性下降
（命中率 0 → 每秒 -2.0；命中率 1 → 每秒 -0.5）。打得准自然少被扣时间。
清房 +40、死亡 -5，最后整体乘 0.1。

### 为此新增的能力

- **模组变成双向**：Python 可以下发 `restart` / `goto d.<id>` / `exec <控制台命令>`
- `room_catalog.py`：探测哪些 `goto d.<variant>` 真的会加载出有敌人的房间
  （模仿 `abplus_room_catalog.py`），把可用房间存成 JSON 供训练随机抽

### 训练管线（已跑通）

```bash
# 1) 探房间：找出哪些 goto d.<variant> 真的会加载出敌人（一次性，约 4 分钟）
#    实测 1201 个编号里挑出 624 个战斗房间
python room_catalog.py --min 0 --max 1200 --out room_catalog.json

# 2) 环境自检：确认能进战斗房间、能发弹、有伤害/命中/受伤信号
python test_combat.py --steps 150

# 3) 多开并行训练（单房间战斗，2 实例约 49 步/秒）
python train_multi.py --attach --combat --steps 20000
```

自检正常输出长这样：

```
reset 1.8s → 进到 variant=986 的战斗房间（5 个敌人，未清空）
AI 在射击（我方泪=2）  被敌人打（HP 6→4）  命中率 0.065
清房次数 0  死亡次数 0
```

### 时序坑（全部实测踩过）

- **`restart` 之后的房间转场会持续一会儿，紧接着发的 `goto` 会被吞掉**
  → reset 后玩家还留在起始房间。必须等转场结束再发 goto。
- **`goto` 造出来的房间 `room.index` 恒为 -3**（不在楼层地图上），
  所以不能用"房间号变了"判断 goto 是否成功 → 改用"房间是否已清空"判断，不行就重发。
- **玩家一旦进入死亡结算画面，`Isaac.ExecuteCommand("restart")` 就彻底失效**，
  会永远卡在 game over。必须在 `IsDead()` 刚为真时就重开，并每秒重试。
- 探测/训练时如果不主动操作，玩家会被敌人打死（没有 AIsaac 那种 `AbpSetInvincible`），
  所以靠上面的自动重开来续命。

### 训练结果（combat_v2，30 万步 / 2.07 小时）

| 指标 | 结果 |
|---|---|
| **清房率** | 4/8 = **50%** |
| 存活率 | 4/8 = 50% |
| 平均回合长度 | 263 步 |
| 平均伤害 | 28 |
| 平均奖励 | **+1.13** |

训练曲线：`ep_rew_mean` 低谷 -0.5 → **+0.66**；`entropy_loss` -3.8 → -2.5（策略收敛）；
`ep_len` 447 → 375（清房更快，而时间是扣分的）。从"随机乱走"到"一半房间能清掉"。

```bash
python eval_combat.py --model models/combat_v2.zip --episodes 10
```

### 课程学习（可选，降低方差）

房间难度差异很大（敌人数 1~10），全难度混训会让奖励噪声很大。实测分布：
1~2 敌 82 个、1~3 敌 160 个、3~6 敌 323 个、6+ 敌 302 个、全难度 624 个。

```bash
# 先只练 1~3 个敌人的房间
python train_multi.py --attach --combat --max-enemies 3 --steps 100000

# 练稳了再放开到全难度
python train_multi.py --attach --combat --steps 300000 --load models/xxx.zip --resume
```

### 待办

- [x] 多开并行训练（2 实例 49 步/秒 → 实测 40 步/秒含 reset）
- [x] 单房间战斗管线 + AIsaac 式奖励
- [x] 规则AI 两个致命 bug（门不可走 / 卡在门上）
- [x] 房间目录 624 个 + 评估脚本
- [ ] **危险地面进观测**（火堆/尖刺的接触伤害）—— 改模组，需重启游戏
- [ ] 课程学习跑一轮（先易后难）
- [ ] action chunking（一次预测 4 步动作序列，借鉴 ACT）

### 关于"模仿学习三代"那篇文章的结论

知乎那篇讲 BC → ACT → Diffusion Policy 的文章，最值钱的实验数据是 action chunking 的消融：

| 配置 | 成功率 |
|---|---|
| 完整 ACT | 44% |
| 去掉 CVAE | 2% |
| **chunk size = 1（单步）** | **1%** |
| chunk size = 100 | 44% |

判断：

- **action chunking 值得借**——我现在是"一次决策重复 2 帧"，是退化的 chunk。
  改成 chunk=4（一段不同动作序列）能缩短有效决策步数（信用分配更容易）、让动作更平滑、减少 Python 往返。
  但 chunk 必须小：ACT 自己测出 chunk=400 反应性就掉了，躲弹幕要快。
- **ACT / Diffusion Policy 本体不适合现在**——它们是模仿学习，必须有优质示范；
  手上最好的示范源是规则AI，克隆它等于把天花板设在规则AI。Diffusion 还有 100ms 推理延迟。
- **顺带印证**：文章讲的 BC `covariate shift` 正是之前人类 BC "像随机操作"的原因，
  所以当时放弃纯 BC 的判断是对的。



## 第三轮训练：chunk=4 是负收益（实测数据）

配置：4 实例 / chunk=4 / 课程 ≤4 敌 / obs 445 维 / 新奖励（hurt+对齐势）

| 迭代 | ep_rew_mean | ep_len | entropy_loss |
|---|---|---|---|
| 1 | -0.347 | 75.7 | **-15.2** |
| 5 | -0.925 | 96.1 | **-15.2** |
| 9 | -0.681 | 89.2 | **-15.1** |

**关键：chunk=4 的熵上限是 4×(ln9+ln5) = 15.22，实测卡在 15.1~15.2 —— 策略几乎完全均匀，
18k 步没学到任何结构。** 对比 chunk=1 那轮：熵上限 3.81，实际降到 **-2.5**（明显有结构）。

而且 fps 从 40 掉到 **15**（慢 4 倍）。

**结论：`action chunking` 在从零 RL 里是负收益。**
它把动作空间从 2 维扩到 8 维，探索难度爆炸。文章里 ACT 能靠 chunking 成功，是因为
它做的是**监督学习**（有示范数据直接回归动作序列），不是从零探索。
**不能把模仿学习的结论直接搬到 RL 探索上。** → 默认退回 `chunk=1`。

## 关于"用 GPU 加速"

结论：**GPU 帮不上，我们的瓶颈不在算力。**

AIsaac 的 450×/1300× 拆开看：

| 来源 | 倍数 | 是什么 |
|---|---|---|
| 单实例虚拟时钟 + 跳渲染 | 20~28× | **改 C 代码**，让游戏逻辑脱离现实时间、且不渲染画面 |
| 并行实例数 | ×16 / ×64 | 多开 |
| **合计** | **450× / 1300×** | |

GPU 在他们项目里用于：CUDA Graph 采样、GPU 版 PPO learner、GPU 观测处理——
**全是加速"学习器"，不是加速"游戏"**。游戏本身靠的是虚拟时钟。

而且 SB3 自己也警告：`MlpPolicy` 在 GPU 上比 CPU **更慢**（本项目已默认 `--device cpu`）。

**我们真正的瓶颈（按影响排序）：**

| 瓶颈 | 影响 | 解法 |
|---|---|---|
| `chunk=4` | 慢 4 倍 | 退回 `chunk=1` |
| 实例数只有 4 | ≈60 子步/秒 | 开到 8~12 实例（接近线性） |
| reset 开销 | 每回合 2.3s / 约 12s 游戏 ≈ 16% | 缩短 settle、房间复用 |
| 游戏锁 60fps 实时 | 单实例上限 | Windows 忏悔做不到虚拟时钟（要逆向 + DLL 注入，月级工程） |

唯一接近前人加速的路径是**往 8~12 实例堆**，但那是线性 N 倍，不是 450×。

## 奖励最终形态

```
hit    +1 每造成 5 点伤害
kill   +0.25 每个击杀
time   -(2.0 + (0.5-2.0)*命中率) 每秒        ← 打得准时间更便宜
hurt   -（凹形血量曲线），丢第一颗心 1.00、最后一颗 4.09
align  gamma*Φ(s') - Φ(s), Φ = -0.2*|到最近敌人距离-150|/40   ← 站在能打到的距离
wall   -0.6 每秒顶着墙/石头/便便推            ← 用户要求，治"怼着障碍物"
clear  +40
death  -5 - 2.0*min(剩余秒数, 30)             ← 早死不该省时间成本
整体 × 0.1
```

## 相关文件

| 文件 | 作用 |
|---|---|
| `multi_instance.py` | 多开核心：PostMessage 控制器 + 多端口桥接 + 实例管理器 + 端口对号 |
| `train_multi.py` | **多开并行训练**：N 个实例同时采样，PPO 一起学 |
| `combat_env.py` | **单房间战斗环境**（AIsaac 式奖励 + 9×5 动作） |
| `room_catalog.py` | 探测哪些 `goto d.<variant>` 真的会加载出敌人 |
| `eval_combat.py` | 量化评估：清房率 / 存活率 / 伤害 / 命中率 |
| `test_combat.py` | 单房间战斗环境自检 |
| `diag_ai.py` | 规则AI 决策诊断（`--watch` 高速观测，避免采样混叠） |
| `test_multi.py` | 多开验证脚本（启动/接管 N 实例 → 等对局 → 对号 → 注入测试） |
| `test_inject.py` | 对已开着的游戏单独测 PostMessage 注入（焦点内 / 失焦） |
| `probe.py` | 探针：只听某个端口，确认模组 socket 通路是否正常 |
| `isaac_ai_bridge/main.lua` | 桥接模组：端口段 + 双向命令通道 + 死亡自动重开 |

## 吞吐实测

| 配置 | 步/秒 |
|---|---|
| 单实例 | ~25-30 |
| **2 实例并行** | **49.2** |

**线性加速确认。** 24 核机器上 8~12 实例是现实上限（受 CPU/GPU/内存限制）。

## 用法

```bash
# 1) 单实例快速验证注入是否可用（游戏已开且在对局中）
python probe.py                # 先确认模组连得上
python test_inject.py          # 再测焦点内 / 失焦注入

# 2) 多开验证（会自己启动 N 个游戏，需要你在每个窗口里手动开一局）
python test_multi.py --launch 2 --keep --run-timeout 600

# 3) 多开训练（接管已经在跑的游戏）
python train_multi.py --attach --steps 20000

# 4) 多开训练（自己启动 3 个实例，从 BC 模型热启动）
python train_multi.py --launch 3 --load models/isaac_ppo_bc --steps 200000
```

> 注意：`--device cpu` 是默认值。MlpPolicy 在 GPU 上反而更慢（SB3 官方建议）。

## 重要教训（血的代价）

- **绝对不要 `Stop-Process -Force` 杀掉正在写存档的游戏**，也不要让多个实例同时跑在对局里再强杀。
  实测这样做会让 `rep_persistentgamedata1.dat` 被重置成 3956 字节的空档。
  正常关窗（游戏自己保存）没问题。
- 万一存档没了：同目录下 `YYYYMMDD.rep_persistentgamedataN.dat` 是历史备份，
  头部都是 `ISAACNGSAVE09R`，可以直接改回正式文件名恢复。
