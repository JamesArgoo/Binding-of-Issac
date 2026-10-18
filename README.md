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

## 相关文件

| 文件 | 作用 |
|---|---|
| `multi_instance.py` | 多开核心：PostMessage 控制器 + 多端口桥接 + 实例管理器 + 端口对号 |
| `train_multi.py` | **多开并行训练**：N 个实例同时采样，PPO 一起学 |
| `test_multi.py` | 多开验证脚本（启动/接管 N 实例 → 等对局 → 对号 → 注入测试） |
| `test_inject.py` | 对已开着的游戏单独测 PostMessage 注入（焦点内 / 失焦） |
| `probe.py` | 探针：只听某个端口，确认模组 socket 通路是否正常 |
| `isaac_ai_bridge/main.lua` | 桥接模组，支持端口段 + 掉线守住原端口 + 死亡自动重开 |

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

## 待办

- [x] 把多开接进训练脚本：N 个 `IsaacEnv` 并行采样（`train_multi.py`，2 实例 49 步/秒）
- [x] 存档串写风险：确认可按弃用处理，训练用全新存档（用户确认）
- [ ] 重新实现 `record_bc.py`，用规则AI当老师录 BC 数据 → BC 热启动 → RL 微调
- [ ] 评估精简 mods 目录能提速多少（mods 路径相对 cwd）
- [ ] 试 `VSync=0` 看单实例能否跑超 60fps
- [ ] 多开规模上限实测（4/6/8 实例的步/秒曲线）
