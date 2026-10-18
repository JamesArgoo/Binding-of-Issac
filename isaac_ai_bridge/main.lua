-- isaac_ai_bridge/main.lua
-- 《以撒的结合：忏悔》-> Python 状态桥接模组
-- 通过 socket 把每帧状态发给本机 Python（需要 --luadebug）。
-- 连上后先发 hello；出错时把错误也通过 socket 发回来，方便排查。

local mod = RegisterMod("Isaac AI Bridge", 1)
local json = require("json")

local HOST = "127.0.0.1"
local BASE_PORT = 48532
local MAX_INSTANCES = 16   -- 多开：端口 48532..48547，各实例自己找空闲端口连
local SEND_EVERY = 2  -- 每 2 帧发一次（RL 决策约 30Hz，够了，减少带宽）
local MAP_EVERY = 30
local GRID_EVERY = 15  -- 每 15 帧刷新一次地形（实时反映炸开石头/便便后的变化）

local sock = nil
local frame = 0
local game = Game()
local last_room_index = -1
local port_cursor = 0   -- 扫描端口段时的游标（成功后 +1，天然避开已被别的实例占用的端口）
local cur_port = nil    -- 当前连接的端口，发给 Python 便于对账
local bound_port = nil  -- 已绑定的端口：掉线后优先回到它，防止被别的实例的空闲端口抢走
local connected_at = 0  -- 连上时的帧号：用来区分"刚连上就被踢"和"真的掉线"

local AUTO_RESTART = true   -- 死亡后自动重开。必须在"刚死"的瞬间重开：
                            -- 等进了死亡结算画面，Isaac.ExecuteCommand("restart") 就无效了，
                            -- 玩家会永远卡在 game over 里（实测踩过）。
                            -- 单房间战斗训练也靠它续命，所以这里保持开启。
local AUTO_START_EVERY = 60 -- 死着就每秒重试一次重开
local was_dead = false
local last_restart_frame = -999

-- 无人值守：新开的实例停在主菜单时，自己尝试把对局开起来
local AUTO_START = true
local START_EVERY = 300        -- 每 300 帧（约 5 秒）试一次
local MAX_START_ATTEMPTS = 12  -- 最多试 12 次（约 1 分钟），不行就放弃，避免一直刷
local start_attempts = 0

local function try_connect()
    local ok, socket = pcall(require, "socket")
    if not ok then
        return false
    end

    -- 已经在用的端口优先重连：这东西守住了，"实例号↔端口"的映射才不会乱
    if bound_port ~= nil then
        local c = socket.tcp()
        if c then
            c:settimeout(0.3)
            if c:connect(HOST, bound_port) then
                sock = c
                cur_port = bound_port
                connected_at = frame
                return true
            end
            c:close()
        end
    end

    -- 首次连接 / 原端口没了：在端口段里挨个试，连上就锁定
    for _ = 1, MAX_INSTANCES do
        local port = BASE_PORT + port_cursor
        port_cursor = (port_cursor + 1) % MAX_INSTANCES
        local client = socket.tcp()
        if client then
            client:settimeout(0.3)
            local r = client:connect(HOST, port)
            if r then
                sock = client
                cur_port = port
                bound_port = port
                connected_at = frame
                return true
            end
            client:close()
        end
    end
    return false
end

local function sock_send(s)
    if not sock then
        return false
    end
    local sent = sock:send(s)
    if sent == #s then
        return true
    end
    -- 部分发送/失败：关连接，避免截断的 JSON 污染数据流（下帧重连，Python 会重新同步）
    sock:close()
    sock = nil
    cur_port = nil
    if frame - connected_at < 30 then
        -- 刚连上就被踢（半秒内）：多半是这端口已被别的实例占着，换下一个端口
        bound_port = nil
    elseif bound_port ~= nil then
        -- 正常掉线：守住自己原来那个端口
        port_cursor = (bound_port - BASE_PORT) % MAX_INSTANCES
    end
    return false
end

-- 接收 Python 下发的命令（restart / goto d.<id> / exec <控制台命令>）。
-- 这是"单房间战斗训练"的关键：reset 时直接把玩家传送进有敌人的房间，
-- 免得 AI 自己在大地图里乱走、半天遇不到敌人。
local function poll_commands()
    if sock == nil then
        return
    end
    sock:settimeout(0)
    while true do
        local line = sock:receive("*l")
        if line == nil then
            break
        end
        line = line:gsub("^%s+", ""):gsub("%s+$", "")
        if line ~= "" then
            if line == "restart" then
                pcall(function() Isaac.ExecuteCommand("restart") end)
            elseif line:sub(1, 5) == "goto " then
                pcall(function() Isaac.ExecuteCommand("goto " .. line:sub(6)) end)
            elseif line:sub(1, 5) == "exec " then
                pcall(function() Isaac.ExecuteCommand(line:sub(6)) end)
            end
        end
    end
    sock:settimeout(0.5)
end

local function build_entities(room)
    local out = {}
    local ents = room:GetEntities()
    for i = 0, ents.Size - 1 do
        local e = ents:Get(i)
        if e and e:Exists() then
            out[#out + 1] = {
                id = e.Index,
                type = e.Type,
                variant = e.Variant,
                subtype = e.SubType,
                x = e.Position.X,
                y = e.Position.Y,
                vx = e.Velocity.X,
                vy = e.Velocity.Y,
                hp = e.HitPoints,
                boss = e:IsBoss(),
                enemy = e:IsEnemy(),
                dead = e:IsDead(),
                spawner_type = e.SpawnerType or 0,
                grid_idx = room:GetGridIndex(e.Position),
                size = e.Size,               -- 碰撞体积（半径）
                height = e.Height or 0,      -- 泪弹高度（Z），非泪弹为 0
                coll = e.CollisionClass or 0,  -- 碰撞类别（EntityCollisionClass）
                cdmg = e.CollisionDamage or 0, -- 接触伤害（>0 表示站上去会掉血：火堆等）
                gcoll = e.GridCollisionClass or 0,
            }
        end
    end
    return out
end

local function build_map(level)
    local out = {}
    local rooms = level:GetRooms()
    for i = 0, rooms.Size - 1 do
        local rd = rooms:Get(i)
        out[#out + 1] = {
            grid = rd.GridIndex,
            type = rd.Data.Type,
            visited = rd.VisitedCount,
        }
    end
    return out
end

local function build_grid(room)
    -- 导出房间网格：可行走标记 + 碰撞等级 + 实体类型 + 世界坐标原点/格子尺寸
    local w = room:GetGridWidth()
    local h = room:GetGridHeight()
    local cells, coll, ent = {}, {}, {}
    local n = w * h
    for i = 0, n - 1 do
        local c = room:GetGridCollision(i)
        -- 可走 = 空地(0) 或 墙但玩家可过(5)。门就是 coll=5，
        -- 只认 0 的话 A* 永远到不了门，AI 会被锁死在房间里。
        cells[i + 1] = (c == 0 or c == 5) and 1 or 0
        coll[i + 1] = c
        ent[i + 1] = 0
    end
    -- 实体类型（单独 pcall，失败不影响走格数据）
    pcall(function()
        for i = 0, n - 1 do
            local ge = room:GetGridEntity(i)
            if ge then
                ent[i + 1] = ge:GetType()
            end
        end
    end)
    local tl = room:GetTopLeftPos()
    local br = room:GetBottomRightPos()
    return {
        w = w,
        h = h,
        cells = cells,
        coll = coll,
        ent = ent,
        ox = tl.X,
        oy = tl.Y,
        tw = (br.X - tl.X) / w,
        th = (br.Y - tl.Y) / h,
    }
end

local function build_doors(room)
    -- 导出 4 个正方向的门：格子索引 + 是否打开
    local out = {}
    for slot = 0, 3 do
        local door = room:GetDoor(slot)
        if door then
            out[#out + 1] = {
                slot = slot,
                grid = door:GetGridIndex(),
                open = door:IsOpen(),
            }
        end
    end
    return out
end

local function build_state()
    local player = Isaac.GetPlayer(0)
    if player == nil then
        return nil
    end
    local room = game:GetRoom()
    local level = game:GetLevel()

    local state = {
        frame = game:GetFrameCount(),
        player = {
            x = player.Position.X,
            y = player.Position.Y,
            vx = player.Velocity.X,
            vy = player.Velocity.Y,
            hp = player:GetHearts(),
            max_hp = player:GetMaxHearts(),
            soul_hp = player:GetSoulHearts(),
            coins = player:GetNumCoins(),
            bombs = player:GetNumBombs(),
            keys = player:GetNumKeys(),
            active_item = player:GetActiveItem(),
            active_charge = player:GetActiveCharge(),
            pocket_item = player:GetPocketItem(0),
            card = player:GetCard(0),
            damage = player.Damage,
            move_speed = player.MoveSpeed,
            tear_range = player.TearRange,
            shot_speed = player.ShotSpeed,
            fire_delay = player.FireDelay,
            size = player.Size,
            grid_idx = room:GetGridIndex(player.Position),
        },
        room = {
            type = room:GetType(),
            index = level:GetCurrentRoomIndex(),
            enemies_alive = room:GetAliveEnemiesCount(),
            bosses_alive = room:GetAliveBossesCount(),
            clear = room:IsClear(),
        },
        entities = build_entities(room),
    }

    -- 朝向 / 无敌帧：没有这些 AI 不知道自己正朝哪射（泪弹方向 = 射击键方向 + 玩家速度），
    -- 所以要靠速度反推，很难瞄得准。AIsaac 的玩家观测也带 head_dir/fire_dir。
    pcall(function()
        local pl = Isaac.GetPlayer(0)
        state.player.fire_dir = pl:GetFireDirection()   -- 0=左 1=上 2=右 3=下 -1=无
        state.player.head_dir = pl:GetHeadDirection()
        state.player.iframes = pl:GetDamageCooldown()   -- 剩余无敌帧
    end)

    -- 门（每帧都发，门开/关会变）
    local okd, doors = pcall(build_doors, room)
    if okd then
        state.doors = doors
    end

    -- 网格：房间变化时 + 每 GRID_EVERY 帧刷新（实时反映炸开石头/便便等地形变化）
    local room_index = level:GetCurrentRoomIndex()
    if room_index ~= last_room_index or frame % GRID_EVERY == 0 then
        last_room_index = room_index
        local okg, g = pcall(build_grid, room)
        if okg then
            state.grid = g
        end
    end

    if frame % MAP_EVERY == 0 then
        local ok, m = pcall(build_map, level)
        if ok then
            state.map = m
        end
    end

    return state
end

function mod:onUpdate()
    frame = frame + 1

    -- 先处理 Python 下发的命令（每帧都收，保证 reset 够快）
    poll_commands()

    -- 死亡自动重开：死了就反复重试（一进结算画面 restart 就失效，所以要抢在刚死时）
    if AUTO_RESTART then
        local pl = Isaac.GetPlayer(0)
        if pl ~= nil and pl:IsDead() then
            if frame - last_restart_frame > AUTO_START_EVERY then
                last_restart_frame = frame
                pcall(function() Isaac.ExecuteCommand("restart") end)
            end
            was_dead = true
        else
            was_dead = false
        end
    end

    if frame % SEND_EVERY ~= 0 then
        return
    end

    if sock == nil then
        try_connect()
        if sock == nil then
            return
        end
        -- 连上后立刻发 hello，验证数据通路（带上端口，方便多开时对账）
        sock_send(json.encode({ hello = true, frame = frame, port = cur_port }) .. "\n")
    end

    local ok, result = pcall(build_state)
    if not ok then
        sock_send('{"error":' .. json.encode(tostring(result)) .. '}\n')
        return
    end
    if result == nil then
        -- 不在对局中（菜单）：无人值守训练时必须自己把对局开起来
        if AUTO_START and start_attempts < MAX_START_ATTEMPTS
                and frame % START_EVERY == 0 then
            start_attempts = start_attempts + 1
            pcall(function() Isaac.ExecuteCommand("restart") end)
        end
        -- 每 30 帧发一次 menu 标记确认还活着（带上尝试次数，方便排查自动开局是否奏效）
        if frame % (MAP_EVERY * 2) == 0 then
            sock_send(json.encode({
                menu = true, frame = frame, start_attempts = start_attempts,
            }) .. "\n")
        end
        return
    end

    sock_send(json.encode(result) .. "\n")
end

mod:AddCallback(ModCallbacks.MC_POST_UPDATE, mod.onUpdate)
