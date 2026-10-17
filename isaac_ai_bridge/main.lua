-- isaac_ai_bridge/main.lua
-- 《以撒的结合：忏悔》-> Python 状态桥接模组
-- 通过 socket 把每帧状态发给本机 Python（需要 --luadebug）。
-- 连上后先发 hello；出错时把错误也通过 socket 发回来，方便排查。

local mod = RegisterMod("Isaac AI Bridge", 1)
local json = require("json")

local HOST = "127.0.0.1"
local PORT = 48532
local SEND_EVERY = 2  -- 每 2 帧发一次（RL 决策约 30Hz，够了，减少带宽）
local MAP_EVERY = 30
local GRID_EVERY = 15  -- 每 15 帧刷新一次地形（实时反映炸开石头/便便后的变化）

local sock = nil
local frame = 0
local game = Game()
local last_room_index = -1

local AUTO_RESTART = true  -- 死亡后自动重开一局（RL 训练用）
local was_dead = false

local function try_connect()
    local ok, socket = pcall(require, "socket")
    if not ok then
        return false
    end
    local client = socket.tcp()
    if not client then
        return false
    end
    client:settimeout(0.5)
    local r = client:connect(HOST, PORT)
    if r then
        sock = client
        return true
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
    return false
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
                size = e.Size,            -- 碰撞体积（半径）
                height = e.Height or 0,   -- 泪弹高度（Z），非泪弹为 0
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
        cells[i + 1] = (c == 0) and 1 or 0
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

    -- 死亡自动重开（RL 训练：让对局无限继续下去）
    if AUTO_RESTART then
        local pl = Isaac.GetPlayer(0)
        local dead = (pl ~= nil and pl:IsDead())
        if dead and not was_dead then
            Isaac.ExecuteCommand("restart")
        end
        was_dead = dead
    end

    if frame % SEND_EVERY ~= 0 then
        return
    end

    if sock == nil then
        try_connect()
        if sock == nil then
            return
        end
        -- 连上后立刻发 hello，验证数据通路
        sock_send('{"hello":true,"frame":' .. frame .. '}\n')
    end

    local ok, result = pcall(build_state)
    if not ok then
        sock_send('{"error":' .. json.encode(tostring(result)) .. '}\n')
        return
    end
    if result == nil then
        -- 不在对局中（菜单），每 30 帧发一次 menu 标记确认还活着
        if frame % (MAP_EVERY * 2) == 0 then
            sock_send('{"menu":true,"frame":' .. frame .. '}\n')
        end
        return
    end

    sock_send(json.encode(result) .. "\n")
end

mod:AddCallback(ModCallbacks.MC_POST_UPDATE, mod.onUpdate)
