local config = require('bots/config')
local dkjson = require('game/dkjson')
local pprint = require('bots/pprint')
local action_proc = require('bots/action_processor')


local ACTION_FILENAME = 'bots/actions_t' .. GetTeam()
local LIVE_CONFIG_FILENAME = 'bots/live_config_auto'

local debug_text = nil
local live_config = nil


local function act(action)
    local tblActions = {}
    if action.actionType == "DOTA_UNIT_ORDER_NONE" then
        tblActions[action.actionType] = {}
    elseif action.actionType == "DOTA_UNIT_ORDER_MOVE_TO_POSITION" then
        -- NOTE: Move To Position is implemented by Dota2 as a path-navigation movement action.
        --       It will create a list of waypoints that the bot will walk in straight lines between.
        --       The waypoints the system creates will guarantee a valid path between current location
        --       and destination location (PROVIDING A VALID PATH EXISTS).
        --       It approximates reaching each "waypoint" (including last one) before moving to the next
        --       waypoint (if it exists) with a granularity tested to be 50 units. So Move To Location is
        --       not a VERY precise movement action, but it's not hugely imprecise either. It is an
        --       important note though, as if you don't check if your position is within the precision
        --       approximiation for movement you could end up instructing the bot to move to the same
        --       location over and over and it just ping-pongs back and forth moving around the precise
        --       location but never directly on it.
        tblActions[action.actionType] = {{action.moveToLocation.location.x, action.moveToLocation.location.y, 0.0}, {0}}
    elseif action.actionType == "DOTA_UNIT_ORDER_MOVE_DIRECTLY" then
         -- NOTE: Move Direclty is implemented by Dota2 as a single point-to-point straight line
        --       movement action. It does not try to path around any obstacles or check for impossible moves.
        --       It has high precision in final position (gut belief is a 1-2 unit approximation).
        --       Because of how it works, it is ill-advised to use Direct movement for long distances as the
        --       probability of hitting a tree or an obstacle are high and with Direct movement you will
        --       not path around it, but rather get stuck trying to move through it and not succeeding.
        tblActions[action.actionType] = {{action.moveDirectly.location.x, action.moveDirectly.location.y, 0.0}, {0}}
    elseif action.actionType == "DOTA_UNIT_ORDER_ATTACK_TARGET" then
        tblActions[action.actionType] = {{action.attackTarget.target}, {action.attackTarget.once}, {0}}
    elseif action.actionType == "DOTA_UNIT_ORDER_TRAIN_ABILITY" then
        tblActions[action.actionType] = {{action.trainAbility.ability}}
    elseif action.actionType == "DOTA_UNIT_ORDER_GLYPH" then
        tblActions[action.actionType] = {}
    elseif action.actionType == "DOTA_UNIT_ORDER_STOP" then
        tblActions[action.actionType] = {{1}}
    elseif action.actionType == "DOTA_UNIT_ORDER_BUYBACK" then
        tblActions[action.actionType] = {}
    elseif action.actionType == "ACTION_CHAT" then
        tblActions[action.actionType] = {{action.chat.message}, {action.chat.toAllchat}}
    elseif action.actionType == "DOTA_UNIT_ORDER_CAST_POSITION" then
        tblActions[action.actionType] = {{action.castLocation.abilitySlot}, {action.castLocation.location.x, action.castLocation.location.y, action.castLocation.location.z}, {0}}
    elseif action.actionType == "DOTA_UNIT_ORDER_CAST_TARGET" then
        tblActions[action.actionType] = {{action.castTarget.abilitySlot}, {action.castTarget.target}, {0}}
    elseif action.actionType == "DOTA_UNIT_ORDER_CAST_TARGET_TREE" then
        tblActions[action.actionType] = {{action.castTree.abilitySlot}, {action.castTree.tree}, {0}}
    elseif action.actionType == "DOTA_UNIT_ORDER_CAST_NO_TARGET" then
        tblActions[action.actionType] = {{action.cast.abilitySlot}, {0}}
    elseif action.actionType == "DOTA_UNIT_ORDER_CAST_TOGGLE" then
        tblActions[action.actionType] = {{action.castToggle.abilitySlot}}
    elseif action.actionType == "DOTA_UNIT_ORDER_PURCHASE_ITEM" then
        tblActions[action.actionType] = {{action.purchaseItem.itemName}}
    elseif action.actionType == "ACTION_COURIER" then
        tblActions[action.actionType] = {{action.courier.action}}
    elseif action.actionType == "DOTA_UNIT_ORDER_PICKUP_RUNE" then
        tblActions[action.actionType] = {{action.pickUpRune.rune}, {0}}
    elseif action.actionType == "DOTA_UNIT_ORDER_PICKUP_ITEM" then
        tblActions[action.actionType] = {{action.pickUpItem.itemId}, {0}}
    elseif action.actionType == "DOTA_UNIT_ORDER_DROP_ITEM" then
        tblActions[action.actionType] = {{action.dropItem.slot}, {action.dropItem.location.x, action.dropItem.location.y, 0, 0}, {0}}
    elseif action.actionType == "ACTION_SWAP_ITEMS" then
        tblActions[action.actionType] = {{action.swapItems.slotA}, {action.swapItems.slotB}}
    end
    action_proc:Run(GetBot(), tblActions)
end


local function get_new_action(dota_time, player_id, latest_action_time, step)
    -- print('(lua) get_new_action @', dota_time, ' player_id=', player_id)
    --print('ACTION_FILENAME=', ACTION_FILENAME)
    local file_fn = nil
    -- Try to load the file first
    file_fn = loadfile(ACTION_FILENAME)
    if file_fn == nil then return nil end

    -- Execute the file_fn; this loads contents into `data`.
    local data = file_fn()
    if data == nil then return nil end

    if data == 'FLUSH' then
        return nil
    end
    local data, pos, err = dkjson.decode(data, 1, nil)
    if err then
        print("(lua) JSON Decode Error=", err, " at pos=", pos)
        return nil
    end

    -- if we has executed this action, skip that
    if latest_action_time ~= nil and math.abs(data.dotaTime - latest_action_time) <= 0.0001 then
        --print("already executed action")
        return nil
    end

    if DotaTime() - data.dotaTime < 0.07 then
        return nil
    end
    -- print("dota_time_diff: ", DotaTime() - data.dotaTime, " team ", GetTeam())


    -- such as chat, train ability
    if data.extra_actions ~= nil and data.extra_actions.actions ~= nil  then
        for _, action in pairs(data.extra_actions.actions) do
            if action.player == player_id then
                act(action)
            end
        end
    end

    -- debug draw pic
    if data.draw ~= nil then
        for _, pic in pairs(data.draw) do
            if pic.type == "text" then
                DebugDrawText(pic.x, pic.y, debug_text, pic.r, pic.g, pic.b)
            elseif pic.type == "line" then
                startLoc = Vector(pic.start_x, pic.start_y, pic.start_z)
                endLoc = Vector(pic.end_x, pic.end_y, pic.end_z)
                DebugDrawLine(startLoc, endLoc, pic.r, pic.g, pic.b)
            elseif pic.type == "circle" then
                vLoc = Vector(pic.x, pic.y, pic.z)
                DebugDrawCircle(vLoc, pic.radius, pic.r, pic.g, pic.b)
            end
        end
    end

    for _, action in pairs(data.actions) do
        --print('step data:',data,data.extraData,action.actionType)
        -- execute only one action at one time
        if action.player == player_id then
            print("sync key:", data.extraData, RealTime(), "###", DotaTime(), "###", step)
            return action, data.dotaTime
        end
    end
end


-- This table keeps track of which time corresponds to which fn_call.
local dota_time_to_step_map = {}
local worldstate_step_offset = nil
local step = 0

local action = nil
local act_at_step = nil

local latest_action_time = nil

function Think()
    step = step + 1
    action = nil

    -- if GetBot():GetPlayerID() ~= 0 and GetBot():GetPlayerID() ~= 2 and GetBot():GetPlayerID() ~= 5 and GetBot():GetPlayerID() ~= 6 then
    --     do return end
    -- end

    local dota_time = DotaTime()
    local game_state = GetGameState()

    -- tell the dotaservice dota2 game has started
    if step == 10 then
        local status = {}
        status.dota_time = dota_time
        status.step = step
        print('LUARDY', json.encode(status))
    end

    action, new_time = get_new_action(dota_time, GetBot():GetPlayerID(), latest_action_time, step)

    if action == nil then
    do return end
    end

    if latest_action_time == nil then
        diff_time = 0
    else
        diff_time = math.abs(new_time - latest_action_time)
    end

    --print('new time', latest_action_time, ' time diff', diff_time)
    latest_action_time = new_time
    print('each step:',step,action.actionType)
    --print(GetBot():GetDifficulty(), DIFFICULTY_HARD)
    --print('(lua) received action =', dump(action))

    debug_text = pprint.pformat(action)

    act(action)
    --ok, err = os.remove(ACTION_FILENAME)
    --print("remove", ok)

    -- if debug_text ~= nil then
    --     DebugDrawText(8, 90, debug_text, 255, 255, 255)
    -- end
end
