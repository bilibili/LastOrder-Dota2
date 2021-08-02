-------------------------------------------------------------------------------
--- AUTHOR: Nostrademous
-------------------------------------------------------------------------------

local pprint                    = require( "bots/pprint" )

-- Atomic Actions
local actionGlyph               = require( "bots/actions/atomic_glyph" )
local actionLevelAbility        = require( "bots/actions/atomic_level_ability" )

-- Hero Functions
local actionNone                = require( "bots/actions/none" )
local actionClearAction         = require( "bots/actions/clear" )
local actionBuyback             = require( "bots/actions/buyback" )
local actionChat                = require( "bots/actions/chat" )

--local actionMoveToLocation      = require( "bots/actions/move_to_location" )
local actionMoveToLocation      = require( "bots/actions/move_to_location" )
local actionMoveDirectly        = require( "bots/actions/move_directly" )
local actionAttackUnit          = require( "bots/actions/attack_unit" )

local actionUseAbilityOnEntity  = require( "bots/actions/use_ability_on_entity" )
local actionUseAbilityOnLocation= require( "bots/actions/use_ability_on_location" )
local actionUseAbilityOnTree    = require( "bots/actions/use_ability_on_tree" )
local actionUseAbility          = require( "bots/actions/use_ability" )
local actionToggleAbility       = require( "bots/actions/toggle_ability" )
local actionUseCourier          = require( "bots/actions/use_courier" )
local actionPurchaseItem        = require( "bots/actions/purchase_item" )
local actionPickUpRune          = require( "bots/actions/pickup_rune" )
local actionPickUpItem          = require( "bots/actions/pickup_item" )
local actionDropItem            = require( "bots/actions/drop_item" )
local actionSwapItems           = require( "bots/actions/swap_item" )

-- Global Variables
ABILITY_STANDARD = 0
ABILITY_PUSH     = 1
ABILITY_QUEUE    = 2

local ActionProcessor = {}

LookUpTable = {
    ['DOTA_UNIT_ORDER_NONE'] = actionNone,
    ['DOTA_UNIT_ORDER_MOVE_TO_POSITION'] = actionMoveToLocation,
    ['DOTA_UNIT_ORDER_MOVE_DIRECTLY'] = actionMoveDirectly,
    ['DOTA_UNIT_ORDER_ATTACK_TARGET'] = actionAttackUnit,
    ['DOTA_UNIT_ORDER_GLYPH'] = actionGlyph,
    ['DOTA_UNIT_ORDER_STOP'] = actionClearAction,
    ['DOTA_UNIT_ORDER_TRAIN_ABILITY'] = actionLevelAbility,
    ['DOTA_UNIT_ORDER_BUYBACK'] = actionBuyback,
    ['ACTION_CHAT'] = actionChat,
    ['DOTA_UNIT_ORDER_CAST_POSITION'] = actionUseAbilityOnLocation,
    ['DOTA_UNIT_ORDER_CAST_TARGET'] = actionUseAbilityOnEntity,
    ['DOTA_UNIT_ORDER_CAST_TARGET_TREE'] = actionUseAbilityOnTree,
    ['DOTA_UNIT_ORDER_CAST_NO_TARGET'] = actionUseAbility,
    ['DOTA_UNIT_ORDER_CAST_TOGGLE'] = actionToggleAbility,
    ['DOTA_UNIT_ORDER_PURCHASE_ITEM'] = actionPurchaseItem,
    ['ACTION_COURIER'] = actionUseCourier,
    ['ACTION_SWAP_ITEMS'] = actionSwapItems,
    ['DOTA_UNIT_ORDER_PICKUP_RUNE'] = actionPickUpRune,
    ['DOTA_UNIT_ORDER_PICKUP_ITEM'] = actionPickUpItem,
    ['DOTA_UNIT_ORDER_DROP_ITEM'] = actionDropItem,
}

local function table_length(tbl)
    local lenNum = 0
    for k,v in pairs(tbl) do
        lenNum = lenNum + 1
    end
    return lenNum
end

function ActionProcessor:Run(hBot, tb)
    if table_length(tb) == 0 then
        print("Got EMPTY tblActions")
        -- If we have no action, draw a big red circle around the hero.
        DebugDrawCircle(hBot:GetLocation(), 200, 255, 0, 0)
        do return end
    end

    -- print('Actions processor: ', pprint.pformat(tblActions))

    for key, value in pairs(tb) do
        local cmd = LookUpTable[key]
        if cmd ~= nil then
            -- print("Executing Action: ", cmd.Name)
            -- NOTE: It is assumed that the first arguement (if args required)
            --       will be the handle to the bot, followed by arguments
            --       provided by the `agent`.
            --       `Agent` args are double nested lists [[]], as some args
            --       specify a location.
            --       Example: [[-7000,-7000,128], [2]]
            --                to do a move_to_location (X,Y,Z) in Queued fashion
            if cmd.NumArgs == 2 then
                cmd:Call(hBot, value[1])
            elseif cmd.NumArgs == 3 then
                cmd:Call(hBot, value[1], value[2])
            elseif cmd.NumArgs == 0 then
                cmd:Call()
            elseif cmd.NumArgs == 1 then
                cmd:Call(hBot)
            elseif cmd.NumArgs == 4 then
                cmd:Call(hBot, value[1], value[2], value[3])
            else
                print("Unimplemented number of Cmd Args for ", cmd.Name, ": ", cmd.NumArgs)
                -- DebugPause()
                do return end
            end
        else
            print("<ERROR> [", key, "] does not exist in action table!")
        end
    end
end

return ActionProcessor
