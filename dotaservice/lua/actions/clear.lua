-------------------------------------------------------------------------------
--- AUTHOR: Nostrademous
-------------------------------------------------------------------------------

local ActionClear = {}

ActionClear.Name = "Clear Action"
ActionClear.NumArgs = 2

local function toboolean(number)
    if number >= 1 then return true end
    return false
end

-------------------------------------------------

function ActionClear:Call( hHero, bStop )
    hHero:Action_ClearActions(toboolean(bStop[1]))
end

-------------------------------------------------

return ActionClear
