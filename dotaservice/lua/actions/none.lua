-------------------------------------------------------------------------------
--- AUTHOR: Nostrademous
-------------------------------------------------------------------------------

local ActionNone = {}

ActionNone.Name = "No Action"
ActionNone.NumArgs = 0

-------------------------------------------------

function ActionNone:Call()
    DebugDrawCircle(GetBot():GetLocation(), 50, 127, 127, 127)
end

-------------------------------------------------

return ActionNone
