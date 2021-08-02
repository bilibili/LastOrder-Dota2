-------------------------------------------------------------------------------
--- AUTHOR: Nostrademous
-------------------------------------------------------------------------------

local AttackUnit = {}

AttackUnit.Name = "Attack Unit"
AttackUnit.NumArgs = 4

-------------------------------------------------
function AttackUnit:Call( hUnit, hTarget, bOnce, iType )
    hTarget = hTarget[1]
    if hTarget == -1 then -- Invalid target. Do nothing.
        do return end
    end
    --print(hTarget)
    hTarget = GetBotByHandle(hTarget)

    iType = iType[1]
 
    -- Note: we do not test if target can be attacked due 
    -- to modifiers (e.g., invulnerable), range, or any
    -- debuffs on the hUnit (e.g., disarmed). We assume
    -- only valid and legal actions are agent selected
    if not hTarget:IsNull() and hTarget:IsAlive() then
        vLoc = hTarget:GetLocation()
        DebugDrawCircle(vLoc, 25, 255, 0, 0)
        DebugDrawLine(hUnit:GetLocation(), vLoc, 255, 0, 0)
        
        bOnce = bOnce[1]
        if iType == nil or iType == ABILITY_STANDARD then
            hUnit:Action_AttackUnit(hTarget, bOnce)
        elseif iType == ABILITY_PUSH then
            hUnit:ActionPush_AttackUnit(hTarget, bOnce)
        elseif iType == ABILITY_QUEUE then
            hUnit:ActionQueue_AttackUnit(hTarget, bOnce)
        end
    end
end
-------------------------------------------------

return AttackUnit
