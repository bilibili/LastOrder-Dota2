-------------------------------------------------------------------------------
--- AUTHOR: Nostrademous
-------------------------------------------------------------------------------

local UseAbilityOnLocation = {}

UseAbilityOnLocation.Name = "Use Ability On Location"
UseAbilityOnLocation.NumArgs = 4

-------------------------------------------------
function UseAbilityOnLocation:Call( hUnit, intAbilitySlot, vLoc, iType )
    local hAbility

    if intAbilitySlot[1] >= 0 then
        hAbility = hUnit:GetAbilityInSlot(intAbilitySlot[1])
    else
        itemSlot = -intAbilitySlot[1] - 1
        hAbility = hUnit:GetItemInSlot(itemSlot)
    end

    if not hAbility then
        print('[ERROR]: ', hUnit:GetUnitName(), " failed to find ability in slot ", intAbilitySlot[1])
        do return end
    end

    vLoc = Vector(vLoc[1], vLoc[2], vLoc[3])

    iType = iType[1]

    -- Note: we do not test if the location can be ability-targeted due
    -- range, mana/cooldowns or any debuffs on the hUnit (e.g., silenced).
    -- We assume only valid and legal actions are agent selected

    DebugDrawCircle(vLoc, 25, 255, 0, 0)
    DebugDrawLine(hUnit:GetLocation(), vLoc, 255, 0, 0)

    if iType == nil or iType == ABILITY_STANDARD then
        hUnit:Action_UseAbilityOnLocation(hAbility, vLoc)
    elseif iType == ABILITY_PUSH then
        hUnit:ActionPush_UseAbilityOnLocation(hAbility, vLoc)
    elseif iType == ABILITY_QUEUE then
        hUnit:ActionQueue_UseAbilityOnLocation(hAbility, vLoc)
    end
end
-------------------------------------------------

return UseAbilityOnLocation
