-------------------------------------------------------------------------------
--- AUTHOR: Nostrademous
-------------------------------------------------------------------------------
local UseAbility = {}

UseAbility.Name = "Use Ability"
UseAbility.NumArgs = 3

-------------------------------------------------
function UseAbility:Call( hUnit, intAbilitySlot, iType )
    local hAbility
    if intAbilitySlot[1] >= 0 then
        hAbility = hUnit:GetAbilityInSlot(intAbilitySlot[1])
    else
        local itemSlot = -intAbilitySlot[1] - 1
        hAbility = hUnit:GetItemInSlot(itemSlot)

        if hAbility:GetName() == 'item_tango' then
            local tree = hUnit:GetNearbyTrees(700)[1]
            if tree ~= nil then
                hUnit:Action_UseAbilityOnTree(hAbility, tree)
            else
                print("UseAbility can not find tree")
            end
            do return end
        end
    end

    if not hAbility then
        print('[ERROR]: ', hUnit:GetUnitName(), " failed to find ability in slot ", intAbilitySlot[1])
        do return end
    end

    iType = iType[1]

    -- Note: we do not test for range, mana/cooldowns or any debuffs on the hUnit (e.g., silenced).

    if iType == nil or iType == ABILITY_STANDARD then
        hUnit:Action_UseAbility(hAbility)
    elseif iType == ABILITY_PUSH then
        hUnit:ActionPush_UseAbility(hAbility)
    elseif iType == ABILITY_QUEUE then
        hUnit:ActionQueue_UseAbility(hAbility)
    end
end
-------------------------------------------------

return UseAbility
