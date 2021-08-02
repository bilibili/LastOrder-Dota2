-------------------------------------------------------------------------------
--- AUTHOR: Nostrademous
-------------------------------------------------------------------------------

local ToggleAbility = {}

ToggleAbility.Name = "Toggle Ability"
ToggleAbility.NumArgs = 2

-------------------------------------------------
function ToggleAbility:Call( hUnit, intAbilitySlot )
    local hAbility = hUnit:GetAbilityInSlot(intAbilitySlot[1])
    if not hAbility then
        print('[ERROR]: ', hUnit:GetUnitName(), " failed to find ability in slot ", intAbilitySlot[1])
        do return end
    end
    hAbility:ToggleAutoCast()
end
-------------------------------------------------

return ToggleAbility
