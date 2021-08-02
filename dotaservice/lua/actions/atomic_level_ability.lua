-------------------------------------------------------------------------------
--- AUTHOR: Nostrademous
-------------------------------------------------------------------------------

local LevelAbility = {}

LevelAbility.Name = "Level Ability"
LevelAbility.NumArgs = 2

-------------------------------------------------

function LevelAbility:Call( hHero, sAbilityName )
    print("Leveling: ", sAbilityName[1])
    -- Sanity Check
    local nAbilityPoints = hHero:GetAbilityPoints()
    if nAbilityPoints > 0 then
        -- Another sanity check
        local hAbility = hHero:GetAbilityByName(sAbilityName[1])
        if hAbility and hAbility:CanAbilityBeUpgraded() then
            -- actually do the leveling
            hHero:ActionImmediate_LevelAbility(sAbilityName[1])
            -- print("EVENT LevelAbility Successful ", hHero:GetPlayerID(), sAbilityName[1])
        else
            print("Trying to level an ability I cannot", sAbilityName[1])
            do return end
        end
    end
end

-------------------------------------------------

return LevelAbility
