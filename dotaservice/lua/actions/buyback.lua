-------------------------------------------------------------------------------
--- AUTHOR: Nostrademous
-------------------------------------------------------------------------------

local ActionBuyback = {}

ActionBuyback.Name = "Buyback Action"
ActionBuyback.NumArgs = 1

-------------------------------------------------

function ActionBuyback:Call( hHero )
    if not hHero:IsAlive() then
        hHero:ActionImmediate_Buyback()
    end
end

-------------------------------------------------

return ActionBuyback
