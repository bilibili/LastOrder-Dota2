-------------------------------------------------------------------------------
--- AUTHOR: Nostrademous
-------------------------------------------------------------------------------

local ActionGlyph = {}

ActionGlyph.Name = "Glyph Action"
ActionGlyph.NumArgs = 1

-------------------------------------------------

function ActionGlyph:Call(hHero)
    hHero:ActionImmediate_Glyph()
end

-------------------------------------------------

return ActionGlyph
