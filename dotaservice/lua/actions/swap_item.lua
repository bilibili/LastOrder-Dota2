local SwapItem = {}
SwapItem.Name = "purchase item"
SwapItem.NumArgs = 3

function SwapItem:Call( hUnit, slotA, slotB )
    hUnit:ActionImmediate_SwapItems(slotA[1], slotB[1])
end

return SwapItem
