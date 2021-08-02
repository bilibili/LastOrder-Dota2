-------------------------------------------------------------------------------
--- AUTHOR: Nostrademous
-------------------------------------------------------------------------------

local Chat = {}

Chat.Name = "Chat"
Chat.NumArgs = 3

-------------------------------------------------

function Chat:Call( hHero, sMsg, bAllChat )
    hHero:ActionImmediate_Chat(sMsg[1], bAllChat[1])
end

-------------------------------------------------

return Chat
