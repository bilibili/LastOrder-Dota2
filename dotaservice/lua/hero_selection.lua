--local config = require("bots/config")
--
--function TeamToChar()
--    -- TEAM_RADIANT==2 TEAM_DIRE==3. Makes total sense!
--    if GetTeam() == TEAM_RADIANT then return 'R' else return 'D' end
--end
--
--function GetBotNames ()
--    truncated_team_id = string.sub(config.game_id, 1, 8) .. "_" .. TeamToChar()
--	return  {
--        truncated_team_id .. "0",
--        truncated_team_id .. "1",
--        truncated_team_id .. "2",
--        truncated_team_id .. "3",
--        truncated_team_id .. "4",
--    }
--end
--
--local bot_heroes = {"npc_dota_hero_nevermore"}
--
--function Think()
--	-- This gets called (every server tick AND until all heroes are picked).
--	-- This needs to gets called at least once if there is no human.
--    local ids = GetTeamPlayers(GetTeam())
--    for i,v in pairs(ids) do
--        -- If the human is in the unassigned slot, the radiant bots start at v = 2
--        -- If the human is in the radiant coach slot, the radiant bots start at v = 2
--        -- If the human is in the first radiant slot, the radiant bots start at v = 0
--        -- If the human is in the second radiant slot, the radiant bots start at v = 1
--        -- If the human is in the third radiant slot, the radiant bots start at v = 2
--		if IsPlayerBot(v) and IsPlayerInHeroSelectionControl(v) then
--        -- if i == 1 and GetTeam() == TEAM_RADIANT then
--            if i == 1 then
--                SelectHero( v, "npc_dota_hero_nevermore" );
--            else
--                SelectHero( v, "npc_dota_hero_sniper" );
--            end
--		end
--	end
--end

local config = require("bots/config")

function TeamToChar()
    -- TEAM_RADIANT==2 TEAM_DIRE==3. Makes total sense!
    if GetTeam() == TEAM_RADIANT then return 'R' else return 'D' end
end

function GetBotNames ()
    -- truncated_team_id = string.sub(config.game_id, 1, 8) .. "_" .. TeamToChar()
    truncated_team_id = "LastOrder" .. "_" .. TeamToChar()
	return  {
        truncated_team_id .. "0",
        truncated_team_id .. "1",
        truncated_team_id .. "2",
        truncated_team_id .. "3",
        truncated_team_id .. "4",
    }
end

local bot_heroes = {"npc_dota_hero_nevermore"}

function Think()
	-- This gets called (every server tick AND until all heroes are picked).
	-- This needs to gets called at least once if there is no human.
    local ids = GetTeamPlayers(GetTeam())
    for i,v in pairs(ids) do
        -- If the human is in the unassigned slot, the radiant bots start at v = 2
        -- If the human is in the radiant coach slot, the radiant bots start at v = 2
        -- If the human is in the first radiant slot, the radiant bots start at v = 0
        -- If the human is in the second radiant slot, the radiant bots start at v = 1
        -- If the human is in the third radiant slot, the radiant bots start at v = 2
		if IsPlayerBot(v) and IsPlayerInHeroSelectionControl(v) then
        -- if i == 1 and GetTeam() == TEAM_RADIANT then
            if i == 1 then
                if GetTeam() == TEAM_RADIANT then
                    SelectHero( v, "npc_dota_hero_nevermore" );
                elseif GetTeam() == TEAM_DIRE then
                    SelectHero( v, "npc_dota_hero_nevermore" );
                end
            else
                SelectHero( v, "npc_dota_hero_wisp" );
            end
		end
	end
end


-- Function below sets the lane assignments for default bots
-- Obviously, our own agents will do what they belive best
function UpdateLaneAssignments()
    if GetTeam() == TEAM_RADIANT then
        return {
            [1] = LANE_MID,
            [2] = LANE_BOT,
            [3] = LANE_BOT,
            [4] = LANE_TOP,
            [5] = LANE_TOP,
        }
    elseif GetTeam() == TEAM_DIRE then
        return {
            [1] = LANE_MID,
            [2] = LANE_BOT,
            [3] = LANE_BOT,
            [4] = LANE_TOP,
            [5] = LANE_TOP,
        }
    end
end

