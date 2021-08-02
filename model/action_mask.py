import numpy as np
from dotaservice.protos.dota_gcmessages_common_bot_script_pb2 import CMsgBotWorldState
from dotaservice.dotautil import cal_distance_with_z
from gym_env.feature_processors.enums import *

CREEP_TYPE = CMsgBotWorldState.UnitType.Value("LANE_CREEP")
HERO_TYPE = CMsgBotWorldState.UnitType.Value("HERO")
TOWER_TYPE = CMsgBotWorldState.UnitType.Value("TOWER")


class ActionMask:
    ability_mask_index = {
        5059: ACTION_NAME_TO_INDEX["ABILITY_Q"],
        5060: ACTION_NAME_TO_INDEX["ABILITY_W"],
        5061: ACTION_NAME_TO_INDEX["ABILITY_E"],
        5064: ACTION_NAME_TO_INDEX["ABILITY_R"]
    }
    ability_mana = {
        5059: {
            1: 75,
            2: 80,
            3: 85,
            4: 90
        },
        5060: {
            1: 75,
            2: 80,
            3: 85,
            4: 90
        },
        5061: {
            1: 75,
            2: 80,
            3: 85,
            4: 90
        },
        5064: {
            1: 150,
            2: 175,
            3: 200
        }
    }

    def __init__(self, team_id, player_id, enemy_player_id):
        self.team_id = team_id
        self.player_id = player_id
        self.enemy_player_id = enemy_player_id

        if self.team_id == 3:
            self.enemy_team_id = 2
            self.enemy_tower_name = 'npc_dota_goodguys_tower1_mid'
            self.self_tower_name = 'npc_dota_badguys_tower1_mid'
        else:
            self.enemy_team_id = 3
            self.enemy_tower_name = 'npc_dota_badguys_tower1_mid'
            self.self_tower_name = 'npc_dota_goodguys_tower1_mid'

        # actions from model/creep_block_model.py
        self.actions = All_ACTION_TYPE
        self.actions_len = len(self.actions)

    def check_enemy_tower_valid(self, self_hero, enemy_tower):
        d, _ = cal_distance_with_z(self_hero.location, enemy_tower.location)
        # sf's night vision 800, attack range 500
        if d > 0 and d < 750:
            return 1
        else:
            return 0

    def unit_attack_valid(self, hero, u):
        if u.is_alive is True and u.HasField('handle'):
            if u.team_id != self.team_id and u.unit_type in [CREEP_TYPE, HERO_TYPE]:
                # 正补
                return 1
            elif u.name == self.enemy_tower_name and self.check_enemy_tower_valid(hero, u):
                return 1
            elif u.team_id == self.team_id and \
                    u.unit_type == CREEP_TYPE and u.health * 2 <= u.health_max:
                # 反补
                return 1
        return 0

    def valid(self, hero_unit, units, cur_dota_time, end_issue_attack_time, next_can_attack_time, end_casting_ability_time,
              dota_map, tango_cd, cur_casting_ability_id, swap_item_cd, pre_action_id, cur_attack_handle):

        mask = np.ones(self.actions_len, dtype=np.int)

        if cur_dota_time < end_casting_ability_time:
            mask[0] = 0

        # 技能 mask
        for ability in hero_unit.abilities:
            ability_id = ability.ability_id
            if ability_id in self.ability_mask_index.keys() and \
                (ability.level == 0 or ability.cooldown_remaining > 0 or
                 hero_unit.mana < self.ability_mana[ability_id][ability.level]
                or (cur_dota_time < end_casting_ability_time and cur_casting_ability_id != ability_id)):
                mask[self.ability_mask_index[ability_id]] = 0

        has_deny_target = False
        has_enemy_target = False
        has_hero_target = False
        has_tower_target = False
        has_flask = False
        has_clarity = False
        has_mango = False
        has_plant_tree = False
        has_tango = False
        # has_nearby_tree = False
        has_magic_stick = False
        has_fire = False
        has_ward_target = False

        has_ward = False
        has_sentry_ward = False

        is_in_attack_interval = False

        for u in units:
            if u.is_alive:
                if u.unit_type == CREEP_TYPE and u.health * 2 < u.health_max and u.team_id == self.team_id:
                    if cur_attack_handle != -1:
                        if u.handle == cur_attack_handle:
                            has_deny_target = True
                    else:
                        has_deny_target = True
                if u.unit_type == CREEP_TYPE and u.team_id != self.team_id:
                    if cur_attack_handle != -1:
                        if u.handle == cur_attack_handle:
                            has_enemy_target = True
                    else:
                        has_enemy_target = True

                if u.unit_type == HERO_TYPE and u.team_id != self.team_id:
                    has_hero_target = True
                if u.name == self.enemy_tower_name and self.check_enemy_tower_valid(hero_unit, u):
                    has_tower_target = True
                if u.unit_type == CMsgBotWorldState.UnitType.Value("WARD") and u.team_id != self.team_id:
                    has_ward_target = True

        if len(hero_unit.items) > 0:
            for i in hero_unit.items:
                if i.slot <= 5:
                    if i.ability_id == ITEM_FLASK_ID and i.cooldown_remaining == 0 and swap_item_cd[i.slot] == 0:
                        has_flask = True
                    if i.ability_id == ITEM_MANGO_ID and i.cooldown_remaining == 0 and swap_item_cd[i.slot] == 0:
                        has_mango = True
                    if i.ability_id == ITEM_BRANCH_ID and i.cooldown_remaining == 0 and swap_item_cd[i.slot] == 0:
                        has_plant_tree = True
                    if i.ability_id == ITEM_TANGO_ID and i.cooldown_remaining == 0 and swap_item_cd[i.slot] == 0:
                        has_tango = True
                    if i.ability_id == ITEM_CLARITY_ID and i.cooldown_remaining == 0 and swap_item_cd[i.slot] == 0:
                        has_clarity = True
                    if i.ability_id in [ITEM_MAGIC_STICK_ID, ITEM_MAGIC_WAND_ID] and i.cooldown_remaining == 0 \
                            and i.charges > 0 and swap_item_cd[i.slot] == 0:
                        has_magic_stick = True
                    if i.ability_id == ITEM_FAERIE_FIRE_ID and i.cooldown_remaining == 0 and swap_item_cd[i.slot] == 0:
                        has_fire = True

                    if (i.ability_id == ITEM_WARD_ID or i.ability_id == ITEM_WARD_DISPENSER_ID) \
                            and i.cooldown_remaining == 0 and swap_item_cd[i.slot] == 0:
                        has_ward = True
                    if i.ability_id == ITEM_WARD_SENTRY_ID and i.cooldown_remaining == 0 and swap_item_cd[i.slot] == 0:
                        has_sentry_ward = True

            if len(hero_unit.modifiers) > 0:
                for m in hero_unit.modifiers:
                    if m.name == 'modifier_flask_healing' or hero_unit.health == hero_unit.health_max:
                        # 身上有药物buff或者满血满蓝就不吃药
                        has_flask = False
                    if m.name == 'modifier_tango_heal' or hero_unit.health == hero_unit.health_max:
                        has_tango = False
                        has_plant_tree = False
                    if hero_unit.mana + 110 >= hero_unit.mana_max:
                        has_mango = False

                    if m.name == 'modifier_clarity_potion' or hero_unit.mana == hero_unit.mana_max:
                        has_clarity = False

        if next_can_attack_time != -1 and cur_dota_time < next_can_attack_time:
            is_in_attack_interval = True

        if is_in_attack_interval or cur_dota_time < end_casting_ability_time:
            mask[ACTION_NAME_TO_INDEX["ATTACK_ENEMY"]] = 0
            mask[ACTION_NAME_TO_INDEX["ATTACK_SELF"]] = 0
            mask[ACTION_NAME_TO_INDEX["ATTACK_HERO"]] = 0
            mask[ACTION_NAME_TO_INDEX["ATTACK_TOWER"]] = 0

        if not has_tower_target:
            mask[ACTION_NAME_TO_INDEX["ATTACK_TOWER"]] = 0
        if not has_hero_target:
            mask[ACTION_NAME_TO_INDEX["ATTACK_HERO"]] = 0
        if not has_deny_target:
            mask[ACTION_NAME_TO_INDEX["ATTACK_SELF"]] = 0
        if not has_enemy_target:
            mask[ACTION_NAME_TO_INDEX["ATTACK_ENEMY"]] = 0

        if not has_flask or cur_dota_time < end_casting_ability_time:
            mask[ACTION_NAME_TO_INDEX["FLASK"]] = 0
        if not has_clarity or cur_dota_time < end_casting_ability_time:
            mask[ACTION_NAME_TO_INDEX["CLARITY"]] = 0

        # if not has_plant_tree or not has_tango:
        # 暂时不考虑种树吃树
        mask[ACTION_NAME_TO_INDEX["PLANT"]] = 0

        if not has_tango or cur_dota_time < tango_cd or cur_dota_time < end_casting_ability_time:
            mask[ACTION_NAME_TO_INDEX["EAT_TREE"]] = 0

        if not has_mango or cur_dota_time < end_casting_ability_time:
            mask[ACTION_NAME_TO_INDEX["MANGO"]] = 0

        if not has_magic_stick or cur_dota_time < end_casting_ability_time:
            mask[ACTION_NAME_TO_INDEX["MAGIC_STICK"]] = 0

        if not has_fire or cur_dota_time < end_casting_ability_time:
            mask[ACTION_NAME_TO_INDEX["FAERIE_FIRE"]] = 0

        if not has_ward_target or cur_dota_time < end_casting_ability_time:
            mask[ACTION_NAME_TO_INDEX["ATTACK_WARD"]] = 0

        if not has_ward or cur_dota_time < end_casting_ability_time:
            mask[ACTION_NAME_TO_INDEX["USE_WARD"]] = 0
        if not has_sentry_ward or cur_dota_time < end_casting_ability_time:
            mask[ACTION_NAME_TO_INDEX["USE_SENTRY_WARD"]] = 0

        if cur_dota_time < end_issue_attack_time:
            for i in range(self.actions_len):
                if i not in [1, pre_action_id]:
                    mask[i] = 0

        return mask
