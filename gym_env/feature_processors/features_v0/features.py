import numpy as np
from dotaservice.protos.dota_gcmessages_common_bot_script_pb2 import CMsgBotWorldState
from dotaservice.dotautil import location_to_degree, cal_distance_with_z, attack_to_death_times, in_facing_distance, \
     attack_time, attack_damage, cal_distance
from model.action_mask import ActionMask
from gym_env.feature_processors.enums import UNIT_NUM, HISTORY_HEALTH_NUM, LEVEL_TO_RESPAWNTIME
import datetime
from gym_env.feature_processors.enums import *
from gym_env.dota_game import logger


def respawn_time(level):
    return LEVEL_TO_RESPAWNTIME[level]


def is_night(dota_time):
    # no test for luna or night stalker or phoniex,day and night is 5min cycle
    d = int(np.ceil(dota_time / 300))
    if d % 2 == 0:
        return 1
    else:
        return 0


# 线上小兵的首次产生是在游戏时间为00: 00时。之后，它们每30秒产生一次。与近战和远程小兵不同的是，攻城兵在第11波才开始产生，而且只会每10波产生一次。简单来说，攻城兵的首次产生是在5: 00时，并且每5分钟产生一次。
# 兵线永远以近战兵为主。刚开始，兵线由3个近战兵和1个远程兵组成（每10波还会有1个攻城兵）。而它们的数量会逐渐提升：
# 第31波以及之后（15: 00）：每条路 + 1个近战兵。一个兵线中近战兵的总数为4个。
# 第61波以及之后（30: 00）：每条路 + 1个近战兵和 + 1个攻城兵。一个兵线中近战兵的总数为5个，攻城兵的总数为2个。
# 第81波以及之后（40: 00）：每条路 + 1个远程兵。一个兵线中远程兵的总数为2个。
# 第91波以及之后（45: 00）：每条路 + 1个近战兵。一个兵线中近战兵的总数为6个
# 45分钟之后，一个兵线将由6个近战兵，2个远程兵（每10波2个攻城兵）组成，总共有8个（加上攻城兵则为10个）小兵。
def is_fresh_creep(dota_time):
    if dota_time < 0:
        return abs(dota_time)
    else:
        return 30 - dota_time % 30


def is_fresh_siege(dota_time):
    if dota_time < 0:
        return abs(dota_time) + 300
    else:
        return 300 - dota_time % 300


def is_fresh_rune(dota_time):
    if dota_time < 0:
        return abs(dota_time) + 120
    else:
        # only consid river rune
        return 120 - dota_time % 120


def is_fresh_junge(dota_time):
    if dota_time < 0:
        return abs(dota_time) + 60
    else:
        return 60 - dota_time % 60


def is_attacking(d):
    flag = 0
    # flag=1 then attacking others,0means no
    if d.attack_target_handle != 0 and d.attack_target_handle != -1 and d.HasField('attack_target_handle'):
        flag = 1
    return flag


def is_attacked(d):
    flag = 0
    if len(d.incoming_tracking_projectiles) > 0:
        for tr in d.incoming_tracking_projectiles:
            if tr.is_attack is True:
                flag = 1
                return flag
    return flag


def get_key_value(dic, key):
    if dic.HasField(key):
        return getattr(dic, key)
    else:
        return 0


# 简单的数据分桶
def normalize_value(value, value_range=[], result_range=[], default_value=0):
    for index, v in enumerate(value_range):
        if value <= v:
            return result_range[index]

    return default_value


def one_hot_with_box(dic, key, value_range=[], result_range=[], default_value=0):
    value = get_key_value(dic, key)
    value = normalize_value(value, value_range=value_range, result_range=result_range, default_value=default_value)
    key = '{}_one_hot_{}'.format(key, value)
    return key, value


def is_creep_type(unit):
    if unit.unit_type == CMsgBotWorldState.UnitType.Value("LANE_CREEP"):
        if 'melee' in unit.name:
            return [1, 0, 0]
        elif 'ranged' in unit.name:
            return [0, 1, 0]
        elif 'siege' in unit.name:
            return [0, 0, 1]
        else:
            return [0, 0, 0]
    else:
        return [0, 0, 0]


# 特征特殊的非零默认值
default_value = {
    'locationx': 9999,
    'locationy': 9999,
    'locationz': 9999,
    'distance': 9999,
    'z_distance': 9999,
    'direct_x': 9999,
    'direct_y': 9999,
    'direct_z': 9999
}


class Feature:
    current_global_features = [
        'dota_time', 'is_night', 'time2refresh_creep', 'game_time', 'enemy_count', 'friends_count', 'creep_count',
        'enemy_hero_count', 'enemy_ability_1', 'enemy_ability_2', 'enemy_ability_3', 'kill_count', 'death_count', 'player_team',
        'avaiable_action_0', 'avaiable_action_1', 'avaiable_action_2', 'avaiable_action_3', 'avaiable_action_4',
        'avaiable_action_5', 'avaiable_action_6', 'avaiable_action_7', 'avaiable_action_8', 'avaiable_action_9',
        'avaiable_action_10', 'avaiable_action_11', 'avaiable_action_12', 'recent_avg_delay', 'enemy_hero_death',
        'enemy_hero_remain_spawn_time', 'action_interval_time', 'ward_observer', 'ward_sentry', 'enemy_ward_observer',
        'enemy_ward_sentry', 'enemy_ward_observer_unknown', 'enemy_ward_sentry_unknown'
    ]

    current_unit_common_features = [
        'level_0', 'level_1', 'level_2', 'level_3', 'level_4', 'level_5', 'level_6', 'level_7', 'level_8', 'level_9',
        'level_10', 'level_11', 'level_12', 'level_13', 'level_14', 'level_15', 'level_16', 'level_17', 'level_18', 'level_19',
        'level_20', 'health', 'distance', 'z_distance', 'mana', 'health_max', 'mana_max', 'xp_needed_to_level',
        'current_movement_speed', 'base_movement_speed', 'attack_range', 'attack_projectile_speed', 'attack_speed',
        'attack_anim_point', 'anim_activity', 'anim_cycle', 'attack_damage', 'bonus_damage', 'armor', 'self_hero_x',
        'self_hero_y', 'locationx', 'locationy', 'locationz', 'degree_to_hero', 'facing', 'magic_resist', 'is_attacked',
        'unit_type_one_hot_0', 'unit_type_one_hot_1', 'unit_type_one_hot_2', 'unit_type_one_hot_3', 'unit_type_one_hot_4',
        'unit_type_one_hot_5', 'unit_type_one_hot_6', 'unit_type_one_hot_7', 'unit_type_one_hot_8', 'unit_type_one_hot_9',
        'unit_type_one_hot_10', 'unit_type_one_hot_11', 'unit_type_one_hot_12', 'last_0_frame_health', 'last_1_frame_health',
        'last_2_frame_health', 'last_3_frame_health', 'last_4_frame_health', 'last_5_frame_health', 'last_6_frame_health',
        'last_7_frame_health', 'last_8_frame_health', 'last_9_frame_health', 'last_10_frame_health', 'last_11_frame_health',
        'direct_x', 'direct_y', 'direct_z', 'in_shadowrize_1', 'in_shadowrize_2', 'in_shadowrize_3', 'in_attack_range',
        'attacks_per_second', 'in_vision_range_daytime', 'in_vision_range_nighttime', 'in_vision',
        "closest_projectiles_distance", "closest_projectiles_get_damage_time", "closest_projectiles_get_damage",
        "max_damage_projectiles_distance", "max_damage_projectiles_get_damage_time",
        "max_damage_projectiles_distance_get_damage", 'self_hero_attacking_this_unit', 'is_attacking',
        'since_issue_attack_time_percent', 'is_in_attack_interval', 'since_attack_interval_time_percent',
        "is_casting_ability_one_hot_1", "is_casting_ability_one_hot_2", "is_casting_ability_one_hot_3",
        "is_casting_ability_one_hot_4", 'since_casting_ability_time_percent', 'is_attacking_me', 'is_attacking_enemy_hero',
        'enemy_hero_padding_seconds', 'creep_type_0', 'creep_type_1', 'creep_type_2', 'in_enemy_shadowrize_1',
        'in_enemy_shadowrize_2', 'in_enemy_shadowrize_3'
    ]

    current_hero_ability_features = [
        'level', 'cast_range', 'cooldown_remaining', 'ability_one_hot_1', 'ability_one_hot_2', 'ability_one_hot_3',
        'ability_one_hot_4', 'ability_one_hot_5', 'ability_one_hot_6'
    ]

    current_hero_modifier_feature = [
        'remaining_duration', 'stack_count', 'modify_one_hot_1', 'modify_one_hot_2', 'modify_one_hot_3', 'modify_one_hot_4',
        'modify_one_hot_5'
    ]

    current_hero_item_feature = [
        'cooldown_remaining', 'charges', 'count', 'item_one_hot_1', 'item_one_hot_2', 'item_one_hot_3', 'item_one_hot_4',
        'item_one_hot_5', 'item_one_hot_6', 'item_one_hot_7', 'item_one_hot_8', 'item_one_hot_9', 'item_one_hot_10',
        'item_one_hot_11', 'item_one_hot_12', 'item_one_hot_13', 'item_one_hot_14', 'item_one_hot_15', 'item_one_hot_16',
        'item_one_hot_17'
    ]

    current_courier_feature = [
        'locationx', 'locationy', 'locationz', 'distance', 'enemy_courier_distance', 'item_one_hot_1', 'item_one_hot_2',
        'item_one_hot_3', 'item_one_hot_4', 'item_one_hot_5', 'item_one_hot_6', 'item_one_hot_7', 'item_one_hot_8',
        'item_one_hot_9', 'item_one_hot_10', 'item_one_hot_11', 'item_one_hot_12', 'item_one_hot_13', 'item_one_hot_14',
        'item_one_hot_15', 'item_one_hot_16', 'item_one_hot_17', 'is_dead', 'state_1', 'state_2', 'state_3'
    ]

    current_home_item_feature = [
        'item_one_hot_1', 'item_one_hot_2', 'item_one_hot_3', 'item_one_hot_4', 'item_one_hot_5', 'item_one_hot_6',
        'item_one_hot_7', 'item_one_hot_8', 'item_one_hot_9', 'item_one_hot_10', 'item_one_hot_11', 'item_one_hot_12',
        'item_one_hot_13', 'item_one_hot_14', 'item_one_hot_15', 'item_one_hot_16', 'item_one_hot_17'
    ]

    hero_ability = {
        5059: 1, # nevermore_shadowraze1
        5060: 2, # nevermore_shadowraze2
        5061: 3, # nevermore_shadowraze3
        5062: 4,
        5063: 5,
        5064: 6
    }

    modifier_name = {
        "modifier_nevermore_necromastery": 1,
        "modifier_nevermore_shadowraze_debuff": 2,
        "modifier_flask_healing": 3,
        "modifier_nevermore_requiem_fear": 4,
        "modifier_tango_heal": 5
    }

    items_name = {
        14: 1, # item_slippers
        20: 2, # item_circlet
        ITEM_FLASK_ID: 3, # item_flask
        9999: 4, # item_branches // mask
        75: 5, # item_wraith_band
        29: 6, # item_boots
        25: 7, # item_gloves
        17: 8, # item_belt_of_strength
        73: 9, # item_bracer
        9998: 10, # item_blades_of_attack
        63: 11, # item_power_treads
        9997: 12, # item_lesser_crit
        ITEM_MANGO_ID: 13, # item_enchanted_mango
        ITEM_TANGO_ID: 14,
        ITEM_MAGIC_STICK_ID: 15,
        ITEM_MAGIC_WAND_ID: 16,
        ITEM_FAERIE_FIRE_ID: 17
    }

    courier_state_d = {
        # 'dead':1,dead已单独标注
        'idle': 1,
        'back': 2,
        'delivering': 3
    }

    def __init__(self, team_id=3, player_id=5, enemy_player_id=0, dotamap=None):
        self.team_id = team_id
        self.player_id = player_id
        self.dotamap = dotamap
        self.enemy_player_id = enemy_player_id

        if self.team_id == 3:
            self.enemy_team_id = 2
            self.enemy_tower_name = 'npc_dota_goodguys_tower1_mid'
            self.self_tower_name = 'npc_dota_badguys_tower1_mid'
        else:
            self.enemy_team_id = 3
            self.enemy_tower_name = 'npc_dota_badguys_tower1_mid'
            self.self_tower_name = 'npc_dota_goodguys_tower1_mid'

        # 只映射影魔的技能,天赋也在ability里面，影魔的天赋一般能反映在基础属性里，所以先不加了，后期5v5其他英雄
        self.buffer = {}
        self.feature_file_name = './saved_data/' + datetime.datetime.now().strftime("%Y-%m-%d_%H:%M:%S_%f") + "_fe.csv"
        self.debug_mode = False
        if self.debug_mode is True:
            open(self.feature_file_name, 'w')

        self.action_mask = ActionMask(team_id=self.team_id, player_id=self.player_id, enemy_player_id=self.enemy_player_id)
        self.handle_index = {}
        self.current_user = None
        self.current_enemy = None
        self.current_global = None
        self.pad_num = 0
        self.not_pad_num = 0
        self.death_not_pad_num = 0

        self.pre_kill_count = 0
        self.enemy_hero_level = 1
        self.enemy_hero_respawn_time = 0
        self.enemy_hero_death_flag = 0
        self.fog_pading = 0

        # before attack point
        self.start_issue_attack_time = -1
        self.end_issue_attack_time = -1
        self.cur_attack_handle = -1

        # attack interval
        self.latest_last_attack_time = 0
        self.start_can_not_attack_time = -1
        self.next_can_attack_time = -1

        # before cast point
        self.start_casting_ability_time = -1
        self.end_casting_ability_time = -1
        self.cur_casting_ability_id = -1

        self.pre_dota_time = -1

    def map_ability(self, ability_id):
        if ability_id in self.hero_ability.keys():
            return self.hero_ability[ability_id]
        else:
            return ability_id

    def save_data_all(self, gf_dic, all_ucf_dic, all_uaf_dic, all_ucategoy_dic, all_uif_dic, all_umf_dic):
        if self.debug_mode is False:
            return
        with open(self.feature_file_name, 'a') as f_w:
            write_str = "\n\n\n"
            write_str += "global feature \n"
            for f, value in gf_dic.items():
                write_str += "%s=%s\n" % (f, str(value))
            write_str += "\n"

            write_str += "unit common feature \n"
            for handle, ucf_dic in all_ucf_dic.items():
                write_str += "unit handle %s \n" % str(handle)
                for f, value in ucf_dic.items():
                    write_str += "%s=%s\n" % (f, str(value))
                write_str += "\n"
            write_str += "\n"

            write_str += "unit ability feature \n"
            for handle, uaf_dic_list in all_uaf_dic.items():
                write_str += "unit handle %s \n" % str(handle)
                for uaf_dic in uaf_dic_list:
                    for f, value in uaf_dic.items():
                        write_str += "%s=%s\n" % (f, str(value))
                    write_str += "\n"
            write_str += "\n"

            write_str += "unit category info \n"
            for handle, category in all_ucategoy_dic.items():
                write_str += "unit handle %s category %s \n" % (str(handle), str(category))

            write_str += "unit modifier feature \n"
            for handle, uaf_dic_list in all_umf_dic.items():
                write_str += "unit handle %s \n" % str(handle)
                for uaf_dic in uaf_dic_list:
                    for f, value in uaf_dic.items():
                        write_str += "%s=%s\n" % (f, str(value))
                    write_str += "\n"
            write_str += "\n"

            write_str += "unit item feature \n"
            for handle, uaf_dic_list in all_uif_dic.items():
                write_str += "unit handle %s \n" % str(handle)
                for uaf_dic in uaf_dic_list:
                    for f, value in uaf_dic.items():
                        write_str += "%s=%s\n" % (f, str(value))
                    write_str += "\n"
            write_str += "\n"

            write_str += "\n\n\n"
            f_w.write(write_str)

    def save_data_array(self, np_array, feature_define_list, feature_type, is_empty):
        if self.debug_mode is False:
            return
        with open(self.feature_file_name, 'a') as f_w:
            write_str = "save_data_array %s\n" % feature_type
            if is_empty:
                write_str += "zero data, length %d\n" % (len(feature_define_list))
            else:
                for index, f in enumerate(feature_define_list):
                    write_str += "%s=%s\n" % (f, np_array[index])
            write_str += "\n \n"
            f_w.write(write_str)

    # main func
    def trans_feature(self, obs_dic, player_events_info, avg_delay, running_mode, pre_action, pre_action_time, enemy_ability,
                      pre_attack_target_distance, tango_cd, courier_state, swap_item_cd, cur_attack_handle):

        self.swap_item_cd = swap_item_cd
        self.recent_avg_delay = avg_delay
        self.current_obs_dic = obs_dic
        self.player_events_info = player_events_info
        self.running_mode = running_mode
        self.enemy_ability = enemy_ability
        self.tango_cd = tango_cd
        self.courier_state = courier_state
        self.pre_action = pre_action
        if self.pre_dota_time == -1:
            self.pre_dota_time = obs_dic.dota_time
        else:
            self.pre_dota_time = self.current_dota_time
        self.current_dota_time = obs_dic.dota_time
        self.update_info(obs_dic, pre_action, pre_action_time, pre_attack_target_distance, running_mode, cur_attack_handle)

        # generate all feature data
        gf_dic, all_ucf_dic, all_uaf_dic, all_uif_dic, all_umf_dic, all_ucategoy_dic, mask, all_unit_attack_mask_dict, \
            myself_courier_f_dict,home_item_f_dict = self.generate_all_feature(obs_dic)

        if self.running_mode == "self_run":
            self.save_data_all(gf_dic, all_ucf_dic, all_uaf_dic, all_ucategoy_dic, all_uif_dic, all_umf_dic)

        # generate selected feature data and padding
        return self.generate_selected_feature(gf_dic, all_ucf_dic, all_uaf_dic, all_uif_dic, all_umf_dic, all_ucategoy_dic,
                                              mask, all_unit_attack_mask_dict, myself_courier_f_dict, home_item_f_dict)

    # select feature from generate feature
    def generate_selected_feature(self, gf_dic, all_ucf_dic, all_uaf_dic, all_uif_dic, all_umf_dic, all_ucategoy_dic, mask,
                                  all_unit_attack_mask_dict, myself_courier_f_dict, home_item_f_dict):
        # global feature
        global_f = np.zeros(len(Feature.current_global_features))
        for index, f in enumerate(Feature.current_global_features):
            if f in gf_dic:
                global_f[index] = gf_dic[f]

        courier_f = np.zeros(len(Feature.current_courier_feature))
        for index, f in enumerate(Feature.current_courier_feature):
            if f in myself_courier_f_dict:
                courier_f[index] = myself_courier_f_dict[f]
        home_item_f = np.zeros(len(Feature.current_home_item_feature))
        for index, f in enumerate(Feature.current_home_item_feature):
            if f in home_item_f_dict:
                home_item_f[index] = home_item_f_dict[f]

        # self.save_data_array(global_f, Feature.current_global_features, "global feature", False)

        # user common feature, shape(UNIT_NUM, ucf_f)
        ucf_f = []
        ucf_row_info = {}
        ucf_row_info_index = 0
        units_mask = np.zeros(UNIT_NUM, dtype=np.int)
        units_mask_index = 0
        unit_category = np.zeros(UNIT_NUM)

        for unit_handle, ucf_dic in all_ucf_dic.items():
            unit_common_f = np.zeros(len(Feature.current_unit_common_features))
            ability_common_f = np.zeros(shape=(len(Feature.hero_ability), len(Feature.current_hero_ability_features)))
            modify_common_f = np.zeros(shape=(len(Feature.modifier_name), len(Feature.current_hero_modifier_feature)))
            item_common_f = np.zeros(shape=(len(Feature.items_name), len(Feature.current_hero_item_feature)))

            units_mask[units_mask_index] = all_unit_attack_mask_dict[unit_handle]
            unit_category[units_mask_index] = all_ucategoy_dic[unit_handle]
            units_mask_index += 1

            for index, f in enumerate(Feature.current_unit_common_features):
                if f in ucf_dic:
                    unit_common_f[index] = ucf_dic[f]
                elif f in default_value:
                    unit_common_f[index] = default_value[f]

            # 技能扩展
            if all_uaf_dic.get(unit_handle) is not None:
                for index, key in enumerate(Feature.current_hero_ability_features):
                    for ab_index, uaf_dic in enumerate(all_uaf_dic.get(unit_handle)):
                        if key in uaf_dic:
                            ability_common_f[ab_index][index] = uaf_dic[key]

            # modifer
            if all_umf_dic.get(unit_handle) is not None:
                for index, key in enumerate(Feature.current_hero_modifier_feature):
                    for m_index, umf_dic in enumerate(all_umf_dic.get(unit_handle)):
                        if key in umf_dic:
                            modify_common_f[m_index][index] = umf_dic[key]

            # item
            if all_uif_dic.get(unit_handle) is not None:
                for index, key in enumerate(Feature.current_hero_item_feature):
                    for m_index, uif_dic in enumerate(all_uif_dic.get(unit_handle)):
                        if key in uif_dic:
                            item_common_f[m_index][index] = uif_dic[key]

            # self.save_data_array(unit_common_f, Feature.current_unit_common_features, "unit common feature", False)

            tmp = np.append(unit_common_f, ability_common_f.reshape(-1))
            tmp = np.append(tmp, modify_common_f.reshape(-1))
            tmp = np.append(tmp, item_common_f.reshape(-1))
            ucf_f.append(tmp)
            if 'distance' in ucf_dic:
                ucf_row_info[ucf_row_info_index] = [unit_handle, ucf_dic['distance']]
            else:
                ucf_row_info[ucf_row_info_index] = [unit_handle, 0]
            ucf_row_info_index += 1

        # padding 补齐空白
        if len(ucf_f) < UNIT_NUM:
            padding_f = np.zeros(
                len(Feature.current_unit_common_features) +
                len(Feature.hero_ability) * len(Feature.current_hero_ability_features) +
                len(Feature.modifier_name) * len(Feature.current_hero_modifier_feature) +
                len(Feature.items_name) * len(Feature.current_hero_item_feature))
            for i in range(UNIT_NUM - len(ucf_f)):
                # self.save_data_array(padding_f, Feature.current_unit_common_features, "unit common feature", True)
                ucf_f.append(padding_f)
                # fake unit handle id
                #ucf_row_info.append(-999999)

        global_f1 = np.append(global_f, courier_f.reshape(-1)).reshape(-1)
        global_f2 = np.append(global_f1, home_item_f.reshape(-1)).reshape(-1)
        ucf_nf = np.array(ucf_f).reshape([
            UNIT_NUM,
            len(Feature.current_unit_common_features) + len(self.hero_ability) * len(self.current_hero_ability_features) +
            len(Feature.modifier_name) * len(Feature.current_hero_modifier_feature) +
            len(Feature.items_name) * len(Feature.current_hero_item_feature)
        ])

        unit_category = np.array(unit_category).reshape(-1)

        if self.running_mode == "self_run":
            self.save_data_array(unit_category, [i for i in range(len(unit_category))], "unit_category", False)
            self.save_data_array(units_mask, [i for i in range(len(units_mask))], "units_mask", False)
            self.save_data_array(mask, [i for i in range(len(mask))], "action_mask", False)
            #self.save_data_array(ucf_row_info, [i for i in range(len(ucf_row_info))], "ucf_row_info", False)

        return global_f2, ucf_nf, unit_category, mask, ucf_row_info, units_mask

    def update_attack_point(self, current_user, current_dota_time, running_mode):
        # attack interval
        has_update = False
        if current_user.last_attack_time != self.latest_last_attack_time:
            self.latest_last_attack_time = current_user.last_attack_time
            self.next_can_attack_time = current_dota_time - (0.5 /
                                                             current_user.attack_speed) + (1 /
                                                                                           (0.625 * current_user.attack_speed))
            self.start_can_not_attack_time = current_dota_time
            has_update = True

            self.start_issue_attack_time = -1
            self.end_issue_attack_time = -1
            self.cur_attack_handle = -1
            if running_mode == "local_test_self":
                logger.info("%s, %f attack interval_start, next_can_attack_time %f, %f" %
                            (running_mode, current_dota_time, self.next_can_attack_time, self.latest_last_attack_time))

        if current_dota_time + 0.1 > self.next_can_attack_time and self.next_can_attack_time != -1:
            self.next_can_attack_time = -1
            self.start_can_not_attack_time = -1

            #if self.running_mode == "self_run":
            #    logger.info("%s, %f attack interval_end %f" % (
            #    self.running_mode, self.current_dota_time, self.next_can_attack_time))
        return has_update

    """
        feature buffer
    """

    def update_latest_health_info(self, d):
        # 每次buffer更新时把那些死掉的，很久不见unit从buffer里删掉，现在死亡复活时间最低是12秒
        buffer_unit_list = list(self.buffer.keys())
        for k in buffer_unit_list:
            if d.dota_time - self.buffer[k]['last_dotatime'] > 10:
                del self.buffer[k]

        for u in d.units:
            self.update_buffer(self.buffer, u, d.dota_time)

        # 检查buffer，给那些没有更新的unit，填充-1
        for k in self.buffer.keys():
            if d.dota_time != self.buffer[k]['last_dotatime']:
                self.buffer[k]['health'][0:11] = self.buffer[k]['health'][1:HISTORY_HEALTH_NUM]
                self.buffer[k]['health'][11] = -1

    def update_info(self, d, pre_action, pre_action_time, pre_attack_target_distance, running_mode, cur_attack_handle):
        self.running_mode = running_mode
        self.handle_index = {}

        for u in d.units:
            if u.HasField('handle'):
                self.handle_index[u.handle] = u

        self.attacking_self_hero_handle = {}
        for u in d.damage_events:
            if u.victim_player_id == self.player_id:
                self.attacking_self_hero_handle[u.attacker_unit_handle] = 1

        for u in d.units:
            if u.player_id == self.player_id and u.team_id == self.team_id \
                    and u.unit_type == CMsgBotWorldState.UnitType.Value("HERO"):
                if u.HasField('location'):
                    self.current_user = u
                if len(u.incoming_tracking_projectiles) > 0:
                    for danmu in u.incoming_tracking_projectiles:
                        self.attacking_self_hero_handle[danmu.caster_handle] = 1

            if u.player_id == self.enemy_player_id and u.team_id == self.enemy_team_id \
                    and u.unit_type == CMsgBotWorldState.UnitType.Value("HERO") and \
                    u.is_alive is True:
                if u.HasField('location'):
                    self.current_enemy = u
                    # logger.info([self.current_dota_time,self.current_enemy.location])

        # before cast point
        if pre_action in ["ABILITY_Q", "ABILITY_W", "ABILITY_E"] and self.end_casting_ability_time == -1:
            self.start_casting_ability_time = pre_action_time
            self.end_casting_ability_time = pre_action_time + 0.55
            if pre_action == "ABILITY_Q":
                self.cur_casting_ability_id = 5059
            elif pre_action == "ABILITY_W":
                self.cur_casting_ability_id = 5060
            elif pre_action == "ABILITY_E":
                self.cur_casting_ability_id = 5061

            if self.running_mode == "local_test_self":
                logger.info("%s, %f start cast" % (self.running_mode, self.current_dota_time))

        elif pre_action in ["STOP"] and self.end_casting_ability_time != -1:
            self.start_casting_ability_time = -1
            self.end_casting_ability_time = -1
            self.cur_casting_ability_id = -1

            if self.running_mode == "local_test_self":
                logger.info("%s, %f cancel cast, pre action %s " % (self.running_mode, self.current_dota_time, pre_action))

        if d.dota_time > self.end_casting_ability_time and self.end_casting_ability_time != -1:
            self.start_casting_ability_time = -1
            self.end_casting_ability_time = -1
            self.cur_casting_ability_id = -1

            if self.running_mode == "local_test_self":
                logger.info("%s, %f end cast" % (self.running_mode, self.current_dota_time))

        # before attack point time
        if pre_action in ["ATTACK_ENEMY", "ATTACK_SELF", "ATTACK_HERO", "ATTACK_TOWER"] \
               and pre_attack_target_distance < 580 and self.start_issue_attack_time == -1 and self.next_can_attack_time == -1:
            self.start_issue_attack_time = pre_action_time
            self.end_issue_attack_time = pre_action_time + (0.5 / self.current_user.attack_speed) + 0.1
            if pre_action in ["ATTACK_ENEMY", "ATTACK_SELF"]:
                self.cur_attack_handle = cur_attack_handle

            if self.running_mode == "local_test_self":
                logger.info("%s, %f start attack, pre action %s, speed %f, end time %f" %
                            (self.running_mode, self.current_dota_time, pre_action, self.current_user.attack_speed,
                             self.end_issue_attack_time))

        elif pre_action in ["STOP"] and self.end_issue_attack_time != -1:
            self.start_issue_attack_time = -1
            self.end_issue_attack_time = -1
            self.cur_attack_handle = -1
            if self.running_mode == "local_test_self":
                logger.info("%s, %f cancel attack, pre action %s" % (self.running_mode, self.current_dota_time, pre_action))

        if d.dota_time > self.end_issue_attack_time and self.end_issue_attack_time != -1:
            self.start_issue_attack_time = -1
            self.end_issue_attack_time = -1
            self.cur_attack_handle = -1
            if self.running_mode == "local_test_self":
                logger.info("%s, %f end attack" % (self.running_mode, self.current_dota_time))

    def select_bat_time(self, u):
        if u.unit_type == CMsgBotWorldState.UnitType.Value("LANE_CREEP"):
            if 'ranged' in u.name or 'melee' in u.name:
                return 1
            else:
                return 2.7
        elif u.unit_type == CMsgBotWorldState.UnitType.Value("HERO"):
            return 1.7
        elif u.unit_type == CMsgBotWorldState.UnitType.Value("TOWER"):
            return 1

    # need_fix: consider the time interval between each buffer update
    def update_buffer(self, buffer, dic, dota_time):
        #为避免unit不在视野很久或者死亡复活归来，造成buffer不连续，所以每个unit若5秒不在视野内则其buffer清零，因为一级死亡复活时间是6秒。对于unit目前基本不会存在长时间丢失视野情况
        #之前没有过滤非1塔和其他非对战英雄，unit死亡后也buffer也没有去掉，这会造成buffer越来越大
        # buffer形如{27：{‘last_attack_time’:170,'health':[...]}}, 单位只存储存活期间,heanth默认值是该单位最大生命值，time默认0
        buffer_unit_list = buffer.keys()
        if dic.unit_type in [CMsgBotWorldState.UnitType.Value("LANE_CREEP")] or dic.name in [
                self.enemy_tower_name, self.self_tower_name
        ] or dic.player_id in [self.enemy_player_id, self.player_id]:
            if dic.is_alive is True:
                if dic.handle in buffer_unit_list:
                    buffer[dic.handle]['last_dotatime'] = dota_time
                    buffer[dic.handle]['health'][0:11] = buffer[dic.handle]['health'][1:HISTORY_HEALTH_NUM]
                    if dic.HasField('health'):
                        buffer[dic.handle]['health'][11] = dic.health
                else:
                    buffer[dic.handle] = {}
                    buffer[dic.handle]['last_dotatime'] = dota_time
                    buffer[dic.handle]['health'] = [dic.health for i in range(HISTORY_HEALTH_NUM)]
                    buffer[dic.handle]['name'] = dic.name
                    if dic.HasField('health'):
                        buffer[dic.handle]['health'][-1] = dic.health

    def get_incoming_feature(self, u, feature_dict):
        if len(u.incoming_tracking_projectiles) > 0:
            closest_distance = 99999
            max_damge = -1
            for danmu in u.incoming_tracking_projectiles:
                if danmu.HasField("caster_handle") is False:
                    continue
                caster = self.handle_index.get(danmu.caster_handle)
                if caster is None:
                    continue
                distance, _ = cal_distance_with_z(danmu.location, u.location) # 弹道到目标的距离
                get_damage_time = distance / danmu.velocity
                get_damage = attack_damage(caster.attack_damage + caster.base_damage_variance, u.armor)
                if caster.handle == self.current_user.handle:
                    feature_dict['self_hero_attacking_this_unit'] = 1

                if distance < closest_distance:
                    feature_dict["closest_projectiles_distance"] = distance
                    feature_dict["closest_projectiles_get_damage_time"] = get_damage_time
                    feature_dict["closest_projectiles_get_damage"] = get_damage
                    closest_distance = distance

                if get_damage > max_damge:
                    feature_dict["max_damage_projectiles_distance"] = distance
                    feature_dict["max_damage_projectiles_get_damage_time"] = get_damage_time
                    feature_dict["max_damage_projectiles_distance_get_damage"] = get_damage
                    max_damge = get_damage

    """
        generate feature from obs dict
    """

    def generate_all_feature(self, dic):
        gf_dic = self.get_global_feature(dic)
        self.current_global = gf_dic
        all_ucf_dic = {}
        all_uaf_dic = {}
        all_uif_dic = {}
        all_umf_dic = {}
        all_ucategoy_dic = {}
        all_unit_attack_mask_dict = {}
        myself_courier_f_dict = {}
        home_item_f_dict = {}

        enemy_count = 0
        friends_count = 0

        has_enemy_hero = False
        has_enemy_tower = False
        enemy_hero_count = 0
        has_our_courier = False

        for u in dic.units:
            # 英雄
            if u.unit_type == CMsgBotWorldState.UnitType.Value("HERO"):
                all_unit_attack_mask_dict[u.handle] = self.action_mask.unit_attack_valid(self.current_user, u)
                # 只记录对战的两个英雄
                if u.player_id not in [self.player_id, self.enemy_player_id]:
                    continue
                #敌方英雄死亡时unit会一直在units中，但信息是其死前最后一拍的数据，所以这里不取，在后面pad
                if u.player_id == self.enemy_player_id and (self.enemy_hero_death_flag == 1 or u.is_alive == False):
                    continue
                # 只记录对战的两个英雄

                if u.player_id == self.enemy_player_id:
                    enemy_hero_count += 1
                    # ensure only one enemy hero unit
                    if has_enemy_hero:
                        continue

                    has_enemy_hero = True
                    self.enemy_hero_handle = u.handle
                    self.enemy_hero_level = u.level
                if u.player_id == self.player_id:
                    home_item_f_dict = self.get_home_item_feature(u)
                ucf, ucateoy = self.get_unit_feature(u, self.buffer)
                all_ucf_dic[u.handle] = ucf
                all_ucategoy_dic[u.handle] = ucateoy

                if len(u.abilities) > 0:
                    af = [{} for i in range(len(Feature.hero_ability))]
                    for i, ability in enumerate(u.abilities):
                        if ability.ability_id in Feature.hero_ability:
                            af[Feature.hero_ability[ability.ability_id] - 1] = self.get_ability_feature(ability)
                    #预测的敌方英雄影压技能cd和技能等级在这里填入
                    if ucateoy == 5:
                        for key, value in self.enemy_ability.items():
                            aid = Feature.hero_ability[key] - 1
                            af[aid]['level'] = value['level']
                            af[aid]['cooldown_remaining'] = value['cooldown']
                            af[aid]['ability_one_hot_%d' % (aid + 1)] = 1
                            if aid == 0:
                                af[aid]['cast_range'] = 200
                            elif aid == 1:
                                af[aid]['cast_range'] = 450
                            elif aid == 2:
                                af[aid]['cast_range'] = 700
                            af[aid]['charges'] = 0
                        #logger.info(af)

                    all_uaf_dic[u.handle] = af

                if len(u.modifiers) > 0:
                    mf = [{} for i in range(len(Feature.modifier_name))]
                    for i, mod in enumerate(u.modifiers):
                        if mod.name in Feature.modifier_name:
                            mf[Feature.modifier_name[mod.name] - 1] = self.get_modifier_feature(mod)
                    all_umf_dic[u.handle] = mf

                if len(u.items) > 0:
                    itemf = [{} for i in range(len(Feature.items_name))]
                    item_count = {}
                    for k, v in Feature.items_name.items():
                        item_count[k] = 0

                    for i, itemd in enumerate(u.items):
                        if itemd.ability_id in Feature.items_name:
                            item_count[itemd.ability_id] += itemd.charges
                            itemf[Feature.items_name[itemd.ability_id] - 1] = self.get_item_feature(
                                itemd, item_count[itemd.ability_id])
                    all_uif_dic[u.handle] = itemf

            # 兵和塔
            if u.is_alive is True and u.HasField('handle'):
                if u.unit_type in [CMsgBotWorldState.UnitType.Value("LANE_CREEP")
                                  ] or u.name in ['npc_dota_badguys_tower1_mid', 'npc_dota_goodguys_tower1_mid']:
                    if self.cur_attack_handle != -1:
                        if u.handle == self.cur_attack_handle:
                            all_unit_attack_mask_dict[u.handle] = self.action_mask.unit_attack_valid(self.current_user, u)
                        else:
                            all_unit_attack_mask_dict[u.handle] = 0
                    else:
                        all_unit_attack_mask_dict[u.handle] = self.action_mask.unit_attack_valid(self.current_user, u)

                    if u.name == self.enemy_tower_name:
                        has_enemy_tower = True
                        self.enemy_tower_handle = u.handle

                    ucf, ucateoy = self.get_unit_feature(u, self.buffer)
                    all_ucf_dic[u.handle] = ucf
                    all_ucategoy_dic[u.handle] = ucateoy

                    if ucateoy == 6 and ucf['distance'] <= 800:
                        enemy_count += 1
                    elif ucateoy == 3 and ucf['distance'] <= 800:
                        friends_count += 1

                if u.unit_type == CMsgBotWorldState.UnitType.Value("COURIER"):
                    if u.player_id == self.player_id:
                        has_our_courier = True
                        myself_courier_f_dict = self.get_courier_feature(u)
                    elif u.player_id == self.enemy_player_id:
                        myself_courier_f_dict["enemy_courier_distance"] = cal_distance(u.location, self.current_user.location)
                        #if self.running_mode == "self_run":
                        #    logger.info("enemy_courier distance %f" %myself_courier_f_dict["enemy_courier_distance"])

        gf_dic['enemy_hero_count'] = enemy_hero_count
        gf_dic['enemy_count'] = enemy_count
        gf_dic['friends_count'] = friends_count
        gf_dic['creep_count'] = 0
        if friends_count > enemy_count:
            gf_dic['creep_count'] = 1

        tmp_dict = {}
        for key, value in list(all_ucf_dic.values())[0].items():
            if key in default_value.keys():
                tmp_dict[key] = default_value[key]
            else:
                tmp_dict[key] = 0
        # logger.info('death flag {0},has enemy {1}'.format(self.enemy_hero_death_flag,has_enemy_hero))
        # padding fake hero and tower
        if has_enemy_hero is False:
            pad_flag, pad_dict = self.checkout_enemey_vision(dic, self.buffer)
            if pad_flag is True:
                all_ucf_dic[9999] = tmp_dict
            else:
                all_ucf_dic[9999] = pad_dict
            all_ucategoy_dic[9999] = 5
            all_unit_attack_mask_dict[9999] = 0
            self.enemy_hero_handle = 9999
        else:
            self.fog_pading = 0
        if has_enemy_tower is False:
            all_ucf_dic[9998] = tmp_dict
            all_ucategoy_dic[9998] = 7
            all_unit_attack_mask_dict[9998] = 0
            self.enemy_tower_handle = 9998

        if has_our_courier is False:
            myself_courier_f_dict['is_dead'] = 1

        return gf_dic, all_ucf_dic, all_uaf_dic, all_uif_dic, all_umf_dic, all_ucategoy_dic, \
            self.current_action_mask, all_unit_attack_mask_dict, myself_courier_f_dict,home_item_f_dict

    def get_home_item_feature(self, d):
        #这里slot位置以7.24新版本为准
        dr = {}
        if len(d.items) > 0:
            for im in d.items:
                if im.ability_id in Feature.items_name and im.slot >= 9 and im.slot < 15:
                    key = Feature.items_name[im.ability_id]
                    dr['item_one_hot_%d' % key] = im.charges
        return dr

    def get_courier_feature(self, d):
        dr = {}
        if d.HasField('location'):
            dr['locationx'] = d.location.x
            dr['locationy'] = d.location.y
            dr['locationz'] = d.location.z
            if self.current_user is not None:
                dr['distance'] = cal_distance(d.location, self.current_user.location)

        if len(d.items) > 0:
            for i in d.items:
                if i.ability_id in Feature.items_name:
                    key = Feature.items_name[i.ability_id]
                    dr['item_one_hot_%d' % key] = i.charges

        if self.courier_state in Feature.courier_state_d:
            s_id = Feature.courier_state_d[self.courier_state]
            dr['state_%d' % s_id] = 1
        return dr

    def get_modifier_feature(self, d):
        dr = {}
        m_id = 0
        if d.name in Feature.modifier_name:
            m_id = Feature.modifier_name[d.name]

        if self.running_mode == "local_test_self":
            if d.name == "modifier_flask_healing":
                logger.info("has flask modifier!!")

        dr['modify_id'] = m_id
        dr['modify_one_hot_%d' % m_id] = 1
        dr['remaining_duration'] = get_key_value(d, 'remaining_duration')
        dr['stack_count'] = get_key_value(d, 'stack_count')
        return dr

    def get_item_feature(self, d, count):
        dr = {}
        dr['count'] = count
        i_id = 0
        if d.ability_id in Feature.items_name:
            i_id = Feature.items_name[d.ability_id]
        dr['item_id'] = i_id
        dr['item_one_hot_%d' % i_id] = 1

        if d.slot in self.swap_item_cd and self.swap_item_cd[d.slot] > 0:
            dr['cooldown_remaining'] = self.swap_item_cd[d.slot] #get_key_value(d, 'cooldown_remaining')
        else:
            if d.slot in self.swap_item_cd:
                dr['cooldown_remaining'] = get_key_value(d, 'cooldown_remaining')
            else:
                dr['cooldown_remaining'] = 6

        dr['charges'] = 0 #get_key_value(d, 'charges')
        return dr

    def get_ability_feature(self, d):
        dr = {}
        a_id = self.map_ability(d.ability_id)
        dr['ability_embedding_id'] = a_id
        dr['ability_one_hot_%d' % a_id] = 1

        dr['level'] = get_key_value(d, 'level')
        dr['cast_range'] = get_key_value(d, 'cast_range')
        dr['cooldown_remaining'] = get_key_value(d, 'cooldown_remaining')
        dr['charges'] = get_key_value(d, 'charges')

        return dr

    def get_unit_feature(self, d, buffer, is_enemy_hero_padding=False):
        dr = {}
        current_level = get_key_value(d, 'level')
        dr['level_%d' % current_level] = 1
        dr['unit_type_embedding_id'] = get_key_value(d, 'unit_type')
        dr['unit_type_one_hot_%d' % dr['unit_type_embedding_id']] = 1

        if d.HasField('team_id') and d.HasField('unit_type'):
            if d.team_id == self.enemy_team_id:
                if d.unit_type == 1: # enemy HERO
                    utid = 5

                elif d.unit_type == 3: # enemy CREEP
                    utid = 6
                else:
                    utid = 7 # enemy tower1
            elif d.team_id == self.team_id:
                if d.unit_type == 1: # HERO
                    utid = 4
                elif d.unit_type == 3: # LANE CREEP
                    utid = 2
                else:
                    utid = 9 # self tower1
            else:
                utid = 3
        else:
            utid = 1

        dr['is_illusion'] = int(get_key_value(d, 'is_illusion'))
        dr['remaining_lifespan'] = get_key_value(d, 'remaining_lifespan')
        dr['health'] = get_key_value(d, 'health')
        dr['mana'] = get_key_value(d, 'mana')
        dr['health_max'] = get_key_value(d, 'health_max')
        dr['mana_max'] = get_key_value(d, 'mana_max')
        dr['xp_needed_to_level'] = get_key_value(d, 'xp_needed_to_level')
        dr['current_movement_speed'] = get_key_value(d, 'current_movement_speed')
        dr['base_movement_speed'] = get_key_value(d, 'base_movement_speed')
        dr['attack_range'] = get_key_value(d, 'attack_range')
        dr['attack_projectile_speed'] = get_key_value(d, 'attack_projectile_speed')
        dr['attack_speed'] = get_key_value(d, 'attack_speed')
        if is_enemy_hero_padding is False:
            dr['attack_anim_point'] = get_key_value(d, 'attack_anim_point')
            dr['anim_activity'] = get_key_value(d, 'anim_activity')
            dr['anim_cycle'] = get_key_value(d, 'anim_cycle')
        dr['attack_damage'] = get_key_value(d, 'attack_damage')
        dr['bonus_damage'] = get_key_value(d, 'bonus_damage')
        dr['attacks_per_second'] = get_key_value(d, 'attacks_per_second')
        creep_type = is_creep_type(d)
        for i, v in enumerate(creep_type):
            dr['creep_type_%d' % i] = v

        if d.handle in self.attacking_self_hero_handle:
            dr['is_attacking_me'] = 1
            #if self.running_mode == "local_test_self":
            #    logger.info("%d, %d is attacking me" %(utid, d.unit_type))

        if buffer.get(d.handle) is not None:
            if 'health' in buffer[d.handle]:
                for i in range(HISTORY_HEALTH_NUM):
                    dr['last_%d_frame_health' % i] = buffer[d.handle]['health'][HISTORY_HEALTH_NUM - i - 1]

        dr['armor'] = get_key_value(d, 'armor')

        if d.HasField('location'):
            dr['locationx'] = d.location.x
            dr['locationy'] = d.location.y
            dr['locationz'] = d.location.z
            if self.current_user is not None:
                dr['direct_x'] = dr['locationx'] - self.current_user.location.x
                dr['direct_y'] = dr['locationy'] - self.current_user.location.y
                dr['direct_z'] = dr['locationz'] - self.current_user.location.z
                dr['distance'], dr['z_distance'] = cal_distance_with_z(d.location, self.current_user.location)
                dr['degree_to_hero'] = location_to_degree(self.current_user.location, d.location)

                if d.team_id == self.enemy_team_id and (d.unit_type == 1 or
                                                        (d.unit_type == 3 and
                                                         'siege' not in d.name)) and is_enemy_hero_padding is False:
                    dr['in_shadowrize_1'] = in_facing_distance(self.current_user, d, 200, r=250, normalization=True)
                    dr['in_shadowrize_2'] = in_facing_distance(self.current_user, d, 450, r=250, normalization=True)
                    dr['in_shadowrize_3'] = in_facing_distance(self.current_user, d, 700, r=250, normalization=True)

                if is_enemy_hero_padding is False:
                    # 在攻击范围内
                    if dr['distance'] > 0 and self.current_user.attack_range > dr['distance']:
                        dr['in_attack_range'] = 1
                    else:
                        dr['in_attack_range'] = 0

                    # 在视野范围内
                    if dr['distance'] > 0 and self.current_user.vision_range_daytime > dr['distance']:
                        dr['in_vision_range_daytime'] = 1
                    else:
                        dr['in_vision_range_daytime'] = 0

                    if dr['distance'] > 0 and self.current_user.vision_range_nighttime > dr['distance']:
                        dr['in_vision_range_nighttime'] = 1
                    else:
                        dr['in_vision_range_nighttime'] = 0

                    if self.current_global['is_night']:
                        dr['in_vision'] = dr['in_vision_range_nighttime']
                    else:
                        dr['in_vision'] = dr['in_vision_range_daytime']

            if self.current_enemy is not None and d.team_id == self.team_id and (d.unit_type == 1 or
                                                                                 (d.unit_type == 3 and 'siege' not in d.name)):
                dr['in_enemy_shadowrize_1'] = in_facing_distance(self.current_enemy, d, 200, r=250, normalization=True)
                dr['in_enemy_shadowrize_2'] = in_facing_distance(self.current_enemy, d, 450, r=250, normalization=True)
                dr['in_enemy_shadowrize_3'] = in_facing_distance(self.current_enemy, d, 700, r=250, normalization=True)

        if self.current_user is not None and d.handle != self.current_user.handle and is_enemy_hero_padding is False:
            if d.HasField('attack_target_handle') and \
                    d.attack_target_handle == self.current_user.handle:
                dr['attack_myself_hero'] = 1
            else:
                dr['attack_myself_hero'] = 0

            if d.bounty_xp >= self.current_user.xp_needed_to_level:
                dr['kill_it_to_lvup'] = 1
            else:
                dr['kill_it_to_lvup'] = 0

        # 是否攻击敌方英雄
        if self.current_enemy is not None and d.handle != self.current_enemy.handle and is_enemy_hero_padding is False:
            if d.HasField('attack_target_handle') and d.attack_target_handle == self.current_enemy.handle:
                dr['is_attacking_enemy_hero'] = 1
            else:
                dr['is_attacking_enemy_hero'] = 0

        if is_enemy_hero_padding is False:
            dr['facing'] = get_key_value(d, 'facing')
            dr['magic_resist'] = get_key_value(d, 'magic_resist')
            dr['is_attacked'] = is_attacked(d)

        if d.unit_type == CMsgBotWorldState.UnitType.Value("HERO"):
            dr['strength'] = get_key_value(d, 'strength')
            dr['agility'] = get_key_value(d, 'agility')
            dr['intelligence'] = get_key_value(d, 'intelligence')
            dr['primary_attribute'] = get_key_value(d, 'primary_attribute')
            dr['health_regen'] = get_key_value(d, 'health_regen')
            dr['mana_regen'] = get_key_value(d, 'mana_regen')
            if d.HasField('reliable_gold') and d.HasField('unreliable_gold'):
                dr['gold'] = d.reliable_gold + d.unreliable_gold
            else:
                dr['gold'] = 0
            dr['reliable_gold'] = get_key_value(d, 'reliable_gold')
            dr['unreliable_gold'] = get_key_value(d, 'unreliable_gold')
            dr['buyback_cost'] = get_key_value(d, 'buyback_cost')
            dr['is_channeling'] = int(get_key_value(d, 'is_channeling'))

            if d.team_id == self.team_id:

                if self.start_issue_attack_time != -1:
                    dr['is_attacking'] = 1
                    dr['since_issue_attack_time_percent'] = float(self.current_dota_time - self.start_issue_attack_time) / (
                        self.end_issue_attack_time - self.start_issue_attack_time)
                    if self.running_mode == "local_test_self":
                        logger.info("%s, %f issue attack feature, %f " %
                                    (self.running_mode, self.current_dota_time, dr['since_issue_attack_time_percent']))

                if self.next_can_attack_time != -1 and self.current_dota_time < self.next_can_attack_time:
                    dr['is_in_attack_interval'] = 1
                    dr['since_attack_interval_time_percent'] = float(self.current_dota_time -
                                                                     self.start_can_not_attack_time) / (
                                                                         self.next_can_attack_time -
                                                                         self.start_can_not_attack_time)

                    #if self.running_mode == "self_run":
                    #    logger.info("%s, %f, current can not attack, %f "  % (self.running_mode, self.current_dota_time, dr['since_attack_interval_time_percent']))
                if self.start_casting_ability_time != -1:
                    #dr["is_casting_ability_one_hot_%d" %self.cur_casting_ability_id] = 1
                    dr['since_casting_ability_time_percent'] = float(self.current_dota_time -
                                                                     self.start_casting_ability_time) / (
                                                                         self.end_casting_ability_time -
                                                                         self.start_casting_ability_time)

                    #if self.running_mode == "self_run":
                    #    logger.info("%s, %f cast feature, %f " % (
                    #    self.running_mode, self.current_dota_time, dr['since_casting_ability_time_percent']))

        if is_enemy_hero_padding is False:
            self.get_incoming_feature(d, dr)

        return dr, utid

    def get_global_feature(self, d):
        dr = {}
        dr['dota_time'] = get_key_value(d, 'dota_time')
        dr['game_time'] = get_key_value(d, 'game_time')
        dr['is_night'] = is_night(dr['dota_time'])
        dr['time2refresh_creep'] = is_fresh_creep(dr['dota_time'])
        dr['time2fresh_siege'] = is_fresh_siege(dr['dota_time'])
        dr['time2fresh_rune'] = is_fresh_rune(dr['dota_time'])
        dr['time2refresh_jungle'] = is_fresh_junge(dr['dota_time'])
        dr['glyph_cooldown'] = get_key_value(d, 'glyph_cooldown')
        dr['glyph_cooldown_enemy'] = get_key_value(d, 'glyph_cooldown_enemy')
        dr['action_interval_time'] = self.current_dota_time - self.pre_dota_time

        dr['mango_num'] = 0
        dr['salve_num'] = 0
        dr['stick_num'] = 0
        dr['wand_num'] = 0

        dr['faerie_num'] = 0

        dr['kill_count'] = self.player_events_info.kills
        dr['death_count'] = self.player_events_info.deaths

        # remember enemy hero's respawn time
        if self.player_events_info.kills - self.pre_kill_count > 0:
            self.enemy_hero_respawn_time = self.current_dota_time + respawn_time(self.enemy_hero_level)
            self.enemy_hero_death_flag = 1
        if self.current_dota_time > self.enemy_hero_respawn_time:
            self.enemy_hero_death_flag = 0
        self.pre_kill_count = self.player_events_info.kills

        if self.enemy_hero_death_flag == 1:
            dr['enemy_hero_death'] = 1
            dr['enemy_hero_remain_spawn_time'] = int(self.enemy_hero_respawn_time - self.current_dota_time)

        if self.team_id == 2:
            dr['player_team'] = 0
        else:
            dr['player_team'] = 1

        # 真假眼
        dr['enemy_ward_observer_unknown'] = 1
        dr['enemy_ward_sentry_unknown'] = 1
        dr['ward_observer'] = 0
        dr['enemy_ward_observer'] = 0
        dr['ward_sentry'] = 0
        dr['enemy_ward_sentry'] = 0
        for u in d.units:
            if u.unit_type == CMsgBotWorldState.UnitType.Value("WARD") and u.is_alive:
                if u.name == "npc_dota_observer_wards":
                    if u.team_id == self.team_id:
                        dr['ward_observer'] = 1
                    else:
                        dr['enemy_ward_observer'] = 1
                        dr['enemy_ward_observer_unknown'] = 0
                elif u.name == "npc_dota_sentry_wards":
                    if u.team_id == self.team_id:
                        dr['ward_sentry'] = 1
                    else:
                        dr['enemy_ward_sentry'] = 1
                        dr['enemy_ward_sentry_unknown'] = 0

        #if self.running_mode == "local_test_self":
        #    logger.info("ward_observer %d, ward_sentry %d, enemy_ward_observer %d, enemy_ward_sentry %d, enemy_ward_observer_unknown %d, enemy_ward_sentry_unknown %d" %(
        #        dr['ward_observer'], dr['ward_sentry'], dr['enemy_ward_observer'], dr['enemy_ward_sentry'], dr['enemy_ward_observer_unknown'], dr['enemy_ward_sentry_unknown']
        #    ))

        self.current_action_mask = self.action_mask.valid(self.current_user, d.units, self.current_dota_time,
                                                          self.end_issue_attack_time, self.next_can_attack_time,
                                                          self.end_casting_ability_time, self.dotamap, self.tango_cd,
                                                          self.cur_casting_ability_id, self.swap_item_cd,
                                                          ACTION_NAME_TO_INDEX[self.pre_action], self.cur_attack_handle)

        for index, i in enumerate(self.current_action_mask.tolist()):
            dr['avaiable_action_%d' % index] = i

        dr["recent_avg_delay"] = self.recent_avg_delay

        # enemy ability
        if len(d.ability_events) > 0:
            for ae in d.ability_events:
                if ae.player_id == self.enemy_player_id:
                    if ae.ability_id == 5059:
                        dr['enemy_ability_1'] = 1
                    elif ae.ability_id == 5060:
                        dr['enemy_ability_2'] = 1
                    elif ae.ability_id == 5061:
                        dr['enemy_ability_3'] = 1
        return dr

    def checkout_enemey_vision(self, obs, buffer):
        # 确定敌方英雄最后一拍位置是否在目前我方可见区域，如果不在则用其最后一拍的位置代替维持至多20拍2.66秒，超过则设为默认值；如果其最后一拍位置已在我方视野内则设为默认值
        # 查看敌方英雄是否在所有友方单位的视野范围内，条件是黑夜白天视野范围内且高低地，先不考虑树木的问题，因为不知道树木的遮掩机制
        # 如果敌方死亡，则直接pad
        # logger.info('death {0},len enemy obs {1}'.format(self.enemy_hero_death_flag,len(self.enemy_history_info)))
        if self.enemy_hero_death_flag == 1:
            # logger.info('not pad,step {0},death'.format(self.fog_pading))
            self.death_not_pad_num += 1
            return True, 0

        in_vision_flag = False
        pad_flag = False
        if self.current_enemy is not None:
            for u in obs.units:
                if u.is_alive is True and u.HasField('handle'):
                    if (u.unit_type in [CMsgBotWorldState.UnitType.Value("LANE_CREEP")] or u.name == self.self_tower_name) and \
                        u.team_id == self.team_id and u.HasField('location') and u.HasField('vision_range_daytime') and \
                            u.HasField('vision_range_nighttime'):
                        if u.location.z < self.current_enemy.location.z:
                            continue
                        else:
                            if self.current_global['is_night']:
                                if cal_distance(self.current_enemy.location, u.location) < 200:
                                    in_vision_flag = True
                                    break
                            else:
                                if cal_distance(self.current_enemy.location, u.location) < 200:
                                    in_vision_flag = True
                                    break

        if in_vision_flag == False and self.fog_pading < 50 and self.current_enemy is not None:
            r, utid = self.get_unit_feature(self.current_enemy, buffer, is_enemy_hero_padding=True)
            r['in_vision_range_nighttime'] = 0
            r['in_vision_range_daytime'] = 0
            r['in_vision'] = 0
            r['enemy_hero_padding_seconds'] = self.fog_pading
            self.fog_pading += 1
        else:
            r = 0
            pad_flag = True
            self.current_enemy = None
        return pad_flag, r
