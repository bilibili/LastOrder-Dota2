import math
from gym_env.feature_processors.enums import ITEM_WARD_SENTRY_ID, ITEM_WARD_ID, ITEM_WARD_DISPENSER_ID
from dotaservice.dotautil import cal_distance
from dotaservice.protos.dota_gcmessages_common_bot_script_pb2 import CMsgBotWorldState
from gym_env.dota_game import logger
import numpy as np

MACHINE_ACTION = [
    'PREPARE', 'MODEL', 'MAX_DOTA_TIME', 'DEAD', 'USE_TP', 'CREEP_BLOCK', 'EAT_TREE', 'FOLLOW', 'USE_WARD', 'USE_SENTRY_WARD',
    'GO_TO_TOWER'
]


class StateMachine:
    dota_state = {
        'action': 'PREPARE', # 动作状态，使用模型还是脚本
        'dota_time': 'READY', # 游戏时间状态
        'live': 'LIVE', # 玩家生命状态
        'health': 'ENOUGH', # 玩家血量状态
        'mana': 'ENOUGH', # 玩家魔法量状态
        'hits': 'EQUAL' # 补刀局面
    }

    def __init__(self, init_position, start_position, stop_position, game_max_time, creep_block_max_time, team_id,
                 self_tower_position, enemy_tower_position, running_mode):
        self.init_position = init_position
        self.start_position = start_position
        self.stop_position = stop_position
        self.game_max_time = game_max_time
        self.creep_block_max_time = creep_block_max_time
        self.team_id = team_id
        self.self_tower_position = self_tower_position
        self.enemy_tower_position = enemy_tower_position
        self.running_mode = running_mode
        self.creeps = []
        self.pre_num = 1

        self.dota_state['action'] = 'PREPARE'

    def get_vector_lenth(self, p):
        return math.sqrt(math.pow(p.x, 2) + math.pow(p.y, 2))

    def get_vector_dot(self, p1, p2):
        return p1.x * p2.x + p1.y * p2.y

    def get_lane_distance(self, p1):
        x = self.start_position.x - self.stop_position.x
        y = self.start_position.y - self.stop_position.y
        lane_vector = CMsgBotWorldState.Vector(x=x, y=y, z=256)
        x = p1.x - self.stop_position.x
        y = p1.y - self.stop_position.y
        p_vector = CMsgBotWorldState.Vector(x=x, y=y, z=256)

        cos_to_lane_axis = self.get_vector_dot(lane_vector, p_vector) / (
            self.get_vector_lenth(lane_vector) * self.get_vector_lenth(p_vector))
        d = self.get_vector_lenth(p_vector)
        return abs(d * cos_to_lane_axis)

    def enter_state(self, state):
        self.dota_state['action'] = state

    def is_action_state(self, state):
        return self.dota_state['action'] == state

    def dispatcher(self, obs, current_player):
        self_player = current_player
        self.self_position = self_player.location
        # self_courier = self.self_courier
        self.creep_lane_dis = {}
        if self_player.is_alive is False:
            self.dota_state['action'] = 'DEAD'
        if obs.dota_time > self.game_max_time:
            self.dota_state['action'] = 'MAX_DOTA_TIME'

        if self.dota_state['action'] == 'PREPARE':
            if obs.dota_time > 0 and obs.dota_time < self.creep_block_max_time:
                self.dota_state['action'] = 'CREEP_BLOCK'
                if self.running_mode == "local_test_self":
                    logger.info("enter CREEP_BLOCK")
            elif obs.dota_time > self.creep_block_max_time:
                self.dota_state['action'] = 'MODEL'
        if self.dota_state['action'] == 'CREEP_BLOCK':
            self_to_lane_distance = self.get_lane_distance(self.self_position)
            num, A_, B_, C_, x, y, double_flag, dist_2tower = self.block(self_player, self_to_lane_distance)
            self.creep_lane_dis['block_ferature'] = [num, A_, B_, C_, x, y, double_flag, dist_2tower]
            self_creep_min_distance = 9999
            enemey_creep_min_distance = 9999
            middle_loc_dis2_tower = 9999
            self_creep_min = None
            enemey_creep_min = None
            creep_block_fail = False
            enemy_hero_insight = False
            self.creep_lane_dis[self_player.handle] = self_to_lane_distance
            for u in obs.units:
                if u.unit_type == CMsgBotWorldState.UnitType.Value("LANE_CREEP"):
                    if u.team_id == self.team_id:
                        unit_distance = self.get_lane_distance(u.location)
                        self.creeps.append((unit_distance, u))
                        self.creep_lane_dis[u.handle] = unit_distance
                        if self_to_lane_distance - unit_distance > 50:
                            creep_block_fail = True
                        dist = cal_distance(u.location, self.enemy_tower_position)
                        if dist < self_creep_min_distance:
                            self_creep_min = u
                            self_creep_min_distance = dist
                        if 'ranged' in u.name:
                            self.first_range_creep = u
                    else:
                        dist = cal_distance(u.location, self.self_tower_position)
                        if dist < enemey_creep_min_distance:
                            enemey_creep_min = u
                            enemey_creep_min_distance = dist
                if u.unit_type == CMsgBotWorldState.UnitType.Value("HERO") and u.team_id != self.team_id:
                    if self.get_vector_lenth(
                            CMsgBotWorldState.Vector(
                                x=self_player.location.x - u.location.x, y=self_player.location.y - u.location.y, z=256)) < 500:
                        enemy_hero_insight = True

            if self_creep_min is not None:
                self.creep_lane_dis['min_creep'] = CMsgBotWorldState.Vector(
                    x=self_creep_min.location.x, y=self_creep_min.location.y, z=256)
            if self_creep_min is not None and enemey_creep_min is not None:
                x = (self_creep_min.location.x + enemey_creep_min.location.x) / 2
                y = (self_creep_min.location.y + enemey_creep_min.location.y) / 2
                middle_loc_dis2_tower = cal_distance(CMsgBotWorldState.Vector(x=x, y=y, z=0), self.self_tower_position)

            if (self_creep_min is not None and enemey_creep_min is not None and cal_distance(enemey_creep_min.location, self_creep_min.location) < 500)\
                    or (obs.dota_time > self.creep_block_max_time) or enemy_hero_insight:
                self.dota_state['action'] = 'MODEL'
                if self.running_mode in ["local_test_self", 'local_test_opponent']:
                    logger.info("{0} enter MODEL,time{1}".format(self.running_mode, obs.dota_time))
            elif middle_loc_dis2_tower <= 700 or creep_block_fail == True:
                #若中点落入我方塔攻击范围内则开始跟随或者卡兵失败
                if self_to_lane_distance > self.get_lane_distance(self.self_tower_position):
                    self.dota_state['action'] = 'FOLLOW'
                    if self.running_mode in ["local_test_self", 'local_test_opponent']:
                        logger.info("{0} enter FOLLOW,time {1}".format(self.running_mode, obs.dota_time))
                else:
                    self.dota_state['action'] = 'MODEL'
                    if self.running_mode in ["local_test_self", 'local_test_opponent']:
                        logger.info("{0} enter MODEL,time {1}".format(self.running_mode, obs.dota_time))
        elif self.dota_state['action'] == 'FOLLOW':
            if cal_distance(self.self_position, self.self_tower_position) < 300 or (obs.dota_time > self.creep_block_max_time):
                self.dota_state['action'] = 'MODEL'
                if self.running_mode in ["local_test_self", 'local_test_opponent']:
                    logger.info("{0} enter MODEL,time{1}".format(self.running_mode, obs.dota_time))

        elif self.dota_state['action'] == 'DEAD':
            if self_player.is_alive is True:
                self.dota_state['action'] = 'USE_TP'

        elif self.dota_state['action'] == 'USE_TP':
            if cal_distance(self.init_position, self.self_position) > 2000:
                self.dota_state['action'] = 'MODEL'

        elif self.dota_state['action'] == 'EAT_TREE':
            for m in self_player.modifiers:
                if m.name == 'modifier_tango_heal':
                    self.dota_state['action'] = 'MODEL'
                    break

        elif self.dota_state['action'] == 'USE_WARD':
            has_ward = False
            if len(self_player.items) != 0:
                for item in self_player.items:
                    if item.ability_id in [ITEM_WARD_ID, ITEM_WARD_DISPENSER_ID] and \
                            item.slot <= 5:
                        has_ward = True
                        break
            if has_ward is False:
                self.dota_state['action'] = 'MODEL'

        elif self.dota_state['action'] == 'USE_SENTRY_WARD':
            has_ward = False
            if len(self_player.items) != 0:
                for item in self_player.items:
                    if item.ability_id in [ITEM_WARD_SENTRY_ID] and \
                            item.slot <= 5:
                        has_ward = True
                        break
            if has_ward is False:
                self.dota_state['action'] = 'MODEL'

        elif self.dota_state['action'] == 'GO_TO_TOWER':
            tower_pos = CMsgBotWorldState.Vector(x=460, y=855, z=256)
            if self.team_id == 2:
                tower_pos = CMsgBotWorldState.Vector(x=-1408, y=-1517, z=256)

            if cal_distance(self.self_position, tower_pos) < 200:
                self.dota_state['action'] = 'MODEL'

            for u in obs.units:
                if u.team_id != self.team_id and u.is_alive and cal_distance(u.location, self.self_position) < 600:
                    self.dota_state['action'] = 'MODEL'
                    break

        elif self.dota_state['action'] == 'MODEL':
            if obs.dota_time < 15:
                enemy_hero_insight = False
                for u in obs.units:
                    if u.unit_type == CMsgBotWorldState.UnitType.Value("HERO") and u.team_id != self.team_id:
                        enemy_hero_insight = True
                if enemy_hero_insight is not True:
                    self.dota_state['action'] = 'GO_TO_TOWER'
        # logger.info(self.creep_lane_dis)
        return self.creep_lane_dis

    def block(self, current_player, self_to_lane_distance):
        min_dist = 9999
        double_flag = False
        if self.creeps == []:
            return 0, 0, 0, 0, 9999, 9999, -1, 9999
        dis_2tower = cal_distance(current_player.location, self.self_tower_position)
        if self.pre_num > 1:
            threshold = 40
        else:
            threshold = 32
        for c in self.creeps:
            if c[0] < min_dist:
                min_dist = c[0]
                min_dist_u = c[1]
        if self.team_id == 2:
            coef = np.polyfit([-869, -4855], [-747, -4380], 1)
        else:
            if current_player.location.x < 2520 or current_player.location.y < 1730:
                coef = np.polyfit([2520, -150], [1730, -50], 1)
            else:
                coef = np.polyfit([2520, 4100], [1730, 3570], 1)
        delta_lane_dis = min_dist - self_to_lane_distance
        num = 1
        bonus_num = 0
        raw_x = min_dist_u.location.x
        raw_y = min_dist_u.location.y
        dis_2forward_creeps = [
            self.get_vector_lenth(
                CMsgBotWorldState.Vector(
                    x=min_dist_u.location.x - c[1].location.x, y=min_dist_u.location.y - c[1].location.y, z=256))
            for c in self.creeps
        ]
        for i, c in enumerate(self.creeps):
            # 判断是否呈两列，主要防范出兵时成两列
            if (min_dist_u.location.x - c[1].location.x) * (
                    min_dist_u.location.y -
                    c[1].location.y) < 0 and self.team_id == 2 and current_player.location.x <= self.self_tower_position.x:
                if dis_2forward_creeps[i] > 200:
                    double_flag = True
                    solo_creep_dist_lane = c[0]
                    if (solo_creep_dist_lane - self_to_lane_distance) - delta_lane_dis < 150 and num == 1:
                        raw_x += c[1].location.x
                        raw_y += c[1].location.y
                        num += 1
                    elif (solo_creep_dist_lane - self_to_lane_distance) - delta_lane_dis > 220 and num == 1:
                        raw_x += 0.5 * min_dist_u.location.x
                        raw_y += 0.5 * min_dist_u.location.y
                        bonus_num += 0.5

            u_2forward_dis = self.get_vector_lenth(
                CMsgBotWorldState.Vector(
                    x=c[1].location.x - min_dist_u.location.x, y=c[1].location.y - min_dist_u.location.y, z=256))
            if c[0] != min_dist and (c[0] - min_dist < threshold):
                if num == 1 and u_2forward_dis < 120 and delta_lane_dis > 90:
                    raw_x += 0.25 * min_dist_u.location.x
                    raw_y += 0.25 * min_dist_u.location.y
                    bonus_num += 0.25
                raw_x += c[1].location.x
                raw_y += c[1].location.y
                num = num + 1
        if double_flag == False and num < 3:
            raw_x = raw_x / (num + bonus_num)
            raw_y = raw_y / (num + bonus_num)
        else:
            raw_x = (raw_x + current_player.location.x) / (num + 1 + bonus_num)
            raw_y = (raw_y + current_player.location.y) / (num + 1 + bonus_num)

        b = raw_y - coef[0] * raw_x
        A_ = coef[0]**2 + 1
        B_ = 2 * (coef[0] * b - coef[0] * raw_y - raw_x)
        C_ = raw_x**2 + (b - raw_y)**2 - 200**2

        root = np.sqrt(B_**2 - 4 * A_ * C_)
        x = (-B_ + root) / (2 * A_)
        if self.team_id == 2:
            if x < raw_x:
                x = (-B_ - root) / (2 * A_)
        else:
            if x > raw_x:
                x = (-B_ - root) / (2 * A_)
        y = coef[0] * x + b

        self.creeps = []
        self.pre_num = num
        return num, A_, B_, C_, x, y, int(double_flag), dis_2tower
