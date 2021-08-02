# -*- coding: utf-8 -*-
import numpy as np
from dotaservice.protos.dota_gcmessages_common_bot_script_pb2 import CMsgBotWorldState
from dotaservice.dotautil import location_to_degree, cal_distance_with_z,  cal_distance

UNIT_NUM=5
def is_night(dota_time):
    # no test for luna or night stalker or phoniex,day and night is 5min cycle
    d = int(np.ceil(dota_time/300))
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
        return 30-dota_time % 30

def get_key_value(dic, key):
    if dic.HasField(key):
        return getattr(dic, key)
    else:
        return 0


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


class CBFeature:
    current_global_features = [
        'dota_time', 'game_time', 'player_team','num','A_','B_','C_','x','y','double_flag','dist_2tower','self_tower_pos_x','self_tower_pos_x',
    'stop_pos_x','stop_pos_y']

    current_unit_common_features = [
        'distance', 'z_distance','current_movement_speed', 'base_movement_speed', 'anim_activity', 'anim_cycle',
        'self_hero_x', 'self_hero_y', 'locationx', 'locationy',
        'locationz', 'degree_to_hero', 'facing','unit_type',
        'direct_x', 'direct_y', 'direct_z','hist_locationx1', 'hist_locationy1','hist_locationx2', 'hist_locationy2','hist_facing1','hist_facing2','creep_type',
        'hist_distance1','hist_distance2','hist_anim_cycle1','hist_anim_cycle2','hist_degree_to_hero1','hist_degree_to_hero2',
        'lane_distance','lane_distance1','lane_distance2','min_creep_dis','min_creep_dis1','min_creep_dis2'
    ]

    def __init__(self, team_id=3, player_id=5):
        self.team_id = team_id
        self.player_id = player_id
        if self.team_id == 3:
            self.enemy_team_id = 2
            self.enemy_player_id = 0
            self.enemy_tower_name = 'npc_dota_goodguys_tower1_mid'
            self.self_tower_name = 'npc_dota_badguys_tower1_mid'
        else:
            self.enemy_player_id = 5
            self.enemy_team_id = 3
            self.enemy_tower_name = 'npc_dota_badguys_tower1_mid'
            self.self_tower_name = 'npc_dota_goodguys_tower1_mid'

        # 只映射影魔的技能,天赋也在ability里面，影魔的天赋一般能反映在基础属性里，所以先不加了，后期5v5其他英雄
        self.buffer = {}

        self.handle_index = {}
        self.current_user = None
        self.current_enemy=None
        self.current_global = None


    def trans_feature(self, obs_dic, player_events_info, avg_delay, running_mode, creep_lane_dis,self_tower_position,stop_position):
        self.recent_avg_delay = avg_delay
        self.current_obs_dic = obs_dic
        self.player_events_info = player_events_info
        self.running_mode = running_mode
        self.creep_lane_dis=creep_lane_dis
        self.self_tower_pos=self_tower_position
        self.stop_pos=stop_position
        self.update_info(obs_dic, running_mode)

        # generate all feature data
        gf_dic, all_ucf_dic,all_ucategoy_dic= self.generate_all_feature(obs_dic)

        # generate selected feature data and padding
        return self.generate_selected_feature(gf_dic, all_ucf_dic,all_ucategoy_dic)

    # select feature from generate feature
    def generate_selected_feature(self, gf_dic, all_ucf_dic,  all_ucategoy_dic):
        # global feature
        global_f = np.zeros(len(CBFeature.current_global_features))
        for index, f in enumerate(CBFeature.current_global_features):
            if f in gf_dic:
                global_f[index] = gf_dic[f]

        # user common feature, shape(UNIT_NUM, ucf_f)
        ucf_f = []
        ucf_row_info = {}
        ucf_row_info_index = 0
        units_mask = np.zeros(UNIT_NUM, dtype=np.int)
        units_mask_index = 0
        unit_category = np.zeros(UNIT_NUM)

        for unit_handle, ucf_dic in all_ucf_dic.items():
            unit_common_f = np.zeros(len(CBFeature.current_unit_common_features))

            unit_category[units_mask_index] = all_ucategoy_dic[unit_handle]
            units_mask_index += 1

            for index, f in enumerate(CBFeature.current_unit_common_features):
                if f in ucf_dic:
                    unit_common_f[index] = ucf_dic[f]
                elif f in default_value:
                    unit_common_f[index] = default_value[f]

            ucf_f.append(unit_common_f)
            if 'distance' in ucf_dic:
                ucf_row_info[ucf_row_info_index] = [unit_handle, ucf_dic['distance']]
            else:
                ucf_row_info[ucf_row_info_index] = [unit_handle, 0]
            ucf_row_info_index += 1

        # padding 补齐空白
        if len(ucf_f) < UNIT_NUM:
            padding_f = np.zeros(len(CBFeature.current_unit_common_features))
            for i in range(UNIT_NUM - len(ucf_f)):
                ucf_f.append(padding_f)
                # fake unit handle id
                #ucf_row_info.append(-999999)

        ucf_f = np.array(ucf_f).reshape([UNIT_NUM, len(CBFeature.current_unit_common_features)])

        unit_category = np.array(unit_category).reshape(-1)


        return global_f, ucf_f, unit_category, ucf_row_info, units_mask



    """
        feature buffer
    """
    def update_info(self, d, running_mode):
        self.running_mode = running_mode
        self.handle_index = {}
        self.current_dota_time = d.dota_time

        for u in d.units:
            if u.HasField('handle'):
                self.handle_index[u.handle] = u

        # 每次buffer更新时把那些死掉的，很久不见unit从buffer里删掉，现在死亡复活时间最低是12秒
        buffer_unit_list = list(self.buffer.keys())
        for k in buffer_unit_list:
            if self.current_dota_time - self.buffer[k]['last_dotatime'] > 10:
                del self.buffer[k]

        for u in d.units:
            self.update_buffer(self.buffer, u)
            if u.player_id == self.player_id and u.team_id == self.team_id \
                    and u.unit_type == CMsgBotWorldState.UnitType.Value("HERO"):
                if u.HasField('location'):
                    self.current_user = u

    # need_fix: consider the time interval between each buffer update
    def update_buffer(self, buffer, dic):
        #为避免unit不在视野很久或者死亡复活归来，造成buffer不连续，所以每个unit若5秒不在视野内则其buffer清零，因为一级死亡复活时间是6秒。对于unit目前基本不会存在长时间丢失视野情况
        #之前没有过滤非1塔和其他非对战英雄，unit死亡后也buffer也没有去掉，这会造成buffer越来越大
        # buffer形如{27：{‘last_attack_time’:170,'health':[...]}}, 单位只存储存活期间,heanth默认值是该单位最大生命值，time默认0
        buffer_unit_list = buffer.keys()
        if (dic.unit_type in [CMsgBotWorldState.UnitType.Value("LANE_CREEP")] and self.team_id==dic.team_id) or (dic.player_id in [self.player_id] and dic.unit_type in [CMsgBotWorldState.UnitType.Value("HERO")]):
            if dic.is_alive is True:
                if dic.handle in buffer_unit_list:
                    buffer[dic.handle]['last_dotatime']=self.current_dota_time
                    buffer[dic.handle]['location'][0:2]=buffer[dic.handle]['location'][1:3]
                    buffer[dic.handle]['location'][2]=dic.location
                    buffer[dic.handle]['facing'][0:2]=buffer[dic.handle]['facing'][1:3]
                    buffer[dic.handle]['facing'][2]=dic.facing
                    buffer[dic.handle]['lane_dis'][0:2]=buffer[dic.handle]['lane_dis'][1:3]
                    buffer[dic.handle]['lane_dis'][2]=self.creep_lane_dis[dic.handle]
                    buffer[dic.handle]['min_creep_dis'][0:2]=buffer[dic.handle]['min_creep_dis'][1:3]
                    buffer[dic.handle]['min_creep_dis'][2]=cal_distance(dic.location,self.creep_lane_dis['min_creep'])
                else:
                    buffer[dic.handle] = {}
                    buffer[dic.handle]['last_dotatime']=self.current_dota_time
                    buffer[dic.handle]['name']=dic.name
                    buffer[dic.handle]['location']=[dic.location for i in range(3)]
                    buffer[dic.handle]['facing']=[dic.facing for i in range(3)]
                    buffer[dic.handle]['lane_dis']=[self.creep_lane_dis[dic.handle] for i in range(3)]
                    if 'min_creep' in self.creep_lane_dis:
                        buffer[dic.handle]['min_creep_dis']=[cal_distance(dic.location,self.creep_lane_dis['min_creep']) for i in range(3)]
                    else:
                        buffer[dic.handle]['min_creep_dis']=[0 for i in range(3)]

    def post_process_buffer(self,buffer,handle,ucf):
        if 'feature' in buffer[handle].keys():
            buffer[handle]['feature'][0:2]=buffer[handle]['feature'][1:3]
            buffer[handle]['feature'][2]=ucf
        else:
            buffer[handle]['feature']=[ucf for i in range(3)]

    """
        generate feature from obs dict
    """
    def generate_all_feature(self, dic):
        gf_dic = self.get_global_feature(dic)
        self.current_global = gf_dic
        all_ucf_dic = {}
        all_ucategoy_dic = {}

        for u in dic.units:
            # 英雄
            if u.unit_type == CMsgBotWorldState.UnitType.Value("HERO") and self.player_id==u.player_id:
                ucf, ucateoy = self.get_unit_feature(u, self.buffer)
                self.post_process_buffer(self.buffer,u.handle,ucf)
                all_ucf_dic[u.handle] = ucf
                all_ucategoy_dic[u.handle] = ucateoy

            # 兵和塔
            if u.is_alive is True and u.HasField('handle'):
                if u.unit_type in [CMsgBotWorldState.UnitType.Value("LANE_CREEP")] and self.team_id==u.team_id and cal_distance(
                    u.location, self.current_user.location)<1500:
                    ucf, ucateoy = self.get_unit_feature(u, self.buffer)
                    self.post_process_buffer(self.buffer, u.handle, ucf)
                    all_ucf_dic[u.handle] = ucf
                    all_ucategoy_dic[u.handle] = ucateoy
        tmp_dict = {}
        for key, value in list(all_ucf_dic.values())[0].items():
            if key in default_value.keys():
                tmp_dict[key]=default_value[key]
            else:
                tmp_dict[key] = 0

        return gf_dic, all_ucf_dic,  all_ucategoy_dic


    def get_unit_feature(self, d, buffer):
        dr = {}
        if d.unit_type==1:
            dr['unit_type']=0
        else:
            dr['unit_type']=1
        if d.HasField('team_id') and d.HasField('unit_type'):
            if d.team_id == self.enemy_team_id:
                if d.unit_type == 1:  # enemy HERO
                    utid = 5
                elif d.unit_type == 3:  # enemy CREEP
                    utid = 6
                else:
                    utid = 7  # enemy tower1
            elif d.team_id == self.team_id:
                if d.unit_type == 1:  # HERO
                    utid = 4
                elif d.unit_type == 3:  # LANE CREEP
                    utid = 2
                else:
                    utid = 9  # self tower1
            else:
                utid = 3
        else:
            utid = 1

        dr['current_movement_speed'] = get_key_value(d, 'current_movement_speed')
        dr['base_movement_speed'] = get_key_value(d, 'base_movement_speed')
        if d.anim_activity==1502:
            dr['anim_activity']=1
        else:
            dr['anim_activity']=0
        dr['anim_cycle'] = get_key_value(d, 'anim_cycle')


        if buffer.get(d.handle) is not None:
            if 'location' in buffer[d.handle]:
                dr['hist_locationx1']=buffer[d.handle]['location'][-2].x
                dr['hist_locationy1']=buffer[d.handle]['location'][-2].y
                dr['hist_locationx2']=buffer[d.handle]['location'][-3].x
                dr['hist_locationy2']=buffer[d.handle]['location'][-3].y
            if 'facing' in buffer[d.handle]:
                dr['hist_facing1']=buffer[d.handle]['facing'][-2]
                dr['hist_facing2']=buffer[d.handle]['facing'][-3]
            if 'lane_dis' in buffer[d.handle]:
                dr['lane_distance']=buffer[d.handle]['lane_dis'][-1]
                dr['lane_distance1']=buffer[d.handle]['lane_dis'][-2]
                dr['lane_distance2']=buffer[d.handle]['lane_dis'][-3]
            if 'min_creep_dis' in buffer[d.handle]:
                dr['min_creep_dis']=buffer[d.handle]['min_creep_dis'][-1]
                dr['min_creep_dis1']=buffer[d.handle]['min_creep_dis'][-2]
                dr['min_creep_dis2']=buffer[d.handle]['min_creep_dis'][-3]
        else:
            dr['hist_locationx1'] = d.location.x
            dr['hist_locationy1'] = d.location.y
            dr['hist_locationx2'] = d.location.x
            dr['hist_locationy2'] = d.location.y
            dr['hist_facing1'] = d.facing
            dr['hist_facing2'] = d.facing
            dr['lane_distance'] = 9999
            dr['lane_distance1'] = 9999
            dr['lane_distance2'] = 9999
            dr['min_creep_dis'] = 9999
            dr['min_creep_dis1'] = 9999
            dr['min_creep_dis2'] = 9999

        if d.HasField('location'):
            dr['locationx'] = d.location.x
            dr['locationy'] = d.location.y
            dr['locationz'] = d.location.z
            if self.current_user is not None:
                dr['direct_x'] = dr['locationx'] - self.current_user.location.x
                dr['direct_y'] = dr['locationy'] - self.current_user.location.y
                dr['direct_z'] = dr['locationz'] - self.current_user.location.z
                dr['distance'], dr['z_distance'] = cal_distance_with_z(
                    d.location, self.current_user.location)
                dr['degree_to_hero'] = location_to_degree(self.current_user.location, d.location)
        dr['facing'] = get_key_value(d, 'facing')
        if 'ranged' in d.name:
            dr['creep_type']=1
        elif 'melee' in d.name:
            dr['creep_type']=2
        else:
            dr['creep_type']=0

        if buffer.get(d.handle) is not None and 'feature' in buffer[d.handle]:
            dr['hist_distance1'] = buffer[d.handle]['feature'][-2]['distance']
            dr['hist_distance2'] = buffer[d.handle]['feature'][-3]['distance']
            dr['hist_anim_cycle1'] = buffer[d.handle]['feature'][-2]['anim_cycle']
            dr['hist_anim_cycle2'] = buffer[d.handle]['feature'][-3]['anim_cycle']
            dr['hist_degree_to_hero1'] = buffer[d.handle]['feature'][-2]['degree_to_hero']
            dr['hist_degree_to_hero2'] = buffer[d.handle]['feature'][-3]['degree_to_hero']

        else:
            dr['hist_distance1'] = dr['distance']
            dr['hist_distance2'] = dr['distance']
            dr['hist_anim_cycle1'] = dr['anim_cycle']
            dr['hist_anim_cycle2'] = dr['anim_cycle']
            dr['hist_degree_to_hero1'] = dr['degree_to_hero']
            dr['hist_degree_to_hero2'] = dr['degree_to_hero']
        # logger.info(dr)
        return dr, utid


    def get_global_feature(self, d):
        dr = {}
        dr['dota_time'] = get_key_value(d, 'dota_time')
        dr['game_time'] = get_key_value(d, 'game_time')
        dr['self_tower_pos_x']=self.self_tower_pos.x
        dr['self_tower_pos_y']=self.self_tower_pos.y
        dr['stop_pos_x']=self.stop_pos.x
        dr['stop_pos_y']=self.stop_pos.y

        if self.team_id == 2:
            dr['player_team'] = 0
        else:
            dr['player_team'] = 1

        if 'block_ferature' in self.creep_lane_dis:
            dr['num']=self.creep_lane_dis['block_ferature'][0]
            dr['A_']=self.creep_lane_dis['block_ferature'][1]
            dr['B_']=self.creep_lane_dis['block_ferature'][2]
            dr['C_']=self.creep_lane_dis['block_ferature'][3]
            dr['x']=self.creep_lane_dis['block_ferature'][4]
            dr['y']=self.creep_lane_dis['block_ferature'][5]
            dr['double_flag']=self.creep_lane_dis['block_ferature'][6]
            dr['dist_2tower']=self.creep_lane_dis['block_ferature'][7]
        else:
            dr['num']=0
            dr['A_']=0
            dr['B_']=0
            dr['C_']=0
            dr['x']=9999
            dr['y']=9999
            dr['double_flag']=-1
            dr['dist_2tower']=9999
        return dr

