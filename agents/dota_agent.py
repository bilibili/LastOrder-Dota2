import math, json, time, pickle, os.path, random, queue
import numpy as np
from dotaservice.state_machine import StateMachine
from dotaservice.protos.dota_gcmessages_common_bot_script_pb2 import CMsgBotWorldState
from model.cb_features import CBFeature
from dotaservice.dotautil import cal_distance
from gym_env.feature_processors.enums import *
from gym_env.feature_processors.features_v0.features import Feature
from agents.base_agent import BaseAgent
from agents.instance import Instance
from model.dota_models.ppo import PPOModel
from model.utils import get_health_reward, sampling_action_cb
from gym_env.dota_game import logger, TEAM_DIRE, TEAM_RADIANT
from nevermore import Nevermore

# 夜宴方塔的坐标
# 基地塔 {'x': 4943.9990234375, 'y': 4776.0, 'z': 383.9998779296875}
# 基地塔 {'x': 5279.99951171875, 'y': 4431.9990234375, 'z': 383.9998779296875}
# 上高塔 {'x': 3551.999755859375, 'y': 5776.0, 'z': 384.0}
# 上二塔 {'x': -0.00034500000765547156, 'y': 6016.0, 'z': 256.0}
# 上一塔 {'x': -4672.00048828125, 'y': 6016.0, 'z': 256.0}
# 中高塔 {'x': 4272.0, 'y': 3759.0, 'z': 384.0}
# 中二塔 {'x': 2496.0, 'y': 2111.999755859375, 'z': 256.0}
# 中一塔 {'x': 523.999755859375, 'y': 651.9998168945312, 'z': 128.0}
# 下高塔 {'x': 6336.0, 'y': 3031.99951171875, 'z': 383.999755859375}
# 下二塔 {'x': 6208.0, 'y': 383.99951171875, 'z': 256.0}
# 下一塔 {'x': 6269.3388671875, 'y': -1728.74169921875, 'z': 256.0}

# load ability KV
with open(os.path.join('data', 'npc_abilities_kv.json'), 'r') as f:
    ABILITY_KV = json.load(f)


class PPOAgent(BaseAgent):

    def __init__(self,
                 dota_game,
                 team_id,
                 player_id,
                 enemy_player_id,
                 play_mode,
                 mode,
                 self_data_queue=None,
                 enemy_data_queue=None,
                 opponent_info_queue=None):

        BaseAgent.__init__(self, dota_game, team_id, player_id, enemy_player_id, mode, self_data_queue, enemy_data_queue)
        self.self_position = None
        self.feature = Feature(team_id=team_id, player_id=player_id, enemy_player_id=enemy_player_id, dotamap=self.dota_map)
        self.cb_feature = CBFeature(team_id=team_id, player_id=player_id)
        self.instances = []
        self.step = 0
        self.opponent_info_queue = opponent_info_queue
        self.current_elo = 0
        self.current_index = 0
        self.game_max_time = 60 * 12
        self.play_mode = play_mode

        # 游戏规则
        self.rule_msg = [
            "dota_lobby_gameplay_rules", # 游戏规则
            "dota_play_1v1_desc", # "与其他玩家一起练习中路对单的技艺。比赛将在其中一方率先阵亡两次或丢失一座防御塔后结束。"
            "DOTA_Hero_Selection_BanTitle", # 禁用
            "====================",
            "dota_hud_error_cant_target_rune", # 不能选定神符
            "DOTA_ActivateGlyph", # 激活符文
            "DOTA_ActivateRadar", # 使用扫描
            "item_infused_raindrop",
            "item_bottle",
            "item_soul_ring",
            "====================",
        ]

        # 欢迎语
        self.wel_msg = [
            "dota_chatwheel_label_rattletrap_4", # 我是个机器人
            "dota_chatwheel_label_chaos_knight_2" # 久仰大名
        ]

        self.chat_flag = False

        if self.running_mode in ["self_eval"]:
            self.test_mode = True # True

            self.has_reload = False

            self.localtion_history = []
            self.near_no_enemy = 0

            self.game_max_time = 60 * 10

        self.first_time = True
        self.episode_reward = []
        self.discount_rate = 0.995
        self.gae_lam = 0.95

        # 卡兵开始位置
        if self.team_id == TEAM_DIRE:
            self.start_position = CMsgBotWorldState.Vector(x=3700, y=3100, z=256)
            self.stop_position = CMsgBotWorldState.Vector(x=-240, y=-110, z=256)

            self.self_tower_position = CMsgBotWorldState.Vector(x=524, y=652, z=256)
            self.enemy_tower_position = CMsgBotWorldState.Vector(x=-1544, y=-1408, z=256)
        else:
            self.start_position = CMsgBotWorldState.Vector(x=-4450, y=-3800, z=256)
            self.stop_position = CMsgBotWorldState.Vector(x=-600, y=-480, z=256)

            self.self_tower_position = CMsgBotWorldState.Vector(x=-1544, y=-1408, z=256)
            self.enemy_tower_position = CMsgBotWorldState.Vector(x=524, y=652, z=256)

        # 真眼位置
        self.sentry_ward = [CMsgBotWorldState.Vector(x=-65, y=-650, z=256)]
        if self.team_id == TEAM_DIRE:
            self.init_position = CMsgBotWorldState.Vector(x=6479, y=6109, z=256)
            self.init_ward = [[CMsgBotWorldState.Vector(x=50, y=-200, z=256), CMsgBotWorldState.Vector(x=50, y=-200, z=256)]]
            self.ward_position = CMsgBotWorldState.Vector(x=50, y=-200, z=256)
        else:
            self.init_position = CMsgBotWorldState.Vector(x=-6871, y=-6427, z=256)
            self.init_ward = [[
                CMsgBotWorldState.Vector(x=-1055, y=-550, z=256),
                CMsgBotWorldState.Vector(x=-1055, y=-550, z=256)
            ]]
            self.ward_position = CMsgBotWorldState.Vector(x=-1055, y=-550, z=256)
        # 2个位置随机选择一个
        self.init_ward = self.init_ward[0]
        self.sentry_ward = self.sentry_ward[0]

        self.use_script = False
        self.bonus_reward = 0
        self.ucf_row_info = {}

        self.next_check_poll_time = time.time()

        # 补刀成绩
        self.performance_rewards = {-2: {'last_hits': 0, 'denies': 0}, -1: {'last_hits': 0, 'denies': 0}}

        self.model_time = 0
        self.last_send_model_time = 0
        self.history_action_info = []
        self.model_get_time_list = []

        self.terminal_reward = -5

        self.cur_lstm_state = np.zeros(shape=(1, 500 * 2), dtype=np.float32)
        self.latest_instances = []
        self.pre_action = 2
        self.pre_move_direction = 0
        self.pre_action_time = -1
        self.pre_attack_target_distance = -1

        self.skip_step_count = 0
        self.buy_init_items = True # 购买出门装
        self.first_death = False
        self.latest_cast_ability_time = -1
        self.latest_cast_count = -1

        # 敌方影魔的影压等级和冷却时间估算
        # 技能没激活时level为0，cooldown999
        self.enemy_hero_ability_pred = {
            5059: {
                'level': 0,
                'cooldown': 999
            },
            5060: {
                'level': 0,
                'cooldown': 999
            },
            5061: {
                'level': 0,
                'cooldown': 999
            }
        }

        self.check_buy_flask_time = 0
        self.check_buy_ward_time = 90 # 购买假眼时间
        self.check_buy_ward_sentry_time = 90 # 购买真眼时间

        self.plant_tree_cd = -90
        self.tango_cd = -90
        self.swap_item_time = -90

        self.pre_tango_flag = False

        self.first_kill_check = True
        self.is_respown = False
        self.creep_block_max_time = 30

        self.state_machine = StateMachine(self.init_position, self.start_position, self.stop_position, self.game_max_time,
                                          self.creep_block_max_time, self.team_id, self.self_tower_position,
                                          self.enemy_tower_position, self.running_mode)

        self.sf = Nevermore(self.running_mode)

        self.ward_cd = -1
        self.sentry_ward_cd = -1

        self.swap_item_cd = {0: 0, 1: 0, 2: 0, 3: 0, 4: 0, 5: 0}

        self.seed_sentry_ward = random.randint(0, 1)
        if self.running_mode == "self_eval" or self.running_mode == "ai_vs_ai":
            self.seed_sentry_ward = 0

        self.cur_attack_handle = -1
        self.pre_death_health_reward = 0
        self.creep_lane_dis = {}

        self.sample_count = 0
        self.predict_timeout = 0

    # 将模型接收到的 action 转成 pb
    def choose_action(self, action_type, params):
        if action_type == ACTION_NAME_TO_INDEX["MOVE"]:
            r = params[0] * 360 / PPOModel.move_x_partition_count
            if r is None:
                r = 0
            x = math.cos(math.radians(r)) * 150
            y = math.sin(math.radians(r)) * 150

            x = self.self_position.x + x
            y = self.self_position.y + y
            z = self.self_position.z

            action_pb = self.basic_action.move(x, y, z)
        elif action_type == ACTION_NAME_TO_INDEX["STOP"]:
            action_pb = self.basic_action.stop()
        elif action_type == ACTION_NAME_TO_INDEX["ATTACK_ENEMY"]:
            handle = self.ucf_row_info[params[0]][0]
            action_pb = self.basic_action.enemy({'handle': handle})
        elif action_type == ACTION_NAME_TO_INDEX["ATTACK_SELF"]:
            handle = self.ucf_row_info[params[0]][0]
            action_pb = self.basic_action.deny({'handle': handle})
        elif action_type == ACTION_NAME_TO_INDEX["ABILITY_Q"]:
            action_pb = self.basic_action.use_ability(0)
        elif action_type == ACTION_NAME_TO_INDEX["ABILITY_W"]:
            action_pb = self.basic_action.use_ability(1)
        elif action_type == ACTION_NAME_TO_INDEX["ABILITY_E"]:
            action_pb = self.basic_action.use_ability(2)
        elif action_type == ACTION_NAME_TO_INDEX["ABILITY_R"]:
            action_pb = self.basic_action.use_ability(5)
        elif action_type == ACTION_NAME_TO_INDEX["ATTACK_HERO"]:
            handle = self.enemy_hero.handle
            action_pb = self.basic_action.enemy({'handle': handle})
        elif action_type == ACTION_NAME_TO_INDEX["ATTACK_TOWER"]:
            handle = self.enemy_tower.handle
            action_pb = self.basic_action.enemy({'handle': handle})
        elif action_type == ACTION_NAME_TO_INDEX["FLASK"]:
            action_pb = self.basic_action.drink_flask()
        elif action_type == ACTION_NAME_TO_INDEX["CLARITY"]:
            action_pb = self.basic_action.drink_clarity()
        elif action_type == ACTION_NAME_TO_INDEX["MANGO"]:
            action_pb = self.basic_action.eat_mango()
        elif action_type == ACTION_NAME_TO_INDEX["PLANT"]:
            action_pb = self.basic_action.plant()
        elif action_type == ACTION_NAME_TO_INDEX["EAT_TREE"]:
            self.state_machine.enter_state('EAT_TREE')
            action_pb = self.go_to_tower_and_eat_tree()

        elif action_type == ACTION_NAME_TO_INDEX["MAGIC_STICK"]:
            action_pb = self.basic_action.use_magic_stick()
        elif action_type == ACTION_NAME_TO_INDEX["FAERIE_FIRE"]:
            action_pb = self.basic_action.eat_faerie_fire()
        elif action_type == ACTION_NAME_TO_INDEX["ATTACK_WARD"]:
            action_pb = self.basic_action.attack({'handle': params})

        elif action_type == ACTION_NAME_TO_INDEX["USE_WARD"]:
            self.state_machine.enter_state('USE_WARD')
            action_pb = self.use_ward()

        elif action_type == ACTION_NAME_TO_INDEX["USE_SENTRY_WARD"]:
            self.state_machine.enter_state('USE_SENTRY_WARD')
            action_pb = self.use_sentry_ward()

        actions_pb = self.basic_action.get_action_message(action_pb, self.cur_dota_time, self.sync_key)
        return actions_pb

    def choose_item(self, obs):
        #update item swap cd:
        for slot, cd in self.swap_item_cd.items():
            if cd > 0:
                self.swap_item_cd[slot] = max(cd - 0.2, 0)

        used_gold = 0
        self_info = self.current_player()

        # 购买出门装
        if self.buy_init_items:
            for item in self.sf.init_item:
                self.ex_actions.append(self.basic_action.buy(item))
            self.buy_init_items = False
            self.check_buy_flask_time = self.cur_dota_time + 5

        if self.cur_dota_time > self.check_buy_flask_time:
            self.check_buy_flask_time = self.cur_dota_time + 1

            # 买药逻辑
            all_flask_items = self.basic_action.where_items(ITEM_FLASK_ID)
            all_tango_items = self.basic_action.where_items(ITEM_TANGO_ID)
            loss_health = self.current_player().health_max - self.current_player().health
            remain_mana = self.current_player().mana
            loss_mana = self.current_player().mana_max - self.current_player().mana
            healing_health = 0
            healing_mana = 0
            for i, mod in enumerate(self.current_player().modifiers):
                if mod.name == "modifier_flask_healing":
                    healing_health += mod.remaining_duration * 40

                if mod.name == "modifier_tango_heal":
                    healing_health += mod.remaining_duration * 7

                if mod.name == "modifier_clarity_potion":
                    healing_mana = mod.remaining_duration * 4.5

            #if healing_health > 0:
            #    logger.info("healing_health %d" %healing_health)
            #if healing_mana > 0:
            #    logger.info("healing_mana %d" %healing_mana)

            if self_info.level < 5:
                max_flask_count = 2
                min_flask_count = 1
                max_mango_count = 3
                min_mana = 360
            elif self_info.level < 8:
                max_flask_count = 3
                min_flask_count = 1
                max_mango_count = 4
                min_mana = 450
            else:
                max_flask_count = 3
                min_flask_count = 1
                max_mango_count = 4
                min_mana = 450

            if self.current_player().mana_max < min_mana:
                min_mana = self.current_player().mana_max

            no_need_to_buy_flask = True
            if (all_flask_items < min_flask_count and all_tango_items == 0) or \
                    (all_flask_items < max_flask_count and loss_health > (400 * all_flask_items + 112 * all_tango_items + healing_health - 100)):
                no_need_to_buy_flask = False
                if self.basic_action.could_buy('item_flask', used_gold=used_gold):
                    self.ex_actions.append(self.basic_action.buy('item_flask'))
                    used_gold += 110

            no_need_to_buy_mana = True
            all_mango_items = self.basic_action.where_items(216)
            if (loss_health < 100 or no_need_to_buy_flask is True
               ) and min_mana > (all_mango_items * 110 + remain_mana) and all_mango_items < max_mango_count:
                no_need_to_buy_mana = False
                if self.basic_action.could_buy('item_enchanted_mango', used_gold=used_gold):
                    self.ex_actions.append(self.basic_action.buy('item_enchanted_mango'))
                    used_gold += 70

            # 购买装备逻辑
            if not self.buy_init_items and self.sf.can_buy_new_item(
                    used_gold, self.current_player()) and no_need_to_buy_mana is True and no_need_to_buy_flask is True:
                items = self.sf.need_item()

                for item in items['items']:
                    self.ex_actions.append(self.basic_action.buy(item))

        sentry_ward_slot = self.basic_action.find_item_slot_by_id(ITEM_WARD_SENTRY_ID)
        ward_slot = self.basic_action.find_item_slot_by_id(ITEM_WARD_ID)
        dispenser_ward_slot = self.basic_action.find_item_slot_by_id(ITEM_WARD_DISPENSER_ID)

        if self.enemy_hero is not None:
            enemy_player_count = 0
            for m in self.enemy_hero.modifiers:
                if m.name == 'modifier_nevermore_necromastery':
                    enemy_player_count = m.stack_count
                    break

            self_player_count = 0
            for m in self.player_history_info[-1].modifiers:
                if m.name == 'modifier_nevermore_necromastery':
                    self_player_count = m.stack_count
                    break

            diff = self_player_count - enemy_player_count
            #if self.running_mode == "local_test_self":
            #    logger.info("state diff, self %d, enemy %d" %(self_player_count, enemy_player_count))
        else:
            diff = 0
            self_player_count = 0

        if self.cur_dota_time > self.check_buy_ward_time and self.has_ward is False \
                and dispenser_ward_slot is None and ward_slot is None:
            self.check_buy_ward_time = self.cur_dota_time + 150 + random.randint(0, 100)
            self.ex_actions.append(self.basic_action.buy('item_ward_observer'))

        if self.has_sentry_ward is False and self.cur_dota_time > self.check_buy_ward_sentry_time and \
                self.basic_action.could_buy('item_ward_sentry', used_gold=used_gold) and sentry_ward_slot is None \
                and dispenser_ward_slot is None and self.seed_sentry_ward == 1:
            if diff > 8 or (diff == 0 and self_player_count == 36):
                self.check_buy_ward_sentry_time = self.cur_dota_time + 250 + random.randint(0, 100)
                self.ex_actions.append(self.basic_action.buy('item_ward_sentry'))

        # 送装备
        if self.basic_action.stash_item_need_take():
            if (self.self_courier_state == 'delivering' and self.courier_to_base_distance < 600) or \
                    self.self_courier_state == 'idle':
                self.ex_actions.append(self.basic_action.take_item_by_courier())
                self.self_courier_state = 'delivering'

        # 如果装备太多，用树枝种下一颗快乐的小树
        equipment_counter = 0
        for i in self_info.items:
            if i.slot <= 8:
                equipment_counter += 1

        # if equipment_counter > 6:
        #     slot = self.find_item_slot_by_id(ITEM_BRANCH_ID)
        #     if slot is not None and slot < 6:
        #         self.ex_actions.append(self.plant(slot=slot))

        # 背包里面的装备放进物品栏
        if len(self_info.items) > 0:
            blank_slots = [0, 1, 2, 3, 4, 5]
            current_items = {}
            storage_slots = [6, 7, 8]
            storage_items = {}
            for i in self_info.items:
                if i.slot in blank_slots:
                    current_items[i.ability_id] = i.slot
                    blank_slots.remove(i.slot)
                elif i.slot in storage_slots:
                    storage_items[i.ability_id] = i.slot

            #if self.running_mode == "local_test_self":
            #    cur_ids = " ".join([str(key) for key, value in current_items.items()])
            #    logger.info("blank slots %d, current item id %s" %(len(blank_slots), cur_ids))

            #need swap items
            if len(storage_items) > 0 and obs.dota_time > self.swap_item_time:
                # if there is no empty slot, swap consume item with circlet
                swap_cd = 6.2
                if len(blank_slots) == 0:
                    if ITEM_MANGO_ID in storage_items:
                        if ITEM_CIRCLET in current_items:
                            self.ex_actions.append(
                                self.basic_action.swap_item(current_items[ITEM_CIRCLET], storage_items[ITEM_MANGO_ID]))
                            self.swap_item_cd[current_items[ITEM_CIRCLET]] = swap_cd
                        elif ITEM_WRAITH in current_items:
                            self.ex_actions.append(
                                self.basic_action.swap_item(current_items[ITEM_WRAITH], storage_items[ITEM_MANGO_ID]))
                            self.swap_item_cd[current_items[ITEM_WRAITH]] = swap_cd
                        elif ITEM_BRACER in current_items:
                            self.ex_actions.append(
                                self.basic_action.swap_item(current_items[ITEM_BRACER], storage_items[ITEM_MANGO_ID]))
                            self.swap_item_cd[current_items[ITEM_BRACER]] = swap_cd
                        elif ITEM_FAERIE_FIRE_ID in current_items:
                            self.ex_actions.append(
                                self.basic_action.swap_item(current_items[ITEM_FAERIE_FIRE_ID], storage_items[ITEM_MANGO_ID]))
                            self.swap_item_cd[current_items[ITEM_FAERIE_FIRE_ID]] = swap_cd

                    elif ITEM_FLASK_ID in storage_items:
                        if ITEM_CIRCLET in current_items:
                            self.ex_actions.append(
                                self.basic_action.swap_item(current_items[ITEM_CIRCLET], storage_items[ITEM_FLASK_ID]))
                            self.swap_item_cd[current_items[ITEM_CIRCLET]] = swap_cd
                        elif ITEM_WRAITH in current_items:
                            self.ex_actions.append(
                                self.basic_action.swap_item(current_items[ITEM_WRAITH], storage_items[ITEM_FLASK_ID]))
                            self.swap_item_cd[current_items[ITEM_WRAITH]] = swap_cd
                        elif ITEM_BRACER in current_items:
                            self.ex_actions.append(
                                self.basic_action.swap_item(current_items[ITEM_BRACER], storage_items[ITEM_FLASK_ID]))
                            self.swap_item_cd[current_items[ITEM_BRACER]] = swap_cd
                        elif ITEM_FAERIE_FIRE_ID in current_items:
                            self.ex_actions.append(
                                self.basic_action.swap_item(current_items[ITEM_FAERIE_FIRE_ID], storage_items[ITEM_FLASK_ID]))
                            self.swap_item_cd[current_items[ITEM_FAERIE_FIRE_ID]] = swap_cd

                    elif ITEM_LESSER_CRIT in storage_items:
                        if ITEM_FAERIE_FIRE_ID in current_items:
                            self.ex_actions.append(
                                self.basic_action.swap_item(current_items[ITEM_FAERIE_FIRE_ID],
                                                            storage_items[ITEM_LESSER_CRIT]))
                    elif ITEM_BOOTS in storage_items:
                        if ITEM_FAERIE_FIRE_ID in current_items:
                            self.ex_actions.append(
                                self.basic_action.swap_item(current_items[ITEM_FAERIE_FIRE_ID], storage_items[ITEM_BOOTS]))

                    elif ITEM_WARD_ID in storage_items:
                        self.ex_actions.append(self.basic_action.swap_item(0, storage_items[ITEM_WARD_ID]))
                        self.swap_item_cd[0] = swap_cd

                    elif ITEM_WARD_DISPENSER_ID in storage_items:
                        self.ex_actions.append(self.basic_action.swap_item(0, storage_items[ITEM_WARD_DISPENSER_ID]))
                        self.swap_item_cd[0] = swap_cd
                    elif ITEM_WARD_SENTRY_ID in storage_items:
                        self.ex_actions.append(self.basic_action.swap_item(1, storage_items[ITEM_WARD_SENTRY_ID]))
                        self.swap_item_cd[1] = swap_cd

                    elif ITEM_MAGIC_STICK_ID in storage_items:
                        if ITEM_FAERIE_FIRE_ID in current_items:
                            self.ex_actions.append(
                                self.basic_action.swap_item(current_items[ITEM_FAERIE_FIRE_ID],
                                                            storage_items[ITEM_MAGIC_STICK_ID]))
                            self.swap_item_cd[current_items[ITEM_FAERIE_FIRE_ID]] = swap_cd

                    elif ITEM_BRACER in storage_items:
                        if ITEM_FAERIE_FIRE_ID in current_items:
                            self.ex_actions.append(
                                self.basic_action.swap_item(current_items[ITEM_FAERIE_FIRE_ID], storage_items[ITEM_BRACER]))
                            self.swap_item_cd[current_items[ITEM_FAERIE_FIRE_ID]] = swap_cd

                else:
                    self.ex_actions.append(self.basic_action.swap_item(blank_slots[0], list(storage_items.values())[0]))
                    self.swap_item_cd[blank_slots[0]] = swap_cd

                self.swap_item_time = obs.dota_time + 1

    # 实现 bot 的 generate_action 返回动作
    def generate_action(self, obs_dict, enemy_observation):
        return self._act(obs_dict, enemy_observation)

    # 聊天，嘲讽
    def nlp(self, obs_dict):
        if len(self.rule_msg) > 0 and self.team_id == 3:
            msg = self.rule_msg.pop(0)
            self.ex_actions.append(self.basic_action.chat(msg))

        if len(self.wel_msg) > 0 and obs_dict.dota_time > 0:
            msg = self.wel_msg.pop(0)
            self.ex_actions.append(self.basic_action.chat(msg))

    def skip_histroy(self):
        self.player_history_info = self.player_history_info[:-1]
        self.player_events_history_info = self.player_events_history_info[:-1]
        self.self_tower1 = self.self_tower1[:-1]

        self.enemy_history_info = self.enemy_history_info[:-1]
        self.enemy_events_history_info = self.enemy_events_history_info[:-1]
        self.enemy_tower1 = self.enemy_tower1[:-1]

        self.check_dota_time_dic['self'] = self.check_dota_time_dic['self'][:-1]
        self.check_dota_time_dic['enemy'] = self.check_dota_time_dic['enemy'][:-1]

    def get_instance(self, obs_dict, is_creep_block=False):
        enemy_action, enemy_anim, self_anim = self.get_enemy_action(obs_dict)
        embedding_dict = {
            "self_pre_action": self.pre_action,
            "self_pre_move_direction": self.pre_move_direction,
            "self_anim": self_anim,
            "enemy_anim": enemy_anim,
            "enemy_action": enemy_action
        }

        if is_creep_block is True:
            mask = np.ones(2, dtype=np.int)
            gf, ucf, ucategory, self.ucf_row_info, units_mask = self.cb_feature.trans_feature(
                obs_dict, self.player_events_history_info[-1], self.latest_avg_delay, self.running_mode, self.creep_lane_dis,
                self.self_tower_position, self.stop_position)

        else:
            gf, ucf, ucategory, mask, self.ucf_row_info, units_mask = self.feature.trans_feature(
                obs_dict, self.player_events_history_info[-1], self.latest_avg_delay, self.running_mode,
                All_ACTION_TYPE[self.pre_action], self.pre_action_time, self.enemy_hero_ability_pred,
                self.pre_attack_target_distance, self.tango_cd, self.self_courier_state, self.swap_item_cd,
                self.cur_attack_handle)

        dota_map_f = self.dota_map.nearby(self.self_position.x, self.self_position.y, size=8)
        instance = Instance(
            obs_dict.dota_time,
            gf,
            ucf,
            ucategory,
            mask,
            model_time=self.model_time,
            units_mask=units_mask,
            lstm_state=self.cur_lstm_state,
            embedding_dict=embedding_dict,
            dota_map=dota_map_f)
        return instance, mask

    # enemy_obs 用于计算reward，不用作feature
    def _act(self, obs_dict, enemy_observation):

        if self.test_mode is not True and (enemy_observation is None and self.cur_dota_time > 60) or \
                (enemy_observation is not None and self.cur_dota_time - enemy_observation.dota_time > 60):
            send_message = {'msg_type': 'error'}
            send_message["dota_time"] = self.cur_dota_time
            send_message["ip"] = self.local_ip
            send_message["error_msg"] = "opponent not start"
            send_message["error_type"] = "msg_delay"
            p = pickle.dumps([send_message])
            self._net.logger_sender.send(p)
            time.sleep(5)
            self.alt_and_f4(force_terminal=True)

        self.ex_actions = []
        self.pics = []

        # TODO runner 给 model

        # 聊天，技能加点等
        actions_pb = self.basic_action.get_action_message(self.basic_action.do_nothing(), self.cur_dota_time, self.sync_key)

        # update info
        self.update_info(obs_dict, enemy_observation)
        self.feature.update_latest_health_info(obs_dict)

        self.cal_enemy_ability(obs_dict)

        # 状态机
        self.creep_lane_dis = self.state_machine.dispatcher(obs_dict, self.current_player())

        # 技能加点
        self_info = self.current_player()
        if self_info.ability_points > 0:
            ability_name = self.sf.ability_route(self.ability_info)
            if ability_name:
                self.ex_actions.append(self.basic_action.train_ability(ability_name))

        # 补刀成绩
        minutes = int(obs_dict.dota_time / 60)
        if minutes not in self.performance_rewards:
            self.performance_rewards[minutes] = {}
        if len(self.player_history_info) > 1:
            self.performance_rewards[minutes]['last_hits'] = self.current_player().last_hits
            self.performance_rewards[minutes]['denies'] = self.current_player().denies
        else:
            self.performance_rewards[minutes]['last_hits'] = 0
            self.performance_rewards[minutes]['denies'] = 0

        self.nlp(obs_dict)

        # buy, swap and use items
        self.choose_item(obs_dict)

        # terminate game
        if self.state_machine.is_action_state('MAX_DOTA_TIME'):
            time.sleep(5)
            self.alt_and_f4(force_terminal=True)

        # dotatime 小于 0 时为准备时间 水走向或者 在泉中路
        if self.state_machine.is_action_state('PREPARE'):
            slot = self.basic_action.find_item_slot_by_id(ITEM_WARD_ID, in_equipment=True)
            if slot is None:
                slot = self.basic_action.find_item_slot_by_id(ITEM_WARD_DISPENSER_ID, in_equipment=True)
            action = self.go_to_start_point()
            if slot is not None:
                g = self.init_ward[0]
                if cal_distance(g, self.self_position) > 300:
                    action = self.basic_action.move(g.x, g.y, g.z)
                else:
                    action = self.basic_action.use_item_on_postion(slot, self.init_ward[1])
            # go back
            actions_pb = self.basic_action.get_action_message(action, self.cur_dota_time, self.sync_key)
        elif self.state_machine.is_action_state('FOLLOW'):
            actions_pb_raw = self.basic_action.move(self.self_tower_position.x, self.self_tower_position.y,
                                                    self.self_tower_position.z)
            actions_pb = self.basic_action.get_action_message(actions_pb_raw, self.cur_dota_time, self.sync_key)

        elif self.state_machine.is_action_state('CREEP_BLOCK'): # 堵兵模型
            # logger.info('mode creep block step:{0},team:{1}'.format(self.step, self.team_id))
            instance, _ = self.get_instance(obs_dict, is_creep_block=True)

            if self.test_mode:
                action_type, action_params, state_value, sub_prob, sub_prob_distribution = self.cbmodel.predict(
                    instance, self.running_mode)
                logger.info([self.team_id, self.step, action_params, sub_prob, sub_prob_distribution])

            actions_pb = self.choose_action(action_type, action_params)
            # actions_pb = self.basic_action.get_action_message(self.go_to_tower(), self.cur_dota_time, self.sync_key)
        elif self.state_machine.is_action_state('DEAD'): # 死亡
            self.skip_histroy()
            self.history_action_info = []
            # first death, end reward -1
            if self.first_death is False:
                self.pre_death_health_reward = -1 * get_health_reward(
                    float(self.player_history_info[-1].health) / self.player_history_info[-1].health_max)
                if self.running_mode in ["local_test_self", "local_test_opponent"]:
                    logger.info("first death !!! %s, pre health reward %f" % (self.running_mode, self.pre_death_health_reward))

                self.first_death = True
                self.is_respown = True
                #logger.info("%s self respawntime %d" % (self.running_mode, respawn_time(self.current_player()["level"])))

        elif self.state_machine.is_action_state('USE_TP'): # 使用 TP
            self.skip_histroy()
            # 死亡后使用 tp 出来节省时间
            if self.first_death and self.current_player().is_alive is True:
                if not self.is_tping():
                    self.ex_actions.append(self.basic_action.use_tp())
                actions_pb = self.basic_action.get_action_message(self.basic_action.do_nothing(), self.cur_dota_time,
                                                                  self.sync_key)
        elif self.state_machine.is_action_state('EAT_TREE'): # 吃树
            self.skip_histroy()
            actions_pb = self.basic_action.get_action_message(self.go_to_tower_and_eat_tree(), self.cur_dota_time,
                                                              self.sync_key)

        elif self.state_machine.is_action_state('USE_SENTRY_WARD'): # 查眼
            self.skip_histroy()
            actions_pb = self.basic_action.get_action_message(self.use_sentry_ward(), self.cur_dota_time, self.sync_key)
            if self.running_mode == "local_test_self":
                logger.info("in USE_SENTRY_WARD state")

        elif self.state_machine.is_action_state('USE_WARD'): # 查眼
            self.skip_histroy()
            actions_pb = self.basic_action.get_action_message(self.use_ward(), self.cur_dota_time, self.sync_key)
            if self.running_mode == "local_test_self":
                logger.info("in USE_WARD state")
        elif self.state_machine.is_action_state('GO_TO_TOWER'): # 跑去塔
            self.skip_histroy()
            actions_pb = self.basic_action.get_action_message(self.go_to_tower(), self.cur_dota_time, self.sync_key)
        elif self.state_machine.is_action_state('MODEL'): # 对战
            if self.running_mode == "local_test_self":
                logger.info("")

            has_update = self.feature.update_attack_point(self.player_history_info[-1], self.cur_dota_time, self.running_mode)
            if self.skip_step_count > 0:
                # if self.running_mode == "local_test_self":
                #     logger.info(
                #         "current time %f, anim_activity: %d, anim_cycle: %f, attack_anim_point: %f, action_type: %d" % (
                #             self.cur_dota_time, self.player_history_info[-1].anim_activity,
                #             self.player_history_info[-1].anim_cycle,
                #             self.player_history_info[-1].attack_anim_point, self.player_history_info[-1].action_type))

                if has_update == False:
                    # if self.running_mode == "local_test_self":
                    #     logger.info("skip one step!!!")
                    self.skip_step_count = self.skip_step_count - 1
                    self.skip_histroy()
                    extra_actions = self.basic_action.get_extra_message(self.ex_actions)
                    return actions_pb, extra_actions
                else:
                    self.skip_step_count = 0

            instant_reward = 0

            if self.running_mode in ["self_eval"] and len(self.player_history_info) > 12:
                x_arr = []
                y_arr = []
                for h in self.player_history_info:
                    x_arr.append(h.location.x)
                    y_arr.append(h.location.y)

                for u in obs_dict.units:
                    if self.is_enemy(u) and u.is_alive and cal_distance(u.location, self.current_player().location) <= 600:
                        self.near_no_enemy = 0
                        break

                self.near_no_enemy += 1
                if self.near_no_enemy > 15 and np.std(x_arr) < 5 and np.std(y_arr) < 5:
                    self.state_machine.enter_state('GO_TO_TOWER')
                    self.near_no_enemy = 0

            instance, mask = self.get_instance(obs_dict, is_creep_block=False)

            # 本地
            if self.test_mode:
                action_type, action_params, state_value, action_prob, sub_prob, \
                final_prob, sub_prob_distribution, self.cur_lstm_state, raw_action_probs = self.model.predict(instance, self.running_mode)
                if self.running_mode in ["local_test_self", "self_eval", "ai_vs_ai"]:
                    raw_prob_str = ','.join(['%d:%.3f' % (index, i) for index, i in enumerate(raw_action_probs)])

                    logger.info("time:%f, choose action:%s, %d, prob: %f" %
                                (self.cur_dota_time, All_ACTION_TYPE[action_type], action_type, action_prob))
                    logger.info("%s" % raw_prob_str)
                    if All_ACTION_TYPE[action_type] in ["ATTACK_ENEMY", "ATTACK_SELF"]:
                        sub_prob_str = ','.join(['%d:%.3f' % (index, i) for index, i in enumerate(sub_prob_distribution)])
                        logger.info(
                            "pre action %s, attack handle %d, current attack handle %d" %
                            (All_ACTION_TYPE[self.pre_action], self.cur_attack_handle, self.ucf_row_info[action_params[0]][0]))
                        logger.info("%s" % sub_prob_str)

                    elif All_ACTION_TYPE[action_type] == "MOVE":
                        sub_prob_str = ','.join(['%d:%.3f' % (index, i) for index, i in enumerate(sub_prob_distribution)])
                        logger.info("move distribution")
                        logger.info("%s" % sub_prob_str)

            # 反眼
            if action_type == ACTION_NAME_TO_INDEX["ATTACK_WARD"]:
                # 仅取敌人第一个
                for u in obs_dict.units:
                    if u.unit_type == CMsgBotWorldState.UnitType.Value("WARD") and u.team_id != self.team_id and u.is_alive:
                        action_params = u.handle
                        self.enemy_ward_location = u.location
                        break

            actions_pb = self.choose_action(action_type, action_params)
            self.pre_action = action_type
            self.pre_action_time = self.cur_dota_time
            if All_ACTION_TYPE[action_type] == "MOVE":
                self.pre_move_direction = action_params[0] + 1
            else:
                self.pre_move_direction = 0
            if All_ACTION_TYPE[action_type] in ["ATTACK_ENEMY", "ATTACK_SELF"]:
                self.pre_attack_target_distance = self.ucf_row_info[action_params[0]][1]
                self.cur_attack_handle = self.ucf_row_info[action_params[0]][0]
            else:
                self.cur_attack_handle = -1
                if All_ACTION_TYPE[action_type] == "ATTACK_HERO":
                    self.pre_attack_target_distance = cal_distance(self.enemy_hero.location,
                                                                   self.player_history_info[-1].location)
                elif All_ACTION_TYPE[action_type] == "ATTACK_TOWER":
                    self.pre_attack_target_distance = cal_distance(self.enemy_tower.location,
                                                                   self.player_history_info[-1].location)
                elif All_ACTION_TYPE[action_type] == "ATTACK_WARD":
                    self.pre_attack_target_distance = cal_distance(self.enemy_ward_location,
                                                                   self.player_history_info[-1].location)
                else:
                    self.pre_attack_target_distance = -1

            # set skip step to prevent cancelling action execution
            if All_ACTION_TYPE[action_type] == "ABILITY_R":
                self.skip_step_count = 9

            self.step += 1

        # 学技能、聊天、画图
        extra_actions = self.basic_action.get_extra_message(self.ex_actions)
        return actions_pb, extra_actions

    def compute_episode_reward(self, reward_list):
        total_episode_reward = 0
        episode_reward_list = [0 for i in range(len(reward_list))]
        health_episode_reward = 0
        health_episode_reward_list = [0 for i in range(len(reward_list))]
        mana_episode_reward = 0
        mana_episode_reward_list = [0 for i in range(len(reward_list))]
        exp_episode_reward = 0
        exp_episode_reward_list = [0 for i in range(len(reward_list))]
        gold_episode_reward = 0
        gold_episode_reward_list = [0 for i in range(len(reward_list))]
        lasthits_episode_reward = 0
        lasthits_episode_reward_list = [0 for i in range(len(reward_list))]
        deny_episode_reward = 0
        deny_episode_reward_list = [0 for i in range(len(reward_list))]
        tower_episode_reward = 0
        tower_episode_reward_list = [0 for i in range(len(reward_list))]

        for index in reversed(range(len(reward_list))):
            total_episode_reward = reward_list[index]['total_reward'] + self.discount_rate * total_episode_reward
            episode_reward_list[index] = total_episode_reward
            #logger.info("index:%d, episode reward:%f" %(index, total_episode_reward))

            if index < len(reward_list) - 1:
                health_episode_reward = reward_list[index]['health'] + self.discount_rate * health_episode_reward
                health_episode_reward_list[index] = health_episode_reward

                mana_episode_reward = reward_list[index]['mana'] + self.discount_rate * mana_episode_reward
                mana_episode_reward_list[index] = mana_episode_reward

                exp_episode_reward = reward_list[index]['experience'] + self.discount_rate * exp_episode_reward
                exp_episode_reward_list[index] = exp_episode_reward

                gold_episode_reward = reward_list[index]['gold'] + self.discount_rate * gold_episode_reward
                gold_episode_reward_list[index] = gold_episode_reward

                lasthits_episode_reward = reward_list[index]['last_hits'] + self.discount_rate * lasthits_episode_reward
                lasthits_episode_reward_list[index] = lasthits_episode_reward

                deny_episode_reward = reward_list[index]['denies'] + self.discount_rate * deny_episode_reward
                deny_episode_reward_list[index] = deny_episode_reward

                tower_episode_reward = reward_list[index]['denies'] + self.discount_rate * tower_episode_reward
                tower_episode_reward_list[index] = tower_episode_reward

        avg_episode_reward = sum(episode_reward_list) / len(episode_reward_list)
        avg_health_episode_reward = sum(health_episode_reward_list) / len(health_episode_reward_list)
        avg_mana_episode_reward = sum(mana_episode_reward_list) / len(mana_episode_reward_list)
        avg_exp_episode_reward = sum(exp_episode_reward_list) / len(exp_episode_reward_list)
        avg_gold_episode_reward = sum(gold_episode_reward_list) / len(gold_episode_reward_list)
        avg_lasthits_episode_reward = sum(lasthits_episode_reward_list) / len(lasthits_episode_reward_list)
        avg_deny_episode_reward = sum(deny_episode_reward_list) / len(deny_episode_reward_list)
        avg_tower_episode_reward = sum(tower_episode_reward_list) / len(tower_episode_reward_list)

        return {
            "total": avg_episode_reward,
            "health": avg_health_episode_reward,
            "mana": avg_mana_episode_reward,
            "exp": avg_exp_episode_reward,
            "gold": avg_gold_episode_reward,
            "lasthits": avg_lasthits_episode_reward,
            "denies": avg_deny_episode_reward,
            "tower": avg_tower_episode_reward
        }

    # 走向起始点
    def go_to_start_point(self):
        return self.basic_action.move(self.start_position.x, self.start_position.y, self.start_position.z)

    def use_ward(self):
        slot = self.basic_action.find_item_slot_by_id(ITEM_WARD_ID, in_equipment=True)
        if slot is None:
            slot = self.basic_action.find_item_slot_by_id(ITEM_WARD_DISPENSER_ID, in_equipment=True)
        if slot is not None and cal_distance(self.self_position, self.ward_position) < 400:
            if self.cur_dota_time > self.ward_cd:
                action = self.basic_action.use_item_on_postion(slot, self.ward_position)
                self.ward_cd = self.cur_dota_time + 3
            else:
                action = self.basic_action.do_nothing()
        else:
            action = self.basic_action.move(self.ward_position.x, self.ward_position.y, self.ward_position.z, directly=False)
        return action

    def use_sentry_ward(self):
        slot = self.basic_action.find_item_slot_by_id(ITEM_WARD_SENTRY_ID, in_equipment=True)
        if slot is not None and cal_distance(self.self_position, self.sentry_ward) < 400:
            if self.cur_dota_time > self.sentry_ward_cd:
                action = self.basic_action.use_item_on_postion(slot, self.sentry_ward)
                self.sentry_ward_cd = self.cur_dota_time + 3
            else:
                action = self.basic_action.do_nothing()
        else:
            action = self.basic_action.move(self.sentry_ward.x, self.sentry_ward.y, self.sentry_ward.z, directly=False)
        return action

    def go_to_tower(self):
        # 防御塔附近的一个点
        tower_pos = CMsgBotWorldState.Vector(x=460, y=855, z=256)
        if self.team_id == 2:
            tower_pos = CMsgBotWorldState.Vector(x=-1408, y=-1517, z=256)
        action = self.basic_action.move(tower_pos.x, tower_pos.y, tower_pos.z, directly=False)

        return action

    # 走到塔下并吃树
    def go_to_tower_and_eat_tree(self):
        # 防御塔附近的一个点
        tower_pos = CMsgBotWorldState.Vector(x=460, y=855, z=256)
        if self.team_id == 2:
            tower_pos = CMsgBotWorldState.Vector(x=-1408, y=-1517, z=256)

        if cal_distance(self.self_position, tower_pos) > 200:
            action = self.basic_action.move(tower_pos.x, tower_pos.y, tower_pos.z, directly=False)
        else:
            if self.cur_dota_time > self.tango_cd:
                self.tango_cd = self.cur_dota_time + 3
                action = self.basic_action.eat_nearest_tree()
            else:
                action = self.basic_action.do_nothing()

        return action

    def get_enemy_action(self, obs):
        # 获得敌我action和anim activity
        units = obs.units
        action = 0 # 此action是pb原生action type，和 bot 里抽过一层的 action 不一样
        self_anim = enemy_anim = 0

        for u in units:
            if u.unit_type == CMsgBotWorldState.UnitType.Value("HERO"):
                if u.player_id == self.enemy_player_id:
                    if u.action_type:
                        action = u.action_type
                    if u.anim_activity:
                        # 就先认为只有20个，其实不知道多少
                        enemy_anim = np.clip(u.anim_activity - 1500, 0, 19)
                if u.player_id == self.player_id:
                    if u.anim_activity:
                        self_anim = np.clip(u.anim_activity - 1500, 0, 19)

        return action, enemy_anim, self_anim

    def cal_enemy_ability(self, obs):
        #先不考虑增加影压伤害的天赋和减技能cd的天赋，因为都是20级以上的天赋
        #魔抗的意思就是抵御多少百分比的魔法伤害，检测到技能事件或者技能事件则将敌方技能cd重置
        level = 0
        if len(obs.damage_events) > 0:
            self_hero = self.current_player()
            stack = 0
            if len(self_hero.modifiers) > 0:
                for md in self_hero.modifiers:
                    if md.name == 'modifier_nevermore_shadowraze_debuff':
                        #logger.info([self.cur_dota_time,md])
                        stack = max(md.stack_count - 1, 0)
                        break

            for de in obs.damage_events:
                if de.HasField('ability_id'):
                    if de.victim_player_id == self.player_id and de.ability_id in self.enemy_hero_ability_pred.keys():
                        #logger.info(de)
                        before_damage = de.damage / (1 - self.current_player().magic_resist)
                        if self.enemy_hero_ability_pred[de.ability_id]['level'] > 0:
                            level = self.enemy_hero_ability_pred[de.ability_id]['level']
                        else:
                            level = 1
                        stack_damage = self.sf.stack_bonus_damage[level] * stack
                        raw_damage = before_damage - stack_damage
                        for k, v in self.sf.shadowraze_damage.items():
                            if raw_damage >= v:
                                level = k
                        self.enemy_hero_ability_pred[de.ability_id]['cooldown'] = 10

        if len(obs.ability_events) > 0:
            for ae in obs.ability_events:
                if ae.HasField('ability_id'):
                    if ae.player_id == self.enemy_player_id and ae.ability_id in self.enemy_hero_ability_pred.keys():
                        self.enemy_hero_ability_pred[ae.ability_id]['cooldown'] = 10

        if self.enemy_hero_ability_pred[5059][
                'cooldown'] == 999 and self.enemy_hero is not None and self.enemy_hero.level == 1 and self.cur_dota_time > 25:
            has_necromastery_buff = False
            if len(self.enemy_hero.modifiers) > 0:
                for i, mod in enumerate(self.enemy_hero.modifiers):
                    if mod.name == "modifier_nevermore_necromastery":
                        has_necromastery_buff = True
                        break
            if has_necromastery_buff == False:
                level = 1
                if self.running_mode == "local_test_self":
                    logger.info("enemy has ability!!!")

        #只要有技能伤害事件或者技能事件，那么影压等级至少为1，其他没有释放的影压的cd就变成0而非默认的999,空压的时候没有伤害，连位置都是000
        for k, v in self.enemy_hero_ability_pred.items():
            if v['cooldown'] != 999:
                v['cooldown'] = max(v['cooldown'] - 0.2, 0)
            else:
                # 只要有一个影压被释放，其他所有影压也被解锁
                if level > 0:
                    v['cooldown'] = 0
            if v['level'] < level:
                v['level'] = level
