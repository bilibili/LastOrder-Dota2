import tensorflow.compat.v1 as tf

tf.disable_v2_behavior()

from multiprocessing import Process, Queue
import os
import queue
import socket
import time
import psutil
from gym_env.dota_game import logger, TEAM_RADIANT, TEAM_DIRE
from dotaservice.protos.dota_gcmessages_common_bot_script_pb2 import CMsgBotWorldState
from dotaservice.net_console_util import monitor_log, worldstate_listener
from dotaservice.dotautil import cal_distance
from google.protobuf.json_format import MessageToDict
from config.config import read_config

from model.dota_models.ppo import PPOModel
from model.dota_models.creep_block_model import CreepBlockModel
from model.dota_map import DotaMap
from model.basic_action import BasicAction


class BaseAgent:
    running_mode = [
        "self_eval",
    ]

    def __init__(self, dota_game, team_id, player_id, enemy_player_id, mode, self_data_queue, enemy_data_queue):
        if mode in BaseAgent.running_mode:
            self.running_mode = mode
        else:
            raise Exception('unknown running_mode')
        self.self_data_queue = self_data_queue
        self.enemy_data_queue = enemy_data_queue

        self.dota_game = dota_game
        self.player_id = player_id
        self.enemy_player_id = enemy_player_id
        self.last_action_execution_time = -99999
        self.action_num = 0
        self.team_id = team_id
        if self.team_id == TEAM_RADIANT:
            self.enemy_team_id = 3
            self.enemy_tower_name = 'npc_dota_badguys_tower1_mid'
            self.self_tower_name = 'npc_dota_goodguys_tower1_mid'
        else:
            self.enemy_team_id = 2
            self.enemy_tower_name = 'npc_dota_goodguys_tower1_mid'
            self.self_tower_name = 'npc_dota_badguys_tower1_mid'

        self.unit_history_info = {}

        self.ability_info = {}
        self.history_max_size = 5
        self.replay_number = None
        self.buffer = {} # save unit history health and last_attack_time

        # use the docker -h parameter value
        # example cds-sh-ai-game-product-50_{local docker id}
        self.local_ip = socket.gethostname()
        self.bot_console_log_path = os.path.join(dota_game.session_folder, "bots", dota_game.CONSOLE_LOG_FILENAME)

        # action time delay
        self.write_new_action_time = int(round(time.time() * 1000))
        self.read_new_action_time = int(round(time.time() * 1000))

        # 记录 10 次操作时间，最后算个apm
        self.action_time_list = []
        self.action_delay_list = []
        self.action_interval_list = []

        self.lua_step_list = []
        self.lua_last_step_value = 0
        self.lua_dota_time_list = []
        self.lua_last_dota_time_value = 0
        self.lua_real_time_list = []
        self.lua_last_real_time_value = 0

        self.last_taken_step = 0
        self.execution_timeout = 0

        self.gen_time_list = []
        self.get_lua_r_list = []

        self.yaml_conf = read_config("config/ppo.yaml")

        # td n
        self.n = self.yaml_conf["n_steps"]
        self.game_end_status = "-1"

        # debug 画图相关
        self.pics = []

        self.enemy_history_info = []
        self.enemy_events_history_info = []

        self.player_history_info = []
        self.player_events_history_info = []

        self.enemy_tower1 = []
        self.self_tower1 = []

        self.latest_avg_delay = 0
        self.self_courier = None
        self.self_courier_history = []
        self.self_courier_state = 'idle'
        self.courier_to_base_distance = 0

        self.dota_map = DotaMap(team_id=self.team_id, enemy_team_id=self.enemy_team_id)

        self.check_dota_time_dic = {'self': [], 'enemy': []}

        self.obs_interval = 0.2
        self.plant_tree_ids = []

        self.latest_enemy_obs = None
        self.opponent_obs_not_update_time = 0

        self.basic_action = BasicAction(self.player_id)

        self.miss_check_length = 5 * 20
        self.self_obs_miss_list = []
        self.opponent_obs_miss_list = []

        self.next_malloc_check_time = time.time()
        self.pre_snap = None

        if self.running_mode in ["self_eval"]:
            self.cbmodel = CreepBlockModel(is_creep_block=True)
            self.cbmodel.tf_session = tf.Session() # Session1
            param = read_config("config/ppo.yaml")
            self.model = PPOModel(param=param)
            self.model.tf_session = tf.Session() # Session2

        self.gc_time = time.time()

        self.first_wait_obs = True
        self.spec_ob_flag = True

    def init_process(self):
        suf = None
        if self.running_mode == "self_eval":
            logger.debug("Bot 初始化")
            consolelog_q_monitor_pattern = Queue()
            consolelog_q_monitor_result = Queue()
            consolelog_q_monitor_p = Process(
                target=monitor_log,
                args=(self.dota_game.session_folder, self.dota_game.CONSOLE_LOG_FILENAME, consolelog_q_monitor_pattern,
                      consolelog_q_monitor_result))
            consolelog_q_monitor_p.daemon = True
            consolelog_q_monitor_p.start()
            # 直接从 steam 里面启动游戏，console.log 日志会被切割

            try:
                consolelog_q_monitor_pattern.put({"pattern": "Tearing"})
                if self.dota_game.host_mode == "HOST_MODE_GUI_MENU" and self.running_mode == "self_eval":
                    line = consolelog_q_monitor_result.get().strip()
                else:
                    line = consolelog_q_monitor_result.get(timeout=10).strip()
                suf = line.split(" ")[-1][1:-1] # '.5131657928'
                # 关闭老日志
                consolelog_q_monitor_pattern.close()
                consolelog_q_monitor_result.close()
                consolelog_q_monitor_p.terminate()
                consolelog_q_monitor_pattern.join_thread()
                consolelog_q_monitor_result.join_thread()
                logger.debug("进程关闭")
            except queue.Empty:
                logger.debug("没有切割日志")

        if suf is not None:
            console_log = f'console{suf}.log'
        else:
            console_log = 'console.log'

        self.q_monitor_pattern = Queue()
        self.q_monitor_result = Queue()
        monitor_p = Process(
            target=monitor_log,
            args=(self.dota_game.session_folder, console_log, self.q_monitor_pattern, self.q_monitor_result))
        monitor_p.daemon = True
        monitor_p.start()

        # pb_file=self.dota_game.session_folder+datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S %f")+"_pb.json"
        self.q_world_state = Queue()
        world_state_p = Process(
            target=worldstate_listener,
            args=(self.dota_game.PORT_WORLDSTATES[self.team_id], self.team_id, self.q_world_state,
                  self.dota_game.session_folder, self.dota_game.CONSOLE_LOG_FILENAME))
        world_state_p.daemon = True
        world_state_p.start()

    def generate_action(self, obs_dict, enemy_observation):
        action_pb = CMsgBotWorldState.Action()
        action_pb.actionType = CMsgBotWorldState.Action.Type.Value("DOTA_UNIT_ORDER_NONE")
        action_pb.player = self.player_id
        actions_pb = CMsgBotWorldState.Actions(actions=[action_pb], dota_time=self.cur_dota_time, extraData=self.sync_key)
        return actions_pb, -1

    def alt_and_f4(self, force_terminal=False):
        if self.running_mode == "self_eval" or force_terminal:
            self.dota_game.stop_dota_pids()
            child_process = psutil.Process().children(recursive=True)
            try:
                for child in child_process:
                    child.terminate()
                _, alive = psutil.wait_procs(child_process, timeout=3)
                for p in alive:
                    p.kill()
            except Exception as e:
                logger.warn(e)

            time.sleep(3)
            os._exit(0)

    def update_info(self, observation, enemy_observation):
        # update self info
        self.has_sentry_ward = False
        self.has_ward = False
        self.enemy_hero = None
        for u in observation.units:
            if u.unit_type == CMsgBotWorldState.UnitType.Value("WARD") and u.is_alive and u.team_id == self.team_id:
                if u.name == "npc_dota_observer_wards":
                    self.has_ward = True
                elif u.name == "npc_dota_sentry_wards":
                    self.has_sentry_ward = True

            if self.is_myself(u):
                if self.is_hero(u):
                    self.self_position = u.location
                    self.facing = u.facing
                    self.player_history_info.append(u)
                    if len(self.player_history_info) > self.history_max_size:
                        self.player_history_info.pop(0)

                    self.check_dota_time_dic['self'].append(observation.dota_time)
                    if len(self.check_dota_time_dic['self']) > self.history_max_size:
                        self.check_dota_time_dic['self'].pop(0)

                elif self.is_courier(u):
                    self.self_courier = u
                    self.self_courier_history.append(u)
                    if len(self.self_courier_history) > self.history_max_size:
                        self.self_courier_history.pop(0)
                    if not u.is_alive:
                        self.self_courier_state = 'dead'
                    elif len(self.self_courier_history) >= 2 and self.self_courier_history[-2].is_alive is False:
                        self.self_courier_state = 'idle'
                    elif self.self_courier_state == 'dead':
                        self.self_courier_state = 'idle'

                    cal = 0
                    # 信使距离泉水
                    if self.team_id == 2:
                        location = CMsgBotWorldState.Vector(x=-7415, y=-6488, z=512.0)
                        cal = cal_distance(location, u.location)
                    elif self.team_id == 3:
                        location = CMsgBotWorldState.Vector(x=7515, y=6211, z=512.0)
                        cal = cal_distance(location, u.location)
                    self.courier_to_base_distance = cal

                    # 信使身上无装备
                    if len(self.self_courier.items) == 0:
                        if self.courier_to_base_distance <= 800:
                            self.self_courier_state = 'idle'
                        else:
                            self.self_courier_state = 'back'

            if u.name == self.self_tower_name:
                #logger.info("self_tower damage %d" %u.attack_damage)

                self.self_tower1.append(u)
                if len(self.self_tower1) > self.history_max_size:
                    self.self_tower1.pop(0)

            if u.name == self.enemy_tower_name:
                self.enemy_tower = u
            if self.is_enemy(u) and self.is_hero(u):
                self.enemy_hero = u

        # KDA info
        for u in observation.players:
            if u.player_id == self.player_id:
                self.player_events_history_info.append(u)
                if len(self.player_events_history_info) > self.history_max_size:
                    self.player_events_history_info.pop(0)

        for x in self.player_history_info[-1].abilities:
            if x.ability_id in [5059, 5060, 5061, 5062, 5063, 5064, 5996, 5949, 6875, 6670, 6912, 5906, 6141, 6070, 6445, 6119]:
                self.ability_info[x.ability_id] = x

        # update dota map unit position
        self.plant_tree_ids = self.dota_map.update(observation)
        self.basic_action.update(self.player_history_info, self.self_courier_history)

        # update enemy info
        if self.running_mode != "self_eval" and enemy_observation is not None:
            for u in enemy_observation.units:
                if u.is_illusion is True:
                    continue

                if self.is_enemy(u) and u.player_id == self.enemy_player_id and self.is_hero(u):
                    self.enemy_history_info.append(u)
                    if len(self.enemy_history_info) > self.history_max_size:
                        self.enemy_history_info.pop(0)
                    self.check_dota_time_dic['enemy'].append(enemy_observation.dota_time)
                    if len(self.check_dota_time_dic['enemy']) > self.history_max_size:
                        self.check_dota_time_dic['enemy'].pop(0)

                if u.name == self.enemy_tower_name:
                    self.enemy_tower1.append(u)
                    if len(self.enemy_tower1) > self.history_max_size:
                        self.enemy_tower1.pop(0)
            # KDA info
            for u in enemy_observation.players:
                if u.player_id == self.enemy_player_id:
                    self.enemy_events_history_info.append(u)
                    if len(self.enemy_events_history_info) > self.history_max_size:
                        self.enemy_events_history_info.pop(0)

    def append_info(self, target_list, target_value, cur_value):
        target_list.append(cur_value - target_value)
        if len(target_list) > self.n:
            target_list.pop(0)

    def end_game(self, force_terminal=False):
        #if game not start
        if len(self.player_history_info) == 0:
            self.alt_and_f4(force_terminal)
            return

        pattern = "good guys win = "
        abs_glob = os.path.join(self.dota_game.session_folder, "bots", self.dota_game.CONSOLE_LOG_FILENAME)
        is_lose = False
        normal_end = False
        with open(abs_glob) as f:
            for line in f.readlines():
                if line.find(pattern) != -1:
                    line_fields = line.split(" ")
                    if str.strip(line_fields[-1]) == '0':
                        if self.team_id == TEAM_DIRE:
                            is_lose = False
                        else:
                            is_lose = True
                    else:
                        if self.team_id == TEAM_DIRE:
                            is_lose = True
                        else:
                            is_lose = False
                    logger.info("game end %s" % line)
                    normal_end = True
                    break

        time.sleep(5)
        self.alt_and_f4(force_terminal)

    def run(self):
        self.init_process()

        if self.running_mode in ["self_eval"]:
            # with self.cbmodel.tf_session.as_default():
            # tf.global_variables_initializer().run()
            with self.model.tf_session.as_default():
                tf.global_variables_initializer().run()
                with open("trained_model/redis_model_f", 'rb') as m:
                    model_str = m.read()
                    self.model.deserializing(model_str)
                print("load redis model !")

            with self.cbmodel.tf_session.as_default():
                tf.global_variables_initializer().run()
                with open("trained_model/redis_creep_block_model_f", "rb") as f:
                    m_f = f.read()
                    self.cbmodel.deserializing(m_f)
                    logger.info("reload creep_block_model file")

        while True:
            # wait for observation after lua executing the latest action
            while True:
                try:
                    t = 20
                    if self.first_wait_obs and self.running_mode == "self_eval":
                        t = 300
                        self.first_wait_obs = False
                    observation = self.q_world_state.get(timeout=t)
                    # print(observation)

                    if self.running_mode != "self_eval":
                        self.self_data_queue.put(observation)

                    if observation.dota_time < -80:
                        if self.running_mode == "local_test_self":
                            logger.info("%s game not start" % self.running_mode)
                    else:
                        if observation.dota_time > 15:
                            if not self.q_world_state.empty():
                                self.self_obs_miss_list.append(1)
                                self.execution_timeout += 1
                            else:
                                self.self_obs_miss_list.append(0)
                            if len(self.self_obs_miss_list) > self.miss_check_length:
                                self.self_obs_miss_list = self.self_obs_miss_list[1:]
                            if len(self.self_obs_miss_list) == self.miss_check_length and sum(self.self_obs_miss_list) > 25:
                                send_message = {'msg_type': 'error'}
                                send_message["dota_time"] = self.cur_dota_time
                                send_message["ip"] = self.local_ip
                                send_message["error_msg"] = "ip:%s, %s, self_obs_miss too many %d" % (
                                    self.local_ip, self.running_mode, sum(self.self_obs_miss_list))
                                # send_log_date(send_message)
                                self.end_game(force_terminal=True)
                                return

                        # get the latest obs
                        if not self.q_world_state.empty():
                            if self.running_mode == "local_test_self":
                                logger.info("%s q_world_state queue is not empty" % self.running_mode)
                            continue
                        self.cur_dota_time = observation.dota_time
                        break
                except queue.Empty:
                    logger.info("%s myself q_world_state timeout" % self.running_mode)
                    self.end_game()
                    return

            self.action_num += 1
            self.sync_key = "###%d_%d###" % (self.action_num, self.team_id)

            sync_time = time.time()

            actions_pb, extra_actions = self.generate_action(observation, self.latest_enemy_obs)

            if actions_pb == -1 or actions_pb is None:
                logger.info("do not has hero!!!")
                continue

            s_write_time = time.time()
            # wait for executing action in lua
            actions = MessageToDict(actions_pb)

            if extra_actions != -1 and extra_actions is not None:
                actions['extra_actions'] = MessageToDict(extra_actions)

            if self.pics is not None and len(self.pics) != 0:
                actions['draw'] = self.pics

            try:
                self.dota_game.write_action(data=actions, team_id=self.team_id)
            except PermissionError as e:
                print(e)
                continue

            self.q_monitor_pattern.put({"pattern": self.sync_key, "dotatime": self.cur_dota_time})

            try:
                result_dict = self.q_monitor_result.get(timeout=20)
            except queue.Empty:
                logger.info("%s q_monitor_result timeout" % self.running_mode)
                self.end_game()
                return

    def current_player(self):
        if len(self.player_history_info) == 0:
            return None
        return self.player_history_info[-1]

    # 判断是否是队友
    def is_friend(self, u):
        return u.team_id == self.team_id

    # 判断是否是敌人
    def is_enemy(self, u):
        return u.team_id != self.team_id

    def is_myself(self, u):
        return u.team_id == self.team_id and u.player_id == self.player_id

    def is_hero(self, u):
        return u.unit_type == CMsgBotWorldState.UnitType.Value("HERO")

    def is_lane_creep(self, u):
        return u.unit_type == CMsgBotWorldState.UnitType.Value("LANE_CREEP")

    def is_tower(self, u):
        return u.unit_type == CMsgBotWorldState.UnitType.Value("TOWER")

    def is_courier(self, u):
        return u.unit_type == CMsgBotWorldState.UnitType.Value("COURIER")

    def is_tping(self):
        if len(self.current_player().modifiers) == 0:
            return False

        for i in self.current_player().modifiers:
            if i.name == 'modifier_teleporting':
                return True

        return False
