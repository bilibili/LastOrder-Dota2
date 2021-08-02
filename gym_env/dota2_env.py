from multiprocessing import Process, Queue
import os
import queue
import time
import psutil
import sys
from sys import platform


from gym_env.dota_game import logger, get_default_game_path, TEAM_RADIANT, TEAM_DIRE

from dotaservice.protos.dota_gcmessages_common_bot_script_pb2 import CMsgBotWorldState
from dotaservice.net_console_util import monitor_log, worldstate_listener
from dotaservice.dotautil import cal_distance
from dotaservice.state_machine import StateMachine, MACHINE_ACTION

from model.painter import draw_circle
from model.cb_features import CBFeature
from model.utils import hero_rewards_all, sampling_action, get_health_reward, sampling_action_cb
from model.dota_map import DotaMap

from gym_env.dota_game import DotaGame
from dotaservice.protos.dota_shared_enums_pb2 import DOTA_GAMEMODE_1V1MID


class Dota2Env:
    def __init__(self, timescale=1, render=False):
        host_timescale = timescale
        if render:
            host_mode = "HOST_MODE_GUI"
        else:
            host_mode = "HOST_MODE_DEDICATED"

        self.dota_game = DotaGame(
            host_timescale=host_timescale,
            ticks_per_observation=6,
            game_mode=DOTA_GAMEMODE_1V1MID,
            host_mode=host_mode,
        )

        self.team_id = TEAM_RADIANT
        self.q_world_state = None

    def reset(self):
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

        # print("dota_env restart after 5s...")
        # time.sleep(5)
        # os._exit(0)

        self.dota_game.run_dota()

        self.q_world_state = Queue()
        world_state_p = Process(
            target=worldstate_listener,
            args=(self.dota_game.PORT_WORLDSTATES[self.team_id], self.team_id, self.q_world_state,
                  self.dota_game.session_folder, self.dota_game.CONSOLE_LOG_FILENAME))
        world_state_p.daemon = True
        world_state_p.start()

        observation = self.q_world_state.get(timeout=300)
        return observation

    def step(self, action):
        try:
            self.dota_game.write_action(data=action, team_id=self.team_id)
        except PermissionError as e:
            print(e)

        observation = self.q_world_state.get(timeout=20)
        reward = 0
        done = 0
        if observation.dota_time > -60:
            done = 1

        info = {"dota_time": observation.dota_time}
        return observation, reward, done, info


if __name__ == '__main__':

    actions = {'dotaTime': -79.8333, 'actions': [{'actionType': 'DOTA_UNIT_ORDER_MOVE_DIRECTLY', 'player': 5,
                                                  'moveDirectly': {
                                                      'location': {'x': 3700.0, 'y': 3100.0, 'z': 256.0}}}],
               'extraData': '###1_3###', 'extra_actions': {'actions': [
            {'actionType': 'DOTA_UNIT_ORDER_TRAIN_ABILITY', 'player': 5,
             'trainAbility': {'ability': 'nevermore_necromastery'}}, {'actionType': 'ACTION_CHAT', 'player': 5,
                                                                      'chat': {
                                                                          'message': 'dota_lobby_gameplay_rules',
                                                                          'toAllchat': True}},
            {'actionType': 'DOTA_UNIT_ORDER_PURCHASE_ITEM', 'player': 5,
             'purchaseItem': {'itemName': 'item_circlet'}},
            {'actionType': 'DOTA_UNIT_ORDER_PURCHASE_ITEM', 'player': 5,
             'purchaseItem': {'itemName': 'item_slippers'}},
            {'actionType': 'DOTA_UNIT_ORDER_PURCHASE_ITEM', 'player': 5,
             'purchaseItem': {'itemName': 'item_flask'}},
            {'actionType': 'DOTA_UNIT_ORDER_PURCHASE_ITEM', 'player': 5,
             'purchaseItem': {'itemName': 'item_faerie_fire'}},
            {'actionType': 'DOTA_UNIT_ORDER_PURCHASE_ITEM', 'player': 5,
             'purchaseItem': {'itemName': 'item_faerie_fire'}},
            {'actionType': 'DOTA_UNIT_ORDER_PURCHASE_ITEM', 'player': 5,
             'purchaseItem': {'itemName': 'item_ward_observer'}}]}}

    dota_env = Dota2Env()
    o = dota_env.reset()

    while True:

        o, r, d, info = dota_env.step(actions)
        print(info["dota_time"])
        if d:
            print("done, reset env")
            o = dota_env.reset()
