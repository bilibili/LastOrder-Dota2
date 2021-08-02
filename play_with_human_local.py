from gym_env.dota_game import DotaGame, TEAM_RADIANT, TEAM_DIRE
from dotaservice.protos.dota_shared_enums_pb2 import DOTA_GAMEMODE_1V1MID
from agents.dota_agent import PPOAgent
from multiprocessing import Process
from sys import platform
import time
import os
import pathlib


# path example
DOTA_CLINET_PATH_MAC = "~/Library/Application Support/Steam/steamapps/common/dota 2 beta/game"
DOTA_CLINET_PATH_WINDOWS = r'E:\SteamLibrary\steamapps\common\dota 2 beta\game'
DOTA_CLINET_PATH_LINUX = "~/.steam/steam/steamapps/common/dota 2 beta/game"


TMP_PATH_WINDOWS = str(pathlib.Path(__file__).parent.resolve()) + r'\tmp'


LAST_ORDER_PROJECT_PATH_MAC = pathlib.Path(__file__).parent.resolve()
LAST_ORDER_PROJECT_PATH_WINDOWS = pathlib.Path(__file__).parent.resolve()
LAST_ORDER_PROJECT_PATH_LINUX = pathlib.Path(__file__).parent.resolve()
print(LAST_ORDER_PROJECT_PATH_WINDOWS)


def dota_process_exists():
    if platform == 'win32':
        return len(os.popen("tasklist /v | findstr dota2.exe").read()) != 0
    else:
        return len(os.popen("ps aux | grep dota2 | grep -v grep").read()) != 0


def run_human_vs_ai(dota_game: DotaGame, team_id: int, player_id: int, opponent_player_id: int):
    if platform == 'darwin':
        dota_game.session_folder = LAST_ORDER_PROJECT_PATH_MAC
    elif platform == 'win32':
        dota_game.session_folder = LAST_ORDER_PROJECT_PATH_WINDOWS
    else:
        dota_game.session_folder = LAST_ORDER_PROJECT_PATH_LINUX

    agent = PPOAgent(
        dota_game,
        team_id,
        player_id,
        opponent_player_id,
        "",
        "self_eval",
    )
    agent.run()


def run_dota():
    dota_game = DotaGame(host_timescale=1, ticks_per_observation=6, game_mode=DOTA_GAMEMODE_1V1MID, host_mode="HOST_MODE_GUI_MENU")
    if platform == 'darwin':
        dota_game.session_folder = LAST_ORDER_PROJECT_PATH_MAC
    elif platform == 'win32':
        dota_game.session_folder = LAST_ORDER_PROJECT_PATH_WINDOWS
    else:
        dota_game.session_folder = LAST_ORDER_PROJECT_PATH_LINUX
    try:
        dota_game.stop_dota_pids()
        dota_game.run_dota()
        time.sleep(10)
    except Exception as e:
        print(e)
        dota_game.stop_dota_pids()


def supervisor():
    while True:
        if not dota_process_exists():
            Process(target=run_dota).run()
            dota_game = DotaGame(host_timescale=1, ticks_per_observation=6, game_mode=DOTA_GAMEMODE_1V1MID, host_mode="HOST_MODE_GUI_MENU")
            dp = Process(target=run_human_vs_ai, args=(dota_game, TEAM_RADIANT, 1, 0))
            dp.start()
        time.sleep(20)


if __name__ == "__main__":
    supervisor()
