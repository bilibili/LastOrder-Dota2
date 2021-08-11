import os
import time
import socket
import sys
from pathlib import Path
from struct import unpack
from sys import platform
from dotaservice.protos.dota_shared_enums_pb2 import DOTA_GAMERULES_STATE_GAME_IN_PROGRESS
from dotaservice.protos.dota_shared_enums_pb2 import DOTA_GAMERULES_STATE_PRE_GAME
from gym_env.dota_game import logger
from dotaservice.protos.dota_gcmessages_common_bot_script_pb2 import CMsgBotWorldState


def monitor_log(session_folder, console_log, pattern_queue, result_queue):
    abs_glob = os.path.join(session_folder, "bots", console_log)

    latest_lua_step = -1
    latest_lua_realtime = -1

    while True:
        if Path(abs_glob).is_file():
            logger.debug("wait file")
            break
        else:
            time.sleep(0.5)

    during_game_max_time = 1

    with open(abs_glob, 'r', encoding="UTF-8") as f:
        f.seek(0, 0)
        while True:
            # pattern format: dict
            raw_pattern = pattern_queue.get()
            during_game = False
            pattern = raw_pattern["pattern"]
            if 'dotatime' in raw_pattern:
                during_game = True
                obs_dotatime = raw_pattern["dotatime"]
                start_time = time.time()

            # Go to the end of file
            while True:
                if during_game:
                    if time.time() - start_time > during_game_max_time:
                        result_queue.put({'reaction_time': -1, 'lua_realtime': -1, 'lua_step': -1, 'error_code': 0})
                        break

                curr_position = f.tell()
                line = f.readline()
                if not line:
                    f.seek(curr_position)
                    time.sleep(0.001)
                else:
                    if line.find(pattern) != -1:
                        if during_game:
                            try:
                                _, _, lua_realtime, lua_dotatime, lua_step = line.split("###")
                            except Exception as e:
                                result_queue.put({'reaction_time': 0, 'lua_realtime': 0, 'lua_step': 0, 'error_code': 1})
                                break

                            if latest_lua_step == -1:
                                lua_realtime_diff = 0
                                lua_step_diff = 0
                            else:
                                lua_realtime_diff = int((float(lua_realtime) - latest_lua_realtime) * 1000)
                                lua_step_diff = int((int(lua_step) - latest_lua_step) * 1000)

                            result_queue.put({
                                'reaction_time': int((float(lua_dotatime) - obs_dotatime) * 1000),
                                'lua_realtime': lua_realtime_diff,
                                'lua_step': lua_step_diff,
                                'error_code': 0
                            })
                            latest_lua_realtime = float(lua_realtime)
                            latest_lua_step = int(lua_step)
                        else:
                            result_queue.put(line)
                        break


def _world_state_from_reader(reader):
    # Receive the package length.
    remain_length = 4
    data = b""
    while True:
        tmp = reader.recv(remain_length)
        if len(tmp) == 0:
            logger.debug("connection close")
            raise Exception
        data = data + tmp
        remain_length = remain_length - len(data)
        if remain_length == 0:
            break

    n_bytes = unpack("@I", data)[0]
    data = b""
    while True:
        # Receive the payload given the length.
        tmp = reader.recv(n_bytes)
        if len(tmp) == 0:
            logger.debug("connection close")
            raise Exception
        data = data + tmp
        n_bytes = n_bytes - len(data)
        if n_bytes == 0:
            break

    # Decode the payload.
    world_state = CMsgBotWorldState()
    world_state.ParseFromString(data)
    logger.debug("Received world_state: dotatime={}, gamestate={}".format(world_state.dota_time, world_state.game_state))

    return world_state


def worldstate_listener(port, team_id, queue, session_folder, console_log):
    retry_counter = 0
    while True:
        client = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        error = client.connect_ex(("127.0.0.1", port))
        if not error:
            retry_counter = 0
            break
        time.sleep(1)
        if retry_counter > 15:
            print("worldstate_listener connect error more than 15 times, EXIT!!!!")
            sys.exit(0)
        else:
            retry_counter += 1

        print("worldstate_listener connect error!")

    while True:
        # This reader is always going to need to keep going to keep the buffers flushed.
        try:
            world_state = _world_state_from_reader(client)
            if world_state is None:
                logger.debug("Finishing worldstate listener (team_id={})".format(team_id))
                return
            is_in_game = world_state.game_state in [DOTA_GAMERULES_STATE_PRE_GAME, DOTA_GAMERULES_STATE_GAME_IN_PROGRESS]
            has_units = len(world_state.units) > 0
            if is_in_game and has_units:
                if (platform == 'darwin') or queue.qsize() <= 1:
                    queue.put(world_state)
            #del(world_state)
        except Exception as e:
            logger.debug(e)
            while True:
                client = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
                error = client.connect_ex(("127.0.0.1", port))
                if not error:
                    retry_counter = 0
                    break
                time.sleep(0.5)
                if retry_counter > 15:
                    sys.exit(0)
                else:
                    retry_counter += 1
            continue
