from sys import platform
import subprocess
import glob
import json
import logging
import os
import re
import shutil
import time
import uuid

from dotaservice.protos.dota_shared_enums_pb2 import DOTA_GAMERULES_STATE_GAME_IN_PROGRESS
from dotaservice.protos.dota_shared_enums_pb2 import DOTA_GAMERULES_STATE_PRE_GAME


logging.basicConfig(format='%(asctime)s %(levelname)-8s %(message)s')
logger = logging.getLogger(__name__)
if platform != 'win32':
    fh = logging.FileHandler('./dotapy.log', mode='w')
else:
    fh = logging.FileHandler('./bots/dotapy%d.log' % int(time.time()), mode='w')
fh.setLevel(logging.INFO)
logger.addHandler(fh)

LUA_FILES_GLOB = "./dotaservice/lua/*.lua"
LUA_FILES_GLOB_ACTIONS = "./dotaservice/lua/actions/*.lua"

TEAM_RADIANT = 2
TEAM_DIRE = 3


def get_default_game_path():
    game_path = None
    from play_with_human_local import DOTA_CLINET_PATH_MAC, DOTA_CLINET_PATH_WINDOWS, DOTA_CLINET_PATH_LINUX
    if platform == "linux" or platform == "linux2":
        game_path = os.path.expanduser(os.getenv('DOTA_GAME_PATH', DOTA_CLINET_PATH_LINUX))
    elif platform == 'darwin':
        game_path = os.path.expanduser(
            os.getenv('DOTA_GAME_PATH', DOTA_CLINET_PATH_MAC))
    elif platform == 'win32':
        game_path = os.path.expanduser(
            os.getenv('DOTA_GAME_PATH', DOTA_CLINET_PATH_WINDOWS))

    return game_path


def get_default_action_path():
    action_path = None
    from play_with_human_local import TMP_PATH_WINDOWS
    if platform == "linux" or platform == "linux2":
        action_path = "/tmp/"
    elif platform == "darwin":
        #action_path = "/Volumes/ramdisk/"
        action_path = "/tmp/"
    elif platform == "win32":
        action_path = TMP_PATH_WINDOWS

    return action_path


class DotaGame:
    ACTIONS_FILENAME_FMT = 'actions_t{team_id}'
    ACTIONABLE_GAME_STATES = [DOTA_GAMERULES_STATE_PRE_GAME, DOTA_GAMERULES_STATE_GAME_IN_PROGRESS]
    BOTS_FOLDER_NAME = 'bots'
    CONFIG_FILENAME = 'config_auto'
    CONSOLE_LOG_FILENAME = 'console.log'
    CONSOLE_LOGS_GLOB = 'console*.log'
    DOTA_SCRIPT_FILENAME = 'dota.sh'
    if platform == 'win32':
        DOTA_SCRIPT_FILENAME = os.path.join('bin', 'win64', 'dota2.exe')
    LIVE_CONFIG_FILENAME = 'live_config_auto'
    PORT_WORLDSTATES = {TEAM_RADIANT: 12120, TEAM_DIRE: 12121}
    RE_DEMO = re.compile(r'playdemo[ \t](.*dem)')
    RE_LUARDY = re.compile(r'LUARDY[ \t](\{.*\})')
    RE_WIN = re.compile(r'good guys win = (\d)')
    WORLDSTATE_PAYLOAD_BYTES = 4

    def __init__(
        self,
        host_timescale,
        ticks_per_observation,
        game_mode,
        host_mode,
        game_id=None,
    ):
        logger.setLevel('INFO')
        self.dota_path = get_default_game_path()
        self.action_folder = get_default_action_path()
        self.remove_logs = True
        self.host_timescale = host_timescale
        self.ticks_per_observation = ticks_per_observation
        self.game_mode = game_mode
        self.host_mode = host_mode
        self.game_id = game_id
        if not self.game_id:
            self.game_id = str(uuid.uuid1())
        self.dota_bot_path = os.path.join(self.dota_path, 'dota', 'scripts', 'vscripts', self.BOTS_FOLDER_NAME)
        self.bot_path = self._create_bot_path()
        self._write_config()

    def _write_config(self):
        # Write out the game configuration.
        config = {
            'game_id': self.game_id,
            'ticks_per_observation': self.ticks_per_observation,
        }
        self.write_static_config(data=config)

    def write_static_config(self, data):
        self._write_bot_data_file(filename_stem=self.CONFIG_FILENAME, data=data)

    def write_live_config(self, data):
        logger.debug('Writing live_config={}'.format(data))
        self._write_bot_data_file(filename_stem=self.LIVE_CONFIG_FILENAME, data=data)

    def write_action(self, data, team_id):
        filename_stem = self.ACTIONS_FILENAME_FMT.format(team_id=team_id)
        self._write_bot_data_file(filename_stem=filename_stem, data=data)

    def _write_bot_data_file(self, filename_stem, data):
        """Write a file to lua to that the bot can read it.

        Although writing atomicly would prevent bad reads, we just catch the bad reads in the
        dota bot client.
        """
        filename = os.path.join(self.bot_path, '{}.lua'.format(filename_stem))
        data = """return '{data}'""".format(data=json.dumps(data, separators=(',', ':')))
        with open(filename, 'w') as f:
            f.write(data)

    def _create_bot_path(self):
        """Remove DOTA's bots subdirectory or symlink and update it with our own."""
        if os.path.exists(self.dota_bot_path) or os.path.islink(self.dota_bot_path):
            if os.path.isdir(self.dota_bot_path) and not os.path.islink(self.dota_bot_path):
                raise ValueError('There is already a bots directory ({})! Please remove manually.'.format(self.dota_bot_path))
            os.remove(self.dota_bot_path)
        self.session_folder = os.path.join(self.action_folder, str(self.game_id))
        logger.debug('session_folder={}'.format(self.session_folder))
        if not os.path.isdir(self.session_folder):
            os.mkdir(self.session_folder)
        bot_path = os.path.join(self.session_folder, self.BOTS_FOLDER_NAME)
        if not os.path.isdir(bot_path):
            os.mkdir(bot_path)

        # Copy all the bot files into the action folder.
        lua_files = glob.glob(LUA_FILES_GLOB)
        for filename in lua_files:
            shutil.copy(filename, bot_path)

        # Copy all the bot action files into the actions subdirectory
        action_path = os.path.join(bot_path, "actions")
        os.mkdir(action_path)
        action_files = glob.glob(LUA_FILES_GLOB_ACTIONS)
        for filename in action_files:
            shutil.copy(filename, action_path)

        # Finally, symlink DOTA to this folder.
        try:
            os.symlink(src=bot_path, dst=self.dota_bot_path)
        except:
            pass
        return bot_path

    def stop_dota_pids(self):
        """Stop all dota processes.

        Stopping dota is nessecary because only one client can be active at a time. So we clean
        up anything that already existed earlier, or a (hanging) mess we might have created.
        """
        if platform != 'win32':
            os.system("pkill dota2")
        else:
            os.system("taskkill /IM dota2.exe")

        #dota_pids = str.split(os.popen("ps -e | grep dota2 | awk '{print $1}'").read())
        #for pid in dota_pids:
        #    try:
        #        os.kill(int(pid), signal.SIGKILL)
        #    except ProcessLookupError:
        #        pass

    def run_dota(self):
        self.stop_dota_pids()
        script_path = os.path.join(self.dota_path, self.DOTA_SCRIPT_FILENAME)

        # TODO(tzaman): all these options should be put in a proto and parsed with gRPC Config.
        args = [
            script_path,
            '-botworldstatesocket_threaded',
            '-botworldstatetosocket_frames',
            '{}'.format(self.ticks_per_observation),
            '-botworldstatetosocket_radiant',
            '{}'.format(self.PORT_WORLDSTATES[TEAM_RADIANT]),
            '-botworldstatetosocket_dire',
            '{}'.format(self.PORT_WORLDSTATES[TEAM_DIRE]),
            '-con_logfile',
            '{}'.format(os.path.join(self.session_folder, 'bots', self.CONSOLE_LOG_FILENAME)),
            # '-con_logfile', '/home/luke/console.log',
            '-con_timestamp',
            '-console',
            '-insecure',
            '-noip',
            '-nowatchdog', # WatchDog will quit the game if e.g. the lua api takes a few seconds.
            '+clientport',
            '27006', # Relates to steam client.
            '+dota_surrender_on_disconnect',
            '0',
            '+host_timescale',
            '{}'.format(self.host_timescale),
            '+hostname',
            'dotaservice',
            '+sv_cheats',
            '1',
            '+sv_hibernate_when_empty',
            '0',
            '+tv_delay',
            '0',
            '+tv_enable',
            '1',
            '+tv_title',
            '{}'.format(self.game_id),
            '+tv_autorecord',
            '1',
            '+tv_transmitall',
            '1', # TODO(tzaman): what does this do exactly?
            '+dota_1v1_skip_strategy',
            '1',
        ]

        if self.host_mode == "HOST_MODE_DEDICATED":
            args.append('-dedicated')
        if self.host_mode == "HOST_MODE_DEDICATED" or \
                self.host_mode == "HOST_MODE_GUI":
            args.append('-fill_with_bots')
            # args.append('-w 720')
            # args.append('-h 480')
            args.extend(['+map', 'start', 'gamemode', str(self.game_mode)])
            args.extend(['+sv_lan', '1'])
        if self.host_mode == "HOST_MODE_GUI_MENU":
            args.append('-novid')
            args.extend(['+sv_lan', '0'])

        cmd = ' '.join(args)
        logger.info(cmd)
        #non-blocking call
        subprocess.Popen(args)
