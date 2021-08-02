import json
import numpy as np
from dotaservice.protos.dota_gcmessages_common_bot_script_pb2 import CMsgBotWorldState


class DotaMap():

    def __init__(self, team_id, enemy_team_id):
        self.team_id = team_id
        self.enemy_team_id = enemy_team_id

        with open('./data/mapdata.7.24b.json') as f:
            self.map_data = json.loads(f.read())['data']
        # [-8288, 8288]
        # Coordinates of all untraversable 64x64 grid tiles.
        with open('./data/gridnavdata.7.24b.json') as f:
            self.grid_data = json.loads(f.read())['data']
        # Elevations of each 64x64 grid tile. height / 128
        with open('./data/elevationdata.7.24b.json') as f:
            self.elevationdata = json.loads(f.read())['data']

        # [阻碍，树木，高度，敌方单位，友方, 敌方英雄]
        self.np_map = np.zeros((150, 150, 6))

        for d in self.grid_data:
            x = int((d['x'] + 7500) / 100)
            y = int((d['y'] + 7500) / 100)
            if x >= 0 and x < 150 and y >= 0 and y < 150:
                self.np_map[x][y][0] += 1

        for index, d in self.map_data['npc_dota_filler'].items():
            x = int((d['x'] + 7500) / 100)
            y = int((d['y'] + 7500) / 100)
            if x >= 0 and x < 150 and y >= 0 and y < 150:
                self.np_map[x][y][0] += 1

        for index, d in self.map_data['npc_dota_tower'].items():
            x = int((d['x'] + 7500) / 100)
            y = int((d['y'] + 7500) / 100)
            if x >= 0 and x < 150 and y >= 0 and y < 150:
                self.np_map[x][y][0] += 1

        for index, d in self.map_data['npc_dota_watch_tower'].items():
            x = int((d['x'] + 7500) / 100)
            y = int((d['y'] + 7500) / 100)
            if x >= 0 and x < 150 and y >= 0 and y < 150:
                self.np_map[x][y][0] += 1

        for index, d in self.map_data['npc_dota_roshan_spawner'].items():
            x = int((d['x'] + 7500) / 100)
            y = int((d['y'] + 7500) / 100)
            if x >= 0 and x < 150 and y >= 0 and y < 150:
                self.np_map[x][y][0] += 1

        for index, d in self.map_data['npc_dota_neutral_spawner'].items():
            x = int((d['x'] + 7500) / 100)
            y = int((d['y'] + 7500) / 100)
            if x >= 0 and x < 150 and y >= 0 and y < 150:
                self.np_map[x][y][0] += 1

        for d in self.map_data['ent_dota_tree']:
            x = int((d['x'] + 7500) / 100)
            y = int((d['y'] + 7500) / 100)
            if x >= 0 and x < 150 and y >= 0 and y < 150:
                self.np_map[x][y][1] += 1

        for index_x, arr in enumerate(self.elevationdata):
            for index_y, height in enumerate(arr):
                x = int((index_x * 64 - 8288 + 7500) / 100)
                y = int((index_y * 64 - 8288 + 7500) / 100)
                if x >= 0 and x < 150 and y >= 0 and y < 150 and height != -128:
                    self.np_map[x][y][2] = height

    def update(self, obs):
        # 暂时没有吃树操作，树木更新暂时不考虑
        self.np_map[:, :, 3] = 0
        self.np_map[:, :, 4] = 0
        self.np_map[:, :, 5] = 0

        # 种树 tree id
        tree_ids = []

        for t in obs.tree_events:
            if t.destroyed is True:
                x = int((t.location.x + 7500) / 100)
                y = int((t.location.y + 7500) / 100)
                if x >= 0 and x < 150 and y >= 0 and y < 150:
                    self.np_map[x][y][1] -= 1

            if t.respawned is True:
                # 种下的树坐标全是 0
                x = int((t.location.x + 7500) / 100)
                y = int((t.location.y + 7500) / 100)
                if x >= 0 and x < 150 and y >= 0 and y < 150:
                    self.np_map[x][y][1] += 1

        for u in obs.units:
            if u.is_alive is not True or u.location is None or \
                 (u.unit_type != CMsgBotWorldState.UnitType.Value("LANE_CREEP") and
                  u.unit_type != CMsgBotWorldState.UnitType.Value("TOWER") and
                  u.unit_type != CMsgBotWorldState.UnitType.Value("HERO")):
                continue
            x = int((u.location.x + 7500) / 100)
            y = int((u.location.y + 7500) / 100)

            if x >= 0 and x < 150 and y >= 0 and y < 150:
                if self.team_id == u.team_id:
                    self.np_map[x][y][3] += 1
                elif self.enemy_team_id == u.team_id:
                    self.np_map[x][y][4] += 1
                    # 敌方英雄单独做一层
                    if u.unit_type == CMsgBotWorldState.UnitType.Value("HERO"):
                        self.np_map[x][y][5] = 1

        return []

    def nearby(self, origin_x, origin_y, size=4):
        origin_x = (origin_x + 7500) / 100
        origin_y = (origin_y + 7500) / 100
        shape_size = 2 * size + 1
        result = np.zeros((shape_size, shape_size, 6))
        for x in range(-size, size + 1):
            for y in range(-size, size + 1):
                offset_x = int(origin_x + x)
                offset_y = int(origin_y + y)
                if offset_x >= 0 and offset_y >= 0 and offset_x < 150 and offset_y < 150:
                    result[x + size][y + size] = self.np_map[offset_x][offset_y]
                else:
                    result[x + size][y + size] = np.zeros(6)
        return result
