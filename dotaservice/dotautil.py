import numpy as np
import math
from dotaservice.protos.dota_gcmessages_common_bot_script_pb2 import CMsgBotWorldState


# 护甲减伤
def armor_filter(armor):
    return 1 - ((0.06 * armor) / (1.0 + 0.06 * abs(armor)))


# 普通攻击
def attack_damage(attack, armor):
    return attack * armor_filter(armor)


# 普通攻击致死次数
def attack_to_death_times(attack, armor, hp):
    if hp <= 0:
        return 0
    damage = attack_damage(attack, armor)
    times = int(hp / damage) + 1
    return times


# 攻击速度
# https://dota2.gamepedia.com/Attack_speed
def attack_per_second(ias, bat=1.7):
    if ias < 0.2:
        ias = 0.2
    if ias > 7:
        ias = 7
    return ias / 1.7


# 每次攻击时间
# https://dota2-zh.gamepedia.com/index.php?title=%E6%94%BB%E5%87%BB%E5%8A%A8%E4%BD%9C&variant=zh
def attack_time(ias, bat=1.7):
    return 1 / attack_per_second(ias, bat=bat)


# 计算平面距离
def cal_distance(p1, p2):
    if p1 is None or p2 is None:
        return -1
    return np.sqrt(np.power(p1.x - p2.x, 2) + np.power(p1.y - p2.y, 2))


# 计算平面距离和纵向距离
def cal_distance_with_z(p1, p2):
    if p1 is None or p2 is None:
        return -1, -1
    else:
        return cal_distance(p1, p2), p2.z - p1.z


# 计算两个坐标之间的角度,保持和action degree的执行方式一致
def location_to_degree(hero_location, target_location):
    direct_x = target_location.x - hero_location.x
    direct_y = target_location.y - hero_location.y

    degree = np.degrees(np.arctan2(direct_y, direct_x))
    if degree < 0:
        return degree + 360
    else:
        return degree


# 在面前距离内
def in_facing_distance(hero, u, distance, r=250, normalization=False):
    x = math.cos(math.radians(hero.facing)) * distance + \
        hero.location.x
    y = math.sin(math.radians(hero.facing)) * distance + \
        hero.location.y

    location = CMsgBotWorldState.Vector(x=x, y=y, z=512.0)

    d = cal_distance(u.location, location) # 技能中心点距离单位的位置
    if d < r and u.team_id != hero.team_id:
        if normalization:
            return (r - d) / r # 在中心点时为1，靠近边缘趋近与0，出范围为0
        else:
            return 1
    else:
        return 0
