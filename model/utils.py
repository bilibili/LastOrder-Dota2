import os, random, shutil
import numpy as np
import math
import yaml
import time
import pickle
import traceback
import socket
import psutil
import random
from gym_env.feature_processors.enums import *


def openai_sample(pd):
    noise = np.random.uniform(pd + 1e-8)
    mask = np.where(pd == 0.0, 0, 1)
    pd_with_noise = pd - np.log(-np.log(noise))
    return np.argmax(pd_with_noise * mask)


def distribution_sampling(distribution, index_range_list=[]):
    r = random.randint(0, 10000) + 1e-5
    accumulated_prob = 0
    last_positive_value_index = -1
    if len(index_range_list) == 0:
        index_range = range(len(distribution))
    else:
        index_range = index_range_list
    for index in index_range:
        i = distribution[index]
        if i > 1e-10:
            last_positive_value_index = index
        accumulated_prob += 10000 * i
        if accumulated_prob >= r:
            return index
    return last_positive_value_index


def multi_distribution_sampling(distribution):
    actions = []
    for pd in distribution:
        r = random.randint(0, 10000) + 1e-5
        accumulated_prob = 0
        last_positive_value_index = -1

        for index in range(len(pd)):
            prob = pd[index]
            if prob > 1e-10:
                last_positive_value_index = index
            accumulated_prob += 10000 * prob
            if accumulated_prob >= r:
                actions.append(index)
                break

            if index == len(pd) - 1:
                actions.append(last_positive_value_index)
    return actions


def get_max_diff(list_a):
    s_a = sorted(list_a)
    if len(s_a) > 2 and s_a[-2] > 0:
        diff_percent = (s_a[-1] - s_a[-2]) / s_a[-2]
    else:
        # get max index
        diff_percent = 1

    return diff_percent > 0.5 and s_a[-1] > 0.4


def create_path(path):
    if os.path.exists(path):
        shutil.rmtree(path)
    os.makedirs(path, exist_ok=True)


def sampling_action(action_type_distribution, move_degree_distribution, enemy_target, self_target, state_out,
                    current_state_value, running_mode):
    action_type_distribution = action_type_distribution.tolist()
    eval_min_prob = 0.5
    if running_mode == "self_eval" or running_mode == "ai_vs_ai":
        max_diff = get_max_diff(action_type_distribution)
        if max_diff:
            action_index = action_type_distribution.index(max(action_type_distribution))
        else:
            action_index = distribution_sampling(action_type_distribution)
    else:
        action_index = distribution_sampling(action_type_distribution)
    # move action
    parameter_list = []
    action_prob = action_type_distribution[action_index]
    final_action_prob = 0
    sub_prob_distribution = []
    sub_prob = 1

    if action_index == ACTION_NAME_TO_INDEX["MOVE"]:
        sub_prob_distribution = move_degree_distribution.tolist()
        if running_mode == "self_eval" or running_mode == "ai_vs_ai":
            #max_diff = get_max_diff(sub_prob_distribution)
            #if max_diff:
            #    degree_index = sub_prob_distribution.index(max(sub_prob_distribution))
            #else:
            degree_index = distribution_sampling(sub_prob_distribution)
        else:
            degree_index = distribution_sampling(sub_prob_distribution)
        sub_prob = sub_prob_distribution[degree_index]
        final_action_prob = sub_prob * action_prob
        parameter_list.append(degree_index)
    elif action_index == ACTION_NAME_TO_INDEX["ATTACK_ENEMY"]:
        sub_prob_distribution = enemy_target.tolist()
        if running_mode == "self_eval" or running_mode == "ai_vs_ai":
            max_diff = get_max_diff(sub_prob_distribution)
            if max_diff:
                target_index = sub_prob_distribution.index(max(sub_prob_distribution))
            else:
                target_index = distribution_sampling(sub_prob_distribution)
        else:
            target_index = distribution_sampling(sub_prob_distribution)
        sub_prob = sub_prob_distribution[target_index]
        final_action_prob = sub_prob * action_prob
        parameter_list.append(target_index)
    elif action_index == ACTION_NAME_TO_INDEX["ATTACK_SELF"]:
        sub_prob_distribution = self_target.tolist()
        if running_mode == "self_eval" or running_mode == "ai_vs_ai":
            max_diff = get_max_diff(sub_prob_distribution)
            if max_diff:
                target_index = sub_prob_distribution.index(max(sub_prob_distribution))
            else:
                target_index = distribution_sampling(sub_prob_distribution)
        else:
            target_index = distribution_sampling(sub_prob_distribution)
        sub_prob = sub_prob_distribution[target_index]
        final_action_prob = sub_prob * action_prob
        parameter_list.append(target_index)
    else:
        parameter_list = [0]
        final_action_prob = action_prob

    return [
        action_index, parameter_list,
        current_state_value.tolist()[0], action_prob, sub_prob, final_action_prob, sub_prob_distribution, state_out,
        action_type_distribution
    ]


def sampling_action_cb(move_degree_distribution, current_state_value, running_mode):
    parameter_list = []

    if running_mode in ['self_eval_double', 'self_eval', 'local_test_self', 'local_test_opponent']:
        if max(move_degree_distribution) > 0.5:
            degree_index = move_degree_distribution.argmax()
        else:
            degree_index = distribution_sampling(move_degree_distribution)
    else:
        degree_index = distribution_sampling(move_degree_distribution)
    action_prob = move_degree_distribution[degree_index]
    if degree_index == 12:
        action_index = 1
    else:
        action_index = 0
    parameter_list.append(degree_index * 3)

    return [action_index, parameter_list, current_state_value.tolist()[0], action_prob, move_degree_distribution]


# def send_log_date(msg_dic):
#     if 'LOCALTEST' not in os.environ.keys():
#         p = pickle.dumps([msg_dic])
#         context = zmq.Context()
#         with open('config.yaml', 'rt', encoding='utf8') as f:
#             conf = yaml.safe_load(f)
#         log_sender = context.socket(zmq.PUSH)
#         log_sender.connect("tcp://%s:%d" % (conf["log_server_address"]["ip"], conf["log_server_address"]["port"]))
#         log_sender.send(p)


def exception_handle(e, dota_game, logger):
    logger.warn(e)
    logger.info(traceback.format_exc())
    if 'LOCALTEST' not in os.environ.keys():
        send_message = {'msg_type': 'error'}
        send_message["error"] = traceback.format_exc()
        send_message["ip"] = socket.gethostname()
        p = pickle.dumps([send_message])

        context = zmq.Context()
        with open('config.yaml', 'rt', encoding='utf8') as f:
            conf = yaml.safe_load(f)
        log_sender = context.socket(zmq.PUSH)
        log_sender.connect("tcp://%s:%d" % (conf["log_server_address"]["ip"], conf["log_server_address"]["port"]))
        log_sender.send(p)
        time.sleep(3)
    dota_game.stop_dota_pids()
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


#only update self score, keep enemy score fix
def elo_cal(s_score, e_score, result):
    score_constant = 400
    learning_constant = 32
    s_e_expect_win_rate = 1 / (1 + math.pow(10, (e_score - s_score) / score_constant))
    s_score_new = s_score + learning_constant * (result - s_e_expect_win_rate)
    return s_score_new


# e_result_dict = {enemy1_score : [win, loss..], ..}
def elo_cal_batch_result(s_score, e_result_dict):
    score_constant = 400
    #evaluator take about 300 matches for one update, set smaller learning rate
    learning_constant = 1
    accumulate_expect_score = 0
    accumulate_true_score = 0
    total_result_count = 0
    for e_score, e_result_list in e_result_dict.items():
        s_e_expect_score = 1 / (1 + math.pow(10, (e_score - s_score) / score_constant))
        accumulate_expect_score += len(e_result_list) * s_e_expect_score
        accumulate_true_score += sum(e_result_list)
        total_result_count += len(e_result_list)
    s_score_new = s_score + learning_constant * (accumulate_true_score - accumulate_expect_score)
    return s_score_new


#input_distribution = {index: prob, ...}
#return output_slot_index = {outindex: inputindex, ...}
def get_slot_assignment(input_distribution, output_slot_count):
    each_slot_prob = 1 / float(output_slot_count)
    min_prob_threshold = each_slot_prob * 0.6
    output_result = {}
    cur_output_index = 0
    max_prob = -1
    max_prob_input_index = -1
    first_run = True

    while cur_output_index < output_slot_count:
        no_candidate = True
        for input_index, prob in input_distribution.items():
            if cur_output_index == output_slot_count:
                break
            if prob > min_prob_threshold:
                output_result[cur_output_index] = input_index
                no_candidate = False
                cur_output_index += 1
            if first_run and prob > max_prob:
                max_prob_input_index = input_index
                max_prob = prob
            input_distribution[input_index] = prob - each_slot_prob
        first_run = False
        if no_candidate is True:
            break

    if len(output_result) < output_slot_count:
        for i in range(len(output_result), output_slot_count):
            output_result[i] = max_prob_input_index

    ouput_list = [0 for i in range(output_slot_count)]
    for outindex, inputindex in output_result.items():
        ouput_list[outindex] = inputindex
    return ouput_list


def discount_with_dones(rewards, dones, gamma):
    discounted = []
    r = 0
    for reward, done in zip(rewards[::-1], dones[::-1]):
        r = reward + gamma * r * (1. - done) # fixed off by one bug
        discounted.append(r)
    return discounted[::-1], r


xp2next_level = {
    1: 230,
    2: 370,
    3: 480,
    4: 580,
    5: 600,
    6: 720,
    7: 750,
    8: 890,
    9: 930,
    10: 970,
    11: 1010,
    12: 1050,
    13: 1225,
    14: 1250,
    15: 1275,
    16: 1300,
    17: 1325,
    18: 1500,
    19: 1590,
    20: 1600,
    21: 1850,
    22: 2100,
    23: 2350,
    24: 2600,
    25: 3500,
    26: 4500,
    27: 5500,
    28: 6500,
    29: 7500,
    30: 9999,
}

reward_weights_all = {
    'experience': 0.002, # Per unit of experience.
    'gold': 0.006, # Per unit of gold gained[1].
    'mana': 0.5, # （0-0.75）
    'health': 1, # （0-2）
    'last_hits': -0.12, # Last Hitting an enemy creep[3].
    'denies': 0.1, # Last Hitting an allied creep[3].
    'kills': 1.5, # Killing an enemy hero[3].
    'deaths': -3, # Dying.
    'tower1': 1.5,
    'item_health': -0.3,
    'item_mana': -0.1,
    'item_fire': -0.3,
    'item_magic_stick': -0.5,
    'item_tango': -0.1,
    'ward': 0,
    's_ward': 0,
    'flask_break': 1
}


def get_flask_break_reward(player_history_info):
    if len(player_history_info[-2].modifiers) > 0:
        for i, mod in enumerate(player_history_info[-2].modifiers):
            if mod.name == 'modifier_flask_healing':
                if player_history_info[-2].health != player_history_info[-2].health_max and mod.remaining_duration > 0.2:
                    not_break = False
                    remaining_duration = mod.remaining_duration
                    if len(player_history_info[-1].modifiers) > 0:
                        for j, mod1 in enumerate(player_history_info[-1].modifiers):
                            if mod1.name == 'modifier_flask_healing':
                                not_break = True
                    if not_break is False:
                        pre_health = min(
                            float(player_history_info[-2].health + 50 * remaining_duration), player_history_info[-2].health_max)
                        pre_reward = get_health_reward(float(pre_health) / player_history_info[-2].health_max)
                        after_reward = get_health_reward(
                            float(player_history_info[-1].health) / player_history_info[-1].health_max)

                        return max(reward_weights_all['health'] * (after_reward - pre_reward), -0.4)

    return 0


def get_item_count(u):
    item_count = {'health': 0, 'mana': 0, 'fire': 0, 'magic_stick': 0, 'tango': 0, 'ward': 0, 's_ward': 0}
    if u.items is not None:
        for i in u.items:
            if i.slot <= 5:
                if i.ability_id == ITEM_FLASK_ID:
                    item_count['health'] += i.charges
                if i.ability_id == ITEM_CLARITY_ID or i.ability_id == ITEM_MANGO_ID:
                    item_count['mana'] += i.charges
                if i.ability_id in [ITEM_MAGIC_STICK_ID, ITEM_MAGIC_WAND_ID]:
                    item_count['magic_stick'] += i.charges
                if i.ability_id == ITEM_FAERIE_FIRE_ID:
                    item_count['fire'] += i.charges
                if i.ability_id == ITEM_TANGO_ID:
                    item_count['tango'] += i.charges
                if i.ability_id == ITEM_WARD_ID or i.ability_id == ITEM_WARD_DISPENSER_ID:
                    item_count['ward'] += i.charges
                if i.ability_id == ITEM_WARD_SENTRY_ID:
                    item_count['s_ward'] += i.charges

    return item_count


def get_health_reward(health_percent, pow=4):
    return 1 * health_percent + 1 - math.pow((1 - health_percent), pow)


def hero_rewards_all(player_history_info,
                     player_events_history_info,
                     self_tower,
                     dota_time,
                     is_enemy=False,
                     logger=False,
                     running_mode=False,
                     is_respown=False,
                     pre_death_health_reward=False):
    reward_dict = {}
    for key, _ in reward_weights_all.items():
        reward_dict[key] = 0
    if len(player_history_info) < 2 or len(player_events_history_info) < 2:
        return 0, reward_dict

    # enemy respown
    #if player_history_info[-1].is_alive is True and player_history_info[-2].is_alive is False:
    #    is_respown = True

    for key in player_history_info[-1].DESCRIPTOR.fields_by_name.keys():
        if key == 'xp_needed_to_level':
            if player_history_info[-1].level != player_history_info[-2].level:
                # 暂时不考虑一下升两级的情况
                reward_dict['experience'] = (xp2next_level[player_history_info[-1].level] -
                                             player_history_info[-1].xp_needed_to_level +
                                             player_history_info[-2].xp_needed_to_level) * reward_weights_all['experience']
            else:
                reward_dict['experience'] = (player_history_info[-2].xp_needed_to_level - player_history_info[-1].xp_needed_to_level) *\
                                           reward_weights_all['experience']

        elif key == 'reliable_gold':
            cur_gold = player_history_info[-1].reliable_gold + player_history_info[-1].unreliable_gold
            pre_gold = player_history_info[-2].reliable_gold + player_history_info[-2].unreliable_gold
            if cur_gold - pre_gold > 0:
                reward_dict['gold'] = reward_weights_all['gold'] * (cur_gold - pre_gold)

        elif key == 'health':
            if is_respown == False:
                cur_health_reward = get_health_reward(
                    float(player_history_info[-1].health) / player_history_info[-1].health_max)
                pre_health_reward = get_health_reward(
                    float(player_history_info[-2].health) / player_history_info[-2].health_max)
                reward_dict['health'] = reward_weights_all[key] * (cur_health_reward - pre_health_reward)
            else:
                if is_enemy == False:
                    reward_dict['health'] = pre_death_health_reward

        elif key == 'mana':
            if is_respown == False:
                reward_dict[key] = reward_weights_all[key] * \
                                   float(player_history_info[-1].mana - player_history_info[-2].mana) / player_history_info[-1].mana_max

        else:
            if key in reward_weights_all.keys():
                delta = (getattr(player_history_info[-1], key) - getattr(player_history_info[-2], key))
                reward_dict[key] = delta * reward_weights_all[key]

    # KDA reward
    for key in player_events_history_info[-1].DESCRIPTOR.fields_by_name.keys():
        if key in ['kills', 'deaths'] and is_enemy:
            continue
        if key in reward_weights_all:
            reward_dict[key] = (getattr(player_events_history_info[-1], key) - getattr(player_events_history_info[-2], key)) * \
                                reward_weights_all[key]

    # tower reward
    if len(self_tower) >= 2 and self_tower[-1].health - self_tower[-2].health < 0:
        cur_health_reward = get_health_reward(float(self_tower[-1].health) / self_tower[-1].health_max, pow=3)
        pre_health_reward = get_health_reward(float(self_tower[-2].health) / self_tower[-2].health_max, pow=3)
        reward_dict['tower1'] = reward_weights_all['tower1'] * (cur_health_reward - pre_health_reward)

    # item reward
    last_items = get_item_count(player_history_info[-1])
    before_items = get_item_count(player_history_info[-2])
    if last_items['health'] - before_items['health'] < 0:
        reward_dict['item_health'] = reward_weights_all['item_health']
    if last_items['mana'] - before_items['mana'] < 0:
        reward_dict['item_mana'] = reward_weights_all['item_mana']
    if last_items['fire'] - before_items['fire'] < 0:
        reward_dict['item_fire'] = reward_weights_all['item_fire']
    if last_items['magic_stick'] - before_items['magic_stick'] < 0:
        reward_dict['item_magic_stick'] = reward_weights_all['item_magic_stick']
    if last_items['tango'] - before_items['tango'] < 0:
        reward_dict['item_tango'] = reward_weights_all['item_tango']

    if last_items['ward'] - before_items['ward'] < 0:
        reward_dict['ward'] = reward_weights_all['ward']
    if last_items['s_ward'] - before_items['s_ward'] < 0:
        reward_dict['s_ward'] = reward_weights_all['s_ward']

    reward_dict['flask_break'] = get_flask_break_reward(player_history_info)

    if running_mode == "local_test_self" and reward_dict['flask_break'] != 0:
        logger.info("flask_break %f" % reward_dict['flask_break'])

    if dota_time is not None:
        decay_count = dota_time // (60 * 6)
        #(0-2), (3-5), (6-8)
        if decay_count > 1:
            decay_count = 1
        for key in reward_dict.keys():
            if key not in ['kills', 'deaths']:
                reward_dict[key] = reward_dict[key] * np.power(DECAY_RATE, decay_count)

    return sum(reward_dict.values()), reward_dict


def create_path(path):
    if os.path.exists(path):
        shutil.rmtree(path)
    os.makedirs(path, exist_ok=True)
