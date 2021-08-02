import numpy as np
from gym_env.feature_processors.enums import ACTION_NAME_TO_INDEX, DOUBLE_ACTION_PARA_TYPE


class Instance:
    # reward is the td n reward plus the target state value
    def __init__(self,
                 dota_time=None,
                 state_gf=None,
                 state_ucf=None,
                 state_ucategory=None,
                 mask=None,
                 reward=0,
                 action=None,
                 action_params=None,
                 state_value=0,
                 dump_path=None,
                 instant_reward=0.,
                 gae_advantage=0,
                 action_prob=None,
                 sub_action_prob=1,
                 final_action_prob=None,
                 model_time=None,
                 units_mask=None,
                 lstm_state=None,
                 lstm_gradient_mask=None,
                 embedding_dict=None,
                 dota_map=None,
                 update_times=0):
        self.dota_time = dota_time
        self.state_gf = state_gf
        self.state_ucf = state_ucf
        self.state_ucategory = state_ucategory
        self.mask = mask

        self.state_value = state_value
        self.action = action
        self.action_params = action_params
        self.q_reward = reward
        self.instant_reward = instant_reward
        self.model_time = model_time
        self.action_prob = action_prob
        self.sub_action_prob = sub_action_prob

        self.gae_advantage = gae_advantage
        self.units_mask = units_mask

        self.lstm_state = lstm_state
        self.lstm_gradient_mask = 1

        self.embedding_dict = embedding_dict
        self.dota_map = dota_map

        self.update_times = update_times

    def zeros_like(self, target_instance):
        self.dota_time = 0
        self.state_gf = np.zeros_like(target_instance.state_gf)
        self.state_ucf = np.zeros_like(target_instance.state_ucf)
        # for ensure there is one enemy hero/tower
        self.state_ucategory = target_instance.state_ucategory
        self.mask = np.zeros_like(target_instance.mask)

        self.state_value = 0
        self.action = ACTION_NAME_TO_INDEX["STOP"]
        self.action_params = {}
        for atype in DOUBLE_ACTION_PARA_TYPE:
            self.action_params[atype] = 0
        self.q_reward = 0
        self.instant_reward = 0
        self.model_time = target_instance.model_time
        self.action_prob = 1
        self.sub_action_prob = 1

        self.gae_advantage = 0
        self.units_mask = np.zeros_like(target_instance.units_mask)

        self.lstm_state = np.zeros_like(target_instance.lstm_state)
        self.lstm_gradient_mask = 1
        self.embedding_dict = target_instance.embedding_dict
        self.dota_map = np.zeros_like(target_instance.dota_map)
        self.update_times = 0


def padding_instance(reward_instance, latest_instance, total_length, exclude_last_instance):
    padding_length = total_length - len(reward_instance)
    if exclude_last_instance:
        start_position = -len(reward_instance) - 1
    else:
        start_position = -len(reward_instance)
    padding_instances = latest_instance[start_position - padding_length:start_position]
    if len(padding_instances) < padding_length:
        zero_instance = Instance()
        zero_instance.zeros_like(reward_instance[0])
        for i in range(padding_length - len(padding_instances)):
            padding_instances.insert(0, zero_instance)

    #padding instance do not compute gradient
    for index, item in enumerate(padding_instances):
        padding_instances[index].lstm_gradient_mask = 0
    for index, item in enumerate(reward_instance):
        reward_instance[index].lstm_gradient_mask = 1
    padding_instances.extend(reward_instance)
    return padding_instances
