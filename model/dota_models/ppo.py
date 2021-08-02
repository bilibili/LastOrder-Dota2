import time
import tensorflow.compat.v1 as tf

tf.disable_v2_behavior()
import numpy as np
from model.basic_model import BasicModel
from gym_env.feature_processors.enums import UNIT_NUM, ABILITY_NUM, All_ACTION_TYPE, ACTION_NAME_TO_INDEX, NEARBY_MAP_SIZE
from gym_env.feature_processors.features_v0.features import Feature
from model.utils import sampling_action


class PPOModel(BasicModel):
    lstm_layer_size = 500
    move_x_partition_count = 36
    ACTION_EMB_SIZE = 16
    use_dropout = False

    def __init__(self, param, use_distribution=False, training_server=False):
        super().__init__()
        self.max_unit_count = UNIT_NUM
        self.max_unit_ability_count = ABILITY_NUM

        self.feature_global_num = len(Feature.current_global_features) + len(Feature.current_courier_feature) + len(
            Feature.current_home_item_feature)
        self.feature_unit_common_num = len(Feature.current_unit_common_features) + \
            len(Feature.current_hero_ability_features) * len(Feature.hero_ability) + \
            len(Feature.current_hero_modifier_feature) * len(Feature.modifier_name) + \
            len(Feature.current_hero_item_feature) * len(Feature.items_name)
        self.feature_unit_ability_num = len(Feature.current_hero_ability_features)
        self.action_types = All_ACTION_TYPE

        self.start_training_time = time.time()
        self.use_distribution = use_distribution
        self.learning_rate = param['learning_rate']
        self.clip_range = param['clip_range']
        self.action_entropy_parameter = param['action_entropy_parameter']
        self.share_input_shape = param['share_feature_shape']
        self.player_input_shape = param['player_feature_shape']
        self.action_shape = len(self.action_types) # param['output_shape']  # 第一层动作
        self.max_action = self.action_shape
        self.nearby_map_fullsize = 2 * NEARBY_MAP_SIZE + 1 # 附近地图大小
        self.pre_action_length = param["pre_action_length"]
        self.player_num = param["player_num"]
        if training_server:
            self.n_steps = param['n_steps']
            self.lstm_step = 10
        else:
            self.n_steps = 1
            self.lstm_step = 1

        self.low_bound_range = 3.0
        self.max_grad_norm = 2.0
        self.unit_feature_common_length = 100
        self.update_times = 0

        with tf.variable_scope("all_shared_part"):

            #action embedding
            self.self_pre_action_id = tf.placeholder(dtype=tf.int32, shape=[None], name='pre_action_id')
            self.action_type_embedding_matrix = tf.get_variable(
                'action_type_embedding_matrix', [self.max_action, self.ACTION_EMB_SIZE],
                initializer=tf.glorot_uniform_initializer())
            self.action_type_embedding = tf.nn.embedding_lookup(self.action_type_embedding_matrix, self.self_pre_action_id)

            self.self_pre_move_id = tf.placeholder(dtype=tf.int32, shape=[None], name='self_pre_move_id')
            self.move_embedding_matrix = tf.get_variable(
                'move_direction_embedding_matrix', [self.move_x_partition_count + 1, self.ACTION_EMB_SIZE],
                initializer=tf.glorot_uniform_initializer())
            self.self_move_embedding = tf.nn.embedding_lookup(self.move_embedding_matrix, self.self_pre_move_id)

            self.enemy_action_type_id = tf.placeholder(dtype=tf.int32, shape=[None], name='enemy_action_id')
            self.enemy_action_type_embedding_matrix = tf.get_variable(
                'enemy_action_type_embedding_matrix', [self.max_action, self.ACTION_EMB_SIZE],
                initializer=tf.glorot_uniform_initializer())
            self.enemy_action_type_embedding = tf.nn.embedding_lookup(self.enemy_action_type_embedding_matrix,
                                                                      self.enemy_action_type_id)

            # anim embeding
            self.max_anim = 20
            self.enemy_anim_type_id = tf.placeholder(dtype=tf.int32, shape=[None], name='enemy_anim_id')
            self.enemy_anim_type_embedding_matrix = tf.get_variable(
                'enemy_anim_type_embedding_matrix', [self.max_anim, self.ACTION_EMB_SIZE],
                initializer=tf.glorot_uniform_initializer())
            self.enemy_anim_type_embedding = tf.nn.embedding_lookup(self.enemy_anim_type_embedding_matrix,
                                                                    self.enemy_anim_type_id)
            self.self_anim_type_id = tf.placeholder(dtype=tf.int32, shape=[None], name='self_anim_id')
            self.self_anim_type_embedding_matrix = tf.get_variable(
                'self_anim_type_embedding_matrix', [self.max_anim, self.ACTION_EMB_SIZE],
                initializer=tf.glorot_uniform_initializer())
            self.self_anim_type_embedding = tf.nn.embedding_lookup(self.self_anim_type_embedding_matrix, self.self_anim_type_id)

            # valid action filled with 1 on corresponding position, invalid action filled with 0
            self.valid_actions_mask = tf.placeholder(dtype=tf.float32, shape=[None, self.max_action], name='actions_mask')
            self.units_mask = tf.placeholder(dtype=tf.float32, shape=[None, self.max_unit_count], name='units_mask')
            self.feature_global = tf.placeholder(tf.float32, [None, self.feature_global_num], name="input_feature_global")

            # hero nearby map info
            self.feature_map_global = tf.placeholder(
                tf.float32, [None, self.nearby_map_fullsize, self.nearby_map_fullsize, 6], name="input_feature_map_global")

            # shape = [batch size, max_unit_count, feature]
            self.feature_unit_global = tf.placeholder(
                tf.float32, [None, self.max_unit_count, self.feature_unit_common_num], name="input_feature_unit_global")
            self.feature_unit_categoy = tf.placeholder(tf.int32, [None, self.max_unit_count], name="input_feature_unit_categoy")
            self.feature_unit_categoy_batch = tf.reshape(self.feature_unit_categoy, [-1, 1])

            self.feature_unit_global_tran = tf.reshape(self.feature_unit_global, [-1, self.feature_unit_common_num])

            self.feature_unit_all_feature_1 = self.fc_layer(
                self.feature_unit_global_tran,
                self.unit_feature_common_length * 2,
                "feature_unit_global_tran_1",
                ln=True,
                activation=tf.nn.relu)

            self.feature_unit_all_feature_2 = self.fc_layer(
                self.feature_unit_all_feature_1,
                self.unit_feature_common_length,
                "feature_unit_global_tran_2",
                ln=True,
                activation=tf.nn.relu)

            #unit out size
            common_unit_out_size = int(self.unit_feature_common_length * 0.8)
            reshaped_units_mask = tf.reshape(self.units_mask, [-1])

            # get self hero, shape [batch * 1, size]
            self.self_hero_feature, _ = self.get_type_fc_and_mask(self.feature_unit_all_feature_2,
                                                                  self.feature_unit_categoy_batch, reshaped_units_mask, 4,
                                                                  "self_hero_feature_raw", common_unit_out_size)
            # get enemy hero, shape [batch * 1, size]
            self.enemy_hero_feature, self.enemy_hero_mask = self.get_type_fc_and_mask(self.feature_unit_all_feature_2,
                                                                                      self.feature_unit_categoy_batch,
                                                                                      reshaped_units_mask, 5,
                                                                                      "enemy_hero_feature_raw",
                                                                                      common_unit_out_size)
            # get self tower, shape [batch * 1, size]
            self.self_tower_feature, _ = self.get_type_fc_and_mask(self.feature_unit_all_feature_2,
                                                                   self.feature_unit_categoy_batch, reshaped_units_mask, 9,
                                                                   "self_tower_feature_raw", common_unit_out_size)
            # get enemy tower, shape [batch * 1, size]
            self.enemy_tower_feature, self.enemy_tower_mask = self.get_type_fc_and_mask(self.feature_unit_all_feature_2,
                                                                                        self.feature_unit_categoy_batch,
                                                                                        reshaped_units_mask, 7,
                                                                                        "enemy_tower_feature_raw",
                                                                                        common_unit_out_size)

            # unit feature
            enemy_units_attention_features, enemy_units_raw_features = self.attention_with_fc(
                self.self_hero_feature, self.feature_unit_all_feature_2,
                int(self.unit_feature_common_length * 0.5), "enemy_attention", self.feature_unit_categoy_batch, 6,
                int(self.unit_feature_common_length * 0.9), common_unit_out_size)

            self_units_maxpool_features, self_units_raw_features = self.attention_with_fc(
                self.self_hero_feature, self.feature_unit_all_feature_2,
                int(self.unit_feature_common_length * 0.5), "self_attention", self.feature_unit_categoy_batch, 2,
                int(self.unit_feature_common_length * 0.9), common_unit_out_size)

            feature_map_global_cnn = self.cnn_layer(self.feature_map_global, scope='dota_map')

            # global and units feature
            feature_global_tran = self.fc_layer(
                self.feature_global, self.unit_feature_common_length, "feature_global", ln=True, activation=tf.nn.relu)

            # concat all feature
            self.feature_concat_raw = tf.concat([
                feature_global_tran, self.self_hero_feature, self.enemy_hero_feature, self.self_tower_feature,
                self.enemy_tower_feature, enemy_units_attention_features, self_units_maxpool_features,
                self.action_type_embedding, self.enemy_action_type_embedding, self.enemy_anim_type_embedding,
                self.self_anim_type_embedding, self.self_move_embedding, feature_map_global_cnn
            ],
                                                axis=-1)

            #lstm
            self.lstm_state = tf.placeholder(tf.float32, [None, 2 * self.lstm_layer_size], name='lstm_state')

            xs = self.batch_to_seq(self.feature_concat_raw, self.lstm_step)
            outputs, self.state_out = self.openai_lstm(xs, self.lstm_state, 'units_lstm', self.lstm_layer_size)
            self.feature_out = self.seq_to_batch(outputs)

        with tf.variable_scope("selected_action_head"):
            self.feature_action_1 = self.fc_layer(
                self.feature_out, self.unit_feature_common_length * 3, "selected_action_head_1", ln=True, activation=tf.nn.relu)

            self.feature_action_2 = self.fc_layer(
                self.feature_action_1,
                self.unit_feature_common_length,
                "selected_action_head_2",
                ln=True,
                activation=tf.nn.relu)

            self.feature_action = self.fc_layer(
                self.feature_action_2, self.max_action, "selected_action_head", ln=False, activation=None)

            self.feature_action_mask = self.add_softmax_mask(self.feature_action, self.valid_actions_mask)
            self.action_prob = tf.nn.softmax(self.feature_action_mask, axis=-1) # TODO 检查这里是否有两层 softmax

        with tf.variable_scope("move_degree_head"): # 移动
            # move degree
            # 360 / 30 = 12
            self.move_degree_fc_out = self.fc_layer(
                self.feature_out, self.unit_feature_common_length * 3, "head_move_degree_fc1", ln=True, activation=tf.nn.relu)
            self.head_move_degree_out = self.parameter_head_op(self.move_degree_fc_out, "head_move_degree",
                                                               self.move_x_partition_count)

        with tf.variable_scope("enemy_target_head"): # 正补
            # enemy target selection
            self.enemy_target_fc_out = self.fc_layer(
                self.feature_out,
                int(self.unit_feature_common_length * 3),
                "enemy_target_fc_out",
                ln=True,
                activation=tf.nn.relu)
            #concat mask and data
            enemy_creep_mask = tf.reshape(tf.math.equal(self.feature_unit_categoy_batch, tf.constant(6)), [-1])
            units_mask = tf.reshape(self.units_mask, [-1])
            enemy_creep_mask = tf.cast(enemy_creep_mask, dtype=tf.int32) * \
                                 tf.cast(units_mask, dtype=tf.int32)
            enemy_creep_mask = tf.reshape(tf.cast(enemy_creep_mask, dtype=tf.bool), [-1])

            self.enemy_target_prob_out = self.attetntion_score(self.enemy_target_fc_out, enemy_units_raw_features,
                                                               int(self.unit_feature_common_length * 3),
                                                               "enemy_target_attention", enemy_creep_mask, self.max_unit_count)

        with tf.variable_scope("self_target_head"): # 反补
            # self target selection
            self.self_target_fc_out = self.fc_layer(
                self.feature_out,
                int(self.unit_feature_common_length * 3),
                "self_target_fc_out",
                ln=True,
                activation=tf.nn.relu)

            self_creep_mask = tf.reshape(tf.math.equal(self.feature_unit_categoy_batch, tf.constant(2)), [-1])
            units_mask = tf.reshape(self.units_mask, [-1])
            self_creep_mask = tf.cast(self_creep_mask, dtype=tf.int32) * \
                              tf.cast(units_mask, dtype=tf.int32)
            self_creep_mask = tf.reshape(tf.cast(self_creep_mask, dtype=tf.bool), [-1])

            self.self_target_prob_out = self.attetntion_score(self.self_target_fc_out, self_units_raw_features,
                                                              int(self.unit_feature_common_length * 3), "self_target_attention",
                                                              self_creep_mask, self.max_unit_count)

        with tf.variable_scope("state_value_head"): # Value
            # state value head
            self.state_value_head_fc1 = self.fc_layer(
                self.feature_out, self.unit_feature_common_length * 3, "state_value_fc1", ln=True, activation=tf.nn.relu)
            self.state_value_head_fc2 = self.fc_layer(
                self.state_value_head_fc1, self.unit_feature_common_length, "state_value_fc2", ln=True, activation=tf.nn.relu)

            self.state_value_head_out = self.fc_layer(
                self.state_value_head_fc2, 1, "state_value_head", ln=False, activation=None)

        with tf.variable_scope("compute_general_loss"): # LOSS
            self.R = tf.placeholder(dtype=tf.float32, shape=[None, 1], name='td_target')
            self.old_state_value = tf.placeholder(dtype=tf.float32, shape=[None, 1], name='old_state_value')
            self.clip_range = tf.placeholder(dtype=tf.float32, shape=None, name='clip_range')
            self.action = tf.placeholder(dtype=tf.int32, shape=None, name="action")

            clipped_pred = self.old_state_value + tf.clip_by_value(self.state_value_head_out - self.old_state_value, -0.2, 0.2)
            clipped_vf_loss = tf.square(clipped_pred - self.R)
            unclipped_vf_loss = tf.square(self.state_value_head_out - self.R)
            self.state_value_loss = 1 * tf.maximum(clipped_vf_loss, unclipped_vf_loss)

            # entropy
            self.action_types_entropy = -1 * tf.reduce_sum(tf.multiply(self.action_prob, self.clip_log(self.action_prob)), -1)
            self.action_types_entropy_summary = tf.reduce_mean(self.action_types_entropy)
            valid_mask = tf.cast(tf.equal(self.action_prob, tf.constant(0, dtype=tf.float32)), dtype=tf.int32)
            self.action_types_entropy_gradient = tf.reduce_mean(
                tf.abs(tf.reduce_sum(tf.cast((valid_mask - 1), dtype=tf.float32) * (self.clip_log(self.action_prob) + 1), -1)))

            # 分开计算不同 sub action 的 entropy
            # move entropy
            self.move_degree_entropy, self.move_degree_entropy_summary, self.move_entropy_gradient = \
                self.get_entropy_and_summary(self.action, ACTION_NAME_TO_INDEX["MOVE"], self.head_move_degree_out)

            # attack enemy entropy
            self.attack_enemy_entropy, self.attack_enemy_entropy_summary, self.attack_enemy_entropy_gradient = \
                self.get_entropy_and_summary(self.action, ACTION_NAME_TO_INDEX["ATTACK_ENEMY"], self.enemy_target_prob_out)

            # attack self entropy
            self.attack_deny_entropy, self.attack_deny_entropy_summary, self.attack_deny_entropy_gradient = \
                self.get_entropy_and_summary(self.action, ACTION_NAME_TO_INDEX["ATTACK_SELF"], self.self_target_prob_out)

            self.move_degree_id = tf.placeholder(dtype=tf.int32, shape=None, name="move_degree_id")
            move_degree = tf.one_hot(self.move_degree_id, self.move_x_partition_count)
            self.degree_prob = tf.reduce_sum(tf.multiply(self.head_move_degree_out, move_degree), axis=-1, keepdims=True)

            self.enemy_target_id = tf.placeholder(dtype=tf.int32, shape=None, name="enemy_target_id")
            # adding enemy hero and enemy tower in the end
            enemy_target = tf.one_hot(self.enemy_target_id, self.max_unit_count)
            self.enemy_target_prob = tf.reduce_sum(
                tf.multiply(self.enemy_target_prob_out, enemy_target), axis=-1, keepdims=True)

            self.self_target_id = tf.placeholder(dtype=tf.int32, shape=None, name="self_target_id")
            self_target = tf.one_hot(self.self_target_id, self.max_unit_count)
            self.self_target_prob = tf.reduce_sum(tf.multiply(self.self_target_prob_out, self_target), axis=-1, keepdims=True)

            # 这里相当于 merge 了所有(移动，正补，反补)的 sub action prob
            self.sub_prob = tf.where(
                tf.math.equal(self.action, tf.constant(ACTION_NAME_TO_INDEX["ATTACK_SELF"])), self.self_target_prob,
                tf.ones_like(self.self_target_prob))
            self.sub_prob = tf.where(
                tf.math.equal(self.action, tf.constant(ACTION_NAME_TO_INDEX["ATTACK_ENEMY"])), self.enemy_target_prob,
                self.sub_prob)
            self.sub_prob = tf.where(
                tf.math.equal(self.action, tf.constant(ACTION_NAME_TO_INDEX["MOVE"])), self.degree_prob, self.sub_prob)

        with tf.name_scope('compute_RL_gradient'):
            self.ADV = tf.placeholder(dtype=tf.float32, shape=[None, 1], name="advantage")
            self.old_action_prob = tf.placeholder(dtype=tf.float32, shape=[None, 1], name="action_prob")
            self.old_sub_action_prob = tf.placeholder(dtype=tf.float32, shape=[None, 1], name="action_sub_prob")
            self.entropy_parameter = tf.placeholder(dtype=tf.float32, shape=None, name="entropy_params")
            self.learning_rate = tf.placeholder(dtype=tf.float32, shape=None, name="learning_rate")

            self.ratio_clipfrac, self.ratio_low_bound_value, self.ratio_low_bound_frac, self.actor_loss = self.action_loss(
                self.action, self.action_prob, self.old_action_prob, self.ADV, self.clip_range, self.low_bound_range,
                self.action_shape)

            # 这里的 sub_prob 已经在上面计算好了
            self.sub_ratio_clipfrac, self.sub_ratio_low_bound_value, self.sub_ratio_low_bound_frac, self.sub_actor_loss = self.action_loss_with_prob(
                self.sub_prob, self.old_sub_action_prob, self.ADV, self.clip_range, self.low_bound_range)

            self.actor_loss = tf.reduce_mean(self.actor_loss)
            self.sub_actor_loss = tf.reduce_mean(self.sub_actor_loss)
            self.state_value_loss = tf.reduce_mean(self.state_value_loss)

            self.move_degree_entropy = tf.reduce_mean(self.move_degree_entropy)
            self.attack_enemy_entropy = tf.reduce_mean(self.attack_enemy_entropy)
            self.attack_deny_entropy = tf.reduce_mean(self.attack_deny_entropy)
            self.action_types_entropy = tf.reduce_mean(self.action_types_entropy)

            self.entropy_loss = self.entropy_parameter * 1.75 * self.move_degree_entropy + \
                      self.entropy_parameter * 0.1 * self.attack_enemy_entropy + \
                      self.entropy_parameter * 0.25 * self.attack_deny_entropy + \
                      self.entropy_parameter * 0.75 * self.action_types_entropy

            self.target_loss = self.actor_loss + self.sub_actor_loss + self.state_value_loss - self.entropy_loss
            # self.all_object = tf.reduce_mean(self.target_loss)

            optimizer = tf.train.AdamOptimizer(self.learning_rate)
            if self.use_distribution:
                import horovod.tensorflow as hvd
                self.hvd_optimizer = hvd.DistributedOptimizer(optimizer)

            # get local gradient
            trainable_params = tf.trainable_variables()
            a_grads_origins = optimizer.compute_gradients(self.target_loss, trainable_params)
            grads, all_vars = zip(*a_grads_origins)

            # get gradient over all gpu
            if self.use_distribution:
                avg_grads = self.hvd_optimizer._allreduce_grads(grads)
            else:
                avg_grads = grads

            # do gradient clip and apply gradient to local model
            clipped_grads, self.a_global_grads = tf.clip_by_global_norm(avg_grads, self.max_grad_norm)
            grads_and_vars = list(zip(clipped_grads, all_vars))
            self.just_train_it = optimizer.apply_gradients(grads_and_vars, global_step=tf.train.get_or_create_global_step())
        self.count_parameters()

    def count_parameters(self):
        parameters = {}
        all_count = 0
        for val in tf.trainable_variables():
            name = val.name.lower()
            shape = val.get_shape()
            count = 1
            for one in shape:
                count *= one.value
            parameters[name] = count
            all_count += count
        print('count_parameters:', all_count, parameters)

    def predict(self, instance, running_mode):
        action_type_distribution, move_degree_distribution, current_state_value, \
            enemy_target, self_target, state_out = \
            self.tf_session.run([self.action_prob,
                                 self.head_move_degree_out,
                                 self.state_value_head_out,
                                 self.enemy_target_prob_out,
                                 self.self_target_prob_out,
                                 self.state_out],
                                feed_dict={
                                    self.valid_actions_mask: np.array(instance.mask).reshape((1, -1)),
                                    self.units_mask: instance.units_mask.reshape(-1, self.max_unit_count),
                                    self.feature_global: np.array(instance.state_gf).reshape((1, -1)),
                                    self.feature_unit_global: np.array(instance.state_ucf).reshape((1, self.max_unit_count, -1)),
                                    self.feature_unit_categoy: np.array(instance.state_ucategory).reshape((1, -1)),
                                    self.lstm_state: instance.lstm_state,
                                    self.feature_map_global: np.array(instance.dota_map).reshape((1, self.nearby_map_fullsize, self.nearby_map_fullsize, 6)),
                                    self.self_pre_action_id: np.array(instance.embedding_dict["self_pre_action"]).reshape(-1),
                                    self.self_anim_type_id: np.array(instance.embedding_dict["self_anim"]).reshape(-1),
                                    self.enemy_action_type_id: np.array(instance.embedding_dict["enemy_action"]).reshape(-1),
                                    self.enemy_anim_type_id: np.array(instance.embedding_dict["enemy_anim"]).reshape(-1),
                                    self.self_pre_move_id: np.array(instance.embedding_dict["self_pre_move_direction"]).reshape(-1),
                                })

        return sampling_action(action_type_distribution[0], move_degree_distribution[0], enemy_target[0], self_target[0],
                               state_out, current_state_value[0], running_mode)

    def predict_batch(self, instances):
        action_type_distribution, move_degree_distribution, current_state_value, \
        enemy_target, self_target, lstm_state_out = \
            self.tf_session.run([self.action_prob,
                                 self.head_move_degree_out,
                                 self.state_value_head_out,
                                 self.enemy_target_prob_out,
                                 self.self_target_prob_out,
                                 self.state_out],
                                feed_dict={
                                    self.valid_actions_mask: instances.mask_list_np,
                                    self.units_mask: instances.units_mask_list_np,
                                    self.feature_global: instances.state_gf_list_np,
                                    self.feature_unit_global: instances.state_ucf_list_np,
                                    self.feature_unit_categoy: instances.state_ucategory_list_np,
                                    self.lstm_state: instances.lstm_state_list_np,
                                    self.feature_map_global: instances.dota_map_list_np,

                                    self.self_pre_action_id: instances.embedding_para_dict_np["self_pre_action"],
                                    self.self_anim_type_id: instances.embedding_para_dict_np["self_anim"],
                                    self.enemy_action_type_id: instances.embedding_para_dict_np["enemy_action"],
                                    self.enemy_anim_type_id: instances.embedding_para_dict_np["enemy_anim"],
                                    self.self_pre_move_id: instances.embedding_para_dict_np["self_pre_move_direction"],
                                })
        result_list = []
        for i in range(instances.len()):
            tmp_dict = {
                "action_type_distribution": action_type_distribution[i],
                "move_degree_distribution": move_degree_distribution[i],
                "enemy_target": enemy_target[i],
                "self_target": self_target[i],
                "lstm_state_out": lstm_state_out[i].reshape(1, -1),
                "current_state_value": current_state_value[i],
                "update_times": self.update_times
            }
            result_list.append(tmp_dict)
        return result_list
