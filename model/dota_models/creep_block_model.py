import tensorflow.compat.v1 as tf
import numpy as np
import random
import time
from model.basic_model import BasicModel
from model.cb_features import CBFeature

#instance复用主模型，所以action的mask也复用主的，但是mask只放开move和stop，所以在action头上的entropy就先不考虑了
#unit数量保持一致，所以第一层的特征层和主模型保持一致，稍微使模型容量扩大有点浪费


class CreepBlockModel(BasicModel):

    def __init__(self, use_distribution=False, is_creep_block=False):
        BasicModel.__init__(self)
        self.start_training_time = time.time()
        self.use_distribution = use_distribution
        self.unit_feature_common_length = 50
        self.move_x_partition_count = 13
        if self.use_distribution:
            import horovod.tensorflow as hvd

        self.feature_global_num = len(CBFeature.current_global_features)
        self.feature_unit_common_num = len(CBFeature.current_unit_common_features)
        self.max_unit_count = 5

        # with tf.device('/device:GPU:2'):
        with tf.variable_scope("creep_block"):
            # index from 1: attack, move, ability(4), stop, do_nothing
            # move, stop, do_nothing
            # valid action filled with 1 on corresponding position, invalid action filled with 0
            self.feature_global = tf.placeholder(tf.float32, [None, self.feature_global_num], name="input_feature_global")
            self.feature_global_tran = self.fc_layer(
                self.feature_global, 15, "feature_global_tran", ln=True, activation=tf.nn.relu)
            # shape = [batch size, max_unit_count, feature]
            self.feature_unit_global = tf.placeholder(
                tf.float32, [None, self.max_unit_count, self.feature_unit_common_num], name="input_feature_unit_global")
            # five unit categoy
            self.feature_unit_categoy = tf.placeholder(tf.int32, [None, self.max_unit_count], name="input_feature_unit_categoy")
            self.feature_unit_categoy_batch = tf.reshape(self.feature_unit_categoy, [-1, 1])

            # unit feature
            self.feature_unit_global_tran = tf.reshape(self.feature_unit_global, [-1, self.feature_unit_common_num])

            self.feature_unit_all_feature = self.fc_layer(
                self.feature_unit_global_tran,
                self.unit_feature_common_length * 4,
                "feature_unit_global_tran_1",
                ln=True,
                activation=tf.nn.relu)

            common_unit_out_size = int(self.unit_feature_common_length * 1.6)

            #get self hero
            feature_bool_mask = tf.reshape(tf.math.equal(self.feature_unit_categoy_batch, tf.constant(4)), [-1])
            feature_raw = tf.boolean_mask(self.feature_unit_all_feature, feature_bool_mask)
            self.self_hero_feature = self.fc_layer(
                feature_raw, common_unit_out_size, 'self_hero_feature_raw', ln=True, activation=tf.nn.relu)
            #get self creep
            self.feature_bool_mask = tf.reshape(tf.math.equal(self.feature_unit_categoy_batch, tf.constant(2)), [-1])
            self_units_maxpool_features, self_units_raw_features = self.attention_with_fc(
                self.self_hero_feature, self.feature_unit_all_feature,
                int(self.unit_feature_common_length * 0.5), "self_attention", self.feature_unit_categoy_batch, 2,
                int(self.unit_feature_common_length * 0.9), common_unit_out_size)

            self.feature_concat_raw = tf.concat([self.feature_global_tran, self_units_maxpool_features, self.self_hero_feature],
                                                axis=-1)

            self.feature_out = self.fc_layer(
                self.feature_concat_raw,
                self.unit_feature_common_length * 2,
                "feature_concat_raw",
                ln=True,
                activation=tf.nn.relu)
            # move degree
            # 360 / 30 = 12
            self.move_degree_fc_out = self.fc_layer(
                self.feature_out,
                int(self.unit_feature_common_length * 1.5),
                "head_move_degree_fc1",
                ln=True,
                activation=tf.nn.relu)
            self.head_move_degree_out = self.parameter_head_op(self.move_degree_fc_out, "head_move_degree",
                                                               self.move_x_partition_count)

            # state value head
            self.state_value_head_fc1 = self.fc_layer(
                self.feature_out, int(self.unit_feature_common_length * 1.5), "state_value_fc1", ln=True, activation=tf.nn.relu)

            self.state_value_head_fc2 = self.fc_layer(
                self.state_value_head_fc1, self.unit_feature_common_length, "state_value_fc2", ln=True, activation=tf.nn.relu)
            self.state_value_head_out = self.fc_layer(
                self.state_value_head_fc2, 1, "state_value_head", ln=False, activation=None)

        with tf.variable_scope("compute_general_loss"):
            # state value loss
            self.td_target = tf.placeholder(dtype=tf.float32, shape=[None, 1], name='td_target')
            self.old_state_value = tf.placeholder(dtype=tf.float32, shape=[None, 1], name='old_state_value')
            self.clip_range = tf.placeholder(dtype=tf.float32, shape=None, name='clip_range')

            clipped_pred = self.old_state_value + tf.clip_by_value(self.state_value_head_out - self.old_state_value, -0.2, 0.2)
            clipped_vf_loss = tf.square(clipped_pred - self.td_target)
            unclipped_vf_loss = tf.square(self.state_value_head_out - self.td_target)
            #not overfitting current instances
            self.state_value_loss = tf.maximum(clipped_vf_loss, unclipped_vf_loss)

            self.train_selected_move_degree_id = tf.placeholder(dtype=tf.int32, shape=None)
            self.train_selected_move_degree = tf.one_hot(self.train_selected_move_degree_id, self.move_x_partition_count)
            self.train_move_action_degree_prob = tf.reduce_sum(
                tf.multiply(self.head_move_degree_out, self.train_selected_move_degree), axis=-1, keepdims=True)
            # entropy = action entropy + move degree entropy

            # move entropy
            self.move_degree_entropy = -1 * tf.reduce_sum(
                tf.multiply(self.head_move_degree_out, self.clip_log(self.head_move_degree_out)), -1)
            self.move_degree_entropy_summary = tf.reduce_mean(self.move_degree_entropy)
            self.entropy_object = self.move_degree_entropy

            self.entropy_parameter = tf.placeholder(dtype=tf.float32, shape=None)

            self.learning_rate = tf.placeholder(dtype=tf.float32, shape=None)

            self.params = tf.trainable_variables()

            with tf.name_scope('compute_RL_gradient'):
                self.train_actor_advantage = tf.placeholder(dtype=tf.float32, shape=[None, 1])
                self.train_actor_advantage_clipped = self.train_actor_advantage
                self.old_action_prob = tf.placeholder(dtype=tf.float32, shape=[None, 1])

                ration = self.train_move_action_degree_prob / self.old_action_prob
                unclipped_actor_loss = self.train_actor_advantage_clipped * ration
                clipped_actor_loss = self.train_actor_advantage_clipped * tf.clip_by_value(
                    ration, 1.0 - self.clip_range, 1.0 + self.clip_range)
                clip_mask = tf.cast(tf.greater(tf.abs(ration - 1.0), self.clip_range), tf.float32)
                self.clipfrac = tf.reduce_mean(clip_mask)

                low_bound_range = 1.2
                less_than_zero_mask = tf.less(self.train_actor_advantage, 0)
                ration_mask = tf.greater(ration, low_bound_range)
                low_bound_mask = tf.logical_and(less_than_zero_mask, ration_mask)
                self.low_bound_frac = tf.reduce_mean(tf.cast(low_bound_mask, tf.float32))
                low_bound_mask.set_shape([None])
                low_bound_value_raw = tf.reduce_mean(tf.boolean_mask(ration, low_bound_mask))
                self.low_bound_value = tf.where(tf.is_nan(low_bound_value_raw), 0., low_bound_value_raw)

                self.actor_loss_raw = -1 * tf.minimum(unclipped_actor_loss, clipped_actor_loss)
                self.actor_loss = tf.where(low_bound_mask, tf.zeros_like(self.actor_loss_raw), self.actor_loss_raw)
                self.target_loss = self.actor_loss + 0.5 * self.state_value_loss - self.entropy_parameter * self.entropy_object
                self.all_object = tf.reduce_mean(self.target_loss)
                self.optimizer = tf.train.AdamOptimizer(self.learning_rate)
                if self.use_distribution:
                    self.hvd_optimizer = hvd.DistributedOptimizer(self.optimizer)
                # gradient clip
                a_grads_origins = self.optimizer.compute_gradients(self.all_object, self.params)
                grads, vars = zip(*a_grads_origins)
                if self.use_distribution:
                    avg_grads = self.hvd_optimizer._allreduce_grads(grads)
                else:
                    avg_grads = grads

                clipped_grads, self.a_global_grads = tf.clip_by_global_norm(avg_grads, 2.0)
                grads_and_vars = list(zip(clipped_grads, vars))
                self.a_server_apply_gradients = self.optimizer.apply_gradients(
                    grads_and_vars, global_step=tf.train.get_or_create_global_step())

    def learn(self,
              data,
              data1,
              queue,
              sess=None,
              epoch_time=5,
              mini_batch_size=2048,
              log_basic_info=None,
              need_log=False,
              mpi_rank=-1,
              sgd_epoch_percent=1,
              sgd_max_instances=10000,
              per_epoch_sgd_time=2,
              entropy_parameter_init=None,
              lr_decay=None,
              ts_mean=None,
              ts_std=None):
        #非lstm，和baseline里标准版不一样，dataset是滚动更新的，所以不需要讲全部data选取
        s_time = time.time()
        inds = np.arange(data.len())
        for i in range(8):
            np.random.shuffle(inds)
            # minibatch_index=inds
            end = int(mini_batch_size)

            minibatch_index = inds[0:end]
            slice = data.slice(minibatch_index, 1)
            if mpi_rank == 0:
                log_basic_info.info("slice time %f" % (time.time() - s_time))
            log_summary = mpi_rank <= 4
            self.sgd(
                slice,
                queue,
                self.start_training_time,
                sess=sess,
                need_log=log_summary,
                log_basic_info=log_basic_info,
                entropy_parameter_init=entropy_parameter_init,
                lr_decay=lr_decay,
                ts_mean=ts_mean,
                ts_std=ts_std)
        if mpi_rank == 0:
            queue({'sgd_round_per_min': [1, time.time()]})
            log_basic_info.info("sgd end")

    def sgd(self,
            training_instances,
            queue,
            start_training_time,
            sess=None,
            need_log=False,
            log_basic_info=None,
            entropy_parameter_init=None,
            lr_decay=None,
            ts_mean=None,
            ts_std=None):
        if entropy_parameter_init == None:
            entropy_parameter_init = 0.001 * 10
        if lr_decay == None:
            lr_decay = 1 * 0.00025 / 25
        if ts_mean is None:
            batch_advantage = (training_instances["gae_advantage_list"] - training_instances["gae_advantage_list"].mean()) / \
                              (training_instances["gae_advantage_list"].std() + 1e-10)
        else:
            batch_advantage = (training_instances["gae_advantage_list"] - ts_mean) / (ts_std + 1e-10)

        _,  degree, \
            move_degree_entropy,  \
            state_value_loss, actor_loss, a_global_grads, lr_decay, clipfrac, \
            low_bound_frac, low_bound_value= \
            self.tf_session.run([self.a_server_apply_gradients,
                      self.head_move_degree_out,
                        self.move_degree_entropy_summary,
                      self.state_value_loss,
                      self.actor_loss,
                      self.a_global_grads,
                      self.learning_rate,
                      self.clipfrac,
                      self.low_bound_frac,
                      self.low_bound_value,
                      ],

                     feed_dict={
                self.feature_global: training_instances["state_gf_list"],
                self.feature_unit_global: training_instances["state_ucf_list"],
                self.feature_unit_categoy: training_instances["state_ucategory_list"],
                self.train_selected_move_degree_id: training_instances["action_para_MOVE"],
                self.train_actor_advantage: batch_advantage,
                self.td_target: training_instances["q_reward_list"],
                self.old_state_value: training_instances["state_value_list"],
                self.old_action_prob: training_instances["sub_action_prob_list"],
                self.clip_range: 0.2,
                self.entropy_parameter: entropy_parameter_init,
                self.learning_rate: lr_decay,
                     })

        if need_log:
            queue({
                'entropy_parameter': entropy_parameter_init,
                'avg_advantage_value': np.average(training_instances["gae_advantage_list"]),
                'avg_Q_value': np.average(training_instances["q_reward_list"]),
                'learning_rate': lr_decay,
                'avg_state_value_loss': sum([i[0] for i in state_value_loss]) / len(state_value_loss),
                'avg_actor_loss': sum([i[0] for i in actor_loss]) / len(actor_loss),
                'avg_move_degree_entropy': move_degree_entropy,
                'actor_global_norm': a_global_grads,
                'clipfrac': clipfrac,
                'avg_choose_prob': np.average(training_instances["action_prob_list"]),
                'max_choose_prob': np.max(training_instances["action_prob_list"]),
                'low_bound_frac': low_bound_frac,
                'low_bound_value': low_bound_value,
                'move_degree_histogram': degree.tolist()
            })

    def predict_batch(self, instances):
        move_degree_distribution, current_state_value = self.tf_session.run(
            [
                self.head_move_degree_out,
                self.state_value_head_out,
            ],
            feed_dict={
                self.feature_global: instances.state_gf_list_np,
                self.feature_unit_global: instances.state_ucf_list_np,
                self.feature_unit_categoy: instances.state_ucategory_list_np,
            })
        result_list = []
        for i in range(instances.len()):
            tmp_dict = {
                "move_degree_distribution": move_degree_distribution[i],
                "current_state_value": current_state_value[i],
            }
            result_list.append(tmp_dict)
        return result_list

    def sampling_action(self, move_degree_distribution, current_state_value, running_mode):
        parameter_list = []

        if running_mode in ['self_eval_double', 'self_eval', 'local_test_self', 'local_test_opponent', '']:
            if max(move_degree_distribution) > 0.3:
                degree_index = move_degree_distribution.argmax()
            else:
                degree_index = self.distribution_sampling(move_degree_distribution)
        else:
            degree_index = self.distribution_sampling(move_degree_distribution)
        action_prob = move_degree_distribution[degree_index]
        if degree_index == 12:
            action_index = 1
        else:
            action_index = 0
        parameter_list.append(degree_index * 3)

        return [action_index, parameter_list, current_state_value.tolist()[0], action_prob, move_degree_distribution]

    def predict(self, instance, running_mode):
        move_degree_distribution, current_state_value, = \
            self.tf_session.run([
                                 self.head_move_degree_out,
                                 self.state_value_head_out,
                                 ],
                                feed_dict={
                                    self.feature_global: np.array(instance.state_gf).reshape((1, -1)),
                                    self.feature_unit_global: np.array(instance.state_ucf).reshape((1, self.max_unit_count, -1)),
                                    self.feature_unit_categoy: np.array(instance.state_ucategory).reshape((1, -1)),
                                })

        return self.sampling_action(move_degree_distribution[0], current_state_value[0], running_mode)

    def distribution_sampling(self, distribution, index_range_list=[]):
        r = random.randint(0, 100) + 1e-5
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
            accumulated_prob += 100 * i
            if accumulated_prob >= r:
                return index
        return last_positive_value_index
