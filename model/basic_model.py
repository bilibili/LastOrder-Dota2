import tensorflow.compat.v1 as tf

tf.disable_v2_behavior()
import tensorflow as tf2
import pickle, time, os
import numpy as np


class BasicModel:

    def __init__(self):
        self.tf_session = None

        self.deserializing_var_placeholder = {}
        self.deserializing_assign_op = []

    def serializing_with_session(self, logger=None, elo_score=-1):
        if logger is not None:
            logger.info("start model serializing")
        with self.tf_session.as_default():
            with self.tf_session.graph.as_default():
                a_vars = tf.trainable_variables()
                a_values = self.tf_session.run(a_vars)
                a_var_values = {}
                for index, vars in enumerate(a_vars):
                    a_var_values[vars.name] = a_values[index]

                a_var_values['model_time'] = int(time.time())
                a_var_values['elo'] = elo_score
                raw_model = pickle.dumps(a_var_values)
                p = raw_model # zlib.compress(raw_model, level=4)
                # if logger is not None:
                #     logger.info("before compress, model size %d" % sys.getsizeof(raw_model))
                #     logger.info("after compress, model size %d" % sys.getsizeof(p))
                return p, a_var_values['model_time']

    # 序列化模型
    def serializing(self, update_times=0, logger=None, elo_score=-1):
        if logger is not None:
            logger.info("start model serializing")
        a_vars = tf.trainable_variables()
        a_values = self.tf_session.run(a_vars)
        a_var_values = {}
        for index, vars in enumerate(a_vars):
            a_var_values[vars.name] = a_values[index]

        a_var_values['model_time'] = int(time.time())
        a_var_values['update_times'] = update_times
        a_var_values['elo'] = elo_score
        raw_model = pickle.dumps(a_var_values)
        p = raw_model # zlib.compress(raw_model, level=4)
        # if logger is not None:
        #     logger.info("before compress, model size %d" % sys.getsizeof(raw_model))
        #     logger.info("after compress, model size %d" % sys.getsizeof(p))
        return p, a_var_values['model_time']

    # 反序列化模型
    def deserializing(self, AC_total):
        model_time = None
        update_times = None
        if type(AC_total) is not dict:
            AC_total = pickle.loads(AC_total)

        if "update_times" in AC_total:
            update_times = AC_total['update_times']
            AC_total.pop('update_times')

        if 'model_time' in AC_total:
            model_time = AC_total['model_time']
            AC_total.pop('model_time')

        if 'elo' in AC_total:
            AC_total.pop('elo')

        if 'global_step:0' in AC_total:
            AC_total.pop('global_step:0')

        feed_dict = {}
        with self.tf_session.as_default():
            with self.tf_session.graph.as_default():
                for var_name, value in AC_total.items():
                    if var_name not in self.deserializing_var_placeholder:
                        var = self.tf_session.graph.get_tensor_by_name(var_name)
                        value = np.array(value)
                        assign_placeholder = tf.placeholder(var.dtype, shape=value.shape)
                        self.deserializing_var_placeholder[var_name] = assign_placeholder
                        self.deserializing_assign_op.append(tf.assign(var, assign_placeholder))
                    feed_dict[self.deserializing_var_placeholder[var_name]] = value
                self.tf_session.run(self.deserializing_assign_op, feed_dict=feed_dict)
        return model_time, update_times

    # 保存 ckp 模型
    def save_ckp_model(self, checkpoint_path):
        saver = tf.train.Saver()
        model_file = os.path.join(checkpoint_path, 'model.ckpt')
        saver.save(self.tf_session, model_file, global_step=tf.train.get_global_step())

    def restore_ckp_model(self, checkpoint_path):
        with self.tf_session.as_default():
            with self.tf_session.graph.as_default():
                latest_ckp = tf.train.latest_checkpoint(checkpoint_path)
                saver = tf.train.Saver()
                saver.restore(self.tf_session, latest_ckp)
                print('load ' + latest_ckp)

    # 加载 checkpoint 模型到 dict 字典中，返回 dict
    def load_ckp_to_dict(self, checkpoint_path):
        value_dict = {}
        latest_ckp = tf.train.latest_checkpoint(checkpoint_path)
        reader = tf.train.NewCheckpointReader(latest_ckp)
        for name in reader.get_variable_to_shape_map():
            value_dict[name + ':0'] = reader.get_tensor(name)
        return value_dict

    def weights_to_dict(self):
        a_vars = tf.trainable_variables()
        a_values = self.tf_session.run(a_vars)
        a_var_values = {}
        for index, vars in enumerate(a_vars):
            a_var_values[vars.name] = a_values[index]
        return a_var_values

    def fc_layer(self, inputs, num, scope, ln=True, activation=None, kernel_initializer=tf2.initializers.GlorotUniform()):
        with tf.variable_scope(scope):
            fc_raw = tf.layers.dense(
                inputs=inputs,
                units=num,
                activation=None,
                kernel_initializer=kernel_initializer,
                bias_initializer=tf.constant_initializer(0.))
            if ln == True:
                layer_norm_out = self.layer_norm(fc_raw, num, scope + "_ln")
                return activation(layer_norm_out)
            else:
                if activation == None:
                    return fc_raw
                else:
                    return activation(fc_raw)

    def cnn_layer(self, image, scope):
        with tf.variable_scope('cnn_' + scope):
            c1 = self.conv(image, out_channel=8, kernel=3, scope='c1')
            c2 = self.conv(c1, out_channel=16, kernel=3, scope='c2')
            c2_shape = np.prod([v.value for v in c2.get_shape()[1:]])
            reshape_c2 = tf.reshape(c2, [-1, c2_shape])
            c2_fc = self.fc_layer(inputs=reshape_c2, num=32, scope=scope + '_fc', ln=True, activation=tf.nn.relu)
            return c2_fc

    def conv(self, image, out_channel, kernel, scope=''):
        with tf.variable_scope('conv_' + scope):
            channel_shape = image.get_shape()[3].value
            w = tf.get_variable('w', [kernel, kernel, channel_shape, out_channel], initializer=tf.constant_initializer(1.0))
            b = tf.get_variable('b', out_channel, initializer=tf.constant_initializer(0.0))
            conv = tf.nn.conv2d(image, w, strides=[1, 2, 2, 1], padding='SAME') + b
            pool = tf.nn.max_pool(tf.nn.relu(conv), [1, 2, 2, 1], [1, 1, 1, 1], padding='SAME')
            return pool

    def layer_norm(self, inputs, shape, scope):
        with tf.variable_scope(scope):
            mean, variance = tf.nn.moments(inputs, [1], keep_dims=True)
            normalised_input = (inputs - mean) / tf.sqrt(variance + 1e-10)
            # init to make zero mean and unit variance
            gains = tf.get_variable("norm_gain", shape, initializer=tf.constant_initializer(1.))
            biases = tf.get_variable("norm_bias", shape, initializer=tf.constant_initializer(0.))
            return normalised_input * gains + biases

    # only can used at one unit per UNIT_NUM scene
    def get_type_fc_and_mask(self, input, unit_categoy_batch, unit_attack_mask, type_constant, scope, layer_size):
        feature_bool_mask = tf.reshape(tf.math.equal(unit_categoy_batch, tf.constant(type_constant)), [-1])
        feature_raw = tf.boolean_mask(input, feature_bool_mask)
        mask = tf.boolean_mask(unit_attack_mask, feature_bool_mask)
        mask = tf.cast(mask, dtype=tf.int32)
        feature_raw = self.fc_layer(feature_raw, layer_size, scope, ln=True, activation=tf.nn.relu)
        return feature_raw, mask

    def softmax_over_valid_position(self, input_tensor, valid_mask):
        self.all_exp_values = tf.exp(input_tensor)
        self.valid_exp_values = tf.multiply(self.all_exp_values, valid_mask)
        # if no valid action, denominator is zero
        self.valid_denominator = tf.reduce_sum(self.valid_exp_values, axis=-1, keep_dims=True)
        return self.valid_exp_values / self.valid_denominator

    def parameter_head_op(self, input_tensor, scope, fc_size):
        with tf.variable_scope(scope):
            feature_head_out = self.fc_layer(input_tensor, fc_size, scope, ln=False, activation=None)
            return tf.nn.softmax(feature_head_out, axis=-1)

    def embedding_op(self, input_tensor, scope, type_embedding_name, type_total_count, type_embedding_size):
        with tf.variable_scope(scope):
            input_tensor = tf.reshape(input_tensor, [-1])
            self.feature_type_embedding = tf.get_variable(type_embedding_name, [type_total_count, type_embedding_size])
            self.embedded_type = tf.nn.embedding_lookup(self.feature_type_embedding, input_tensor)
            return self.embedded_type

    # mask[true, false,....]
    def add_softmax_mask(self, input_tensor, mask_tensor):
        int_mask = tf.cast(mask_tensor, dtype=tf.int32)
        self.scaled_input_tensor = input_tensor + tf.cast((1 - int_mask) * -10000000, dtype=tf.float32)
        return self.scaled_input_tensor

    # action entropy
    def action_entropy(self, p):
        return -1 * tf.reduce_sum(tf.multiply(p, self.clip_log(p)), -1, keep_dims=True)

    def attention_with_fc(self, query, key_raw, layer_size, scope, feature_unit_categoy_batch, attention_unit_type, layersize_1,
                          laysize_2):
        with tf.variable_scope(scope):
            feature_bool_mask = tf.reshape(tf.math.equal(feature_unit_categoy_batch, tf.constant(attention_unit_type)), [-1])
            feature_filtered_unit = tf.where(feature_bool_mask, key_raw, tf.zeros_like(key_raw))

            feature_fitered_fc1_out = self.fc_layer(
                feature_filtered_unit, layersize_1, scope + "_fc1", ln=True, activation=tf.nn.relu)
            feature_fitered_fc2_out = self.fc_layer(
                feature_fitered_fc1_out, laysize_2, scope + "_fc2", ln=True, activation=tf.nn.relu)

            attention_1, _ = self.attention_op(query, feature_fitered_fc2_out, layer_size, scope + "_attention_1",
                                               feature_unit_categoy_batch, attention_unit_type)
            attention_2, _ = self.attention_op(query, feature_fitered_fc2_out, layer_size, scope + "_attention_2",
                                               feature_unit_categoy_batch, attention_unit_type)
            attention_3, _ = self.attention_op(query, feature_fitered_fc2_out, layer_size, scope + "_attention_3",
                                               feature_unit_categoy_batch, attention_unit_type)
            attention_concat = tf.concat([attention_1, attention_2, attention_3], axis=-1)
            return attention_concat, feature_fitered_fc2_out

    def get_entropy_and_summary(self, train_selected_action_type_id, action_id, raw_input):
        attack_action_mask = tf.math.equal(train_selected_action_type_id, tf.constant(action_id, dtype=tf.int32))
        masked_out = tf.where(attack_action_mask, raw_input, tf.zeros_like(raw_input))
        input_entropy = -1 * tf.reduce_sum(tf.multiply(masked_out, self.clip_log(masked_out)), -1, keepdims=True)

        attack_action_mask.set_shape([None])
        boolen_masked_out = tf.boolean_mask(raw_input, attack_action_mask)
        entropy_summary = tf.reduce_mean(-1 *
                                         tf.reduce_sum(tf.multiply(boolen_masked_out, self.clip_log(boolen_masked_out)), -1))

        valid_mask = tf.cast(tf.equal(boolen_masked_out, tf.constant(0, dtype=tf.float32)), dtype=tf.int32)
        entropy_gradient = tf.reduce_mean(
            tf.abs(tf.reduce_sum(tf.cast((valid_mask - 1), dtype=tf.float32) * (self.clip_log(boolen_masked_out) + 1), -1)))
        return input_entropy, entropy_summary, entropy_gradient

    # query shape:(batch, d), key shape(batch, key_num, d), mask shape (batch, key_num)
    def attention_op(self, query, key_raw, layer_size, scope, feature_unit_categoy_batch, attention_unit_type):
        with tf.variable_scope(scope):
            self.feature_bool_mask = tf.reshape(
                tf.math.equal(feature_unit_categoy_batch, tf.constant(attention_unit_type)), [-1])
            self.feature_filtered_unit = tf.where(self.feature_bool_mask, key_raw, tf.zeros_like(key_raw))
            self.reshaped_key = self.feature_filtered_unit

            self.query_encoded = self.fc_layer(query, layer_size, scope + "_query_layer", ln=False, activation=tf.nn.relu)
            # reshape to [batch, 1, d]
            self.query_encoded = tf.reshape(self.query_encoded, [-1, 1, layer_size])

            self.key_encoded = self.fc_layer(
                self.reshaped_key, layer_size, scope + "_key_layer", ln=False, activation=tf.nn.relu)
            # reshape to [batch, max_unit_count, layer_size]
            self.key_encoded = tf.reshape(self.key_encoded, [-1, self.max_unit_count, layer_size])

            self.value_encoded = self.fc_layer(
                self.reshaped_key, layer_size, scope + "_value_layer", ln=False, activation=tf.nn.relu)
            self.value_encoded = tf.reshape(self.value_encoded, [-1, self.max_unit_count, layer_size])

            # scaled dot-product attention
            layer_weight = tf.constant(layer_size, dtype=tf.float32)

            self.query_key_dot_elmentwise = tf.multiply(self.query_encoded, self.key_encoded)
            self.query_key_dot = tf.reduce_sum(
                self.query_key_dot_elmentwise, axis=-1, keepdims=False) / tf.math.sqrt(layer_weight)

            self.feature_bool_mask = tf.reshape(self.feature_bool_mask, [-1, self.max_unit_count])
            self.query_key_dot_scaled = self.add_softmax_mask(self.query_key_dot, self.feature_bool_mask)
            self.query_key_score = tf.nn.softmax(self.query_key_dot_scaled, axis=-1)
            self.query_key_score = tf.reshape(self.query_key_score, [-1, self.max_unit_count, 1])

            self.weighted_value_element = tf.multiply(self.query_key_score, self.value_encoded)
            # reduce sum over different key, result shape: [batch, layer_size]
            self.attention_value = tf.reduce_sum(self.weighted_value_element, axis=1, keepdims=False)
            return self.attention_value, self.query_key_score

    def attetntion_score(self, query, key_raw, layer_size, scope, units_mask, unit_num):
        with tf.variable_scope(scope):
            self.feature_bool_mask = units_mask

            self.feature_filtered_unit = tf.where(self.feature_bool_mask, key_raw, tf.zeros_like(key_raw))
            self.reshaped_key = self.feature_filtered_unit

            if self.use_dropout:
                query = tf.nn.dropout(query, self.dropout_keep_prob)
            self.query_encoded = self.fc_layer(query, layer_size, scope + "_query_layer", ln=False, activation=tf.nn.relu)
            # reshape to [batch, 1, d]
            self.query_encoded = tf.reshape(self.query_encoded, [-1, 1, layer_size])

            if self.use_dropout:
                self.reshaped_key = tf.nn.dropout(self.reshaped_key, self.dropout_keep_prob)
            self.key_encoded = self.fc_layer(
                self.reshaped_key, layer_size, scope + "_key_layer", ln=False, activation=tf.nn.relu)

            # reshpa to [batch, key_num, d]
            self.key_encoded = tf.reshape(self.key_encoded, [-1, unit_num, layer_size])

            # scaled dot-product attention
            layer_weight = tf.constant(layer_size, dtype=tf.float32)
            self.query_key_dot_elmentwise = tf.multiply(self.query_encoded, self.key_encoded)
            self.query_key_dot = tf.reduce_sum(
                self.query_key_dot_elmentwise, axis=-1, keepdims=False) / tf.math.sqrt(layer_weight)

            self.feature_bool_mask = tf.reshape(self.feature_bool_mask, [-1, unit_num])
            self.query_key_dot_scaled = self.add_softmax_mask(self.query_key_dot, self.feature_bool_mask)
            self.query_key_score = tf.nn.softmax(self.query_key_dot_scaled, axis=-1)
            return self.query_key_score

    def mask_fc_max_pool_op(self, input_tensor, feature_unit_categoy_batch, scope, constant_value, fc_lay1_size, fc_lay2_size):
        with tf.variable_scope(scope):
            self.feature_bool_mask = tf.reshape(tf.math.equal(feature_unit_categoy_batch, tf.constant(constant_value)), [-1])
            self.feature_filtered_unit = tf.where(self.feature_bool_mask, input_tensor, tf.zeros_like(input_tensor))

            self.feature_fitered_fc1_out = self.fc_layer(
                self.feature_filtered_unit, fc_lay1_size, scope + "_fc1", ln=True, activation=tf.nn.relu)
            feature_fitered_fc2_out = self.fc_layer(
                self.feature_fitered_fc1_out, fc_lay2_size, scope + "_fc2", ln=True, activation=tf.nn.relu)

            feature_unit_all_feature_trans = tf.reshape(feature_fitered_fc2_out, [-1, self.max_unit_count, fc_lay2_size])
            feature_max_indices = tf.argmax(feature_unit_all_feature_trans, axis=1)
            indices = tf.cast(feature_max_indices, dtype=tf.int32)
            indices_shape = tf.shape(feature_max_indices)

            R, _ = tf.meshgrid(tf.range(indices_shape[0]), tf.range(indices_shape[1]), indexing='ij')
            coords = tf.stack([R, indices], axis=2)

            # shape [batch, fc_lay2_size, fc_lay2_size]
            feature_unit_all_feature_trans_selected = tf.gather_nd(feature_unit_all_feature_trans, coords)
            feature_unit_all_feature_trans_selected = tf.reshape(feature_unit_all_feature_trans_selected,
                                                                 [-1, fc_lay2_size * fc_lay2_size])
            return feature_unit_all_feature_trans_selected, feature_fitered_fc2_out

    # input tensor shape [batch_size, unit_size, XX_size, feature]
    def max_pool_structure(self, input_tensor, scope, type_embedding_name, type_total_count, type_embedding_size,
                           type_max_shape, type_origin_feature_count, fc_lay1_size, fc_lay2_size):
        # no reuse for different max_pool_structure
        with tf.variable_scope(scope):
            # reshape to 1D tensor
            self.feature_type = input_tensor[:, :, :, 0]
            self.feature_other = input_tensor[:, :, :, 1:]
            # embedding
            self.feature_embedded_type = self.embedding_op(self.feature_type, type_embedding_name + "_embedding",
                                                           type_embedding_name, type_total_count + 1, type_embedding_size)

            self.feature_embedded_type = tf.reshape(self.feature_embedded_type,
                                                    [-1, self.max_unit_count, type_max_shape, type_embedding_size])
            # concat along -1 axis
            self.feature_other = tf.cast(self.feature_other, tf.float32)
            self.feature_concat = tf.concat([self.feature_embedded_type, self.feature_other], -1)
            self.feature_fc1_input = tf.reshape(self.feature_concat, [-1, type_origin_feature_count - 1 + type_embedding_size])

            self.feature_fc1_output = tf.layers.dense(
                inputs=self.feature_fc1_input,
                units=fc_lay1_size,
                activation=tf.nn.relu,
                kernel_initializer=tf.contrib.layers.xavier_initializer(),
                bias_initializer=tf.constant_initializer(0.))

            # input shape must be 3-dimension
            self.feature_input_max_pool = tf.reshape(self.feature_fc1_output, [-1, type_max_shape, fc_lay1_size])
            # return shape[-1, 1, fc_lay2_size]
            self.feature_max_pool = tf.layers.max_pooling1d(self.feature_input_max_pool, type_max_shape, 1)
            self.feature_out = tf.reshape(self.feature_max_pool, [-1, self.max_unit_count, fc_lay1_size])
            return self.feature_out

    def clip_log(self, x, min=1e-4, max=1.0):
        return tf.log(tf.clip_by_value(x, min, max))

    def get_l2_loss(self, x):
        return tf.add_n([tf.nn.l2_loss(v) for v in x if 'bias' not in v.name])

    def exp_log(self, x):
        return tf.log(tf.exp(x))

    def batch_to_seq(self, h, nsteps):
        layer_size = h.get_shape()[-1]
        h = tf.reshape(h, [-1, nsteps, layer_size])
        return [tf.squeeze(v, [1]) for v in tf.split(axis=1, num_or_size_splits=nsteps, value=h)]

    def seq_to_batch(self, h):
        shape = h[0].get_shape().as_list()
        assert (len(shape) > 1)
        nh = h[0].get_shape()[-1].value
        return tf.reshape(tf.concat(axis=1, values=h), [-1, nh])

    def openai_lstm(self, xs, s, scope, nh, init_scale=1.0):
        nbatch, nin = [v.value for v in xs[0].get_shape()]
        with tf.variable_scope(scope):
            wx = tf.get_variable("wx", shape=[nin, nh * 4], initializer=self.ortho_init)
            wh = tf.get_variable("wh", shape=[nh, nh * 4], initializer=self.ortho_init)
            b = tf.get_variable("b", [nh * 4], initializer=tf.constant_initializer(0.0))

            c, h = tf.split(axis=1, num_or_size_splits=2, value=s) # 上一步的 memory cell 和 output
            for idx, x in enumerate(xs):
                # c = c*(1-m)
                # h = h*(1-m)
                z = tf.matmul(x, wx) + tf.matmul(h, wh) + b
                i, f, o, u = tf.split(axis=1, num_or_size_splits=4, value=z)
                i = tf.nn.sigmoid(i) # 输入
                f = tf.nn.sigmoid(f) # 遗忘
                o = tf.nn.sigmoid(o) # 输出
                u = tf.tanh(u)
                c = f * c + i * u
                h = o * tf.tanh(c)
                xs[idx] = h
            s = tf.concat(axis=1, values=[c, h]) # 上一步的 memory cell 和 output
            return xs, s

    def ortho_init(self, shape, dtype, partition_info):
        # lasagne ortho init for tf
        shape = tuple(shape)
        if len(shape) == 2:
            flat_shape = shape
        elif len(shape) == 4: # assumes NHWC
            flat_shape = (np.prod(shape[:-1]), shape[-1])
        else:
            raise NotImplementedError
        a = np.random.normal(0.0, 1.0, flat_shape)
        u, _, v = np.linalg.svd(a, full_matrices=False)
        q = u if u.shape == flat_shape else v # pick the one with the correct shape
        q = q.reshape(shape)
        return (1.0 * q[:shape[0], :shape[1]]).astype(np.float32)

    def action_head(self, inputs, input_mask, layer_size, output_shape, scope):
        logits = self.fc_layer(inputs, output_shape, scope + '_fc1', ln=False, activation=None)
        head_masked = self.add_softmax_mask(logits, input_mask)
        prob = tf.nn.softmax(head_masked, axis=-1)
        entropy = self.action_entropy(prob)
        return prob, entropy, head_masked, logits

    def action_loss(self, action, prob, old_prob, adv, clip_range, low_bound_range, output_shape):
        action_one_hot = tf.one_hot(action, output_shape)
        action_prob = tf.reduce_sum(tf.multiply(prob, action_one_hot), axis=-1, keepdims=True)

        return self.action_loss_with_prob(action_prob, old_prob, adv, clip_range, low_bound_range)

    def action_loss_with_prob(self, action_prob, old_prob, adv, clip_range, low_bound_range):
        ratio = action_prob / old_prob
        unclipped_actor_loss = adv * ratio
        clipped_actor_loss = adv * tf.clip_by_value(ratio, 1.0 - clip_range, 1.0 + clip_range)
        actor_loss_raw = -1 * tf.minimum(unclipped_actor_loss, clipped_actor_loss)

        clip_mask = tf.cast(tf.greater(tf.abs(ratio - 1.0), clip_range), tf.float32)
        clipfrac = tf.reduce_mean(clip_mask)

        less_than_zero_mask = tf.less(adv, 0)
        ration_mask = tf.greater(ratio, low_bound_range)
        low_bound_mask = tf.logical_and(less_than_zero_mask, ration_mask)
        low_bound_frac = tf.reduce_mean(tf.cast(low_bound_mask, tf.float32))
        low_bound_mask.set_shape([None])
        low_bound_value_raw = tf.reduce_mean(tf.boolean_mask(ratio, low_bound_mask))
        low_bound_value = tf.where(tf.is_nan(low_bound_value_raw), 0.0, low_bound_value_raw)

        actor_loss = tf.where(low_bound_mask, -low_bound_range * adv, actor_loss_raw)
        return clipfrac, low_bound_value, low_bound_frac, actor_loss
