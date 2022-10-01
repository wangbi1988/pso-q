# -*- coding: utf-8 -*-
"""
Created on Sun Jul 19 18:08:29 2020

@author: 王碧
"""


from .deep_agents import DeepAgent, tf, tf_util, np, tf_layers;
from .deep_mellow import DeepMellow;

class DeepHesv(object):
    def __init__(self, *params, mode = 'DQN', **args):
        if mode == 'DQN':
            self.model = _DeepHesv(*params, **args);
        elif mode == 'MM':
            self.model = _DeepMellowHesv(*params, **args);
        else:
            raise NotImplementedError();
        
    def __getattr__(self, name):
        return getattr(self.model, name);
    
#    pi = tf.constant(np.pi);
    
    @staticmethod
    @tf.function
    def getHESV(sum_x, var, M):
        return sum_x + tf.sqrt(2 * np.pi * var) / 4 + 2 / 3 * (M**2 * tf.exp(1 - 2 * M**2) * (3 * sum_x - 2 * tf.sqrt(var) * M));

class _dict_for_statistic_info(object):
    def __init__(self):
        self._dict = {};
        
    def getter(self, key):
        if key in self._dict.keys():
            return self._dict[key];
        return -np.inf;
    
    def setter(self, key, value):
        self._dict[key] = value;
        
    @staticmethod
    def query(dict_list, keys):
#        ret = ([],) * len(dict_list);
        ret = [[] for i in range(len(dict_list))];
        if type(keys) is not list:
            for r, d in zip(ret, dict_list):
                r.append(d.getter(keys));
            return ret;
        
        for key in keys:
            tmp = _dict_for_statistic_info.query(dict_list, key);
            for r, t in zip(ret, tmp):
                r.append(t);
        return ret;

class _DeepHesv(DeepAgent):
    def __init__(self, *params, **args):
        super(_DeepHesv, self).__init__(*params, **args);
        self.interval_table_for_sum = _dict_for_statistic_info();
        self.interval_table_for_max = _dict_for_statistic_info();
        self.interval_table_for_min = _dict_for_statistic_info();
        self.interval_table_for_count = _dict_for_statistic_info();
    
    def update(self, trajectories, gamma, *params, lr_factor = 1.0, **args):
#        trajectory = trajectories[-1];
#        (obs, reward, next_obs, done) = trajectory.values;
#        self.count(([obs[0]],));
#        self.count(([obs[0]], [obs[1]]));
        return super(_DeepHesv, self).update(trajectories, gamma, *params, lr_factor = lr_factor, **args);
    
    def _do_update(self, experience, *params,  **args):
#                    self._obs_ph,
#                    act_t_ph,
#                    rew_t_ph,
#                    self._nobs_ph,
#                    done_mask_ph,
#                    self.config.prob_ph,
#                    self.config.omega_ph,
#                    self.config.varsigma_ph,
#                    self.config.c_ph,
#                    hesv_obs_count_t_ph,
#                    hesv_next_obs_count_t_ph,
#                    hesv_sum_t_ph,
#                    hesv_max_t_ph,
#                    hesv_min_t_ph,
#            
        (obses_t, actions, rewards, obses_tp1, dones) = experience;
        
#        need the input of counts 
#        keys = self.obs2index((obses_t, actions));
#        hesv_obs_count_t_ph = np.asarray(self.count_list.query(keys));
        keys_tp1 = self.obs2index((obses_tp1, (None,) * len(obses_tp1)));
        hesv_next_obs_count_t_ph = np.asarray(self.count_list.query(keys_tp1)) + 1e-10;
        
        self.count((obses_t,));
        self.count((obses_t, actions));
#        print(np.sum(hesv_next_obs_count_t_ph))
#        k = [[self.interval_table.get(k) for k in ks] for ks in keys_tp1];
        hesv_sum_t_ph, hesv_max_t_ph, hesv_min_t_ph = _dict_for_statistic_info.query([self.interval_table_for_sum, self.interval_table_for_max, self.interval_table_for_min], keys_tp1);
        hesv_sum_t_ph, hesv_max_t_ph, hesv_min_t_ph  = np.reshape(hesv_sum_t_ph, (len(rewards), self.actionSpace.n)), np.reshape(hesv_max_t_ph, (len(rewards), self.actionSpace.n)), np.reshape(hesv_min_t_ph, (len(rewards), self.actionSpace.n));
#        hesv_max_t_ph, hesv_min_t_ph = _dict_for_statistic_info.query([self.interval_table_for_max, self.interval_table_for_min], keys_tp1);
#        hesv_max_t_ph, hesv_min_t_ph  = np.reshape(hesv_max_t_ph, (len(rewards), self.actionSpace.n)), np.reshape(hesv_min_t_ph, (len(rewards), self.actionSpace.n));

        ret = self.train_func(obses_t, actions, rewards, obses_tp1, dones, *params, 
#                               hesv_obs_count_t_ph, 
                               hesv_next_obs_count_t_ph, 
                               hesv_sum_t_ph, 
                               hesv_max_t_ph, hesv_min_t_ph,
                               sess=self.sess, **args);
#        print(ret[-2], ret[-1])           
        return ret;
       
    def _setup_loss(self):
        print('buliding loss');
        with self.graph.as_default():
            q_func_vars = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, "deepq");
            obs2map = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, "obsmap");
            q_func_vars = q_func_vars + obs2map;
            
        with tf.variable_scope("loss"):
            act_t_ph = tf.placeholder(tf.int32, [None], name="action");
            rew_t_ph = tf.placeholder(tf.float32, [None], name="reward");
            done_mask_ph = tf.placeholder(tf.float32, [None], name="done");
            
#            hesv_obs_count_t_ph = tf.placeholder(tf.float32, [None], name="hesv_obs_count");
            hesv_next_obs_count_t_ph = tf.placeholder(tf.float32, [None, self.actionSpace.n], name="hesv_next_obs_count");
            hesv_sum_t_ph = tf.placeholder(tf.float32, [None, self.actionSpace.n], name="hesv_sum");
            hesv_max_t_ph = tf.placeholder(tf.float32, [None, self.actionSpace.n], name="hesv_max");
            hesv_min_t_ph = tf.placeholder(tf.float32, [None, self.actionSpace.n], name="hesv_min");
#            hesv_count_t_ph = tf.placeholder(tf.float32, [None, self.actionSpace.n], name="hesv_count");
            
            
            q_t_selected = tf.reduce_sum(self.x2q_op * tf.one_hot(act_t_ph, self.actionSpace.n), axis=1);
            
#            #without replay
#            var = hesv_next_obs_count_t_ph * tf.pow(hesv_max_t_ph - hesv_min_t_ph, 2);
#            var = tf.where_v2(tf.is_nan(var), 0.0, var);
#            M = tf.pow(var / hesv_next_obs_count_t_ph, 0.5) * self.config.varsigma_ph;
#            M = tf.where_v2(tf.is_nan(M), 0.0, M);
#            hesv = DeepHesv.getHESV(hesv_sum_t_ph, var, M) / hesv_next_obs_count_t_ph;
            
#            #with replay
            var = tf.multiply(hesv_next_obs_count_t_ph, tf.pow(hesv_max_t_ph - hesv_min_t_ph, 2));
            var = tf.divide(var, tf.reduce_max(var, axis = 1, keepdims = True));
#            var = tf.where_v2(tf.math.is_nan(var), 0.0, var);
            M = tf.abs(hesv_max_t_ph - hesv_min_t_ph) * self.config.varsigma_ph;
#            M = tf.where_v2(tf.math.is_nan(M), 0.0, M);
            sum_x = tf.multiply(self.nx2nq_op, hesv_next_obs_count_t_ph);
            hesv__ = DeepHesv.getHESV(sum_x, var, M) / hesv_next_obs_count_t_ph;
            hesv = tf.where_v2(tf.math.is_nan(hesv__), self.nx2nq_op, hesv__);
            
            idx = tf.greater_equal(hesv, tf.reduce_max(hesv, axis = 1, keepdims = True));
            
            prob_true = 1.0 / tf.reduce_sum(tf.cast(idx, tf.float32), axis = 1, keepdims = True);
            prob = tf.where_v2(idx, prob_true, 0.0);
            
            q_tp1 = self.config.c_ph * self.config._setup_operator(self.nx2nq_op) + (1 - self.config.c_ph) * tf.reduce_sum(tf.multiply(self.nx2nq_op, prob), axis = 1);
            
            q_tp1 = (1.0 - done_mask_ph) * q_tp1;
            q_t_selected_target = rew_t_ph + self.gamma * q_tp1;
            
            td_error = q_t_selected - tf.stop_gradient(q_t_selected_target);
            print('td error', td_error);
            errors = tf_util.huber_loss(td_error);
#            errors = tf.square(td_error);
            weighted_error = tf.reduce_mean(errors);
        
            tf.summary.scalar("td_error", tf.reduce_mean(td_error));
            tf.summary.scalar("loss", weighted_error);
            
            optimizer = tf.train.AdamOptimizer(learning_rate = self.learning_rate, name = 'adam');
            
            gradients = optimizer.compute_gradients(weighted_error, var_list=q_func_vars);
            if self.grad_norm_clipping is not None:
                for i, (grad, var) in enumerate(gradients):
                    if grad is not None:
                        gradients[i] = (tf.clip_by_norm(grad, self.grad_norm_clipping), var);
            optimize_expr = optimizer.apply_gradients(gradients);
            
            train = tf_util.function(
                inputs=[
                    self._obs_ph,
                    act_t_ph,
                    rew_t_ph,
                    self._nobs_ph,
                    done_mask_ph,
                    self.config.prob_ph,
                    self.config.omega_ph,
                    self.config.varsigma_ph,
                    self.config.c_ph,
#                    hesv_obs_count_t_ph,
                    hesv_next_obs_count_t_ph,
                    hesv_sum_t_ph,
                    hesv_max_t_ph,
                    hesv_min_t_ph,
                ],
                outputs=[td_error, prob, hesv],
                updates=[optimize_expr]
            );
            return train;
        
        
    def episode_done(self, trajectories):
        G = 0;
        for traj in trajectories[::-1]:
            (obs, reward, next_obs, done) = traj.values;
            x = self.obs2index(([obs[0]], [obs[1]]))[0]; # return the key of state-action-pair
            
            infos_sum = self.interval_table_for_sum.getter(x);
            infos_max = self.interval_table_for_max.getter(x);
            infos_min = self.interval_table_for_min.getter(x);
            infos_count = self.interval_table_for_min.getter(x);
            G = self.gamma * G + reward;
#            G += (reward / 100);
#            sum max min
            if infos_sum == -np.inf:
                infos_sum = G;
            else:
                infos_sum += G;
            if G > infos_max:
                infos_max = G;
            if (G < infos_min) or (infos_min == -np.inf):
                infos_min = G;
            if infos_count == -np.inf:
                infos_count = 1;
            else:
                infos_count = infos_count + 1;
            self.interval_table_for_sum.setter(x, infos_sum);
            self.interval_table_for_max.setter(x, infos_max);
            self.interval_table_for_min.setter(x, infos_min);
            self.interval_table_for_count.setter(x, infos_count);
        trajectories = [];
    
#    deepmellowhesv.super -> deepmellow....super -> deephesv
class _DeepMellowHesv(DeepMellow, _DeepHesv):
    def __init__(self, *params, **args):
        super(DeepMellow, self).__init__(*params, **args);

    def _setup_loss(self):
        print('buliding loss');
        with self.graph.as_default():
            q_func_vars = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, "deepq");
            obs2map = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, "obsmap");
            q_func_vars = q_func_vars + obs2map;
            
        with tf.variable_scope("loss"):
            act_t_ph = tf.placeholder(tf.int32, [None], name="action");
            rew_t_ph = tf.placeholder(tf.float32, [None], name="reward");
            done_mask_ph = tf.placeholder(tf.float32, [None], name="done");
            
#            hesv_obs_count_t_ph = tf.placeholder(tf.float32, [None], name="hesv_obs_count");
            hesv_next_obs_count_t_ph = tf.placeholder(tf.float32, [None, self.actionSpace.n], name="hesv_next_obs_count");
            hesv_sum_t_ph = tf.placeholder(tf.float32, [None, self.actionSpace.n], name="hesv_sum");
            hesv_max_t_ph = tf.placeholder(tf.float32, [None, self.actionSpace.n], name="hesv_max");
            hesv_min_t_ph = tf.placeholder(tf.float32, [None, self.actionSpace.n], name="hesv_min");
            
            q_t_selected = tf.reduce_sum(self.x2q_op * tf.one_hot(act_t_ph, self.actionSpace.n), axis=1);
            
            var = tf.multiply(hesv_next_obs_count_t_ph, tf.pow(hesv_max_t_ph - hesv_min_t_ph, 2));
            var = tf.divide(var, tf.reduce_max(var, axis = 1, keepdims = True));
            M = tf.abs(hesv_max_t_ph - hesv_min_t_ph) * self.config.varsigma_ph;
            sum_x = tf.multiply(self.nx2nq_op, hesv_next_obs_count_t_ph);
            hesv__ = DeepHesv.getHESV(sum_x, var, M) / hesv_next_obs_count_t_ph;
            hesv = tf.where_v2(tf.math.is_nan(hesv__), self.nx2nq_op, hesv__);
            
            idx = tf.greater_equal(hesv, tf.reduce_max(hesv, axis = 1, keepdims = True));
            
            prob_true = (1.0 - self.config.c_ph) / tf.reduce_sum(tf.cast(idx, tf.float32), axis = 1, keepdims = True);
            prob = tf.where_v2(idx, prob_true, 0.0);
            prob = prob + self.config.prob_ph * self.config.c_ph;
            
#            q_tp1 = self.config.c_ph * self.config._setup_operator(self.nx2nq_op) + (1 - self.config.c_ph) * tf.reduce_sum(tf.multiply(self.nx2nq_op, prob), axis = 1);
            q_tp1 = self.config._setup_operator(self.nx2nq_op, prob = prob);
            
            q_tp1 = (1.0 - done_mask_ph) * q_tp1;
            q_t_selected_target = rew_t_ph + self.gamma * q_tp1;
            
            td_error = q_t_selected - tf.stop_gradient(q_t_selected_target);
            print('td error', td_error);
            errors = tf_util.huber_loss(td_error);
#            errors = tf.square(td_error);
            weighted_error = tf.reduce_mean(errors);
        
            tf.summary.scalar("td_error", tf.reduce_mean(td_error));
            tf.summary.scalar("loss", weighted_error);
            
            optimizer = tf.train.AdamOptimizer(learning_rate = self.learning_rate, name = 'adam');
            
            gradients = optimizer.compute_gradients(weighted_error, var_list=q_func_vars);
            if self.grad_norm_clipping is not None:
                for i, (grad, var) in enumerate(gradients):
                    if grad is not None:
                        gradients[i] = (tf.clip_by_norm(grad, self.grad_norm_clipping), var);
            optimize_expr = optimizer.apply_gradients(gradients);
            
            train = tf_util.function(
                inputs=[
                    self._obs_ph,
                    act_t_ph,
                    rew_t_ph,
                    self._nobs_ph,
                    done_mask_ph,
                    self.config.prob_ph,
                    self.config.omega_ph,
                    self.config.varsigma_ph,
                    self.config.c_ph,
#                    hesv_obs_count_t_ph,
                    hesv_next_obs_count_t_ph,
                    hesv_sum_t_ph,
                    hesv_max_t_ph,
                    hesv_min_t_ph,
                ],
                outputs=[td_error, prob, hesv],
                updates=[optimize_expr]
            );
            return train;
        