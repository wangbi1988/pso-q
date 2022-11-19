# -*- coding: utf-8 -*-
"""
Created on Tue Jun 16 11:00:47 2020

@author: bb
"""

from . import tf_util, approximatedAgent, ReplayBuffer, observation_input;

import marshal;
import numpy as np;
import tensorflow as tf;
import tensorflow.contrib.slim as slim;
import tensorflow.contrib.layers as tf_layers;
from tensorflow.contrib.layers.python.layers import initializers;

import multiprocessing;

class CountBasedList(object):
    def __init__(self):
        self.dict = {};
        
    def _get(self, idx):
        if idx in self.dict.keys():
            return self.dict[idx];
        else:
            return 0;
    
    def _add(self, idx):
        if idx in self.dict.keys():
            nums = self.dict[idx] + 1;
        else:
            nums = 1;
        self.dict[idx] = nums;
        
    def count(self, idxs):
        for i in idxs:
            self._add(i);
        
    def query(self, idxs):
        if type(idxs) is not list:
            return self._get(idxs);
        ret = [self.query(i) for i in idxs];
        return ret;


class DeepAgent(approximatedAgent):
    def __init__(self, obsSpace, actionSpace, config = None,
                 gamma=0.99, learning_rate=5e-4, buffer_tol_size = None, train_freq=1, batch_size = 32,
                 learning_starts=1000, target_network_update_freq=500, 
                 n_cpu_tf_sess=None, verbose=0, seed=None, output_graph = True,
                 scale = False, grad_norm_clipping = None, obsmap_layers = (24, ), qnet_layers = (12, ),
                 countbased_features = 12, buffer_size = 256, name = 'dqn',
                 ):
        super(DeepAgent, self).__init__(obsSpace, actionSpace);
        self.actionSpace.n = self.actionSpace.shape[0];
        self.gamma = gamma;
        self.learning_starts = learning_starts;
        self.train_freq = train_freq;
        self.batch_size = batch_size;
        self.target_network_update_freq = target_network_update_freq;
        self.replay_buffer = ReplayBuffer(10000) if buffer_tol_size is None else ReplayBuffer(buffer_tol_size);
        self.buffer_size = buffer_size;
                
        self.countbased_features = countbased_features;
        self.learning_rate = learning_rate;
        
        self.config = config;
        
        self.obsmap_layers = obsmap_layers;
        self.qnet_layers = qnet_layers;
        
        self.count_list = CountBasedList();
        self.scale = scale;
        self.grad_norm_clipping = grad_norm_clipping;
        self.learn_step_counter = self.learn_step_counter_pre = 0;
        self.n_cpu_tf_sess = n_cpu_tf_sess if n_cpu_tf_sess else multiprocessing.cpu_count();
        
        self.init_seed = 0;
        
        self.setup_model();

        if output_graph:
            self.print_model();
            
        self.episode_leng = [];
        
        self.name = name;
            
    def print_model(self):
        with self.graph.as_default():
            slim.model_analyzer.analyze_vars(tf.trainable_variables(), print_info=True);

    def obs2x(self, obs):
        return self.obs2x_func(obs, sess = self.sess);
        
    def obs2index(self, obs):
        ids = self.obs2index_func(obs[0], sess = self.sess)[0];
    
        keys = None;
        if len(obs) == 1:
            keys = [(marshal.dumps(i) + marshal.dumps('-state')) for i in ids];
        else:
            keys = [((marshal.dumps(i) + marshal.dumps('-act%s'%action)) if action is not None else [(marshal.dumps(i) + marshal.dumps('-act%s'%j)) for j in self.legal_actions(state)]) for state, action, i in zip(obs[0], obs[1], ids)];
        return keys;
        
    def count(self, obs, query = False):
        assert hasattr(obs, '__iter__'), 'obs in tabular agent need be iterable';
        keys = self.obs2index(obs);
        if not query:
            self.count_list.count(keys);
        else:
            return np.asarray(self.count_list.query(keys));
    
    def lookup(self, obs):
        assert hasattr(obs, '__iter__'), 'obs in tabular agent need be iterable'
        state, action =  obs;
        qvalue = self.x2q_func(state, sess = self.sess)[0];
        ret = [(qs if act is None else qs[act]) for qs, act in zip(qvalue, action)];
        return ret, state;
        
    def update(self, trajectories, gamma, *params, lr_factor = 1.0, **args):
        trajectory = trajectories[-1];
        (obs, reward, next_obs, done) = trajectory.values;
        if self.replay_buffer is not None:
            self.replay_buffer.add(obs[0], obs[1], reward, next_obs[0], float(done));
        if done:
            self.episode_leng.append(self.learn_step_counter - self.learn_step_counter_pre);
            self.learn_step_counter_pre = self.learn_step_counter;
#        exp-replay
        if self.learn_step_counter >= self.learning_starts:
            if self.learn_step_counter % self.train_freq == 0:
                can_sample = self.replay_buffer.can_sample(self.buffer_size) if self.replay_buffer is not None else True;
                if can_sample:
                    experience = self.replay_buffer.sample(self.buffer_size) if self.replay_buffer is not None else trajectory.values;
                    self._do_update(experience, *params,  **args);
            if (self.target_network_update_freq > 0) and (self.learn_step_counter % self.target_network_update_freq == 0):
                self.copy_net_func(sess = self.sess);
                print('copy....');
        self.learn_step_counter += 1;
        return next_obs[1];
    
    def _do_update(self, experience, *params,  **args):
#                    self.config.prob_ph,
#                    self.config.omega_ph,
        (obses_t, actions, rewards, obses_tp1, dones) = experience;
        ret = self.train_func(obses_t, actions, rewards, obses_tp1, dones, *params, sess=self.sess, **args);
        return ret;         
    
    def episode_done(self, trajectories):
        pass;
        
        
    def pi(self, obs, judge, *params, **args):
        prob = self.behavior_policy(obs, judge, * params, **args);
        return DeepAgent.choice_v1(len(prob), size = 1, p = prob);
    
    def behavior_policy(self, obs, judge, *params, **args):
        return self.target_policy_probability(obs, judge, *params, **args);
    
    def target_policy_probability(self, obs, judge, *params, **args):
        return self.config.judge(self.pred_prob_op, self.sess, 
                                 feed_dict = {self._obs_ph: obs}, 
                                 prob_ph = np.ones((len(obs), self.actionSpace.n)) / self.actionSpace.n, 
                                 **args);
        
    def setup_model(self):
        self.graph = tf.Graph()
        with self.graph.as_default():
            self.sess = tf_util.make_session(num_cpu = self.n_cpu_tf_sess, graph = self.graph);
            
            
            with tf.variable_scope("input", reuse=False):
                self._obs_ph, self._processed_obs = observation_input(self.obsSpace, None, scale = self.scale, name = "ob");
                self._nobs_ph, self._nprocessed_obs = observation_input(self.obsSpace, None, scale = self.scale, name = "nob");
            
            
            self.obs2index_op, self.nobs2index_op, self.obs2index_func = self._setup_countbased();
            self.obs2x_op, self.nobs2nx_op, self.obs2x_func = self._setup_obsmap();
            self.x2q_op, self.nx2nq_op, self.x2q_func = self._setup_x2qvalue();
            
            with tf.variable_scope("hyperparams"):
                self.config._setup_hyperparams(self.x2q_op);
                self.pred_prob_op = self._setup_policy();
                
            print('buliding loss begin......');
            self.train_func = self._setup_loss();
            
            self.copy_net_func = self._setup_copy();
#            self.copy_net_func = self._setup_soft_update();

            tf_util.initialize(self.sess);
            self.copy_net_func(sess = self.sess);
            save_path = 'ckpt/ckpt';
            writer = tf.summary.FileWriter(save_path);
            writer.add_graph(self.graph);
    
    def _setup_countbased(self):
        with tf.variable_scope("countbased"):
            self.hashmatrix = tf.Variable(tf.random_normal((self._obs_ph.shape[1], self.countbased_features), 
                                                           seed = self.init_seed), 
                                          dtype = tf.float32, name = "hash_matrix", trainable = False);
            obs2index_op = tf.sign(tf.matmul(self._processed_obs, self.hashmatrix), name = "obs2index");
            nobs2index_op = tf.sign(tf.matmul(self._nprocessed_obs, self.hashmatrix), name = "nobs2index");
            return obs2index_op, nobs2index_op, tf_util.function(
                    inputs=[self._obs_ph],
                    outputs=[obs2index_op],
                    updates=[]
                    );
    
    def _setup_obsmap(self):
        def create(_input, reuse = False):
            action_out = tf.layers.flatten(_input);
            for index, layer_size in enumerate(self.obsmap_layers):
                action_out = tf_layers.fully_connected(action_out, num_outputs = layer_size, 
                                                       activation_fn = None, 
                                                       weights_initializer = initializers.xavier_initializer(seed = self.init_seed),
                                                       scope = 'fc%s'%index, reuse = reuse);
                action_out = tf_layers.layer_norm(action_out, center=True, scale=True, scope = 'norm%s'%index);
                action_out = tf.nn.relu(action_out, name = 'relu%s'%index);
            return action_out;
        
        with tf.variable_scope("obsmap"):
            obs2x_op = create(self._processed_obs);
        with tf.variable_scope("obsmap", reuse = True):
            nobs2nx_op = create(self._nprocessed_obs, reuse = True);
        return obs2x_op, nobs2nx_op, tf_util.function(
                    inputs=[self._obs_ph],
                    outputs=[obs2x_op],
                    updates=[]
                    );
    
    def _setup_x2qvalue(self):
        def create(_input, trainable = True, reuse = False):
            action_out = tf.layers.flatten(_input);
            for index, layer_size in enumerate(self.qnet_layers):
                action_out = tf_layers.fully_connected(action_out, num_outputs = layer_size, 
                                                       activation_fn = None, scope = 'fc%s'%index, 
                                                       trainable = trainable, 
                                                       weights_initializer = initializers.xavier_initializer(seed = self.init_seed), 
                                                       reuse = reuse,);
                action_out = tf_layers.layer_norm(action_out, center=True, 
                                                  scale=True, scope = 'norm%s'%index, trainable = trainable);
                action_out = tf.nn.relu(action_out, name = 'relu%s'%index);
            action_out = tf_layers.fully_connected(action_out, num_outputs = self.actionSpace.n, 
                                                   scope = 'output', trainable = trainable, 
                                                   weights_initializer = initializers.xavier_initializer(seed = self.init_seed), 
                                                   reuse = reuse,);
            return action_out;
        
        with tf.variable_scope("deepq"):
            x2q_op = create(self.obs2x_op);
        
        if self.target_network_update_freq > 0:
            with tf.variable_scope("target_deepq"):
                nx2nq_op = create(self.nobs2nx_op, trainable = False, reuse = False);
        else:
            with tf.variable_scope("deepq", reuse = True):
                nx2nq_op = create(self.nobs2nx_op, trainable = True, reuse = True);
                
        return x2q_op, nx2nq_op, tf_util.function(
                    inputs=[self._obs_ph],
                    outputs=[x2q_op],
                    updates=[]
                    );
    
    def _setup_loss(self):
        with self.graph.as_default():
            q_func_vars = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, "deepq");
            obs2map = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, "obsmap");
            q_func_vars = q_func_vars + obs2map;
            
        with tf.variable_scope("loss"):
            act_t_ph = tf.placeholder(tf.int32, [None], name="action");
            rew_t_ph = tf.placeholder(tf.float32, [None], name="reward");
            done_mask_ph = tf.placeholder(tf.float32, [None], name="done");
            
            q_t_selected = tf.reduce_sum(self.x2q_op * tf.one_hot(act_t_ph, self.actionSpace.n), axis=1);
            
            q_tp1 = self.config._setup_operator(self.nx2nq_op);
            q_tp1 = (1.0 - done_mask_ph) * q_tp1;
            q_t_selected_target = rew_t_ph + self.gamma * q_tp1;
            td_error = q_t_selected - tf.stop_gradient(q_t_selected_target);
            errors = tf_util.huber_loss(td_error);
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
#                    lr_ph,
                ],
                outputs=[td_error, weighted_error, q_t_selected, q_t_selected_target, gradients],
                updates=[optimize_expr]
            );
            return train;
    
    def _setup_policy(self):
        pred_prob = self.config._setup_judge(self.x2q_op);
        return pred_prob;
        
    def _setup_copy(self):
        with self.graph.as_default():
            q_func_vars = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, "deepq");
            target_q_func_vars = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, "target_deepq");
        with tf.variable_scope("copy_net"):
            update_target_expr = [];
            for var, var_target in zip(sorted(q_func_vars, key=lambda v: v.name),
                                       sorted(target_q_func_vars, key=lambda v: v.name)):
                update_target_expr.append(var_target.assign(var));
            update_target_expr = tf.group(*update_target_expr);
        return tf_util.function([], [], updates=[update_target_expr]);
    
    def _setup_soft_update(self, tau = 1e-3):
        """Soft update model parameters.
        θ_target = τ*θ_local + (1 - τ)*θ_target
        Params
        ======
            local_model (PyTorch model): weights will be copied from
            target_model (PyTorch model): weights will be copied to
            tau (float): interpolation parameter 
        """
        with self.graph.as_default():
            q_func_vars = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, "deepq");
            target_q_func_vars = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, "target_deepq");
            
        with tf.variable_scope("soft_copy_net"):
            update_target_expr = [];
            for var, var_target in zip(sorted(q_func_vars, key=lambda v: v.name),
                                       sorted(target_q_func_vars, key=lambda v: v.name)):
                update_target_expr.append(var_target.assign((1. - tau) * var_target + tau * var));
            update_target_expr = tf.group(*update_target_expr);
        return tf_util.function([], [], updates=[update_target_expr]);

    def save(self, file = None):
        import pickle;
        with self.graph.as_default():
            model_vars_gpu = tf.global_variables();
            model_cpu = {};
            for var in model_vars_gpu:
                model_cpu[var.name] = var.value().eval(session = self.sess);
            
        if file is None:
            file = self.name;
        with open('trained-model/%s.pkl'%file, 'wb') as f:
            pickle.dump(model_cpu, f);
            
    def load(self, file = None):
        import pickle;
        model_cpu = {};
        if file is None:
            file = self.name;
        with open('trained-model/%s.pkl'%file, 'rb') as f:
            model_cpu = pickle.load(f);
        
        with self.graph.as_default():
            model_vars_gpu = tf.global_variables();
            
            ops = [];
            for var in model_vars_gpu:
                if var.name in model_cpu.keys():
                    ops.append(var.assign(model_cpu[var.name]));
            self.sess.run(ops);
