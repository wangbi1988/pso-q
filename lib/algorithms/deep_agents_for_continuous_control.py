# -*- coding: utf-8 -*-
"""
Created on Tue Jun 16 11:00:47 2020

@author: bb
"""

from . import tf_util, approximatedAgent, CountBasedList, observation_input, ReplayBuffer;

import numpy as np;
import tensorflow as tf;

if not tf.__version__.startswith('1'):
    import tensorflow.compat.v1 as tf
    tf.disable_v2_behavior()
    import tensorflow.keras.layers as tf_layers
    import tf_slim as slim;
    import tensorflow.keras.initializers as initializers
else:
    import tensorflow.contrib.slim as slim;
    import tensorflow.contrib.layers as tf_layers;
    from tensorflow.contrib.layers.python.layers import initializers;



import marshal;
import multiprocessing;

class DeepAgentContinuous(approximatedAgent):
    def __init__(self, obsSpace, actionSpace, act_bounds, pso, config = None,
                 gamma = 0.99, learning_rate = 5e-4, buffer_tol_size = None, 
                 train_freq = 1, batch_size = 32,
                 learning_starts = 1000, target_network_update_freq = 500, 
                 n_cpu_tf_sess = None, verbose = 0, output_graph = True,
                 scale = True, grad_norm_clipping = None, obsmap_layers = (24, ), qnet_layers = (12, ),
                 countbased_features = 12, name = 'dqn', init_seed = None,
                 K = 1,
                 ):
        super(DeepAgentContinuous, self).__init__(obsSpace, actionSpace);
        self.act_bounds = act_bounds;
        self.pso = pso;
        self.gamma = gamma;
        self.learning_starts = learning_starts;
        self.train_freq = train_freq;
        self.batch_size = batch_size;
        self.target_network_update_freq = target_network_update_freq;
        self.replay_buffer = None if buffer_tol_size is None else ReplayBuffer(buffer_tol_size);
        self.buffer_size = batch_size;
        
        self.K = K;
                
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
        
        self.init_seed = init_seed;
        
        self.setup_model();

        if output_graph:
            self.print_model();
            
        self.episode_leng = [];
        
        self.name = name;
            
    def print_model(self):
        with self.graph.as_default():
            slim.model_analyzer.analyze_vars(tf.trainable_variables(), print_info = True);

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
        qvalue = self.x2q_func(state, action, sess = self.sess)[0];
        return qvalue, state;
        
    def update(self, trajectories, gamma, *params, lr_factor = 1.0, **args):
        trajectory = trajectories[-1];
        (obses_t, actions), rewards, (obses_tp1, actions_tp1), dones = trajectory.values;
        if self.replay_buffer is not None:
            self.replay_buffer.add(obses_t[0], actions[0], rewards[0], obses_tp1[0], actions_tp1[0], float(dones[0]));
        if dones:
            self.episode_leng.append(self.learn_step_counter - self.learn_step_counter_pre);
            self.learn_step_counter_pre = self.learn_step_counter;
#        exp-replay
        if self.learn_step_counter >= self.learning_starts:
            if self.learn_step_counter % self.train_freq == 0:
                can_sample = self.replay_buffer.can_sample(self.buffer_size) if self.replay_buffer is not None else True;
                if can_sample:
                    experience = self.replay_buffer.sample(self.buffer_size) if self.replay_buffer is not None else (
                            obses_t, actions, rewards, obses_tp1, actions_tp1, np.asarray(dones, dtype = np.float32));
                    self._do_update(experience, *params,  **args);
            if (self.target_network_update_freq > 0) and (self.learn_step_counter % self.target_network_update_freq == 0):
                self.copy_net_func(sess = self.sess);
        self.learn_step_counter += 1;
        return actions_tp1[0];
    
    def _do_update(self, experience, *params,  **args):
        obses_t, actions, rewards, obses_tp1, acts_1, dones = experience;
        ret = self.train_func(obses_t, actions, rewards, obses_tp1, acts_1, dones, *params, sess=self.sess, **args);
        td_error, weighted_error, q_t_selected, q_t_selected_target, gradients = ret
#        print('weighted_error', weighted_error)
#        print('q_t_selected', q_t_selected)
#        print('q_t_selected_target', q_t_selected_target)
#        print('td_error', td_error)
        return ret;         
    
    def episode_done(self, trajectories):
        pass;
        
        
    def pi(self, obs, judge, *params, **args):
        prob, acts = self.behavior_policy(obs, judge, * params, **args);
        idx = DeepAgentContinuous.choice_v1(len(prob), size = 1, p = prob);
        act = acts[idx, :];
#        act = [([act[i]] if np.random.random() > args['eps_ph']\
#                else (np.random.random(len(self.act_bounds[0]))) * (self.act_bounds[1] - self.act_bounds[0]) + self.act_bounds[0]) for i in range(len(obs))];
#        act = np.reshape(np.asarray(act), newshape = (len(obs) * len(self.act_bounds[0])));
        return act;
    
    def decide(self, obs, *params, **args):
        action = [];
        for o in obs:
            position, velocity = o
            if position > -4 * velocity or position < 13 * velocity - 0.6:
                force = 1.
            else:
                force = -1.
            action.append(force)
        return action
    
    def behavior_policy(self, obs, judge, *params, **args):
        return self.target_policy_probability(obs, judge, *params, **args);
    
    def target_policy_probability(self, obs, judge, *params, **args):
        return self.config.judge(self.pred_prob_op, self.sess, 
                                 feed_dict = {self._test_obs_ph: obs}, 
                                 prob_ph = np.ones(shape = (len(obs), self.K)) / self.K, 
                                 **args);
        
    def setup_model(self):
        self.graph = tf.Graph()
        with self.graph.as_default():
            self.sess = tf_util.make_session(num_cpu = self.n_cpu_tf_sess, graph = self.graph);
            
            
            with tf.variable_scope("input", reuse=False):
                self._obs_ph, self._processed_obs = observation_input(self.obsSpace, self.batch_size, 
                                                                      scale = self.scale, name = "batch_ob");
                self._nobs_ph, self._nprocessed_obs = observation_input(self.obsSpace, self.batch_size, 
                                                                        scale = self.scale, name = "batch_nob");
                self._act_ph, self._processed_act = observation_input(self.actionSpace, self.batch_size, 
                                                                      scale = self.scale, name = "batch_act");
                self._nact_ph, self._nprocessed_act = observation_input(self.actionSpace, self.batch_size, 
                                                                        scale = self.scale, name = "batch_nact");
                
                
                self._rew_ph = tf.placeholder(tf.float32, [self.batch_size], name="batch_reward");
                self._processed_rew_ph = tf.reshape(self._rew_ph, shape = (self.batch_size, 1));
                                                                            
                self._done_mask_ph = tf.placeholder(tf.float32, [self.batch_size], name="done");
                self._processed_done_mask_ph = tf.reshape(self._done_mask_ph, shape = (self.batch_size, 1));
            
                self._test_obs_ph, self._test_processed_obs = observation_input(self.obsSpace, 1, 
                                                                      scale = self.scale, name = "test_ob");
                self._test_act_ph, self._test_processed_act = observation_input(self.actionSpace, 1, 
                                                                      scale = self.scale, name = "test_act");
                                                                                
                                                                                
                
            inputs = (self._obs_ph, self._processed_obs, 
                      self._nobs_ph, self._nprocessed_obs, 
                      self._act_ph, self._processed_act, 
                      self._nact_ph, self._nprocessed_act);
                      
            (self.obs2index_op, self.nobs2index_op, self.obs2index_func,
             self.obs2x_op, self.nobs2nx_op, self.obs2x_func,
             self.x2q_op, self.nx2nq_op, self.x2q_func) = self._bulid_training_scheme(inputs, 
                                                      reuse = False, name = 't1');
            print('+'*30)
            
            self.global_particle, self.global_eval = self._buildPSO(self.pso, 
                                                                    (self._test_obs_ph, self._test_processed_obs), 
                                                                    1, maxiter = 5, N = 20, k = self.K);
            print('global_particle', self.global_particle, self.global_eval)
            
            self.nglobal_particle, self.nglobal_eval = self._buildPSO(self.pso, 
                                                                      (self._nobs_ph, self._nprocessed_obs),
                                                                      self.batch_size, 
                                                                      targetNet = True,
                                                                      maxiter = 20, N = 50, k = self.K);
                                                                         
            print('nglobal_particle', self.nglobal_particle, self.nglobal_eval);
            self.pred_prob_op = self._setup_policy((self.global_particle, self.global_eval));
            
            print('pred_prob_op', self.pred_prob_op);
        
            print('buliding loss begin......');
            self.train_func = self._setup_loss();
            
#            self.copy_net_func = self._setup_copy();
            self.copy_net_func = self._setup_soft_update();

            tf_util.initialize(self.sess);
            self.copy_net_func(sess = self.sess);
            
            save_path = 'ckpt/ckpt';
            writer = tf.summary.FileWriter(save_path);
            writer.add_graph(self.graph);
    
    
    def _bulid_training_scheme(self, inputs, reuse = False, name = 't1'):
        (_obs_ph, _processed_obs, 
         _nobs_ph, _nprocessed_obs, 
         _act_ph, _processed_act, 
         _nact_ph, _nprocessed_act
         ) = inputs;
         
        obs2index_op, obs2index_func = self._setup_countbased((_obs_ph, _processed_obs,), 
                                                              reuse = reuse, name = name);
        nobs2index_op, _ = self._setup_countbased(( _nobs_ph, _nprocessed_obs,), reuse = True,);
        
        obs2x_op, obs2x_func = self._setup_obsmap((_obs_ph, _processed_obs,), 
                                                  reuse = reuse, name = 'obs_%s'%(name));
        nobs2nx_op, _ = self._setup_obsmap(( _nobs_ph, _nprocessed_obs,), 
                                           reuse = True, name = 'obs_%s'%(name));
        
        
        act2x_op, act2x_func = self._setup_obsmap((_act_ph, _processed_act,), 
                                                  reuse = reuse, name = 'act_%s'%(name));
                                                  
        nact2x_op, _ = self._setup_obsmap((_nact_ph, _nprocessed_act,), 
                                                  reuse = True, name = 'act_%s'%(name));
        
        x2q_op, x2q_func = self._setup_x2qvalue((_obs_ph, obs2x_op, _act_ph, act2x_op), 
                                                reuse = reuse, name = name);
                                                
        nx2nq_op, _ = self._setup_x2qvalue((_nobs_ph, nobs2nx_op, _nact_ph, nact2x_op), 
                                           reuse = False, name = 'n' + name,
                                           targetNet = True);
        
        return (obs2index_op, nobs2index_op, obs2index_func, 
                obs2x_op, nobs2nx_op, obs2x_func, 
                x2q_op, nx2nq_op, x2q_func);
                
    def _buildPSO(self, pso, inputs, batch_size, targetNet = False, k = 1,
                  maxiter = 50, N = 1024, paras = {'W': 0.5,
                                                   'C1': 1,
                                                   'C2': 2}, name = 't1'):
        _obs_ph, _processed_obs = inputs;
        def _inner_func(_setup_obsmap, _setup_x2qvalue):
            def __func(act, obs):
                obs2x_op, _ = _setup_obsmap((obs, obs,), 
                                            reuse = True, name = 'obs_%s'%(name));
                _processed_act = ((act - self.act_bounds[0]) / (self.act_bounds[1] - self.act_bounds[0]));
                act2x_op, _ = _setup_obsmap((_processed_act, _processed_act,), 
                                            reuse = True, name = 'act_%s'%(name));
                x2q_op, _ = _setup_x2qvalue((obs, obs2x_op, act, act2x_op), reuse = True, 
                                            name = 'PSO_x2q', targetNet = targetNet);
                return tf.stop_gradient(x2q_op);
            return __func
        
        def _inner_test_func(_setup_obsmap, _setup_x2qvalue):
            def __func(act, obs):
                act = tf.math.pow(act - obs, 2)
                total = tf.reduce_sum(act, axis = 1 if len(act.shape) == 2 else 0, keep_dims = True)
                return -total
            return __func
        
        D = self.actionSpace.shape[0];
        bounds = self.act_bounds;
        
        p = pso(N = N * D, D = D, **paras);
        func = _inner_func(self._setup_obsmap, self._setup_x2qvalue);
        global_particle, global_eval = p.build_batch_v4(func = func, 
                                                        y = _processed_obs, 
                                                        bounds = bounds, maxiter = maxiter, 
                                                        batch_size = batch_size, 
                                                        k = k);
        return tf.stop_gradient(tf.reshape(global_particle, shape = (batch_size, k, D,))), \
                tf.stop_gradient(tf.reshape(global_eval, shape = (batch_size, k)));
    
    def _setup_countbased(self, x, name = 'obs2index', reuse = False):
        _obs_ph, _processed_obs = x;
        with tf.variable_scope("countbased", reuse = reuse):
            hashmatrix = tf.get_variable(name = "hash_matrix", shape = ((_obs_ph.shape[1], self.countbased_features)), 
                                         initializer = tf.random_normal_initializer(seed = self.init_seed));
        obs2index_op = tf.sign(tf.matmul(_processed_obs, hashmatrix));
        return obs2index_op, tf_util.function(
                inputs=[_obs_ph],
                outputs=[obs2index_op],
                updates=[]
                );
    
    
    def _setup_obsmap(self, x, reuse = False, name = 'obsmap'):
        _obs_ph, _processed_obs = x;
        def create(_input, reuse = reuse):
            action_out = tf_layers.flatten(_input);
            for index, layer_size in enumerate(self.obsmap_layers):
                action_out = tf_layers.fully_connected(action_out, num_outputs = layer_size, 
                                                       activation_fn = None, 
                                                       weights_initializer = initializers.xavier_initializer(seed = self.init_seed),
                                                       scope = 'fc%s'%index, reuse = reuse,
                                                       #weights_regularizer = tf.contrib.layers.l2_regularizer(0.01),
                                                       );
                #action_out = tf_layers.layer_norm(action_out, center = True, scale = True, scope = 'norm%s'%index);
                action_out = tf.nn.leaky_relu(action_out, name = 'relu%s'%index);
            return action_out;
        
        with tf.variable_scope("obsmap_%s"%(name), reuse = reuse):
            obs2x_op = create(_processed_obs);
        return obs2x_op, tf_util.function(
                    inputs=[_obs_ph],
                    outputs=[obs2x_op],
                    updates=[]
                    );
    
    def _setup_x2qvalue(self, x, reuse = False, name = 'x2qv', targetNet = False):
        _obs_ph, obs2x_op, _act_ph, act2x_op = x;
        def create(_input, trainable = True, reuse = False):
            action_out = tf.layers.flatten(_input);
            for index, layer_size in enumerate(self.qnet_layers):
                action_out = tf_layers.fully_connected(action_out, num_outputs = layer_size, 
                                                       activation_fn = None, scope = 'fc%s'%index, 
                                                       trainable = trainable, 
                                                       weights_initializer = initializers.xavier_initializer(seed = self.init_seed), 
                                                       reuse = reuse,
                                                       #weights_regularizer = tf.contrib.layers.l2_regularizer(0.01),
                                                       );
                #action_out = tf_layers.layer_norm(action_out, center=True, 
                                                  #scale=True, scope = 'norm%s'%index, trainable = trainable);
                action_out = tf.nn.leaky_relu(action_out, name = 'relu%s'%index);
            action_out = tf_layers.fully_connected(action_out, num_outputs = 1, 
                                                   activation_fn = None,
                                                   scope = 'output', trainable = trainable, 
                                                   #weights_initializer = initializers.xavier_initializer(seed = self.init_seed), 
                                                   reuse = reuse,);
            return action_out;
        
        
        _input = obs2x_op + act2x_op;
        #_input = tf.concat([obs2x_op, act2x_op], axis = 1);
        
        if targetNet:
            if self.target_network_update_freq > 0:
                with tf.variable_scope("target_deepq", reuse = reuse):
                    x2q_op = create(_input, trainable = False, reuse = reuse);
            else:
                with tf.variable_scope("deepq", reuse = True):
                    x2q_op = create(_input, trainable = True, reuse = True);
            x2q_op = tf.stop_gradient(x2q_op);
        else:
            with tf.variable_scope("deepq", reuse = reuse):
                x2q_op = create(_input, reuse = reuse);
                
        return x2q_op, tf_util.function(
                    inputs=[_obs_ph, _act_ph],
                    outputs=[x2q_op],
                    updates=[]
                    );
    
    def _setup_loss(self):
        with self.graph.as_default():
            q_func_vars = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, "deepq");
            obs2map = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, "obsmap");
            q_func_vars = q_func_vars + obs2map;
            
        with tf.variable_scope("loss"):
            q_t_selected = self.x2q_op;
            
            q_tp1 = self.config._setup_operator(self.nglobal_eval);
            #q_tp1 = self.nx2nq_op;
            
            q_tp1 = (1.0 - self._processed_done_mask_ph) * q_tp1;
            q_t_selected_target = self._processed_rew_ph + self.gamma * q_tp1;
            td_error = q_t_selected - tf.stop_gradient(q_t_selected_target);
            print('td_error gradients', tf.gradients(td_error, [self._obs_ph, self._act_ph, self._nobs_ph, self._nact_ph]))
            
            errors = tf_util.huber_loss(td_error);
            #errors = tf.square(td_error);
            weighted_error = tf.reduce_mean(errors);
        
            tf.summary.scalar("td_error", tf.reduce_mean(td_error));
            tf.summary.scalar("loss", weighted_error);
            
            optimizer = tf.train.AdamOptimizer(learning_rate = self.learning_rate, name = 'adam');
            #optimizer = tf.train.RMSPropOptimizer(learning_rate = self.learning_rate, name = 'adam');
            
            gradients = optimizer.compute_gradients(weighted_error, var_list=q_func_vars);
            if self.grad_norm_clipping is not None:
                for i, (grad, var) in enumerate(gradients):
                    print(var.name)
                    if grad is not None:
                        gradients[i] = (tf.clip_by_norm(grad, self.grad_norm_clipping), var);
            optimize_expr = optimizer.apply_gradients(gradients);
            
            train = tf_util.function(
                inputs=[
                    self._obs_ph,
                    self._act_ph,
                    self._rew_ph,
                    self._nobs_ph,
                    self._nact_ph,
                    self._done_mask_ph,
                ],
                outputs=[td_error, weighted_error, q_t_selected, q_t_selected_target, gradients],
                updates=[optimize_expr]
            );
            return train;
    
    def _setup_policy(self, x, reuse = False, name = 'policy'):
        global_particle, global_eval = x;
        with tf.variable_scope("hyperparams"):
            self.config._setup_hyperparams(global_eval);
        pred_prob = self.config._setup_judge(x);
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
        return model_cpu;
            
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
                if var.name.startswith('swarm_intelligence'):
                    continue;
                if var.name in model_cpu.keys():
                    ops.append(var.assign(model_cpu[var.name], name = var.name[:-3]));
            self.sess.run(ops);
