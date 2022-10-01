# -*- coding: utf-8 -*-
"""
Created on Sun Jul 19 18:08:29 2020

@author: 王碧
"""


from .deep_agents import DeepAgent, tf, tf_util, np, tf_layers;
from scipy import optimize;

class DeepMellow(DeepAgent):
    def fx(sub_values):
        def ffx(beta):
            return np.sum(np.multiply(np.exp(beta * sub_values), sub_values)) ** 2;
        return lambda beta: ffx(beta);
    
    def target_policy_probability(self, obs, judge, beta_scale = 1.0, *params, **args):
        sub_values, x = self.config.judge([self.pred_prob_op, self.x2q_op], self.sess, feed_dict = {self._obs_ph: obs}, 
                                       prob_ph = np.ones((len(obs), self.actionSpace.n)) / self.actionSpace.n, **args);
        beta = optimize.minimize_scalar(DeepMellow.fx(sub_values), method="Brent", options={"maxiter": 5}).x;
        x = beta * x;
#        tmp = np.exp(x);
        tmp = np.exp(x - x.max());
        prob = (tmp / np.sum(tmp));
        return prob;
    
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
            
#            lr_ph = tf.constant(self.learning_rate, name = 'lr');
#            lr_ph = tf.placeholder(tf.int32, name = 'lr', shape = []);
            
            
            q_t_selected = tf.reduce_sum(self.x2q_op * tf.one_hot(act_t_ph, self.actionSpace.n), axis=1);
            
            q_tp1 = self.config._setup_operator(self.nx2nq_op);
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
#                    lr_ph,
                ],
                outputs=[td_error, weighted_error, rew_t_ph, q_t_selected_target],
                updates=[optimize_expr]
            );
            return train;
