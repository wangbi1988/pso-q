# -*- coding: utf-8 -*-
"""
Created on Thu Jul  9 15:57:36 2020

@author: 王碧
"""

from . import def_dtype;
import tensorflow as tf;
#import tensorflow_probability as tfp;
import numpy as np;

@tf.function
def operator_max(x, **args):
    q_tp1 = tf.reduce_max(x, axis = 1, keepdims = True);
    return q_tp1;

@tf.function
def operator_expection(x, prob_ph, **args):
    q_tp1 = tf.reduce_sum(tf.multiply(prob_ph, x), axis = 1, keepdims = True);
    return q_tp1;

@tf.function
def operator_mellowmax(x, omega_ph, prob_ph, **args):
    c = tf.reduce_max(x, axis = 1, keepdims = True); 
    print(c)
    y = c + tf.math.divide(tf.log(tf.reduce_sum(tf.multiply(prob_ph, tf.exp(omega_ph * (x - c))), 
                                                axis = 1, keepdims = True)), omega_ph);
    y = tf.reshape(y, (-1,))
    return y;

@tf.function
def judge_epsilon_greedy(x, eps_ph, **args):
    acts, vals = x;
    if len(vals.shape) < 2:
        vals = tf.reshape(vals, shape = (tf.shape(vals)[0], 1))
    batch_size, n_actions = vals.shape;
    n_actions = tf.cast(n_actions, def_dtype);
    print(vals)
    probs = tf.zeros_like(vals, dtype = def_dtype);
    probs = tf.add(probs, tf.math.divide(eps_ph, n_actions));
    maxq = tf.reduce_max(vals, axis = 1, keepdims = True);
    t = tf.greater_equal(vals, maxq);
    tt = tf.add(probs, tf.math.divide(1 - eps_ph, tf.reduce_sum(tf.cast(t, dtype = def_dtype), 
                                                                axis = 1, keepdims = True)));
    true_fun = tf.where_v2(t, tt, probs);
    eps_ph = tf.reshape(eps_ph, (-1,))
    pred_probs = tf.where_v2(tf.less(eps_ph, 1), true_fun, probs);
    return pred_probs, acts;
    
def fx(sub_values):
    @tf.function
    def ffx(beta):
        return tf.reduce_sum(tf.multiply(tf.exp(beta * sub_values), sub_values), axis = 1) ** 2;
    return lambda beta: ffx(beta);

@tf.function
def judge_mellowmax(x, omega_ph, **args):
    acts, vals = x;
    mm = operator_mellowmax(vals, omega_ph, **args);
    sub_values = vals - mm;
    return sub_values, acts;

class BasicDeepConfig(object):
    def __init__(self, operator, judge):
        self._operatorr = operator if callable(operator) else BasicDeepConfig._registered_operators[operator];
        self._judge = judge if callable(judge) else BasicDeepConfig._registered_judges[judge];
        self._initialized = False;
        
    def _setup_hyperparams(self, x):
        self.eps_ph = tf.placeholder(def_dtype, (), name = "eps_ph");
        self.omega_ph = tf.placeholder(def_dtype, (), name = "omega_ph");
        self.eta_ph = tf.placeholder(def_dtype, [x.shape[0], 1], name = "eta_ph");
        self.prob_ph = tf.placeholder(def_dtype, [x.shape[0], x.shape[1]], name = "prob_ph");
        
        self.str2hyperparams = {'eps_ph': self.eps_ph, 'omega_ph': self.omega_ph, 
                                'eta_ph': self.eta_ph, 'prob_ph': self.prob_ph};
                                
        self.prob_ph_default = np.ones((1, x.shape[1].value)) / x.shape[1].value;
        
#        hesv
        self.varsigma_ph = tf.placeholder(def_dtype, (), name = "varsigma_ph");
        self.c_ph = tf.placeholder(def_dtype, (), name = "c_ph");
        
        self._initialized = True;
        
    def judge(self, judge, sess, feed_dict, **args):
        for key, val in args.items():
            if key in self.str2hyperparams.keys():
                feed_dict[self.str2hyperparams[key]] = val;
        vals = sess.run(judge, feed_dict = feed_dict);
        if type(judge) in (list, tuple):
            # return (vals[i][0] for i in range(len(judge)));  
            # return [vals[i][0] for i in range(len(judge))];    
            return vals[0], vals[1];       
        else:
            return vals;
    
    def _setup_operator(self, x, **args):
        assert self._initialized, 'uninitialized config';
        tmp = self.str2hyperparams.copy();
        tmp.update(args);
        operator = self._operatorr(x, **tmp);
        return operator;
    
    def _setup_judge(self, x, **args):
        assert self._initialized, 'uninitialized config';
        tmp = self.str2hyperparams.copy();
        tmp.update(args);
        #judge = self._judge(x, **self.str2hyperparams, **args);
        judge = self._judge(x, **tmp);
        return judge;
        
    _registered_judges = {
        'epsilon-greedy': judge_epsilon_greedy,
    #    'boltzman': judge_boltzman,
        'mellowmax': judge_mellowmax,
    #    'g-learning': judge_g_learning,
    };
            
    _registered_operators = {
        'max': operator_max,
    #    'boltzman': operator_boltz,
        'mellowmax': operator_mellowmax,
    #    'expected': operator_expected,
    #    'glearning': operator_glearning,
    };
            
class ConfigWithHyperParams(BasicDeepConfig):
#    allowed_keys = set('max_time_steps', 'gamma', 'omega', 'varsigma', 'c', 'prob');
    default_kvargs = {'max_time_steps': 20000, 
                      'gamma': 0.99, 'prob': 0.25,
                      'omega': 10, 
                      'varsigma': 0.1, 'c': 0.1};
    def __init__(self, operator, judge, 
                 epsilon_func = lambda x: 0.1, 
                 omega_func = None, 
                 **args):
        self._operatorr = operator if callable(operator) else BasicDeepConfig._registered_operators[operator];
        self._judge = judge if callable(judge) else BasicDeepConfig._registered_judges[judge];
        self._initialized = False;
        _default = ConfigWithHyperParams.default_kvargs.keys() - args.keys();
        self.__dict__.update((k, ConfigWithHyperParams.default_kvargs[k]) for k in _default);
        if len(args) > 0:
            self.__dict__.update((k, v) for k, v in args.items());
        
        self.epsilon_func = epsilon_func;
        if omega_func is None:
            self.omega_func = lambda x: self.omega;
        else:
            self.omega_func = omega_func;
            
            
#    return dict for action selection
    def pi_args(self, steps):
#        if not hasattr(steps , '__iter__'):
#            steps = [steps];
#        steps = np.reshape(steps, (len(steps), 1));
        return {'eps_ph': self.epsilon_func(steps), 'omega_ph': self.omega_func(steps)};
    
#    return list for update
    def update_para(self, buffer_size, timestep = 0):
        #   self.config.prob_ph,
        #   self.config.omega_ph,
        x = np.ones((buffer_size, 1));
        return (
                np.matmul(x, self.prob_ph_default), 
#                np.matmul(x, self.omega_ph_default * self.omega), 
#                np.matmul(x, self.varsigma_ph_default * self.varsigma), 
#                np.matmul(x, self.c_ph_default) * self.c(timestep) if callable(self.c) else np.matmul(x, self.c_ph_default) * self.c,
                self.omega,
                self.varsigma, 
                self.c(timestep) if callable(self.c) else self.c, 
                );
