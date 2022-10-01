# -*- coding: utf-8 -*-
"""
Created on Thu Jun 11 11:31:34 2020

@author: bb
"""

import numpy as np;

class trajectory(object):
    def __init__(self, obs, reward, next_obs, done):
        self.obs = obs;
        self.reward = reward;
        self.next_obs = next_obs;
        self.done = done;
    
    @property
    def obs(self):
        return self._obs;
        
    @obs.setter
    def obs(self, obs):
        self._obs = obs;
        
    @property
    def reward(self):
        return self._reward;
        
    @reward.setter
    def reward(self, reward):
        self._reward = reward;
        
    @property
    def next_obs(self):
        return self._next_obs;
        
    @next_obs.setter
    def next_obs(self, next_obs):
        self._next_obs = next_obs;
        
    @property
    def done(self):
        return self._done;
        
    @done.setter
    def done(self, done):
        self._done = done;
        
    @property
    def values(self):
        return (self.obs, self.reward, self.next_obs, self.done);
        
        
class abstractAgent(object):
    @staticmethod
    def choice_v1(values, size, p):
        x = np.random.rand();
        cum = 0;
        for i, t in enumerate(p):
            cum += t;
            if x < cum:
                break;
        if isinstance(values, int):
            return i;
        elif hasattr(values, '__iter__'):
            return values[i];
        else:
            raise NotImplementedError();
    
    def lookup(self, obs):
        raise NotImplementedError();
        
    def update(self, trajectories, *params, **args):
        raise NotImplementedError();
        
    def episode_done(self, trajectories):
        pass;
        
    def behavior_policy(self, obs, judge, *params, **args):
        raise NotImplementedError();
        
    def target_policy_probability(self, obs, judge, *params, **args):
        raise NotImplementedError();
        
    def pi(self, obs, judge, *params, **args):
        prob = self.behavior_policy(obs, judge, * params, **args);
        return abstractAgent.choice_v1(len(prob), size = 1, p = prob);
    
    def config_args(self, trajectory, args = None):
        if args is None:
            args = {};
        for clbs in self.callbacks:
            args = clbs(trajectory, args);
            
        (obs, reward, next_obs, done) = trajectory.values;
        if len(next_obs) == 2:
            args['next_a'] = next_obs[1];
        return args;
    
    @staticmethod
    def return_max(values):
        _max = np.max(values);
        idx = np.arange(0, len(values), dtype = np.int);
        return idx[values == _max];

    
class approximatedAgent(abstractAgent):
    def __init__(self, obsSpace, actionSpace):
        self.obsSpace = obsSpace;
        self.actionSpace = actionSpace;
        self.callbacks = [];
        self.legal_actions = lambda state: np.arange(self.actionSpace);
        
    def obs2x(self, obs, onehot):
        raise NotImplementedError();
        
    def lookup(self, obs):
        raise NotImplementedError();

    
class constrainedActionAgent(abstractAgent):
    def reshapeActionSpace(self, env):
        if hasattr(env, 'constrainedActions'):
            self.legal_actions = env.constrainedActions;
        else:
            print('unsuitable env');
        
        
class countBasedAgent(abstractAgent):
    def __init__(self, countTableShape):
        self.countTableShape = countTableShape;
        self.count_list = np.zeros(self.countTableShape);
        self.callbacks.append(self.countbasedcallback);
        
    def obs2index(self, obs):
        raise NotImplementedError();
    
    def count(self, obs, query = False):
        id_ = self.obs2index(obs);
        if not query:
            self.count_list[id_] += 1;
        return self.count_list[id_];
    
    def countbasedcallback(self, trajectory, args):
        (obs, reward, next_obs, done) = trajectory.values;
        args['obs_count'] = self.count(obs, query = True);
        args['next_obs_count'] = self.count((next_obs[0], None), query = True);
        return args;
        

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
