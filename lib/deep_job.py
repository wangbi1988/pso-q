# -*- coding: utf-8 -*-
"""
Created on Thu Jul 30 10:09:22 2020

@author: 王碧
"""

from .multiprocess import abstractJob;
from .envs import gym;
from .algorithms import trajectory;
from .algorithms import DeepAgent, BasicDeepConfig, ReplayBuffer, DeepMellow;
import numpy as np;

from joblib import Parallel, delayed;

from lib.multiprocess import JobPoolForUnSerialize;
class _subJob(object):
    @staticmethod
    def run(env, act, idx):
        v = env.step(act);
        return (idx, v);
    
class Wrapper(object):
    def __init__(self, env, action_repeat):
        self._env = env;
        self.action_repeat = action_repeat;

    def __getattr__(self, name):
        return getattr(self._env, name);
    
    def step(self, action):
        r = 0.0;
        for _ in range(self.action_repeat):
            obs_, reward_, done_, info_ = self._env.step(action);
            r = r + reward_;
            if done_:
                return obs_, r, done_, info_;
        return obs_, r, done_, info_;

class EnvBlock(object):
    unzip_step_return = lambda w, x, y, z: (w, x, y, z);
    def __init__(self, envName, seeds, max_timesteps_limit):
        self.envblock = [];
        self.env_need_step = {};
        self.seeds = seeds;
        for i in seeds:
            env = gym.make(envName);
            env.seed(int(i));
            env.action_space.seed(int(i));
            env.observation_space.seed(int(i));
            if max_timesteps_limit is not None:
                env._max_episode_steps = max_timesteps_limit;
            self.envblock.append(env);
            
    def step(self, action):
        returns = ([], np.zeros(len(self.envblock)), [], []);
        pops = [];
        for a, (idx, env) in enumerate(self.env_need_step.items()):
            tmp = env.step(action[a]);
            returns[0].append(tmp[0]);
            returns[2].append(tmp[2]);
            if tmp[2]:
                pops.append(idx);
            returns[3].append(tmp[3]);
            returns[1][idx] = tmp[1];
        for i in pops[::-1]:
            self.env_need_step.pop(i);
        return returns;
    
    def reset(self):
        returns = [];
        for idx, env in enumerate(self.envblock):
            env.seed(int(self.seeds[idx]));
            env.action_space.seed(int(self.seeds[idx]));
            env.observation_space.seed(int(self.seeds[idx]));
            returns.append(env.reset());
            self.env_need_step[idx] = env;
        return returns;
            
    def close(self):
        self.env_need_step = {};
        for env in self.envblock:
            try:
                env.close();
            except:
                continue;

class DeepDiscretedJobwithInjectors(abstractJob):
    def __init__(self, envName, config, model,  model_args,
                 lr_factor = 1.0, 
                 testenvSeeds = None,
                 injectors = [], **args):
        self.args = args;
        self.injectors = injectors;
        self.config = config;
        self.lr_factor = lr_factor;
        self.envName, self.testenvSeeds, self.model, self.model_args = envName, testenvSeeds, model, model_args;
        self.env, self.testenv = None, None;
    
    def __check_env__(self):
        self.agent = self.model(**self.model_args);
        self.env = gym.make(self.envName);
        self.env.seed(0);
        self.env.action_space.seed(0);
        self.env.observation_space.seed(0);
        assert isinstance(self.env.action_space, gym.spaces.Discrete), 'Error';
        self.env = Wrapper(self.env, 3);
        self.testenv = EnvBlock('LunarLander-v2', 
                            seeds = self.testenvSeeds,
                            max_timesteps_limit = None);
    
    def __check_injectors__(self, timestep, **args):
        for i in self.injectors:
            if i.trigger(timestep, **args):
                i.callback(env = self.testenv, table = self.agent, timestep = timestep, **args);
                
    def getSummary(self):
        summary = {};
        for i in self.injectors:
            summary[i.name] = i.getSummary();
        return summary;
    
    def run(self):
        try:
            self.__check_env__();
            obs = self.env.reset();
            done = False;
            tol_reward = 0;
            a = None;
            trajectories = [];
            for timestep in range(self.config.max_time_steps):
                self.args['timestep'] = timestep;
                self.args['modelname'] = self.agent.name;
                self.__check_injectors__(timestep);
                pi_args = self.config.pi_args(timestep);
                if a is None:
                    a = self.agent.pi([obs], self.config.judge, 
                                      times = self.agent.count(([obs], [None]), query = True), **pi_args);
                                      
                next_obs, reward, done, info = self.env.step(a);
                
                next_a = self.agent.pi([next_obs], self.config.judge, 
                                      times = self.agent.count(([next_obs], [None]), query = True), **pi_args);
                                       
                traj = trajectory((obs, a), reward, (next_obs, next_a), done);
                trajectories.append(traj);
                
                lr_factor = self.lr_factor;
                if callable(lr_factor):
                    lr_factor = lr_factor(timestep);
                
                a = self.agent.update(trajectories, self.agent.gamma, 
                                      *self.config.update_para(self.agent.buffer_size, timestep),
                                      lr_factor = lr_factor,
                                      );
                                      
                tol_reward += reward;
                if done:
                    obs = self.env.reset();
                    a = None;
                    self.agent.episode_done(trajectories);
                    trajectories = [];
                else:
                    obs = next_obs;
            self.agent.save();
            return self.getSummary();
        finally:
            self.close();
                
    def close(self):
        if self.env:
            self.env.close();
        if self.testenv:
            self.testenv.close();
            
        