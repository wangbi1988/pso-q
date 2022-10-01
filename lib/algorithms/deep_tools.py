# -*- coding: utf-8 -*-
"""
Created on Tue Jun 16 11:10:00 2020

@author: bb
"""
import numpy as np;
from stable_baselines.common.buffers import ReplayBuffer, PrioritizedReplayBuffer;


def exampleER():
    batch_size = 64;
    obs_, action, reward_, new_obs_, done = (1,) * 5;
    done = float(done);
    replay_buffer = ReplayBuffer(size = 100);
    replay_buffer.add(obs_, action, reward_, new_obs_, done);
    can_sample = replay_buffer.can_sample(batch_size);
    if can_sample:
        experience = replay_buffer.sample(batch_size, env = None);
        obses_t, actions, rewards, obses_tp1, dones = replay_buffer.sample(batch_size, env = None);
        weights, batch_idxes = np.ones_like(rewards), None;