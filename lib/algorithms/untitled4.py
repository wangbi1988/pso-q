# -*- coding: utf-8 -*-
"""
Created on Mon Jul 13 06:10:49 2020

@author: 王碧
"""

#traj1 = trajectories[-1];
#traj2 = trajectory((obs, a), (reward - 1) if not done else -1, (next_obs, next_a), 1.0);
##
#for i in range(100):
#    dqn.update([traj1], GAMMA, alpha(dqn.count(([obs], [a]), query = True)), operator = 'max');
#    dqn.update([traj1], GAMMA, alpha(dqn.count(([obs], [a]), query = True)), operator = 'max');
#    dqn.update([traj1], GAMMA, alpha(dqn.count(([obs], [a]), query = True)), operator = 'max');
#    dqn.update([traj1], GAMMA, alpha(dqn.count(([obs], [a]), query = True)), operator = 'max');
#    dqn.update([traj1], GAMMA, alpha(dqn.count(([obs], [a]), query = True)), operator = 'max');
#    dqn.update([traj2], GAMMA, alpha(dqn.count(([obs], [a]), query = True)), operator = 'max');
#    f,g =dqn.lookup(([obs],[a]))
#    print(f)

#obs = env.reset();
#f,g =dqn.lookup(([obs],[1]))
#h =dqn.obs2x([obs, next_obs])
try:
    test(env, dqn);
finally:
    env.close();