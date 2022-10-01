# -*- coding: utf-8 -*-
"""
Created on Mon Apr 12 15:53:12 2021

@author: bb
"""
import time
import tensorflow as tf
from functools import wraps
from PSO_RL.lib.swarm_intelligence.pso.pso import PSOMP, tfPSOMP
from PSO_RL.lib.swarm_intelligence.pso.cpu_pso import cpuPSOMP
from PSO_RL.lib.swarm_intelligence.pso.pso_matrix_parallel import tfPSOMP as v1tfPSOMP
from PSO_RL.lib.tools import tf_util
import numpy as np

def fn_timer(function):
  @wraps(function)
  def function_timer(*args, **kwargs):
    t0 = time.time()
    result = function(*args, **kwargs)
    t1 = time.time()
    print ("Total time running %s: %s seconds" %
        (function, str(t1-t0))
        )
    return result
  return function_timer

def sphere(x, y):
    x = np.power(x - y, 2);
    total = np.sum(x, axis = 1 if len(x.shape) == 2 else 0);
    return total

@tf.function
def sphere_tf(x, y):
    x = tf.math.pow(x - y, 2)
    total = tf.reduce_sum(x, axis = 1 if len(x.shape) == 2 else 0, keep_dims = True)
    return -total


def test_tfpsomp(D, N, batch_size, maxiter, loops):
    bounds= np.asarray([[-5]*D, [5]*D]).astype(np.float32);
    y = tf.placeholder(dtype = tf.float32, shape = (batch_size, D,))
    feed_y = np.zeros(shape = (batch_size, D))
    p = tfPSOMP(W = 0.5, C1 = 1, C2 = 2, N = N, D = D);
    res = p.build_batch_v4(func = sphere_tf, y = y, 
                     bounds=bounds, maxiter = maxiter, batch_size = batch_size, k = 1)
    
    sess = tf_util.make_session(num_cpu = 6);
    sess.run(tf.global_variables_initializer())
    def min_(loops):
        t0 = time.time()
        xs = [];
        for i in range(loops):
            x = p.minimize(res, sess = sess, paras_dict = {y: feed_y})
            xs.append(x)
        t1 = time.time()
        time_slap = t1 - t0
        t0 = t1
        return time_slap
#        sess.close();
    return min_

def test_cpupso(D, N, batch_size, maxiter, loops):
    y = np.zeros(shape = (batch_size, D))
    bounds = np.asarray([[-5]*D, [5]*D]).astype(np.float32);
    p = cpuPSOMP(1, 2, 0.5, N, D);
    def min_(loops):
        t0 = time.time()
        xs = [];
        for i in range(loops):
            x = p.minimize(y, bounds, batch_size, maxiter, sphere)
            xs.append(x)
        t1 = time.time()
        time_slap = t1 - t0
        t0 = t1
        return time_slap
    return min_

def test_v1tfpsomp(D, N, batch_size, maxiter, loops):
    bounds= np.asarray([[-5]*D, [5]*D]).astype(np.float32);
    y = tf.placeholder(dtype = tf.float32, shape = (batch_size, D,))
    feed_y = np.zeros(shape = (batch_size, D))
    
    p = v1tfPSOMP(W = 0.5, C1 = 1, C2 = 2, N = N, D = D);
    res = p.build_batch_v2(func = sphere_tf, y = y, 
                     bounds=bounds, maxiter = maxiter, batch_size = batch_size)
    
    sess = tf_util.make_session(num_cpu = 6);
    sess.run(tf.global_variables_initializer())
    def min_(loops):
        t0 = time.time()
        xs = [];
        for i in range(loops):
            x = p.minimize(res, sess = sess, paras_dict = {y: feed_y})
            xs.append(x)
        t1 = time.time()
        time_slap = t1 - t0
        t0 = t1
        return time_slap
#    sess.close();
    return min_

import tikzplotlib

def save2tex(filename):
    tikzplotlib.save(filename, encoding = 'utf-8');

if __name__ == "__main__":
#    test_funcs = [test_v1tfpsomp];
    import pandas as pd;
    import seaborn as sns;
    import matplotlib.pyplot as plt;
    
    test_funcs = [test_tfpsomp, test_cpupso, test_v1tfpsomp];
    D, N, batch_size, maxiter, loops = 5, 100, 64, 10, 100;
    
    Ds = [16, 32, 64, 128];
    Ns = [5, 10, 50, 100, 200];
    
    results = []
    
    # DNs = [(i, j) for i in Ds for j in Ns]
    # for func in test_funcs:
    #     print(func.__name__)
    #     for (batch_size, N) in DNs:
    #         sub_func = func(D, N, batch_size, maxiter, loops)
    #         times = sub_func(loops)
    #         results.append((func.__name__, D, N, batch_size, maxiter, loops, times / loops))

    # table = pd.DataFrame(results);
    # table.columns = ['Algs.', 'D', 'N', 'batch_size', 'maxiter', 'loops', 'times'];
    # table.to_pickle('pso_test.pkl');
    table = pd.read_pickle('pso_test.pkl');
    algs = table['Algs.'].unique();
    idx = table['Algs.'] == algs[0];
    table.loc[idx, 'N'] =  table[idx]['N'] * 0.9;
    idx = table['Algs.'] == algs[2];
    table.loc[idx, 'N'] =  table[idx]['N'] * 1.1;
#    del table['batch_size']
#    del table['maxiter']
#    del table['loops']
    
    # def_fig_size_width, def_fig_size_height = plt.rcParams['figure.figsize'];
    
    # f, axs = plt.subplots(1, 1, 
    #                       figsize=(def_fig_size_width * 1, 
    #                                def_fig_size_height * 1), 
    #                                );
    # sns.scatterplot(
    #     data=table, x="N", y="times", hue="Algs.", 
    #     style="batch_size", s=100,
    #     ax = axs
    # )
    # plt.savefig('results-pso.pdf', dpi = 300, bbox_inches = 'tight');
    # save2tex('results-pso.tex');
    # plt.show()    
#    sns.lineplot(data = table, x = 'N', y = 'times', hue = 'Algs.')
#    g = sns.catplot(x="N", hue="Algs.", col="batch_size",
#                data=table, y="times");
#    fig = plt.figure()
#    ax = fig.add_subplot(111, projection = '3d')
#    
#    x = table['N']
#    y = table['batch_size']
#    z = table['times']
#    
#    ax.set_xlabel("N")
#    ax.set_ylabel("batch_size")
#    ax.set_zlabel("times")
#    
#    ax.scatter(x, y, z)
#    
#    plt.show()