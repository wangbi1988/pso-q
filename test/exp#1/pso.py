# -*- coding: utf-8 -*-
"""
Created on Mon Apr 12 15:53:12 2021

@author: bb
"""
import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import warnings

warnings.filterwarnings("ignore")

import sys;

print('Python %s on %s' % (sys.version, sys.platform))
sys.path.extend(['D:\\papers\\codes\\PSO_RL'])

import time
import tensorflow as tf
from functools import wraps
from lib.swarm_intelligence.pso.pso import PSOMP, tfPSOMP
from lib.swarm_intelligence.pso.cpu_pso import cpuPSOMP
from lib.swarm_intelligence.pso.pso_matrix_parallel import tfPSOMP as v1tfPSOMP
from lib.tools import tf_util
import numpy as np


def fn_timer(function):
    @wraps(function)
    def function_timer(*args, **kwargs):
        t0 = time.time()
        result = function(*args, **kwargs)
        t1 = time.time()
        print("Total time running %s: %s seconds" %
              (function, str(t1 - t0))
              )
        return result

    return function_timer


def sphere(x, y):
    x = np.power(x - y, 2);
    total = np.sum(x, axis=1 if len(x.shape) == 2 else 0);
    return total


@tf.function
def sphere_tf(x, y):
    x = tf.math.pow(x - y, 2)
    total = tf.reduce_sum(x, axis=1 if len(x.shape) == 2 else 0, keep_dims=True)
    return -total


def test_tfpsomp(D, N, batch_size, maxiter, loops):
    bounds = np.asarray([[-5] * D, [5] * D]).astype(np.float32);
    y = tf.placeholder(dtype=tf.float32, shape=(batch_size, D,))
    feed_y = np.zeros(shape=(batch_size, D))
    p = tfPSOMP(W=0.5, C1=1, C2=2, N=N, D=D);
    res = p.build_batch_v4(func=sphere_tf, y=y,
                           bounds=bounds, maxiter=maxiter, batch_size=batch_size, k=1)

    sess = tf_util.make_session(num_cpu=6);
    sess.run(tf.global_variables_initializer())

    def min_(loops):
        t0 = time.time()
        xs = [];
        for i in range(loops):
            x = p.minimize(res, sess=sess, paras_dict={y: feed_y})
            xs.append(x)
        t1 = time.time()
        time_slap = t1 - t0
        t0 = t1
        return time_slap

    #        sess.close();
    return min_


def test_cpupso(D, N, batch_size, maxiter, loops):
    y = np.zeros(shape=(batch_size, D))
    bounds = np.asarray([[-5] * D, [5] * D]).astype(np.float32);
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
    bounds = np.asarray([[-5] * D, [5] * D]).astype(np.float32);
    y = tf.placeholder(dtype=tf.float32, shape=(batch_size, D,))
    feed_y = np.zeros(shape=(batch_size, D))

    p = v1tfPSOMP(W=0.5, C1=1, C2=2, N=N, D=D);
    res = p.build_batch_v2(func=sphere_tf, y=y,
                           bounds=bounds, maxiter=maxiter, batch_size=batch_size)

    sess = tf_util.make_session(num_cpu=6);
    sess.run(tf.global_variables_initializer())

    def min_(loops):
        t0 = time.time()
        xs = [];
        for i in range(loops):
            x = p.minimize(res, sess=sess, paras_dict={y: feed_y})
            xs.append(x)
        t1 = time.time()
        time_slap = t1 - t0
        t0 = t1
        return time_slap

    #    sess.close();
    return min_


import tikzplotlib


def save2tex(filename):
    tikzplotlib.save(filename, encoding='utf-8');


if __name__ == "__main__":
    #    test_funcs = [test_v1tfpsomp];
    import pandas as pd;
    import seaborn as sns;
    import matplotlib.pyplot as plt;

    test_funcs = [test_tfpsomp, test_cpupso, test_v1tfpsomp];
    # test_funcs = [test_tfpsomp];
    loops = 100;

    Ds = [1, 5, 10, 20, 50];
    Ls = [1, 5, 10, 20, 50, 100];
    Bs = [1, 16, 32, 64, 128]
    Ns = [1, 5, 10, 50, 100, 200]
    # Ds = [1, 5];
    # Ls = [1, 5];
    # Bs = [1, 16]
    # Ns = [1, 5]

    results = []

    DLs = [(i, j, k) for i in Ds for (j, k) in enumerate(Ls)]
    NBs = [(i, j) for i in Ns for j in Bs]

    # D iL L N B
    exp1 = [(5, 2, 10, iN, N, B) for (iN, N) in enumerate(Ns) for B in Bs]
    exp2 = [(D, iL, L, 4, 100, 64) for D in Ds for (iL, L) in enumerate(Ls)]
    exps = (exp1, exp2)

    for func in test_funcs:
        for exp in exps:
            for (D, iL, L, iN, N, B) in exp:
                print(func.__name__, D, L, N, B)
                sub_func = func(D, N, B, L, loops)
                times = sub_func(loops)
                results.append((func.__name__, D, N, B, iN,
                                iL, loops, times / loops))

    table = pd.DataFrame(results);
    table.columns = ['Algs.', 'D', 'N', 'batch_size', 'maxpop', 'maxiter', 'loops', 'times'];
    table.to_pickle('pso_test_mi.pkl');

    table = pd.read_pickle('pso_test_mi.pkl');

    # table = table[(table["D"] == 5) & (table['maxiter'] == 2)]
    x_lab = "maxpop"
    # x_lab = "maxiter"
    table[x_lab] = table[x_lab] * 5;
    algs = table['Algs.'].unique();
    idx = table['Algs.'] == algs[0];
    table.loc[idx, x_lab] = table[idx][x_lab] - 1;
    idx = table['Algs.'] == algs[2];
    table.loc[idx, x_lab] = table[idx][x_lab] + 1;

    # for i, j in enumerate(Ns):
    #     table.loc[table["N"] == j, 'N'] = i
    # del table['batch_size']
    # del table['N']
    # del table['loops']

    def_fig_size_width, def_fig_size_height = plt.rcParams['figure.figsize'];

    f, axs = plt.subplots(1, 1, figsize=(def_fig_size_width * 1,
                                         def_fig_size_height * 1), )
    sns.scatterplot(
        data=table, x="N", y="times", hue="Algs.",
        style="batch_size", s=100,
        ax=axs
    )
    plt.savefig('results-pso.pdf', dpi=300, bbox_inches='tight');
    save2tex('results-pso.tex');
    plt.show()

    # sns.lineplot(data=table, x="maxiter", y="times", hue="Algs.", )
    # plt.savefig('results-pso-mi.pdf', dpi = 300, bbox_inches = 'tight');
    # save2tex('results-pso-mi.tex');
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
