# -*- coding: utf-8 -*-
"""
Created on Sun Jul 12 10:00:26 2020

@author: 王碧
"""


import tensorflow as tf;

eps_ph = tf.placeholder(dtype = tf.float32);
batch_size, n_actions = (2, 3,);
probs = tf.constant(0, dtype = tf.float32, shape = [batch_size, n_actions]);
probs = tf.add(probs, tf.div(1., n_actions));
def true_fn(x):
    return lambda: x;

def false_fun(x):
    return lambda: tf.add(x, 1);

if eps_ph != 1.:
    probs = tf.cond(tf.less(eps_ph, 1), true_fn(probs), false_fun(probs));

with tf.Session() as sess:
    for i in range(10000000):
        a = sess.run(probs, feed_dict = {eps_ph: 1.})