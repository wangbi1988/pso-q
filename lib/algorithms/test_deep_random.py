# -*- coding: utf-8 -*-
"""
Created on Tue Aug 11 14:21:49 2020

@author: bb
"""
import numpy as np;
import tensorflow as tf;
tf.set_random_seed(1234);
print(tf.random_normal((5, 5)).eval(session=tf.Session()))
print(tf.Session().run(tf.random_normal((5, 5))))