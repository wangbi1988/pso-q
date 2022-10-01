# -*- coding: utf-8 -*-
"""
Created on Sun Jul 12 20:25:13 2020

@author: 王碧
"""

import numpy as np;
import marshal;
import pickle;

l = np.random.randint(0, 100, 400);
def join():
    d = {};
    s = ''.join([str(j) for j in l]);
    d[s] = 1;
        
def str_():
    d = {};
    s = str(l);
    d[s] = 1;

def mar():
    d = {};
    s = marshal.dumps(l);
    d[s] = 1;
    
def pik():
    d = {};
    s = pickle.dumps(l);
    d[s] = 1;
        
def tuple_():
    d = {};
    d[tuple(l)] = 1;
        
import timeit;

num = 100;
print(timeit.timeit(join, number=num));
print(timeit.timeit(pik, number=num));
print(timeit.timeit(mar, number=num));
print(timeit.timeit(tuple_, number=num));

def aa():
    a = [];
    for i in range(int(1e+4)):
        a.append(i);
        
print(timeit.timeit(lambda: [i for i in range(int(1e+4))], number=num));
print(timeit.timeit(aa, number=num));