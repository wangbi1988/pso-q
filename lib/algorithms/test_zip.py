# -*- coding: utf-8 -*-
"""
Created on Sun Aug  9 23:09:40 2020

@author: bb
"""

a = [[[1,2,3],[4,5,6]], [['a', 'b', 'c'], ['d', 'e', 'f']]];
#want [[1,4], ['a', 'd']] 

z = ([], [], [])
for i in zip(*a):
    for l, j in enumerate(zip(*i)):
        z[l % len(z)].append(j)
#        z[0].append(j);
#        z[1].append(k);
#        z[2].append(l);
z = ([], [], [])
for i in a:
    for j in i:
        for l, k in enumerate(j):
            z[l % len(z)].append(k)