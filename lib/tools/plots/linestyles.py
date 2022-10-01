# -*- coding: utf-8 -*-
"""
Created on Fri Nov 29 14:31:15 2019

@author: bb
"""
# For example, (0, (3, 10, 1, 15)) means (3pt line, 10pt space, 1pt line, 15pt space) with no offset
linestyle_tuple = dict([
     ('solid',        (0, (1, 0))),
     ('long dashed',                (0, (10, 1))),
     ('dotted',                (0, (1, 1))),
     ('dashdot',            (0, (3, 2, 1, 2))),
     
     ('loosely dotted',        (0, (1, 10))),
     ('densely dotted',        (0, (1, 1))),

     ('loosely dashed',        (0, (5, 10))),
     ('dashed',                (0, (5, 5))),
     ('densely dashed',        (0, (5, 1))),

     ('loosely dashdotted',    (0, (3, 10, 1, 10))),
     ('dashdotted',            (0, (3, 5, 1, 5))),
     ('densely dashdotted',    (0, (3, 2, 1, 1, 1, 2))),

     ('dashdotdotted',         (0, (3, 5, 1, 5, 1, 5))),
     ('loosely dashdotdotted', (0, (3, 10, 1, 10, 1, 10))),
     ('densely dashdotdotted', (0, (3, 1, 1, 1, 1, 1)))]);
