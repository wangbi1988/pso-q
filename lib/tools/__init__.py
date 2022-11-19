# -*- coding: utf-8 -*-
"""
Created on Sat Jun 13 11:03:56 2020

@author: bb
"""

from lib.tools.plots.usetex import save2tex;


import platform;
system_ = platform.system();
file_slash = '\\' if system_ == 'Windows' else '/';