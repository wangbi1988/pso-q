# -*- coding: utf-8 -*-
"""
Created on Sat Jun 13 10:59:05 2020

@author: bb
"""

class abstractInjector(object):
    def __init__(self, name, **args):
        self.name = name;
        
    def callback(self, **args):
        raise NotImplementedError();
        
    def trigger(self, **args):
        raise NotImplementedError();
        
    def getSummary(self):
        return self.summary;
        

class timestepBasedInjector(abstractInjector):
    def __init__(self, name, freq, **args):
        assert isinstance(freq, int), 'parameter freq need be int';
        self.freq = freq;
        self.name = name;
    
    def trigger(self, timestep, **args):
        return (timestep == 0) or ((timestep + 1) % self.freq == 0);

