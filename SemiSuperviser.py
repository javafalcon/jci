# -*- coding: utf-8 -*-
"""
Created on Thu Jun 20 10:53:13 2019

@author: falcon1

define semi supervise learning
"""

import math
class SemiSuperviser:
    def __init__(self, **param):
        self.rampup_length = param["rampup_length"]
        self.rampdown_length = param["rampdown_length"]
        self.num_epochs = param["num_epochs"]
        self.learning_rate_max = param["learning_rate_max"]
        self.scaled_unsup_weight_max = param["scaled_unsup_weight_max"]
        self.gammer=param["gammer"]
        self.beita=param["beita"]
    def rampup(self, epoch):
        if epoch < self.rampup_length:
            p = 1.0 - float(epoch)/self.rampup_length
            return math.exp(-p * p * self.gammer)
        else:
            return 1.0
    
    def rampdown(self, epoch):
        if epoch >= self.num_epochs - self.rampdown_length:
            ep = (epoch - (self.num_epochs - self.rampdown_length)) * self.beita
            return math.exp(-(ep*ep) / self.rampdown_length)
        return 1.0
    
    def unsupWeight(self, epoch):
        return self.rampup(epoch) * self.scaled_unsup_weight_max
    
    # learning rate
    def learningRate(self, epoch):
        return self.rampup(epoch) * self.rampdown(epoch) * self.learning_rate_max