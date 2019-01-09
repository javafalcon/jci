# -*- coding: utf-8 -*-
"""
Created on Wed Jan  9 13:48:34 2019

@author: Administrator
"""
import tflearn
import tensorflow as tf
from tflearn.data_utils import shuffle, to_categorical
from tflearn.layers.core import input_data, dropout, fully_connected
from tflearn.layers.normalization import local_response_normalization
from tflearn.layers.normalization import batch_normalization
from tflearn.layers.conv import conv_2d, max_pool_2d
from tflearn.layers.estimator import regression
from tflearn.data_preprocessing import ImagePreprocessing
from tflearn.data_augmentation import ImageAugmentation
from sklearn.model_selection import LeaveOneOut, KFold
from sklearn.metrics import accuracy_score, auc, roc_curve, matthews_corrcoef

class CNNnet:
    def _init_(self, width, height, channels, classes, 
               finalAct="softmax", lossFun='categorical_crossentropy',
               lr=0.001):
        self.width = width
        self.height = height
        self.channels = channels
        self.classes = classes
        self.finalAct = finalAct
        self.lossFun = lossFun
        self.lr = lr
        
    def create_Alex(self):
        network = input_data(shape=[None, self.width, self.height, self.channels])
        network = conv_2d(network, 96, 11, strides=4, activation='relu')
        network = max_pool_2d(network, 3, strides=2)#[-1,10,10,96]
        network = local_response_normalization(network)
        network = conv_2d(network, 256, 5, activation='relu')
        network = max_pool_2d(network, 3, strides=2)#[-1,5,5,256]
        network = local_response_normalization(network)
        network = conv_2d(network, 384, 3, activation='relu')
        network = conv_2d(network, 384, 3, activation='relu')
        network = conv_2d(network, 256, 3, activation='relu')
        network = max_pool_2d(network, 3, strides=2)#[-1,3,3,256]
        network = local_response_normalization(network)
        network = fully_connected(network, 4096, activation='tanh')
        network = dropout(network, 0.5)
        network = fully_connected(network, 4096, activation='tanh')
        network = dropout(network, 0.5)
        network = fully_connected(network, self.classes, activation=self.finalAct)
        
        return network   
    
    def create_VGG(self):
        network = input_data(shape=[None, self.width, self.height, self.channels])

        network = conv_2d(network, 64, 3, activation='relu')
        network = conv_2d(network, 64, 3, activation='relu')
        network = max_pool_2d(network, 2, strides=2)#[-1,25,10,64]
    
        network = conv_2d(network, 128, 3, activation='relu')
        network = conv_2d(network, 128, 3, activation='relu')
        network = max_pool_2d(network, 2, strides=2)#[-1,13,5,128]
    
        network = conv_2d(network, 256, 3, activation='relu')
        network = conv_2d(network, 256, 3, activation='relu')
        network = conv_2d(network, 256, 3, activation='relu')
        network = max_pool_2d(network, 2, strides=2)#[-1,7,3,256]
        
        network = conv_2d(network, 512, 3, activation='relu')
        network = conv_2d(network, 512, 3, activation='relu')
        network = conv_2d(network, 512, 3, activation='relu')
        network = max_pool_2d(network, 2, strides=2)
    
        network = conv_2d(network, 512, 3, activation='relu')
        network = conv_2d(network, 512, 3, activation='relu')
        network = conv_2d(network, 512, 3, activation='relu')
        network = max_pool_2d(network, 2, strides=2)
        
        network = fully_connected(network, 2048, activation='relu')
        network = dropout(network, 0.5)
        network = fully_connected(network, 4096, activation='relu')
        network = dropout(network, 0.5)
        network = fully_connected(network, self.classes, activation=self.finalAct)
    
        return network

    def create_smallerVGG(self):
        network = input_data(shape=[None, self.width, self.height, self.channels])
        
        network = conv_2d(network, 32, 2, activation='relu')
        network = batch_normalization(network)
        network = max_pool_2d(network, 3)
        network = dropout(network, 0.25)
        
        network = conv_2d(network, 64, 3, activation='relu')
        network = batch_normalization(network)
        network = conv_2d(network, 64, 3, activation='relu')
        network = batch_normalization(network)
        network = max_pool_2d(network,2)
        network = dropout(network, 0.25)
        
        network = conv_2d(network, 128, 3, activation='relu')
        network = batch_normalization(network)
        network = conv_2d(network, 128, 3, activation='relu')
        network = batch_normalization(network)
        network = max_pool_2d(network,2)
        network = dropout(network, 0.25)
        
        network = fully_connected(network, 1024, activation='relu')
        network = dropout(network, 0.5)
        network = fully_connected(network, self.classes, activatioin=self.finalAct)
        
        return network
        
    def create_Cifar(self):
        network = input_data(shape=[None, self.width, self.height, self.channels])
        network = conv_2d(network, 32, 3, activation='relu')
        network = max_pool_2d(network, 2)
        network = dropout(network, 0.75)
        network = conv_2d(network, 64, 3, activation='relu')
        network = conv_2d(network, 64, 3, activation='relu')
        network = max_pool_2d(network, 2)
        network = dropout(network, 0.5)
        network = fully_connected(network, 512, activation='sigmoid')
        network = dropout(network, 0.8)
        network = fully_connected(network, 512, activation='sigmoid')
        network = dropout(network, 0.8)
        network = fully_connected(network, self.classes, activation=self.finalAct)
        
        return network
    
        