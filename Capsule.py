# -*- coding: utf-8 -*-
"""
Created on Sun May  3 17:08:05 2020

@author: lwzjc
"""
import tensorflow as tf
from tensorflow.keras import backend as K
from tensorflow.keras import layers, initializers,regularizers

def squash(vectors, axis=-1):
    s_squared_norm = K.sum(K.square(vectors), axis, keepdims=True)
    scale = s_squared_norm / (1 + s_squared_norm) / K.sqrt(s_squared_norm + K.epsilon())
    return scale * vectors

def margin_loss(y_true, y_pred):
    L =  y_true * K.square(K.maximum(0., 0.9 - y_pred)) + \
         14*(1 - y_true) * K.square(K.maximum(0., y_pred - 0.1))
    return K.mean(K.sum(L,1))

class CapsuleLayer(layers.Layer):
    def __init__(self, num_capsule, dim_vector, num_routing=3,
                kernel_initializer='glorot_uniform',
                bias_initializer='zeros',
                kernel_regularizer=None,
                **kwargs):
        super(CapsuleLayer, self).__init__(**kwargs)
        self.num_capsule = num_capsule
        self.dim_vector = dim_vector
        self.num_routing = num_routing
        self.kernel_initializer = initializers.get(kernel_initializer)
        self.kernel_regularizer = regularizers.get(kernel_regularizer)
        self.bias_initializer = initializers.get(bias_initializer)
        
        
    def build(self, input_shape):
        super(CapsuleLayer, self).build(input_shape)

        self.input_num_capsule = input_shape[1]
        self.input_dim_vector = input_shape[2]
        
        # Transform matrix
        self.W = self.add_weight(shape=[self.input_num_capsule*self.num_capsule, self.input_dim_vector, self.dim_vector],
                                initializer=self.kernel_initializer,
                                regularizer=self.kernel_regularizer,
                                name='W', trainable=True)
        
        self.bias = self.add_weight(shape=[1,self.input_num_capsule,self.num_capsule,1,1],
                                   initializer=self.bias_initializer,
                                   name='bias',
                                   trainable=False)
        self.built = True
                
    def call(self, inputs):
        # inputs.shape=[None,input_num_capsule,input_dim_vector]
        # Expand dims to [None,input_num_capsule,1,1,input_dim_vector]
        inputs_expand = K.expand_dims(K.expand_dims(inputs,2),2)
        
        # Replicate num_capsule dimension to prepare being multiplied by W
        # Now it has shape = [None,input_num_capsule,num_capsule,1,input_dim_vector]
        inputs_tiled = K.tile(inputs_expand, [1,1,self.num_capsule,1,1])
        
        # Begin: inputs_hat computation V2
        # Compute 'inputs * W' by scanning inputs_tiled on dimension 0. 
        # inputs_hat.shape = [None, input_num_capsule,num_capsule,1,dim_vector]
        inp = K.reshape(inputs_tiled,(-1, self.input_num_capsule*self.num_capsule,1,self.input_dim_vector))
        
        inputs_hat = tf.map_fn(lambda x: K.batch_dot(x,self.W,[2,1]), elems=inp)
        inputs_hat = K.reshape(inputs_hat, (-1, self.input_num_capsule, self.num_capsule, 1, self.dim_vector))
        
        # Begin: routing algorithm V2
        # Routing alogrithm V2. Use iteration. V2 and V1 both work without much difference on performace
        assert self.num_routing > 0, 'The num_routing should be > 0'
        for i in range(self.num_routing):
            c = tf.nn.softmax(self.bias, axis=2)
            outputs = squash(K.sum(c*inputs_hat, 1, keepdims=True))
            
            # last iteration needs not compute the bias which will not be passed to the graph any more anyway.
            if i != self.num_routing - 1:
                (self.bias).assign_add(K.sum(inputs_hat*outputs, [0,-1], keepdims=True))
                
        return K.reshape(outputs, [-1, self.num_capsule, self.dim_vector])
    
    def compute_output_shape(self,input_shape):
        return tuple([None, self.num_capsule, self.dim_vector])
    
class Length(layers.Layer):    
    def call(self, inputs, **kwargs):
        return K.sqrt(K.sum(K.square(inputs), -1))
    
    def compute_output_shape(self, input_shape):
        return input_shape[:-1]

class Mask(layers.Layer):   
    def call(self, inputs, **kwargs):
        # use true label to select target capsule, shape=[batch_size, num_capsule]
        if type(inputs) is list: # true label is provided with shape=[batch_size, n_classes]
            assert len(inputs)==2
            inputs, mask = inputs
        else: # if no true label, mask by the max length of vectors of capsule
            x = inputs
            x = (x - K.max(x, 1, True)) / K.epsilon() + 1
            mask = K.clip(x, 0, 1)
        
        # masked inputs, shape=[batch_size, dim_vector]
        inputs_masked = K.batch_dot(inputs, mask, [1,1])
        return inputs_masked
    
    def computer_output_shape(self, input_shape):
        if type(input_shape[0]) is tuple: # true lable provided
            return tuple([None, input_shape[0][-1]])
        else:
            return tuple([None, input_shape[-1]])   
def PrimaryCap(inputs, dim_vector, n_channels, kernel_size, strides, padding, name='primarycap'):
    """
    Apply Conv2D `n_channels` times and concatenate all capsules
    :param inputs: 4D tensor, shape=[None, width, height, channels]
    :param dim_vector: the dim of the output vector of capsule
    :param n_channels: the number of types of capsules
    :return: output tensor, shape=[None, num_capsule, dim_vector]
    """
    output = layers.Conv2D(filters=dim_vector*n_channels, kernel_size=kernel_size, strides=strides, padding=padding,
                           name=name+'_conv2d')(inputs)
    outputs = layers.Reshape(target_shape=[-1, dim_vector], name=name+'_reshape')(output)
    return layers.Lambda(squash, name=name+'_squash')(outputs)
