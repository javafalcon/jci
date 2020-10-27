# -*- coding: utf-8 -*-
"""
Created on Tue Oct 27 21:05:51 2020
transformer_v3,含mask掩码，位置编码使用0~maxlen的embedding层，
@author: lwzjc
"""
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers

# 将padding位mark，原来为0的padding项的mark输出为1
def create_padding_mask(x):
    # 获取为0的padding项
    seq = tf.cast(tf.math.equal(tf.reduce_sum(x, axis=-1), 0), tf.float32)
    # 扩充维度以便用于attention矩阵
    return seq[:, np.newaxis, np.newaxis, :] # (batch_size, 1, 1, seq_len)

# 自注意力机制
def scaled_dot_product_attention(q, k, v, mask=None):
    
    # query key 相乘获取匹配关系
    matmul_qk = tf.matmul(q, k, transpose_b=True)
    
    # 使用dk进行缩放
    dk = tf.cast(tf.shape(k)[-1], tf.float32)
    scaled_attention_logits = matmul_qk / tf.math.sqrt(dk)
    
    # 掩码.将被掩码的token乘以-1e9（表示负无穷），这样
    # softmax之后就为0， 不对其它token产生影响
    if mask is not None:
        scaled_attention_logits += (mask * -1e9)
    # 通过softmax获取attention权重
    attention_weights = tf.nn.softmax(scaled_attention_logits, axis=-1)
    # attention乘上value
    output = tf.matmul(attention_weights, v) # (..., seq_len_v, depth)
    
    return output, attention_weights

class MultiHeadAttention(layers.Layer):
    def __init__(self, embed_dim, num_heads=8):
        super(MultiHeadAttention, self).__init__()
        self.num_heads = num_heads
        self.embed_dim = embed_dim
        
        # embd_dim必须可以被num_heads整除
        assert embed_dim % num_heads == 0
        # 分头后的维度
        self.projection_dim = embed_dim // num_heads
        self.wq = layers.Dense(embed_dim)
        self.wk = layers.Dense(embed_dim)
        self.wv = layers.Dense(embed_dim)
        self.dense = layers.Dense(embed_dim)
        
    def separate_heads(self, x, batch_size):
        x = tf.reshape(x, (batch_size, -1, self.num_heads, self.projection_dim))
        return tf.transpose(x, perm=[0, 2, 1, 3])
    
    def call(self, v, k, q, mask=None):
        batch_size = tf.shape(q)[0]
        
        # 分头前的前向网络，获取q, k, v语义
        q = self.wq(q)
        k = self.wq(k)
        v = self.wv(v)
        
        # 分头
        q = self.separate_heads(q, batch_size) # [batch_size, num_heads, seq_len_q, projection_dim]
        k = self.separate_heads(k, batch_size)
        v = self.separate_heads(v, batch_size)
        
        # 通过缩放点积注意力层
        scaled_attention, attention_weights = scaled_dot_product_attention(q, k, v, mask)
        # 把多头维度后移
        scaled_attention = tf.transpose(scaled_attention, [0, 2, 1, 3])
        # 合并多头
        concat_attention = tf.reshape(scaled_attention, (batch_size, -1, self.embed_dim))
        
        # 全连接重塑
        output = self.dense(concat_attention)
        
        return output, attention_weights
    
# 构造前向网络
def point_wise_feed_forward_network(d_model, diff):
    # d_model 即embed_dim
    return tf.keras.Sequential([
        layers.Dense(diff, activation='relu'),
        layers.Dense(d_model)])    
    
# transformer编码层
class EncoderLayer(layers.Layer):
    def __init__(self, d_model, n_heads, ffd, dropout_rate=0.1):
        super(EncoderLayer, self).__init__()
        self.mha = MultiHeadAttention(d_model, n_heads)
        self.ffn = point_wise_feed_forward_network(d_model, ffd)
        
        self.layernorm1 = layers.LayerNormalization(epsilon=1e-6)
        self.layernorm2 = layers.LayerNormalization(epsilon=1e-6)
        
        self.dropout1 = layers.Dropout(dropout_rate)
        self.dropout2 = layers.Dropout(dropout_rate)
        
    def call(self, inputs, training, mask=None):
        att_output, _ = self.mha(inputs, inputs, inputs, mask)
        att_output = self.dropout1(att_output, training=training)
        out1 = self.layernorm1(inputs + att_output)
        # 前向网络
        ffn_output = self.ffn(out1)
        ffn_output = self.dropout2(ffn_output, training=training)
        out2 = self.layernorm2(out1 + ffn_output)
        
        return out2

class Encoder(layers.Layer):
    def __init__(self, n_layers, d_model, n_heads, ffd,
                 max_seq_len, dropout_rate=0.1):
        super(Encoder, self).__init__()
        self.seq_len = max_seq_len
        self.n_layers = n_layers
        self.d_model = d_model
        self.pos_emb = layers.Embedding(max_seq_len, d_model)
        self.encoder_layer = [EncoderLayer(d_model, n_heads, ffd, dropout_rate)
                              for _ in range(n_layers)]
        self.dropout = layers.Dropout(dropout_rate)
        
    def call(self, inputs, training, mask=None):
        word_emb = tf.cast(inputs, tf.float32)
        positions = tf.range(start=0, limit=self.seq_len, delta=1)
        positions = self.pos_emb(positions)
        emb = word_emb + positions
        
        x = self.dropout(emb, training=training)
        for i in range(self.n_layers):
            x = self.encoder_layer[i](x, training, mask)
        return x

