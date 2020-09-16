# -*- coding: utf-8 -*-
"""
Created on Tue Aug 25 09:15:27 2020

@author: lwzjc
"""

import numpy as np
import tensorflow as tf
import tensorflow.keras.layers as layers
def get_angles(pos, i, embed_dim):
    """
    基于角度的位置编码方法。计算位置编码矢量的长度
    Parameters
    ----------
    pos : 
        在句子中字的位置序号，取值范围是[0, max_sequence_len).
    i   : int
        字向量的维度，取值范围是[0, embedding_dim).
    embedding_dim : int
        字向量最大维度， 即embedding_dim的最大值.

    Returns
    -------
    float32
        第pos位置上对应矢量的长度.

    """
    angel_rates = 1 / np.power(10000, (2 * (i // 2)) / np.float32(embed_dim))
    return pos * angel_rates

def position_encoding(position, embed_dim):
    angel_rads = get_angles(np.arange(-position,position+1)[:, np.newaxis], 
                            np.arange(embed_dim)[np.newaxis, :], 
                            embed_dim)
    sines = np.sin(angel_rads[:, 0::2])
    cones = np.cos(angel_rads[:, 1::2])
    pos_encoding = np.concatenate([sines, cones], axis=-1)
    pos_encoding = pos_encoding[np.newaxis, ...]
    return tf.cast(pos_encoding, dtype=tf.float32)

# 将padding位mark，原来为0的padding项的mark输出为1
def create_padding_mask(seq):
    # 获取为0的padding项
    seq = tf.cast(tf.math.equal(seq, 0), tf.float32)
    # 扩充维度以便用于attention矩阵
    return seq[:, np.newaxis, np.newaxis, :] # (batch_size, 1, 1, seq_len)
    
def scaled_dot_product_attention(q, k, v, mask):
    # query key 相乘获取匹配关系
    matmul_qk = tf.matmul(q, k, transpose_b=True)
    # 使用dk进行缩放
    dk = tf.cast(tf.shape(k)[-1], tf.float32)
    scaled_attention_logits = matmul_qk / tf.math.sqrt(dk)
    
    # 掩码.将被掩码的token乘以-1e9（表示负无穷），这样
    # softmax之后就为0， 不对其它token产生印象
    if mask is not None:
        scaled_attention_logits += (mask * -1e9)
    # 通过softmax获取attention权重
    attention_weights = tf.nn.softmax(scaled_attention_logits, axis=-1)
    # attention乘上value
    output = tf.matmul(attention_weights, v) # (..., seq_len_v, depth)
    return output, attention_weights

    
# 构造mutil head attention层
"""
multi-head attention包含3部分：线性层与分头-缩放点积注意力-头链接-末尾线性层
每个多头注意块有三个输入：Q（查询），K（密钥），V（值）。它们通过第一层线性层并分成多个头
Q，K，V不是一个单独的注意头，而是分成多个头，因为它允许模型共同参与来自不同表征空间的
不同信息。在拆分后，每个头部具有降低的维度，总计算成本与具有全维度的单个头部注意力相同
"""
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
    
    def call(self, v, k, q, mask):
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
    
def point_wise_feed_forward_network(d_model, diff):
    # d_model 即embed_dim
    return tf.keras.Sequential([
        layers.Dense(diff, activation='relu'),
        layers.Dense(d_model)])

class EncoderLayer(layers.Layer):
    def __init__(self, d_model, n_heads, ffd, dropout_rate=0.1):
        super(EncoderLayer, self).__init__()
        self.mha = MultiHeadAttention(d_model, n_heads)
        self.ffn = point_wise_feed_forward_network(d_model, ffd)
        
        self.layernorm1 = layers.LayerNormalization(epsilon=1e-6)
        self.layernorm2 = layers.LayerNormalization(epsilon=1e-6)
        
        self.dropout1 = layers.Dropout(dropout_rate)
        self.dropout2 = layers.Dropout(dropout_rate)
        
    def call(self, inputs, training, mask):
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
                 input_vocab_size, max_seq_len, dropout_rate=0.1):
        super(Encoder, self).__init__()
        
        self.n_layers = n_layers
        self.d_model = d_model
        self.embedding = layers.Embedding(input_vocab_size, d_model)
        self.pos_embedding = position_encoding(max_seq_len, d_model)
        self.encoder_layer = [EncoderLayer(d_model, n_heads, ffd, dropout_rate)
                              for _ in range(n_layers)]
        self.dropout = layers.Dropout(dropout_rate)
        
    def call(self, inputs, training, mask):
        seq_len = inputs.shape[-1]
        word_emb = self.embedding(inputs)       
        word_emb = tf.cumsum(word_emb, axis=1)
        #word_emb *= (tf.cast(self.d_model, tf.float32))
        emb = word_emb + self.pos_embedding[:, :seq_len, :]
        x = self.dropout(emb, training=training)
        for i in range(self.n_layers):
            x = self.encoder_layer[i](x, training, mask)
        return x
    

