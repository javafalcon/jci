# -*- coding: utf-8 -*-
"""
Created on Thu Nov  1 14:31:08 2018

@author: Administrator
"""

import tensorflow as tf
import numpy as np

# 导入 MNIST 数据
from tensorflow.examples.tutorials.mnist import input_data

mnist = input_data.read_data_sets("E:/Repoes/pythonProgram/Tensor/mnist_data", one_hot=True)
trainX = mnist.train.images
testX = mnist.test.images
# 参数
learning_rate = 0.61  # 学习速率
training_epochs = 3  # 训练批次
batch_size = 128  # 随机选择训练数据大小
display_step = 10  # 展示步骤


# 网络参数
# 我这里采用了三层编码，实际针对mnist数据，隐层两层，分别为256，128效果最好
n_hidden_1 = 512  # 第一隐层神经元数量
n_hidden_2 = 128  # 第二
n_hidden_3 = 64  # 第三
n_input = 784  # 输入
scale = 0.1
# tf Graph输入
X = tf.placeholder("float32", [None, n_input])

# 权重初始化
weights = {
    'encoder_h1': tf.Variable(tf.random_normal([n_input, n_hidden_1])),
    'encoder_h2': tf.Variable(tf.random_normal([n_hidden_1, n_hidden_2])),
    'encoder_h3': tf.Variable(tf.random_normal([n_hidden_2, n_hidden_3])),
    'decoder_h1': tf.Variable(tf.random_normal([n_hidden_3, n_hidden_2])),
    'decoder_h2': tf.Variable(tf.random_normal([n_hidden_2, n_hidden_1])),
    'decoder_h3': tf.Variable(tf.random_normal([n_hidden_1, n_input])),
}

# 偏置值初始化
biases = {
    'encoder_b1': tf.Variable(tf.random_normal([n_hidden_1])),
    'encoder_b2': tf.Variable(tf.random_normal([n_hidden_2])),
    'encoder_b3': tf.Variable(tf.random_normal([n_hidden_3])),
    'decoder_b1': tf.Variable(tf.random_normal([n_hidden_2])),
    'decoder_b2': tf.Variable(tf.random_normal([n_hidden_1])),
    'decoder_b3': tf.Variable(tf.random_normal([n_input])),
}


# 开始编码
def encoder(x):
    # sigmoid激活函数，layer = x*weights['encoder_h1']+biases['encoder_b1']
    layer_1 = tf.nn.relu(tf.add(tf.matmul(x + scale * tf.random_normal((n_input,)), 
                                   weights['encoder_h1']),
                                   biases['encoder_b1']))
    layer_2 = tf.nn.relu(tf.add(tf.matmul(layer_1, weights['encoder_h2']),
                                   biases['encoder_b2']))
    layer_3 = tf.nn.relu(tf.add(tf.matmul(layer_2, weights['encoder_h3']),
                                   biases['encoder_b3']))
    return layer_3


# 开始解码
def decoder(x):
    # sigmoid激活函数,layer = x*weights['decoder_h1']+biases['decoder_b1']
    layer_1 = tf.nn.relu(tf.add(tf.matmul(x, weights['decoder_h1']),
                                   biases['decoder_b1']))#sigmoid()
    layer_2 = tf.nn.relu(tf.add(tf.matmul(layer_1, weights['decoder_h2']),
                                   biases['decoder_b2']))#sigmoid
    layer_3 = tf.nn.relu(tf.add(tf.matmul(layer_2, weights['decoder_h3']),
                                   biases['decoder_b3']))#sigmoid
    return layer_3

def gethiddenfeature(sess, ds):
    decoder_op = decoder(X)
    sess.run(decoder_op, feed_dict={X: ds})


train_size = len(trainX)
# 构造模型
encoder_op = encoder(X)
#encoder_result = encoder_op
decoder_op = decoder(encoder_op)

# 预测
y_pred = decoder_op
# 实际输入数据当作标签
y_true = X

# 定义代价函数和优化器，最小化平方误差,这里可以根据实际修改误差模型
loss = tf.reduce_mean(tf.pow(y_true - y_pred, 2))
#optimizer = tf.train.GradientDescentOptimizer(learning_rate).minimize(loss)
optimizer = tf.train.AdagradOptimizer(learning_rate).minimize(loss)
# 初始化变量
init = tf.global_variables_initializer()

# 运行
with tf.Session() as sess:
    sess.run(init)
    # 总的batch
    total_batch = int( np.ceil(train_size / batch_size)) #int(mnist.train.num_examples / batch_size)
    # 开始训练
    for epoch in range(training_epochs):
        for i in range(total_batch):
            start = (i * batch_size) % train_size
            end = min(start + batch_size, train_size)
            batch_xs = trainX[start:end]
            #batch_xs, batch_ys = mnist.train.next_batch(batch_size)
            _, c = sess.run([optimizer, loss], feed_dict={X: batch_xs})
        # 展示每次训练结果
            if epoch % display_step == 0:
                print("Epoch:", '%02d' % (epoch + 1),
                      "cost=", "{:.9f}".format(c))

 
    tx = gethiddenfeature(sess, testX)
