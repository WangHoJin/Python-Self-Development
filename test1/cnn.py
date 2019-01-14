import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets("./mnist/data/", one_hot=True, reshape=False)

X = tf.placeholder(tf.float32, shape=[None, 28, 28, 1])
Y_label = tf.placeholder(tf.float32, shape=[None, 10])

Kernel1 = tf.Variable(tf.truncated_normal(shape=[4, 4, 1, 4], stddev=0.1))
"""
(4x4x1)필터를 4개 사용 -> shape = [4,4,1,4]
tf.truncated_normal()을 이용해서 초기화
"""
Bias1 = tf.Variable(tf.truncated_normal(shape=[4], stddev=0.1))
"""
이미지와 Kernel을 Conv한 후에 같은 사이즈만큼 더해주기 위한 변수
"""
Conv1 = tf.nn.conv2d(X, Kernel1, strides=[1, 1, 1, 1], padding='SAME') + Bias1
"""
strides=[1,1,1,1] 1칸씩 이동
padding='SAME' stride에 의존하여 padding값이 결정됨
"""
Activation1 = tf.nn.relu(Conv1)

Pool1 = tf.nn.max_pool(Activation1, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')
"""
(14 x 14 x 4) 사이즈를 만들어준다
"""

W1 = tf.Variable(tf.truncated_normal(shape=[14*14*4, 10]))
B1 = tf.Variable(tf.truncated_normal(shape=[10]))
Pool1_flat = tf.reshape(Pool1, [-1, 14*14*4])
OutputLayer = tf.matmul(Pool1_flat, W1)+B1

Loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=Y_label, logits=OutputLayer))
train_step = tf.train.AdadeltaOptimizer(0.005).minimize(Loss)

#정확도 확인
correct_prediction = tf.equal(tf.argmax(OutputLayer, 1), tf.argmax(Y_label, 1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

with tf.Session() as sess:
    print("Start...")
    sess.run(tf.global_variables_initializer())
    for i in range(100):
        trainingData, Y = mnist.train.next_batch(64)
        sess.run(train_step, feed_dict={X: trainingData, Y_label: Y})
        if i % 100 == 0:
            print("%d :"%i, sess.run(accuracy, feed_dict={X: mnist.test.images, Y_label: mnist.test.labels}))
