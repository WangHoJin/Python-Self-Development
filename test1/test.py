import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np

W = tf.Variable([[3,3,3],[3,3,3],[3,3,3]])
b = tf.Variable([[3],[3],[3]])







init = tf.global_variables_initializer()
sess = tf.Session()
sess.run(init)

result = tf.nn.relu(tf.matmul(W,b))



print(sess.run(W))
print(sess.run(b))
print(sess.run(result))
