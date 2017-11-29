import tensorflow as tf
import numpy as np

g1 = tf.Graph()
g2 = tf.Graph()

print('default_graph=' , tf.get_default_graph())
print('g1=' , g1)
print('g2=' , g2)

with g1.as_default():
    x_data = np.random.rand(100).astype(np.float32)
    y_data = x_data * 0.1 + 0.3
    W = tf.Variable(tf.random_uniform([1] , -1.0 , 1.0))
    b = tf.Variable(tf.zeros([1]))
    y = W * x_data + b

    loss = tf.reduce_mean(tf.square(y - y_data))
    print('num of trainable variables=' , len(tf.trainable_variables()))
    print('num of global variables=' , len(tf.global_variables()))
    print('g1=' , g1)
    print('default graph=' , tf.get_default_graph())

print('default graph=' , tf.get_default_graph())
W2 = tf.Variable(tf.random_uniform([1] , -1.0 , 1.0))
print('num of trainable variables=', len(tf.trainable_variables()))
print('num of global variables=', len(tf.global_variables()))

# 在各图下建立会话进行计算
with g1.as_default():
    sess1 = tf.Session(graph=g1)
    print('sess1=' , sess1)
    init = tf.global_variables_initializer()
    sess1.run(init)
    train = tf.train.GradientDescentOptimizer(0.5).minimize(loss)
    for step in range(201):
        sess1.run(train)
        if step %100 ==0:
            print('step=',step , 'W=',sess1.run(W) , 'b=',sess1.run(b))
