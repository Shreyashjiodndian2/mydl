import tensorflow as tf
import tensorboard
import pandas as pd
import numpy as np
import time

with tf.name_scope("input_layer") as scope:
    x1 = tf.placeholder(shape=[None,1], name="input", dtype="float32")
    x2 = tf.placeholder ( shape=[ None , 1 ] , name="input" , dtype="float32")
    x3 = tf.placeholder ( shape=[ None , 1 ] , name="input" , dtype="float32")
    x4 = tf.placeholder ( shape=[ None , 1 ] , name="input" , dtype="float32")
    x5 = tf.placeholder ( shape=[ None , 1 ] , name="input" , dtype="float32")
    x6 = tf.placeholder ( shape=[ None , 1 ] , name="input" , dtype="float32")

with tf.name_scope("target") as scope:
    Target = tf.placeholder(shape=[None,1], name="Target", dtype="float")

with tf.name_scope(name="weights") as scope:
    w11 = tf.Variable(initial_value=tf.random_normal(shape=[1,1], stddev=2), name="w11", trainable=True, dtype="float32")
    w12 = tf.Variable(initial_value=tf.random_normal(shape=[1,1], stddev=2), name="w11", trainable=True, dtype="float32")
    w21 = tf.Variable(initial_value=tf.random_normal(shape=[1,1], stddev=2), name="w11", trainable=True, dtype="float32")
    w22 = tf.Variable(initial_value=tf.random_normal(shape=[1,1], stddev=2), name="w11", trainable=True, dtype="float32")
    w31 = tf.Variable(initial_value=tf.random_normal(shape=[1,1], stddev=2), name="w11", trainable=True, dtype="float32")
    w32 = tf.Variable(initial_value=tf.random_normal(shape=[1,1], stddev=2), name="w11", trainable=True, dtype="float32")
    w41 = tf.Variable(initial_value=tf.random_normal(shape=[1,1], stddev=2), name="w11", trainable=True, dtype="float32")
    w42 = tf.Variable(initial_value=tf.random_normal(shape=[1,1], stddev=2), name="w11", trainable=True, dtype="float32")
    w51 = tf.Variable(initial_value=tf.random_normal(shape=[1,1], stddev=2), name="w11", trainable=True, dtype="float32")
    w52 = tf.Variable(initial_value=tf.random_normal(shape=[1,1], stddev=2), name="w11", trainable=True, dtype="float32")
    w61 = tf.Variable(initial_value=tf.random_normal(shape=[1,1], stddev=2), name="w61", trainable=True, dtype="float32")
    w62 = tf.Variable(initial_value=tf.random_normal(shape=[1,1], stddev=2), name="w62", trainable=True, dtype="float32")

with tf.name_scope(name="biases") as scope:
    b = tf.Variable(0, name="biases", trainable=True)

with tf.name_scope(name="output") as scope:
    y11 = tf.pow(x1, w12)
    y12 = tf.matmul(y11, w11)
    y21 = tf.pow(x2, w22)
    y22 = tf.matmul(w21, y21)
    y31 = tf.pow(x3, w32)
    y32 = tf.matmul(y31, w31)
    y41 = tf.pow(x4, w42)
    y42 = tf.matmul(y41, w41)
    y51 = tf.pow(x5, w52)
    y52 = tf.matmul(y51, w51)
    y61 = tf.pow(x6, w62)
    y62 = tf.matmul(y61, w61)
    add12 = tf.add(y12, y22)
    add34 = tf.add(y32,y42)
    add56 = tf.add(y52,y62)
    add1234 = tf.add(add12,add34)
    y = tf.add(add1234, add56)

with tf.name_scope(name="optimizer") as scope:
    global_itr = tf.Variable ( 0 , name="global_itr" , trainable=False )
    cost = tf.nn.softmax_cross_entropy_with_logits ( logits=y , labels=Target , name="softmax_cost_function" )
    cost = tf.reduce_mean ( cost )
    tf.summary.scalar ( "cost" , cost )
    optimizer = tf.train.AdamOptimizer ().minimize ( cost , global_step=global_itr )

db = pd.read_csv("E:/datasets/datathon/train/features/features.csv", skiprows=1, na_filter=True)
db = np.array(db)
#db = np.loadtxt("E:/datasets/datathon/train/features/features.csv", skiprows=1, delimiter=",")
tar = pd.read_csv("E:/datasets/datathon/train/labels/labels.csv",  na_filter=True)
print(tar)
tar = np.array(tar)
epochs = 10
tar1 = np.zeros(shape=[3790, 1], dtype="float")
v1 = np.zeros(shape=[3790, 1], dtype="float")
v2 = np.zeros(shape=[3790, 1], dtype="float")
v3 = np.zeros(shape=[3790, 1], dtype="float")
v4 = np.zeros(shape=[3790, 1], dtype="float")
v5 = np.zeros(shape=[3790, 1], dtype="float")
v6 = np.zeros(shape=[3790, 1], dtype="float")
print(tar)
for i in range(3789):
    v1[i][0] = db[:,0][i]
    v2[i][0] = db[:,1][i]
    v3[i][0] = db[:,2][i]
    v4[i][0] = db[:,3][i]
    v5[i][0] = db[:,4][i]
    v6[i][0] = db[:,5][i]

with tf.name_scope("optimizer") as scope:
    global_itr= tf.Variable(0, name="global_itr", trainable=False)
    cost = tf.nn.softmax_cross_entropy_with_logits(logits=y, labels=Target, name="softmax_cost_function")
    cost = tf.reduce_mean(cost)
    tf.summary.scalar("cost",cost)

    optimizer = tf.train.AdamOptimizer().minimize(cost, global_step=global_itr)

with tf.name_scope('accuracy') as scope:
    correct_pred = tf.equal(tf.argmax(y, 1), tf.argmax(Target, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))


with tf.Session() as sess:
    summaryMerged = tf.summary.merge_all()

    filename = "E:/run/run/yesbank" + time.strftime("%H+%M+%S")
    tf.global_variables_initializer().run()

    for i in range(epochs):
        print(sess.run(y,feed_dict={x1: v1, x2: v2, x3:v3, x4:v4, x5:v5, x6:v6 ,Target:tar}), cost, accuracy)
